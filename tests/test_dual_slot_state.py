"""CUDA-free tests of the real control plane (also executable with unittest)."""
import importlib.util
from pathlib import Path
import sys
import unittest
from collections import deque
from enum import Enum, auto
from types import SimpleNamespace

# nano_qwen.__init__ imports the GPU runner. Load this deliberately CUDA-free
# production module directly so lifecycle regressions need no torch install.
path = Path(__file__).resolve().parents[1] / "src/nano_qwen/engine/pipeline.py"
spec = importlib.util.spec_from_file_location("_dual_slot_state_tests", path)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


class Status(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Request:
    def __init__(self, ident=0, prompt=3, limit=6):
        self.seq_id, self.status = ident, Status.WAITING
        self.token_ids = [2] * prompt
        self.num_tokens = self.num_prompt_tokens = prompt
        self.num_cached_tokens = self.num_scheduled_tokens = 0
        self.last_token, self.temperature = 2, 0
        self.max_tokens, self.ignore_eos = limit, False
        self.is_prefill, self.block_table = True, []

    def __len__(self): return self.num_tokens
    @property
    def is_finished(self): return self.status is Status.FINISHED
    @property
    def num_completion_tokens(self): return self.num_tokens-self.num_prompt_tokens
    @property
    def completion_token_ids(self): return self.token_ids[self.num_prompt_tokens:]
    def append_token(self, token):
        self.token_ids.append(token)
        self.last_token = token
        self.num_tokens += 1


class Blocks:
    def __init__(self, count=32, size=4):
        self.free, self.size = deque(range(count)), size
    def can_allocate(self, seq):
        return 0 if len(self.free) >= (len(seq)+self.size-1)//self.size else -1
    def allocate(self, seq, cached):
        assert cached == 0
        assert self.reserve_tokens(seq, len(seq))
    def reserve_tokens(self, seq, length):
        extra = max(0, (length+self.size-1)//self.size-len(seq.block_table))
        if len(self.free) < extra: return False
        seq.block_table.extend(self.free.popleft() for _ in range(extra))
        return True
    def deallocate(self, seq):
        self.free.extend(seq.block_table)
        seq.block_table.clear()
        seq.num_cached_tokens = 0


class Runner:
    def __init__(self, eos_at=None):
        self.slots, self.trace, self.views = {}, [], []
        self.sample_count, self.eos_at, self.retired = {}, eos_at or {}, []
    def call(self, name, *args): return getattr(self, name)(*args)
    def launch_slot(self, views, prefill, slot):
        assert slot not in self.slots
        if prefill: assert not self.slots, "prefill must drain the pipeline"
        self.slots[slot] = dict(views=views, read=False)
        self.views.extend(views)
        self.trace.append(("launch", slot, prefill))
    def sample_slot(self, slot, live):
        self.trace.append(("sample", slot, tuple(live)))
        values = []
        for seq in self.slots[slot]["views"]:
            final_prompt = seq.num_cached_tokens+seq.num_scheduled_tokens == seq.num_tokens
            if seq.seq_id not in live or (seq.is_prefill and not final_prompt):
                values.append(-1)
                continue
            n = self.sample_count.get(seq.seq_id, 0)+1
            self.sample_count[seq.seq_id] = n
            values.append(1 if n == self.eos_at.get(seq.seq_id) else 10+n)
        def get():
            assert slot in self.slots
            self.trace.append(("read", slot))
            self.slots[slot]["read"] = True
            return values
        return SimpleNamespace(get_output=get)
    def release_slot(self, slot):
        assert self.slots[slot]["read"]
        self.trace.append(("release", slot))
        del self.slots[slot]
    def remove_request(self, ident):
        assert all(state["read"] or all(v.seq_id != ident for v in state["views"])
                   for state in self.slots.values())
        self.retired.append(ident)


def engine(requests, *, budget=64, slots=4, blocks=32, eos_at=None):
    scheduler = SimpleNamespace(waiting=deque(requests), running=deque(), in_flight=set(),
        max_num_seqs=slots, max_num_batched_tokens=budget, block_manager=Blocks(blocks))
    runner = Runner(eos_at)
    pipeline = core.DualSlotPipeline(scheduler, runner,
                                    SimpleNamespace(max_model_len=256, eos=1))
    return pipeline, runner, scheduler


def drain(pipeline, maximum=100):
    outputs = []
    for _ in range(maximum):
        result, _ = pipeline.step()
        outputs.extend(result)
        s = pipeline.scheduler
        if not (s.waiting or s.running or s.in_flight or pipeline.pending): return outputs
    raise AssertionError("pipeline did not drain")


class PipelineTests(unittest.TestCase):
    def test_launch_precedes_commit_and_sampling_follows_commit(self):
        p,r,s = engine([Request(limit=4)])
        p.step()  # serialized prefill
        p.step()  # prime slot 0
        r.trace.clear()
        p.step()
        kinds = [x[0] for x in r.trace]
        self.assertEqual(kinds, ["launch", "read", "release", "sample"])
        self.assertEqual(p.requests[0].inflight_refs, 1)

    def test_frontier_reserves_page_before_cpu_commits_and_views_do_not_drift(self):
        seq = Request(prompt=3, limit=5)
        p,r,s = engine([seq])
        p.step(); p.step(); p.step()
        decode = [v for v in r.views if not v.is_prefill]
        self.assertEqual([v.num_cached_tokens for v in decode], [3,4])
        self.assertEqual([len(v.block_table) for v in decode], [1,2])
        self.assertEqual(decode[0].num_scheduled_tokens, 1)
        self.assertEqual(seq.num_cached_tokens, 4)
        drain(p)
        self.assertEqual(seq.num_completion_tokens, 5)

    def test_eos_emits_early_but_releases_only_after_zombie_drains(self):
        seq = Request(limit=8)
        p,r,s = engine([seq], slots=1, eos_at={0:2})
        p.step(); p.step()
        out,_ = p.step()
        self.assertEqual(out, [(0,[11,1])])
        self.assertTrue(seq.is_finished)
        self.assertTrue(seq.block_table)
        self.assertEqual(r.retired, [])
        self.assertEqual(p.requests[0].inflight_refs, 1)
        p.step()
        self.assertEqual(seq.completion_token_ids, [11,1])
        self.assertEqual(r.retired, [0])
        self.assertEqual(p.stats["zombie_rows"], 1)
        self.assertEqual(r.sample_count[0], 2)

    def test_new_request_cannot_reuse_zombie_request_slot(self):
        a,b = Request(limit=8),Request(1,limit=1)
        p,r,s = engine([a], slots=1, eos_at={0:2})
        p.step(); p.step(); p.step()
        s.waiting.append(b)
        p.step()
        self.assertEqual([v.seq_id for v in r.views], [0,0,0])
        out,_ = p.step()
        self.assertEqual(out,[(1,[11])])
        self.assertEqual(r.retired,[0,1])

    def test_known_length_limit_does_not_launch_extra_forward(self):
        for limit in (1,2,5):
            with self.subTest(limit=limit):
                seq = Request(limit=limit)
                p,r,s = engine([seq])
                drain(p)
                self.assertEqual(seq.num_completion_tokens,limit)
                self.assertEqual(p.stats["decode_forwards"],limit-1)
                self.assertEqual(p.stats["zombie_rows"],0)

    def test_chunked_prefill_uses_frozen_counts_and_samples_only_final_chunk(self):
        seq = Request(prompt=7,limit=3)
        p,r,s = engine([seq],budget=2)
        result = drain(p)
        prefill = [v for v in r.views if v.is_prefill]
        self.assertEqual([v.num_scheduled_tokens for v in prefill],[2,2,2,1])
        self.assertEqual([v.num_cached_tokens for v in prefill],[0,2,4,6])
        self.assertEqual(result,[(0,[11,12,13])])

    def test_multiple_requests_with_different_completion_lengths(self):
        seqs = [Request(i,prompt=2+i,limit=1+i) for i in range(3)]
        p,r,s = engine(seqs,slots=2,budget=4)
        outputs = dict(drain(p))
        self.assertEqual({i:len(v) for i,v in outputs.items()}, {0:1,1:2,2:3})
        self.assertEqual(sorted(r.retired),[0,1,2])
        self.assertEqual(len(s.block_manager.free),32)

    def test_capacity_failure_does_not_free_live_state_and_poison_is_sticky(self):
        seq = Request(prompt=4,limit=3)
        p,r,s = engine([seq],blocks=1)
        p.step()
        with self.assertRaisesRegex(RuntimeError,'KV capacity'):
            p.step()
        self.assertTrue(seq.block_table)
        self.assertEqual(r.retired,[])
        with self.assertRaisesRegex(RuntimeError,'close this engine'):
            p.step()

    def test_blocked_new_prompt_does_not_stop_an_admitted_decode(self):
        a,b = Request(prompt=3,limit=5),Request(1,prompt=8,limit=1)
        p,r,s = engine([a],slots=2,blocks=2)
        p.step()
        s.waiting.append(b)
        outputs = dict(drain(p))
        self.assertEqual([len(outputs[0]),len(outputs[1])],[5,1])
        self.assertEqual(r.retired,[0,1])


if __name__ == '__main__':
    unittest.main()
