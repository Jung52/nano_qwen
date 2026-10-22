"""Control-plane regression tests; no CUDA work or model checkpoint required.

Run in the project's normal environment:
  PYTHONPATH=src python -m unittest discover -s tests -p test_async_decode_pipeline.py -v
"""
from collections import deque
from types import SimpleNamespace
import random
import unittest

from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.engine.sequence import Sequence, SequenceStatus
from nano_qwen.sampling_params import SamplingParams
from nano_qwen.scheduler import Scheduler


class FakeOutput:
    def __init__(self, tokens, events):
        self.tokens, self.events = tokens, events

    def get_output(self):
        self.events.append(("wait",))
        return self.tokens


class FakeRunner:
    """Models ordered GPU token feedback; consumes no CPU output to dispatch."""
    async_output = True

    def __init__(self, token_fn=None):
        self.events, self.gpu_tokens, self.positions, self.removed = [], {}, {}, []
        self.output = None
        self.token_fn = token_fn or (lambda seq, position: 7 + position % 11)

    def call(self, method, *args):
        if method == "execute_model":
            rows, prefill = args
            self.events.append(("execute", tuple(s.seq_id for s in rows)))
            tokens = []
            for seq in rows:
                position = seq.num_cached_tokens + seq.num_scheduled_tokens - 1
                if getattr(seq, "_async_decode_view", False):
                    assert not prefill and seq.seq_id in self.gpu_tokens
                    assert position == self.positions[seq.seq_id] + 1
                token = self.token_fn(seq, position)
                tokens.append(token)
                if not seq.is_prefill or position + 1 == len(seq):
                    self.gpu_tokens[seq.seq_id] = token
                    self.positions[seq.seq_id] = position
            self.output = FakeOutput(tokens, self.events)
        elif method == "sample_tokens":
            self.events.append(("sample",))
            return self.output
        elif method == "remove_request":
            self.events.append(("remove", args[0]))
            self.removed.append(args[0])
            self.gpu_tokens.pop(args[0], None)
            self.positions.pop(args[0], None)
        else:
            raise AssertionError(method)


def engine(enabled=True, blocks=32, budget=512, token_fn=None):
    config = SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=budget,
        eos=1, kvcache_block_size=256, enable_prefix_cache=False,
        num_kvcache_blocks=blocks, async_scheduling=enabled)
    e = LLMEngine.__new__(LLMEngine)
    e.config, e.scheduler, e.batch_queue = config, Scheduler(config), deque()
    e.max_concurrent_batches = 2
    e._coalesce_armed = False
    e.coalesce_stats = dict(coalesce_attempts=0, coalesce_successes=0,
        merged_batches=0, merged_requests=0, batch_size_before={}, batch_size_after={})
    e.model_runner = FakeRunner(token_fn)
    return e


def add(e, length=8, output=6, ignore_eos=True):
    seq = Sequence([3] * length, SamplingParams(temperature=1e-8,
        max_tokens=output, ignore_eos=ignore_eos))
    e.scheduler.add(seq)
    return seq


def finish(e, limit=4000):
    outputs = []
    for _ in range(limit):
        if e.is_finished():
            return outputs
        values, _ = e.step()
        outputs.extend(values)
        assert len(e.batch_queue) <= 2
        assert not (e.scheduler.in_flight & {s.seq_id for s in e.scheduler.running})
    raise AssertionError("engine did not drain")


class AsyncDecodeTests(unittest.TestCase):
    def test_two_dispatches_precede_first_result_wait(self):
        e = engine()
        seq = add(e)
        e.step()  # finish prefill, establish GPU token slot
        e.model_runner.events.clear()
        e.step()
        names = [v[0] for v in e.model_runner.events]
        self.assertEqual(names[:5], ["execute", "sample", "execute", "sample", "wait"])
        self.assertEqual(len(e.batch_queue), 1)
        self.assertEqual(seq.num_completion_tokens, 2)
        self.assertEqual(e.batch_queue[0][0][0].num_tokens, seq.num_tokens)
        self.assertEqual(e.scheduler.async_decode.pending[seq.seq_id], 1)
        finish(e)

    def test_off_remains_serial_between_decode_rounds(self):
        e = engine(False)
        add(e)
        e.step()
        e.model_runner.events.clear()
        e.step()
        self.assertEqual([v[0] for v in e.model_runner.events], ["execute", "sample", "wait"])
        self.assertFalse(e.batch_queue)
        finish(e)

    def test_boundary_reservation_preserves_queued_table_snapshot(self):
        e = engine()
        seq = add(e, length=254)
        e.step()
        e.step()  # inputs at 254 and 255, both use first block
        queued = e.batch_queue[0][0][0]
        self.assertEqual(len(queued.block_table), 1)
        e.step()  # reserve input at 256; second block is necessary
        self.assertEqual(len(seq.block_table), 2)
        self.assertEqual(len(queued.block_table), 1)
        finish(e)
        self.assertEqual(len(e.scheduler.block_manager.free_block_ids), 32)

    def test_eos_discards_lookahead_and_defers_resource_release(self):
        e = engine(token_fn=lambda s, p: 1 if p == 8 else 7)
        seq = add(e, length=8, ignore_eos=False)
        e.step()
        outputs, _ = e.step()  # first decode EOS, next decode is already queued
        self.assertEqual(outputs, [(seq.seq_id, [7, 1])])
        self.assertTrue(seq.is_finished)
        self.assertTrue(seq.block_table)
        self.assertIn(seq.seq_id, e.scheduler.in_flight)
        self.assertFalse(e.model_runner.removed)
        tail = finish(e)
        self.assertFalse(tail)
        self.assertEqual(seq.completion_token_ids, [7, 1])
        self.assertFalse(seq.block_table)
        self.assertEqual(e.model_runner.removed, [seq.seq_id])

    def test_max_tokens_has_no_extra_dispatch_or_duplicate_output(self):
        e = engine()
        a, b = add(e, output=2), add(e, output=5)
        outputs = finish(e)
        self.assertEqual(sorted(len(tokens) for _, tokens in outputs), [2, 5])
        executions = [v[1] for v in e.model_runner.events if v[0] == "execute"]
        self.assertEqual(sum(a.seq_id in ids for ids in executions), 2)
        self.assertEqual(sum(b.seq_id in ids for ids in executions), 5)
        self.assertEqual(len(e.model_runner.removed), 2)

    def test_new_arrival_drains_before_admission(self):
        e = engine()
        a = add(e, output=8)
        e.step()
        e.step()
        self.assertEqual(len(e.batch_queue), 1)
        b = add(e, length=12, output=3)
        e.model_runner.events.clear()
        e.step()
        self.assertEqual([v[0] for v in e.model_runner.events], ["wait"])
        self.assertIsNone(e.scheduler.async_decode)
        finish(e)
        self.assertEqual((a.num_completion_tokens, b.num_completion_tokens), (8, 3))

    def test_kv_pressure_drains_before_existing_preemption_policy(self):
        e = engine(blocks=1)
        seq = add(e, length=254, output=5)
        e.step()
        e.step()
        self.assertEqual(len(e.batch_queue), 1)
        e.model_runner.events.clear()
        e.step()  # next input would require an unavailable block
        self.assertEqual([v[0] for v in e.model_runner.events], ["wait"])
        self.assertFalse(e.batch_queue)
        self.assertIsNone(e.scheduler.async_decode)
        self.assertIn(seq, e.scheduler.running)
        self.assertFalse(e.scheduler.in_flight)
        # The whole request now exceeds this intentionally tiny KV pool;
        # the normal admission path reports that rather than freeing live KV.
        with self.assertRaisesRegex(RuntimeError, "KV cache exhausted"):
            e.step()

    def test_inflight_preemption_and_out_of_order_commit_rejected(self):
        e = engine()
        seq = add(e)
        e.step()
        self.assertTrue(e.scheduler.begin_async_decode())
        a, _ = e.scheduler.schedule()
        b, _ = e.scheduler.schedule()
        with self.assertRaisesRegex(ValueError, "flight"):
            e.scheduler.preempt(seq)
        with self.assertRaisesRegex(RuntimeError, "dispatch order"):
            e.scheduler.postprocess(b, [7], False)
        e.scheduler.postprocess(a, [7], False)
        e.scheduler.postprocess(b, [8], False)
        e.scheduler.end_async_decode()
        self.assertEqual(seq.completion_token_ids[-2:], [7, 8])

    def test_randomized_cohorts_match_serial_committed_tokens(self):
        rng = random.Random(42)
        for _ in range(40):
            specs = [(rng.choice([3, 127, 254, 255, 256, 257]), rng.randint(1, 12),
                      rng.choice([True, False])) for _ in range(rng.randint(1, 6))]
            results = []
            for enabled in (False, True):
                e = engine(enabled, blocks=64, budget=rng.choice([32, 256, 1024]))
                e.model_runner.token_fn = lambda s, p: 1 if p % 17 == 0 else 7 + p % 11
                seqs = [add(e, length=n, output=k, ignore_eos=ignore) for n, k, ignore in specs]
                outputs = finish(e)
                self.assertEqual(len(outputs), len(seqs))
                self.assertEqual(len(e.model_runner.removed), len(seqs))
                self.assertEqual(len(e.scheduler.block_manager.free_block_ids), 64)
                results.append([s.completion_token_ids for s in seqs])
            self.assertEqual(*results)


if __name__ == "__main__":
    unittest.main()
