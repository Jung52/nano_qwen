"""CPU control-flow gates for decode latency under continuous prompt arrivals."""
from collections import deque
from types import SimpleNamespace

import pytest

from nano_qwen.engine.execution import RequestOutput
from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.engine.sequence import Sequence
from nano_qwen.sampling_params import SamplingParams
from nano_qwen.scheduler import Scheduler
from nano_qwen.utils.context import BatchDescriptor


def scheduler(mtp=False):
    return Scheduler(SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=128,
        eos=1, kvcache_block_size=256, enable_prefix_cache=False,
        enable_mtp=mtp, num_kvcache_blocks=128))


def consume(owner, rows, prefill):
    results = [RequestOutput(
        () if s.is_prefill and s.num_cached_tokens+s.num_scheduled_tokens < s.num_tokens else (7,),
        0, None, s.num_scheduled_tokens) for s in rows]
    owner.postprocess(rows, results, prefill)


def ready_decode(owner):
    seq = Sequence([2], SamplingParams(temperature=1e-6, max_tokens=20))
    owner.add(seq)
    rows, prefill = owner.schedule()
    consume(owner, rows, prefill)
    return seq


def test_ready_decode_gets_budget_before_long_prefill():
    owner = scheduler()
    a = ready_decode(owner)
    for _ in range(2):
        owner.add(Sequence([2]*1024))
    rows, prefill = owner.schedule()
    assert rows[0] is a and a.num_scheduled_tokens == 1
    assert sum(s.num_scheduled_tokens for s in rows) == 128
    assert prefill and rows[1].num_scheduled_tokens == 127


def test_verify_remains_separate_and_cannot_be_starved_by_prefill():
    owner = scheduler(mtp=True)
    a = ready_decode(owner)
    a.draft_token = 9
    owner.add(Sequence([2]*1024))
    rows, prefill = owner.schedule()
    assert rows == [a] and prefill and a.is_speculative


def test_engine_continuous_arrivals_bound_decode_wait():
    owner = scheduler()
    a = ready_decode(owner)
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler, engine.batch_queue = owner, deque()
    engine.max_concurrent_batches = 2
    engine._coalesce_armed = False
    engine.coalesce_stats = dict(coalesce_attempts=0, coalesce_successes=0,
        merged_batches=0, merged_requests=0, batch_size_before={}, batch_size_after={})
    trace, pending = [], None

    def call(method, *args):
        nonlocal pending
        if method == 'execute_model':
            rows, prefill = args
            trace.append(tuple(s.seq_id for s in rows))
            results = [RequestOutput(
                () if s.is_prefill and s.num_cached_tokens+s.num_scheduled_tokens < s.num_tokens else (7,),
                0, None, s.num_scheduled_tokens) for s in rows]
            pending = SimpleNamespace(get_results=lambda: results)
        elif method == 'sample_tokens':
            return pending
    engine.model_runner = SimpleNamespace(call=call)
    for _ in range(8):
        owner.add(Sequence([2]*1024))
        engine.step()
    appearances = [i for i, ids in enumerate(trace) if a.seq_id in ids]
    assert appearances[0] == 0
    assert len(appearances) >= 4
    assert all(b-a <= 2 for a, b in zip(appearances, appearances[1:]))


def test_descriptor_rejects_mixed_verify_and_prefill():
    a, b = Sequence([2]), Sequence([3])
    a.num_scheduled_tokens, a.is_speculative, a.draft_token = 2, True, 9
    b.num_scheduled_tokens = 1
    with pytest.raises(ValueError, match='verify|speculative'):
        BatchDescriptor.for_requests([a, b])


def test_mtp_isolates_decode_but_batches_fresh_prefills():
    owner = scheduler(mtp=True)
    seqs = [Sequence([2]*16) for _ in range(4)]
    for seq in seqs:
        owner.add(seq)
    rows, prefill = owner.schedule()
    assert rows == seqs and prefill
