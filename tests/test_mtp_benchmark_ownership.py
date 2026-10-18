"""Borrowed-engine validation and cleanup, with real scheduler/slot ownership."""

from collections import deque
from types import SimpleNamespace

import pytest
import torch

import bench_qwen3_5_mtp_production as bench
from nano_qwen.config import Config
from nano_qwen.engine.model_runner import InputBatch, ModelRunner
from nano_qwen.engine.sequence import Sequence
from nano_qwen.engine.execution import RequestOutput
from nano_qwen.scheduler import Scheduler
from test_mtp_weights import checkpoint_pair


@pytest.fixture
def engine(tmp_path, monkeypatch):
    target, source = checkpoint_pair(tmp_path)
    config = Config(
        str(target),
        enable_mtp=True,
        max_num_seqs=1,
        mtp_verify_mode="reuse",
        mtp_model=str(source),
        num_kvcache_blocks=8,
    )
    runner = ModelRunner.__new__(ModelRunner)
    runner.input_batch = InputBatch(1)
    runner._mtp_last_hidden, runner._mtp_cache_lens = {}, {}
    runner._pending = None
    runner.use_prefill_cudagraph = False
    runner.mtp_stats = dict(
        draft_tokens=9, accepted_tokens=7, rejected_tokens=2, verify_steps=9
    )
    value = SimpleNamespace(
        config=config,
        model_runner=runner,
        scheduler=Scheduler(config),
        batch_queue=deque(),
        closed=False,
        fail=False,
    )

    def step():
        seqs, prefill = value.scheduler.schedule()
        runner.input_batch.update(seqs)
        if value.fail:
            runner._pending = (None, None, seqs, prefill)
            raise RuntimeError("injected execution failure")
        count = sum(s.num_scheduled_tokens for s in seqs) if prefill else 0
        if not prefill:
            runner.mtp_stats["accepted_tokens"] += 1
            runner.mtp_stats["verify_steps"] += 1
        value.scheduler.postprocess(seqs, [RequestOutput((2,), 0, None, s.num_scheduled_tokens) for s in seqs], prefill)
        return [], count

    value.step = step
    value.exit = lambda: setattr(value, "closed", True)
    # This suite tests CPU ownership; numerical execution uses CUDA integration tests.
    for name in ("reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    return value


def run(engine, **kwargs):
    return bench.run_mode(
        engine.config.model,
        True,
        [2, 3],
        3,
        60.0,
        False,
        mtp_model=kwargs.pop("mtp_model", engine.config.mtp_model),
        mtp_verify_mode="reuse",
        engine=engine,
        **kwargs,
    )


@pytest.mark.parametrize(
    "where", ["waiting", "running", "in_flight", "batch_queue", "pending", "slot"]
)
def test_borrowed_busy_engine_is_rejected_without_consuming_other_requests(
    engine, where
):
    other = Sequence([3, 4])
    if where in ("waiting", "running"):
        getattr(engine.scheduler, where).append(other)
    elif where == "in_flight":
        engine.scheduler.in_flight.add(other.seq_id)
    elif where == "batch_queue":
        engine.batch_queue.append(([other], True, 2, None))
    elif where == "pending":
        engine.model_runner._pending = (None, None, [other], True)
    else:
        engine.model_runner.input_batch.update([other])
    before = (
        list(engine.scheduler.waiting),
        list(engine.scheduler.running),
        set(engine.scheduler.in_flight),
        list(engine.batch_queue),
        dict(engine.model_runner.input_batch.seq_id_to_slot),
        engine.model_runner._pending,
    )
    with pytest.raises(ValueError, match="idle"):
        run(engine)
    after = (
        list(engine.scheduler.waiting),
        list(engine.scheduler.running),
        set(engine.scheduler.in_flight),
        list(engine.batch_queue),
        dict(engine.model_runner.input_batch.seq_id_to_slot),
        engine.model_runner._pending,
    )
    assert before == after and not engine.closed


def test_borrowed_engine_cannot_silently_use_another_mtp_source(engine):
    with pytest.raises(ValueError, match="MTP.*(source|checkpoint)"):
        run(engine, mtp_model="another-checkpoint")
    assert engine.scheduler.is_finished() and not engine.closed


def test_borrowed_engine_must_fit_the_requested_context(engine):
    engine.config.max_model_len = 4
    with pytest.raises(ValueError, match="(length|capacity)"):
        run(engine)
    assert engine.scheduler.is_finished() and not engine.closed


def test_borrowed_engine_returns_per_run_stats_and_survives_failure(engine):
    metrics, tokens = run(engine)
    assert tokens == [2, 2, 2]
    assert metrics["accepted_tokens"] == metrics["verify_steps"] == 2
    assert engine.model_runner.mtp_stats["accepted_tokens"] == 9
    engine.fail = True
    with pytest.raises(RuntimeError, match="injected execution failure"):
        run(engine)
    assert engine.scheduler.is_finished() and not engine.batch_queue
    assert not engine.model_runner.input_batch.seq_id_to_slot
    assert engine.model_runner._pending is None
    assert len(engine.scheduler.block_manager.free_block_ids) == 8
    assert not engine.closed
    engine.fail = False
    second, tokens = run(engine)
    assert tokens == [2, 2, 2]
    assert second["accepted_tokens"] == second["verify_steps"] == 2
