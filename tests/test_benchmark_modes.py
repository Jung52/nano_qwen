"""Benchmark policy must never infer lookahead from queue size alone."""
from types import SimpleNamespace

import pytest
from runtime_modes import MODES, PERF_MODES, ModeConfig, PROTOCOL
import bench_one_batch


@pytest.mark.parametrize("mode", MODES+PERF_MODES, ids=lambda m:m.name)
def test_mode_policy_is_explicit_and_verified(mode):
    engine=SimpleNamespace(config=SimpleNamespace(async_scheduling=mode.async_scheduling),
                           model_runner=SimpleNamespace())
    mode.apply(engine)
    assert engine.max_concurrent_batches == mode.queue_depth
    assert engine.model_runner.async_output == mode.async_output
    assert engine.model_runner.use_prefill_cudagraph == mode.prefill_cudagraph
    assert mode.async_scheduling == (mode.name not in ("BASELINE","P0"))
    assert PROTOCOL == "explicit-async-v2"


def test_queue_two_does_not_implicitly_enable_lookahead():
    policy=ModeConfig("queue-only",False,2,True,False)
    assert policy.async_scheduling is False


@pytest.mark.parametrize("queue,output",[(1,True),(2,False)])
def test_impossible_lookahead_policy_is_rejected(queue,output):
    with pytest.raises(ValueError):
        ModeConfig("invalid",False,queue,output,False,True)


def test_silent_serial_fallback_is_rejected():
    engine=SimpleNamespace(config=SimpleNamespace(async_scheduling=False),
                           model_runner=SimpleNamespace())
    with pytest.raises(RuntimeError,match="async_scheduling"):
        PERF_MODES[2].apply(engine)


def test_one_batch_flag_is_explicit_and_incompatible_flags_fail(tmp_path):
    args=bench_one_batch.parse_args(["--model",str(tmp_path)])
    assert args.async_scheduling is False
    args=bench_one_batch.parse_args(["--model",str(tmp_path),"--async-scheduling"])
    assert args.async_scheduling is True
    with pytest.raises(SystemExit):
        bench_one_batch.parse_args(["--model",str(tmp_path),"--async-scheduling","--sync-output"])


def test_engine_factory_passes_scheduling_policy_at_construction(monkeypatch):
    pytest.importorskip("torch")
    import engine_bench_utils as utils
    assert utils.MODE_NAMES == {mode.name: mode for mode in MODES}
    assert utils.PERF_MODE_NAMES == {mode.name: mode for mode in PERF_MODES}
    def factory(model, **kwargs):
        return SimpleNamespace(config=SimpleNamespace(async_scheduling=kwargs.get("async_scheduling")),
                               model_runner=SimpleNamespace(), kwargs=kwargs)
    monkeypatch.setattr(utils,"LLMEngine",factory)
    for mode in PERF_MODES:
        engine=utils.make_engine("checkpoint",mode)
        assert engine.kwargs["async_scheduling"] == mode.async_scheduling
        assert engine.max_concurrent_batches == mode.queue_depth
