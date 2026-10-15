"""Prevent double counting GPU annotations and mixing profile with latency."""

from types import SimpleNamespace

import pytest

import bench_one_batch as bench


@pytest.mark.parametrize("extra,kwargs", [((), {}), ((False,), {}), ((), {"is_verify": False})])
def test_path_audit_forwards_runner_optional_verify_argument(extra, kwargs):
    import torch
    from bench_fp8_pressure import path_audit
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return 17
    runner = SimpleNamespace(enforce_eager=True, run_model=original)
    ids, positions = torch.tensor([2]), torch.tensor([3])
    with path_audit(runner) as counts:
        assert runner.run_model(ids, positions, False, *extra, **kwargs) == 17
    assert calls == [((ids, positions, False, *extra), kwargs)]
    assert counts == {"eager/rows1": 1}
    assert runner.run_model is original


def event(name, category, start, duration):
    return dict(ph="X", name=name, cat=category, ts=start, dur=duration)


def test_phase_breakdown_counts_physical_events_once_and_reports_union():
    events = [event("nq/prefill/bs8", "user_annotation", 0, 100),
              event("nq/decode/window/bs8", "user_annotation", 100, 200),
              event("nq/prefill/bs8", "gpu_user_annotation", 10, 80),
              event("_per_channel_mm", "kernel", 10, 20),
              event("_quantize_rows", "kernel", 20, 20),
              event("gated_delta_rule_decode", "kernel", 130, 30),
              event("Memcpy DtoH", "gpu_memcpy", 170, 10)]
    result = bench.phase_breakdown(events)
    assert result["prefill"]["device_event_count"] == 2
    assert result["prefill"]["device_sum_ms"] == pytest.approx(.04)
    assert result["prefill"]["device_busy_ms"] == pytest.approx(.03)
    assert result["prefill"]["categories_ms"] == {"fp8_gemm": .02, "activation_quantization": .02}
    assert result["decode"]["device_sum_ms"] == pytest.approx(.04)
    assert result["decode"]["device_span_ms"] == pytest.approx(.05)
    assert result["decode"]["device_internal_gap_ms"] == pytest.approx(.01)
    assert result["unassigned_device_event_count"] == 0


def test_phase_breakdown_keeps_unassigned_work_visible():
    result = bench.phase_breakdown([event("unknown", "kernel", 100, 30)])
    assert result["unassigned_device_event_count"] == 1
    assert result["unassigned_device_sum_ms"] == pytest.approx(.03)


def test_large_batch_gather_is_counted_with_state_index_copy():
    events = [event("nq/decode/window/bs64", "user_annotation", 0, 100),
              event("void at::native::vectorized_gather_kernel<16, long>", "kernel", 10, 20),
              event("void at::native::index_elementwise_kernel<index_copy_kernel_impl>", "kernel", 35, 30)]
    result = bench.phase_breakdown(events)
    assert result["decode"]["categories_ms"] == {"state_index_copy": .05}


def test_cli_supports_separate_unprofiled_and_short_profile_runs(tmp_path):
    args = bench.parse_args(["--model", str(tmp_path), "--timing-rounds", "2",
                             "--profile-rounds", "0", "--workload-memory"])
    assert args.timing_rounds == 2
    assert args.profile_rounds == 0
    with pytest.raises(SystemExit):
        bench.parse_args(["--model", str(tmp_path), "--timing-rounds", "0", "--profile-rounds", "0"])
