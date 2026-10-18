# ruff: noqa: F811
"""Stage accounting must not count annotations or inactive graph banks as work."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import profile_mtp_reuse as profile


def event(name, cat, ts, dur, **kwargs):
    return dict(name=name, cat=cat, ph="X", ts=ts, dur=dur, pid=0, tid=7, **kwargs)


def test_trace_disjoint_heads_overlap_and_physical_union():
    assert hasattr(profile, "summarize_trace"), "B2/B4 attribution is not implemented"
    signature = dict(grid=[10, 1, 1], block=[128, 1, 1])
    events = [event("mtp.target", "gpu_user_annotation", 0, 100),
              event("mtp.snapshot", "gpu_user_annotation", 20, 5),
              event("mtp.target_head", "gpu_user_annotation", 70, 30),
              event("mtp.draft", "gpu_user_annotation", 110, 40),
              event("mtp.decision_wait", "user_annotation", 15, 30),
              event("dense", "kernel", 10, 20),
              event("copy_state", "kernel", 20, 5),
              event("vocab", "kernel", 75, 10, args=signature),
              event("vocab", "kernel", 115, 10, args=signature),
              event("draft_dense", "kernel", 130, 5),
              event("D2H", "gpu_memcpy", 165, 5)]
    result = profile.summarize_trace(events)
    assert result["kernel_groups_us"] == {"target": 20, "snapshot": 5, "target_head": 10,
                                           "draft_head": 10, "draft": 5}
    assert result["kernel_sum_us"] == 50
    assert result["device_busy_union_us"] == 50
    assert result["device_internal_gap_us"] == 110
    assert result["decision_wait_cpu_us"] == 30
    assert result["decision_wait_gpu_overlap_us"] == 15
    assert result["decision_wait_without_gpu_us"] == 15


def test_unassigned_device_work_stays_visible():
    assert hasattr(profile, "summarize_trace"), "Unassigned GPU accounting is missing"
    result = profile.summarize_trace([event("unknown", "kernel", 1, 7)])
    assert result["kernel_groups_us"] == {"other": 7}
    assert result["device_internal_gap_us"] == 0


def test_only_replayed_exact_size_events_are_read():
    assert hasattr(profile, "GraphEventBank"), "Exact graph-bank event selection is missing"
    bank = profile.GraphEventBank()
    start = SimpleNamespace(elapsed_time=lambda end: end.value)
    bank.add(("verify", 2), "snapshot", start, SimpleNamespace(value=.2))
    bank.add(("verify", 4), "snapshot", start, SimpleNamespace(value=99.))
    bank.add(("draft", 1), "draft_head", start, SimpleNamespace(value=.3))
    assert bank.collect([("verify", 2), ("draft", 1)]) == {"snapshot": .2, "draft_head": .3}
    # One CUDA event pair stores the most recent replay. Reusing it twice in
    # one step must fail visibly instead of silently charging the last replay twice.
    with pytest.raises(ValueError, match="replay"):
        bank.collect([("verify", 2), ("verify", 2)])


def test_host_pass_adds_no_cuda_events_or_synchronization(monkeypatch):
    import torch
    def forbidden(*args, **kwargs):
        raise AssertionError("host diagnostics must not add GPU events or synchronization")
    monkeypatch.setattr(torch.cuda, "Event", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    recorder = profile.PhaseRecorder(SimpleNamespace(), kind="host")
    recorder.begin()
    with recorder.scope("decision_wait"):
        pass
    row = recorder.finish()
    assert row["host_ms"]["decision_wait"] >= 0
    assert row["host_calls"]["decision_wait"] == 1


def test_event_instrumentation_preserves_prefix_states_and_covers_rejection(engine):
    # The existing real hybrid-engine fixture covers both FP8s and eager/Graph.
    import torch
    from test_mtp_batch2 import compare_modes
    assert hasattr(profile, "PhaseRecorder"), "Graph-aware stage instrumentation is missing"
    recorder = profile.PhaseRecorder(engine.model_runner, kind="events")
    rows = []
    step = engine.step

    def measured_step():
        recorder.begin()
        result = step()
        rows.append(recorder.finish())
        return result

    with torch.inference_mode(), recorder.install():
        with patch.object(engine, "step", measured_step):
            compare_modes(engine, [(True, False), (False, True)])
    groups = {label for row in rows for label in row["gpu_ms"]}
    assert {"target_body", "target_head", "snapshot", "rollback", "draft_body", "draft_head", "cache_only"}.issubset(groups)
    assert all(value >= 0 for row in rows for value in row["gpu_ms"].values())


# These fixtures allocate a small real model, not production-size weights.
from test_mtp_batch2 import engine  # noqa: E402,F401
from test_mtp_verify_reuse import tiny_checkpoint, fp8_checkpoint  # noqa: E402,F401
