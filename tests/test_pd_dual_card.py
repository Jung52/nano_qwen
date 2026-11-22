"""CLI and pure helpers of the dual-card PD driver (no GPU needed)."""
import json
import math

import pytest

import bench_pd_dual_card as driver


def parse_args(*extra):
    return driver.driver_arguments([
        "--config", "pd-serial", "--model", "/unused/model",
        "--output", "/unused/out.json", *extra,
    ])


def test_serial_forces_queue_depth_one_in_section_config():
    args = parse_args()
    assert args.config == "pd-serial"
    assert args.queue_depth == 2  # CLI default; the section forces 1


def test_pipeline_accepts_bounded_queue_and_shm_budget():
    args = parse_args("--config", "pd-pipeline", "--queue-depth", "3",
                      "--shm-budget-gb", "0.5")
    assert args.queue_depth == 3
    assert args.shm_budget_gb == 0.5


def test_single_concurrency_limited_to_two():
    with pytest.raises(SystemExit):
        parse_args("--config", "single", "--concurrency", "3")


def test_invalid_lens_and_budget_rejected():
    # Duplicate lens are allowed for mixed workloads; negatives are not.
    args = parse_args("--lens", "257", "257", "257", "257", "2048",
                      "--outs", "128", "128", "128", "128", "32")
    assert args.lens == [257, 257, 257, 257, 2048]
    assert args.outs == [128, 128, 128, 128, 32]
    with pytest.raises(SystemExit):
        parse_args("--lens", "0")
    with pytest.raises(SystemExit):
        parse_args("--outs", "128")
    with pytest.raises(SystemExit):
        parse_args("--shm-budget-gb", "0")


def test_kernel_interval_parsing_merges_and_filters(tmp_path):
    trace = {"traceEvents": [
        {"ph": "X", "cat": "kernel", "ts": 100.0, "dur": 50.0},
        {"ph": "X", "cat": "kernel", "ts": 120.0, "dur": 30.0},  # overlaps -> merge
        {"ph": "X", "cat": "kernel", "ts": 300.0, "dur": 10.0},
        {"ph": "X", "cat": "cpu_op", "ts": 0.0, "dur": 999.0},    # filtered out
        {"ph": "X", "cat": "gpu_memcpy", "ts": 400.0, "dur": 20.0},
        {"ph": "X", "cat": "kernel", "ts": 500.0, "dur": 0.0},    # zero dur filtered
    ]}
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace))
    assert driver.trace_kernel_intervals(path) == [
        [100.0, 150.0], [300.0, 310.0], [400.0, 420.0]]


def test_trace_overlap_intersection():
    left = [[0.0, 100.0], [200.0, 300.0]]
    right = [[50.0, 250.0], [280.0, 400.0]]
    assert driver.overlap_us(left, right) == 100.0 + 20.0


def test_trace_overlap_missing_inputs():
    assert driver.overlap_us(None, [[0.0, 1.0]]) is None
    assert driver.trace_kernel_intervals("/nonexistent/trace.json") is None


def test_stats_and_p95():
    assert driver.p95([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == pytest.approx(9.55)
    assert driver.stats([2.0, 4.0])["median"] == 3.0
    assert driver.stats([])["median"] is None
    assert driver.p95([]) is None
