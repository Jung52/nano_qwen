"""Synthetic traces cover attribution boundaries without requiring CUDA."""
import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest
import trace_tools as trace


def event(name, cat, start, duration, corr=None, pid=1, tid=1):
    return dict(name=name, cat=cat, ph="X", ts=start, dur=duration, pid=pid, tid=tid,
                args={} if corr is None else {"correlation": corr})


def batch(bid, launch, graph, *, head=None, sample=None, pid=1, tid=1, driver=False):
    head = head or (graph[1], graph[1]+1)
    sample = sample or (head[1], head[1]+1)
    def scope(stage, start, end):
        return event(f"nq/{stage}/b{bid:05d}", "user_annotation", start, end-start, pid=pid, tid=tid)
    def api(name, times, corr):
        return event(name, "cuda_driver" if driver else "cuda_runtime", times[0], times[1]-times[0], corr, pid, tid)
    return [scope("schedule", launch[0]-2, launch[0]-1),
            scope("execute", launch[0]-.5, launch[1]+3),
            scope("launch", launch[0], launch[1]),
            scope("lm_head", launch[1]+.1, launch[1]+1),
            scope("sample_call", launch[1]+1.1, launch[1]+2),
            scope("commit", sample[1]+1, sample[1]+2),
            api("cuGraphLaunch" if driver else "cudaGraphLaunch", launch, bid*10+1),
            api("head", (launch[1]+.2, launch[1]+.3), bid*10+2),
            api("sample", (launch[1]+1.2, launch[1]+1.3), bid*10+3),
            event("graph", "kernel", graph[0], graph[1]-graph[0], bid*10+1),
            event("head", "kernel", head[0], head[1]-head[0], bid*10+2),
            event("sample", "kernel", sample[0], sample[1]-sample[0], bid*10+3)]


def report(events):
    return trace.summarize_frames(trace.analyze({"traceEvents": events}))


def test_launch_before_gpu_start_is_prequeued_not_overlap():
    result = report(batch(0, (1, 2), (20, 30)) + batch(1, (5, 6), (35, 45)))
    assert result["pairs"][0]["verdict"] == "PREQUEUED"
    assert result["pairs"][0]["launch_overlap_physical_kernel_us"] == 0


def test_two_dispatches_in_one_consumer_scope_remain_separate():
    events = [event("nqc/step/1", "user_annotation", 0, 60)]
    events += batch(0, (1, 2), (10, 20)) + batch(1, (25, 26), (35, 45))
    frames = trace.analyze({"traceEvents": events})
    assert frames[0]["graph"] == [10, 20]
    assert frames[1]["graph"] == [35, 45]
    assert report(events)["pairs"][0]["verdict"] == "AFTER_FORWARD"


def test_actual_cpu_launch_gpu_kernel_intersection_is_overlap():
    result = report(batch(0, (1, 2), (10, 30)) + batch(1, (20, 21), (35, 45)))
    assert result["pairs"][0]["verdict"] == "OVERLAP"
    assert result["pairs"][0]["launch_overlap_physical_kernel_us"] == 1
    assert result["launch_next_overlaps_physical_kernel"] == 1


def test_internal_graph_gap_is_not_physical_kernel_overlap():
    events = batch(0, (1, 2), (10, 30)) + batch(1, (20, 21), (35, 45))
    for e in events:
        if e["cat"] == "kernel" and e["args"].get("correlation") == 1:
            e["dur"] = 5
    events.append(event("graph-tail", "kernel", 25, 5, 1))
    summary = report(events)
    assert summary["pairs"][0]["verdict"] == "WITHIN_FORWARD_SPAN"
    assert summary["pairs"][0]["launch_overlap_physical_kernel_us"] == 0
    assert summary["launch_next_overlaps_forward"] == 1
    assert summary["launch_next_overlaps_physical_kernel"] == 0


def test_one_kernel_tiny_graph_is_supported_without_threshold():
    frames = trace.analyze({"traceEvents": batch(0, (1, 2), (10, 15))})
    assert len(frames) == 1 and frames[0]["graph_kernel_count"] == 1


def test_unlabelled_micrographs_do_not_create_new_dispatches():
    events = batch(0, (1, 2), (10, 15))
    events += [event("cudaGraphLaunch", "cuda_runtime", 3, .1, 99),
               event("micro", "kernel", 30, 1, 99)]
    frames = trace.analyze({"traceEvents": events})
    assert len(frames) == 1 and frames[0]["graph"] == [10, 15]


def test_micrograph_inside_head_remains_head_work():
    events = batch(0, (1, 2), (10, 15))
    events += [event("cudaGraphLaunch", "cuda_runtime", 2.4, .1, 99),
               event("head-micro", "kernel", 16, .5, 99)]
    assert trace.analyze({"traceEvents": events})[0]["lm_head"] == [15, 16.5]


def test_driver_graph_api_supported():
    assert len(trace.analyze({"traceEvents": batch(0, (1, 2), (10, 15), driver=True)})) == 1


def test_nested_runtime_driver_graph_calls_do_not_duplicate_replay():
    events = batch(0, (1, 3), (10, 15))
    events += [event("cuGraphLaunch", "cuda_driver", 1.2, .8, 91),
               event("graph-driver", "kernel", 10, 5, 91)]
    frames = trace.analyze({"traceEvents": events})
    assert len(frames) == 1 and frames[0]["launch"] == [1, 3]


def test_same_timestamp_on_other_thread_does_not_inherit_scope():
    events = batch(0, (1, 2), (10, 15))
    for e in events:
        if e["cat"] == "cuda_runtime":
            e["tid"] = 99
    assert trace.analyze({"traceEvents": events}) == []


def test_reused_correlation_across_dispatches_is_unknown():
    events = batch(0, (1, 2), (20, 25)) + batch(1, (5, 6), (30, 35))
    for e in events:
        if e["args"].get("correlation") == 11:
            e["args"]["correlation"] = 1
    assert trace.analyze({"traceEvents": events}) == []


def test_reused_correlation_and_batch_ids_across_processes_are_unknown():
    events = batch(0, (1, 2), (20, 25), pid=1) + batch(0, (5, 6), (30, 35), pid=2)
    assert trace.analyze({"traceEvents": events}) == []


def test_missing_kernel_correlations_are_not_inferred_from_host_scope():
    events = batch(0, (1, 2), (10, 15))
    for e in events:
        if e["cat"] == "kernel":
            e["args"].clear()
    assert trace.analyze({"traceEvents": events}) == []


def test_copy_and_memset_count_as_device_activity():
    data = {"traceEvents": batch(0, (1, 2), (10, 15), head=(18, 19), sample=(20, 21))}
    data["traceEvents"] += [event("copy", "gpu_memcpy", 15, 3), event("zero", "gpu_memset", 19, 1)]
    result = trace.device_activity(data, trace.analyze(data))
    assert result["device_idle_us"] == 0


def test_union_does_not_double_count_overlapping_device_events():
    assert trace.overlap((1, 7), [(0, 5), (3, 8)]) == 6
    assert trace.overlap((0, 1), [(2, 3)]) == 0


def test_export_unknown_without_dispatch_markers_is_explicit(tmp_path):
    path = tmp_path/"legacy.json"
    path.write_text(json.dumps({"traceEvents": [event("nqc/step/0", "user_annotation", 0, 100)]}))
    summary, result = trace.report_trace(path)
    assert summary["status"] == "unknown" and summary["pairs"] == []
    assert result.is_file() and (tmp_path/"legacy.timeline.svg").is_file()


def test_gzip_offline_analysis_and_cli_need_no_torch(tmp_path):
    path = tmp_path/"turn001.trace.json.gz"
    with gzip.open(path,"wt") as handle:
        json.dump({"traceEvents": batch(0, (1, 2), (10, 15))}, handle)
    script = Path(__file__).resolve().parents[1]/"benchmarks/bench_one_batch_trace.py"
    result = subprocess.run([sys.executable, str(script), "--analyze", str(path)],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["status"] == "ok"
    assert (tmp_path/"turn001.summary.json").is_file()


def test_bounded_export_preserves_every_selected_native_graph_kernel():
    events = []
    for i in range(7):
        events += batch(i, (i*30+1, i*30+2), (i*30+10, i*30+15))
    data = {"traceEvents":events}
    encoded, selected, window = trace.bounded_trace(data, trace.analyze(data), "async", 5)
    exported = json.loads(encoded)
    assert window["complete_batch_ids"] == [1,2,3,4,5]
    assert len(selected) == 5
    for frame in selected:
        corr=frame["graph_correlations"][0]
        assert [e for e in events if e["cat"]=="kernel" and e["args"].get("correlation")==corr] == [
            e for e in exported["traceEvents"] if e.get("cat")=="kernel" and e["args"].get("correlation")==corr]


def test_sample_order_is_reported_as_gpu_dependency():
    result = report(batch(0,(1,2),(10,15),sample=(25,26)) + batch(1,(20,21),(22,23)))
    assert result["sample_before_next_graph"] == 0
