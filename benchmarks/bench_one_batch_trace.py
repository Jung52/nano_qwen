#!/usr/bin/env python3
"""Three real nano_qwen scheduling paths, short CPU/CUDA decode traces.

Run from nano_qwen (PyTorch + CUDA + model dependencies required):
  CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_one_batch_trace.py \
      --model /models/Qwen3.5-4B --output-dir logs/one_batch_trace

no_async/async use async_scheduling=False/True on the current source tree
(--same-source semantics; no git-ref fallback). No fetch, fake GPU delay,
replay repetition, or per-step synchronization is added. Only the diagnostic
KV capacity and annotations are changed; engine.step drives every scheduling
decision.

Each directory contains trace.json (<= --max-trace-mb), summary.json,
metadata.json, check_outputs.json, consistency.json and run.log. Checks run
without profiler or per-step synchronization. A serial reference on the same
source tree is used to distinguish scheduling errors from source differences.
The trace preserves native operators, CUDA graph
kernels and correlation IDs. Size reduction selects a shorter continuous
window, never a sample of individual kernels. At least four complete
decode batches are required. Pure-Python analysis is importable without torch.
"""
from __future__ import annotations

import argparse
import ast
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from contextlib import contextmanager, ExitStack
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

MODES = ("no_async", "async")
MARKER = re.compile(r"^nq/([a-z_]+)/b(\d+)(?:/slot([01]))?$")
MIB = 1024 * 1024


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE).strip()


def source_info(root, ref=None):
    info = {"root": str(root), "ref": ref, "files_sha256": {}}
    try:
        info.update(commit=git(root, "rev-parse", "HEAD"), dirty=bool(git(root, "status", "--porcelain")))
    except (OSError, subprocess.CalledProcessError):
        info.update(commit=None, dirty=None)
    for name in ("config/config.py", "engine/llm_engine.py", "engine/model_runner.py",
                 "engine/cuda_graph.py", "engine/pipeline.py", "scheduler/async_decode.py"):
        path = root / "src/nano_qwen" / name
        if path.is_file():
            info["files_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return info


def config_fields(root):
    path = root / "src/nano_qwen/config/config.py"
    if not path.is_file():
        raise RuntimeError(f"Not a nano_qwen source tree: {root}")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Config")
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}


def find_repo(explicit=None):
    candidates = [Path(explicit)] if explicit else [Path.cwd(), Path(__file__).resolve().parent.parent,
                                                  Path(__file__).resolve().parent]
    for path in candidates:
        if (path / "src/nano_qwen/engine/llm_engine.py").is_file():
            return path.resolve()
    raise RuntimeError("Run inside nano_qwen, or supply --repo-root /path/to/nano_qwen")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--repo-root")
    p.add_argument("--batch-size", type=int, default=64, help="Real decode batch, not queue depth; try 128 if desired")
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--trace-steps", type=int, default=6, help="Requested complete decode batches in saved trace")
    p.add_argument("--max-trace-mb", type=float, default=20.0, help="Hard cap on each uncompressed trace.json, MiB")
    p.add_argument("--settle-steps", type=int, default=4)
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="logs/one_batch_trace")
    p.add_argument("--mode", choices=("all",) + MODES, default="all")
    p.add_argument("--check-steps", type=int, default=64, help="Greedy correctness output length; >=8")
    p.add_argument("--check-seeds", nargs="+", type=int, default=[42, 43])
    p.add_argument("--check-repeats", type=int, default=2, help="Repeated workloads in one engine, >=2")
    p.add_argument("--checks-only", action="store_true", help="Run correctness suite without collecting a trace")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--worker", choices=MODES, help=argparse.SUPPRESS)
    p.add_argument("--source-root", help=argparse.SUPPRESS)
    p.add_argument("--source-json", help=argparse.SUPPRESS)
    p.add_argument("--worker-output", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.batch_size not in [1, 2, 4, 8] + list(range(16, 513, 16)):
        p.error("Use an exact decode graph bucket: 1/2/4/8, or a multiple of 16 up to 512")
    if a.trace_steps < 4 or a.max_trace_mb <= 0:
        p.error("--trace-steps must be >=4; --max-trace-mb must be positive")
    if min(a.input_len, a.settle_steps, a.warmup_rounds, a.timeout) < 1:
        p.error("Input, settling, warmup and timeout must be positive")
    if not 0 < a.gpu_memory_utilization < 1:
        p.error("--gpu-memory-utilization must be between 0 and 1")
    if a.check_steps < 8 or a.check_repeats < 2:
        p.error("--check-steps must be >=8 and --check-repeats must be >=2")
    if len(set(a.check_seeds)) != len(a.check_seeds):
        p.error("--check-seeds must not contain duplicates")
    a.model = str(Path(a.model).resolve())
    a.output_dir = str(Path(a.output_dir).resolve())
    return a


@contextmanager
def workload_kv_pool(ModelRunner, torch, batch_size, max_model_len):
    """Same capacity policy as benchmarks/bench_fp8_pressure.py, operators unchanged."""
    original = ModelRunner.allocate_kv_cache
    def allocate(runner):
        config, hf = runner.config, runner.config.hf_config
        layers = [m for m in runner.model.modules() if hasattr(m, "k_cache") and hasattr(m, "v_cache")]
        if not layers:
            raise RuntimeError("No KV-bearing attention layers")
        blocks = batch_size * (math.ceil(max_model_len / runner.block_size) + 1)
        heads = hf.num_key_value_heads // runner.world_size
        dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        block_bytes = 2 * len(layers) * runner.block_size * heads * dim * hf.dtype.itemsize
        free, total = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats()
        transient = stats["allocated_bytes.all.peak"] - stats["allocated_bytes.all.current"]
        budget = int(total * config.gpu_memory_utilization) - (total - free) - transient
        if blocks * block_bytes > budget:
            raise RuntimeError("Workload KV pool exceeds available budget; reduce batch/input or use a smaller checkpoint")
        config.num_kvcache_blocks = blocks
        runner.kv_cache = torch.empty((2, len(layers), blocks, runner.block_size, heads, dim),
                                      dtype=hf.dtype, device="cuda")
        for i, layer in enumerate(layers):
            layer.k_cache, layer.v_cache = runner.kv_cache[0, i], runner.kv_cache[1, i]
    ModelRunner.allocate_kv_cache = allocate
    try:
        yield
    finally:
        ModelRunner.allocate_kv_cache = original


def audit_graphs(engine, mode, batch_size):
    r, config = engine.model_runner, engine.config
    graphs_by_bs = r.cuda_graphs.decode_graphs
    if batch_size not in graphs_by_bs:
        raise RuntimeError(f"Expected a decode graph with exact BS={batch_size}")
    graphs = [graphs_by_bs[batch_size]]
    if bool(getattr(config, "async_scheduling", False)) != (mode == "async"):
        raise RuntimeError("Actual async_scheduling differs from requested mode")
    return graphs, {"graph_banks": 1, "active_graphs_at_batch": 1,
                    "captured_batch_sizes": sorted(graphs_by_bs),
                    "async_scheduling": bool(getattr(config, "async_scheduling", False)),
                    "queue_depth": engine.max_concurrent_batches, "async_output": r.async_output,
                    "static_buffer_ptrs": {k: int(v.data_ptr()) for k, v in r.cuda_graphs.decode_graph_vars.items()
                                           if hasattr(v, "data_ptr")}}


class Labels:
    """Annotations only. Preserve every original call, argument, return and wait."""
    def __init__(self, engine, torch, graphs):
        self.engine, self.torch, self.graphs = engine, torch, graphs
        self.runner = engine.model_runner
        self.undo = ExitStack()
        self.next_id = 0
        self.active = None
        self.executed = deque()
        self.last_waited = None
        self.slots = {}
        self.batches = []

    def mark(self, stage, bid, slot=None):
        suffix = f"/slot{slot}" if slot is not None else ""
        return self.torch.profiler.record_function(f"nq/{stage}/b{bid:05d}{suffix}")

    def patch(self, obj, attr, value):
        own = attr in vars(obj)
        old = getattr(obj, attr)
        setattr(obj, attr, value)
        self.undo.callback(setattr if own else delattr, obj, attr, *([old] if own else []))

    @contextmanager
    def activate(self, bid):
        old, self.active = self.active, bid
        try:
            yield
        finally:
            self.active = old

    def __enter__(self):
        r = self.runner
        original_execute = r.execute_model
        def execute(seqs, is_prefill):
            if is_prefill:
                raise RuntimeError("Prefill entered the decode-only diagnostic window")
            bid = self.next_id
            self.next_id += 1
            self.batches.append({"batch_id": bid, "batch_size": len(seqs)})
            with self.activate(bid), self.mark("execute", bid):
                result = original_execute(seqs, is_prefill)
            self.executed.append(bid)
            return result
        self.patch(r, "execute_model", execute)

        original_plan = self.engine.scheduler.schedule
        def plan(*args, **kwargs):
            with self.mark("schedule", self.next_id):
                return original_plan(*args, **kwargs)
        self.patch(self.engine.scheduler, "schedule", plan)

        for graph in self.graphs:
            original_replay = graph.replay
            def replay(*args, _original=original_replay, **kwargs):
                if self.active is None:
                    raise RuntimeError("Graph replay outside a tracked decode execution")
                with self.mark("launch", self.active):
                    return _original(*args, **kwargs)
            self.patch(graph, "replay", replay)

        original_logits = r.model.compute_logits
        def logits(*args, **kwargs):
            if self.active is None:
                raise RuntimeError("LM head outside tracked execute/sample")
            with self.mark("lm_head", self.active):
                return original_logits(*args, **kwargs)
        self.patch(r.model, "compute_logits", logits)

        original_sample = r.sample_tokens
        def sample(*args, **kwargs):
            bid = self.executed.popleft()
            with self.activate(bid), self.mark("sample_call", bid):
                output = original_sample(*args, **kwargs)
            if output is None:
                raise RuntimeError("Rank-0 sample_tokens returned no output")
            output._nq_trace_batch_id = bid
            return output
        self.patch(r, "sample_tokens", sample)

        from nano_qwen.engine.async_output import AsyncModelOutput
        original_get = AsyncModelOutput.get_output
        def get_output(output, *args, **kwargs):
            bid = getattr(output, "_nq_trace_batch_id", None)
            if bid is None:  # Already queued before profiler activation.
                return original_get(output, *args, **kwargs)
            with self.mark("result_wait", bid):
                value = original_get(output, *args, **kwargs)
            self.last_waited = bid
            return value
        self.patch(AsyncModelOutput, "get_output", get_output)

        original_commit = self.engine.scheduler.postprocess
        def commit(*args, **kwargs):
            if self.last_waited is None:
                return original_commit(*args, **kwargs)
            with self.mark("commit", self.last_waited):
                self.last_waited = None
                return original_commit(*args, **kwargs)
        self.patch(self.engine.scheduler, "postprocess", commit)
        return self

    def __exit__(self, *exc):
        self.undo.close()


def end(event):
    return event["ts"] + event.get("dur", 0)


def merge(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if result and lo <= result[-1][1]:
            result[-1][1] = max(result[-1][1], hi)
        else:
            result.append([lo, hi])
    return result


def overlap(interval, windows):
    return sum(max(0, min(interval[1], hi) - max(interval[0], lo)) for lo, hi in merge(windows))


def median_ms(values):
    return statistics.median(values) / 1000 if values else None


def analyze(data):
    """Use native correlation IDs, never CPU-scope duration as GPU duration."""
    events = data["traceEvents"]
    scopes = defaultdict(dict)
    all_markers = []
    for e in events:
        m = MARKER.fullmatch(e.get("name", ""))
        if e.get("cat") == "user_annotation" and e.get("ph") == "X" and m:
            stage, bid, slot = m[1], int(m[2]), int(m[3] or 0)
            scopes[bid].setdefault(stage, e)
            all_markers.append((e, stage, bid, slot))
    # Sample scopes may contain an LM head in the dual path. More specific
    # scopes take priority so GPU sampling never includes the LM head.
    attributed = {}
    launch_api = {}
    indexed = {}
    for stage in ("launch", "lm_head", "sample_call"):
        marked = sorted((e["ts"], end(e), bid, slot) for e, s, bid, slot in all_markers if s == stage)
        indexed[stage] = ([x[0] for x in marked], marked)
    for e in events:
        if e.get("cat") not in ("cuda_runtime", "cuda_driver") or e.get("ph") != "X":
            continue
        corr = e.get("args", {}).get("correlation")
        if corr is None:
            continue
        for stage in ("launch", "lm_head", "sample_call"):
            starts, marked = indexed[stage]
            i = bisect_right(starts, e["ts"]) - 1
            if i >= 0 and e["ts"] <= marked[i][1] and end(e) <= marked[i][1] + 1:
                _, _, bid, slot = marked[i]
                if stage == "launch" and e.get("name") != "cudaGraphLaunch":
                    continue
                attributed[corr] = (stage, bid, slot)
                if e.get("name") == "cudaGraphLaunch":
                    launch_api[bid] = e
                break
    kernels = defaultdict(lambda: defaultdict(list))
    for e in events:
        if e.get("cat") != "kernel" or e.get("ph") != "X":
            continue
        tag = attributed.get(e.get("args", {}).get("correlation"))
        if tag:
            stage, bid, slot = tag
            kernels[bid][stage].append(e)
    frames = []
    for bid in sorted(scopes):
        s, k = scopes[bid], kernels[bid]
        if bid not in launch_api or not all(name in s for name in ("schedule", "execute", "sample_call", "commit")):
            continue
        if not k["launch"] or not k["sample_call"] or not k["lm_head"]:
            continue
        launch = launch_api[bid]
        if s["schedule"]["ts"] > launch["ts"]:
            raise RuntimeError("Invalid plan/launch ordering")
        graph = [min(e["ts"] for e in k["launch"]), max(end(e) for e in k["launch"])]
        head = [min(e["ts"] for e in k["lm_head"]), max(end(e) for e in k["lm_head"])]
        sample = [min(e["ts"] for e in k["sample_call"]), max(end(e) for e in k["sample_call"])]
        frame = dict(batch_id=bid,
                     plan=[s["schedule"]["ts"], launch["ts"]], launch=[launch["ts"], end(launch)],
                     graph=graph, lm_head=head, forward=merge([graph, head]), sample=sample,
                     commit=[s["commit"]["ts"], end(s["commit"])],
                     graph_kernel_count=len(k["launch"]),
                     graph_correlations=sorted({e["args"]["correlation"] for e in k["launch"]}),
                     forward_kernel_intervals=merge([(e["ts"], end(e)) for stage in ("launch", "lm_head") for e in k[stage]]))
        if "result_wait" in s:
            frame["result_wait"] = [s["result_wait"]["ts"], end(s["result_wait"])]
        frames.append(frame)
    return frames


def summarize_frames(frames):
    adjacent = [(a, b) for a, b in zip(frames, frames[1:]) if b["batch_id"] == a["batch_id"] + 1]
    result = {"complete_batches": len(frames), "adjacent_pairs": len(adjacent),
              "graph_kernel_counts": [f["graph_kernel_count"] for f in frames]}
    for name in ("plan", "launch"):
        result[name + "_next_overlaps_forward"] = sum(overlap(b[name], a["forward"]) > 0 for a, b in adjacent)
    result["commit_overlaps_next_forward"] = sum(overlap(a["commit"], b["forward"]) > 0 for a, b in adjacent)
    result["commit_overlaps_next_physical_kernel"] = sum(
        overlap(a["commit"], b["forward_kernel_intervals"]) > 0 for a, b in adjacent)
    result["sample_before_next_graph"] = sum(a["sample"][1] <= b["graph"][0] for a, b in adjacent)
    result["graph_ms_median"] = median_ms([f["graph"][1] - f["graph"][0] for f in frames])
    result["forward_active_span_ms_median"] = median_ms([sum(hi-lo for lo,hi in f["forward"]) for f in frames])
    result["plan_ms_median"] = median_ms([f["plan"][1] - f["plan"][0] for f in frames])
    result["launch_api_ms_median"] = median_ms([f["launch"][1] - f["launch"][0] for f in frames])
    result["sample_ms_median"] = median_ms([f["sample"][1] - f["sample"][0] for f in frames])
    result["result_wait_ms_median"] = median_ms([f["result_wait"][1] - f["result_wait"][0] for f in frames if "result_wait" in f])
    result["gpu_step_interval_ms_median"] = median_ms([b["graph"][0] - a["graph"][0] for a, b in adjacent])
    result["sample_to_next_graph_ms_median"] = median_ms([b["graph"][0] - a["sample"][1] for a, b in adjacent])
    result["diagnostic_only"] = True
    return result


def derived_events(frames, pid, mode):
    result = [{"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
               "args": {"name": f"{mode}: plan includes prepare; launch is CPU; forward/sample are GPU"}}]
    tracks = {1: "CPU: plan / launch / commit", 2: "GPU: forward / sample", 3: "GPU: cuda_graph"}
    for tid, label in tracks.items():
        result.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": label}})
    for i, frame in enumerate(frames):
        def add(name, tid, span):
            result.append({"ph": "X", "cat": "nq_derived", "pid": pid, "tid": tid,
                           "name": name, "ts": span[0], "dur": span[1] - span[0],
                           "args": {"batch_id": frame["batch_id"],
                                    "source": "native CUDA correlation / CPU markers", "may_contain_kernel_gaps": True}})
        for name in ("plan", "launch", "commit"):
            add(f"{name} {i}", 1, frame[name])
        for interval in frame["forward"]:
            add(f"forward {i}", 2, interval)
        add(f"sample {i}", 2, frame["sample"])
        add("cuda_graph", 3, frame["graph"])
    return result


def bounded_trace(data, frames, mode, max_steps=6, max_bytes=20*MIB):
    """Keep a continuous time window and complete correlation groups."""
    # Start after the first profiled execution, allowing pipeline startup at
    # the profiler boundary to stay outside the saved comparison window.
    candidates = frames[1:1+max_steps]
    if len(candidates) < 4:
        raise RuntimeError(f"Only {len(candidates)} complete steady batches; need at least 4. Check CUPTI/markers")
    original = data["traceEvents"]
    numeric_pids = [e["pid"] for e in original if isinstance(e.get("pid"), int)]
    pid = max(numeric_pids, default=0) + 100
    for count in range(len(candidates), 3, -1):
        selected = candidates[:count]
        lo = min(f["plan"][0] for f in selected)
        hi = max(max(f["commit"][1], f["sample"][1], f["graph"][1], f["lm_head"][1]) for f in selected)
        # Keep original X events that intersect the window, with unmodified
        # timestamps/durations. A few edge context events may extend past it.
        kept = [e for e in original if e.get("ph") not in ("M", "s", "f", "t")
                and "ts" in e and e["ts"] <= hi and end(e) >= lo]
        flows = defaultdict(list)
        for e in original:
            if e.get("ph") in ("s", "f", "t"):
                flows[(e.get("cat"), e.get("name"), str(e.get("id")), str(e.get("id2")))].append(e)
        for group in flows.values():
            if all(lo <= e["ts"] <= hi for e in group) and any(e["ph"] == "s" for e in group) and any(e["ph"] == "f" for e in group):
                kept.extend(group)
        active_pids = {e.get("pid") for e in kept}
        kept = [e for e in original if e.get("ph") == "M" and e.get("pid") in active_pids] + kept
        # A selected graph must retain EVERY native kernel from the recording.
        corr = {c for f in selected for c in f["graph_correlations"]}
        expected = Counter(e["args"]["correlation"] for e in original if e.get("cat") == "kernel" and e.get("args", {}).get("correlation") in corr)
        actual = Counter(e["args"]["correlation"] for e in kept if e.get("cat") == "kernel" and e.get("args", {}).get("correlation") in corr)
        if expected != actual:
            raise RuntimeError("Window dropped kernels from a selected graph")
        kept.extend(derived_events(selected, pid, mode))
        output = {k: v for k, v in data.items() if k != "traceEvents"}
        output.update(traceEvents=kept, nq_window={"mode": mode, "complete_batch_ids": [f["batch_id"] for f in selected],
                      "start_us": lo, "end_us": hi, "edge_context_may_be_partial": True,
                      "policy": "continuous window; all selected graph kernels; unchanged timestamps and durations",
                      "requested_complete_batches": max_steps, "max_uncompressed_bytes": int(max_bytes)})
        encoded = json.dumps(output, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) <= max_bytes:
            return encoded, selected, output["nq_window"]
    raise RuntimeError("Even four complete batches exceed --max-trace-mb; increase the size cap. No kernels were sampled")


def output_records(prompts, seqs):
    if len(prompts) != len(seqs):
        raise RuntimeError("Prompt/request count mismatch")
    return [{"request_index": i, "runtime_seq_id": seq.seq_id,
             "prompt_sha256": digest(prompt), "input_len": len(prompt),
             "max_tokens": seq.max_tokens, "ignore_eos": seq.ignore_eos,
             "tokens": [int(t) for t in seq.completion_token_ids]}
            for i, (prompt, seq) in enumerate(zip(prompts, seqs))]


def compare_outputs(reference, actual, max_examples=8):
    """Compare actual token arrays, including lengths; runtime sequence IDs may differ."""
    def indexed(rows):
        result = {r["request_index"]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate request_index in output records")
        return result
    expected, observed = indexed(reference), indexed(actual)
    report = dict(passed=True, compared_requests=0, mismatching_requests=0,
                  mismatching_tokens=0, input_mismatches=0, examples=[])
    for index in sorted(set(expected) | set(observed)):
        x, y = expected.get(index), observed.get(index)
        if x is None or y is None:
            report["mismatching_requests"] += 1
            report["mismatching_tokens"] += len((x or y)["tokens"])
            if len(report["examples"]) < max_examples:
                report["examples"].append(dict(request_index=index, kind="missing_request",
                                                reference_present=x is not None, actual_present=y is not None))
            continue
        report["compared_requests"] += 1
        input_fields = ("prompt_sha256", "input_len", "max_tokens", "ignore_eos")
        wrong_input = [k for k in input_fields if x[k] != y[k]]
        a, b = x["tokens"], y["tokens"]
        positions = [i for i in range(max(len(a), len(b)))
                     if i >= len(a) or i >= len(b) or a[i] != b[i]]
        if wrong_input:
            report["input_mismatches"] += 1
        if positions or wrong_input:
            report["mismatching_requests"] += 1
            report["mismatching_tokens"] += len(positions)
            if len(report["examples"]) < max_examples:
                first = positions[0] if positions else None
                report["examples"].append(dict(request_index=index, input_fields_differ=wrong_input,
                    first_token_index=first, reference_length=len(a), actual_length=len(b),
                    reference_token=a[first] if first is not None and first < len(a) else None,
                    actual_token=b[first] if first is not None and first < len(b) else None,
                    reference_context=a[max(0,(first or 0)-4):(first or 0)+5],
                    actual_context=b[max(0,(first or 0)-4):(first or 0)+5]))
    report["passed"] = bool(expected) and not (report["mismatching_requests"] or report["input_mismatches"])
    return report


def compare_suites(reference, actual):
    def index(report):
        result = {(r["case"], r["seed"], r["repeat"]): r for r in report["runs"]}
        if len(result) != len(report["runs"]):
            raise ValueError("Duplicate correctness run keys")
        return result
    x, y = index(reference), index(actual)
    reports = []
    for key in sorted(set(x) | set(y)):
        if key not in x or key not in y:
            reports.append(dict(case=key[0], seed=key[1], repeat=key[2], passed=False, error="missing_run"))
            continue
        comparison = compare_outputs(x[key]["requests"], y[key]["requests"])
        comparison.update(case=key[0], seed=key[1], repeat=key[2])
        comparison["passed"] &= x[key].get("passed", False) and y[key].get("passed", False)
        reports.append(comparison)
    return {"passed": bool(reports) and reference.get("status") == actual.get("status") == "ok"
                       and reference.get("passed", True) and actual.get("passed", True)
                       and all(r["passed"] for r in reports),
            "compared_runs": len(reports), "failed_runs": [r for r in reports if not r["passed"]],
            "mismatching_tokens": sum(r.get("mismatching_tokens", 0) for r in reports)}


def consistency_report(mode, suite, reference, reference_path, other_suites):
    same_source = compare_suites(reference, suite)
    cross_mode = {name: compare_suites(other, suite) for name, other in other_suites.items() if name != mode}
    repeat = check_repeats(suite)
    local_pass = bool(suite.get("passed", False) and repeat["passed"])
    return dict(mode=mode, same_source_serial_reference=str(reference_path),
                same_source_tokens=same_source, repeated_outputs=repeat,
                profile_vs_unprofiled=suite.get("profile_vs_unprofiled"),
                cross_mode_tokens=cross_mode,
                scheduling_checks_passed=local_pass and same_source["passed"],
                all_compared_tokens_equal=all(r["passed"] for r in cross_mode.values()),
                passed=local_pass and same_source["passed"] and all(r["passed"] for r in cross_mode.values()),
                scope="Greedy token identity and lifecycle checks; not a proof of bitwise logits/KV equality")


def check_repeats(suite):
    groups = defaultdict(list)
    for run in suite["runs"]:
        groups[(run["case"], run["seed"])].append(run)
    failures, comparisons = [], 0
    planned = {(c["name"], c["seed"]) for c in suite.get("case_plan", [])}
    if planned and planned != set(groups):
        failures.append(dict(error="case_plan_incomplete", missing_cases=sorted(planned-set(groups)),
                             unexpected_cases=sorted(set(groups)-planned)))
    for key, runs in groups.items():
        runs.sort(key=lambda r: r["repeat"])
        if len(runs) != suite["repeats"] or [r["repeat"] for r in runs] != list(range(suite["repeats"])):
            failures.append(dict(case=key[0], seed=key[1], error="missing_repeat"))
            continue
        for run in runs[1:]:
            comparison = compare_outputs(runs[0]["requests"], run["requests"])
            comparisons += 1
            if not comparison["passed"]:
                comparison.update(case=key[0], seed=key[1], repeat=run["repeat"])
                failures.append(comparison)
    return dict(passed=bool(groups) and suite.get("status") == "ok" and not failures,
                comparisons=comparisons, failures=failures)


class SamplerPathAudit:
    """Record the sampler's real dispatch: the all_greedy argmax branch.

    The wrapper counts every production sampler call and, while the equality
    check is enabled (correctness phase only, never inside the profiler),
    compares returned tokens with logits.argmax on device. Tokens, logits and
    temperatures are passed through untouched.
    """

    def __init__(self, engine, torch):
        runner = engine.model_runner
        self._original = runner.sampler
        self._torch = torch
        self._mismatch = torch.zeros((), dtype=torch.int64, device="cuda")
        self.stats = {"calls": 0, "all_greedy_calls": 0, "mixed_calls": 0,
                      "argmax_checks": 0, "verify_argmax_equality": False,
                      "argmax_mismatches": 0}
        audit = self
        original = self._original
        class _SamplerAuditProxy:
            def __call__(self, logits, temperatures, all_greedy=False, **kwargs):
                audit.stats["calls"] += 1
                audit.stats["all_greedy_calls" if all_greedy else "mixed_calls"] += 1
                tokens = original(logits, temperatures, all_greedy=all_greedy, **kwargs)
                if audit.stats["verify_argmax_equality"]:
                    audit.stats["argmax_checks"] += 1
                    audit._mismatch += (tokens != logits.argmax(dim=-1)).sum()
                return tokens
            def __getattr__(self, name):
                return getattr(original, name)
        runner.sampler = _SamplerAuditProxy()

    def finalize(self):
        self._torch.cuda.synchronize()
        self.stats["argmax_mismatches"] = int(self._mismatch.item())
        if self.stats["calls"] == 0:
            raise RuntimeError("The sampler was never called; argmax path unverified")
        if self.stats["mixed_calls"]:
            raise RuntimeError(f"{self.stats['mixed_calls']} sampler call(s) left the "
                               "all_greedy argmax path; temperature alone is not evidence")
        if self.stats["argmax_checks"] and self.stats["argmax_mismatches"]:
            raise RuntimeError(f"{self.stats['argmax_mismatches']} sampled token(s) differ "
                               "from logits.argmax; the argmax path was not taken")
        return self.stats


def correctness_cases(a, block_size, trace_output_len):
    cases = [dict(name="trace_replay", seed=a.seed, input_len=a.input_len,
                  max_tokens=trace_output_len, policy="uniform")]
    for seed in a.check_seeds:
        cases.extend([
            dict(name="steady", seed=seed, input_len=a.input_len, max_tokens=a.check_steps, policy="uniform"),
            dict(name="kv_boundary", seed=seed, input_len=block_size-2, max_tokens=a.check_steps, policy="uniform"),
            dict(name="mixed_lengths", seed=seed, input_len=a.input_len, max_tokens=a.check_steps, policy="mixed"),
            dict(name="forced_eos", seed=seed, input_len=a.input_len, max_tokens=a.check_steps, policy="forced_eos"),
        ])
    return cases


def eos_actions(seq_ids, counts, selected, at=3):
    force, suppress = [], []
    for row, seq_id in enumerate(seq_ids):
        counts[seq_id] = counts.get(seq_id, 0) + 1
        if seq_id in selected:
            if counts[seq_id] == at:
                force.append(row)
            elif counts[seq_id] < at:
                suppress.append(row)
    return force, suppress


@contextmanager
def forced_eos_policy(runner, seqs, eos, vocab_size, torch):
    """EOS lifecycle fault injection ONLY, outside profiler. Other tests use the real sampler unchanged."""
    if not isinstance(eos, int) or not 0 <= eos < vocab_size:
        raise RuntimeError("A valid model eos_token_id is required for the forced_eos test")
    original = runner.sampler
    selected = {s.seq_id for s in seqs[:max(1, len(seqs)//2)]}
    class SamplerProxy:
        def __init__(self):
            self.context, self.counts, self.forced = [], {}, set()
        def __getattr__(self, name):
            return getattr(original, name)
        def set_batch_context(self, sampled_seqs, is_prefill):
            self.context = list(sampled_seqs)
            setter = getattr(original, "set_batch_context", None)
            if setter is not None:
                setter(sampled_seqs, is_prefill)
        def __call__(self, logits, temperatures, *args, **kwargs):
            tokens = original(logits, temperatures, *args, **kwargs)
            if len(self.context) != tokens.shape[0]:
                raise RuntimeError("Sampler row mapping does not match requests")
            ids = [s.seq_id for s in self.context]
            force, suppress = eos_actions(ids, self.counts, selected)
            if force or suppress:
                tokens = tokens.clone()
            if suppress:
                rows = torch.tensor(suppress, dtype=torch.int64, device=tokens.device)
                old = tokens[rows]
                tokens[rows] = torch.where(old == eos, (eos+1) % vocab_size, old)
            if force:
                rows = torch.tensor(force, dtype=torch.int64, device=tokens.device)
                tokens[rows] = eos
                self.forced.update(ids[row] for row in force)
            return tokens
    proxy = SamplerProxy()
    runner.sampler = proxy
    try:
        yield proxy
    finally:
        runner.sampler = original


def lifecycle_state(engine):
    scheduler, runner = engine.scheduler, engine.model_runner
    manager = scheduler.block_manager
    state = dict(engine_finished=engine.is_finished(), waiting=len(scheduler.waiting),
                 running=len(scheduler.running), in_flight=len(scheduler.in_flight),
                 batch_queue=len(engine.batch_queue), input_slots=len(runner.input_batch.seq_id_to_slot),
                 used_kv_blocks=len(manager.used_block_ids), free_kv_blocks=len(manager.free_block_ids),
                 expected_free_kv_blocks=engine.config.num_kvcache_blocks,
                 unique_free_kv_blocks=len(set(manager.free_block_ids)),
                 async_cohort_open=getattr(scheduler, "async_decode", None) is not None,
                 pending_sampling=getattr(runner, "_pending", None) is not None,
                 pending_sampling_plan=getattr(runner, "_sampling_plan", None) is not None,
                 retained_target_hidden=getattr(runner, "_target_hidden", None) is not None)
    zero = ("waiting", "running", "in_flight", "batch_queue", "input_slots", "used_kv_blocks")
    state["passed"] = bool(state["engine_finished"] and not state["async_cohort_open"]
                           and not state["pending_sampling"] and not state["pending_sampling_plan"]
                           and not state["retained_target_hidden"] and all(state[k] == 0 for k in zero)
                           and state["free_kv_blocks"] == state["unique_free_kv_blocks"] == state["expected_free_kv_blocks"])
    return state


def run_correctness_suite(engine, a, torch, Sequence, SamplingParams, out, trace_outputs, trace_output_len):
    cases = correctness_cases(a, engine.config.kvcache_block_size, trace_output_len)
    suite = dict(status="running", mode=a.worker, repeats=a.check_repeats, runs=[],
                 checks_profiler_enabled=False, no_per_step_synchronize=True,
                 greedy_temperature=1e-8, case_plan=cases,
                 coverage=["long_decode", "kv_block_boundary", "different_max_tokens", "forced_early_eos",
                           "slot_reuse_between_workloads", "repeated_determinism", "profile_vs_unprofiled"])
    path = out / "check_outputs.json"
    save(path, suite)
    vocab = int(engine.config.hf_config.vocab_size)
    from contextlib import nullcontext
    with torch.inference_mode():
        for case in cases:
            for repeat in range(a.check_repeats):
                if not lifecycle_state(engine)["passed"]:
                    raise RuntimeError("Correctness workload started with unretired state")
                rng = random.Random(case["seed"])
                prompts = [[rng.randrange(vocab) for _ in range(case["input_len"])] for _ in range(a.batch_size)]
                seqs, expected_lengths = [], []
                for i, prompt in enumerate(prompts):
                    short = i < max(1, a.batch_size//2)
                    cap = 2 if case["policy"] == "mixed" and short else case["max_tokens"]
                    force = case["policy"] == "forced_eos" and short
                    seqs.append(Sequence(prompt, SamplingParams(temperature=1e-8, max_tokens=cap, ignore_eos=not force)))
                    expected_lengths.append(3 if force else cap)
                policy = (forced_eos_policy(engine.model_runner, seqs, engine.config.eos, vocab, torch)
                          if case["policy"] == "forced_eos" else nullcontext())
                engine.model_runner.cuda_graphs.reset_dispatch_stats()
                run = dict(case=case["name"], seed=case["seed"], repeat=repeat, policy=case["policy"],
                           passed=False, expected_lengths=expected_lengths)
                try:
                    with policy as injection:
                        for seq in seqs:
                            engine.scheduler.add(seq)
                        ticks = 0
                        while not engine.is_finished():
                            engine.step()
                            ticks += 1
                            if ticks > case["max_tokens"] * 4 + 32:
                                raise RuntimeError("Correctness workload did not drain")
                        torch.cuda.synchronize()  # Once per entire workload, AFTER the pipeline drains.
                    run.update(requests=output_records(prompts, seqs), lifecycle=lifecycle_state(engine),
                               host_ticks=ticks, dispatch=engine.model_runner.cuda_graphs.dispatch_stats())
                    run["lengths_match"] = [len(r["tokens"]) for r in run["requests"]] == expected_lengths
                    run["eos_policy_verified"] = True
                    if case["policy"] == "forced_eos":
                        selected = seqs[:max(1, a.batch_size//2)]
                        run["eos_policy_verified"] = (injection.forced == {s.seq_id for s in selected}
                            and all(s.completion_token_ids[-1] == engine.config.eos for s in selected))
                    decode = run["dispatch"].get("decode", {})
                    run["decode_graph_observed"] = decode.get("graph_calls", 0) > 0
                    run["passed"] = bool(run["lifecycle"]["passed"] and run["lengths_match"]
                                         and run["eos_policy_verified"] and run["decode_graph_observed"]
                                         and all(s.is_finished for s in seqs))
                    if not run["passed"]:
                        raise RuntimeError("Correctness lifecycle, output-length, EOS or graph-path check failed")
                except Exception:
                    run.update(error=traceback.format_exc(), passed=False)
                    run.setdefault("requests", output_records(prompts, seqs))
                    suite["runs"].append(run)
                    suite["status"] = "failed"
                    save(path, suite)
                    raise
                suite["runs"].append(run)
                save(path, suite)
                print(f"check {case['name']} seed={case['seed']} repeat={repeat}: lifecycle/length PASS", flush=True)
    suite["status"] = "ok"
    suite["repeat_consistency"] = check_repeats(suite)
    if trace_outputs is not None:
        rows = [r for r in suite["runs"] if r["case"] == "trace_replay"]
        comparisons = [compare_outputs(trace_outputs, r["requests"]) for r in rows]
        suite["profile_vs_unprofiled"] = dict(passed=all(r["passed"] for r in comparisons), comparisons=comparisons)
    else:
        suite["profile_vs_unprofiled"] = {"status": "not_recorded"}
    suite["passed"] = bool(suite["repeat_consistency"]["passed"] and suite["profile_vs_unprofiled"].get("passed", True))
    save(path, suite)
    return suite


def run_worker(a):
    root = Path(a.source_root).resolve()
    sys.path.insert(0, str(root / "src"))
    import torch
    from nano_qwen.config import Config
    from nano_qwen.engine.llm_engine import LLMEngine
    from nano_qwen.engine.model_runner import ModelRunner
    from nano_qwen.engine.sequence import Sequence
    from nano_qwen.sampling_params import SamplingParams
    from nano_qwen.engine import llm_engine as engine_module
    if not Path(engine_module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Imported a different nano_qwen checkout")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real model checks")
    if not a.checks_only and torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError("CUPTI profiler support required for trace collection")
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") != "0":
        raise RuntimeError("Unset CUDA_LAUNCH_BLOCKING before overlap profiling")
    mode = a.worker
    out = Path(a.worker_output) if a.worker_output else Path(a.output_dir) / mode
    out.mkdir(parents=True, exist_ok=True)
    if "async_scheduling" not in Config.__dataclass_fields__:
        raise RuntimeError("This source does not implement async_scheduling; refusing a silent fallback")
    capture_ticks = a.trace_steps + 4
    output_len = a.settle_steps + capture_ticks + 9
    block_size = Config.__dataclass_fields__["kvcache_block_size"].default
    max_input = max(a.input_len, block_size-2)
    max_model_len = math.ceil((max_input + max(output_len, a.check_steps) + 2) / block_size) * block_size
    metadata = dict(mode=mode, args=vars(a), source=json.loads(a.source_json),
                    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    imported_engine=engine_module.__file__, output_len=output_len, max_model_len=max_model_len,
                    benchmark_policy={"prefill_graph": False, "kv_pool": "workload capacity + one block/request",
                                      "temperature": 1e-8, "ignore_eos": True, "no_per_tick_sync": True,
                                      "no_artificial_gpu_delay": True, "record_shapes": False,
                                      "with_stack": False, "profile_memory": False})
    save(out / "metadata.json", metadata)
    kwargs = dict(tensor_parallel_size=1, max_num_seqs=a.batch_size,
                  max_num_batched_tokens=a.batch_size * max_input, max_model_len=max_model_len,
                  enable_prefix_cache=False, enforce_eager=False, use_prefill_cudagraph=False,
                  gpu_memory_utilization=a.gpu_memory_utilization)
    kwargs["async_scheduling"] = mode == "async"
    engine = None
    try:
        torch.manual_seed(a.seed)
        torch.cuda.set_per_process_memory_fraction(a.gpu_memory_utilization, device=0)
        with torch.inference_mode(), workload_kv_pool(ModelRunner, torch, a.batch_size, max_model_len):
            engine = LLMEngine(a.model, **kwargs)
        engine.max_concurrent_batches = 1 if mode == "no_async" else 2
        engine.model_runner.async_output = True  # Keep D2H policy identical in both modes.
        graphs, audit = audit_graphs(engine, mode, a.batch_size)
        metadata.update(actual=audit, kv_blocks=engine.config.num_kvcache_blocks,
                        quantization_methods=dict(Counter(
                            type(getattr(m, "quant_method", None)).__name__
                            for m in engine.model_runner.model.modules()
                            if getattr(m, "quant_method", None) is not None)))
        sampler_audit = SamplerPathAudit(engine, torch)
        save(out / "metadata.json", metadata)

        def enqueue():
            if not engine.is_finished():
                raise RuntimeError("Previous workload did not drain")
            rng = random.Random(a.seed)
            vocab = int(engine.config.hf_config.vocab_size)
            prompts = [[rng.randrange(vocab) for _ in range(a.input_len)] for _ in range(a.batch_size)]
            seqs = [Sequence(prompt, SamplingParams(temperature=1e-8, max_tokens=output_len, ignore_eos=True)) for prompt in prompts]
            for seq in seqs:
                engine.scheduler.add(seq)
            return prompts, seqs

        def prefill(seqs):
            engine.step()
            torch.cuda.synchronize()  # Outside recording; common prefill/decode boundary.
            if any(s.num_completion_tokens != 1 for s in seqs):
                raise RuntimeError("Expected one complete prefill of the entire fixed batch")

        def drain(seqs):
            ticks = 0
            while not engine.is_finished():
                engine.step()
                ticks += 1
                if ticks > output_len * 4 + 32:
                    raise RuntimeError("Engine failed to drain")
            torch.cuda.synchronize()
            if any(not s.is_finished or s.num_completion_tokens != output_len for s in seqs):
                raise RuntimeError("Unexpected token count")
            runner = engine.model_runner
            if runner.input_batch.seq_id_to_slot or engine.scheduler.block_manager.used_block_ids:
                raise RuntimeError("Persistent request slots or KV pages leaked")

        warmup_ms = []
        trace_outputs = None
        summary = dict(status="ok", mode=mode, trace_collected=False)
        with torch.inference_mode():
            for _ in range(a.warmup_rounds):
                _, seqs = enqueue()
                prefill(seqs)
                started = time.perf_counter()
                drain(seqs)
                warmup_ms.append((time.perf_counter() - started) * 1000 / (output_len - 1))
            if not a.checks_only:
                prompts, seqs = enqueue()
                prefill(seqs)
                for _ in range(a.settle_steps):
                    engine.step()
                torch.cuda.synchronize()  # Once before recording, never between recorded ticks.
                engine.model_runner.cuda_graphs.reset_dispatch_stats()
                with Labels(engine, torch, graphs) as labels:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA],
                                                record_shapes=False, with_stack=False, profile_memory=False) as prof:
                        for _ in range(capture_ticks):
                            engine.step()
                        torch.cuda.synchronize()  # Recording-end drain only.
                dispatch = engine.model_runner.cuda_graphs.dispatch_stats().get("decode", {})
                if (dispatch.get("graph_calls") != labels.next_id or dispatch.get("eager_calls", 0)
                        or dispatch.get("batch_counts") != {a.batch_size: labels.next_id}):
                    raise RuntimeError(f"Unexpected graph dispatch / batch size: {dispatch}")
                drain(seqs)
                trace_outputs = output_records(prompts, seqs)
                save(out / "trace_outputs.json", {"requests": trace_outputs})
        if not a.checks_only:
            with tempfile.TemporaryDirectory(prefix="nq-raw-trace-") as temporary:
                raw = Path(temporary) / "raw.json"
                prof.export_chrome_trace(str(raw))
                data = json.loads(raw.read_text(encoding="utf-8"))
            frames = analyze(data)
            encoded, selected, window = bounded_trace(data, frames, mode, a.trace_steps, int(a.max_trace_mb*MIB))
            summary = summarize_frames(selected)
            summary.update(status="ok", mode=mode, trace_bytes=len(encoded), trace_mib=len(encoded)/MIB,
                           warmup_unprofiled_mean_step_ms=warmup_ms, prompt_sha256=digest(prompts),
                           output_sha256=digest([s.completion_token_ids for s in seqs]),
                           captured_dispatch=dispatch, captured_complete_batches=len(frames), saved_window=window,
                           notes=["Trace diagnostics include profiler overhead; do not use them as formal speedup.",
                                  "Forward consists of graph and LM-head GPU segments; gaps between segments are excluded.",
                                  "Original native kernels/operators are preserved; edge context may be partial."])
            if summary["sample_before_next_graph"] != summary["adjacent_pairs"]:
                raise RuntimeError("GPU token dependency order failed: sample(n) must precede graph(n+1)")
            summary["pipeline_observed"] = bool(summary["launch_next_overlaps_forward"])
            summary["expected_overlap_observed"] = (not summary["pipeline_observed"] if mode == "no_async" else summary["pipeline_observed"])
            (out / "trace.json").write_bytes(encoded)
            save(out / "summary.json", summary)
            print(f"{mode}: {len(selected)} complete batches; {len(encoded)/MIB:.2f} MiB; "
                  f"launch overlap {summary['launch_next_overlaps_forward']}/{summary['adjacent_pairs']}; "
                  f"GPU step {summary['gpu_step_interval_ms_median']:.2f} ms", flush=True)
            if not summary["expected_overlap_observed"]:
                print("WARNING: actual overlap differs from this mode's intention; inspect the recorded waits.", flush=True)
            summary["trace_collected"] = True
        sampler_audit.stats["verify_argmax_equality"] = True
        suite = run_correctness_suite(engine, a, torch, Sequence, SamplingParams, out, trace_outputs, output_len)
        sampler_stats = sampler_audit.finalize()
        metadata["sampler_path"] = sampler_stats
        save(out / "metadata.json", metadata)
        summary.update(output_checks_passed=suite["passed"], correctness_runs=len(suite["runs"]),
                       repeat_consistency=suite["repeat_consistency"], profile_vs_unprofiled=suite["profile_vs_unprofiled"],
                       sampler_path=sampler_stats)
        save(out / "summary.json", summary)
    finally:
        if engine is not None:
            engine.exit()


def parent(a):
    repo = find_repo(a.repo_root)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    modes = MODES if a.mode == "all" else (a.mode,)
    for mode in modes:
        destination = out / mode
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(f"Output directory is not empty: {destination}; use a new --output-dir")
    if "async_scheduling" not in config_fields(repo):
        raise RuntimeError("The current checkout does not implement async_scheduling")
    info = source_info(repo)
    root = repo

    def launch(mode, destination, checks_only):
        destination.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--model", a.model,
                   "--worker", mode, "--source-root", str(root), "--source-json", json.dumps(info),
                   "--worker-output", str(destination),
                   "--batch-size", str(a.batch_size), "--input-len", str(a.input_len),
                   "--trace-steps", str(a.trace_steps), "--max-trace-mb", str(a.max_trace_mb),
                   "--settle-steps", str(a.settle_steps), "--warmup-rounds", str(a.warmup_rounds),
                   "--gpu-memory-utilization", str(a.gpu_memory_utilization), "--seed", str(a.seed),
                   "--check-steps", str(a.check_steps), "--check-repeats", str(a.check_repeats),
                   "--check-seeds", *map(str, a.check_seeds), "--output-dir", str(out)]
        if checks_only:
            command.append("--checks-only")
        label = f"{destination.relative_to(out)}"
        print(f"[{label}] source={info['commit'] or root}; log={destination/'run.log'}", flush=True)
        env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
        failed = False
        with (destination / "run.log").open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(command, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
                failed = result.returncode != 0
            except subprocess.TimeoutExpired:
                log.write("\nWorker timeout\n")
                failed = True
        path = destination / "check_outputs.json"
        if failed or not path.is_file():
            failures.append(label)
            print(f"[{label}] FAILED; inspect run.log / failure.json", flush=True)
            return None
        suite = json.loads(path.read_text(encoding="utf-8"))
        summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        if summary.get("trace_collected"):
            print(f"[{label}] {summary['complete_batches']} batches, {summary['trace_mib']:.2f} MiB, "
                  f"launch overlap {summary['launch_next_overlaps_forward']}/{summary['adjacent_pairs']}", flush=True)
        print(f"[{label}] output workloads={len(suite['runs'])}; local checks={suite['passed']}", flush=True)
        return suite

    # Both modes share one source and one serial reference.
    failures, completed = [], {}
    reference_suite = None
    reference_path = None
    for mode in modes:
        suite = launch(mode, out / mode, a.checks_only)
        if suite is None:
            continue
        completed[mode] = suite
        if mode == "no_async":
            reference_suite, reference_path = suite, out / "no_async" / "check_outputs.json"
    if "no_async" not in completed and "async" in completed:
        reference_suite = launch("no_async", out / "no_async" / "serial_reference", True)
        if reference_suite is not None:
            reference_path = out / "no_async" / "serial_reference" / "check_outputs.json"
    if reference_suite is not None:
        for mode in modes:
            if mode not in completed:
                continue
            report = consistency_report(mode, completed[mode], reference_suite, reference_path, completed)
            report.update(source=info, compared_modes=list(completed), requested_modes=list(modes),
                          all_requested_modes_completed=len(completed) == len(modes))
            report["passed"] &= report["all_requested_modes_completed"]
            save(out / mode / "consistency.json", report)
            summary_path = out / mode / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update(consistency_passed=report["passed"], scheduling_checks_passed=report["scheduling_checks_passed"])
            save(summary_path, summary)
            print(f"[{mode}] serial reference={report['same_source_tokens']['passed']}; "
                  f"cross-mode tokens={report['all_compared_tokens_equal']}; repeats={report['repeated_outputs']['passed']}", flush=True)
            if not report["passed"]:
                failures.append(mode + "_token_consistency")
    if failures:
        raise RuntimeError("Failed checks: " + ", ".join(failures))


def main():
    a = parse_args()
    try:
        if a.worker:
            run_worker(a)
        else:
            parent(a)
    except Exception:
        if a.worker:
            directory = Path(a.worker_output) if a.worker_output else Path(a.output_dir) / a.worker
            directory.mkdir(parents=True, exist_ok=True)
            save(directory / "failure.json", {"status": "failed", "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
