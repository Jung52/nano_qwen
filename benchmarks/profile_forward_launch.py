#!/usr/bin/env python3
"""Profile nano_qwen's real engine loop without changing its scheduling.

Copy this ONE file into nano_qwen/bench/ (benchmarks/ also works).
Run from the repository root:
  CUDA_VISIBLE_DEVICES=0 python bench/profile_forward_launch.py \
      --model /path/to/Qwen3.5-4B --batch-size 8 --input-len 128 --steps 32

Outputs: raw Chrome trace, annotated trace (all original events retained),
pairs.csv, phases.csv, summary.json, and a measured two-lane timeline.svg.
The raw cudaGraphLaunch/cuGraphLaunch API interval, NOT the host forward
scope, is compared against GPU kernel intervals from the preceding step.
Missing CUPTI correlations yield UNKNOWN; host durations are never used as
GPU durations. LM-head projection outside the graph is reported separately.
No CUDA events, queries, .item(), sleeps or per-step synchronize are added.
Model initialization/capture, full warmup, prefill and settling are unprofiled.
One final synchronize drains the profiled window after all steps are submitted.

Re-analyze a trace + its metadata without importing torch or loading a model:
  python bench/profile_forward_launch.py --analyze logs/launch_test/RUN.raw.trace.json

Source API checked against Jung52/nano_qwen main db22de322bf9 (2026-10-22).
This measures your local checkout, whose revision and dirty state are recorded.
TP=1. A fixed cohort avoids changing batch size, EOS, prefix reuse and preemption.
"""

from __future__ import annotations

import argparse
import csv
import functools
import gzip
import html
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path


PREFIX = "nqp"
SCOPE_RE = re.compile(r"^nqp/([a-z_]+)/b(\d+)$")
CHECKED_COMMIT = "db22de322bf98e4a5707ea95c0d378d7bc6e03ad"


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def find_repo():
    starts = (Path(__file__).resolve().parent, Path.cwd().resolve())
    for start in starts:
        for root in (start, *start.parents):
            if (root / "src/nano_qwen/engine/llm_engine.py").is_file():
                sys.path.insert(0, str(root / "src"))
                return root
    return None  # Also supports an installed/editable nano_qwen package.


def git_info(root):
    if root is None:
        return {"root": None, "commit": None}
    def git(*argv):
        p = subprocess.run(["git", "-C", str(root), *argv], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {"root": str(root), "commit": git("rev-parse", "HEAD"),
            "status": git("status", "--porcelain"), "checked_api_commit": CHECKED_COMMIT}


class Instrumentation:
    """Host-only record_function scopes, installed after graph capture/warmup."""
    def __init__(self, torch, engine, output_cls):
        self.torch, self.engine = torch, engine
        self.runner = engine.model_runner
        self.output_cls = output_cls
        self.patches, self.batches = [], []
        self.outputs = {}
        self.graph_ids = {}
        self.serial = 0
        self.current = self.planned = self.last_execute = self.last_consumed = None

    @contextmanager
    def scope(self, phase, bid=None):
        bid = self.current if bid is None else bid
        if bid is None:
            yield
        else:
            with self.torch.profiler.record_function(f"{PREFIX}/{phase}/b{bid:05d}"):
                yield

    def patch(self, obj, name, replacement):
        existed = name in getattr(obj, "__dict__", {})
        old_local = getattr(obj, "__dict__", {}).get(name)
        self.patches.append((obj, name, existed, old_local))
        setattr(obj, name, replacement)

    def __enter__(self):
        r, sched = self.runner, self.engine.scheduler
        required = ((r, "execute_model"), (r, "prepare_inputs"), (r, "run_model"),
                    (r, "sample_tokens"), (sched, "schedule"), (sched, "postprocess"))
        for obj, name in required:
            if not callable(getattr(obj, name, None)):
                raise RuntimeError(f"Unsupported local API: missing {type(obj).__name__}.{name}")
        schedule = sched.schedule
        def plan(*args, **kwargs):
            bid = self.serial
            self.serial += 1  # Empty attempts have distinct IDs too.
            with self.scope("plan", bid):
                seqs, prefill = schedule(*args, **kwargs)
            row = {"batch_id": bid, "is_prefill": bool(prefill), "empty": not bool(seqs),
                   "seq_ids": [s.seq_id for s in seqs], "batch_size": len(seqs),
                   "replays": []}
            self.batches.append(row)
            if seqs:
                self.planned = bid
            return seqs, prefill
        self.patch(sched, "schedule", plan)

        execute = r.execute_model
        def execute_model(seqs, is_prefill, *args, **kwargs):
            if self.planned is None:
                raise RuntimeError("execute_model was called without a measured schedule")
            bid, old = self.planned, self.current
            self.planned, self.current = None, bid
            self.last_execute = bid
            row = self.batches[bid]
            row["actual_is_prefill"] = bool(is_prefill)
            row["context_lengths"] = [len(s) for s in seqs]
            try:
                with self.scope("execute", bid):
                    return execute(seqs, is_prefill, *args, **kwargs)
            finally:
                self.current = old
        self.patch(r, "execute_model", execute_model)

        for obj, name, phase in ((r, "prepare_inputs", "prepare"),
                                  (r, "run_model", "forward_submit"),
                                  (r.model, "compute_logits", "lm_head")):
            original = getattr(obj, name)
            def make_wrapper(fn, label):
                @functools.wraps(fn)
                def wrapper(*args, **kwargs):
                    with self.scope(label):
                        return fn(*args, **kwargs)
                return wrapper
            self.patch(obj, name, make_wrapper(original, phase))

        # CUDAGraph.replay is a Python method; wrap the actual graph object
        # rather than the broader run_decode() that also stages static inputs.
        original_replay = self.torch.cuda.CUDAGraph.replay
        def replay(graph, *args, **kwargs):
            if self.current is None:
                return original_replay(graph, *args, **kwargs)
            gid = self.graph_ids.setdefault(id(graph), len(self.graph_ids))
            self.batches[self.current]["replays"].append({"graph_object": gid})
            with self.scope("launch"):
                return original_replay(graph, *args, **kwargs)
        self.patch(self.torch.cuda.CUDAGraph, "replay", replay)

        sample = r.sample_tokens
        def sample_tokens(*args, **kwargs):
            bid, old = self.last_execute, self.current
            self.current = bid
            try:
                with self.scope("sample", bid):
                    out = sample(*args, **kwargs)
                if out is not None:
                    self.outputs[id(out)] = bid
                return out
            finally:
                self.current = old
        self.patch(r, "sample_tokens", sample_tokens)

        get_output = self.output_cls.get_output
        def consume(output, *args, **kwargs):
            bid = self.outputs.pop(id(output), None)
            if bid is None:
                return get_output(output, *args, **kwargs)
            with self.scope("commit_wait", bid):
                result = get_output(output, *args, **kwargs)
            self.last_consumed = bid
            return result
        self.patch(self.output_cls, "get_output", consume)

        postprocess = sched.postprocess
        def commit(*args, **kwargs):
            bid = self.last_consumed
            with self.scope("commit", bid):
                return postprocess(*args, **kwargs)
        self.patch(sched, "postprocess", commit)
        return self

    def __exit__(self, *exc):
        for obj, name, existed, old in reversed(self.patches):
            if existed:
                setattr(obj, name, old)
            else:
                delattr(obj, name)


def event_end(e):
    return float(e["ts"]) + float(e.get("dur", 0))


def enclosing(scope, event):
    return (scope.get("pid") == event.get("pid") and scope.get("tid") == event.get("tid")
            and scope["ts"] <= event["ts"] and event_end(event) <= event_end(scope) + 0.01)


def correlation(e):
    for key, value in e.get("args", {}).items():
        if key.lower().replace("_", " ") in ("correlation", "correlation id"):
            return str(value) if str(value) not in ("0", "-1", "None") else None
    return None


def interval(events):
    if not events:
        return None
    return [min(float(e["ts"]) for e in events), max(event_end(e) for e in events)]


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def draw_svg(path, cpu, gpu, batches, draw_steps):
    """Exact shared time axis. Never stretch short launches for legibility."""
    chosen = [b for b in batches if not b.get("empty") and not b["is_prefill"]][:draw_steps]
    ids = {b["batch_id"] for b in chosen}
    c = [(p, bid, e) for p, bid, e in cpu if bid in ids
         and p in ("plan", "prepare", "launch_api", "commit_wait", "commit")]
    g = [(p, bid, span) for p, bid, span in gpu if bid in ids]
    if not c:
        return
    points = [(e["ts"], event_end(e)) for _, _, e in c] + [tuple(s) for _, _, s in g]
    lo, hi = min(a for a, _ in points), max(b for _, b in points)
    width, left, plot = 1800, 150, 1600
    scale = plot / max(hi - lo, 1)
    colors = {"plan": "#5ea6ff", "prepare": "#9ba9bf", "launch_api": "#ff9a4a",
              "commit_wait": "#814854", "commit": "#edcf62", "forward": "#31bf80",
              "forward_graph": "#31bf80", "sample": "#ac84ed"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="390" viewBox="0 0 {width} 390">',
           '<rect width="100%" height="100%" fill="#11151c"/>',
           '<style>text{fill:#e5eaf3;font:15px sans-serif}</style>',
           '<text x="25" y="32" font-size="23">nano_qwen: measured CPU launch vs GPU forward</text>',
           '<text x="25" y="57">Shared trace clock; widths preserve measured durations; hover for exact intervals.</text>',
           '<text x="25" y="122">CPU</text>', '<text x="25" y="222">GPU spans</text>']
    for i in range(9):
        x = left + plot * i / 8
        out += [f'<path d="M{x} 78 V255" stroke="#354052" stroke-dasharray="3 5"/>',
                f'<text x="{x}" y="280" text-anchor="middle">{(hi-lo)*i/8000:.3f} ms</text>']
    n_by_id = {b["batch_id"]: n for n, b in enumerate(chosen)}
    def block(phase, bid, a, b, y):
        x, w = left + (a-lo)*scale, (b-a)*scale
        label = f'{phase} n={n_by_id[bid]}'
        tooltip = html.escape(f'{label}; start={a-lo:.3f} us; end={b-lo:.3f} us; duration={b-a:.3f} us')
        color = colors[phase]
        if w < 0.8:
            out.append(f'<path d="M{x:.3f} {y} v34" stroke="{color}"><title>{tooltip}</title></path>')
        else:
            out.append(f'<rect x="{x:.3f}" y="{y}" width="{w:.3f}" height="34" fill="{color}" fill-opacity="0.55" stroke="{color}"><title>{tooltip}</title></rect>')
        if w > max(62, len(label)*7):
            out.append(f'<text x="{x+w/2:.3f}" y="{y+22}" text-anchor="middle">{label}</text>')
    for p, bid, e in c:
        block(p, bid, float(e["ts"]), event_end(e), 100)
    for p, bid, s in g:
        block(p, bid, *s, 200)
    out += ['<text x="25" y="320">CPU: blue plan / gray prepare / orange actual GraphLaunch / red D2H wait / yellow commit</text>',
            '<text x="25" y="345">GPU: green forward (graph + LM head when correlated) / purple sample. Raw kernels remain in the trace.</text>',
            '<text x="25" y="370">Diagnostic profiler run, not unprofiled throughput. Missing GPU correlations are not inferred from CPU scopes.</text>', '</svg>']
    Path(path).write_text("\n".join(out), encoding="utf-8")


def analyze(trace_path, meta, stem, draw_steps=8, tolerance_us=2.0):
    data = read_json(trace_path)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    timed = [e for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]
    scopes, cpu = [], []
    for e in timed:
        m = SCOPE_RE.match(e.get("name", ""))
        if m:
            phase, bid = m.group(1), int(m.group(2))
            scopes.append((phase, bid, e))
            cpu.append((phase, bid, e))
    thread_scopes = defaultdict(list)
    for p, b, e in scopes:
        thread_scopes[e.get("pid"), e.get("tid")].append((p, b, e))
    apis = [e for e in timed if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    kernels = [e for e in timed if e.get("cat") == "kernel"]
    graph_apis, launches_by_id, correlated = defaultdict(list), defaultdict(list), defaultdict(list)
    for api in apis:
        parents = [(p, b, s) for p, b, s in thread_scopes[api.get("pid"), api.get("tid")]
                   if enclosing(s, api)]
        if not parents:
            continue
        phase, bid, _ = min(parents, key=lambda p: p[2]["dur"])
        if re.search(r"(?:cuda|cu)GraphLaunch", api.get("name", ""), re.I):
            graph_apis[bid].append(api)
            cpu.append(("launch_api", bid, api))
            phase = "launch"
        cid = correlation(api)
        if cid is not None:
            launches_by_id[cid].append((phase, bid, api))
    ambiguous = 0
    for kernel in kernels:
        cid = correlation(kernel)
        candidates = [(p, b, a) for p, b, a in launches_by_id.get(cid, [])
                      if a["ts"] <= kernel["ts"] + tolerance_us]
        # IDs can be reused; prefer the latest preceding API, never a later one.
        if not candidates:
            continue
        latest = max(float(a["ts"]) for _, _, a in candidates)
        candidates = [(p, b, a) for p, b, a in candidates if a["ts"] >= latest-0.01]
        owners = {(p, b) for p, b, _ in candidates}
        if len(owners) != 1:
            ambiguous += 1
            continue
        phase, bid = next(iter(owners))
        correlated[bid, phase].append(kernel)

    batches = [b for b in meta.get("batches", []) if not b.get("empty") and not b["is_prefill"]]
    by_id = {b["batch_id"]: b for b in batches}
    windows, graph_windows, gpu, measurements = {}, {}, [], []
    for b in batches:
        bid = b["batch_id"]
        body = interval(correlated[bid, "launch"])
        head = interval(correlated[bid, "lm_head"])
        has_head = any(p == "lm_head" and ident == bid for p, ident, _ in scopes)
        full = interval(correlated[bid, "launch"] + correlated[bid, "lm_head"]) if body and (head or not has_head) else None
        sample = interval(correlated[bid, "sample"])
        graph_windows[bid], windows[bid] = body, full
        if full:
            gpu.append(("forward", bid, full))
        elif body:
            gpu.append(("forward_graph", bid, body))
        if sample:
            gpu.append(("sample", bid, sample))
        measurements.append({"batch_id": bid, "graph_kernels": len(correlated[bid, "launch"]),
                             "lm_head_kernels": len(correlated[bid, "lm_head"]),
                             "sample_kernels": len(correlated[bid, "sample"]),
                             "graph_gpu_us": body, "lm_head_gpu_us": head,
                             "forward_gpu_us": full, "sample_gpu_us": sample,
                             "actual_graph_launch_api_count": len(graph_apis[bid])})

    pairs = []
    for n, (prev, nxt) in enumerate(zip(batches, batches[1:])):
        a, b = prev["batch_id"], nxt["batch_id"]
        raw_calls = graph_apis[b]
        # Some exporters emit nested runtime and driver APIs. Use runtime when
        # available; never union them into an artificially longer CPU launch.
        calls = [e for e in raw_calls if e.get("cat") == "cuda_runtime"] or raw_calls
        launch = interval(calls) if len(calls) == 1 else None
        fwd, body = windows.get(a), graph_windows.get(a)
        row = {"n": n, "batch_n": a, "batch_n_plus_1": b,
               "same_request_cohort": prev["seq_ids"] == nxt["seq_ids"],
               "launch_next_start_us": launch[0] if launch else None,
               "launch_next_end_us": launch[1] if launch else None,
               "launch_duration_us": launch[1]-launch[0] if launch else None,
               "forward_n_start_us": fwd[0] if fwd else None,
               "forward_n_end_us": fwd[1] if fwd else None,
               "graph_n_end_us": body[1] if body else None,
               "launch_after_graph_end_us": launch[0]-body[1] if launch and body else None,
               "launch_after_forward_end_us": None,
               "launch_overlap_forward_us": None,
               "launch_uncovered_by_forward_n_us": None,
               "launch_overlap_graph_us": overlap(launch, body) if launch and body else None,
               "verdict": "UNKNOWN"}
        if launch and fwd:
            delta = launch[0] - fwd[1]
            shared = overlap(launch, fwd)
            row.update(launch_after_forward_end_us=delta, launch_overlap_forward_us=shared,
                       launch_uncovered_by_forward_n_us=launch[1]-launch[0]-shared)
            row["verdict"] = ("OVERLAP" if shared > tolerance_us else
                              "AFTER_FORWARD" if delta > tolerance_us else
                              "BEFORE_FORWARD" if launch[1] < fwd[0]-tolerance_us else
                              "BOUNDARY_WITHIN_TOLERANCE")
        pairs.append(row)

    annotated = dict(data) if isinstance(data, dict) else {"traceEvents": list(events)}
    annotated["traceEvents"] = list(events)  # Preserve EVERY raw event unchanged.
    ann = annotated["traceEvents"]
    pid = max((e.get("pid", 0) for e in events if isinstance(e.get("pid"), int)), default=0) + 100
    ann += [{"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
             "args": {"name": "nano_qwen: measured GPU spans (derived from CUPTI kernels)"}},
            {"ph": "M", "name": "thread_name", "pid": pid, "tid": 1,
             "args": {"name": "forward / sample GPU spans; native kernels retained"}}]
    decode_n = {b["batch_id"]: n for n, b in enumerate(batches)}
    for phase, bid, span in gpu:
        ann.append({"ph": "X", "cat": "nqp_measured_gpu_span", "pid": pid, "tid": 1,
                    "name": f"{phase} n={decode_n[bid]} (b{bid:05d})", "ts": span[0],
                    "dur": span[1]-span[0], "args": {"source": "min/max correlated kernel timestamps",
                    "contains_internal_gaps": True, "batch_id": bid}})
    dump_json(str(stem)+".timeline.trace.json", annotated)
    write_csv(str(stem)+".pairs.csv", pairs)
    phase_rows = [{"phase": p, "batch_id": b, "decode_n": decode_n.get(b),
                   "start_us": e["ts"], "end_us": event_end(e), "duration_us": e["dur"],
                   "clock": "Chrome trace CPU/GPU common clock"}
                  for p, b, e in cpu if b in by_id]
    phase_rows += [{"phase": p+"_gpu", "batch_id": b, "decode_n": decode_n[b],
                    "start_us": s[0], "end_us": s[1], "duration_us": s[1]-s[0],
                    "clock": "Chrome trace CPU/GPU common clock"} for p, b, s in gpu]
    write_csv(str(stem)+".phases.csv", phase_rows)
    draw_svg(str(stem)+".timeline.svg", cpu, gpu, batches, draw_steps)
    counts = dict(Counter(p["verdict"] for p in pairs))
    warnings = []
    if not kernels:
        warnings.append("NO GPU KERNELS: no timing conclusion is valid. Check CUPTI/profiler installation.")
    if not any(graph_apis.values()):
        warnings.append("No cudaGraphLaunch/cuGraphLaunch API captured: verify graph path and CUPTI.")
    if not any(graph_windows.values()):
        warnings.append("No graph kernels linked to replay APIs. GPU attribution is UNKNOWN; raw trace is retained.")
    if any(m["forward_gpu_us"] is None for m in measurements):
        warnings.append("Some complete forward spans are missing; partial graph-only spans are labeled separately.")
    if any(len(b.get("replays", [])) != 1 for b in batches):
        warnings.append("Not exactly one graph replay per decode. Inspect local path before interpreting this as single-graph decode.")
    graph_objects = sorted({r["graph_object"] for b in batches for r in b.get("replays", [])})
    if len(graph_objects) != 1:
        warnings.append(f"Observed {len(graph_objects)} graph objects in the decode window; not one reused graph.")
    summary = {"trace": str(trace_path), "metadata": meta, "verdict_counts": counts,
               "tolerance_us": tolerance_us, "device_kernel_count": len(kernels),
               "graph_objects": graph_objects, "ambiguous_kernel_owners": ambiguous,
               "measurements": measurements, "pairs": pairs, "warnings": warnings,
               "notes": ["CPU record_function scopes are not GPU durations.",
                         "launch means actual cudaGraphLaunch/cuGraphLaunch, not run_model wall time.",
                         "forward includes captured model and separately correlated LM head; static input staging is excluded.",
                         "Derived GPU spans include gaps between their first/last kernels, not sums of kernel times.",
                         "Only direct CUPTI correlation IDs are used. Missing data remains UNKNOWN.",
                         "commit is scheduler.postprocess; commit_wait separately includes D2H event wait and token tolist.",
                         "The profiled window can be perturbed by profiler/Python annotation overhead."]}
    dump_json(str(stem)+".summary.json", summary)
    print("\nVerdicts:", counts, flush=True)
    values = [p["launch_after_forward_end_us"] for p in pairs if p["launch_after_forward_end_us"] is not None]
    if values:
        print(f"Median launch(n+1).start - forward(n).end: {statistics.median(values):.3f} us")
    for warning in warnings:
        print("WARNING:", warning, flush=True)
    print(f"Open: {stem}.timeline.trace.json\nPlot: {stem}.timeline.svg\nPairs: {stem}.pairs.csv", flush=True)
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", "--model-path")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--input-len", type=int, default=128)
    p.add_argument("--steps", type=int, default=32, help="Measured decode steps")
    p.add_argument("--settle-steps", type=int, default=3, help="Unprofiled decode steps after prefill")
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--queue-depth", type=int, choices=(1, 2), default=2)
    p.add_argument("--sync-output", action="store_true", help="Explicit blocking-D2H control, changes runner behavior")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="logs/launch_test")
    p.add_argument("--draw-steps", type=int, default=8)
    p.add_argument("--tolerance-us", type=float, default=2.0)
    p.add_argument("--record-shapes", action="store_true")
    p.add_argument("--with-stack", action="store_true")
    p.add_argument("--analyze", type=Path, help="Re-analyze a raw trace without torch/GPU")
    p.add_argument("--metadata", type=Path, help="Optional metadata JSON for --analyze")
    a = p.parse_args(argv)
    if min(a.batch_size, a.input_len, a.warmup_rounds, a.draw_steps) < 1 or a.steps < 2 or a.settle_steps < 0:
        p.error("Positive batch/input/warmup/draw values, steps>=2, settle-steps>=0 required")
    if not 0 < a.gpu_memory_utilization < 1 or a.tolerance_us < 0:
        p.error("0 < gpu-memory-utilization < 1 and tolerance-us >= 0 required")
    if not a.analyze:
        if not a.model or not Path(a.model).is_dir():
            p.error("--model must be an existing local model directory")
        if a.input_len + a.settle_steps + a.steps + 1 > a.max_model_len:
            p.error("max-model-len must cover input-len + settle-steps + steps + 1")
    return a


def main(argv=None):
    args = parse_args(argv)
    if args.analyze:
        name = str(args.analyze)
        stem = Path(re.sub(r"\.raw\.trace\.json(?:\.gz)?$", "", name))
        if str(stem) == name:
            stem = args.analyze.with_suffix("")
        meta_path = args.metadata or Path(str(stem)+".metadata.json")
        analyze(args.analyze, read_json(meta_path), stem, args.draw_steps, args.tolerance_us)
        return

    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") not in ("", "0"):
        raise RuntimeError("Unset CUDA_LAUNCH_BLOCKING: it forces synchronous launches and invalidates this experiment")
    repo = find_repo()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; run inside your nano_qwen environment")
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError("CUDA profiler activities unavailable; check PyTorch/CUPTI")
    # Initialize Kineto before graph capture. Documented CUDA Graph workaround;
    # never toggle or start profiling inside a captured graph.
    try:
        from torch.profiler._utils import _init_for_cuda_graphs
    except ImportError:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]):
            torch.empty(1, device="cuda").zero_()
            torch.cuda.synchronize()
    else:
        _init_for_cuda_graphs()

    from nano_qwen.engine.llm_engine import LLMEngine
    from nano_qwen.engine.sequence import Sequence
    from nano_qwen.engine.async_output import AsyncModelOutput
    from nano_qwen.sampling_params import SamplingParams

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    total_decode = args.settle_steps + args.steps
    length = total_decode + 1
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"bs{args.batch_size}_i{args.input_len}_q{args.queue_depth}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata = {"args": options, "source": git_info(repo), "torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "tp": 1,
                "checked_api_commit": CHECKED_COMMIT, "added_per_step_cuda_sync": False,
                "batches": [], "status": "initializing"}
    print(f"Loading {args.model}; B={args.batch_size}, input={args.input_len}, decode={total_decode}, TP=1", flush=True)
    with torch.inference_mode():
        engine = LLMEngine(args.model, tensor_parallel_size=1,
                           max_num_seqs=max(8, args.batch_size),
                           max_num_batched_tokens=max(8, args.batch_size)*args.input_len,
                           max_model_len=args.max_model_len,
                           gpu_memory_utilization=args.gpu_memory_utilization,
                           enable_prefix_cache=False, enforce_eager=False,
                           use_prefill_cudagraph=False,
                           async_scheduling=args.queue_depth == 2)
    try:
        engine.max_concurrent_batches = args.queue_depth
        if bool(getattr(engine.config, "async_scheduling", False)) != (args.queue_depth == 2):
            raise RuntimeError("queue_depth did not map to async_scheduling; refusing a silent serial fallback")
        runner = engine.model_runner
        if not hasattr(runner, "async_output"):
            raise RuntimeError("Local runner has no async_output setting; inspect API before profiling")
        runner.async_output = not args.sync_output
        if args.batch_size not in runner.graphs:
            raise RuntimeError(f"BS={args.batch_size} has no exact decode graph. Available: {sorted(runner.graphs)}")
        block = engine.config.kvcache_block_size
        needed = args.batch_size*((args.input_len+length+block-1)//block)
        if engine.config.max_model_len < args.input_len+length or engine.config.num_kvcache_blocks < needed:
            raise RuntimeError("Allocated context/KV pool cannot hold the workload")
        def reject_preempt(seq):
            raise RuntimeError(f"Preemption would invalidate the fixed cohort: seq {seq.seq_id}")
        engine.scheduler.preempt = reject_preempt

        def enqueue(seed):
            if not engine.is_finished():
                raise RuntimeError("Previous workload did not drain")
            rng = random.Random(seed)
            vocab = int(engine.config.hf_config.vocab_size)
            seqs = [Sequence([rng.randrange(vocab) for _ in range(args.input_len)],
                             SamplingParams(temperature=args.temperature, max_tokens=length, ignore_eos=True))
                    for _ in range(args.batch_size)]
            for seq in seqs:
                engine.scheduler.add(seq)
            return seqs

        def check_end(seqs):
            if not engine.is_finished() or any(s.num_completion_tokens != length for s in seqs):
                raise RuntimeError("Fixed-batch/one-step assumption violated; inspect local scheduler")

        with torch.inference_mode():
            for n in range(args.warmup_rounds):
                print(f"Full unprofiled warmup {n+1}/{args.warmup_rounds}", flush=True)
                seqs = enqueue(args.seed+n)
                for _ in range(total_decode+1):
                    engine.step()
                torch.cuda.synchronize()
                check_end(seqs)
            seqs = enqueue(args.seed+args.warmup_rounds)
            engine.step()  # Unprofiled prefill.
            if any(s.num_completion_tokens != 1 for s in seqs):
                raise RuntimeError("Prefill did not consume the fixed cohort in one step")
            for _ in range(args.settle_steps):
                engine.step()
            torch.cuda.synchronize()  # One clean profiling boundary, not inside the step loop.
            metadata.update(status="profiling", graph_sizes=sorted(runner.graphs))
            kwargs = dict(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                          record_shapes=args.record_shapes, with_stack=args.with_stack, profile_memory=False)
            try:
                kwargs["experimental_config"] = torch.profiler._ExperimentalConfig(enable_cuda_sync_events=True)
            except (AttributeError, TypeError):
                pass  # CUDA sync runtime calls are still visible in ordinary runtime activity.
            print("Recording real engine.step(); no extra per-step device barriers", flush=True)
            with Instrumentation(torch, engine, AsyncModelOutput) as hooks:
                with torch.profiler.profile(**kwargs) as prof:
                    for n in range(args.steps):
                        with torch.profiler.record_function(f"nqp/engine_tick/{n:05d}"):
                            engine.step()
                    with torch.profiler.record_function("nqp/final_drain_NOT_A_STEP"):
                        torch.cuda.synchronize()
                metadata["batches"] = hooks.batches
                metadata["observed_graph_objects"] = len(hooks.graph_ids)
            check_end(seqs)
            metadata["status"] = "ok"
            raw_path = Path(str(stem)+".raw.trace.json")
            prof.export_chrome_trace(str(raw_path))
            dump_json(str(stem)+".metadata.json", metadata)
            print(f"Raw trace: {raw_path}", flush=True)
            analyze(raw_path, metadata, stem, args.draw_steps, args.tolerance_us)
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
