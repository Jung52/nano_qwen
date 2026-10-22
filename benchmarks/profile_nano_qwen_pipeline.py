#!/usr/bin/env python3
"""Diagnose nano_qwen scheduling and CPU/CUDA synchronization, TP=1.

  CUDA_VISIBLE_DEVICES=0 python bench/profile_nano_qwen_pipeline.py \
      --model /path/to/Qwen3.5-4B --batch-size 8 --input-len 128 \
      --output-len 128 --steps 32 --output-dir logs/nano_pipeline

Copy into bench/ or benchmarks/. --async-scheduling on needs the decode patch;
off also works with the unmodified project and preserves asynchronous output.
Uses the REAL engine.step(), scheduler, runner and sampler; adds only reversible
host annotations. No added per-step synchronize, CUDA events, queries or .item().
Preserves the local queue depth and async-output setting unless explicitly changed.
Warmup, prefill and settling are outside the profile; final drain is labelled.

Outputs: raw.trace.json, timeline.trace.json, sync_calls.csv, pairs.csv,
phases.csv, ticks.csv, queue_journal.csv, metadata.json, summary.json, source_snapshot.txt.
All raw operators and graph-internal kernels are retained. Missing GPU
correlations are UNKNOWN. CPU call duration is never used as GPU duration.
The derived CPU plan spans schedule start to actual GraphLaunch start and
includes preparation/waiting; the raw trace exposes its individual functions.
Forward GPU spans include the graph body plus separately correlated LM head.

CPU-only reanalysis:
  python bench/profile_nano_qwen_pipeline.py --analyze RUN/raw.trace.json

API checked against Jung52/nano_qwen main on 2026-10-22. Your local commit,
source hashes, installed hooks and missing hooks are saved in metadata.
"""

from __future__ import annotations

import argparse
import csv
import functools
import gzip
import hashlib
import inspect
import json
import os
import random
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path


SCOPE = re.compile(r"^nqd/([a-z_]+)/b(\d+)$")
SYNC = re.compile(r"^(?:cuda(?:Event|Stream|Device)Synchronize|cu(?:Event|Stream|Ctx)Synchronize)$")


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def csv_write(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)


def end(e):
    return float(e["ts"]) + float(e.get("dur", 0))


def span(events):
    return [min(float(e["ts"]) for e in events), max(end(e) for e in events)] if events else None


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def union_us(intervals):
    total, right = 0.0, float("-inf")
    for a, b in sorted(intervals):
        total += max(0.0, b - max(a, right))
        right = max(right, b)
    return total


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def find_repo():
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for root in (start, *start.parents):
            if (root / "src/nano_qwen/engine/llm_engine.py").is_file():
                sys.path.insert(0, str(root / "src"))
                return root
    return None


def source_info(repo, objects, outdir):
    def git(*args):
        if repo is None:
            return None
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    files, chunks = {}, []
    for obj in objects:
        name = inspect.getsourcefile(obj)
        if name and name not in files:
            p = Path(name)
            content = p.read_bytes()
            files[name] = hashlib.sha256(content).hexdigest()
            chunks.append(f"\n===== {name} =====\n" + content.decode("utf-8"))
    (outdir / "source_snapshot.txt").write_text("".join(chunks), encoding="utf-8")
    engine_source = inspect.getsource(objects[0])
    return {"root": str(repo) if repo else None, "commit": git("rev-parse", "HEAD"),
            "dirty_status": git("status", "--porcelain"), "file_sha256": files,
            "decode_queue_gate_found_in_source": bool(re.search(r"if\s+decode_batches\s*:", engine_source))}


class Hooks:
    """Host-only instrumentation; preserves original calls, values and exceptions."""

    def __init__(self, torch, engine, output_cls):
        self.torch, self.engine, self.runner = torch, engine, engine.model_runner
        self.output_cls = output_cls
        self.active = False
        self.current = self.planned = self.last_execute = self.last_consumed = None
        self.in_sample = False
        self.tick = -1
        self.rows, self.ticks, self.journal = [], [], []
        self.outputs, self.graph_ids = {}, {}
        self.patches, self.installed, self.missing = [], [], []

    def scope(self, phase, bid=None):
        bid = self.current if bid is None else bid
        if not self.active or bid is None:
            return nullcontext()
        return self.torch.profiler.record_function(f"nqd/{phase}/b{bid:05d}")

    def patch(self, obj, name, fn):
        own = name in getattr(obj, "__dict__", {})
        old = getattr(obj, "__dict__", {}).get(name)
        setattr(obj, name, fn)
        self.patches.append((obj, name, own, old))
        self.installed.append(f"{getattr(obj, '__name__', type(obj).__name__)}.{name}")

    def wrap(self, obj, name, phase):
        fn = getattr(obj, name, None)
        if not callable(fn):
            self.missing.append(f"{type(obj).__name__}.{name}")
            return
        @functools.wraps(fn)
        def call(*args, **kwargs):
            with self.scope(phase):
                return fn(*args, **kwargs)
        self.patch(obj, name, call)

    def snapshot(self, where, bid=None):
        sched, q = self.engine.scheduler, self.engine.batch_queue
        decode = any(any(not s.is_prefill for s in entry[0]) for entry in q)
        row = {"tick": self.tick, "where": where, "batch_id": bid,
               "measured": self.active, "queue_len": len(q), "queued_decode": decode,
               "running": len(sched.running), "waiting": len(sched.waiting),
               "in_flight": len(sched.in_flight)}
        self.journal.append(row)
        return row

    def __enter__(self):
        try:
            self.install()
        except BaseException:
            self.__exit__()
            raise
        return self

    def install(self):
        r, s = self.runner, self.engine.scheduler
        for obj, name in ((self.engine, "step"), (s, "schedule"), (s, "postprocess"),
                          (r, "execute_model"), (r, "sample_tokens"), (r, "prepare_inputs"),
                          (r, "run_model"), (r.model, "compute_logits")):
            if not callable(getattr(obj, name, None)):
                raise RuntimeError(f"Unsupported local API: missing {type(obj).__name__}.{name}")
        original_step = self.engine.step
        def step(*args, **kwargs):
            self.tick += 1
            tick, first = self.tick, len(self.rows)
            before = self.snapshot("tick_begin")
            ctx = self.torch.profiler.record_function(f"nqd/engine_tick/t{tick:05d}") if self.active else nullcontext()
            with ctx:
                result = original_step(*args, **kwargs)
            after = self.snapshot("tick_end")
            self.ticks.append({"tick": tick, "measured": self.active,
                               "dispatches": sum(not b["empty"] for b in self.rows[first:]),
                               "queue_begin": before["queue_len"], "queue_end": after["queue_len"],
                               "queued_decode_begin": before["queued_decode"],
                               "running_begin": before["running"], "in_flight_begin": before["in_flight"]})
            return result
        self.patch(self.engine, "step", step)

        schedule = s.schedule
        def plan(*args, **kwargs):
            bid = len(self.rows)
            self.snapshot("before_schedule", bid)
            old, self.current = self.current, bid
            try:
                with self.scope("schedule", bid):
                    seqs, prefill = schedule(*args, **kwargs)
            finally:
                self.current = old
            self.rows.append({"batch_id": bid, "tick": self.tick, "measured": self.active,
                              "empty": not bool(seqs), "is_prefill": bool(prefill),
                              "batch_size": len(seqs), "seq_ids": [x.seq_id for x in seqs],
                              "context_lengths": [len(x) for x in seqs], "graph_objects": []})
            if seqs:
                self.planned = bid
            self.snapshot("after_schedule", bid)
            return seqs, prefill
        self.patch(s, "schedule", plan)

        execute = r.execute_model
        def execute_model(seqs, is_prefill, *args, **kwargs):
            if self.planned is None:
                raise RuntimeError("execute_model without a corresponding schedule()")
            old, self.current = self.current, self.planned
            self.last_execute, self.planned = self.current, None
            try:
                with self.scope("execute"):
                    return execute(seqs, is_prefill, *args, **kwargs)
            finally:
                self.current = old
        self.patch(r, "execute_model", execute_model)

        for obj, name, phase in [
            (r, "prepare_inputs", "prepare_inputs"), (r, "prepare_decode", "prepare_decode"),
            (r, "prepare_prefill", "prepare_prefill"), (r, "prepare_block_tables", "prepare_block_tables"),
            (r, "prepare_sample", "prepare_temperatures"), (r, "run_model", "forward_call"),
            (r, "project_target_hidden", "project_hidden"), (r.model, "compute_logits", "lm_head"),
            (r.sampler, "forward", "sampler"), (r.cuda_graphs, "run_decode", "graph_decode"),
            (r.input_batch, "remove_finished", "input_remove_finished"),
            (r.input_batch, "update", "input_update"), (r.input_batch, "slots_for", "input_slots"),
            (s, "preempt", "preempt")]:
            self.wrap(obj, name, phase)

        guard = getattr(r, "synchronize_input_prep", None)
        if callable(guard):
            @contextmanager
            def prep_guard(*args, **kwargs):
                cm = guard(*args, **kwargs)
                with self.scope("input_guard_wait"):
                    value = cm.__enter__()
                try:
                    yield value
                except BaseException:
                    info = sys.exc_info()
                    with self.scope("input_guard_record"):
                        handled = cm.__exit__(*info)
                    if not handled:
                        raise
                else:
                    with self.scope("input_guard_record"):
                        cm.__exit__(None, None, None)
            self.patch(r, "synchronize_input_prep", prep_guard)
        else:
            self.missing.append("ModelRunner.synchronize_input_prep")

        replay = self.torch.cuda.CUDAGraph.replay
        def launch(graph, *args, **kwargs):
            if self.current is not None:
                gid = self.graph_ids.setdefault(id(graph), len(self.graph_ids))
                self.rows[self.current]["graph_objects"].append(gid)
            with self.scope("sample_graph_launch" if self.in_sample else "launch"):
                return replay(graph, *args, **kwargs)
        self.patch(self.torch.cuda.CUDAGraph, "replay", launch)

        sampling = r.sample_tokens
        def sample(*args, **kwargs):
            old, old_sample = self.current, self.in_sample
            self.current, self.in_sample = self.last_execute, True
            try:
                with self.scope("sample_call"):
                    out = sampling(*args, **kwargs)
                if out is not None:
                    self.outputs[id(out)] = self.current
                return out
            finally:
                self.current, self.in_sample = old, old_sample
        self.patch(r, "sample_tokens", sample)

        consume = self.output_cls.get_output
        def wait(output, *args, **kwargs):
            bid = self.outputs.get(id(output))
            if bid is None:
                return consume(output, *args, **kwargs)
            self.snapshot("before_result_wait", bid)
            old, self.current = self.current, bid
            try:
                with self.scope("result_wait"):
                    result = consume(output, *args, **kwargs)
            finally:
                self.current = old
            self.outputs.pop(id(output), None)
            self.last_consumed = bid
            self.snapshot("after_result_wait", bid)
            return result
        self.patch(self.output_cls, "get_output", wait)

        postprocess = s.postprocess
        def commit(*args, **kwargs):
            old, self.current = self.current, self.last_consumed
            try:
                with self.scope("commit"):
                    result = postprocess(*args, **kwargs)
                self.snapshot("after_commit", self.current)
                return result
            finally:
                self.current = old
        self.patch(s, "postprocess", commit)

        # Preserve existing trace_event behavior while bridging its named blocks
        # (especially sampler/D2H) into this SAME torch-profiler clock.
        for module in dict.fromkeys([inspect.getmodule(type(r)), inspect.getmodule(type(self.engine))]):
            original = getattr(module, "trace_event", None)
            if callable(original):
                def make_bridge(fn):
                    @contextmanager
                    def bridge(name, *args, **kwargs):
                        label = "src_" + re.sub(r"[^a-z_]+", "_", name.lower())
                        with self.scope(label), fn(name, *args, **kwargs):
                            yield
                    return bridge
                self.patch(module, "trace_event", make_bridge(original))

        # Annotate ORIGINAL CUDA sync calls; never add a new synchronization.
        for obj, name, phase in [(self.torch.cuda.Event, "synchronize", "cuda_event_sync_call"),
                                  (self.torch.cuda.Stream, "synchronize", "cuda_stream_sync_call"),
                                  (self.torch.cuda, "synchronize", "cuda_device_sync_call")]:
            self.wrap(obj, name, phase)

    def __exit__(self, *exc):
        for obj, name, own, old in reversed(self.patches):
            if own:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        self.patches.clear()


def analyze(raw_path, metadata, outdir, slow_us=100.0):
    data = read(raw_path)
    events = data["traceEvents"]
    timed = [e for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]
    scopes, threads = [], defaultdict(list)
    for e in timed:
        m = SCOPE.match(e.get("name", ""))
        if m and e.get("cat") == "user_annotation":
            row = (m[1], int(m[2]), e)
            scopes.append(row)
            threads[e.get("pid"), e.get("tid")].append(row)
    def parents(e):
        return sorted([(p, b, s) for p, b, s in threads[e.get("pid"), e.get("tid")]
                       if s["ts"] <= e["ts"] and end(e) <= end(s) + 0.01], key=lambda v: v[2]["dur"])
    by_corr, launches, sync_rows = defaultdict(list), defaultdict(list), []
    for e in timed:
        cat, name = e.get("cat"), e.get("name", "")
        if cat not in ("cuda_runtime", "cuda_driver", "cpu_op"):
            continue
        ps = parents(e)
        if not ps:
            continue
        labels = {p for p, _, _ in ps}
        phase, bid, _ = ps[0]
        in_sample = "sample_call" in labels
        graph = bool(re.search(r"(?:cuda|cu)GraphLaunch", name))
        if cat in ("cuda_runtime", "cuda_driver"):
            if "lm_head" in labels:
                phase = "lm_head"
            elif in_sample:
                phase = "sample"
            elif graph and "launch" in labels:
                phase = "launch"
                launches[bid].append(e)
            cid = e.get("args", {}).get("correlation")
            if cid not in (None, 0, -1):
                by_corr[str(cid)].append((phase, bid, e))
        kind = ("native_sync" if SYNC.match(name) else
                "copy_api" if cat in ("cuda_runtime", "cuda_driver") and "Memcpy" in name else
                "cpu_copy" if cat == "cpu_op" and name in ("aten::copy_", "aten::_to_copy", "aten::to") else None)
        if kind and (kind == "native_sync" or e["dur"] >= slow_us):
            site = next((p for p, _, _ in ps if not p.startswith("cuda_")), phase)
            sync_rows.append({"batch_id": bid, "kind": kind, "name": name, "site": site,
                              "start_us": e["ts"], "end_us": end(e), "duration_us": e["dur"],
                              "slow": e["dur"] >= slow_us,
                              "parent_scopes": " > ".join(p for p, _, _ in reversed(ps))})
    owned = defaultdict(list)
    kernels = [e for e in timed if e.get("cat") == "kernel"]
    for e in kernels:
        candidates = [r for r in by_corr.get(str(e.get("args", {}).get("correlation")), [])
                      if r[2]["ts"] <= e["ts"] + 2]
        if candidates:
            p, b, _ = max(candidates, key=lambda r: r[2]["ts"])
            owned[b, p].append(e)
    batches = [b for b in metadata["batches"] if b["measured"] and not b["empty"] and not b["is_prefill"]]
    cpu = {(b, p): e for p, b, e in scopes}
    full, graph, plans, gpu = {}, {}, {}, []
    def launch_span(b):
        calls = launches[b]
        calls = [e for e in calls if e["cat"] == "cuda_runtime"] or calls
        return span(calls) if len(calls) == 1 else None
    for row in batches:
        b = row["batch_id"]
        graph[b] = span(owned[b, "launch"])
        head = span(owned[b, "lm_head"])
        full[b] = span(owned[b, "launch"] + owned[b, "lm_head"]) if graph[b] and head else None
        call, sched = launch_span(b), cpu.get((b, "schedule"))
        plans[b] = [sched["ts"], call[0]] if sched and call else None
        for p, value in [("forward", full[b]), ("forward_graph", graph[b]),
                         ("sample", span(owned[b, "sample"]))]:
            if value:
                gpu.append((p, b, value))
    pairs = []
    for prev, nxt in zip(batches, batches[1:]):
        a, b = prev["batch_id"], nxt["batch_id"]
        f, l, p = full[a], launch_span(b), plans[b]
        sched, c = cpu.get((b, "schedule")), cpu.get((a, "commit"))
        delta, shared = (l[0]-f[1], overlap(l, f)) if l and f else (None, None)
        verdict = "UNKNOWN"
        if f and l:
            verdict = ("OVERLAP" if shared > 2 else "AFTER_FORWARD" if delta > 2 else
                       "BEFORE_FORWARD" if l[1] < f[0]-2 else "BOUNDARY_WITHIN_TOLERANCE")
        pairs.append({"batch_n": a, "batch_next": b,
                      "same_request_cohort": set(prev["seq_ids"]) == set(nxt["seq_ids"]),
                      "schedule_next_overlap_forward_us": overlap([sched["ts"], end(sched)], f) if sched and f else None,
                      "plan_next_overlap_forward_us": overlap(p, f) if p and f else None,
                      "launch_next_after_forward_end_us": delta, "launch_next_overlap_forward_us": shared,
                      "launch_next_overlap_graph_us": overlap(l, graph[a]) if l and graph[a] else None,
                      "commit_n_before_next_schedule": end(c) <= sched["ts"] if c and sched else None,
                      "commit_n_overlap_forward_next_us": overlap([c["ts"], end(c)], full[b]) if c and full[b] else None,
                      "verdict": verdict})
    sync_by_site = defaultdict(list)
    for row in sync_rows:
        if row["kind"] == "native_sync":
            sync_by_site[row["site"]].append(row)
    site_stats = []
    for site, rows in sync_by_site.items():
        by_batch = defaultdict(list)
        for row in rows:
            by_batch[row["batch_id"]].append((row["start_us"], row["end_us"]))
        site_stats.append({"site": site, "calls": len(rows),
                           "median_call_us": median([r["duration_us"] for r in rows]),
                           "max_call_us": max(r["duration_us"] for r in rows),
                           "median_per_batch_union_us": median([union_us(v) for v in by_batch.values()])})
    site_stats.sort(key=lambda r: r["median_per_batch_union_us"], reverse=True)
    phase_rows = [{"batch_id": b, "resource": "CPU", "phase": p, "start_us": e["ts"],
                   "end_us": end(e), "duration_us": e["dur"]} for p, b, e in scopes]
    phase_rows += [{"batch_id": b, "resource": "GPU", "phase": p, "start_us": v[0],
                    "end_us": v[1], "duration_us": v[1]-v[0]} for p, b, v in gpu]
    for b, value in plans.items():
        if value:
            phase_rows.append({"batch_id": b, "resource": "CPU", "phase": "plan_including_prepare",
                               "start_us": value[0], "end_us": value[1], "duration_us": value[1]-value[0]})
    csv_write(outdir / "pairs.csv", pairs)
    csv_write(outdir / "phases.csv", phase_rows)
    csv_write(outdir / "sync_calls.csv", sorted(sync_rows, key=lambda r: r["duration_us"], reverse=True))
    csv_write(outdir / "ticks.csv", metadata.get("ticks", []))
    csv_write(outdir / "queue_journal.csv", metadata.get("queue_journal", []))
    pid = max((e.get("pid", 0) for e in events if isinstance(e.get("pid"), int)), default=0) + 100
    added = [{"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
              "args": {"name": "nano_qwen derived ranges; original CPU/GPU events retained"}}]
    number = {b["batch_id"]: i for i, b in enumerate(batches)}
    for p, b, value in gpu + [("plan", b, v) for b, v in plans.items() if v]:
        added.append({"ph": "X", "cat": "nqd_derived_range", "pid": pid,
                      "tid": 3 if p == "plan" else 2 if p == "forward_graph" else 1,
                      "name": f"{p} n={number[b]} (b{b:05d})", "ts": value[0], "dur": value[1]-value[0],
                      "args": {"batch_id": b, "contains_internal_waits_or_gaps": True,
                               "source": "CPU schedule-to-GraphLaunch" if p == "plan" else "correlated CUPTI kernels"}})
    save(outdir / "timeline.trace.json", dict(data, traceEvents=events + added))
    warnings = []
    if not kernels:
        warnings.append("No GPU kernels: no GPU timing conclusion is valid; check CUPTI/PyTorch.")
    if not launches:
        warnings.append("No forward GraphLaunch captured: check graph dispatch and capture size.")
    if not any(full.values()):
        warnings.append("Full forward attribution unavailable; use raw trace, do not infer GPU duration from CPU scopes.")
    if any(b["is_prefill"] for b in metadata["batches"] if b["measured"] and not b["empty"]):
        warnings.append("Prefill appeared inside the measurement window; check preemption/context/KV capacity.")
    if any(b["batch_size"] != metadata.get("batch_size", b["batch_size"]) for b in batches):
        warnings.append("Decode batch size changed; inspect scheduling/KV pressure before comparing rounds.")
    summary = {"decode_batches": len(batches), "pair_verdicts": dict(Counter(r["verdict"] for r in pairs)),
               "cuda_kernels": len(kernels), "forward_graph_launches": sum(map(len, launches.values())),
               "commit_before_next_schedule_pairs": sum(r["commit_n_before_next_schedule"] is True for r in pairs),
               "median_launch_next_after_forward_end_us": median([r["launch_next_after_forward_end_us"] for r in pairs]),
               "native_sync_sites": site_stats, "warnings": warnings,
               "notes": ["plan includes preparation and waiting; it is elapsed time, not useful CPU work.",
                         "Nested CPU scopes/copy durations overlap; do not add them together.",
                         "Long MemcpyAsync/copy_ calls are observations, not proof of their exact synchronization cause.",
                         "Profiler overhead affects timing; this run is diagnostic, not a throughput benchmark."]}
    save(outdir / "summary.json", summary)
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", "--model-path")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--input-len", type=int, default=128)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--steps", "--profile-steps", type=int, default=32)
    p.add_argument("--settle-steps", type=int, default=4)
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--queue-depth", type=int, choices=(1, 2), default=None,
                   help="Default: preserve engine's native queue depth. This is NOT a vLLM async-scheduler switch.")
    p.add_argument("--sync-output", action="store_true", help="Explicit blocking-D2H control; default preserves runner")
    p.add_argument("--async-scheduling", choices=("on", "off"), default="off",
                   help="Requires the async decode patch for on; both cases keep asynchronous D2H output")
    p.add_argument("--temperature", type=float, default=1e-8,
                   help="1e-8 passes legacy SamplingParams validation and selects the current runner's greedy path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--slow-us", type=float, default=100.0)
    p.add_argument("--with-stack", action="store_true", help="Additional stack data; higher profiler overhead")
    p.add_argument("--record-shapes", action="store_true")
    p.add_argument("--output-dir", default="logs/nano_pipeline")
    p.add_argument("--analyze", type=Path)
    p.add_argument("--metadata", type=Path)
    a = p.parse_args(argv)
    if a.async_scheduling == "on" and (a.queue_depth == 1 or a.sync_output):
        p.error("async-scheduling=on requires queue depth 2 and asynchronous output")
    if min(a.batch_size, a.input_len, a.output_len, a.warmup_rounds) < 1 or a.steps < 2 or a.settle_steps < 2:
        p.error("Positive batch/input/output/warmup, steps>=2 and settle-steps>=2 required")
    if not 0 < a.gpu_memory_utilization < 1 or a.slow_us < 0 or a.temperature <= 1e-10:
        p.error("Require 0<gpu-memory-utilization<1, slow-us>=0 and temperature>1e-10")
    if not a.analyze:
        if not a.model or not Path(a.model).is_dir():
            p.error("--model must be an existing local model directory")
        if a.output_len < a.settle_steps + a.steps + 8:
            p.error("output-len must be >= settle-steps + steps + 8")
        if a.max_model_len < a.input_len + a.output_len + 2:
            p.error("max-model-len must exceed input-len + output-len")
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.analyze:
        out = a.analyze.resolve().parent
        meta = read(a.metadata or out / "metadata.json")
        print(json.dumps(analyze(a.analyze, meta, out, a.slow_us), ensure_ascii=False, indent=2))
        return
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") not in ("", "0"):
        raise RuntimeError("Unset CUDA_LAUNCH_BLOCKING before async profiling")
    repo = find_repo()
    import torch
    if not torch.cuda.is_available() or torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError("CUDA GPU and PyTorch CUDA profiler/CUPTI required")
    try:
        from torch.profiler._utils import _init_for_cuda_graphs
    except ImportError:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]):
            torch.empty(1, device="cuda").zero_()
            torch.cuda.synchronize()
    else:
        _init_for_cuda_graphs()
    from nano_qwen.engine.llm_engine import LLMEngine
    from nano_qwen.config import Config
    from dataclasses import fields
    supports_async = "async_scheduling" in {f.name for f in fields(Config)}
    if a.async_scheduling == "on" and not supports_async:
        raise RuntimeError("Local Config has no async_scheduling field; apply the async decode patch first")
    from nano_qwen.engine.async_output import AsyncModelOutput
    from nano_qwen.engine.sequence import Sequence
    from nano_qwen.sampling_params import SamplingParams
    out = Path(a.output_dir).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    torch.manual_seed(a.seed)
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    meta = {"arguments": options, "batch_size": a.batch_size, "batches": [], "status": "loading",
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "added_per_step_cuda_sync": False}
    save(out / "metadata.json", meta)
    async_options = {"async_scheduling": a.async_scheduling == "on"} if supports_async else {}
    with torch.inference_mode():
        engine = LLMEngine(a.model, tensor_parallel_size=1, max_num_seqs=max(8, a.batch_size),
                           max_num_batched_tokens=max(8, a.batch_size)*a.input_len,
                           max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization,
                           enable_prefix_cache=False, enforce_eager=False, use_prefill_cudagraph=False,
                           **async_options)
    try:
        r = engine.model_runner
        if a.queue_depth is not None:
            engine.max_concurrent_batches = a.queue_depth
        if a.sync_output:
            r.async_output = False
        if a.batch_size not in r.graphs:
            raise RuntimeError(f"Exact decode graph missing for B={a.batch_size}; available {sorted(r.graphs)}")
        block = engine.config.kvcache_block_size
        need = a.batch_size*((a.input_len+a.output_len+block-1)//block)
        if engine.config.num_kvcache_blocks < need or engine.config.max_model_len < a.input_len+a.output_len:
            raise RuntimeError("Insufficient KV/context capacity for the fixed cohort")
        meta.update(queue_depth=engine.max_concurrent_batches, async_output=bool(r.async_output),
                    async_scheduling=bool(getattr(engine.config, "async_scheduling", False)),
                    graph_sizes=sorted(r.graphs), source=source_info(repo, [type(engine), type(r),
                    type(engine.scheduler), type(r.cuda_graphs), AsyncModelOutput], out))
        rng = random.Random(a.seed)
        vocab = int(engine.config.hf_config.vocab_size)
        prompts = [[rng.randrange(vocab) for _ in range(a.input_len)] for _ in range(a.batch_size)]
        meta["prompts_sha256"] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
        params = SamplingParams(temperature=a.temperature, max_tokens=a.output_len, ignore_eos=True)
        def enqueue():
            if not engine.is_finished():
                raise RuntimeError("Previous workload not drained")
            seqs = [Sequence(prompt[:], params) for prompt in prompts]
            for seq in seqs:
                engine.scheduler.add(seq)
            return seqs
        def finish():
            limit = max(1000, a.batch_size*(a.input_len+a.output_len+16))
            for _ in range(limit):
                if engine.is_finished():
                    return
                engine.step()
            raise RuntimeError("Engine failed to drain; inspect scheduler/KV state")
        with torch.inference_mode():
            for n in range(a.warmup_rounds):
                print(f"Unprofiled full warmup {n+1}/{a.warmup_rounds}", flush=True)
                enqueue()
                finish()
            torch.cuda.synchronize()
            with Hooks(torch, engine, AsyncModelOutput) as hooks:
                seqs = enqueue()
                for _ in range(a.settle_steps):
                    engine.step()
                if any(s.num_completion_tokens < 1 for s in seqs):
                    raise RuntimeError("Prefill not complete after settling; increase settle-steps")
                torch.cuda.synchronize()  # Profiling entry boundary only.
                opts = dict(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                            with_stack=a.with_stack, record_shapes=a.record_shapes, profile_memory=False)
                try:
                    opts["experimental_config"] = torch._C._profiler._ExperimentalConfig(enable_cuda_sync_events=True)
                except (AttributeError, TypeError):
                    pass
                print(f"Recording {a.steps} real engine steps; queue={engine.max_concurrent_batches}, async_output={r.async_output}", flush=True)
                with torch.profiler.profile(**opts) as prof:
                    hooks.active = True
                    for _ in range(a.steps):
                        if engine.is_finished():
                            raise RuntimeError("Requests ended during measurement; increase output-len")
                        engine.step()
                    with torch.profiler.record_function("nqd/final_drain_NOT_A_STEP"):
                        torch.cuda.synchronize()
                    hooks.active = False
                prof.export_chrome_trace(str(out / "raw.trace.json"))
                meta.update(batches=hooks.rows, ticks=hooks.ticks, queue_journal=hooks.journal,
                            installed_hooks=hooks.installed, missing_optional_hooks=hooks.missing, status="ok")
            save(out / "metadata.json", meta)
            finish()  # Unprofiled tail; no cancellation inserted in measured work.
        summary = analyze(out / "raw.trace.json", meta, out, a.slow_us)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"\nPerfetto: {out / 'timeline.trace.json'}\nBlocking calls: {out / 'sync_calls.csv'}", flush=True)
        if not summary["cuda_kernels"] or not summary["forward_graph_launches"]:
            raise RuntimeError("Trace lacks required CUDA kernels/GraphLaunch; see saved warnings")
    except BaseException as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(out / "metadata.json", meta)
        raise
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
