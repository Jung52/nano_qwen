#!/usr/bin/env python3
"""Run 3 FULL B8/I128/O256 requests per mode; sample bounded CPU/CUDA windows.

Place in nano_qwen/benchmarks. Requires the dual-slot branch and its existing
bench_fp8_pressure.py workload-KV helper. Default: decode CUDA Graph, eager
prefill, greedy sampling, TP=1. Default: record 16 steady decode steps in each
round, with 2 guard steps on each side; run all 256 output tokens to completion.
Write detail + overview .trace.json.gz files per mode, with GPU slot ranges.
Also time three separate full runs WITHOUT profiler. --full-trace records
entire workloads; default bounded windows keep profiler memory manageable.
--compact FILE... reduces existing traces without loading torch or running GPU.
"""
from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager, nullcontext
from collections import Counter
from pathlib import Path

from slot_trace import APIS, FLOWS, PHYSICAL, scan_slot_groups

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
BATCH, INPUT, OUTPUT, ROUNDS = 8, 128, 256, 3
CONTEXT = INPUT + OUTPUT


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@contextmanager
def labels(engine, torch):
    """Restore all temporary labels, including the shared get_output method."""
    from nano_qwen.engine.async_output import AsyncModelOutput
    r, pipeline = engine.model_runner, engine._dual_pipeline
    with ExitStack() as undo:
        def patch(obj, name, value):
            own, old = name in vars(obj), getattr(obj, name)
            setattr(obj, name, value)
            undo.callback(setattr if own else delattr, obj, name, *([old] if own else []))

        def wrap(obj, name, label):
            original = getattr(obj, name)
            @functools.wraps(original)
            def call(*args, **kwargs):
                title = label(*args, **kwargs) if callable(label) else label
                with torch.profiler.record_function(title):
                    return original(*args, **kwargs)
            patch(obj, name, call)

        def slot():
            return f"/slot{r._active_slot_id if pipeline is not None else 0}"

        # Reuse existing prepare/forward/sampler/copy scopes in Kineto's clock.
        for module in (sys.modules[type(r).__module__], sys.modules[type(engine).__module__]):
            original = module.trace_event
            @contextmanager
            def bridge(name, cat="bench", args=None, original=original):
                name_map = {"run_model": "forward", "d2h_copy": "d2h_enqueue"}
                with torch.profiler.record_function("nq/cpu/" + name_map.get(name, name) + slot()):
                    with original(name, cat, args):
                        yield
            patch(module, "trace_event", bridge)
        wrap(AsyncModelOutput, "get_output", "nq/cpu/get_output_wait")
        wrap(r.model, "compute_logits", lambda *a, **kw: "nq/cpu/lm_head" + slot())
        if pipeline is None:
            wrap(engine.scheduler, "postprocess", "nq/cpu/commit/slot0")
        else:
            wrap(pipeline, "_plan_decode", "nq/cpu/schedule")
            wrap(pipeline, "_commit_oldest", lambda: f"nq/cpu/commit/slot{pipeline.pending[0].slot_id}")
            wrap(r, "launch_slot", lambda seqs, prefill, slot_id: f"nq/cpu/launch_forward/slot{slot_id}")
            wrap(r, "sample_slot", lambda slot_id, ids: f"nq/cpu/sample_slot/slot{slot_id}")
        yield


def run_batch(engine, torch, seed, annotate, capture=None, window=None, return_tokens=False):
    from nano_qwen.engine.sequence import Sequence
    from nano_qwen.sampling_params import SamplingParams
    rng = random.Random(seed)
    prompts = [[rng.randrange(engine.config.hf_config.vocab_size) for _ in range(INPUT)] for _ in range(BATCH)]
    seqs = [Sequence(row, SamplingParams(temperature=1e-6, max_tokens=OUTPUT, ignore_eos=True)) for row in prompts]
    mark = torch.profiler.record_function if annotate else lambda name: nullcontext()
    if not engine.is_finished():
        raise RuntimeError("Previous round has not drained")
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    engine.model_runner.cuda_graphs.reset_dispatch_stats()
    started = time.perf_counter()
    with mark("nq/prefill"):
        for seq in seqs:
            engine.scheduler.add(seq)
        engine.step()
        first_lengths = [s.num_completion_tokens for s in seqs]
        torch.cuda.synchronize()
    prefill_end = time.perf_counter()
    ticks = 0
    def step(record=False):
        nonlocal ticks
        with torch.profiler.record_function(f"nq/tick/{ticks:03d}") if record else nullcontext():
            engine.step()
        ticks += 1

    def advance(target, record=False):
        while seqs[0].num_completion_tokens < target:
            step(record)
        if any(s.num_completion_tokens != target for s in seqs):
            raise RuntimeError("Requests advanced to different lengths")

    if capture is not None:
        start, length = window
        # Counts refer to COMMITTED decode tokens, so both modes use the same
        # context lengths despite dual-slot prime/drain host ticks.
        advance(1 + start - 2)
        torch.cuda.synchronize()
        with capture():
            with torch.profiler.record_function("nq/window/leading_guard"):
                advance(1 + start, True)
            with torch.profiler.record_function("nq/window/focus"):
                advance(1 + start + length, True)
            with torch.profiler.record_function("nq/window/trailing_guard"):
                advance(1 + start + length + 2, True)
            torch.cuda.synchronize()
        # Profiling never truncates generation or abandons in-flight work.
        while not engine.is_finished():
            step()
        torch.cuda.synchronize()
    else:
        with mark("nq/decode"):
            while not engine.is_finished():
                step(annotate)
            torch.cuda.synchronize()  # Phase boundary only; never synchronize every tick.
    ended = time.perf_counter()
    if first_lengths != [1]*BATCH or any(not s.is_finished or s.num_completion_tokens != OUTPUT for s in seqs):
        raise RuntimeError("Expected one complete prefill and 256 output tokens per request")
    r, p = engine.model_runner, engine._dual_pipeline
    if r.input_batch.seq_id_to_slot or engine.scheduler.block_manager.used_block_ids:
        raise RuntimeError("Persistent slots/KV pages did not drain")
    if p is not None and (p.pending or p.requests or any(s.busy for s in r.decode_slots)):
        raise RuntimeError("Dual execution slots did not drain")
    stats = r.cuda_graphs.dispatch_stats()
    dec = stats.get("decode", {})
    expected_kind = "eager_calls" if r.enforce_eager else "graph_calls"
    if dec.get(expected_kind) != OUTPUT-1 or dec.get("batch_counts") != {BATCH: OUTPUT-1}:
        raise RuntimeError(f"Unexpected decode dispatch/batch: {dec}")
    hash_json = lambda x: hashlib.sha256(json.dumps(x).encode()).hexdigest()
    result = dict(seed=seed, output_tokens=BATCH*OUTPUT, decode_tokens=BATCH*(OUTPUT-1),
                host_decode_ticks=ticks, dispatch=stats, prompt_sha256=hash_json(prompts),
                output_sha256=hash_json([s.completion_token_ids for s in seqs]))
    if capture is None and not annotate:
        dec, e2e = ended-prefill_end, ended-started
        result.update(ttft_ms=(prefill_end-started)*1000, decode_ms=dec*1000,
            tpot_ms=dec*1000/(OUTPUT-1), decode_tok_s=BATCH*(OUTPUT-1)/dec,
            e2e_ms=e2e*1000, output_tok_s=BATCH*OUTPUT/e2e)
    if return_tokens:
        result["token_ids"] = [s.completion_token_ids for s in seqs]
    return result


def trace_records(path):
    """Stream a Chrome JSON object; reject truncated traces, including footer.

    Memory is bounded by one event plus a 1MiB read buffer, not the file size.
    Yields header first, then events; works for raw and gzip-compressed input.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    decoder = json.JSONDecoder()
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        buffer = ""
        while True:
            chunk = stream.read(1024*1024)
            if not chunk:
                raise ValueError(f"No traceEvents array (possibly truncated): {path}")
            buffer += chunk
            match = re.search(r'"traceEvents"\s*:\s*\[', buffer)
            if match:
                prefix = buffer[:match.end()-1]
                header = json.loads(prefix + "[]}")
                header.pop("traceEvents", None)
                yield "header", header
                buffer = buffer[match.end():]
                break
            if len(buffer) > 2*1024*1024:
                raise ValueError("Trace header exceeds 2MiB")
        position = 0
        need_comma = after_comma = False
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position == len(buffer):
                buffer = stream.read(1024*1024)
                position = 0
                if not buffer:
                    raise ValueError(f"Truncated traceEvents array: {path}")
                continue
            if buffer[position] == "]":
                if after_comma:
                    raise ValueError(f"Trailing comma in traceEvents: {path}")
                footer = buffer[position+1:] + stream.read(2*1024*1024+1)
                if len(footer) > 2*1024*1024:
                    raise ValueError("Trace footer exceeds 2MiB")
                json.loads(prefix + "[]" + footer)
                return
            if need_comma:
                if buffer[position] != ",":
                    raise ValueError(f"Missing event separator: {path}")
                position += 1
                need_comma, after_comma = False, True
                continue
            try:
                event, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                more = stream.read(1024*1024)
                if not more:
                    raise ValueError(f"Truncated or invalid event: {path}") from None
                buffer = buffer[position:] + more
                position = 0
                continue
            if not isinstance(event, dict):
                raise ValueError("Trace events must be objects")
            yield "event", event
            position, need_comma, after_comma = end, True, False
            if position > 1024*1024:
                buffer, position = buffer[position:], 0


class CompactTrace:
    """Export GPU slot groups + physical detail + an automatic overview."""
    def __init__(self, path, keep_cpu_ops=False, namespace_flows=True, keep_flows=False):
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        self.keep_cpu_ops = keep_cpu_ops
        self.namespace_flows = namespace_flows
        self.keep_flows = keep_flows
        self.overview_path = self.path.with_name(self.path.name.replace(".trace.json.gz", ".overview.trace.json.gz"))
        if self.overview_path == self.path:
            self.overview_path = self.path.with_name("overview." + self.path.name)
        self.overview_tmp = self.overview_path.with_name(self.overview_path.name + ".tmp")
        self.base = None
        self.started = self.wrote_event = False
        self.flow_ids = {}
        self.seen, self.kept, self.rounds = Counter(), Counter(), Counter()
        self.source_bytes = self.sources = 0
        self.group_counts = Counter()

    def __enter__(self):
        self.output = gzip.open(self.tmp, "wt", encoding="utf-8", compresslevel=6)
        self.overview = gzip.open(self.overview_tmp, "wt", encoding="utf-8", compresslevel=6)
        self.overview_wrote = False
        return self

    def emit(self, event, overview=False):
        target = self.overview if overview else self.output
        flag = self.overview_wrote if overview else self.wrote_event
        if flag:
            target.write(",")
        target.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        if overview:
            self.overview_wrote = True
        else:
            self.wrote_event = True

    def add(self, source, label):
        self.sources += 1
        self.source_bytes += Path(source).stat().st_size
        groups, active, counts = scan_slot_groups(source, trace_records)
        self.group_counts.update(counts)
        active_pids = {track[0] for track in active}
        offset = 0
        for kind, event in trace_records(source):
            if kind == "header":
                current = int(event.get("baseTimeNanoseconds", 0))
                if not self.started:
                    self.base = current
                    self.started = True
                    event["nano_qwen_compact"] = dict(keep_cpu_ops=self.keep_cpu_ops,
                        namespace_flows=self.namespace_flows, keep_flows=self.keep_flows,
                        gpu_ranges="CUPTI GPU timestamps joined by CUDA correlation; display envelopes, not fused kernels",
                        notes="Original pids/tids/durations retained. Gaps between merged windows are unrecorded, not device idle.")
                    header = json.dumps(event, ensure_ascii=False, separators=(",", ":"))[:-1] + ',"traceEvents":['
                    self.output.write(header)
                    self.overview.write(header)
                offset = (current-self.base)/1000
                continue
            cat, phase = event.get("cat", "uncategorized"), event.get("ph")
            self.seen[cat] += 1
            if phase in FLOWS and not self.keep_flows:
                continue
            if phase == "M" and (event.get("pid") not in active_pids or
                    (event.get("tid", 0) != 0 and (event.get("pid"), event.get("tid")) not in active)):
                continue
            # Rebuild our derived ranges if re-exporting an already grouped file.
            if cat == "gpu_user_annotation" and event.get("args", {}).get("derived_from", "").startswith("CUPTI"):
                continue
            keep = (self.keep_cpu_ops or phase == "M" or (phase in FLOWS and self.keep_flows)
                    or cat in PHYSICAL | APIS | {"gpu_user_annotation"}
                    or (cat == "user_annotation" and event.get("name", "").startswith("nq/")))
            if not keep:
                continue
            self.kept[cat] += 1
            if event.get("name", "").startswith("nq/round/"):
                self.rounds[event["name"]] += 1
            if "ts" in event:
                event["ts"] += offset
            if self.namespace_flows and phase in ("s", "f", "t", "S", "T", "F", "b", "e", "n"):
                def namespace(value, scope):
                    key = label, cat, event.get("scope"), scope, json.dumps(value, sort_keys=True)
                    return self.flow_ids.setdefault(key, len(self.flow_ids)+1)
                if "id" in event:
                    event["id"] = namespace(event["id"], "id")
                if isinstance(event.get("id2"), dict):
                    event["id2"] = {k: namespace(v, k) for k,v in event["id2"].items()}
            self.emit(event)
            if (phase == "M" or cat == "gpu_user_annotation" or
                (cat == "user_annotation" and event.get("name", "").startswith("nq/")) or
                (cat in APIS and event.get("name") in ("cudaGraphLaunch", "cudaEventSynchronize"))):
                self.emit(event, overview=True)
        for event in groups:
            event["ts"] += offset
            self.emit(event)
            self.emit(event, overview=True)
            self.kept["gpu_user_annotation"] += 1

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                if not self.started:
                    raise ValueError("No trace sources")
                self.output.write("]}")
                self.overview.write("]}")
            self.output.close()
            self.overview.close()
            if exc_type is None:
                self.tmp.replace(self.path)
                self.overview_tmp.replace(self.overview_path)
            else:
                self.tmp.unlink(missing_ok=True)
                self.overview_tmp.unlink(missing_ok=True)
        except BaseException:
            self.output.close()
            self.overview.close()
            self.tmp.unlink(missing_ok=True)
            self.overview_tmp.unlink(missing_ok=True)
            raise

    def stats(self):
        return dict(source_count=self.sources, source_bytes=self.source_bytes,
                    compressed_bytes=self.path.stat().st_size, events_before=dict(self.seen),
                    events_after=dict(self.kept), round_markers=dict(self.rounds),
                    gpu_grouping=dict(self.group_counts), overview=str(self.overview_path))


def worker(args):
    if args.fused_ops != "auto":
        os.environ["NANO_QWEN_FUSED_OPS"] = args.fused_ops
    mode = args.worker
    out = Path(args.output_dir)
    engine = None
    report = dict(status="failed", mode=mode, batch=BATCH, input=INPUT, output=OUTPUT,
                  warmup=1, recorded_rounds=ROUNDS, diagnostic_only=True, rounds=[],
                  capture="full" if args.full_trace else "decode_window",
                  window=None if args.full_trace else dict(start_decode=args.trace_start,
                  focus_steps=args.trace_steps, guard_steps_each_side=2), keep_cpu_ops=args.keep_cpu_ops,
                  profile_windows=[], measurements=[], fused_ops=os.environ.get("NANO_QWEN_FUSED_OPS", "1"))
    try:
        import torch
        from nano_qwen.config import Config
        from nano_qwen.engine.llm_engine import LLMEngine
        from bench_fp8_pressure import git_info, kv_allocation_policy
        if not torch.cuda.is_available() or torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
            raise RuntimeError("CUDA profiler unavailable; activate nano-qwen-fp8-test and check CUPTI")
        if "dual_slot_decode" not in Config.__dataclass_fields__:
            raise RuntimeError("The dual-slot branch/patch is required")
        report.update(source=git_info(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      model=args.model, torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(), eager=args.eager)
        torch.cuda.set_per_process_memory_fraction(args.mem_util)
        torch.manual_seed(42)
        policy = dict(batch_size=BATCH, max_model_len=CONTEXT, options=dict(kv_cache_policy="workload"))
        with torch.inference_mode(), kv_allocation_policy(policy, torch):
            engine = LLMEngine(args.model, dual_slot_decode=mode == "dual_slot", enforce_eager=args.eager,
                use_prefill_cudagraph=False, enable_prefix_cache=False, tensor_parallel_size=1,
                max_num_seqs=BATCH, max_num_batched_tokens=BATCH*INPUT, max_model_len=CONTEXT,
                gpu_memory_utilization=args.mem_util)
        engine.max_concurrent_batches = 2
        r = engine.model_runner
        r.async_output = True
        if engine.config.max_model_len < CONTEXT:
            raise RuntimeError("Model context is shorter than the fixed workload")
        if not args.eager:
            banks = r.cuda_graphs.decode_graph_banks
            if len(banks) != (2 if mode == "dual_slot" else 1) or BATCH not in r.cuda_graphs.decode_graphs:
                raise RuntimeError("Wrong graph bank count or missing exact B8 graph")
        report["kv_blocks"] = engine.config.num_kvcache_blocks
        def reject_preempt(seq):
            raise RuntimeError(f"Unexpected request preemption: {seq.seq_id}")
        engine.scheduler.preempt = reject_preempt
        with torch.inference_mode():
            print(f"[{mode}] Warmup 1/1 (not recorded)", flush=True)
            run_batch(engine, torch, seed=42, annotate=False)
            # Formal timings are entirely outside torch.profiler/record_function.
            for i in range(1, ROUNDS+1):
                print(f"[{mode}] Unprofiled {i}/{ROUNDS}", flush=True)
                report["measurements"].append(run_batch(engine, torch, seed=42+i,
                    annotate=False, return_tokens=True))
            trace = out / f"{mode}.trace.json.gz"
            with CompactTrace(trace, args.keep_cpu_ops, namespace_flows=not args.full_trace,
                              keep_flows=args.keep_flows) as sink:
                @contextmanager
                def capture(round_id):
                    # Window contexts bound profiler memory as well as disk
                    # size; export/filter/gzip occurs after recording stops.
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=False, with_stack=False, profile_memory=False) as prof, labels(engine, torch):
                        with torch.profiler.record_function(f"nq/round/{round_id}") if round_id else nullcontext():
                            yield
                    raw = out / f".{mode}.r{round_id}.raw.json"
                    prof.export_chrome_trace(str(raw))
                    before_kernels = sink.kept["kernel"]
                    sink.add(raw, round_id)
                    kernels = sink.kept["kernel"] - before_kernels
                    if not kernels:
                        raise RuntimeError(f"Round {round_id}: no physical CUDA kernels captured; inspect CUPTI/Graph profiling support")
                    report["profile_windows"].append(dict(round=round_id, kernel_events=kernels))
                    raw.unlink()

                if args.full_trace:
                    # Opt in to the original full recording duration. All three
                    # rounds remain in one Kineto cycle, then one compact file.
                    with capture(0):
                        for i in range(1, ROUNDS+1):
                            with torch.profiler.record_function(f"nq/round/{i}"):
                                sample = run_batch(engine, torch, seed=42+i, annotate=True)
                            report["rounds"].append(sample)
                else:
                    for i in range(1, ROUNDS+1):
                        print(f"[{mode}] Round {i}/{ROUNDS}: full O256, trace focus {args.trace_steps} decode steps", flush=True)
                        sample = run_batch(engine, torch, seed=42+i, annotate=False,
                            capture=lambda i=i: capture(i), window=(args.trace_start, args.trace_steps))
                        report["rounds"].append(sample)
                if not sink.kept["kernel"] or any(sink.rounds[f"nq/round/{i}"] < 1 for i in range(1,ROUNDS+1)):
                    raise RuntimeError("Trace is missing CUDA kernels or round labels")
            report.update(trace=str(trace), trace_stats=sink.stats())
            if [s["output_sha256"] for s in report["measurements"]] != [s["output_sha256"] for s in report["rounds"]]:
                raise RuntimeError("Profiled outputs differ from unprofiled outputs for the same seeds")
            if not args.eager and (not sink.group_counts["graph_launches"] or
                    sink.group_counts["graph_launches"] != sink.group_counts["graph_groups"]):
                raise RuntimeError("Some Graph launches lack CUPTI GPU ranges; cannot validate slot grouping")
            if report["fused_ops"] == "1" and not sink.group_counts["fused_norm_kernels"]:
                raise RuntimeError("Fusion requested but no _norm_kernel captured; check the installed source/path")
        report["status"] = "ok"
        print(f"[{mode}] Trace saved: {trace} ({trace.stat().st_size/2**20:.2f} MiB)", flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr)
    finally:
        if engine is not None:
            try:
                engine.exit()
            except Exception:
                report.update(status="failed", cleanup_error=traceback.format_exc())
        write_json(out / f"{mode}.report.json", report)
    return 0 if report["status"] == "ok" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/wei/code/models/4b-channel-fp8")
    parser.add_argument("--output-dir")
    parser.add_argument("--mem-util", type=float, default=0.65)
    parser.add_argument("--eager", action="store_true", help="Default uses decode CUDA Graph")
    parser.add_argument("--trace-start", type=int, default=64, help="Committed decode tokens before focus window")
    parser.add_argument("--trace-steps", type=int, default=16, help="Focus window size; generation still runs 256 tokens")
    parser.add_argument("--full-trace", action="store_true", help="Record all three full workloads (larger profiler memory)")
    parser.add_argument("--keep-cpu-ops", action="store_true", help="Retain low-level CPU aten operators in exported trace")
    parser.add_argument("--keep-flows", action="store_true", help="Keep per-kernel CPU/GPU flow arrows in detail trace")
    parser.add_argument("--fused-ops", choices=["auto", "0", "1"], default="auto",
                        help="Reference=0, Triton fusion=1; set before importing the model")
    parser.add_argument("--compact", nargs="+", metavar="TRACE", help="Filter/gzip existing JSON traces without GPU")
    parser.add_argument("--worker", choices=["single_slot", "dual_slot"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.mem_util < 1:
        parser.error("mem-util must be between 0 and 1")
    if args.trace_start < 2 or args.trace_steps < 1 or args.trace_start+args.trace_steps+2 > OUTPUT-1:
        parser.error("Need start>=2, steps>=1, start+steps+2<=255")
    if args.worker:
        return worker(args)
    if not args.compact and not (ROOT/"benchmarks/bench_fp8_pressure.py").is_file():
        parser.error("Place this script in nano_qwen/benchmarks")
    args.model = str(Path(args.model).expanduser().resolve())
    out = Path(args.output_dir or f"reports/slots_b8_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error(f"Use a new/empty output directory: {out}")
    if args.compact:
        stats = []
        for source in args.compact:
            source = Path(source).expanduser().resolve()
            name = source.name if source.suffix == ".gz" else source.name + ".gz"
            dest = out/name
            if dest.exists():
                parser.error(f"Duplicate output filename: {dest.name}")
            with CompactTrace(dest, args.keep_cpu_ops, namespace_flows=False, keep_flows=args.keep_flows) as sink:
                sink.add(source, 0)
            row = dict(source=str(source), output=str(dest), **sink.stats())
            stats.append(row)
            print(f"{source.name}: {source.stat().st_size/2**20:.2f} -> {dest.stat().st_size/2**20:.2f} MiB", flush=True)
        write_json(out/"compact_summary.json", stats)
        return 0
    reports = []
    for mode in ("single_slot", "dual_slot"):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--model", args.model,
               "--output-dir", str(out), "--mem-util", str(args.mem_util), "--worker", mode,
               "--trace-start", str(args.trace_start), "--trace-steps", str(args.trace_steps),
               "--fused-ops", args.fused_ops]
        if args.eager:
            cmd.append("--eager")
        if args.full_trace:
            cmd.append("--full-trace")
        if args.keep_cpu_ops:
            cmd.append("--keep-cpu-ops")
        if args.keep_flows:
            cmd.append("--keep-flows")
        # Each process creates exactly one engine; CUDA context ends before
        # the next mode starts. Preserve the active conda Python executable.
        result = subprocess.run(cmd, cwd=ROOT, check=False)
        path = out / f"{mode}.report.json"
        report = json.loads(path.read_text()) if path.exists() else dict(status="failed", mode=mode)
        report["returncode"] = result.returncode
        reports.append(report)
    ok = all(r["status"] == "ok" and r["returncode"] == 0 for r in reports)
    if ok:
        a,b = reports
        if (a["kv_blocks"] != b["kv_blocks"] or a["source"]["source_sha256"] != b["source"]["source_sha256"]
                or [s["prompt_sha256"] for s in a["rounds"]] != [s["prompt_sha256"] for s in b["rounds"]]):
            raise RuntimeError("The two runs used different KV capacity, source or prompts")
    write_json(out/"summary.json", dict(status="ok" if ok else "failed", reports=reports,
        notes=["Each mode executes three full B8/I128/O256 runs; warmup excluded.",
               "Default trace samples 16 committed decode steps per round plus 2 guard steps per side; prefill excluded.",
               "Export automatically adds GPU slot ranges and an overview; detail preserves all physical GPU events.",
               "Flow arrows hidden by default; --keep-flows retains them in detail.",
               "Gaps between sampled windows are unrecorded generation/export time, not GPU idle time.",
               "CPU thread and GPU CUDA streams are distinct tracks.",
               "Profiler traces are diagnostics, not unprofiled speed measurements."]))
    print(f"Results: {out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
