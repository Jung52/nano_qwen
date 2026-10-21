#!/usr/bin/env python3
"""Fixed-batch real-engine profiling: CPU replay -> alternating GPU Graph slots.

Default B8/I128/O256, warmup once, 3 separate 16-step decode windows with
2 guard steps each side. Each full request still generates 256 tokens and
drains. Each round directly exports ONE full-detail trace: no operator filter,
no overview file, no external postprocessing command. TP=1, decode Graph,
eager prefill. Single/dual modes use separate processes and the same KV budget.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack, nullcontext
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"benchmarks"))
from slot_trace_full import export_full_trace


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


@contextmanager
def replay_labels(engine, torch):
    """Temporary instrumentation around ACTUAL graph.replay(), never a fake op.

    Installed after warmup and only inside profiling. No graph capture, stream,
    event or scheduling behavior is changed. Every patch is restored on exit.
    """
    runner, manager, pipeline = engine.model_runner, engine.model_runner.cuda_graphs, engine._dual_pipeline
    counts = Counter()
    with ExitStack() as undo:
        def patch(obj, name, value):
            own, old = name in vars(obj), getattr(obj, name)
            setattr(obj, name, value)
            undo.callback(setattr if own else delattr, obj, name, *([old] if own else []))
        def wrap(obj, method, title):
            old = getattr(obj, method)
            @functools.wraps(old)
            def call(*args, **kwargs):
                name = title(*args, **kwargs) if callable(title) else title
                with torch.profiler.record_function(name):
                    return old(*args, **kwargs)
            patch(obj, method, call)
        banks = manager.decode_graph_banks or [(manager.decode_graphs, manager.decode_graph_vars, manager.decode_graph_pool)]
        seen = set()
        for slot, (graphs, _, _) in enumerate(banks):
            for batch, graph in graphs.items():
                if id(graph) in seen:
                    raise RuntimeError("Graph banks unexpectedly share a graph object")
                seen.add(id(graph))
                original = graph.replay
                def replay(*args, _old=original, _slot=slot, _batch=batch, **kwargs):
                    counts[(_slot, _batch)] += 1
                    with torch.profiler.record_function(f"CUDAGraph.replay/slot{_slot}/bs{_batch}"):
                        return _old(*args, **kwargs)
                patch(graph, "replay", replay)
        def active():
            return getattr(runner, "_active_slot_id", 0) if pipeline is not None else 0
        wrap(runner, "prepare_inputs", lambda *a, **kw: f"prepare_inputs/slot{active()}")
        wrap(runner, "sample_tokens", lambda *a, **kw: f"sample_tokens/slot{active()}")
        wrap(runner.model, "compute_logits", lambda *a, **kw: f"LMHead/slot{active()}")
        from nano_qwen.engine.async_output import AsyncModelOutput
        wrap(AsyncModelOutput, "get_output", "get_output_wait")
        if pipeline is not None:
            wrap(runner, "launch_slot", lambda seqs, is_prefill, slot_id: f"forward/slot{slot_id}")
            wrap(runner, "sample_slot", lambda slot_id, ids: f"sample/slot{slot_id}")
            wrap(pipeline, "_commit_oldest", lambda: f"commit/slot{pipeline.pending[0].slot_id}")
            wrap(pipeline, "_plan_decode", "schedule_decode")
        else:
            wrap(engine.scheduler, "postprocess", "commit/slot0")
        yield counts


def run_batch(engine, torch, args, seed, capture=None):
    from nano_qwen.engine.sequence import Sequence
    from nano_qwen.sampling_params import SamplingParams
    if not engine.is_finished():
        raise RuntimeError("Previous round did not drain")
    rng = random.Random(seed)
    prompts = [[rng.randrange(engine.config.hf_config.vocab_size) for _ in range(args.input_len)]
               for _ in range(args.batch_size)]
    seqs = [Sequence(p, SamplingParams(temperature=1e-6, max_tokens=args.output_len, ignore_eos=True)) for p in prompts]
    runner, pipeline = engine.model_runner, engine._dual_pipeline
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    runner.cuda_graphs.reset_dispatch_stats()
    started = time.perf_counter()
    for seq in seqs:
        engine.scheduler.add(seq)
    engine.step()
    torch.cuda.synchronize()
    prefill_end = time.perf_counter()
    if any(s.num_completion_tokens != 1 for s in seqs):
        raise RuntimeError("Expected exactly one complete prefill for the whole batch")
    ticks = 0
    def step(record=False):
        nonlocal ticks
        with torch.profiler.record_function(f"engine.step/{ticks:04d}") if record else nullcontext():
            engine.step()
        ticks += 1
        if ticks > args.output_len + 4:
            raise RuntimeError("Decode failed to drain within the fixed-workload bound")
    def advance(target, record=False):
        while min(s.num_completion_tokens for s in seqs) < target:
            step(record)
        if any(s.num_completion_tokens != target for s in seqs):
            raise RuntimeError("Fixed batch fragmented or advanced to different lengths")
    if capture is not None:
        # Use committed output lengths, not a fixed number of engine.step()
        # calls: dual-slot has prime/drain ticks with different host counts.
        advance(1 + args.skip_decode - 2)
        torch.cuda.synchronize()
        with capture():
            with torch.profiler.record_function("decode/leading_guard"):
                advance(1 + args.skip_decode, True)
            with torch.profiler.record_function("decode/focus"):
                advance(1 + args.skip_decode + args.profile_decode_steps, True)
            with torch.profiler.record_function("decode/trailing_guard"):
                advance(1 + args.skip_decode + args.profile_decode_steps + 2, True)
            torch.cuda.synchronize()
    while not engine.is_finished():
        step()
    torch.cuda.synchronize()
    ended = time.perf_counter()
    if any(not s.is_finished or s.num_completion_tokens != args.output_len for s in seqs):
        raise RuntimeError("Wrong output count or unfinished request")
    if runner.input_batch.seq_id_to_slot or engine.scheduler.block_manager.used_block_ids:
        raise RuntimeError("Persistent request slots or KV pages leaked")
    if pipeline is not None and (pipeline.pending or pipeline.requests or any(s.busy for s in runner.decode_slots)):
        raise RuntimeError("Dual execution slots did not drain")
    dispatch = runner.cuda_graphs.dispatch_stats()
    dec = dispatch.get("decode", {})
    if dec.get("graph_calls") != args.output_len-1 or dec.get("eager_calls", 0) or dec.get("batch_counts") != {args.batch_size: args.output_len-1}:
        raise RuntimeError(f"Expected exact-batch decode Graph for every forward: {dec}")
    digest = lambda x: hashlib.sha256(json.dumps(x).encode()).hexdigest()
    result = dict(seed=seed, host_decode_ticks=ticks, dispatch=dispatch, prompt_sha256=digest(prompts),
        output_sha256=digest([s.completion_token_ids for s in seqs]), output_tokens=args.batch_size*args.output_len)
    if capture is None:
        dec_ms, e2e_ms = (ended-prefill_end)*1000, (ended-started)*1000
        result.update(prefill_ms=(prefill_end-started)*1000, decode_ms=dec_ms, e2e_ms=e2e_ms,
            decode_tok_s=args.batch_size*(args.output_len-1)*1000/dec_ms)
    return result


def worker(args):
    if args.fused_ops != "auto":
        os.environ["NANO_QWEN_FUSED_OPS"] = args.fused_ops
    out, engine = Path(args.output_dir), None
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status="failed", mode=args.worker, args=vars(args), timings=[], profiles=[],
        full_detail=True, operators_filtered=False, fused_ops=os.environ.get("NANO_QWEN_FUSED_OPS", "1"))
    try:
        import torch
        from nano_qwen.engine.llm_engine import LLMEngine
        from bench_fp8_pressure import git_info, kv_allocation_policy
        if not torch.cuda.is_available() or torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
            raise RuntimeError("CUDA + CUPTI profiler required; use nano-qwen-fp8-test")
        report.update(source=git_info(), torch=torch.__version__, cuda=torch.version.cuda,
            entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        torch.cuda.set_per_process_memory_fraction(args.mem_util)
        torch.manual_seed(args.seed)
        context = args.input_len + args.output_len
        policy = dict(batch_size=args.batch_size, max_model_len=context, options=dict(kv_cache_policy="workload"))
        with torch.inference_mode(), kv_allocation_policy(policy, torch):
            engine = LLMEngine(args.model, dual_slot_decode=args.worker == "dual", enforce_eager=False,
                use_prefill_cudagraph=False, enable_prefix_cache=False, tensor_parallel_size=1,
                max_num_seqs=args.batch_size, max_num_batched_tokens=args.batch_size*args.input_len,
                max_model_len=context, gpu_memory_utilization=args.mem_util)
        runner = engine.model_runner
        engine.max_concurrent_batches, runner.async_output = 2, True
        if bool(engine._dual_pipeline is not None) != (args.worker == "dual"):
            raise RuntimeError("dual_slot_decode flag did not reach the engine; wrong source/environment")
        banks = runner.cuda_graphs.decode_graph_banks
        if len(banks) != (2 if args.worker == "dual" else 1) or any(args.batch_size not in b[0] for b in banks):
            raise RuntimeError("Wrong graph bank count or missing exact-batch graph")
        report["kv_blocks"] = engine.config.num_kvcache_blocks
        def reject_preempt(seq):
            raise RuntimeError(f"Unexpected request preemption: {seq.seq_id}")
        engine.scheduler.preempt = reject_preempt
        with torch.inference_mode():
            print(f"[{args.worker}] Warmup once, full fixed batch", flush=True)
            run_batch(engine, torch, args, args.seed)
            for i in range(1, args.timing_rounds+1):
                report["timings"].append(run_batch(engine, torch, args, args.seed+i))
            for i in range(1, args.profile_rounds+1):
                @contextmanager
                def capture():
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=args.record_shapes, with_stack=args.with_stack,
                            profile_memory=False) as prof, replay_labels(engine, torch) as counts:
                        with torch.profiler.record_function(f"fixed_batch/round{i}/{args.worker}"):
                            yield
                    raw = out/f".{args.worker}.r{i}.raw.trace.json"
                    trace = out/f"{args.worker}.r{i}.trace.json.gz"
                    prof.export_chrome_trace(str(raw))
                    # Integrated export only adds display lanes/ranges. NO op filtering.
                    stats = export_full_trace(raw, trace, mode=args.worker)
                    if stats["cpu_graph_launches"] != sum(counts.values()):
                        raise RuntimeError("CPU replay wrappers and CUPTI launches disagree")
                    if not stats["original_counts"].get("cpu_op") or not stats["original_counts"].get("kernel"):
                        raise RuntimeError("Missing real CPU operators or GPU kernels; inspect profiler/CUPTI")
                    stats["wrapped_replays"] = {f"slot{s}/bs{b}": n for (s,b),n in counts.items()}
                    if args.keep_raw:
                        import gzip
                        import shutil
                        retained = out/f"{args.worker}.r{i}.raw.trace.json.gz"
                        with raw.open("rb") as src, gzip.open(retained,"wb") as dst:
                            shutil.copyfileobj(src, dst)
                        stats["raw_trace"] = str(retained)
                    raw.unlink()
                    report["profiles"].append(stats)
                    print(f"[{args.worker}] {trace.name}: launches={stats['cpu_graph_launches']}, "
                        f"GPU kernels={stats['original_counts'].get('kernel',0)}, CPU ops={stats['original_counts'].get('cpu_op',0)}", flush=True)
                sample = run_batch(engine, torch, args, args.seed+i, capture=capture)
                report["profiles"][-1]["workload"] = sample
                save(out/f"{args.worker}.summary.json", report)
        report["status"] = "ok"
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        if engine is not None:
            try:
                engine.exit()
            except Exception:
                report.update(status="failed", cleanup_error=traceback.format_exc())
        save(out/f"{args.worker}.summary.json", report)
    return 0 if report["status"] == "ok" else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="/home/wei/code/models/4b-channel-fp8")
    p.add_argument("--mode", choices=["single", "dual", "both"], default="both")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--input-len", type=int, default=128)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--profile-rounds", type=int, default=3)
    p.add_argument("--timing-rounds", type=int, default=0, help="Optional unprofiled measurements, separate from traces")
    p.add_argument("--skip-decode", type=int, default=64)
    p.add_argument("--profile-decode-steps", type=int, default=16)
    p.add_argument("--mem-util", type=float, default=0.65)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fused-ops", choices=["auto", "0", "1"], default="auto")
    p.add_argument("--record-shapes", action="store_true")
    p.add_argument("--with-stack", action="store_true")
    p.add_argument("--keep-raw", action="store_true", help="Also gzip Kineto's untouched original trace")
    p.add_argument("--output-dir", default="reports/one_batch_slots")
    p.add_argument("--worker", choices=["single", "dual"], help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.batch_size not in [1, 2, 4, 8]+list(range(16,513,16)):
        p.error("Need an exact decode Graph batch")
    if min(a.input_len, a.profile_rounds) < 1 or a.timing_rounds < 0 or a.profile_decode_steps < 2 or a.skip_decode < 2:
        p.error("Need input/profile rounds>=1, timing>=0, skip>=2, profile steps>=2")
    if a.skip_decode+a.profile_decode_steps+2 > a.output_len-1:
        p.error("Trace window + guards must fit output_len-1 decode forwards")
    if not 0 < a.mem_util < 1:
        p.error("mem-util must lie in (0,1)")
    return a


def main():
    args = parse_args()
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    args.model = str(Path(args.model).expanduser().resolve())
    if args.worker:
        return worker(args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for mode in (["single", "dual"] if args.mode == "both" else [args.mode]):
        target = out/mode
        target.mkdir(exist_ok=True)
        path = target/f"{mode}.summary.json"
        path.unlink(missing_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", mode,
               "--model", args.model, "--output-dir", str(target)]
        for key in ("batch_size", "input_len", "output_len", "profile_rounds", "timing_rounds",
                    "skip_decode", "profile_decode_steps", "mem_util", "seed", "fused_ops"):
            cmd += ["--"+key.replace("_","-"), str(getattr(args,key))]
        for key in ("record_shapes", "with_stack", "keep_raw"):
            if getattr(args,key):
                cmd.append("--"+key.replace("_","-"))
        result = subprocess.run(cmd, cwd=ROOT)
        report = json.loads(path.read_text()) if path.exists() else dict(status="failed", mode=mode)
        report["returncode"] = result.returncode
        reports.append(report)
    ok = all(r.get("status") == "ok" and r.get("returncode") == 0 for r in reports)
    parity = None
    if ok and len(reports) == 2:
        x,y = reports
        parity = all(a["workload"]["prompt_sha256"] == b["workload"]["prompt_sha256"] and
                     a["workload"]["output_sha256"] == b["workload"]["output_sha256"]
                     for a,b in zip(x["profiles"], y["profiles"]))
        ok = (parity and x["kv_blocks"] == y["kv_blocks"] and
              x["source"]["source_sha256"] == y["source"]["source_sha256"])
    save(out/"summary.json", dict(status="ok" if ok else "failed", single_dual_token_parity=parity,
        reports=reports, notes=["Open *.rN.trace.json.gz, not the previous overview trace.",
        "No CPU operator or GPU kernel filtering. Graph replay does not execute its captured CPU aten calls again.",
        "CPU cudaGraphLaunch display lane duplicates real runtime intervals, not a second physical CPU thread.",
        "GPU cuda_graph_slot0/1 envelopes use actual GPU kernel timestamps; do not add parent and child durations.",
        "Separate profiler windows, one trace per round; requests run to full completion outside each window."]))
    print(f"Results: {out}/summary.json", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
