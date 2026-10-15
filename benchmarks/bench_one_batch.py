#!/usr/bin/env python3
"""Profile one fixed batch through nano_qwen's real LLMEngine.step().

Unprofiled timing (--timing-rounds) is separate from diagnostic profiling
(--profile-rounds, default 1). Each profiled workload produces one trace. GPU events are
normalized onto their own Perfetto process/thread tracks, so the UI shows the
host thread and GPU stream threads as separate tracks:

  *.trace.json      CPU ops/runtime + GPU kernels, with CPU->GPU arrows

Examples:
  CUDA_VISIBLE_DEVICES=0 python bench_one_batch.py --model /path/to/Qwen3.5
  python bench_one_batch.py --model /path/to/Qwen3.5 --enforce-eager --with-stack

Defaults: BS=64, input=64 tokens/request, 32 decode forwards/request.
Prefill produces token #1; max_tokens=decode_steps+1, ignore_eos=True.
Warmup runs unprofiled. --profile-rounds 0 selects timing only.
--workload-memory uses a workload-sized KV pool, a CUDA allocator budget, and
captures only the exact prefill bucket; production operators are unchanged.

Trace reading:
  Open *.trace.json in https://ui.perfetto.dev (Open trace file). CPU and GPU
  events live under different tracks; each CUDA stream gets its own GPU thread
  track. CPU record_function durations are host scopes, not GPU kernel time.
  For CUDA Graph replay, op attribution requires --enforce-eager --with-stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from collections import Counter
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", "--model-path", required=True, help="Local Hugging Face model directory")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--input-len", type=int, default=64)
    p.add_argument("--decode-steps", type=int, default=32, help="Decode forwards; output length is this + 1")
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--timing-rounds", type=int, default=0, help="Formal passes with profiler disabled")
    p.add_argument("--profile-rounds", type=int, default=1)
    p.add_argument("--profile-decode-steps", type=int, help="Trace only this many decode steps; drain the rest unprofiled")
    p.add_argument("--profile-skip-decode", type=int, default=0, help="Decode steps before the aggregated window; still traced with --trace-stage all")
    p.add_argument("--workload-memory", action="store_true", help="Controlled KV pool and exact prefill bucket; allocator capped at memory utilization")
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--enforce-eager", action="store_true", help="Disable prefill and decode CUDA Graphs")
    p.add_argument("--disable-prefill-cudagraph", action="store_true")
    p.add_argument("--queue-depth", type=int, choices=(1, 2), default=2)
    p.add_argument("--sync-output", action="store_true", help="Use the runner's blocking D2H baseline")
    p.add_argument("--trace-stage", choices=("all", "decode"), default="all")
    p.add_argument("--record-shapes", action="store_true")
    p.add_argument("--with-stack", action="store_true")
    p.add_argument("--profile-memory", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="logs/traces")
    args = p.parse_args(argv)
    for name in ("batch_size", "input_len", "decode_steps", "warmup_rounds", "max_model_len"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be >=1")
    if not 0 < args.gpu_memory_utilization < 1:
        p.error("--gpu-memory-utilization must be between 0 and 1")
    if min(args.timing_rounds, args.profile_rounds, args.profile_skip_decode) < 0 or not (args.timing_rounds + args.profile_rounds):
        p.error("Nonnegative round counts and at least one timing/profile pass required")
    if args.profile_decode_steps is None:
        args.profile_decode_steps = args.decode_steps - args.profile_skip_decode
    if args.profile_decode_steps < 1 or args.profile_decode_steps + args.profile_skip_decode > args.decode_steps:
        p.error("Profile window must fit inside decode_steps")
    if args.input_len + args.decode_steps + 1 > args.max_model_len:
        p.error("--max-model-len must cover input_len + decode_steps + 1")
    if not Path(args.model).is_dir():
        p.error("--model must be a local model directory")
    return args


def find_repo():
    here = Path(__file__).resolve().parent
    for root in (here, here.parent, Path.cwd()):
        if (root / "src/nano_qwen/engine/llm_engine.py").is_file():
            sys.path.insert(0, str(root / "src"))
            return root
    return None  # An editable/installed nano_qwen package is also supported.


def enqueue(engine, args, Sequence, SamplingParams, seed):
    if not engine.is_finished():
        raise RuntimeError("Previous workload has not drained")
    rng = random.Random(seed)
    vocab = int(engine.config.hf_config.vocab_size)
    seqs = [Sequence([rng.randrange(vocab) for _ in range(args.input_len)],
                     SamplingParams(temperature=1.0, max_tokens=args.decode_steps + 1, ignore_eos=True))
            for _ in range(args.batch_size)]
    for seq in seqs:
        engine.scheduler.add(seq)
    return seqs


def check_finished(engine, seqs, args):
    if not engine.is_finished():
        raise RuntimeError("Batch not drained after expected prefill + decode steps")
    if any(not s.is_finished or s.num_completion_tokens != args.decode_steps + 1 for s in seqs):
        raise RuntimeError("Unexpected completion count; fixed-batch assumption violated")
    if engine.model_runner.input_batch.seq_id_to_slot:
        raise RuntimeError("Finished requests still occupy persistent input slots")


def full_run(engine, seqs, args, torch):
    torch.cuda.synchronize()
    start = time.perf_counter()
    engine.step()
    torch.cuda.synchronize()  # Do not charge pending prefill work to decode.
    prefill_end = time.perf_counter()
    if any(s.num_completion_tokens != 1 for s in seqs):
        raise RuntimeError("Prefill did not process the entire fixed cohort in one forward")
    for _ in range(args.decode_steps):
        engine.step()
    torch.cuda.synchronize()
    end = time.perf_counter()
    check_finished(engine, seqs, args)
    return {"wall_ms": (end - start) * 1000,
            "prefill_engine_step_ms": (prefill_end - start) * 1000,
            "decode_wall_ms": (end - prefill_end) * 1000,
            "decode_mean_step_ms": (end - prefill_end) * 1000 / args.decode_steps,
            "decode_tokens_per_s": args.batch_size * args.decode_steps / (end - prefill_end)}


def physical_gpu_event(event):
    return event.get("ph") == "X" and event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")


def kernel_category(name):
    name = name.lower()
    if "_quantize_rows" in name or "_quantize_groups" in name:
        return "activation_quantization"
    if "_per_channel_mm" in name or "_block_mm" in name:
        return "fp8_gemm"
    if any(s in name for s in ("delta_rule", "gdn", "chunk_fwd", "wy_fast", "chunk_gated")):
        return "gdn"
    if any(s in name for s in ("flash", "attention", "fmha")):
        return "attention"
    if any(s in name for s in ("gemm", "cublas", "cutlass", "gemv")):
        return "bf16_gemm"
    if any(s in name for s in ("indexselect", "index_select", "indexcopy", "index_copy", "vectorized_gather_kernel")):
        return "state_index_copy"
    if "memcpy" in name or "memset" in name:
        return "copy_memset"
    if "conv" in name:
        return "convolution"
    if any(s in name for s in ("norm", "rotary", "rope", "silu", "gate_projection")):
        return "norm_gate_activation_rope"
    return "other"


def phase_breakdown(events):
    """Physical device durations in synchronized host phase windows, in ms.

    Kernel sums can overlap across streams; the interval union is GPU busy
    time. Gaps inside the first/last device event are not called CPU time.
    GPU annotations are deliberately excluded to avoid nested double counts.
    """
    phases = {stage: [e for e in events if e.get("ph") == "X" and e.get("cat") == "user_annotation"
                      and e.get("name", "").startswith(f"nq/{stage}/")
                      and (stage == "prefill" or "/window/" in e["name"])]
              for stage in ("prefill", "decode")}
    rows = {stage: [] for stage in phases}
    unassigned = []
    for event in events:
        if not physical_gpu_event(event):
            continue
        stage = next((s for s, windows in phases.items() if any(
            w["ts"] <= event["ts"] and event["ts"] + event["dur"] <= w["ts"] + w["dur"] + 1
            for w in windows)), None)
        (rows[stage] if stage else unassigned).append(event)
    result = {}
    for stage, device in rows.items():
        categories, kernels, intervals = {}, {}, []
        for event in device:
            name, ms = event["name"], event["dur"] / 1000
            category = kernel_category(name)
            categories[category] = categories.get(category, 0) + ms
            row = kernels.setdefault(name, dict(name=name, category=category, calls=0, total_ms=0))
            row["calls"] += 1
            row["total_ms"] += ms
            intervals.append((event["ts"], event["ts"] + event["dur"]))
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        busy = sum(end - start for start, end in merged) / 1000
        span = (merged[-1][1] - merged[0][0]) / 1000 if merged else 0
        result[stage] = dict(device_event_count=len(device), device_sum_ms=sum(categories.values()),
                             device_busy_ms=busy, device_span_ms=span, device_internal_gap_ms=span - busy,
                             profiled_host_scope_ms=sum(w["dur"] for w in phases[stage]) / 1000,
                             categories_ms=categories, kernels=sorted(kernels.values(), key=lambda r: -r["total_ms"]))
    result.update(unassigned_device_event_count=len(unassigned),
                  unassigned_device_sum_ms=sum(e["dur"] for e in unassigned) / 1000)
    return result


def is_gpu_event(event):
    category = str(event.get("cat", ""))
    return category == "kernel" or category.startswith("gpu_")


def ensure_gpu_tracks(source, device_label):
    """Put GPU device events on their own Perfetto process/thread tracks.

    PyTorch normally exports kernels under a separate GPU pid and one tid per
    CUDA stream. This pass also handles exporters that share the host pid and
    always adds explicit track names so Perfetto renders distinct CPU/GPU
    threads instead of overlapping them.
    """
    data = json.loads(source.read_text(encoding="utf-8"))
    events = data.get("traceEvents", [])
    timed = [e for e in events if e.get("ph") in ("X", "i")]
    host_pids = {e["pid"] for e in timed if not is_gpu_event(e) and e.get("pid") is not None}
    gpu_events = [e for e in timed if is_gpu_event(e)]
    gpu_pids = {e["pid"] for e in gpu_events if e.get("pid") is not None}

    if gpu_events and (gpu_pids & host_pids):
        # Pathological exporter: remap device events to a dedicated GPU pid,
        # keeping one thread per CUDA stream.
        gpu_pid = max(host_pids | gpu_pids, default=0) + 1
        stream_tids = {}
        for event in gpu_events:
            stream = event.get("args", {}).get("stream", event.get("tid", 0))
            event["pid"] = gpu_pid
            event["tid"] = stream_tids.setdefault(stream, 1000 + len(stream_tids))
        gpu_pids = {gpu_pid}

    metadata = {(m.get("pid"), m.get("tid"), m.get("name"))
                for m in events if m.get("ph") == "M"}
    for pid in sorted(gpu_pids):
        if (pid, 0, "process_name") not in metadata:
            events.append({"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
                           "args": {"name": f"{device_label} (pid {pid})"}})
    for pid, tid in sorted({(e["pid"], e["tid"]) for e in gpu_events}):
        if (pid, tid, "thread_name") not in metadata:
            events.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid,
                           "args": {"name": f"stream {tid}"}})

    source.write_text(json.dumps(data), encoding="utf-8")
    tracks = ({"pid": p, "tid": t} for p, t in {(e["pid"], e["tid"]) for e in gpu_events})
    return len(gpu_events), sorted(tracks, key=lambda track: (track["pid"], track["tid"]))


def main():
    args = parse_args()
    repo = find_repo()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; run in the nano_qwen inference environment")
    if args.profile_rounds and torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError("CUDA profiler activity unavailable; check PyTorch/CUPTI installation")
    from nano_qwen.engine.llm_engine import LLMEngine
    from nano_qwen.engine.sequence import Sequence
    from nano_qwen.sampling_params import SamplingParams
    from bench_fp8_pressure import git_info, kv_allocation_policy, memory_snapshot, path_audit, telemetry

    source = git_info()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.workload_memory:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_utilization, device=0)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mode = "eager" if args.enforce_eager else "graph"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = out / f"bs{args.batch_size}_in{args.input_len}_dec{args.decode_steps}_{mode}_{args.trace_stage}_{stamp}_{os.getpid()}"
    summary_path = Path(str(stem) + ".summary.json")
    summary = dict(args=vars(args), source=source,
                   entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   telemetry_before=telemetry(), torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(), tensor_parallel_size=1, timing_samples=[], profiles=[],
                   notes=["Formal timing has profiler disabled and a synchronized prefill/decode boundary.",
                          "TPOT = decode wall time / decode steps, without division by batch.",
                          "Profile device sums and gaps are diagnostic; not unprofiled end-to-end latency.",
                          "Synthetic fixed-length workload; no model-quality assessment."])

    def checkpoint():
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    job = dict(batch_size=args.batch_size, max_model_len=args.max_model_len,
               options=dict(kv_cache_policy="workload"))
    policy = kv_allocation_policy(job, torch) if args.workload_memory else nullcontext()
    print(f"Loading model; BS={args.batch_size}, input={args.input_len}, decode={args.decode_steps}, TP=1", flush=True)
    # Avoid retaining autograd state in model initialization and graph capture.
    with torch.inference_mode(), policy:
        engine = LLMEngine(args.model, tensor_parallel_size=1, max_num_seqs=args.batch_size,
                           max_num_batched_tokens=args.batch_size * args.input_len,
                           max_model_len=args.max_model_len, enforce_eager=args.enforce_eager,
                           use_prefill_cudagraph=not (args.disable_prefill_cudagraph or args.workload_memory),
                           enable_prefix_cache=False, gpu_memory_utilization=args.gpu_memory_utilization)
    try:
        with torch.inference_mode():
            engine.max_concurrent_batches = args.queue_depth
            runner = engine.model_runner
            runner.async_output = not args.sync_output
            runner.use_prefill_cudagraph = not args.disable_prefill_cudagraph
            if args.workload_memory and not args.enforce_eager and runner.use_prefill_cudagraph:
                runner.cuda_graphs.capture_prefill([args.batch_size * args.input_len])
            block_size = engine.config.kvcache_block_size
            needed = args.batch_size * ((args.input_len + args.decode_steps + 1 + block_size - 1) // block_size)
            if engine.config.num_kvcache_blocks < needed:
                raise RuntimeError(f"Need {needed} KV blocks; allocated {engine.config.num_kvcache_blocks}")
            if engine.config.max_model_len < args.input_len + args.decode_steps + 1:
                raise RuntimeError("Model maximum context is shorter than workload")
            if not args.enforce_eager and args.batch_size not in runner.graphs:
                raise RuntimeError(f"No exact CUDA Graph for BS={args.batch_size}")
            summary["engine"] = dict(kv_blocks=engine.config.num_kvcache_blocks,
                                     prefill_graph_sizes=runner.prefill_graph_sizes, decode_graph_sizes=runner.graph_bs,
                                     inference_mode_initialization=True,
                                     quantization_methods=dict(Counter(type(m.quant_method).__name__
                                         for m in runner.model.modules() if hasattr(m, "quant_method"))))

            def reject_preempt(seq):
                raise RuntimeError(f"Request {seq.seq_id} would be preempted")
            engine.scheduler.preempt = reject_preempt
            for r in range(args.warmup_rounds):
                print(f"Warmup {r + 1}/{args.warmup_rounds} (full workload, profiler off)", flush=True)
                torch.manual_seed(args.seed + r)
                seqs = enqueue(engine, args, Sequence, SamplingParams, args.seed + r)
                summary["last_warmup_pass"] = full_run(engine, seqs, args, torch)
            for r in range(args.timing_rounds):
                torch.manual_seed(args.seed + args.warmup_rounds + r)
                seqs = enqueue(engine, args, Sequence, SamplingParams, args.seed + args.warmup_rounds + r)
                torch.cuda.reset_peak_memory_stats()
                with path_audit(runner) as counts:
                    timing = full_run(engine, seqs, args, torch)
                timing.update(round=r + 1, seed=args.seed + args.warmup_rounds + r,
                              paths=dict(counts), memory=memory_snapshot(torch))
                summary["timing_samples"].append(timing)
                checkpoint()  # Preserve each original sample before another pass.
                print(f"Timing {r + 1}/{args.timing_rounds}: {json.dumps(timing)}", flush=True)

            # These labels wrap only diagnostics, never formal timing.
            original_logits = runner.model.compute_logits
            def logits(hidden):
                with torch.profiler.record_function(f"nq/lm_head/rows{hidden.shape[0]}"):
                    return original_logits(hidden)
            if args.profile_rounds:
                runner.model.compute_logits = logits
            for r in range(args.profile_rounds):
                torch.manual_seed(args.seed + args.warmup_rounds)
                seqs = enqueue(engine, args, Sequence, SamplingParams, args.seed + args.warmup_rounds)
                if args.trace_stage == "decode":
                    engine.step()
                torch.cuda.synchronize()
                if args.trace_stage == "decode":
                    for _ in range(args.profile_skip_decode):
                        engine.step()
                    torch.cuda.synchronize()
                print(f"Recording profile {r + 1}/{args.profile_rounds}", flush=True)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=args.record_shapes, with_stack=args.with_stack,
                                            profile_memory=args.profile_memory) as prof:
                    if args.trace_stage == "all":
                        with torch.profiler.record_function(f"nq/prefill/bs{args.batch_size}"):
                            engine.step()
                            torch.cuda.synchronize()
                    # With --trace-stage all, skipped steps remain in the trace
                    # but are excluded from both aggregated phase windows.
                    if args.trace_stage == "all":
                        for _ in range(args.profile_skip_decode):
                            engine.step()
                        torch.cuda.synchronize()
                    with torch.profiler.record_function(f"nq/decode/window/bs{args.batch_size}"):
                        for i in range(args.profile_decode_steps):
                            with torch.profiler.record_function(f"nq/step/{i:03d}/bs{args.batch_size}"):
                                engine.step()
                        torch.cuda.synchronize()
                for _ in range(args.decode_steps - args.profile_skip_decode - args.profile_decode_steps):
                    engine.step()
                torch.cuda.synchronize()
                check_finished(engine, seqs, args)
                trace_path = Path(str(stem) + f".r{r + 1}.trace.json")
                prof.export_chrome_trace(str(trace_path))
                gpu_event_count, gpu_tracks = ensure_gpu_tracks(trace_path, torch.cuda.get_device_name())
                breakdown = phase_breakdown(json.loads(trace_path.read_text())["traceEvents"])
                profile = dict(round=r + 1, trace=str(trace_path), gpu_device_events=gpu_event_count,
                               gpu_tracks=gpu_tracks, breakdown=breakdown,
                               decode_steps=args.profile_decode_steps,
                               decode_context_start=args.input_len + 1 + args.profile_skip_decode,
                               decode_context_end=args.input_len + args.profile_skip_decode + args.profile_decode_steps)
                summary["profiles"].append(profile)
                averages = prof.key_averages(group_by_input_shape=args.record_shapes)
                Path(str(stem) + f".r{r + 1}.operators.txt").write_text(
                    averages.table(sort_by="self_device_time_total", row_limit=50), encoding="utf-8")
                checkpoint()
                if not gpu_event_count:
                    raise RuntimeError("Trace has no CUDA device events; do not infer GPU idle gaps")
            runner.model.compute_logits = original_logits
            summary["memory_after"] = memory_snapshot(torch)
            summary["telemetry_after"] = telemetry()
            summary["status"] = "ok"
            checkpoint()
            print(f"Summary: {summary_path}", flush=True)
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
