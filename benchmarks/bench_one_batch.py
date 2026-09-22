#!/usr/bin/env python3
"""Profile one fixed batch through nano_qwen's real LLMEngine.step().

Simplified version: no engine monkey-patching and no extra unprofiled timing
pass. One profiled workload produces ONE Chrome trace file. GPU events are
normalized onto their own Perfetto process/thread tracks, so the UI shows the
host thread and GPU stream threads as separate tracks:

  *.trace.json      CPU ops/runtime + GPU kernels, with CPU->GPU arrows

Examples:
  CUDA_VISIBLE_DEVICES=0 python bench_one_batch.py --model /path/to/Qwen3.5
  python bench_one_batch.py --model /path/to/Qwen3.5 --enforce-eager --with-stack

Defaults: BS=64, input=64 tokens/request, 32 decode forwards/request.
Prefill produces token #1; max_tokens=decode_steps+1, ignore_eos=True.
Warmup runs unprofiled. Only the profiled pass is traced.

Trace reading:
  Open *.trace.json in https://ui.perfetto.dev (Open trace file). CPU and GPU
  events live under different tracks; each CUDA stream gets its own GPU thread
  track. CPU record_function durations are host scopes, not GPU kernel time.
  For CUDA Graph replay, op attribution requires --enforce-eager --with-stack.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", "--model-path", required=True, help="Local Hugging Face model directory")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--input-len", type=int, default=64)
    p.add_argument("--decode-steps", type=int, default=32, help="Decode forwards; output length is this + 1")
    p.add_argument("--warmup-rounds", type=int, default=1)
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
    args = p.parse_args()
    for name in ("batch_size", "input_len", "decode_steps", "warmup_rounds", "max_model_len"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be >=1")
    if not 0 < args.gpu_memory_utilization < 1:
        p.error("--gpu-memory-utilization must be between 0 and 1")
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
    prefill_end = time.perf_counter()  # step consumed prefill's D2H event.
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
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError("CUDA profiler activity unavailable; check PyTorch/CUPTI installation")
    from nano_qwen.engine.llm_engine import LLMEngine
    from nano_qwen.engine.sequence import Sequence
    from nano_qwen.sampling_params import SamplingParams

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mode = "eager" if args.enforce_eager else "graph"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = out / f"bs{args.batch_size}_in{args.input_len}_dec{args.decode_steps}_{mode}_{args.trace_stage}_{stamp}_{os.getpid()}"
    print(f"Loading model; BS={args.batch_size}, input={args.input_len}, decode={args.decode_steps}, TP=1", flush=True)
    engine = LLMEngine(args.model, tensor_parallel_size=1, max_num_seqs=args.batch_size,
                       max_num_batched_tokens=args.batch_size * args.input_len,
                       max_model_len=args.max_model_len, enforce_eager=args.enforce_eager,
                       use_prefill_cudagraph=not args.disable_prefill_cudagraph,
                       enable_prefix_cache=False, gpu_memory_utilization=args.gpu_memory_utilization)
    try:
        engine.max_concurrent_batches = args.queue_depth
        runner = engine.model_runner
        runner.async_output = not args.sync_output
        runner.use_prefill_cudagraph = not args.disable_prefill_cudagraph
        block_size = engine.config.kvcache_block_size
        needed = args.batch_size * ((args.input_len + args.decode_steps + 1 + block_size - 1) // block_size)
        if engine.config.num_kvcache_blocks < needed:
            raise RuntimeError(f"Need at least {needed} KV blocks for fixed batch; allocated {engine.config.num_kvcache_blocks}")
        if engine.config.max_model_len < args.input_len + args.decode_steps + 1:
            raise RuntimeError("Model's actual maximum context is shorter than the workload")
        if not args.enforce_eager and args.batch_size not in runner.graphs:
            raise RuntimeError(f"No exact CUDA Graph for BS={args.batch_size}; use --enforce-eager or a captured BS")

        # Reject preemption rather than silently measuring a different workload.
        def reject_preempt(seq):
            raise RuntimeError(f"Request {seq.seq_id} would be preempted; insufficient KV capacity for fixed batch")
        engine.scheduler.preempt = reject_preempt

        with torch.inference_mode():
            for r in range(args.warmup_rounds):
                print(f"Warmup {r + 1}/{args.warmup_rounds} (full workload, profiler off)", flush=True)
                seqs = enqueue(engine, args, Sequence, SamplingParams, args.seed + r)
                timing = full_run(engine, seqs, args, torch)
            seqs = enqueue(engine, args, Sequence, SamplingParams, args.seed + args.warmup_rounds)
            if args.trace_stage == "decode":
                engine.step()  # Populate KV/GDN state and consume the prefill sample.
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            print(f"Recording {args.trace_stage} trace", flush=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=args.record_shapes, with_stack=args.with_stack,
                                        profile_memory=args.profile_memory) as prof:
                if args.trace_stage == "all":
                    with torch.profiler.record_function(f"nq/prefill/bs{args.batch_size}"):
                        engine.step()
                    prof.step()
                for i in range(args.decode_steps):
                    with torch.profiler.record_function(f"nq/decode/step{i:03d}/bs{args.batch_size}"):
                        engine.step()
                    prof.step()
                with torch.profiler.record_function("nq/benchmark_boundary/final_synchronize"):
                    torch.cuda.synchronize()
            check_finished(engine, seqs, args)

        trace_path = Path(str(stem) + ".trace.json")
        prof.export_chrome_trace(str(trace_path))
        gpu_event_count, gpu_tracks = ensure_gpu_tracks(trace_path, torch.cuda.get_device_name())
        averages = prof.key_averages(group_by_input_shape=args.record_shapes)
        table = "CPU self time\n" + averages.table(sort_by="self_cpu_time_total", row_limit=40)
        table += "\nGPU self time\n" + averages.table(sort_by="self_device_time_total", row_limit=40)
        Path(str(stem) + ".operators.txt").write_text(table, encoding="utf-8")
        try:
            commit = subprocess.check_output(["git", "-C", str(repo or Path.cwd()), "rev-parse", "HEAD"],
                                             text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
        summary = {"args": vars(args), "git_commit": commit,
                   "torch": torch.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(), "tensor_parallel_size": 1,
                   "trace": str(trace_path), "gpu_device_events": gpu_event_count,
                   "gpu_tracks": gpu_tracks,
                   "last_warmup_pass": timing,
                   "allocated_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
                   "notes": ["Warmup timing is not a statistical benchmark.",
                             "Synthetic fixed-length workload; no numerical correctness validation.",
                             "CPU and GPU events are normalized onto separate Perfetto tracks.",
                             "One GPU thread track is emitted per CUDA stream."]}
        Path(str(stem) + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(table)
        print(json.dumps({"last_warmup_pass": timing,
                          "gpu_device_events": gpu_event_count,
                          "gpu_tracks": gpu_tracks}, indent=2))
        print(f"Trace (CPU + separate GPU tracks): {trace_path}")
        print(f"Summary: {stem}.summary.json\nOperators: {stem}.operators.txt")
        if not gpu_event_count:
            raise RuntimeError("Trace saved but has no CUDA device events. Check CUPTI/profiler permissions; do not use it to infer GPU idle gaps.")
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
