#!/usr/bin/env python3
"""Profile one fixed batch through nano_qwen's real LLMEngine.step().

Place in the repository root or benchmarks/. No engine_bench_utils dependency.
Checked against Jung52/nano_qwen commit b50ac456ab0c2e1e809154d6d7f18d34a6d05e3c.
Inspired by SGLang's synthetic batch / warmup / Chrome trace workflow:
https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/one_batch.py
Profiler API: https://docs.pytorch.org/docs/stable/profiler.html

Examples (local model directory, one visible GPU, TP=1):
  CUDA_VISIBLE_DEVICES=0 python bench_one_batch.py --model /path/to/Qwen3.5
  python bench_one_batch.py --model /path/to/Qwen3.5 --trace-stage decode
  python bench_one_batch.py --model /path/to/Qwen3.5 --enforce-eager --with-stack

Defaults: BS=64, input=64 tokens/request, 32 decode forwards/request.
Prefill produces token #1; max_tokens=decode_steps+1, ignore_eos=True.
One full warmup and one unprofiled timing pass precede the profiled pass.
No extra synchronization or per-step logging inside the decode loop.
Only benchmark boundaries synchronize; the engine's own waits stay intact.

Trace reading:
  Open *.trace.json in https://ui.perfetto.dev (Open trace file).
  Look for nq/engine/{schedule,execute_model,sample_tokens,d2h_wait},
  nq/runner/{prepare_inputs,prepare_decode,run_model,sampler,d2h_copy},
  nq/graph/decode/bs64, nq/scheduler/postprocess, and CUDA runtime/GPU streams.
  CPU record_function durations measure host scopes, NOT GPU kernel duration.
  CUDA Graph replay does not re-execute Python layers; inspect GPU kernels and
  use a separate --enforce-eager trace with --with-stack for op attribution.
  record_shapes/stack/memory tracing add overhead and are opt-in.

Current-repo caveats:
  in_flight requests cannot be rescheduled until postprocess. A single full
  batch may therefore have queue occupancy=1 even with queue depth=2.
  Packed prefill >512 tokens takes eager fallback in current run_model;
  BS64 x input64=4096 is eager prefill even when decode graphs are enabled.
  This script observes those paths; it does not implement a different pipeline.
  Fixed-length synthetic tokens test performance, not generation correctness.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
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
    p.add_argument("--output-dir", default="logs/bench_one_batch")
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


@contextmanager
def annotate_engine(engine, torch, batch_size):
    """Temporary profiler annotations, with no change to scheduling semantics."""
    stats = Counter()
    runner = engine.model_runner

    @contextmanager
    def trace_event(name, cat="bench", args=None):
        suffix = "" if not args else " " + " ".join(f"{k}={v}" for k, v in args.items())
        with torch.profiler.record_function(f"nq/{cat}/{name}{suffix}"):
            yield

    with ExitStack() as cleanup:
        def patch(obj, name, replacement):
            # Remove temporary instance overrides on exit, restoring descriptors.
            owned = name in vars(obj)
            old = getattr(obj, name)
            setattr(obj, name, replacement)
            cleanup.callback(setattr if owned else delattr, obj, name, *([old] if owned else []))
            return old

        # These files import trace_event directly: patch both aliases, not only
        # nano_qwen.utils.trace.trace_event. Do not enable the CPU-only collector.
        for module in ("nano_qwen.engine.llm_engine", "nano_qwen.engine.model_runner"):
            patch(importlib.import_module(module), "trace_event", trace_event)

        def wrap(obj, name, label):
            old = getattr(obj, name)
            def call(*a, **kw):
                with torch.profiler.record_function(label):
                    return old(*a, **kw)
            patch(obj, name, call)

        wrap(engine.scheduler, "postprocess", "nq/scheduler/postprocess")
        for name in ("prepare_prefill", "prepare_decode", "prepare_block_tables", "prepare_sample"):
            wrap(runner, name, f"nq/runner/{name}")
        wrap(runner.model, "compute_logits", "nq/model/compute_logits")
        wrap(runner.input_batch, "update", "nq/input_batch/update")
        wrap(runner, "remove_request", "nq/runner/remove_request")

        original_execute = runner.execute_model
        def execute(seqs, is_prefill):
            if len(seqs) != batch_size:
                raise RuntimeError(f"Expected fixed BS={batch_size}, got {len(seqs)}; batch was split")
            stats["prefill_batches" if is_prefill else "decode_batches"] += 1
            return original_execute(seqs, is_prefill)
        patch(runner, "execute_model", execute)

        original_postprocess = engine.scheduler.postprocess
        def postprocess(*a, **kw):
            # step() pops the consumed batch just before postprocess.
            stats["max_queue_occupancy"] = max(stats["max_queue_occupancy"], len(engine.batch_queue) + 1)
            return original_postprocess(*a, **kw)
        patch(engine.scheduler, "postprocess", postprocess)

        # Scope the actual replay call; counters are actual replay invocations.
        def graph_range(graph, label, counter):
            replay = graph.replay
            def call():
                stats[counter] += 1
                with torch.profiler.record_function(label):
                    return replay()
            patch(graph, "replay", call)

        for bs, graph in getattr(runner, "graphs", {}).items():
            graph_range(graph, f"nq/graph/decode/bs{bs}", "decode_graph_replays")
        for bucket, entries in getattr(runner, "prefill_piecewise_graphs", {}).items():
            for layer, segments in entries.items():
                for part in ("pre", "post"):
                    graph_range(segments[part]["graph"], f"nq/graph/prefill/bucket{bucket}/layer{layer}/{part}",
                                "prefill_segment_replays")
        yield stats


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
    # This separate, uninstrumented pass supplies timing unaffected by profiler.
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
                full_run(engine, seqs, args, torch)
            print("Unprofiled timing pass", flush=True)
            seqs = enqueue(engine, args, Sequence, SamplingParams, args.seed + args.warmup_rounds)
            timing = full_run(engine, seqs, args, torch)
            seqs = enqueue(engine, args, Sequence, SamplingParams, args.seed + args.warmup_rounds + 1)
            if args.trace_stage == "decode":
                engine.step()  # Populate KV/GDN state and consume the prefill sample.
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            print(f"Recording {args.trace_stage} trace", flush=True)
            with annotate_engine(engine, torch, args.batch_size) as counters:
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

        trace_path = str(stem) + ".trace.json"
        prof.export_chrome_trace(trace_path)
        averages = prof.key_averages(group_by_input_shape=args.record_shapes)
        table = "CPU self time\n" + averages.table(sort_by="self_cpu_time_total", row_limit=40)
        table += "\nGPU self time\n" + averages.table(sort_by="self_device_time_total", row_limit=40)
        Path(str(stem) + ".operators.txt").write_text(table, encoding="utf-8")
        try:
            commit = subprocess.check_output(["git", "-C", str(repo or Path.cwd()), "rev-parse", "HEAD"],
                                             text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
        gpu_events = sum(getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA for e in prof.events())
        summary = {"args": vars(args), "git_commit": commit,
                   "torch": torch.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(), "tensor_parallel_size": 1,
                   "trace": trace_path, "cuda_device_events": gpu_events,
                   "unprofiled_single_pass": timing, "profiled_counters": dict(counters),
                   "allocated_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
                   "notes": ["Timing is a single separate unprofiled pass, not a statistical benchmark.",
                             "Synthetic fixed-length workload; no numerical correctness validation.",
                             "Prefill >512 packed tokens is eager in the checked repository.",
                             "Single in-flight batch cannot demonstrate overlap between independent batches."]}
        Path(str(stem) + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(table)
        print(json.dumps({"unprofiled": timing, "profiled_counters": dict(counters)}, indent=2))
        print(f"Trace: {trace_path}\nSummary: {stem}.summary.json\nOperators: {stem}.operators.txt")
        if not gpu_events:
            raise RuntimeError("Trace saved but has no CUDA device events. Check CUPTI/profiler permissions; do not use it to infer GPU idle gaps.")
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
