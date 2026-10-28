#!/usr/bin/env python3
"""Unprofiled wall-time benchmark for the three unified scheduling modes.

Reuses the real engine paths from bench_one_batch_trace (same source root,
same KV capacity policy, same graph audit) but never wraps the engine in
labels or a profiler. Per (batch-size, mode) an isolated process:
loads the model once, warms up >=3 full workloads, then measures >=10
fixed-seed workloads end to end:

    torch.cuda.synchronize(); t0 = perf_counter()
    enqueue fixed prompts; while not engine.is_finished(): engine.step()
    torch.cuda.synchronize(); t1 = perf_counter()

Model load, compilation, graph capture and all output verification stay
outside the timed window. The main metric is the wall time of the fixed
request set from enqueue to full drain. Tokens/s uses completed tokens.
Measured rounds use identical prompts across modes (seed+round), so greedy
outputs must match exactly across the three modes.

Modes rotate their run order across batch sizes to cancel drift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_one_batch_trace import (MODES, audit_graphs, config_fields, find_repo,
                                   save, source_info, workload_kv_pool)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--repo-root")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 32, 64])
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--warmup-rounds", type=int, default=3)
    p.add_argument("--measure-rounds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.99)
    p.add_argument("--mode", choices=("all",) + MODES, default="all")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--output-dir", default="logs/one_batch_perf")
    p.add_argument("--worker", choices=MODES, help=argparse.SUPPRESS)
    p.add_argument("--source-root", help=argparse.SUPPRESS)
    p.add_argument("--source-json", help=argparse.SUPPRESS)
    p.add_argument("--worker-output", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    for bs in a.batch_sizes:
        if bs not in [1, 2, 4, 8] + list(range(16, 513, 16)):
            p.error(f"Use an exact decode graph bucket, not batch size {bs}")
    if min(a.input_len, a.output_len, a.warmup_rounds, a.measure_rounds, a.timeout) < 1:
        p.error("Input, output, warmup, measure and timeout must be positive")
    if not 0 < a.gpu_memory_utilization < 1:
        p.error("--gpu-memory-utilization must be between 0 and 1")
    a.model = str(Path(a.model).resolve())
    a.output_dir = str(Path(a.output_dir).resolve())
    return a


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


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
        raise RuntimeError("CUDA is required")
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") != "0":
        raise RuntimeError("Unset CUDA_LAUNCH_BLOCKING before wall-time measurement")
    mode = a.worker
    if len(a.batch_sizes) != 1:
        raise RuntimeError("A perf worker measures exactly one batch size")
    a.batch_size = a.batch_sizes[0]
    out = Path(a.worker_output) if a.worker_output else Path(a.output_dir) / f"bs{a.batch_size}_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    block_size = Config.__dataclass_fields__["kvcache_block_size"].default
    max_model_len = math.ceil((a.input_len + a.output_len + 2) / block_size) * block_size
    metadata = dict(mode=mode, batch_size=a.batch_size, args=vars(a), source=json.loads(a.source_json),
                    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    max_model_len=max_model_len,
                    benchmark_policy={"prefill_graph": False, "kv_pool": "workload capacity + one block/request",
                                      "temperature": 1e-8, "ignore_eos": True, "profiler": False,
                                      "per_step_sync": False, "timing": "enqueue-to-drain wall time"})
    kwargs = dict(tensor_parallel_size=1, max_num_seqs=a.batch_size,
                  max_num_batched_tokens=a.batch_size * a.input_len, max_model_len=max_model_len,
                  enable_prefix_cache=False, enforce_eager=False, use_prefill_cudagraph=False,
                  gpu_memory_utilization=a.gpu_memory_utilization)
    if "async_scheduling" in Config.__dataclass_fields__:
        kwargs["async_scheduling"] = mode == "async"
    engine = None
    try:
        torch.manual_seed(a.seed)
        torch.cuda.set_per_process_memory_fraction(a.gpu_memory_utilization, device=0)
        with torch.inference_mode(), workload_kv_pool(ModelRunner, torch, a.batch_size, max_model_len):
            engine = LLMEngine(a.model, **kwargs)
        engine.max_concurrent_batches = 1 if mode == "no_async" else 2
        engine.model_runner.async_output = True
        graphs, audit = audit_graphs(engine, mode, a.batch_size)
        metadata.update(actual=audit, kv_blocks=engine.config.num_kvcache_blocks)
        save(out / "metadata.json", metadata)

        vocab = int(engine.config.hf_config.vocab_size)

        def enqueue(seed):
            if not engine.is_finished():
                raise RuntimeError("Previous workload did not drain")
            rng = random.Random(seed)
            prompts = [[rng.randrange(vocab) for _ in range(a.input_len)] for _ in range(a.batch_size)]
            seqs = [Sequence(prompt, SamplingParams(
                temperature=1e-8, max_tokens=a.output_len, ignore_eos=True)) for prompt in prompts]
            for seq in seqs:
                engine.scheduler.add(seq)
            return prompts, seqs

        def drain():
            ticks = 0
            while not engine.is_finished():
                engine.step()
                ticks += 1
                if ticks > a.output_len * 4 + 32:
                    raise RuntimeError("Engine failed to drain")
            torch.cuda.synchronize()

        def one_workload(seed):
            prompts, seqs = enqueue(seed)
            torch.cuda.synchronize()  # Fence before the timed window.
            start = time.perf_counter()
            drain()
            torch.cuda.synchronize()  # Fence after full drain.
            wall_ms = (time.perf_counter() - start) * 1000
            # Verification happens after the fence: no readback inside timing.
            if engine.scheduler.waiting or engine.scheduler.running:
                raise RuntimeError("Scheduler not empty after drain")
            if any(not s.is_finished or s.num_completion_tokens != a.output_len for s in seqs):
                raise RuntimeError("Unexpected completion count")
            if engine.model_runner.input_batch.seq_id_to_slot or engine.scheduler.block_manager.used_block_ids:
                raise RuntimeError("Persistent request slots or KV pages leaked")
            digest = hashlib.sha256(
                json.dumps([s.completion_token_ids for s in seqs]).encode()).hexdigest()
            return dict(wall_ms=wall_ms, tokens_per_s=a.batch_size * a.output_len / (wall_ms / 1000),
                        step_ms=wall_ms / (a.output_len - 1), output_sha256=digest)

        with torch.inference_mode():
            for i in range(a.warmup_rounds):
                enqueue(a.seed + 1000 + i)
                drain()
            samples = [dict(round=r, **one_workload(a.seed + r))  # Same prompts across modes.
                       for r in range(a.measure_rounds)]
        walls = [s["wall_ms"] for s in samples]
        report = dict(mode=mode, batch_size=a.batch_size, status="ok",
                      warmup_rounds=a.warmup_rounds, measure_rounds=a.measure_rounds,
                      wall_ms_median=statistics.median(walls),
                      wall_ms_p95=percentile(walls, 0.95), wall_ms_min=min(walls), wall_ms_max=max(walls),
                      tokens_per_s_median=statistics.median([s["tokens_per_s"] for s in samples]),
                      step_ms_median=statistics.median([s["step_ms"] for s in samples]),
                      output_sha256=samples[0]["output_sha256"], samples=samples)
        save(out / "perf.json", report)
        print(f"[bs{a.batch_size}/{mode}] median {report['wall_ms_median']:.1f} ms, "
              f"p95 {report['wall_ms_p95']:.1f} ms, {report['tokens_per_s_median']:.0f} tok/s", flush=True)
    finally:
        if engine is not None:
            engine.exit()


def parent(a):
    repo = find_repo(a.repo_root)
    if "async_scheduling" not in config_fields(repo):
        raise RuntimeError("The current checkout lacks async_scheduling; this benchmark compares "
                           "the unified modes and never falls back to other refs")
    info = source_info(repo)
    if info.get("dirty"):
        print(f"WARNING: uncommitted changes at {repo} are included in every mode", flush=True)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    modes = MODES if a.mode == "all" else (a.mode,)
    results = {}
    failures = []
    for i, bs in enumerate(a.batch_sizes):
        # Rotate mode order across batch sizes to cancel thermal/state drift.
        order = [modes[(i + k) % len(modes)] for k in range(len(modes))]
        print(f"[bs{bs}] mode order: {order}", flush=True)
        for mode in order:
            destination = out / f"bs{bs}_{mode}"
            destination.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--model", a.model,
                       "--worker", mode, "--source-root", str(repo), "--source-json", json.dumps(info),
                       "--worker-output", str(destination), "--batch-sizes", str(bs),
                       "--input-len", str(a.input_len), "--output-len", str(a.output_len),
                       "--warmup-rounds", str(a.warmup_rounds), "--measure-rounds", str(a.measure_rounds),
                       "--seed", str(a.seed), "--gpu-memory-utilization", str(a.gpu_memory_utilization),
                       "--output-dir", str(out)]
            env = dict(os.environ, PYTHONPATH=str(repo / "src"), PYTHONDONTWRITEBYTECODE="1")
            with (destination / "run.log").open("w", encoding="utf-8") as log:
                try:
                    result = subprocess.run(command, env=env, cwd=repo, stdout=log,
                                            stderr=subprocess.STDOUT, timeout=a.timeout)
                    failed = result.returncode != 0
                except subprocess.TimeoutExpired:
                    log.write("\nWorker timeout\n")
                    failed = True
            if failed or not (destination / "perf.json").is_file():
                failures.append(f"bs{bs}/{mode}")
                print(f"[bs{bs}/{mode}] FAILED; inspect {destination/'run.log'}", flush=True)
                continue
            results[(bs, mode)] = json.loads((destination / "perf.json").read_text(encoding="utf-8"))
    summary = dict(source=info, args=vars(a), batches=[])
    for bs in a.batch_sizes:
        rows = {mode: results.get((bs, mode)) for mode in modes if (bs, mode) in results}
        if not rows:
            continue
        baseline = rows.get("no_async") or next(iter(rows.values()))
        block = dict(batch_size=bs, modes={})
        for mode, report in rows.items():
            block["modes"][mode] = {k: report[k] for k in
                                    ("wall_ms_median", "wall_ms_p95", "tokens_per_s_median", "step_ms_median")}
            block["modes"][mode]["wall_ms_vs_no_async"] = (
                report["wall_ms_median"] / baseline["wall_ms_median"] if baseline is not report else 1.0)
        # Greedy outputs must be identical across modes for identical inputs.
        digests = {mode: r["output_sha256"] for mode, r in rows.items()}
        block["cross_mode_outputs_identical"] = len(set(digests.values())) == 1
        block["output_sha256"] = digests
        summary["batches"].append(block)
    save(out / "perf_summary.json", summary)
    for block in summary["batches"]:
        print(f"\nBS={block['batch_size']}  (median wall ms | p95 | tok/s | vs no_async)")
        for mode, row in block["modes"].items():
            print(f"  {mode:9s} {row['wall_ms_median']:8.1f} | {row['wall_ms_p95']:8.1f} | "
                  f"{row['tokens_per_s_median']:7.0f} | {row['wall_ms_vs_no_async']:.3f}")
        print(f"  cross-mode greedy outputs identical: {block['cross_mode_outputs_identical']}")
    if failures:
        raise RuntimeError("Failed workers: " + ", ".join(failures))
    if any(not b["cross_mode_outputs_identical"] for b in summary["batches"]):
        raise RuntimeError("Cross-mode greedy outputs differ; perf numbers are not comparable")


def main():
    a = parse_args()
    if a.worker:
        run_worker(a)
    else:
        parent(a)


if __name__ == "__main__":
    main()
