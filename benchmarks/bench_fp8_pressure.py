#!/usr/bin/env python3
"""Compare production per-channel/per-token FP8, block-128 FP8 and optional BF16.

Run on a Linux/WSL CUDA inference environment, preferably an otherwise idle 5090.
Default: BS {4,8} x input {256,512,1024} x output {512,1024}: 12 cases.
One engine and ONE benchmark warmup per model/mode/pass, then direct measurement
of the entire matrix. Default: one measured sweep, no stress or diagnostics.
The parent and SVG report require only Python's standard library.

Examples:
  python benchmarks/bench_fp8_pressure.py --per-channel-model /models/pc \
      --blockwise-model /models/block --bf16-model /models/bf16
  python benchmarks/bench_fp8_pressure.py --per-channel-model /models/pc \
      --blockwise-model /models/block --preset stress --profile --microbench
  python benchmarks/bench_fp8_pressure.py --self-test
  python benchmarks/bench_fp8_pressure.py --report-only logs/bench_fp8_pressure/RUN
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MIB = 2**20
PRESETS = {
    name: dict(batches=[4, 8], inputs=[256, 512, 1024], outputs=[512, 1024],
               rounds=rounds, warmup=1, passes=passes, stress_seconds=stress)
    for name, rounds, passes, stress in (("matrix", 1, 1, 0), ("quick", 1, 1, 0),
                                        ("stress", 3, 1, 30), ("full", 5, 2, 60))
}


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * p / 100
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def stats(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not values:
        return {"n": 0}
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0
    return dict(n=len(values), mean=mean, median=statistics.median(values),
                p90=percentile(values, 90), p95=percentile(values, 95),
                p99=percentile(values, 99), min=min(values), max=max(values),
                stdev=sd, cv=sd / mean if mean else 0)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def git_info():
    def git(*args):
        try:
            return subprocess.check_output(["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT), *args],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    source_hash = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        source_hash.update(path.relative_to(ROOT).as_posix().encode())
        source_hash.update(path.read_bytes())
    return dict(commit=git("rev-parse", "HEAD"), status=git("status", "--short"),
                diff_sha256=hashlib.sha256(git("diff", "--", "src").encode()).hexdigest(),
                source_sha256=source_hash.hexdigest(),
                benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def telemetry():
    fields = "name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem"
    try:
        result = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5)
        return dict(fields=fields.split(","), rows=list(csv.reader(result.stdout.splitlines())),
                    error=result.stderr.strip() if result.returncode else None)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-channel-model", help="nano_qwen_fp8_v1 checkpoint")
    parser.add_argument("--blockwise-model", help="HF dynamic FP8 block128 checkpoint")
    parser.add_argument("--bf16-model", help="Optional original BF16/FP16 checkpoint")
    parser.add_argument("--preset", choices=PRESETS, default="matrix")
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--input-lens", type=int, nargs="+")
    parser.add_argument("--output-lens", type=int, nargs="+", help="Includes the first token produced by prefill")
    parser.add_argument("--modes", nargs="+", choices=("eager", "decode_graph", "piecewise_graph"), default=["decode_graph"])
    parser.add_argument("--rounds", type=int, help="Measured matrix sweeps; default 1; every sample is retained")
    parser.add_argument("--warmup-rounds", type=int, choices=(0, 1), help="ONE global warmup per engine (default 1); never per case")
    parser.add_argument("--passes", type=int, help="Alternate model order across independent engine processes")
    parser.add_argument("--stress-seconds", type=float)
    parser.add_argument("--stress-warmup-seconds", type=float, default=0,
                        help="Must be 0: use the single global engine warmup")
    parser.add_argument("--skip-fixed", action="store_true", help="Skip fixed E2E; isolated decode still runs")
    parser.add_argument("--max-fixed-prefill-tokens", type=int, default=8192,
                        help="Skip full-batch prefill above this cap; stress still runs")
    parser.add_argument("--stress-prefill-budget", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--kv-cache-policy", choices=("workload", "auto"), default="workload",
                        help="Identical workload-sized KV pools (default), or production auto-fill for capacity pressure")
    parser.add_argument("--queue-depth", type=int, choices=(1, 2), default=2)
    parser.add_argument("--sync-output", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cooldown-seconds", type=float, default=2)
    parser.add_argument("--job-timeout-seconds", type=float, default=1800)
    parser.add_argument("--profile", action="store_true", help="Separate eager diagnostic traces, not timing rounds")
    parser.add_argument("--profile-decode-steps", type=int, default=8)
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--microbench", action="store_true", help="Separate per-shape quantize/GEMM/full-linear CUDA graph benchmark")
    parser.add_argument("--micro-iters", type=int, default=100)
    parser.add_argument("--micro-repeats", type=int, default=5)
    parser.add_argument("--output-dir", default="logs/bench_fp8_pressure")
    parser.add_argument("--resume", help="Resume an existing timestamp directory using its saved plan")
    parser.add_argument("--report-only", help="Regenerate compact JSON, CSV, Markdown and SVG from saved results; no CUDA needed")
    parser.add_argument("--retry-failed", action="store_true", help="With --resume, also retry completed failed jobs")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="CPU-only checks of accounting and aggregation")
    parser.add_argument("--worker-job", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_job or args.self_test or args.resume or args.report_only:
        return args
    preset = PRESETS[args.preset]
    for key, default in (("batch_sizes", "batches"), ("input_lens", "inputs"), ("output_lens", "outputs"),
                         ("rounds", "rounds"), ("warmup_rounds", "warmup"), ("passes", "passes"),
                         ("stress_seconds", "stress_seconds")):
        if getattr(args, key) is None:
            setattr(args, key, preset[default])
    for key in ("batch_sizes", "input_lens", "output_lens"):
        values = sorted(set(getattr(args, key)))
        if not values or min(values) < (2 if key == "output_lens" else 1):
            parser.error(f"Invalid --{key.replace('_', '-')}; outputs must be >=2, other sizes >=1")
        setattr(args, key, values)
    for key in ("rounds", "passes", "max_fixed_prefill_tokens", "stress_prefill_budget",
                "micro_iters", "micro_repeats", "profile_decode_steps"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be >=1")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if args.warmup_rounds < 0 or min(args.stress_seconds, args.stress_warmup_seconds, args.cooldown_seconds) < 0 or args.job_timeout_seconds <= 0:
        parser.error("Invalid duration")
    if args.stress_warmup_seconds != 0:
        parser.error("--stress-warmup-seconds must be 0; only one global warmup is supported")
    if args.skip_fixed:
        parser.error("The matrix separates prefill/decode/E2E; --skip-fixed is incompatible")
    if max(args.batch_sizes) * max(args.input_lens) > args.max_fixed_prefill_tokens:
        parser.error("Raise --max-fixed-prefill-tokens to at least max(batch sizes) * max(input lengths)")
    for key in ("per_channel_model", "blockwise_model", "bf16_model"):
        value = getattr(args, key)
        if value is None and key != "bf16_model":
            parser.error(f"--{key.replace('_', '-')} is required")
        if value:
            model = Path(value).resolve()
            if not (model / "config.json").is_file():
                parser.error(f"Missing model/config.json: {model}")
            setattr(args, key, str(model))
    if len({args.per_channel_model, args.blockwise_model, args.bf16_model} - {None}) != (3 if args.bf16_model else 2):
        parser.error("Each variant must use a different checkpoint directory")
    return args


def model_spec(path, variant):
    config = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    text = dict(config.get("text_config", config))
    nano_path = Path(path) / "nano_qwen_quantization.json"
    nano = json.loads(nano_path.read_text(encoding="utf-8")) if nano_path.is_file() else None
    hf = config.get("quantization_config") or text.get("quantization_config")
    if variant == "per_channel" and (not nano or nano.get("format") != "nano_qwen_fp8_v1"):
        raise ValueError(f"{path} is not a nano_qwen per-channel checkpoint")
    if variant == "blockwise" and (nano or not hf or hf.get("weight_block_size") != [128, 128]
                                  or hf.get("activation_scheme") != "dynamic" or hf.get("quant_method") != "fp8"):
        raise ValueError(f"{path} is not a dynamic block128 FP8 checkpoint")
    if variant == "bf16" and (nano or hf):
        raise ValueError("BF16 reference must be unquantized")
    for key in ("quantization_config", "_name_or_path", "transformers_version"):
        text.pop(key, None)
    signature = hashlib.sha256(json.dumps(text, sort_keys=True).encode()).hexdigest()
    return dict(path=path, architecture_sha256=signature, text_config=text,
                quantization=nano or hf, checkpoint_file_bytes=sum(p.stat().st_size for p in Path(path).glob("*.safetensors")),
                tokenizer_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(path) / "tokenizer.json", Path(path) / "tokenizer_config.json") if p.is_file()})


def activation_accounting(m, k, itemsize, scheme):
    source = m * k * itemsize
    if scheme not in ("per_channel", "blockwise"):
        return dict(source_bytes=source, fp8_bytes=0, scale_bytes=0, fp8_representation_bytes=0,
                    representation_saved_bytes=0, input_plus_quantized_bytes=source)
    quantized = m * k
    scale = m * (k // 128 if scheme == "blockwise" else 1) * 4
    return dict(source_bytes=source, fp8_bytes=quantized, scale_bytes=scale,
                fp8_representation_bytes=quantized + scale,
                representation_saved_bytes=source - quantized - scale,
                input_plus_quantized_bytes=source + quantized + scale)


def storage_inventory(runner, torch):
    seen = set()
    rows = []
    for name, module in runner.model.named_modules():
        params = list(module.named_parameters(recurse=False, remove_duplicate=False))
        buffers = list(module.named_buffers(recurse=False, remove_duplicate=False))
        if not params and not buffers:
            continue
        row = dict(module=name or "<root>", module_type=type(module).__name__,
                   method=type(getattr(module, "quant_method", None)).__name__,
                   weight_bytes=0, scale_bytes=0, other_parameter_bytes=0, buffer_bytes=0,
                   physical_bytes=0, equivalent_bf16_bytes=0, tensors=[])
        for field, tensor in params + buffers:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            physical = 0 if key in seen else storage.nbytes()
            seen.add(key)
            logical = tensor.numel() * tensor.element_size()
            scale = field in ("weight_scale", "weight_scale_inv")
            is_buffer = any(field == f for f, _ in buffers)
            category = "buffer_bytes" if is_buffer else "scale_bytes" if scale else "weight_bytes" if field == "weight" else "other_parameter_bytes"
            row[category] += physical
            row["physical_bytes"] += physical
            equivalent = 0 if scale else (2 * tensor.numel() if tensor.dtype == torch.float8_e4m3fn else logical)
            row["equivalent_bf16_bytes"] += equivalent if physical else 0
            row["tensors"].append(dict(name=field, shape=list(tensor.shape), dtype=str(tensor.dtype),
                                       logical_bytes=logical, physical_bytes=physical, shared_storage=physical == 0))
            if field == "weight":
                row.update(weight_shape=list(tensor.shape), weight_dtype=str(tensor.dtype))
        if hasattr(module, "input_size") and hasattr(module, "output_size"):
            row.update(k=module.input_size, n=module.output_size,
                       compute_itemsize=module.compute_dtype.itemsize)
        row["saved_vs_equivalent_bf16_bytes"] = row["equivalent_bf16_bytes"] - row["physical_bytes"]
        rows.append(row)
    totals = {key: sum(row[key] for row in rows) for key in
              ("weight_bytes", "scale_bytes", "other_parameter_bytes", "buffer_bytes", "physical_bytes",
               "equivalent_bf16_bytes", "saved_vs_equivalent_bf16_bytes")}
    states = [t for layer in runner.gdn_layers for t in (layer.conv_states, layer.recurrent_states)]
    totals.update(kv_cache_bytes=runner.kv_cache.untyped_storage().nbytes(),
                  gdn_state_bytes=sum(t.untyped_storage().nbytes() for t in states),
                  kv_blocks=runner.config.num_kvcache_blocks,
                  state_slots=runner.config.max_num_seqs)
    # Small shared-weight probes catch accidentally comparing different model families.
    anchors = {}
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        tensor = runner.model.get_parameter(name).detach().reshape(-1)
        # Integer arithmetic avoids float32 linspace rounding past the last
        # element for the large (vocab * hidden) tied embedding matrix.
        indices = torch.arange(128, device=tensor.device, dtype=torch.long)
        indices = indices * (tensor.numel() - 1) // 127
        anchors[name] = hashlib.sha256(tensor[indices].float().cpu().numpy().tobytes()).hexdigest()
    return dict(totals=totals, modules=rows, unquantized_weight_probes=anchors)


def memory_snapshot(torch):
    free, total = torch.cuda.mem_get_info()
    return dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                device_used_bytes=total - free, device_total_bytes=total,
                allocator_retries=torch.cuda.memory_stats().get("num_alloc_retries", 0),
                allocator_ooms=torch.cuda.memory_stats().get("num_ooms", 0))


@contextmanager
def kv_allocation_policy(job, torch):
    from nano_qwen.engine.model_runner import ModelRunner
    original = ModelRunner.allocate_kv_cache

    def allocate(runner):
        config, hf = runner.config, runner.config.hf_config
        modules = [m for m in runner.model.modules() if hasattr(m, "k_cache") and hasattr(m, "v_cache")]
        if not modules:
            raise RuntimeError("No KV-cache-bearing attention layers")
        heads = hf.num_key_value_heads // runner.world_size
        dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        blocks = job["batch_size"] * math.ceil(job["max_model_len"] / runner.block_size)
        # All variants reserve the same worst-case request capacity, plus one block per slot.
        blocks += job["batch_size"]
        block_bytes = 2 * len(modules) * runner.block_size * heads * dim * hf.dtype.itemsize
        free, total = torch.cuda.mem_get_info()
        memory = torch.cuda.memory_stats()
        transient = memory["allocated_bytes.all.peak"] - memory["allocated_bytes.all.current"]
        budget = int(total * config.gpu_memory_utilization) - (total - free) - transient
        if blocks * block_bytes > budget:
            raise RuntimeError(f"Workload KV pool exceeds VRAM budget: need {blocks * block_bytes / MIB:.1f} MiB, "
                               f"available {budget / MIB:.1f} MiB; reduce batch/input/output or use a smaller model")
        config.num_kvcache_blocks = blocks
        runner.kv_cache = torch.empty((2, len(modules), blocks, runner.block_size, heads, dim),
                                      dtype=hf.dtype, device="cuda")
        for index, module in enumerate(modules):
            module.k_cache = runner.kv_cache[0, index]
            module.v_cache = runner.kv_cache[1, index]

    if job["options"]["kv_cache_policy"] == "workload":
        ModelRunner.allocate_kv_cache = allocate
    try:
        yield
    finally:
        ModelRunner.allocate_kv_cache = original


def make_sequences(engine, job, output_len, seed):
    from nano_qwen.engine.sequence import Sequence
    from nano_qwen.sampling_params import SamplingParams
    rng = random.Random(seed)
    vocab = int(engine.config.hf_config.vocab_size)
    return [Sequence([rng.randrange(vocab) for _ in range(job["input_len"])],
                     SamplingParams(temperature=1, max_tokens=output_len, ignore_eos=True))
            for _ in range(job["batch_size"])]


def verify_drained(engine, seqs):
    if not engine.is_finished() or engine.model_runner.input_batch.seq_id_to_slot:
        raise RuntimeError("Engine or persistent slots did not drain")
    if any(not seq.is_finished or seq.num_completion_tokens != seq.max_tokens for seq in seqs):
        raise RuntimeError("Unexpected output length")


def fixed_run(engine, job, output_len, seed, torch):
    seqs = make_sequences(engine, job, output_len, seed)
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = memory_snapshot(torch)
    enqueue_start = time.perf_counter()
    for seq in seqs:
        engine.scheduler.add(seq)
    enqueue_ms = (time.perf_counter() - enqueue_start) * 1000
    started = time.perf_counter()
    engine.step()
    # Async output must not shift unfinished prefill GPU work into decode.
    torch.cuda.synchronize()
    prefill_end = time.perf_counter()
    if any(seq.num_completion_tokens != 1 for seq in seqs):
        raise RuntimeError("Fixed run requires a single complete prefill of all requests")
    step_ms = []
    for _ in range(output_len - 1):
        step_start = time.perf_counter()
        engine.step()
        step_ms.append((time.perf_counter() - step_start) * 1000)
    torch.cuda.synchronize()
    ended = time.perf_counter()
    verify_drained(engine, seqs)
    elapsed, prefill, decode = ended - enqueue_start, prefill_end - started, ended - prefill_end
    bs = job["batch_size"]
    after = memory_snapshot(torch)
    return dict(output_len=output_len, seed=seed, e2e_ms=elapsed * 1000,
                enqueue_ms=enqueue_ms, prefill_ms=prefill * 1000, decode_ms=decode * 1000,
                ttft_ms=(prefill_end - enqueue_start) * 1000,
                decode_context_start=job["input_len"] + 1,
                decode_context_end=job["input_len"] + output_len - 1,
                decode_steps=output_len - 1, input_tokens=bs * job["input_len"], output_tokens=bs * output_len,
                decode_tokens=bs * (output_len - 1),
                e2e_output_tokens_per_s=bs * output_len / elapsed,
                e2e_total_tokens_per_s=bs * (job["input_len"] + output_len) / elapsed,
                requests_per_s=bs / elapsed, prefill_input_tokens_per_s=bs * job["input_len"] / prefill,
                decode_tokens_per_s=bs * (output_len - 1) / decode,
                tpot_ms=decode * 1000 / (output_len - 1),
                decode_step_ms=step_ms, decode_step_distribution=stats(step_ms),
                decode_step_timing="host engine.step wall time; no per-step CUDA synchronization; not GPU time or ITL",
                memory_before=before, memory_after=after,
                transient_peak_bytes=max(0, after["peak_allocated_bytes"] - before["allocated_bytes"]))


def isolated_decode_run(engine, job, output_len, seed, torch):
    seqs = make_sequences(engine, job, output_len, seed)
    torch.manual_seed(seed)
    old_depth = engine.max_concurrent_batches
    engine.max_concurrent_batches = 1
    for seq in seqs:
        engine.scheduler.add(seq)
    held = []
    prepare_steps = 0
    torch.cuda.synchronize()
    prepare_start = time.perf_counter()
    try:
        # Hold completed prefills outside the ready queue so later prefills
        # cannot mix decode tokens into the preparation phase.
        while engine.scheduler.waiting or engine.batch_queue:
            engine.step()
            prepare_steps += 1
            held.extend(engine.scheduler.running)
            engine.scheduler.running.clear()
            if prepare_steps > job["batch_size"] * (job["input_len"] + 1):
                raise RuntimeError("Isolated decode prefill preparation is stuck")
    finally:
        engine.scheduler.running.extend(held)
        engine.max_concurrent_batches = old_depth
    torch.cuda.synchronize()
    prepared_ms = (time.perf_counter() - prepare_start) * 1000
    if len(held) != len(seqs) or any(seq.num_completion_tokens != 1 for seq in seqs):
        raise RuntimeError("Isolated decode preparation did not produce exactly one token per request")
    torch.cuda.reset_peak_memory_stats()
    before = memory_snapshot(torch)
    old_budget = engine.scheduler.max_num_batched_tokens
    engine.scheduler.max_num_batched_tokens = max(old_budget, job["batch_size"])
    started = time.perf_counter()
    steps = []
    try:
        for _ in range(output_len - 1):
            step_start = time.perf_counter()
            engine.step()
            steps.append((time.perf_counter() - step_start) * 1000)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    finally:
        engine.scheduler.max_num_batched_tokens = old_budget
    verify_drained(engine, seqs)
    after = memory_snapshot(torch)
    return dict(output_len=output_len, seed=seed, prepared_prefill_ms=prepared_ms,
                prepared_prefill_steps=prepare_steps, decode_ms=elapsed * 1000,
                decode_steps=output_len - 1, decode_tokens=job["batch_size"] * (output_len - 1),
                decode_tokens_per_s=job["batch_size"] * (output_len - 1) / elapsed,
                tpot_ms=elapsed * 1000 / (output_len - 1), decode_step_ms=steps,
                memory_before=before, memory_after=after,
                transient_peak_bytes=max(0, after["peak_allocated_bytes"] - before["allocated_bytes"]))


@contextmanager
def path_audit(runner):
    from nano_qwen.utils.context import get_context
    counts = Counter()
    original = runner.run_model

    def wrapped(ids, positions, is_prefill):
        descriptor = get_context().batch_descriptor
        mode = descriptor.mode if descriptor else ("prefill" if is_prefill else "decode")
        rows = ids.shape[0]
        if runner.enforce_eager:
            path = "eager"
        elif mode == "mixed":
            path = "mixed_eager"
        elif mode == "prefill" and runner.use_prefill_cudagraph:
            graph_size = next((n for n in runner.prefill_graph_sizes if n >= rows), None)
            path = "prefill_piecewise" if graph_size in runner.prefill_piecewise_graphs else "prefill_eager_fallback"
        elif mode == "prefill":
            path = "prefill_eager"
        else:
            path = "decode_graph" if rows in runner.graphs else "decode_eager_fallback"
        counts[f"{path}/rows{rows}"] += 1
        return original(ids, positions, is_prefill)

    runner.run_model = wrapped
    try:
        yield counts
    finally:
        runner.run_model = original


def sustained_run(engine, job, torch):
    from nano_qwen.engine.sequence import Sequence
    from nano_qwen.sampling_params import SamplingParams
    options = job["options"]
    torch.manual_seed(options["seed"] + job["pass"] * 10000)
    rng = random.Random(options["seed"] + job["pass"] * 10000)
    vocab = int(engine.config.hf_config.vocab_size)
    # Pre-generate prompts outside the measured interval; admission overhead is included.
    buckets = [max(1, job["input_len"] * numerator // 4) for numerator in (2, 3, 4)]
    pool = [[rng.randrange(vocab) for _ in range(buckets[i % len(buckets)])] for i in range(128)]
    active, completed = {}, []
    ttft, itl, latency, step_times = [], [], [], []
    generated = admitted_input = admitted = preemptions = 0
    original_preempt = engine.scheduler.preempt

    def preempt(seq):
        nonlocal preemptions
        preemptions += 1
        return original_preempt(seq)

    engine.scheduler.preempt = preempt
    engine.scheduler.max_num_batched_tokens = options["stress_prefill_budget"]
    engine.max_concurrent_batches = options["queue_depth"]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = memory_snapshot(torch)
    start = time.perf_counter()
    deadline = start + options["stress_seconds"]
    running_window = None
    try:
        while active or time.perf_counter() < deadline:
            now = time.perf_counter()
            while len(active) < job["batch_size"] and now < deadline:
                prompt = pool[admitted % len(pool)]
                output_len = options["output_lens"][admitted % len(options["output_lens"])]
                seq = Sequence(prompt, SamplingParams(temperature=1, max_tokens=output_len, ignore_eos=True))
                submitted = time.perf_counter()
                engine.scheduler.add(seq)
                active[seq.seq_id] = [seq, submitted, submitted, 0]
                admitted += 1
                admitted_input += len(prompt)
                now = time.perf_counter()
            if not active:
                break
            step_start = time.perf_counter()
            engine.step()
            now = time.perf_counter()
            step_times.append((now - step_start) * 1000)
            finished = []
            for seq_id, state in active.items():
                seq, submitted, last_time, last_count = state
                count = seq.num_completion_tokens
                if count > last_count:
                    if last_count == 0:
                        ttft.append((now - submitted) * 1000)
                    else:
                        itl.append((now - last_time) * 1000)
                    generated += count - last_count
                    state[2:] = [now, count]
                if seq.is_finished:
                    if count != seq.max_tokens:
                        raise RuntimeError("Sustained request stopped early")
                    latency.append((now - submitted) * 1000)
                    completed.append(dict(input_tokens=seq.num_prompt_tokens, output_tokens=count,
                                          submitted_ms=(submitted - start) * 1000, finished_ms=(now - start) * 1000))
                    finished.append(seq_id)
            for seq_id in finished:
                del active[seq_id]
            if running_window is None and now >= deadline:
                running_window = dict(elapsed_s=now - start, output_tokens=generated,
                                      completed_requests=len(completed), active_requests=len(active))
            if now - start > options["stress_seconds"] + options["job_timeout_seconds"] / 2:
                raise RuntimeError("Sustained workload exceeded drain timeout")
        torch.cuda.synchronize()
        end = time.perf_counter()
        verify_drained(engine, [])
        if running_window is None:
            running_window = dict(elapsed_s=end - start, output_tokens=generated,
                                  completed_requests=len(completed), active_requests=0)
        elapsed = end - start
        after = memory_snapshot(torch)
        expected = sum(row["output_tokens"] for row in completed)
        if generated != expected or len(completed) != admitted:
            raise RuntimeError("Sustained token/request accounting mismatch")
        return dict(duration_with_drain_s=elapsed, admission_window_s=options["stress_seconds"],
                    drain_s=max(0, elapsed - options["stress_seconds"]), requests=admitted,
                    input_tokens=admitted_input, output_tokens=generated,
                    output_tokens_per_s=generated / elapsed,
                    total_tokens_per_s=(admitted_input + generated) / elapsed,
                    requests_per_s=admitted / elapsed,
                    window_output_tokens_per_s=running_window["output_tokens"] / running_window["elapsed_s"],
                    window=running_window, ttft_ms=stats(ttft), itl_ms=stats(itl), request_e2e_ms=stats(latency),
                    engine_step_ms=stats(step_times), preemptions=preemptions,
                    memory_before=before, memory_after=after,
                    ttft_samples_ms=ttft, itl_samples_ms=itl, request_e2e_samples_ms=latency,
                    completions=completed)
    finally:
        engine.scheduler.preempt = original_preempt


def kernel_category(name):
    name = name.lower()
    if "_quantize_rows" in name or "_quantize_groups" in name:
        return "activation_quantization"
    if "_block_mm" in name or ("cutlass" in name and ("e4m3" in name or "float8" in name)):
        return "fp8_gemm_including_scales"
    if any(key in name for key in ("gdn", "delta_rule", "chunk_delta", "chunk_fwd", "wy_fast")):
        return "gdn"
    if any(key in name for key in ("flash_fwd", "flash_bwd", "fmha", "attention", "batchprefill", "batchdecode")):
        return "attention"
    if "cutlass" in name or "gemm" in name or "cublas" in name:
        return "other_gemm_check_lm_head_scope"
    if "memcpy" in name:
        return "copy"
    return "other"


@contextmanager
def profile_scopes(runner, torch):
    original = runner.model.compute_logits
    hooks = []

    def logits(hidden):
        with torch.profiler.record_function(f"nq/lm_head/rows{hidden.shape[0]}"):
            return original(hidden)

    runner.model.compute_logits = logits
    import nano_qwen.quantization.fp8 as row_fp8
    import nano_qwen.quantization.fp8_block as block_fp8
    original_quantizers = [(row_fp8, "quantize_fp8_per_token", row_fp8.quantize_fp8_per_token),
                           (block_fp8, "quantize_fp8_per_group", block_fp8.quantize_fp8_per_group)]
    for owner, name, function in original_quantizers:
        def quantized(x, original=function, scope_name=name):
            with torch.profiler.record_function("nq/activation_quantization/" + scope_name):
                return original(x)
        setattr(owner, name, quantized)
    from nano_qwen.layers.linear import LinearBase
    stacks = {}
    for name, module in runner.model.named_modules():
        if not isinstance(module, LinearBase):
            continue

        def before(layer, inputs, scope_name=name):
            scope = torch.profiler.record_function("nq/linear/" + scope_name)
            scope.__enter__()
            stacks.setdefault(id(layer), []).append(scope)

        def after(layer, inputs, output):
            stacks[id(layer)].pop().__exit__(None, None, None)

        hooks.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))
    try:
        yield
    finally:
        runner.model.compute_logits = original
        for owner, name, function in original_quantizers:
            setattr(owner, name, function)
        for hook in hooks:
            hook.remove()
        for scopes in stacks.values():
            for scope in reversed(scopes):
                scope.__exit__(None, None, None)


def profile_workload(engine, job, torch, output_dir):
    runner = engine.model_runner
    original_eager = runner.enforce_eager
    original_budget = engine.scheduler.max_num_batched_tokens
    runner.enforce_eager = True
    engine.scheduler.max_num_batched_tokens = job["batch_size"] * job["input_len"]
    decode_steps = job["options"]["profile_decode_steps"]
    results = {}
    try:
        # Warm the eager path even if the timed engine used graph replay.
        fixed_run(engine, job, decode_steps + 1, job["options"]["seed"], torch)
        for stage in ("prefill", "decode"):
            seqs = make_sequences(engine, job, decode_steps + 1, job["options"]["seed"])
            for seq in seqs:
                engine.scheduler.add(seq)
            if stage == "decode":
                engine.step()
            torch.cuda.synchronize()
            with profile_scopes(runner, torch), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True, with_stack=job["options"]["with_stack"], profile_memory=False) as prof:
                for i in range(1 if stage == "prefill" else decode_steps):
                    with torch.profiler.record_function(f"nq/{stage}/step{i}"):
                        engine.step()
                    prof.step()
                torch.cuda.synchronize()
            if stage == "prefill":
                for _ in range(decode_steps):
                    engine.step()
            verify_drained(engine, seqs)
            trace = output_dir / f"{stage}.eager.trace.json"
            prof.export_chrome_trace(str(trace))
            rows, category_rows, scopes = summarize_trace(trace)
            total = sum(row["total_us"] for row in rows)
            save_csv(output_dir / f"{stage}.kernels.csv", rows)
            save_csv(output_dir / f"{stage}.kernel_categories.csv", category_rows)
            save_csv(output_dir / f"{stage}.scopes.csv", scopes)
            results[stage] = dict(trace=str(trace), gpu_event_count=sum(row["calls"] for row in rows),
                                  gpu_kernel_sum_ms=total / 1000, kernel_categories=category_rows, scopes=scopes)
            if not total:
                raise RuntimeError("Trace has no CUDA events; check CUPTI")
        return results
    finally:
        runner.enforce_eager = original_eager
        engine.scheduler.max_num_batched_tokens = original_budget


def summarize_trace(trace):
    kernels, scope_rows = {}, {}
    events = json.loads(Path(trace).read_text(encoding="utf-8"))["traceEvents"]
    for event in events:
        name, category = event.get("name", ""), event.get("cat")
        if event.get("ph") != "X":
            continue
        # GPU user annotations are CUDA events too. They must not be added
        # to kernel time, which would count nested operations repeatedly.
        if category == "kernel":
            row = kernels.setdefault(name, dict(name=name, calls=0, total_us=0, category=kernel_category(name)))
            row["calls"] += 1
            row["total_us"] += event["dur"]
        if name.startswith("nq/") and category in ("user_annotation", "gpu_user_annotation"):
            row = scope_rows.setdefault(name, dict(name=name, cpu_calls=0, gpu_calls=0,
                                                   inclusive_cpu_ms=0, inclusive_gpu_span_ms=0))
            if category == "user_annotation":
                row["cpu_calls"] += 1
                row["inclusive_cpu_ms"] += event["dur"] / 1000
            else:
                row["gpu_calls"] += 1
                row["inclusive_gpu_span_ms"] += event["dur"] / 1000
    total = sum(row["total_us"] for row in kernels.values())
    rows = sorted(kernels.values(), key=lambda row: -row["total_us"])
    categories = {}
    for row in rows:
        row.update(avg_us=row["total_us"] / row["calls"], gpu_sum_pct=100 * row["total_us"] / total if total else 0)
        categories[row["category"]] = categories.get(row["category"], 0) + row["total_us"]
    category_rows = [dict(category=name, total_us=value, gpu_sum_pct=100 * value / total if total else 0)
                     for name, value in sorted(categories.items(), key=lambda item: -item[1])]
    return rows, category_rows, list(scope_rows.values())


def graph_microtime(fn, torch, iterations, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Amortize host replay submission cost over several operations.
        retained = [fn() for _ in range(10)]
    values = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / (10 * iterations))
    del retained, graph
    return stats(values)


def microbench(runner, job, torch):
    from nano_qwen.layers.linear import LinearBase
    from nano_qwen.quantization.fp8 import Fp8LinearMethod, quantize_fp8_per_token
    from nano_qwen.quantization.fp8_block import Fp8BlockLinearMethod, _block_mm, quantize_fp8_per_group
    import triton
    import torch.nn.functional as functional
    options = job["options"]
    rows = []
    seen = set()
    for name, layer in runner.model.named_modules():
        if not isinstance(layer, LinearBase):
            continue
        if (layer.input_size, layer.output_size, type(layer.quant_method)) in seen:
            continue
        seen.add((layer.input_size, layer.output_size, type(layer.quant_method)))
        scheme = ("per_channel" if isinstance(layer.quant_method, Fp8LinearMethod) else
                  "blockwise" if isinstance(layer.quant_method, Fp8BlockLinearMethod) else "unquantized")
        for m in sorted({job["batch_size"], min(job["batch_size"] * job["input_len"], job["prefill_budget"])}):
            torch.manual_seed(options["seed"])
            x = torch.randn((m, layer.input_size), dtype=layer.compute_dtype, device="cuda")
            timings = {}
            if scheme != "unquantized":
                quantize = quantize_fp8_per_token if scheme == "per_channel" else quantize_fp8_per_group
                q, scale = quantize(x)
                if scheme == "per_channel":
                    ws = layer.weight_scale.t().contiguous()

                    def gemm():
                        return torch._scaled_mm(q, layer.weight.t(), scale, ws,
                                                out_dtype=layer.compute_dtype, use_fast_accum=False)
                else:
                    y = torch.empty((m, layer.output_size), dtype=layer.compute_dtype, device="cuda")

                    def gemm():
                        _block_mm[(triton.cdiv(m, 16), triton.cdiv(layer.output_size, 64))](
                            q, layer.weight, scale, layer.weight_scale_inv, y,
                            m, layer.output_size, layer.input_size, 16, 64, num_warps=4)
                        return y

                timings["activation_quantization_us"] = graph_microtime(lambda: quantize(x), torch, options["micro_iters"], options["micro_repeats"])
                timings["gemm_with_scales_us"] = graph_microtime(gemm, torch, options["micro_iters"], options["micro_repeats"])
            else:
                gemm = lambda: functional.linear(x, layer.weight, layer.bias)
                timings["gemm_us"] = graph_microtime(gemm, torch, options["micro_iters"], options["micro_repeats"])
            timings["full_linear_us"] = graph_microtime(lambda: layer(x), torch, options["micro_iters"], options["micro_repeats"])
            output = layer(x)
            if not torch.isfinite(output).all():
                raise RuntimeError(f"Nonfinite microbench output: {name}")
            # Arithmetic check against reconstructed FP32 values, NOT BF16 quantization quality.
            a, w = x[:16].float(), layer.weight[:128].float()
            if scheme == "per_channel":
                a = q[:16].float() * scale[:16]
                w *= layer.weight_scale[:128]
            elif scheme == "blockwise":
                a = q[:16].float() * scale[:16].repeat_interleave(128, dim=1)
                w *= layer.weight_scale_inv[:math.ceil(w.shape[0] / 128)].repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:w.shape[0], :w.shape[1]]
            reference = a @ w.t()
            if layer.bias is not None:
                reference += layer.bias[:w.shape[0]].float()
            actual = output[:a.shape[0], :w.shape[0]].float()
            error = (actual - reference).norm() / reference.norm().clamp_min(1e-12)
            arithmetic_error = error.item()
            if not math.isfinite(arithmetic_error) or arithmetic_error > 0.05:
                raise RuntimeError(f"Large GEMM arithmetic error at {name}: relative L2={arithmetic_error}")
            gemm_key = "gemm_with_scales_us" if scheme != "unquantized" else "gemm_us"
            elapsed_us = timings[gemm_key]["median"]
            rows.append(dict(module=name, scheme=scheme, m=m, n=layer.output_size, k=layer.input_size,
                             timings=timings, arithmetic_relative_l2=arithmetic_error,
                             useful_gemm_tflops=2 * m * layer.output_size * layer.input_size / elapsed_us / 1e6,
                             activation_bytes=activation_accounting(m, layer.input_size, x.element_size(), scheme)))
            print(f"  micro {scheme} {name} M={m} N={layer.output_size} K={layer.input_size}", flush=True)
    return rows


def workload_cases(batch_sizes, input_lens, output_lens):
    return [dict(batch_size=bs, input_len=length, output_len=output)
            for bs, length, output in itertools.product(batch_sizes, input_lens, output_lens)]


def case_job(job, case):
    return {**job, **case, "max_model_len": case["input_len"] + case["output_len"]}


def measure_matrix(engine, job, torch, report, checkpoint, run=fixed_run):
    """One global warmup, then rounds of ALL cases. Injection enables CPU checks."""
    options = job["options"]
    cases = job["cases"]
    warmed_shapes = set()
    report["warmup"] = dict(requested_runs=options["warmup_rounds"], completed_runs=0,
                            policy="once per engine at largest geometry; never per case")
    if options["warmup_rounds"]:
        warm = max(cases, key=lambda c: (c["batch_size"], c["input_len"], c["output_len"]))
        report["active_phase"] = "global_warmup"
        checkpoint()
        sample = run(engine, case_job(job, warm), warm["output_len"], options["seed"] + 1000000, torch)
        report["warmup"].update(completed_runs=1, case=warm, e2e_ms=sample["e2e_ms"])
        warmed_shapes.add((warm["batch_size"], warm["input_len"]))
        checkpoint()
    measured = 0
    for round_id in range(options["rounds"]):
        for case_id, case in enumerate(cases):
            shape = (case["batch_size"], case["input_len"])
            report["active_phase"] = f"matrix/round{round_id}/case{case_id}"
            report["active_case"] = case
            checkpoint()
            seed = options["seed"] + job["pass"] * 10000 + round_id
            row = run(engine, case_job(job, case), case["output_len"], seed, torch)
            row.update(case, **{"round": round_id}, case_id=case_id, measurement_order=measured,
                       first_run_of_prefill_geometry=shape not in warmed_shapes,
                       benchmark_warmup_completed=report["warmup"]["completed_runs"])
            warmed_shapes.add(shape)
            measured += 1
            report["fixed"].append(row)
            checkpoint()
            print(f"  [{measured}/{len(cases) * options['rounds']}] bs={case['batch_size']} "
                  f"in={case['input_len']} out={case['output_len']} round={round_id + 1}: "
                  f"prefill={row['prefill_ms']:.2f}ms decode={row['decode_tokens_per_s']:.1f}tok/s "
                  f"E2E={row['e2e_ms']:.2f}ms ({row['e2e_output_tokens_per_s']:.1f}tok/s)", flush=True)
    report.pop("active_case", None)


def run_worker(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    report_path = Path(job["result"])
    output_dir = report_path.parent
    report = dict(job=job, status="starting", fixed=[], decode_only=[], fixed_skips=[], diagnostics={})
    save_json(report_path, report)
    engine = None
    phase = "import"
    try:
        import torch
        from nano_qwen.engine.llm_engine import LLMEngine
        from nano_qwen.quantization.fp8 import Fp8LinearMethod
        from nano_qwen.quantization.fp8_block import Fp8BlockLinearMethod
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; use the Linux/WSL inference environment")
        options = job["options"]
        torch.manual_seed(options["seed"])
        packages = {}
        for package in ("transformers", "triton", "flashinfer-python", "flash-attn", "flash-qla", "tilelang", "safetensors"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = "not installed as this distribution"
        report["environment"] = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                                      gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                                      cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                      triton=__import__("triton").__version__, packages=packages, before=telemetry(), git=git_info(),
                                      cuda_environment={key: os.environ.get(key) for key in
                                                        ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "OMP_NUM_THREADS", "CUDA_MODULE_LOADING")})
        phase = "engine_initialization"
        torch.cuda.reset_peak_memory_stats()
        init_start = time.perf_counter()
        with kv_allocation_policy(job, torch):
            engine = LLMEngine(job["model"], tensor_parallel_size=1, max_num_seqs=job["batch_size"],
                               max_num_batched_tokens=job["prefill_budget"], max_model_len=job["max_model_len"],
                               gpu_memory_utilization=options["gpu_memory_utilization"],
                               enforce_eager=job["mode"] == "eager", use_prefill_cudagraph=job["mode"] == "piecewise_graph",
                               enable_prefix_cache=False)
        runner = engine.model_runner
        engine.max_concurrent_batches = options["queue_depth"]
        runner.async_output = not options["sync_output"]
        torch.cuda.synchronize()
        phase = "storage_inventory"
        report["init_ms"] = (time.perf_counter() - init_start) * 1000
        report["memory_after_init"] = memory_snapshot(torch)
        report["storage"] = storage_inventory(runner, torch)
        report["engine"] = dict(max_num_seqs=engine.config.max_num_seqs, max_model_len=engine.config.max_model_len,
                                max_num_batched_tokens=engine.config.max_num_batched_tokens,
                                kv_blocks=engine.config.num_kvcache_blocks, block_size=engine.config.kvcache_block_size,
                                decode_graph_sizes=runner.graph_bs, prefill_graph_sizes=runner.prefill_graph_sizes,
                                kv_cache_policy=options["kv_cache_policy"])
        expected = {"per_channel": Fp8LinearMethod, "blockwise": Fp8BlockLinearMethod}.get(job["variant"])
        methods = [module.quant_method for module in runner.model.modules() if hasattr(module, "quant_method")]
        if expected and not any(isinstance(method, expected) for method in methods):
            raise RuntimeError("Requested quantization backend not loaded")
        if engine.config.max_model_len < job["max_model_len"]:
            raise RuntimeError("Model context limit is shorter than requested workload")
        cases = job["cases"]
        block = engine.config.kvcache_block_size
        required = max(c["batch_size"] * math.ceil((c["input_len"] + c["output_len"]) / block) for c in cases)
        report["fixed_required_kv_blocks"] = required
        fixed_allowed = (not options["skip_fixed"]
                         and all(c["batch_size"] * c["input_len"] <= job["prefill_budget"] for c in cases)
                         and max(c["batch_size"] * c["input_len"] for c in cases) <= options["max_fixed_prefill_tokens"])
        if required > engine.config.num_kvcache_blocks:
            raise RuntimeError(f"Full matrix needs {required} KV blocks; have {engine.config.num_kvcache_blocks}")
        if not fixed_allowed:
            raise RuntimeError("The complete matrix requires fixed prefill: remove --skip-fixed and raise "
                               "--max-fixed-prefill-tokens to at least max(batch * input)")
        save_json(report_path, report)
        with torch.inference_mode():
            original_preempt = engine.scheduler.preempt

            def reject_preempt(seq):
                raise RuntimeError(f"Fixed-batch request {seq.seq_id} would be preempted")

            engine.scheduler.preempt = reject_preempt
            try:
                phase = "matrix"
                measure_matrix(engine, job, torch, report, lambda: save_json(report_path, report))
            finally:
                engine.scheduler.preempt = original_preempt
            # Optional diagnostics are AFTER every measured matrix case.
            if options["stress_seconds"] > 0:
                phase = "sustained"
                report["active_phase"] = phase
                report["telemetry_before_stress"] = telemetry()
                report["sustained"] = sustained_run(engine, job, torch)
                report["telemetry_after_stress"] = telemetry()
                save_json(report_path, report)
            if job["diagnostic"]:
                engine.scheduler.max_num_batched_tokens = job["prefill_budget"]
                if options["microbench"]:
                    phase = "microbench"
                    report["active_phase"] = phase
                    report["diagnostics"]["microbench"] = microbench(runner, job, torch)
                    save_json(report_path, report)
                if options["profile"]:
                    phase = "profiler"
                    report["active_phase"] = phase
                    report["diagnostics"]["profile"] = profile_workload(engine, job, torch, output_dir)
                    save_json(report_path, report)
            report["status"] = "ok"
            report["environment"]["after"] = telemetry()
    except Exception as error:
        kind = "oom" if "out of memory" in str(error).lower() else "capacity" if "VRAM budget" in str(error) else "runtime"
        report.update(status="failed", failure_phase=report.get("active_phase", phase), failure_kind=kind, error=str(error), traceback=traceback.format_exc())
        print(report["traceback"], file=sys.stderr, flush=True)
        if "torch" in locals() and torch.cuda.is_initialized():
            try:
                report["memory_at_failure"] = memory_snapshot(torch)
            except Exception:
                pass
    finally:
        save_json(report_path, report)
        if engine is not None:
            try:
                engine.exit()
            except Exception as error:
                report["cleanup_error"] = str(error)
                save_json(report_path, report)
    return 0 if report["status"] == "ok" else 1


FIXED_METRICS = ("e2e_ms", "prefill_ms", "decode_ms", "e2e_output_tokens_per_s", "e2e_total_tokens_per_s",
                 "prefill_input_tokens_per_s", "decode_tokens_per_s", "tpot_ms", "transient_peak_bytes",
                 "enqueue_ms", "ttft_ms")


def aggregate(reports):
    fixed, decode_only, stress, layers, activation, micros = [], [], [], [], [], []
    for report in reports:
        job = report["job"]
        common = {key: job[key] for key in ("variant", "mode", "batch_size", "input_len", "pass")}
        for row in report.get("fixed", []):
            fixed.append({**common, **row})
        for row in report.get("decode_only", []):
            decode_only.append({**common, **row})
        if "sustained" in report:
            stress.append({**common, **report["sustained"]})
        for row in report.get("storage", {}).get("modules", []):
            layers.append({**common, **row})
            if "k" not in row:
                continue
            scheme = {"Fp8LinearMethod": "per_channel", "Fp8BlockLinearMethod": "blockwise"}.get(row["method"], "unquantized")
            geometries = sorted({(c["batch_size"], c["input_len"]) for c in job.get("cases", [])}
                                or {(job["batch_size"], job["input_len"])})
            for bs, length in geometries:
                for stage, m in (("decode", bs), ("full_prefill", bs * length)):
                    activation.append({**common, "batch_size": bs, "input_len": length,
                                       "module": row["module"], "stage": stage, "m": m,
                                       "k": row["k"], "n": row["n"], "scheme": scheme,
                                       **activation_accounting(m, row["k"], row["compute_itemsize"], scheme)})
        for row in report.get("diagnostics", {}).get("microbench", []):
            micros.append({**common, **row})
    groups = {}
    for row in fixed:
        key = tuple(row[k] for k in ("variant", "mode", "batch_size", "input_len", "output_len"))
        groups.setdefault(key, []).append(row)
    summary = []
    for key, rows in groups.items():
        common = dict(zip(("variant", "mode", "batch_size", "input_len", "output_len"), key))
        summary.append({**common, "metrics": {metric: stats([row.get(metric) for row in rows]) for metric in FIXED_METRICS},
                        "decode_step_ms": stats([value for row in rows for value in row["decode_step_ms"]]),
                        "peak_allocated_bytes": max(row["memory_after"]["peak_allocated_bytes"] for row in rows)})
    decode_groups = {}
    for row in fixed + decode_only:
        key = tuple(row[k] for k in ("variant", "mode", "batch_size", "input_len", "output_len"))
        decode_groups.setdefault(key, []).append(row)
    decode_summary = []
    for key, rows in decode_groups.items():
        decode_summary.append(dict(zip(("variant", "mode", "batch_size", "input_len", "output_len"), key)) |
                              dict(metrics={metric: stats([r[metric] for r in rows]) for metric in
                                            ("decode_ms", "decode_tokens_per_s", "tpot_ms", "transient_peak_bytes")},
                                   decode_step_ms=stats([value for row in rows for value in row["decode_step_ms"]]),
                                   source="fixed_e2e" if "e2e_ms" in rows[0] else "isolated_after_chunked_prefill"))
    decode_index = {tuple(row[k] for k in ("variant", "mode", "batch_size", "input_len", "output_len", "pass", "round")): row
                    for row in fixed + decode_only}
    decode_comparisons = []
    for baseline, candidate in (("per_channel", "blockwise"), ("bf16", "per_channel"), ("bf16", "blockwise")):
        groups = {}
        for key, base in decode_index.items():
            other = decode_index.get((candidate, *key[1:]))
            if key[0] == baseline and other and ("e2e_ms" in base) == ("e2e_ms" in other):
                groups.setdefault(key[1:5], []).append(base["decode_ms"] / other["decode_ms"])
        for key, ratios in groups.items():
            decode_comparisons.append(dict(baseline=baseline, candidate=candidate,
                                           **dict(zip(("mode", "batch_size", "input_len", "output_len"), key)),
                                           metrics={"decode_speedup": stats(ratios)}))
    comparisons = []
    indexed = {tuple(row[k] for k in ("variant", "mode", "batch_size", "input_len", "output_len", "pass", "round")): row
               for row in fixed}
    for baseline, candidate in (("per_channel", "blockwise"), ("bf16", "per_channel"), ("bf16", "blockwise")):
        ratios = {}
        for key, base in indexed.items():
            if key[0] != baseline:
                continue
            other = indexed.get((candidate, *key[1:]))
            if not other:
                continue
            group = key[1:5]
            rows = ratios.setdefault(group, [])
            rows.append(dict(e2e_speedup=base["e2e_ms"] / other["e2e_ms"],
                             decode_speedup=base["decode_ms"] / other["decode_ms"],
                             prefill_speedup=base["prefill_ms"] / other["prefill_ms"],
                             e2e_time_saved_pct=100 * (1 - other["e2e_ms"] / base["e2e_ms"]),
                             decode_time_saved_pct=100 * (1 - other["decode_ms"] / base["decode_ms"]),
                             peak_allocated_saved_bytes=base["memory_after"]["peak_allocated_bytes"] - other["memory_after"]["peak_allocated_bytes"],
                             baseline_ms=base["e2e_ms"], candidate_ms=other["e2e_ms"], **{k: base[k] for k in ("pass", "round")}))
        for key, rows in ratios.items():
            comparisons.append(dict(baseline=baseline, candidate=candidate,
                                    **dict(zip(("mode", "batch_size", "input_len", "output_len"), key)),
                                    paired_n=len(rows), e2e_win_fraction=sum(r["e2e_speedup"] > 1 for r in rows) / len(rows),
                                    metrics={k: stats([r[k] for r in rows]) for k in
                                             ("e2e_speedup", "decode_speedup", "prefill_speedup", "e2e_time_saved_pct",
                                              "decode_time_saved_pct", "peak_allocated_saved_bytes")},
                                    paired_samples=rows))
    layer_index = {(r["variant"], r["mode"], r["batch_size"], r["input_len"], r["pass"], r["module"]): r for r in layers}
    decoder_groups = {}
    for row in layers:
        parts = row["module"].split(".")
        if len(parts) < 3 or parts[:2] != ["model", "layers"] or not parts[2].isdigit():
            continue
        key = tuple(row[k] for k in ("variant", "mode", "batch_size", "input_len", "pass")) + (int(parts[2]),)
        group = decoder_groups.setdefault(key, dict(zip(("variant", "mode", "batch_size", "input_len", "pass", "layer"), key)))
        for field in ("physical_bytes", "weight_bytes", "scale_bytes", "buffer_bytes", "equivalent_bf16_bytes", "saved_vs_equivalent_bf16_bytes"):
            group[field] = group.get(field, 0) + row[field]
        if row.get("weight_dtype") == "torch.float8_e4m3fn":
            group["quantized_projections"] = group.get("quantized_projections", 0) + 1
    layer_comparisons = []
    for key, row in layer_index.items():
        if key[0] != "blockwise":
            continue
        base = layer_index.get(("per_channel", *key[1:]))
        bf = layer_index.get(("bf16", *key[1:]))
        if not base:
            continue
        layer_comparisons.append({k: row[k] for k in ("mode", "batch_size", "input_len", "pass", "module")} |
                                 dict(per_channel_bytes=base["physical_bytes"], blockwise_bytes=row["physical_bytes"],
                                      blockwise_saved_vs_per_channel_bytes=base["physical_bytes"] - row["physical_bytes"],
                                      per_channel_scale_bytes=base["scale_bytes"], blockwise_scale_bytes=row["scale_bytes"],
                                      bf16_actual_bytes=bf["physical_bytes"] if bf else None,
                                      blockwise_saved_vs_actual_bf16_bytes=bf["physical_bytes"] - row["physical_bytes"] if bf else None,
                                      blockwise_saved_vs_equivalent_bf16_bytes=row["saved_vs_equivalent_bf16_bytes"],
                                      blockwise_saved_vs_equivalent_bf16_pct=100 * row["saved_vs_equivalent_bf16_bytes"] / row["equivalent_bf16_bytes"] if row["equivalent_bf16_bytes"] else 0,
                                      per_channel_saved_vs_equivalent_bf16_pct=100 * base["saved_vs_equivalent_bf16_bytes"] / base["equivalent_bf16_bytes"] if base["equivalent_bf16_bytes"] else 0))
    memory_comparisons = []
    report_index = {tuple(r["job"][k] for k in ("variant", "mode", "batch_size", "input_len", "pass")): r
                    for r in reports if "storage" in r}
    for baseline, candidate in (("per_channel", "blockwise"), ("bf16", "per_channel"), ("bf16", "blockwise")):
        for key, base in report_index.items():
            if key[0] != baseline:
                continue
            other = report_index.get((candidate, *key[1:]))
            if not other:
                continue
            bt, ct = base["storage"]["totals"], other["storage"]["totals"]
            saved = bt["physical_bytes"] - ct["physical_bytes"]
            memory_comparisons.append(dict(baseline=baseline, candidate=candidate,
                                          **dict(zip(("mode", "batch_size", "input_len", "pass"), key[1:])),
                                          model_saved_bytes=saved, model_saved_pct=100 * saved / bt["physical_bytes"],
                                          scale_saved_bytes=bt["scale_bytes"] - ct["scale_bytes"],
                                          kv_saved_bytes=bt["kv_cache_bytes"] - ct["kv_cache_bytes"],
                                          gdn_state_saved_bytes=bt["gdn_state_bytes"] - ct["gdn_state_bytes"],
                                          allocated_after_init_saved_bytes=base["memory_after_init"]["allocated_bytes"] - other["memory_after_init"]["allocated_bytes"],
                                          kv_blocks_baseline=bt["kv_blocks"], kv_blocks_candidate=ct["kv_blocks"]))
    warnings = []
    anchors = [r["storage"]["unquantized_weight_probes"] for r in reports if "storage" in r]
    if anchors and any(a != anchors[0] for a in anchors):
        warnings.append("Unquantized embedding/LM-head probes differ. Check that all checkpoints derive from the same original weights.")
    coverage = [{row["module"] for row in r["storage"]["modules"]
                 if row.get("weight_dtype") == "torch.float8_e4m3fn"}
                for r in reports if "storage" in r and r["job"]["variant"] != "bf16"]
    if coverage and any(c != coverage[0] for c in coverage):
        warnings.append("FP8 module coverage differs between variants; inspect layers.csv before attributing differences to the scale format.")
    for row in summary:
        if row["metrics"]["e2e_ms"]["cv"] > 0.10:
            warnings.append(f"High E2E variation (>10% CV): {row['variant']} {row['mode']} bs{row['batch_size']} in{row['input_len']} out{row['output_len']}")
    sustained_groups = {}
    for row in stress:
        key = tuple(row[k] for k in ("variant", "mode", "batch_size", "input_len"))
        sustained_groups.setdefault(key, []).append(row)
    sustained_summary = []
    for key, rows in sustained_groups.items():
        sustained_summary.append(dict(zip(("variant", "mode", "batch_size", "input_len"), key)) |
                                 dict(metrics={metric: stats([row[metric] for row in rows]) for metric in
                                               ("output_tokens_per_s", "window_output_tokens_per_s", "requests_per_s", "preemptions")},
                                      pooled_ttft_ms=stats([v for r in rows for v in r["ttft_samples_ms"]]),
                                      pooled_itl_ms=stats([v for r in rows for v in r["itl_samples_ms"]]),
                                      pooled_request_e2e_ms=stats([v for r in rows for v in r["request_e2e_samples_ms"]])))
    sustained_index = {tuple(row[k] for k in ("variant", "mode", "batch_size", "input_len", "pass")): row for row in stress}
    sustained_comparisons = []
    for baseline, candidate in (("per_channel", "blockwise"), ("bf16", "per_channel"), ("bf16", "blockwise")):
        groups = {}
        for key, base in sustained_index.items():
            other = sustained_index.get((candidate, *key[1:]))
            if key[0] == baseline and other:
                groups.setdefault(key[1:4], []).append(dict(
                    output_throughput_speedup=other["output_tokens_per_s"] / base["output_tokens_per_s"],
                    window_throughput_speedup=other["window_output_tokens_per_s"] / base["window_output_tokens_per_s"],
                    ttft_p95_ratio=other["ttft_ms"]["p95"] / max(base["ttft_ms"]["p95"], 1e-12)))
        for key, rows in groups.items():
            sustained_comparisons.append(dict(baseline=baseline, candidate=candidate,
                                              **dict(zip(("mode", "batch_size", "input_len"), key)),
                                              metrics={k: stats([r[k] for r in rows]) for k in rows[0]}))
    pressure_envelope = []
    for variant, mode in {(r["variant"], r["mode"]) for r in sustained_summary}:
        for input_len in {r["input_len"] for r in sustained_summary if r["variant"] == variant and r["mode"] == mode}:
            candidates = [r for r in sustained_summary if (r["variant"], r["mode"], r["input_len"]) == (variant, mode, input_len)]
            best = max(candidates, key=lambda r: r["metrics"]["output_tokens_per_s"]["median"])
            pressure_envelope.append(dict(variant=variant, mode=mode, input_len=input_len,
                                          best_tested_batch=best["batch_size"],
                                          best_output_tokens_per_s=best["metrics"]["output_tokens_per_s"]["median"],
                                          largest_completed_tested_batch=max(r["batch_size"] for r in candidates),
                                          note="Completed samples only; largest tested batch is not a measured absolute capacity limit"))
    return dict(fixed_raw=fixed, fixed_summary=summary, comparisons=comparisons, sustained=stress,
                decode_only_raw=decode_only, decode_summary=decode_summary, decode_comparisons=decode_comparisons,
                layers=layers, layer_comparisons=layer_comparisons, activation_representation=activation,
                decoder_layers=list(decoder_groups.values()), microbench=micros,
                memory_comparisons=memory_comparisons, sustained_summary=sustained_summary,
                sustained_comparisons=sustained_comparisons, pressure_envelope=pressure_envelope, warnings=warnings)


def flatten_metrics(rows):
    result = []
    for row in rows:
        row = dict(row)
        metrics = {**row.pop("metrics", {}), **row.pop("timings", {})}
        for metric, values in metrics.items():
            for name, value in values.items():
                row[f"{metric}_{name}"] = value
        result.append(row)
    return result


METRIC_DEFINITIONS = {
    "prefill_ms": "Synchronized wall time of one full-batch engine.step(), producing token #1; excludes enqueue.",
    "prefill_input_tokens_per_s": "batch_size * input_len / (prefill_ms / 1000); aggregate across batch.",
    "ttft_ms": "Time from starting scheduler enqueue to all first tokens being ready = enqueue_ms + prefill_ms.",
    "decode_ms": "Synchronized wall time for output_len - 1 decode steps after the first token.",
    "decode_tokens_per_s": "batch_size * (output_len - 1) / (decode_ms / 1000); aggregate across batch.",
    "tpot_ms": "decode_ms / (output_len - 1); mean synchronized wall time per decode step, not per-request ITL.",
    "e2e_ms": "Time from starting scheduler enqueue to all requested output tokens being ready; includes enqueue + prefill + decode.",
    "e2e_output_tokens_per_s": "batch_size * output_len / (e2e_ms / 1000); primary E2E throughput.",
    "e2e_total_tokens_per_s": "batch_size * (input_len + output_len) / (e2e_ms / 1000); includes prompt tokens, not decode throughput.",
    "peak_allocated_mib": "PyTorch allocation peak during this sample; all cases share the engine's maximum-sized pools.",
    "first_run_of_prefill_geometry": "Geometry was not benchmark-warmed in this process; possible first-use compilation is retained, not proven.",
    "speedup": "baseline latency / candidate latency; >1 means candidate faster; only matched pass/round/geometry are paired.",
}


def compact_summary(plan, result):
    dimensions = ("variant", "mode", "batch_size", "input_len", "output_len")
    index = {tuple(r[k] for k in dimensions): r for r in result["fixed_summary"]}
    samples = {}
    for row in result["fixed_raw"]:
        samples.setdefault(tuple(row[k] for k in dimensions), []).append(row)
    expected = set()
    target_n = {}
    for job in plan["jobs"]:
        cases = job.get("cases") or workload_cases([job["batch_size"]], [job["input_len"]], plan["options"]["output_lens"])
        for case in cases:
            key = (job["variant"], job["mode"], case["batch_size"], case["input_len"], case["output_len"])
            expected.add(key)
            target_n[key] = target_n.get(key, 0) + job["options"]["rounds"]
    rows = []
    for key in sorted(expected):
        summary = index.get(key)
        raw = samples.get(key, [])
        row = dict(zip(dimensions, key))
        row.update(n=len(raw), expected_n=target_n[key],
                   status="complete" if len(raw) == target_n[key] else "partial" if raw else "missing")
        for metric in FIXED_METRICS:
            values = summary["metrics"].get(metric, {}) if summary else {}
            if metric == "transient_peak_bytes":
                row["transient_peak_mib"] = values.get("median", 0) / MIB if values.get("n") else None
            else:
                row[metric] = values.get("median")
        row.update(peak_allocated_mib=summary["peak_allocated_bytes"] / MIB if summary else None,
                   prefill_share_pct=statistics.median(100 * r["prefill_ms"] / r["e2e_ms"] for r in raw) if raw else None,
                   decode_share_pct=statistics.median(100 * r["decode_ms"] / r["e2e_ms"] for r in raw) if raw else None,
                   first_geometry_samples=sum(bool(r.get("first_run_of_prefill_geometry")) for r in raw),
                   measurement_orders=[dict(pass_id=r["pass"], round_id=r["round"], order=r.get("measurement_order")) for r in raw],
                   decode_context_start=key[3] + 1, decode_context_end=key[3] + key[4] - 1,
                   e2e_cv=summary["metrics"]["e2e_ms"]["cv"] if summary else None)
        rows.append(row)
    return rows


def trend_deltas(rows):
    """Adjacent sizes only; hold the other two workload dimensions constant."""
    records = []
    metrics = ("prefill_ms", "prefill_input_tokens_per_s", "decode_ms", "decode_tokens_per_s", "tpot_ms",
               "e2e_ms", "e2e_output_tokens_per_s")
    for axis in ("batch_size", "input_len", "output_len"):
        held = [k for k in ("variant", "mode", "batch_size", "input_len", "output_len") if k != axis]
        groups = {}
        for row in rows:
            if row["n"]:
                groups.setdefault(tuple(row[k] for k in held), []).append(row)
        for key, group in sorted(groups.items()):
            group.sort(key=lambda r: r[axis])
            for before, after in zip(group, group[1:]):
                record = dict(zip(held, key))
                record.update(axis=axis, from_size=before[axis], to_size=after[axis],
                              workload_size_ratio=after[axis] / before[axis],
                              from_status=before["status"], to_status=after["status"])
                for metric in metrics:
                    record[metric + "_ratio"] = after[metric] / before[metric] if before[metric] else None
                records.append(record)
    return records


def svg_trends(rows, mode):
    """Dependency-free, static SVG: batch rows x prefill/decode/E2E columns."""
    data = [r for r in rows if r["mode"] == mode]
    batches = sorted({r["batch_size"] for r in data})
    inputs = sorted({r["input_len"] for r in data})
    variants = sorted({r["variant"] for r in data})
    outputs = sorted({r["output_len"] for r in data})
    colours = {name: colour for name, colour in zip(variants, ("#2563eb", "#dc2626", "#059669"))}
    panels = (("Prefill latency (ms) / lower better", "prefill_ms"),
              ("Decode tok/s / higher better", "decode_tokens_per_s"),
              ("E2E output tok/s / higher better", "e2e_output_tokens_per_s"))
    width, row_height = 1200, 285
    height = 125 + row_height * len(batches) + 55
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#334155;font-size:12px}.title{font-size:19px;font-weight:bold}.panel{font-size:13px;font-weight:bold}</style>']
    def label(x, y, value, cls="", anchor="start"):
        parts.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{html.escape(str(value))}</text>')
    label(32, 30, f"FP8 matrix trends | {mode}", "title")
    label(32, 52, "One global warmup per engine; all measured samples retained. Throughput is aggregate across batch.")
    legend = [(variant, out) for variant in variants for out in outputs]
    for i, (variant, out) in enumerate(legend):
        x, y = 32 + (i % 4) * 290, 78 + (i // 4) * 22
        dash = ' stroke-dasharray="7 4"' if outputs.index(out) % 2 else ''
        parts.append(f'<line x1="{x}" x2="{x+30}" y1="{y-4}" y2="{y-4}" stroke="{colours[variant]}" stroke-width="2.5"{dash}/>')
        label(x + 37, y, f"{variant}, output={out}")
    panel_top = 100 + 22 * math.ceil(len(legend) / 4)
    height = panel_top + row_height * len(batches) + 55
    parts[0] = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    for bi, batch in enumerate(batches):
        for pi, (title, metric) in enumerate(panels):
            x0, y0, pw, ph = 65 + pi * 395, panel_top + bi * row_height + 34, 302, 188
            label(x0, y0 - 16, f"BS={batch} | {title}", "panel")
            panel_rows = [r for r in data if r["batch_size"] == batch and r[metric] is not None]
            max_y = max((r[metric] for r in panel_rows), default=1) * 1.15 or 1
            def xcoord(length):
                return x0 + (length - inputs[0]) / (inputs[-1] - inputs[0]) * pw if len(inputs) > 1 else x0 + pw / 2
            def ycoord(value):
                return y0 + ph - value / max_y * ph
            for tick in range(5):
                value = max_y * tick / 4
                y = ycoord(value)
                parts.append(f'<line x1="{x0}" x2="{x0+pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>')
                label(x0 - 8, y + 4, f"{value:.1f}" if max_y < 100 else f"{value:.0f}", anchor="end")
            parts.append(f'<path d="M{x0} {y0} V{y0+ph} H{x0+pw}" stroke="#94a3b8" fill="none"/>')
            for length in inputs:
                label(xcoord(length), y0 + ph + 20, length, anchor="middle")
            label(x0 + pw / 2, y0 + ph + 40, "Input tokens / request", anchor="middle")
            for variant, out in legend:
                selected = [r for r in data if (r["variant"], r["batch_size"], r["output_len"]) == (variant, batch, out)]
                selected.sort(key=lambda r: r["input_len"])
                segment = []
                # Missing measurements split lines rather than silently bridging gaps.
                for r in selected + [None]:
                    if r is None or r[metric] is None:
                        if len(segment) > 1:
                            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in segment)
                            dash = ' stroke-dasharray="7 4"' if outputs.index(out) % 2 else ''
                            parts.append(f'<polyline points="{points}" fill="none" stroke="{colours[variant]}" stroke-width="2"{dash}/>')
                        segment = []
                        continue
                    x, y = xcoord(r["input_len"]), ycoord(r[metric])
                    segment.append((x, y))
                    tip = html.escape(f"{variant}; BS={batch}; input={r['input_len']}; output={out}; {metric}={r[metric]:.3f}; n={r['n']}; {r['status']}")
                    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colours[variant]}"><title>{tip}</title></circle>')
            if not panel_rows:
                label(x0 + pw / 2, y0 + ph / 2, "No measured data", anchor="middle")
    label(32, height - 29, "Solid/dashed lines separate output lengths. SVG point tooltips show exact values. Missing cases stay empty.")
    label(32, height - 10, "Output length also changes average decode context. Prefill should be compared at the same BS/input; n=1 is a single sample.")
    parts.append('</svg>')
    return "\n".join(parts)


def write_outputs(output, plan, reports):
    output.mkdir(parents=True, exist_ok=True)
    result = aggregate(reports)
    save_json(output / "results.json", dict(plan=plan, reports=reports, aggregate=result))
    for key in ("fixed_raw", "fixed_summary", "comparisons", "sustained", "layers", "layer_comparisons",
                "activation_representation", "microbench", "memory_comparisons", "decoder_layers",
                "decode_only_raw", "decode_summary", "decode_comparisons", "sustained_summary",
                "sustained_comparisons", "pressure_envelope"):
        save_csv(output / f"{key}.csv", flatten_metrics(result[key]))
    rows = compact_summary(plan, result)
    trends = trend_deltas(rows)
    paired = []
    for item in result["comparisons"]:
        row = {k: item[k] for k in ("baseline", "candidate", "mode", "batch_size", "input_len", "output_len", "paired_n")}
        row.update({k: v["median"] for k, v in item["metrics"].items()})
        paired.append(row)
    coverage = dict(expected_cases=len(rows), complete_cases=sum(r["status"] == "complete" for r in rows),
                    partial_cases=sum(r["status"] == "partial" for r in rows), missing_cases=sum(not r["n"] for r in rows),
                    measured_samples=sum(r["n"] for r in rows), expected_samples=sum(r["expected_n"] for r in rows))
    notes = ["Each row fixes model, mode, batch, input AND output length. No averaging across workload geometries.",
             "One largest-geometry benchmark warmup per engine; initialization may do its own graph capture/warmup and is separate.",
             "All measured samples are retained, including first-use compile costs in later geometries. No automatic outlier removal.",
             "Default n=1 provides a direct measurement, not a distribution or a statistically established speedup.",
             "Decode excludes token #1. Longer output raises average attended context; its throughput difference is not a pure length-independent effect.",
             "Every sample synchronizes before timing, at the prefill/decode boundary, and after decode; not after every decode step.",
             "Shared engine capacity: max BS/input/context KV and GDN pools stay allocated for every case.",
             "Per-channel and blockwise FP8 use different kernels; results compare these implementations as a whole.",
             "This is in-process inference E2E, without tokenization, model loading, networking or client rendering.",
             "Cases are measured in saved order; model order alternates only if additional --passes are explicitly requested."]
    if plan.get("schema_version", 1) < 2:
        notes = ["LEGACY PLAN: old per-geometry workers/timing/warmup policy retained. New timing definitions do not retroactively apply."] + notes
    jobs = [dict(variant=r["job"]["variant"], mode=r["job"]["mode"], pass_id=r["job"]["pass"],
                 status=r["status"], init_ms=r.get("init_ms"), warmup=r.get("warmup"),
                 failure_phase=r.get("failure_phase"), active_case=r.get("active_case"), error=r.get("error"),
                 memory_after_init=r.get("memory_after_init"), environment=r.get("environment")) for r in reports]
    save_csv(output / "summary.csv", rows)
    save_csv(output / "speedups.csv", paired)
    save_csv(output / "trends.csv", trends)
    save_json(output / "ai_summary.json", dict(schema_version=2, benchmark="FP8 Cartesian workload matrix",
              source=plan["git"], timing_protocol=plan.get("timing_protocol"), options=plan["options"],
              models=plan.get("models", {}), coverage=coverage, metric_definitions=METRIC_DEFINITIONS,
              interpretation_notes=notes, warnings=result["warnings"], jobs=jobs,
              summary=rows, paired_speedups=paired, trend_deltas=trends,
              samples=[{k: v for k, v in row.items() if k not in ("decode_step_ms", "decode_step_distribution")}
                       for row in result["fixed_raw"]]))
    modes = sorted({r["mode"] for r in rows})
    charts = []
    for mode in modes:
        name = "trends.svg" if len(modes) == 1 else f"trends_{mode}.svg"
        svg = svg_trends(rows, mode)
        (output / name).write_text(svg, encoding="utf-8")
        charts.append((name, svg))
    if len(modes) > 1:
        (output / "trends.svg").write_text(charts[0][1], encoding="utf-8")
    def num(value, decimals=2):
        return "—" if value is None else f"{value:.{decimals}f}"
    lines = ["# FP8完整组合测试", "",
             f"完成{coverage['complete_cases']}/{coverage['expected_cases']}个模型×工作负载组合；"
             f"保留{coverage['measured_samples']}/{coverage['expected_samples']}次实测。", "",
             "模型每次只建一个engine、进行一次全局warmup；默认12组各直接测一次。", "",
             "prefill包含首token，decode统计剩余output−1步，E2E包含入队、prefill和decode。"
             "吞吐均为整个batch的总吞吐；下表为各组样本中位数，n=1时就是该次实测。", "",
             "## 趋势图", ""]
    lines.extend(f"![{name}]({name})" for name, _ in charts)
    headers = ["模型", "模式", "BS", "输入", "输出", "n", "状态", "Prefill ms", "Prefill tok/s",
               "Decode ms", "Decode tok/s", "TPOT ms", "E2E ms", "E2E output tok/s", "Decode占比%"]
    table_rows = [[r["variant"], r["mode"], r["batch_size"], r["input_len"], r["output_len"], r["n"], r["status"],
                   num(r["prefill_ms"]), num(r["prefill_input_tokens_per_s"], 1), num(r["decode_ms"]),
                   num(r["decode_tokens_per_s"], 1), num(r["tpot_ms"]), num(r["e2e_ms"]),
                   num(r["e2e_output_tokens_per_s"], 1), num(r["decode_share_pct"], 1)] for r in rows]
    lines.extend(["", "## 按组合分列的结果", "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)])
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in table_rows)
    lines.extend(["", "## 同组合模型对比", "", "倍率=基准耗时/候选耗时，>1表示候选更快。", "",
                  "| 基准→候选 | 模式 | BS | 输入 | 输出 | 配对数 | Prefill倍率 | Decode倍率 | E2E倍率 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in paired:
        lines.append(f"| {r['baseline']}→{r['candidate']} | {r['mode']} | {r['batch_size']} | {r['input_len']} | {r['output_len']} | "
                     f"{r['paired_n']} | {r['prefill_speedup']:.3f} | {r['decode_speedup']:.3f} | {r['e2e_speedup']:.3f} |")
    lines.extend(["", "## 读数说明", "",
                  "- 按固定batch和输出长度观察输入长度变化：prefill看ms与输入tok/s；decode看tok/s与TPOT；E2E看ms与输出tok/s。",
                  "- 改变输出长度也会改变decode期间的平均上下文长度，因此单独保留512/1024两条线。",
                  "- 后续新shape首次运行可能包含编译开销，不丢弃；first_geometry_samples和测量顺序可用于辨认。",
                  "- decode_step_ms是未逐步同步的主机step耗时，不是GPU耗时或真实ITL；趋势以同步总decode耗时为准。",
                  "- 默认每组n=1，不能据此判断稳定性或显著性。--rounds可增加完整矩阵实测轮次，不会再次warmup。",
                  "- 每组共享最大容量的KV/GDN池，显存读数表示该engine配置下的占用。", "",
                  "## 交给AI分析的文件", "",
                  "首选ai_summary.json：含口径、环境、覆盖情况、所有样本、组合中位数、配对倍率和逐维度变化率。",
                  "summary.csv便于Excel查看；results.json含完整原始记录；fixed_raw.csv保留每次实测。", "",
                  "## 失败及缺失", ""])
    for job in jobs:
        if job["status"] != "ok":
            lines.append(f"- {job['variant']} {job['mode']} pass{job['pass_id']}: {job['failure_phase']} {job['error']}")
    if not any(job["status"] != "ok" for job in jobs) and not coverage["missing_cases"] and not coverage["partial_cases"]:
        lines.append("无。")
    else:
        lines.append(f"缺失{coverage['missing_cases']}组，部分完成{coverage['partial_cases']}组；不将缺失当作0。")
    lines.extend(["", "## 提示", "", *["- " + warning for warning in result["warnings"]]])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in table_rows)
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>FP8组合测试报告</title>'
                '<style>body{font:14px system-ui;margin:28px;color:#1e293b}svg{width:100%;height:auto;max-width:1200px}'
                'table{border-collapse:collapse;font-size:12px}th,td{padding:7px;border:1px solid #cbd5e1;text-align:right}'
                'th{background:#f1f5f9;position:sticky;top:0}a{color:#2563eb}.table{overflow:auto}</style>'
                '<h1>FP8完整组合测试</h1>'
                f'<p>完成{coverage["complete_cases"]}/{coverage["expected_cases"]}组；每个engine全局warmup一次，保留全部实测。</p>'
                '<p>prefill包含首token，decode排除首token；E2E包含入队、prefill和decode。'
                '每组n=1时为单次结果。新shape的首次编译可能计入实测。</p>'
                '<p><a href="ai_summary.json">AI数据JSON</a> · <a href="summary.csv">结果CSV</a> · '
                '<a href="speedups.csv">模型倍率</a> · <a href="report.md">完整说明</a></p>'
                + ''.join(svg for _, svg in charts) + '<h2>各组合实测</h2><div class="table"><table><thead><tr>'
                + th + '</tr></thead><tbody>' + tr + '</tbody></table></div></html>')
    (output / "report.html").write_text(document, encoding="utf-8")
    return result


def execute_plan(output, plan, retry_failed=False, plan_only=False):
    jobs, options = plan["jobs"], plan["options"]
    print(f"Plan: {len(jobs)} engine processes; {len(plan.get('cases', []))} cases per model; "
          f"{options['warmup_rounds']} global warmup; {options['rounds']} measured matrix sweeps; "
          f"{options['stress_seconds']}s sustained load per process.\nOutput: {output}", flush=True)
    if plan_only:
        return 0
    reports = []
    for i, job in enumerate(jobs):
        job_dir = output / job["name"]
        report_path = job_dir / "result.json"
        old_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
        if old_report and (old_report["status"] == "ok" or (old_report["status"] == "failed" and not retry_failed)):
            reports.append(old_report)
            print(f"[{i + 1}/{len(jobs)}] saved {old_report['status']}: {job['name']}", flush=True)
            continue
        print(f"[{i + 1}/{len(jobs)}] {job['name']}", flush=True)
        job_dir.mkdir(parents=True, exist_ok=True)
        job_file = job_dir / "job.json"
        save_json(job_file, job)
        before = telemetry()
        interrupted = False
        with (job_dir / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker-job", str(job_file)],
                                       stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            deadline = time.perf_counter() + options["job_timeout_seconds"]
            try:
                while True:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        process.kill()
                        process.wait()
                        returncode = -1
                        break
                    try:
                        returncode = process.wait(timeout=min(30, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        print(f"  running: {job['name']}; log={job_dir / 'worker.log'}", flush=True)
            except KeyboardInterrupt:
                interrupted = True
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                returncode = process.returncode
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else dict(job=job, status="starting", fixed=[])
        report.update(worker_returncode=returncode, parent_telemetry_before=before)
        if interrupted:
            report.update(status="interrupted", error="Interrupted by user; resume reruns this job")
        elif returncode and report["status"] in ("ok", "starting"):
            report.update(status="failed", error="Worker timed out or exited unexpectedly; inspect worker.log")
        save_json(report_path, report)
        reports.append(report)
        write_outputs(output, plan, reports)
        print(f"  {report['status']}; fixed rounds={len(report.get('fixed', []))}; log={job_dir / 'worker.log'}", flush=True)
        if interrupted:
            return 130
        if options["cooldown_seconds"]:
            time.sleep(options["cooldown_seconds"])
    write_outputs(output, plan, reports)
    failed = sum(r["status"] != "ok" for r in reports)
    print(f"Report: {output / 'report.md'}\nAI data: {output / 'ai_summary.json'}\nCharts: {output / 'trends.svg'}\nFailed jobs: {failed}/{len(reports)}", flush=True)
    return 1 if failed else 0


def build_jobs(options, variants, output):
    cases = workload_cases(options["batch_sizes"], options["input_lens"], options["output_lens"])
    max_bs = max(options["batch_sizes"])
    max_input = max(options["input_lens"])
    max_output = max(options["output_lens"])
    budget = max_bs * max_input
    if options["stress_seconds"]:
        budget = max(budget, options["stress_prefill_budget"])
    max_length = max_input + max(max_output, options["profile_decode_steps"] + 1 if options["profile"] else 0)
    jobs = []
    for mode in options["modes"]:
        for pass_id in range(options["passes"]):
            order = list(variants)
            if pass_id % 2:
                order.reverse()
            for variant in order:
                name = f"{mode}_matrix_pass{pass_id}_{variant}"
                jobs.append(dict(name=name, variant=variant, model=variants[variant], mode=mode,
                                 batch_size=max_bs, input_len=max_input, **{"pass": pass_id},
                                 cases=cases, prefill_budget=budget, max_model_len=max_length,
                                 diagnostic=pass_id == 0, options=options,
                                 result=str(output / name / "result.json")))
    return jobs


def run_parent(args):
    variants = {"per_channel": args.per_channel_model, "blockwise": args.blockwise_model}
    if args.bf16_model:
        variants["bf16"] = args.bf16_model
    specs = {name: model_spec(path, name) for name, path in variants.items()}
    if len({s["architecture_sha256"] for s in specs.values()}) != 1:
        raise ValueError("Checkpoint text configs differ. Convert both FP8 formats from the SAME original checkpoint.")
    if len({json.dumps(s["tokenizer_sha256"], sort_keys=True) for s in specs.values()}) != 1:
        raise ValueError("Tokenizer files differ between checkpoints")
    options = vars(args).copy()
    output = Path(args.output_dir).resolve() / (time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    output.mkdir(parents=True, exist_ok=False)
    jobs = build_jobs(options, variants, output)
    plan = dict(schema_version=2, options=options, git=git_info(), models=specs,
                cases=workload_cases(args.batch_sizes, args.input_lens, args.output_lens),
                jobs=jobs, output=str(output),
                timing_protocol="One fresh process per variant/mode/pass; shared max-capacity engine for all cases; "
                                "one benchmark warmup at largest geometry; measured full matrix sweeps; "
                                "identical seeds and case order across variants; no per-case warmup; "
                                "subsequent first-use compilation is retained")
    save_json(output / "plan.json", plan)
    return execute_plan(output, plan, plan_only=args.plan_only)


def resume_plan(args):
    output = Path(args.resume).resolve()
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    current = git_info()
    if any(current[key] != plan["git"].get(key) for key in ("commit", "source_sha256", "benchmark_sha256")):
        raise ValueError("Source or benchmark changed since this plan; start a new run to keep results comparable")
    for variant, spec in plan["models"].items():
        if model_spec(spec["path"], variant) != spec:
            raise ValueError(f"Checkpoint metadata changed since the saved plan: {variant}")
    if str(output) != plan["output"]:
        raise ValueError("Resume in the original output directory; job artifacts use absolute paths")
    return execute_plan(output, plan, retry_failed=args.retry_failed, plan_only=args.plan_only)


def regenerate_report(directory):
    output = Path(directory).resolve()
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if plan.get("schema_version") != 2:
        raise ValueError("This report format requires a new matrix plan (schema_version=2); legacy timings use different definitions")
    reports = []
    for job in plan["jobs"]:
        path = output / job["name"] / "result.json"
        if path.is_file():
            reports.append(json.loads(path.read_text(encoding="utf-8")))
    if not reports:
        raise ValueError("No saved worker result.json files found")
    write_outputs(output, plan, reports)
    print(f"AI data: {output / 'ai_summary.json'}\nCSV: {output / 'summary.csv'}\nCharts: {output / 'trends.svg'}")
    return 0


def self_test():
    import io
    import tempfile
    import xml.etree.ElementTree as ET
    from contextlib import redirect_stdout
    from types import SimpleNamespace
    from unittest.mock import patch

    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert stats([]) == {"n": 0}
    pc = activation_accounting(4, 2048, 2, "per_channel")
    block = activation_accounting(4, 2048, 2, "blockwise")
    assert pc["scale_bytes"] == 16 and block["scale_bytes"] == 256
    assert pc["representation_saved_bytes"] == 8176
    assert block["input_plus_quantized_bytes"] > block["source_bytes"]
    assert kernel_category("_block_mm") == "fp8_gemm_including_scales"
    assert kernel_category("kernel_cutlass_gdn_decode_bf16state") == "gdn"
    cases = workload_cases([4, 8], [256, 512, 1024], [512, 1024])
    assert len(cases) == 12 and len({tuple(c.values()) for c in cases}) == 12
    options = dict(batch_sizes=[4, 8], input_lens=[256, 512, 1024], output_lens=[512, 1024],
                   modes=["decode_graph"], passes=1, rounds=1, warmup_rounds=1,
                   stress_seconds=0, stress_prefill_budget=2048, profile=False,
                   profile_decode_steps=8, seed=42, kv_cache_policy="workload")
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        jobs = build_jobs(options, {"per_channel": "pc", "blockwise": "block"}, output)
        assert len(jobs) == 2  # NOT one engine per geometry/output.
        assert all(j["batch_size"] == 8 and j["prefill_budget"] == 8192 and j["max_model_len"] == 2048 for j in jobs)
        reports = []
        for job in jobs:
            calls = []
            def fake_run(engine, current, output_len, seed, torch):
                calls.append((current["batch_size"], current["input_len"], output_len, seed))
                bs, length = current["batch_size"], current["input_len"]
                scale = 1 if current["variant"] == "per_channel" else 0.5
                prefill, decode, enqueue = length * bs * 0.1 * scale, (output_len - 1) * 2 * scale, 0.1 * scale
                total = prefill + decode + enqueue
                return dict(output_len=output_len, seed=seed, e2e_ms=total, enqueue_ms=enqueue,
                            prefill_ms=prefill, decode_ms=decode, ttft_ms=enqueue + prefill,
                            input_tokens=bs * length, output_tokens=bs * output_len,
                            decode_tokens=bs * (output_len - 1), decode_steps=output_len - 1,
                            e2e_output_tokens_per_s=bs * output_len * 1000 / total,
                            e2e_total_tokens_per_s=bs * (length + output_len) * 1000 / total,
                            prefill_input_tokens_per_s=bs * length * 1000 / prefill,
                            decode_tokens_per_s=bs * (output_len - 1) * 1000 / decode,
                            tpot_ms=decode / (output_len - 1), transient_peak_bytes=100,
                            memory_after=dict(peak_allocated_bytes=1000), decode_step_ms=[1, 1])
            report = dict(job=job, fixed=[], status="ok")
            with redirect_stdout(io.StringIO()):
                measure_matrix(object(), job, None, report, lambda: None, run=fake_run)
            assert len(calls) == 13 and calls[0][:3] == (8, 1024, 1024)
            assert report["warmup"]["completed_runs"] == 1
            assert len(report["fixed"]) == 12
            assert [r["measurement_order"] for r in report["fixed"]] == list(range(12))
            assert sum(r["first_run_of_prefill_geometry"] for r in report["fixed"]) == 5
            assert all(r["decode_tokens"] == r["batch_size"] * (r["output_len"] - 1) for r in report["fixed"])
            reports.append(report)
        result = aggregate(reports)
        assert len(result["fixed_summary"]) == 24 and len(result["comparisons"]) == 12
        assert all(r["metrics"]["e2e_speedup"]["median"] == 2 for r in result["comparisons"])
        plan = dict(schema_version=2, options=options, jobs=jobs, cases=cases, git=dict(commit="TEST ONLY"))
        summary = compact_summary(plan, result)
        assert len(summary) == 24 and all(r["status"] == "complete" for r in summary)
        # Correct aggregation uses each measured BS/input, not the engine's max capacity.
        bs4 = next(r for r in summary if r["variant"] == "per_channel" and r["batch_size"] == 4 and r["input_len"] == 256 and r["output_len"] == 512)
        assert bs4["prefill_ms"] == 102.4 and bs4["decode_context_end"] == 767
        assert len(trend_deltas(summary)) == 40
        write_outputs(output, plan, reports)
        ET.parse(output / "trends.svg")
        data = json.loads((output / "ai_summary.json").read_text())
        assert data["coverage"]["complete_cases"] == 24 and len(data["samples"]) == 24
        with (output / "summary.csv").open(encoding="utf-8-sig") as stream:
            assert len(list(csv.DictReader(stream))) == 24
        partial = [{**reports[1], "fixed": reports[1]["fixed"][:-1]}]
        incomplete = compact_summary(plan, aggregate(reports[:1] + partial))
        assert sum(r["status"] == "missing" for r in incomplete) == 1
        assert len(aggregate(reports[:1] + partial)["comparisons"]) == 11
        # Repeated full sweeps still perform exactly ONE warmup.
        repeat_job = {**jobs[0], "options": {**options, "rounds": 2}}
        repeated = dict(fixed=[])
        calls.clear()
        with redirect_stdout(io.StringIO()):
            measure_matrix(object(), repeat_job, None, repeated, lambda: None, run=fake_run)
        assert len(calls) == 25 and len(repeated["fixed"]) == 24
        assert [r["case_id"] for r in repeated["fixed"]] == list(range(12)) * 2
        # An interrupted case leaves previous samples and the failing geometry reviewable.
        def failing_run(*args):
            if len(calls) == 3:
                raise RuntimeError("synthetic failure")
            return fake_run(*args)
        calls.clear()
        failed = dict(fixed=[])
        try:
            with redirect_stdout(io.StringIO()):
                measure_matrix(object(), jobs[0], None, failed, lambda: None, run=failing_run)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Failure was swallowed")
        assert len(failed["fixed"]) == 2 and failed["active_case"] == cases[2]

    # Simulate delayed GPU completion: prefill work must finish BEFORE decode starts.
    clock = SimpleNamespace(now=0.0, pending=0.0, syncs=0)
    seqs = [SimpleNamespace(num_completion_tokens=0, max_tokens=3, is_finished=False) for _ in range(2)]
    def synchronize():
        clock.now += clock.pending
        clock.pending = 0
        clock.syncs += 1
    def add(seq):
        clock.now += 0.0001
    def step():
        clock.now += 0.001
        clock.pending += 0.005 if seqs[0].num_completion_tokens == 0 else 0.003
        for seq in seqs:
            seq.num_completion_tokens += 1
            seq.is_finished = seq.num_completion_tokens == seq.max_tokens
    engine = SimpleNamespace(scheduler=SimpleNamespace(add=add), step=step, is_finished=lambda: True,
                             model_runner=SimpleNamespace(input_batch=SimpleNamespace(seq_id_to_slot={})))
    torch = SimpleNamespace(manual_seed=lambda seed: None,
                            cuda=SimpleNamespace(synchronize=synchronize, reset_peak_memory_stats=lambda: None))
    snapshot = dict(allocated_bytes=0, peak_allocated_bytes=100)
    module = sys.modules[__name__]
    with patch.object(module, "make_sequences", return_value=seqs), \
         patch.object(module, "memory_snapshot", return_value=snapshot), \
         patch.object(time, "perf_counter", side_effect=lambda: clock.now):
        row = fixed_run(engine, dict(batch_size=2, input_len=256), 3, 42, torch)
    assert clock.syncs == 3
    assert math.isclose(row["prefill_ms"], 6) and math.isclose(row["decode_ms"], 8)
    assert math.isclose(row["e2e_ms"], 14.2)
    assert math.isclose(row["e2e_ms"], row["enqueue_ms"] + row["prefill_ms"] + row["decode_ms"])
    assert row["decode_tokens"] == 4 and math.isclose(row["tpot_ms"], 4)
    print("CPU self-test passed: 12-case matrix, one global warmup, shared capacity, synchronized stage timing, "
          "token counts, repeated sweeps, missing/failed cases, paired speedups and JSON/CSV/SVG export.")
    return 0


def main():
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.worker_job:
        return run_worker(args.worker_job)
    if args.report_only:
        return regenerate_report(args.report_only)
    if args.resume:
        return resume_plan(args)
    return run_parent(args)


if __name__ == "__main__":
    sys.exit(main())
