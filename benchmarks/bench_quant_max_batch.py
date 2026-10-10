#!/usr/bin/env python3
"""nano_qwen: BF16 / per-channel FP8 / block-wise FP8 throughput comparison.

Targets repository API at fcaebf0d0bbc262c79828d21157e0fb4f3b356f3.
Put this file in nano_qwen/benchmarks/, then run from the repository root:

  python benchmarks/bench_quant_max_batch.py \
      --bf16 ./9b --per-channel-fp8 ./fp8_channel --block-wise-fp8 ./fp8_block

Default matrix: input={256,512,1024}, output={512,1024}, 3 measured repeats.
Search finds the largest verified candidate for input=max(input lengths),
output=max(output lengths), with the same engine allocation used for the sweep.
All workers size the KV pool from the workload rather than remaining VRAM, so
every variant gets identical KV capacity and FP8 weight savings become actual
headroom instead of a bloated KV cache.
Two comparisons: the common minimum of the three maxima, and each model's own
maximum. Max batch means simultaneous fixed-batch requests: no preemption,
chunked prefill, prefix reuse or admission in waves. It is capacity, not the
batch that necessarily maximizes tokens/s. CUDA Graph candidates are 1,2,4,8
and multiples of 16 up to 512. Eager mode searches integer batch sizes.

Each probe uses a fresh subprocess and completes the longest workload once;
probe timings are discarded. Each final model process has ONE extra full
workload warmup, then directly measures all combinations without case warmups.
The engine's built-in initialization warmup/graph capture still runs. Shapes
not covered by this warmup may compile on their first repeat; all raw repeats
are retained. Median-of-3 is the default summary; use --repeats 1 for one pass.

Outputs: summary.csv/json, raw.jsonl, search.json, worker logs, summary.html,
common_batch.svg and own_max_batch.svg. No plotting dependency is needed.
Controller and --help use only the Python standard library; workers use the
repository's installed torch/transformers/Triton/FlashAttention environment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import itertools
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

API_COMMIT = "fcaebf0d0bbc262c79828d21157e0fb4f3b356f3"
VARIANTS = ("bf16", "per_channel_fp8", "block_wise_fp8")
DEFINITIONS = {
    "prefill_ms": "Wall time of the single full-batch engine.step through first-token readiness; includes sampling/CPU/D2H.",
    "prefill_tok_s": "batch * input_len / prefill_seconds; engine prefill throughput, not pure GPU kernel throughput.",
    "decode_ms": "Wall time of output_len-1 decode steps, including final CUDA synchronization.",
    "decode_tok_s": "batch * (output_len-1) / decode_seconds; first output token belongs to prefill.",
    "e2e_ms": "Prefill + decode engine wall time; model load, request construction, enqueue, tokenization and detokenization excluded.",
    "e2e_output_tok_s": "batch * output_len / e2e_seconds.",
    "e2e_total_tok_s": "batch * (input_len+output_len) / e2e_seconds; includes input tokens, use consistent metric when comparing.",
    "decode_first_quarter_ms": "Mean host step wall time in the first quarter of decode steps.",
    "decode_last_quarter_ms": "Mean host step wall time in the last quarter of decode steps.",
    "peak_allocated_mib": "Per-trial torch.cuda.max_memory_allocated; includes model, all KV/GDN pools and graphs, excludes non-PyTorch GPU allocations.",
    "max_batch": "Largest passing candidate found by an approximate monotonic capacity search, verified at worst input/output; bounded by batch_cap and candidate grid.",
}
TIMING_KEYS = ("prefill_ms", "prefill_tok_s", "decode_ms", "decode_tok_s",
               "decode_mean_step_ms", "decode_p50_step_ms", "decode_p95_step_ms",
               "decode_first_quarter_ms", "decode_last_quarter_ms", "e2e_ms",
               "e2e_output_tok_s", "e2e_total_tok_s")


class CapacityError(RuntimeError):
    pass


@contextmanager
def workload_kv_policy(job, torch):
    """Size the KV pool from the workload, not from remaining VRAM.

    The engine default fills the mem_util budget with KV blocks, so a variant
    with smaller weights (FP8) ends up with a much larger KV pool than BF16 and
    no headroom for transient prefill tensors. All variants instead get the
    same worst-case capacity (every slot full to max_model_len, plus one extra
    block per slot), so FP8 weight savings become actual headroom.
    """
    from nano_qwen.engine.model_runner import ModelRunner
    original = ModelRunner.allocate_kv_cache

    def allocate(runner):
        config, hf = runner.config, runner.config.hf_config
        modules = [m for m in runner.model.modules() if hasattr(m, "k_cache") and hasattr(m, "v_cache")]
        if not modules:
            raise RuntimeError("No KV-cache-bearing attention layers")
        max_len = max(job["input_lens"]) + max(job["output_lens"])
        blocks = job["max_batch"] * math.ceil(max_len / runner.block_size)
        blocks += job["max_batch"]  # one extra block per slot
        heads = hf.num_key_value_heads // runner.world_size
        dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        block_bytes = 2 * len(modules) * runner.block_size * heads * dim * hf.dtype.itemsize
        # Reserve headroom for the largest transient the workload itself allocates:
        # the full-batch prefill logits tensor (B x max_input x vocab x fp32).
        # The engine's built-in init warmup also spikes memory, but its peak has
        # already been freed by the time the KV pool is sized, so it is not
        # subtracted here (subtracting it wrongly rejected feasible batches).
        logits_bytes = job["max_batch"] * max(job["input_lens"]) * hf.vocab_size * 4
        free, total = torch.cuda.mem_get_info()
        budget = total - (total - free) - logits_bytes
        if blocks * block_bytes > budget:
            raise CapacityError(f"Workload KV pool exceeds VRAM budget: need {blocks * block_bytes / 2**20:.1f} MiB, "
                                f"available {budget / 2**20:.1f} MiB; reduce batch/input/output or use a smaller model")
        config.num_kvcache_blocks = blocks
        runner.kv_cache = torch.empty((2, len(modules), blocks, runner.block_size, heads, dim),
                                      dtype=hf.dtype, device="cuda")
        for index, module in enumerate(modules):
            module.k_cache = runner.kv_cache[0, index]
            module.v_cache = runner.kv_cache[1, index]

    ModelRunner.allocate_kv_cache = allocate
    try:
        yield
    finally:
        ModelRunner.allocate_kv_cache = original


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def candidates(cap, mode):
    if mode == "eager":
        return list(range(1, cap + 1))
    return [b for b in [1, 2, 4, 8, *range(16, 513, 16)] if b <= cap]


def repo_root(explicit=None):
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "src/nano_qwen/engine/llm_engine.py").is_file():
            raise ValueError(f"Invalid --repo: {root}")
        return root
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        for root in (base, *base.parents):
            if (root / "src/nano_qwen/engine/llm_engine.py").is_file():
                return root
    raise ValueError("Cannot find nano_qwen. Run from its root or set --repo.")


def revision(root):
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
        return {"head": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None}


def checkpoint_info(path, expected):
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    text_cfg = cfg.get("text_config", cfg)
    quant = cfg.get("quantization_config") or text_cfg.get("quantization_config")
    marker = path / "nano_qwen_quantization.json"
    meta = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
    dtype = text_cfg.get("dtype") or text_cfg.get("torch_dtype") or cfg.get("dtype") or cfg.get("torch_dtype")
    if dtype != "bfloat16":
        raise ValueError(f"{expected}: BF16 compute dtype required; config dtype={dtype!r}")
    if expected == "bf16":
        valid = not quant and meta is None
    elif expected == "per_channel_fp8":
        valid = bool(meta and meta.get("format") == "nano_qwen_fp8_v1"
                     and meta.get("weight_scale") == "per_output_channel_float32" and not quant)
    else:
        valid = bool(quant and quant.get("quant_method") == "fp8"
                     and quant.get("weight_block_size") == [128, 128]
                     and quant.get("activation_scheme") == "dynamic" and meta is None)
    if not valid:
        raise ValueError(f"{path} is not the expected {expected} checkpoint. Use the repository converter with the correct --scheme.")
    architecture = {k: text_cfg.get(k) for k in (
        "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
        "layer_types", "linear_num_key_heads", "linear_num_value_heads",
        "linear_key_head_dim", "linear_value_head_dim", "tie_word_embeddings")}
    return {"path": str(path), "dtype": dtype, "architecture": architecture,
            "quantization": meta or quant, "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest()}


def find_max(grid, probe):
    """Exponential search followed by binary search. probe raises on real errors."""
    cache = {}

    def passing(i):
        if i not in cache:
            cache[i] = probe(grid[i])
        return cache[i]

    if not passing(0):
        raise CapacityError("Batch=1 cannot complete the requested longest workload.")
    lo, hi = 0, len(grid)
    i = 1
    while i < len(grid):
        if not passing(i):
            hi = i
            break
        lo = i
        i = max(i + 1, i * 2)
    if hi == len(grid) and lo != len(grid) - 1:
        if passing(len(grid) - 1):
            lo = len(grid) - 1
        else:
            hi = len(grid) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if passing(mid):
            lo = mid
        else:
            hi = mid
    return {"max_batch": grid[lo], "next_failed_batch": grid[hi] if hi < len(grid) else None,
            "candidate_limit_reached": lo == len(grid) - 1,
            "tested_candidates": [{"batch": grid[i], "passed": cache[i]} for i in sorted(cache)]}


def percentile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    i = int(x)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (x - i)


def fixed_batch_run(engine, batch, input_len, output_len, seed, torch, Sequence, SamplingParams):
    if not engine.is_finished() or engine.model_runner.input_batch.seq_id_to_slot:
        raise RuntimeError("Previous workload did not drain all requests/slots.")
    rng = random.Random(seed)
    vocab = int(engine.config.hf_config.vocab_size)
    seqs = [Sequence([rng.randrange(vocab) for _ in range(input_len)],
                     SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True))
            for _ in range(batch)]
    for seq in seqs:
        engine.scheduler.add(seq)
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    _, num_tokens = engine.step()
    prefill_end = time.perf_counter()
    if num_tokens != batch * input_len or any(s.num_completion_tokens != 1 for s in seqs):
        raise RuntimeError("Prefill was split/chunked; fixed-batch comparison is invalid.")
    # Input validation is kept outside each individual decode step timestamp.
    decode_start = time.perf_counter()
    step_seconds = []
    for _ in range(output_len - 1):
        step_start = time.perf_counter()
        _, num_tokens = engine.step()
        step_seconds.append(time.perf_counter() - step_start)
        if num_tokens != -batch:
            raise RuntimeError(f"Decode batch changed: expected {batch}, got num_tokens={num_tokens}.")
    torch.cuda.synchronize()
    end = time.perf_counter()
    if not engine.is_finished() or any(not s.is_finished or s.num_completion_tokens != output_len for s in seqs):
        raise RuntimeError("Output length or completion count does not match the fixed workload.")
    if engine.model_runner.input_batch.seq_id_to_slot:
        raise RuntimeError("Finished requests retained persistent slots.")
    if len(engine.scheduler.block_manager.free_block_ids) != engine.config.num_kvcache_blocks:
        raise RuntimeError("KV blocks leaked after the workload.")
    prefill = prefill_end - start
    # Include the boundary validation overhead so e2e == prefill + decode.
    decode = end - prefill_end
    e2e = end - start
    quarter = max(1, len(step_seconds) // 4)
    return {
        "batch": batch, "input_len": input_len, "output_len": output_len,
        "input_tokens": batch * input_len, "output_tokens": batch * output_len,
        "decode_tokens": batch * (output_len - 1),
        "prefill_ms": prefill * 1000, "prefill_tok_s": batch * input_len / prefill,
        "decode_ms": decode * 1000, "decode_tok_s": batch * (output_len - 1) / decode,
        "decode_mean_step_ms": decode * 1000 / (output_len - 1),
        "decode_p50_step_ms": percentile(step_seconds, .50) * 1000,
        "decode_p95_step_ms": percentile(step_seconds, .95) * 1000,
        "decode_first_quarter_ms": statistics.mean(step_seconds[:quarter]) * 1000,
        "decode_last_quarter_ms": statistics.mean(step_seconds[-quarter:]) * 1000,
        "decode_loop_ms": (end - decode_start) * 1000,
        "e2e_ms": e2e * 1000, "e2e_output_tok_s": batch * output_len / e2e,
        "e2e_total_tok_s": batch * (input_len + output_len) / e2e,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "decode_step_ms": [t * 1000 for t in step_seconds],
    }


def capacity_failure(exc, tb):
    if isinstance(exc, CapacityError):
        return True
    msg = str(exc).lower()
    return ("out of memory" in msg or type(exc).__name__ == "OutOfMemoryError"
            or (isinstance(exc, AssertionError) and "assert config.num_kvcache_blocks > 0" in tb))


def worker(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    result = {"status": "error", "job": job, "trials": []}
    engine = None
    stage = "import"
    try:
        sys.path.insert(0, str(Path(job["repo"]) / "src"))
        import torch
        from nano_qwen.engine.llm_engine import LLMEngine
        from nano_qwen.engine.sequence import Sequence
        from nano_qwen.sampling_params import SamplingParams
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment.")
        torch.manual_seed(job["seed"])
        stage = "initialize"
        with workload_kv_policy(job, torch):
            engine = LLMEngine(job["model"], tensor_parallel_size=1,
                               max_num_seqs=job["max_batch"],
                               max_num_batched_tokens=job["max_batch"] * max(job["input_lens"]),
                               max_model_len=max(job["input_lens"]) + max(job["output_lens"]),
                               gpu_memory_utilization=job["mem_util"],
                               enforce_eager=job["mode"] == "eager",
                               use_prefill_cudagraph=False, enable_prefix_cache=False)
        engine.max_concurrent_batches = job["queue_depth"]
        runner = engine.model_runner
        expected = {"bf16": None, "per_channel_fp8": "Fp8Config", "block_wise_fp8": "Fp8BlockConfig"}[job["variant"]]
        actual = type(engine.config.quant_config).__name__ if engine.config.quant_config is not None else None
        if actual != expected:
            raise RuntimeError(f"Quantization backend mismatch: expected {expected}, loaded {actual}.")
        required_len = max(job["input_lens"]) + max(job["output_lens"])
        if engine.config.max_model_len < required_len:
            raise ValueError(f"Model context {engine.config.max_model_len} < requested {required_len}.")
        stage = "capacity"
        block_size = engine.config.kvcache_block_size
        # Conservative: reserve space for all requested output tokens, including
        # the final token (which normally never needs a KV write).
        needed = job["max_batch"] * math.ceil(required_len / block_size)
        if engine.config.num_kvcache_blocks < needed:
            raise CapacityError(f"Need {needed} KV blocks, allocated {engine.config.num_kvcache_blocks}.")
        batches = sorted(set(job["batches"]))
        if job["mode"] == "decode_graph" and any(b not in runner.graphs for b in batches):
            raise RuntimeError(f"Missing exact decode CUDA Graph for batches={batches}; refusing eager fallback.")

        def reject_preemption(seq):
            raise CapacityError(f"Request {seq.seq_id} would be preempted.")

        engine.scheduler.preempt = reject_preemption
        params = list(runner.model.parameters())
        gdn_bytes = sum(t.numel() * t.element_size() for layer in runner.gdn_layers
                        for t in (layer.conv_states, layer.recurrent_states))
        props = torch.cuda.get_device_properties(0)
        result["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                                 "cuda": torch.version.cuda, "gpu": props.name,
                                 "gpu_total_mib": props.total_memory / 2**20,
                                 "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                 "quant_config": actual, "compute_dtype": str(engine.config.hf_config.dtype),
                                 "repo": revision(job["repo"])}
        result["allocation"] = {"max_num_seqs": job["max_batch"],
                                "max_num_batched_tokens": engine.config.max_num_batched_tokens,
                                "max_model_len": engine.config.max_model_len,
                                "kv_blocks": engine.config.num_kvcache_blocks, "kv_block_size": block_size,
                                "worst_required_kv_blocks": needed,
                                "weights_mib": sum(t.numel() * t.element_size() for t in params) / 2**20,
                                "kv_mib": runner.kv_cache.numel() * runner.kv_cache.element_size() / 2**20,
                                "gdn_mib": gdn_bytes / 2**20,
                                "torch_allocated_mib": torch.cuda.memory_allocated() / 2**20,
                                "decode_graph_sizes": list(runner.graphs)}
        result["environment"]["linear_methods"] = dict(Counter(
            type(module.quant_method).__name__ for module in runner.model.modules()
            if hasattr(module, "quant_method")))
        with torch.inference_mode():
            stage = "probe" if job["kind"] == "probe" else "warmup"
            longest = (job["max_batch"], max(job["input_lens"]), max(job["output_lens"]))
            print(f"BENCH {stage} {job['variant']} batch={longest[0]} input={longest[1]} output={longest[2]}", flush=True)
            result["warmup_or_probe"] = fixed_batch_run(engine, *longest, job["seed"], torch, Sequence, SamplingParams)
            result["extra_benchmark_warmups"] = 0 if job["kind"] == "probe" else 1
            if job["kind"] == "measure":
                stage = "measure"
                # Give the common comparison identical relative ordering even
                # when some variants have a distinct own-max batch. Alternate
                # which comparison goes first across repeats.
                cases = []
                for repeat in range(job["repeats"]):
                    shapes = list(itertools.product(job["input_lens"], job["output_lens"]))
                    random.Random(job["seed"] + repeat).shuffle(shapes)
                    ordered_batches = batches if repeat % 2 == 0 else list(reversed(batches))
                    cases.extend((repeat, batch, inp, out) for batch in ordered_batches for inp, out in shapes)
                for repeat, batch, input_len, output_len in cases:
                    seed = job["seed"] + repeat * 1000003 + input_len * 1009 + output_len
                    row = fixed_batch_run(engine, batch, input_len, output_len, seed, torch, Sequence, SamplingParams)
                    row.update(variant=job["variant"], repeat=repeat,
                               comparison=[name for name, b in job["comparison_batches"].items() if b == batch])
                    result["trials"].append(row)
                    write_json(job["result"], result)
                    print(f"BENCH {job['variant']} B={batch} I={input_len} O={output_len} repeat={repeat+1} "
                          f"prefill={row['prefill_tok_s']:.1f} decode={row['decode_tok_s']:.1f} e2e={row['e2e_output_tok_s']:.1f}", flush=True)
        result["status"] = "ok"
    except Exception as exc:
        tb = traceback.format_exc()
        result.update(status="capacity_error" if capacity_failure(exc, tb) else "error",
                      error_type=type(exc).__name__, error=str(exc), traceback=tb, failure_stage=stage)
        print(tb, file=sys.stderr, flush=True)
    finally:
        if engine is not None:
            try:
                engine.exit()
            except Exception:
                result["cleanup_error"] = traceback.format_exc()
                if result["status"] == "ok":
                    result["status"] = "error"
                    result["error"] = "Engine cleanup failed; see cleanup_error."
        write_json(job["result"], result)
    return 0 if result["status"] == "ok" else 2


def run_job(job, out, timeout):
    tag = f"{job['variant']}_{job['kind']}_b{job['max_batch']}"
    config = out / "jobs" / (tag + ".job.json")
    result_path = out / "jobs" / (tag + ".result.json")
    log = out / "logs" / (tag + ".log")
    job = dict(job, result=str(result_path))
    write_json(config, job)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    env["PYTHONPATH"] = str(Path(job["repo"]) / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print(f"[{job['variant']}] {job['kind']} B={job['max_batch']}; log={log}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        try:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(config)],
                                     cwd=job["repo"], env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Worker timeout after {timeout}s. See {log}") from exc
    if not result_path.exists():
        raise RuntimeError(f"Worker exited {process.returncode} without a result. See {log}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if ((result["status"] not in ("ok", "capacity_error") and job["kind"] != "measure")
            or (process.returncode and result["status"] == "ok")):
        raise RuntimeError(f"{job['variant']} failed at {result.get('failure_stage')}: {result.get('error')}. See {log}")
    print(f"  {result['status']}" + (f": {result.get('error')}" if result['status'] != "ok" else ""), flush=True)
    return result


def aggregate(trials):
    groups = {}
    for t in trials:
        for comparison in t["comparison"]:
            key = (comparison, t["variant"], t["batch"], t["input_len"], t["output_len"])
            groups.setdefault(key, []).append(t)
    rows = []
    for (comparison, variant, batch, input_len, output_len), values in sorted(groups.items()):
        row = dict(comparison=comparison, variant=variant, batch=batch,
                   input_len=input_len, output_len=output_len, repeats=len(values))
        for key in TIMING_KEYS:
            row[key] = statistics.median(v[key] for v in values)
        for key in ("prefill_tok_s", "decode_tok_s", "e2e_output_tok_s"):
            row[key + "_min"] = min(v[key] for v in values)
            row[key + "_max"] = max(v[key] for v in values)
        row["peak_allocated_mib"] = max(v["peak_allocated_mib"] for v in values)
        row["peak_reserved_mib"] = max(v["peak_reserved_mib"] for v in values)
        rows.append(row)
    baseline = {(r["comparison"], r["input_len"], r["output_len"]): r for r in rows if r["variant"] == "bf16"}
    for r in rows:
        base = baseline.get((r["comparison"], r["input_len"], r["output_len"]))
        if base:
            for metric in ("prefill_tok_s", "decode_tok_s", "e2e_output_tok_s"):
                r[metric + "_vs_bf16"] = r[metric] / base[metric]
    return rows


def chart_svg(rows, comparison):
    """Exact-data SVG line charts, with three phases in separate panels."""
    rows = [r for r in rows if r["comparison"] == comparison]
    outputs = sorted({r["output_len"] for r in rows})
    inputs = sorted({r["input_len"] for r in rows})
    panel_w, panel_h = 380, 255
    width, height = panel_w * 3, 85 + panel_h * len(outputs)
    colors = {"bf16": "#334155", "per_channel_fp8": "#2563eb", "block_wise_fp8": "#e76f23"}
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<g font-family="Arial,sans-serif" font-size="12" fill="#334155">',
         f'<text x="20" y="27" font-size="20" font-weight="bold">{html.escape(comparison)}: median throughput</text>']
    for i, v in enumerate(VARIANTS):
        s.append(f'<rect x="{20+i*345}" y="45" width="14" height="9" fill="{colors[v]}"/><text x="{40+i*345}" y="54">{v}</text>')
    metrics = [("prefill_tok_s", "Prefill input tok/s"), ("decode_tok_s", "Decode output tok/s"), ("e2e_output_tok_s", "E2E output tok/s")]
    for row_i, output_len in enumerate(outputs):
        for col, (metric, title) in enumerate(metrics):
            x0, y0 = col * panel_w, 80 + row_i * panel_h
            left, top, pw, ph = x0 + 65, y0 + 37, 290, 155
            data = [r for r in rows if r["output_len"] == output_len]
            ymax = max(r[metric] for r in data) * 1.20
            s.append(f'<text x="{x0+20}" y="{y0+18}" font-weight="bold">{title}; output={output_len}</text>')
            for tick in range(5):
                y = top + ph * (1 - tick / 4)
                s.append(f'<line x1="{left}" y1="{y}" x2="{left+pw}" y2="{y}" stroke="#e2e8f0"/><text x="{left-7}" y="{y+4}" text-anchor="end">{ymax*tick/4:.0f}</text>')
            def xpos(value):
                return left + pw * ((value - inputs[0]) / (inputs[-1] - inputs[0]) if len(inputs) > 1 else .5)
            for inp in inputs:
                x = xpos(inp)
                s.append(f'<text x="{x}" y="{top+ph+19}" text-anchor="middle">{inp}</text>')
            s.append(f'<text x="{left+pw/2}" y="{top+ph+42}" text-anchor="middle">Input tokens/request</text>')
            for v in VARIANTS:
                points = sorted([r for r in data if r["variant"] == v], key=lambda r:r["input_len"])
                coordinates = [(xpos(r["input_len"]), top + ph * (1 - r[metric] / ymax)) for r in points]
                line = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
                s.append(f'<polyline points="{line}" fill="none" stroke="{colors[v]}" stroke-width="2"/>')
                for r, (x, y) in zip(points, coordinates):
                    s.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{colors[v]}"><title>{v}; batch={r["batch"]}; input={r["input_len"]}; output={output_len}; {metric}={r[metric]:.3f}</title></circle>')
    s.extend(["</g>", "</svg>"])
    return "\n".join(s)


def save_reports(out, meta, trials):
    rows = aggregate(trials)
    write_json(out / "summary.json", dict(meta, metric_definitions=DEFINITIONS, rows=rows))
    with (out / "raw.jsonl").open("w", encoding="utf-8") as handle:
        for row in trials:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    charts = []
    for comparison in ("common_batch", "own_max_batch"):
        if any(r["comparison"] == comparison for r in rows):
            svg = chart_svg(rows, comparison)
            (out / (comparison + ".svg")).write_text(svg, encoding="utf-8")
            charts.append(svg)
    heads = ("comparison", "variant", "batch", "input_len", "output_len", "prefill_tok_s", "decode_tok_s", "e2e_output_tok_s", "decode_tok_s_vs_bf16")
    table = '<table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in heads) + '</tr></thead><tbody>'
    for row in rows:
        table += '<tr>' + ''.join('<td>' + html.escape(f'{row[h]:.2f}' if isinstance(row.get(h), float) else str(row.get(h, ""))) + '</td>' for h in heads) + '</tr>'
    table += '</tbody></table>'
    notes = html.escape(json.dumps({"status":meta.get("status"), "max_batches":meta.get("max_batches"),
                                    "common_batch":meta.get("common_batch"), "search":meta.get("search")}, indent=2))
    page = ('<!doctype html><html><meta charset="utf-8"><title>nano_qwen quantization benchmark</title>'
            '<style>body{font:14px Arial;margin:30px;color:#334155}svg{max-width:100%;height:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border:1px solid #cbd5e1;text-align:right}pre{white-space:pre-wrap}</style>'
            '<h1>nano_qwen: BF16 / per-channel FP8 / block-wise FP8</h1>'
            '<p>Common batch compares execution at equal concurrency. Own max batch combines execution speed and memory capacity. '
            'E2E excludes loading/tokenization. Decode counts output_len−1 tokens per request. '
            'Prefill uses engine wall time through the first token. Charts use independent vertical scales. '
            'All repetitions are retained; summaries are medians. Maxima are verified within the configured candidate cap.</p>'
            + '<pre>' + notes + '</pre>' + ''.join(charts) + table + '</html>')
    (out / "summary.html").write_text(page, encoding="utf-8")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bf16", required=True)
    parser.add_argument("--per-channel-fp8", required=True)
    parser.add_argument("--block-wise-fp8", required=True)
    parser.add_argument("--repo")
    parser.add_argument("--input-lens", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--output-lens", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--batch-cap", type=int, default=512)
    parser.add_argument("--max-batches", type=int, nargs=3, metavar=("BF16", "CHANNEL", "BLOCK"),
                        help="Skip search using known maxima; final warmup still validates capacity.")
    parser.add_argument("--mode", choices=("decode_graph", "eager"), default="decode_graph")
    parser.add_argument("--mem-util", type=float, default=.80)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--queue-depth", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds per worker process.")
    parser.add_argument("--output-dir", help="Must be empty/new; default benchmarks/reports/quant_max_BATCH_TIMESTAMP.")
    args = parser.parse_args()
    if any(v <= 0 for v in args.input_lens) or any(v <= 1 for v in args.output_lens):
        parser.error("Input lengths must be positive and output lengths must be >=2.")
    if args.batch_cap < 1 or args.repeats < 1 or args.timeout < 1 or not 0 < args.mem_util < 1:
        parser.error("Require batch-cap/repeats/timeout>=1 and 0<mem-util<1.")
    if args.mode == "decode_graph" and args.batch_cap > 512:
        parser.error("This repository captures exact decode graphs only up to batch=512.")
    if args.max_batches and any(b not in candidates(args.batch_cap, args.mode) for b in args.max_batches):
        parser.error("Every --max-batches value must belong to the selected mode's candidate grid and cap.")
    args.input_lens = sorted(set(args.input_lens))
    args.output_lens = sorted(set(args.output_lens))
    return args


def main():
    args = parse_args()
    root = repo_root(args.repo)
    paths = [Path(p).expanduser().resolve() for p in (args.bf16, args.per_channel_fp8, args.block_wise_fp8)]
    checkpoints = {v:checkpoint_info(p, v) for v, p in zip(VARIANTS, paths)}
    if any(info["architecture"] != checkpoints["bf16"]["architecture"] for info in checkpoints.values()):
        raise ValueError("Checkpoint architectures differ. Convert both FP8 models from the same BF16 source.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "benchmarks/reports" / f"quant_max_{stamp}_{os.getpid()}"
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}")
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    meta = {"status": "running", "timestamp_utc": stamp, "api_commit": API_COMMIT,
            "repo": revision(root), "settings": vars(args), "checkpoints": checkpoints,
            "search": {}, "max_batches": {}, "workers": {}}
    trials = []
    base = {"repo": str(root), "input_lens": args.input_lens, "output_lens": args.output_lens,
            "mem_util": args.mem_util, "mode": args.mode, "repeats": args.repeats,
            "queue_depth": args.queue_depth, "seed": args.seed}
    try:
        grid = candidates(args.batch_cap, args.mode)
        for idx, (variant, model) in enumerate(zip(VARIANTS, paths)):
            if args.max_batches:
                maximum = args.max_batches[idx]
                meta["search"][variant] = {"max_batch": maximum, "source": "user_supplied", "candidate_limit_reached": None}
            else:
                def probe(batch, variant=variant, model=model):
                    job = dict(base, variant=variant, model=str(model), kind="probe", max_batch=batch, batches=[batch])
                    result = run_job(job, out, args.timeout)
                    return result["status"] == "ok"
                info = find_max(grid, probe)
                maximum = info["max_batch"]
                meta["search"][variant] = dict(info, source="capacity_search")
            meta["max_batches"][variant] = maximum
            write_json(out / "search.json", meta["search"])
        common = min(meta["max_batches"].values())
        meta["common_batch"] = common
        print(f"Max verified batches: {meta['max_batches']}; common batch={common}", flush=True)
        failures = []
        for variant, model in zip(VARIANTS, paths):
            maximum = meta["max_batches"][variant]
            job = dict(base, variant=variant, model=str(model), kind="measure", max_batch=maximum,
                       batches=sorted({common, maximum}),
                       comparison_batches={"common_batch":common, "own_max_batch":maximum})
            try:
                result = run_job(job, out, args.timeout)
            except RuntimeError as exc:
                # Worker crashed/timeout: record and keep going so the other
                # variants still get measured.
                failures.append(dict(variant=variant, error=str(exc)))
                print(f"[{variant}] measure aborted: {exc}", flush=True)
                continue
            trials.extend(result["trials"])
            meta["workers"][variant] = {k: result.get(k) for k in ("environment", "allocation", "extra_benchmark_warmups")}
            if result["status"] != "ok":
                failures.append(dict(variant=variant, stage=result.get("failure_stage"),
                                     error=result.get("error"), partial_trials=len(result.get("trials", []))))
                print(f"[{variant}] measure failed ({result.get('failure_stage')}): {result.get('error')}", flush=True)
                continue
            save_reports(out, meta, trials)
        meta["failures"] = failures
        meta["status"] = "completed" if not failures else "completed_with_failures"
    except Exception as exc:
        meta.update(status="failed", error=str(exc))
        raise
    finally:
        rows = save_reports(out, meta, trials)
        print(f"Reports: {out}", flush=True)
        for row in rows:
            print(f"{row['comparison']:13s} {row['variant']:16s} B={row['batch']:3d} "
                  f"I={row['input_len']:4d} O={row['output_len']:4d} "
                  f"prefill={row['prefill_tok_s']:9.1f} decode={row['decode_tok_s']:8.1f} "
                  f"e2e={row['e2e_output_tok_s']:8.1f} decode_vs_bf16={row.get('decode_tok_s_vs_bf16',0):.3f}x")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        sys.exit(worker(sys.argv[2]))
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
