"""Engine-level performance benchmark for the Qwen3.5 inference chain.

Measures four engine modes and attributes each gain to one feature::

    P0  eager, queue=1, sync D2H                  (conservative baseline)
    P1  eager, queue=2, async D2H                 -> P1-P0: MRV2 queue + async D2H
    P2  decode CUDA Graph, eager prefill, queue=2 -> P2-P1: decode CUDA Graph
    P3  piecewise prefill + decode CUDA Graph     -> P3-P2: piecewise prefill graph

Uses the production random Sampler with fixed seeds (never ArgmaxSampler).
No per-step ``torch.cuda.synchronize()`` — only before/after each measured
run, so the async pipeline stays intact.  Kernel-level GDN perf lives in
``bench_gdn_flashqla.py`` and is not duplicated here.

Each mode runs in its own subprocess; ``--mode parent`` spawns P0..P3 and
prints the comparison tables.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Callable

import torch

from engine_bench_utils import (
    PERF_MODES,
    GraphPathCounter,
    ModeConfig,
    env_info,
    make_engine,
    make_params,
    make_random_prompt,
    percentile,
    repo_log_path,
    run_until_idle,
    save_json,
)

try:
    from nano_qwen.engine.sequence import Sequence
except ImportError:  # pragma: no cover
    Sequence = None


def reset_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def med(values: list[float]) -> float:
    return percentile(values, 50)


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------

def measure(
    engine,
    build: Callable[[], list[Sequence]],
    warmup: int,
    rounds: int,
    seed_per_run: bool = True,
) -> list[tuple[float, "RunResultType", list[Sequence]]]:
    """Run `build` warmup times then `rounds` times; returns (wall, result, seqs).

    ``build(round_idx)`` receives the round index — builders must vary their
    prompt tokens per round, because the engine's block manager hashes filled
    blocks (prefix cache); re-submitting an identical prompt hits the cache
    and prepare_prefill raises ``prefix-cache prefill is not supported``.
    """
    for r in range(warmup):
        seqs = build(r)
        for s in seqs:
            engine.scheduler.add(s)
        run_until_idle(engine, seqs)
    out = []
    for r in range(rounds):
        if seed_per_run:
            reset_seed(r)
        seqs = build(warmup + r)  # unique prompts per run: never repeat a warmup prompt
        for s in seqs:
            engine.scheduler.add(s)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = run_until_idle(engine, seqs)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        out.append((wall, res, seqs))
    return out


def workload_prefill_latency(engine, counter: GraphPathCounter, args) -> dict[str, Any]:
    """Single-request prefill latency; 128..2048 tokens (graph bucket tops at 512)."""
    vocab = engine.config.hf_config.vocab_size
    lengths = [128, 256, 512, 1024, 2048]
    runs: list[dict[str, Any]] = []
    snapshot = counter.as_dict()
    for length in lengths:
        def build(r, L=length):
            return [Sequence(make_random_prompt(L, seed=400 + L + r * 1000, vocab_size=vocab), make_params(1))]
        measured = measure(engine, build, args.warmup_rounds, args.rounds)
        for i, (wall, res, seqs) in enumerate(measured):
            ttft = res.ttft_s[seqs[0].seq_id]
            runs.append({
                "prompt_len": length,
                "cold": i == 0,
                "ttft_s": round(ttft, 4),
                "tokens_per_s": round(length / ttft, 1),
                "wall_s": round(wall, 4),
            })
    delta = _counter_delta(snapshot, counter.as_dict())
    return {"runs": runs, "graph_stats": delta}


def workload_varlen_prefill(engine, counter: GraphPathCounter, args) -> dict[str, Any]:
    """Packed-varlen prefill throughput across request layouts."""
    vocab = engine.config.hf_config.vocab_size
    layouts = [[64, 64], [32, 96], [17, 31, 80], [128, 256], [64, 128, 256, 512]]
    runs: list[dict[str, Any]] = []
    snapshot = counter.as_dict()
    for i, layout in enumerate(layouts):
        def build(r, layout=layout):
            return [
                Sequence(make_random_prompt(L, seed=500 + i * 10 + j + r * 1000, vocab_size=vocab), make_params(2))
                for j, L in enumerate(layout)
            ]
        measured = measure(engine, build, args.warmup_rounds, args.rounds)
        total_tokens = sum(layout)
        for wall, res, seqs in measured:
            runs.append({
                "layout": layout,
                "total_tokens": total_tokens,
                "requests": len(layout),
                "wall_s": round(wall, 4),
                "tokens_per_s": round(total_tokens / wall, 1),
            })
    delta = _counter_delta(snapshot, counter.as_dict())
    return {"runs": runs, "graph_stats": delta}


def workload_decode(engine, counter: GraphPathCounter, args) -> dict[str, Any]:
    """Decode throughput per batch size; 1/2/4/8/16 graph hits, 3/5 eager fallback."""
    vocab = engine.config.hf_config.vocab_size
    batch_sizes = [1, 2, 3, 4, 5, 8, 16]
    runs: list[dict[str, Any]] = []
    for bs in batch_sizes:
        snapshot = counter.as_dict()

        def build(r, bs=bs):
            return [
                Sequence(make_random_prompt(128, seed=600 + bs + i + r * 1000, vocab_size=vocab),
                         make_params(args.decode_tokens))
                for i in range(bs)
            ]
        measured = measure(engine, build, args.warmup_rounds, args.rounds)
        delta = _counter_delta(snapshot, counter.as_dict())
        graph_hits = delta["decode_graph_hits"]
        fallbacks = delta["decode_eager_fallbacks"]
        status = "graph" if graph_hits and not fallbacks else ("fallback" if fallbacks and not graph_hits else "mixed")
        for wall, res, seqs in measured:
            ttfts = [res.ttft_s[s.seq_id] for s in seqs]
            itls = [v for s in seqs for v in res.itl_s[s.seq_id]]
            e2es = [res.e2e_s[s.seq_id] for s in seqs]
            # first completion token comes from prefill; the rest are decode steps
            decode_latency = [e2es[i] - ttfts[i] for i in range(len(seqs))]
            runs.append({
                "batch_size": bs,
                "graph": status,
                "wall_s": round(wall, 4),
                "ttft_s": round(med(ttfts), 4),
                "tpot_s": round(med(itls), 4),
                "decode_latency_s": round(med(decode_latency), 4),
                "decode_tokens_per_sec": round(
                    (args.decode_tokens - 1) * bs / med(decode_latency), 1
                ) if med(decode_latency) > 0 else 0.0,
            })
    return {"runs": runs}


def workload_mixed(engine, counter: GraphPathCounter, args) -> dict[str, Any]:
    """Mixed request pool: TTFT / TPOT / E2E / throughput percentiles."""
    vocab = engine.config.hf_config.vocab_size
    prompt_lengths = [32, 64, 128, 256, 512]
    runs: list[dict[str, Any]] = []
    snapshot = counter.as_dict()
    for n_requests in (8, 16):
        for out_len in (32, 64):
            def build(r, n_requests=n_requests, out_len=out_len):
                seqs = []
                for i in range(n_requests):
                    L = prompt_lengths[i % len(prompt_lengths)]
                    seqs.append(
                        Sequence(make_random_prompt(L, seed=700 + n_requests * 10 + out_len + i + r * 1000, vocab_size=vocab),
                                 make_params(out_len))
                    )
                return seqs
            measured = measure(engine, build, args.mixed_warmup_rounds, args.mixed_rounds)
            for wall, res, seqs in measured:
                ttfts = [res.ttft_s[s.seq_id] for s in seqs]
                itls = [v for s in seqs for v in res.itl_s[s.seq_id]]
                e2es = [res.e2e_s[s.seq_id] for s in seqs]
                input_tokens = sum(s.num_prompt_tokens for s in seqs)
                output_tokens = sum(len(s.completion_token_ids) for s in seqs)
                runs.append({
                    "requests": n_requests,
                    "output_len": out_len,
                    "wall_s": round(wall, 4),
                    "requests_per_sec": round(n_requests / wall, 2),
                    "input_tokens_per_sec": round(input_tokens / wall, 1),
                    "output_tokens_per_sec": round(output_tokens / wall, 1),
                    "ttft_mean": round(sum(ttfts) / len(ttfts), 4),
                    "ttft_p50": round(med(ttfts), 4),
                    "ttft_p95": round(percentile(ttfts, 95), 4),
                    "tpot_mean": round(sum(itls) / len(itls), 4),
                    "tpot_p50": round(med(itls), 4),
                    "tpot_p95": round(percentile(itls, 95), 4),
                    "e2e_mean": round(sum(e2es) / len(e2es), 4),
                    "e2e_p50": round(med(e2es), 4),
                    "e2e_p95": round(percentile(e2es, 95), 4),
                })
    delta = _counter_delta(snapshot, counter.as_dict())
    return {"runs": runs, "graph_stats": delta}


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {k: after[k] - before.get(k, 0) for k in after}


# ---------------------------------------------------------------------------
# Child / parent
# ---------------------------------------------------------------------------

def run_child(args) -> None:
    mode = next(m for m in PERF_MODES if m.name == args.mode)
    print(f"[{mode.name}] building engine (max_num_seqs={args.max_num_seqs})...")
    engine = make_engine(
        args.model, mode,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    mem_after_init = {
        "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "max_memory_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }

    results: dict[str, Any] = {
        "meta": {
            **env_info(),
            "mode": mode.name,
            "model": args.model,
            "config": {
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            },
        },
        "memory": {"after_init": mem_after_init},
        "workloads": {},
    }

    with GraphPathCounter(engine.model_runner) as counter:
        reset_seed(0)
        # Largest prefill run doubles as the prefill memory-peak measurement.
        torch.cuda.reset_peak_memory_stats()
        results["workloads"]["prefill_latency"] = workload_prefill_latency(engine, counter, args)
        results["memory"]["prefill_peak"] = {
            "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "max_memory_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        }
        results["workloads"]["varlen_prefill"] = workload_varlen_prefill(engine, counter, args)
        results["workloads"]["decode"] = workload_decode(engine, counter, args)
        results["workloads"]["mixed"] = workload_mixed(engine, counter, args)
        results["graph_stats"] = counter.as_dict()

    results["memory"]["overall_peak"] = {
        "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "max_memory_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }
    engine.exit()

    out_path = args.json_out or repo_log_path(f"bench_engine_qwen35_{mode.name}", subdir="bench")
    save_json(out_path, results)
    print(f"[{mode.name}] results -> {out_path}")
    _print_workloads(results["workloads"])


def _print_workloads(workloads: dict[str, Any]) -> None:
    for name, wl in workloads.items():
        print(f"\n== {name} ==")
        for run in wl.get("runs", []):
            print("  " + " ".join(f"{k}={v}" for k, v in run.items()))


def run_parent(args) -> None:
    mode_names = [m.name for m in PERF_MODES]
    per_mode: dict[str, dict[str, Any]] = {}
    base = args.json_out or None
    for name in mode_names:
        out_path = (base + f".{name}.json") if base else repo_log_path(f"bench_engine_qwen35_{name}", subdir="bench")
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--model", args.model,
            "--mode", name,
            "--json-out", out_path,
            "--max-num-seqs", str(args.max_num_seqs),
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            "--max-model-len", str(args.max_model_len),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--rounds", str(args.rounds),
            "--warmup-rounds", str(args.warmup_rounds),
            "--mixed-rounds", str(args.mixed_rounds),
            "--mixed-warmup-rounds", str(args.mixed_warmup_rounds),
            "--decode-tokens", str(args.decode_tokens),
        ]
        print(f"=== spawning mode {name}")
        completed = subprocess.run(cmd, text=True)
        if completed.returncode != 0:
            print(f"!!! mode {name} subprocess failed (rc={completed.returncode})")
            per_mode[name] = {"child_rc": completed.returncode}
            continue
        with open(out_path, encoding="utf-8") as f:
            per_mode[name] = json.load(f)

    print("\n" + "=" * 110)
    print("PERFORMANCE SUMMARY (median over rounds)")
    print("=" * 110)

    def wl_med(mode: str, workload: str, key: str, filt: dict | None = None):
        runs = per_mode.get(mode, {}).get("workloads", {}).get(workload, {}).get("runs", [])
        if filt:
            runs = [r for r in runs if all(r.get(k) == v for k, v in filt.items())]
        vals = [r[key] for r in runs if key in r and r[key] is not None]
        return med(vals) if vals else float("nan")

    headers = ["workload", "P0", "P1", "P2", "P3", "P1-P0", "P2-P1", "P3-P2"]
    print("".join(f"{h:>22}" for h in headers))

    def row(label: str, getter: Callable[[str], float], unit_fmt: str = "{:.1f}"):
        vals = [getter(m) for m in mode_names]
        deltas = [vals[i + 1] - vals[i] if not any(v != v for v in vals[i:i + 2]) else float("nan") for i in range(3)]
        cells = [label] + vals + deltas
        print("".join((f"{c:>22}" if isinstance(c, str) else f"{unit_fmt.format(c):>22}") for c in cells))

    row("prefill_512_tok/s", lambda m: wl_med(m, "prefill_latency", "tokens_per_s", {"prompt_len": 512, "cold": False}), "{:.1f}")
    row("prefill_2048_tok/s", lambda m: wl_med(m, "prefill_latency", "tokens_per_s", {"prompt_len": 2048, "cold": False}), "{:.1f}")
    row("varlen_960_tok/s", lambda m: wl_med(m, "varlen_prefill", "tokens_per_s", {"total_tokens": 960}), "{:.1f}")
    for bs in (1, 2, 3, 4, 5, 8, 16):
        row(f"decode_bs{bs}_tok/s", lambda m, bs=bs: wl_med(m, "decode", "decode_tokens_per_sec", {"batch_size": bs}), "{:.1f}")
    for cfg in ((8, 32), (16, 64)):
        row(
            f"mixed_{cfg[0]}x{cfg[1]}_out_tok/s",
            lambda m, cfg=cfg: wl_med(m, "mixed", "output_tokens_per_sec", {"requests": cfg[0], "output_len": cfg[1]}),
            "{:.1f}",
        )
    row("mixed_ttft_p50_s", lambda m: wl_med(m, "mixed", "ttft_p50", {"requests": 16, "output_len": 64}), "{:.3f}")
    row("mixed_tpot_p50_s", lambda m: wl_med(m, "mixed", "tpot_p50", {"requests": 16, "output_len": 64}), "{:.3f}")

    print("\nGRAPH PATH STATISTICS (per mode)")
    for m in mode_names:
        gs = per_mode.get(m, {}).get("graph_stats", {})
        print(f"  {m}: {gs}")

    print("\nGPU MEMORY (per mode, MiB)")
    for m in mode_names:
        mem = per_mode.get(m, {}).get("memory", {})
        print(f"  {m}: {mem}")

    agg_path = save_json(base or repo_log_path("bench_engine_qwen35_aggregate", subdir="bench"), {
        "meta": env_info(),
        "modes": per_mode,
    })
    print(f"\naggregate JSON: {agg_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Engine-level performance benchmark (Qwen3.5)")
    parser.add_argument("--model", default=os.environ.get("NANO_QWEN_MODEL", "/home/wei/code/models/qwen"))
    parser.add_argument("--mode", default="parent", choices=["parent"] + [m.name for m in PERF_MODES])
    parser.add_argument("--json-out", default="")
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--mixed-rounds", type=int, default=3)
    parser.add_argument("--mixed-warmup-rounds", type=int, default=1)
    parser.add_argument("--decode-tokens", type=int, default=64)
    args = parser.parse_args()

    if args.mode == "parent":
        run_parent(args)
    else:
        run_child(args)


if __name__ == "__main__":
    main()
