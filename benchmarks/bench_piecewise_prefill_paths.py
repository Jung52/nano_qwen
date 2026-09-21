"""Focused P2/P3 benchmark for piecewise-prefill routing.

The script keeps each mode in a separate process and covers the cases that
distinguish graph benefit from scheduler interaction:

* pure prefill buckets: [512], [128] * 4, [32, 64, 128, 256]
* chunked 960-token varlen prefill with a 512-token budget
* a late 448-token prefill inserted after decode requests are running

It reports wall time, path counters, selected graph buckets, and the lazy
per-layer graph-capture time for P3.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from statistics import median
from typing import Any, Callable

import torch

from engine_bench_utils import (
    PERF_MODE_NAMES,
    ModeConfig,
    GraphPathCounter,
    env_info,
    make_engine,
    make_params,
    make_random_prompt,
    repo_log_path,
    run_until_idle,
    save_json,
)
from nano_qwen.engine.sequence import Sequence


PURE_LAYOUTS: list[tuple[str, list[int]]] = [
    ("pure_512", [512]),
    ("pure_128x4", [128, 128, 128, 128]),
    ("pure_varlen_480", [32, 64, 128, 256]),
    ("chunked_varlen_960", [64, 128, 256, 512]),
]
DEFAULT_PREFILL_GRAPH_BUCKETS = [1, 2, 4, 8] + list(range(16, 513, 16))


def uses_expandable_segments() -> bool:
    setting = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    for item in setting.split(";"):
        key, separator, value = item.partition(":")
        if not separator or key.strip() != "expandable_segments":
            continue
        if value.strip().lower() not in {"0", "false", "no", "off"}:
            return True
    return False


def validate_allocator_for_cuda_graphs() -> None:
    if uses_expandable_segments():
        raise SystemExit(
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is not supported "
            "by this CUDA Graph benchmark; captured graph replay can become "
            "unstable and fail late with 'CUDA driver error: device not ready'. "
            "Unset PYTORCH_CUDA_ALLOC_CONF (or set expandable_segments:False) "
            "and rerun."
        )


def reset_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def remove_finished_requests(engine) -> None:
    engine.model_runner.input_batch.remove_finished()


def memory_snapshot() -> dict[str, float]:
    torch.cuda.synchronize()
    mib = 1024 * 1024
    allocator_stats = torch.cuda.memory_stats()
    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / mib, 1),
        "reserved_mib": round(torch.cuda.memory_reserved() / mib, 1),
        "max_allocated_mib": round(torch.cuda.max_memory_allocated() / mib, 1),
        "max_reserved_mib": round(torch.cuda.max_memory_reserved() / mib, 1),
        "num_alloc_retries": allocator_stats.get("num_alloc_retries", 0),
        "num_ooms": allocator_stats.get("num_ooms", 0),
        "allocation_count": allocator_stats.get("allocation.all.current", 0),
        "segment_count": allocator_stats.get("segment.all.current", 0),
    }


def counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, after_value in after.items():
        before_value = before.get(key, 0)
        if isinstance(after_value, int):
            result[key] = after_value - before_value
        elif isinstance(after_value, dict):
            nested = {
                str(item_key): item_after - before_value.get(item_key, 0)
                for item_key, item_after in after_value.items()
                if item_after - before_value.get(item_key, 0)
            }
            if nested:
                result[key] = nested
    return result


def capture_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"captures": 0, "total_ms": 0.0, "median_ms": 0.0, "buckets": {}}

    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        grouped[record["bucket"]].append(record["ms"])
    buckets = {
        str(bucket): {
            "captures": len(values),
            "total_ms": round(sum(values), 3),
            "median_ms": round(median(values), 3),
        }
        for bucket, values in sorted(grouped.items())
    }
    values = [record["ms"] for record in records]
    return {
        "captures": len(records),
        "total_ms": round(sum(values), 3),
        "median_ms": round(median(values), 3),
        "buckets": buckets,
    }


def instrument_capture(runner) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    original = runner.capture_prefill_layer_segments

    def wrapped(layer, graph_size: int, has_residual: bool):
        started = time.perf_counter()
        output = original(layer, graph_size, has_residual)
        records.append({
            "bucket": graph_size,
            "ms": (time.perf_counter() - started) * 1000.0,
        })
        return output

    runner.capture_prefill_layer_segments = wrapped
    return records


def selected_bucket_counts(runner, delta: dict[str, Any]) -> dict[str, int]:
    sizes = getattr(runner, "prefill_graph_sizes", [])
    selections: dict[str, int] = defaultdict(int)
    for token_count, calls in delta.get("prefill_token_counts", {}).items():
        bucket = next((size for size in sizes if size >= int(token_count)), None)
        label = "none" if bucket is None else str(bucket)
        selections[label] += calls
    return dict(sorted(selections.items()))


def run_measurements(
    engine,
    counter: GraphPathCounter,
    capture_records: list[dict[str, Any]],
    builder: Callable[[int], list[Sequence]],
    warmup_rounds: int,
    rounds: int,
) -> tuple[list[float], dict[str, Any]]:
    for round_index in range(warmup_rounds):
        reset_seed(round_index)
        sequences = builder(round_index)
        for sequence in sequences:
            engine.scheduler.add(sequence)
        run_until_idle(engine, sequences)
        remove_finished_requests(engine)

    counter_before = counter.as_dict()
    capture_before = len(capture_records)
    walls: list[float] = []
    for round_index in range(rounds):
        reset_seed(warmup_rounds + round_index)
        sequences = builder(warmup_rounds + round_index)
        for sequence in sequences:
            engine.scheduler.add(sequence)
        torch.cuda.synchronize()
        started = time.perf_counter()
        run_until_idle(engine, sequences)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - started)
        remove_finished_requests(engine)

    stats = {
        "graph": counter_delta(counter_before, counter.as_dict()),
        "capture": capture_summary(capture_records[capture_before:]),
    }
    return walls, stats


def workload_pure(
    engine,
    counter: GraphPathCounter,
    capture_records: list[dict[str, Any]],
    layout: list[int],
    args,
) -> dict[str, Any]:
    vocab_size = engine.config.hf_config.vocab_size

    def builder(round_index: int) -> list[Sequence]:
        return [
            Sequence(
                make_random_prompt(
                    prompt_length,
                    seed=1000 + sum(layout) + round_index * 100 + request_index,
                    vocab_size=vocab_size,
                ),
                make_params(1),
            )
            for request_index, prompt_length in enumerate(layout)
        ]

    walls, stats = run_measurements(
        engine,
        counter,
        capture_records,
        builder,
        args.warmup_rounds,
        args.rounds,
    )
    stats["selected_buckets"] = selected_bucket_counts(engine.model_runner, stats["graph"])
    return {
        "layout": layout,
        "total_tokens": sum(layout),
        "requests": len(layout),
        "runs": [
            {
                "wall_s": round(wall, 4),
                "tokens_per_s": round(sum(layout) / wall, 1),
            }
            for wall in walls
        ],
        **stats,
    }


def run_mixed_once(engine, round_index: int, args) -> tuple[float, float]:
    vocab_size = engine.config.hf_config.vocab_size
    decode_sequences = [
        Sequence(
            make_random_prompt(
                args.decode_prompt_tokens,
                seed=8000 + round_index * 100 + request_index,
                vocab_size=vocab_size,
            ),
            make_params(args.decode_tokens),
        )
        for request_index in range(args.mixed_decode_requests)
    ]
    for sequence in decode_sequences:
        engine.scheduler.add(sequence)

    steps = 0
    while not any(sequence.num_completion_tokens for sequence in decode_sequences):
        engine.step()
        steps += 1
        if steps > args.mixed_decode_requests * args.decode_tokens + 20:
            raise RuntimeError("decode requests did not produce their first token")

    prefill_sequence = Sequence(
        make_random_prompt(
            args.mixed_prefill_tokens,
            seed=9000 + round_index,
            vocab_size=vocab_size,
        ),
        make_params(1),
    )
    engine.scheduler.add(prefill_sequence)
    torch.cuda.synchronize()
    started = time.perf_counter()
    ttft = None
    steps = 0
    while not engine.is_finished():
        completions_before = prefill_sequence.num_completion_tokens
        engine.step()
        steps += 1
        now = time.perf_counter()
        if completions_before == 0 and prefill_sequence.num_completion_tokens:
            ttft = now - started
        if steps > 4 * args.decode_tokens * args.mixed_decode_requests + 100:
            raise RuntimeError("mixed workload did not drain")
    wall = time.perf_counter() - started
    remove_finished_requests(engine)
    return wall, ttft if ttft is not None else float("nan")


def workload_mixed(
    engine,
    counter: GraphPathCounter,
    capture_records: list[dict[str, Any]],
    args,
) -> dict[str, Any]:
    for round_index in range(args.mixed_warmup_rounds):
        reset_seed(round_index)
        run_mixed_once(engine, round_index, args)

    counter_before = counter.as_dict()
    capture_before = len(capture_records)
    runs: list[dict[str, Any]] = []
    for round_index in range(args.mixed_rounds):
        reset_seed(args.mixed_warmup_rounds + round_index)
        wall, ttft = run_mixed_once(
            engine,
            args.mixed_warmup_rounds + round_index,
            args,
        )
        output_tokens = args.mixed_decode_requests * (args.decode_tokens - 1) + 1
        runs.append({
            "wall_s": round(wall, 4),
            "ttft_s": round(ttft, 4),
            "output_tokens_per_s": round(output_tokens / wall, 1),
        })

    graph = counter_delta(counter_before, counter.as_dict())
    return {
        "layout": [args.mixed_prefill_tokens]
        + [args.decode_prompt_tokens] * args.mixed_decode_requests,
        "runs": runs,
        "graph": graph,
        "capture": capture_summary(capture_records[capture_before:]),
        "selected_buckets": selected_bucket_counts(engine.model_runner, graph),
    }


def run_child(args) -> dict[str, Any]:
    mode: ModeConfig = PERF_MODE_NAMES[args.mode]
    print(f"[{mode.name}] building engine...", flush=True)
    engine = make_engine(
        args.model,
        mode,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    engine.model_runner.prefill_graph_sizes = list(args.prefill_graph_buckets)
    memory = {"after_init": memory_snapshot()}
    torch.cuda.reset_peak_memory_stats()
    capture_records = instrument_capture(engine.model_runner)
    workloads: dict[str, Any] = {}
    with GraphPathCounter(engine.model_runner) as counter:
        if args.suite == "full":
            for name, layout in PURE_LAYOUTS:
                print(f"[{mode.name}] {name}...", flush=True)
                workloads[name] = workload_pure(
                    engine,
                    counter,
                    capture_records,
                    layout,
                    args,
                )
            memory["after_pure"] = memory_snapshot()
            torch.cuda.reset_peak_memory_stats()
            if args.disable_piecewise_before_mixed:
                engine.model_runner.use_prefill_cudagraph = False
        print(f"[{mode.name}] mixed_late_prefill...", flush=True)
        workloads["mixed_late_prefill"] = workload_mixed(
            engine,
            counter,
            capture_records,
            args,
        )
        total_graph = counter.as_dict()

    memory["after_mixed"] = memory_snapshot()
    engine.exit()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return {
        "meta": {**env_info(), "mode": mode.name, "model": args.model},
        "config": {
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "warmup_rounds": args.warmup_rounds,
            "rounds": args.rounds,
            "mixed_warmup_rounds": args.mixed_warmup_rounds,
            "mixed_rounds": args.mixed_rounds,
            "suite": args.suite,
            "disable_piecewise_before_mixed": args.disable_piecewise_before_mixed,
            "prefill_graph_buckets": args.prefill_graph_buckets,
        },
        "workloads": workloads,
        "graph_stats": total_graph,
        "capture_stats": capture_summary(capture_records),
        "memory": memory,
    }


def print_parent(results: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 112)
    print("PIECEWISE PATH SUMMARY (median; speedup = P2 wall / P3 wall)")
    print("=" * 112)
    workload_names = list(results["P2"]["workloads"])
    for name in workload_names:
        p2_runs = results["P2"]["workloads"][name]["runs"]
        p3_runs = results["P3"]["workloads"][name]["runs"]
        p2_wall = median(run["wall_s"] for run in p2_runs)
        p3_wall = median(run["wall_s"] for run in p3_runs)
        if name == "mixed_late_prefill":
            p2_rate = median(run["output_tokens_per_s"] for run in p2_runs)
            p3_rate = median(run["output_tokens_per_s"] for run in p3_runs)
        else:
            p2_rate = median(run["tokens_per_s"] for run in p2_runs)
            p3_rate = median(run["tokens_per_s"] for run in p3_runs)
        speedup = p2_wall / p3_wall
        p3_graph = results["P3"]["workloads"][name]["graph"]
        print(
            f"{name:24s} P2={p2_rate:9.1f} tok/s P3={p3_rate:9.1f} tok/s "
            f"speedup={speedup:6.3f} hits={p3_graph.get('prefill_piecewise_hits', 0):3d} "
            f"mixed_fb={p3_graph.get('prefill_piecewise_mixed_fallbacks', 0):3d} "
            f"large_fb={p3_graph.get('prefill_piecewise_large_fallbacks', 0):3d}"
        )

    capture = results["P3"]["capture_stats"]
    print(
        f"\nP3 lazy captures: calls={capture['captures']} "
        f"total={capture['total_ms']:.1f}ms median={capture['median_ms']:.3f}ms"
    )
    for bucket, values in capture["buckets"].items():
        print(
            f"  bucket={bucket:>3s}: captures={values['captures']:3d} "
                f"total={values['total_ms']:9.3f}ms median={values['median_ms']:8.3f}ms"
            )

    print("\nCUDA MEMORY")
    for mode_name in ("P2", "P3"):
        memory = results[mode_name]["memory"]
        snapshots = ", ".join(
            f"{label}: alloc={values['allocated_mib']:.1f}MiB "
            f"reserved={values['reserved_mib']:.1f}MiB "
            f"peak={values['max_allocated_mib']:.1f}MiB "
            f"retries={values['num_alloc_retries']} "
            f"ooms={values['num_ooms']} "
            f"allocs={values['allocation_count']}"
            for label, values in memory.items()
        )
        print(f"  {mode_name}: {snapshots}")


def run_parent(args) -> None:
    results: dict[str, dict[str, Any]] = {}
    passthrough = [
        "--model", args.model,
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--warmup-rounds", str(args.warmup_rounds),
        "--rounds", str(args.rounds),
        "--mixed-warmup-rounds", str(args.mixed_warmup_rounds),
        "--mixed-rounds", str(args.mixed_rounds),
        "--decode-prompt-tokens", str(args.decode_prompt_tokens),
        "--decode-tokens", str(args.decode_tokens),
        "--mixed-decode-requests", str(args.mixed_decode_requests),
        "--mixed-prefill-tokens", str(args.mixed_prefill_tokens),
        "--suite", args.suite,
        "--prefill-graph-buckets", ",".join(map(str, args.prefill_graph_buckets)),
    ]
    if args.disable_piecewise_before_mixed:
        passthrough.append("--disable-piecewise-before-mixed")
    mode_names = ("P2", "P3") if args.order == "p2-first" else ("P3", "P2")
    for mode_name in mode_names:
        child_path = f"{args.json_out}.{mode_name}.json"
        completed = subprocess.run(
            [
                sys.executable,
                os.path.abspath(__file__),
                *passthrough,
                "--mode", mode_name,
                "--json-out", child_path,
            ],
            cwd=os.getcwd(),
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{mode_name} child failed with exit code {completed.returncode}"
            )
        with open(child_path, encoding="utf-8") as output_file:
            results[mode_name] = json.load(output_file)

    aggregate_path = save_json(args.json_out, {
        "meta": env_info(),
        "modes": results,
    })
    print_parent(results)
    print(f"\naggregate JSON: {aggregate_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("NANO_QWEN_MODEL", "/home/wei/code/models/qwen"))
    parser.add_argument("--mode", choices=("parent", "P2", "P3"), default="parent")
    parser.add_argument("--suite", choices=("full", "mixed"), default="full")
    parser.add_argument("--order", choices=("p2-first", "p3-first"), default="p2-first")
    parser.add_argument("--disable-piecewise-before-mixed", action="store_true")
    parser.add_argument(
        "--prefill-graph-buckets",
        type=lambda value: [int(item) for item in value.split(",")],
        default=DEFAULT_PREFILL_GRAPH_BUCKETS,
        help="Comma-separated active prefill graph buckets",
    )
    parser.add_argument("--json-out", default="")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--mixed-warmup-rounds", type=int, default=2)
    parser.add_argument("--mixed-rounds", type=int, default=5)
    parser.add_argument("--decode-prompt-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--mixed-decode-requests", type=int, default=7)
    parser.add_argument("--mixed-prefill-tokens", type=int, default=448)
    args = parser.parse_args()
    if not args.json_out:
        args.json_out = repo_log_path("piecewise_prefill_paths", subdir="bench")
    if args.mixed_decode_requests >= args.max_num_seqs:
        raise SystemExit(
            "--mixed-decode-requests must be less than --max-num-seqs "
            "to leave one slot for the late prefill request"
        )
    if not args.prefill_graph_buckets or any(
        size <= 0 for size in args.prefill_graph_buckets
    ):
        raise SystemExit("--prefill-graph-buckets must contain positive bucket sizes")
    if args.prefill_graph_buckets != sorted(set(args.prefill_graph_buckets)):
        raise SystemExit("--prefill-graph-buckets must be unique and ascending")
    return args


def main() -> None:
    args = parse_args()
    validate_allocator_for_cuda_graphs()
    if args.mode == "parent":
        run_parent(args)
        return

    result = run_child(args)
    output_path = save_json(args.json_out, result)
    print(f"[{args.mode}] results -> {output_path}")


if __name__ == "__main__":
    main()
