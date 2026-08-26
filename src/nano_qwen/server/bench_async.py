"""Benchmark the async output/batch pipeline.

Run in WSL with the ``triton`` environment, for example::

    PYTHONPATH=src python -m nano_qwen.server.bench_async \
        --model /mnt/d/nano-vllm/Qwen3.5-0.8B

The model is created once. Each depth is warmed up before timing, so model
loading and Triton compilation are excluded from the reported numbers.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from functools import wraps

import torch

from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.sampling_params import SamplingParams


DEFAULT_MODEL = os.environ.get("NANO_QWEN_MODEL", "/mnt/d/nano-vllm/Qwen3.5-0.8B")


def run_once(engine: LLMEngine, prompts: list[str], params: SamplingParams) -> tuple[float, int, list[dict]]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = engine.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, sum(len(item["token_ids"]) for item in outputs), outputs


def install_trace(engine: LLMEngine) -> list[tuple[int, bool]]:
    """Trace dispatched batch sizes without changing ModelRunner behavior."""
    calls: list[tuple[int, bool]] = []
    runner = engine.model_runner
    original = runner.execute_model

    @wraps(original)
    def traced(seqs, is_prefill):
        calls.append((len(seqs), is_prefill))
        return original(seqs, is_prefill)

    runner.execute_model = traced
    return calls


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark nano-qwen async pipeline")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--depths", default="1,2", help="comma-separated queue depths")
    parser.add_argument(
        "--d2h-modes",
        default="sync,async",
        help="comma-separated D2H modes (sync, async)",
    )
    parser.add_argument("--trace", action="store_true", help="print dispatched batch sizes")
    args = parser.parse_args()

    prompts = [f"请简要介绍第 {i} 个主题。" for i in range(args.num_seqs)]
    params = SamplingParams(temperature=0.7, max_tokens=args.max_tokens)
    depths = [int(value) for value in args.depths.split(",")]
    d2h_modes = [value.strip() for value in args.d2h_modes.split(",")]
    invalid_modes = set(d2h_modes) - {"sync", "async"}
    if invalid_modes:
        parser.error(f"invalid --d2h-modes: {sorted(invalid_modes)}")

    engine = LLMEngine(
        args.model,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.50,
    )
    try:
        print(f"model={args.model}")
        print(f"requests={args.num_seqs} max_num_seqs={args.max_num_seqs} max_tokens={args.max_tokens}")
        print(f"depths={depths} d2h_modes={d2h_modes} repeats={args.repeats}")
        results: dict[tuple[str, int], float] = {}
        trace = install_trace(engine) if args.trace else None

        for mode in d2h_modes:
            engine.model_runner.async_output = mode == "async"
            for depth in depths:
                engine.max_concurrent_batches = depth
                # Warmup also catches invalid configurations before timing.
                run_once(engine, prompts, params)
                times: list[float] = []
                token_counts: list[int] = []
                if trace is not None:
                    trace.clear()
                for _ in range(args.repeats):
                    elapsed, tokens, _ = run_once(engine, prompts, params)
                    times.append(elapsed)
                    token_counts.append(tokens)
                median = statistics.median(times)
                results[(mode, depth)] = median
                tokens = statistics.median(token_counts)
                print(
                    f"d2h={mode} depth={depth} median_s={median:.3f} "
                    f"tok_per_s={tokens / median:.1f} "
                    f"runs={[round(x, 3) for x in times]}"
                )
                if trace is not None:
                    prefill = sum(is_prefill for _, is_prefill in trace)
                    decode = len(trace) - prefill
                    batch_sizes = sorted(set(size for size, _ in trace))
                    prefill_sizes = [size for size, is_prefill in trace if is_prefill]
                    split_prefill = bool(prefill_sizes) and max(prefill_sizes) < args.num_seqs
                    print(
                        f"trace d2h={mode} depth={depth}: dispatches={len(trace)} "
                        f"prefill={prefill} decode={decode} "
                        f"batch_sizes={batch_sizes} split_prefill={split_prefill}"
                    )

        for depth in depths:
            sync = results.get(("sync", depth))
            async_time = results.get(("async", depth))
            if sync is not None and async_time is not None:
                speedup = sync / async_time
                print(
                    f"speedup_async_vs_sync_depth{depth}={speedup:.3f} "
                    f"({(speedup - 1) * 100:.1f}%)"
                )
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
