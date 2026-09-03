"""Benchmark the async output/batch pipeline.

Run in WSL with the ``triton`` environment, for example::

    PYTHONPATH=src python -m nano_qwen.server.bench_async \
        --model /mnt/d/nano-vllm/Qwen3.5-0.8B

The model is created once. Each (d2h mode, queue depth) configuration is
warmed up, then stressed for --duration seconds while finished requests are
immediately replaced; the speedup compares sustained tokens-per-second.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

import torch

from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.sampling_params import SamplingParams
from nano_qwen.utils.trace import enable_tracing, save_trace, trace_event


DEFAULT_MODEL = os.environ.get("NANO_QWEN_MODEL", "/mnt/d/nano-vllm/Qwen3.5-0.8B")


def default_trace_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    log_dir = repo_root / "logs" / "traces"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"bench_async_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def run_once(engine: LLMEngine, prompts: list[str], params: SamplingParams) -> tuple[float, int, list[dict]]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = engine.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, sum(len(item["token_ids"]) for item in outputs), outputs


def stress_once(
    engine: LLMEngine,
    prompts: list[str],
    params: SamplingParams,
    duration: float,
) -> tuple[float, int, int, int]:
    """Keep requests continuously in flight for ``duration`` seconds.

    Finished requests are immediately replaced from ``prompts`` so the batch
    stays saturated. Returns (elapsed, processed_tokens, output_tokens,
    finished_requests) for the timed window only; the in-flight tail is then
    drained untimed so the next config starts from an idle engine.
    """
    torch.cuda.synchronize()
    for prompt in prompts:
        engine.add_request(prompt, params)
    start = time.perf_counter()
    deadline = start + duration
    processed = 0
    output_tokens = 0
    finished = 0
    next_prompt = 0
    while time.perf_counter() < deadline:
        outputs, num_tokens = engine.step()
        processed += abs(num_tokens)  # prefill tokens + decode batch size
        for _, token_ids in outputs:
            output_tokens += len(token_ids)
            finished += 1
            engine.add_request(prompts[next_prompt % len(prompts)], params)
            next_prompt += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    while not engine.is_finished():
        engine.step()
    return elapsed, processed, output_tokens, finished


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
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="stress duration per (d2h mode, depth) config, in seconds",
    )
    parser.add_argument("--depths", default="1,2", help="comma-separated queue depths")
    parser.add_argument(
        "--d2h-modes",
        default="sync,async",
        help="comma-separated D2H modes (sync, async)",
    )
    parser.add_argument("--trace", action="store_true", help="print dispatched batch sizes")
    parser.add_argument(
        "--trace-out",
        default="",
        help="path for the Perfetto-compatible JSON trace "
        "(default: <repo>/logs/traces/bench_async_<timestamp>.json)",
    )
    args = parser.parse_args()
    enable_tracing()

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
        gpu_memory_utilization=0.90,
    )
    try:
        print(f"model={args.model}")
        print(f"requests={args.num_seqs} max_num_seqs={args.max_num_seqs} max_tokens={args.max_tokens}")
        print(f"depths={depths} d2h_modes={d2h_modes} duration_s={args.duration:g}")
        results: dict[tuple[str, int], float] = {}
        trace = install_trace(engine) if args.trace else None

        for mode in d2h_modes:
            engine.model_runner.async_output = mode == "async"
            for depth in depths:
                engine.max_concurrent_batches = depth
                # Warmup also catches invalid configurations before timing.
                with trace_event("warmup", "bench", {"d2h": mode, "depth": depth}):
                    run_once(engine, prompts, params)
                if trace is not None:
                    trace.clear()
                with trace_event("stress", "bench", {"d2h": mode, "depth": depth}):
                    elapsed, processed, output_tokens, finished = stress_once(
                        engine, prompts, params, args.duration
                    )
                tok_per_s = processed / elapsed
                results[(mode, depth)] = tok_per_s
                print(
                    f"d2h={mode} depth={depth} duration_s={elapsed:.1f} "
                    f"processed_tokens={processed} output_tokens={output_tokens} "
                    f"requests={finished} tok_per_s={tok_per_s:.1f}"
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
            sync_tps = results.get(("sync", depth))
            async_tps = results.get(("async", depth))
            if sync_tps is not None and async_tps is not None:
                speedup = async_tps / sync_tps
                print(
                    f"speedup_async_vs_sync_depth{depth}={speedup:.3f} "
                    f"({(speedup - 1) * 100:.1f}%)"
                )
    finally:
        engine.exit()
        trace_path = args.trace_out or default_trace_path()
        saved = save_trace(str(trace_path))
        if saved:
            print(f"trace saved: {saved}")


if __name__ == "__main__":
    main()
