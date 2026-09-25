"""End-to-end Qwen3.5 MTP2 production benchmark.

Runs one baseline engine and one production MTP engine, counts only committed
completion tokens after prefill, and verifies that greedy outputs match. Use
``--stress`` for a full two-minute decode per mode with 30-second intervals.
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nano_qwen.engine.llm_engine import LLMEngine  # noqa: E402
from nano_qwen.engine.sequence import Sequence  # noqa: E402
from nano_qwen.sampling_params import SamplingParams  # noqa: E402


def make_prompt(model_path: str, prompt_tokens: int) -> list[int]:
    seed = AutoTokenizer.from_pretrained(model_path, use_fast=True).encode(
        "Qwen3.5 MTP2 production throughput verification.",
        add_special_tokens=False,
    )
    if not seed:
        raise RuntimeError("tokenizer returned an empty prompt")
    return (seed * ((prompt_tokens + len(seed) - 1) // len(seed)))[:prompt_tokens]


def release(engine, seq: Sequence) -> None:
    if seq.block_table:
        engine.scheduler.block_manager.deallocate(seq)
    engine.batch_queue.clear()
    engine.scheduler.waiting.clear()
    engine.scheduler.running.clear()
    engine.scheduler.in_flight.clear()
    engine.model_runner.input_batch.remove(seq.seq_id)


@torch.inference_mode()
def run_mode(model_path: str, enable_mtp: bool, prompt: list[int], max_tokens: int,
             duration: float, enforce_eager: bool,
             window_seconds: float | None = None) -> tuple[dict, list[int]]:
    engine = LLMEngine(
        model_path,
        tensor_parallel_size=1,
        max_num_seqs=1,
        max_num_batched_tokens=max(512, len(prompt) + 1),
        max_model_len=len(prompt) + max_tokens + 2,
        gpu_memory_utilization=0.85,
        enforce_eager=enforce_eager,
        use_prefill_cudagraph=False,
        enable_mtp=enable_mtp,
    )
    try:
        seq = Sequence(
            prompt,
            SamplingParams(
                temperature=1e-6,
                max_tokens=max_tokens,
                ignore_eos=True,
            ),
        )
        engine.scheduler.add(seq)
        torch.manual_seed(0)

        start = deadline = window_start = None
        window_start_tokens = 0
        window_start_accepted = 0
        window_start_verifies = 0
        windows = []

        def record_window(now: float) -> None:
            nonlocal window_start, window_start_tokens
            nonlocal window_start_accepted, window_start_verifies
            elapsed_window = now - window_start
            if elapsed_window <= 0:
                return
            tokens = max(0, seq.num_completion_tokens - 1)
            accepted = engine.model_runner.mtp_stats["accepted_tokens"] if enable_mtp else 0
            verifies = engine.model_runner.mtp_stats["verify_steps"] if enable_mtp else 0
            windows.append({
                "start_s": window_start - start,
                "end_s": now - start,
                "tokens": tokens - window_start_tokens,
                "tokens_per_s": (tokens - window_start_tokens) / elapsed_window,
                "accepted_tokens": accepted - window_start_accepted,
                "verify_steps": verifies - window_start_verifies,
            })
            window_start = now
            window_start_tokens = tokens
            window_start_accepted = accepted
            window_start_verifies = verifies

        while not seq.is_finished:
            _, scheduled_tokens = engine.step()
            if scheduled_tokens > 0:
                start = perf_counter()
                deadline = start + duration
                window_start = start
                continue
            now = perf_counter()
            if window_seconds is not None and now - window_start >= window_seconds:
                record_window(now)
            if now >= deadline:
                break

        if start is None:
            raise RuntimeError("benchmark ended before prefill completed")
        elapsed = perf_counter() - start
        if window_seconds is not None and elapsed - (window_start - start) >= 0.5:
            record_window(start + elapsed)
        token_ids = seq.completion_token_ids
        result = {
            "tokens": max(0, len(token_ids) - 1),
            "elapsed_s": elapsed,
            "tokens_per_s": max(0, len(token_ids) - 1) / elapsed,
            "stop_reason": "max_tokens" if seq.is_finished else "duration",
            "windows": windows,
        }
        if enable_mtp:
            result.update(engine.model_runner.mtp_stats)
        release(engine, seq)
        torch.cuda.synchronize()
        return result, token_ids
    finally:
        engine.exit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=129)
    parser.add_argument("--max-tokens", type=int, default=10240)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument(
        "--stress", action="store_true",
        help="run a 120-second decode per mode and report 30-second windows",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--allow-numerical-drift",
        action="store_true",
        help=(
            "continue after a greedy top-1 change caused by the different "
            "verify/decode kernel path; report the matching prefix instead"
        ),
    )
    args = parser.parse_args()
    if args.prompt_tokens < 2:
        parser.error("--prompt-tokens must be >= 2")
    if args.max_tokens < 2:
        parser.error("--max-tokens must be >= 2")
    if args.duration <= 0:
        parser.error("--duration must be > 0")

    if args.stress:
        args.duration = 120.0
        args.max_tokens = max(args.max_tokens, 32768)
    window_seconds = 30.0 if args.stress else None

    prompt = make_prompt(args.model, args.prompt_tokens)

    baseline, baseline_tokens = run_mode(
        args.model, False, prompt, args.max_tokens, args.duration,
        args.enforce_eager, window_seconds,
    )
    mtp, mtp_tokens = run_mode(
        args.model, True, prompt, args.max_tokens, args.duration,
        args.enforce_eager, window_seconds,
    )

    compare_len = min(len(baseline_tokens), len(mtp_tokens))
    mismatch = next(
        (
            i for i, (expected, actual) in enumerate(
                zip(baseline_tokens, mtp_tokens)
            ) if expected != actual
        ),
        None,
    )
    divergence = (
        f"token={mismatch} baseline={baseline_tokens[mismatch]} "
        f"mtp={mtp_tokens[mismatch]}"
        if mismatch is not None
        else "none"
    )
    print(
        f"BASELINE tokens={baseline['tokens']} elapsed={baseline['elapsed_s']:.3f}s "
        f"throughput={baseline['tokens_per_s']:.2f}tok/s "
        f"stop={baseline['stop_reason']}"
    )
    acceptance = mtp["accepted_tokens"] / max(mtp["verify_steps"], 1)
    print(
        f"MTP2 tokens={mtp['tokens']} elapsed={mtp['elapsed_s']:.3f}s "
        f"throughput={mtp['tokens_per_s']:.2f}tok/s "
        f"acceptance={acceptance:.4%} verify_steps={mtp['verify_steps']} "
        f"stop={mtp['stop_reason']}"
    )
    if args.stress:
        for index, (base_window, mtp_window) in enumerate(
            zip(baseline["windows"], mtp["windows"]), start=1
        ):
            window_acceptance = (
                mtp_window["accepted_tokens"] / max(mtp_window["verify_steps"], 1)
            )
            print(
                f"WINDOW {index} "
                f"baseline={base_window['tokens_per_s']:.2f}tok/s "
                f"mtp={mtp_window['tokens_per_s']:.2f}tok/s "
                f"speedup={mtp_window['tokens_per_s'] / base_window['tokens_per_s']:.3f}x "
                f"acceptance={window_acceptance:.4%} "
                f"tokens={base_window['tokens']}/{mtp_window['tokens']}"
            )
    print(f"MATCH prefix={compare_len if mismatch is None else mismatch} divergence={divergence}")
    if args.stress and mismatch is not None:
        print(
            "NOTE throughput after divergence compares different generated "
            "continuations; acceptance and speedup can depend on that path"
        )
    print(f"RESULT speedup={mtp['tokens_per_s'] / baseline['tokens_per_s']:.3f}x")
    if mismatch is not None and not (args.allow_numerical_drift or args.stress):
        raise AssertionError(
            f"MTP2 production output diverged at completion token {mismatch}: "
            f"baseline={baseline_tokens[mismatch]}, "
            f"mtp={mtp_tokens[mismatch]}; "
            f"lengths={len(baseline_tokens)}/{len(mtp_tokens)}. "
            "Use --allow-numerical-drift only for long throughput runs."
        )


if __name__ == "__main__":
    main()
