"""Terminal chat for main's single-buffer CUDA Graph decode pipeline.

Run from the repository root:
    PYTHONPATH=src python -m nano_qwen --model /path/to/qwen3.5
"""
from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path
import sys
from time import perf_counter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="nano_qwen: interactive streaming chat")
    parser.add_argument("--model", required=True, help="Local BF16 or converted FP8 checkpoint")
    parser.add_argument("--prompt", help="Generate once and exit; omit for interactive chat")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 selects the engine's deterministic greedy path")
    parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction,
                        default=True, help="One-batch decode lookahead (default: enabled)")
    parser.add_argument("--eager", action="store_true", help="Use eager execution for debugging")
    parser.add_argument("--prefill-cudagraph", action=argparse.BooleanOptionalAction,
                        default=False, help="Also capture piecewise prefill graphs")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction,
                        default=False, help="Pass enable_thinking to the chat template")
    parser.add_argument("--raw", action="store_true",
                        help="With --prompt, bypass the chat template")
    parser.add_argument("--stats", action=argparse.BooleanOptionalAction, default=True,
                        help="Print per-turn token counts, latency and Graph hits to stderr")
    parser.add_argument("--profile-steps", type=int, default=0,
                        help="Profile the first N tokens of every turn with torch.profiler; "
                             "export labelled traces for offline analysis")
    parser.add_argument("--profile-dir", default="logs/chat_trace",
                        help="Directory for per-turn raw traces (--profile-steps)")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("max_tokens", "max_model_len", "max_num_batched_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if not 0 <= args.temperature < float("inf"):
        parser.error("--temperature must be finite and nonnegative")
    if args.profile_steps < 0:
        parser.error("--profile-steps must be >= 0")
    if args.raw and args.prompt is None:
        parser.error("--raw requires --prompt")
    if not Path(args.model).is_dir():
        parser.error(f"Model directory does not exist: {args.model}")


def _load_runtime():
    # Keep --help available without initializing/importing the CUDA stack.
    import torch
    from transformers import TextStreamer
    from nano_qwen.engine.llm_engine import LLMEngine
    from nano_qwen.sampling_params import SamplingParams
    return torch, TextStreamer, LLMEngine, SamplingParams


def _prompt_ids(tokenizer, messages: list[dict[str, str]], args) -> list[int]:
    if args.raw:
        return tokenizer.encode(messages[-1]["content"])
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=args.thinking,
    )
    # transformers 5.x returns a BatchEncoding instead of list[int]; older
    # tokenizers and test doubles return the flat id list directly.
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def _stream_turn(engine, messages, args, torch, streamer_type, sampling_type,
                 turn_no: int = 1) -> str:
    ids = _prompt_ids(engine.tokenizer, messages, args)
    # Validate before creating a streamer: a long conversation can be reset
    # without touching GPU state or printing part of an assistant response.
    if not ids:
        raise ValueError("The prompt template produced an empty input")
    if len(ids) + args.max_tokens > engine.config.max_model_len:
        raise ValueError(
            f"Context budget exceeded: {len(ids)} input + {args.max_tokens} output "
            f"> {engine.config.max_model_len}. Use /reset or reduce --max-tokens."
        )
    params = sampling_type(
        temperature=1e-6 if args.temperature == 0 else args.temperature,
        max_tokens=args.max_tokens, ignore_eos=False,
    )
    # Buffer incomplete byte/word pieces via the tokenizer's supported
    # streamer; decoding each token independently corrupts multibyte text.
    streamer = streamer_type(
        engine.tokenizer, skip_prompt=False, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    graph_manager = engine.model_runner.cuda_graphs
    before = graph_manager.dispatch_stats().get("decode", {})
    output = []
    started = perf_counter()
    first = last = None
    reason = None
    session = None
    profiled_tokens = 0
    if args.profile_steps > 0:
        from .utils.profiling import ProfileSession
        session = ProfileSession(torch, engine, args.profile_dir, turn_no).start()
    try:
        with closing(engine.generate_stream(ids, params)) as chunks:
            for chunk in chunks:
                if chunk.token_ids:
                    last = perf_counter()
                    if first is None:
                        first = last
                    output.extend(chunk.token_ids)
                    streamer.put(torch.tensor(chunk.token_ids, dtype=torch.int64, device="cpu"))
                    if session is not None and session.active:
                        profiled_tokens += len(chunk.token_ids)
                        if profiled_tokens >= args.profile_steps:
                            session.stop()
                if chunk.finished:
                    reason = chunk.finish_reason
    finally:
        try:
            streamer.end()
        finally:
            if session is not None:
                trace_line = session.finish()
                if trace_line:
                    print(trace_line, file=sys.stderr, flush=True)

    ended = perf_counter()
    if args.stats and output:
        decode_seconds = 0 if first is None or last is None else last - first
        decode_rate = (len(output) - 1) / decode_seconds if decode_seconds > 0 else 0.0
        after = graph_manager.dispatch_stats().get("decode", {})
        graph = after.get("graph_calls", 0) - before.get("graph_calls", 0)
        eager = after.get("eager_calls", 0) - before.get("eager_calls", 0)
        print(
            f"[tokens={len(output)} | first-token={(first-started)*1000:.1f} ms | "
            f"decode={decode_rate:.1f} tok/s | total={ended-started:.3f} s | "
            f"decode graph/eager={graph}/{eager} | stop={reason}]",
            file=sys.stderr, flush=True,
        )
    return engine.tokenizer.decode(
        output, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        torch, streamer_type, engine_type, sampling_type = _load_runtime()
    except ImportError as exc:
        print(f"Missing inference dependency: {exc}. Use the repository's inference environment.",
              file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("A CUDA GPU is required.", file=sys.stderr)
        return 1

    engine = None
    try:
        if args.profile_steps > 0:
            # Initialize Kineto before the engine captures decode CUDA Graphs.
            from .utils.profiling import init_kineto
            try:
                init_kineto(torch)
            except Exception as exc:
                print(f"[trace] capture disabled: {exc!r}", file=sys.stderr)
                args.profile_steps = 0
        print(f"Loading {args.model} ...", file=sys.stderr, flush=True)
        with torch.inference_mode():
            engine = engine_type(
                args.model, tensor_parallel_size=1, max_num_seqs=1,
                max_model_len=args.max_model_len,
                max_num_batched_tokens=args.max_num_batched_tokens,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=args.eager, async_scheduling=args.async_scheduling,
                use_prefill_cudagraph=args.prefill_cudagraph,
            )
        if args.async_scheduling and not (
            engine.config.async_scheduling and engine.max_concurrent_batches == 2
            and engine.model_runner.async_output
        ):
            raise RuntimeError("The engine did not enable the requested async decode pipeline")
        graphs = engine.model_runner.cuda_graphs.decode_graphs
        if not args.eager and set(graphs) != {1}:
            raise RuntimeError("Expected the current main's single B1 decode CUDA Graph")
        print(
            f"Ready: B1 | decode={'eager' if args.eager else 'single CUDA Graph'} | "
            f"async={'on' if args.async_scheduling else 'off'} | "
            f"prefill={'piecewise Graph' if args.prefill_cudagraph and not args.eager else 'eager'}",
            file=sys.stderr, flush=True,
        )
        initial = [] if args.system_prompt is None else [
            {"role": "system", "content": args.system_prompt}
        ]
        messages = initial.copy()
        if args.prompt is not None:
            messages.append({"role": "user", "content": args.prompt})
            _stream_turn(engine, messages, args, torch, streamer_type, sampling_type)
            return 0

        print("Commands: /reset, /exit. Ctrl+C exits. Context is re-prefilled each turn.",
              file=sys.stderr, flush=True)
        turn_no = 0
        while True:
            try:
                prompt = input("\n你> ").strip()
            except EOFError:
                print()
                break
            if prompt in ("/exit", "/quit"):
                break
            if prompt == "/reset":
                messages = initial.copy()
                print("对话已清空。", flush=True)
                continue
            if not prompt:
                continue
            candidate = messages + [{"role": "user", "content": prompt}]
            print("助手> ", end="", flush=True)
            turn_no += 1
            try:
                answer = _stream_turn(engine, candidate, args, torch, streamer_type,
                                      sampling_type, turn_no)
            except ValueError as exc:
                print(f"\n{exc}", file=sys.stderr, flush=True)
                continue
            messages = candidate + [{"role": "assistant", "content": answer}]
        return 0
    except KeyboardInterrupt:
        print("\n已退出。", file=sys.stderr, flush=True)
        return 130
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        if engine is not None:
            engine.exit()


if __name__ == "__main__":
    raise SystemExit(main())
