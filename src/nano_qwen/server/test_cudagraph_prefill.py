"""Compare prefill+decode CUDA Graphs with decode-only CUDA Graphs.

Run in WSL with the ``triton`` environment::

    PYTHONPATH=src python -m nano_qwen.server.test_cudagraph_prefill \
        --model /mnt/d/nano-vllm/Qwen3.5-0.8B

``all_graph`` uses the piecewise prefill graphs plus the existing decode
graphs. ``decode_only`` disables only prefill graphs and keeps decode graphs
enabled. The first request is reported separately because it includes lazy
prefill graph capture in ``all_graph`` mode.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict

import torch

from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.sampling_params import SamplingParams


DEFAULT_MODEL = os.environ.get("NANO_QWEN_MODEL", "/mnt/d/nano-vllm/Qwen3.5-0.8B")


def make_prompts(num_seqs: int, repeat: int) -> list[str]:
    base = "请分析人工智能的发展、应用和未来挑战。"
    return [base * repeat + f"问题编号{i}。" for i in range(num_seqs)]


def timed_generate(engine, prompts, params, phase_totals):
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = engine.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, sum(len(item["token_ids"]) for item in outputs)


def install_phase_trace(engine):
    """Measure run_model wall time while preserving the original call path."""
    totals = defaultdict(float)
    counts = defaultdict(int)
    runner = engine.model_runner
    original = runner.run_model

    def timed(input_ids, positions, is_prefill):
        label = "prefill" if is_prefill else "decode"
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original(input_ids, positions, is_prefill)
        torch.cuda.synchronize()
        totals[label] += time.perf_counter() - start
        counts[label] += 1
        return result

    runner.run_model = timed
    return totals, counts


def run_mode(args, mode, prompts, params):
    engine = LLMEngine(
        args.model,
        enforce_eager=False,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.50,
    )
    try:
        engine.model_runner.use_prefill_cudagraph = mode == "all_graph"
        totals, counts = install_phase_trace(engine)

        # Cold run includes the first piecewise prefill capture when enabled.
        cold_time, cold_tokens = timed_generate(engine, prompts, params, totals)

        totals.clear()
        counts.clear()
        steady_times = []
        steady_tokens = []
        for _ in range(args.repeats):
            elapsed, tokens = timed_generate(engine, prompts, params, totals)
            steady_times.append(elapsed)
            steady_tokens.append(tokens)

        steady_median = statistics.median(steady_times)
        phase_time = dict(totals)
        phase_counts = dict(counts)
        steady_token_median = statistics.median(steady_tokens)
        print(
            f"mode={mode} cold_s={cold_time:.3f} "
            f"steady_median_s={steady_median:.3f} "
            f"tok_per_s={steady_token_median / steady_median:.1f} "
            f"steady_runs={[round(x, 3) for x in steady_times]}"
        )
        print(
            f"phase mode={mode} prefill_s={phase_time.get('prefill', 0.0):.3f} "
            f"decode_s={phase_time.get('decode', 0.0):.3f} "
            f"prefill_calls={phase_counts.get('prefill', 0)} "
            f"decode_calls={phase_counts.get('decode', 0)}"
        )
        return {
            "mode": mode,
            "cold": cold_time,
            "steady": steady_median,
            "cold_tokens": cold_tokens,
            "steady_tokens": steady_token_median,
            "steady_runs": steady_times,
            "prefill_s": phase_time.get("prefill", 0.0),
            "decode_s": phase_time.get("decode", 0.0),
            "prefill_calls": phase_counts.get("prefill", 0),
            "decode_calls": phase_counts.get("decode", 0),
        }
    finally:
        engine.exit()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prefill CUDA Graph modes")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--prompt-repeat", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=("all_graph", "decode_only"),
        help="Run one mode in the current process; parent mode runs both in isolated subprocesses.",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Emit a RESULT_JSON line for the parent benchmark process.",
    )
    args = parser.parse_args()

    prompts = make_prompts(args.num_seqs, args.prompt_repeat)
    params = SamplingParams(temperature=0.7, max_tokens=args.max_tokens, ignore_eos=True)
    print(
        f"requests={args.num_seqs} max_num_seqs={args.max_num_seqs} "
        f"prompt_repeat={args.prompt_repeat} max_tokens={args.max_tokens} "
        f"repeats={args.repeats}"
    )

    if args.mode is not None:
        result = run_mode(args, args.mode, prompts, params)
        if args.json_output:
            print(f"RESULT_JSON={json.dumps(result, separators=(',', ':'))}")
        return

    # CUDA Graph pools and the model's KV cache can remain reserved until the
    # CUDA context is destroyed. Run each mode in a fresh process so the two
    # measurements do not compete for memory or inherit allocator state.
    results = {}
    passthrough = [
        "--model", args.model,
        "--num-seqs", str(args.num_seqs),
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--max-model-len", str(args.max_model_len),
        "--prompt-repeat", str(args.prompt_repeat),
        "--max-tokens", str(args.max_tokens),
        "--repeats", str(args.repeats),
        "--json-output",
    ]
    for mode in ("all_graph", "decode_only"):
        command = [
            sys.executable,
            "-m",
            "nano_qwen.server.test_cudagraph_prefill",
            *passthrough,
            "--mode",
            mode,
        ]
        completed = subprocess.run(
            command,
            cwd=os.getcwd(),
            text=True,
            capture_output=True,
            env=os.environ.copy(),
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                f"mode={mode} subprocess failed with exit code {completed.returncode}"
            )
        result_line = next(
            (line for line in completed.stdout.splitlines()
             if line.startswith("RESULT_JSON=")),
            None,
        )
        if result_line is None:
            raise RuntimeError(f"mode={mode} did not emit RESULT_JSON")
        results[mode] = json.loads(result_line.removeprefix("RESULT_JSON="))

    all_graph = results["all_graph"]
    decode_only = results["decode_only"]
    steady_speedup = decode_only["steady"] / all_graph["steady"]
    cold_speedup = decode_only["cold"] / all_graph["cold"]
    prefill_speedup = decode_only["prefill_s"] / all_graph["prefill_s"]
    decode_ratio = decode_only["decode_s"] / all_graph["decode_s"]
    print(
        f"steady_speedup_prefill_graph={steady_speedup:.3f} "
        f"({(steady_speedup - 1) * 100:.1f}%)"
    )
    print(
        f"cold_speedup_prefill_graph={cold_speedup:.3f} "
        f"({(cold_speedup - 1) * 100:.1f}%)"
    )
    print(
        f"prefill_phase_speedup={prefill_speedup:.3f} "
        f"({(prefill_speedup - 1) * 100:.1f}%) "
        f"decode_phase_ratio={decode_ratio:.3f}"
    )


if __name__ == "__main__":
    main()
