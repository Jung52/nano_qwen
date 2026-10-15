"""Strict B1 workload baseline before expanding production MTP batches."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench_qwen3_5_mtp_production import run_mode  # noqa: E402
from nano_qwen.engine.llm_engine import LLMEngine  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def completion_check(expected, actual, count):
    mismatch = next(
        (i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), None
    )
    if mismatch is None and len(expected) != len(actual):
        mismatch = min(len(expected), len(actual))
    return {
        "strict_pass": mismatch is None and len(actual) == len(expected) == count,
        "first_difference": mismatch,
    }


def make_cases(tokenizer):
    seed = tokenizer.encode(
        "Qwen3.5 MTP2 production throughput verification.", add_special_tokens=False
    )
    cases = [
        dict(
            name=f"synthetic-{n}-{o}",
            prompt=(seed * ((n + len(seed) - 1) // len(seed)))[:n],
            output=o,
        )
        for n in (64, 256, 1024)
        for o in (128, 512)
    ]
    for name, text in (
        (
            "chinese",
            "请解释大语言模型推理中 prefill 和 decode 的区别，并以一个中文聊天请求为例，说明延迟和吞吐为什么会随 batch 大小变化。",
        ),
        (
            "code",
            "Write a Python function that merges overlapping intervals. Include type hints, explain the algorithm, and show tests for empty input, touching intervals, and nested intervals.",
        ),
    ):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        cases.append(dict(name=name, prompt=prompt, output=128))
    return cases


def matrix(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    cases = make_cases(tokenizer)
    report = {
        "model": args.model,
        "mtp_model": args.mtp_model,
        "batch": 1,
        "prefill_graph": False,
        "decode_graph": True,
        "repeats": 2,
        "cases": cases,
        "samples": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    for mode in ("off", "reuse"):
        with_engine = LLMEngine(
            args.model,
            max_num_seqs=1,
            max_num_batched_tokens=1024,
            max_model_len=1540,
            gpu_memory_utilization=0.85,
            enforce_eager=False,
            use_prefill_cudagraph=False,
            enable_mtp=mode != "off",
            mtp_model=args.mtp_model if mode != "off" else None,
            mtp_verify_mode=mode if mode != "off" else "serial",
        )
        try:
            for case in cases:
                common = dict(
                    model_path=args.model,
                    enable_mtp=mode != "off",
                    prompt=case["prompt"],
                    duration=3600,
                    enforce_eager=False,
                    mtp_model=args.mtp_model if mode != "off" else None,
                    mtp_verify_mode=mode if mode != "off" else "serial",
                    engine=with_engine,
                )
                # Short unmeasured warmup compiles each prefill shape and executes verify.
                run_mode(**common, max_tokens=4)
                for repeat in (1, 2):
                    metrics, tokens = run_mode(**common, max_tokens=case["output"])
                    row = dict(
                        case=case["name"],
                        input=len(case["prompt"]),
                        output=case["output"],
                        mode=mode,
                        repeat=repeat,
                        metrics=metrics,
                        completion=tokens,
                    )
                    if mode == "reuse":
                        ref = next(
                            s
                            for s in report["samples"]
                            if s["case"] == case["name"]
                            and s["mode"] == "off"
                            and s["repeat"] == repeat
                        )
                        row.update(
                            completion_check(ref["completion"], tokens, case["output"])
                        )
                    report["samples"].append(row)
                    save()
                    print(
                        f"{mode} {case['name']} repeat={repeat} tok/s={metrics['tokens_per_s']:.2f} strict={row.get('strict_pass')}",
                        flush=True,
                    )
        finally:
            with_engine.exit()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mtp-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix(args)


if __name__ == "__main__":
    main()
