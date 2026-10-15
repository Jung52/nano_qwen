"""Strict real-model B2 gates and two-repeat off/reuse production timing."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench_mtp_small_batch import completion_check, make_cases  # noqa: E402
from bench_qwen3_5_mtp_production import release  # noqa: E402
from nano_qwen.engine.llm_engine import LLMEngine  # noqa: E402
from nano_qwen.engine.sequence import Sequence  # noqa: E402
from nano_qwen.sampling_params import SamplingParams  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


@torch.inference_mode()
def run_batch(engine, prompts, output):
    if not engine.is_finished() or engine.model_runner.input_batch.seq_id_to_slot:
        raise ValueError("Benchmark engine must be idle")
    seqs = [
        Sequence(
            p, SamplingParams(temperature=1e-6, max_tokens=output, ignore_eos=True)
        )
        for p in prompts
    ]
    stats = dict(engine.model_runner.mtp_stats)
    first, ended, steps = {}, {}, []
    start = None
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    request_start = perf_counter()
    try:
        for seq in seqs:
            engine.scheduler.add(seq)
        while not all(s.is_finished for s in seqs):
            before = sum(s.num_completion_tokens for s in seqs)
            step_start = perf_counter()
            engine.step()
            now = perf_counter()
            for i, seq in enumerate(seqs):
                if seq.num_completion_tokens:
                    first.setdefault(i, now)
                if seq.is_finished:
                    ended.setdefault(i, now)
            if len(first) == len(seqs) and start is None:
                start = now
            elif start is not None:
                steps.append(
                    dict(
                        elapsed_ms=1000 * (now - step_start),
                        committed=sum(s.num_completion_tokens for s in seqs) - before,
                    )
                )
        elapsed = perf_counter() - start
        completions = [s.completion_token_ids for s in seqs]
        tokens = sum(len(c) - 1 for c in completions)
        delta = {k: v - stats[k] for k, v in engine.model_runner.mtp_stats.items()}
        metrics = dict(
            tokens=tokens,
            elapsed_s=elapsed,
            tokens_per_s=tokens / elapsed,
            request_tpot_ms=[
                1000 * (ended[i] - first[i]) / (len(c) - 1)
                for i, c in enumerate(completions)
            ],
            request_ttft_ms=[
                1000 * (first[i] - request_start) for i in range(len(seqs))
            ],
            e2e_s=perf_counter() - request_start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            steps=steps,
            **delta,
        )
        return metrics, completions
    finally:
        for seq in seqs:
            release(engine, seq)


def measure(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    cases = make_cases(tokenizer)
    prompt = {c["name"]: c["prompt"] for c in cases}
    batches = [
        dict(name="same64", prompts=[prompt["synthetic-64-128"]] * 2),
        dict(
            name="unequal64-256",
            prompts=[prompt["synthetic-64-128"], prompt["synthetic-256-128"]],
        ),
        dict(name="chinese-code", prompts=[prompt["chinese"], prompt["code"]]),
    ]
    report = dict(
        model=args.model,
        mtp_model=args.mtp_model,
        batch=2,
        output=128,
        prefill_graph=False,
        repeats=2,
        cases=batches,
        gates=[],
        samples=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def engine(mode, eager):
        return LLMEngine(
            args.model,
            max_num_seqs=2,
            max_num_batched_tokens=1024,
            max_model_len=1024,
            gpu_memory_utilization=0.85,
            enforce_eager=eager,
            use_prefill_cudagraph=False,
            enable_mtp=mode != "off",
            mtp_model=args.mtp_model if mode != "off" else None,
            mtp_verify_mode=mode if mode != "off" else "serial",
        )

    expected = None
    # Full-output eager gate on unequal lengths; serial is the independent reference.
    for mode in ("off", "serial", "reuse"):
        value = engine(mode, True)
        try:
            _, completions = run_batch(value, batches[1]["prompts"], 128)
            expected = completions if expected is None else expected
            gate = dict(
                mode=mode,
                case=batches[1]["name"],
                completion=completions,
                strict_pass=all(
                    completion_check(a, b, 128)["strict_pass"]
                    for a, b in zip(expected, completions)
                ),
            )
            report["gates"].append(gate)
            save()
            print("eager", mode, "strict", gate["strict_pass"], flush=True)
            if not gate["strict_pass"]:
                raise RuntimeError(
                    "B2 eager completion gate failed; no validated performance"
                )
        finally:
            value.exit()

    for mode in ("off", "reuse"):
        value = engine(mode, False)
        try:
            for case in batches:
                run_batch(value, case["prompts"], 4)
                for repeat in (1, 2):
                    metrics, completions = run_batch(value, case["prompts"], 128)
                    row = dict(
                        mode=mode,
                        case=case["name"],
                        repeat=repeat,
                        metrics=metrics,
                        completion=completions,
                    )
                    if mode == "reuse":
                        ref = next(
                            s
                            for s in report["samples"]
                            if s["mode"] == "off"
                            and s["case"] == case["name"]
                            and s["repeat"] == repeat
                        )
                        row["requests"] = [
                            completion_check(a, b, 128)
                            for a, b in zip(ref["completion"], completions)
                        ]
                        row["strict_pass"] = all(
                            r["strict_pass"] for r in row["requests"]
                        )
                    report["samples"].append(row)
                    save()
                    print(
                        mode,
                        case["name"],
                        repeat,
                        metrics["tokens_per_s"],
                        row.get("strict_pass"),
                        flush=True,
                    )
        finally:
            value.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mtp-model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    measure(parser.parse_args())


if __name__ == "__main__":
    main()
