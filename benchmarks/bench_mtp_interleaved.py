"""Two-repeat interleaved MTP timing, with a controlled full-sampling reference."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench_mtp_batch2 import run_batch  # noqa: E402
from bench_mtp_small_batch import make_cases, completion_check  # noqa: E402
from nano_qwen.engine.llm_engine import LLMEngine  # noqa: E402
from nano_qwen.layers.sampler import Sampler  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class FullSamplingReference(Sampler):
    """The 7dc09b9 sampling algorithm, retaining softmax/RNG even for greedy."""

    def forward(self, logits, temperatures, all_greedy=False):
        return super().forward(logits, temperatures)


@contextmanager
def execution_mode(engine, enabled, sampler):
    if not engine.is_finished() or engine.model_runner.input_batch.seq_id_to_slot:
        raise ValueError("Mode switches require an idle engine")
    runner = engine.model_runner
    mtp, original_sampler = runner.mtp, runner.sampler
    original_enabled = engine.scheduler.enable_mtp
    runner.mtp = mtp if enabled else None
    runner.sampler = sampler
    engine.scheduler.enable_mtp = enabled
    try:
        yield
    finally:
        runner.mtp, runner.sampler = mtp, original_sampler
        engine.scheduler.enable_mtp = original_enabled


def measure(args):
    root = Path(__file__).resolve().parents[1]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    cases = {c["name"]: c["prompt"] for c in make_cases(tokenizer)}
    samplers = {"fast": Sampler(), "reference": FullSamplingReference()}
    files = [*root.joinpath("src/nano_qwen").rglob("*.py"), Path(__file__).resolve()]
    report = dict(model=args.model, mtp_model=args.mtp_model, output=args.output_tokens,
                  prefill_graph=False, decode_graph=True, repeats=2,
                  resident_engine=True, samples=[], cases={},
                  source=dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                              sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    for batch in args.batches:
        engine = LLMEngine(args.model, max_num_seqs=batch, max_num_batched_tokens=max(1024, batch * 64),
                           max_model_len=1024, gpu_memory_utilization=.85,
                           enforce_eager=False, use_prefill_cudagraph=False,
                           enable_mtp=True, mtp_model=args.mtp_model, mtp_verify_mode="reuse")
        try:
            for case in args.cases:
                seed = [cases["synthetic-64-128"]] if case == "same64" else [cases["chinese"], cases["code"]]
                prompts = [seed[i % len(seed)] for i in range(batch)]
                report["cases"][f"{batch}:{case}"] = prompts
                for variant in args.samplers:
                    for mode in ("off", "reuse"):
                        with execution_mode(engine, mode == "reuse", samplers[variant]):
                            run_batch(engine, prompts, 4)
                reference = None
                for repeat in (1, 2):
                    for variant in args.samplers:
                        for mode in ("off", "reuse"):
                            with execution_mode(engine, mode == "reuse", samplers[variant]):
                                metrics, completion = run_batch(engine, prompts, args.output_tokens)
                            if reference is None:
                                reference = completion
                            checks = [completion_check(a, b, args.output_tokens) for a, b in zip(reference, completion)]
                            strict = len(completion) == len(reference) == batch and all(c["strict_pass"] for c in checks)
                            row = dict(batch=batch, case=case, sampler=variant, mode=mode, repeat=repeat,
                                       execution_order=len(report["samples"]) + 1, metrics=metrics,
                                       completion=completion, requests=checks, strict_pass=strict)
                            report["samples"].append(row)
                            save()
                            print(f"B{batch} {case} {variant}/{mode} repeat={repeat} tok/s={metrics['tokens_per_s']:.2f} strict={strict}", flush=True)
        finally:
            engine.exit()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mtp-model", required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[2])
    parser.add_argument("--cases", nargs="+", choices=["same64", "chinese-code"], default=["same64", "chinese-code"])
    parser.add_argument("--samplers", nargs="+", choices=["reference", "fast"], default=["reference", "fast"])
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output_tokens < 2 or any(b < 1 for b in args.batches):
        parser.error("outputs must be >=2 and batch sizes positive")
    report = measure(args)
    if any(not s["strict_pass"] for s in report["samples"]):
        raise SystemExit("Strict completion gate failed; diagnostic results retained")


if __name__ == "__main__":
    main()
