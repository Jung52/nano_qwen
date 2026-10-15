"""Strict real-model, unequal-context B4/B8 gates; diagnostic runs, not timing."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from bench_mtp_interleaved import execution_mode
from bench_mtp_batch2 import run_batch
from bench_mtp_small_batch import completion_check, make_cases
from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.layers.sampler import Sampler
from transformers import AutoTokenizer


@torch.inference_mode()
def measure(args):
    root = Path(__file__).resolve().parents[1]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    seed = next(c["prompt"] for c in make_cases(tokenizer) if c["name"] == "synthetic-256-128")
    report = dict(model=args.model, mtp_model=args.mtp_model, output=128,
                  purpose="strict correctness only; timings are not formal samples",
                  source=dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                              sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in [*root.joinpath("src/nano_qwen").rglob("*.py"), Path(__file__).resolve()]}),
                  gates=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in (4, 8):
        prompts = [seed[:64 + 24*i] for i in range(batch)]
        reference = None
        for eager in (True, False):
            engine = LLMEngine(args.model, max_num_seqs=batch, max_num_batched_tokens=2048,
                               max_model_len=1024, gpu_memory_utilization=.85,
                               enforce_eager=eager, use_prefill_cudagraph=False,
                               enable_mtp=True, mtp_model=args.mtp_model, mtp_verify_mode="reuse")
            try:
                for mode in ("off", "serial", "reuse"):
                    if mode != "off":
                        engine.config.mtp_verify_mode = mode
                        if not eager:
                            engine.model_runner.cuda_graphs._capture_verify()
                    with execution_mode(engine, mode != "off", Sampler()):
                        metrics, completions = run_batch(engine, prompts, 128)
                    if reference is None:
                        reference = completions
                    checks = [completion_check(a, b, 128) for a, b in zip(reference, completions)]
                    strict = len(reference) == len(completions) == batch and all(c["strict_pass"] for c in checks)
                    report["gates"].append(dict(batch=batch, eager=eager, mode=mode,
                                                prompts=prompts, completion=completions,
                                                requests=checks, strict_pass=strict,
                                                diagnostic_metrics=metrics))
                    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                    print(f"B{batch} {'eager' if eager else 'graph'} {mode}: strict={strict}", flush=True)
                    if not strict:
                        raise RuntimeError("Strict completion gate failed; diagnostic records retained")
            finally:
                engine.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mtp-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    measure(parser.parse_args())
