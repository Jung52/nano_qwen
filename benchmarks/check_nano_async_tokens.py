#!/usr/bin/env python3
"""Compare real greedy CUDA results with lookahead off/on, in fresh processes.

  python bench/check_nano_async_tokens.py --model /path/to/model

Default prompt lengths 127, 255, 256, 257 exercise KV block boundaries. This is
a numerical/lifecycle check, not a profiler or throughput benchmark.
"""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import random
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-len", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=.9)
    p.add_argument("--output-dir", default="logs/nano_async_tokens")
    p.add_argument("--eager", action="store_true")
    p.add_argument("--respect-eos", action="store_true")
    p.add_argument("--child", choices=("off", "on"), help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.batch_size < 1 or a.output_len < 2 or a.max_model_len < 257 + a.output_len:
        p.error("Require batch-size>=1, output-len>=2, max-model-len>=257+output-len")
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") not in ("0", ""):
        p.error("Unset CUDA_LAUNCH_BLOCKING")
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.child is None:
        # Each subprocess uses the same seed, prompts and model, without
        # retaining the previous engine's allocator or graph state.
        for mode in ("off", "on"):
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                *sys.argv[1:], "--child", mode], check=True)
        off, on = [json.loads((out / f"{m}.json").read_text()) for m in ("off", "on")]
        if off["prompts"] != on["prompts"] or off["tokens"] != on["tokens"]:
            for i, (x, y) in enumerate(zip(off["tokens"], on["tokens"])):
                if x != y:
                    print(f"Mismatch at request {i}: off={x}, on={y}")
            raise RuntimeError("Greedy output differs; inspect saved off/on JSON before relying on the patch")
        print(f"PASS: identical token IDs for {len(off['tokens'])} requests; {out}")
        return

    root = next((r for r in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
                 if (r / "src/nano_qwen").is_dir()), None)
    if root is None:
        raise RuntimeError("Put this script under nano_qwen/bench or benchmarks")
    sys.path.insert(0, str(root / "src"))
    import torch
    from nano_qwen.config import Config
    from nano_qwen.engine.llm_engine import LLMEngine
    from nano_qwen.engine.sequence import Sequence
    from nano_qwen.sampling_params import SamplingParams
    if "async_scheduling" not in {f.name for f in fields(Config)}:
        raise RuntimeError("Apply the async decode patch first")
    torch.manual_seed(42)
    rng = random.Random(42)
    with torch.inference_mode():
        e = LLMEngine(a.model, async_scheduling=a.child == "on",
            max_num_seqs=max(8, a.batch_size), max_num_batched_tokens=max(8, a.batch_size)*257,
            max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization,
            enable_prefix_cache=False, enforce_eager=a.eager, use_prefill_cudagraph=False)
    try:
        vocab = e.config.hf_config.vocab_size
        lengths = [127, 255, 256, 257]
        prompts = [[rng.randrange(vocab) for _ in range(lengths[i % 4])]
                   for i in range(a.batch_size)]
        sp = SamplingParams(temperature=1e-8, max_tokens=a.output_len, ignore_eos=not a.respect_eos)
        seqs = [Sequence(prompt, sp) for prompt in prompts]
        with torch.inference_mode():
            for seq in seqs:
                e.scheduler.add(seq)
            for _ in range(10000):
                if e.is_finished():
                    break
                e.step()
            else:
                raise RuntimeError("Engine did not drain within 10000 steps")
            torch.cuda.synchronize()
        if not all(s.is_finished for s in seqs):
            raise RuntimeError("Engine drained before every request finished")
        if not a.respect_eos and any(s.num_completion_tokens != a.output_len for s in seqs):
            raise RuntimeError("A request did not produce its configured output length")
        if e.model_runner.input_batch.seq_id_to_slot:
            raise RuntimeError("Request slots remain after engine drain")
        if e.scheduler.block_manager.used_block_ids:
            raise RuntimeError("KV blocks remain after engine drain")
        payload = {"mode": a.child, "prompts": prompts,
                   "tokens": [s.completion_token_ids for s in seqs], "eager": a.eager}
        (out / f"{a.child}.json").write_text(json.dumps(payload, indent=2))
    finally:
        e.exit()


if __name__ == "__main__":
    main()
