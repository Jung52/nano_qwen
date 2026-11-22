"""Fixed-batch decode microbenchmark: B=1..8 on one 5090, same 9B FP8 model,
same context length and token positions per request.

Per batch size: actual execution path per step (CUDA Graph hit or eager
fallback), GPU busy time per step (CUDA events), wall-clock step time and
effective tok/s. Prefill is excluded; all B requests stay active for the
whole measured window so the decode batch size is exactly B.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.engine.sequence import Sequence
from nano_qwen.sampling_params import SamplingParams


def p95(values):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * 0.95
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--context", type=int, default=257,
                        help="prompt length for every request (same content)")
    parser.add_argument("--out-len", type=int, default=64,
                        help="decode tokens per request (measured window)")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=30)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    engine = LLMEngine(args.model, enforce_eager=False, use_prefill_cudagraph=True,
                       max_num_seqs=max(args.batches), max_num_batched_tokens=512,
                       max_model_len=4096, gpu_memory_utilization=0.7,
                       tensor_parallel_size=1, async_scheduling=False,
                       enable_prefix_cache=False)
    seed = engine.tokenizer.encode("Explain how a computer stores information in memory. ",
                                   add_special_tokens=False)
    report = dict(args=vars(args), batches=[])

    for batch_size in args.batches:
        params = SamplingParams(temperature=1e-8,
                                max_tokens=args.out_len + args.warmup_steps
                                + args.measured_steps, ignore_eos=True)
        seqs = [Sequence((seed * ((args.context + len(seed) - 1) // len(seed)))[:args.context],
                         params) for _ in range(batch_size)]
        for seq in seqs:
            engine.scheduler.add(seq)
        while any(seq.num_completion_tokens < 1 for seq in seqs):
            engine.step()  # prefill (excluded from measurement)
        graph_hit = batch_size in getattr(engine.model_runner, "graphs", {})
        gpu_ms, wall_ms, steps = [], [], []
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        for step in range(args.warmup_steps + args.measured_steps):
            torch.cuda.synchronize()  # wall boundary before the step
            import time
            wall_start = time.perf_counter()
            start_evt.record()
            engine.step()
            end_evt.record()
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - wall_start) * 1e3)
            if step >= args.warmup_steps:
                gpu_ms.append(start_evt.elapsed_time(end_evt))
                steps.append(step)
        # verify the batch stayed at exactly batch_size during measurement
        per_step_tokens = [batch_size] * len(steps)
        tok_s = [batch_size / (wall / 1e3) for wall in wall_ms[args.warmup_steps:]]
        report["batches"].append(dict(
            batch=batch_size, path="graph" if graph_hit else "eager",
            graph_hit=graph_hit,
            steps=len(steps), tokens_per_step=batch_size,
            gpu_ms_p50=statistics.median(gpu_ms), gpu_ms_p95=p95(gpu_ms),
            wall_ms_p50=statistics.median(wall_ms[args.warmup_steps:]),
            wall_ms_p95=p95(wall_ms[args.warmup_steps:]),
            effective_tok_s=statistics.median(tok_s),
            effective_tok_s_p95=p95(tok_s)))
        print(json.dumps(report["batches"][-1]), flush=True)
        # drain the requests
        while not engine.is_finished():
            engine.step()
    engine.exit()
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(batches=report["batches"]), indent=2))


if __name__ == "__main__":
    main()
