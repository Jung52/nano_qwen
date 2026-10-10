"""BF16/FP8 压力测试:更大 batch、更长 prompt、单次大规模 prefill。

Run: python benchmarks/stress_bench.py --model MODEL --json-out OUT.json \
     [--cases "2048x8,4096x8,8192x4,16384x2"] [--mode decode_graph]

与 validate_fp8_model.py 的区别:
- 不限制 prompt <=2048 / batch <=8,engine 参数随用例放大;
- prefill 允许 chunked(引擎默认 max_num_batched_tokens=16384),按整段计时;
- 轮数可减(warmup/rounds),默认 1+2。
"""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.engine.sequence import Sequence
from nano_qwen.sampling_params import SamplingParams


PROMPT_BASE = (
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning systems read input tokens, encode context, and generate output tokens. "
    "Attention layers weigh relevant history while feed-forward layers transform representations. "
)


def storage_bytes(tensors):
    storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                for t in tensors}
    return sum(storages.values())


def stress_case(engine, prompt, batch, decode_tokens):
    """单次全量 prefill(可 chunk)+ 固定 decode 步数,容忍 chunked prefill。"""
    seqs = [Sequence(prompt, SamplingParams(temperature=1.0, max_tokens=decode_tokens,
                                            ignore_eos=True))
            for _ in range(batch)]
    for seq in seqs:
        engine.scheduler.add(seq)
    torch.cuda.synchronize()
    start = time.perf_counter()
    # 直到每个序列都产出第一个 token(prefill 阶段,可能多步 chunk)
    while not all(seq.num_completion_tokens >= 1 for seq in seqs):
        engine.step()
    torch.cuda.synchronize()
    prefill_end = time.perf_counter()
    while not engine.is_finished():
        engine.step()
    torch.cuda.synchronize()
    end = time.perf_counter()
    assert all(seq.num_completion_tokens == decode_tokens for seq in seqs)
    assert not engine.model_runner.input_batch.seq_id_to_slot, "finished batch leaked slots"
    prefill_s = prefill_end - start
    decode_s = end - prefill_end
    return {
        "ttft_ms": prefill_s * 1000,
        "prefill_tok_per_s": batch * len(prompt) / prefill_s,
        "decode_tok_per_s": batch * (decode_tokens - 1) / decode_s,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--mode", choices=("eager", "decode_graph"), default="decode_graph")
    parser.add_argument("--cases", default="2048x8,4096x8,8192x4,16384x2",
                        help="promptxbatch 列表,如 2048x8,4096x8")
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--gpu-util", type=float, default=0.9)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    args = parser.parse_args()

    cases = []
    for spec in args.cases.split(","):
        p, b = spec.split("x")
        cases.append((int(p), int(b)))
    max_prompt, max_batch = max(p for p, _ in cases), max(b for _, b in cases)

    engine = LLMEngine(
        args.model,
        enforce_eager=args.mode == "eager",
        tensor_parallel_size=1,
        max_num_seqs=max_batch * 2,
        max_model_len=max_prompt + args.decode_tokens,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_util,
    )
    runner = engine.model_runner
    report = {
        "model": args.model, "mode": args.mode, "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "quantization": engine.config.quantization,
        "parameters_mib": storage_bytes(runner.model.parameters()) / 2**20,
        "gdn_state_mib": storage_bytes(t for l in runner.gdn_layers
                                       for t in (l.conv_states, l.recurrent_states)) / 2**20,
        "kv_mib": runner.kv_cache.numel() * runner.kv_cache.element_size() / 2**20,
        "num_kvcache_blocks": engine.config.num_kvcache_blocks,
        "decode_tokens": args.decode_tokens, "warmup": args.warmup, "rounds": args.rounds,
        "workloads": [],
    }
    print(f"MODEL_READY mode={args.mode} quantization={engine.config.quantization}", flush=True)
    try:
        block = engine.config.kvcache_block_size
        for prompt_len, batch in cases:
            base_ids = engine.tokenizer.encode(PROMPT_BASE, add_special_tokens=False)
            prompt = (base_ids * math.ceil(prompt_len / len(base_ids)))[:prompt_len]
            needed = batch * math.ceil((prompt_len + args.decode_tokens) / block)
            if needed > engine.config.num_kvcache_blocks:
                print(f"SKIP prompt={prompt_len} bs={batch}: KV 容量不足 "
                      f"({needed} > {engine.config.num_kvcache_blocks} blocks)", flush=True)
                continue
            runs = []
            for index in range(args.warmup + args.rounds):
                torch.manual_seed(20261006 + index)
                result = stress_case(engine, prompt, batch, args.decode_tokens)
                if index >= args.warmup:
                    runs.append(result)
            medians = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
            report["workloads"].append({"prompt_tokens": prompt_len, "batch": batch,
                                        "runs": runs, **medians})
            print(f"STRESS prompt={prompt_len} bs={batch} ttft={medians['ttft_ms']:.1f}ms "
                  f"prefill={medians['prefill_tok_per_s']:.0f}tok/s "
                  f"decode={medians['decode_tok_per_s']:.2f}t/s", flush=True)
        report["status"] = "stress_completed"
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"REPORT {args.json_out}", flush=True)
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
