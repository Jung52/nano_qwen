"""Two-sample full-sampling versus greedy-fast control for ordinary decode."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nano_qwen.engine.llm_engine import LLMEngine  # noqa: E402
from nano_qwen.engine.sequence import Sequence  # noqa: E402
from nano_qwen.layers.sampler import Sampler  # noqa: E402
from nano_qwen.sampling_params import SamplingParams  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class FullSamplingReference(Sampler):
    """Keep deterministic greedy ties but execute the full sampling algorithm."""

    def forward(self, logits, temperatures, all_greedy=False):
        return super().forward(logits, temperatures)


@torch.inference_mode()
def run(engine, prompts, output):
    if not engine.is_finished() or engine.model_runner.input_batch.seq_id_to_slot:
        raise ValueError("An idle engine is required")
    seqs = [Sequence(p, SamplingParams(temperature=1e-6, max_tokens=output, ignore_eos=True)) for p in prompts]
    torch.manual_seed(0)
    start = perf_counter()
    for seq in seqs:
        engine.scheduler.add(seq)
    engine.step()
    assert all(s.num_completion_tokens == 1 for s in seqs)
    first = perf_counter()
    while not all(s.is_finished for s in seqs):
        engine.step()
    end = perf_counter()
    assert engine.is_finished() and not engine.model_runner.input_batch.seq_id_to_slot
    completions = [s.completion_token_ids for s in seqs]
    assert all(len(c) == output for c in completions)
    return dict(decode_tokens=len(seqs)*(output-1), decode_s=end-first,
                tokens_per_s=len(seqs)*(output-1)/(end-first), tpot_ms=1000*(end-first)/(output-1),
                ttft_ms=1000*(first-start), e2e_s=end-start), completions


def measure(args):
    root = Path(__file__).resolve().parents[1]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    seed = tokenizer.encode("Qwen3.5 ordinary decode throughput verification.", add_special_tokens=False)
    prompt = (seed*64)[:64]
    texts = ["请解释大语言模型推理中 prefill 和 decode 的区别，并以一个中文聊天请求为例，说明延迟和吞吐为什么会随 batch 大小变化。",
             "Write a Python function that merges overlapping intervals. Include type hints, explain the algorithm, and show tests for empty input, touching intervals, and nested intervals."]
    chat = [tokenizer.apply_chat_template([dict(role="user", content=t)], tokenize=True,
                                         add_generation_prompt=True, enable_thinking=False, return_dict=False) for t in texts]
    report = dict(model=args.model, batch=2, input_cases={"same64": [prompt, prompt], "chinese-code": chat},
                  output=128, repeats=2, prefill_graph=False, decode_graph=True,
                  environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda),
                  source=dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                              sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in [*root.joinpath("src/nano_qwen").rglob("*.py"), Path(__file__).resolve()]}), samples=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    engine = LLMEngine(args.model, max_num_seqs=2, max_num_batched_tokens=1024, max_model_len=1024,
                       gpu_memory_utilization=.85, enforce_eager=False, use_prefill_cudagraph=False)
    try:
        for case, prompts in report["input_cases"].items():
            samplers = dict(reference=FullSamplingReference(), fast=Sampler())
            for sampler in samplers.values():
                engine.model_runner.sampler = sampler
                run(engine, prompts, 4)
            expected = None
            for repeat in (1, 2):
                for name, sampler in samplers.items():
                    engine.model_runner.sampler = sampler
                    metrics, completions = run(engine, prompts, 128)
                    expected = completions if expected is None else expected
                    strict = expected == completions
                    report["samples"].append(dict(case=case, sampler=name, repeat=repeat, execution_order=len(report["samples"])+1,
                                                  metrics=metrics, completion=completions, strict_pass=strict))
                    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
                    print(case, name, repeat, metrics["tokens_per_s"], "strict", strict, flush=True)
                    if not strict:
                        raise RuntimeError("Strict greedy completion mismatch")
    finally:
        engine.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    measure(parser.parse_args())
