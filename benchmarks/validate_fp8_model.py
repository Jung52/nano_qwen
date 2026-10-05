"""Complete-model FP8 loading, common-input logits, generation and graph checks.

Run each mode separately to fit a 12GB GPU. Saved common-input logits allow
BF16/FP8 comparison without keeping two complete models in GPU memory.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engine_bench_utils import run_until_idle
from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.engine.sequence import Sequence
from nano_qwen.layers.attention import Attention
from nano_qwen.sampling_params import SamplingParams
from nano_qwen.utils.context import set_context, reset_context


TEXTS = [
    "The sky appears blue because sunlight scatters in Earth's atmosphere. Shorter wavelengths scatter more strongly, so blue light reaches our eyes from many directions.",
    "量化通过减少模型权重和激活使用的位数，降低显存占用。缩放系数用于在低精度数值与原始数值之间转换。评估量化需要同时检查准确率和推理速度。",
    "def add(a, b):\n    return a + b\n\nassert add(2, 3) == 5\n# Adding two integers returns their sum.\n",
]
QUESTIONS = ["Explain briefly why the sky appears blue.", "什么是模型量化？用两句话说明。"]


def storage_bytes(tensors):
    storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                for t in tensors}
    return sum(storages.values())


class GreedySampler(torch.nn.Module):
    def forward(self, logits, temperatures):
        assert torch.isfinite(logits).all(), "Non-finite generation logits"
        return logits.argmax(dim=-1)


class GraphCounter:
    def __init__(self, runner):
        self.manager = runner.cuda_graphs
        self.hits = {"decode": 0, "prefill": 0}

    def __enter__(self):
        self.originals = {}
        for kind in self.hits:
            name = f"run_{kind}"
            original = getattr(self.manager, name)
            self.originals[name] = original

            def counted(*args, _kind=kind, _original=original, **kwargs):
                self.hits[_kind] += 1
                return _original(*args, **kwargs)

            setattr(self.manager, name, counted)
        return self

    def __exit__(self, *exc):
        for name, original in self.originals.items():
            setattr(self.manager, name, original)


def batch(engine, prompts, tokens):
    seqs = [Sequence(p, SamplingParams(temperature=1.0, max_tokens=tokens, ignore_eos=True))
            for p in prompts]
    for seq in seqs:
        engine.scheduler.add(seq)
    torch.cuda.synchronize()
    result = run_until_idle(engine, seqs)
    torch.cuda.synchronize()
    assert all(len(s.completion_token_ids) == tokens for s in seqs)
    assert not engine.model_runner.input_batch.seq_id_to_slot
    window = max(result.e2e_s.values()) - min(result.ttft_s.values())
    return {"outputs": result.outputs, "ttft_ms": statistics.median(result.ttft_s.values()) * 1000,
            "decode_tokens_per_s": (tokens - 1) * len(seqs) / window if tokens > 1 else None}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("eager", "decode_graph", "piecewise_graph"), default="eager")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    engine = LLMEngine(
        args.model, enforce_eager=args.mode == "eager",
        use_prefill_cudagraph=args.mode == "piecewise_graph",
        tensor_parallel_size=1, max_num_seqs=4,
        max_model_len=1024, max_num_batched_tokens=512, gpu_memory_utilization=0.65,
    )
    runner = engine.model_runner
    print(f"MODEL_READY mode={args.mode} quantization={engine.config.quantization}", flush=True)
    report = {"model": args.model, "mode": args.mode, "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "quantization": engine.config.quantization,
              "parameters_mib": storage_bytes(runner.model.parameters()) / 2**20,
              "gdn_state_mib": storage_bytes(t for l in runner.gdn_layers
                                            for t in (l.conv_states, l.recurrent_states)) / 2**20,
              "kv_mib": runner.kv_cache.numel() * runner.kv_cache.element_size() / 2**20,
              "generation": [], "timings": []}
    saved_logits = []
    try:
        # Same tokens in all model variants. Disable paged KV writes for this
        # standalone full-sequence prefill; the GDN state is reset per text.
        attentions = [m for m in runner.model.modules() if isinstance(m, Attention)]
        caches = [(m, m.k_cache, m.v_cache) for m in attentions]
        for m, _, _ in caches:
            m.k_cache = m.v_cache = torch.empty(0, device="cuda")
        try:
            for text in TEXTS:
                ids = torch.tensor(engine.tokenizer.encode(text), device="cuda", dtype=torch.int64)
                for layer in runner.gdn_layers:
                    layer.reset_state([0])
                boundaries = torch.tensor([0, ids.numel()], device="cuda", dtype=torch.int32)
                set_context(True, cu_seqlens_q=boundaries, cu_seqlens_k=boundaries,
                            max_seqlen_q=ids.numel(), max_seqlen_k=ids.numel(),
                            state_indices=torch.zeros(1, device="cuda", dtype=torch.int64),
                            prefill_slices=[(0, ids.numel())])
                hidden = runner.model(ids, torch.arange(ids.numel(), device="cuda"))
                logits = runner.model.compute_logits(hidden[-8:])
                assert torch.isfinite(logits).all()
                saved_logits.append({"ids": ids.cpu(), "logits": logits.float().cpu()})
                if args.mode == "piecewise_graph":
                    expected = logits.clone()
                    for layer in runner.gdn_layers:
                        layer.reset_state([0])
                    graph_logits = runner.cuda_graphs.run_prefill(
                        ids, torch.arange(ids.numel(), device="cuda"),
                    )[-8:]
                    assert torch.isfinite(graph_logits).all()
                    report.setdefault("prefill_graph_vs_eager", []).append({
                        "tokens": ids.numel(),
                        "relative_l2": float((graph_logits.float() - expected.float()).norm()
                                             / expected.float().norm()),
                        "top1_agreement": float((graph_logits.argmax(-1) == expected.argmax(-1))
                                                 .float().mean()),
                        "max_abs": float((graph_logits.float() - expected.float()).abs().max()),
                    })
                for layer in runner.gdn_layers:
                    assert torch.isfinite(layer.conv_states).all()
                    assert torch.isfinite(layer.recurrent_states).all()
        finally:
            reset_context()
            for m, k, v in caches:
                m.k_cache, m.v_cache = k, v
            for layer in runner.gdn_layers:
                layer.reset_state([0])
        torch.save(saved_logits, str(output) + ".logits.pt")
        production_sampler = runner.sampler
        runner.sampler = GreedySampler()
        for question in QUESTIONS:
            prompt = engine.tokenizer.encode(engine.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            ))
            with GraphCounter(runner) as counter:
                result = batch(engine, [prompt], 48)
            report["generation"].append({"question": question, "ids": result["outputs"][0],
                                         "text": engine.tokenizer.decode(result["outputs"][0]),
                                         "paths": dict(counter.hits)})
        runner.sampler = production_sampler
        # Warm timings with production async sampling; no per-token diagnostics.
        for length, size, tokens in [(128, 1, 32), (128, 4, 32), (512, 1, 1)]:
            prompt = (engine.tokenizer.encode(TEXTS[0]) * 20)[:length]
            runs = []
            for i in range(args.rounds + 1):
                torch.manual_seed(20261005 + i)
                result = batch(engine, [prompt] * size, tokens)
                if i:
                    runs.append(result)
            report["timings"].append({"prompt_tokens": length, "batch": size, "completion_tokens": tokens,
                "ttft_ms": statistics.median(r["ttft_ms"] for r in runs),
                "decode_tokens_per_s": statistics.median(r["decode_tokens_per_s"] for r in runs)
                    if tokens > 1 else None})
        report["status"] = "runtime_checks_passed"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
