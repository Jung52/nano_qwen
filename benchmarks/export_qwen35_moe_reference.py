"""Export a local Transformers baseline for the pinned Qwen3.5-2B-MoE checkpoint."""

import argparse
import json
from collections import Counter
from hashlib import sha1, sha256
from pathlib import Path

from safetensors import safe_open


MODEL_ID = "bayliner1980/Qwen3.5-2B-MoE-A1.3B"
REVISION = "0ab0f2c0b07497038ac8f5dd15d133574413da28"
LFS_SHA256 = {
    "model.safetensors": "dfe68d3bb33b7505a300fc753a536d51880b0e04d16b8c56d446030ff8c956c7",
    "tokenizer.json": "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
}
CONFIG_BLOBS = {
    "config.json": "ae24095046f14b990b7ef38eb3ed8b63c1872c02",
    "generation_config.json": "5522f62a006c2f55e12c20e7f23d203c17fcd1f5",
    "tokenizer_config.json": "46fb74b44dc6db0f15216921f2ba5f98ceb03c86",
}
CASES = (
    ("short_english", "The capital of France is", " Paris, and the capital of Italy is Rome."),
    ("multilingual", "用一句话解释混合专家模型。", "混合专家模型按输入选择部分专家进行计算。"),
)
DECODE_STEPS = 8


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_revision(checkpoint):
    for filename, expected in CONFIG_BLOBS.items():
        data = (checkpoint / filename).read_bytes()
        digest = sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if digest != expected:
            raise ValueError(f"{filename} does not match revision {REVISION}")
    for filename, expected in LFS_SHA256.items():
        digest = sha256()
        with (checkpoint / filename).open("rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024**2), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"{filename} does not match revision {REVISION}")


def inspect_checkpoint(checkpoint, output):
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config["model_type"] != "qwen3_5_moe_text":
        raise ValueError(f"Unexpected model type: {config['model_type']}")
    h = config["hidden_size"]
    e = config["num_experts"]
    width = config["moe_intermediate_size"]
    shared_width = config["shared_expert_intermediate_size"]
    expected = {
        "gate.weight": [e, h],
        "shared_expert.gate_proj.weight": [shared_width, h],
        "shared_expert.up_proj.weight": [shared_width, h],
        "shared_expert.down_proj.weight": [h, shared_width],
        "shared_expert_gate.weight": [1, h],
    }
    for expert in range(e):
        expected[f"experts.{expert}.gate_proj.weight"] = [width, h]
        expected[f"experts.{expert}.up_proj.weight"] = [width, h]
        expected[f"experts.{expert}.down_proj.weight"] = [h, width]
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
        tensors = {}
        for name in weights.keys():
            tensor = weights.get_slice(name)
            tensors[name] = {"shape": tensor.get_shape(), "dtype": tensor.get_dtype()}
        metadata = weights.metadata()
    moes = {
        f"model.language_model.layers.{layer}.mlp.{suffix}": shape
        for layer in range(config["num_hidden_layers"])
        for suffix, shape in expected.items()
    }
    moes["model.language_model.embed_tokens.weight"] = [config["vocab_size"], h]
    write_json(output / "checkpoint_schema.json", {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "file_bytes": (checkpoint / "model.safetensors").stat().st_size,
        "weights_sha256": LFS_SHA256["model.safetensors"],
        "config": config,
        "safetensors_metadata": metadata,
        "tensors": tensors,
    })
    for name, shape in moes.items():
        if tensors.get(name) != {"shape": shape, "dtype": "BF16"}:
            raise ValueError(f"{name}: expected BF16 {shape}, found {tensors.get(name)}")
    dtypes = Counter(entry["dtype"] for entry in tensors.values())
    print(f"Checkpoint verified: {len(tensors)} tensors, {dict(dtypes)}", flush=True)
    return config


def run_baseline(checkpoint, output, config):
    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The reference uses CUDA BF16")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        experts_implementation="eager",
        local_files_only=True,
        output_loading_info=True,
    )
    model.to("cuda:0").eval()
    load_issues = {key: value for key, value in loading_info.items() if value}
    if load_issues:
        raise RuntimeError(f"Transformers loading issues: {load_issues}")
    loading_info = {key: list(value) for key, value in loading_info.items()}
    tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    if not tied:
        raise RuntimeError("Embedding and LM head do not share storage")

    logits = {}
    cases = {}
    with torch.inference_mode():
        for name, prompt, continuation in CASES:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            fixed_ids = tokenizer.encode(continuation, add_special_tokens=False)[:DECODE_STEPS]
            if len(fixed_ids) != DECODE_STEPS:
                raise ValueError(f"{name} must have {DECODE_STEPS} fixed continuation IDs")
            prompt_tensor = torch.tensor([prompt_ids], device="cuda:0")
            prefill = model(input_ids=prompt_tensor, use_cache=True, logits_to_keep=1)
            rows = [prefill.logits[0, -1].cpu()]
            cache = prefill.past_key_values
            for token in fixed_ids:
                step = model(
                    input_ids=torch.tensor([[token]], device="cuda:0"),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = step.past_key_values
                rows.append(step.logits[0, -1].cpu())
            rows = torch.stack(rows)
            if rows.shape != (DECODE_STEPS + 1, config["vocab_size"]) or not torch.isfinite(rows).all():
                raise RuntimeError(f"Invalid logits for {name}: {rows.shape}")

            greedy = model.generate(
                input_ids=prompt_tensor,
                attention_mask=torch.ones_like(prompt_tensor),
                do_sample=False,
                max_new_tokens=DECODE_STEPS,
                pad_token_id=tokenizer.pad_token_id,
            )
            greedy_ids = greedy[0, len(prompt_ids):].cpu().tolist()
            logits[name] = rows
            cases[name] = {
                "prompt": prompt,
                "prompt_token_ids": prompt_ids,
                "fixed_decode_token_ids": fixed_ids,
                "reference_argmax_ids": rows.argmax(dim=1).tolist(),
                "greedy_token_ids": greedy_ids,
                "greedy_text": tokenizer.decode(greedy_ids, skip_special_tokens=True),
                "logits_shape": list(rows.shape),
                "logits_dtype": str(rows.dtype),
            }
            print(f"{name}: {len(prompt_ids)} prompt tokens and {len(fixed_ids)} decode steps", flush=True)

    temporary = output / "reference_logits.safetensors.tmp"
    save_file(logits, temporary, metadata={"revision": REVISION})
    temporary.replace(output / "reference_logits.safetensors")
    write_json(output / "reference.json", {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "model_dtype": str(model.dtype),
        "attention_implementation": model.config._attn_implementation,
        "experts_implementation": model.config._experts_implementation,
        "loading_info": loading_info,
        "embeddings_tied": tied,
        "row_order": "prefill last position, then one row after consuming each fixed decode token",
        "cases": cases,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/qwen35-moe-reference"))
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    verify_revision(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    config = inspect_checkpoint(args.checkpoint, args.output)
    if not args.inspect_only:
        run_baseline(args.checkpoint, args.output, config)


if __name__ == "__main__":
    main()
