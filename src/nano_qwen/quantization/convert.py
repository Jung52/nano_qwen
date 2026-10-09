"""Convert a local BF16/FP16 Qwen3.5 dense checkpoint to nano_qwen FP8.

Run: python -m nano_qwen.quantization.convert --model SOURCE --output DEST
Block-128: add --scheme block_wise (HF-compatible weights/scales/config).
Weights are quantized offline; activations remain dynamic during inference.
"""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.distributed as dist
from transformers import AutoConfig

from nano_qwen.models.qwen3_5 import Qwen3_5ForCausalLM
from .checkpoint import (
    FP8_METADATA, METADATA_FILE, checkpoint_spec, normalize_weight_name,
    read_quantization_metadata, validate_tensor,
)
from .fp8 import Fp8Config
from .fp8_block import Fp8BlockConfig


def _quantize_weight(weight, device, scheme, name):
    values = weight.to(device=device, dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError(f"Non-finite source weight: {name}")
    if scheme == "block_wise":
        rows, columns = values.shape
        blocks = values.reshape(rows // 128, 128, columns // 128, 128)
        scale = blocks.abs().amax(dim=(1, 3)) / 448.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        normalized = (blocks / scale[:, None, :, None]).reshape(rows, columns)
    else:
        scale = values.abs().amax(dim=1, keepdim=True) / 448.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        normalized = values / scale
    quantized = normalized.clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized.cpu(), scale.cpu()


def convert_checkpoint(model_path, output_path, device="cpu", shard_mib=256,
                       scheme="per_channel"):
    """Write a separate text-only FP8 checkpoint without modifying the source.

    block_wise stores FP32 dequantization multipliers for each 128x128 weight
    block and an HF quantization_config; per_channel retains the v1 format.
    """
    if scheme not in ("per_channel", "block_wise"):
        raise ValueError(f"Unsupported FP8 conversion scheme: {scheme}")
    if shard_mib < 1:
        raise ValueError("shard_mib must be positive")
    source, destination = Path(model_path).resolve(), Path(output_path).resolve()
    if read_quantization_metadata(source) is not None:
        raise ValueError("Source is already a quantized checkpoint")
    if destination == source or (destination.exists() and any(destination.iterdir())):
        raise ValueError("Output must be a separate, empty directory")
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint in {source}")
    full_config = AutoConfig.from_pretrained(source)
    config = getattr(full_config, "text_config", full_config)
    if (getattr(full_config, "quantization_config", None)
            or getattr(config, "quantization_config", None)):
        raise ValueError("Source is already a quantized checkpoint")
    if config.model_type != "qwen3_5_text" or getattr(config, "num_experts", 0):
        raise ValueError("Conversion supports Qwen3.5 dense text backbones only")
    if config.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Source model compute dtype must be BF16 or FP16")
    if dist.is_initialized():
        raise ValueError("Run conversion in a separate TP=1 process")
    block_quant = {"quant_method": "fp8", "activation_scheme": "dynamic",
                   "weight_block_size": [128, 128], "modules_to_not_convert": []}
    quant_config = Fp8BlockConfig(block_quant) if scheme == "block_wise" else Fp8Config()
    with tempfile.TemporaryDirectory() as rendezvous:
        dist.init_process_group(
            "gloo", init_method=Path(rendezvous, "store").as_uri(), rank=0, world_size=1,
        )
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(config.dtype)
            with torch.device("meta"):
                model = Qwen3_5ForCausalLM(config, quant_config=quant_config)
            spec = checkpoint_spec(model)
        finally:
            torch.set_default_dtype(previous)
            dist.destroy_process_group()
    if config.tie_word_embeddings:
        spec.pop("lm_head.weight", None)
    scale_field = "weight_scale_inv" if scheme == "block_wise" else "weight_scale"
    expected_sources = {key for key in spec if not key.endswith("." + scale_field)}
    if scheme == "block_wise":
        block_quant["modules_to_not_convert"] = sorted({
            name.removesuffix(".weight") for name, (_, _, _, dtype) in spec.items()
            if name.endswith(".weight") and dtype != torch.float8_e4m3fn
        } | {"lm_head"})
    destination.mkdir(parents=True, exist_ok=True)
    pending, pending_bytes, total_bytes, index, seen = {}, 0, 0, {}, set()
    quantized_layers = 0

    def flush():
        nonlocal pending, pending_bytes
        if not pending:
            return
        filename = f"model-{len(set(index.values())) + 1:05d}.safetensors"
        save_file(pending, str(destination / filename), metadata={"format": "pt"})
        index.update({name: filename for name in pending})
        pending, pending_bytes = {}, 0

    for file in files:
        with safe_open(file, "pt", "cpu") as checkpoint:
            for raw_name in checkpoint.keys():
                name = normalize_weight_name(raw_name)
                if name not in expected_sources:
                    continue  # Exclude vision/MTP and the redundant tied LM head.
                if name in seen:
                    raise ValueError(f"Duplicate source tensor: {name}")
                seen.add(name)
                weight = checkpoint.get_tensor(raw_name)
                _, _, shape, dtype = spec[name]
                if tuple(weight.shape) != shape or weight.dtype not in (
                    torch.bfloat16, torch.float16, torch.float32,
                ):
                    raise ValueError(f"Invalid source weight {raw_name}: {weight.shape}/{weight.dtype}")
                if dtype == torch.float8_e4m3fn:
                    quantized, scale = _quantize_weight(weight, device, scheme, raw_name)
                    tensors = {name: quantized, name.removesuffix("weight") + scale_field: scale}
                    quantized_layers += 1
                else:
                    tensors = {name: weight.to(dtype=dtype).contiguous()}
                for key, tensor in tensors.items():
                    validate_tensor(key, tensor, spec[key][2], spec[key][3])
                    size = tensor.numel() * tensor.element_size()
                    if pending_bytes + size > shard_mib * 2**20:
                        flush()
                    pending[key] = tensor
                    pending_bytes += size
                    total_bytes += size
                if quantized_layers and quantized_layers % 24 == 0 and dtype == torch.float8_e4m3fn:
                    print(f"Converted {quantized_layers} projections", flush=True)
    missing = expected_sources - seen
    if missing:
        raise ValueError(f"Missing source tensors: {sorted(missing)}")
    flush()
    for filename in (
        "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja", "generation_config.json",
    ):
        if filename == "config.json" and scheme == "block_wise":
            continue  # Write the HF quantization marker only after every shard/index.
        if (source / filename).is_file():
            shutil.copy2(source / filename, destination / filename)
    (destination / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_bytes}, "weight_map": index,
    }, indent=2), encoding="utf-8")
    format_metadata = ({"format": "hf_fp8_block128", **block_quant}
                       if scheme == "block_wise" else FP8_METADATA)
    metadata = {**format_metadata, "scheme": scheme, "source": str(source),
                "quantized_projections": quantized_layers,
                "tensor_bytes": total_bytes, "text_only": True}
    # Write the success marker last so incomplete conversion is not recognized as FP8.
    if scheme == "block_wise":
        saved_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        saved_config["quantization_config"] = block_quant
        (destination / "config.json").write_text(json.dumps(saved_config, indent=2), encoding="utf-8")
    else:
        (destination / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--shard-mib", type=int, default=256)
    parser.add_argument("--scheme", choices=("per_channel", "block_wise"), default="per_channel",
                        help="FP8 weight scaling: per output channel (default) or 128x128 blocks")
    args = parser.parse_args()
    if args.shard_mib < 1:
        parser.error("--shard-mib must be positive")
    print(json.dumps(convert_checkpoint(args.model, args.output, args.device, args.shard_mib,
                                        scheme=args.scheme), indent=2))


if __name__ == "__main__":
    main()
