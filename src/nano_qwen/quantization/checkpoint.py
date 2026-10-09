"""Versioned nano_qwen checkpoint metadata and unfused parameter layout."""

import json
from pathlib import Path

import torch


METADATA_FILE = "nano_qwen_quantization.json"
FP8_METADATA = {
    "format": "nano_qwen_fp8_v1",
    "quant_method": "fp8",
    "weight_dtype": "float8_e4m3fn",
    "weight_scale": "per_output_channel_float32",
    "activation_scale": "dynamic_per_token_float32",
}


def read_quantization_metadata(path):
    file = Path(path) / METADATA_FILE
    if not file.exists():
        return None
    metadata = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or any(
        metadata.get(key) != value for key, value in FP8_METADATA.items()
    ):
        raise ValueError(f"Unsupported nano_qwen quantization metadata: {file}")
    return metadata


def read_hf_fp8_config(path):
    file = Path(path) / "config.json"
    if not file.exists():
        return None
    config = json.loads(file.read_text(encoding="utf-8"))
    quant = config.get("quantization_config") or config.get("text_config", {}).get("quantization_config")
    if quant is None:
        return None
    if (quant.get("quant_method") != "fp8"
            or quant.get("weight_block_size") != [128, 128]
            or quant.get("activation_scheme") != "dynamic"
            or quant.get("fmt", "e4m3") != "e4m3"):
        raise ValueError("Only HF dynamic E4M3 FP8 with weight_block_size=[128,128] is supported")
    return quant


def normalize_weight_name(name):
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    return name


def checkpoint_spec(model):
    """Map unfused checkpoint keys to (parameter name, shard, shape, dtype)."""
    modules = dict(model.named_modules())
    packed = getattr(model, "packed_modules_mapping", {})
    spec = {}
    for name, param in model.named_parameters():
        module_name, field = name.rsplit(".", 1)
        parent, _, leaf = module_name.rpartition(".")
        sources = [(source, shard) for source, (target, shard) in packed.items()
                   if target == leaf]
        if sources:
            for source, shard in sources:
                rows = modules[module_name].output_sizes[shard]
                if field == "weight_scale_inv":
                    rows //= 128
                shape = (rows, *param.shape[1:])
                key = f"{parent}.{source}.{field}"
                spec[key] = (name, shard, shape, param.dtype)
        else:
            spec[name] = (name, None, tuple(param.shape), param.dtype)
    return spec


def tied_head(model):
    return (
        model.lm_head.weight.device.type != "meta"
        and model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    )


def validate_tensor(name, tensor, shape, dtype):
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise ValueError(
            f"Invalid tensor {name}: expected {tuple(shape)} / {dtype}, "
            f"got {tuple(tensor.shape)} / {tensor.dtype}"
        )
    values = tensor.float()
    if not torch.isfinite(values).all():
        raise ValueError(f"Non-finite tensor: {name}")
    if name.endswith((".weight_scale", ".weight_scale_inv")) and not (values > 0).all():
        raise ValueError(f"Weight scales must be positive: {name}")
