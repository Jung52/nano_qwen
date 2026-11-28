import os
from contextlib import ExitStack
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

from nano_qwen.quantization import UnquantizedLinearMethod
from nano_qwen.quantization.checkpoint import (
    checkpoint_spec, normalize_weight_name, read_quantization_metadata, read_hf_fp8_config,
    tied_head, validate_tensor,
)


_EXPERT_GATE_UP = ".mlp.experts.gate_up_proj"
_EXPERT_DOWN = ".mlp.experts.down_proj"


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    quantized = False
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if method is not None and not isinstance(method, UnquantizedLinearMethod):
            quantized = True
    metadata = read_quantization_metadata(path)
    hf_quant = read_hf_fp8_config(path) if metadata is None else None
    if hf_quant is not None:
        if not quantized or not any(hasattr(m, "weight_scale_inv") for m in model.modules()):
            raise ValueError("HF block FP8 checkpoint requires a block FP8 model")
        return _load_fp8_model(model, path, hf_block=True)
    if quantized or metadata is not None:
        if not quantized or metadata is None:
            raise ValueError("FP8 model and converted checkpoint metadata must match")
        return _load_fp8_model(model, path)
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    model_params = dict(model.named_parameters())
    if any(name.endswith(_EXPERT_GATE_UP) for name in model_params):
        return _load_bf16_moe_model(model, model_params, path)
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for checkpoint_name in f.keys():
                # Qwen3.5 multimodal checkpoints nest the text backbone below
                # ``model.language_model`` while nano_qwen exposes it as
                # ``model``.  The visual and MTP weights are intentionally
                # outside this text-only model and must be ignored.
                weight_name = checkpoint_name
                if weight_name.startswith("model.language_model."):
                    weight_name = "model." + weight_name[len("model.language_model."):]
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model_params.get(param_name)
                        if param is None:
                            break
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(checkpoint_name), shard_id)
                        break
                else:
                    param = model_params.get(weight_name)
                    if param is None:
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(checkpoint_name))


def _moe_checkpoint_spec(model, params):
    spec = checkpoint_spec(model)
    for name, param in params.items():
        if name.endswith(_EXPERT_GATE_UP):
            width = param.shape[1] // 2
            projections = (("gate_proj", slice(0, width)), ("up_proj", slice(width, None)))
        elif name.endswith(_EXPERT_DOWN):
            projections = (("down_proj", slice(None)),)
        else:
            continue

        del spec[name]
        prefix = name.rsplit(".", 1)[0]
        for expert_id in range(param.shape[0]):
            for projection, rows in projections:
                index = (expert_id, rows)
                key = f"{prefix}.{expert_id}.{projection}.weight"
                spec[key] = (name, index, tuple(param[index].shape), param.dtype)
    return spec


def _load_bf16_moe_model(model, params, path):
    spec = _moe_checkpoint_spec(model, params)
    embed_name = "model.embed_tokens.weight"
    head_name = "lm_head.weight"
    is_tied = tied_head(model)
    if is_tied:
        spec[head_name] = spec[embed_name]

    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise ValueError(f"No .safetensors files found in {path}")

    entries = {}
    with ExitStack() as stack:
        # Check every expert slice and text tensor before writing any weights.
        for file in files:
            checkpoint = stack.enter_context(safe_open(file, "pt", "cpu"))
            for raw_name in checkpoint.keys():
                if raw_name.startswith(("model.visual.", "visual.", "model.mtp.", "mtp.")):
                    continue
                name = normalize_weight_name(raw_name)
                if name not in spec:
                    raise ValueError(f"Unexpected MoE checkpoint tensor: {raw_name}")
                if name in entries:
                    raise ValueError(f"Duplicate MoE checkpoint tensor: {raw_name}")
                param_name, _, shape, dtype = spec[name]
                expected_dtypes = ("BF16",)
                # GDN stores A_log in FP32 at runtime, even when serialized as BF16.
                if dtype == torch.float32 and param_name.endswith(".linear_attn.A_log"):
                    expected_dtypes = ("BF16", "F32")
                tensor = checkpoint.get_slice(raw_name)
                actual_shape = tuple(tensor.get_shape())
                actual_dtype = tensor.get_dtype()
                if actual_shape != shape or actual_dtype not in expected_dtypes:
                    raise ValueError(
                        f"Invalid MoE tensor {raw_name}: expected {shape} / {' or '.join(expected_dtypes)}, "
                        f"got {actual_shape} / {actual_dtype}"
                    )
                entries[name] = (checkpoint, raw_name)

        missing = spec.keys() - entries.keys()
        if is_tied:
            missing.discard(head_name)
        if missing:
            raise ValueError(f"Missing {len(missing)} MoE checkpoint tensors: {sorted(missing)[:8]}")
        if is_tied and head_name in entries:
            head, head_key = entries.pop(head_name)
            embedding, embed_key = entries[embed_name]
            if not torch.equal(head.get_tensor(head_key), embedding.get_tensor(embed_key)):
                raise ValueError("Tied embedding and lm_head checkpoint tensors disagree")

        for name, (checkpoint, raw_name) in entries.items():
            param_name, shard, _, _ = spec[name]
            param = params[param_name]
            tensor = checkpoint.get_tensor(raw_name)
            if isinstance(shard, tuple):
                param.data[shard].copy_(tensor)
            else:
                loader = getattr(param, "weight_loader", default_weight_loader)
                if shard is None:
                    loader(param, tensor)
                else:
                    loader(param, tensor, shard)


def _load_fp8_model(model: nn.Module, path: str, hf_block=False):
    spec = checkpoint_spec(model)
    params = dict(model.named_parameters())
    entries = {}
    # Validate the complete schema before copying any weights into the model.
    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as checkpoint:
            for raw_name in checkpoint.keys():
                if hf_block and raw_name.startswith(("model.visual.", "visual.", "model.mtp.", "mtp.")):
                    continue
                name = normalize_weight_name(raw_name)
                if name not in spec or name in entries:
                    raise ValueError(f"Unexpected or duplicate FP8 tensor: {raw_name}")
                _, _, shape, dtype = spec[name]
                tensor = checkpoint.get_tensor(raw_name)
                if hf_block and name.endswith(".weight_scale_inv"):
                    if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                        raise ValueError(f"Invalid block FP8 scale dtype: {raw_name}")
                    tensor = tensor.float()
                validate_tensor(name, tensor, shape, dtype)
                entries[name] = (file, raw_name)
    missing = set(spec) - entries.keys()
    is_tied = tied_head(model)
    if is_tied and "model.embed_tokens.weight" in entries:
        missing.discard("lm_head.weight")
    if missing:
        raise ValueError(f"Missing FP8 checkpoint tensors: {sorted(missing)}")
    if is_tied and "lm_head.weight" in entries:
        tensors = []
        for name in ("lm_head.weight", "model.embed_tokens.weight"):
            file, raw_name = entries[name]
            with safe_open(file, "pt", "cpu") as checkpoint:
                tensors.append(checkpoint.get_tensor(raw_name))
        if not torch.equal(*tensors):
            raise ValueError("Tied embedding and lm_head checkpoint tensors disagree")
    for name, (file, raw_name) in entries.items():
        param_name, shard, _, _ = spec[name]
        param = params[param_name]
        with safe_open(file, "pt", "cpu") as checkpoint:
            tensor = checkpoint.get_tensor(raw_name)
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard is None:
                loader(param, tensor)
            else:
                loader(param, tensor, shard)
