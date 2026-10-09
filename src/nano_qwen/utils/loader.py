import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

from nano_qwen.quantization import UnquantizedLinearMethod
from nano_qwen.quantization.checkpoint import (
    checkpoint_spec, normalize_weight_name, read_quantization_metadata, read_hf_fp8_config,
    tied_head, validate_tensor,
)


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
