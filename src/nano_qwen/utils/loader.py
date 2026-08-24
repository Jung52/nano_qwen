import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
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
