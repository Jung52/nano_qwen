"""One-step Qwen3.5 MTP head used by the production speculative decoder.

Usage after loading the target with ModelRunner (TP=1)::

    mtp = Qwen3_5MTP(runner.config.hf_config).eval()
    load_mtp_weights(mtp, runner.config.model)
    # With a causal prefill Context, target_hidden[i] predicts token[i+1]
    # when MTP receives token[i+1] at position i+1.
    draft_hidden = mtp(shifted_ids, shifted_positions, target_hidden, runner.model.model.embed_tokens)
    draft_logits = runner.model.compute_logits(draft_hidden)

ModelRunner owns the MTP KV pool and lifecycle. The scheduler appends one
greedy draft, the target verifies two rows, and rejected GDN state is rolled
back to the state after the first (authoritative) token.
"""

from copy import copy
from glob import glob
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from safetensors import safe_open

from nano_qwen.layers.layernorm import GemmaRMSNorm
from nano_qwen.models.qwen3_5 import Qwen3_5DecoderLayer
from nano_qwen.utils.loader import default_weight_loader


class Qwen3_5MTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not dist.is_initialized() or dist.get_world_size() != 1:
            raise ValueError("This MTP baseline requires initialized TP=1")
        if getattr(config, "mtp_num_hidden_layers", 0) != 1:
            raise ValueError("Only Qwen3.5 9B's one MTP layer is supported")
        if getattr(config, "mtp_use_dedicated_embeddings", False):
            raise ValueError("Dedicated MTP embeddings are unsupported")

        h = config.hidden_size
        self.pre_fc_norm_embedding = GemmaRMSNorm(h, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(h, eps=config.rms_norm_eps)
        self.fc = nn.Linear(2 * h, h, bias=False)
        # Qwen3_5DecoderLayer normally indexes config.layer_types. Append a
        # full-attention entry on a shallow copy; do not mutate target config.
        mtp_config = copy(config)
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            mtp_config.layer_types = list(layer_types) + ["full_attention"]
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(mtp_config, config.num_hidden_layers)
        ])
        if self.layers[0].block_type != "full_attention":
            raise ValueError("MTP layer must use full attention")
        self.norm = GemmaRMSNorm(h, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, target_hidden, shared_embedding):
        """Predict the token after `input_ids` from aligned target states.

        Input row i uses the target's hidden state at position i and the
        *actual next token* at position i+1. Caller supplies causal Context.
        """
        if input_ids.ndim != 1 or positions.shape != input_ids.shape:
            raise ValueError("input_ids and positions must be packed 1-D tensors")
        if target_hidden.shape != (input_ids.numel(), self.fc.out_features):
            raise ValueError("target_hidden must have one row per MTP input")
        embeddings = self.pre_fc_norm_embedding(shared_embedding(input_ids))
        hidden = self.pre_fc_norm_hidden(target_hidden)
        hidden = self.fc(torch.cat((embeddings, hidden), dim=-1))
        hidden, residual = self.layers[0](positions, hidden, None)
        hidden, _ = self.norm(hidden, residual)
        return hidden


def load_mtp_weights(module: Qwen3_5MTP, model_path: str | Path) -> None:
    """Load all `mtp.*` tensors, including both halves of packed gate/up.

    `nano_qwen.utils.loader.load_model` intentionally ignores `mtp.*`; call
    this once after the target model is loaded. Fails on missing/extra keys.
    """
    parameters = dict(module.named_parameters())
    expected = set()
    for name in parameters:
        if ".gate_up_proj." in name:
            expected.add(name.replace("gate_up_proj", "gate_proj"))
            expected.add(name.replace("gate_up_proj", "up_proj"))
        else:
            expected.add(name)

    seen = set()
    files = sorted(glob(str(Path(model_path) / "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors weights in {model_path}")
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if not key.startswith("mtp."):
                    continue
                name = key.removeprefix("mtp.")
                if name not in expected or name in seen:
                    raise ValueError(f"Unexpected or duplicate MTP tensor: {key}")
                seen.add(name)
                shard = None
                if ".gate_proj." in name:
                    name = name.replace("gate_proj", "gate_up_proj")
                    shard = 0
                elif ".up_proj." in name:
                    name = name.replace("up_proj", "gate_up_proj")
                    shard = 1
                param = parameters[name]
                weight = checkpoint.get_tensor(key)
                loader = getattr(param, "weight_loader", default_weight_loader)
                if shard is None:
                    if param.shape != weight.shape:
                        raise ValueError(f"Shape mismatch for {key}: {weight.shape} vs {param.shape}")
                    loader(param, weight)
                else:
                    loader(param, weight, shard)

    if missing := sorted(expected - seen):
        raise ValueError(f"Missing MTP tensors: {missing}")
