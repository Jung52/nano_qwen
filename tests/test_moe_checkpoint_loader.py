from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from transformers import AutoConfig

from nano_qwen.models.qwen3_5 import Qwen3_5ForCausalLM
from nano_qwen.utils.loader import load_model


EXPERT_UP = "model.language_model.layers.0.mlp.experts.2.up_proj.weight"
EMBEDDING = "model.language_model.embed_tokens.weight"


@pytest.fixture
def moe_checkpoint():
    config = AutoConfig.for_model(
        "qwen3_5_moe_text", hidden_size=128, vocab_size=256,
        num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2, num_key_value_heads=1, head_dim=128,
        linear_num_key_heads=1, linear_num_value_heads=2,
        linear_key_head_dim=128, linear_value_head_dim=128,
        num_experts=3, num_experts_per_tok=2,
        moe_intermediate_size=8, shared_expert_intermediate_size=16,
        max_position_embeddings=128, tie_word_embeddings=True, dtype=torch.bfloat16,
    )
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with patch.object(dist, "get_rank", return_value=0), patch.object(dist, "get_world_size", return_value=1):
            model = Qwen3_5ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)

    expected = {}
    source = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.zero_()
            if name == "lm_head.weight":
                continue
            tensor = torch.randn(param.shape, dtype=torch.bfloat16)
            expected[name] = tensor
            key = name.replace("model.", "model.language_model.", 1)
            if name.endswith(".experts.gate_up_proj"):
                prefix = key.removesuffix(".gate_up_proj")
                for expert_id, expert in enumerate(tensor):
                    gate, up = expert.chunk(2, dim=0)
                    source[f"{prefix}.{expert_id}.gate_proj.weight"] = gate.clone()
                    source[f"{prefix}.{expert_id}.up_proj.weight"] = up.clone()
            elif name.endswith(".experts.down_proj"):
                prefix = key.removesuffix(".down_proj")
                for expert_id, expert in enumerate(tensor):
                    source[f"{prefix}.{expert_id}.down_proj.weight"] = expert.clone()
            elif name.endswith(".shared_expert.gate_up_proj.weight"):
                gate, up = tensor.chunk(2, dim=0)
                source[key.replace("gate_up_proj", "gate_proj")] = gate.clone()
                source[key.replace("gate_up_proj", "up_proj")] = up.clone()
            else:
                source[key] = tensor
    return model, source, expected


@pytest.mark.parametrize("explicit_head", [False, True])
def test_loads_moe_checkpoint_across_shards(tmp_path, moe_checkpoint, explicit_head):
    model, source, expected = moe_checkpoint
    if explicit_head:
        source["lm_head.weight"] = source[EMBEDDING].clone()
    entries = list(source.items())
    save_file(dict(entries[::2]), tmp_path / "model-00001-of-00002.safetensors")
    save_file(dict(entries[1::2]), tmp_path / "model-00002-of-00002.safetensors")

    load_model(model, str(tmp_path))

    params = dict(model.named_parameters())
    for name, tensor in expected.items():
        torch.testing.assert_close(params[name], tensor.to(params[name].dtype), rtol=0, atol=0)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


@pytest.mark.parametrize("fault", ["missing", "shape", "dtype"])
def test_rejects_invalid_expert_slice_before_copying(tmp_path, moe_checkpoint, fault):
    model, source, _ = moe_checkpoint
    expert = source.pop(EXPERT_UP)
    if fault == "shape":
        source[EXPERT_UP] = expert[:1].clone()
    elif fault == "dtype":
        source[EXPERT_UP] = expert.float()
    save_file(source, tmp_path / "model.safetensors")

    match = "Missing.*up_proj" if fault == "missing" else "Invalid MoE tensor.*up_proj"
    with pytest.raises(ValueError, match=match):
        load_model(model, str(tmp_path))
    assert all(param.count_nonzero().item() == 0 for param in model.parameters())


def test_rejects_duplicate_normalized_keys(tmp_path, moe_checkpoint):
    model, source, _ = moe_checkpoint
    save_file(source, tmp_path / "model-00001-of-00002.safetensors")
    alias = EXPERT_UP.replace("model.language_model.", "model.", 1)
    save_file({alias: source[EXPERT_UP]}, tmp_path / "model-00002-of-00002.safetensors")

    with pytest.raises(ValueError, match="Duplicate MoE checkpoint tensor.*up_proj"):
        load_model(model, str(tmp_path))


def test_rejects_conflicting_tied_head(tmp_path, moe_checkpoint):
    model, source, _ = moe_checkpoint
    source["lm_head.weight"] = torch.zeros_like(source[EMBEDDING])
    save_file(source, tmp_path / "model.safetensors")

    with pytest.raises(ValueError, match="Tied embedding and lm_head checkpoint tensors disagree"):
        load_model(model, str(tmp_path))
