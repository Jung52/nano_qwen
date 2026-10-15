"""An FP8 target can load its unquantized MTP head from a separate checkpoint."""

import json
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from transformers import AutoConfig

from nano_qwen.config import Config
from nano_qwen.models.qwen3_5_mtp import Qwen3_5MTP, load_mtp_weights
from nano_qwen.quantization.checkpoint import FP8_METADATA, METADATA_FILE


def config():
    value = AutoConfig.for_model("qwen3_5_text")
    for name, item in dict(hidden_size=128, intermediate_size=256, vocab_size=256,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=64, linear_num_key_heads=1, linear_num_value_heads=2,
        linear_key_head_dim=128, linear_value_head_dim=128,
        layer_types=["linear_attention", "full_attention"], mtp_num_hidden_layers=1).items():
        setattr(value, name, item)
    return value


def checkpoint_pair(tmp_path):
    target, source = tmp_path / "fp8", tmp_path / "original"
    target.mkdir(); source.mkdir()
    config().save_pretrained(target)
    config().save_pretrained(source)
    (target / METADATA_FILE).write_text(json.dumps(FP8_METADATA))
    return target, source


def test_separate_mtp_model_is_accepted_with_fp8_target(tmp_path):
    target, source = checkpoint_pair(tmp_path)
    value = Config(str(target), enable_mtp=True, max_num_seqs=1, mtp_model=str(source))
    assert value.quantization == "fp8"
    assert value.mtp_model == str(source)


@pytest.mark.parametrize("field,value", [("hidden_size", 256), ("vocab_size", 512),
                                        ("num_attention_heads", 4), ("mtp_num_hidden_layers", 2),
                                        ("linear_num_value_heads", 4),
                                        ("layer_types", ["full_attention"] * 2)])
def test_separate_mtp_model_rejects_incompatible_architecture(tmp_path, field, value):
    target, source = checkpoint_pair(tmp_path)
    other = config()
    setattr(other, field, value)
    other.save_pretrained(source)
    with pytest.raises(ValueError, match="MTP.*" + field):
        Config(str(target), enable_mtp=True, max_num_seqs=1, mtp_model=str(source))


@pytest.mark.parametrize("field,value", [("rope_theta", 10000.), ("partial_rotary_factor", .5)])
def test_separate_mtp_model_checks_effective_rope_fallback(tmp_path, field, value):
    target, source = checkpoint_pair(tmp_path)
    for path in (target, source):
        data = json.loads((path / "config.json").read_text())
        data["rope_parameters"] = {}
        data["rope_theta"] = 1e7
        data["partial_rotary_factor"] = .25
        if path == source:
            data[field] = value
        (path / "config.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="MTP.*RoPE"):
        Config(str(target), enable_mtp=True, max_num_seqs=1, mtp_model=str(source))


def test_separate_mtp_model_rejects_missing_path(tmp_path):
    target, _ = checkpoint_pair(tmp_path)
    with pytest.raises(ValueError, match="MTP"):
        Config(str(target), enable_mtp=True, max_num_seqs=1, mtp_model=str(tmp_path / "missing"))


def test_mtp_weight_loading_uses_original_weights_and_rejects_raw_fp8(tmp_path):
    with patch.object(dist, "is_initialized", return_value=True), \
         patch.object(dist, "get_rank", return_value=0), \
         patch.object(dist, "get_world_size", return_value=1):
        module = Qwen3_5MTP(config())
    tensors = {}
    for name, parameter in module.named_parameters():
        value = torch.full_like(parameter, .125)
        if ".gate_up_proj." in name:
            a, b = value.chunk(2, dim=0)
            tensors["mtp." + name.replace("gate_up_proj", "gate_proj")] = a.clone()
            tensors["mtp." + name.replace("gate_up_proj", "up_proj")] = b.clone()
        else:
            tensors["mtp." + name] = value
    save_file(tensors, str(tmp_path / "model.safetensors"))
    load_mtp_weights(module, tmp_path)
    for parameter in module.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, .125), rtol=0, atol=0)
    key = next(iter(tensors))
    tensors[key] = tensors[key].to(torch.float8_e4m3fn)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="MTP.*(BF16|unquantized)"):
        load_mtp_weights(module, tmp_path)


@pytest.mark.parametrize("allow_drift", [False, True])
def test_production_benchmark_cannot_report_success_after_token_mismatch(monkeypatch, capsys, allow_drift):
    import sys
    import bench_qwen3_5_mtp_production as bench
    monkeypatch.setattr(bench, "make_prompt", lambda *a: [2, 3])
    def run(model, enable_mtp, *args, **kwargs):
        data = dict(tokens=2, elapsed_s=1., tokens_per_s=2., stop_reason="max_tokens", windows=[],
                    accepted_tokens=1, verify_steps=1)
        return data, [2, 5 if enable_mtp else 3, 4]
    monkeypatch.setattr(bench, "run_mode", run)
    monkeypatch.setattr(sys, "argv", ["bench", "--model", "unused"] +
                        (["--allow-numerical-drift"] if allow_drift else []))
    with pytest.raises(AssertionError, match="diverg"):
        bench.main()
    assert "RESULT speedup=" not in capsys.readouterr().out
