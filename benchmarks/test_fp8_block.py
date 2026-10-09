"""HF block FP8 loading on CPU; kernel math/graph checks on CUDA SM89+."""

import json
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
from safetensors.torch import save_file
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nano_qwen.config import Config
from nano_qwen.quantization import get_quantization_config, UnquantizedLinearMethod
from nano_qwen.quantization.checkpoint import (
    checkpoint_spec, FP8_METADATA, METADATA_FILE, read_hf_fp8_config,
)
from nano_qwen.quantization.fp8_block import (
    Fp8BlockConfig, Fp8BlockLinearMethod, quantize_fp8_per_group,
)
from nano_qwen.utils.loader import load_model


# Load the actual linear.py in isolation: layers/__init__.py eagerly imports
# FlashAttention, which the CPU checkpoint tests do not need.
_spec = importlib.util.spec_from_file_location(
    "_nano_qwen_test_linear", Path(__file__).resolve().parents[1] / "src/nano_qwen/layers/linear.py",
)
_linear = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_linear)
MergedColumnParallelLinear = _linear.MergedColumnParallelLinear
ReplicatedLinear = _linear.ReplicatedLinear
RowParallelLinear = _linear.RowParallelLinear


HF_QUANT = {"quant_method": "fp8", "activation_scheme": "dynamic",
            "weight_block_size": [128, 128], "modules_to_not_convert": []}


@pytest.fixture(autouse=True)
def local_layer_environment():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with patch.object(dist, "get_rank", return_value=0), patch.object(
            dist, "get_world_size", return_value=1,
        ):
            yield
    finally:
        torch.set_default_dtype(previous)


class TinyModel(nn.Module):
    def __init__(self, quant):
        super().__init__()
        self.packed_modules_mapping = {"gate_proj": ("gate_up_proj", 0),
                                       "up_proj": ("gate_up_proj", 1)}
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(256, 256)
        self.model.mlp = nn.Module()
        self.model.mlp.gate_up_proj = MergedColumnParallelLinear(
            256, [128, 256], quant_config=quant, prefix="model.mlp.gate_up_proj",
        )
        self.model.mlp.down_proj = RowParallelLinear(
            256, 256, quant_config=quant, prefix="model.mlp.down_proj",
        )
        self.lm_head = nn.Linear(256, 256, bias=False)
        self.lm_head.weight.data = self.model.embed_tokens.weight.data


def checkpoint_tensors(model, hf=True):
    tensors = {}
    for name, (_, _, shape, dtype) in checkpoint_spec(model).items():
        if name == "lm_head.weight":
            continue
        if name.endswith(("weight_scale", "weight_scale_inv")):
            # Match lovedheart's on-disk BF16 block scales.
            dtype = torch.bfloat16 if hf else torch.float32
            value = (torch.rand(shape, dtype=torch.float32) * 0.02 + 0.001).to(dtype)
        else:
            value = torch.randn(shape, dtype=torch.float32).to(dtype)
        if hf:
            name = name.replace("model.", "model.language_model.", 1)
        tensors[name] = value
    return tensors


def write_hf_config(path, quant=HF_QUANT):
    (path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_text", "max_position_embeddings": 1024,
        "quantization_config": quant,
    }))


def test_config_autodetection_and_legacy(tmp_path):
    write_hf_config(tmp_path)
    config = Config(str(tmp_path))
    assert config.quantization == "fp8"
    assert isinstance(config.quant_config, Fp8BlockConfig)
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        Config(str(tmp_path), tensor_parallel_size=2)
    write_hf_config(tmp_path, {**HF_QUANT, "weight_block_size": [64, 128]})
    with pytest.raises(ValueError, match="weight_block_size"):
        read_hf_fp8_config(tmp_path)
    (tmp_path / METADATA_FILE).write_text(json.dumps(FP8_METADATA))
    # The original nano_qwen metadata keeps precedence over copied HF config.
    assert type(Config(str(tmp_path)).quant_config).__name__ == "Fp8Config"


def test_excluded_fused_projections():
    ignored = ["model.language_model.mlp.gate_proj", "model.language_model.mlp.up_proj"]
    model = TinyModel(Fp8BlockConfig({**HF_QUANT, "modules_to_not_convert": ignored}))
    assert isinstance(model.model.mlp.gate_up_proj.quant_method, UnquantizedLinearMethod)
    assert isinstance(model.model.mlp.down_proj.quant_method, Fp8BlockLinearMethod)
    with pytest.raises(ValueError, match="same quantization"):
        TinyModel(Fp8BlockConfig({**HF_QUANT, "modules_to_not_convert": ignored[:1]}))


def test_hf_loads_bf16_block_scales_in_packed_order(tmp_path):
    model = TinyModel(Fp8BlockConfig(HF_QUANT))
    tensors = checkpoint_tensors(model)
    tensors["model.visual.weight"] = torch.ones(1)
    tensors["mtp.layers.0.weight"] = torch.ones(1)
    write_hf_config(tmp_path)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    load_model(model, str(tmp_path))
    params = dict(model.named_parameters())
    for name, (target, shard, shape, _) in checkpoint_spec(model).items():
        if name == "lm_head.weight":
            continue
        actual = params[target]
        if shard is not None:
            # Unequal gate/up sizes catch offsets accidentally measured in tokens.
            offset = 0 if shard == 0 else (1 if name.endswith("weight_scale_inv") else 128)
            actual = actual.narrow(0, offset, shape[0])
        source = name.replace("model.", "model.language_model.", 1)
        torch.testing.assert_close(actual.float(), tensors[source].float(), rtol=0, atol=0)
    assert model.model.mlp.gate_up_proj.weight_scale_inv.shape == (3, 2)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    with pytest.raises(ValueError, match="block FP8 model"):
        load_model(TinyModel(get_quantization_config("fp8")), str(tmp_path))


@pytest.mark.parametrize("corruption", ["missing", "zero", "scale_dtype", "weight_dtype", "text_extra", "duplicate"])
def test_hf_rejects_corrupted_checkpoint(tmp_path, corruption):
    model = TinyModel(Fp8BlockConfig(HF_QUANT))
    tensors = checkpoint_tensors(model)
    scale = "model.language_model.mlp.gate_proj.weight_scale_inv"
    if corruption == "missing":
        tensors.pop(scale)
    elif corruption == "zero":
        tensors[scale].zero_()
    elif corruption == "scale_dtype":
        tensors[scale] = tensors[scale].to(torch.int32)
    elif corruption == "weight_dtype":
        key = scale.removesuffix("weight_scale_inv") + "weight"
        tensors[key] = tensors[key].to(torch.bfloat16)
    elif corruption == "text_extra":
        tensors["model.language_model.unexpected.weight"] = torch.ones(1)
    elif corruption == "duplicate":
        save_file({scale: tensors[scale]}, str(tmp_path / "duplicate.safetensors"))
    write_hf_config(tmp_path)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError):
        load_model(model, str(tmp_path))


def test_legacy_per_channel_loader_still_works(tmp_path):
    model = TinyModel(get_quantization_config("fp8"))
    tensors = checkpoint_tensors(model, hf=False)
    (tmp_path / METADATA_FILE).write_text(json.dumps(FP8_METADATA))
    save_file(tensors, str(tmp_path / "model.safetensors"))
    load_model(model, str(tmp_path))
    torch.testing.assert_close(model.model.mlp.down_proj.weight_scale,
                               tensors["model.mlp.down_proj.weight_scale"], rtol=0, atol=0)


def require_fp8_cuda():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("requires CUDA SM89+ hardware and a CUDA-enabled PyTorch")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_quantization_preserves_fp32_values_above_fp8_midpoint(dtype):
    require_fp8_cuda()
    x = torch.zeros(1, 256, device="cuda", dtype=dtype)
    x[0, :4] = torch.tensor([100, -100, 386, -386], device="cuda", dtype=dtype)
    q, scales = quantize_fp8_per_group(x)
    # 100 / (386 / 448) is about 116.062, above the 112/120 midpoint.
    # Truncating it to FP16 first produces exactly 116 and wrongly rounds to 112.
    torch.testing.assert_close(
        q[0, :4].float(), torch.tensor([120, -120, 448, -448], device="cuda", dtype=torch.float32),
        rtol=0, atol=0,
    )
    assert not q[0, 4:].float().count_nonzero()
    torch.testing.assert_close(
        scales, torch.tensor([[386 / 448, 1]], device="cuda", dtype=torch.float32),
        rtol=1e-6, atol=0,
    )


@pytest.mark.parametrize("batch", [1, 17, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_kernel_against_explicit_group_dequantization(batch, dtype):
    require_fp8_cuda()
    torch.manual_seed(17)
    layer = ReplicatedLinear(256, 384, bias=True, quant_config=Fp8BlockConfig(HF_QUANT)).cuda()
    layer._compute_dtype = dtype
    with torch.no_grad():
        layer.weight.copy_(torch.randn(384, 256, device="cuda", dtype=torch.float32))
        layer.weight_scale_inv.copy_(torch.tensor(
            [[0.001, 0.03], [0.002, 0.006], [0.04, 0.005]], device="cuda", dtype=torch.float32,
        ))
        layer.bias.zero_()
    x = torch.randn(batch, 256, device="cuda", dtype=dtype)
    x[:, :128] *= 0.001
    x[:, 128:] *= 30
    x[0].zero_()
    q, scales = quantize_fp8_per_group(x)
    blocks = x.float().reshape(batch, 2, 128)
    reference_scales = blocks.abs().amax(-1) / 448
    reference_scales[reference_scales == 0] = 1
    reference_q = (blocks / reference_scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.testing.assert_close(scales, reference_scales, rtol=1e-6, atol=0)
    torch.testing.assert_close(q.float(), reference_q.reshape_as(q).float(), rtol=0, atol=0)
    x_dequant = q.float() * scales.repeat_interleave(128, 1)
    w_dequant = layer.weight.float() * layer.weight_scale_inv.repeat_interleave(128, 0).repeat_interleave(128, 1)
    reference = F.linear(x_dequant, w_dequant).to(dtype)
    actual = layer(x)
    assert actual.dtype == dtype
    error = (actual.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12)
    assert error < 0.004
    torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)


def test_cuda_graph_replay():
    require_fp8_cuda()
    layer = ReplicatedLinear(256, 256, quant_config=Fp8BlockConfig(HF_QUANT)).cuda()
    with torch.no_grad():
        layer.weight.copy_(torch.randn(256, 256, device="cuda"))
        layer.weight_scale_inv.fill_(0.01)
    x = torch.randn(8, 256, device="cuda")
    for _ in range(3):
        layer(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = layer(x)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.testing.assert_close(output, layer(x), rtol=0, atol=0)
