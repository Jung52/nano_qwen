"""Quantization interfaces; run with pytest in the existing WSL CUDA env.

Includes versioned checkpoint loading and packed weight/scale validation.
"""

from pathlib import Path
import json
import sys
from unittest.mock import patch
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nano_qwen.config import Config
from nano_qwen.engine.cuda_graph import CudaGraphManager
from nano_qwen.layers.attention import Attention
from nano_qwen.layers.gated_delta_net import GatedDeltaNet
from nano_qwen.layers.linear import (
    ColumnParallelLinear, LinearBase, MergedColumnParallelLinear,
    QKVParallelLinear, ReplicatedLinear, RowParallelLinear,
)
from nano_qwen.models.qwen3_5 import Qwen3_5ForCausalLM
from nano_qwen.quantization import get_quantization_config, UnquantizedLinearMethod
from nano_qwen.quantization.fp8 import (
    Fp8Config, Fp8LinearMethod, quantize_fp8_per_token, stable_gate_projection,
)
from nano_qwen.quantization.checkpoint import FP8_METADATA, METADATA_FILE, checkpoint_spec
from nano_qwen.utils.loader import load_model
from nano_qwen.utils.context import set_context, reset_context


@pytest.fixture(autouse=True)
def local_rank():
    # Layer tests isolate local TP storage/math; NCCL communication is not tested.
    with patch.object(dist, "get_rank", return_value=0), patch.object(
        dist, "get_world_size", return_value=1,
    ):
        yield


@pytest.fixture
def bf16_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def text_config():
    config = AutoConfig.for_model("qwen3_5_text")
    config.hidden_size = 128
    config.intermediate_size = 256
    config.vocab_size = 256
    config.tie_word_embeddings = True
    config.num_hidden_layers = 2
    config.num_attention_heads = 2
    config.num_key_value_heads = 1
    config.head_dim = 64
    config.linear_num_key_heads = 1
    config.linear_num_value_heads = 2
    config.linear_key_head_dim = 128
    config.linear_value_head_dim = 128
    config.layer_types = ["linear_attention", "full_attention"]
    return config


def test_config_and_unsupported_combinations(tmp_path):
    text_config().save_pretrained(tmp_path)
    assert Config(str(tmp_path)).quant_config is None
    with pytest.raises(ValueError, match="converted"):
        Config(str(tmp_path), quantization="fp8")
    (tmp_path / METADATA_FILE).write_text(json.dumps(FP8_METADATA))
    assert Config(str(tmp_path)).quantization == "fp8"
    assert isinstance(Config(str(tmp_path), quantization="fp8").quant_config, Fp8Config)
    with pytest.raises(ValueError, match="Unsupported quantization"):
        Config(str(tmp_path), quantization="int8")
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        Config(str(tmp_path), quantization="fp8", tensor_parallel_size=2)
    (tmp_path / METADATA_FILE).write_text(json.dumps({**FP8_METADATA, "format": "other"}))
    with pytest.raises(ValueError, match="Unsupported"):
        Config(str(tmp_path))


@pytest.mark.parametrize("rank", [0, 1])
def test_bf16_tp_sharding_and_bias(rank, bf16_default):
    torch.manual_seed(11)
    x = torch.randn(3, 16)
    weight = torch.randn(32, 16)
    bias = torch.randn(32)
    with patch.object(dist, "get_rank", return_value=rank), patch.object(
        dist, "get_world_size", return_value=2,
    ):
        column = ColumnParallelLinear(16, 32, bias=True)
        column.weight_loader(column.weight, weight)
        column.weight_loader(column.bias, bias)
        expected = F.linear(x, weight.chunk(2)[rank], bias.chunk(2)[rank])
        torch.testing.assert_close(column(x), expected, rtol=0, atol=0)

        row = RowParallelLinear(16, 32, bias=True)
        row.weight_loader(row.weight, weight)
        row.weight_loader(row.bias, bias)
        local_x = x.chunk(2, dim=1)[rank]
        expected_local = F.linear(
            local_x, weight.chunk(2, dim=1)[rank], bias if rank == 0 else None,
        )
        with patch.object(dist, "all_reduce") as reduce:
            actual = row(local_x)
            reduce.assert_called_once()
        torch.testing.assert_close(actual, expected_local, rtol=0, atol=0)

        merged = MergedColumnParallelLinear(16, [16, 32])
        shards = [torch.randn(16, 16), weight]
        for shard_id, shard in enumerate(shards):
            merged.weight_loader(merged.weight, shard, shard_id)
        expected_weight = torch.cat([w.chunk(2)[rank] for w in shards])
        torch.testing.assert_close(merged.weight, expected_weight, rtol=0, atol=0)
        torch.testing.assert_close(merged(x), F.linear(x, expected_weight), rtol=0, atol=0)

        qkv = QKVParallelLinear(16, 4, 4, 2)
        shards = [torch.randn(n, 16) for n in (16, 8, 8)]
        for name, shard in zip(("q", "k", "v"), shards):
            qkv.weight_loader(qkv.weight, shard, name)
        expected_weight = torch.cat([w.chunk(2)[rank] for w in shards])
        torch.testing.assert_close(qkv.weight, expected_weight, rtol=0, atol=0)
        torch.testing.assert_close(qkv(x), F.linear(x, expected_weight), rtol=0, atol=0)


def test_bf16_loader_preserves_names_and_packed_order(tmp_path, bf16_default):
    model = Qwen3_5ForCausalLM(text_config())
    params = dict(model.named_parameters())
    checkpoint = {}
    for name, param in params.items():
        if ".gate_up_proj." in name:
            gate, up = torch.randn_like(param).chunk(2)
            checkpoint[name.replace("gate_up_proj", "gate_proj")] = gate.contiguous()
            checkpoint[name.replace("gate_up_proj", "up_proj")] = up.contiguous()
        else:
            checkpoint[name] = torch.randn_like(param)
    checkpoint = {
        name.replace("model.", "model.language_model.", 1): value
        for name, value in checkpoint.items()
    }
    checkpoint["lm_head.weight"] = checkpoint["model.language_model.embed_tokens.weight"].clone()
    save_file(checkpoint, str(tmp_path / "model.safetensors"))
    load_model(model, str(tmp_path))
    for name, param in params.items():
        key = name.replace("model.", "model.language_model.", 1)
        if ".gate_up_proj." in name:
            expected = torch.cat([
                checkpoint[key.replace("gate_up_proj", part)]
                for part in ("gate_proj", "up_proj")
            ])
        else:
            expected = checkpoint[key]
        torch.testing.assert_close(param, expected, rtol=0, atol=0)


def test_fp8_model_scope_and_loading_guard(tmp_path, bf16_default):
    model = Qwen3_5ForCausalLM(text_config(), quant_config=get_quantization_config("fp8"))
    assert model.model.layers[1].self_attn.attn.decode_num_splits == 1
    assert Qwen3_5ForCausalLM(text_config()).model.layers[1].self_attn.attn.decode_num_splits == 0
    for name, layer in model.named_modules():
        if isinstance(layer, LinearBase):
            assert isinstance(layer.quant_method, Fp8LinearMethod), name
            assert layer.prefix == name
            assert layer.weight.dtype == torch.float8_e4m3fn
            assert layer.weight_scale.dtype == torch.float32
            assert layer.compute_dtype == torch.bfloat16
    gdn = model.model.layers[0].linear_attn
    for layer in (gdn.in_proj_a, gdn.in_proj_b, gdn.conv1d,
                  model.model.embed_tokens, model.lm_head):
        assert layer.weight.dtype == torch.bfloat16
    assert gdn.A_log.dtype == torch.float32
    with pytest.raises(ValueError, match="metadata must match"):
        load_model(model, str(tmp_path))
    with pytest.raises(ValueError, match="divisible by 16"):
        ReplicatedLinear(17, 32, quant_config=Fp8Config())


def fp8_checkpoint(model):
    torch.manual_seed(7)
    tensors = {}
    for name, (_, _, shape, dtype) in checkpoint_spec(model).items():
        if name.endswith("weight_scale"):
            tensors[name] = torch.rand(shape, dtype=dtype) + 0.01
        else:
            tensors[name] = torch.randn(shape).to(dtype)
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
    return tensors


def test_fp8_checkpoint_loads_packed_scales_and_weights(tmp_path, bf16_default):
    model = Qwen3_5ForCausalLM(text_config(), quant_config=Fp8Config())
    tensors = fp8_checkpoint(model)
    # Tied head can be omitted, and source HF text-prefix names are accepted.
    tensors.pop("lm_head.weight")
    save_file({key.replace("model.", "model.language_model.", 1): value
               for key, value in tensors.items()}, str(tmp_path / "model.safetensors"))
    (tmp_path / METADATA_FILE).write_text(json.dumps(FP8_METADATA))
    load_model(model, str(tmp_path))
    params = dict(model.named_parameters())
    for key, (target, shard, shape, _) in checkpoint_spec(model).items():
        if key == "lm_head.weight":
            continue
        actual = params[target] if shard is None else params[target].narrow(0, shard * shape[0], shape[0])
        torch.testing.assert_close(actual.float(), tensors[key].float(), rtol=0, atol=0)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


@pytest.mark.parametrize("error", ["missing_scale", "negative_scale", "nonfinite", "wrong_dtype", "wrong_shape", "extra", "duplicate", "tied_head"])
def test_fp8_checkpoint_rejects_corruption(tmp_path, bf16_default, error):
    model = Qwen3_5ForCausalLM(text_config(), quant_config=Fp8Config())
    tensors = fp8_checkpoint(model)
    scale_name = "model.layers.0.mlp.up_proj.weight_scale"
    if error == "missing_scale":
        tensors.pop(scale_name)
    elif error == "negative_scale":
        tensors[scale_name].fill_(-1)
    elif error == "nonfinite":
        tensors[scale_name].fill_(float("nan"))
    elif error == "wrong_dtype":
        tensors[scale_name] = tensors[scale_name].to(torch.bfloat16)
    elif error == "wrong_shape":
        tensors[scale_name] = tensors[scale_name].flatten()
    elif error == "extra":
        tensors["unexpected.weight"] = torch.ones(1)
    elif error == "duplicate":
        tensors[scale_name.replace("model.", "model.language_model.", 1)] = tensors[scale_name].clone()
    else:
        tensors["lm_head.weight"].add_(1)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / METADATA_FILE).write_text(json.dumps(FP8_METADATA))
    with pytest.raises(ValueError):
        load_model(model, str(tmp_path))


def test_offline_conversion_roundtrip(tmp_path, bf16_default):
    from nano_qwen.quantization.convert import convert_checkpoint

    source, output = tmp_path / "bf16", tmp_path / "fp8"
    source.mkdir()
    config = text_config()
    config.dtype = torch.bfloat16
    config.save_pretrained(source)
    original = Qwen3_5ForCausalLM(config)
    raw = {name: torch.randn(shape, dtype=dtype)
           for name, (_, _, shape, dtype) in checkpoint_spec(original).items()
           if name != "lm_head.weight"}
    weight_name = "model.layers.0.mlp.gate_proj.weight"
    raw[weight_name][0].zero_()
    raw["model.visual.unused"] = torch.ones(1)
    save_file(raw, str(source / "model.safetensors"))
    (source / "chat_template.jinja").write_text("test template")
    metadata = convert_checkpoint(source, output, device="cpu", shard_mib=1)
    assert metadata["quantized_projections"] == 13
    assert (output / "chat_template.jinja").read_text() == "test template"
    quantized = Qwen3_5ForCausalLM(config, quant_config=Fp8Config())
    load_model(quantized, str(output))
    layer = quantized.model.layers[0].mlp.gate_up_proj
    assert layer.weight_scale[0].item() == 1.0
    assert torch.count_nonzero(layer.weight[0].float()) == 0
    reference = torch.cat([raw[weight_name], raw[weight_name.replace("gate_proj", "up_proj")]])
    restored = layer.weight.float() * layer.weight_scale
    assert ((restored - reference.float()).norm() / reference.float().norm()).item() < 0.04
    torch.testing.assert_close(quantized.model.layers[0].linear_attn.in_proj_a.weight,
                               raw["model.layers.0.linear_attn.in_proj_a.weight"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="empty"):
        convert_checkpoint(source, output)


@pytest.mark.parametrize("source_layout", ["text", "multimodal"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_offline_block_conversion_roundtrip(tmp_path, bf16_default, source_layout, device):
    from nano_qwen.quantization.convert import convert_checkpoint
    from nano_qwen.quantization.fp8_block import Fp8BlockConfig

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    source, output = tmp_path / "bf16", tmp_path / "block_fp8"
    source.mkdir()
    config = text_config()
    config.hidden_size = 256
    config.head_dim = 128
    config.dtype = torch.bfloat16
    source_config = (AutoConfig.for_model("qwen3_5", text_config=config.to_dict())
                     if source_layout == "multimodal" else config)
    source_config.save_pretrained(source)
    original = Qwen3_5ForCausalLM(config)
    raw = {name: torch.randn(shape, dtype=dtype)
           for name, (_, _, shape, dtype) in checkpoint_spec(original).items()
           if name != "lm_head.weight"}
    weight_name = "model.layers.0.mlp.gate_proj.weight"
    # Four independently scaled blocks, including a zero block and negatives.
    raw[weight_name][:128, :128] = 0
    raw[weight_name][:128, 128:] = 448
    raw[weight_name][128:, :128] = 896
    raw[weight_name][128:, 128:] = -1344
    raw["model.visual.unused"] = torch.ones(1)
    source_tensors = ({key.replace("model.", "model.language_model.", 1): value
                       for key, value in raw.items()}
                      if source_layout == "multimodal" else raw)
    save_file(source_tensors, str(source / "model.safetensors"))
    (source / "chat_template.jinja").write_text("test template")

    metadata = convert_checkpoint(source, output, device=device, shard_mib=1,
                                  scheme="block_wise")
    detected = Config(str(output))
    assert isinstance(detected.quant_config, Fp8BlockConfig)
    assert metadata["quantized_projections"] == 13
    assert not (output / METADATA_FILE).exists()
    assert (output / "chat_template.jinja").read_text() == "test template"
    assert (output / "model.safetensors.index.json").is_file()
    quantized = Qwen3_5ForCausalLM(config, quant_config=detected.quant_config)
    load_model(quantized, str(output))
    layer = quantized.model.layers[0].mlp.gate_up_proj
    # CUDA constant division may differ from the CPU scale by one FP32 ULP.
    torch.testing.assert_close(layer.weight_scale_inv[:2],
                               torch.tensor([[1., 1.], [2., 3.]], dtype=torch.float32),
                               rtol=1e-6, atol=0)
    assert layer.weight_scale_inv[0, 0].item() == 1.0
    assert layer.weight_scale_inv.dtype == torch.float32
    expected = torch.zeros(256, 256, dtype=torch.float32)
    expected[:128, 128:] = 448
    expected[128:, :128] = 448
    expected[128:, 128:] = -448
    torch.testing.assert_close(layer.weight[:256].float(), expected, rtol=0, atol=0)
    restored = (layer.weight.float()
                * layer.weight_scale_inv.repeat_interleave(128, 0).repeat_interleave(128, 1))
    reference = torch.cat([raw[weight_name], raw[weight_name.replace("gate_proj", "up_proj")]])
    assert ((restored - reference.float()).norm() / reference.float().norm()).item() < 0.04
    torch.testing.assert_close(restored[:256], raw[weight_name].float(), rtol=1e-6, atol=0)
    torch.testing.assert_close(quantized.model.layers[0].linear_attn.in_proj_a.weight,
                               raw["model.layers.0.linear_attn.in_proj_a.weight"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="empty"):
        convert_checkpoint(source, output, scheme="block_wise")


@pytest.mark.parametrize("scheme", ["per_channel", "block_wise"])
@pytest.mark.parametrize("source_layout", ["text", "multimodal"])
def test_offline_conversion_rejects_quantized_hf_source(tmp_path, bf16_default,
                                                       scheme, source_layout):
    from nano_qwen.quantization.convert import convert_checkpoint

    source, output = tmp_path / "hf_fp8", tmp_path / "converted"
    source.mkdir()
    config = text_config()
    config.dtype = torch.bfloat16
    config.quantization_config = {"quant_method": "fp8", "activation_scheme": "dynamic",
                                  "weight_block_size": [128, 128]}
    source_config = (AutoConfig.for_model("qwen3_5", text_config=config.to_dict())
                     if source_layout == "multimodal" else config)
    source_config.save_pretrained(source)
    save_file({"placeholder": torch.zeros(1)}, str(source / "model.safetensors"))
    with pytest.raises(ValueError, match="already.*quantized"):
        convert_checkpoint(source, output, scheme=scheme)
    assert not output.exists()


def test_offline_conversion_rejects_unknown_scheme_before_writing(tmp_path):
    from nano_qwen.quantization.convert import convert_checkpoint

    output = tmp_path / "converted"
    with pytest.raises(ValueError, match="scheme"):
        convert_checkpoint(tmp_path / "missing", output, scheme="unknown")
    assert not output.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("quantized", [False, True])
def test_gdn_state_uses_compute_dtype(quantized, monkeypatch):
    monkeypatch.delenv("NANO_QWEN_GDN_FP32_STATE", raising=False)
    if quantized:
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                gdn = GatedDeltaNet(text_config(), 0, quant_config=Fp8Config())
        finally:
            torch.set_default_dtype(previous)
    else:
        # Existing callers may cast the module after creating it in FP32.
        gdn = GatedDeltaNet(text_config(), 0).to(device="cuda", dtype=torch.bfloat16)
        assert isinstance(gdn.in_proj_qkv.quant_method, UnquantizedLinearMethod)
    gdn.allocate_state_pool(2)
    assert gdn.conv_states.dtype == torch.bfloat16
    assert gdn.recurrent_states.dtype == torch.bfloat16
    gdn.conv_states.fill_(3)
    gdn.recurrent_states.fill_(5)
    gdn.reset_state([1])
    assert torch.count_nonzero(gdn.conv_states[1]) == 0
    assert torch.count_nonzero(gdn.recurrent_states[1]) == 0
    assert torch.all(gdn.conv_states[0] == 3)
    assert torch.all(gdn.recurrent_states[0] == 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("n,k,m", [(6144, 2048, 1), (512, 2048, 4), (2048, 6144, 128)])
def test_fp8_layer_and_graph(n, k, m, bf16_default):
    torch.manual_seed(23)
    with torch.device("cuda"):
        layer = ReplicatedLinear(k, n, bias=True, quant_config=Fp8Config())
        weight = torch.randn(n, k)
        scale = weight.float().abs().amax(dim=1, keepdim=True) / 448.0
        layer.weight.data.copy_((weight.float() / scale).to(torch.float8_e4m3fn))
        layer.weight_scale.data.copy_(scale)
        layer.bias.data.copy_(torch.randn(n))
        # 3-D and non-contiguous input; includes a zero row when M>1.
        x = torch.randn(1, k, m).transpose(1, 2)
        if m > 1:
            x[:, 0].zero_()
        with torch.inference_mode():
            q, scales = quantize_fp8_per_token(x.reshape(m, k))
            if m > 1:
                assert scales[0].item() == 1.0
                assert torch.count_nonzero(q[0].float()) == 0
            dequant_x = q.float() * scales
            dequant_w = layer.weight.float() * layer.weight_scale
            expected = F.linear(dequant_x, dequant_w).to(torch.bfloat16)
            expected.add_(layer.bias)
            actual = layer(x).reshape(m, n)
            error = (actual.float() - expected.float()).norm() / expected.float().norm()
            assert torch.isfinite(actual).all()
            assert error.item() < 0.005
            empty = layer(torch.empty(0, k))
            assert empty.shape == (0, n) and empty.dtype == torch.bfloat16

            for _ in range(3):
                layer(x)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = layer(x)
            x.mul_(torch.linspace(0.1, 2.0, m, device="cuda").view(1, m, 1))
            eager = layer(x)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, eager, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_gdn_mixed_decode_matches_decode(bf16_default):
    """A decode request keeps its kernel, state and output in a mixed batch."""
    torch.manual_seed(83)
    with torch.device("cuda"):
        gdn = GatedDeltaNet(text_config(), 0, quant_config=Fp8Config())
        for module in gdn.modules():
            if isinstance(module, LinearBase):
                weight = torch.randn(module.weight.shape) * 0.02
                scales = weight.float().abs().amax(dim=1, keepdim=True) / 448
                module.weight.data.copy_((weight.float() / scales).to(module.weight.dtype))
                module.weight_scale.data.copy_(scales)
            elif isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(std=0.02)
        gdn.A_log.data.zero_()
        gdn.dt_bias.data.zero_()
        gdn.conv1d.weight.data.normal_(std=0.02)
        gdn.allocate_state_pool(3)
        x = torch.randn(1, 130, gdn.hidden_size)
        conv = torch.randn_like(gdn.conv_states) * 0.1
        state = torch.randn_like(gdn.recurrent_states) * 0.01
    try:
        with torch.inference_mode():
            gdn.conv_states.copy_(conv)
            gdn.recurrent_states.copy_(state)
            set_context(False, state_indices=torch.tensor([1, 2], device="cuda"))
            expected = gdn(x[:, -2:].transpose(0, 1))
            expected_conv = gdn.conv_states[1:].clone()
            expected_state = gdn.recurrent_states[1:].clone()
            gdn.conv_states.copy_(conv)
            gdn.recurrent_states.copy_(state)
            cu = torch.tensor([0, 128, 129, 130], device="cuda", dtype=torch.int32)
            set_context(True, cu_seqlens_q=cu, state_indices=torch.arange(3, device="cuda"),
                prefill_slices=[(0, 128), (128, 129), (129, 130)], prefill_decode_rows=[1, 2])
            actual = gdn(x)
            torch.testing.assert_close(actual[:, -2:].transpose(0, 1), expected, rtol=0, atol=0)
            torch.testing.assert_close(gdn.conv_states[1:], expected_conv, rtol=0, atol=0)
            torch.testing.assert_close(gdn.recurrent_states[1:], expected_state, rtol=0, atol=0)
    finally:
        reset_context()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_gdn_chunk_boundaries_preserve_state():
    """Aligned external chunks use the same BF16 state rounding as one shot."""
    from nano_qwen.layers.gated_delta_net import chunk_gated_delta_rule

    torch.manual_seed(61)
    shape = (1, 256, 2, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn_like(query) * 0.1
    decay = -torch.rand(shape[:-1], device="cuda", dtype=torch.float32) * 0.05
    beta = torch.rand_like(decay)
    initial = torch.randn(1, 2, 128, 128, device="cuda", dtype=torch.bfloat16) * 0.01
    slots = torch.zeros(1, device="cuda", dtype=torch.int64)
    with torch.inference_mode():
        state = initial.clone()
        expected = chunk_gated_delta_rule(query, key, value, decay, beta,
            initial_state=state, initial_state_indices=slots,
            round_state_each_chunk=True)
        final_state = state.clone()
        for chunk in (64, 128):
            state.copy_(initial)
            outputs = []
            for start in range(0, shape[1], chunk):
                stop = start + chunk
                outputs.append(chunk_gated_delta_rule(
                    query[:, start:stop], key[:, start:stop], value[:, start:stop],
                    decay[:, start:stop], beta[:, start:stop],
                    initial_state=state, initial_state_indices=slots,
                    round_state_each_chunk=True))
            torch.testing.assert_close(torch.cat(outputs, dim=1), expected, rtol=0, atol=0)
            torch.testing.assert_close(state, final_state, rtol=0, atol=0)
        # The old carry keeps additional precision until the external boundary.
        state.copy_(initial)
        chunk_gated_delta_rule(query, key, value, decay, beta,
            initial_state=state, initial_state_indices=slots)
        assert not torch.equal(state, final_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_attention_singleton_prefill_and_mixed():
    """A paged one-token tail and a mixed decode row match ordinary prefill."""
    torch.manual_seed(71)
    attention = Attention(8, 256, 256**-0.5, 2, decode_num_splits=1)
    query = torch.randn(65, 8, 256, device="cuda", dtype=torch.bfloat16) * 3
    key = torch.randn(65, 2, 256, device="cuda", dtype=torch.bfloat16) * 3
    value = torch.randn_like(key)
    cu = torch.tensor([0, 65], device="cuda", dtype=torch.int32)
    try:
        with torch.inference_mode():
            set_context(True, cu_seqlens_q=cu, cu_seqlens_k=cu,
                        max_seqlen_q=65, max_seqlen_k=65, prefill_slices=[(0, 65)])
            expected = attention(query, key, value)[-1:]
            attention.k_cache = torch.zeros(2, 256, 2, 256, device="cuda", dtype=query.dtype)
            attention.v_cache = torch.zeros_like(attention.k_cache)
            attention.k_cache[:, :65] = key
            attention.v_cache[:, :65] = value
            for lengths in ([1], [1, 8]):
                q_cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()],
                                    device="cuda", dtype=torch.int32)
                k_cu = torch.arange(len(lengths) + 1, device="cuda", dtype=torch.int32) * 65
                table = torch.arange(len(lengths), device="cuda", dtype=torch.int32).reshape(-1, 1)
                packed = torch.cat([query[-n:] for n in lengths])
                packed_key = torch.cat([key[-n:] for n in lengths])
                packed_value = torch.cat([value[-n:] for n in lengths])
                set_context(True, cu_seqlens_q=q_cu, cu_seqlens_k=k_cu,
                    max_seqlen_q=max(lengths), max_seqlen_k=65, block_tables=table,
                    slot_mapping=torch.full((sum(lengths),), -1, device="cuda", dtype=torch.int32),
                    prefill_slices=[(0, 1)] + ([(1, 9)] if len(lengths) > 1 else []))
                actual = attention(packed, packed_key, packed_value)
                torch.testing.assert_close(actual[:1], expected, rtol=0, atol=0)
                assert actual.shape == packed.shape
    finally:
        reset_context()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_attention_decode_batch_and_graph():
    """KV reduction stays fixed when a request is batched or graph-replayed."""
    torch.manual_seed(53)
    attention = Attention(8, 256, 256**-0.5, 2, decode_num_splits=1)
    attention.k_cache = torch.randn(16, 256, 2, 256, device="cuda", dtype=torch.bfloat16)
    attention.v_cache = torch.randn_like(attention.k_cache)
    query = torch.randn(8, 8, 256, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(8, 2, 256, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    tables = torch.arange(16, device="cuda", dtype=torch.int32).reshape(8, 2)
    slots = torch.full((8,), -1, device="cuda", dtype=torch.int32)
    lengths = torch.full((8,), 129, device="cuda", dtype=torch.int32)
    try:
        with torch.inference_mode():
            for length in (63, 129, 257, 511):
                lengths.fill_(length)
                set_context(False, slot_mapping=slots[:1], context_lens=lengths[:1],
                            block_tables=tables[:1])
                expected = attention(query[:1], key[:1], value[:1])
                for bs in (2, 3, 4, 5, 8):
                    set_context(False, slot_mapping=slots[:bs], context_lens=lengths[:bs],
                                block_tables=tables[:bs])
                    actual = attention(query[:bs], key[:bs], value[:bs])
                    torch.testing.assert_close(actual[:1], expected, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                captured = attention(query, key, value)
            query.mul_(0.5)
            lengths.fill_(130)
            expected = attention(query, key, value)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    finally:
        reset_context()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_gate_projection_is_row_independent():
    torch.manual_seed(41)
    x = torch.randn(38, 2048, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(16, 2048, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = F.linear(x.float(), weight.float()).to(x.dtype)
        result = stable_gate_projection(x, weight)
        torch.testing.assert_close(result.float(), expected.float(), rtol=0.008, atol=0.002)
        for rows in (1, 30, 128, 384):
            padded = x[:rows] if rows < len(x) else torch.cat([
                x, torch.zeros(rows - len(x), x.shape[1], device=x.device, dtype=x.dtype),
            ])
            count = min(rows, len(x))
            torch.testing.assert_close(stable_gate_projection(padded, weight)[:count],
                                       result[:count], rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        stable_gate_projection(x, weight)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            captured = stable_gate_projection(x, weight)
        x.mul_(0.5)
        expected = stable_gate_projection(x, weight)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_piecewise_prefill_matches_eager(bf16_default):
    """Padding, graph reuse and packed request boundaries preserve FP8 logits."""
    torch.manual_seed(37)
    config = text_config()
    with torch.device("cuda"):
        model = Qwen3_5ForCausalLM(config, quant_config=Fp8Config())
        for layer in model.modules():
            if isinstance(layer, LinearBase):
                weight = torch.randn(layer.weight.shape) * 0.02
                scale = weight.float().abs().amax(dim=1, keepdim=True) / 448.0
                layer.weight.data.copy_((weight.float() / scale).to(torch.float8_e4m3fn))
                layer.weight_scale.data.copy_(scale)
                layer.quant_method.process_weights_after_loading(layer)
            elif isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(std=0.02)
            if isinstance(layer, GatedDeltaNet):
                layer.allocate_state_pool(2)
    model.model.embed_tokens.weight.data.normal_(std=0.02)
    model.model.layers[0].linear_attn.A_log.data.zero_()
    model.model.layers[0].linear_attn.dt_bias.data.zero_()
    model.model.layers[0].linear_attn.conv1d.weight.data.normal_(std=0.02)
    for module in model.modules():
        if isinstance(module, Attention):
            module.k_cache = module.v_cache = torch.empty(0, device="cuda")
    runner = SimpleNamespace(model=model, use_prefill_cudagraph=True,
        config=SimpleNamespace(quant_config=Fp8Config(), max_num_batched_tokens=128,
                               hf_config=config))
    graphs = CudaGraphManager(runner)
    gdn = model.model.layers[0].linear_attn
    try:
        with torch.inference_mode():
            graphs.capture_prefill([128])
            assert 128 in graphs.prefill_graphs
            # Different content and layouts replay the same captured bucket.
            for lengths in ([30], [64], [128], [32, 96], [63, 65]):
                count = sum(lengths)
                ids = torch.randint(config.vocab_size, (count,), device="cuda")
                positions = torch.cat([torch.arange(n, device="cuda") for n in lengths])
                boundaries = [0]
                for length in lengths:
                    boundaries.append(boundaries[-1] + length)
                cu = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
                set_context(True, cu_seqlens_q=cu, cu_seqlens_k=cu,
                    max_seqlen_q=max(lengths), max_seqlen_k=max(lengths),
                    state_indices=torch.arange(len(lengths), device="cuda"),
                    prefill_slices=list(zip(boundaries[:-1], boundaries[1:])))
                gdn.reset_state([0, 1])
                expected = model.compute_logits(model(ids, positions))
                conv = gdn.conv_states.clone()
                recurrent = gdn.recurrent_states.clone()
                gdn.reset_state([0, 1])
                actual = graphs.run_prefill(ids, positions)
                torch.cuda.synchronize()
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(gdn.conv_states, conv, rtol=0, atol=0)
                torch.testing.assert_close(gdn.recurrent_states, recurrent, rtol=0, atol=0)
    finally:
        reset_context()
        graphs.clear()
