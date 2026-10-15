"""Production FP8 Triton kernels: math, tails and graph/autotune lifecycle."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="CUDA SM89+ required",
)


@pytest.mark.parametrize('mode', ['auto', 'legacy'])
def test_block_large_rows_dispatch_is_tuned_and_matches_dequantization(mode):
    from nano_qwen.quantization.fp8_block import (
        block_fp8_matmul, _block_mm_tuned, _block_mm_prefill_tuned,
    )
    _, q, w, sa, sw, expected = fixture(True, m=137, n=384, k=256)
    output = torch.empty_like(expected, dtype=torch.bfloat16)
    tuner = _block_mm_prefill_tuned if mode == 'auto' else _block_mm_tuned
    tuner.cache.clear()
    block_fp8_matmul(q, w, sa, sw, output, mode=mode)
    assert len(tuner.cache) == 1
    assert (output.float()-expected).norm()/expected.norm() < .005


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_packed_prefill_quantization_preserves_legacy_bytes_and_group_tails(dtype):
    from nano_qwen.quantization.fp8_block import quantize_fp8_per_group
    x = torch.randn(129, 128*10, device='cuda', dtype=dtype)
    x[0].zero_()
    qa, sa = quantize_fp8_per_group(x, mode='auto')
    qb, sb = quantize_fp8_per_group(x, mode='legacy')
    torch.testing.assert_close(qa.view(torch.uint8), qb.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(sa, sb, rtol=0, atol=0)


def test_shared_projection_quantizes_once_and_refreshes_graph_input(monkeypatch):
    from nano_qwen.quantization import fp8
    x, _, w, _, sw, _ = fixture(False, m=3, n=128, k=128)
    class Layer:
        input_size, output_size, tp_size, tp_dim = 128, 128, 1, None
        compute_dtype = torch.bfloat16
        quant_method = fp8.Fp8LinearMethod()
        def __init__(self, value):
            self.weight, self.weight_scale = w, sw
            self.bias = torch.full((128,), value, device='cuda', dtype=self.compute_dtype)
        def __call__(self, values):
            return self.quant_method.apply(self, values, self.bias)
    layers = (Layer(1), Layer(2))
    reference = tuple(layer(x) for layer in layers)
    original = fp8.quantize_fp8_per_token
    calls = []
    def counted(values):
        calls.append(1)
        return original(values)
    monkeypatch.setattr(fp8, 'quantize_fp8_per_token', counted)
    monkeypatch.setattr(fp8, '_SHARE_FP8_INPUT', True)
    actual = fp8.project_shared_fp8_input(x, *layers)
    assert len(calls) == 1
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fp8.project_shared_fp8_input(x, *layers)
    x.zero_()
    graph.replay()
    torch.cuda.synchronize()
    for a, layer in zip(captured, layers):
        torch.testing.assert_close(a, layer.bias.expand_as(a), rtol=0, atol=0)


def fixture(block, m=17, n=384, k=256, dtype=torch.bfloat16):
    from nano_qwen.quantization.fp8 import quantize_fp8_per_token
    from nano_qwen.quantization.fp8_block import quantize_fp8_per_group
    from nano_qwen.quantization.convert import _quantize_weight

    torch.manual_seed(31)
    x = torch.randn(1, k, m, device="cuda", dtype=dtype).transpose(1, 2)
    x[:, 0].zero_()
    w = torch.randn(n, k, device="cuda", dtype=dtype)
    wq, sw = _quantize_weight(w, 'cuda', 'block_wise' if block else 'per_channel', 'test.weight')
    wq, sw = wq.to(w.device), sw.to(w.device)
    q, sa = (quantize_fp8_per_group if block else quantize_fp8_per_token)(x.reshape(m, k))
    dx = q.float() * (sa.repeat_interleave(128, 1) if block else sa)
    dw = wq.float() * (sw.repeat_interleave(128, 0).repeat_interleave(128, 1) if block else sw)
    return x, q, wq, sa, sw, dx @ dw.T


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_linear_uses_triton_with_bias_strides_zero_rows_and_empty_input(monkeypatch, block, dtype):
    from nano_qwen.quantization.fp8 import Fp8LinearMethod
    from nano_qwen.quantization.fp8_block import Fp8BlockLinearMethod

    def forbid_scaled_mm(*args, **kwargs):
        raise AssertionError("Production FP8 GEMM must use Triton")

    monkeypatch.setattr(torch, "_scaled_mm", forbid_scaled_mm)
    x, q, w, sa, sw, expected = fixture(block, dtype=dtype)
    layer = SimpleNamespace(input_size=256, output_size=384, compute_dtype=dtype,
                            weight=w, weight_scale=sw, weight_scale_inv=sw)
    bias = torch.randn(384, device="cuda", dtype=dtype)
    method = Fp8BlockLinearMethod() if block else Fp8LinearMethod()
    actual = method.apply(layer, x, bias).reshape(17, 384)
    reference = expected.to(dtype) + bias
    assert actual.dtype == dtype and torch.isfinite(actual).all()
    assert (actual.float() - reference.float()).norm() / reference.float().norm() < 0.005
    torch.testing.assert_close(actual[0], bias, rtol=0, atol=0)
    assert method.apply(layer, x[:, :0], bias).shape == (1, 0, 384)


def test_per_channel_handles_k_and_n_tails():
    from nano_qwen.quantization.fp8 import fp8_per_channel_mm

    _, q, w, sa, sw, expected = fixture(False, m=3, n=48, k=80)
    actual = fp8_per_channel_mm(q, w, sa, sw, torch.bfloat16)
    assert (actual.float() - expected).norm() / expected.norm() < 0.005
    assert torch.count_nonzero(actual[0]) == 0


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("capturing", [False, True])
def test_reuse_launch_selects_m1_tile_without_overwriting_m2(monkeypatch, block, capturing):
    from nano_qwen.quantization.fp8 import fp8_per_channel_mm, _per_channel_mm_tuned
    from nano_qwen.quantization.fp8_block import fp8_block_mm, _block_mm_tuned
    from nano_qwen.quantization.triton_gemm import GEMM_CONFIGS, gemm_key
    from nano_qwen.utils.context import set_context, reset_context

    mm = fp8_block_mm if block else fp8_per_channel_mm
    tuner = _block_mm_tuned if block else _per_channel_mm_tuned
    _, q, w, sa, sw, _ = fixture(block, m=2, n=128, k=128)
    expected = torch.cat([mm(q[i:i+1], w, sa[i:i+1], sw) for i in range(2)])
    m1_key = gemm_key(q[:1], w, sa[:1], sw, expected[:1])
    m2_key = gemm_key(q, w, sa, sw, expected)
    chosen = tuner.cache[m1_key]
    sentinel = next(c for c in GEMM_CONFIGS if c is not chosen)
    tuner.cache[m2_key] = sentinel
    before = dict(tuner.cache)
    original = tuner.fn
    launches = []

    class RecordingKernel:
        __name__ = original.__name__
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(kwargs)
                return original[grid](*args, **kwargs)
            return launch

    monkeypatch.setattr(tuner, "fn", RecordingKernel())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    set_context(False, gemm_reference_m=1)
    try:
        actual = mm(q, w, sa, sw)
    finally:
        reset_context()
    assert launches == [{"DEVICE": q.device.index, **chosen.all_kwargs()}]
    assert tuner.cache == before
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("block", [False, True])
def test_reuse_cold_eager_warms_m1_and_cold_capture_fails_closed(monkeypatch, block):
    from nano_qwen.quantization.fp8 import fp8_per_channel_mm, _per_channel_mm_tuned
    from nano_qwen.quantization.fp8_block import fp8_block_mm, _block_mm_tuned
    from nano_qwen.quantization.triton_gemm import gemm_key
    from nano_qwen.utils.context import set_context, reset_context

    mm = fp8_block_mm if block else fp8_per_channel_mm
    tuner = _block_mm_tuned if block else _per_channel_mm_tuned
    _, q, w, sa, sw, _ = fixture(block, m=2, n=128, k=128)
    output = torch.empty(2, 128, device="cuda", dtype=torch.bfloat16)
    key1, key2 = gemm_key(q[:1], w, sa[:1], sw, output[:1]), gemm_key(q, w, sa, sw, output)
    tuner.cache.pop(key1, None)
    tuner.cache.pop(key2, None)
    set_context(False, gemm_reference_m=1)
    try:
        with monkeypatch.context() as cold:
            cold.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
            with pytest.raises(RuntimeError, match="M1"):
                mm(q, w, sa, sw)
        assert key1 not in tuner.cache and key2 not in tuner.cache
        actual = mm(q, w, sa, sw)
        assert key1 in tuner.cache and key2 not in tuner.cache
    finally:
        reset_context()
    expected = torch.cat([mm(q[i:i+1], w, sa[i:i+1], sw) for i in range(2)])
    assert torch.equal(actual, expected)


def test_reuse_keeps_unquantized_checkpoint_exceptions_on_single_row_math(monkeypatch):
    from nano_qwen.quantization.unquantized import UnquantizedLinearMethod
    from nano_qwen.utils.context import set_context, reset_context
    import torch.nn.functional as functional

    x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    layer = SimpleNamespace(weight=torch.randn(512, 256, device="cuda", dtype=x.dtype))
    bias = torch.randn(512, device="cuda", dtype=x.dtype)
    original = functional.linear
    expected = torch.cat([original(row.unsqueeze(0), layer.weight, bias) for row in x])
    rows = []
    def linear(values, weight, bias=None):
        rows.append(values.shape[0])
        return original(values, weight, bias)
    monkeypatch.setattr(functional, "linear", linear)
    set_context(False, gemm_reference_m=1)
    try:
        actual = UnquantizedLinearMethod().apply(layer, x, bias)
    finally:
        reset_context()
    assert rows == [1, 1]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("family", ['per_channel', 'block', 'block_prefill'])
def test_all_candidate_tiles_match_explicit_dequantization(family):
    from nano_qwen.quantization.triton_gemm import GEMM_CONFIGS
    from nano_qwen.quantization.fp8 import _per_channel_mm
    from nano_qwen.quantization.fp8_block import _block_mm, _block_mm_prefill

    m = 137 if family == 'block_prefill' else 17
    _, q, w, sa, sw, expected = fixture(family != 'per_channel', m=m)
    kernel = {'per_channel': _per_channel_mm, 'block': _block_mm, 'block_prefill': _block_mm_prefill}[family]
    output = torch.empty_like(expected, dtype=torch.bfloat16)
    for config in GEMM_CONFIGS:
        grid = (triton.cdiv(m, config.kwargs["BM"]), triton.cdiv(384, config.kwargs["BN"]))
        extra = {}
        if family == 'block_prefill':
            grid = (grid[0]*grid[1],)
            extra['STAGES'] = config.num_stages
        kernel[grid](q, w, sa, sw, output, m, 384, 256, DEVICE=q.device.index,
                     **extra, **config.all_kwargs())
        assert torch.isfinite(output).all()
        assert (output.float() - expected).norm() / expected.norm() < 0.005
        assert torch.count_nonzero(output[0]) == 0


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("m", [7, 137])
def test_cold_and_warm_graph_capture_never_autotunes_inside_capture(monkeypatch, block, m):
    from nano_qwen.quantization.fp8 import fp8_per_channel_mm, _per_channel_mm_tuned
    from nano_qwen.quantization.fp8_block import fp8_block_mm, _block_mm_tuned, _block_mm_prefill_tuned

    mm = fp8_block_mm if block else fp8_per_channel_mm
    tuner = (_block_mm_prefill_tuned if m >= 128 else _block_mm_tuned) if block else _per_channel_mm_tuned
    _, q, w, sa, sw, expected = fixture(block, m=m, n=128, k=128)
    tuner.cache.clear()
    original_bench = tuner.do_bench

    def forbid_bench(*args, **kwargs):
        raise AssertionError("Capture or cached launches must not search configs")

    monkeypatch.setattr(tuner, "do_bench", forbid_bench)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cold = mm(q, w, sa, sw, torch.bfloat16)
    graph.replay()
    torch.cuda.synchronize()
    assert not tuner.cache
    assert (cold.float() - expected).norm() / expected.norm() < 0.005

    monkeypatch.setattr(tuner, "do_bench", original_bench)
    warm = mm(q, w, sa, sw, torch.bfloat16)
    assert len(tuner.cache) == 1
    monkeypatch.setattr(tuner, "do_bench", forbid_bench)
    mm(q, w, sa, sw, torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mm(q, w, sa, sw, torch.bfloat16)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, warm, rtol=0, atol=0)


def test_autotune_callback_uses_device_local_stream_and_two_measured_replays(monkeypatch):
    from nano_qwen.quantization.triton_gemm import bench_gemm_config

    # PyTorch's implicit capture stream is process-global and may belong to another GPU.
    # An unusable implicit stream proves the callback supplies its own device-local one.
    monkeypatch.setattr(torch.cuda.graph, "default_capture_stream", object())
    counter = torch.zeros(1, device="cuda")
    times = bench_gemm_config(lambda: counter.add_(1), quantiles=(0.5, 0.2, 0.8))
    # One warmup executes; graph capture records without executing; two timed replays.
    assert counter.item() == 3
    assert len(times) == 3 and all(t > 0 for t in times)


@pytest.mark.parametrize("capturing", [False, True])
def test_cached_launch_reuses_selected_tile(monkeypatch, capturing):
    from nano_qwen.quantization.triton_gemm import GEMM_CONFIGS, gemm_key, launch_gemm

    _, q, w, sa, sw, expected = fixture(False, m=7, n=128, k=128)
    output = torch.empty_like(expected, dtype=torch.bfloat16)
    selected = GEMM_CONFIGS[-1]  # Deliberately different from the cold-capture fallback.
    launches = []

    class RecordingKernel:
        def __getitem__(self, grid):
            def record(*args, **kwargs):
                launches.append((grid, kwargs))
            return record

    tuner = SimpleNamespace(cache={gemm_key(q, w, sa, sw, output): selected},
                            fn=RecordingKernel())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    launch_gemm(tuner, q, w, sa, sw, output)
    assert launches == [((triton.cdiv(7, selected.kwargs["BM"]),
                          triton.cdiv(128, selected.kwargs["BN"])),
                         {"DEVICE": q.device.index, **selected.all_kwargs()})]


def test_validated_block_prefill_policy_applies_to_eager_and_cold_graph_but_preserves_decode(monkeypatch):
    from nano_qwen.quantization.fp8_block import fp8_block_mm, _block_mm_tuned, _block_mm_prefill_tuned
    from nano_qwen.quantization.triton_gemm import GEMM_CONFIGS, gemm_key

    if torch.cuda.get_device_properties(0).name != 'NVIDIA GeForce RTX 4070 Ti SUPER':
        pytest.skip('Preference was validated only on RTX 4070 Ti SUPER')
    _, q, w, sa, sw, expected = fixture(True, m=512, n=2560, k=4096)
    tuner = _block_mm_tuned
    tuner.cache.clear()
    def forbid_bench(*args, **kwargs):
        raise AssertionError('Validated prefill preference must bypass fresh search')
    monkeypatch.setattr(tuner, 'do_bench', forbid_bench)
    eager = fp8_block_mm(q, w, sa, sw, mode='legacy')
    key = gemm_key(q, w, sa, sw, eager)
    chosen = tuner.cache[key]
    assert chosen.kwargs == {'BM': 32, 'BN': 128}
    assert chosen.num_warps == 4 and chosen.num_stages == 3
    assert (eager.float()-expected).norm()/expected.norm() < .005
    tuner.cache.pop(key)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fp8_block_mm(q, w, sa, sw, mode='legacy')
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
    assert tuner.cache[key].kwargs == {'BM': 32, 'BN': 128}
    output = torch.empty((8, 2560), device='cuda', dtype=torch.bfloat16)
    decode_key = gemm_key(q[:8], w, sa[:8], sw, output)
    decode_config = GEMM_CONFIGS[0]
    _block_mm_tuned.cache[decode_key] = decode_config
    decoded = fp8_block_mm(q[:8], w, sa[:8], sw)
    assert _block_mm_tuned.cache[decode_key] is decode_config
    assert (decoded.float()-expected[:8]).norm()/expected[:8].norm() < .005
