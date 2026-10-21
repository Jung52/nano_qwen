"""Real CUDA math, rounding, strided head norms and replay buffer independence."""
import pytest
import torch

from nano_qwen.layers import fused_norm as backend
from nano_qwen.layers.layernorm import GemmaRMSNorm, RMSNorm, RMSNormGated

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def close(actual, expected, dtype):
    # BF16/FP16 reductions may cross one rounding boundary; FP32 remains tight.
    atol, rtol = {torch.bfloat16: (0.016, 0.012), torch.float16: (0.002, 0.002),
                  torch.float32: (2e-6, 2e-6)}[dtype]
    for a, b in zip(actual if isinstance(actual, tuple) else (actual,),
                    expected if isinstance(expected, tuple) else (expected,)):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def reference(x, weight, eps, residual=None, gemma=False, gate=None):
    dtype = x.dtype
    v = x.float() if residual is None else x.float() + residual.float()
    norm = (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)).to(dtype)
    w = weight.to(dtype) if gate is not None else (1.0 + weight if gemma else weight)
    y = norm * w
    if gate is not None:
        y = (y * torch.nn.functional.silu(gate.float())).to(dtype)
    return (y, v.to(dtype)) if residual is not None else y


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("kind", ["rms", "residual", "gemma", "gemma_residual", "gated"])
@pytest.mark.parametrize("shape,strided", [((8, 128), True), ((8, 2560), False),
                                        ((2, 16, 128), False), ((3, 127), False)])
@torch.inference_mode()
def test_math(dtype, kind, shape, strided, monkeypatch):
    torch.manual_seed(10)
    def rand():
        if strided:
            return torch.randn((shape[0], shape[-1]*2), device="cuda", dtype=dtype)[:, :shape[-1]]
        return torch.randn(shape, device="cuda", dtype=dtype)
    x = rand()
    r = rand() if "residual" in kind else None
    gate = rand() if kind == "gated" else None
    klass = RMSNormGated if gate is not None else GemmaRMSNorm if "gemma" in kind else RMSNorm
    layer = klass(shape[-1]).cuda().to(dtype)
    if gate is not None:
        layer.weight.data = layer.weight.float()  # Real GDN FP32 checkpoint weight.
    layer.weight.uniform_(-0.75, 0.75) if "gemma" in kind else layer.weight.uniform_(0.5, 1.5)
    snapshots = [t.clone() for t in (x, r, gate) if t is not None]
    monkeypatch.setattr(backend, "ENABLED", True)
    assert backend.can_fuse(x, layer.weight, r, gate)
    actual = layer(x, gate) if gate is not None else layer(x, r)
    expected = reference(x, layer.weight, layer.eps, r, "gemma" in kind, gate)
    close(actual, expected, dtype)
    if r is not None:
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    for t, before in zip((t for t in (x, r, gate) if t is not None), snapshots):
        torch.testing.assert_close(t, before, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["gemma_residual", "gated"])
@torch.inference_mode()
def test_two_graph_banks_keep_independent_outputs(kind, monkeypatch):
    """Two static input/output banks, repeated replay with changed inputs."""
    monkeypatch.setattr(backend, "ENABLED", True)
    layer = (RMSNormGated(128) if kind == "gated" else GemmaRMSNorm(128)).cuda().bfloat16()
    banks = []
    for _ in range(2):
        x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
        aux = torch.randn_like(x)
        def call():
            return layer(x, aux)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call()
        banks.append((graph, x, aux, out))
    outputs = [b[3][0] if isinstance(b[3], tuple) else b[3] for b in banks]
    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    for step in range(10):
        slot = step % 2
        graph, x, aux, out = banks[slot]
        other_before = outputs[1-slot].clone()
        x.normal_()
        aux.normal_()
        graph.replay()
        close(out, reference(x, layer.weight, layer.eps,
            residual=aux if kind != "gated" else None, gemma=kind != "gated",
            gate=aux if kind == "gated" else None), x.dtype)
        torch.testing.assert_close(outputs[1-slot], other_before, rtol=0, atol=0)


def test_training_and_unsupported_layout_use_reference(monkeypatch):
    monkeypatch.setattr(backend, "ENABLED", True)
    layer = GemmaRMSNorm(128).cuda()
    x = torch.randn(8, 128, device="cuda", requires_grad=True)
    assert not backend.can_fuse(x, layer.weight)
    layer(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    with torch.inference_mode():
        assert not backend.can_fuse(x[:, ::2], layer.weight[:64])
        monkeypatch.setattr(backend, "ENABLED", False)
        assert not backend.can_fuse(x, layer.weight)
