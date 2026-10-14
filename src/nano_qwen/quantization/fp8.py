import os

import torch
from torch import nn
import triton
import triton.language as tl

from .base import LinearMethodBase, QuantizationConfig
from .triton_gemm import autotune_gemm, launch_gemm


# Read once before compilation/capture; use a fresh process for an A/B run.
_SHARE_FP8_INPUT = os.environ.get("NANO_QWEN_FP8_SHARE_INPUT", "1") != "0"


@triton.jit
def _gate_projection(X, W, Y, K: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row, column = tl.program_id(0), tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + offsets, offsets < K, 0).to(tl.float32)
    w = tl.load(W + column * K + offsets, offsets < K, 0).to(tl.float32)
    tl.store(Y + row * N + column, tl.sum(x * w, axis=0))


def stable_gate_projection(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Small unquantized GDN gates with row-independent FP32 accumulation."""
    shape = x.shape[:-1] + (weight.shape[0],)
    x = x.reshape(-1, x.shape[-1]).contiguous()
    output = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=x.dtype)
    if x.shape[0]:
        _gate_projection[(x.shape[0], weight.shape[0])](
            x, weight, output, x.shape[1], weight.shape[0],
            triton.next_power_of_2(x.shape[1]), num_warps=4,
        )
    return output.reshape(shape)


@triton.jit
def _quantize_rows(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(X + row * K + offsets, offsets < K, 0).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=0)
    scale = tl.where(amax == 0, 1.0, amax / 448.0)
    values = tl.minimum(tl.maximum(values / scale, -448.0), 448.0)
    tl.store(Q + row * K + offsets, values, offsets < K)
    tl.store(S + row, scale)


def quantize_fp8_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("FP8 activation quantization expects a nonempty feature dimension")
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("FP8 activation quantization requires CUDA BF16 or FP16 inputs")
    x = x.contiguous()
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    if x.shape[0] == 0:
        return q, scales
    block = triton.next_power_of_2(x.shape[1])
    _quantize_rows[(x.shape[0],)](
        x, q, scales, x.shape[1], block,
        num_warps=8 if block >= 4096 else 4,
    )
    return q, scales


@triton.jit
def _per_channel_mm(A, W, AS, WS, Y, M: tl.constexpr, N: tl.constexpr,
                    K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                    DEVICE: tl.constexpr = 0):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, 128)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for group in range(tl.cdiv(K, 128)):
        offsets = group * 128 + kk
        a = tl.load(A + rows[:, None] * K + offsets[None, :],
                    (rows[:, None] < M) & (offsets[None, :] < K), 0.0)
        w = tl.load(W + cols[None, :] * K + offsets[:, None],
                    (cols[None, :] < N) & (offsets[:, None] < K), 0.0)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32, max_num_imprecise_acc=0)
    sa = tl.load(AS + rows, rows < M, 0)
    sw = tl.load(WS + cols, cols < N, 0)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc * sa[:, None] * sw[None, :],
             (rows[:, None] < M) & (cols[None, :] < N))


_per_channel_mm_tuned = autotune_gemm(_per_channel_mm)


def fp8_per_channel_mm(q, weight, scales, weight_scale, out_dtype=torch.bfloat16):
    output = torch.empty((q.shape[0], weight.shape[0]), device=q.device, dtype=out_dtype)
    if q.shape[0]:
        launch_gemm(_per_channel_mm_tuned, q, weight, scales, weight_scale, output)
    return output


class Fp8LinearMethod(LinearMethodBase):
    """E4M3 weights per output channel and dynamic per-token inputs.

    Uses converted FP8 weights and dequantization scales from a versioned
    nano_qwen checkpoint, never ordinary BF16 weights.
    """

    def create_weights(self, layer, input_size, output_size, params_dtype):
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("FP8 linear computation requires BF16 or FP16")
        if layer.tp_size != 1:
            raise NotImplementedError("FP8 linear layers currently require TP=1")
        if input_size % 16 or output_size % 16:
            raise ValueError("The FP8 backend requires input/output sizes divisible by 16")
        layer.weight = nn.Parameter(torch.empty(
            output_size, input_size, dtype=torch.float8_e4m3fn,
        ), requires_grad=False)
        layer.weight.weight_loader = layer.weight_loader
        # FP32 scales remain separate from the compute and storage dtypes.
        layer.weight_scale = nn.Parameter(torch.empty(
            output_size, 1, dtype=torch.float32,
        ), requires_grad=False)
        layer.weight_scale.weight_loader = layer.weight_loader

    def process_weights_after_loading(self, layer):
        # Prepare contiguous FP32 scales before GEMM/graph capture.
        layer.weight_scale.data = layer.weight_scale.data.contiguous()

    def apply(self, layer, x, bias=None):
        if x.ndim == 0 or x.shape[-1] != layer.input_size:
            raise ValueError("FP8 linear input has the wrong feature dimension")
        shape = x.shape[:-1] + (layer.output_size,)
        q, scales = quantize_fp8_per_token(x.reshape(-1, x.shape[-1]))
        return self.apply_quantized(layer, q, scales, shape, bias)

    def apply_quantized(self, layer, q, scales, output_shape, bias=None):
        """Consume call-local quantized activations shared by sibling projections."""
        if q.shape[0] == 0:
            return torch.empty(output_shape, device=q.device, dtype=layer.compute_dtype)
        output = fp8_per_channel_mm(
            q, layer.weight, scales, layer.weight_scale, layer.compute_dtype,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(output_shape)


def project_shared_fp8_input(x: torch.Tensor, *layers) -> tuple[torch.Tensor, ...]:
    """Project one input through sibling replicated/column-parallel layers.

    Quantize once, keeping separate GEMMs and their existing accumulation.
    No tensors are cached across calls or CUDA Graph replays. Other backends
    and row-parallel layers retain their normal forward/communication path.
    """
    if not layers:
        return ()
    if not _SHARE_FP8_INPUT or not all(
        isinstance(layer.quant_method, Fp8LinearMethod)
        and layer.tp_size == 1 and layer.tp_dim in (None, 0)
        for layer in layers
    ):
        return tuple(layer(x) for layer in layers)
    if x.ndim == 0 or any(x.shape[-1] != layer.input_size for layer in layers):
        raise ValueError("FP8 linear input has the wrong feature dimension")
    q, scales = quantize_fp8_per_token(x.reshape(-1, x.shape[-1]))
    return tuple(
        layer.quant_method.apply_quantized(
            layer, q, scales, x.shape[:-1] + (layer.output_size,), layer.bias,
        )
        for layer in layers
    )


class Fp8Config(QuantizationConfig):
    def get_quant_method(self, layer, prefix):
        return Fp8LinearMethod()
