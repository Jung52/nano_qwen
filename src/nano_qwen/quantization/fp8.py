import torch
from torch import nn
import triton
import triton.language as tl

from .base import LinearMethodBase, QuantizationConfig


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
        # Contiguous rowwise scales for _scaled_mm; prepare before graph capture.
        layer.weight_scale.data = layer.weight_scale.data.contiguous()

    def apply(self, layer, x, bias=None):
        if x.ndim == 0 or x.shape[-1] != layer.input_size:
            raise ValueError("FP8 linear input has the wrong feature dimension")
        shape = x.shape[:-1] + (layer.output_size,)
        q, scales = quantize_fp8_per_token(x.reshape(-1, x.shape[-1]))
        if q.shape[0] == 0:
            return torch.empty(shape, device=x.device, dtype=layer.compute_dtype)
        output = torch._scaled_mm(
            q, layer.weight.t(), scales,
            layer.weight_scale.t().contiguous(),
            out_dtype=layer.compute_dtype, use_fast_accum=False,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(shape)


class Fp8Config(QuantizationConfig):
    def get_quant_method(self, layer, prefix):
        return Fp8LinearMethod()
