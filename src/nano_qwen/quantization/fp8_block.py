"""HF FP8 block-128 weights and dynamic per-token, per-128-channel inputs."""

from functools import partial
import os

import torch
from torch import nn
import triton
import triton.language as tl

from .base import LinearMethodBase, QuantizationConfig
from .checkpoint import normalize_weight_name
from .unquantized import UnquantizedLinearMethod


# Read once before graph capture. No device queries or online autotuning.
_BLOCK_MODE = os.environ.get("NANO_QWEN_FP8_BLOCK_MODE", "auto")
if _BLOCK_MODE not in ("auto", "legacy"):
    raise ValueError("NANO_QWEN_FP8_BLOCK_MODE must be auto or legacy")
_PREFILL_MIN_ROWS = 128
# (BM, BN, warps, stages): candidates for the separate GPU microbenchmark.
PREFILL_CONFIGS = ((32, 128, 4, 3), (64, 128, 4, 3),
                   (128, 128, 8, 2), (64, 256, 8, 2))


@triton.jit
def _quantize_groups(X, Q, S, K: tl.constexpr, GROUPS: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    offsets = group * 128 + tl.arange(0, 128)
    values = tl.load(X + row * K + offsets).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=0)
    scale = tl.where(amax == 0, 1.0, amax / 448.0)
    values = tl.minimum(tl.maximum(values / scale, -448.0), 448.0)
    # Avoid Triton's FP32 -> FP16 (round toward zero) -> FP8 lowering on SM89.
    # Convert directly with round-to-nearest-even, packing four FP8 outputs.
    quantized = tl.inline_asm_elementwise(
        "{ .reg .b16 lo, hi; "
        "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; "
        "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; "
        "mov.b32 $0, {lo, hi}; }",
        constraints="=r,f,f,f,f", args=[values],
        dtype=tl.float8e4nv, is_pure=True, pack=4,
    )
    tl.store(Q + row * K + offsets, quantized)
    tl.store(S + row * GROUPS + group, scale)


@triton.jit
def _quantize_groups_prefill(X, Q, S, K: tl.constexpr, GROUPS: tl.constexpr,
                            GROUPS_PER_CTA: tl.constexpr):
    row = tl.program_id(0)
    groups = tl.program_id(1) * GROUPS_PER_CTA + tl.arange(0, GROUPS_PER_CTA)
    offsets = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    valid = groups[:, None] < GROUPS
    values = tl.load(X + row * K + offsets, valid, 0.0).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=1)
    scale = tl.where(amax == 0, 1.0, amax / 448.0)
    values = tl.minimum(tl.maximum(values / scale[:, None], -448.0), 448.0)
    # Keep the existing direct FP32 -> FP8 RNE conversion, including on SM89.
    quantized = tl.inline_asm_elementwise(
        "{ .reg .b16 lo, hi; "
        "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; "
        "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; "
        "mov.b32 $0, {lo, hi}; }",
        constraints="=r,f,f,f,f", args=[values],
        dtype=tl.float8e4nv, is_pure=True, pack=4,
    )
    tl.store(Q + row * K + offsets, quantized, valid)
    tl.store(S + row * GROUPS + groups, scale, groups < GROUPS)


def _resolve_mode(mode):
    mode = _BLOCK_MODE if mode is None else mode
    if mode not in ("auto", "legacy"):
        raise ValueError("Block FP8 mode must be auto or legacy")
    return mode


def quantize_fp8_per_group(x, *, mode=None):
    if x.ndim != 2 or not x.shape[1] or x.shape[1] % 128:
        raise ValueError("Block FP8 expects 2D inputs with K divisible by 128")
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Block FP8 requires CUDA BF16/FP16 inputs")
    x = x.contiguous()
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((x.shape[0], x.shape[1] // 128),
                         device=x.device, dtype=torch.float32)
    if x.shape[0]:
        groups = x.shape[1] // 128
        if _resolve_mode(mode) == "legacy" or x.shape[0] < _PREFILL_MIN_ROWS:
            _quantize_groups[(x.shape[0], groups)](
                x, q, scales, x.shape[1], groups, num_warps=4,
            )
        else:
            packed_groups = min(8, triton.next_power_of_2(groups))
            _quantize_groups_prefill[(x.shape[0], triton.cdiv(groups, packed_groups))](
                x, q, scales, x.shape[1], groups, packed_groups, num_warps=4,
            )
    return q, scales


@triton.jit
def _block_mm(A, W, AS, WS, Y, M: tl.constexpr, N: tl.constexpr,
              K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, 128)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # Each K block has its own activation AND weight dequantization scale.
    for group in range(K // 128):
        offsets = group * 128 + kk
        a = tl.load(A + rows[:, None] * K + offsets[None, :],
                    rows[:, None] < M, 0.0)
        w = tl.load(W + cols[None, :] * K + offsets[:, None],
                    cols[None, :] < N, 0.0)
        sa = tl.load(AS + rows * (K // 128) + group, rows < M, 0)
        sw = tl.load(WS + (cols // 128) * (K // 128) + group, cols < N, 0)
        product = tl.dot(a, w, out_dtype=tl.float32, max_num_imprecise_acc=0)
        acc += product * sa[:, None] * sw[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _block_mm_prefill(A, W, AS, WS, Y, M: tl.constexpr, N: tl.constexpr,
                      K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      STAGES: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m, num_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group_id = pid // (GROUP_M * num_n)
    first_m = group_id * GROUP_M
    group_m = tl.minimum(num_m - first_m, GROUP_M)
    within = pid % (GROUP_M * num_n)
    tile_m = first_m + within % group_m
    tile_n = within // group_m
    rows = tile_m * BM + tl.arange(0, BM)
    cols = tile_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, 128)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # Preserve each 128-wide partial dot and the original scale/accumulation order.
    for group in tl.range(0, K // 128, num_stages=STAGES):
        offsets = group * 128 + kk
        a = tl.load(A + rows[:, None] * K + offsets[None, :], rows[:, None] < M, 0.0)
        w = tl.load(W + cols[None, :] * K + offsets[:, None], cols[None, :] < N, 0.0)
        sa = tl.load(AS + rows * (K // 128) + group, rows < M, 0)
        sw = tl.load(WS + (cols // 128) * (K // 128) + group, cols < N, 0)
        product = tl.dot(a, w, out_dtype=tl.float32, max_num_imprecise_acc=0)
        acc += product * sa[:, None] * sw[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


def select_block_fp8_config(m, n, k, *, mode=None):
    """None selects the unchanged decode/reference kernel; no timing side effects."""
    if _resolve_mode(mode) == "legacy" or m < _PREFILL_MIN_ROWS:
        return None
    # Conservative starting point. Benchmark 128-row tiles separately on the target GPU.
    return PREFILL_CONFIGS[0] if m < 512 else PREFILL_CONFIGS[1]


def block_fp8_matmul(q, weight, scales, weight_scales, output, *, mode=None, config=None):
    """Write a contiguous (M,N) output using the production block-FP8 dispatch."""
    m, k = q.shape
    n = weight.shape[0]
    if not m:
        return output
    selected = select_block_fp8_config(m, n, k, mode=mode) if config is None else config
    if selected is None:
        _block_mm[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
            q, weight, scales, weight_scales, output, m, n, k, 16, 64, num_warps=4,
        )
    else:
        if selected not in PREFILL_CONFIGS:
            raise ValueError(f"Unsupported block FP8 prefill config: {selected}")
        bm, bn, warps, stages = selected
        _block_mm_prefill[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            q, weight, scales, weight_scales, output, m, n, k, bm, bn, stages, 8,
            num_warps=warps, num_stages=stages,
        )
    return output


class Fp8BlockLinearMethod(LinearMethodBase):
    def create_weights(self, layer, input_size, output_size, params_dtype):
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Block FP8 requires BF16/FP16 computation")
        if layer.tp_size != 1:
            raise NotImplementedError("Block FP8 currently requires TP=1")
        if input_size % 128 or output_size % 128:
            raise ValueError("Block FP8 requires input/output sizes divisible by 128")
        if any(size % 128 for size in getattr(layer, "output_sizes", [])):
            raise ValueError("Each packed FP8 projection must be block-aligned")
        layer.weight = nn.Parameter(torch.empty(
            output_size, input_size, dtype=torch.float8_e4m3fn,
        ), requires_grad=False)
        layer.weight.weight_loader = layer.weight_loader
        layer.weight_scale_inv = nn.Parameter(torch.empty(
            output_size // 128, input_size // 128, dtype=torch.float32,
        ), requires_grad=False)
        layer.weight_scale_inv.weight_loader = partial(self.load_scale, layer)

    @staticmethod
    def load_scale(layer, param, loaded_weight, shard=None):
        target = param.data
        if shard is not None:
            # Packed gate/up offsets are measured in scale rows, not weight rows.
            offset = sum(layer.output_sizes[:shard]) // 128
            size = layer.output_sizes[shard] // 128
            target = target.narrow(0, offset, size)
        if target.shape != loaded_weight.shape:
            raise ValueError("Invalid block FP8 weight scale shape")
        # The lovedheart checkpoint stores these scales in BF16; keep FP32 on GPU.
        target.copy_(loaded_weight)

    def apply(self, layer, x, bias=None):
        if x.ndim == 0 or x.shape[-1] != layer.input_size:
            raise ValueError("Block FP8 linear input has the wrong feature dimension")
        shape = x.shape[:-1] + (layer.output_size,)
        q, scales = quantize_fp8_per_group(x.reshape(-1, x.shape[-1]))
        output = torch.empty((q.shape[0], layer.output_size),
                             device=x.device, dtype=layer.compute_dtype)
        if q.shape[0]:
            block_fp8_matmul(q, layer.weight, scales, layer.weight_scale_inv, output)
        if bias is not None:
            output.add_(bias)
        return output.reshape(shape)


class Fp8BlockConfig(QuantizationConfig):
    def __init__(self, checkpoint_config):
        self.ignored = [normalize_weight_name(name) for name in
                        checkpoint_config.get("modules_to_not_convert", [])]

    def get_quant_method(self, layer, prefix):
        sources = [prefix]
        if prefix.endswith(".gate_up_proj"):
            sources = [prefix.removesuffix("gate_up_proj") + name
                       for name in ("gate_proj", "up_proj")]
        ignored = [any(name == skip or name.startswith(skip + ".")
                       for skip in self.ignored) for name in sources]
        if any(ignored) and not all(ignored):
            raise ValueError("Packed gate/up projections need the same quantization scheme")
        return UnquantizedLinearMethod() if all(ignored) else Fp8BlockLinearMethod()
