"""Inference-only RMSNorm fusion, including the checkpoint's rounding points.

Set NANO_QWEN_FUSED_OPS=0 BEFORE importing nano_qwen to use the torch reference
for correctness/performance comparisons. Each norm invocation launches one
kernel; this does not change FP8 GEMMs, GDN recurrence or request ownership.

Row statistics are bit-exact versus torch's ``pow(2).mean(-1).rsqrt()``:
for N % 128 == 0 the kernel reproduces torch's contiguous inner-dim reduction
(32 threads per row, vec4 loads, per-thread accumulation over ``{4t+i+128k}``,
l2r slot merge, 32-lane shuffle tree). Other shapes fall back to torch ops for
the statistics (still one extra small kernel chain, correctness first).
"""
import os
import struct

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


ENABLED = os.environ.get("NANO_QWEN_FUSED_OPS", "1").lower() not in ("0", "false", "off")


def _fp32(v):
    return struct.unpack("f", struct.pack("f", v))[0]


@triton.jit
def _norm_kernel(X, W, R, G, Y, R_OUT, INV, XS: tl.constexpr, RS: tl.constexpr,
                 GS: tl.constexpr, N: tl.constexpr, EPS: tl.constexpr,
                 FACTOR: tl.constexpr, RESIDUAL: tl.constexpr,
                 GEMMA: tl.constexpr, GATED: tl.constexpr,
                 EXTERNAL_INV: tl.constexpr, STRIDED_REDUCE: tl.constexpr,
                 K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N
    dtype = X.dtype.element_ty
    x = tl.load(X + row * XS + col, mask, other=0).to(tl.float32)
    if RESIDUAL:
        x += tl.load(R + row * RS + col, mask, other=0).to(tl.float32)
        tl.store(R_OUT + row * N + col, x.to(dtype), mask)
    # Normalize the unrounded FP32 residual sum, not R_OUT.
    if EXTERNAL_INV:
        # Non-128-multiple shapes: the caller supplies torch's exact
        # rsqrt(mean+eps) computed with torch ops.
        inv = tl.load(INV + row)
    elif STRIDED_REDUCE:
        # Torch's reduce config for N % 128 == 0: block_width 32, vec4, each
        # thread accumulates elements {4t+i+128k} sequentially per slot,
        # merges slots left-to-right, then reduces 32 lanes via shuffles.
        # tl.sum over the merged [32] tensor uses that same 32-lane tree
        # (verified bit-exact against torch.mean for N=128).
        t = tl.arange(0, 32)
        off = 4 * t
        base = row * XS + off
        rbase = row * RS + off
        v = tl.load(X + base).to(tl.float32)
        if RESIDUAL:
            v += tl.load(R + rbase).to(tl.float32)
        s0 = v * v
        v = tl.load(X + base + 1).to(tl.float32)
        if RESIDUAL:
            v += tl.load(R + rbase + 1).to(tl.float32)
        s1 = v * v
        v = tl.load(X + base + 2).to(tl.float32)
        if RESIDUAL:
            v += tl.load(R + rbase + 2).to(tl.float32)
        s2 = v * v
        v = tl.load(X + base + 3).to(tl.float32)
        if RESIDUAL:
            v += tl.load(R + rbase + 3).to(tl.float32)
        s3 = v * v
        for k in tl.static_range(1, K):
            b = base + 128 * k
            rb = rbase + 128 * k
            v = tl.load(X + b).to(tl.float32)
            if RESIDUAL:
                v += tl.load(R + rb).to(tl.float32)
            s0 += v * v
            v = tl.load(X + b + 1).to(tl.float32)
            if RESIDUAL:
                v += tl.load(R + rb + 1).to(tl.float32)
            s1 += v * v
            v = tl.load(X + b + 2).to(tl.float32)
            if RESIDUAL:
                v += tl.load(R + rb + 2).to(tl.float32)
            s2 += v * v
            v = tl.load(X + b + 3).to(tl.float32)
            if RESIDUAL:
                v += tl.load(R + rb + 3).to(tl.float32)
            s3 += v * v
        s = ((s0 + s1) + s2) + s3
        if K == 1:
            # N=128: torch's 32-lane tree matches tl.sum's butterfly (verified).
            s = tl.sum(s, 0)
        else:
            # N>=256: torch's cross-lane tree is the adjacent-pair binary tree.
            s = tl.reshape(s, (16, 2))
            a, b = tl.split(s)
            s = a + b
            s = tl.reshape(s, (8, 2))
            a, b = tl.split(s)
            s = a + b
            s = tl.reshape(s, (4, 2))
            a, b = tl.split(s)
            s = a + b
            s = tl.reshape(s, (2, 2))
            a, b = tl.split(s)
            s = a + b
            s = tl.sum(s, 0)
        inv = tl.rsqrt(s * FACTOR + EPS)
    else:
        inv = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    norm = (x * inv).to(dtype).to(tl.float32)
    w = tl.load(W + col, mask, other=0).to(tl.float32)
    if GEMMA:
        w = (1.0 + w).to(dtype).to(tl.float32)
    if GATED:
        w = w.to(dtype).to(tl.float32)
    y = norm * w
    if GATED:
        # Torch rounds the norm*weight product before its FP32 SiLU gate.
        y = y.to(dtype).to(tl.float32)
        g = tl.load(G + row * GS + col, mask, other=0).to(tl.float32)
        gate = tl.div_rn(g, 1.0 + libdevice.exp(-g))
        y = y * gate
    tl.store(Y + row * N + col, y, mask)


def can_fuse(x, weight, residual=None, gate=None):
    def rows(t):
        return (t.device == x.device and t.dtype == x.dtype and t.shape == x.shape
                and t.stride(-1) == 1 and (t.ndim <= 2 or t.is_contiguous()))
    return (ENABLED and not torch.is_grad_enabled() and x.is_cuda and x.ndim >= 1
            and x.numel() > 0 and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and x.shape[-1] <= 16384 and rows(x)
            and weight.device == x.device and weight.ndim == 1
            and weight.numel() == x.shape[-1] and weight.is_contiguous()
            and (weight.dtype == x.dtype if gate is None else weight.dtype in
                 (x.dtype, torch.float32))
            and (residual is None or rows(residual))
            and (gate is None or rows(gate)))


def fused_norm(x, weight, eps, residual=None, *, gemma=False, gate=None):
    """Call only after can_fuse; preserve input and residual storage."""
    n = x.shape[-1]
    flat = x.reshape(-1, n)
    r = residual.reshape(-1, n) if residual is not None else flat
    g = gate.reshape(-1, n) if gate is not None else flat
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    r_out = torch.empty_like(y) if residual is not None else y
    block = triton.next_power_of_2(n)
    strided = n % 128 == 0 and n >= 128
    external_inv = not strided
    if external_inv:
        # Rare non-128-multiple shapes keep torch's exact statistics.
        base = flat.float()
        if residual is not None:
            base = base + residual.reshape(-1, n).float()
        inv = torch.rsqrt(base.pow(2).mean(dim=-1) + eps)
    else:
        inv = flat  # unused pointer on the strided path
    factor = _fp32(1.0 / n) if strided else 0.0
    k = n // 128 if strided else 1
    _norm_kernel[(flat.shape[0],)](flat, weight, r, g, y, r_out, inv,
        flat.stride(0), r.stride(0), g.stride(0), n, eps, factor,
        residual is not None, gemma, gate is not None, external_inv, strided, k,
        block, num_warps=4 if block <= 2048 else 8, enable_fp_fusion=False)
    return (y, r_out) if residual is not None else y
