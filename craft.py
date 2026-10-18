import triton
import triton.language as tl

@triton.jit
def _quantize_groups(x, q, s,
                     K: tl.constexpr,
                     G: tl.constexpr):
    row, group = tl.program(0), tl.program_id(1)
    offsets = group * 128 + tl.arange(0, 128)
    values = tl.load(x + row * K + offsets).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=0)
    scale = tl.where(amax == 0, 1.0, amax/448.0)
    values = tl.minimum(tl.maximum(values / scale, -448.0), 448.0)
    quantized = tl.inline_asm_elementwise(
        "{ .reg .b16 lo, hi; "
        "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; "
        "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; "
        "mov.b32 $0, {lo, hi}; }",
        constraints="=r,f,f,f,f", args=[values],
        dtype=tl.float8e4nv, is_pure=True, pack=4,
    )
    tl.store(q + row * K + offsets, quantized)
    tl.store(s + row * G + group, scale)