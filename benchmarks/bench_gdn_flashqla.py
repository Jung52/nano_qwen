"""Benchmark: nano_qwen's GDN chunked-prefill kernel vs FlashQLA forward.

Baseline = the lowest-level kernel the production Qwen3.5 GatedDeltaNet
prefill actually calls:

    nano_qwen.layers.fla.chunk.chunk_gated_delta_rule

with the production flags (``use_qk_l2norm_in_kernel=True``, ``cu_seqlens=None``,
real state pool + ``initial_state_indices``).  No GDN math is reimplemented
here; inputs are constructed in the exact layout the layer uses right before
the chunk call (see ``GatedDeltaNet._forward_prefill``).

FlashQLA = ``from flash_qla import chunk_gated_delta_rule``.  The GPU this
script targets (RTX 4070 Ti SUPER, SM89) is NOT in FlashQLA v0.1.2's
official SM list (90/100/103/120/121), so the script never assumes that a
successful import means kernel support: FlashQLA is probed with one real
kernel execution and marked UNSUPPORTED/FAILED on the first failure, after
which the baseline keeps benchmarking alone.

Usage (on a CUDA box, from the repo root):

    python benchmarks/bench_gdn_flashqla.py
"""

from __future__ import annotations

import argparse
import inspect
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

# Make the repo package importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nano_qwen.layers.fla.chunk import (  # noqa: E402  (baseline kernel)
    chunk_gated_delta_rule as baseline_chunk,
)

# ---------------------------------------------------------------------------
# Qwen3.5 GDN head config, taken from the project's model config
# (models/qwen config.json text_config: linear_num_key_heads=16,
# linear_num_value_heads=32, linear_key/value_head_dim=128) and the GQA
# repeat in GatedDeltaNet._forward_prefill.
# ---------------------------------------------------------------------------
H_QK = 16          # linear_num_key_heads (before GQA repeat)
H_V = 32           # linear_num_value_heads
HEAD_DIM = 128     # linear_key_head_dim == linear_value_head_dim
GQA_RATIO = 2      # H_V // H_QK, applied via repeat_interleave in production

DTYPE = torch.bfloat16
DEFAULT_SEQ_LENS = [1024, 2048, 4096, 8192]
WARMUP = 20
REPEAT = 100

FLASHQLA_STATUS: str | None = None   # None = untested, else see main()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def get_env_info() -> dict[str, str]:
    info = {}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        info["cc"] = f"{major}.{minor}"
    else:
        info["gpu"] = "N/A"
        info["cc"] = "N/A"
    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda or "N/A"
    return info


def get_flashqla_version() -> str:
    try:
        import flash_qla
        return str(getattr(flash_qla, "__version__", "unknown"))
    except Exception:
        return "NOT INSTALLED"


def print_env() -> None:
    info = get_env_info()
    print("Environment")
    print(f"GPU: {info['gpu']}")
    print(f"CC: {info['cc']}")
    print(f"Torch: {info['torch']}")
    print(f"CUDA: {info['cuda']} (torch runtime)")
    print(f"FlashQLA: {get_flashqla_version()}")
    print()


# ---------------------------------------------------------------------------
# Inputs — identical tensors for both kernels.
#
# Mirrors GatedDeltaNet._forward_prefill right before the chunk call:
#   q/k: bf16 (1, S, H_QK, 128) -> repeat_interleave(GQA_RATIO) -> (1, S, 32, 128)
#   v:   bf16 (1, S, 32, 128)
#   g:   fp32 (1, S, 32), log-space decay, negative
#        (production: g = -exp(A_log) * softplus(a + dt_bias) <= 0)
#   beta: fp32 (1, S, 32) in (0, 1)  (production: sigmoid(b))
#   initial_state: (1, 32, 128, 128) K-last pool; FP32 by default (FlashQLA
#        official convention; the baseline kernel casts it to fp32 internally
#        and its legacy path uses fp32 pools).  Pass --state-dtype bf16 to
#        mirror the current repo pool rule for K=V=128.
#   state_indices: (1,) int64 on GPU — production passes Context.state_indices.
# ---------------------------------------------------------------------------

def make_inputs(
    seq_len: int,
    state_dtype: torch.dtype,
    norm_in_kernel: bool,
) -> dict:
    device = "cuda"
    q = torch.randn(1, seq_len, H_QK, HEAD_DIM, dtype=DTYPE, device=device)
    k = torch.randn(1, seq_len, H_QK, HEAD_DIM, dtype=DTYPE, device=device)
    v = torch.randn(1, seq_len, H_V, HEAD_DIM, dtype=DTYPE, device=device)
    # Production GQA handling: q/k are repeated to H_V heads before the call.
    q = q.repeat_interleave(GQA_RATIO, dim=2)
    k = k.repeat_interleave(GQA_RATIO, dim=2)

    if not norm_in_kernel:
        # Fallback used only when FlashQLA lacks use_qk_l2norm_in_kernel:
        # normalize once, identically, for both kernels (same eps as the
        # baseline's l2norm_fwd).
        q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6).to(DTYPE)
        k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6).to(DTYPE)

    g = -(0.1 + 0.9 * torch.rand(1, seq_len, H_V, dtype=torch.float32, device=device))
    beta = torch.rand(1, seq_len, H_V, dtype=torch.float32, device=device)
    # Nonzero on purpose: a zero state would hide gather/scatter bugs.
    initial_state = torch.randn(
        1, H_V, HEAD_DIM, HEAD_DIM, device=device
    ).to(state_dtype)
    state_indices = torch.tensor([0], dtype=torch.int64, device=device)
    scale = HEAD_DIM ** -0.5
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "state_indices": state_indices,
        "scale": scale,
    }


# ---------------------------------------------------------------------------
# Kernel wrappers — both return (output, final_state) in the same convention:
#   output:      (B, S, H, V) in q.dtype
#   final_state: (B, H, V, K) K-last
# The baseline kernel writes the final state back into the state pool
# in-place, so each call gets a fresh clone.
# ---------------------------------------------------------------------------

def run_baseline(inputs: dict, norm_in_kernel: bool) -> tuple[torch.Tensor, torch.Tensor]:
    # The chunk kernel writes the TRUE final state into the state pool
    # in-place (INPLACE_UPDATE epilogue); ``h`` only holds per-chunk states
    # BEFORE each chunk, so h[:, -1] is NOT the final state.
    state = inputs["initial_state"].clone()
    o, _, h = baseline_chunk(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        scale=inputs["scale"],
        initial_state=state,
        initial_state_indices=inputs["state_indices"],
        cu_seqlens=None,
        use_qk_l2norm_in_kernel=norm_in_kernel,
    )
    return o, state.contiguous()


def _flashqla_params():
    try:
        import flash_qla
    except Exception:
        return None
    try:
        return inspect.signature(flash_qla.chunk_gated_delta_rule).parameters
    except (TypeError, ValueError):
        return None  # opaque extension; treat as unsupported


def run_flashqla(inputs: dict, norm_in_kernel: bool) -> tuple[torch.Tensor, torch.Tensor]:
    import flash_qla

    params = _flashqla_params()
    if params is None:
        raise RuntimeError(
            "flash_qla.chunk_gated_delta_rule signature is not introspectable"
        )
    kwargs: dict = {
        "q": inputs["q"],
        "k": inputs["k"],
        "v": inputs["v"],
        "g": inputs["g"],
        "beta": inputs["beta"],
    }
    if "scale" in params:
        kwargs["scale"] = inputs["scale"]
    if "initial_state" in params:
        kwargs["initial_state"] = inputs["initial_state"].clone()
    if "cu_seqlens" in params:
        kwargs["cu_seqlens"] = None
    if "use_qk_l2norm_in_kernel" in params:
        kwargs["use_qk_l2norm_in_kernel"] = norm_in_kernel
    if "output_final_state" in params:
        kwargs["output_final_state"] = True
    if "state_v_first" in params:
        # FlashQLA's default state layout is K-first [N, HV, K, V]; pass
        # state_v_first=True so in/out states follow our K-last [V, K].
        kwargs["state_v_first"] = True

    res = flash_qla.chunk_gated_delta_rule(**kwargs)

    # Normalize the return into (output, final_state) without touching the
    # kernel: accept (o, ht), (o, None, h), or per-chunk h with 5 dims.
    if isinstance(res, torch.Tensor):
        out, state_like = res, None
    elif isinstance(res, (tuple, list)):
        out = res[0]
        tensors = [t for t in res[1:] if isinstance(t, torch.Tensor) and t.dim() >= 4]
        state_like = tensors[-1] if tensors else None
    else:
        raise RuntimeError(f"unexpected return type from flash_qla: {type(res)!r}")
    if state_like is None:
        raise RuntimeError("flash_qla did not return a final state tensor")
    final_state = (
        state_like[:, -1].contiguous() if state_like.dim() == 5
        else state_like.contiguous()
    )
    return out, final_state


# ---------------------------------------------------------------------------
# Correctness — compare in fp32, before any timing.
# ---------------------------------------------------------------------------

def check_correctness(
    out_a: torch.Tensor,
    state_a: torch.Tensor,
    out_b: torch.Tensor,
    state_b: torch.Tensor,
) -> dict:
    metrics = {}

    def _compare(name: str, a: torch.Tensor, b: torch.Tensor):
        a = a.detach().float().reshape(-1)
        b = b.detach().float().reshape(-1)
        if a.numel() != b.numel():
            metrics[f"{name} shape mismatch"] = f"{tuple(a.shape)} vs {tuple(b.shape)}"
            return
        diff = (a - b).abs()
        cos = F.cosine_similarity(a, b, dim=0).item()
        metrics[f"{name} max_abs_diff"] = diff.max().item()
        metrics[f"{name} mean_abs_diff"] = diff.mean().item()
        metrics[f"{name} cosine_similarity"] = cos

    _compare("output", out_a, out_b)
    _compare("final_state", state_a, state_b)
    return metrics


# ---------------------------------------------------------------------------
# Timing — CUDA events, synchronize after every round.
# ---------------------------------------------------------------------------

def benchmark(fn, warmup: int = WARMUP, repeat: int = REPEAT) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global FLASHQLA_STATUS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="initial_state dtype: fp32 (FlashQLA official / legacy path) or "
             "bf16 (current repo pool rule for K=V=128)",
    )
    parser.add_argument(
        "--seq-lens",
        type=lambda s: [int(x) for x in s.split(",")],
        default=DEFAULT_SEQ_LENS,
        help="comma-separated sequence lengths",
    )
    args = parser.parse_args()
    state_dtype = torch.float32 if args.state_dtype == "fp32" else torch.bfloat16

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark requires CUDA")
    print_env()

    # Decide the q/k normalization path once, based on FlashQLA's API.
    params = _flashqla_params()
    norm_in_kernel = params is not None and "use_qk_l2norm_in_kernel" in params
    if not norm_in_kernel:
        print("note: flash_qla has no use_qk_l2norm_in_kernel param; "
              "pre-normalizing q/k with torch for both kernels\n")

    for seq_len in args.seq_lens:
        print(f"seq={seq_len}")
        inputs = make_inputs(seq_len, state_dtype, norm_in_kernel)

        # Correctness first: one real call of each kernel on identical inputs.
        base_out, base_state = run_baseline(inputs, norm_in_kernel)
        print(f"baseline out: {tuple(base_out.shape)} "
              f"state: {tuple(base_state.shape)} {base_state.dtype}")

        flashqla_ms: str | float = "UNSUPPORTED"
        if FLASHQLA_STATUS is None:
            try:
                fla_out, fla_state = run_flashqla(inputs, norm_in_kernel)
                metrics = check_correctness(base_out, base_state, fla_out, fla_state)
                for name, value in metrics.items():
                    if isinstance(value, float):
                        print(f"{name}: {value:.6f}")
                    else:
                        print(f"{name}: {value}")
                FLASHQLA_STATUS = "OK"
            except Exception as e:
                # A real kernel execution failed (e.g. SM89 not in the
                # supported arch list).  Print the full error, keep going
                # with the baseline only, and do not try to patch around it.
                FLASHQLA_STATUS = "UNSUPPORTED/FAILED"
                print("FlashQLA: UNSUPPORTED/FAILED — first real execution raised:")
                traceback.print_exc()
        else:
            print("FlashQLA: verified on seq #1 (skipped re-check)")

        baseline_ms = benchmark(
            lambda: run_baseline(inputs, norm_in_kernel)
        )
        print(f"baseline: {baseline_ms:.4f} ms")

        if FLASHQLA_STATUS == "OK":
            flashqla_ms = benchmark(
                lambda: run_flashqla(inputs, norm_in_kernel)
            )
            print(f"FlashQLA: {flashqla_ms:.4f} ms")
            print(f"speedup: {baseline_ms / flashqla_ms:.2f}x")
        else:
            print("FlashQLA: UNSUPPORTED")
        print()


if __name__ == "__main__":
    main()
