import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm used by Qwen3.5 attention and decoder layers.

    Same RMS normalization as RMSNorm, but the learned weight is an
    *increment* around 1.0: output = x_normed * (1 + weight).
    weight starts at 0, so the module is a no-op before loading.

    Checkpoint evidence (Qwen3.5-0.8B): q_norm/k_norm weights are
    bf16 tensors centered near 0 with range ~[-1, 1], i.e. the
    1+weight semantics; linear_attn.norm (standard semantics) is a
    different layer written in the next step.

    Note: unlike upstream RMSNorm, this does NOT normalize in-place.
    Upstream's x.float() only copies when x is not float32; on CPU
    float32 inputs x.float() returns the same tensor and the
    in-place mul corrupts the caller's input. Non-inplace is safe
    for every dtype.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(orig_dtype) * (1.0 + self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float() + residual.float()
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(orig_dtype) * (1.0 + self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


class RMSNormGated(nn.Module):
    """Gated RMSNorm used at the GDN (linear attention) output.

    Normalizes hidden_states, scales by the learned weight (standard
    semantics: multiply directly, init 1), then gates with silu(gate):

        y = RMSNorm(hidden_states) * weight * silu(gate)

    In GatedDeltaNet this is fed (out, z) where z = in_proj_z(hidden).
    Checkpoint evidence (Qwen3.5-0.8B): linear_attn.norm.weight is
    float32, 128 dims (= linear_value_head_dim), centered near 1
    (0.52~1.06) -> standard multiply semantics, unlike GemmaRMSNorm's
    1+weight. Kept non-inplace: x.float() aliases float32 inputs, so
    in-place ops would corrupt the caller's tensor.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        # Matches official Qwen3_5RMSNormGated exactly: normalize in fp32,
        # multiply the learned weight in the input dtype, then gate with
        # silu(gate) in fp32 (bf16 * fp32 promotes to fp32), cast back.
        orig_dtype = hidden_states.dtype
        x = hidden_states.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(orig_dtype) * self.weight.to(orig_dtype)
        x = x * torch.nn.functional.silu(gate.float())
        return x.to(orig_dtype)
