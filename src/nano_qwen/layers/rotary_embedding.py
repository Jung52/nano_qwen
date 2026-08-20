from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb


class InterleavedMRoPE(nn.Module):
    """Text-only MRoPE for Qwen3.5 (partial rotary, interleaved layout).

    Semantics verified against transformers Qwen3_5RotaryEmbedding
    (modeling_qwen3_5.py:91-186, 562-610):

    1. Partial rotary: only the first rotary_dim = int(head_dim *
       partial_rotary_factor) = 64 dims are rotated; the remaining
       head_dim - 64 = 192 dims pass through untouched.
    2. inv_freq = base ** (-arange(0, 64, 2) / 64) -> 32 freqs;
       cos/sin are 32-dim, applied to each half of the 64-dim rot slice
       (equivalent to the official rotate_half scheme: official cos is
       two concatenated identical 32-dim segments, so
       upstream apply_rotary_emb gives identical numbers).
    3. mrope_section [11, 11, 10] is an *interleaved* layout for 3D
       (T/H/W) position streams. For text-only inference all three
       streams share the same 1D positions, so the interleaved
       reordering is a numerical identity (freqs[0] == freqs[1] ==
       freqs[2]); the parameter is kept for API parity with the config.
    """

    def __init__(
        self,
        head_size: int,
        partial_rotary_factor: float,
        mrope_section: list[int],
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = int(head_size * partial_rotary_factor)
        self.is_partial = self.rotary_dim < head_size
        self.mrope_section = mrope_section
        inv_freq = 1.0 / (base**(torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_partial:
            q_rot, q_pass = query[..., :self.rotary_dim], query[..., self.rotary_dim:]
            k_rot, k_pass = key[..., :self.rotary_dim], key[..., self.rotary_dim:]
            q_rot = apply_rotary_emb(q_rot, cos, sin)
            k_rot = apply_rotary_emb(k_rot, cos, sin)
            query = torch.cat((q_rot, q_pass), dim=-1)
            key = torch.cat((k_rot, k_pass), dim=-1)
        else:
            query = apply_rotary_emb(query, cos, sin)
            key = apply_rotary_emb(key, cos, sin)
        return query, key
