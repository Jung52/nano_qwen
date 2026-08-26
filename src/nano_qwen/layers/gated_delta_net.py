"""Gated DeltaNet linear attention layer for Qwen3.5 (text-only).

Kernels follow SGLang's GDN design no pure-torch fallback:
    * prefill (extend): SGLang's self-written Triton chunk kernel, ported to
      ``nano_qwen.layers.fla`` from ``sglang.kernels.ops.attention.fla``
      (itself a flash-linear-attention adaptation): chunk-local cumsum +
      fused intra-chunk (kkt + solve_tril + recompute) + cross-chunk state
      recurrence + output.  q/k are L2-normalized and scaled in-kernel.
    * decode: FlashInfer ``gated_delta_rule_decode_pretranspose``
      (``flashinfer.gdn_decode``), which computes the sigmoid gating
      (g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)) inside
      the kernel from the raw ``a`` / ``b`` projections.

Requires CUDA; q/k/v must be bf16/fp16.

State layout: V-major / K-last ``[N, HV, V, K]``  the SGLang and FlashInfer
convention (K-last).

Layer forward math:
    mixed = silu(conv1d(in_proj_qkv(x)))            # causal depthwise conv
    q, k, v = split(mixed)                          # -> heads
    a, b, z  = in_proj_a/b/z(x)
    y      = delta_rule(q, k, v, a, b, A_log, dt_bias, state)
    y      = RMSNormGated(y, z)
    out    = out_proj(y)

State (conv_states / recurrent_states) is allocated by GatedDeltaNet under
the ModelRunner-managed request lifecycle. Each sequence reads/writes its own
persistent slot via context.state_indices.
"""

import torch
import torch.nn.functional as F
from torch import nn

from nano_qwen.layers.layernorm import RMSNormGated
from nano_qwen.utils.context import get_context


# ---------------------------------------------------------------------------
# Delta-rule kernels (no torch fallback)
# ---------------------------------------------------------------------------


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked delta rule prefill via the SGLang Triton chunk kernel.

    Args:
        query/key/value: (B, S, H, Dk / Dv) bf16/fp16, head-last.
        g: (B, S, H) per-step log decay (fp32 preferred; cast internally)
        beta: (B, S, H) write gate
        initial_state: (N, H, V, K) K-last state pool
        initial_state_indices: (B,) request slots into the state pool
    Returns:
        out: (B, S, H, Dv), final_state: (B, H, V, K) K-last
    """
    from nano_qwen.layers.fla.chunk import chunk_gated_delta_rule as fla_chunk

    assert query.dtype != torch.float32, "Triton chunk kernel requires bf16/fp16 q/k/v"
    o, _, h = fla_chunk(
        q=query,
        k=key,
        v=value,
        g=g.to(torch.float32),
        beta=beta.to(torch.float32),
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=None,
        use_qk_l2norm_in_kernel=True,
    )
    # h: (B, NT, H, V, K) per-chunk states; last chunk is the final state.
    return o, h[:, -1].contiguous()


def decode_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token delta rule decode via FlashInfer (in-kernel gating).

    Args:
        query/key/value: (B, S, H, Dk / Dv) bf16/fp16, S must be 1.
        a: (B, S, Hv) raw input-dependent decay projection
        b: (B, S, Hv) raw update-gate projection
        A_log: (Hv,) log-space decay parameter (fp32)
        dt_bias: (Hv,) time-step bias
        initial_state: (B, H, V, K) K-last (fp32 legacy path, or bf16 for
            K=V=128 and T<=4)
    Returns:
        out: (B, S, H, Dv), final_state: (B, H, V, K) K-last
    """
    from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

    B, S, H, K = query.shape
    _, _, HV, V = value.shape
    assert S == 1, f"FlashInfer decode requires S=1, got S={S}"
    out, new_state = gated_delta_rule_decode_pretranspose(
        q=query.view(B, 1, H, K),
        k=key.view(B, 1, H, K),
        v=value.view(B, 1, HV, V),
        state=initial_state.contiguous(),
        A_log=A_log.detach().to(torch.float32),
        a=a.view(B, 1, HV),
        dt_bias=dt_bias.detach(),
        b=b.view(B, 1, HV),
        scale=None,
        output=None,
        use_qk_l2norm=True,
    )
    return out.view(B, S, HV, V), new_state


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------


class GatedDeltaNet(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.layer_idx = layer_idx
        self.gqa_ratio = self.num_v_heads // self.num_k_heads

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.norm.weight = nn.Parameter(self.norm.weight.data.to(torch.float32))
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # State pools, allocated by ModelRunner after init.
        self.conv_states: torch.Tensor = torch.tensor([])
        self.recurrent_states: torch.Tensor = torch.tensor([])

    def allocate_state_pool(self, num_slots: int):
        """Allocate this layer's persistent runtime state on its CUDA device."""
        device = self.in_proj_qkv.weight.device
        if device.type != "cuda":
            raise RuntimeError("GatedDeltaNet state pool must be allocated on CUDA")

        # The convolution cache stores raw qkv projection values, so it uses
        # the projection/compute dtype. FlashInfer's pretranspose decode uses
        # its bf16-state backend for Qwen3.5's K=V=128 layout; other supported
        # layouts fall back to the legacy fp32-state path.
        conv_dtype = self.in_proj_qkv.weight.dtype
        recurrent_dtype = (
            torch.bfloat16
            if self.head_k_dim == 128 and self.head_v_dim == 128
            else torch.float32
        )
        self.conv_states = torch.zeros(
            num_slots,
            self.conv_dim,
            self.conv_kernel_size - 1,
            dtype=conv_dtype,
            device=device,
        )
        self.recurrent_states = torch.zeros(
            num_slots,
            self.num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            dtype=recurrent_dtype,
            device=device,
        )

    def reset_state(self, slots: int | list[int] | tuple[int, ...] | torch.Tensor):
        """Clear one or more persistent request slots entirely on the GPU."""
        if self.conv_states.numel() == 0 or self.recurrent_states.numel() == 0:
            raise RuntimeError("GatedDeltaNet state pool has not been allocated")

        if isinstance(slots, torch.Tensor):
            slot_indices = slots.to(
                device=self.conv_states.device,
                dtype=torch.int64,
            )
        else:
            slot_indices = torch.as_tensor(
                slots,
                device=self.conv_states.device,
                dtype=torch.int64,
            )
        slot_indices = slot_indices.reshape(-1)
        if slot_indices.numel() == 0:
            return
        self.conv_states.index_fill_(0, slot_indices, 0)
        self.recurrent_states.index_fill_(0, slot_indices, 0)

    def _read_state(self, idx: torch.Tensor):
        # index_select/index_copy are graph-capturable; advanced indexing can
        # require a host-side scalar conversion during CUDA Graph capture.
        idx = idx.reshape(-1)
        return (
            self.conv_states.index_select(0, idx).squeeze(0),
            self.recurrent_states.index_select(0, idx).squeeze(0),
        )

    def _write_state(self, idx: torch.Tensor, conv_state, rec_state):
        idx = idx.reshape(-1)
        self.conv_states.index_copy_(0, idx, conv_state.unsqueeze(0))
        self.recurrent_states.index_copy_(0, idx, rec_state.unsqueeze(0))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            return self._forward_prefill(hidden_states)
        else:
            return self._forward_decode(hidden_states)

    def _forward_prefill(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # (B, C, S)
        qkv = F.silu(self.conv1d(qkv)[:, :, :S])  # causal conv, drop right pad
        qkv = qkv.transpose(1, 2)  # (B, S, C)
        query, key, value = torch.split(
            qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )
        query = query.reshape(B, S, -1, self.head_k_dim)  # (B, S, Hk, Dk)
        key = key.reshape(B, S, -1, self.head_k_dim)
        value = value.reshape(B, S, -1, self.head_v_dim)

        z = self.in_proj_z(hidden_states).reshape(B, S, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)  # (B, S, Hv)
        a = self.in_proj_a(hidden_states)
        beta = torch.sigmoid(b)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        if self.gqa_ratio > 1:
            query = query.repeat_interleave(self.gqa_ratio, dim=2)
            key = key.repeat_interleave(self.gqa_ratio, dim=2)

        ctx = get_context()
        if ctx.state_indices is None or self.recurrent_states.numel() == 0:
            raise RuntimeError(
                "GDN prefill requires the allocated recurrent state pool "
                "and Context.state_indices"
            )
        out, final_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=self.recurrent_states,
            initial_state_indices=ctx.state_indices,
        )
        out = out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm(out, z)
        out = out.reshape(B, S, -1)

        # Persist conv/recurrent state for the following decode steps.
        # conv_state = last kernel-1 raw qkv values (pre-conv1d, pre-silu),
        # matching the decode path's kernel-1 left-context width.
        raw_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # (B, C, S)
        state_width = self.conv_kernel_size - 1
        if raw_qkv.size(-1) < state_width:
            padding = torch.zeros(
                raw_qkv.size(0),
                raw_qkv.size(1),
                state_width - raw_qkv.size(-1),
                dtype=raw_qkv.dtype,
                device=raw_qkv.device,
            )
            new_conv = torch.cat((padding, raw_qkv), dim=-1)
        else:
            new_conv = raw_qkv[:, :, -state_width:]
        new_conv = new_conv.clone()  # (B, C, k-1)
        if ctx.state_indices is not None and self.conv_states.numel() > 0:
            for j, idx in enumerate(ctx.state_indices):
                self._write_state(idx, new_conv[j], final_state[j])
        return self.out_proj(out)

    def _forward_decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Single seq decode: batch B = 1 (engine runs one seq per slot).
        idx = get_context().state_indices[0]
        conv_state, rec_state = self._read_state(idx)  # (C,k-1), (Hv,V,K)
        conv_state = conv_state.unsqueeze(0)  # (1, C, k-1)
        rec_state = rec_state.unsqueeze(0)  # (1, Hv, V, K)

        qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # (1, C, 1)
        # causal conv with cached left-context (kernel-1 values)
        x = torch.cat([conv_state, qkv], dim=-1)  # (1, C, kernel)
        out = F.silu(
            F.conv1d(x, self.conv1d.weight, padding=0, groups=self.conv_dim)
        )
        out = out[:, :, -1:]  # last position
        new_conv_state = x[:, :, -(self.conv_kernel_size - 1):].clone()  # (1, C, k-1)
        self._write_state(idx, new_conv_state.squeeze(0), rec_state.squeeze(0))

        out = out.transpose(1, 2)  # (1, 1, C)
        query, key, value = torch.split(
            out, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )
        query = query.reshape(1, 1, -1, self.head_k_dim)  # (B, S, Hk, Dk)
        key = key.reshape(1, 1, -1, self.head_k_dim)
        value = value.reshape(1, 1, -1, self.head_v_dim)

        z = self.in_proj_z(hidden_states).reshape(1, 1, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)  # (1, 1, Hv)
        a = self.in_proj_a(hidden_states)
        if self.gqa_ratio > 1:
            query = query.repeat_interleave(self.gqa_ratio, dim=2)
            key = key.repeat_interleave(self.gqa_ratio, dim=2)

        out, new_rec = decode_gated_delta_rule(
            query, key, value, a, b, self.A_log, self.dt_bias, rec_state,
        )
        self._write_state(idx, new_conv_state.squeeze(0), new_rec.squeeze(0))

        # Per-head gated norm, then merge heads -> value_dim (matches official:
        # core_attn_out.reshape(-1, head_v_dim) -> norm -> reshape(batch, seq, -1))
        out = out.reshape(-1, self.head_v_dim)  # (Hv, Dv)
        z = z.reshape(-1, self.head_v_dim)  # (Hv, Dv)
        out = self.norm(out, z)
        out = out.reshape(1, 1, -1)  # (B, S, value_dim)
        return self.out_proj(out)
