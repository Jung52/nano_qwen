"""Gated DeltaNet linear attention layer for Qwen3.5 (text-only).

Kernels follow SGLang's GDN design no pure-torch fallback:
    * prefill (extend): FlashQLA ``flash_qla.chunk_gated_delta_rule``
      (TileLang/Triton chunked scan).  FlashQLA has no state-pool indexing,
      so the layer gathers per-slot initial states before the call and
      writes the returned final state back afterwards (both via
      index_select/index_copy_, which are CUDA-Graph-capturable).
      ``state_v_first=True`` selects our K-last [V, K] state layout.
      ``auto_cp`` / ``enable_fwd_cp_cache`` are single-GPU-irrelevant and
      disabled.  The SGLang-derived FLA chunk kernel below
      (``chunk_gated_delta_rule``) is kept as a reference implementation
      for tests/benchmarks.
    * decode: FlashInfer ``gated_delta_rule_decode_pretranspose``
      (``flashinfer.gdn_decode``), which computes the sigmoid gating
      (g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)) inside
      the kernel from the raw ``a`` / ``b`` projections.

Requires CUDA; q/k/v must be bf16/fp16.  The first FlashQLA call JIT-compiles
TileLang kernels (host-side, seconds); the ModelRunner's eager warmup runs
before CUDA Graph capture, so compilation never happens inside a capture.

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
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Chunked delta rule prefill via the SGLang Triton chunk kernel.

    Args:
        query/key/value: (B, S, H, Dk / Dv) bf16/fp16, head-last.
        g: (B, S, H) per-step log decay (fp32 preferred; cast internally)
        beta: (B, S, H) write gate
        initial_state: (N, H, V, K) K-last state pool
        initial_state_indices: (B,) request slots into the state pool
        cu_seqlens: (B + 1,) packed sequence boundaries for variable lengths
    Returns:
        out: (B, S, H, Dv).  The final recurrent state is written in-place
        into ``initial_state`` by the kernel (INPLACE_UPDATE epilogue); the
        kernel's per-chunk ``h`` tensor only holds states *entering* each
        chunk, so it must not be used as the final state.
    """
    from nano_qwen.layers.fla.chunk import chunk_gated_delta_rule as fla_chunk

    assert query.dtype != torch.float32, "Triton chunk kernel requires bf16/fp16 q/k/v"
    # The FLA kernels hard-code contiguous strides (e.g. stride_v = H*V); a
    # strided view (e.g. v split from the packed conv output) would be read
    # at wrong addresses. Materialize contiguous copies — q/k usually already
    # are (GQA repeat_interleave), v is the one that needs the copy.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    o, _, _ = fla_chunk(
        q=query,
        k=key,
        v=value,
        g=g.to(torch.float32),
        beta=beta.to(torch.float32),
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_qk_l2norm_in_kernel=True,
    )
    return o


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
            self.conv_states.index_select(0, idx),       # (B, C, k-1)
            self.recurrent_states.index_select(0, idx),  # (B, H, V, K)
        )

    def _write_state(self, idx: torch.Tensor, conv_state, rec_state):
        idx = idx.reshape(-1)
        # prefill writes one request at a time (2D/3D), batched decode writes
        # many rows at once (3D/4D) — promote single rows to a batch dim.
        if conv_state.dim() == 2:
            conv_state = conv_state.unsqueeze(0)
        if rec_state.dim() == 3:
            rec_state = rec_state.unsqueeze(0)
        self.conv_states.index_copy_(0, idx, conv_state)
        self.recurrent_states.index_copy_(0, idx, rec_state)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            return self._forward_prefill(hidden_states)
        else:
            return self._forward_decode(hidden_states)

    def _forward_prefill(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.prefill_slices is None or context.cu_seqlens_q is None:
            raise RuntimeError(
                "GDN prefill requires packed slices and cu_seqlens"
            )
        if hidden_states.dim() != 3 or hidden_states.shape[0] != 1:
            raise ValueError(
                "GDN prefill expects packed hidden states shaped "
                "[1, total_tokens, hidden_size]"
            )

        _, total_tokens, _ = hidden_states.shape
        raw_qkv_packed = self.in_proj_qkv(hidden_states)
        raw_qkv = raw_qkv_packed.transpose(1, 2)  # (1, C, total_tokens)

        # Depthwise causal convolution has no cu_seqlens argument. Run it on
        # a padded batch so each request starts with an independent zero
        # history, then pack only the valid token ranges for the chunk scan.
        lengths = [end - start for start, end in context.prefill_slices]
        max_len = max(lengths, default=0)
        padded_qkv = raw_qkv.new_zeros(
            len(lengths), self.conv_dim, max_len,
        )
        for batch_idx, (start, end) in enumerate(context.prefill_slices):
            padded_qkv[batch_idx, :, : end - start] = raw_qkv[0, :, start:end]
        padded_conv = F.silu(
            self.conv1d(padded_qkv)[:, :, :max_len]
        )
        qkv = torch.cat(
            [
                padded_conv[batch_idx, :, :length].transpose(0, 1)
                for batch_idx, length in enumerate(lengths)
            ],
            dim=0,
        ).unsqueeze(0)  # (1, total_tokens, C)

        query = qkv[..., : self.key_dim].reshape(1, total_tokens, -1, self.head_k_dim)
        key = qkv[..., self.key_dim:self.key_dim * 2].reshape(
            1, total_tokens, -1, self.head_k_dim
        )
        value = qkv[..., self.key_dim * 2:].reshape(
            1, total_tokens, -1, self.head_v_dim
        )
        z = self.in_proj_z(hidden_states).reshape(
            1, total_tokens, -1, self.head_v_dim
        )
        b = self.in_proj_b(hidden_states)  # (1, total_tokens, Hv)
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
        out = chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=self.recurrent_states,
            initial_state_indices=ctx.state_indices,
            cu_seqlens=ctx.cu_seqlens_q,
            chunk_indices=ctx.prefill_chunk_indices,
        )
        out = out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm(out, z)
        out = out.reshape(1, total_tokens, -1)

        # Persist conv state for the following decode steps: the last
        # kernel-1 raw qkv values (pre-conv1d, pre-silu), matching the decode
        # path's kernel-1 left-context width.  The recurrent final state was
        # already written into the pool in-place by the chunk kernel's
        # INPLACE_UPDATE epilogue — do NOT write h[:, -1] back (it is the
        # state entering the last chunk, not the final state).
        state_width = self.conv_kernel_size - 1
        conv_states = []
        for start, end in context.prefill_slices:
            sequence_qkv = raw_qkv[0, :, start:end]
            if sequence_qkv.size(-1) < state_width:
                padding = raw_qkv.new_zeros(
                    self.conv_dim, state_width - sequence_qkv.size(-1)
                )
                sequence_qkv = torch.cat((padding, sequence_qkv), dim=-1)
            else:
                sequence_qkv = sequence_qkv[:, -state_width:]
            conv_states.append(sequence_qkv)
        new_conv = torch.stack(conv_states, dim=0).clone()
        self.conv_states.index_copy_(0, ctx.state_indices, new_conv)
        return self.out_proj(out)

    def _forward_decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Batched decode: all B sequences in ONE kernel launch per layer. Each
        # row reads and writes its own persistent pool slot via
        # context.state_indices (index_select/index_copy are graph-capturable).
        B = hidden_states.shape[0]
        idx = get_context().state_indices
        if idx is None or idx.numel() == 0:
            raise RuntimeError(
                "GDN decode requires Context.state_indices; "
                "ModelRunner must pass persistent request slots"
            )
        conv_state, rec_state = self._read_state(idx)  # (B,C,k-1), (B,Hv,V,K)

        qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # (B, C, 1)
        # causal conv with cached left-context (kernel-1 values)
        x = torch.cat([conv_state, qkv], dim=-1)  # (B, C, kernel)
        out = F.silu(
            F.conv1d(x, self.conv1d.weight, padding=0, groups=self.conv_dim)
        )
        out = out[:, :, -1:]  # last position
        new_conv_state = x[:, :, -(self.conv_kernel_size - 1):].clone()  # (B, C, k-1)

        out = out.transpose(1, 2)  # (B, 1, C)
        query, key, value = torch.split(
            out, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )
        query = query.reshape(B, 1, -1, self.head_k_dim)  # (B, S, Hk, Dk)
        key = key.reshape(B, 1, -1, self.head_k_dim)
        value = value.reshape(B, 1, -1, self.head_v_dim)

        z = self.in_proj_z(hidden_states).reshape(B, 1, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)  # (B, 1, Hv)
        a = self.in_proj_a(hidden_states)
        if self.gqa_ratio > 1:
            query = query.repeat_interleave(self.gqa_ratio, dim=2)
            key = key.repeat_interleave(self.gqa_ratio, dim=2)

        out, new_rec = decode_gated_delta_rule(
            query, key, value, a, b, self.A_log, self.dt_bias, rec_state,
        )
        self._write_state(idx, new_conv_state, new_rec)

        # Per-head gated norm, then merge heads -> value_dim (matches official:
        # core_attn_out.reshape(-1, head_v_dim) -> norm -> reshape(batch, seq, -1))
        out = out.reshape(-1, self.head_v_dim)  # (B*Hv, Dv)
        z = z.reshape(-1, self.head_v_dim)  # (B*Hv, Dv)
        out = self.norm(out, z)
        out = out.reshape(B, 1, -1)  # (B, S, value_dim)
        return self.out_proj(out)
