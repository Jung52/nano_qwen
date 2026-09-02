# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# Inference-only entry point: the autograd.Function wrapper and all training
# decorators are removed.  Kept the FLA return convention (o, None, h) so the
# caller in gated_delta_net.py unpacks the same triple.

from typing import Optional

import torch

from nano_qwen.layers.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from nano_qwen.layers.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from nano_qwen.layers.fla.chunk_o import chunk_fwd_o
from nano_qwen.layers.fla.cumsum import chunk_local_cumsum
from nano_qwen.layers.fla.index import prepare_chunk_indices
from nano_qwen.layers.fla.l2norm import l2norm_fwd

CHUNK_SIZE = 64


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: torch.LongTensor | None = None,
):
    g = chunk_local_cumsum(
        g, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )

    # fused kkt + solve_tril + recompute_w_u
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return g, o, A, w, h, v_new


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: torch.LongTensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Chunked GDN prefill kernel (inference).

    Args:
        q (torch.Tensor): queries of shape [B, T, H, K] (head-last).
        k (torch.Tensor): keys of shape [B, T, H, K].
        v (torch.Tensor): values of shape [B, T, H, V].
        g (torch.Tensor): (forget) gating tensor in log space, [B, T, H].
        beta (torch.Tensor): write gates, [B, T, H].
        scale: attention scale; defaults to ``1 / sqrt(K)``.
        initial_state: [N, H, V, K] K-last (or None).
        initial_state_indices: [N] state-pool indices when using a pool.
        cu_seqlens: [N+1] for variable-length inputs (B must be 1 then).
    Returns:
        o: [B, T, H, V]
        h: per-chunk states [B, NT, H, V, K]; ``h[:, -1]`` is the final state.
    """
    assert q.dtype == k.dtype == v.dtype
    assert (
        q.dtype != torch.float32
    ), "chunk_gated_delta_rule does not support float32. Please use bfloat16."
    assert (
        len(beta.shape) == 3
    ), "beta must be of shape [B, T, H] (head-last)."
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    _, o, _, _, h, _ = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return o.to(q.dtype), None, h
