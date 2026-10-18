from dataclasses import dataclass, field
import torch


@dataclass(slots=True)
class BatchDescriptor:
    """Structured batch description for graph dispatch and spec decoding.

    Replaces the bare ``is_prefill`` boolean. ``uniform_token_count`` is 1 for
    standard decode, ``spec_steps`` for speculative decoding, and ``None``
    for prefill (variable token counts per request).
    """
    mode: str = "decode"  # "prefill" | "decode" | "spec_decode" | "draft"
    num_tokens: int = 0
    num_reqs: int = 0
    uniform_token_count: int | None = None
    max_query_len: int = 0

    @classmethod
    def for_requests(cls, seqs):
        counts = [s.num_scheduled_tokens for s in seqs]
        if not counts or min(counts) <= 0:
            raise ValueError("Batch requires positive scheduled row counts")
        speculative = [s.is_speculative for s in seqs]
        if any(speculative):
            if not all(speculative) or any(n != 2 for n in counts):
                raise ValueError("Speculative verify requires two rows for every request")
            mode, uniform = "spec_decode", 2
        elif any(s.is_prefill for s in seqs):
            mode, uniform = "prefill", None
        else:
            if any(n != 1 for n in counts):
                raise ValueError("Ordinary decode requires one row per request")
            mode, uniform = "decode", 1
        return cls(mode, sum(counts), len(seqs), uniform, max(counts))


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    batch_descriptor: BatchDescriptor | None = None
    cudagraph_mode: str | None = None  # "full" | "piecewise" | None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    state_indices: torch.Tensor | None = None
    # Python-side packed ranges are fixed during CUDA Graph capture. Keeping
    # them in the context avoids calling CUDA-tensor.tolist() in model.forward.
    prefill_slices: list[tuple[int, int]] | None = None
    prefill_chunk_indices: torch.Tensor | None = None
    # True decode requests packed beside prefill; a one-token prefill tail
    # is different and must retain the prefill state-update kernel.
    prefill_decode_rows: list[int] = field(default_factory=list)
    is_verify: bool = False
    # Reuse dispatch: 2B dense rows use this device's tuned ordinary M=B tiles.
    gemm_reference_m: int | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, state_indices=None, prefill_slices=None, prefill_chunk_indices=None, batch_descriptor=None, cudagraph_mode=None, prefill_decode_rows=None, is_verify=False, gemm_reference_m=None):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        batch_descriptor,
        cudagraph_mode,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        state_indices,
        prefill_slices,
        prefill_chunk_indices,
        [] if prefill_decode_rows is None else prefill_decode_rows,
        is_verify,
        gemm_reference_m,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
