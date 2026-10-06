import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nano_qwen.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        decode_num_splits=0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.decode_num_splits = decode_num_splits
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # chunked prefill / prefix cache
                k, v = k_cache, v_cache
                if self.decode_num_splits == 1 and context.max_seqlen_q == 1:
                    return flash_attn_with_kvcache(
                        q.unsqueeze(1), k_cache, v_cache,
                        cache_seqlens=context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1],
                        block_table=context.block_tables, softmax_scale=self.scale,
                        causal=True, num_splits=1,
                    ).squeeze(1)
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            if self.decode_num_splits == 1 and context.block_tables is not None:
                # Varlen's single-query GQA shortcut rounds differently from
                # ordinary prefill. Use the same fixed-split decode path for
                # singleton rows, including decode rows in a mixed batch.
                for row, (start, end) in enumerate(context.prefill_slices):
                    if end - start != 1:
                        continue
                    o[start:end] = flash_attn_with_kvcache(
                        q[start:end].unsqueeze(1), k_cache, v_cache,
                        cache_seqlens=context.cu_seqlens_k[row + 1:row + 2]
                            - context.cu_seqlens_k[row:row + 1],
                        block_table=context.block_tables[row:row + 1],
                        softmax_scale=self.scale, causal=True, num_splits=1,
                    ).squeeze(1)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True,
                                        num_splits=self.decode_num_splits)
        return o
