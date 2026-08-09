import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_decode_kernel(
    last_token_ids,
    seq_lens,
    last_block_ids,
    input_ids,
    positions,
    context_lens,
    slot_mapping,
    num_seqs,
    block_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_seqs

    seq_len = tl.load(seq_lens + idx, mask=mask)
    last_block_id = tl.load(last_block_ids + idx, mask=mask)

    tl.store(input_ids + idx, tl.load(last_token_ids + idx, mask=mask), mask=mask)
    tl.store(positions + idx, seq_len - 1, mask=mask)
    tl.store(context_lens + idx, seq_len, mask=mask)
    tl.store(
        slot_mapping + idx,
        last_block_id * block_size + (seq_len - 1) % block_size,
        mask=mask,
    )


def prepare_decode(
    last_token_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    last_block_ids: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    context_lens: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_seqs: int,
    block_size: int,
) -> None:
    block = 128
    grid = (triton.cdiv(num_seqs, block),)
    _prepare_decode_kernel[grid](
        last_token_ids,
        seq_lens,
        last_block_ids,
        input_ids,
        positions,
        context_lens,
        slot_mapping,
        num_seqs,
        block_size,
        BLOCK_SIZE=block,
    )