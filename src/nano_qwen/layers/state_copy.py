"""Direct slot-local copies between GDN state pools and their snapshots."""

import torch
import triton
import triton.language as tl


@triton.jit
def _restore_slots(
    conv, recurrent, saved_conv, saved_recurrent, slots,
    CONV_SIZE: tl.constexpr, RECURRENT_SIZE: tl.constexpr,
    CONV_BLOCKS: tl.constexpr, BLOCK: tl.constexpr,
):
    slot = tl.load(slots + tl.program_id(0))
    tile = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    if tile < CONV_BLOCKS:
        offsets += tile * BLOCK
        address = slot * CONV_SIZE + offsets
        conv_values = tl.load(saved_conv + address, mask=offsets < CONV_SIZE, other=0)
        tl.store(conv + address, conv_values, mask=offsets < CONV_SIZE)
    else:
        offsets += (tile - CONV_BLOCKS) * BLOCK
        address = slot * RECURRENT_SIZE + offsets
        recurrent_values = tl.load(saved_recurrent + address, mask=offsets < RECURRENT_SIZE, other=0)
        tl.store(recurrent + address, recurrent_values, mask=offsets < RECURRENT_SIZE)


def restore_state_slots(
    conv: torch.Tensor, recurrent: torch.Tensor,
    saved_conv: torch.Tensor, saved_recurrent: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Restore unique valid slots; all pools and the 1D CUDA indices are contiguous.

    The runner owns index validation/lifetime. No gather tensors, dtype
    conversions or host reads are needed, and the copy is graph-capturable.
    """
    if slots.numel() == 0:
        return
    conv_size = conv.numel() // conv.shape[0]
    recurrent_size = recurrent.numel() // recurrent.shape[0]
    block = 1024
    conv_blocks = triton.cdiv(conv_size, block)
    grid = (slots.numel(), conv_blocks + triton.cdiv(recurrent_size, block))
    _restore_slots[grid](conv, recurrent, saved_conv, saved_recurrent, slots,
                         conv_size, recurrent_size, conv_blocks, block, num_warps=4)
