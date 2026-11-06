# Vendored FLA kernels (adapted from fla-org/flash-linear-attention).
# The kernel family still uses tl.make_block_ptr, which Triton>=3.3 marks
# deprecated in favour of TensorDescriptor/tl.make_tensor_descriptor.
# Suppressing on purpose:
#   - This repo targets Ada (sm89) GPUs, which have no Hopper (sm90+) TMA
#     unit; Triton lowers TensorDescriptor to plain global loads there, so
#     migrating the 60+ call sites buys no performance.
#   - The new API needs a global triton.set_allocator hook and changes
#     out-of-bounds semantics, a real numerics risk for these tuned kernels.
# Revisit when re-syncing with upstream fla (which still uses make_block_ptr
# today) or when running on Hopper+ hardware.
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"tl\.make_block_ptr is deprecated\.",
    category=UserWarning,
)
