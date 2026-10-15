# MTP verification modes

`mtp_verify_mode="serial"` is the default and correctness reference.
`mtp_verify_mode="reuse"` enables the production two-token reuse path.
Both require the existing greedy TP1/B1 MTP setup; reuse additionally requires
an FP8 target. MTP remains opt-in through `enable_mtp=True`.

```python
from nano_qwen.engine.llm_engine import LLMEngine

engine = LLMEngine(
    "FP8_TARGET_DIR",
    mtp_model="ORIGINAL_BF16_DIR",
    enable_mtp=True,
    mtp_verify_mode="reuse",  # "serial" retains two ordinary B1 forwards
    tensor_parallel_size=1,
    max_num_seqs=1,
    max_num_batched_tokens=512,
    max_model_len=2048,
    use_prefill_cudagraph=False,  # Matches the current benchmark protocol.
)
```

Reuse batches dense projections, output projections and FP8 MLP matrices into
two rows of the same request. GDN conv/recurrent updates remain sequential,
with the first update saved for rejection/EOS rollback. Attention remains
single-query per row; input/post/final norms and the vocabulary head keep M1
arithmetic. Unquantized linear exceptions in a mixed FP8 checkpoint also keep
M1 arithmetic. Accepted-draft cache-only catch-up is shared by both modes.

`GatedDeltaNet.forward_dense_pre()` supplies qkv/z/b/a;
`forward_decode_core()` updates one token per request and returns the gated
core before `out_proj`. Attention exposes the equivalent core before `o_proj`.
Ordinary GDN decode and reuse call the same state-update implementation.

The call-local Context policy selects the current device/shape/dtype's tuned
M1 configuration to launch M2, without changing the M2 autotuner entry. Eager
execution warms a missing M1 selection; cold capture fails explicitly. Normal
engine graph setup warms ordinary B1 decode before capturing verification.
No 4070 configuration is transplanted to another GPU.

Eager and CUDA Graph call the same configured implementation. Set the mode
when creating the engine. Changing the config after capture without recapturing
raises an error rather than silently replaying another mode's graph.

```bash
PYTHONPATH=src python benchmarks/bench_qwen3_5_mtp_production.py --model FP8_TARGET_DIR --mtp-model ORIGINAL_BF16_DIR --mtp-verify-mode reuse --prompt-tokens 64 --max-tokens 128
PYTHONPATH=src python benchmarks/bench_mtp_verify_reuse.py --model FP8_TARGET_DIR --mtp-model ORIGINAL_BF16_DIR --output logs/reuse-layers.json
PYTHONPATH=src python benchmarks/bench_mtp_verify_reuse.py --model FP8_TARGET_DIR --mtp-model ORIGINAL_BF16_DIR --timing --output logs/reuse-timing.json
```

The first command compares off with the selected production mode. The second
compares serial/reuse layers and state for one eager/Graph verification window.
The third first checks a full eager completion, then times off/serial/reuse
twice each, with Graph and strict full-completion equality. It also records
amortized decode TPOT, TTFT, request time and allocator memory. Neither command
uses projection patches or overwrites M2 tuning entries.

See [production results](../benchmarks/reports/mtp_reuse_production/results.md)
and the [earlier experiment](../benchmarks/reports/mtp_verify_reuse/results.md).
The length/content matrix, new hotspot trace, B2/4/8 and 5090 measurements are
subsequent work; the B1 guard remains in place.
