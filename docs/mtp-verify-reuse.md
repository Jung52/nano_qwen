# MTP verification modes

`mtp_verify_mode="serial"` is the default and correctness reference.
`mtp_verify_mode="reuse"` enables the production two-token reuse path.
Both require greedy TP1 MTP. FP8 supports B1–8; BF16 remains B1 serial only.
Reuse requires an FP8 target. MTP remains opt-in through `enable_mtp=True`.

```python
from nano_qwen.engine.llm_engine import LLMEngine

engine = LLMEngine(
    "FP8_TARGET_DIR",
    mtp_model="ORIGINAL_BF16_DIR",
    enable_mtp=True,
    mtp_verify_mode="reuse",  # "serial" retains two ordinary B-request forwards
    tensor_parallel_size=1,
    max_num_seqs=8,  # Exact live/subset batches 1..8; B16 remains guarded.
    max_num_batched_tokens=512,
    max_model_len=2048,
    use_prefill_cudagraph=False,  # Matches the current benchmark protocol.
)
```

Reuse batches dense projections, output projections and FP8 MLP matrices into
2B rows across B requests and two time steps. External rows are request-major
`[A0,A1,B0,B1]`; dense matrices are time-major `[A0,B0,A1,B1]`. Each GDN time
step updates B indexed slots and saves the first state independently. Attention
uses one query per request per step; norms, vocabulary head and BF16 linear
exceptions keep ordinary M=B arithmetic. Each request's accept mask controls
its rollback, effective cache length and accepted-draft cache-only catch-up.

`GatedDeltaNet.forward_dense_pre()` supplies qkv/z/b/a;
`forward_decode_core()` updates one token per request and returns the gated
core before `out_proj`. Attention exposes the equivalent core before `o_proj`.
Ordinary GDN decode and reuse call the same state-update implementation.

The call-local Context policy selects the current device/shape/dtype's tuned
M=B configuration to launch M=2B, without changing the M=2B autotuner entry.
Eager execution warms a missing reference selection; cold capture fails
explicitly. Engine setup warms every ordinary shape up to the configured
capacity before capturing verification.
No 4070 configuration is transplanted to another GPU.

Eager and CUDA Graph call the same configured implementation. Set the mode
when creating the engine. Changing the config after capture without recapturing
raises an error rather than silently replaying another mode's graph.

Verify, full MTP draft and cache-only each have exact graph banks for every
size 1..capacity. Ordinary decode also captures every exact size when capacity
is at most 8, so B3/5/6/7 use the same actual M=B numerical reference.
Accepted subsets and dynamic B8→1 transitions select the actual size without padding.
Verification is scheduled separately from prefill/ordinary final rows. New
admissions respect slots held by in-flight/partial requests; a KV-blocked replay
allows an allocated continuation to finish and free capacity.

`temperature <= 1e-6` uses deterministic argmax, including tied highest logits;
the runner supplies an all-greedy hint from CPU Sequence temperatures so a
fully greedy batch skips softmax, RNG and stochastic sampling entirely.
Mixed/non-greedy batches retain the existing sampling path. This adds no GPU
temperature scalar read. Tiny-temperature categorical sampling was insufficient to match the verify path when BF16 top logits tied.

```bash
PYTHONPATH=src python benchmarks/bench_qwen3_5_mtp_production.py --model FP8_TARGET_DIR --mtp-model ORIGINAL_BF16_DIR --mtp-verify-mode reuse --prompt-tokens 64 --max-tokens 128
PYTHONPATH=src python benchmarks/bench_mtp_verify_reuse.py --model FP8_TARGET_DIR --mtp-model ORIGINAL_BF16_DIR --output logs/reuse-layers.json
PYTHONPATH=src python benchmarks/bench_mtp_verify_reuse.py --model FP8_TARGET_DIR --mtp-model ORIGINAL_BF16_DIR --timing --output logs/reuse-timing.json
PYTHONPATH=src python benchmarks/bench_mtp_small_batch.py --model FP8_TARGET_DIR --mtp-model ORIGINAL_BF16_DIR --output logs/mtp-b1.json
PYTHONPATH=src python benchmarks/bench_mtp_batch2.py --model FP8_TARGET_DIR --mtp-model ORIGINAL_BF16_DIR --output logs/mtp-b2.json
PYTHONPATH=src python benchmarks/bench_mtp_batch_gates.py --model FP8_TARGET_DIR --mtp-model ORIGINAL_BF16_DIR --output logs/mtp-b8-gates.json
PYTHONPATH=src python benchmarks/bench_mtp_interleaved.py --model FP8_TARGET_DIR --mtp-model ORIGINAL_BF16_DIR --batches 1 2 4 8 --cases same64 --samplers fast --output logs/mtp-batch-curve.json
PYTHONPATH=src python benchmarks/profile_mtp_reuse.py --model FP8_TARGET_DIR --mtp-model ORIGINAL_BF16_DIR --output logs/mtp-profile.json
PYTHONPATH=src python benchmarks/bench_mtp_interleaved.py --model FP8_TARGET_DIR --mtp-model ORIGINAL_BF16_DIR --output logs/mtp-greedy.json
```

The first command compares off with the selected production mode. The second
compares serial/reuse layers and state for one eager/Graph verification window.
The third first checks a full eager completion, then times off/serial/reuse
twice each, with Graph and strict full-completion equality. It also records
amortized decode TPOT, TTFT, request time and allocator memory. Neither command
uses projection patches or overwrites M2 tuning entries.

See [B1 matrix, reuse trace and B2 results](../benchmarks/reports/mtp_small_batch/results.md),
[original B1 production results](../benchmarks/reports/mtp_reuse_production/results.md)
and the [earlier experiment](../benchmarks/reports/mtp_verify_reuse/results.md).
The [B1/2/4/8 report](../benchmarks/reports/mtp_batch_curve/results.md) uses
the new greedy fast baseline with two interleaved samples per configuration.
The resident off control retains MTP weights/graphs in memory but disables their
execution. Read gains per tested workload, rather than treating sparse points as
a continuous break-even threshold. B16 and 5090 remain untested; ordinary
decode B64 is a separate baseline. Future kernel work needs a new B2/B4 trace;
old serial/B1 trace percentages do not describe these larger batches.
