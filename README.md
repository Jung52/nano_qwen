# nano_qwen

`nano_qwen` is a lightweight LLM inference engine built on top of
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

The goal of this project is not to build another production-scale inference
framework, but to provide a clean and readable engine for learning and
experimenting with modern LLM inference techniques.

Compared with the original nano-vllm, nano_qwen will progressively introduce
several features from modern inference systems such as vLLM and SGLang, while
trying to keep the implementation small and easy to understand.

## Roadmap

### 1. ModelRunnerV2
- [x] Separate `prepare / execute / sample`
- [x] Persistent InputBatch
- [x] Async-first execution
- [x] GPU-native decode input preparation
- [ ] Async scheduling / CPU-GPU overlap

### 2. Qwen3.5
- [ ] Qwen3.5 Dense support
- [ ] Qwen3.5 MoE support
- [ ] Gated DeltaNet / hybrid model architecture

### 3. Advanced Inference
- [x] Qwen3.5 MTP2 greedy speculative decoding (opt-in, TP=1, FP8 batch size 1/2)
- [ ] FP8 / INT8 / INT4 quantization
- [ ] CUDA Graph optimization
- [ ] Efficient KV Cache management

### 4. Parallelism
- [ ] Tensor Parallelism (TP)
- [ ] Data Parallelism (DP)
- [ ] Expert Parallelism (EP)
- [ ] Pipeline Parallelism (PP)

### 5. Distributed Serving
- [ ] Prefill-Decode disaggregation
- [ ] Distributed inference
- [ ] Advanced scheduling

## Qwen3.5 MTP2

MTP2 is integrated behind an explicit production switch:

```python
from nano_qwen.engine.llm_engine import LLMEngine

engine = LLMEngine(
    model_path,
    tensor_parallel_size=1,
    max_num_seqs=1,
    enable_mtp=True,
)
```

The path loads `mtp.*` weights, allocates a separate MTP KV cache, drafts one
token, verifies two target tokens, and rolls back both full-attention
KV bookkeeping and GDN recurrent state after rejection. It currently supports
greedy sampling only (`temperature <= 1e-6`, deterministic argmax including
ties). Fully greedy batches go directly to argmax using CPU request metadata.
FP8 supports every live batch size 1–8; BF16 remains B1 serial. Stochastic
rejection sampling, tensor parallelism and B16 are not implemented.

`mtp_verify_mode="serial"` captures two ordinary B-request target forwards.
`mtp_verify_mode="reuse"` shares 2B dense rows using ordinary M=B tile choices,
with B-row norms/head and sequential state/attention steps. Each request has
its own first-step snapshot, acceptance and cache catch-up. Verify and MTP
draft/cache-only use exact graphs for every size 1..capacity without padding
state writes. Ordinary decode also captures every exact size when capacity ≤ 8.
See [interfaces](docs/mtp-verify-reuse.md), the [greedy fast-path comparison](benchmarks/reports/mtp_greedy_fast_path/results.md)
and [B1/2/4/8 throughput/TPOT curves](benchmarks/reports/mtp_batch_curve/results.md).
The [B2/B4 cost report](benchmarks/reports/mtp_batch_stages/results.md) records
new off/reuse traces and the cost of per-request rollback. Functional support
is separate from measured speedups; these results cover 4B/4070 Ti SUPER.

MTP RoPE positions start at 1, while its cache rows start at 0. Accepted draft
and bonus pairs are appended in order. Chunked prefill preserves the preceding
hidden state and cumulative cache length; preemption replay resets recurrent
and MTP state before rebuilding the prefix.

Converted FP8 targets can use `mtp_model="/path/to/original/checkpoint"` to
load the original BF16 MTP weights independently. The source must use the same
model architecture and effective RoPE configuration. Run a strict comparison:

```bash
python benchmarks/bench_qwen3_5_mtp_production.py \
  --model /path/to/fp8-target \
  --mtp-model /path/to/original/checkpoint \
  --prompt-tokens 64 \
  --max-tokens 128 \
  --duration 60
```

The benchmark counts actual committed decode tokens, excluding the first
token produced by prefill. Any greedy token mismatch fails, including with
the legacy `--allow-numerical-drift` flag. A speedup is reported only after the
compared prefix matches; duration-limited runs may compare different lengths.

For a production-path long-decode stress test, run:

```bash
python benchmarks/bench_qwen3_5_mtp_production.py \
  --model /path/to/qwen3.5 \
  --stress
```

This runs baseline and MTP for 120 seconds each, allows up to 32768 completion
tokens so the token cap does not end the test early, and prints throughput and
acceptance in 30-second windows. Token divergence makes the comparison fail.

## Project Goal

The final goal is to provide a small but relatively modern inference engine:

**simple enough to learn, modern enough to experiment with.**

## FP8 implementation and measurements

See [optimization summary](docs/fp8-performance-2026-10-14.md),
[benchmark entry points](benchmarks/README.md), and
[retained measurement reports](benchmarks/reports/README.md).
Production regression tests live in `tests/`.
