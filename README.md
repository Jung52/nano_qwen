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
- [x] Qwen3.5 MTP2 greedy speculative decoding (opt-in, TP=1, batch size 1)
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
token, verifies a two-token target batch, and rolls back both full-attention
KV bookkeeping and GDN recurrent state after rejection. It currently supports
greedy sampling only (`temperature <= 1e-6`); stochastic rejection sampling,
tensor parallelism, and multi-request batching are not implemented.

For hybrid GDN checkpoints, verify uses a two-row CUDA Graph. The GDN layers
process the two rows recurrently and keep the state after the first row for
rejection rollback. Full-attention layers use two-query paged attention, and
the one-layer MTP decode has its own CUDA Graph. Run the production comparison
with:

```bash
python benchmarks/bench_qwen3_5_mtp_production.py \
  --model /path/to/qwen3.5 \
  --prompt-tokens 129 \
  --max-tokens 10240 \
  --duration 120 \
  --allow-numerical-drift
```

The benchmark fails on greedy token mismatch by default. Long runs may use
`--allow-numerical-drift` because a two-query verification kernel and a
one-query decode kernel can select a different top-1 token after a close
logit comparison; the benchmark reports the matching prefix and divergence.

For a production-path long-decode stress test, run:

```bash
python benchmarks/bench_qwen3_5_mtp_production.py \
  --model /path/to/qwen3.5 \
  --stress
```

This runs baseline and MTP for 120 seconds each, allows up to 32768 completion
tokens so the token cap does not end the test early, and prints throughput and
acceptance in 30-second windows. A numerical divergence is reported without
ending the throughput test.

## Project Goal

The final goal is to provide a small but relatively modern inference engine:

**simple enough to learn, modern enough to experiment with.**
