# nano_qwen

`nano_qwen` is a lightweight LLM inference engine built on top of
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

The goal of this project is not to build another production-scale inference
framework, but to provide a clean and readable engine for learning and
experimenting with modern LLM inference techniques.

Compared with the original nano-vllm, nano_qwen will progressively introduce
several features from modern inference systems such as vLLM and SGLang, while
trying to keep the implementation small and easy to understand.

模块演进、实测收益、性能回退和正确性修复过程见[开发日志](DEVELOPMENT_LOG.md)。

## Greedy sampling

Requests with `temperature <= 1e-6` use deterministic argmax, including ties.
The runner selects the all-greedy fast path from CPU request metadata; mixed
batches retain stochastic sampling for other temperatures. This path skips
softmax and random-number generation for fully greedy batches and works
with ordinary decode. See the [experiment summary](docs/experiments-summary.md).

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
- [ ] DFlash2（独立实验分支，尚未实现）
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

## Project Goal

The final goal is to provide a small but relatively modern inference engine:

**simple enough to learn, modern enough to experiment with.**

## FP8 implementation and measurements

See [optimization summary](docs/fp8-performance-2026-10-14.md),
[experiment results](docs/experiments-summary.md), and
[benchmark entry points](benchmarks/README.md).
Production regression tests live in `tests/`.
