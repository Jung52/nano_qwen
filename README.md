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
- [ ] Async-first execution
- [ ] GPU-native input preparation
- [ ] Async scheduling / CPU-GPU overlap

### 2. Qwen3.5
- [ ] Qwen3.5 Dense support
- [ ] Qwen3.5 MoE support
- [ ] Gated DeltaNet / hybrid model architecture

### 3. Advanced Inference
- [ ] MTP / speculative decoding
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
