# nano_qwen

`nano_qwen` is a lightweight LLM inference engine built on top of
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

The project provides a clean and readable engine for learning and experimenting
with modern LLM inference techniques. The current focus is single-GPU
Qwen3.5 Dense text inference, including hybrid attention, FP8 quantization,
CUDA Graphs, and asynchronous execution.

**simple enough to learn, modern enough to experiment with.**

## 已实现

以下状态以当前 `main` 分支为准。

### Qwen3.5 Dense 与混合注意力

- [x] Qwen3.5 Dense 纯文本模型与 BF16 推理路径。
- [x] Gated DeltaNet 线性注意力与 gated GQA 全注意力混合架构。
- [x] 按请求维护 GDN conv/recurrent state，支持批量 decode。
- [x] 纯 decode 的 GDN recurrent state 按 slot 原地更新，减少状态搬运。

### 执行与调度

- [x] ModelRunnerV2 风格的输入准备、模型执行与采样分离。
- [x] Persistent InputBatch 与稳定的请求到 slot 映射。
- [x] GPU 上准备 decode 输入，采样 token 留在 GPU 供下一步使用。
- [x] 双缓冲异步 D2H 输出与批次队列。
- [x] Continuous batching、chunked prefill 与 decode 优先的 token 预算。
- [x] 可选的稳定 decode 一批前瞻调度，为 CPU 调度与 GPU 执行重叠提供路径。
- [x] 确定性 greedy fast path 与混合温度采样。

`async_scheduling=True` 显式启用一批前瞻，默认关闭。当前覆盖已完成
prefill、没有等待请求的稳定 decode 阶段；新请求到达或 KV block 不足时，
先排空在途批次，再回到常规调度。该路径不代表所有负载都能获得相同的重叠收益。

### CUDA Graph、FP8 与状态管理

- [x] Decode CUDA Graph 与按 token bucket 捕获的 piecewise prefill CUDA Graph。
- [x] Prefill 投影输出 buffer 复用与 Graph/eager 路径统计。
- [x] FP8 W8A8：per-channel 与 block-wise 两条量化路径。
- [x] Triton FP8 GEMM、tile 调优与按 shape 缓存。
- [x] FP8 checkpoint 转换与共享输入量化。
- [x] 分页 KV block 分配、扩展与回收，以及独立的 GDN 状态池。
- [x] 固定 batch 计时、Prefill/Decode 分析、Perfetto trace 与正确性回归入口。

当前支持范围为 `tensor_parallel_size=1`。混合模型的 prefix cache 尚未接通：
只复用 attention KV 不足以恢复 GDN 状态，需同时处理 conv/recurrent state
快照；当前配置会拒绝该组合。

## 后续计划

### 已落地：LMCache PD 状态交接（单请求同步原型）

- [x] 接入真实 LMCache 0.5.6 L1：KV/GDN/请求进度直接拷入共享 CPU tensor 池，
      decode 端按描述符 mmap 恢复；HTTP 仅承载 JSON 控制消息（无 payload 序列化/传输）。
- [x] 实测单卡串行交接（CUDA Graph）：暂停 14–17ms 且几乎不随 prompt 长度增长
      （对比序列化+HTTP 路径 252–464ms），恢复后 TPOT 与 baseline 持平。
- [x] 回归覆盖：真实 L1 分配/租约、跨进程交接、输出一致、fd/泄漏回收与通用状态校验。
      用法见 [docs/lmcache-pd-shm.md](docs/lmcache-pd-shm.md)，开发记录见 DEVELOPMENT_LOG.md 第 11–12 节。

### 已落地：decode 合批（D 上限可配置 1/2/4/8）

- [x] D 侧恢复入口改为 `require_resume_ready()`（无在途 batch、decode 边界）+
      `require_resume_capacity()`（空闲 GDN slot + 全部活跃请求与恢复请求的 KV 页预算）。
- [x] D worker 按步推进、步间恢复 READY 传输；B1→B2→B3→B4→…→B1 步型与 slot 复用
      经正确性矩阵验证（含逐位恢复校验与 replay 确定性）。
- [x] decode CUDA Graph 捕获扩至 [1..16]+32 档（此前 B3/B5-B7 缺图走 eager，
      单步 65ms 为 graph 批次的 6 倍，是 D8 掉速根因）。
- [x] GDN 分块预填充非末块 64 对齐（非 64 对齐切点导致递推状态分歧，BF16 下同样
      复现；对齐后各路径与参考逐 token 一致，副本混跑 divergence 归零）。
- [x] 双 5090 逐档加压：D8 稳定扛到 200ms 到达节拍（4.1 req/s，节奏 12ms 恒稳）；
      合批上限 8 为当前负载饱和点。开发记录见 DEVELOPMENT_LOG.md 第 15 节。

### 下一步（PD 方向）

- [ ] 更高负载档（>5 req/s）：先补单卡 decode 吞吐（B8 步 11ms 已接近单卡极限）；
      B12/B16 固定批仍有收益（1044/1329 tok/s），留作储备。
- [ ] 双副本侧长 prefill 竞争导致短请求节奏恶化（10.6→71.6ms），考虑 prefill
      抢占或独立 prefill 通道。
- [ ] 继续分析异步调度的 CPU/GPU 空隙与适用负载，完善正确性和性能验证。

### 后续探索

- [ ] DFlash / speculative decoding，作为独立实验方向；当前 `main` 尚未实现。
- [ ] 按需要评估 INT8 / INT4 量化。
- [ ] Qwen3.5 MoE 模型支持。

### 暂缓

- [ ] Tensor Parallelism（TP）。
- [ ] Expert Parallelism（EP）。
- [ ] Data Parallelism（DP）。
- [ ] Pipeline Parallelism（PP）与更完整的多 GPU 分布式执行。

近期优先完善单 GPU 路径和缓存/PD 实验，多 GPU 并行暂不作为当前开发重点。

## 文档与实验

- [开发日志](DEVELOPMENT_LOG.md)：模块演进、性能回退与正确性修复过程。
- [FP8 实现与优化总结](docs/fp8-performance-2026-10-14.md)：量化、
  Triton GEMM 和 CUDA Graph 的实现细节。
- [优化结果总结](docs/experiments-summary.md)：关键测试条件、实测指标与限制。
- [Benchmark 与测试入口](benchmarks/README.md)：运行脚本、计时与 trace 使用方式。

原始报告、图表、JSON/CSV 与 trace 保留在本地实验目录，不随 Git 分发。
历史结果对应各自的源码版本与测试配置，具体口径见上述文档。
生产回归测试位于 `tests/`：

```bash
PYTHONPATH=src python -m pytest -q tests
```
