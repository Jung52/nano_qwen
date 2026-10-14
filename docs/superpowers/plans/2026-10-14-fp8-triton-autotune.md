# FP8 Triton Autotune Implementation Plan

> 已完成的历史实验计划。专项脚本及其测试已归档清理；当前实现与数据索引见 [FP8总结](../../fp8-performance-2026-10-14.md)，生产回归位于 `tests/`。旧命令仅记录实验过程。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将两种 FP8 Linear 的生产 GEMM 统一到 Triton，加入有限 autotune 并用同 shape 的两次正式测量验证。

**Architecture:** per-channel 使用 FP32 K 累加和最后的行列 scale；block-wise 保留每 128 个 K 通道的缩放累加。共享六组配置和固定预算调优计时，按 M/N/K、device 和 dtype 缓存选择；capture 使用缓存，未调优的首次 capture 使用保守配置，不在 capture 内搜索。

**Tech Stack:** 当前 PyTorch 2.12.1、Triton 3.7.1、CUDA Graph/Event，TP=1。

**Spec:** 本次用户要求两种 GEMM 使用 Triton、加入适当 autotune 后测试；正式性能样本仍是每配置两次，autotune 搜索独立列出。

## Global Constraints

- 激活粒度、权重格式、FP32 scale、输出 dtype 和 bias 行为保持现有接口。
- per-channel 继续支持 K/N 为 16 的倍数；block 继续为 128 的倍数；保留空输入与非连续输入处理。
- 候选为 `(BM,BN,warps,stages)`：`(16,32,4,2)`、`(16,64,4,2)`、`(16,128,4,2)`、`(32,64,4,3)`、`(32,128,4,3)`、`(64,128,8,3)`，BK=128。
- 每个候选一次预热、两次单次 graph replay；采用与正式测试相同的 eviction 策略，避开默认自适应长时间 benchmark。
- 正式测试仅两种 MLP 投影、M=8/512；Torch 作为同轮旧实现对照，生产路径完全使用 Triton。
- 结果写入新的 triton_autotuned 文件，保留前轮原始报告。

## Review Focus

- K/N 尾部 masking 与跨 BM 的输出行。
- 不同 FP16/BF16 输出 dtype、零行、bias 和非连续输入。
- 首次冷 capture 不搜索，预热后的 capture 使用选定配置，重复调用不重新搜索。
- 固定控制核与 autotuned 生产核的计时 ID 必须区分。
- 每项仍仅两次正式测量，记录选中配置与调优预算。

### Task 1: 生产 Triton GEMM 和有限 autotune

**Files:** 修改 `src/nano_qwen/quantization/fp8.py`、`fp8_block.py`；新增 `src/nano_qwen/quantization/triton_gemm.py` 与 `benchmarks/test_fp8_triton_gemm.py`。

- [x] 先写测试并观察失败：禁用 torch._scaled_mm 后 per-channel 仍工作；两种方法对显式 FP32 反量化参考的 relative L2 <0.005；K=80/N=48、零行、bias、两种输出 dtype；逐候选校验；冷热 capture 和缓存复用。
- [x] 实现 `fp8_per_channel_mm(q,weight,sa,sw,out_dtype)` 与 `fp8_block_mm(q,weight,sa,sw,out_dtype)`，返回 `[M,N]`。
- [x] 共用配置与两次计时的 `triton.autotune` 回调；保留 raw `_block_mm` 供固定控制实验。
- [x] 运行新测试及现有量化/graph 测试，确认输出与缓存行为。

### Task 2: 同轮测试与报告

**Files:** 修改 `benchmarks/bench_fp8_granularity.py` 和对应测试；生成 `benchmarks/reports/fp8_granularity/triton_autotuned.{json,csv,md}`。

- [x] 原生 GEMM 阶段调用生产 wrapper；加入旧 Torch 的相同 q/weight/scales 对照和完整 Linear 对照。
- [x] 固定控制 D 不再复用 autotuned block 的 GEMM/Linear；量化阶段继续复用。
- [x] 写入每 shape 的选中 BM/BN/warps/stages 与搜索预算。
- [x] 预热完毕后仅运行两轮正式测量，第二轮倒序；检查精度、波动、计时 ID 和字节账本。
- [x] 一次独立只读审查，修复实际问题，复核完整测试后把文件放回用户工作区。

Triton 的配置、key 与自定义 do_bench 接口依据：[官方 autotune 文档](https://triton-lang.org/main/python-api/generated/triton.autotune.html)；缓存实现与 graph 行为以已安装版本源码和实测验证为准。

## Execution Result

- 两种生产 GEMM 均为 Triton；六组配置按 shape、device、dtype 缓存。调优 capture 显式使用设备对应的 stream；冷 capture 使用保守配置，warm capture 复用已选 tile。
- 工作区完整验证：`/home/wei/miniconda3/envs/code/bin/python -m pytest -q`，86 passed；精度、尾部、bias、dtype、逐配置、冷热 capture 和缓存选择检查通过。
- 一次 native sweep：12 行结果、32 个独立计时 ID、每 ID 两个正式样本；未重复性能 sweep。报告：[triton_autotuned.md](../../../benchmarks/reports/fp8_granularity/triton_autotuned.md)。原 results.* 报告保留。
- M=512 的 per-channel GEMM：gate_up 旧 Torch 941.248 μs → 新 Triton 310.784 μs；down 492.480 μs → 184.640 μs。部分小 M 与 block 样本波动超过 5%，不用于推断整模型排序。
- 独立只读审查提出设备 stream 问题，修复并通过复核。实际 DRAM/L2 counter 仍受 `ERR_NVGPUCTRPERM` 限制；本次未执行多 GPU 硬件验证。
- 日志和旧文件备份保存于 `logs/triton-autotune-2026-10-14/`；未提交更改，保留工作区中无关改动。
