# FP8 Granularity Performance Attribution Implementation Plan

> 已完成的历史实验计划。专项脚本及其测试已归档清理；当前实现与数据索引见 [FP8总结](../../fp8-performance-2026-10-14.md)，生产回归位于 `tests/`。旧命令仅记录实验过程。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用最少的控制变量实验区分激活量化、权重量化粒度、GEMM 实现和缓存造成的速度差异，并给出 scale 体积与实际显存流量的数值。

**Architecture:** 先分段测量现有两条 FP8 路径，再用固定 Triton GEMM 交叉激活与权重粒度。仅对差异最大的 shape 采集访存计数器；每个配置只做两轮正式测量。

**Tech Stack:** 当前推理环境中的 PyTorch、Triton、CUDA Event、CUDA Graph、可用时的 Nsight Compute。

**Spec:** 用户本次要求：控制变量、控制 shape，弄清 per_channel 与 block_wise 的速度差异和访存量；测试计划尽量简洁，同一个实验仅仅执行两次。下文定义具体执行规格。本轮只写计划。

## Global Constraints

- 每个 `(实验, shape, 变体, 阶段)` 恰好保留两次正式测量；不因波动自动补跑。
- 相同输入、kernel 与缓存条件的重复配置直接复用两次结果，避免不同实验名称下重复执行。
- 编译、固定三次预热、graph capture 和正确性检查属于准备阶段，单独记账；不使用 autotune 或自适应 do_bench 隐藏增加测量次数。
- CUDA Event 的每个正式计时窗口只包含一次 graph replay，graph 内只包含一次待测操作；不另设数百次迭代的内循环。
- 两轮使用相同输入、相同 BF16 原始权重和 seed=23；第二轮倒序运行变体。每次正式测量前恢复相同缓存条件。
- TP=1，BF16 输入和输出，FP8 E4M3 权重与量化激活，FP32 scale，连续输入，权重存储为 `[N,K]`，无 bias。
- 使用 `/home/wei/miniconda3/envs/code/bin/python`；当前该环境为 PyTorch 2.12.1+cu130、Triton 3.7.1。shell 默认 Python 是另一套旧环境，不能混用。
- 只新增 benchmark 与报告；不修改生产量化实现，不重新转换整模型，不重跑长输出吞吐或完整精度评估。

## Review Focus

- 权重必须从同一个 BF16 来源生成两种量化数据，避免比较不同来源 checkpoint。
- `M` 必须记录算子实际行数；decode 是当前 batch，prefill 是实际 token 数，不能只按请求 batch 比较。
- 诊断实验统一直接 FP32 到 FP8 的 RN 转换；原生量化转换差异单独保留。
- 微基准缓存与 NCU cache/replay 设置必须一致；热缓存数据不能当成整模型 DRAM 需求。
- 只保留两个原始样本；计数器不可用或样本波动时，报告必须保留未解决的归因。

## 已核对的实现与待验证假设

| 项目 | per_channel | block_wise |
|---|---|---|
| 激活 scale | 每 token 一份，`[M,1]` | 每 token、每 128 个 K 通道一份，`[M,K/128]` |
| 权重 scale | 每输出通道一份，`[N,1]` | 每 128×128 权重块一份，`[N/128,K/128]` |
| 激活量化 grid | `M` 个 program；整行归约 | `M×K/128` 个 program；128 元素归约 |
| GEMM | `torch._scaled_mm(use_fast_accum=False)` | `_block_mm`，固定 `BM=16, BN=64, BK=128, num_warps=4` |
| scale 应用 | per-token/per-channel scale 传入库 GEMM | 每个 K 块的 dot 后应用两种 scale，再累加 |
| FP8 转换 | `tl.store` 隐式转换 | inline asm 直接 FP32 到 FP8 RN 转换 |

代码依据：[fp8.py](../../../src/nano_qwen/quantization/fp8.py)、[fp8_block.py](../../../src/nano_qwen/quantization/fp8_block.py)、[convert.py](../../../src/nano_qwen/quantization/convert.py)。

假设一：decode 的整行量化只启动 4/8/16 个 program，并行工作不足；分组量化增加 program 数后可能更快。假设二：GEMM 后端与 tile 对小 M 的处理是主要差异。假设三：prefill 时分组 scale 的加载与逐 K 块缩放累加会增加成本。这些都是待测假设，现有吞吐表不能直接确认。

block 权重量化同时缩短 K 分组、合并 128 个输出通道，与 per-channel 并非严格嵌套粒度；不能预设其权重精度一定更高。精度只在同一输入/权重上做算子误差比较，不推导模型质量。

## 实验 shape 与字节账本

定义 `Y[M,N] = X[M,K] @ W[N,K].T`。本地 4B 的 `hidden_size=2560`、`intermediate_size=9216`，所以只选两种主要 MLP 投影：

| 投影 | K | N | M |
|---|---:|---:|---|
| gate_up_proj | 2560 | 18432 | 4、8、16、512 |
| down_proj | 9216 | 2560 | 4、8、16、512 |

模型配置依据：[4b/config.json](../../../../models/4b/config.json)；投影组合依据：[qwen3_5.py](../../../src/nano_qwen/models/qwen3_5.py)。`M=512` 是固定 prefill 算子形状，不能与 batch=512 等同。

对于 128 整除的 K、N，scale 唯一数据体积为：

```text
Sa_channel = 4M              Sa_block = 4M(K/128)
Sw_channel = 4N              Sw_block = 4(N/128)(K/128)

激活量化的一次逻辑读写：B_quant = 3MK + Sa
预量化 GEMM 的一次逻辑读写：B_mm = MK + NK + 2MN + Sa + Sw
完整 Linear 的一次逻辑读写：B_linear = 4MK + NK + 2MN + 2Sa + Sw

ΔB_linear = 8M(K/128 - 1) + 4N(K/16384 - 1)
```

这些数值按张量元素各读一次/写一次计算，不代表实际 DRAM 字节数；GEMM 的跨 tile 重复加载、重复 scale 读取和缓存命中另行测量。两种方案的 FP8 数据部分体积相同。

| M=8 的投影 | FP8 权重 | 激活 scale channel → block | 权重 scale channel → block | 完整 Linear 逻辑流量增量 |
|---|---:|---:|---:|---:|
| gate_up_proj | 45 MiB | 32 B → 640 B | 72 KiB → 11.25 KiB | −60,992 B，约 −0.128% |
| down_proj | 22.5 MiB | 32 B → 2304 B | 10 KiB → 5.625 KiB | +64 B，约 +0.00027% |

由上式推导，权重 scale 的体积比是 `K/16384`，所以本例 block 权重 scale 更少。到 `M=512`，完整 Linear 逻辑增量分别为 15,616 B（约 0.0219%）和 286,336 B（约 0.635%）。最终报告同时列出这个账本和实测计数器，避免把 scale 增长倍数当成总访存增长倍数。

### Task 1: 准备最小实验工具

**Files:**
- Create: `benchmarks/bench_fp8_granularity.py`，包含数据准备、诊断 kernel、阶段计时与结果输出。
- Create at execution time: `benchmarks/reports/fp8_granularity/` 下的 JSON、CSV 和短报告。
- Reference only: `src/nano_qwen/quantization/fp8.py`、`fp8_block.py`、`convert.py`、`benchmarks/test_fp8_block.py`。

**Interfaces:**
- CLI：`--experiment native|controlled|traffic`、`--model`、`--rounds 2`、`--json-out`；shape 表固定，不另做参数搜索。
- 每条结果含 `experiment, M, N, K, variant, stage, round, gpu_us, cache_policy`；计数器与正确性指标分字段存放，不能把缺失计数器填成零。

- [ ] 从 `/home/wei/code/models/4b` 读取第 0 层 gate/up/down BF16 权重；gate/up 按现有顺序拼接。用同一份权重离线生成两种量化数据，所有转换在计时之外。
- [ ] 为两种 K 生成最大 `M=512` 的固定 BF16 输入，小 M 使用同一前缀；统一复用数据，不在两轮间重新采样。
- [ ] 准备 cache eviction buffer，至少为 `2×torch.cuda.get_device_properties(0).L2_cache_size`。当前 GPU 为 66 SM、48 MiB L2，取 96 MiB buffer。每次计时前访问 buffer 并同步，操作在 Event 窗口外；缓存策略记为 `evicted`，不宣称完全清空 L1。
- [ ] 用显式 FP32 反量化后的矩阵乘验证诊断 GEMM，relative L2 < 0.005，检查零行和有限输出。另记录相对同一 BF16 数据的 FP32 GEMM 参考误差；参考计算关闭 TF32，均在计时之外。

### Task 2: 两轮完成原生路径和控制变量实验

**Files:** 使用 Task 1 的脚本；输出 `native.json`、`controlled.json`。

- [ ] **实验 1，原生路径分段计时。** 两种投影 × `M={4,8,16,512}` × 两种原生方案。分别 capture `quant_only`、`gemm_only`（提前提供 q/scale）、`linear` 三个 graph，每个阶段做两次单次 replay 的 Event 计时。记录 `t_quant`、`t_mm`、`t_linear`；三者独立测量，缓存/调度差异可能使 `t_quant+t_mm` 不等于 `t_linear`。
- [ ] **实验 2，同一 GEMM 的 2×2 对照。** 只测两种投影 × `M={8,512}`。在 benchmark 内从 `_block_mm` 扩展共同 Triton GEMM，固定 `BM=16, BN=64, BK=128, num_warps=4`、FP32 累加与 `max_num_imprecise_acc=0`。仅通过 constexpr 的激活/权重 scale 索引改变粒度；四种情况均逐 K 块应用 scale，不对某个变体改成 epilogue。D 分支直接使用原生 `_block_mm`，复用实验 1 对应样本。

| 控制变体 | 激活量化 | 权重量化 |
|---|---|---|
| A | per-token | per-channel |
| B | per-token per-128-channel | per-channel |
| C | per-token | 128×128 block |
| D | per-token per-128-channel | 128×128 block |

- [ ] 两种诊断激活量化均使用与 block 路径相同的直接 FP32 到 FP8 RN 转换；整行与 128 元素归约使用相同转换逻辑。对 A/B/C/D 收集 `quant_only`、`gemm_only`、`linear`，每配置两轮：A/C 共享整行 RN 量化结果，B/D 复用实验 1 分组量化结果，D 的 GEMM 与 Linear 也直接复用。A→B 和 C→D 检查激活粒度，A→C 和 B→D 检查权重粒度，并保留交互效应。
- [ ] 同一份 A 的 q、权重与两种 scale，再喂给原生 `torch._scaled_mm` 做两轮 `gemm_only` 对照。该对照与共同 A GEMM 的差值归为后端路径差异，包含 tile、scale 应用位置和累加策略；不把共同 A 当成优化后的生产实现。
- [ ] 第二轮倒序运行；输出两个原始值、算术均值和 `abs(t1-t2)/mean`。差异超过 5% 标记为不稳定，仍不补跑；方案差异与样本波动相当时不做明确排序。

以下命令是执行阶段新增脚本的约定，当前脚本尚未实现：

```bash
PYTHONPATH=src /home/wei/miniconda3/envs/code/bin/python benchmarks/bench_fp8_granularity.py --experiment native --model /home/wei/code/models/4b --rounds 2 --json-out benchmarks/reports/fp8_granularity/native.json
PYTHONPATH=src /home/wei/miniconda3/envs/code/bin/python benchmarks/bench_fp8_granularity.py --experiment controlled --model /home/wei/code/models/4b --rounds 2 --json-out benchmarks/reports/fp8_granularity/controlled.json
```

### Task 3: 重点 shape 的实际流量与归因报告

**Files:** 使用同一脚本的 `traffic` 模式；输出 `traffic.json`、两轮 `.ncu-rep`、`summary.md`。

- [ ] **实验 3，访存采样。** 从实验 1 选择原生两方案差距最大的一个 `(K,N)`，仅测 `M={8,512}`、两种原生方案、两轮采集。NVTX 范围内执行一次完整 Linear，并分别统计量化与 GEMM 的 DRAM read/write、L2 bytes，以及 kernel 名称、grid/block、寄存器数。
- [ ] NCU 优先只收 DRAM/L2 必要指标；具体指标名由已安装版本确认，例如 `dram__bytes_read.sum`、`dram__bytes_write.sum`、`lts__t_bytes.sum`。每轮使用相同 cache/replay 设置与 Task 1 eviction，eviction 位于 NVTX 范围外；采用 `--cache-control none --clock-control none`，graph 按 node 采集。NCU 的时间不混入实验 1/2 的速度结果。
- [ ] NCU 可能为多个计数器内部 replay；报告实际 pass 数。若需要多 pass，缩减指标到单 pass；不能满足时保留缺失指标，不暗中多执行。这样两轮正式采集不会变成多轮应用 replay。[NVIDIA Profiling Guide，查阅 2026/10，§2.2.3 与 §2.6.3](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay)
- [ ] 当前只读 `ncu --query-metrics` 已返回 `ERR_NVGPUCTRPERM`。计数器访问需要管理员权限或管理员配置；执行阶段若仍不可用，完成实验 1/2 和逻辑账本，实际 DRAM/L2 字段记为 unavailable，保留此项未解决，不更改驱动设置。[NVIDIA Performance Counter Permissions，查阅 2026/10](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
- [ ] 生成一个短报告：每 shape 的原生量化/GEMM/完整 Linear 延迟占比、共同 kernel 四格结果、scale 字节数、可用的 DRAM/L2 数值、算子误差、两次测量波动。只对这两类 MLP 投影归因；若它们不能解释整模型差距，明确剩余投影和执行路径尚未覆盖，不自动扩展实验矩阵。

报告按以下证据判断：量化差异是否占原生 Linear 差异的大部分；完全相同 q/weight/scale 下后端差异是否明显；共同 kernel 中哪一粒度变化影响最大；实际 DRAM 增量是否与逻辑账本不同。并行度、逐块缩放和缓存分别作为与证据相符的解释，两个样本和 grid 数本身不证明唯一瓶颈。

## Sources

[nano_qwen，2026/10 查阅] 本地量化与模型代码及 4B 配置。上文相应文件链接。

[NVIDIA Profiling Guide，2026/10 查阅] NVIDIA. “Nsight Compute Profiling Guide.” [官方文档](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。

[NVIDIA Performance Counter Permissions，2026/10 查阅] NVIDIA. “ERR_NVGPUCTRPERM Permission issue with Performance Counters.” [官方说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)。
