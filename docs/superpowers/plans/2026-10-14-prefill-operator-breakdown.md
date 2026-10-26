# Prefill 算子耗时检查计划

> 已完成的历史实验计划。专项脚本及其测试已归档清理；当前实现与数据索引见 [FP8总结](../../fp8-performance-2026-10-14.md)，生产回归位于 `tests/`。旧命令仅记录实验过程。

> **For agentic workers:** 实施时使用 superpowers:executing-plans，按下列任务逐项执行。已执行，结果见 benchmarks/reports/prefill_breakdown/results.md。

**Goal:** 找出 prefill 的主要耗时算子，以及 block-wise 相对 per-channel 的额外时间分别来自量化、GEMM、其他模型算子还是执行路径。

**Architecture:** 先测真实 prefill 的分段时间，再对耗时或差值最大的算子使用保存的真实输入做微基准，最后仅对确认的 FP8 热点做相同 tile 的机制对照。

**Tech Stack:** 仓库当前 PyTorch/Triton、CUDA Event、CPU/CUDA profiler、piecewise CUDA Graph；TP=1。

**Spec:** 本次用户要求检查 prefill 耗时并给出精简测试计划；沿用每个实验配置仅两次正式执行的限制。

## 已有证据与待验证原因

已有算子报告（本地产物 `benchmarks/reports/fp8_granularity/triton_autotuned.md`） 中，M=512 的耗时如下，单位 μs：

| 投影 | per-channel 量化 | block-wise 量化 | per-channel GEMM | block-wise GEMM |
|---|---:|---:|---:|---:|
| gate_up，K=2560 | 10.752 | 18.800 | 310.784 | 338.784 |
| down，K=9216 | 26.112 | 59.392 | 184.640 | 211.072 |

这些是合成激活的独立分段测量；不能直接相加还原完整 Linear 或整模型时间。block-wise down GEMM 的两次波动约 11.8%，差值需进一步验证。

当前两种 GEMM 都以 BK=128 遍历 K。区别是 per-channel 在 K 累加结束后乘 scale，而 block-wise 在每个 K 块内读取 scale、缩放并累加。gate_up 有 20 个 K 块，down 有 72 个。block-wise 激活量化分别启动 M×20、M×72 个 program，per-token 为 M 个；program 增多不能等同于时间按倍数增加。

已有账本的 Linear 唯一数据逻辑增量仅约 0.022% / 0.635%。这不是实测 DRAM 流量，也没有计入跨 tile 重读。分组缩放、scale 访问、累加依赖与资源占用均是待验证原因，不能仅凭分组数归因。

## Global Constraints

- 第一轮只用两种 FP8 方案和一个 workload：batch=8，每请求 64 token，M=512；M 是本次 packed 输入的 token 总数。真实生产场景若不是该长度，优先替换为原场景，不同时扫描 batch/长度。
- 使用同一原始模型派生的 checkpoint、相同 token ID、BF16 compute dtype、TP=1、相同 KV 容量和运行参数；记录每个 Linear 的实际 quant_method，核对 ignored modules。
- 新请求、无 prefix cache、无混合 decode、无抢占；token budget 至少 512，确保一次完整 prefill。每次重置 KV/GDN 状态，并核对 cu_seqlens、实际 M、graph bucket 和路径命中。
- 模型加载、JIT、autotune、CUDA Graph capture 和预热单独记录，不计入稳态结果；预热后核对候选选择已经缓存。
- 每个实验配置只保留两次正式样本；两轮顺序为 per-channel→block-wise、block-wise→per-channel。预热次数和诊断执行次数也显式记账，不用循环隐藏额外采样。
- 正式计时关闭 profiler；profiler 另设诊断实验，每配置也仅两次，时间不与正式结果混用。样本波动超过 5% 时标记不稳定，不自动增加重复次数。
- 全模型不人为清空 L2；微基准沿用既有 eviction 策略且放在计时范围之外。两类结果分开解释。
- 不改生产算子。实际 DRAM/L2 counter 当前受 ERR_NVGPUCTRPERM 限制，不把分配量、逻辑字节或 profiler memory 字段称为实际访存量。

## Review Focus

- profiler 的 CPU scope 时间不能代替 GPU kernel 时间；子算子与父阶段不重复求和，跨 stream 的重叠和空隙分别报告。
- Graph 可能把 Linear 的 M 扩大到 bucket；比较记录 actual M 与 bucket M，确认两方案没有不同 padding 或 fallback。
- GDN 的 recurrent/conv state 会被更新，预热和前一次样本不能改变下一次样本的初始状态。
- 全模型耗时排行和两方案差值排行分别输出：共同的大算子可能占时很高，却不解释量化方案的差距。
- 机制对照必须统一 tile、warps、stages、dtype 和累加精度；比较缩放位置时复用完全相同的 q/weight/scales。

## 对应代码与检查范围

| 阶段 | 具体算子 | 代码入口 |
|---|---|---|
| FP8 Linear | 激活量化、GEMM、必要的 contiguous/copy、bias | [fp8.py](../../../src/nano_qwen/quantization/fp8.py)、[fp8_block.py](../../../src/nano_qwen/quantization/fp8_block.py) |
| MLP | gate_up_proj、SiluAndMul、down_proj | [Qwen3_5MLP](../../../src/nano_qwen/models/qwen3_5.py) |
| GDN 投影和核心 | in_proj_qkv/z、a/b gate、padding/pack、depthwise conv、chunk scan、gated norm、状态读写、out_proj | [gated_delta_net.py](../../../src/nano_qwen/layers/gated_delta_net.py)、[FLA chunk](../../../src/nano_qwen/layers/fla/chunk.py) |
| Full attention | q/k/v/o projection、Q/K norm、RoPE、KV 写入、FlashAttention、gate | [qwen3_5.py](../../../src/nano_qwen/models/qwen3_5.py)、[attention.py](../../../src/nano_qwen/layers/attention.py) |
| 公共阶段 | embedding、residual/RMSNorm、final norm、LM head | [qwen3_5.py](../../../src/nano_qwen/models/qwen3_5.py) |
| 执行开销 | 输入准备、graph buffer zero/copy、replay、调度、sampling、D2H | [model_runner.py](../../../src/nano_qwen/engine/model_runner.py)、[cuda_graph.py](../../../src/nano_qwen/engine/cuda_graph.py) |

LM head 要单列：当前 compute_logits 使用 F.linear 对全部 prefill hidden rows 投影，随后 ModelRunner 才选每请求的尾部 logits。GDN prefill 实际调用本地 FLA chunk 路径，细分为 l2norm/cumsum、kkt_solve/recompute、state recurrence、output。

### Task 1: 真实 prefill 耗时与差值排行

**Files:** 实施时新增 `benchmarks/bench_prefill_breakdown.py`，复用 `benchmarks/bench_one_batch.py` 的模型/trace 思路及 `benchmarks/engine_bench_utils.py` 的路径计数；输出 `benchmarks/reports/prefill_breakdown/`。

**Interfaces:** JSON 样本记录 scheme、execution_mode、actual_M、bucket_M、layer/module 名、quant_method、M/N/K、两次 samples_ms、GPU 时间、调用次数、稳态检查结果。

- [x] 用当前生产执行模式做两种方案各两次纯 prefill，采样/输出单列；同时记录 engine step 墙钟与 model forward 的 CUDA Event 区间。输出无 profiler 的稳态差值。
- [x] 对相同输入做两种方案各两次独立 profiler 诊断。先在 eager 下保留 module/shape 对应关系，分组统计上表所有阶段；不运行长 decode。
- [x] 如生产使用 piecewise graph，复用真实模式的分段 Event 和路径计数，记录 pre graph、attention core、post graph、buffer copy 与 LM head；不能用 eager 的 launch 开销解释 graph 时间。
- [x] 输出两张表：按总 GPU 耗时排序；按 block-wise 减 per-channel 的 GPU 时间排序。保存调用次数、输入 shape、原始两次值和 trace，而非只给均值。阶段占比以互斥阶段计算，并列出未归属时间/空隙。

**完成标准：** 能指出 prefill 最费时的阶段及解释方案差距的阶段；真实 bucket、初始状态与稳态缓存一致。

### Task 2: 只对热点做真实输入微基准

**Files:** 在新 benchmark 内保存/回放热点输入；参考 `benchmarks/bench_fp8_granularity.py` 的 quant/GEMM/Linear 分拆方法。

- [x] 从 Task 1 按累计额外时间选最多三个热点；若共同的 LM head/GDN 核心是总时间第一名，额外检查该共同热点，不重复扫描所有层。记录每种 shape 在全模型中的调用次数。
- [x] 从同一份参考 forward 保存真实 BF16 输入，给两种量化方法共用；权重来自同一原始 BF16 源。FP8 热点分别测 quant-only、预量化 GEMM-only、完整 Linear；每配置两次，校验显式反量化参考。
- [x] 对 GDN 或 attention 热点固定 batch、cu_seqlens、状态和相同输入，只测该阶段。保存的各方案自身输入另用于核对微基准是否能复现 trace 时间。
- [x] 比较完整 Linear 与真实 trace 的差值是否一致，不将独立 quant/GEMM 均值机械相加。需要观察长度增长时只增加同 batch、每请求 256 token 的一个点（M=2048），不展开 batch×长度矩阵。

**完成标准：** 识别是激活量化慢、GEMM 慢、copy/执行开销慢，还是共同的 prefill 算子占总时间较多。

### Task 3: FP8 热点的最小机制对照（仅在需要归因时执行）

**Files:** benchmark 专用 wrapper/诊断核；生产 kernel 不变。

- [x] 对已确认热点统一使用 BM=32、BN=128、BK=128、warps=4、stages=3，跳过新的 autotune 搜索；检查精度并记录编译后的寄存器、shared memory、spill 信息，无法获得的字段标记缺失。
- [x] 用同一份 per-channel q/weight/scales 比较“最后缩放”与“每个 K 块缩放累加”，每配置两次；这一对照检验逐块累加方式的成本。
- [x] 若差距仍未解释，再用相同逐块累加 kernel 做 A=token/channel、B=group/channel、C=token/block、D=group/block 的 2×2 对照。沿用同一 BF16 源和转换舍入方式，分别观察激活 scale 粒度与权重 scale 粒度的影响；每配置两次。
- [x] 结合 Task 1 的调用次数和差值排序作归因；编译器可能移动 scale load 或改变流水线，因此不同 kernel 的时间差不能单独称为 DRAM 访问耗时。

**完成标准：** 报告明确区分量化开销、逐块缩放/累加开销、scale 粒度与 tile/资源因素；证据不足的原因保持待验证。

## 交付与停止条件

交付一页报告：prefill 时间分布、两方案差值、最多三个热点的拆分结果、两次原始样本、bucket/autotune 参数和必要的机制对照。先执行 Task 1，热点拆分能解释差距时停止扩展实验；LM head 若占时靠前，后续再独立验证只投影请求尾部 rows 的收益与 logits 一致性。

## 执行结果

- ABBA 的四个无 profiler 正式 prefill 完成；batch=8、input=64、M/bucket=512、相同 token 与32个KV blocks。per-channel 59.416 ms（6.21%波动），block-wise 75.476 ms（2.44%）。每方案两次；独立诊断每方案每模式也两次。
- 选出 GDN in_proj_qkv/out_proj、MLP gate_up 三个差值热点，共享真实 BF16 输入与原始 BF16 权重；完成量化/GEMM/Linear及相同tile的缩放位置对照。仅 out_proj 追加四种粒度交叉；共31微基准、62个正式replay，精度relative L2均<0.005。
- 未扩展 M=2048、未替换 LM head rows：当前热点已定位；GDN chunk/conv和LM head的两方案kernel耗时相近。
- 主要差值：graph诊断 GEMM约+11.485 ms，激活量化约+1.514 ms。out_proj缓存BW 16×32约201.056 μs，固定32×128后约90.624 μs，PC固定约87.424 μs，说明调优配置是实际因素。
- 独立只读审查核对全部raw trace、样本数与主要结论。补充GPU union空隙/重叠统计和实验身份复用校验，CPU重解析，没有重测性能；B/C/D逐specialization资源信息明确缺失。
- 工作区完整测试95 passed；原始trace/input/cache及日志保留在 logs/prefill-breakdown-2026-10-14/。未修改生产算子或提交无关改动。

报告：results.md（本地产物 `benchmarks/reports/prefill_breakdown/results.md`）。原脚本已归档清理，当前入口见 [benchmarks/README.md](../../../benchmarks/README.md)。
