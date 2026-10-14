# FP8 优化与测试总结（2026-10-14）

这一轮把 per-channel 的 GEMM 从 `torch._scaled_mm` 改为 Triton，与 block-wise 共用调优和 CUDA Graph 缓存机制；同时复用 prefill 投影缓冲，降低大 token batch 的 graph 驻留显存。

## 生产实现

- [fp8.py](../src/nano_qwen/quantization/fp8.py)：激活仍为 per-token，权重仍为 per-output-channel。FP8 dot 使用 FP32 累加，完整 K 归约后应用两种 scale，保持 bias、输出 dtype、非连续输入和空输入支持。
- [fp8_block.py](../src/nano_qwen/quantization/fp8_block.py)：激活仍为每 token、每128通道一组，权重仍为128×128块。每个128宽 K 分块 dot 后应用 scale，再累加；直接 FP32→FP8 RNE 的量化转换保留。整合远端后保留M≥128的多group量化与 grouped tile prefill kernel，普通/大M两条GEMM均使用统一调优（按M分流，decode batch≥128也会进入大M路径）；显式config和legacy模式仍可用于对照。
- [triton_gemm.py](../src/nano_qwen/quantization/triton_gemm.py)：六组 tile/warp/stage 配置，按 M/N/K、device 和 tensor dtype 缓存。每候选一次预热、两次单次 graph replay；L2 eviction 位于计时外。已调优 shape 直接使用缓存，冷 capture 使用固定兜底，避免图捕获中启动调优。
- 对 RTX 4070 Ti SUPER、legacy block GEMM `M/N/K=512/2560/4096`、FP8输入/权重、FP32 scale、BF16输出，优先使用 `BM32/BN128/warps4/stages3`。这是 GDN `out_proj` 和 full-attention `o_proj` 的共同 shape，不是所有 shape 的通用配置。grouped prefill kernel独立调优，未把旧kernel的特例强制套用到新kernel。
- [cuda_graph.py](../src/nano_qwen/engine/cuda_graph.py)：顺序执行的 layer 按 projection shape/dtype/device 共享 prefill 输出 buffer，保留独立 KV/GDN 状态；避免每层都常驻一份完整 M 的 QKV/z 输出。实测支持单个完整 prefill bucket 到 M=8192。

远端已有的共享输入量化也被保留：per-channel attention q/k/v和GDN qkv/z在同一次调用中共享quantized input/scale，仍分别执行GEMM。`NANO_QWEN_FP8_SHARE_INPUT=0`可关闭，非FP8/TP/row-parallel仍走原投影路径；不同调用及graph replay会刷新输入。

## 测量与归因

`M` 是 Linear 实际输入行数：decode 为当前活跃 batch，prefill 为本次 forward 的实际 token 总数，例如 `batch8 × input64 = M512`。

以下为整合远端更新前的历史测量，模型/源码/缓存指纹和逐次数据随报告保留。最终提交中的远端共享量化和 prefill kernel 也要通过正确性回归；这些历史数据不视为整合后版本新增的性能测量。

| 实验 | 覆盖和主要结论 | 数据 |
|---|---|---|
| 粒度、GEMM与逻辑流量 | 固定 M/N/K，分拆量化/GEMM/Linear，控制激活/权重 scale 粒度；逻辑字节数不能直接当实际 DRAM 流量 | [granularity](../benchmarks/reports/fp8_granularity/results.md)、[autotuned](../benchmarks/reports/fp8_granularity/triton_autotuned.md) |
| Prefill算子分解 | M512时 block 比 per-channel 的 GEMM 累计多约11.485ms，激活量化多约1.514ms。GDN QKV/out 投影是主要差值热点；固定 tile 后 out_proj GEMM 约201→91μs，配置选择是明确因素 | [breakdown](../benchmarks/reports/prefill_breakdown/results.md) |
| M512端到端与精度 | tile调整前后同量化结果1024位置逐位一致；两次端到端样本未证明该 tile 单项调整带来额外提速，TTFT波动较大 | [e2e](../benchmarks/reports/fp8_e2e/results.md) |
| 完整graph矩阵 | batch4/8 × input256/512/1024 × output512/1024，三方案共72次正式测量。PC总耗时比BF16减少约29.6%–33.6%，BW约27.9%–33.3%；PC的TTFT在12个case均更低，decode中10个case略快 | [matrix](../benchmarks/reports/fp8_matrix/results.md) |
| 大batch decode | context65→192，128步，统一90%显存allocator预算；BF16最大16对齐batch96，两种FP8为256。已测峰值1877/2476/2488 tok/s，PC/BW比BF16峰值约+31.9%/+32.5%；同batch96约+20.4%/+14.2% | [capacity](../benchmarks/reports/fp8_decode_capacity/results.md) |

每个正式实验配置保留两次样本，不因波动补第三次；编译、调优、capture、预热和独立精度诊断另计。BF16 batch96两次吞吐波动5.6%，收益按近似值解读。大batch实验只捕获目标batch，短上下文和该显存预算内的容量不代表所有部署设置。

GDN recurrent+conv状态与KV在三方案中相同，每请求约25.125+8=33.125MiB；增加batch会线性增加这部分显存。权重统计需去除 embedding/head 共享 storage 的重复引用。WSL2显存接近满载时曾观察到异常慢的未计时预热，是否为驱动系统内存回退未直接验证，后续采用统一allocator预算。

scale的逻辑额外访问、逐块缩放以及 tile/调度共同影响耗时。NCU的DRAM/L2计数器受权限限制，没有把理论字节量写成实测带宽。精度结论仅覆盖报告中的共同轨迹，不等同于任务准确率或数据集perplexity。

## 整合和回归

同步远端 `origin/main@e014816` 的共享输入量化、prefill专用kernel、压力基准与LICENSE后处理同名实现冲突。grouped GEMM使用一维grouped grid，普通GEMM使用二维grid；stage参数随候选配置传入。新增回归覆盖共享量化次数、graph输入刷新、packed量化与legacy逐字节一致、两条block路径及全部候选tile、冷热capture。

校验脚本同时保留 `--perf-profile/--perf-cases` 与 `--workloads/--chunked-prefill`，默认使用原legacy六组workload；保留失败结果的原子写盘、容量跳过和fixed-batch的prefill-ready同步边界。chunked窗口明确标注active batch变化，避免与纯decode窗口混用。

## 仓库整理

- `benchmarks/` 保留通用吞吐、profile和模型质量校验入口；本轮5个专项实验脚本及其5个脚本测试已归档清理。
- `tests/` 保留checkpoint、GEMM数学/尾部、全部候选配置、bias/dtype、冷热capture、缓存复用、GDN混合批次、prefill buffer复用及校验CLI的回归测试，脱离临时benchmark helper。
- `benchmarks/reports/` 中Markdown/JSON/CSV和吞吐图保留；原始trace、输入、调优缓存和执行源码位于本地被Git忽略的 `logs/`，不提交大型运行产物。
- 清理前源码备份：`logs/fp8-release-cleanup-2026-10-14/before-cleanup.zip`。`docs/superpowers/plans/` 是已完成实验的历史计划，命令不作为当前运行入口。

收尾验证：整合后90项测试通过；只读审查未发现Critical/Important问题。合并遗留的无用容量helper、重复参数表/解析函数和重复JSON字段已清理；容量回归直接调用真实fixed-batch入口，缺KV错误同时报告required/available。数值报告与图表SHA保持清理前原值。未修改本地无关的 `linear.py` 注释及 `craft.py`。
