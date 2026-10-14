# Prefill 算子耗时检查结果

GPU：NVIDIA GeForce RTX 4070 Ti SUPER；batch=8，每请求64 token，M=512，BF16，TP=1。
两种 checkpoint 的 168 个运行时 Linear 全部分别为 per-channel/block-wise；使用相同 token、32 个 KV blocks、无 prefix cache/混合 decode。

## 真实稳态 prefill

正式计时关闭 profiler，生产 piecewise graph 模式，按 ABBA 执行。每方案仅两次；模型加载、JIT、调优、capture 和预热另计。

| 方案 | 两次 engine step 墙钟 ms | 平均 ms | 波动 | model CUDA Event 平均 ms |
|---|---|---:|---:|---:|
| per_channel | 61.261, 57.572 | 59.416 | 6.21% | 61.889 |
| block_wise | 74.555, 76.397 | 75.476 | 2.44% | 78.642 |

这两次样本的 block-wise 平均多用 16.060 ms（约 27.0%）。per-channel 波动超过5%，不追加样本，比例暂作观察。
本次 CUDA Event 读数略高于 engine step 墙钟；两种计时源均保留原始值，总时间使用墙钟，不能相减推断 CPU 开销。

## GPU kernel 时间分布

下面来自独立 piecewise graph profiler 诊断，每方案两次。仅累计 kernel/memcpy/memset，排除 gpu_user_annotation 范围；不是无 profiler 总延迟。

| 互斥类别 | per-channel ms | block-wise ms | block 减 channel ms |
|---|---:|---:|---:|
| gemm | 22.263 | 33.748 | +11.485 |
| lm_head | 6.593 | 6.809 | +0.215 |
| attention_core | 4.226 | 4.296 | +0.070 |
| graph_pre | 3.522 | 3.496 | -0.027 |
| gdn_chunk | 2.701 | 2.763 | +0.062 |
| graph_post | 2.575 | 2.637 | +0.062 |
| normalization | 2.323 | 2.371 | +0.049 |
| gdn_conv | 2.256 | 2.299 | +0.043 |
| quant | 0.550 | 2.064 | +1.514 |
| model_overhead | 0.650 | 0.660 | +0.010 |
| unattributed | 0.401 | 0.409 | +0.008 |
| sampling_output | 0.156 | 0.158 | +0.003 |
| full_attention | 0.133 | 0.136 | +0.003 |
| rope | 0.132 | 0.135 | +0.003 |
| final_norm | 0.046 | 0.046 | +0.001 |
| prepare_inputs | 0.009 | 0.008 | -0.001 |
| embedding | 0.006 | 0.006 | -0.000 |

graph_pre/post 类别是图内未进一步归类的 kernel；attention_core 是核心内 padding/pack/gating/state 等未归类部分。GDN chunk、conv、norm 和 LM head 单列。互斥类别不是完整阶段的 inclusive 时间。

诊断 trace 的跨 stream union 与空隙（不是正式墙钟中的 CPU 开销）：

| 模式 | 方案 | busy 平均 ms | span 平均 ms | 两次空隙 ms | overlap 平均 ms |
|---|---|---:|---:|---|---:|
| eager | per_channel | 48.836 | 133.823 | 105.711, 64.264 | 0.000 |
| eager | block_wise | 61.050 | 140.448 | 94.843, 63.953 | 0.000 |
| piecewise_graph | per_channel | 48.534 | 64.535 | 12.021, 19.981 | 0.008 |
| piecewise_graph | block_wise | 62.031 | 89.446 | 35.956, 18.875 | 0.010 |

## 投影差值排行

eager 诊断保留完整 module/shape 对应关系；每列累计该投影的量化/GEMM/Linear 周边 GPU kernel 时间。

| 投影 | 次数/forward | per-channel ms | block-wise ms | 差值 ms |
|---|---:|---:|---:|---:|
| in_proj_qkv | 24 | 3.004 | 6.459 | +3.454 |
| out_proj | 24 | 1.789 | 4.396 | +2.607 |
| gate_up_proj | 32 | 9.698 | 11.594 | +1.896 |
| down_proj | 32 | 5.402 | 6.951 | +1.550 |
| q_proj | 8 | 0.997 | 2.152 | +1.155 |
| o_proj | 8 | 0.593 | 1.473 | +0.879 |
| in_proj_z | 24 | 1.655 | 2.204 | +0.549 |
| k_proj | 8 | 0.172 | 0.343 | +0.171 |
| v_proj | 8 | 0.174 | 0.345 | +0.171 |

## 共享真实输入微基准

三个差值热点共用 per-channel 参考 forward 的真实 BF16 输入，权重均从原始 BF16 checkpoint 转换。每项两次单次 graph replay；eviction 在计时外。单位 μs，星号表示波动超过5%。

| 热点 | PC量化 | BW量化 | PC GEMM | BW GEMM | PC Linear | BW Linear |
|---|---:|---:|---:|---:|---:|---:|
| in_proj_qkv | 11.8* | 17.3 | 146.4 | 301.1 | 154.6 | 315.9 |
| out_proj | 14.8* | 27.5* | 85.9 | 201.1 | 98.3 | 220.7 |
| gate_up_proj | 72.2* | 18.8* | 346.0 | 401.1 | 358.8 | 422.8 |

## 固定 tile 的机制对照

统一 BM=32、BN=128、BK=128、warps=4、stages=3。PC 两种缩放位置复用完全相同 q/weight/scales；BW 使用同一 BF16 源的 block 数据。单位 μs。

| 热点 | PC最后缩放 | PC逐块缩放累加 | BW逐块缩放累加 | 缓存PC tile | 缓存BW tile |
|---|---:|---:|---:|---|---|
| in_proj_qkv | 162.2 | 163.2 | 212.0* | 64×128 | 16×64 |
| out_proj | 87.4 | 85.5 | 90.6 | 32×128 | 16×32 |
| gate_up_proj | 370.1* | 357.7 | 399.9* | 32×128 | 32×64 |

out_proj 的同算法 2×2 对照：A=token/channel，B=group/channel，C=token/block，D=group/block。激活转换均采用直接 RN，权重来自同一 BF16 源。

| A μs | B μs | C μs | D μs |
|---:|---:|---:|---:|
| 91.1 | 123.9* | 94.1 | 96.3 |

B 波动较大，不能据此单独量化激活 scale 的因果贡献。固定三种 kernel 的寄存器/shared/spill 和缓存配置保存在 JSON 中；2×2 的 B/C/D 未逐 specialization 采集资源信息，字段缺失。所有 GEMM/Linear 对显式反量化参考的 relative L2 <0.005。

## 判断

- 两方案差值主要集中在 FP8 GEMM，其次是激活量化；GDN chunk/conv、norm、LM head 的 kernel 时间相近。GDN 输入/输出投影贡献了比 MLP gate_up 更大的累计差值。
- 对 GDN out_proj，缓存 BW 配置为16×32，固定32×128后 GEMM 明显加快、与PC接近。这是明确的 tile/配置因素，不能把原差距全归因于 block 分组或访存。
- 最后缩放与逐块缩放的固定配置对照没有显示同量级成本；分组 scale 的额外访问/计算与编译器调度仍有影响，但当前样本不能分别换算为 DRAM 耗时。
- LM head 当前对512个 token 投影，单个 kernel 约6–7 ms，是共同的大开销；本次未修改投影 rows 或生产算子。
- profiler 的 eager/graph GPU span 包含诊断带来的 host 间隙；分段 Event 是 inclusive 区间，也可能波动，不与子 kernel 时间相加。实际 DRAM/L2 counter 仍受 ERR_NVGPUCTRPERM 限制。

完整两次原始样本、调用次数、shape、cache keys 和 trace 路径见 results.json；原始 trace 与真实输入目录为 `logs/prefill-breakdown-2026-10-14/data`。
