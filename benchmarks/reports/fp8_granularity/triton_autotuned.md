# FP8 量化粒度精简测试

GPU：NVIDIA GeForce RTX 4070 Ti SUPER。每配置两次正式单次 replay，三次预热另计。

输入是共享的合成 BF16 激活；结果代表 GPU 算子，不等同整模型吞吐或质量。

| 实验 | 投影 | M | 方案 | 量化 μs | GEMM μs | Linear μs | 最大波动 | BF16参考相对L2 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| native | gate_up | 8 | per_channel | 5.632 | 109.056 | 111.616 | 54.5% 不稳定 | 0.0377 |
| native | gate_up | 8 | block_wise | 5.920 | 101.232 | 103.904 | 42.2% 不稳定 | 0.0372 |
| native | gate_up | 8 | torch_reference | 5.632 | 123.168 | 122.368 | 54.5% 不稳定 | 0.0377 |
| native | gate_up | 512 | per_channel | 10.752 | 310.784 | 321.120 | 9.5% 不稳定 | 0.0375 |
| native | gate_up | 512 | block_wise | 18.800 | 338.784 | 350.208 | 14.8% 不稳定 | 0.0370 |
| native | gate_up | 512 | torch_reference | 10.752 | 941.248 | 955.904 | 9.5% 不稳定 | 0.0375 |
| native | down | 8 | per_channel | 5.424 | 60.416 | 63.488 | 11.2% 不稳定 | 0.0381 |
| native | down | 8 | block_wise | 5.584 | 62.464 | 57.856 | 51.0% 不稳定 | 0.0370 |
| native | down | 8 | torch_reference | 5.424 | 65.280 | 125.440 | 54.7% 不稳定 | 0.0380 |
| native | down | 512 | per_channel | 26.112 | 184.640 | 209.920 | 3.9% | 0.0376 |
| native | down | 512 | block_wise | 59.392 | 211.072 | 235.008 | 34.5% 不稳定 | 0.0369 |
| native | down | 512 | torch_reference | 26.112 | 492.480 | 507.904 | 3.9% | 0.0375 |

scale 唯一数据体积与逻辑流量（单位 B，block 减 channel）：

| 投影 | M | 激活 scale | 权重 scale | Linear 逻辑增量 | 增量比例 |
|---|---:|---:|---:|---:|---:|
| gate_up | 8 | 32 → 640 | 73728 → 11520 | -60992 | -0.12804% |
| gate_up | 512 | 2048 → 40960 | 73728 → 11520 | +15616 | +0.02188% |
| down | 8 | 32 → 2304 | 10240 → 5760 | +64 | +0.00027% |
| down | 512 | 2048 → 147456 | 10240 → 5760 | +286336 | +0.63485% |

逻辑流量按输入 BF16、量化数据 FP8、输出 BF16、scale FP32 各读写一次估算，
包含激活 scale 的写入与读取，不包含跨 tile 重复读取或缓存命中。

生产 GEMM 选中的 autotune 参数（BK=128）：

| 投影 | M | 方案 | BM | BN | warps | stages |
|---|---:|---|---:|---:|---:|---:|
| gate_up | 8 | per_channel | 16 | 32 | 4 | 2 |
| gate_up | 8 | block_wise | 16 | 32 | 4 | 2 |
| gate_up | 512 | per_channel | 64 | 128 | 8 | 3 |
| gate_up | 512 | block_wise | 64 | 128 | 8 | 3 |
| down | 8 | per_channel | 32 | 64 | 4 | 3 |
| down | 8 | block_wise | 32 | 64 | 4 | 3 |
| down | 512 | per_channel | 32 | 128 | 4 | 3 |
| down | 512 | block_wise | 32 | 128 | 4 | 3 |

每个 shape、每种生产 GEMM 最多搜索六组配置，每个有效候选一次预热、
两次单次 replay。搜索与正式测量分开，重复调用及 warm capture 使用缓存。

归因对照（分别按被比较阶段的两次波动判断）：

- gate_up，M=8：相同 per-channel 量化数据，旧 Torch GEMM 123.168 μs，新 Triton GEMM 109.056 μs，旧/新=1.129。
- gate_up，M=512：相同 per-channel 量化数据，旧 Torch GEMM 941.248 μs，新 Triton GEMM 310.784 μs，旧/新=3.029。
- down，M=8：相同 per-channel 量化数据，旧 Torch GEMM 65.280 μs，新 Triton GEMM 60.416 μs，旧/新=1.081。样本波动超过 5%，此比值暂作观察。
- down，M=512：相同 per-channel 量化数据，旧 Torch GEMM 492.480 μs，新 Triton GEMM 184.640 μs，旧/新=2.667。

这些对照只覆盖两类 MLP 投影；不能直接解释整模型 prefill/decode 的排序。

A/B/C/D 使用相同 Triton tile 与逐 K 块缩放累加；
A=token/channel，B=group/channel，C=token/block，D=group/block。
A_torch 与 A 使用完全相同的预量化数据，仅 GEMM 后端不同。

相同的量化阶段复用测量；固定 D 与 autotuned 原生 block 的 GEMM/Linear 分别计时。

实际 DRAM/L2 流量：unavailable: ERR_NVGPUCTRPERM。JSON 中的 linear_logical_bytes 是逻辑账本，不是实测流量。
