# FP8 量化粒度精简测试

GPU：NVIDIA GeForce RTX 4070 Ti SUPER。每配置两次正式单次 replay，三次预热另计。

输入是共享的合成 BF16 激活；结果代表 GPU 算子，不等同整模型吞吐或质量。

| 实验 | 投影 | M | 方案 | 量化 μs | GEMM μs | Linear μs | 最大波动 | BF16参考相对L2 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| native | gate_up | 8 | per_channel | 6.688 | 115.584 | 122.208 | 16.3% 不稳定 | 0.0377 |
| native | gate_up | 8 | block_wise | 4.096 | 105.936 | 106.800 | 4.7% | 0.0372 |
| controlled | gate_up | 8 | A | 7.008 | 104.288 | 103.936 | 4.6% | 0.0377 |
| controlled | gate_up | 8 | B | 4.096 | 104.448 | 105.760 | 5.9% 不稳定 | 0.0372 |
| controlled | gate_up | 8 | C | 7.008 | 120.272 | 105.920 | 22.9% 不稳定 | 0.0377 |
| controlled | gate_up | 8 | D | 4.096 | 105.936 | 106.800 | 4.7% | 0.0372 |
| controlled | gate_up | 8 | A_torch | 7.008 | 124.416 | — | 4.6% | 0.0377 |
| native | gate_up | 512 | per_channel | 10.752 | 944.672 | 958.416 | 9.5% 不稳定 | 0.0375 |
| native | gate_up | 512 | block_wise | 16.192 | 654.432 | 665.088 | 2.4% | 0.0370 |
| controlled | gate_up | 512 | A | 14.256 | 643.472 | 655.152 | 1.1% | 0.0374 |
| controlled | gate_up | 512 | B | 16.192 | 650.240 | 663.552 | 2.4% | 0.0370 |
| controlled | gate_up | 512 | C | 14.256 | 644.496 | 657.008 | 1.1% | 0.0375 |
| controlled | gate_up | 512 | D | 16.192 | 654.432 | 665.088 | 2.4% | 0.0370 |
| controlled | gate_up | 512 | A_torch | 14.256 | 956.288 | — | 1.3% | 0.0374 |
| native | down | 8 | per_channel | 14.336 | 65.920 | 86.384 | 128.6% 不稳定 | 0.0380 |
| native | down | 8 | block_wise | 6.304 | 60.832 | 55.760 | 45.7% 不稳定 | 0.0370 |
| controlled | down | 8 | A | 8.160 | 56.784 | 57.344 | 9.2% 不稳定 | 0.0381 |
| controlled | down | 8 | B | 6.304 | 57.200 | 57.856 | 45.7% 不稳定 | 0.0368 |
| controlled | down | 8 | C | 8.160 | 54.176 | 63.488 | 14.8% 不稳定 | 0.0379 |
| controlled | down | 8 | D | 6.304 | 60.832 | 55.760 | 45.7% 不稳定 | 0.0370 |
| controlled | down | 8 | A_torch | 8.160 | 76.208 | — | 25.3% 不稳定 | 0.0381 |
| native | down | 512 | per_channel | 28.672 | 501.024 | 527.872 | 9.1% 不稳定 | 0.0375 |
| native | down | 512 | block_wise | 57.344 | 358.912 | 414.208 | 35.7% 不稳定 | 0.0369 |
| controlled | down | 512 | A | 46.464 | 364.032 | 373.760 | 68.5% 不稳定 | 0.0375 |
| controlled | down | 512 | B | 57.344 | 358.720 | 420.784 | 35.7% 不稳定 | 0.0369 |
| controlled | down | 512 | C | 46.464 | 364.240 | 389.632 | 68.5% 不稳定 | 0.0375 |
| controlled | down | 512 | D | 57.344 | 358.912 | 414.208 | 35.7% 不稳定 | 0.0369 |
| controlled | down | 512 | A_torch | 46.464 | 514.960 | — | 68.5% 不稳定 | 0.0375 |

scale 唯一数据体积与逻辑流量（单位 B，block 减 channel）：

| 投影 | M | 激活 scale | 权重 scale | Linear 逻辑增量 | 增量比例 |
|---|---:|---:|---:|---:|---:|
| gate_up | 8 | 32 → 640 | 73728 → 11520 | -60992 | -0.12804% |
| gate_up | 512 | 2048 → 40960 | 73728 → 11520 | +15616 | +0.02188% |
| down | 8 | 32 → 2304 | 10240 → 5760 | +64 | +0.00027% |
| down | 512 | 2048 → 147456 | 10240 → 5760 | +286336 | +0.63485% |

逻辑流量按输入 BF16、量化数据 FP8、输出 BF16、scale FP32 各读写一次估算，
包含激活 scale 的写入与读取，不包含跨 tile 重复读取或缓存命中。

归因对照（分别按被比较阶段的两次波动判断）：

- gate_up，M=8：相同 q/weight/scale，Torch GEMM 124.416 μs，共同 Triton GEMM 104.288 μs，Torch/Triton=1.193。这一对照支持后端实现对该 shape 的速度影响。
  统一 RN 转换的激活量化：per-token 7.008 μs，per-token-per-128-channel 4.096 μs。
- gate_up，M=512：相同 q/weight/scale，Torch GEMM 956.288 μs，共同 Triton GEMM 643.472 μs，Torch/Triton=1.486。这一对照支持后端实现对该 shape 的速度影响。
  固定 Triton 后，四种粒度 GEMM 为 643.472–654.432 μs，最大跨度 1.70%；两个样本不足以对接近的数值排序。
  统一 RN 转换的激活量化：per-token 14.256 μs，per-token-per-128-channel 16.192 μs。
- down，M=8：相同 q/weight/scale，Torch GEMM 76.208 μs，共同 Triton GEMM 56.784 μs，Torch/Triton=1.342。对照有超过 5% 的样本波动，暂不据此明确归因。
- down，M=512：相同 q/weight/scale，Torch GEMM 514.960 μs，共同 Triton GEMM 364.032 μs，Torch/Triton=1.415。对照有超过 5% 的样本波动，暂不据此明确归因。

这些对照只覆盖两类 MLP 投影；不能直接解释整模型 prefill/decode 的排序。

A/B/C/D 使用相同 Triton tile 与逐 K 块缩放累加；
A=token/channel，B=group/channel，C=token/block，D=group/block。
A_torch 与 A 使用完全相同的预量化数据，仅 GEMM 后端不同。

D 与原生 block、相同的量化阶段复用测量，不另跑两次。

实际 DRAM/L2 流量：unavailable: ERR_NVGPUCTRPERM。JSON 中的 linear_logical_bytes 是逻辑账本，不是实测流量。
