# 已保留的FP8测量数据

| 报告 | 内容 |
|---|---|
| [fp8_granularity](fp8_granularity/results.md) | 粒度/shape受控对比、scale体积与逻辑流量 |
| [triton_autotuned](fp8_granularity/triton_autotuned.md) | 两类Triton GEMM调优后的量化/GEMM/Linear计时 |
| [prefill_breakdown](prefill_breakdown/results.md) | 投影热点、kernel累计时间与固定tile对照 |
| [fp8_e2e](fp8_e2e/results.md) | M512端到端与共同轨迹精度回归 |
| [fp8_matrix](fp8_matrix/results.md) | batch4/8、input256/512/1024、output512/1024完整graph矩阵 |
| [fp8_decode_capacity](fp8_decode_capacity/results.md) | 大batch decode容量边界、吞吐、曲线 |

同目录JSON/CSV及已有precision/rounding诊断JSON全部保留。它们记录各自实验时的源码和测量协议；不能把不同版本、shape、prefill/graph设置的数值直接混成一个实验。

专项脚本已经完成用途并清理；当前运行入口和回归测试见 [benchmarks/README.md](../README.md)。原始trace、模型中间结果和源码备份保留在执行机器的 `logs/`，不随Git提交。
