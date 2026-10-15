# 已保留的FP8测量数据

| 报告 | 内容 |
|---|---|
| [mtp_verify_reuse](mtp_verify_reuse/results.md) | 独立两token主干复用，norm首差异及修复、全轨迹状态门禁、两种FP8 B1每模式两次真实decode对照 |
| [mtp_cache_only](mtp_cache_only/results.md) | 单层MTP仅补KV、独立Graph、逐位KV/后续预测门禁及两种FP8 B1各两次实际吞吐对照 |
| [mtp_cache_lifecycle](mtp_cache_lifecycle/results.md) | MTP完整缓存生命周期、独立BF16 MTP权重、两种FP8 B1严格输出对照与各两次实际decode吞吐、短trace及off路径回归 |
| [gdn_state_pool](gdn_state_pool/results.md) | GDN recurrent-state原地更新，B8/64两种FP8的旧/新路径各两次计时与复制消失的profile证据 |
| [fp8_phase_profile](fp8_phase_profile/results.md) | GDN原地更新后，batch32/64/96/128/192/256、input64/output128的graph矩阵、阶段分解与容量失败记录 |
| [fp8_phase_profile优化前归档](fp8_phase_profile_before_gdn_pool_20261015/results.md) | 优化前batch1–64的阶段分解、曲线和GDN状态复制热点 |
| [fp8_granularity](fp8_granularity/results.md) | 粒度/shape受控对比、scale体积与逻辑流量 |
| [triton_autotuned](fp8_granularity/triton_autotuned.md) | 两类Triton GEMM调优后的量化/GEMM/Linear计时 |
| [prefill_breakdown](prefill_breakdown/results.md) | 投影热点、kernel累计时间与固定tile对照 |
| [fp8_e2e](fp8_e2e/results.md) | M512端到端与共同轨迹精度回归 |
| [fp8_matrix](fp8_matrix/results.md) | batch4/8、input256/512/1024、output512/1024完整graph矩阵 |
| [fp8_decode_capacity](fp8_decode_capacity/results.md) | 大batch decode容量边界、吞吐、曲线 |

同目录JSON/CSV及已有precision/rounding诊断JSON全部保留。它们记录各自实验时的源码和测量协议；不能把不同版本、shape、prefill/graph设置的数值直接混成一个实验。

专项脚本已经完成用途并清理；当前运行入口和回归测试见 [benchmarks/README.md](../README.md)。原始trace、模型中间结果和源码备份保留在执行机器的 `logs/`，不随Git提交。
