已完成12种batch/input/output组合 × 3种方案 × 2次正式测量，共72次。Prefill采用对应完整M的piecewise CUDA Graph，decode也使用CUDA Graph。

[results.md](results.md)包含总耗时、TTFT、decode吞吐和相对BF16收益；[results.csv](results.csv)及[results.json](results.json)保存两次原值、波动和峰值显存。原始数据、调优缓存、执行源码及恢复加固记录位于 `/home/wei/code/nano_qwen/logs/fp8-matrix-2026-10-14`。

本轮专项脚本已归档清理，原始测量和统计保留。当前入口见 [benchmarks/README.md](../../README.md)；清理前源码备份位于本地 `logs/fp8-release-cleanup-2026-10-14/before-cleanup.zip`。

本轮总耗时相对BF16：per-channel减少29.63%–33.59%，block-wise减少27.85%–33.27%。Block-wise的平均TTFT比per-channel高4.38%–15.43%；decode吞吐差异较小。只有batch4/input256/output512的block-wise TTFT两次波动超过5%（5.19%），未追加正式实验。

为支持M=8192，生产图缓冲按匹配的shape/dtype/device在顺序执行的层之间共享。BF16、block-wise各1024位置的额外回归与此前原图hidden/logits逐位一致；此次未扩展量化质量数据集评估。
