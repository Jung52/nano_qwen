本次测试：batch=8，每请求输入64 token，prefill M=512；输出128 token，含prefill首个输出和127步decode（M=8）。四组各两次正式计时，加载、capture、调优、预热和精度诊断另计。

结果见 [results.md](results.md)，逐次数据见 [results.json](results.json) 和 [results.csv](results.csv)。原始数据保存在 `/home/wei/code/nano_qwen/logs/fp8-e2e-2026-10-14/data`；执行源码、旧identity及完整head校验保存在其上级目录。`provenance.json`记录元数据修正和原始样本哈希。

本轮专项脚本已归档清理，原始测量和统计保留。当前入口见 [benchmarks/README.md](../../README.md)；清理前源码备份位于本地 `logs/fp8-release-cleanup-2026-10-14/before-cleanup.zip`。

内核配置调整前后1024个位置的logits完全一致。本次两样本未证实额外端到端提速，优化组TTFT为68.568/100.864ms，波动38.12%。
