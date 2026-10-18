# 执行、提交和调度边界检查

2026-10-18，RTX 4070 Ti SUPER，4B per-channel FP8，TP1。

```bash
/home/wei/miniconda3/envs/code/bin/python benchmarks/bench_runtime_contracts.py \
  --model /home/wei/code/models/4b-channel-fp8 \
  --mtp-model /home/wei/code/models/4b \
  --output benchmarks/reports/runtime_contracts/results.json
```

正式性能：B1/B2 × input64/1024 × output128，各配置恰好两次。
同 resident engine 按 off1→reuse1→reuse2→off2 运行，保留 MTP 权重、缓存和快照；独立 off engine 不加载这些资源，并启用默认 prefill piecewise Graph。两者均使用 decode Graph。每种 input/mode 先做 output4 的编译预热。正式计时关闭 profiler、状态拷贝和诊断 hooks，包含实际提交 token 的 CPU/GPU 完整请求执行，不含加载模型、捕获 Graph 或分词。

另有独立的完整128输出数值诊断：比较 target 全行词表投影与只投影采样行，检查 first draft、tail top1、严格 completion、完整 target hidden，以及相同有效前缀下的 GDN / target KV / MTP KV。诊断结果不计入两次正式性能样本。

CPU 调度复现保留在 `scheduler_repro.json`。动态 B64/63/48/47/32/31/1 的 Graph 分派由 `tests/test_graph_dispatch_stats.py` 验证；其中 B63 安全回退 eager。该项是分派正确性检查，没有声称 B63 的 GPU 性能。

历史 rollback 对比脚本中的旧 sampler 已增加 benchmark 专用元数据适配，仅用于状态参考。需要复现旧的性能归因时，应检出对应报告记录的提交；适配器分配开销不能计入新的 rollback-only 优化收益。
