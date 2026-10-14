# Benchmark 与测试入口

本轮优化内容和历史测量见 [FP8总结](../docs/fp8-performance-2026-10-14.md)。数据索引见 [reports/README.md](reports/README.md)。

## 保留的入口

| 文件 | 用途 |
|---|---|
| `bench_engine_qwen35.py`、`engine_bench_utils.py` | 通用engine吞吐、请求延迟、调度与输出链路诊断 |
| `bench_one_batch.py` | 固定batch的真实engine profile，导出Perfetto trace |
| `bench_piecewise_prefill_paths.py` | piecewise prefill路径对比 |
| `bench_gdn_flashqla.py` | GDN kernel对比 |
| `validate_engine_qwen35.py` | engine运行路径与正确性校验 |
| `validate_fp8_model.py` | 完整模型smoke、共同轨迹质量与可配置性能工作负载 |
| `bench_fp8_pressure.py` | 远端保留的FP8压力与profile基准 |
| `bench_quant_max_batch.py` | 远端保留的BF16/FP8最大batch搜索与对比 |
| `bench_bf16_vs_fp8.sh`、`bench_stress.sh`、`stress_bench.py` | 远端压力测试入口及包装脚本 |

模型相关命令需在已有CUDA、PyTorch、Triton和模型依赖的环境运行；各入口的 `--help` 给出参数。原始日志和trace写入Git忽略的 `logs/`。

生产回归测试已移到 `tests/`：

```bash
PYTHONPATH=src python -m pytest -q tests
```

本轮 `bench_fp8_granularity`、`bench_prefill_breakdown`、`bench_fp8_e2e`、`bench_fp8_matrix`、`bench_fp8_decode_capacity` 专项脚本和脚本测试已经归档清理。既有报告保留两次测量的原值和统计；清理前源码在本地 `logs/fp8-release-cleanup-2026-10-14/before-cleanup.zip`。历史计划中的旧命令用于说明测量过程。

校验CLI支持 `--perf-profile batch/context/legacy` 和 `--perf-cases PROMPT:BATCH`，`--workloads`是显式cases的兼容别名；`--chunked-prefill`采用生产调度，decode统计窗口可能包含后来请求的prefill。旧共享输入检查笔记见 [归档](../docs/archived/fp8-shared-input-notes.md)。
