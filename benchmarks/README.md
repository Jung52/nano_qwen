# Benchmark 与测试入口

本轮优化内容和历史测量见 [FP8总结](../docs/fp8-performance-2026-10-14.md)。数据索引见 [reports/README.md](reports/README.md)。

## 保留的入口

| 文件 | 用途 |
|---|---|
| `bench_engine_qwen35.py`、`engine_bench_utils.py` | 通用engine吞吐、请求延迟、调度与输出链路诊断 |
| `bench_one_batch.py` | 固定batch的正式计时与独立profile，按prefill/decode聚合GPU事件并导出Perfetto trace |
| `bench_qwen3_5_mtp_production.py` | B1 greedy MTP生产对照；独立MTP权重、实际提交token计数及严格completion比较 |
| `bench_mtp_verify_reuse.py` | 正式serial/reuse路径的逐层/状态严格诊断；`--timing`每模式两次实际decode计时 |
| `bench_mtp_small_batch.py` | B1长度矩阵及中文/代码，off/reuse每配置两次，保留全部completion与严格门禁 |
| `bench_mtp_batch2.py` | B2异长eager off/serial/reuse门禁；三类Graph工作负载off/reuse各两次实际提交token计时 |
| `bench_mtp_batch_gates.py` | 真实4B两种FP8的B4/B8异长eager/Graph off/serial/reuse严格completion门禁；仅诊断，不作正式计时 |
| `bench_mtp_interleaved.py` | 同resident engine交错off/reuse；可选旧完整采样/greedy fast对照，每配置2次；实际提交token与严格completion门禁 |
| `bench_mtp_rollback.py` | 同engine、固定调优配置交错off/旧逐请求rollback/新合批rollback，每配置2次；独立host诊断与严格completion门禁 |
| `bench_mtp_prefill_tail.py` | 回滚合批作为共有基线；B2/B4×输入64/256×输出128/512，off/旧MTP prefill head/tail head各2次；独立阶段、首draft及最终状态诊断 |
| `profile_mtp_reuse.py` | B2/B4 off/reuse：正式两次计时、独立CUDA event阶段、无event节点短trace及轻量host诊断；记录动态活跃batch和实际提交tokens |
| `bench_piecewise_prefill_paths.py` | piecewise prefill路径对比 |
| `bench_gdn_flashqla.py` | GDN kernel对比 |
| `validate_engine_qwen35.py` | engine运行路径与正确性校验 |
| `validate_fp8_model.py` | 完整模型smoke、共同轨迹质量与可配置性能工作负载 |
| `bench_fp8_pressure.py` | 远端保留的FP8压力与profile基准 |
| `bench_quant_max_batch.py` | 远端保留的BF16/FP8最大batch搜索与对比 |
| `bench_bf16_vs_fp8.sh`、`bench_stress.sh`、`stress_bench.py` | 远端压力测试入口及包装脚本 |

模型相关命令需在已有CUDA、PyTorch、Triton和模型依赖的环境运行；各入口的 `--help` 给出参数。原始日志和trace写入Git忽略的 `logs/`。

最新两种FP8的阶段分解与batch–TPOT–吞吐曲线见 [fp8_phase_profile](reports/fp8_phase_profile/results.md)。`bench_one_batch.py --timing-rounds 2 --profile-rounds 0`只做两次正式计时；`--timing-rounds 0 --profile-rounds 2 --profile-decode-steps 8`单独做短窗口诊断。加`--workload-memory`可固定KV容量、限制allocator显存预算，并仅捕获精确prefill bucket。

GDN大batch的recurrent-state复制已改为按slot原地更新；旧/新路径B8/64的对照与原始数据见 [gdn_state_pool](reports/gdn_state_pool/results.md)。

MTP缓存索引、接受后补写、chunked prefill及抢占重放修复见 [mtp_cache_lifecycle](reports/mtp_cache_lifecycle/results.md)。该历史报告的serial路径没有吞吐收益；报告保留各两次正式测量及独立短trace，不能用接受率或projected吞吐替代生产结果。

接受draft后的第一行MTP已增加cache-only前向与独立Graph，见 [mtp_cache_only](reports/mtp_cache_only/results.md)。两行主干复用已通过 `mtp_verify_mode="reuse"` 正式接入，默认仍为serial；保留顺序GDN、单query attention、单行norm和LM head。配置和命令见 [正式接口说明](../docs/mtp-verify-reuse.md)，实测见 [生产路径报告](reports/mtp_reuse_production/results.md)。当前支持FP8、B1–8/TP1，每个活跃/接受子集大小都有精确Graph；不提供drift放行参数。

当前阶段报告覆盖batch32/64/96/128/192/256、输入64/输出128，显存不足的配置保留实际完成次数与失败状态。优化前的batch1–64报告见 [归档](reports/fp8_phase_profile_before_gdn_pool_20261015/results.md)。

生产回归测试已移到 `tests/`：

```bash
PYTHONPATH=src python -m pytest -q tests
```

本轮 `bench_fp8_granularity`、`bench_prefill_breakdown`、`bench_fp8_e2e`、`bench_fp8_matrix`、`bench_fp8_decode_capacity` 专项脚本和脚本测试已经归档清理。既有报告保留两次测量的原值和统计；清理前源码在本地 `logs/fp8-release-cleanup-2026-10-14/before-cleanup.zip`。历史计划中的旧命令用于说明测量过程。

校验CLI支持 `--perf-profile batch/context/legacy` 和 `--perf-cases PROMPT:BATCH`，`--workloads`是显式cases的兼容别名；`--chunked-prefill`采用生产调度，decode统计窗口可能包含后来请求的prefill。旧共享输入检查笔记见 [归档](../docs/archived/fp8-shared-input-notes.md)。

B2/B4最新成本报告见[mtp_batch_stages](reports/mtp_batch_stages/results.md)：CPU逐请求rollback随batch放大；下一项只选择批量按slot恢复。profile默认仅两次正式计时，诊断pass独立，不将event/CPU/GPU重叠时间相加。main独立greedy已在`e936f58`接入，数据见[main报告](https://github.com/Jung52/nano_qwen/blob/e936f58/benchmarks/reports/greedy_main/results.md)。
