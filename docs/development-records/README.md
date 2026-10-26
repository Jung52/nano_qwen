# 开发日志的历史证据

供根目录[开发日志](../../DEVELOPMENT_LOG.md)引用。这批小文件原先只在本机被Git忽略的`logs/`中，归档以避免开发日志在其他checkout里只剩无法访问的本机路径。10月14–15日的正式报告曾位于`benchmarks/reports/`；该目录自2026-10-26起不再随Git分发（本地保留，公开结论汇总见[experiments-summary.md](../experiments-summary.md)），不在这里重复复制。

| 文件 | 原始来源 | 口径 |
|---|---|---|
| [engine-validation-2026-09-03.md](engine-validation-2026-09-03.md) | `logs/engine_validation_report_20260903.md` | `a19088d`的历史engine报告；正式性能主要为5次中位数，含失败与当时的诊断解释 |
| `fp8-format-comparison-2026-10-09.csv` / `fp8-format-comparison-2026-10-09.summary.json` | `logs/fp8_format_comparison_20261009/` | HF block与原生per-channel，whole-batch prefill，eager/decode Graph，3次中位数 |
| `fp8-e2e-three-models-2026-10-09.csv` / `fp8-e2e-three-models-2026-10-09.summary.json` | `logs/graph_e2e_three_models_20261009/` | BF16/HF block/per-channel，chunked eager prefill、decode Graph，每项2次端到端计时 |
| `fp8-hf-vs-converted-block-2026-10-09.csv` / `fp8-hf-vs-converted-block-2026-10-09.summary.json` | `logs/graph_hf_vs_block_fp8_20261009_211456/` | HF与自转block同后端、GPU scale均FP32，每项2次端到端计时 |

`manifest.json`记录来源路径、字节数、原始SHA256与归档SHA256。归档只将CRLF统一为LF，没有改动数据内容。这批 CSV/JSON 已随 commit `031e88c` 从仓库与磁盘删除，可从 `git show 031e88c^:docs/development-records/<文件>` 恢复；原始本机日志仍在 `logs/` 对应目录。

9月3日报告按原文保留，其中已修复的问题、当时的数值归因、建议和预期收益不代表当前状态，也不代表后续已经实现或通过。10月9日的eager/decode Graph、whole-batch/chunked、median/mean口径不可混用。详见开发日志中的重新核对与限制。
