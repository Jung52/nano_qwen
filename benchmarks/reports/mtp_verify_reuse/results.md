# 两token主干复用独立实验（2026-10-15）

生产已接入cache-only（`3f8a3ba`）；本实验不修改默认serial B1 verify。两行共享dense投影、输出投影、MLP的矩阵调用；GDN conv/recurrent仍顺序执行、第一步保存snapshot；full attention仍逐行单query，LM head逐行。所有input/post/final norm也保留M1归约路径。不是合并两个独立请求。

## 为什么不能只测第一个窗口

初版两种FP8的auto/pinned单窗口均通过522项逐位比较，独立Graph首窗口也通过；小模型FP8的强制决策链路通过。但是per-channel完整128-token gate发生分歧。沿serial轨迹定位到位置69/70、第26层post-attention norm：1个元素不同，最大绝对差0.015625；该层此前的dense/core输出完全一致。M1/M2的norm归约形状也会改变舍入，不能只固定GEMM tile。另一个BF16小模型最早在第1层gate_up出现差异，这不支持把FP8实验泛化到BF16。

修正版固定norm为单行，并将M2 FP8 GEMM的配置固定为同shape M1已调优的配置；继续按两行执行矩阵计算。初版失败计时不报告加速、不使用drift豁免。当前逐层诊断也拒绝非有限值，即使两边相同Inf也不能过门禁。

## 修正后的正确性范围

- 真实4B：沿完整128-token serial参考轨迹，per-channel 66个verify窗口、block-wise 70个窗口，每窗522项全部逐位相同：各层dense/core/norm/MLP/hidden/residual、最终hidden/logits、GDN最终状态、首行snapshot、有效target KV。该全轨迹逐层检查的候选前向为eager。
- 小型真实FP8混合模型：eager/Graph的逐层门禁，及强制接受、拒绝、连续接受后拒绝、接受/拒绝交替；129-token prompt按128+1分块，按相同已消费前缀比较GDN、target KV和MTP KV，全部逐位一致。明确检查norm仍按M1调用。
- 完整回归286pass；本次新增11项（含CPU非有限值门禁）。真实两种FP8另有完整128-token eager completion gate，并且下面每一个正式Graph样本都要求128个completion token严格等于关闭MTP；未对4B候选Graph轨迹的每个窗口重复做全部逐层状态检查。

## 正式decode吞吐

B1/TP1、输入64/输出128、greedy、ignore EOS，独立BF16 MTP来源相同；prefill eager、decode Graph，profiler关闭。正式计时使用`bench_qwen3_5_mtp_production.run_mode`，只计首token之后实际提交的127个decode token。每个量化/模式2次；serial/reused在同一个MTP engine按ABBA顺序执行，切换graph、预热和prefill均在计时外；请求真实结束、slot状态清理，stats逐次清零。off另用一个engine两次。

| FP8 | MTP关闭 tok/s | serial + cache-only tok/s | 复用 + cache-only tok/s | 相对serial | 相对MTP关闭 |
|---|---:|---:|---:|---:|---:|
| per_channel | 84.46 | 69.17 | 91.86 | +32.8% | 1.088× |
| block_wise | 79.96 | 62.98 | 83.82 | +33.1% | 1.048× |

数值为两次tok/s算术均值；原值见[CSV](timing_samples.csv)，模型/源码指纹、诊断摘要及接受统计见[JSON](results.json)。只有两个样本、一个prompt和一张4070Ti SUPER，结论局限于这组B1工作负载；不把历史83tok/s当同次对照，不推断其他batch/硬件或任务质量。cache-only收益见[独立报告](../mtp_cache_only/results.md)，不能将两个实验的百分比直接相加。

## 复现与保留边界

```bash
PYTHONPATH=src python benchmarks/bench_mtp_verify_reuse.py --model FP8_DIR --mtp-model ORIGINAL_DIR --output logs/reuse-check.json
PYTHONPATH=src python benchmarks/bench_mtp_verify_reuse.py --model FP8_DIR --mtp-model ORIGINAL_DIR --timing --output logs/reuse-timing.json
```

第一条检查一个verify窗口的auto/pinned eager及pinned Graph，不等同于完整轨迹；第二条先做128-token eager严格gate，再各模式两次Graph正式计时，失败立即停止。窗口逐层轨迹诊断编排和失败原值保留在本机`logs/mtp-cache-only/`；本报告保留每窗结果和首差异，不提交大量中间张量。

投影注入使用局部patch，仅用于单线程实验，退出即恢复；不作为生产并发接口。生产仍使用serial verify；snapshot/rollback的实现和`.item()`同步未调整，B4/8、TP>1和5090均未测试。
