# MTP reuse正式接入（2026-10-15）

本轮完成正式API和执行开关，并复测原B1工作负载。`enable_mtp=True, mtp_verify_mode="reuse"`启用复用，默认仍为`serial`；reuse限制FP8/TP1/B1。配置与命令见[接口说明](../../../docs/mtp-verify-reuse.md)。长度/内容矩阵、新热点trace、B2/4/8和5090留到下一阶段。

GDN正式拆分dense、单步decode core和out projection；普通decode与reuse共用状态核心。Attention也暴露out projection之前的core。两行dense/输出投影/FP8 MLP合并，norm、GDN状态更新和单query attention按行，保留首步snapshot和逐行LM head。混合checkpoint中未量化的linear按M1执行。接受draft后的cache-only路径不变。

Context中的局部策略读取本卡、同N/K/dtype的M1 autotune选择，直接执行M2，不写入M2缓存。冷eager先正常调优M1；冷Graph不调优且明确失败，正常engine先预热普通B1再capture。没有投影/方法patch，也没有把4070的配置硬塞给5090。模式变更后不重新capture会报错，避免旧图静默覆盖新配置。

## 正确性与回归

- 完整383项通过。两种FP8 × eager/Graph均覆盖正式M2分派、逐行norm、逐层hidden/residual、logits、GDN最终/首步snapshot和有效target KV。
- 强制接受、拒绝、连续/交替，以及127/128/129、255/256/257、511/512/513块边界，128-token分块；EOS出现在accepted draft、bonus或拒绝分支，剩余1/2/3输出预算；重复请求复用slot。在相同已消费前缀，GDN、target KV和MTP KV与serial严格一致。较早的serial生命周期/抢占回归也继续运行，没有移除旧门禁。
- 真实4B：两种FP8的首窗口eager/Graph各522项严格通过；完整128-token serial参考轨迹中PC66、BW70个verify窗口，每窗522项逐位一致。此全轨迹逐层候选为eager；真实Graph另以正式样本的完整completion门禁验证，不扩大为每个Graph窗口已逐层检查。
- 两种FP8完整128-token eager gate通过；下列12个正式Graph样本均要求全部128个completion token与off一致。非有限值直接失败，无drift放行。
- 9项CPU回归约束借用engine的空闲、MTP来源、长度、异常清理、ownership与每次统计。清理只移除本次请求，成功/失败均不关闭借用engine；累计stats保留，报告返回本次差值。

## 正式decode性能

B1、输入64/输出128、TP1、greedy、ignore EOS；两种FP8和独立BF16 MTP路径见JSON。prefill eager、decode Graph、profiler关闭，排除首token，计127个真实提交decode token。每个模式2次；off一个engine，serial/reuse共享另一个engine按ABBA顺序执行。模式切换/capture/初始化/prefill均在decode计时外，真实请求结束后清理slot。使用`bench_qwen3_5_mtp_production.run_mode`，没有projected吞吐。

| FP8 | off tok/s | serial tok/s | reuse tok/s | reuse相对off | reuse相对serial |
|---|---:|---:|---:|---:|---:|
| per_channel | 83.39 | 69.84 | 94.06 | +12.8% | +34.7% |
| block_wise | 83.26 | 65.92 | 88.31 | +6.1% | +34.0% |

各指标为两次算术均值；TPOT逐次由decode时间/127计算，再取均值。

| FP8 | 模式 | TPOT ms | TTFT ms | 请求耗时 s | 接受率 | allocator峰值 MiB |
|---|---|---:|---:|---:|---:|---:|
| per_channel | off | 11.992 | 63.19 | 1.5862 | — | 12380.8 |
| per_channel | serial | 14.322 | 111.41 | 1.9303 | 90.91% | 12369.3 |
| per_channel | reuse | 10.632 | 64.76 | 1.4150 | 90.91% | 12369.3 |
| block_wise | off | 12.011 | 65.47 | 1.5908 | — | 12385.3 |
| block_wise | serial | 15.173 | 85.92 | 2.0129 | 81.43% | 12373.8 |
| block_wise | reuse | 11.323 | 70.12 | 1.5082 | 81.43% | 12373.8 |

PC每次60/66接受，BW每次57/70，serial/reuse相同。TTFT从入队到首token所在step返回，请求耗时排除engine初始化并包含TTFT与decode；TTFT只有两个样本，存在首请求、capture后状态和缓存影响，不据此归因prefill加速。显存是包含预分配KV、权重与Graph的allocator数据；预算相同的engine会调整KV容量，不能把总量差当MTP权重的净开销。原始allocated/reserved/peak均保留。

本轮PC复现了原工作负载上的收益，不能由两个样本宣布稳定10%，也不能把本轮与[旧实验](../mtp_verify_reuse/results.md)的绝对值之差归为API本身的额外优化。未独立计时每轮GPU verify；decode总耗时/verify次数包含MTP和调度等，不能冒充verify算子耗时。下一步小矩阵和新的reuse短trace再判断收益区间及热点。

源码指纹、每窗诊断、12个正式样本及汇总见[JSON](results.json)，逐次数据见[CSV](timing_samples.csv)。RED/GREEN、完整测试和诊断日志保留在本机`logs/mtp-reuse-production/`；此报告不覆盖历史实验结果。
