# Greedy fast path：B2交错复测（2026-10-15）

CPU侧根据Sequence温度传入all_greedy提示。全greedy直接argmax，不再计算softmax、随机数及随机采样结果；MTP verify保持直接argmax。非greedy/混合批次沿用7dc09b9算法，greedy并列取最小argmax；fast不消耗RNG，因此后续随机请求的全局RNG偏移会与旧实现不同，给定相同RNG状态的混合/随机批次采样语义不变。没有从GPU温度张量取标量或新增同步。

4070 Ti SUPER，TP1，两种FP8，B2，输出128，prefill eager/decode Graph，profiler关闭。每个case按reference-off、reference-reuse、fast-off、fast-reuse顺序执行，再重复一次，每配置正式2次；各组合预先4-token短预热。reference仅在benchmark中忽略CPU fast hint，调用未改的完整采样算法，不修改dense路径或autotuner缓存。所有组合共享同一resident engine、权重、调优与graph；off暂停MTP执行，MTP权重/graph仍驻留，所以这不是卸载MTP后的显存容量对照。

每样本计2×127=254个实际提交decode tokens，所有32个正式样本完整completion相同。same64为两个64-token合成输入；chinese-code为45-token中文与43-token代码。TPOT为两个请求各自decode时间/127的均值。完整执行顺序、completion、每请求延迟、显存和源码指纹见[JSON](results.json)。

| FP8 | 输入 | reference off | reference reuse | reference MTP收益 | fast off | fast reuse | fast MTP收益 | off提速 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| channel | same64 | 110.54 | 157.34 | +42.3% | 152.28 | 157.69 | +3.6% | +37.8% |
| channel | chinese-code | 112.72 | 156.95 | +39.2% | 153.98 | 156.47 | +1.6% | +36.6% |
| block | same64 | 110.22 | 158.54 | +43.8% | 152.87 | 155.09 | +1.5% | +38.7% |
| block | chinese-code | 111.49 | 147.49 | +32.3% | 152.11 | 152.60 | +0.3% | +36.4% |

全greedy减少的工作主要改善off，reuse大部分verify轮次早已不执行通用采样器，因此MTP相对收益收缩。两个样本不足以给小幅正负差确定排序；不能将历史未交错、不同engine/调优的绝对值直接相减归因。B4/8与收益分界应使用fast后的off基线。

验证：原451加2个fast-path/CPU路由回归，完整453通过；混合温度原回归继续通过。实际4B交错样本严格completion一致，未使用数值漂移放行。
