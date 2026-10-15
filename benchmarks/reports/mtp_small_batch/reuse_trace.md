# Reuse短trace：B2开发前基线

2026-10-15，4070 Ti SUPER，B1 FP8 TP1，input64。每种量化分别取Graph与eager一段12轮verify诊断；先预热9个engine step。profiler与formal timing分开。Graph为准确拆分snapshot/MTP内置LM head，空闲engine重新capture并插入external CUDA timing events；这些event及profiler会增加开销。以下不是正常TPOT，也不与正式样本混算。

| FP8 | 执行 | GPU kernel累计 ms | target verify（含snapshot，不含head） | 全部LM head | MTP cache-only | MTP draft（不含head） | 其他 | snapshot event ms/轮 | MTP head event ms/轮 | GPU活动区间空隙 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| channel | graph | 228.80 | 144.67 | 76.12 | 1.43 | 6.40 | 0.18 | 0.330 | 2.142 | 19.3% |
| channel | eager | 257.25 | 172.69 | 75.57 | 1.79 | 6.96 | 0.24 | 0.000 | 0.000 | 82.0% |
| block | graph | 237.36 | 152.20 | 76.40 | 1.52 | 7.03 | 0.20 | 0.343 | 2.147 | 12.3% |
| block | eager | 266.52 | 180.34 | 76.76 | 1.93 | 7.23 | 0.27 | 0.000 | 0.000 | 88.5% |

物理kernel按最内层GPU phase归属且仅计一次，CPU/GPU annotation和嵌套operator不作为kernel重复加总。snapshot包含在target verify中，不应再次加到总量；全部LM head利用eager独立scope内实际kernel的name/grid/block指纹识别，Graph中内置draft head也归入LM head列，并从draft列移除。指纹保存在JSON，不以嵌套operator时间重复相加。Graph事件时间包含被包围kernel间的间隙，与kernel累计不是完全相同口径；eager表内event列为0表示未插入captured events，真实snapshot耗时见JSON operator统计。原始Chrome trace在本地logs/mtp-small-batch/profile-{channel,block}.{graph,eager}.trace.json。

Graph中snapshot约0.3ms/轮，占GPU工作约1%～2%；target矩阵与三次LM head占主要时间。本轮短窗口均接受，rollback未触发，不能由0次推断拒绝场景开销。CPU aten::item 包含等待verify完成的时间，与GPU工作重叠，不可将整段当成可消除收益。活动区间空隙是profiler/event instrumentation下的诊断值，不能直接预测生产加速。

决策：先推进B2密集计算/请求映射，不先改snapshot kernel或完整GPU状态选择。后续在B2真实收益和接受率明确后重新排序热点。
