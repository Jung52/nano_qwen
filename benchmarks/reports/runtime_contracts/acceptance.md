# 验收与本轮取舍

当前改动均保留本地，未提交、未推送。主要工作位于 `feat/qwen35-mtp2`，起点 `4eb43c1f7c8bfd77a2095a563bee501be1d0c550`。`main` 的独立工作树 `/home/wei/code/nano_qwen-main-greedy` 只修改 Sequence 序列化、Config 安全边界及对应测试；起点 `e936f58a09eec1f8cc531b0f25090870e473e3f1`。

## 验证

- main 完整测试107通过。
- mtp2 最终完整测试：707通过、32跳过（150.81s），额外跳过为B4容量下不适用的更大接受子集大小。
- 独立只读审查：没有 Critical / Important / Minor 问题，另独立执行27项CPU回归并通过。
- 正式性能24个样本，12个配置各恰好两次；各请求128输出完全一致。每个样本实际 decode 提交数及逐步提交数之和均为127×B。
- 4B target 全行 head 与采样行 head 的4组独立诊断：完整 hidden、GDN、有效 target/MTP KV 逐字节一致；tail 有限、top1 和 first draft 一致。BF16 tail 最大差0.0625，未使用 numerical drift 豁免。
- 真实GPU专项覆盖两种FP8、eager/Graph、混合/交替接受拒绝、EOS、不同结束时间、slot复用、chunked prefill、B1..8及所有接受子集大小。新增加 ordinary decode 与部分 prefill 混合的全行/采样行比较。
- CPU复现原调度16批/2048 prefill token等待，新调度首批进入decode；另验证持续到达和实际engine队列控制流。
- 动态B64/63/48/47/32/31/1的分派检查：4次Graph、3次eager，B63没有padding到B64。该项没有做B63 GPU计时。
- 原始报告记录81份生产/benchmark源文件哈希；计时后全部一致。resident与独立off共有的decode调优配置，B1共6项、B2共12项，均一致。
- scoped Ruff、compileall、两工作树 diff --check通过。用户原有README、linear注释、DEVELOPMENT_LOG、craft及本地开发记录保留；前四者哈希与开始时一致。

## 做出的取舍

1. 完整保存 worker 元数据，同时禁用TP>1。CPU roundtrip不能证明TP2执行正确；代价是双GPU验收前BF16 TP2不可用。
2. 三列GPU token传输仍为私有载荷，AsyncModelOutput用CPU OutputSpec转为具名RequestOutput。engine/scheduler不按输出列位置或长度猜接受；代价是内部仍保留一种紧凑传输布局。
3. runner的私有pending tuple保留，只在外部边界使用SamplingPlan/OutputSpec/RequestOutput。这样避免无关的存储迁移；代价是未把runner内部待处理数据也统一成dataclass。
4. 历史rollback sampler通过benchmark专用shadow Sequence和载荷适配作状态参考。额外适配分配会污染新计时；复现旧性能归因必须使用原报告提交。
5. 普通decode可以与部分prefill共享余量，例如预算128时1行decode+127行prompt。移除旧的“只有第一个请求允许chunk”限制；真实GPU状态门禁覆盖了新布局，但未将这组控制流测试当成动态负载性能结论。
6. 已排队decode先消费，再调度更多工作，防止持续到达拆分decode cohort。可能减少prefill的CPU/GPU重叠；下一轮动态负载计时才能量化这一取舍。

## 审查未独立判断的范围

1. 审查者未与根任务同时执行GPU工作；GPU证据来自根任务完整套件和4B独立门禁，尚没有第二次独立GPU复现。
2. TP2、没有API的取消行为、hybrid prefix状态恢复没有实现或验收。前两类不因此宣称已支持；执行中preempt被显式拒绝，hybrid prefix报错。
3. 覆盖所有接受子集大小、稀疏选择与live B1..8，不声称穷举B8的256种mask；尚有未测试排列。
4. 两次样本只描述4070TiSUPER上的这组4B PC工作负载，不给统计置信度，也不外推5090、9B或其他prompt。MTP继续可选，没有依据这4个case自动改变默认策略。
5. 用户预先存在的改动不纳入实现审查，旧rollback适配器不作旧性能的新证据；对应文件保持不变，历史报告保留。

独立审查没有发现需要延后处理的Minor。

## 使用Graph统计

```python
engine.model_runner.cuda_graphs.reset_dispatch_stats()
# 运行实际请求
stats = engine.model_runner.cuda_graphs.dispatch_stats()
```

统计按prefill/decode/verify/draft/cache-only分别给出graph_calls、eager_calls、batch_counts和token_counts。prefill命中指piecewise图段被复用，attention core仍在图外执行。指标表示调用与shape分布，不等价于GPU耗时占比。
