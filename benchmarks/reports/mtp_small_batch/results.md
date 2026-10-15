# FP8 small-batch MTP（2026-10-15）

后续已加入全greedy fast path，off基线显著加快。这里保留`7dc09b9`阶段的原始结果；新的受控交错对照见[greedy fast path报告](../mtp_greedy_fast_path/results.md)，不能继续把下表提升直接套到新版本。

先完成[B1长度/实际内容矩阵](b1_results.md)及[reuse新trace](reuse_trace.md)，再将B2接入正式生产路径。本轮为4070 Ti SUPER、TP1、FP8 target + 独立BF16 MTP权重；B4/8和5090尚未测量，B4/8继续guard，普通decode B64峰值仍是独立历史基线。

## 行为与严格门禁

外部verify rows保持request-major `[A0,A1,B0,B1]`，dense内部转换time-major `[A0,B0,A1,B1]`。FP8投影共享2B行、使用本卡普通M=B调优配置而不覆盖2B缓存；norm/BF16例外/head按B行执行。GDN分两时间步批量更新B个pool slot，分别保存首步snapshot；attention逐时间步每请求单query。每请求accept mask、MTP有效长度及接受后的cache-only补写独立。verify/full draft/cache-only均有精确B1/B2图，B2→B1及只接受一个请求时无padding状态写。

两种FP8 × eager/Graph覆盖AA/AR/RA/RR、交替接受、129/257异长、128分块、EOS、输出预算1/2及不同结束时间、另一请求活跃时slot复用、三个排队请求与两个slot。serial/reuse在相同已消费前缀逐位比较hidden、GDN、有效target KV与MTP KV；inactive slot写入会失败。B1既有逐层/完整缓存链路门禁保留。完整451项测试通过，现有依赖deprecation warnings保留。

审查额外发现并修复KV压力生命周期：等待in-flight请求释放块；让持有KV的续块绕过被容量阻塞的重放；新请求不能抢占in-flight/partial请求的state slot，队首新请求也不能饿死已有续块。上述边界先RED后GREEN，并经独立CPU复核。

真实4B：两种FP8各在异长64/256、输出128完成eager off/serial/reuse严格completion门禁；该case的Graph off/reuse也与eager逐token一致。下列每case每模式正式2次，prefill eager、decode Graph、ignore EOS、profiler关闭；每样本计2×(128−1)=254个真实提交decode tokens。先做不计时4-token短预热。吞吐为batch聚合，TPOT为两个请求各自decode耗时/127后平均；TTFT从入队到首token所在step返回。engine初始化不计入请求时间。所有Graph样本必须两请求各128个completion与off完全相同，失败不作为有效速度收益。最终数据在调度边界修复后重新测量；此前诊断修订留在本地logs，未混入均值。

## B2正式结果

| FP8 | 用例 | off tok/s | reuse tok/s | 提升 | off TPOT ms | reuse TPOT ms | 接受率 | 严格token |
|---|---|---:|---:|---:|---:|---:|---:|---|
| channel | same64 | 112.10 | 169.53 | +51.2% | 17.842 | 11.798 | 81.4% | 2/2 |
| channel | unequal64-256 | 111.05 | 168.73 | +51.9% | 18.011 | 11.699 | 84.1% | 2/2 |
| channel | chinese-code | 111.44 | 156.98 | +40.9% | 17.948 | 12.449 | 82.6% | 2/2 |
| block | same64 | 110.77 | 145.65 | +31.5% | 18.056 | 13.743 | 80.0% | 2/2 |
| block | unequal64-256 | 109.95 | 137.53 | +25.1% | 18.190 | 14.221 | 75.7% | 2/2 |
| block | chinese-code | 112.71 | 151.28 | +34.2% | 17.744 | 12.654 | 83.3% | 2/2 |

same64为两个相同64-token合成prompt；unequal64-256为同种合成内容的64/256-token请求；chinese-code为45-token中文对话与43-token代码请求。全部输出128；完整prompt/completion、逐step真实提交数与耗时、每请求TPOT/TTFT、总耗时和显存均保留在[b2_results.json](b2_results.json)，逐次汇总见[b2_samples.csv](b2_samples.csv)。verify_steps计batch轮次，接受率按accepted/(accepted+rejected)请求对数计算，不使用accepted/轮次。step耗时包含调度、MTP与D2H，不冒充独立GPU verify算子耗时。

## 结论与限制

B1收益依内容、上下文与接受率变化，负收益也保留。新trace中target verify和LM head占主导，snapshot约1%～2%；item等待与GPU verify重叠。于是优先完成B2计算/映射，而未先实现直接snapshot kernel或完整GPU接受/回滚。B2用一个GPU accept mask生成每请求decision table并一次回传，host继续拥有EOS/调度生命周期，不宣称已无CPU同步。

每项只有两次正式样本，不能证明收益的统计稳定性，也不能将本轮绝对速度与旧报告跨轮相减归因于某个改动。allocator显存包含预分配KV与Graph；没有把它当MTP净开销。B1与B2是分别建立的实验，尚未形成完整batch收益分界曲线；B4/8必须继续验证混合接受、结束与动态映射后再开放。

接口/命令见[docs/mtp-verify-reuse.md](../../../docs/mtp-verify-reuse.md)。B1修复前的失败数据、修复后的64正式样本、新trace摘要均单独保留；原始trace及RED/GREEN日志保留在执行机logs/mtp-small-batch/。

计时源码指纹在JSON中保留。计时后仅补了普通decode非标准容量3/5/17的最终精确bucket，防止收紧capture列表时漏掉最大请求数；本轮B1/B2及独立B64的capture列表不变，并补7个容量覆盖回归。
