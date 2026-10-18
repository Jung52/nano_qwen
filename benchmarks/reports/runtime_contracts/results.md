# 执行、状态提交与调度边界结果

2026-10-18，RTX 4070 Ti SUPER 16GB，4B per-channel FP8，TP1，输出128。所有性能值取两次正式样本算术平均；正收益表示更快。原始 completion、每步实际提交 token、Graph 分布、源文件哈希和调优配置在 [results.json](results.json)，命令和采样口径在 [README.md](README.md)。

本轮只有 B1、input64 在独立 off 对照中取得明确正向样本：端到端+10.5%。B2、input64 为-6.1%；input1024 下 B1/B2 为-0.6%/-7.3%。MTP继续保持可选，现有结果不足以支持默认开启或扩大适用范围；B2短输入同 engine 的+0.48%端到端差值也应视为接近持平。

## 本轮修改

- main 与 mtp2 均完整序列化 Sequence 执行元数据。由于没有双 GPU 实测，TP>1 显式报错；没有将 CPU roundtrip 当成 BF16 TP2 验收。
- 两分支均禁止 hybrid prefix cache，等待 KV、conv、recurrent 状态一起恢复的实现。mtp2 修复纯 attention prefix 的登记区间，在更新/释放后仍按本轮实际消费的区间登记。
- target 保留完整 hidden，仅把最终 prefill 尾行及 decode 行送入 LM head；非最终 chunk 零投影、零采样。verify 的每请求两行继续使用已经验证的两次 M=B 词表投影。
- RequestOutput 明确 committed_tokens、accepted_count、next_draft、cache_advance。异步三列 GPU 载荷留在私有传输内，engine/scheduler 消费具名结果。draft 不进入正式 token_ids，postprocess 才统一提交。拒绝与 draft EOS 推进一步，接受并提交 bonus 推进两步。
- BatchDescriptor 校验实际执行模式；验证行输入独立追加 draft。拒绝重复/非法提交和执行中请求的提前抢占。
- 优先给已有 decode 分配 token 预算，普通 decode 使用余量接 prefill；MTP verify 保持隔离。已排队 decode 先消费，防止持续到达进一步拆散 decode cohort。
- Graph 记录 prefill/decode/verify/draft/cache-only 的 graph/eager 次数、请求 batch 与执行 token 分布。没有增加捕获 shape 或不安全的状态 padding。

## 正式性能

`resident_off`：同一个已加载 MTP 的 engine 临时关闭 MTP，prefill eager。
`reuse`：同 engine 开启 MTP reuse，prefill eager。
`independent_off`：独立关闭 MTP 的 engine，无 MTP 权重/缓存/快照，prefill piecewise Graph。
三者都使用普通 decode Graph；reuse 还使用 verify/draft/cache-only Graph。计时包含调度、结果消费与状态提交；不含模型加载、Graph 捕获、编译预热或分词。端到端收益含 TTFT，decode 吞吐只统计首 token 后实际提交的127×B token。

| B | input | 模式 | TTFT ms | TPOT ms | decode tok/s | 请求总耗时 ms |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 64 | resident_off | 85.91 | 12.403 | 80.63 | 1661.13 |
| 1 | 64 | reuse | 58.70 | 10.863 | 92.06 | 1438.30 |
| 1 | 64 | independent_off | 33.04 | 12.256 | 81.61 | 1589.56 |
| 1 | 1024 | resident_off | 102.30 | 12.691 | 78.79 | 1714.12 |
| 1 | 1024 | reuse | 106.58 | 12.718 | 78.65 | 1721.72 |
| 1 | 1024 | independent_off | 104.71 | 12.658 | 79.02 | 1712.22 |
| 2 | 64 | resident_off | 75.08 | 12.727 | 157.16 | 1691.41 |
| 2 | 64 | reuse | 107.80 | 12.405 | 161.24 | 1683.26 |
| 2 | 64 | independent_off | 34.69 | 12.179 | 164.22 | 1581.43 |
| 2 | 1024 | resident_off | 215.51 | 13.002 | 153.84 | 1866.81 |
| 2 | 1024 | reuse | 228.61 | 13.107 | 152.59 | 1893.23 |
| 2 | 1024 | independent_off | 213.34 | 12.142 | 164.85 | 1755.40 |

| B | input | reuse 接受率 | 对 resident off 的 decode 收益 | 对 resident off 的端到端收益 | 对 independent off 的 decode 收益 | 对 independent off 的端到端收益 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 90.9% | +14.17% | +15.49% | +12.80% | +10.52% |
| 1 | 1024 | 70.3% | -0.19% | -0.44% | -0.47% | -0.55% |
| 2 | 64 | 81.4% | +2.60% | +0.48% | -1.82% | -6.05% |
| 2 | 1024 | 75.0% | -0.81% | -1.40% | -7.44% | -7.28% |

`verify_steps` 原始计数单位为 batch；上表接受率使用 Graph 分布还原的实际验证请求次数，B2 每轮计两个验证机会。独立 off engine 与 reuse engine 的捕获图、KV 容量和资源分配不同，因此对独立 off 的差值代表实际部署开关代价，不能只归因为单个 kernel 或 MTP 矩阵复用。两次样本能检查方向，不能建立稳定性置信区间；本轮没有扩大 B4/B8、9B 或5090结论。

## 正确性与显存

24个正式样本的完整128输出严格一致。另有4组独立诊断，各比较 target 全行 head 与采样行 head：first draft、tail top1 和 completion 一致，完整 hidden、GDN 与有效 target/MTP KV 逐字节一致。BF16 tail logits 允许在 top1 不变时记录舍入差值，没有使用 drift waiver。

- B1-I64：tail 最大绝对差 0.031250；逐字节检查 32.15 MiB；有效前缀 [{'slot': 0, 'target_prefix': 191, 'mtp_prefix': 190, 'committed': 128}]。
- B1-I1024：tail 最大绝对差 0.062500；逐字节检查 70.59 MiB；有效前缀 [{'slot': 0, 'target_prefix': 1151, 'mtp_prefix': 1150, 'committed': 128}]。
- B2-I64：tail 最大绝对差 0.000000；逐字节检查 64.29 MiB；有效前缀 [{'slot': 0, 'target_prefix': 191, 'mtp_prefix': 189, 'committed': 128}, {'slot': 1, 'target_prefix': 191, 'mtp_prefix': 189, 'committed': 128}]。
- B2-I1024：tail 最大绝对差 0.000000；逐字节检查 141.17 MiB；有效前缀 [{'slot': 0, 'target_prefix': 1151, 'mtp_prefix': 1150, 'committed': 128}, {'slot': 1, 'target_prefix': 1151, 'mtp_prefix': 1150, 'committed': 128}]。

理论上，248320词表的 BF16 logits 每行485KiB。B1,input1024 从485MiB降至约0.474MiB；B2,input1024 从970MiB降至约0.947MiB。这只是 logits 张量本身，不等于完整峰值显存节省。KV 池会按剩余预算扩张，不能把 engine 总 allocated 的差值直接解释为权重成本。

| engine | MTP权重 MiB | MTP KV MiB | target KV MiB | KV blocks | 初始化后 allocated MiB | prefill Graph |
|---|---:|---:|---:|---:|---:|---|
| resident-B1 | 230.03 | 750.00 | 6000.00 | 750 | 12395.00 | False |
| independent-off-B1 | 0.00 | 0.00 | 6848.00 | 856 | 12322.89 | True |
| resident-B2 | 230.03 | 728.00 | 5824.00 | 728 | 12281.34 | False |
| independent-off-B2 | 0.00 | 0.00 | 6696.00 | 837 | 12228.54 | True |

## 调度与 Graph

使用实际 Scheduler，复现 A 已 decode、B/C 各1024输入、预算128：旧版先执行16个 prefill batch/2048 token；新版第一批执行 A 的1个decode token和127个prefill token，A之前的纯prefill等待为0。持续到达和异步 engine 队列另有CPU回归。这里证明了调度顺序，没有声称动态负载的 GPU TPOT 数值。

Graph B64/63/48/47/32/31/1 的 CPU 分派回归记录4次 Graph、3次 eager；B63保持 eager。真实本轮B1/B2分布如下，命中只按对应执行模式的调用计数，不把 prefill 图外的 attention core 误称为 full Graph。

| 正式样本组 | 模式 | Graph调用 | eager调用 | 请求batch分布 |
|---|---|---:|---:|---|
| B1/I64/independent_off | prefill | 2 | 0 | {'1': 2} |
| B1/I64/independent_off | decode | 254 | 0 | {'1': 254} |
| B1/I64/resident_off | prefill | 0 | 2 | {'1': 2} |
| B1/I64/resident_off | decode | 254 | 0 | {'1': 254} |
| B1/I64/reuse | prefill | 0 | 2 | {'1': 2} |
| B1/I64/reuse | verify | 132 | 0 | {'1': 132} |
| B1/I64/reuse | cache_only | 120 | 0 | {'1': 120} |
| B1/I64/reuse | draft | 132 | 0 | {'1': 132} |
| B1/I64/reuse | decode | 2 | 0 | {'1': 2} |
| B1/I1024/independent_off | prefill | 2 | 0 | {'1': 2} |
| B1/I1024/independent_off | decode | 254 | 0 | {'1': 254} |
| B1/I1024/resident_off | prefill | 0 | 2 | {'1': 2} |
| B1/I1024/resident_off | decode | 254 | 0 | {'1': 254} |
| B1/I1024/reuse | prefill | 0 | 2 | {'1': 2} |
| B1/I1024/reuse | verify | 148 | 0 | {'1': 148} |
| B1/I1024/reuse | cache_only | 104 | 0 | {'1': 104} |
| B1/I1024/reuse | draft | 148 | 0 | {'1': 148} |
| B1/I1024/reuse | decode | 2 | 0 | {'1': 2} |
| B2/I64/independent_off | prefill | 2 | 0 | {'2': 2} |
| B2/I64/independent_off | decode | 254 | 0 | {'2': 254} |
| B2/I64/resident_off | prefill | 0 | 2 | {'2': 2} |
| B2/I64/resident_off | decode | 254 | 0 | {'2': 254} |
| B2/I64/reuse | prefill | 0 | 2 | {'2': 2} |
| B2/I64/reuse | verify | 140 | 0 | {'2': 140} |
| B2/I64/reuse | cache_only | 112 | 0 | {'2': 112} |
| B2/I64/reuse | draft | 138 | 0 | {'2': 138} |
| B2/I1024/independent_off | prefill | 2 | 0 | {'2': 2} |
| B2/I1024/independent_off | decode | 254 | 0 | {'2': 254} |
| B2/I1024/resident_off | prefill | 0 | 2 | {'2': 2} |
| B2/I1024/resident_off | decode | 254 | 0 | {'2': 254} |
| B2/I1024/reuse | prefill | 0 | 2 | {'2': 2} |
| B2/I1024/reuse | verify | 144 | 0 | {'2': 144} |
| B2/I1024/reuse | cache_only | 108 | 0 | {'2': 108} |
| B2/I1024/reuse | draft | 144 | 0 | {'2': 144} |
| B2/I1024/reuse | decode | 2 | 0 | {'2': 2} |

## 验证及边界

完整测试和最终审查记录见 [acceptance.md](acceptance.md)。main 只落安全边界改动，mtp2 包含执行、提交、调度与统计。两处均保持本地，未自动提交或推送。历史 rollback 的 pinned sampler 现经 benchmark 专用适配用于状态参考，新的适配开销不用于旧性能归因；旧报告数据保留。
