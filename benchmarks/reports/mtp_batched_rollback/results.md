# MTP rollback 合批：同 engine 旧/新对照

2026-10-18，以 `f443cb3` 为基线的回滚合批独立对照。源码 SHA256、固定参考版本及全部原值保存在 [results.json](results.json)。

同 engine、同调优配置下，B2/B4 新路径相对旧路径 decode 吞吐提高 **5.4%/4.0%**。B4 每拒绝轮层级调用从 **96→24** 次，恢复 API 的 host 耗时约 **15.10→1.27ms**。相对 MTP off，新路径平均仅提高 **2.8%/0.6%**；尤其 B4 的收益小于样本间波动，不视为稳定超过 off。

## 改动与控制变量

- runner 收集 draft 被拒绝及接受 draft 本身为 EOS 的 slot；在后续 MTP cache-only/draft 前统一恢复。bonus 为 EOS 时保留第二步状态。
- 每轮只构造一次 CUDA int64 slot 索引，各 GDN 层复用；每层一次 Triton kernel 同时直接恢复 conv/recurrent，不创建 gather 临时张量。单 slot 兼容入口保留。
- 不改 snapshot 保存、LM head、GEMM 配置、接受判断、MTP 缓存或 graph 算法；profile 只把 rollback scope 改为批量 API。

## 测量协议

RTX 4070 Ti SUPER、4B per-channel FP8 target、独立 BF16 MTP、TP1、greedy、reuse；输入 same64，输出128，decode Graph、prefill eager，ignore_eos=True。EOS/slot 生命周期另由 tiny hybrid 状态回归覆盖。
每个 batch 建立一个常驻 engine，顺序固定 `off1 → legacy1 → batched1 → off2 → legacy2 → batched2`，每配置恰好两次正式测量，无 profiler/host hook。旧路径从固定提交提取原 `sample_tokens` 和原逐 slot rollback；因此没有用新 runner 的调度顺序冒充旧实现。
各路径先做8-token非计时 warmup；直接恢复 kernel 在无活跃请求时预热所有实际 slot 数。正式样本及后续诊断前后的 Triton GEMM tuner cache 完全一致，未修改或重捕获 graph，未手动更换配置。MTP关闭路径仍常驻 MTP 权重，保证旧/新与 off 的资源条件一致。
TPOT 是每请求 decode 秒数 / 实际提交 token 数；吞吐统计合计提交 tokens，不是 projected tokens。首 token 不计入 decode：B2每次254 tokens，B4每次508 tokens。TTFT/请求总耗时/显存在原始JSON/CSV中保留。

## 正式结果

| Batch | 路径 | 两次 decode tok/s | 平均 tok/s | 平均 TPOT ms |
|---|---|---|---:|---:|
| 2 | off | 179.49 / 181.96 | 180.72 | 11.067 |
| 2 | legacy | 172.72 / 179.79 | 176.25 | 11.352 |
| 2 | batched | 182.21 / 189.30 | 185.76 | 10.771 |
| 4 | off | 344.22 / 349.39 | 346.81 | 11.534 |
| 4 | legacy | 335.29 / 335.60 | 335.45 | 11.924 |
| 4 | batched | 355.24 / 342.54 | 348.89 | 11.469 |

| Batch | 新/旧吞吐 | 新/旧TPOT | 旧/off吞吐 | 新/off吞吐 | 接受数/draft数 |
|---|---:|---:|---:|---:|---:|
| 2 | +5.39% | -5.12% | -2.47% | +2.79% | 81.4% |
| 4 | +4.01% | -3.82% | -3.28% | +0.60% | 84.1% |

两轮配对的新/旧 decode 吞吐变化均为正：B2为+5.49%/+5.29%，B4为+5.95%/+2.07%。但B4第二轮新路径比off慢约2.0%，因此不能把两次平均的+0.6%解读为稳定获益。历史报告的绝对吞吐不拼入本轮均值。

prefill未改动。TTFT波动也影响整请求收益，不能用decode吞吐替代端到端结果：

| Batch | 路径 | TTFT均值 ms | 请求总耗时均值 s |
|---|---|---:|---:|
| 2 | off | 61.66 | 1.4672 |
| 2 | legacy | 70.73 | 1.5124 |
| 2 | batched | 99.48 | 1.4674 |
| 4 | off | 72.60 | 1.5375 |
| 4 | legacy | 145.38 | 1.6598 |
| 4 | batched | 179.39 | 1.6359 |

以上TTFT差异没有额外归因实验；本轮不把它归因于未修改的prefill算子。B2新路径请求总耗时与off基本相同，B4仍比off长约6.4%，保留这项负结果。

## 独立 host-only 诊断

正式计时结束后，各路径另跑一个诊断 cohort，只用 CPU perf_counter_ns。没有 profiler、CUDA timing event、额外设备读取或同步，诊断不作为第三次正式样本。下表为层级恢复 API 的 CPU wall time：旧 API 包含每层/每 slot 索引构造，新 API 不含 runner 外部的一次共享索引构造；正式 TPOT 包含两者全部成本。CPU scope 也可能包含运行时等待，不能把整个差值当成可回收 GPU 时间。

| Batch | 路径 | 拒绝轮 | 拒绝请求 | 层级调用数 | 每拒绝轮调用数 | API总计 ms | 每拒绝轮 API ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 2 | legacy | 13 | 26 | 624 | 48 | 65.485 | 5.037 |
| 2 | batched | 13 | 26 | 312 | 24 | 11.896 | 0.915 |
| 4 | legacy | 10 | 40 | 960 | 96 | 150.970 | 15.097 |
| 4 | batched | 10 | 40 | 240 | 24 | 12.716 | 1.272 |

same64 中同一批请求的接受/拒绝同步：B4整批拒绝时96→24次，B2为48→24次。诊断逐轮检查新路径每层仅恢复一次，且所有层共享同一 data_ptr。稀疏乱序 slot 和混合决策由专项回归验证。

## 正确性与验收

- 新增37个回归 case：空集合、单 slot、全 slot、乱序 `[6,1,4]`、masked tail、实际4B状态shape、BF16及FP16/FP32混合存储、snapshot和未回滚slot逐位不变、Graph多次重放与索引更新。
- 两种 FP8、eager/Graph，使用含两层 GDN + 一层 attention 的真实 tiny hybrid 模型，对照固定旧路径；覆盖全部接受/拒绝、混合及交替决策、draft EOS/bonus EOS、不同长度、结束与新请求 slot 复用。completion、逐行 hidden/logits、GDN和有效 target/MTP KV要求 finite 且逐位相同。
- 全尺寸4B另做旧/新 correctness-only cohort：在首次实际接受、首次实际拒绝轮，按相同已消费前缀比较整个 GDN pool 及各请求有效 target/MTP KV；CPU复制与门禁完全独立于正式计时和host诊断，completion也严格相同。

| Batch | 轮类型 | 检查tensor数 | 检查字节数 | 结果 |
|---|---|---:|---:|---|
| 2 | accepted | 84 | 57,556,992 | finite、bitwise均通过 |
| 2 | rejected | 84 | 57,630,720 | finite、bitwise均通过 |
| 4 | accepted | 120 | 115,113,984 | finite、bitwise均通过 |
| 4 | rejected | 120 | 115,261,440 | finite、bitwise均通过 |

全套回归的最终命令与结果见本地 `logs/mtp-batched-rollback/final-tests.log`。基线627通过、32跳过；改动后664通过、32跳过，跳过为已有超出B4容量的参数case，20个第三方弃用警告保留。

## 结论边界

本实验同时包含“回滚请求合批”和“直接按slot拷贝”，收益归因于这条完整回滚路径；没有拆成两个独立消融。正式结果只有每配置两次，保留全部原值及负收益，不能推断统计显著性。
该结果限于当前卡、4B per-channel、same64/128及对应 tuner 配置；block-wise 的正确性门禁通过，不据此声称性能收益，5090/9B/更大batch也未外推。
LM head 和其他矩阵工作保持原样。后续是否评估 LM head，应依据这里的端到端余量单独决定；本轮没有增加其他优化。

## 复现

```bash
PYTHONPATH=src python -m pytest -q --ignore=logs
python benchmarks/bench_mtp_rollback.py \
  --model /home/wei/code/models/4b-channel-fp8 \
  --mtp-model /home/wei/code/models/4b \
  --batches 2 4 --output logs/mtp-batched-rollback/channel.json
```

原始正式样本：[timing.csv](timing.csv)；host汇总：[host.csv](host.csv)；完整逐轮、状态hash、调优配置：[results.json](results.json)。
