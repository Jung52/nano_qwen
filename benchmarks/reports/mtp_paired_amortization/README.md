# MTP 独立 off / reuse 配对实验

本轮用于回答两个问题：长输出能否摊薄 MTP 启动成本；B2/B4/B8 的损益是否已经出现在 decode 本身。生产代码冻结，不调整 LM head、GEMM、接受判断或回滚实现。

## 正式计时

```bash
/home/wei/miniconda3/envs/code/bin/python benchmarks/bench_mtp_paired.py \
  --model /home/wei/code/models/4b-channel-fp8 \
  --mtp-model /home/wei/code/models/4b \
  --output benchmarks/reports/mtp_paired_amortization/results.json
```

固定矩阵，共90个正式样本：

| Batch | 输入 | 输出 | 次数 |
| --- | --- | --- | --- |
| 1 | 64、1024 | 128、512 | 每配置5对 off/reuse |
| 2 | 64 | 512 | 5对 off/reuse |
| 4、8 | 64 | 128、512 | 每配置5对 off/reuse |

运行顺序为 batch→输入/输出配置→pair→mode。奇数对 off→reuse，偶数对 reuse→off，每对相邻；不删除负收益或异常偏慢样本。

- TP1、4B per-channel target、BF16 独立 MTP 权重、greedy、ignore_eos，固定 synthetic prompt。同一 batch 中各请求 prompt 相同。
- 两个完整 KV pool 不能同时驻留16GiB，因此每个正式样本顺序构造独立 engine；off 确实关闭 MTP，不分配 MTP 权重、KV 和 speculative snapshot。未使用同一个 MTP engine 的关闭开关冒充 off。
- `max_num_seqs=B`、`max_num_batched_tokens=2048`、`max_model_len=1538`、显存比例0.85。off 开启生产 prefill piecewise Graph；reuse 保持生产限制，prefill eager，verify/draft/cache-only 使用各自 Graph。
- 每个 batch 先对两种模式预热并记录完整 Triton autotune 配置；每个新 engine 对该 batch 涉及的各输入长度固定预热两次、输出8。B1 每个样本均预热64及1024输入。正式样本前后验证选择不增加、不替换。没有按结果改变预热、配置或重复次数。
- 模型加载、Graph capture、预热均不计入请求耗时；正式段没有 profiler、事件插桩或额外逐步同步。
- 请求端到端从入队开始计时，TTFT 到首次实际提交 token，TPOT 为每请求首 token 后的实际耗时除以实际后续输出 token 数。吞吐分母为本次实际提交的 `(output-1)*B`，不是 projected tokens/s。
- 接受率为 accepted request-trials / verified request-trials；后者由 verify 实际 batch×调用数求和，不能直接除以 verify batch 轮数。
- 每个配置10个样本全部严格比较完整 completion，保存实际逐步提交数、Graph dispatch、内存、源码 SHA256、调优配置和全部 token。

`results.json` 是原始证据，`samples.csv` 为便于分析的每样本指标；`measurements.md` 是自动审计、表格和全部配对波动。`results.md` 记录基于本轮数据的解释。

加速百分比定义为 `off/reuse-1`，正值表示 reuse 更快；这不是延迟下降百分比。每对先取比值，再取几何均值；区间是在5对 log-ratio 上计算的 df=4 t 区间，属于小样本描述。绝对耗时另报告算术均值，所有配对保留。

### 中断与恢复记录

首进程在18个完成样本后超过5分钟无进展，SIGINT未能退出，最终仅对该benchmark进程发送SIGTERM。未完成的尝试另存 `interrupted_attempt1.json`；这些18个已完成样本原样保留在最终数据中，未重抽、替换或择优。新进程两次相同off构造/预热/释放探针正常，卡住根因尚未确认。

恢复命令在原命令后加 `--resume`。恢复器只接受原计划的连续前缀，校验源码、模型、batch计划、严格门禁及engine对齐，恢复已记录调优选择，按原序补剩余样本；`continuations`保存恢复位置和新driver SHA。新增日志和120秒Python栈转储定时器仅覆盖未计时的engine构造，在预热及正式计时前取消。全部旧样本与中断记录保留，正式配置及样本数未改变。

正式90样本完成且续跑进程以0退出后，代码审查发现未来恢复流程需要收紧：原driver在保存样本之后检查source/tuning，新driver改为先记录独立 `invariants_pass` 再保存，恢复时拒绝该字段缺失或失败的样本；同时比较GPU/Torch/CUDA环境并记录续跑环境。新增回归先失败3项、修复后9项通过。该改动发生在正式计时结束后，不更改原始样本，也不把旧样本回填为新格式。实际运行的两个旧driver均按原SHA保存在 `drivers/`，`timing_completion.json`记录成功退出、全部门禁和18条原样保留审计。旧格式文件不能直接使用最新版driver继续恢复。

## 独立诊断

完成正式计时后才运行：

```bash
/home/wei/miniconda3/envs/code/bin/python benchmarks/profile_mtp_paired.py \
  --model /home/wei/code/models/4b-channel-fp8 \
  --mtp-model /home/wei/code/models/4b \
  --timing benchmarks/reports/mtp_paired_amortization/results.json \
  --output benchmarks/reports/mtp_paired_amortization/profiles.json

/home/wei/miniconda3/envs/code/bin/python benchmarks/report_mtp_paired.py \
  --directory benchmarks/reports/mtp_paired_amortization
```

B2/B4/B8、I64/O512，两种独立 engine。构造前恢复正式计时记录的 Triton 候选配置；遇到已有选择不一致时报错，不覆写不同配置。

每个配置分4个独立 pass：

1. 首轮 prefill trace，包含 target prefill、target head、MTP prefill/首 draft。
2. completion 计数112→144的稳态 decode trace，记录 target verify/head、draft/head/cache-only、snapshot/rollback、决策表回传、物理设备时间线。reuse 最后一轮可能多提交1 token，按实际数量归一化。
3. 全512 token 的 host scope 诊断。
4. 全512 token 的 CUDA event 诊断，重新capture并逐步同步，用于区分捕获图内部各阶段，不用于正式性能和GPU空隙结论。

前两个 trace 在事件重新 capture 前完成，保持未加 event nodes 的生产 Graph。所有 pass 均继续完成512 token并与正式 off completion严格一致。主机等待与GPU时间存在重叠；CUDA父子scope也存在重叠，不得相加当作总时间。`.tolist()`等待中的GPU执行不能全部归因于可消除CPU开销。

压缩 trace 可解压后在 Perfetto 中打开。`profiles.json` 保存每个 trace 的原文/压缩 SHA256、物理kernel组及设备忙碌并集、首尾跨度内空隙、实际提交数和活跃batch/slot记录。词表投影的 Graph kernel 指纹识别有口径限制，应与独立 event pass交叉判断。

## main 独立改进

本轮 generic 改进位于独立 main worktree，未参与这轮 mtp2 源码冻结的 A/B：

| 提交 | 内容 | main完整回归 |
| --- | --- | --- |
| a91c95a | 完整 worker 元数据序列化；TP1/hybrid prefix能力限制 | 107通过 |
| cfda31c | 只投影/采样指定目标行，非最终chunk不投影 | 118通过 |
| 02d9e18 | decode优先预算、partial prompt/slot ownership | 123通过 |
| 4c43f71 | 精确shape Graph/eager调用统计及batch直方图 | 124通过 |

MTP draft、提交与回滚协议留在 mtp2。main 提交均为本地提交，本轮未执行 push。
