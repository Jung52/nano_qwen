# MTP prefill：仅投影最终 tail 的端到端对照

2026-10-18。回滚合批（`ac2e76c`）是 off/old/new 共有基线；本报告独立评估MTP prefill的tail head裁剪。

**有明确的head阶段收益，尚未形成稳定端到端收益。** 输入256时，B4的head区间合计约从15.69～16.12ms降至8.00～8.26ms；但8组请求耗时均值只有3组改善、5组变差，新/旧收益范围为−3.81%～+4.42%。不据此宣称端到端优化已达标。

48次正式completion全部严格一致；8组长短输出的最终有效GDN/KV与旧路径逐位一致且finite。tail logits不是逐位相同（最大绝对差0.0625），本矩阵首draft与top-1全部相同。

## 范围与门禁

生产代码只改 `_run_mtp_prefill()`：先保存有效cache长度；非最终chunk直接返回；最终chunk仅对`mtp_hidden[-1:]`调用LM head和argmax。MTP hidden/KV、target、所有decode head、GEMM分派、接受判断和回滚均不修改。没有实现prompt cache-only。

旧路径仅从固定`f443cb3`提取原prefill方法；其余代码全部使用同一工作区的回滚合批版本。源码hash、调优前后配置、逐轮数据和完整completion见[results.json](results.json)。

## 协议

RTX 4070 Ti SUPER，4B per-channel FP8 target、独立BF16 MTP、TP1、greedy reuse。B={2,4} × 输入={64,256} × 输出={128,512}。decode Graph、prefill eager；max_num_batched_tokens=1024、max_model_len=1024、gpu_memory_utilization=.85、ignore_eos=True。输入沿用same64 seed并重复截取至指定长度。

每个batch容量使用同一常驻engine，先预热所有输入和off/old/new路径，GEMM tuner配置固定。每case交错`off1→old1→new1→off2→old2→new2`，每配置严格两次正式计时，共48样本。off仍常驻MTP权重。无profiler、stage hook或状态复制进入正式测量。

各batch容量正式样本全部结束后，再分别运行old/new独立诊断cohort；它们不计入正式均值、不作为第三次性能样本。诊断记录MTP prefill/body/head CUDA event区间，结束事件先于捕获用clone；在最终请求释放前按相同已消费前缀复制有效GDN/target KV/MTP KV，比较finite和逐位一致。EOS/chunk/slot复用另由真实tiny hybrid生命周期测试覆盖。

## 端到端结果（两次算术均值）

“请求收益”正值表示耗时降低。TPOT按每请求实际提交decode token计算，首token排除；TTFT和完整请求耗时直接来自真实engine。

| B | 输入 | 输出 | TTFT old→new ms | 请求 old→new s | 新/旧请求收益 | 新/off请求收益 |
|---|---|---|---|---|---:|---:|
| 2 | 64 | 128 | 108.77→132.41 | 1.6733→1.7371 | -3.81% | -0.86% |
| 2 | 64 | 512 | 140.49→180.89 | 6.7527→6.6177 | +2.00% | -1.98% |
| 2 | 256 | 128 | 129.76→123.24 | 1.6534→1.6635 | -0.61% | +3.50% |
| 2 | 256 | 512 | 109.02→157.96 | 6.3226→6.4835 | -2.55% | +2.13% |
| 4 | 64 | 128 | 143.05→140.20 | 1.6991→1.7074 | -0.49% | +1.04% |
| 4 | 64 | 512 | 150.21→112.70 | 6.9181→6.6123 | +4.42% | +0.55% |
| 4 | 256 | 128 | 179.74→157.71 | 1.9092→1.8592 | +2.62% | -3.84% |
| 4 | 256 | 512 | 184.51→170.09 | 6.8878→7.1175 | -3.33% | -6.08% |

| B | 输入 | 输出 | off TPOT ms | old TPOT ms | new TPOT ms | new/off decode吞吐 |
|---|---|---|---:|---:|---:|---:|
| 2 | 64 | 128 | 12.477 | 12.319 | 12.635 | -1.18% |
| 2 | 64 | 512 | 12.514 | 12.940 | 12.596 | -0.65% |
| 2 | 256 | 128 | 12.460 | 11.997 | 12.128 | +2.74% |
| 2 | 256 | 512 | 12.736 | 12.160 | 12.379 | +2.89% |
| 4 | 64 | 128 | 12.658 | 12.252 | 12.340 | +2.63% |
| 4 | 64 | 512 | 12.802 | 13.244 | 12.719 | +0.65% |
| 4 | 256 | 128 | 13.033 | 13.617 | 13.397 | -2.73% |
| 4 | 256 | 512 | 12.887 | 13.118 | 13.596 | -5.23% |

各两次原值完整保存在[timing.csv](timing.csv)。decode没有改动；其波动不直接归因于prefill head优化。不能用decode吞吐代替整请求结果。

## MTP prefill/head 独立诊断

表中为一个cohort中B个请求的CUDA event区间之和，单位ms；prefill父区间包含body/head及输入准备的排队空隙，不能把三列相加，也不等同于纯kernel累计时间。诊断每case一次、与正式TTFT独立；不以此估计统计显著性。

| Case | 投影行 old→new | MTP prefill old→new ms | MTP body old→new ms | MTP head old→new ms |
|---|---|---|---|---|
| B2-I64-O128 | 128→2 | 19.025→8.274 | 11.871→3.292 | 4.792→4.168 |
| B2-I64-O512 | 128→2 | 8.510→14.213 | 3.106→8.077 | 4.700→4.543 |
| B2-I256-O128 | 512→2 | 12.546→8.959 | 3.709→3.812 | 7.474→3.957 |
| B2-I256-O512 | 512→2 | 21.130→9.777 | 11.166→3.708 | 7.647→4.723 |
| B4-I64-O128 | 256→4 | 23.446→20.494 | 11.763→9.582 | 9.449→9.142 |
| B4-I64-O512 | 256→4 | 27.217→15.472 | 14.128→5.452 | 9.297→7.848 |
| B4-I256-O128 | 1024→4 | 48.389→32.536 | 26.487→19.955 | 16.116→7.996 |
| B4-I256-O512 | 1024→4 | 44.850→30.710 | 24.340→16.989 | 15.688→8.255 |

输入64时，head减少的绝对时间较小：B2每cohort约0.16～0.62ms，B4约0.31～1.45ms。输入256时分别减少约2.92～3.52ms、7.43～8.12ms。

行数减少64/256倍不意味着耗时等比例下降：这个模型的BF16词表权重形状仍是`248320×2560`，逻辑尺寸1,271,398,400字节（约1.184GiB），每次投影仍要访问同一组权重。此处是实现与尺寸分析，没有把逻辑权重体积当成实测DRAM流量。

MTP body代码、输入和hidden结果均不变，其event区间仍有明显波动。因此prefill父区间减少的全部时间不能都归因于head裁剪；本轮没有trace去进一步拆分CPU排队与纯kernel时间。正式TTFT/TPOT的正负变化也不作单一因果解释。

## 数值与状态结果

LM head从M=64/256改为M=1会触发不同BF16 GEMM形状。tail logits记录实际逐位结果和最大误差；不把top-1相同写成logits逐位相同。无数值drift放行参数：首draft、完整completion、MTP hidden、有效GDN/KV均按各自严格门禁判定。

| Case | tail logits逐位相同 | tail最大绝对差 | 首draft/top1 | 最终GDN/KV逐位且finite | 检查字节 |
|---|---|---:|---|---|---:|
| B2-I64-O128 | False | 0.03125 | 全部一致 | True | 66,756,608 |
| B2-I64-O512 | False | 0.03125 | 全部一致 | True | 95,068,160 |
| B2-I256-O128 | False | 0.0625 | 全部一致 | True | 80,912,384 |
| B2-I256-O512 | False | 0.0625 | 全部一致 | True | 109,232,128 |
| B4-I64-O128 | False | 0.03125 | 全部一致 | True | 133,529,600 |
| B4-I64-O512 | False | 0.03125 | 全部一致 | True | 190,136,320 |
| B4-I256-O128 | False | 0.0625 | 全部一致 | True | 161,824,768 |
| B4-I256-O512 | False | 0.0625 | 全部一致 | True | 218,447,872 |

新增12项回归覆盖两种FP8、eager/Graph：full/chunked prefill、非最终chunk无head、最终tail唯一一行、完整MTP hidden/KV不变、混合接受/拒绝、EOS与slot复用。全套最终 **676通过、32跳过、20个第三方弃用警告，150.81s**，见本地`logs/mtp-prefill-tail/full-tests.log`。跳过项为已有超出B4容量的参数case。基线相关87项通过，新测试先8项RED确认旧实现计算多余行，再12项GREEN。

## 边界与后续

每配置只有两次正式测量，全部正负结果保留；TTFT与CPU调度的样本波动仍可能影响整请求排名。所有结论限于当前GPU、4B per-channel与本矩阵，不外推5090、9B或其他内容。block-wise仅做正确性回归。

本轮未改prompt为cache-only，也未修改decode中的有效LM head。后续prompt cache-only应作为独立实验评估。

```bash
PYTHONPATH=src python -m pytest -q --ignore=logs
python benchmarks/bench_mtp_prefill_tail.py \
  --model /home/wei/code/models/4b-channel-fp8 \
  --mtp-model /home/wei/code/models/4b \
  --output logs/mtp-prefill-tail/channel.json
```
