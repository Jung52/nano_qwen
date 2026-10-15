# MTP cache 生命周期修复与 FP8 验证（2026-10-15）

基于 `feat/qwen35-mtp2` 的 `8322fe2395f398b25bce11677265d3491cfc8d6c`。GPU：RTX 4070 Ti SUPER 16 GiB；PyTorch 2.12.1+cu130，Triton 3.7.1，FlashInfer 0.6.12。仅支持 greedy、TP=1、B=1；B4/8 未实现，保留生产 guard。

## 缓存修复

- RoPE 位置 `p` 对应 MTP 物理逻辑行 `p-1`，有效缓存为 `[0,p)`；读取长度为 `p`。256-token prompt 的 MTP 缓存正好占一个 256 行 block。
- verify 的 target 两行和 MTP 补写两行占用相同容量范围；预留 2 行，去掉原来的额外第 3 行。
- 接受 draft 后，先用第一行 target hidden 与已接受 draft 补写，再用第二行 target hidden 与 bonus 补写。拒绝时只追加 corrected token 对应的一行。
- chunked prefill 使用累计 KV 长度。129-token prompt 按 128+1 执行时，第二块查询两行，KV 长度为 129；第一块只有 1 token 也保存 previous hidden。
- draft 为 EOS 时回滚到第一行 GDN snapshot。结束时清理 MTP metadata；抢占后从零重放时同时重置已有 slot 的 GDN 和 MTP 状态。

## 正确性证据

完整 `PYTHONPATH=src:benchmarks python -m pytest -q tests`：**261 passed**。

| 专项 | 数量 | 覆盖 |
|---|---:|---|
| CPU 生命周期与元数据 | 105 | 9 个 prompt 边界、连续/交替接受拒绝、最终输出预算、EOS、非连续物理 block、容量 |
| 真实 CUDA 混合模型 | 36 | eager/graph；完整/分块 prefill；相同已消费前缀下 completion、GDN、target KV、MTP KV；1-token chunk；slot 复用与抢占重放 |
| 独立权重与 benchmark gate | 13 | FP8 target 加载原始 MTP、架构/有效 RoPE 不匹配、raw FP8 MTP 拒绝、drift flag 无法绕过失败 |

其余107项覆盖FP8、GDN、graph及benchmark接口等回归。

同一 prefill 分块方式下，MTP 与普通 decode 的 GDN/target KV 逐位相同；verify logits 也与两次普通 B1 decode 逐位相同。MTP 输出及有效 KV 另用相同 target hidden 的完整前缀独立前向校验（BF16 `rtol=.02, atol=.003`）。完整 prefill 与 chunked prefill 的不同 GEMM/scan 形状有 BF16 舍入差异，状态比较采用归一化 L2 误差 <1%；索引和有效长度仍精确检查。真实模型的 completion 比较不允许 token 差异。

发现额外问题：旧两行并行 target verify 的 GEMM/attention 形状与普通 B1 decode 不同，4B BF16 在第 6 个 completion token 已分歧。修复采用 **两次 B1 target forward + 两次单行 LM head**；graph 将两次 target 前向一起捕获。它保留基线算术，但不再拥有两行 target GEMM 并行收益，不能假定会加速。独立的宽投影测试复现并覆盖了 GEMV/GEMM 舍入差异。

真实 BF16 4B graph，以及 per-channel / block-wise FP8 的 eager、graph 均完成 64-input / 16-output 严格短输出 gate，completion 全部一致。这些短 gate 的时间包括冷启动影响，仅用于正确性，不作为性能结论。

## 独立 MTP 权重

转换后的 FP8 target 不含 `mtp.*`，用 `Config.mtp_model` / CLI `--mtp-model` 指向原始 BF16 checkpoint。架构字段、GDN 参数及模型实际使用的归一化 RoPE 一并校验；不支持直接把未解量化的 FP8 MTP 张量当作 BF16 权重加载。架构一致不证明 checkpoint 权重来源相同，调用者需选择对应原始模型。

## 正式性能协议

入口为 `benchmarks/bench_qwen3_5_mtp_production.py` 的生产 `run_mode()`；输入64、输出128、B1，greedy，忽略 EOS，decode graph 开启。MTP prompt prefill 目前为 eager，两组保持一致。每种 FP8 两次，每次分别创建 MTP off/on engine；仅计 prefill 首 token 后实际提交的 **127 个 decode token**。每一对均要求完整128个 completion token相同。不开 profiler，不使用 projected throughput 或 drift 放行。

MTP off 回归另用相同 `bench_one_batch.py` 比较基准8322与修复后代码：输入64、decode127、B1，prefill/decode 均 graph，固定工作负载 KV 容量，1次 warmup、2次正式计时。两种基准的 prefill 设置不同，因此 off 回归吞吐不与上面的 MTP 比较表混用。

### MTP on/off 正式结果

| 量化 | 次数 | MTP关闭 tok/s | MTP开启 tok/s | 开启/关闭 | 接受 draft / verify |
|---|---:|---:|---:|---:|---:|
| per_channel | 1 | 83.09 | 64.15 | 0.772× | 60/66 |
| block_wise | 1 | 83.17 | 61.13 | 0.735× | 57/70 |
| per_channel | 2 | 83.17 | 61.61 | 0.741× | 60/66 |
| block_wise | 2 | 81.39 | 59.86 | 0.735× | 57/70 |

两次 tok/s 的算术均值：

| 量化 | MTP关闭 | MTP开启 | 开启/关闭 | 吞吐变化 |
|---|---:|---:|---:|---:|
| per_channel | 83.13 | 62.88 | 0.756× | -24.4% |
| block_wise | 82.28 | 60.50 | 0.735× | -26.5% |

**这条严格一致的 MTP 路径没有加速。** 较高接受率没有抵消两次串行 target、额外 MTP/LM head、snapshot/rollback 与调度开销。`enable_mtp` 继续默认关闭；当前结果不能宣传 FP8+MTP 加速。

### MTP关闭的版本对照

| 量化 | 版本 | prefill两次 ms | decode两次 tok/s |
|---|---|---|---|
| per_channel | base8322 | 32.81/33.95 | 82.37/81.99 |
| block_wise | base8322 | 33.69/34.53 | 85.22/85.22 |
| per_channel | fixed | 63.30/33.36 | 83.24/83.51 |
| block_wise | fixed | 38.17/50.25 | 83.23/83.07 |

per_channel decode均值变化 +1.5%（82.18→83.38 tok/s）。block_wise decode均值变化 -2.4%（85.22→83.15 tok/s）。 两个样本不足以证明严格零回退；prefill存在33–63ms的明显波动，因此不把它描述为已通过的性能等价结论。两版本实际路径都是1次prefill graph、127次decode graph，allocator分配/峰值相同；这不替代更稳定环境下的性能验证，也不扩展为大batch回归结论。

版本对照最初因旧`path_audit`不能接收runner的`is_verify`参数，在正式计时前失败。只修复benchmark透明转发参数；修复前失败日志保留。后续两版本使用完全相同的修正harness，8322生产代码没有改动；每配置仍仅2个有效正式样本。

### 短 trace：block-wise，8次verify，提交14个decode token

两条诊断 trace 都实际接受6次、拒绝2次；它们带profiler，不参与上表吞吐统计。

| Graph GPU类别 | 累计kernel ms | 占物理GPU事件累计时间 |
|---|---:|---:|
| fp8_gemm | 144.048 | 52.5% |
| bf16_gemm | 68.711 | 25.0% |
| state_index_copy | 5.582 | 2.0% |

索引读写组包括原有conv-state更新、MTP快照/回滚及embedding，不能把它全部归因于新增快照。eager trace中单独标记的快照：192次层调用、768个物理设备事件，kernel累计5.465ms；拒绝回滚48次层调用、240个物理设备事件，kernel累计1.360ms。Graph里的回滚kernel累计0.997ms；捕获时没有Python快照标记，所以不单独推断graph快照耗时。annotation区间不当作kernel时间叠加。

每请求24层GDN的conv/recurrent state合计25.125MiB。当前快照经过`index_select`临时张量再`index_copy_`，完整一次快照约100.5MiB逻辑读写，完整一次回滚同样约100.5MiB；这是逻辑流量估算，不是实测DRAM流量。两次B1 target forward的原始state更新另计。

采样8次`aten::item`在graph trace累计CPU区间237.947ms，包含等待前序GPU计算，**与GPU时间重叠，不能直接相加或称为可全部消除的CPU开销**。trace确认它强制每次verify同步，尚未证明消除同步能单独带来多少收益。

下一步优先从矩阵计算减少重复工作：恢复两行target并行时逐算子验证与B1基线的一致性；接受draft的第一行MTP只需补KV，目前仍计算最终被丢弃的词表预测，可考虑cache-only前向/graph。随后减少snapshot临时张量和同步；不能仅靠接受率或去掉B1 guard推断收益。


## 复现

```bash
PYTHONPATH=src:benchmarks python -m pytest -q tests
for run in 1 2; do
  PYTHONPATH=src python benchmarks/bench_qwen3_5_mtp_production.py \
    --model /path/to/fp8-target --mtp-model /path/to/original-bf16 \
    --prompt-tokens 64 --max-tokens 128 --duration 60
done
```

原始 stdout、短 trace、冻结的8322源码和本机执行编排在 `logs/mtp-cache-lifecycle/`（Git忽略）。报告 JSON/CSV 保留测量原值、源码指纹及诊断摘要；临时编排脚本不作为新的生产 benchmark 入口。
