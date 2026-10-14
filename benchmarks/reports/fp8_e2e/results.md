# FP8 精简端到端测试

batch=8，输入64 token/请求（prefill M=512），输出128 token/请求，其中prefill产生第1个输出，随后127次decode（M=8）。
真实 engine.step，piecewise prefill graph + decode graph，queue=2，async output，TP=1，相同32个KV blocks，prefix cache关闭。每配置两次；加载/JIT/capture/调优和预热另计。

| 方案 | TTFT ms | decode tokens/s | 总耗时 ms | 峰值 allocated MiB |
|---|---:|---:|---:|---:|
| bf16 | 81.22* | 395.89 | 2647.57 | 10268.45 |
| per_channel | 59.06 | 593.39 | 1771.61 | 6273.51 |
| block_current | 77.29* | 589.77 | 1800.00 | 6270.05 |
| block_optimized | 84.72* | 592.54* | 1800.63* | 6270.05 |

* 表示两样本波动超过5%；原始值、reserved显存及各项波动见JSON/CSV。
block优化前后：TTFT变化 +9.61%；总耗时变化 +0.03%；decode吞吐变化 +0.47%。

优化仅限本GPU、BF16输出、block GEMM M/N/K=512/2560/4096：BM32/BN128/warps4/stages3。GDN out_proj和full-attention o_proj共32层命中相同shape；decode选择保持基线缓存。

## 精度回归

使用BF16生成的同一轨迹作teacher forcing，每方案检查prefill尾部8个位置及127轮decode，共1024个位置。logits比较覆盖完整词表；诊断独立于正式计时。未测数据集perplexity或任务分数。

| 方案/参考 | top1一致率 | logits相对L2 | 平均KL |
|---|---:|---:|---:|
| per_channel/vs_bf16 | 0.985352 | 0.122056 | 0.00551463 |
| block_current/vs_bf16 | 0.994141 | 0.106592 | 0.00393994 |
| block_optimized/vs_bf16 | 0.994141 | 0.106592 | 0.00393994 |
| block_optimized/vs_block_current | 1.000000 | 0 | 0 |

同量化方案的优化前后要求relative L2<0.005、top1一致率≥0.99；BF16与FP8差异单独报告，不把量化误差算成tile优化引入的误差。
基线调优缓存从前次prefill诊断恢复；测量期间无新的GEMM搜索。四组的调优缓存、质量hidden/teacher token及完整worker日志保存在输出目录。
结果JSON包含模型、源码、缓存指纹和完整实验设置；数据目录中的identity.json用于拒绝不兼容复用。

本次两样本未证实block配置调整带来额外端到端提速。优化组TTFT两次差异较大，不能由此确定prefill收益。

| 方案 | 两次TTFT ms | 两次总耗时 ms | TTFT/总耗时波动 % |
|---|---|---|---|
| bf16 | 77.657 / 84.775 | 2647.525 / 2647.616 | 8.76 / 0.00 |
| per_channel | 58.554 / 59.575 | 1747.120 / 1796.093 | 1.73 / 2.76 |
| block_current | 80.977 / 73.599 | 1802.324 / 1797.681 | 9.55 / 0.26 |
| block_optimized | 68.568 / 100.864 | 1737.984 / 1863.273 | 38.12 / 6.96 |

首次执行的源码和旧identity已归档；provenance.json记录种子元数据及指纹校验的修正。正式样本保持原值，未重新计时。
