# main独立greedy fast path（2026-10-18）

本轮在main80ad401上仅移植采样器、CPU温度路由及验证采样器接口，未引入MTP代码、权重、KV或Graph。全greedy直接argmax；混合批次<=1e-6行也确定性argmax，其余温度保留随机算法。全greedy不再消耗RNG，后续随机请求的全局RNG偏移会变化。main原来在并列最高logits时随机选择，确定性最小argmax是有意修正的行为。

4070TiSUPER，4B，两种FP8，B2，same64及中文45/代码43，输出128，prefill eager/decode Graph。每case同engine reference1/fast1/reference2/fast2，各配置正式2次，预先4-token短预热，profiler关闭。reference保留确定性greedy tie规则，但执行完整softmax/RNG；它是控制fast分支的完整采样参考，并非声称未修改80ad401的随机tie行为与新实现相同。

16个正式样本full completion严格相同，每次计254实际提交decode tokens；源码SHA、prompt、完整tokens、顺序和逐次延迟见[JSON](results.json)，逐次数据见[CSV](samples.csv)。

| FP8 | 输入 | 完整采样 tok/s | fast tok/s | 平均收益 | fast TPOT ms |
|---|---|---:|---:|---:|---:|
| channel | same64 | 161.08 | 161.89 | +0.50% | 12.355 |
| channel | chinese-code | 172.15 | 174.78 | +1.53% | 11.444 |
| block | same64 | 176.47 | 180.54 | +2.31% | 11.078 |
| block | chinese-code | 184.99 | 185.17 | +0.10% | 10.801 |

这些小幅差异仅是两次测量，不能据此宣称稳定提升；此前mtp2上的36%～39%是另一条runner/输出路径的同engine消融，不能外推成main的固定收益。main保留独立实现与测试，使确定性greedy和省略无用采样不依赖MTP。

验证：干净main基线98通过；4项新回归先RED，移植后完整102通过，覆盖混合温度并列规则、跳过softmax/RNG/设备标量读取、decode/prefill尾行与slot映射、CPU温度路由不推进RNG。未改变非greedy采样分布算法。生产路径无需额外开关：由CPU请求温度自动选择。

复现：`python benchmarks/bench_greedy_sampling.py --model FP8_DIR --output logs/greedy-main.json`。MTP在独立feat/qwen35-mtp2分支保持enable_mtp可选；功能支持FP8 TP1 greedy B1–8，性能适用范围以该分支各GPU/模型/工作负载的实测为准，不能把功能支持范围当加速范围。
