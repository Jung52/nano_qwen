# MTP cache-only（2026-10-15）

一层MTP接受draft后的第一行仅补KV：embedding/pre-norm→fc→input norm→K/V投影→K norm/key RoPE→store。跳过Q、attention、out projection、MLP、final norm及LM head。第二行仍完整前向产生下一draft；串行B1 target verify不变，未优化snapshot或同步，保留TP1/B1 guard。

独立cache-only CUDA Graph使用独立输入buffer，生命周期沿用逻辑位置p映射物理行p−1。14项新增真实CUDA测试覆盖eager/graph、非连续block及位置1/128/129/256/257/512/513：整份cache（包含未写行）与完整MTP逐位相同，后续完整前向的draft相同。eager禁止调用Q/attention/out/MLP/final norm/LM head；graph范围经独立代码审查。修复前14fail，修复后14pass；完整回归275pass（基线261）。

真实4B两类FP8的B1 eager16-token gate通过；下列正式graph对照每配置两次，完整128个completion token都与MTP关闭逐位相同，计时只含prefill首token之后实际提交的127个decode token。输入64/输出128，TP1、greedy、ignore EOS，prefill均eager，profiler关闭。使用原production run_mode计数；同一MTP engine按完整补写/cache-only/cache-only/完整补写执行，真实结束及slot重置，逐次清零stats；off engine另做两次。初始化/capture/prefill不计入decode，未增加正式样本。

| FP8 | MTP关闭 tok/s | 完整补写 tok/s | cache-only tok/s | 相对完整补写 | 相对MTP关闭 |
|---|---:|---:|---:|---:|---:|
| per_channel | 79.74 | 63.10 | 68.79 | +9.0% | 0.863× |
| block_wise | 82.87 | 61.21 | 67.39 | +10.1% | 0.813× |

表中是两次tok/s算术均值，原值见[CSV](timing_samples.csv)和[JSON](results.json)。cache-only降低了补写成本，仍未超过同次关闭MTP的吞吐。接受率/verify次数在两种补写方式中一致；不能把本轮与历史83tok/s直接作为同环境A/B，也不宣称模型任务质量或大batch收益。

原始日志、测试RED/GREEN、同engine ABBA编排在`logs/mtp-cache-only/`（Git忽略）。仍使用`bench_qwen3_5_mtp_production.py`的真实token计数，不用projected吞吐，不以drift放行。下一步two-token主干复用单独实验、单独门禁，不合并为cache-only收益。
