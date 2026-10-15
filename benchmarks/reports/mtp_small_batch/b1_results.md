# Small-batch MTP: B1 regression baseline

2026-10-15，RTX 4070 Ti SUPER，FP8 TP1，prefill eager / decode CUDA Graph。每配置两次正式计时，profiler关闭，短预热不计入正式样本。逐次数据及完整prompt/completion见JSON。decode排除prefill首token，只数实际提交的 output−1 个token。TTFT从入队到首token所在step返回；请求耗时不含engine构建。

## Greedy语义修复

旧off采样器在temperature=1e-6时仍随机打破并列最高分，verify使用argmax。PC输入256的第57个completion，token3158/4100同为21.125；off选4100，verify选3158。现统一<=1e-6为argmax，其余请求仍随机采样。旧失败矩阵保留在[b1_before_greedy_fix.json](b1_before_greedy_fix.json)，其吞吐不能作为通过正确性门禁的性能结果。新旧序列/接受率可能不同，不能直接跨修订比较。

## 修复后B1矩阵

| FP8 | 用例 | 输入/输出 | off tok/s | reuse tok/s | 收益 | 接受率 | 严格token |
|---|---|---:|---:|---:|---:|---:|---|
| channel | synthetic-64-128 | 64/128 | 82.58 | 83.74 | +1.4% | 90.9% | 2/2 |
| channel | synthetic-64-512 | 64/512 | 81.05 | 78.60 | -3.0% | 79.9% | 2/2 |
| channel | synthetic-256-128 | 256/128 | 81.04 | 80.63 | -0.5% | 82.6% | 2/2 |
| channel | synthetic-256-512 | 256/512 | 80.72 | 77.57 | -3.9% | 80.9% | 2/2 |
| channel | synthetic-1024-128 | 1024/128 | 79.20 | 75.67 | -4.5% | 70.3% | 2/2 |
| channel | synthetic-1024-512 | 1024/512 | 77.96 | 79.47 | +1.9% | 76.2% | 2/2 |
| channel | chinese | 45/128 | 81.57 | 80.82 | -0.9% | 77.5% | 2/2 |
| channel | code | 43/128 | 81.97 | 88.28 | +7.7% | 88.1% | 2/2 |
| block | synthetic-64-128 | 64/128 | 77.71 | 82.69 | +6.4% | 81.4% | 2/2 |
| block | synthetic-64-512 | 64/512 | 77.04 | 79.51 | +3.2% | 76.8% | 2/2 |
| block | synthetic-256-128 | 256/128 | 78.40 | 75.97 | -3.1% | 68.0% | 2/2 |
| block | synthetic-256-512 | 256/512 | 76.79 | 78.94 | +2.8% | 76.2% | 2/2 |
| block | synthetic-1024-128 | 1024/128 | 76.48 | 81.45 | +6.5% | 81.4% | 2/2 |
| block | synthetic-1024-512 | 1024/512 | 74.68 | 79.54 | +6.5% | 80.2% | 2/2 |
| block | chinese | 45/128 | 77.73 | 83.23 | +7.1% | 77.5% | 2/2 |
| block | code | 43/128 | 78.94 | 92.97 | +17.8% | 95.4% | 2/2 |

各项为两次算术均值；负收益保留。两次测量只能描述本轮结果，不能提供统计显著性。CSV还含逐次TPOT、TTFT、请求耗时与显存；显存包含预分配KV/Graph，不等于MTP净开销。当前未连接5090。

## 每case延迟

| FP8 | 用例 | 模式 | TPOT ms | TTFT ms | 请求耗时 s |
|---|---|---|---:|---:|---:|
| channel | synthetic-64-128 | off | 12.109 | 74.13 | 1.6120 |
| channel | synthetic-64-128 | reuse | 11.942 | 102.94 | 1.6196 |
| channel | synthetic-64-512 | off | 12.338 | 116.29 | 6.4210 |
| channel | synthetic-64-512 | reuse | 12.726 | 97.07 | 6.6003 |
| channel | synthetic-256-128 | off | 12.341 | 78.41 | 1.6457 |
| channel | synthetic-256-128 | reuse | 12.402 | 76.23 | 1.6513 |
| channel | synthetic-256-512 | off | 12.390 | 77.95 | 6.4091 |
| channel | synthetic-256-512 | reuse | 12.915 | 114.44 | 6.7138 |
| channel | synthetic-1024-128 | off | 12.626 | 123.25 | 1.7268 |
| channel | synthetic-1024-128 | reuse | 13.218 | 146.68 | 1.8254 |
| channel | synthetic-1024-512 | off | 12.829 | 114.26 | 6.6699 |
| channel | synthetic-1024-512 | reuse | 12.583 | 144.09 | 6.5742 |
| channel | chinese | off | 12.262 | 59.52 | 1.6168 |
| channel | chinese | reuse | 12.379 | 140.12 | 1.7122 |
| channel | code | off | 12.201 | 81.33 | 1.6309 |
| channel | code | reuse | 11.328 | 97.55 | 1.5362 |
| block | synthetic-64-128 | off | 12.869 | 58.03 | 1.6924 |
| block | synthetic-64-128 | reuse | 12.094 | 98.13 | 1.6341 |
| block | synthetic-64-512 | off | 12.981 | 140.86 | 6.7740 |
| block | synthetic-64-512 | reuse | 12.578 | 116.07 | 6.5432 |
| block | synthetic-256-128 | off | 12.756 | 84.70 | 1.7047 |
| block | synthetic-256-128 | reuse | 13.169 | 69.90 | 1.7424 |
| block | synthetic-256-512 | off | 13.024 | 107.95 | 6.7632 |
| block | synthetic-256-512 | reuse | 12.668 | 99.75 | 6.5730 |
| block | synthetic-1024-128 | off | 13.075 | 125.20 | 1.7857 |
| block | synthetic-1024-128 | reuse | 12.279 | 148.55 | 1.7080 |
| block | synthetic-1024-512 | off | 13.392 | 123.69 | 6.9669 |
| block | synthetic-1024-512 | reuse | 12.573 | 150.52 | 6.5754 |
| block | chinese | off | 12.866 | 117.87 | 1.7518 |
| block | chinese | reuse | 12.015 | 67.85 | 1.5938 |
| block | code | off | 12.668 | 60.65 | 1.6695 |
| block | code | reuse | 10.758 | 66.13 | 1.4323 |
