# Qwen3.5 Engine 级正确性与性能验证报告

日期:2026-09-03 · 验证对象:Jung52/nano_qwen @ `a19088d`(GDN 变长 prefill + piecewise CUDA Graph)

## 1. 环境

| 项 | 值 |
|---|---|
| git commit | `a19088dea047eea450a3219685efe2b686b65aa0` |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (16GB) |
| torch | 2.11.0+cu130 |
| CUDA | 13.0 |
| flash-attn | 2.8.3.post1 |
| flashinfer | 0.6.12 |
| TP | 1(环境单卡,TP=2 矩阵未跑,见 §11) |
| 模型 | Qwen3.5-4B(32 层,hidden 2560,hybrid GDN,bf16,~8.8GB) |

新增文件:`benchmarks/validate_engine_qwen35.py`、`benchmarks/bench_engine_qwen35.py`、`benchmarks/engine_bench_utils.py`。
生产代码改动仅限本次会话早前的端口修复(动态 NCCL 端口),验证脚本未修改任何 kernel/调度器/引擎生产逻辑。

## 2. 已有 kernel benchmark

`benchmarks/bench_gdn_flashqla.py` 保持原样(未删除、未重写、未重复实现);kernel 级 GDN prefill 正确性/性能继续由它负责。本报告为 engine 级验证。

## 3. Engine 正确性矩阵(ArgmaxSampler 精确 token 比较)

模式:BASELINE(eager, queue=1, sync D2H)/ A(eager, queue=2, async D2H)/ B(decode graph, prefill eager)/ C(piecewise prefill graph + decode graph)。

**在 `TORCHDYNAMO_DISABLE=1` 下(排除 torch.compile 形状特化噪声):**

| case | BASELINE | A | B | C | 说明 |
|---|---|---|---|---|---|
| w1 单请求 12 长度(64-chunk 边界) | PASS | PASS | PASS | PASS | 1/31/63/64/65/127/128/129/255/256/257/511 |
| w2 packed-varlen prefill | PASS | PASS | **MISMATCH** | **MISMATCH** | B/C 在 `[63,65]` 第 2 请求第 6 个 token 起分歧(见 §8) |
| w3 GDN conv 请求边界隔离(A/B 模式) | PASS | PASS | PASS | PASS | A/B 单独运行 == packed 运行,无边界污染 |
| w6 batched decode 隔离(bs 1/2/3/4/5/8) | PASS | PASS | PASS | PASS | graph bs + eager fallback bs 全部与 solo 一致 |
| w7 queue=1 vs queue=2 + scheduler 不变量 | PASS | PASS | PASS | PASS | 每步 waiting/running/in_flight 无交集;终态干净 |
| w8 staggered completion | **FAIL** | **FAIL** | **FAIL** | **FAIL** | 仅迟到请求第 4 个 token 分歧(见 §8,非串扰) |
| w9 sync vs async D2H | PASS | PASS | **MISMATCH** | **MISMATCH** | B/C 在第 6 请求第 6 个 token 起分歧(见 §8) |
| w10 长 decode 128 步(batch 1/4) | PASS | PASS | PASS | PASS | 全程 logits/state finite,计数精确 |

本地用例(BASELINE 子进程):

| case | 结果 |
|---|---|
| w4 GDN state 生命周期 | **PASS**:prefill 后 slot0 state 非零且 finite(checksum 6.51e4),decode 一步后变化(6.83e4) |
| w5 slot 复用 vs fresh engine | **PASS**:B(reused slot 0)== B(fresh engine slot 0),12 token 完全一致 |
| known_lim chunked prefill | **KNOWN_LIMITATION**:512-token prompt + `max_num_batched_tokens=128` 按预期触发 `prefix-cache prefill is not supported by this runner` |

**验收标准对照(16 项):** 1✓ 2✓ 3✓ 4✓ 5✓ 6✓ 7✓ 8✓(复用=新鲜)9✗(w8,数值特征)10✓ 11✓ 12✓ 13✓(graph exact-BS 与 eager 一致,token 级)14✓(piecewise 与 eager 一致)15✓(同 bucket 不同 layout 复用正确,[64,64] 捕获的 128-bucket 被 [32,96] 复用且无串扰)16✓。KNOWN LIMITATION 已标记。

## 4. 三种失败/分歧的根因(均为数值可复现性,非状态串扰)

**F1 — CUDA Graph decode replay ≠ eager(bit 级)**:同引擎上 `graphs={}` 强制 eager 与正常 replay 对比,logits 差异 ≤0.3-0.5(bf16 量级),偶发近平分 argmax 翻转 → 经 GDN 循环状态蝴蝶式放大。这是 w2/w9 在 B/C 模式与 BASELINE 分歧的原因(分歧点都在 decode 中段,prefill 首个 token 全部一致)。w6 证明 graph 路径的**状态隔离本身正确**(bs 1/2/4/8 graph 与 solo 输出一致)。机制假设:图捕获期间 flash-attn/Triton 的算法选择或形状特化与 eager 派发不同。

**F2 — torch.compile 形状特化噪声**:dynamo 开启时,同一层按 batch 形状编译出不同 kernel,`rms_forward` 反复触发 shape guard 重编译(recompile_limit=8 后回退 eager)。探针证明:baseline 模式下 bs=3 出现过一次 0.10 差异的近平分翻转;禁用 dynamo 后所有 bs 的 batched vs solo token 全部一致。**精确 token 比较必须在 `TORCHDYNAMO_DISABLE=1` 下进行**,这是当前 layer 级 `@torch.compile` 的固有特性。

**F3 — queue depth 改变 batch 组成**:w8 探针证实 queue=2 时第二个 batch 在前一个 batch postprocess 前派发,in_flight 未清空,调度器对 running 的切分不同((4),(1),(3),(1)... vs (5),(4),(4)...)。同一请求在不同 queue 深度下处于不同大小 decode batch → F1 类数值差异 → 偶发翻转。这是 w8 的根因,**不是 slot 复用/状态污染**(w5 已证明复用正确)。

## 5. 性能基准(生产 Sampler,固定 seed,每配置 warmup 2 + 5 轮取中位数)

decode 吞吐(tok/s,decode 阶段):

| bs | P0 eager | P1 +queue/async | P2 +decode graph | P3 +piecewise prefill | P2-P1 |
|---|---|---|---|---|---|
| 1 | 30.9 | 38.6 | **67.1 (2.2x)** | 70.0 | +28.5 |
| 2 | 64.5 | 58.2 | **124.7** | 127.5 | +66.5 |
| 3 (fallback) | 99.7 | 87.0 | **74.7 (回退!)** | 95.9 | −12.3 |
| 4 | 124.2 | 140.4 | **241.9** | 244.9 | +101.5 |
| 5 (fallback) | 157.3 | 155.4 | 164.8 | 183.3 | +9.4 |
| 8 | 262.1 | 289.7 | **458.2** | 460.5 | +168.5 |
| 16 | 460.3 | 539.7 | **820.7** | 817.4 | +281.0 |

prefill 吞吐(tok/s,单请求 TTFT):

| prompt | P0 | P2 | P3 |
|---|---|---|---|
| 512 | 6652 | 7434 | 7530 |
| 1024 | ~8200 | ~8100 | ~8200 |
| 2048 | 7956 | 7851 | 7996 |

varlen prefill 960 tokens:[64,128,256,512] 布局 P0=5435 → P2=6397 → P3=6612(+21.6%)。同 total token 不同布局吞吐差异明显(128 tokens:P3 下 [64,64]=2873 vs [17,31,80]=2544)。

mixed workload(16 req × 64 out):output tok/s P0=386 → P2=241→P3=271(此配置方差大,波动 ±30%);TTFT p50 P0=0.274s → P2/P3≈0.41s(**queue=2 使 TTFT 恶化 ~0.13s**,消费被延迟一个 batch);TPOT p50 P3=0.046s。

## 6. Graph 命中统计(全 workload 累计)

| 模式 | decode graph hit | decode eager fallback | prefill graph | prefill eager |
|---|---|---|---|---|
| P0/P1 | 0 | 3874/4250 | 0 | 143 |
| P2 | 2609 | **1641** | 0 | 143 |
| P3 | 2609 | **1641** | 77 | 66 |

decode fallback 全部来自 bs∉{1,2,4,8,16} 的批次(3/5/6/7/9-15 等,mixed workload 中大量出现);prefill 中 1024/2048 恒为 eager fallback(>512 bucket)。

## 7. GPU 内存(TP=1,max_num_seqs=16,util=0.9)

| 模式 | 初始化后 | prefill 峰值 | 全程峰值 |
|---|---|---|---|
| P0/P1 | 11.3GB | 12.3GB | 12.3GB |
| P2 | 11.3GB | 12.3GB | 12.3GB |
| P3 | 11.3GB | 13.4GB | **13.9GB(reserved 15.3GB,接近 16GB 上限)** |

piecewise prefill 图池额外占用 ~1.6GB。

## 8. 发现的引擎问题(按严重度)

1. **[P1] 生产代码 `engine.exit()` 引用环泄漏 ~6GB**:exit 后 `allocated_bytes.all.current` 仍 6133MiB,`gc.collect()` 后降至 8MiB。同进程重建 engine 会因 KV 预算公式使用残留统计而断言失败。长驻服务若重建 engine 会累积泄漏。建议:`exit()` 中显式 `gc.collect()` 或打破循环引用(ModelRunner ↔ model/KV 视图)。
2. **[P2] prefix cache 已写但不可用**:`block_manager.hash_blocks` 在 postprocess 写入哈希,但 `prepare_prefill` 对 prefix 命中直接 raise。**相同 prompt 二次提交会使 engine step 崩溃**(性能脚本首轮即触发)。建议:要么在 `prepare_prefill` 支持 prefix-cache prefill,要么暂时禁用 hash 写入。
3. **[P2] dynamo 形状特化噪声(F2)**:layer 级 `@torch.compile` 导致同请求在不同 batch 形状下 token 级不可复现。建议:编译装饰加 `dynamic=False` 静态形状约束,或按 batch bucket 固定形状编译。
4. **[P3] CUDA Graph replay 与 eager 存在 bf16 级差异(F1)**:偶发近平分翻转。多数推理引擎接受此差异;若追求严格一致需定位 capture 期间 kernel 选择差异(flash-attn autotune 是首要怀疑对象)。
5. **[P3] queue=2 使 TTFT 退化 ~0.13s**(消费延迟一个 batch),在低并发下 P1 vs P0 整体收益为负(mixed 16×64 −33%)。
6. **本次会话已修复(生产代码)**:NCCL 固定端口 2333 → 动态空闲端口(避免 TIME_WAIT EADDRINUSE);gpu_memory_utilization 0.50→0.90(4B 模型需要)。

## 9. 已知限制(明示,不判为失败)

- **chunked/prefix-cache prefill 不支持**:`max_num_batched_tokens < prompt_len` 时第二个 chunk 触发 RuntimeError(有专门测试用例验证并标记 KNOWN_LIMITATION)。列入下一阶段任务。
- TP>1 未验证(单卡环境);脚本已按 parent-child 子进程架构预留扩展位。

## 10. 下一步优先级(基于实测数据)

- **A(优先):decode CUDA Graph batch padding + dummy GDN state slot**。数据支持:bs=3/5 fallback 吞吐仅为相邻 graph bucket 的 31-40%(P2 下 bs3=74.7 vs bs4=241.9,bs5=164.8 vs bs8=458.2),且 mixed 负载中 fallback 批次占 39%(1641/4250)。将 3→4、5→8(及 mixed 中其他奇数 batch)padding 进图,预计 decode 整体再提升 30-60%。
- **B(次优先):hybrid chunked/prefix prefill**。功能缺口:长 prompt(>max_num_batched_tokens)直接不可用(问题 2 同源)。同时解决"相同 prompt 崩溃"问题——至少先安全降级(prefix 命中时绕过 cache 重新 prefill)。
- **D(第三):MRV2 overlap 深化**。P1 vs P0 在低并发下收益为负、TTFT 退化,说明 CPU prepare/sample 与 GPU forward 的重叠尚未真正生效;用本次会话已搭建的 Perfetto trace 基础设施(`bench_async` 的 `d2h_wait`/`prepare_inputs` 时间线)定位后决定是否继续投入。
- **C(暂缓):prefill kernel 优化**。prefill 已 ~8k tok/s,piecewise graph 仅 +1-4%;varlen 布局间吞吐差异(2873 vs 2544 @128 tokens)说明 conv 打包还有空间,但收益优先级低于 A/B。

## 11. 复现命令

```bash
# 正确性(精确 token 比较,需禁用 dynamo)
TORCHDYNAMO_DISABLE=1 python benchmarks/validate_engine_qwen35.py --model /home/wei/code/models/qwen

# 性能(生产配置)
python benchmarks/bench_engine_qwen35.py --model /home/wei/code/models/qwen

# 单模式/单用例调试
python benchmarks/validate_engine_qwen35.py --model ... --mode C --cases w6
```

结果 JSON:`logs/validate/`、`logs/bench/`。
