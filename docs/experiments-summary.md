# 优化结果总结（截至 2026-10）

本文是仓库实验结果的**独立总结**：优化了什么、为什么有效、在什么条件下测得、优化前后指标、正确性验证与已知限制。逐次原始数据与完整图表不在 Git 中分发（见[本地实验产物](#本地实验产物)），各版本、各测试配置的数据在下方分节说明，**不同版本的绝对数值不能拼接成一条控制变量曲线**。

更长的开发过程复盘见根目录 [DEVELOPMENT_LOG.md](../DEVELOPMENT_LOG.md)；FP8 生产实现与整合细节见 [fp8-performance-2026-10-14.md](fp8-performance-2026-10-14.md)；9 月与 10-09 的历史证据见 [development-records/](development-records/README.md)。

## 优化内容与原因

### 1. FP8 W8A8 量化：per-channel 与 block-wise 两条 GEMM 路径（10-05 至 10-14）

- **改了什么**：全模型 168 个 Linear 的 GEMM 支持两种 FP8 粒度——per-channel（权重 per-output-channel、激活 per-token）与 block-wise（权重 128×128 块、激活每 token 每 128 通道一组）。激活量化用 Triton、完整 K 归约后（或逐 K 块）应用 scale；两条路径共用有限候选的 tile 调优、按 shape 缓存与 CUDA Graph capture 处理。后续整合远端共享输入量化与 grouped prefill kernel。
- **为什么有效**：受控实验表明，同量化数据的 Torch GEMM 与 Triton GEMM 在 M512 `gate_up` 上为 956.3μs vs 643.5μs（约 1.49×），固定 Triton 后四种粒度仅差约 1.7%——**后端与 tile 配置是明确因素，不是粒度本身**。调优后 M512 `gate_up`/`down` 的 GEMM 约 941.2→310.8μs、492.5→184.6μs（约 3.03×/2.67×）。权重从 BF16 变 FP8 还显著降低常驻显存（见指标节）。

### 2. Prefill 投影输出 buffer 复用（10-14）

- **改了什么**：顺序执行的层按 projection shape/dtype/device 共享 prefill 输出 buffer（保留独立 KV/GDN 状态），避免每层常驻一份完整 M 的 QKV/z 输出。
- **为什么有效**：M 越大驻留显存越大；复用后单个完整 prefill bucket 实测支持到 M=8192（B8×输入1024），使大 M 的完整 graph 矩阵可测、可跑。

### 3. GDN recurrent-state 按 slot 原地更新（10-15，commit `80ad401`）

- **改了什么**：纯 decode 不再先 `index_select` 取出请求的 recurrent state、更新后再 `index_copy_` 写回 pool，而是把完整 pool 与 GPU slot 索引传给 FlashInfer，由 kernel 原地读写对应请求；仅保留较小的 conv-state 取出/写回，mixed 路径保留原 dense 调用。改动仅涉及 `gated_delta_net.py` 与 `cuda_graph.py`。
- **为什么有效**：profile 显示 B64 decode 时 recurrent gather/scatter 占 GPU 累计时间约 33%–34%，超过全部 FP8 GEMM 合计（约 26%–27%），是大 batch decode 的最大共享开销。BF16 recurrent state 每请求每层 1MiB，B64、24 层共 1536MiB；gather+scatter 合计约 4×1536MiB=6GiB/步 的额外逻辑读写。优化后这两项操作次数降为 0。

### 4. 确定性 greedy fast path（10-18）

- **改了什么**：CPU 请求温度路由选择全 greedy 路径：全 greedy batch 直接 argmax，跳过 softmax 与随机数生成；混合批次中 temperature≤1e-6 的行同样确定性 argmax，其余温度保留随机采样。并列最高 logits 时取确定性最小 argmax（相对旧行为的有意修正）。
- **为什么有效**：全 greedy 负载下省略 softmax/RNG 与设备标量读取。收益较小（见指标节），main 保留它主要是让确定性 greedy 不依赖 MTP 分支。

## 测试环境

- **硬件**：NVIDIA GeForce RTX 4070 Ti SUPER 16 GiB，driver 616.92；WSL2（kernel 6.6.87.2）；AMD Ryzen 7 9700X。
- **模型**：Qwen3.5-4B 类架构（24 层 Gated DeltaNet + 8 层全注意力，每层 MLP），checkpoint 位于 `/home/wei/code/models/`：`4b`（BF16）、`4b-channel-fp8`、`4b_block_fp8`。
- **软件**：Python 3.10；按实验分别为 PyTorch 2.11.0+cu130 / Triton 3.6.0 或 PyTorch 2.12.1+cu130 / Triton 3.7.1。
- **通用口径**：TP=1；engine queue depth 2、异步输出、关闭 prefix cache、ignore EOS；prefill/decode 均 CUDA Graph（按 M 分 bucket 的 piecewise prefill + 精确 batch decode graph）；每配置预热后正式计时 2 次，波动>5% 保留原样本、不补第三次；模型加载/JIT/autotune/capture/预热不计入结果。
- **定义**：`M` 是 Linear 实际输入行数（完整 prefill = batch×input_len，decode = batch）；输出 128 含 prefill 产生的首 token、随后 127 步 decode；TPOT=decode时间/127，吞吐=batch×127/decode秒。

## 优化前后主要指标

### GDN 原地更新（受控 A/B，B8/64，输入64/输出128）

旧/新路径每配置两次正式计时、交替顺序，仅 GDN 与 Graph 相关文件不同：

| 方案 | B | 旧 TPOT ms | 新 TPOT ms | 旧 token/s | 新 token/s | 吞吐变化 |
|---|---:|---:|---:|---:|---:|---:|
| per_channel | 8 | 13.308 | 12.069 | 601.2 | 662.9 | +10.3% |
| block_wise | 8 | 13.536 | 12.146 | 591.0 | 658.6 | +11.4% |
| per_channel | 64 | 30.205 | 20.557 | 2118.9 | 3114.2 | +47.0% |
| block_wise | 64 | 31.269 | 20.257 | 2048.2 | 3159.5 | +54.3% |

B64 独立 profile 的 8 步窗口中，旧版每窗口 192 次 recurrent gather + 192 次 scatter（约 9.7–10.9ms/步），新版两项均为 0，核心计算时间不变（约 4.4–5.0ms/步）。这是本轮收益最大的一项优化。

### FP8 相对 BF16（完整 graph 矩阵，B4/8 × 输入256/512/1024 × 输出512/1024）

三方案 72 次正式测量：per-channel 总耗时比 BF16 减少约 29.6%–33.6%，block-wise 减少约 27.9%–33.3%；per-channel 在 12 个 case 的 TTFT 均更低，decode 吞吐 10 个 case 略快。**这版源码里 block-wise 没有稳定的 decode 优势**。代表值（B8、输入512、输出1024）：总耗时 BF16 21876.22ms → per-channel 14720.44 / block-wise 15011.99ms；TTFT 586.71 → 507.27 / 546.61ms；decode 384.41 → 575.84 / 565.88 token/s。

### 大 batch decode 容量（decode-only，128 步，context 65→192）

仅捕获当前 batch、KV 每请求 1 个 256-token block、90% allocator 显存预算：

| 方案 | 16 对齐容量边界 | 吞吐最高已测 batch | 已测峰值 token/s | 相对 BF16 峰值 |
|---|---:|---:|---:|---:|
| BF16 | 96 | 96 | 1877.0 | — |
| per-channel | 256 | 256 | 2476.4 | +31.9% |
| block-wise | 256 | 256 | 2487.5 | +32.5% |

同为 batch96 时 per-channel/block-wise 相对 BF16 约 +20.4% / +14.2%。峰值比包含容量收益，不能当同 batch 加速。显存节省主要来自权重：扣除 embedding/head 共享 storage 重复引用后，BF16 约 7.83GiB，两种 FP8 约 4.52/4.51GiB；每请求 GDN 状态约 25.125MiB、窗口 KV 约 8MiB，不随权重量化缩小。

### GDN 优化后的完整 prefill 矩阵（B32–256，输入64/输出128）

| B | PC TPOT ms | BW TPOT ms | PC token/s | BW token/s |
|---:|---:|---:|---:|---:|
| 32 | 15.56 | 16.05 | 2057.13 | 1993.38 |
| 64 | 20.39 | 20.44 | 3138.47 | 3130.85 |
| 96 | 39.42 | 36.76 | 2435.35 | 2611.70 |
| 128/192/256 | 容量失败 | 容量失败 | 不生成数字 | 不生成数字 |

B64 是已测吞吐最高点；B128 在 KV budget guard 失败，B192/256 在初始化 warmup 处 OOM——完整 prefill 的 `[B×64, 248320]` BF16 logits 在 B128/192/256 约需 3.79/5.68/7.58GiB。这与 decode-only 能到 B256 不矛盾（后者不计时完整 prefill、每请求 1 个 KV block）。

### 单算子与失败结果（保留，不夸大）

- **Prefill 分解（B8/输入64/M512）**：block-wise 比 per-channel 的 GEMM 累计多约 11.485ms、激活量化多约 1.514ms，热点在 GDN in_proj_qkv/out_proj；固定 tile 后 out_proj GEMM 约 201→91μs——配置选择是明确因素。
- **端到端验证（B8/输入64/输出128）**：把该 tile 调整带回真实 engine，block 优化前后总耗时 1800.00→1800.63ms，decode 589.77→592.54 token/s，**两次样本未证实额外端到端提速**（TTFT 77.29→84.72ms 且波动大）。
- **greedy fast path（两种 FP8，B2，same64/中文代码输入，输出128）**：相对完整采样 reference 的 fast 吞吐收益 0.10%–2.31%，仅两次测量，不能宣称稳定提升。

## 正确性验证

- **FP8 量化精度**：FP32→E4M3 改为直接 RNE 转换后，6 组真实 checkpoint 矩阵的最大 relative L2 约 0.000761，两条 48-token 轨迹在 FP8 方案内 eager/graph 完全一致。
- **端到端轨迹回归**：用 BF16 生成的共同轨迹 teacher forcing，每方案比较 prefill 尾部 8 位置 + 127 轮 decode 共 1024 位置完整词表 logits：per-channel top1 一致率 0.985352、logits 相对 L2 0.122056；block-wise 0.994141 / 0.106592；block 优化前后逐位一致（top1 1.000000、L2 0）。
- **Graph buffer 复用回归**：BF16 与 block-wise 各 1024 位置与改造前原图 hidden/logits 逐位一致。
- **GDN 原地更新回归**：B8/B64 稀疏乱序 slot、slot0、slot>batch、多次 graph replay 变更索引，输出与完整 BF16 状态池对 dense 参考逐位一致；实际 GDN forward 禁止 pool 的 index_select/index_copy_。
- **greedy fast path**：16 个正式样本 full completion 严格相同；混合温度并列规则、跳过 softmax/RNG/设备标量读取、decode/prefill 尾行与 slot 映射、CPU 路由不推进 RNG 的回归先 RED 后通过。
- **回归套件**：各轮分别 94/98/102/114/124 项通过；均为该轮当时的数量，不是全模型质量分数。

## 已知限制

- **样本与波动**：正式实验每配置仅 2 次，波动>5% 保留不补测；BF16 B96 两次吞吐差异 5.62%，所有收益按近似值解读。
- **无实测带宽**：NCU DRAM/L2 计数器受 `ERR_NVGPUCTRPERM` 限制，逻辑字节账本不能换算成实际 DRAM 流量。
- **WSL2 显存回退**：接近显存满载时曾观察到异常慢的未计时预热，是否为驱动系统内存回退未直接验证；后续统一 90% allocator 预算。
- **容量口径**：容量边界只属于该短上下文、固定 KV 与仅捕获当前 batch 的协议；部署默认捕获全部小 batch 会更高。
- **路径覆盖**：GDN 原地更新只覆盖纯 decode，mixed 路径仍按请求取出/写回，不享受相同收益；chunked prefill 的 decode 统计窗口可能包含后来请求的 prefill。
- **精度结论范围**：只覆盖报告中的共同轨迹与 logits 位置，不等同数据集 perplexity 或任务准确率。
- **版本差异**：各实验的源码、torch/triton 版本、KV 预算与测量口径不同，跨版本绝对数值不可直接相减。

## 本地实验产物

2026-10-26 起 `benchmarks/reports/` 不再随 Git 分发（历史提交中的文件仍可追溯）。原始报告、图表与日志的完整备份位于本机仓库外：

- `/home/wei/code/nano_qwen_experiment_archive/2026-10-26_031e88c/`（含 `benchmarks-reports/`、`logs/`、`benchmarks-logs/` 与 `MANIFEST.md`，记录了源码 commit、硬件/软件环境与执行命令）
- 各报告对应的 JSON/CSV 逐次数据在 commit `031e88c^` 的 Git 历史中（`git show 031e88c^:benchmarks/reports/<路径>`）
- 原始 trace、源码快照与运行日志在本机 `logs/`、`benchmarks/logs/`（被 .gitignore 忽略）

本地报告目录（本机路径，供查证）：

```
benchmarks/reports/gdn_state_pool/results.md
benchmarks/reports/fp8_phase_profile/results.md
benchmarks/reports/fp8_phase_profile_before_gdn_pool_20261015/results.md
benchmarks/reports/fp8_decode_capacity/results.md
benchmarks/reports/fp8_matrix/results.md
benchmarks/reports/fp8_e2e/results.md
benchmarks/reports/prefill_breakdown/results.md
benchmarks/reports/fp8_granularity/results.md
benchmarks/reports/fp8_granularity/triton_autotuned.md
benchmarks/reports/greedy_main/results.md
```

在其他机器上拉取删除提交前，应先在原机器上备份这些目录。
