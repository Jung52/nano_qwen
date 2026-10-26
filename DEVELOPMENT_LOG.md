# nano_qwen 开发日志

记录截至 **2026-10-26，main通用推理路径**。这是开发过程的复盘：为什么加入一个模块、实现后遇到了什么、哪些实验改变了原来的判断，以及目前还没有解决什么。README 保留项目介绍和入口，本文件保留过程。

## 记录范围与读数口径

当前本地 Git 历史在 `2b3693f` 处存在 shallow boundary；现存第一份较完整的 engine 测量报告是 **2026-09-03**。日志从这份证据开始，更早没有保留下来的性能数据不补写，也不按提交标题推算收益。

正文区分三类信息：**实测**来自保留的报告、JSON/CSV；**源码事实**来自具体提交或当前实现；**推测/待验证**只用于解释下一步要查什么。一个功能被提交，不等于其性能已经验证。早期报告里的解释也保留其历史身份，不能自动成为现在的正确性结论。

主要本机数据使用 Qwen3.5-4B、RTX 4070 Ti SUPER 16 GiB、TP=1。9月3日报告的 PyTorch 为2.11.0+cu130，10月记录为2.12.1+cu130；模型、源码、调度、Graph、KV预算和测量口径也有变化。**不同章节的绝对吞吐不是一条控制变量的版本增长曲线。**

- `M` 是本次 Linear 的实际输入行数。完整 prefill 通常为 `batch×input_length`；chunked prefill 是当前实际派发的 token 总数；普通 decode 为活跃 batch。
- 输出128包含 prefill 产生的首 token，随后是127步 decode。TPOT=`decode时间/127`，batch总吞吐=`batch×127/decode秒`；TPOT不再除batch。
- 端到端输出吞吐、纯decode吞吐、单算子μs、profiler累计kernel时间分别记录，不能相互替代。吞吐提高比例和耗时降低比例也分别计算。
- 9月3日主要性能表是预热2次、正式5次取中位数；10月9日格式对照是3次取中位数，另外两组端到端对照是2次。10月14日以后的专项正式实验通常每配置2次；波动大的样本保留，不补第三次挑结果。
- 编译、autotune、capture和预热通常不计入稳态性能。短正确性gate不当性能结论；带profiler的诊断不替代关闭profiler的正式计时。

早期仅存本机`logs/`的7份小报告已整理到[历史证据目录](docs/development-records/README.md)。保留原始来源、原文件与归档文件SHA256；只统一了CRLF换行。后续数据沿用`benchmarks/reports/`——该目录自2026-10-26起是**本地实验产物**，不再随Git分发；正文中的报告路径均为本机路径，公开的优化前后指标、正确性与限制汇总见[优化结果总结](docs/experiments-summary.md)，没有重新测试或重写旧样本。

## 1. 2026-09-03：第一次把异步执行、Graph和完整engine放在一起检查

当时已经有`prepare / execute / sample`拆分、persistent InputBatch、GPU常驻token、异步D2H、双pinned buffer与Event，以及GDN批处理和piecewise prefill。第一次完整报告检查的版本是`a19088d`，同时记录了NCCL动态端口与显存预算修正。它测的是完整engine，不是只测GDN kernel。[原始报告](docs/development-records/engine-validation-2026-09-03.md)

### 1.1 异步队列：解决了数据生命周期，但没有自然获得CPU/GPU重叠收益

报告定义了四种性能模式：P0=eager/queue1/同步输出，P1=eager/queue2/异步输出，P2=再启用decode Graph，P3=再启用piecewise prefill Graph。固定batch的decode结果如下，单位为整个batch的token/s：

| Batch | P0 | P1：queue/async | P2：decode Graph | P3：再加prefill Graph |
|---:|---:|---:|---:|---:|
| 1 | 30.9 | 38.6 | 67.1 | 70.0 |
| 2 | 64.5 | 58.2 | 124.7 | 127.5 |
| 3 | 99.7 | 87.0 | 74.7 | 95.9 |
| 4 | 124.2 | 140.4 | 241.9 | 244.9 |
| 8 | 262.1 | 289.7 | 458.2 | 460.5 |
| 16 | 460.3 | 539.7 | 820.7 | 817.4 |

**新模块加入后不增反降，从这一阶段就出现了。** queue2/async在B1、B4、B8有收益，在B2、B3反而下降。原因不能简化为“异步复制不好”：队列深度还改变了请求何时消费、何时回到running、下一轮凑出多大的batch。双缓冲和Event保证的是输出可安全消费；能否与GPU计算重叠，是另一个需要时间线验证的问题。

正确性检查也暴露了同一现象：queue1与queue2可能把同一个请求放进不同大小的decode batch。早期报告将若干token分歧归为数值特征，并观察到形状特化、接近并列的logits等因素。这些是**当时的诊断**，不能据此认定此后所有分歧都是数值误差，更不能替代完整的缓存生命周期与状态检查。

### 1.2 CUDA Graph：固定shape收益大，混合负载却明显回退

当时decode只捕获B=1/2/4/8/16。表中B3没有图，P2甚至比P1更慢；打开Graph开关不代表每个forward都在replay。全workload统计中，P2/P3只有2609次decode Graph命中，另有1641次eager fallback，约占39%。prefill也只有77次Graph、66次eager，1024/2048 token超出了当时512的bucket上限。

更明显的反例来自mixed workload（16个请求、每请求64输出token）：

| 模式 | 输出吞吐 token/s | 相对P0 |
|---|---:|---:|
| P0 eager | 386 | — |
| P2 decode Graph | 241 | 约−37.6% |
| P3 再加piecewise prefill | 271 | 约−29.8% |

当时该负载波动约±30%，所以降幅只能作近似观察；TTFT p50从约0.274s增至约0.41s，也是报告记录的回退。固定batch里成功的Graph路径，在动态到达、batch变化和队列消费共同作用下，没有直接转化为服务吞吐收益。

这里要保留一个容易混淆的历史细节：**9月3日报告的mixed workload是混合请求负载，不是后来“一个forward同时打包prefill和decode”的mixed batch。** `a19088d`的scheduler在选到prefill后就返回；真正同forward混排是在9月14日引入。不能拿这组数字声称已经测出了9月14日混排功能的因果开销。[当时scheduler源码](https://github.com/Jung52/nano_qwen/blob/a19088dea047eea450a3219685efe2b686b65aa0/src/nano_qwen/scheduler/scheduler.py)

### 1.3 Piecewise prefill与GDN：batch化之后，还要支付图边界和显存成本

单请求prefill在512 tokens时，P2→P3约7434→7530 token/s；1024/2048没有显示同量级收益。变长布局`[64,128,256,512]`共960 tokens，P0→P3为5435→6612 token/s。但同为128 tokens，P3的`[64,64]`为2873 token/s，`[17,31,80]`为2544 token/s。总token数一致，并不能消掉GDN按请求分段、卷积打包和padding布局的差异。

piecewise图池还增加了约1.6GB驻留内存，P3全程峰值约13.9GB、reserved约15.3GB，已经接近16GB显卡上限。这时“多捕获几个shape”同时是性能选择和容量选择。

这份报告改变了早期优先级：先修状态/资源生命周期和Graph命中问题，再决定是否继续增加prefill kernel复杂度；不能只盯单个成功的固定batch。

## 2. 2026-09-04至09-10：先把退出、KV与验证基础补齐

9月3日还记录了两个比微小加速更直接的问题：`exit()`后约6133MiB仍被引用，`gc.collect()`后才降到约8MiB；prefix hash已经写入，但命中后的prefill尚不支持，相同prompt再次提交会失败。两者会污染同进程重复建engine和重复workload测试。

[`70db14a`](https://github.com/Jung52/nano_qwen/commit/70db14a4e63580603638a5be3f27c2d050b8b1fd)处理退出清理、关闭未完成的prefix复用路径，并调整编译形状特化；[`b50ac45`](https://github.com/Jung52/nano_qwen/commit/b50ac456ab0c2e1e809154d6d7f18d34a6d05e3c)继续加强runner清理和engine验证。9月10日扩展大batch profiling，9月14日又调整trace的CPU/GPU轨道显示。

现存记录不足以为这些提交分别填一项吞吐提升，所以这里只记修复内容。它们让后续基准能更可靠地跑完、隔离OOM与资源残留，但不应包装成已经量化的性能优化。

prefix cache的经历也说明：Qwen3.5不仅有全注意力KV，还有GDN recurrent/conv state。仅有KV block的prefix hash，不足以恢复完整模型状态。当前`enable_prefix_cache`仍默认关闭；源码允许opt-in不等于已完成混合模型prefix状态复用的验证。[当前配置](src/nano_qwen/config/config.py)

## 3. 2026-09-14至09-23：chunked prefill、mixed batch与Graph路由逐步相遇

### 3.1 从“先prefill后decode”变成同forward混排

[`0fa22d7`](https://github.com/Jung52/nano_qwen/commit/0fa22d75ca3d0fd7477e003595ed938e50aaa254)加入chunked prefill和mixed调度：选完prefill后，用剩余token预算继续放入decode请求；同时准备累计KV长度、paged读取及GDN延续状态。

功能上的收益是长prompt可以分块，已有请求可以在prefill间隙继续推进。但decode row一旦混入prefill，就不再天然享有普通纯decode的一次完整Graph replay。GDN还必须区分“正在decode的一行”与“prefill最后一块恰好只有一个token”：两者长度都为1，却可能需要不同状态更新/舍入路径。

这里没有找到完整的、同环境同workload的mixed开关前后正式计时，因此不填一个想当然的提速或回退百分比。9月3日的动态负载回退作为前面的历史证据保留，不挪作本提交的实验结果。

### 3.2 Coalesce：尝试修复异步队列拆小decode batch

[`fea78f4`](https://github.com/Jung52/nano_qwen/commit/fea78f4dd854f03ff34d6729b88b43381acbd49f)在下一轮decode调度前先消费已有decode结果，使ready请求有机会重新合成较大batch，并记录coalesce次数、合批前后batch分布。它也补了in-flight与ready集合不能重叠、同一请求不能重复派发，以及关闭prefix cache时仍必须检查KV准入等约束。

这解决的是“队列中有请求，但scheduler暂时看不到完整ready集合”的结构问题；它也可能增加CPU等待。现存提交说明能证明实现意图和约束，不能证明所有负载吞吐提高。BF16 GDN state仍为默认，FP32 state作为debug选项，而不是为了过测试直接增加默认状态流量。

### 3.3 Graph管理拆分、粗bucket与结构化batch描述

[`7b78e98`](https://github.com/Jung52/nano_qwen/commit/7b78e989c59ac4c14969cb7fc51650f9b582fac7)曾显式用`pure_prefill`限制piecewise路由，避免把mixed误当普通prefill；[`4ccf696`](https://github.com/Jung52/nano_qwen/commit/4ccf6968f0d112093430c2e1a97967b54b4a3a5d)把捕获、buffer和replay拆到独立`CudaGraphManager`，引入按token总数的粗bucket、共享buffer及`torch.compile`；[`5cf92bf`](https://github.com/Jung52/nano_qwen/commit/5cf92bf4bf7e3a697d5154e88dc131027ca723e3)再加入`BatchDescriptor`，为普通decode、prefill和spec decode提供结构化描述。

这些改动让功能继续扩展，但**结构更完整不等于Graph覆盖率或性能已经提升**。当前实现仍有几条要分开看的路径：

- 纯decode按精确batch查图，未捕获的batch走eager；不能仅从“已启用CUDA Graph”推断没有fallback。
- prefill/mixed可走按总token数选bucket的piecewise路径：每层dense pre/post重放，attention/GDN core留在外部执行，中间有清零、拷贝和切片。它与整个decode模型一次replay的开销不同。
- FP8 mixed的GDN用`prefill_decode_rows`区分实际decode row与prefill row。当前mixed分支仍有按请求处理、dense state取出/写回；10月15日优化的pool原地更新覆盖的是**纯decode**，不能声称mixed也获得相同收益。
- bucket太粗会多做padding，bucket太多增加capture、编译及驻留buffer成本；不同请求布局还会改变core工作量。

以上是[当前Graph实现](src/nano_qwen/engine/cuda_graph.py)、[输入描述](src/nano_qwen/engine/model_runner.py)和[GDN分支](src/nano_qwen/layers/gated_delta_net.py)可以确认的代码事实。更新后的mixed吞吐仍需要专门对照，不能用“已有bucket实现”认定9月3日观察到的动态负载问题已经消失。

## 4. 2026-10-05至10-09：FP8首先是量化路径与正确性问题

### 4.1 加入W8A8，权重减少不代表整个engine立刻变快

[`a321fa3`](https://github.com/Jung52/nano_qwen/commit/a321fa3f9701762785eda58ef7706420a4b2d817)加入量化配置、离线转换、checkpoint loader、Linear量化方法及per-channel FP8：权重为per-output-channel，激活实际为per-token。最初激活量化是Triton，GEMM却是`torch._scaled_mm`；不能因为`fp8.py`里有Triton kernel，就说整条GEMM已与block-wise同后端。

embedding、LM head、norm、GDN state/卷积/门控及采样没有全部变成FP8。一次Linear还多了动态量化和scale处理。这些额外工作在小batch、eager下可能抵消矩阵计算收益。

10月9日的正确性smoke确实保留了一个反例：128输入、32输出、两边都是eager时，B1的BF16/FP8 decode为21.26/19.40 token/s，B4为123.83/68.01 token/s。这里FP8为HF block checkpoint；这些只是诊断脚本中的短计时，不是配对性能实验，不能用于给整个FP8方案定性，但足以说明“加入量化就应该更快”并不成立。BF16原记录（本地产物 `benchmarks/reports/precision_bf16_eager.json`）、FP8原记录（本地产物 `benchmarks/reports/precision_fp8_eager.json`）

### 4.2 为保持路径一致，部分更激进的编译/派发被收回

[`42c20f3`](https://github.com/Jung52/nano_qwen/commit/42c20f3546ed0c46a459223acf8dd6e3ce079abb)修复FP8推理一致性：piecewise的FP8 dense段保留原始操作进行Graph capture，而不让Inductor融合改变BF16归一化/转换后进入FP8量化的bin；attention固定相应decode split策略；GDN加入稳定门控投影、chunk状态舍入和mixed decode row的专用路径。

这类修复可能保留额外kernel、逐行工作或拷贝。它们是正确性取舍，不是已经证明免费的开关。后续若改回更快的融合或auto split，需要重新对齐输入、kernel算术和状态，不能只比较最终一段文本是否看起来正常。

### 4.3 Block-wise支持与舍入修复

[`45265de`](https://github.com/Jung52/nano_qwen/commit/45265de37f94fb8d74838bb2eb8412525090d12bb)加入HF block-128识别与block-scaled Triton GEMM，随后`fcaebf0`补离线128×128权重转换。block-wise的激活是每token、每128通道一组，与per-channel的per-token激活不同。

10月9日的本地诊断发现：FP32到E4M3的转换经由FP16中间值，会在FP8中点附近丢失信息；仅调整除法精度并未降低原始mismatch数量。改为直接FP32→FP8 RNE转换后，6组真实checkpoint矩阵的最大relative L2约0.000761，两条48-token生成轨迹在该FP8方案内eager/graph完全一致。该轮全套57项通过。舍入修复记录（本地产物 `benchmarks/reports/precision_check_summary.json`）

这证明的是实现与显式反量化参考的误差受控、所测路径一致；3段短文本、24个logits位置的检查不等于数据集质量或perplexity评估。

## 5. 2026-10-09：HF block、原生per-channel与自转block的第一次对照

### 5.1 当时的两种FP8比较，混合了粒度和GEMM后端

本机比较了HF block checkpoint与原生per-channel checkpoint。相同4B模型、TP1，decode Graph，128次decode、总输出129；每workload预热1次、正式3次取中位数，prefill为whole-batch。下表两个shape说明排序并不固定：

| Batch / input | per-channel TTFT ms | HF block TTFT ms | per-channel decode tok/s | HF block decode tok/s |
|---|---:|---:|---:|---:|
| 1 / 128 | 48.00 | 97.68 | 87.21 | 92.08 |
| 8 / 512 | 762.67 | 642.45 | 586.46 | 639.88 |

当时per-channel用`torch._scaled_mm`，block用Triton；激活和权重scale粒度也不同。它能说明“这两个实现的实际表现”，不能将差值全归因于block更精确或scale访存更多。CSV（`docs/development-records/fp8-format-comparison-2026-10-09.csv`，已随 031e88c 删除，可从 `git show 031e88c^:docs/development-records/fp8-format-comparison-2026-10-09.csv` 恢复）、协议与环境（`docs/development-records/fp8-format-comparison-2026-10-09.summary.json`，已随 031e88c 删除，可从 `git show 031e88c^:docs/development-records/fp8-format-comparison-2026-10-09.summary.json` 恢复）

### 5.2 三模型端到端：decode受益，prefill仍可能恶化

同日另一轮改为1024-token预算的chunked eager prefill、decode Graph，每case2次；不能与上表whole-batch/3次中位数混用。B8、输入512、输出129的结果为：

| 方案 | TTFT ms | 端到端总耗时 ms | 端到端输出 tok/s |
|---|---:|---:|---:|
| BF16 | 460.99 | 2945.36 | 350.38 |
| HF block FP8 | 566.32 | 2490.69 | 414.34 |
| 原生per-channel FP8 | 657.66 | 2584.13 | 399.36 |

两种FP8总耗时更低，但TTFT更高。到B8、输入4096时，原生per-channel端到端输出吞吐为127.89 token/s，低于BF16的139.72；HF block为147.68。不能把一次“FP8更快”推广到所有prompt长度和阶段。CSV（`docs/development-records/fp8-e2e-three-models-2026-10-09.csv`，已随 031e88c 删除，可从 `git show 031e88c^:docs/development-records/fp8-e2e-three-models-2026-10-09.csv` 恢复）、协议与限制（`docs/development-records/fp8-e2e-three-models-2026-10-09.summary.json`，已随 031e88c 删除，可从 `git show 031e88c^:docs/development-records/fp8-e2e-three-models-2026-10-09.summary.json` 恢复）

### 5.3 自己转换block checkpoint，没有额外性能收益

HF原始block与自转block都走同一block-128后端，GPU scale都为FP32。B1/4/8×输入512/1024/2048/4096、输出129，每项2次端到端对照：自转block在12个case中3个更快、9个更慢；按相同workload总耗时合并，输出吞吐约低1.07%。

这轮主要完成checkpoint格式与转换链路验证，没有产生一个更快的GEMM实现。原始HF scale与自转scale的磁盘dtype不同，也不意味着运行时是不同dtype；小差值还包含顺序运行波动，不能反推出哪种磁盘格式更快。CSV（`docs/development-records/fp8-hf-vs-converted-block-2026-10-09.csv`，已随 031e88c 删除，可从 `git show 031e88c^:docs/development-records/fp8-hf-vs-converted-block-2026-10-09.csv` 恢复）、scale与协议记录（`docs/development-records/fp8-hf-vs-converted-block-2026-10-09.summary.json`，已随 031e88c 删除，可从 `git show 031e88c^:docs/development-records/fp8-hf-vs-converted-block-2026-10-09.summary.json` 恢复）

## 6. 2026-10-10至10-14：从比较两种量化，转向控制shape、后端与配置

10月10日的`1362bca`加入共享输入量化与block prefill专用kernel：attention的q/k/v、GDN的qkv/z可以共享同一份量化输入，但仍分别做GEMM；block在较大M采用多group量化与grouped GEMM。10月14日的`eb3ba3e`统一两类Triton GEMM的调优、缓存和capture处理，并在整合中保留这些已有功能。[生产与整合记录](docs/fp8-performance-2026-10-14.md)

### 6.1 先拆掉“block访存更多，所以一定更慢”的假设

受控实验固定M/N/K和Triton算法，分别切换per-token/per-group激活、per-channel/per-block权重，另外用完全相同的量化数据比较Torch与Triton。MLP `gate_up`在M512时：同一数据的Torch GEMM约956.3μs，共同Triton约643.5μs；固定Triton后四种粒度为643.5–654.4μs，跨度约1.7%。这说明原比较中后端是明确因素，两个样本不足以给接近的粒度结果稳定排序。粒度实验（本地产物 `benchmarks/reports/fp8_granularity/results.md`）

scale的唯一数据量也推翻了一个直觉：block权重scale不一定更多。以`gate_up`的N=18432、K=2560为例：

- per-channel权重scale为18432个FP32，73728B；128×128 block scale为144×20个FP32，11520B，反而更少。
- M8激活scale从32B增加到640B，但权重scale减少得更多，整个Linear逻辑流量账本约减少60992B。
- M512激活scale从2048B增到40960B，整体逻辑账本仅约增加15616B，即0.02188%；`down`在M512的逻辑增量约0.63485%。

账本按输入、量化结果、输出和scale各读写一次计算，不包含跨tile重复加载、L2命中和编译器调度。NCU计数器受`ERR_NVGPUCTRPERM`限制，所以这里没有实测DRAM流量，不能把这些字节直接除带宽当成耗时。

### 6.2 统一Triton和autotune，主要改善了特定GEMM shape

生产两条GEMM都使用有限的六组tile/warp/stage候选，按shape、device和dtype缓存；正式计时与搜索分开，Graph capture中不临时启动搜索。per-channel在完整K归约后缩放，block仍逐K块应用scale，量化语义没有为了共用框架而改成一样。

调优后微基准中，M512的`gate_up`约941.2→310.8μs，`down`约492.5→184.6μs，分别观察到约3.03×/2.67×的GEMM比值。部分样本波动超过5%，而且这只是两种MLP投影，不能写成整个模型提速3倍。调优记录（本地产物 `benchmarks/reports/fp8_granularity/triton_autotuned.md`）

### 6.3 Prefill真正差在哪里：GDN投影比“多切几个块”更具体

固定B8、每请求64输入，prefill M512，真实engine稳态Graph计时PC/BW平均59.42/75.48ms；两个PC样本波动超过5%，比例作观察。独立profile中，BW比PC的GEMM累计多11.485ms、激活量化多1.514ms，而GDN核心、卷积、norm和attention核心差值很小。算子分解（本地产物 `benchmarks/reports/prefill_breakdown/results.md`）

这些投影属于模型的不同部分，不能只看名称里的`in/out`：

| 模块/投影 | 作用 | 每次完整forward调用数 | BW相对PC累计差值 ms |
|---|---|---:|---:|
| GDN `in_proj_qkv` | 将hidden投影为GDN的q/k/v，后续进入因果卷积与delta-rule | 24 | +3.454 |
| GDN `out_proj` | GDN核心输出经门控norm后，投回hidden维度 | 24 | +2.607 |
| MLP `gate_up_proj` | 融合gate/up两路，再做SwiGLU | 32 | +1.896 |
| MLP `down_proj` | MLP中间维度回到hidden维度 | 32 | +1.550 |
| 全注意力 `q_proj` / `o_proj` | gated GQA的query/gate投影及attention输出投影 | 各8 | +1.155 / +0.879 |

模型是24层GDN+8层全注意力，每层都有MLP；`gate_up`不是GDN内部的gate。这里的差值是eager投影诊断的累计kernel时间，不是可直接相加到正式Graph墙钟的延迟。

GDN `out_proj`的M/N/K=512/2560/4096，BW缓存tile为16×32时GEMM约201.1μs；固定32×128后约90.6μs，与PC接近。这个shape还有8层全注意力`o_proj`共用。因此至少这部分差距是配置选择问题，不能全部归因于block scale或更高精度。

这是当时legacy GEMM路径的配置对照；随后整合的grouped大M路径独立调优，没有把这个tile当作所有shape和kernel的通用最优配置。

### 6.4 单算子好结果，没有自动变成端到端好结果

把这个out_proj/o_proj配置带回B8、输入64、输出128的真实engine，block优化前后总耗时1800.00→1800.63ms，decode589.77→592.54 token/s；TTFT77.29→84.72ms且波动很大。共同轨迹1024个位置上，两版block logits完全相同，但**两次样本没有证明额外端到端提速**。这条失败结果保留，没有用单算子201→91μs替代它。端到端与精度回归（本地产物 `benchmarks/reports/fp8_e2e/results.md`）

### 6.5 Graph输出buffer复用：先让完整prefill跑得下

顺序层此前会各自持有完整M的投影输出。后来按projection shape/dtype/device复用prefill输出buffer，同时保留各层独立KV/GDN状态，实测支持完整M到8192。这里复用的是顺序执行的临时dense输出，不是让不同请求或层共享状态。

B4/8×输入256/512/1024×输出512/1024的完整Graph矩阵，BF16/PC/BW共72次正式测量：PC总耗时比BF16减少约29.6%–33.6%，BW约27.9%–33.3%；PC在12个case的TTFT都更低，decode中10个case略快。完整矩阵（本地产物 `benchmarks/reports/fp8_matrix/results.md`）

这更新了“PC prefill快、BW decode快”的早期印象：**在这版源码和矩阵里，BW并没有稳定的decode优势。** 后端、tile、共享量化、Graph以及具体batch/context共同决定排序。这批测量发生在当时的源码快照，后续整合保留数据，不冒充整合后全部重测。

## 7. 2026-10-14：decode容量上去了，吞吐却逐渐饱和

decode-only实验先建立真实KV/GDN状态，计时128步、context65→192；仅捕获当前batch，KV每请求1个256-token block，统一90% allocator预算。预热、prefill和收尾不计入decode。容量实验（本地产物 `benchmarks/reports/fp8_decode_capacity/results.md`）

| 方案 | 当次16对齐容量边界 | 吞吐最高的已测batch | 已测峰值 token/s |
|---|---:|---:|---:|
| BF16 | 96 | 96 | 1877.0 |
| per-channel | 256 | 256 | 2476.4 |
| block-wise | 256 | 256 | 2487.5 |

两种FP8相对BF16各自已测峰值高约31.9%/32.5%，其中包含容量不同带来的收益；同为B96时只有约20.4%/14.2%。不能把跨batch的峰值比说成同batch加速。

显存节省主要来自权重。扣除embedding/head共享storage的重复引用后，BF16约7.83GiB，PC/BW约4.52/4.51GiB；每请求GDN状态约25.125MiB、此窗口KV约8MiB，并不随权重量化缩小。B128→B256虽然请求数翻倍，PC吞吐只从2425.5升到2476.4，已经很接近平台区间。

BF16 B128一次未限额预热在WSL接近显存满载时明显变慢，正式计时前终止；是否为驱动系统内存回退没有直接证据，后续统一预算。BF16 B96两次差约5.6%，没有补测第三次。这里的“最大batch”仅属于这一短上下文和capture/KV策略，不能直接移用于完整prefill或默认捕获全部小batch的部署。

## 8. 2026-10-15：profile把大batch热点从FP8转到了GDN状态复制

先用`bench_one_batch`对两种FP8做B1/4/8/16/32/64、输入64、输出128，每配置2次，prefill/decode均Graph；另外独立采prefill和最初8步decode的trace。旧版正式结果中B64的PC/BW约2102/2079 token/s，TPOT约30.44/30.80ms。优化前报告（本地产物 `benchmarks/reports/fp8_phase_profile_before_gdn_pool_20261015/results.md`）

在小batch和prefill，FP8 GEMM仍是大项；B8 eager投影诊断里，`gate_up_proj`占FP8 GEMM约42%。但这是“FP8 GEMM内部占比”，不是整模型占比。到B64，GDN recurrent-state gather/scatter占GPU累计时间约33%–34%，比全部FP8 GEMM的约26%–27%还高。这时继续只调FP8 tile，并没有覆盖最大的共享开销。

旧路径每层先`index_select`取出请求状态，核心更新后再`index_copy_`写回pool。BF16 recurrent state每请求每层`32×128×128×2B=1MiB`；24层、B64共1536MiB。gather读写、scatter读写合计约`4×1536MiB=6GiB/步`额外逻辑流量，还不含核心更新本身和conv state。这比小规模scale元数据更能解释大batch的共同瓶颈，仍不能写成实测DRAM流量。

### 8.1 改为按slot直接更新pool，保留原后端和BF16算术

`80ad401`中的优化给FlashInfer传完整recurrent pool和GPU slot索引，让kernel原地读写对应请求；仅保留较小的conv-state取出/写回，mixed路径仍保留dense调用。小batch ABI要求的索引转为int32，Graph预热/捕获改用互异合法slot，避免多个请求别名到同一状态。

两份受控源码快照只改变GDN与Graph相关文件；旧/新路径各B8/64、各2次正式计时，完整预热，交替先后顺序：

| 方案 / B | 旧TPOT ms | 新TPOT ms | 旧/新 token/s | 吞吐变化 |
|---|---:|---:|---:|---:|
| PC / 8 | 13.308 | 12.069 | 601.2 / 662.9 | +10.3% |
| BW / 8 | 13.536 | 12.146 | 591.0 / 658.6 | +11.4% |
| PC / 64 | 30.205 | 20.557 | 2118.9 / 3114.2 | +47.0% |
| BW / 64 | 31.269 | 20.257 | 2048.2 / 3159.5 | +54.3% |

独立B64 profile每个8步窗口中，旧版192次recurrent gather、192次scatter，新版两项都为0；旧版这部分约9.7–10.9ms/步，核心时间仍约4.4–5.0ms/步。这次有“操作消失”与正式吞吐共同支持归因，不只是整模型一次变快。受控A/B与trace证据（本地产物 `benchmarks/reports/gdn_state_pool/results.md`）

实际CUDA回归覆盖稀疏乱序slot、slot0、slot大于当前batch、多次Graph replay改变索引，输出与完整BF16状态池对dense参考逐位一致。优化没有换成低精度state，也没有删除请求隔离；该阶段98项通过是当时的回归数量，不是全模型质量分数。

### 8.2 完整prefill矩阵再次说明：最大batch不等于最快batch

之后扩到B32/64/96/128/192/256、输入64、输出128。B96只新增2次，其他batch沿用原样本；均完整prefill与decode Graph、固定每slot两个KV block、90% allocator预算。

| B | PC TPOT ms | BW TPOT ms | PC token/s | BW token/s |
|---:|---:|---:|---:|---:|
| 32 | 15.56 | 16.05 | 2057.13 | 1993.38 |
| 64 | 20.39 | 20.44 | 3138.47 | 3130.85 |
| 96 | 39.42 | 36.76 | 2435.35 | 2611.70 |
| 128/192/256 | 未完成 | 未完成 | 不生成数字 | 不生成数字 |

这版完整流程中B64是已测吞吐最高点；B96能完成但更慢，并且部分样本device-used达到显卡容量。没有分页/DRAM计数证据，不能断言具体驱动回退机制。现阶段矩阵与曲线（本地产物 `benchmarks/reports/fp8_phase_profile/results.md`）

B128在KV预算guard处失败，B192/256在初始化warmup的LM head或GDN chunk状态处OOM。一个共同问题是prefill仍对所有token做完整词表投影：`[B×64,248320]`的BF16 logits，仅B128/192/256就约需3.79/5.68/7.58GiB，还没加权重、GDN、KV和图内临时量。

这与前一天decode-only能到B256并不矛盾：那次不计时完整prefill、只捕获当前batch、每请求1个KV block；这一轮有完整prefill且KV策略不同。容量失败配置保留0/2、错误阶段和日志，不按“每项2次”机械重复OOM，也不补造两个耗时。

## 9. 2026-10-18：将通用改进独立保留在 main

确定性greedy fast path已独立保留：CPU请求温度选择全greedy路径，直接argmax；混合温度仍执行原采样语义。基线和原始数据见独立控制实验（本地产物 `benchmarks/reports/greedy_main/results.md`）。

随后分别提交四项通用改进：

| 提交 | 改动 | 当次main完整回归 |
| --- | --- | --- |
| a91c95a | 序列化保存完整worker执行元数据；对未验证TP及hybrid prefix cache明确限制 | 107通过 |
| cfda31c | 保留需要的hidden，只对最终采样行投影；非最终prefill chunk不计算词表logits | 118通过 |
| 02d9e18 | 优先预留decode预算，余量分配prefill；保留partial prompt及in-flight slot所有权 | 123通过 |
| 4c43f71 | 记录Graph/eager调用、实际batch和token直方图，保持原精确shape路由 | 124通过 |

采样行与调度回归覆盖BF16、两种FP8和eager/Graph下的真实小模型，比较输出及有效GDN/KV状态。新增Graph计数不代表动态负载命中率已经提高；未捕获shape仍回退eager，没有不安全padding。

实验性分支及完整历史已独立归档，main保留普通推理框架。下一阶段从这里单独建立DFlash2实验，不预先加入执行接口或声明收益。

## 10. 后续记录方式

继续保留模块引入后回退、单算子加速未转化为端到端收益、正确性修复牺牲性能等结果。每项新实验记录源码、模型、GPU、预热和计时口径、实际token数、原始数据、状态门禁及未验证范围；不同版本的绝对吞吐不拼接成控制变量曲线。

## 11. 2026-10-26：LMCache CPU RAM 状态交接原型（PD 交接第一步）

`e6ab0b1` 加入单请求同步的 PD 状态交接原型：P 实例完成完整 prompt 并采样首 token 后，把 attention KV、全部 GDN conv/recurrent state 和请求进度序列化为完整快照，经本机二进制 HTTP 服务提交到真实 LMCache 0.5.6 CPU L1 内存池；D 实例分配自己的 block/slot，恢复 GPU 输入后从下一步 decode 继续。原始推理路径不变；新路径要求 `async_scheduling=False`、`enable_prefix_cache=False`、`ignore_eos=True`、`max_tokens>=2`，用于固定长度正确性验证。

- 接入代码在 `src/nano_qwen/cache/`（L1 存储、HTTP 服务、connector 与 engine_ops）；引擎新增 `prefill_to_lmcache` / `resume_from_lmcache` 两个入口，实现与使用说明见 [docs/lmcache-pd.md](docs/lmcache-pd.md)。
- 传输是本项目自己的二进制 HTTP 协议，**不是**官方 LMCache MP 协议、RDMA 或共享内存零拷贝；未实现并发路由、自动 P/D 池调度、部分前缀命中；快照不自动 LRU 淘汰，满容量明确失败，D 成功恢复后 ACK 删除，遗留快照默认 900 秒后在下一次服务操作时回收。缓存服务重启会丢失待交接快照。
- 回归：`tests/test_lmcache_pd_state.py` 用 AST 加载真实的 Sequence/BlockManager/Scheduler/LLMEngine 类、以确定性 CPU 状态替代模型执行；`tests/test_lmcache_pd_store.py` 覆盖真实 L1 分配、提交/校验/删除、容量耗尽、过期回收与 HTTP 交接。`REQUIRE_LMCACHE=1` 使缺依赖直接失败，防止 skip-only 的假通过。

实测（2026-10-26，RTX 4070 Ti SUPER 16GiB，`code` 环境 PyTorch 2.12.1+cu130、LMCache 0.5.6，模型 `4b-channel-fp8`）：

| 测试 | 结果 |
|---|---|
| CPU 回归 | 30 passed |
| store-smoke | passed，backend=lmcache_l1，真实 L1 分配 |
| roundtrip eager（prompt 64/256/257/512，输出32） | 4/4 passed：state_bytes_equal、tokens_equal 全真，prefill_dispatches=0、decode_dispatches=31 |
| roundtrip CUDA Graph | 4/4 passed，同上；TPOT 约 11.6ms（eager 同机约 60.7ms，仅为路径参考，不代表交接加速） |
| 双进程交接（P 进程退出后 D 进程独立恢复） | passed：prompt 257 快照 41.1MiB，P 提交 405ms、D 恢复 218ms，输出与不中断 reference 的 32 token 完全相同 |

口径与限制：交接耗时包含同步、序列化与 HTTP 副本，只用于诊断原型，**不作为生产 PD 加速结论**。256/257 长度覆盖完整 KV 页与尾页，防止对齐边界多分配 block。两端必须同权重、同配置、同量化格式与同推理软件版本，快照不支持跨格式转换；模型身份哈希读取全部 checkpoint 权重，位于计时外。逐次 JSON 保留在本地 `logs/lmcache_pd/`。

## 12. 2026-10-26：交接开销定位与共享内存直通重构（移除 HTTP payload 路径）

### v2：开销定位（测量脚本已随后续清理移除）

同机单卡串行测量（同一常驻引擎，baseline 与交接交替，10 组配置 × 1 预热 + 5 次）：CUDA Graph 下交接额外 E2E 224–442ms、暂停 252–464ms，随 prompt 长度线性增长。阶段分解（ms，中位数，64→1024）：

| 阶段 | 耗时 | 说明 |
|---|---|---|
| serialize | 46–115 | `state.pack`：张量逐块转字节 + SHA256 |
| PUT | 79–133 | HTTP 上传 + 服务端 L1 写入 |
| GET | 52–93 | HTTP 下载 + 客户端校验 |
| deserialize | 34–80 | `state.unpack`：字节重建张量 |
| GPU 导出/恢复 | 5–18 | 真正的 D2H/H2D 拷贝与状态管理 |
| ACK | ~1 | DELETE 确认 |

约 95% 的交接开销在 CPU 序列化与 HTTP 副本，GPU 状态搬运几乎免费；恢复后 TPOT 与 baseline 持平（11.5–12.0ms）。eager 对照的 E2E 差值被 decode 抖动（TPOT 51–73ms）淹没、出现负值，仅其暂停时间与 Graph 一致，不可作结论依据。

### v3：GPU↔共享池直通（`lmcache_shm` + `tensor_ops`）

P 端把 KV/GDN 逐张量直接拷入 LMCache 0.5.6 L1 的共享 CPU tensor 池并 commit；D 端按描述符 mmap 同一池、校验后一次 H2D 散射到自己分配的 block/slot。HTTP 只承载约 3KiB 的 JSON 控制消息（reserve/commit/acquire/ack/release），payload pack/unpack 调用为 0、payload HTTP 字节为 0。租约用 pidfd 检测对端存活：活进程的映射不被 TTL 回收，进程死亡自动释放租约且不 unlink 共享池。

实测（同机同模型同 Graph 配置，prompt 64/257/1024 × 输出 128，5 次中位数）：

| Input | v2 暂停 | v3 暂停 | 提升 | v3 额外 E2E | v3 阶段分解（ms） |
|---:|---:|---:|---:|---:|---|
| 64 | 251.8 | **14.2** | ~17× | 18.1 | reserve/map 1.6 + D2H 5.0 + commit 0.7 + acquire 0.9 + H2D 4.2 + ACK 0.8 |
| 257 | 304.1 | **15.0** | ~20× | 25.4 | 同构，D2H 5.4、H2D 4.2 |
| 1024 | 448.6 | **17.3** | ~26× | 15.8 | D2H 5.5、H2D 6.5 |

关键性质：开销几乎不随 prompt 长度增长（16 倍状态量仅 +3ms），因为大张量驻留共享池按 mmap 映射，不存在随体积缩放的打包/传输。正确性：roundtrip（64/256/257/512 × 32，Graph）4/4 输出逐 token 一致；独立进程 P 退出→D 恢复通过；CPU 回归 71/71。限制不变：单请求同步、TP=1、同主机同 IPC/PID namespace；pageable 映射内存走阻塞拷贝，非纯 PCIe 带宽测试。

环境注记：conda 构建的 CPython 缺 `os.pidfd_open`（内核 syscall 本身可用，系统 Python 有），以 ctypes 包装 `SYS_pidfd_open` 的 sitecustomize shim 补齐（`/tmp/pidfd_shim`，经活/死进程语义验证），不影响仓库代码。

### 清理：移除 HTTP payload 路径

应用户要求删除 v1/v2 的字节快照路径：`cache/connector.py`、`cache/service.py`、`state.pack/unpack`、`LMCacheL1Store` 字节存储、v1/v2 benchmark 与测试及两份文档。被 SHM 复用的部分保留：`state.py` 的 Snapshot/capture/validate/restore/model_identity、`lmcache_l1.validate_key`；公共基准工具提取到 `benchmarks/pd_common.py`（Recorder 不再插桩 codec），CPU 引擎 fixture 提取到 `tests/pd_cpu_fixture.py`。`engine_ops` 只保留 direct-tensor 分支，非 `LMCacheShmConnector` 明确报错；测试与 benchmark 守卫升级为断言 `pack/unpack` 属性不存在。清理后回归 23/23（71 项中 48 项随路径移除），GPU 真实模型 roundtrip 复测通过。现存唯一交接路径：GPU → LMCache L1 共享池 → GPU。
