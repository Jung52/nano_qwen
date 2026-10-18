# nano_qwen 开发日志

记录截至 **2026-10-18，`feat/qwen35-mtp2@f443cb3`、`main@e936f58`**。这是开发过程的复盘：为什么加入一个模块、实现后遇到了什么、哪些实验改变了原来的判断，以及目前还没有解决什么。README 保留项目介绍和入口，本文件保留过程。

## 记录范围与读数口径

当前本地 Git 历史在 `2b3693f` 处存在 shallow boundary；现存第一份较完整的 engine 测量报告是 **2026-09-03**。日志从这份证据开始，更早没有保留下来的性能数据不补写，也不按提交标题推算收益。

正文区分三类信息：**实测**来自保留的报告、JSON/CSV；**源码事实**来自具体提交或当前实现；**推测/待验证**只用于解释下一步要查什么。一个功能被提交，不等于其性能已经验证。早期报告里的解释也保留其历史身份，不能自动成为现在的正确性结论。

主要本机数据使用 Qwen3.5-4B、RTX 4070 Ti SUPER 16 GiB、TP=1。9月3日报告的 PyTorch 为2.11.0+cu130，10月记录为2.12.1+cu130；模型、源码、调度、Graph、KV预算和测量口径也有变化。**不同章节的绝对吞吐不是一条控制变量的版本增长曲线。**

- `M` 是本次 Linear 的实际输入行数。完整 prefill 通常为 `batch×input_length`；chunked prefill 是当前实际派发的 token 总数；普通 decode 为活跃 batch。
- 输出128包含 prefill 产生的首 token，随后是127步 decode。TPOT=`decode时间/127`，batch总吞吐=`batch×127/decode秒`；TPOT不再除batch。
- 端到端输出吞吐、纯decode吞吐、单算子μs、profiler累计kernel时间分别记录，不能相互替代。吞吐提高比例和耗时降低比例也分别计算。
- 9月3日主要性能表是预热2次、正式5次取中位数；10月9日格式对照是3次取中位数，另外两组端到端对照是2次。10月14日以后的专项正式实验通常每配置2次；波动大的样本保留，不补第三次挑结果。
- 编译、autotune、capture和预热通常不计入稳态性能。短正确性gate不当性能结论；带profiler的诊断不替代关闭profiler的正式计时。

早期仅存本机`logs/`的7份小报告已整理到[历史证据目录](docs/development-records/README.md)。保留原始来源、原文件与归档文件SHA256；只统一了CRLF换行。后续数据沿用[现有报告目录](benchmarks/reports/README.md)，没有重新测试或重写旧样本。

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

正确性检查也暴露了同一现象：queue1与queue2可能把同一个请求放进不同大小的decode batch。早期报告将若干token分歧归为数值特征，并观察到形状特化、接近并列的logits等因素。这些是**当时的诊断**，不能据此认定此后所有分歧都是数值误差，更不能替代后来MTP的完整缓存生命周期检查。

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

10月9日的正确性smoke确实保留了一个反例：128输入、32输出、两边都是eager时，B1的BF16/FP8 decode为21.26/19.40 token/s，B4为123.83/68.01 token/s。这里FP8为HF block checkpoint；这些只是诊断脚本中的短计时，不是配对性能实验，不能用于给整个FP8方案定性，但足以说明“加入量化就应该更快”并不成立。[BF16原记录](benchmarks/reports/precision_bf16_eager.json)、[FP8原记录](benchmarks/reports/precision_fp8_eager.json)

### 4.2 为保持路径一致，部分更激进的编译/派发被收回

[`42c20f3`](https://github.com/Jung52/nano_qwen/commit/42c20f3546ed0c46a459223acf8dd6e3ce079abb)修复FP8推理一致性：piecewise的FP8 dense段保留原始操作进行Graph capture，而不让Inductor融合改变BF16归一化/转换后进入FP8量化的bin；attention固定相应decode split策略；GDN加入稳定门控投影、chunk状态舍入和mixed decode row的专用路径。

这类修复可能保留额外kernel、逐行工作或拷贝。它们是正确性取舍，不是已经证明免费的开关。后续若改回更快的融合或auto split，需要重新对齐输入、kernel算术和状态，不能只比较最终一段文本是否看起来正常。

### 4.3 Block-wise支持与舍入修复

[`45265de`](https://github.com/Jung52/nano_qwen/commit/45265de37f94fb8d74838bb2eb8412525090d12bb)加入HF block-128识别与block-scaled Triton GEMM，随后`fcaebf0`补离线128×128权重转换。block-wise的激活是每token、每128通道一组，与per-channel的per-token激活不同。

10月9日的本地诊断发现：FP32到E4M3的转换经由FP16中间值，会在FP8中点附近丢失信息；仅调整除法精度并未降低原始mismatch数量。改为直接FP32→FP8 RNE转换后，6组真实checkpoint矩阵的最大relative L2约0.000761，两条48-token生成轨迹在该FP8方案内eager/graph完全一致。该轮全套57项通过。[舍入修复记录](benchmarks/reports/precision_check_summary.json)

这证明的是实现与显式反量化参考的误差受控、所测路径一致；3段短文本、24个logits位置的检查不等于数据集质量或perplexity评估。

## 5. 2026-10-09：HF block、原生per-channel与自转block的第一次对照

### 5.1 当时的两种FP8比较，混合了粒度和GEMM后端

本机比较了HF block checkpoint与原生per-channel checkpoint。相同4B模型、TP1，decode Graph，128次decode、总输出129；每workload预热1次、正式3次取中位数，prefill为whole-batch。下表两个shape说明排序并不固定：

| Batch / input | per-channel TTFT ms | HF block TTFT ms | per-channel decode tok/s | HF block decode tok/s |
|---|---:|---:|---:|---:|
| 1 / 128 | 48.00 | 97.68 | 87.21 | 92.08 |
| 8 / 512 | 762.67 | 642.45 | 586.46 | 639.88 |

当时per-channel用`torch._scaled_mm`，block用Triton；激活和权重scale粒度也不同。它能说明“这两个实现的实际表现”，不能将差值全归因于block更精确或scale访存更多。[CSV](docs/development-records/fp8-format-comparison-2026-10-09.csv)、[协议与环境](docs/development-records/fp8-format-comparison-2026-10-09.summary.json)

### 5.2 三模型端到端：decode受益，prefill仍可能恶化

同日另一轮改为1024-token预算的chunked eager prefill、decode Graph，每case2次；不能与上表whole-batch/3次中位数混用。B8、输入512、输出129的结果为：

| 方案 | TTFT ms | 端到端总耗时 ms | 端到端输出 tok/s |
|---|---:|---:|---:|
| BF16 | 460.99 | 2945.36 | 350.38 |
| HF block FP8 | 566.32 | 2490.69 | 414.34 |
| 原生per-channel FP8 | 657.66 | 2584.13 | 399.36 |

两种FP8总耗时更低，但TTFT更高。到B8、输入4096时，原生per-channel端到端输出吞吐为127.89 token/s，低于BF16的139.72；HF block为147.68。不能把一次“FP8更快”推广到所有prompt长度和阶段。[CSV](docs/development-records/fp8-e2e-three-models-2026-10-09.csv)、[协议与限制](docs/development-records/fp8-e2e-three-models-2026-10-09.summary.json)

### 5.3 自己转换block checkpoint，没有额外性能收益

HF原始block与自转block都走同一block-128后端，GPU scale都为FP32。B1/4/8×输入512/1024/2048/4096、输出129，每项2次端到端对照：自转block在12个case中3个更快、9个更慢；按相同workload总耗时合并，输出吞吐约低1.07%。

这轮主要完成checkpoint格式与转换链路验证，没有产生一个更快的GEMM实现。原始HF scale与自转scale的磁盘dtype不同，也不意味着运行时是不同dtype；小差值还包含顺序运行波动，不能反推出哪种磁盘格式更快。[CSV](docs/development-records/fp8-hf-vs-converted-block-2026-10-09.csv)、[scale与协议记录](docs/development-records/fp8-hf-vs-converted-block-2026-10-09.summary.json)

## 6. 2026-10-10至10-14：从比较两种量化，转向控制shape、后端与配置

10月10日的`1362bca`加入共享输入量化与block prefill专用kernel：attention的q/k/v、GDN的qkv/z可以共享同一份量化输入，但仍分别做GEMM；block在较大M采用多group量化与grouped GEMM。10月14日的`eb3ba3e`统一两类Triton GEMM的调优、缓存和capture处理，并在整合中保留这些已有功能。[生产与整合记录](docs/fp8-performance-2026-10-14.md)

### 6.1 先拆掉“block访存更多，所以一定更慢”的假设

受控实验固定M/N/K和Triton算法，分别切换per-token/per-group激活、per-channel/per-block权重，另外用完全相同的量化数据比较Torch与Triton。MLP `gate_up`在M512时：同一数据的Torch GEMM约956.3μs，共同Triton约643.5μs；固定Triton后四种粒度为643.5–654.4μs，跨度约1.7%。这说明原比较中后端是明确因素，两个样本不足以给接近的粒度结果稳定排序。[粒度实验](benchmarks/reports/fp8_granularity/results.md)

scale的唯一数据量也推翻了一个直觉：block权重scale不一定更多。以`gate_up`的N=18432、K=2560为例：

- per-channel权重scale为18432个FP32，73728B；128×128 block scale为144×20个FP32，11520B，反而更少。
- M8激活scale从32B增加到640B，但权重scale减少得更多，整个Linear逻辑流量账本约减少60992B。
- M512激活scale从2048B增到40960B，整体逻辑账本仅约增加15616B，即0.02188%；`down`在M512的逻辑增量约0.63485%。

账本按输入、量化结果、输出和scale各读写一次计算，不包含跨tile重复加载、L2命中和编译器调度。NCU计数器受`ERR_NVGPUCTRPERM`限制，所以这里没有实测DRAM流量，不能把这些字节直接除带宽当成耗时。

### 6.2 统一Triton和autotune，主要改善了特定GEMM shape

生产两条GEMM都使用有限的六组tile/warp/stage候选，按shape、device和dtype缓存；正式计时与搜索分开，Graph capture中不临时启动搜索。per-channel在完整K归约后缩放，block仍逐K块应用scale，量化语义没有为了共用框架而改成一样。

调优后微基准中，M512的`gate_up`约941.2→310.8μs，`down`约492.5→184.6μs，分别观察到约3.03×/2.67×的GEMM比值。部分样本波动超过5%，而且这只是两种MLP投影，不能写成整个模型提速3倍。[调优记录](benchmarks/reports/fp8_granularity/triton_autotuned.md)

### 6.3 Prefill真正差在哪里：GDN投影比“多切几个块”更具体

固定B8、每请求64输入，prefill M512，真实engine稳态Graph计时PC/BW平均59.42/75.48ms；两个PC样本波动超过5%，比例作观察。独立profile中，BW比PC的GEMM累计多11.485ms、激活量化多1.514ms，而GDN核心、卷积、norm和attention核心差值很小。[算子分解](benchmarks/reports/prefill_breakdown/results.md)

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

把这个out_proj/o_proj配置带回B8、输入64、输出128的真实engine，block优化前后总耗时1800.00→1800.63ms，decode589.77→592.54 token/s；TTFT77.29→84.72ms且波动很大。共同轨迹1024个位置上，两版block logits完全相同，但**两次样本没有证明额外端到端提速**。这条失败结果保留，没有用单算子201→91μs替代它。[端到端与精度回归](benchmarks/reports/fp8_e2e/results.md)

### 6.5 Graph输出buffer复用：先让完整prefill跑得下

顺序层此前会各自持有完整M的投影输出。后来按projection shape/dtype/device复用prefill输出buffer，同时保留各层独立KV/GDN状态，实测支持完整M到8192。这里复用的是顺序执行的临时dense输出，不是让不同请求或层共享状态。

B4/8×输入256/512/1024×输出512/1024的完整Graph矩阵，BF16/PC/BW共72次正式测量：PC总耗时比BF16减少约29.6%–33.6%，BW约27.9%–33.3%；PC在12个case的TTFT都更低，decode中10个case略快。[完整矩阵](benchmarks/reports/fp8_matrix/results.md)

这更新了“PC prefill快、BW decode快”的早期印象：**在这版源码和矩阵里，BW并没有稳定的decode优势。** 后端、tile、共享量化、Graph以及具体batch/context共同决定排序。这批测量发生在当时的源码快照，后续整合保留数据，不冒充整合后全部重测。

## 7. 2026-10-14：decode容量上去了，吞吐却逐渐饱和

decode-only实验先建立真实KV/GDN状态，计时128步、context65→192；仅捕获当前batch，KV每请求1个256-token block，统一90% allocator预算。预热、prefill和收尾不计入decode。[容量实验](benchmarks/reports/fp8_decode_capacity/results.md)

| 方案 | 当次16对齐容量边界 | 吞吐最高的已测batch | 已测峰值 token/s |
|---|---:|---:|---:|
| BF16 | 96 | 96 | 1877.0 |
| per-channel | 256 | 256 | 2476.4 |
| block-wise | 256 | 256 | 2487.5 |

两种FP8相对BF16各自已测峰值高约31.9%/32.5%，其中包含容量不同带来的收益；同为B96时只有约20.4%/14.2%。不能把跨batch的峰值比说成同batch加速。

显存节省主要来自权重。扣除embedding/head共享storage的重复引用后，BF16约7.83GiB，PC/BW约4.52/4.51GiB；每请求GDN状态约25.125MiB、此窗口KV约8MiB，并不随权重量化缩小。B128→B256虽然请求数翻倍，PC吞吐只从2425.5升到2476.4，已经很接近平台区间。

BF16 B128一次未限额预热在WSL接近显存满载时明显变慢，正式计时前终止；是否为驱动系统内存回退没有直接证据，后续统一预算。BF16 B96两次差约5.6%，没有补测第三次。这里的“最大batch”仅属于这一短上下文和capture/KV策略，不能直接移用于完整prefill或默认捕获全部小batch的部署。

## 8. 2026-10-15：profile把大batch热点从FP8转到了GDN状态复制

先用`bench_one_batch`对两种FP8做B1/4/8/16/32/64、输入64、输出128，每配置2次，prefill/decode均Graph；另外独立采prefill和最初8步decode的trace。旧版正式结果中B64的PC/BW约2102/2079 token/s，TPOT约30.44/30.80ms。[优化前报告](benchmarks/reports/fp8_phase_profile_before_gdn_pool_20261015/results.md)

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

独立B64 profile每个8步窗口中，旧版192次recurrent gather、192次scatter，新版两项都为0；旧版这部分约9.7–10.9ms/步，核心时间仍约4.4–5.0ms/步。这次有“操作消失”与正式吞吐共同支持归因，不只是整模型一次变快。[受控A/B与trace证据](benchmarks/reports/gdn_state_pool/results.md)

实际CUDA回归覆盖稀疏乱序slot、slot0、slot大于当前batch、多次Graph replay改变索引，输出与完整BF16状态池对dense参考逐位一致。优化没有换成低精度state，也没有删除请求隔离；该阶段98项通过是当时的回归数量，不是全模型质量分数。

### 8.2 完整prefill矩阵再次说明：最大batch不等于最快batch

之后扩到B32/64/96/128/192/256、输入64、输出128。B96只新增2次，其他batch沿用原样本；均完整prefill与decode Graph、固定每slot两个KV block、90% allocator预算。

| B | PC TPOT ms | BW TPOT ms | PC token/s | BW token/s |
|---:|---:|---:|---:|---:|
| 32 | 15.56 | 16.05 | 2057.13 | 1993.38 |
| 64 | 20.39 | 20.44 | 3138.47 | 3130.85 |
| 96 | 39.42 | 36.76 | 2435.35 | 2611.70 |
| 128/192/256 | 未完成 | 未完成 | 不生成数字 | 不生成数字 |

这版完整流程中B64是已测吞吐最高点；B96能完成但更慢，并且部分样本device-used达到显卡容量。没有分页/DRAM计数证据，不能断言具体驱动回退机制。[现阶段矩阵与曲线](benchmarks/reports/fp8_phase_profile/results.md)

B128在KV预算guard处失败，B192/256在初始化warmup的LM head或GDN chunk状态处OOM。一个共同问题是prefill仍对所有token做完整词表投影：`[B×64,248320]`的BF16 logits，仅B128/192/256就约需3.79/5.68/7.58GiB，还没加权重、GDN、KV和图内临时量。

这与前一天decode-only能到B256并不矛盾：那次不计时完整prefill、只捕获当前batch、每请求1个KV block；这一轮有完整prefill且KV策略不同。容量失败配置保留0/2、错误阶段和日志，不按“每项2次”机械重复OOM，也不补造两个耗时。

## 9. MTP：先补全缓存正确性，严格通过后仍然比关闭更慢

MTP最初作者时间为9月25日，原分支提交为`817db88`；10月15日rebase到main后变成`8322fe2`，其committer时间才是10月15日。它只在MTP分支上，不能写成同期main已经默认启用。当前opt-in边界是greedy、TP1、B1。

### 9.1 最初链路加入了draft/verify，却漏了完整cache生命周期

代码审查与元数据复现发现了几类独立问题：

- MTP RoPE从1开始，物理cache却也从1写，而paged读取声明从0开始：129-token prompt写1…129、读0…128；256-token prompt会越过一个block的边界。
- 接受draft后仅把bonus送入下一次MTP，跳过已接受draft对应的hidden/token pair，造成缓存中间空洞。
- 129按128+1分块时，最后一块只声明两行KV，没有包含此前缓存；第一块若只有1 token，又没保存previous hidden。
- EOS、终止和slot复用需要对齐真实已消费前缀；进一步检查还发现抢占重放保留InputBatch slot，旧GDN state未重置。

`0ba199f`统一为RoPE位置p存cache行p−1，使用累计有效长度，verify预留2行；接受后先补draft、再补bonus，EOS回滚到第一行snapshot；结束清metadata，抢占从零重放时清GDN和MTP状态。

这次验证不能只看最终target token：新测试强制接受、拒绝、连续和交替决策，覆盖127/128/129、255/256/257、511/512/513、完整/分块prefill、EOS、剩余1/2输出、slot复用和抢占恢复。在相同已消费token前缀比较completion、GDN、target KV、MTP KV；MTP有效KV还与相同target hidden的独立前缀前向比较。[完整修复和测试口径](benchmarks/reports/mtp_cache_lifecycle/results.md)

这些状态对照主要由小型真实混合checkpoint的CUDA集成测试完成：同一prefill分块方式下，GDN/target KV逐位比较；独立MTP前缀参考使用BF16容差；完整prefill与分块prefill的不同算术形状采用归一化状态误差限制。4B完整模型另做greedy输出gate与正式128-token对照，不把它等同于4B每层全量cache已逐位验证。

### 9.2 Cache修好后，还遇到两行verify与普通B1 decode的数值差异

仅修索引和补写，真实BF16 4B仍在第6个completion token分歧。额外探针和宽投影回归表明，两行target verify与普通单行decode的GEMV/GEMM等算术形状不同，可能翻转接近并列的贪心结果。

当前正确性路径把verify改成**两次顺序B1 target forward，LM head也逐行处理**；Graph将两次target前向一起捕获。这样保住了基线算术和第一行回滚snapshot，但失去了原来两行target并行的潜在收益。这是已知取舍，不能再按“一次target forward出两个token”的理想成本估算吞吐。

FP8转换目录不含`mtp.*`时，通过独立`mtp_model`加载原始BF16 MTP权重，并校验架构、GDN参数与实际有效RoPE配置。raw FP8 MTP张量不直接当BF16权重读入。完整回归261项通过；两种FP8 B1 eager/graph短gate通过，正式4组对照的128个completion token也全部相同。

### 9.3 实际提交token的性能，否定了“高接受率就会快”的判断

B1、输入64、输出128，decode Graph，每种FP8正式2次；两边prefill均eager，排除首token，计时实际提交的127个decode token：

| 量化 | MTP关闭 token/s | MTP开启 token/s | 开启/关闭 | draft接受情况 |
|---|---:|---:|---:|---:|
| per-channel | 83.13 | 62.88 | 0.756× | 60/66 verify，约90.9% |
| block-wise | 82.28 | 60.50 | 0.735× | 57/70 verify，约81.4% |

**当前严格一致的MTP路径吞吐下降约24%–27%。** 两次target、额外MTP/LM head、状态snapshot/rollback和调度同步没有被接受率抵消。`enable_mtp`继续默认关闭；旧`projected_tokens_per_s`仅可作诊断，`--allow-numerical-drift`或stress均不能绕过正确性失败输出最终有效加速结论。[原始两次样本与协议](benchmarks/reports/mtp_cache_lifecycle/results.json)

短block-wise Graph trace记录8次verify、6次接受/2次拒绝、提交14个decode token；物理GPU事件累计中FP8 GEMM约52.5%，BF16 GEMM约25.0%，索引读写组约2.0%。后者还包含原有conv-state更新和embedding，不能全算新增snapshot。每请求GDN state共25.125MiB，当前snapshot经临时张量搬运，完整一次约100.5MiB逻辑读写；完整一次rollback同量级。

每次verify仍有`.item()`同步。8次`aten::item`在该trace的CPU区间累计237.947ms，但包含等待前序GPU，和GPU时间重叠；不能声称删掉它就能节省237.947ms。源码还确认：接受draft后第一行MTP本来只需补KV，目前仍计算被丢弃的词表预测，是后续可考虑的cache-only路径。该轮尚未执行这项优化，也没有预支收益；后续实际结果见9.4。

MTP关闭的单独版本回归使用prefill/decode都Graph的`bench_one_batch`，与上表协议不同。PC decode均值+1.5%、BW−2.4%，prefill约33–63ms波动；2个样本不足以确认严格零回退。该轮还修复了旧benchmark包装器不转发runner第4个`is_verify`参数的问题，旧/新版本用同一修正harness，失败发生在正式计时前。

B4/8没有通过移除guard来“支持”；批量MTP仍未实现。后续若恢复并行verify，必须继续比较实际消费前缀下的状态/缓存，而不是退回只看最终target token或容许drift。

### 9.4 先删除被丢弃预测的计算：cache-only有收益，但仍不抵消两次target

`3f8a3ba`把接受draft后第一行补写改为独立cache-only：保留embedding/pre-norm/fc/input norm、K/V投影、K norm和key RoPE/store。跳过Q、attention、out projection、MLP、final norm和LM head；下一行仍完整前向预测draft。一层MTP的KV只依赖这段前缀，因此可以删除后面的计算而不更改有效KV。

新建独立CUDA Graph及输入buffer；14项真实CUDA回归覆盖eager/Graph、非连续block、位置1/128/129/256/257/512/513，整份KV含未写行与完整MTP逐位相同，后续draft也相同。完整测试从261增加到275。实际两种FP8另做短eager gate，正式每配置两次完整128-token Graph对照严格相同。

输入64/输出128、B1、prefill eager/decode Graph、只计实际提交的127个decode token：PC完整补写63.10→cache-only 68.79 tok/s（+9.0%），BW 61.21→67.39（+10.1%）。同轮关闭MTP分别79.74/82.87，所以cache-only本身仍未使MTP超过off。完整补写/cache-only/cache-only/完整补写在同一engine执行；这是独立A/B，不与前一轮83tok/s混算。[报告与原值](benchmarks/reports/mtp_cache_only/results.md)

### 9.5 两token主干复用：首先遇到的FP8差异来自norm，不是投影

随后只在独立benchmark尝试把两行dense、输出投影和MLP按M2计算，GDN conv/recurrent仍按token顺序执行并保存首步snapshot，full attention保留逐行单query，LM head也逐行。生产继续保留serial B1实现作为参考，没有添加默认启用开关。

两种FP8最初的首窗口auto/pinned和独立Graph均通过逐层522项检查，小FP8强制决策生命周期也通过，但PC完整128-token输出检查仍发生分歧。沿serial参考轨迹定位到位置69/70、第26层post-attention norm：1个元素不同，最大绝对差0.015625；此前dense/core均逐位一致。另一个BF16小模型则最早在第1层gate_up分歧。**首窗口通过不能证明完整轨迹一致，只固定GEMM配置也不够。** 失败计时没有被保留为有效收益。

修正版仍让矩阵工作处理M2，将FP8 M2 tile固定到对应M1已调优配置，但input/post/final norm恢复逐行M1归约。真实4B完整serial轨迹PC 66个verify窗口、BW 70个窗口，每窗522项逐位通过，比较各层hidden/residual、dense/core/MLP、最终logits、GDN最终/首步snapshot及有效target KV。该全轨迹逐层诊断的候选为eager；Graph状态门禁由小FP8集成测试覆盖，真实4B Graph另做完整completion检查，没有把它说成每窗逐层Graph状态检查。

新增11项回归（10项小FP8 eager/Graph，1项CPU非有限值gate），完整286pass。强制接受、拒绝、连续接受后拒绝和交替模式，129按128+1分块，在相同已消费前缀逐位比较GDN、target KV及MTP KV；诊断遇到相同Inf也不能放行。

下表是修正版新一轮独立计时，不覆盖9.4。B1、输入64/输出128、相同模型/MTP来源，prefill eager、decode Graph、profiler关闭。每模式2次，serial/reused在同一engine按ABBA；每次必须完整128个completion token与off严格相同，计时127个真实提交token。

| 量化 | MTP关闭 tok/s | serial + cache-only tok/s | 两token复用 + cache-only tok/s | 复用相对off |
|---|---:|---:|---:|---:|
| per_channel | 84.46 | 69.17 | 91.86 | +8.8% |
| block_wise | 79.96 | 62.98 | 83.82 | +4.8% |

这证明重复主干矩阵工作在当前工作负载上确实值得先处理；cache-only收益和本表百分比不能直接相加。结果只来自一个prompt、一个B1形状和4070Ti SUPER、两个样本。两行复用仍是单线程局部patch实验，未接入生产并发配置、未推广BF16；生产当前仍是serial verify + cache-only。没有继续做snapshot/rollback和`.item()`优化，也没有扩展B4/8、TP>1或5090。[独立实验、首差异及两次原值](benchmarks/reports/mtp_verify_reuse/results.md)

### 9.6 从实验patch变成正式开关：复用收益在原B1工作负载上得到复现

`be9b5dd`正式加入`mtp_verify_mode="serial" / "reuse"`，默认serial，整体MTP仍由`enable_mtp`显式开启；reuse当前限定FP8、TP1、B1。上一节描述的是独立实验当时的状态，这一轮将它接入正式runner和CUDA Graph。

GDN拆成dense投影、单token状态core与out projection；普通decode和reuse共用同一状态更新代码。Attention也暴露输出投影之前的core，因此不再通过`patch.object()`替换投影和forward。两行dense/输出投影/FP8 MLP合并；input/post/final norm仍按M1，GDN两步依次执行并保留首步snapshot，attention保留单query，LM head逐行。混合FP8 checkpoint中未量化的linear也保留M1算术。

M1 tile选择变为本次Context的执行策略，按当前卡、N/K和dtype读取M1调优结果，直接launch M2，不改写M2 autotuner项。缺少M1结果时先在eager调优，cold capture明确失败；正常engine先预热普通B1再捕获verify。Graph记录自己捕获的模式，配置变更后不重新capture会报错，避免静默重放serial。5090将使用自身M1调优结果，但本轮没有在5090执行。

从286项基线开始，先让配置、M1分派、M2行数、冷capture、混合BF16例外和旧图误用门禁失败，再接入API。随后将原有边界/终止门禁覆盖到两种FP8与eager/Graph：127/128/129、255/256/257、511/512/513，128-token分块；EOS出现在accepted draft、bonus或拒绝分支，剩余1/2/3个输出，重复请求slot复用。按相同已消费前缀比较GDN、target KV和MTP KV，完整回归最终383pass。

真实4B两种FP8首窗口的eager/Graph各522项逐位通过；完整serial参考轨迹PC66个、BW70个verify窗口，每窗522项严格一致。这里全轨迹逐层候选仍为eager；真实Graph另由完整completion门禁约束，没有声称每个Graph窗口都做了逐层检查。

代码复核发现新增benchmark借用engine的接口可能混入已有请求或另一个MTP来源，异常时也可能遗留请求。补9项CPU回归后，借用前检查空闲、有效MTP来源和长度；finally定向移除本次请求，保留调用者engine；统计返回本次差值，保留engine累计值。benchmark直接走正式配置/API，不再靠替换方法或改写tuner cache来切换模式，同时记录TPOT、TTFT、请求总耗时和allocator显存。

正式对照仍为B1、输入64/输出128、prefill eager、decode Graph、关闭profiler；每模式两次，MTP engine按serial/reuse/reuse/serial执行。全部128个completion token必须严格等于off，计时127个真实提交decode token：

| FP8 | off token/s | serial token/s | reuse token/s | reuse相对off |
|---|---:|---:|---:|---:|
| per_channel | 83.39 | 69.84 | 94.06 | +12.8% |
| block_wise | 83.26 | 65.92 | 88.31 | +6.1% |

PC每次60/66接受，BW每次57/70，serial和reuse接受情况一致。本轮在原工作负载上复现了收益；此前PC约8.8%、本轮12.8%，但两轮绝对值不构成API改造的额外加速对照，也不能由每项两次推断不同输入都稳定提升10%。TTFT两次有明显波动，不用它宣称prefill优化；allocator总量包含预分配KV与Graph，不当作MTP权重净开销。[正式报告、源码指纹和逐次数据](benchmarks/reports/mtp_reuse_production/results.md)

本轮保持snapshot/rollback和`.item()`实现不变。下一阶段先以同prompt测试64/256/1024输入×128/512输出，再补中文对话/代码输入；之后取reuse新trace排序热点。B2/4/8需要混合接受/拒绝和新graph布局，继续保留guard，不把开关接入等同于已支持多请求。

### 9.7 B1泛化验收先暴露greedy并列问题，再正式扩展B2（2026-10-15，`7dc09b9`）

这一步先没有继续做kernel。`be9b5dd`的收益只来自B1输入64/输出128，先按同一种合成prompt测试输入64/256/1024×输出128/512，再加入中文对话和代码请求。两种FP8分别off/reuse、每配置正式2次，prefill eager、decode Graph；完整prompt、completion和正负结果保留。[B1矩阵](benchmarks/reports/mtp_small_batch/b1_results.md)

**先遇到的不是速度，而是严格completion失败。** 旧采样器把`temperature=1e-6`当近似greedy，却仍对softmax概率做随机采样；verify直接argmax。PC输入256的第57个completion，token3158与4100的最高logit恰好同为21.125：off随机选4100、verify选3158。PC64/512在第314个输出、PC1024在第108个输出也分歧，两次位置一致。初始PC只有6/16个reuse样本严格通过，BW只有2/16。不能把这些结果当作长度泛化的性能证据，更不能据此先放开B2。

`layers/sampler.py`现在对<=1e-6的请求确定性argmax，其他温度继续随机采样。混合greedy/stochastic并列分数回归先失败再通过。重新执行的B1矩阵64个正式样本全部通过严格门禁，原失败矩阵单独保存。PC收益约−4.5%～+7.7%，BW约−3.1%～+17.8%；负收益没有删掉，也没有追加第三次挑结果。接受率、上下文和内容都会改变收益，所以原64/128的12.8%不能当作所有请求都能达到的固定加速。

**新trace与serial旧trace分开。** 两种FP8分别取reuse Graph/eager短窗口，各预热9个engine step、诊断12轮verify，不与正式计时混算。Graph通过空闲engine重新capture插入external CUDA events，拆出首步snapshot和MTP图内LM head；实际物理kernel按最内层phase归属一次，利用eager中head的name/grid/block指纹识别Graph内head，避免重复累加嵌套scope。Graph里target verify约63%～64%、三次LM head约32%～33%；snapshot事件约0.33/0.34ms每轮，只占GPU工作约1%～2%。这段短窗口均接受，未测出rollback比例，不能把0次外推成拒绝免费。`.item()`的CPU等待与GPU verify重叠，设备空隙还受profiler/event影响，不能整段算可消除收益。因此没有先做snapshot融合或完整GPU接受判定，而继续B2。[reuse新trace](benchmarks/reports/mtp_small_batch/reuse_trace.md)

**B2的难点是请求×时间的映射，而不是删guard。** `models/mtp_verify.py`对外保留request-major `[A0,A1,B0,B1]`，内部dense转为time-major `[A0,B0,A1,B1]`，2B矩阵使用普通M=B配置。norm、未量化例外和词表head仍按B行，GDN两步分别批量更新B个pool slot；第一步每请求保存snapshot，attention每步每请求单query。`model_runner.py`修正了第二请求误取累计KV长度的问题，改为相邻累计长度之差；GPU按请求形成accept mask和预测对，只回传一份小decision table，host继续维护EOS、接受/拒绝和调度。这里仍有D2H同步，没有宣称已实现纯GPU决策。

接受请求补cache-only行后再算bonus/draft；拒绝请求回滚自己的GDN slot，MTP有效长度分别维护。`cuda_graph.py`的verify、full MTP和cache-only均建立精确B1/B2图，处理只接受一个请求、先结束一个请求和剩余B1，不用padding行接触活跃状态。调度上将verify与prompt/普通尾步分开。保留serial作为参考，BF16依然只支持B1 serial，FP8 B4/8仍guard。

**扩大生命周期后，审查又逼出了容量/slot问题。** 两个255-token prompt、仅2个KV块时，A会因verify跨块被preempt，B借到块且仍in-flight；旧准入检查只看running为空就报容量不足，没等B结束释放。另一个例子中，被preempt的A重放需要2块，只有1块空闲，队列后B却已经持块且可完成最后prefill chunk；仅把异常改break会死锁。现在等待in-flight释放，且让已持块续块绕过受阻重放。新请求也不能占用in-flight或partial prompt保留的state slot；当新C挡在已接纳A/B之前时，先继续A/B。上述例子先RED再GREEN，并由审查独立CPU复现确认。三个排队请求、两个slot及分块prefill也加入真实eager/Graph测试。

最终451项测试通过；两种FP8×eager/Graph覆盖AA/AR/RA/RR、交替、129/257异长、分块、EOS、剩余预算、不同结束时间，以及另一请求活跃时slot复用。相同已消费前缀上的hidden、GDN、有效target/MTP KV与serial逐位一致，inactive slot发生写入会失败；B1原有逐层与完整缓存门禁保留。

真实4B先做两种FP8的异长64/256、输出128 eager off/serial/reuse门禁，再做三种工作负载的Graph off/reuse，各配置2次；24个正式样本每个计254个实际提交decode token，全部严格一致。异长case还直接确认了eager与Graph completion相同。调度边界修复后重测的数据如下；修复前诊断修订没有混入均值。

| FP8 | B2输入 | off token/s | reuse token/s | 收益 | off / reuse TPOT ms |
|---|---|---:|---:|---:|---:|
| per_channel | 64/64 | 112.10 | 169.53 | +51.2% | 17.842 / 11.798 |
| per_channel | 64/256 | 111.05 | 168.73 | +51.9% | 18.011 / 11.699 |
| per_channel | 中文45 / 代码43 | 111.44 | 156.98 | +40.9% | 17.948 / 12.449 |
| block_wise | 64/64 | 110.77 | 145.65 | +31.5% | 18.056 / 13.743 |
| block_wise | 64/256 | 109.95 | 137.53 | +25.1% | 18.190 / 14.221 |
| block_wise | 中文45 / 代码43 | 112.71 | 151.28 | +34.2% | 17.744 / 12.654 |

TPOT是两个请求各自decode耗时/127后平均；batch聚合吞吐使用总实际提交数/整批decode时间。输出均128、ignore EOS，prefill eager、decode Graph、profiler关闭，engine初始化不计入请求时间。每项只有2次，不把这组收益称为统计稳定的全输入结论。逐请求TTFT、总耗时、显存、逐step提交数及源码指纹见[B2报告与原值](benchmarks/reports/mtp_small_batch/results.md)。收紧普通decode capture列表时还补了3/5/17等非标准容量的最大精确bucket，防止漏掉最大请求数；B1/B2/B64列表不变，此计时后边界补丁在报告明确注明。

当前证据支持FP8 TP1 B2正式reuse；还没有B4/8或5090结果，也没有找到完整batch收益分界。普通decode的B64吞吐峰值仍是另一个工作负载的独立基线。下一阶段应在同样严格状态语义下继续B4/8，随后按新batch的trace决定优化head、顺序核心或状态复制，不能只延长数组就宣称多batch生产化。

### 9.8 greedy fast path：改善普通decode基线，也收窄MTP相对收益（2026-10-15，`4c83dac`）

上一版虽然对greedy最终返回argmax，却仍计算softmax、指数随机数和随机采样候选；MTP verify已经直接argmax，两个路径多做的工作不同。`layers/sampler.py`新增可选`all_greedy`提示，由`model_runner.py`读取CPU侧Sequence温度决定；全批<=1e-6时直接argmax，没有GPU温度归约、`.item()`或新增同步。内部验证采样器同时适配提示参数。混合/非greedy沿用原算法；fast不消耗RNG，因此后续随机请求的全局RNG偏移可能不同，给定相同初始RNG的混合采样语义不变。

新增两项回归先RED，检查不调用softmax/RNG/设备标量读取、不修改logits，runner按CPU温度路由且不推进RNG；原混合温度并列分数测试继续通过。完整测试从451增至453。

正式复测没有拿两个不同进程的旧吞吐相减，而在同一resident engine里用benchmark-only FullSamplingReference忽略fast hint，调用保留的旧完整采样算法。每轮reference-off/reference-reuse/fast-off/fast-reuse，再重复一轮；两种FP8、B2、same64及中文45/代码43、输出128、prefill eager/decode Graph，32个正式样本严格completion一致。每样本计254个实际decode tokens。off暂停MTP执行，但其权重和graph仍驻留，不能把allocator数值当成卸载MTP后的容量差。

同轮受控实验中off提升36.4%～38.7%；PC same64的MTP相对收益从42.3%收窄到3.6%，中文/代码从39.2%到1.6%；BW相应从43.8%到1.5%、32.3%到0.3%。这不是MTP实现变慢，而是普通路径原先做了额外采样工作，修好后比较基线更强。两个样本也不足以给很小差异稳定排序。不同engine/调优的后续B2曲线绝对值不能与这组数据直接相减。[受控报告](benchmarks/reports/mtp_greedy_fast_path/results.md)

### 9.9 B4/8先补齐实际shape与生命周期，再看收益是否仍成立（2026-10-15，`1479335`）

B2已有通用request-major↔time-major布局，继续扩batch的主要风险在真实活跃数和接受子集。B4只接受3个、B8只接受5个时，cache-only/full MTP都必须按实际大小执行；请求结束还会出现B3/5/6/7。旧ordinary decode在容量8时只有1/2/4/8图，B3会向上padding到4，无法作为reuse真正M=B的数值参考。

`engine/cuda_graph.py`因此对容量<=8捕获每个实际ordinary batch；verify、full draft和cache-only沿用1..capacity的精确图库。`config.py`、serial/reuse入口在验证后开放FP8 TP1的capacity1..8，BF16继续只限B1 serial，B16保留guard。M=2B仍选当前设备、shape、dtype的M=B调优配置，不改写全局autotuner缓存；norm、head和BF16例外保持B行算术，GDN两步按B个pool indices顺序更新，没有借扩batch修改已验证的状态语义。

新增真实小型混合模型回归覆盖两种FP8/eager/Graph，接受数0..B及非连续mask、B8→1每个活跃数、EOS、不同结束时间、新请求到达及slot复用。对齐相同已消费前缀，逐位比较hidden、未被替换的真实logits、GDN状态、target KV和MTP KV，记录完真实logits后才用测试候选强制接受/拒绝；inactive slot不得写入。每个subset1..8的full/cache-only整个KV张量也要求逐位相同。

审查发现初版B4/B8用例没有真正跨chunk，补上chunk128、B+1排队请求、129/257交替输入及127-token尾请求；新增8项全部通过。最终完整619项通过，32项skip仅是B4下subset>4的不可行参数，对应B8参数实际执行，未把缺失GPU验证伪装成通过。

真实4B门禁再覆盖两种FP8×B4/B8×eager/Graph×off/serial/reuse，共24组异长输入64/88/112/136…232，输出128；全部完整completion严格一致。这些只作诊断，不纳入正式性能均值。随后固定4070TiSUPER、TP1、input64/output128、prefill eager/decode Graph、关闭profiler，每个B1/2/4/8点交错off1/reuse1/off2/reuse2，仅正式两次。32个样本全部严格一致，计数B×127实际提交tokens，没有projected吞吐。

| FP8 | B | off / reuse tok/s | 吞吐收益 | off / reuse TPOT ms |
|---|---:|---:|---:|---:|
| channel | 1 | 81.60 / 90.30 | +10.7% | 12.256 / 11.074 |
| channel | 2 | 146.67 / 161.86 | +10.4% | 13.637 / 12.375 |
| channel | 4 | 289.30 / 279.77 | -3.3% | 13.826 / 14.297 |
| channel | 8 | 555.42 / 521.00 | -6.2% | 14.407 / 15.361 |
| block | 1 | 82.41 / 83.04 | +0.8% | 12.136 / 12.045 |
| block | 2 | 151.09 / 156.99 | +3.9% | 13.237 / 12.756 |
| block | 4 | 293.09 / 289.76 | -1.1% | 13.648 / 13.858 |
| block | 8 | 560.05 / 567.37 | +1.3% | 14.286 / 14.100 |

这轮per-channel只在B1/B2有约10%平均decode收益，B4/B8回退3.3%/6.2%。block-wise没有单调的收益分界；B4两次reuse为307.73/271.79tok/s，一次高于、一次低于off约293tok/s，不能用均值−1.1%断言所有重复都慢。MTP的TTFT还会抵消decode收益：BW B2平均decode+3.9%，请求总耗时仅1.744→1.738s。误差棒只表示两样本范围，不是置信区间。

因此“支持B4/8”和“值得在B4/8开启”被分开记录：当前这组PC工作负载优先B1/B2，B4/B8倾向off；BW小幅差异继续保留不确定性，没有硬编码自动策略或宣布全模型/全上下文统一阈值。暂不扩B16，5090未测。B2/B4新trace尚未采集，下一次选择LM head、同步、快照或GDN核心优化时必须重新profile，不能继续套旧B1比例。[曲线、每次原值及门禁](benchmarks/reports/mtp_batch_curve/results.md)

### 9.10 将greedy fast path独立移植main，重新测真实基础路径（2026-10-18，main `e936f58`）

main仍在80ad401，没有整套MTP实现。这次没有把mtp2合回main，而仅移植`layers/sampler.py`、`model_runner.py`的CPU温度路由及两种验证sampler签名。全greedy直接确定性argmax；混合批次的<=1e-6行也使用同样tie规则，其他温度保留随机算法。全greedy不再推进RNG，后续随机请求的全局RNG偏移可能变化，这与给定相同RNG状态下保留混合采样算法是两个不同问题。

main原本对并列最高BF16 logit仍随机抽样，不能声称这个行为与新argmax相同。4项回归先RED，覆盖混合tie、跳过softmax/RNG/设备标量读取、CPU路由，以及prefill尾行和请求slot映射；干净main98项，移植后完整102项通过。没有新GPU温度归约或同步，也没有MTP模型、KV或graph依赖。

新增`bench_greedy_sampling.py`在真正独立main engine内，B2、same64及中文45/代码43、输出128，prefill eager/decode Graph。每个case完整采样reference1/fast1/reference2/fast2，先4-token短预热，每配置仅2次。reference保留确定性tie，但仍执行softmax/RNG，控制的是fast分支而非旧main随机tie行为。两种FP8共16个正式样本严格completion一致，每次254个实际decode tokens。

本轮PC same64/中文代码平均+0.50%/+1.53%，BW为+2.31%/+0.10%。这些小幅变化不能称为稳定收益，更不能把10月15日mtp2另一条runner/输出路径中的36%～39%外推成main的固定加速。main已有独立正确性和性能证据，MTP仍留在可选feature分支。[main报告与原值](https://github.com/Jung52/nano_qwen/blob/e936f58/benchmarks/reports/greedy_main/results.md)

### 9.11 B2/B4新成本报告：GPU省下的余量与串行rollback（2026-10-18，mtp2 `f443cb3`）

这次只补profiling，没有改MTP生产计算。`profile_mtp_reuse.py`从固定B1扩到B2/B4 off/reuse，并将CPU分析、Graph event插桩分到两个benchmark模块。所有生产src文件仍与1479335逐字节一致。主验收使用4070TiSUPER/4B/per-channel/Graph；两种FP8与eager/Graph的小模型继续做serial/reuse逐前缀hidden、真实logits、GDN和target/MTP KV门禁，没有借功能支持范围宣称所有模型都已测性能。

先运行无profiler、无插桩的两次正式样本；再用完整128输出的CUDA event pass拆target body/head、draft body/head、cache-only、snapshot/rollback。event只读取本step真正replay的精确bank，重复同bank不能误读最后一次时间。小模型Graph回归抓到了重捕获ordinary decode时默认dtype已恢复为FP32的问题；改为只在benchmark重捕获期间使用模型dtype并还原，不靠cast输出掩盖错误。

随后恢复不含external event节点的原始graph，取completion约48→80短trace，再取不同输出预算的动态尾段。后者实际经历B4→3→2→1和B2→1；逐step保留stable slot、cached前缀、实际提交tokens及accept/reject。物理GPU kernel计一次，target head独立，draft图内head仅按同trace target head的name/grid/block指纹识别。snapshot已包含在target body，展示独立列前先减掉；GPU event区间包含enqueue空隙，不能等同纯kernel时间。

| B | off / reuse tok/s | 平均收益 | off / reuse TPOT ms | accepted/drafted |
|---|---:|---:|---:|---:|
| 2 | 173.48 / 168.94 | -2.6% | 11.531 / 11.848 | 81.4% |
| 4 | 290.99 / 274.23 | -5.8% | 13.748 / 14.593 | 84.1% |

旧图的“B2正、B4负”没有在本轮稳定复现：B2两次一正一负，均值−2.6%，B4均值−5.8%。保留这个结果，没有加第三次挑数。当前报告能解释成本机制，不能倒推10月15日运行的唯一原因；旧运行缺少全部tuner选择记录，且本轮primary与独立host进程的M=B参考tile也有变化（B2 4/6、B4 3/6个shape），跨pass绝对值不相减归因。

本轮短trace中B4 target kernel从2.1995降到1.6860ms/聚合token，但额外head约0.3404ms，加上draft/cache/rollback后，全部GPU kernel仅从2.7030降到2.6363ms/聚合token。接受率84.1%对应的是平均每轮能提交多少token，不会自动减少两次target词表投影和后续draft词表投影。GPU余量已经很薄，不能只看主干权重复用或接受率。

**再用轻量host pass检查真正的提交成本。** 独立进程只读perf_counter_ns，不开profiler、不加CUDA event或额外GPU同步；这不是第三次正式计时。B2有13个拒绝轮，24层×2请求=48次回滚/轮，共624次；host共约80.55ms，约6.20ms/拒绝轮。B4有10个拒绝轮，每轮24×4=96次，共960次、约123.59ms，约12.36ms/拒绝轮。调用均约129µs；每请求token对应scope从0.634增到0.973ms，决策table等待仅约0.091/0.077ms。

同一个host进程内，off→reuse诊断TPOT差为B2约0.48ms、B4约0.83ms，增加约0.36ms；rollback scope增加约0.34ms，与这部分增长接近。每请求本身可能少拒绝几次，但同一批所有请求的逐层恢复串行提交，其他请求也要等。物理copy kernel很短，host tensor构造、index_select临时分配、copy提交及等待却占掉余量。scope仍可能覆盖GPU工作，不把整段当成可直接删除的收益。

因此下一项只选择**按slot批量恢复rollback，减少逐请求逐层的临时gather与host提交**，本轮没有实现这个优化。保持现有host接受判断和数值路径，不同时修改LM head、GEMM或GPU决策。LM head也是主要额外GPU项，但预测都被使用，不能直接删除一次head。B16继续等，5090和9B仍需独立基线。

最终mtp2完整627通过、32项历史B4不可行子集skip；8个正式样本、4个CUDA event完整cohort、8个短trace/动态尾段和4个host完整cohort，off/reuse严格completion门禁均通过。Ruff、差异和独立复核通过。原始大trace保留本地，报告保留其SHA256/大小、完整tokens、配置与step元数据。[成本报告及图](benchmarks/reports/mtp_batch_stages/results.md)

## 10. 目前各模块走到哪里

| 模块 | 已形成的实现与经验 | 尚未被现有数据证明的部分 |
|---|---|---|
| Engine / InputBatch / D2H | 请求slot、GPU token、Event和双缓冲保证生命周期；异步/coalesce提供合批机制 | 普遍的CPU/GPU overlap收益；动态负载TTFT与吞吐同时改善 |
| Scheduler / chunked / mixed | 分块延续、同forward混排、in-flight防重复、KV准入；MTP抢占重放重置 | 新版mixed在相同负载下消除早期回退；公平性/延迟的更大矩阵 |
| CudaGraphManager | decode精确batch图、prefill/mixed的piecewise dense图、bucket/共享临时buffer、MTP verify/full draft/cache-only独立图 | 任意batch都命中；粗bucket没有padding代价；更多图不增加显存 |
| Qwen3.5 / GDN / attention | 混合模型与投影边界明确；稳定数值路径；纯decode recurrent pool原地更新有受控收益 | mixed获得相同pool收益；所有输入/硬件的全模型质量等价 |
| FP8 / checkpoint / GEMM | 两粒度、独立转换、舍入修复、共享量化、统一调优与大M路径 | 更细粒度必然更慢/快；所有batch都优于BF16；真实DRAM流量及任务质量 |
| KV / GDN state / prefix | 容量检查、合法slot和结束清理；MTP行号/长度一致 | 只缓存KV就能恢复GDN prefix；decode-only容量能代表完整服务容量 |
| LM head / sampler | CPU全greedy直达argmax；混合采样保留；保持B行算术及实际提交token语义；cache-only补写跳过head | 全prompt词表投影的容量开销已解决；同步可免费移出 |
| MTP | 完整缓存链路、独立BF16权重、cache-only；FP8正式B1–8映射、accept mask与全部实际大小精确图；两次交错收益曲线；B2/B4阶段与host回滚归因 | 更多输入稳定收益；BW清晰收益分界；BF16泛化；B16、5090、TP>1或随机采样已支持 |
| Benchmark / profile | 正式计时与诊断分开，保存失败、源码指纹、原值；临时专项脚本归档清理 | 一张短trace可以代表整段TPOT，或两个样本足以给近似结果稳定排序 |

## 11. 后续开发沿用的记录方式

后续新增模块时，接着追加实际发生的记录，不把计划写成完成结果。每条至少保留：日期/分支/提交或源码指纹、触发问题、改动路径、测量配置和计数口径、原始两次数据或历史实际协议、正确性范围、失败/反例、当时结论及尚未验证项。

特别保留三类没有“优化成功”的结果：功能加入后回退、单算子加速没有端到端收益、正确性修复牺牲性能。它们决定了下一次该控制哪个变量，比只列成功百分比更能解释当前实现。

涉及数据的更新应链接原始报告；新的kernel/graph/调度实验另留一组结果，不覆盖旧数据再把不同版本连成一条曲线。下一阶段最具体的待验证点，是按slot批量rollback、mixed路径、全prompt LM head与5090；B2/B4新trace已补，B1/2/4/8仍未获得跨运行、跨模型通用的收益分界。cache-only和独立两token实验已有各自对照，不预支尚未验证部分的收益。
