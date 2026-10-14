本次仅测decode，比较BF16、per-channel FP8、block-wise FP8。

协议：batch=64/128/256/512逐档增加，遇真实CUDA OOM后按16个请求细化容量边界。每个可用粗档、各方案最大batch以及BF16最大batch上的三方案各保留两次正式样本；已完成样本不覆盖、不因波动增加第三次。

每请求同一个起始token，prefill一次后63步不计时decode建立真实KV/GDN状态。正式计时128步、context_lens=65→192，随后独立收尾一步。原生engine.step、decode CUDA Graph、LM head、随机采样、async output和queue=2，TP=1，prefix cache关闭。加载、三次model kernel warmup、autotune、capture、完整生成预热均在正式计时外。

为控制显存，仅在benchmark中固定KV为每请求一个256-token block，并按生产graph输入/输出契约只捕获当前batch；部署默认还捕获其它batch，其显存上限可能更低。不修改生产路径。本测试的最大batch受短上下文协议约束，不代表长上下文容量。

历史执行脚本已归档清理；协议和所有正式样本保持原值。当前入口见 [benchmarks/README.md](../../README.md)，清理前源码位于本地 `logs/fp8-release-cleanup-2026-10-14/before-cleanup.zip`。

记录：隔离worktree源代码与主目录生产源码逐文件一致；新增窗口、边界和同batch比较测试先失败再通过。原始数据保存模型/源码/软件环境指纹、CUDA OOM日志、两次样本和autotune缓存。

收尾审查发现FP32 GDN调试环境变量会改变容量；已加入BF16 state约束和身份记录，新增回归先失败再通过。测试时关闭该调试模式。

WSL2实测调整：未限制allocator的BF16 batch128预热时FB显存仅余约41MiB、耗时明显异常，疑似驱动系统内存回退；在产生正式样本前停止。三方案改为统一90%总显存的PyTorch allocator预算，让容量边界表现为OOM。最大batch结论限定于该预算及16请求对齐。此前BF16 batch64两次样本峰值reserved=12610MiB低于该预算，保留原值及原源码/身份归档；provenance.json记录适配，计时窗口和kernel不变，不重跑该对照。

完成：10配置/20正式样本，M/Graph/phase/完成token数全部匹配；两次原始BF16 batch64 SHA不变，现场源码身份匹配。BF16容量96、两种FP8容量256，相邻失败点112/272。114项测试通过；最终只读审查的FP32调试状态约束已修复，其余未发现实质问题。

权重口径修正：raw weight_mib逐Parameter累加包含tied embedding/head重复引用；扣除共享storage后BF16约7.83GiB、两种FP8约4.52GiB。结果页的曲线、完整身份以及weight_accounting为CPU后处理附加内容；原始计时JSON保持原值。
