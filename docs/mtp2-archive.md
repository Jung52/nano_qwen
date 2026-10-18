# MTP2 实验归档

2026-10-18：本轮停止继续优化 MTP，完整代码、未提交的执行/提交协议修改、生命周期测试、开发日志及测量数据保留在 `archive/mtp2`。该分支用于复查与参考，不合并整套 MTP 实现到 main。

main 独立保留普通推理需要的通用改进：确定性 greedy fast path、完整请求元数据序列化、能力限制、按采样行进行词表投影、decode 优先预算与 slot 所有权、精确 shape 的 Graph 统计。MTP 的 draft、verify、提交、回滚、额外权重和 KV 不进入 main。

本地归档工作区为 `/home/wei/code/nano_qwen-mtp-archive`。完整历史见本分支的 `DEVELOPMENT_LOG.md`，最新独立 off/reuse 实验见 `benchmarks/reports/mtp_paired_amortization/`，此前缓存和执行边界验证见 `benchmarks/reports/runtime_contracts/`。

最后一轮已完成9配置×5配对，共90个正式样本；B2/B4/B8各有独立 off/reuse 的 prefill、稳态decode、host及CUDA事件诊断，共24个pass、12个压缩trace。正式计时未启用profiler。全部完成样本及诊断的completion严格一致，按实际提交token计吞吐；没有使用数值漂移豁免。原18条中断前样本原样保留，恢复及driver源码指纹见报告README和timing_completion.json。

当前证据只支持该卡4B per-channel、B1短输入存在清晰收益；长输入有较大波动，B2/B4没有稳定净收益，B8长输出端到端退化。它不支持默认开启，也不能外推到5090、9B或真实动态服务。停止继续投入不等于推翻已有缓存正确性修复。

DFlash2 后续从 main 的普通decode基线开始单独实验。当前仅准备分支，不添加执行接口、权重或性能承诺。
