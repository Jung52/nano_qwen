# GDN recurrent-state 原地更新（2026-10-15）

将纯decode的recurrent-state gather/scatter改为FlashInfer pool+indices原地更新；两份源码快照只有 `gated_delta_net.py` 和 `cuda_graph.py` 不同。当前工作区已采用新路径。

RTX4070TiSUPER / WSL2，输入64、输出128（prefill产生首token，decode127步），B=8/64，per-channel/block-wise。旧/新路径每配置完整预热一次后正式计时**两次**，各配置独立进程，交替先后顺序。prefill/decode均用graph，queue depth2、异步输出、禁用prefix cache、ignore EOS、固定KV容量、90% allocator预算。

## 关闭profiler的正式计时

| 方案 | B | 旧TPOT ms | 新TPOT ms | TPOT降低 | 旧token/s | 新token/s | 吞吐提升 |
|---|---:|---:|---:|---:|---:|---:|---:|
| per_channel | 8 | 13.308 | 12.069 | 9.3% | 601.2 | 662.9 | 10.3% |
| per_channel | 64 | 30.205 | 20.557 | 31.9% | 2118.9 | 3114.2 | 47.0% |
| block_wise | 8 | 13.536 | 12.146 | 10.3% | 591.0 | 658.6 | 11.4% |
| block_wise | 64 | 31.269 | 20.257 | 35.2% | 2048.2 | 3159.5 | 54.3% |

表中为两次算术平均，[全部16次原值](timing_samples.csv)另存；[均值与比例](comparison.csv)可复算。未因波动追加第三次。prefill数值也保留在原始数据中，但这次改动针对decode。所有正式样本均确认1轮完整prefill和127轮精确batch的decode graph，无回退、抢占、OOM或allocator重试。

## B=64复制消失的证据

另作两次decode-only profile，每次记录最初8步、context65…72；prefill和余下119步在profiler外完成。这些窗口只用于归因，不代替完整127步TPOT。按kernel名称及grid区分recurrent state与conv state/embedding；旧路径recurrent gather的grid=[64,256,1]，scatter的grid=[65536,1,1]。

| 方案 | 路径 | 轮次 | recurrent gather次数 | scatter次数 | 复制ms/步 | GDN核心ms/步 |
|---|---|---:|---:|---:|---:|---:|
| block_wise | after | 1 | 0 | 0 | 0.000 | 4.624 |
| block_wise | after | 2 | 0 | 0 | 0.000 | 4.572 |
| block_wise | before | 1 | 192 | 192 | 10.938 | 5.021 |
| block_wise | before | 2 | 192 | 192 | 9.784 | 4.646 |
| per_channel | after | 1 | 0 | 0 | 0.000 | 4.483 |
| per_channel | after | 2 | 0 | 0 | 0.000 | 4.410 |
| per_channel | before | 1 | 192 | 192 | 9.861 | 4.669 |
| per_channel | before | 2 | 192 | 192 | 9.719 | 4.565 |

旧路径每个窗口192次gather与192次scatter，即24层×8步；新路径两项都为0。卷积状态的小拷贝仍存在。按BF16状态尺寸计算，B64的24层recurrent state共1536MiB，移除gather/scatter减少约6GiB/步的额外逻辑读写；GDN核心自身的必要读写仍在，此数字不是DRAM计数。

## 实现与验证

- `decode_gated_delta_rule`增加可选GPU slot索引，传完整pool给FlashInfer；引擎int64索引转换为小batch ABI要求的int32，保留原dense调用供mixed路径使用。
- 纯decode直接更新recurrent pool，仅conv state继续取出/写回；删除不再调用的旧状态复制helper。
- CUDA Graph预热/捕获使用互异合法slot，replay覆盖同一GPU索引缓冲区。
- 新增4项CUDA回归：B8/B64稀疏乱序slot、slot0与slot>B、多次graph replay变更索引，输出和完整BF16状态池与dense参考逐位一致；实际GDN forward禁止recurrent pool的index_select/index_copy_；slot重置不影响其他请求；捕获预热覆盖独立slot。
- 完整回归：`python -m pytest -q`，98 passed（20条依赖库告警）。
- 依赖本机FlashInfer的BF16 pool+indices接口，沿用同一GDN算术后端；不修改量化粒度和状态dtype。当前模型测试是性能smoke，逐位验证覆盖的是GDN核心与既有混合路径回归，不是完整模型质量评测。

## 原始记录

- [results.json](results.json)：完整计时、profile、内存、graph路径、源文件hash和GPU遥测。
- [profile_samples.csv](profile_samples.csv)：两次profile的复制次数/时间及GDN核心时间。
- `logs/gdn-state-pool-2026-10-15/`：before/after源码快照、运行脚本、两份源码manifest、全部日志及Perfetto trace。
- 入口复用`benchmarks/bench_one_batch.py`：`--batch-size 64 --input-len 64 --decode-steps 127 --max-model-len 256 --timing-rounds 2 --profile-rounds 2 --profile-decode-steps 8 --trace-stage decode --workload-memory`，分别指向两种模型；B8设置`--profile-rounds 0`。
