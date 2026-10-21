# 归一化融合与slot分组实验

基于当前`feat/moondream-dual-slot`工作区。只增加RMSNorm系列的融合和实验工具，
保留现有双slot调度、KV/GDN生命周期和Graph bank。此补丁要求原双slot实现已经存在。

## 使用

在仓库根目录，用项目已有的`nano-qwen-fp8-test`环境执行：

```bash
conda activate nano-qwen-fp8-test
python benchmarks/bench_fused_slots.py \
  --model /home/wei/code/models/4b-channel-fp8 \
  --output-dir reports/fused_slots_b8
```

一条命令依次完成：

1. CUDA数值测试、两组独立Graph输入/输出测试、trace导出测试。
2. 独立进程的归一化微基准，输出`norm_micro.json`。
3. 四组独立进程实验：`reference_single`、`fused_single`、`fused_dual`、`reference_dual`。
   每组B8/I128/O256，预热1轮，正式计时3轮，再单独生成3轮profiler窗口。
4. 自动比较相同prompt下的所有输出token，并生成`summary.json`。

默认decode使用CUDA Graph、prefill使用eager，TP=1。加`--eager`可单独验证eager模式。
各组使用相同的工作负载KV容量和生产sampler；测量包含双slot的启动及排空步骤。
不在每一步插入`cuda.synchronize()`，仅在共同的prefill/decode阶段边界同步。
正式计时没有profiler；采集窗口中的耗时不用于计算加速比。
三轮结果仅供对照，不能替代更多重复实验来判断微小差异。

只运行某部分：

```bash
python -m pytest -q tests/test_fused_norm.py tests/test_slot_trace.py
python benchmarks/bench_fused_norm.py --output reports/norm_micro.json
python benchmarks/profile_slots_b8.py --fused-ops 1 --output-dir reports/fused_slot_traces
```

## 修改内容

| 文件 | 作用 |
| --- | --- |
| `src/nano_qwen/layers/fused_norm.py` | 单kernel完成RMSNorm、GemmaRMSNorm、残差加归一化、GDN门控归一化 |
| `src/nano_qwen/layers/layernorm.py` | 满足CUDA推理条件时调用融合kernel，保留原PyTorch路径 |
| `benchmarks/slot_trace.py` | 按CUPTI关联号直接生成GPU slot范围，无torch依赖 |
| `benchmarks/profile_slots_b8.py` | 采集结束直接导出分组detail和overview，分离无profiler计时 |
| `benchmarks/bench_fused_slots.py` | 四组实验及逐token验收，自动输出加速比 |
| `benchmarks/bench_fused_norm.py` | 原实现/融合实现的Graph replay微基准 |
| `tests/test_fused_norm.py` | 63个CUDA数值、布局、残差、门控及双Graph输出独立性用例 |
| `tests/test_slot_trace.py` | 9个CPU用例，检查GPU时间、关联、跨窗口时钟、事件保留及验收判定 |

融合保留这些计算边界：

- FP32计算方差。带残差时，对**未提前舍入的FP32相加结果**计算方差。
- 归一化后先转回输入dtype，再乘权重。Gemma使用`1+weight`，并保留其dtype舍入。
- GDN门控归一化先完成输入dtype下的权重乘法，再乘FP32 SiLU gate，最后转回输入dtype。
- Q/K head norm的二维行跨度可以大于head维度，支持现有reshape后的布局。
- 训练、CPU及不支持的dtype/布局走原路径。

`NANO_QWEN_FUSED_OPS=0`在**import模型前**设置，即可退回参考实现；默认启用融合。
GPU reduction的运算顺序可能与PyTorch不同。单算子测试检查数值误差，整机实验仍然
要求greedy输出逐token一致；任何mismatch都会令总验收失败，不会被性能收益抵消。

本轮未融合GDN的conv/state更新、FP8 GEMM或LM Head；这些算子仍会出现在detail中。
先通过本轮实验判断归一化实际收益，再决定下一组融合目标。

## 实验输出与阅读方式

每个配置目录直接获得：

- `<single_slot或dual_slot>.overview.trace.json.gz`：CPU语义范围与GPU slot/Graph/LM Head/sample/D2H范围。
- `<single_slot或dual_slot>.trace.json.gz`：上述范围加全部物理GPU kernel、拷贝和CUDA API。
- `<single_slot或dual_slot>.report.json`：三轮无profiler数据、输出token、窗口调度及kernel统计。

用Perfetto先打开overview，查看`cuda_graph_slot_0/1`在compute stream上交替，
以及copy stream上的`D2H_slot_0/1`。再打开detail展开内部kernel。
默认隐藏大量CPU→GPU关联箭头；需要时`profile_slots_b8.py --keep-flows`保留在detail中。

GPU父范围在**本次采集的自动导出流程**中生成：
CPU slot范围→其中CUDA API的correlation→物理GPU事件→GPU起止时间。
它们使用GPU时钟，保留原始pid/tid与物理事件，既不代表新kernel，也不能和子kernel时长相加。
导出采用两次流式扫描，只保存关联及区间摘要，不一次性载入完整大JSON。
无需再执行后处理命令。此方式不要求升级到支持原生Graph annotation的新CUDA版本。

默认每轮记录16个稳态decode步骤，前后各保留2步guard；完整256token仍会生成并排空。
三轮窗口之间的空白是未采集的生成/导出时间，不能当成GPU空闲。

`summary.json`给出三轮中位数，以及四种独立对照：单slot融合收益、双slot融合收益、
参考实现的双slot收益、融合实现的双slot收益。查看`graph_kernels_per_replay`确认Graph内
kernel数量是否降低，查看`fused_norm_kernels`确认实际使用了新kernel。
`norm_micro.json`的微基准加速比只代表归一化，不能直接当成整机加速比。

## 当前验证范围

交付环境没有PyTorch、Triton或CUDA，无法执行新kernel、GPU数值用例或真实吞吐实验。
已通过Python语法检查、9个新CPU用例和9个原双slot状态机用例。
用已上传的真实trace检查自动导出：60次Graph replay全部获得slot范围，
120300个kernel与720个拷贝事件全部保留，Graph范围包含119700个kernel。
这是导出链路验证，不是融合后的新GPU实验结果。

参考：
- Triton LayerNorm示例：https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
- Triton精确除法接口：https://triton-lang.org/main/python-api/triton.language.html
