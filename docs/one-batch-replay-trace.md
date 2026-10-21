# 固定batch：CPU replay与GPU双slot完整trace

## 先说明这次上传的文件

`dual_slot.trace.json(1).gz`中存在60次真实CPU `cudaGraphLaunch`、91800个GPU kernel、
720个拷贝事件，以及slot0/1各30个Graph范围。原文件的header明确为`keep_cpu_ops=false`，
所以CPU `aten`算子已过滤；GPU kernel没有全部删除。

GPU轨道还叠加了720个自动或手工annotation，容易把Graph范围挤在多层范围之间。
本包将annotation放到单独显示轨道，Graph父范围和内部kernel留在同一个原始CUDA stream。
已经从旧文件过滤掉的CPU算子不能靠导出恢复，需要按下面的程序重新采集。

## 放入仓库的文件

无需修改Scheduler、ModelRunner、CUDA Graph manager或融合kernel。只新增：

| 路径 | 用途 |
| --- | --- |
| `benchmarks/slot_trace_full.py` | 完整导出，不过滤CPU算子、CUDA API、GPU kernel、copy或flow |
| `benchmarks/bench_one_batch_slots.py` | 类似bench_one_batch的固定负载实验，采集时临时标记真实replay |
| `tests/test_replay_trace.py` | 10个无需CUDA的完整性与临时包装恢复测试 |
| `tests/test_replay_trace_cuda.py` | 不加载模型的真实CUDA Graph replay/profiler冒烟测试 |

依赖仓库已有`benchmarks/bench_fp8_pressure.py`的工作负载KV分配工具，
以及已经验证过的双slot分支。默认不改变融合开关；`--fused-ops 0/1`可明确选择。
本包不修改之前融合路径的数值验收标准，也不解决尚未定位的融合token差异。

## 测试后采集

将包内文件按目录放到仓库，在仓库根目录运行：

```bash
conda activate nano-qwen-fp8-test
python -m pytest -q tests/test_replay_trace.py tests/test_replay_trace_cuda.py
python benchmarks/bench_one_batch_slots.py \
  --model /home/wei/code/models/4b-channel-fp8 \
  --mode both \
  --output-dir reports/one_batch_slots
```

默认B8、输入128、输出256，每种模式预热1次、采集3轮。single和dual各用独立进程，
避免两个engine在一个进程里反复初始化时遗留Graph pool占用。

每轮完整请求会生成256token并排空，只将稳态16步decode及前后各2步guard纳入profiler。
三个轮次分别导出，避免把未采集或导出耗时混在一份合并trace里。
这里控制的是采集长度，采集窗口内部的CPU算子和GPU kernel全保留。

只看双slot、先试1轮：

```bash
python benchmarks/bench_one_batch_slots.py \
  --model /home/wei/code/models/4b-channel-fp8 \
  --mode dual --profile-rounds 1 \
  --output-dir reports/one_batch_dual
```

需要覆盖几乎整个decode时可设置：

```bash
python benchmarks/bench_one_batch_slots.py \
  --mode both --skip-decode 2 --profile-decode-steps 251 \
  --output-dir reports/one_batch_long
```

长窗口会显著增加profiler内存、文件体积及采样开销。默认16步足够观察slot交替。
`--record-shapes`和`--with-stack`按需开启，默认关闭；它们不决定是否保留算子。
`--keep-raw`会额外保留Kineto原始trace的gzip副本，用于对照。
若需要正式计时，加`--timing-rounds 3`；计时轮次在profiler以外单独运行。

## 直接打开哪些文件

默认`--mode both`的输出结构为：

```text
reports/one_batch_slots/
  summary.json
  single/
    single.summary.json
    single.r1.trace.json.gz
    single.r2.trace.json.gz
    single.r3.trace.json.gz
  dual/
    dual.summary.json
    dual.r1.trace.json.gz
    dual.r2.trace.json.gz
    dual.r3.trace.json.gz
```

使用Perfetto打开`dual.r1.trace.json.gz`，不需要再运行任何后处理命令，也不再区分overview/detail。

| 显示位置 | 能看到的内容 |
| --- | --- |
| CPU engine | `engine.step`、prepare、forward、`CUDAGraph.replay/slot0/bs8`、真实`cudaGraphLaunch`、sample、commit及CPU aten |
| CPU cudaGraphLaunch显示轨道 | 将真实CUDA API区间额外列出并带上slot，便于直接观察发射顺序 |
| CUDA compute stream | `cuda_graph_slot0`与`cuda_graph_slot1`交替，内部保留原始kernel |
| CUDA copy stream | 真实D2H及其他copy事件 |
| GPU profiler scopes | 原有GPU annotation，移到这里以免遮挡kernel与Graph范围 |

CPU的Graph launch显示轨道是同一批真实API事件的显示副本，不代表第二条CPU执行线程。
原API仍保留在真实CPU线程上，显示副本使用独立category，不计为又一次实际Graph launch。
GPU父范围由CUPTI correlation关联Graph的真实kernel，再取GPU起止时间生成。
导出时将父范围放在同起点的子kernel前面，使Perfetto正确处理这些相同开始时间的嵌套事件。
它们是显示包络，不能与内部kernel时长重复相加；未把CPU replay耗时当成GPU运行时间。

Graph replay会跳过捕获时的Python与逐算子CPU dispatch，因而不能要求每次replay里再次出现
Graph内部的CPU `aten::pow/mean/...`调用。内部计算仍以GPU kernel形式存在。
本程序保留本次实际发生的CPU算子，不伪造Graph内部CPU算子。
若要看Python算子与Graph节点的来源对应关系，这是另一类capture归因诊断。

## 自动检查

每轮检查真实replay包装次数、CUPTI launch次数和GPU kernel关联是否一致，
并检查dual的slot0/1是否交替；缺GPU kernel时直接失败，不用CPU时长伪造GPU范围。
检查全部请求完成、KV页和请求slot释放，以及双slot prime/drain完成。
推进窗口依据已提交token数，而非固定调用N次engine.step，适配双slot额外的启动与排空。
两种模式都执行时，额外核对prompt、token输出、源码及KV容量一致。

## 交付前检查

10个新CPU测试、9个既有双slot状态机测试通过；Python语法检查通过。
对上传的真实trace重新走导出流程，60次launch全部关联，slot各30次且严格交替；
全部原始kernel、copy、CUDA API和CPU语义事件逐条保持一致。
本地环境没有PyTorch、Triton和CUDA，新的真实CUDA冒烟测试与整机采集未执行，需在项目环境运行。

参考：
- PyTorch CUDA Graph replay：https://docs.pytorch.org/docs/stable/notes/cuda
- PyTorch Graph kernel annotation：https://docs.pytorch.org/tutorials/advanced/cuda_graph_annotations_tutorial.html
