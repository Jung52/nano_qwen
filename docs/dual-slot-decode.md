# Moondream式双slot decode实验分支

基线：`Jung52/nano_qwen@db22de322bf98e4a5707ea95c0d378d7bc6e03ad`。
本地分支：`feat/moondream-dual-slot`。默认关闭，尚未执行CUDA动态验收或性能测量。

## 核心边界

- `DecodeSlot 0/1`是临时执行槽，拥有独立输入暂存、block table、temperature、GPU/CPU输出及Event。
- 每个执行槽独立捕获decode Graph，使用独立Graph pool、静态输入和hidden输出。
- 两槽forward与sampling使用同一个compute stream；D2H使用另一个copy stream，等待本步`sample_done` Event。
- `InputBatch persistent slot`继续表示请求身份。KV和GDN state pool不复制，不增加另一套请求slot分配器。
- 循环为`launch forward t+1 -> commit t -> sample t+1`。一个执行槽要等CPU提交完旧结果才释放。
- `RequestLease.frontier`表示已预留的计算位置；`StepRow`保存本轮的count、起点及私有Sequence执行视图。CPU提交旧步不会覆盖新步元数据。
- EOS以已有`SequenceStatus.FINISHED`表示finalized；`inflight_refs`归零后才释放KV和InputBatch slot。已结束请求随下一轮运行的结果被丢弃，不再追加token。
- 已知的`max_tokens`会计入pending sample预算，避免为确定的长度上限多派发一次。

## 第一版范围

支持TP=1、普通文本采样、eager及精确batch decode Graph；沿用BF16及两种FP8模型路径，但CUDA验收仍需在实际环境执行。

新请求prefill和chunked prefill均作为排空边界，尚未把prefill/mixed forward放入双slot重叠流水。原有piecewise prefill可以使用，但运行前会排空pending decode。此版本没有实现grammar mask接口、抢占恢复或多卡执行；KV不足时先排空并尝试让已有decode推进，仍无法推进则明确报错。Prefix cache模式拒绝启用。

解码forward已使用两套工作区；LM head、sampler仍沿用现有算子及其临时分配，因此不宣称热路径零分配。新增Graph pool增加驻留显存，先以小batch和保守显存预算验收。

## 修改入口

| 文件 | 职责 |
|---|---|
| `engine/decode_slot.py` | 两套暂存区、输出和Event |
| `engine/pipeline.py` | 计算frontier、step快照、FIFO提交与延迟释放 |
| `engine/model_runner.py` | `launch_slot/sample_slot/release_slot`及slot内输入准备 |
| `engine/cuda_graph.py` | 两个独立decode Graph bank |
| `engine/block_manager.py` | 按计划位置预留KV页，覆盖未提交token |
| `config/config.py` | `dual_slot_decode=False`开关及prefix guard |
| `engine/llm_engine.py` | 选择实验流水；原执行路径保持默认 |

## 使用

```python
from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.sampling_params import SamplingParams

engine = LLMEngine(
    "/path/to/Qwen3.5-4B",
    dual_slot_decode=True,
    tensor_parallel_size=1,
    max_num_seqs=4,
    max_num_batched_tokens=2048,
    max_model_len=512,
    gpu_memory_utilization=0.65,
    use_prefill_cudagraph=False,
    enable_prefix_cache=False,
)
try:
    result = engine.generate(
        ["解释CUDA Graph的作用。"],
        SamplingParams(temperature=1e-6, max_tokens=32),
        use_tqdm=False,
    )
finally:
    engine.exit()
```

`max_model_len`必须覆盖`prompt_tokens + max_tokens - 1`；计算frontier不会使用尚未提交的CPU token作为输入，decode输入来自共享GPU sampled-token pool。

## 验证

不需要torch/CUDA的真实控制平面测试：

```bash
python tests/test_dual_slot_state.py -v
```

本次结果：9项通过；Python语法编译、`git diff --check`通过。没有安装torch/CUDA，因此既有GPU回归、模型输出、Graph replay和加速收益尚未执行。

实际环境先eager，再Graph；对BF16与各FP8 checkpoint分别执行：

```bash
PYTHONPATH=src python benchmarks/validate_dual_slot.py \
  --model /path/to/model --eager --json-out reports/dual_slot_eager.json

PYTHONPATH=src python benchmarks/validate_dual_slot.py \
  --model /path/to/model --json-out reports/dual_slot_graph.json

PYTHONPATH=src python -m pytest -q
```

验证脚本比较原路径与双slot的greedy输出，覆盖B1/2/4、255/256/257-token KV页边界、max_tokens=1、重复请求复用；另外强制第二个token为EOS，检查zombie排空和资源回收。Graph模式检查两个bank的地址和pool互异。它不测性能；数值分歧保留在JSON，不自动归类为near-tie或放宽判定。

性能阶段另做相同配置的独立A/B，固定batch、生成长度和预热，记录真实CUDA kernel及memcpy时间线。重点检查CPU commit是否落在下一forward期间，以及D2H、H2D、slot复用有无引入新等待。GPU kernel流依然串行，不能用两个slot推断GPU并行或承诺吞吐提升。

参考：[Moondream官方Popping the GPU Bubble](https://moondream.ai/blog/popping-the-gpu-bubble)、[官方Pipeline状态机](https://github.com/m87-labs/kestrel/blob/8e6fb899bb8a076f3235b6a5ab832a78c031c77b/kestrel/scheduler/pipeline.py)。本补丁按nano_qwen的Sequence/KV/GDN接口独立实现。
