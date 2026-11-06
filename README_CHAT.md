# nano_qwen 终端流式交互入口

基于 `main@6c49814` 的调度器、runner、模型和量化实现；没有改动算子、采样、
CUDA Graph 捕获或异步调度算法。落地时的适配与新增能力：transformers 5.x
`apply_chat_template` 返回 `BatchEncoding` 的兼容（`chat._prompt_ids`）、
`--profile-steps` 每轮 trace 采样与 overlap 分析（`chat_trace.py`）。

## 放置文件

将包内文件按相同路径放入仓库根目录。生产文件为：

- 替换 `src/nano_qwen/engine/llm_engine.py`：保留原接口，新增 `StreamChunk`、
  `generate_stream()` 及流式请求退场处理。
- 替换 `src/nano_qwen/__init__.py`：把已有 `nano-qwen` console script 接到聊天入口。
- 新增 `src/nano_qwen/chat.py`：交互循环、聊天模板、多轮历史和文本流式解码。
- 新增 `src/nano_qwen/__main__.py`：支持 `python -m nano_qwen`。

如果本地有其他修改，优先检查包内补丁，避免直接覆盖自己的文件：

```bash
git apply --check /path/to/nano_qwen_interactive.patch
git apply /path/to/nano_qwen_interactive.patch
```

补丁针对上述 main 版本，不适用于旧 mtp2 或双 slot 分支的整文件覆盖。

## 启动

在项目原有推理环境、仓库根目录运行。模型目录需含 tokenizer 和 chat template。
BF16、per-channel FP8、block-wise FP8 沿用仓库的 checkpoint 自动识别。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m nano_qwen \
  --model /path/to/qwen3.5-fp8 \
  --max-tokens 512
```

默认配置：

- `max_num_seqs=1`、TP1：只捕获 B1 的一份 decode CUDA Graph。
- `enforce_eager=False`、`async_scheduling=True`，保留异步 D2H、双输出缓冲和一批前瞻。
- prefill 默认 eager；若需要 main 的 piecewise prefill Graph，追加 `--prefill-cudagraph`。
- temperature=0 映射到当前 sampler 接受的 `1e-6`，走 deterministic greedy。
- 默认 `max_model_len=4096`、prefill token budget=1024、显存比例0.9。
- thinking 默认关闭；可通过 `--thinking` 传入模型聊天模板。

启动后出现 `Ready: B1 | decode=single CUDA Graph | async=on`。
输入问题后逐步输出；`/reset` 清空历史，`/exit` 或 Ctrl+C 退出。

```text
你> 简单解释一下CUDA Graph有什么作用。
助手> ...逐步输出...

你> 结合上一轮解释它与异步调度的区别。
助手> ...逐步输出...
```

每轮根据完整对话重新 prefill；当前未实现跨轮 KV/GDN 前缀复用。
输入历史加输出预算超出上下文时会提示 `/reset`，不会静默截断历史。

也支持单次运行和已有安装入口：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m nano_qwen \
  --model /path/to/qwen3.5-fp8 --prompt '用中文解释异步推理流水' --max-tokens 128

# 已安装项目时，原有 console script 也进入相同界面。
CUDA_VISIBLE_DEVICES=0 nano-qwen --model /path/to/qwen3.5-fp8
```

没有聊天模板的 checkpoint 可用 `--raw --prompt '...'` 做单次生成。

## 输出与异步语义

`generate_stream()` 返回新增、已经提交的 token IDs，包含用于终止的 EOS；
界面用 Transformers `TextStreamer` 缓冲不完整的字节片段和词，隐藏特殊 token。
不会对每个 token 单独 decode，也不读取异步 execution view 中的乐观长度。

默认 pipeline 使用 main 的单套 decode Graph 输入/输出 buffer 和同一 compute stream。
稳定 decode 阶段保留一批前瞻，CPU 输出/处理当前结果时下一批 GPU 工作可能已在执行；
GPU forward 之间仍按依赖顺序执行，不承诺两个 forward 并发或所有阶段都能重叠。
CPU 每次 `yield` 仅在现有 `step()` 消费结果后发生，没有增加逐 token CUDA synchronize。

流式接口要求调用开始时引擎空闲，消费结束或 `close()` 后才能提交下一请求。
EOS 后可能仍有一批在途工作；接口会先排空，再允许下一轮复用 block/slot。
在 yield 处提前关闭迭代器会取消该请求，丢弃前瞻结果并释放资源。
若模型执行中途异常或 Ctrl+C，CLI 通过 `finally: engine.exit()` 结束引擎。

每轮统计写到 stderr；正文写到 stdout。`decode graph/eager` 是该轮真实分派次数。
`first-token` 是首个 token 提交时间，decode 速率按实际新增 token（排除首 token）计数。
这些数据包含终端输出开销，用于交互观察，不替代独立性能 benchmark。
通过 `--no-stats` 隐藏统计。

参考路径：

```bash
# 关闭前瞻，仍使用单 decode Graph。
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m nano_qwen \
  --model /path/to/qwen3.5-fp8 --no-async-scheduling

# eager + 关闭前瞻，便于定位执行差异。
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m nano_qwen \
  --model /path/to/qwen3.5-fp8 --eager --no-async-scheduling
```

## 边聊天边采 trace（--profile-steps）

加 `--profile-steps N` 后，每轮对话的前 N 个 token 会在 torch.profiler 下执行，
流式输出不受影响； profiler 在第 N 个 token 提交后停止，之后的 token 不再计入。
Kineto 在引擎捕获 CUDA Graph 之前初始化，符合官方 CUDA Graph 注意事项。

每轮结束向 stderr 打一行 verdict，并在 `--profile-dir`（默认 `logs/chat_trace`）
写出 `turnNNN_*.trace.json.gz` 与同名 `summary.json`：

```text
[trace] decode_steps=10 overlap=9/9 | step=12.1 ms | gpu_idle max 1.23 ms, total 3.6% | logs/chat_trace/turn001_....trace.json.gz
```

`overlap=k/n` 表示 n 对相邻 decode replay 中 k 对的 `cudaGraphLaunch` 完整落在
前一步 forward 的 GPU 窗口内（异步前瞻生效）。分析只把拥有 ≥50 个 kernel 的
graph launch 当作 decode replay；channel-fp8 的 `_per_channel_mm` 单 kernel
微 graph 计入 GPU busy 但不参与配对。GPU idle 只统计首个 decode replay 之后的
decode 阶段，不含 tokenizer/prefill 准备空隙。分析是纯 Python，可离线重跑
（`nano_qwen.chat_trace.analyze_trace`）。profiler 会拖慢被采样的十几步
（kernel 级 CUPTI 开销），只作调度诊断，不作性能结论。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m nano_qwen \
  --model /path/to/qwen3.5-fp8 --max-tokens 2048 --profile-steps 12
```

token 正确性由两部分保证：肉眼可见的流式正文，以及 greedy 确定性——
同一 prompt 在 `--no-async-scheduling` 下重跑应逐 token 一致；
引擎级 async vs 串行参考的逐位对比见 `benchmarks/bench_one_batch_trace.py`。

## 验证

已完成语法检查、`python -m nano_qwen --help`、32项控制流/CLI测试与5项参数子测试。
本次环境没有 PyTorch/CUDA；本地控制流测试仅替换不可用的 GPU/model/config 导入边界，
实际执行新的 engine、原 scheduler、原 AsyncDecodeCohort、block manager 和真实 TextStreamer。
未运行真实 GPU 推理，不把上述结果写作 GPU 数值或性能验收。

在项目正常推理环境执行：

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_async_decode_pipeline.py tests/test_chat_streaming.py tests/test_chat_cli.py

# 真实 tiny hybrid：B1 Graph、串行/异步的 completion 与 GDN 状态逐位比较，
# 包含长度63/255/257、chunk=64、资源释放与Graph命中；无需下载完整模型。
PYTHONPATH=src python -m pytest -q tests/test_chat_cuda.py
```

完整模型 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m nano_qwen \
  --model /path/to/qwen3.5-fp8 --prompt '请只回答：你好，世界。' \
  --max-tokens 64 --max-model-len 2048
```
