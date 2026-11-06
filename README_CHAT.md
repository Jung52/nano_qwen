# 终端流式交互

在仓库原有 CUDA 推理环境、仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m nano_qwen \
  --model /path/to/qwen3.5-fp8 --max-tokens 512
```

支持 BF16 和仓库已支持的 FP8 checkpoint；模型目录需包含 tokenizer 和聊天模板。
入口默认 B1、单套 decode Graph buffer、`async_scheduling=True`、prefill eager、greedy。
B1 是单终端每次一个活跃请求的入口策略，不是消费卡限制，也不是异步流水的必要条件。
`/reset` 清空历史，`/exit` 或 Ctrl+C 退出；多轮会重新 prefill 完整历史，尚未跨轮复用 KV/GDN。

```bash
# 单次生成
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m nano_qwen \
  --model /path/to/qwen3.5-fp8 --prompt '简单解释CUDA Graph' --max-tokens 128

# 关闭前瞻，保留 Graph；eager 参考再追加 --eager。
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m nano_qwen \
  --model /path/to/qwen3.5-fp8 --no-async-scheduling
```

`--thinking` 传给聊天模板；默认关闭。`--prefill-cudagraph` 可开启 piecewise prefill Graph。
`--max-model-len` 默认4096，`--max-num-batched-tokens` 默认1024，显存比例默认0.9。
temperature=0 映射到当前 sampler 的1e-6 greedy 路径。
正文写到 stdout，统计写到 stderr；`--no-stats` 可隐藏统计。
交互统计包含终端输出开销，不能替代独立性能 benchmark。

## 可选 trace 采集

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m nano_qwen \
  --model /path/to/qwen3.5-fp8 --profile-steps 16 --profile-dir logs/chat_trace
```

Kineto 在 Graph 捕获前初始化；前 N 个已提交 token 的窗口结束时停止 profiler，
本轮生成结束后才导出 raw trace 和 metadata。正常聊天不开 profiler、不安装诊断 wrapper。
采集、停止或导出失败会报告诊断错误，并保留聊天流程；被采集轮次不作性能结论。

每次实际 `execute_model` 有独立 `nq/.../bNNNNN` batch ID。同一次消费调用内提交两批
也有两个 ID，不再用 `next(chunks)` 的范围充当 GPU forward 范围。
只给真实 decode Graph 对象的 replay 标记 `launch`；FP8 微 Graph 不按 kernel 数量猜测。

分析和绘图统一到 benchmark 入口，离线运行不需要 CUDA/PyTorch：

```bash
python benchmarks/bench_one_batch_trace.py \
  --analyze logs/chat_trace/turn001_XXXXXXXX.trace.json.gz
```

输出 summary.json 和 timeline.svg。`OVERLAP` 要求下一次 CPU launch 与前一步实际
GPU kernel 有区间交集；`PREQUEUED` 表示 launch 早于前一步 GPU 开始；
`WITHIN_FORWARD_SPAN` 表示位于 forward 时间跨度内但没有物理 kernel 交集。
缺少 dispatch 标记或 CUPTI correlation 时报告 unknown，不根据 CPU scope 猜测 GPU 时间。
旧版仅含 `nqc/step` 的 trace 无法可靠恢复多 dispatch 的归属，需重采集。

`device_idle` 统计 kernel、memcpy、memset 的活动并集；仍需结合原始 Perfetto 时间线看原因。
计算流按依赖顺序执行，单 Graph async 不表示相邻两个 forward 在 GPU 上并发。

## 验证

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_async_decode_pipeline.py tests/test_chat_streaming.py tests/test_chat_cli.py \
  tests/test_profiling.py tests/test_trace_tools.py tests/test_benchmark_modes.py

# 真实 tiny hybrid：串行/异步 completion、GDN 状态和 Graph 命中。
PYTHONPATH=src python -m pytest -q tests/test_chat_cuda.py
```

`generate_stream()` 只输出权威请求已提交的 token，EOS 后排空在途结果再释放 slot。
调用开始要求 engine 空闲；消费结束或 close 后再发下一请求。执行中途异常由 CLI 退出 engine。
这些调度、状态和采样语义在本轮诊断清理中保持原实现。
