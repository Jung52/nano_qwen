# LMCache CPU RAM 状态交接原型

基线：`Jung52/nano_qwen@7049b3b5f54f9a13ceab2bc26914a4cf792854c8`。

这一版将完整请求快照保存到真实的 LMCache 0.5.6 CPU L1 内存池，再由 Decode
实例恢复。缓存池使用 LMCache MP 存储核心的 `L1Manager`；传输采用本项目的二进制
HTTP 协议，**不是官方 LMCacheMPConnector 协议、RDMA 或共享内存零拷贝**。
服务不含字典/文件数据后备路径；小型目录只记录对象大小、校验值和过期时间。

## 范围

- 单请求、TP=1，同步交接；原有普通推理路径不变。
- 支持 Qwen3.5 attention KV、全部 GDN conv/recurrent state 和请求进度。
- `async_scheduling=False`、`enable_prefix_cache=False`；初版要求
  `ignore_eos=True`、`max_tokens>=2`，用于固定长度正确性验证。
- P 完成完整 prompt，采样首 token，提交快照后释放 GPU 状态。
- D 分配自己的 block/slot，恢复 GPU token 输入，从下一步 decode 开始。
- 未实现并发路由、自动 P/D 池调度、部分前缀命中、RDMA、零拷贝、层间流水。
- CPU RAM 设有容量上限；待交接快照不自动被 LRU 淘汰，满时明确失败。
  D 成功恢复后 ACK 删除；遗留快照默认 900 秒后在下一次服务操作时回收。

## 1. 安装修改

在解压的交付包目录执行（先替换仓库路径）：

```bash
python apply_lmcache_pd.py --repo /path/to/nano_qwen --check
python apply_lmcache_pd.py --repo /path/to/nano_qwen
```

安装器先检查全部目标文件。已有核心文件必须与基线匹配；会自动兼容 CRLF/LF。
如存在其他本地修改则停止，不覆盖。备份位于仓库的 `logs/lmcache-pd-backup-*`。
包内 `overlay/` 同时包含完整修改文件；不要使用覆盖解压来绕过基线检查。

## 2. 独立安装缓存服务环境

建议推理环境继续使用现有 PyTorch/CUDA，缓存服务单独建 CPU 环境。
LMCache wheel 含编译扩展，这里固定其版本和构建匹配的 PyTorch 2.13。
从 nano_qwen 仓库根目录执行：

```bash
python3.12 -m venv /tmp/nano-lmcache-env
/tmp/nano-lmcache-env/bin/python -m pip install --upgrade pip
/tmp/nano-lmcache-env/bin/python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
/tmp/nano-lmcache-env/bin/python -m pip install -r requirements/lmcache-pd-server.txt
```

服务依赖安装可能下载 CUDA 相关 Python 包；CPU 缓存测试不要求 GPU。
采用独立环境是为了不改变推理进程的 Torch/FlashInfer/Triton 依赖。
使用其他 LMCache 版本会明确报错，需要重新验证 L1 API，不能静默切换后端。

## 3. 先跑 CPU 回归

```bash
REQUIRE_LMCACHE=1 PYTHONPATH=src /tmp/nano-lmcache-env/bin/python -m pytest -q tests/test_lmcache_pd_state.py tests/test_lmcache_pd_store.py
```

`REQUIRE_LMCACHE=1` 使缺依赖直接失败。状态测试通过 AST 加载仓库的实际
Sequence/InputBatch/BlockManager/Scheduler/LLMEngine 类，避开 GPU 模块初始化；
模型执行用确定性 CPU 状态替代，不声称完成真实模型验证。
存储测试实际分配 LMCache L1 内存，并验证提交、校验、删除、容量耗尽和 HTTP 交接。

## 4. 启动缓存服务

终端 A，仓库根目录：

```bash
PYTHONPATH=src /tmp/nano-lmcache-env/bin/python -m nano_qwen.cache.service --size-gb 1 --port 8765
```

这是实验用本机服务，默认只监听 `127.0.0.1`。第一次先使用默认绑定。
`--size-gb` 指 LMCache L1 池；HTTP/序列化还会产生临时副本，进程 RSS 会高于此值。
服务必须在 P 退出和 D 启动期间持续运行，重启会丢失待交接快照。

终端 B，在已有推理环境运行 1 MiB 存储烟测（也可用缓存服务环境运行）：

```bash
PYTHONPATH=src python benchmarks/test_lmcache_pd.py --mode store-smoke --output logs/lmcache_pd/store_smoke.json
```

必须看到 `passed=true`、`backend=lmcache_l1`、`real_l1_allocation=true`。

## 5. 一张 GPU 的真实模型测试

终端 B，使用 nano_qwen 原推理环境，把模型路径换成实际 BF16/FP8 checkpoint：

```bash
PYTHONPATH=src python benchmarks/test_lmcache_pd.py --mode roundtrip --model /path/to/4b-channel-fp8 --prompt-lens 64 256 257 512 --output-len 32 --output logs/lmcache_pd/roundtrip_eager.json
```

默认 eager，先验证它，再加 `--cuda-graph` 单独验证图路径：

```bash
PYTHONPATH=src python benchmarks/test_lmcache_pd.py --mode roundtrip --model /path/to/4b-channel-fp8 --prompt-lens 64 256 257 512 --output-len 32 --cuda-graph --output logs/lmcache_pd/roundtrip_graph.json
```

每组先运行不中断 reference，再执行 P→LMCache→D。
测试占用一个目标 block/slot，强制恢复到另一位置；检查：

- `state_bytes_equal=true`：KV、GDN、token 进度完整往返一致。
- `tokens_equal=true`：与不中断 reference 的固定长度 greedy 输出完全相同。
- `prefill_dispatches=0`：恢复后没有重新执行 prompt。
- `decode_dispatches=output_len-1`：首 token 已在 P 端生成，D 不重复处理它。
- 整体 `passed=true`；任一断言失败则进程非零退出，并保留失败 JSON。

模型身份第一次计算会读取所有 checkpoint 权重并哈希，放在计时外。
两端必须使用相同权重、模型配置、量化方式和推理软件版本；快照不支持跨格式转换。
256/257 等长度同时覆盖完整 KV 页和尾页，防止对齐边界多分配一个 block。

## 6. 两个独立进程测试

下面命令可以先后在同一 GPU 上运行，因此单卡也能测试进程边界；它不代表双卡并行收益。
P 退出前已把快照提交到仍在运行的缓存服务：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python benchmarks/test_lmcache_pd.py --mode prefill --model /path/to/4b-channel-fp8 --prompt-lens 257 --output-len 32 --transfer-id pd257 --with-reference --output logs/lmcache_pd/prefill257.json
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python benchmarks/test_lmcache_pd.py --mode decode --model /path/to/4b-channel-fp8 --transfer-id pd257 --reference-json logs/lmcache_pd/prefill257.json --output logs/lmcache_pd/decode257.json
```

两张 GPU 时，第二条命令可改为 `CUDA_VISIBLE_DEVICES=1`。不需要启用 TP/EP/DP。
输出 JSON 记录 prefill+store、restore、decode 耗时与 TPOT。
这些时间包含同步、序列化和 HTTP 副本，只用于诊断本原型，不作为生产 PD 加速结论。

## 7. 接入代码

```python
from nano_qwen.cache.connector import LMCacheConnector
from nano_qwen.sampling_params import SamplingParams

cache = LMCacheConnector("http://127.0.0.1:8765")
params = SamplingParams(temperature=1e-8, max_tokens=32, ignore_eos=True)
# producer 和 decoder 均为已初始化、空闲的 LLMEngine。
receipt = producer.prefill_to_lmcache("Hello", params, cache, "request-1")
seq = decoder.resume_from_lmcache(cache, "request-1")
while not decoder.is_finished():
    outputs, _ = decoder.step()
print(seq.completion_token_ids)
```

存储提交失败时，P 的请求仍在 running 队列，GPU 状态保留，可继续 step 或退出。
恢复失败时，不 ACK 缓存，释放本次新分配的 block/slot。
恢复已成功但 ACK 失败时，返回的 seq 仍可继续 decode，清理错误记录在
`seq.pd_handoff['ack_error']`，可在完成后重试 `cache.delete(transfer_id)`。

参考接口来源：LMCache `v0.5.6` 的
`lmcache/v1/distributed/{api.py,config.py,l1_manager.py}`。
