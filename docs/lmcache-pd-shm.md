# LMCache 原生 tensor CPU RAM 中转（v3）

## 数据路径

```text
P GPU KV / GDN → LMCache 分配的共享 CPU tensor buffer → D GPU KV / GDN
```

LMCache 0.5.6 的 `L1Manager` 使用命名 POSIX SHM arena，按真实 shape/dtype 分配多 tensor `MemoryObj`。
P、D 直接 mmap 该 arena，通过 arena 内的 offset 构造 CPU tensor 视图。
没有另建 payload 存储池，没有 KV/GDN 的 `pack/unpack`、`.tobytes()`、HTTP 上传/下载或整包 SHA256。
HTTP 8766 仅传少量 JSON 元数据，以及 reserve/commit/acquire/release 控制操作。
同一个 arena 在两个进程中的虚拟地址可以不同；协议传 offset，不传可直接解引用的指针。

v1/v2 的 `LMCacheConnector`、8765 字节服务和相关 benchmark/测试已随清理移除；
仓库中不存在其他数据通道，也不提供回退。

## 运行条件

- 同一 Linux 主机，服务和推理进程共享 `/dev/shm`、PID namespace，并具有相应读写权限。
- 内核支持 `pidfd_open`（CPython 构建缺 `os.pidfd_open` 时自动退回 syscall 直调，
  服务启动时探测一次并快速失败）；Docker 中需要足够的 `--shm-size`。服务启动前检查可用 SHM 容量。
- LMCache 服务仍用原来的独立 CPU Python 环境，固定 `lmcache==0.5.6`；推理环境不需要安装 LMCache。
- TP=1、async_scheduling=False、prefix cache=False，单请求空闲边界交接。
- 传完整 attention KV、GDN conv/recurrent state、请求进度与首个生成 token。

CPU 映射在推理进程中不假定已注册为 CUDA pinned memory。本版使用 blocking copy 并在提交/ACK 前同步。
支持 GPU→CPU→GPU 直接 tensor 中转，但不宣称纯 PCIe 带宽、RDMA 或异步 copy overlap。

## 安装

在仓库根目录运行包内安装器：

```bash
python /tmp/nano_qwen_lmcache_shm_v3/install.py --repo "$PWD" --check
python /tmp/nano_qwen_lmcache_shm_v3/install.py --repo "$PWD"
```

只修改 `cache/engine_ops.py` 的 transport dispatch，并添加新 adapter、测试、文档和 benchmark。
不修改模型、CUDA Graph、量化或调度器。安装器会备份被修改的文件，并拒绝覆盖内容不同的本地版本。
后续清理已移除 v1/v2 的 HTTP payload 路径（`connector.py`/`service.py`/`state.pack`/`unpack` 及其测试、
benchmark 与文档）；公共 baseline/统计逻辑提取到 `benchmarks/pd_common.py`，
CPU 引擎 fixture 提取到 `tests/pd_cpu_fixture.py`，`engine_ops` 只保留直接共享张量路径。

## 服务与 CPU 回归

终端一，服务端 CPU 环境：

```bash
PYTHONPATH=src /tmp/nano-lmcache-env/bin/python -m nano_qwen.cache.shm_service \
  --host 127.0.0.1 --port 8766 --size-gb 1
```

终端二，运行回归：

```bash
REQUIRE_LMCACHE=1 PYTHONPATH=src /tmp/nano-lmcache-env/bin/python -m pytest -q tests/test_lmcache_pd_shm.py
```

先运行不需要 GPU/模型的 smoke：

```bash
PYTHONPATH=src python benchmarks/bench_lmcache_pd_shm.py --mode store-smoke \
  --output logs/lmcache_pd/shm_store_smoke.json
```

应看到 `backend=lmcache_l1_shm`、真实 L1 分配与回收、`payload_pack_calls=0`、
`payload_unpack_calls=0`、`payload_http_bytes=0`。`control_sent_bytes/control_received_bytes`
是 JSON 控制消息体字节数，不包括 HTTP 头，通常远小于 payload。

## GPU 正确性：先执行本项

推理环境使用你的模型路径。Graph 和 eager 各跑一次：

```bash
PYTHONPATH=src python benchmarks/bench_lmcache_pd_shm.py --mode roundtrip \
  --model /home/wei/code/models/4b-channel-fp8 --cuda-graph \
  --prompt-lens 64 256 257 512 --output-lens 32 \
  --output logs/lmcache_pd/shm_roundtrip_graph.json
```

Eager 将 `--cuda-graph` 改为 `--no-cuda-graph`，输出另存。
此模式额外复制 CPU 状态作逐字节校验，并占用 guard slot/block 强制重定位。
必须满足状态一致、greedy 输出一致、D 0 次 prefill、31 次 decode、ACK 成功和资源回收。
`performance_valid=false`，它的额外检查不参与正式性能统计。

## 独立进程 P 退出后由 D 恢复

服务保持运行。下面两条命令顺序执行，使用同一 Graph 设置：

```bash
PYTHONPATH=src python benchmarks/bench_lmcache_pd_shm.py --mode prefill \
  --model /home/wei/code/models/4b-channel-fp8 --cuda-graph \
  --prompt-lens 257 --output-lens 32 --transfer-id shm-pd257 --with-reference \
  --output logs/lmcache_pd/shm_prefill257.json

PYTHONPATH=src python benchmarks/bench_lmcache_pd_shm.py --mode decode \
  --model /home/wei/code/models/4b-channel-fp8 --cuda-graph \
  --transfer-id shm-pd257 --reference-json logs/lmcache_pd/shm_prefill257.json \
  --output logs/lmcache_pd/shm_decode257.json
```

这些模式验证跨进程正确性，未做正式预热；不将其首轮 decode 耗时与 perf 中位数混合比较。

## 正式性能

先做三组小矩阵：

```bash
PYTHONPATH=src python benchmarks/bench_lmcache_pd_shm.py --mode perf \
  --model /home/wei/code/models/4b-channel-fp8 --cuda-graph \
  --prompt-lens 64 257 1024 --output-lens 128 \
  --warmup-rounds 1 --repeats 5 --gpu-memory-utilization 0.7 \
  --output logs/lmcache_pd/perf_shm_quick.json
```

通过后再跑与 v2 相同的矩阵：

```bash
PYTHONPATH=src python benchmarks/bench_lmcache_pd_shm.py --mode perf \
  --model /home/wei/code/models/4b-channel-fp8 --cuda-graph \
  --prompt-lens 64 256 257 512 1024 --output-lens 128 512 \
  --warmup-rounds 1 --repeats 5 --token-budget 512 --max-model-len 4096 \
  --gpu-memory-utilization 0.7 --output logs/lmcache_pd/perf_shm_graph.json
```

正式模式在一个常驻引擎上，先做一轮完整矩阵预热，之后五轮测量，交替 baseline/handoff 顺序。
每次交接只有一轮 reserve→copy→commit→acquire→copy→ACK，完整输出数组在两次计时结束后比较。
运行期间把旧 `state/engine_ops.pack/unpack` 替换为报错函数，保证新路径不能调用 payload codec。
JSON 原子保存，自动生成同名 Markdown 中位数表，并记录真实源码 SHA256、Git 状态、模块路径、模型/量化信息。
perf 中 `state_bytes_equal=null` 表示未额外回导状态校验，不代表失败；完整状态校验由 roundtrip 执行，
`tokens_equal` 仍是每对正式样本的实际比较结果。
原来的 `perf_graph.json` 对应 HTTP 快照路径；新结果是 `perf_shm_graph.json`，文件名明确区分。

| 字段 | 范围 |
|---|---|
| `prefill_ms` | P 的计算、调度与首次采样 |
| `describe_ms` | 生成小型状态元数据与 tensor layout |
| `reserve_write_ms` | LMCache 分配、JSON 描述符、首次 arena mmap/CPU 视图构造 |
| `d2h_ms` | GPU KV gather、直接写共享 CPU tensor、GDN copy、尾部清零、校验与同步 |
| `commit_ms` | 完成写入后的 L1 commit 控制操作 |
| `acquire_ms` | L1 read reservation、JSON 描述符、CPU tensor 视图；不复制 payload |
| `h2d_ms` | 原地恢复 GPU KV/GDN、slot 准备、首 token 初始化与同步 |
| `ack_ms` | GPU copy 完成后的 reader release 与 L1 delete |
| `handoff_pause_ms` | P 首次采样之后，到 D 恢复并 ACK 返回 |
| `e2e_ms` | 从 P 入口到全部 token 生成的单个 wall timer，含全部交接与 ACK |

不再报告 Serialize/Deserialize/PUT/GET 大数据阶段。仍有少量 JSON 元数据编码、状态检查和 CPU 内存管理。

## 租约与失败行为

写对象在 commit 前不可读；每个请求只允许一个活跃 reader。
P 完成全部 copy 和 GPU 同步后才 commit，收到回执才释放本地 GPU 状态。
D 先验证元数据/shape/dtype，再分配本地 GPU blocks；恢复同步完成后才 release/ACK。
ACK 失败保留可运行的 GPU 请求，租约在 `seq.pd_handoff['lease']` 中，可同步后调用
`connector.release(lease, delete=True)` 重试。

服务端用 pidfd 追踪活跃写/读进程；活跃 mapping 不因普通 TTL 被回收。
进程退出后，在下一次操作时中止未提交写，或释放死 reader 的锁使快照可重试。
READY 对象才按 TTL 清理。客户端使用 mmap，不创建 Python SharedMemory tracker，P 退出不会 unlink 服务的 arena。
arena 随服务退出释放，不提供服务重启后的持久化。不要让 CPU tensor 视图逃逸租约生命周期。
数据平面没有原始快照的整包 SHA256 协议；正确性测试单独做逐字节比较。
