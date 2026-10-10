# nano_qwen FP8优化检查与最小补丁

检查版本：Jung52/nano_qwen，main@42c20f3546ed0c46a459223acf8dd6e3ce079abb。
这是基于该版本准备的补丁，没有向GitHub提交修改。远端机器若已有其他改动，先执行git apply --check检查兼容性。

## 结论

现有FP8是动态输入量化的W8A8：每个Linear先调用Triton的_quantize_rows，再调用torch._scaled_mm；没有在每步把整个FP8权重还原为BF16。
但并非全部推理工作都能从FP8受益。当前日志证实的是权重减少38.6%、同batch吞吐提升约1.7%～5%、batch=1降低11.4%，以及FP8能完成batch=64而BF16由于KV容量跳过。

## 代码中确认的优化点

| 优先级 | 代码事实 | 优化与边界 |
|---|---|---|
| 1 | attention的q/k/v和GDN的qkv/z分别量化同一份hidden states | 先共享一次输入量化；本补丁实现。保留独立GEMM和原来的累加规则 |
| 2 | FP8的GDN a/b使用stable_gate_projection；BF16使用普通Linear | profile确定耗时后，尝试融合a/b投影或复用输入加载；保留逐行独立FP32归约与一致性测试 |
| 3 | decode GDN先index_select状态，kernel后index_copy_写回；卷积有cat、clone、depthwise conv和silu | profile确认后研究直接按slot读写状态的kernel和融合decode卷积。不能直接删掉gather/scatter，否则请求slot映射会出错 |
| 4 | decode CUDA Graph只捕获模型主体，compute_logits在replay后执行，采样也在图外 | profile观察图外工作及CPU/GPU间隙，再考虑把LM head或采样纳入graph；处理输出缓冲区生命周期和随机数，不先猜收益 |
| 5 | prefill compute_logits对所有prompt token做BF16 LM head，sample_tokens才选择末尾行 | 推理采样路径先选hidden states尾行，再做LM head；保留质量评估需要的全行logits接口，同时调整pending输出契约。主要改善prefill，不解释decode收益小 |
| 6 | FP8 attention固定decode_num_splits=1，BF16为自动0；FP8 _scaled_mm使用use_fast_accum=False | 两者是已存在的数值稳定性策略。可做受控A/B实验，必须同时测质量、跨batch/混合prefill/decode一致性，不能直接当无成本开关 |

MLP的gate/up已经合并，不要重复做同一项优化。embedding、LM head、GDN状态、卷积、门控、norm和采样仍有未量化的工作。
哪些部分占了主要时间仍需GPU trace；上表区分已确认的代码行为与待验证的运行时瓶颈。

## 补丁范围

- attention共享q/k/v的输入FP8激活与scale；GDN共享qkv/z的输入FP8激活与scale，覆盖eager和piecewise路径。
- 调用结束后不持久缓存激活，每次调用和CUDA Graph replay仍重新量化，避免过期数据。
- 保持checkpoint格式、独立GEMM、bias、输出dtype、use_fast_accum=False、稳定gate和attention split策略。
- 非FP8、TP非1和row-parallel投影走普通forward，保留通信。
- NANO_QWEN_FP8_SHARE_INPUT=0关闭复用，默认1。变量在模块导入时读取，A/B必须分别启动新进程。
- 新增GPU测试检查非连续特征、三维输入、空batch、bias、关闭开关和更新输入后的graph replay，要求与原有独立投影逐元素一致。

这一步减少量化kernel及激活buffer分配，不合并GEMM，因此不能承诺解决全部1.81ms的batch=1额外耗时。

## 应用与正确性验证

在nano_qwen仓库根目录执行，将补丁路径改成实际下载位置：

```bash
git apply --check /path/to/fp8_shared_input.patch
git apply /path/to/fp8_shared_input.patch
python -m compileall -q src benchmarks/test_quantization.py
PYTHONPATH=src python -m pytest -q benchmarks/test_quantization.py
```

pytest应在已有nano_qwen依赖和CUDA环境中执行。FP8投影与graph测试、GDN跨路径一致性、attention混合批次、piecewise graph测试都应通过，再接受性能结果。
另外运行项目已有的质量评估，并比较补丁开/关两种FP8结果：

```bash
mkdir -p benchmarks/reports/fp8_reuse_quality
NANO_QWEN_FP8_SHARE_INPUT=0 PYTHONPATH=src python benchmarks/validate_fp8_model.py \
  --model /root/autodl-tmp/nano_qwen/fp8 --mode decode_graph --suite quality \
  --json-out benchmarks/reports/fp8_reuse_quality/off.json
NANO_QWEN_FP8_SHARE_INPUT=1 PYTHONPATH=src python benchmarks/validate_fp8_model.py \
  --model /root/autodl-tmp/nano_qwen/fp8 --mode decode_graph --suite quality \
  --json-out benchmarks/reports/fp8_reuse_quality/on.json
```

## 性能A/B

保留此前高负载bench_bf16_vs_fp8.sh及其validate脚本；本补丁不覆盖它们。两次都用相同profile、mem_util、生成长度和GPU，后台不运行其他GPU任务：

```bash
NANO_QWEN_FP8_SHARE_INPUT=0 bash benchmarks/bench_bf16_vs_fp8.sh decode_graph
NANO_QWEN_FP8_SHARE_INPUT=1 bash benchmarks/bench_bf16_vs_fp8.sh decode_graph
```

再按1→0顺序重复，保留每次独立报告目录。观察FP8 batch=1/4/8/16/32的step_ms，BF16作为环境稳定性对照。
比较中位数及波动，不用单次小于几个百分点的差异判断有效。GPU峰值显存仍受自动KV预分配和graph池影响，不能直接当权重显存节省。

## 定位剩余耗时

仓库已经有benchmarks/bench_one_batch.py，无需另造profiler。先生成BF16/FP8的batch=1和32 decode trace：

```bash
for model in 9b fp8; do
  for batch in 1 32; do
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python benchmarks/bench_one_batch.py \
      --model "/root/autodl-tmp/nano_qwen/$model" \
      --batch-size "$batch" --input-len 128 --decode-steps 32 \
      --warmup-rounds 3 --max-model-len 2048 --gpu-memory-utilization 0.65 \
      --disable-prefill-cudagraph --trace-stage decode \
      --output-dir "benchmarks/reports/profile_$model"
  done
done
```

打开*.trace.json，查看真实CUDA设备轨道上的_quantize_rows、_gate_projection、FP8/BF16 GEMM、gdn_decode、index_select/index_copy、卷积、LM head及sampling；另看CUDA Graph之间的设备空闲和CPU等待。
graph模式下CPU operator汇总不能完整归因graph内重放的kernel，主要看设备轨道。需要op/shape归因时另跑相同命令加--enforce-eager --record-shapes --with-stack；eager结果用于定位，不能替代decode_graph性能结论。
这里profiler按单个batch配置max_num_seqs，和原高负载sweep统一max_num_seqs=64的状态/KV预算不同；不要混用它们的绝对延迟做同条件性能结论。

## 本地已完成验证

语法编译通过；依赖隔离的路由检查通过，包括一次量化、同一激活/scale复用、输出形状、关闭开关、row-parallel回退和错误输入维度；补丁对检查版本git apply --check通过。
当前分析环境没有PyTorch、Triton、CUDA及模型权重，GPU测试、质量评估与实测加速尚未运行。新增GPU测试是供你的5090环境执行的验证，不代表已经通过。

## 源码位置

- https://github.com/Jung52/nano_qwen/blob/42c20f3546ed0c46a459223acf8dd6e3ce079abb/src/nano_qwen/quantization/fp8.py
- https://github.com/Jung52/nano_qwen/blob/42c20f3546ed0c46a459223acf8dd6e3ce079abb/src/nano_qwen/layers/gated_delta_net.py
- https://github.com/Jung52/nano_qwen/blob/42c20f3546ed0c46a459223acf8dd6e3ce079abb/src/nano_qwen/models/qwen3_5.py
- https://github.com/Jung52/nano_qwen/blob/42c20f3546ed0c46a459223acf8dd6e3ce079abb/src/nano_qwen/engine/cuda_graph.py
- https://github.com/Jung52/nano_qwen/blob/42c20f3546ed0c46a459223acf8dd6e3ce079abb/src/nano_qwen/engine/model_runner.py
