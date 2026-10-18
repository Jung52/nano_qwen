# 稳态decode的CPU API及未分类kernel补充

CPU API耗时包含等待GPU的区间，与GPU重合部分不能当作可消除host开销。这里只按trace中实际事件统计；没有启用Python栈，aten父范围不能单独证明某个具体Python调用点。

| 配置 | CUDA API | 次数 | CPU累计ms | 与设备忙碌重合ms |
| --- | --- | --- | --- | --- |
| B2-off | cudaEventSynchronize | 64 | 355.50 | 350.94 |
| B2-off | cudaGraphLaunch | 32 | 28.04 | 0.14 |
| B2-off | cudaMemcpyAsync | 448 | 3.88 | 1.09 |
| B2-off | cudaLaunchKernel | 224 | 1.36 | 0.45 |
| B2-off | cuLaunchKernel | 64 | 0.99 | 0.77 |
| B2-reuse | cudaStreamSynchronize | 164 | 322.65 | 316.33 |
| B2-reuse | cudaGraphLaunch | 49 | 47.08 | 0.09 |
| B2-reuse | cudaMemcpyAsync | 696 | 13.07 | 0.97 |
| B2-reuse | cudaLaunchKernel | 364 | 7.43 | 1.23 |
| B2-reuse | cuLaunchKernel | 34 | 1.92 | 1.80 |
| B4-off | cudaEventSynchronize | 64 | 372.16 | 361.31 |
| B4-off | cudaGraphLaunch | 32 | 43.41 | 1.62 |
| B4-off | cudaMemcpyAsync | 512 | 7.37 | 1.61 |
| B4-off | cudaLaunchKernel | 224 | 2.50 | 0.55 |
| B4-off | cudaEventRecordWithFlags | 192 | 1.26 | 0.62 |
| B4-reuse | cudaStreamSynchronize | 189 | 353.65 | 330.96 |
| B4-reuse | cudaGraphLaunch | 48 | 52.26 | 0.77 |
| B4-reuse | cudaMemcpyAsync | 783 | 12.53 | 2.87 |
| B4-reuse | cudaLaunchKernel | 359 | 6.45 | 2.24 |
| B4-reuse | cuLaunchKernel | 34 | 2.59 | 1.82 |
| B8-off | cudaEventSynchronize | 64 | 377.97 | 363.96 |
| B8-off | cudaGraphLaunch | 32 | 46.98 | 4.30 |
| B8-off | cudaMemcpyAsync | 640 | 6.83 | 3.15 |
| B8-off | cudaLaunchKernel | 224 | 2.43 | 0.94 |
| B8-off | cuLaunchKernel | 64 | 1.71 | 0.32 |
| B8-reuse | cudaStreamSynchronize | 257 | 427.61 | 417.69 |
| B8-reuse | cudaGraphLaunch | 53 | 60.66 | 0.21 |
| B8-reuse | cudaMemcpyAsync | 1095 | 12.69 | 0.97 |
| B8-reuse | cudaLaunchKernel | 405 | 7.42 | 2.45 |
| B8-reuse | cuLaunchKernelEx | 168 | 2.05 | 0.36 |

## 最长同步调用的aten父范围

| 配置 | API | CPU ms | 外到内的aten范围 |
| --- | --- | --- | --- |
| B2-off | cudaEventSynchronize | 11.76 | 无aten父范围 |
| B2-off | cudaEventSynchronize | 11.39 | 无aten父范围 |
| B2-off | cudaEventSynchronize | 11.32 | 无aten父范围 |
| B2-reuse | cudaStreamSynchronize | 16.55 | aten::to → aten::_to_copy → aten::copy_ |
| B2-reuse | cudaStreamSynchronize | 16.50 | aten::to → aten::_to_copy → aten::copy_ |
| B2-reuse | cudaStreamSynchronize | 16.48 | aten::to → aten::_to_copy → aten::copy_ |
| B4-off | cudaEventSynchronize | 13.44 | 无aten父范围 |
| B4-off | cudaEventSynchronize | 12.74 | 无aten父范围 |
| B4-off | cudaEventSynchronize | 12.66 | 无aten父范围 |
| B4-reuse | cudaStreamSynchronize | 19.32 | aten::to → aten::_to_copy → aten::copy_ |
| B4-reuse | cudaStreamSynchronize | 19.31 | aten::to → aten::_to_copy → aten::copy_ |
| B4-reuse | cudaStreamSynchronize | 19.06 | aten::to → aten::_to_copy → aten::copy_ |
| B8-off | cudaEventSynchronize | 13.44 | 无aten父范围 |
| B8-off | cudaEventSynchronize | 13.07 | 无aten父范围 |
| B8-off | cudaEventSynchronize | 12.93 | 无aten父范围 |
| B8-reuse | cudaStreamSynchronize | 20.53 | aten::to → aten::_to_copy → aten::copy_ |
| B8-reuse | cudaStreamSynchronize | 19.17 | aten::to → aten::_to_copy → aten::copy_ |
| B8-reuse | cudaStreamSynchronize | 19.15 | aten::to → aten::_to_copy → aten::copy_ |

## other类别的最大kernel

名称截取前140字符，完整名称及全部累计值保存在trace_details.json。这里主要用于检查copy/cast、reduction及逐元素计算是否遗漏，不把未知项自动归入GDN。

| 配置 | kernel名称前缀 | 累计ms |
| --- | --- | --- |
| B2-off | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const: | 10.63 |
| B2-off | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::ar | 6.13 |
| B2-off | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, 4,  | 5.95 |
| B2-off | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::bi | 5.94 |
| B2-reuse | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const: | 12.06 |
| B2-reuse | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::ar | 6.64 |
| B2-reuse | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, 4,  | 6.61 |
| B2-reuse | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::bi | 6.51 |
| B4-off | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const: | 10.83 |
| B4-off | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, 4,  | 6.86 |
| B4-off | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::ar | 6.68 |
| B4-off | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::bi | 6.06 |
| B4-reuse | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const: | 13.08 |
| B4-reuse | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, 4,  | 7.67 |
| B4-reuse | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::ar | 7.10 |
| B4-reuse | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::bi | 6.89 |
| B8-off | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const: | 11.51 |
| B8-off | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, 4,  | 7.84 |
| B8-off | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::ar | 6.22 |
| B8-off | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::bi | 5.97 |
| B8-reuse | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const: | 13.66 |
| B8-reuse | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, 4,  | 10.55 |
| B8-reuse | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::ar | 8.18 |
| B8-reuse | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::bi | 7.77 |
