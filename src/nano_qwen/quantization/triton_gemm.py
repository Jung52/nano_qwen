"""Small, reproducible autotune budget shared by the FP8 GEMM families."""

import torch
import triton

from nano_qwen.utils.context import get_context


GEMM_CONFIGS = [
    triton.Config({"BM": bm, "BN": bn}, num_warps=warps, num_stages=stages)
    for bm, bn, warps, stages in (
        (16, 32, 4, 2), (16, 64, 4, 2), (16, 128, 4, 2),
        (32, 64, 4, 3), (32, 128, 4, 3), (64, 128, 8, 3),
    )
]
_eviction_buffers = {}
_capture_streams = {}
_block_prefill_config = triton.Config({"BM": 32, "BN": 128}, num_warps=4, num_stages=3)


def _preferred_config(tuned, a, w, sa, sw, output):
    # Validated with real-input out_proj/o_proj shape on this GPU. Quantization
    # and BK stay unchanged; other shapes/devices/dtypes retain normal tuning.
    if ((a.shape[0], w.shape[0], a.shape[1]) != (512, 2560, 4096)
            or getattr(tuned.fn, "__name__", None) != "_block_mm"
            or tuple(t.dtype for t in (a, w, sa, sw, output)) != (
                torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float32,
                torch.float32, torch.bfloat16)):
        return None
    if torch.cuda.get_device_properties(a.device).name == "NVIDIA GeForce RTX 4070 Ti SUPER":
        return _block_prefill_config
    return None


def bench_gemm_config(fn, quantiles=None):
    """One warmup, two single replays, with eviction outside timing nodes."""
    device = torch.cuda.current_device()
    if device not in _eviction_buffers:
        size = max(96 * 2**20, 2 * torch.cuda.get_device_properties(device).L2_cache_size)
        # A later eager caller must be able to reuse a buffer created during inference.
        with torch.inference_mode(False):
            _eviction_buffers[device] = torch.zeros(size, device=device, dtype=torch.uint8)
    fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    if device not in _capture_streams:
        _capture_streams[device] = torch.cuda.Stream(device=device)
    # PyTorch's implicit capture stream is shared across devices in this process.
    with torch.cuda.graph(graph, stream=_capture_streams[device]):
        start.record()
        fn()
        end.record()
    samples = []
    for _ in range(2):
        _eviction_buffers[device].add_(1)
        torch.cuda.synchronize()
        graph.replay()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    low, high = sorted(samples)
    if quantiles is None:
        return (low + high) / 2
    return [low + (high - low) * q for q in quantiles]


def autotune_gemm(kernel):
    configs = GEMM_CONFIGS
    if kernel.__name__ == '_block_mm_prefill':
        configs = [triton.Config({**c.kwargs, 'STAGES': c.num_stages},
                                num_warps=c.num_warps, num_stages=c.num_stages)
                   for c in GEMM_CONFIGS]
    return triton.autotune(configs=configs, key=["M", "N", "K", "DEVICE"],
                           do_bench=bench_gemm_config)(kernel)


def gemm_key(a, w, sa, sw, output):
    # Matches the installed Triton Autotuner key: explicit keys, then tensor dtypes.
    return (a.shape[0], w.shape[0], a.shape[1], a.device.index,
            *(str(t.dtype) for t in (a, w, sa, sw, output)))


def launch_gemm(tuned, a, w, sa, sw, output):
    m, n, k, device = a.shape[0], w.shape[0], a.shape[1], a.device.index
    key = gemm_key(a, w, sa, sw, output)
    if m == 2 and get_context().gemm_reference_m == 1:
        # Read the local M1 selection; never transplant it into the M2 cache.
        reference_key = (1, *key[1:])
        config = tuned.cache.get(reference_key)
        if config is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("MTP reuse requires eager M1 GEMM warmup before capture")
            tuned[lambda meta: (1, triton.cdiv(n, meta['BN']))](
                a[:1], w, sa[:1], sw, output[:1], 1, n, k, DEVICE=device,
            )
            config = tuned.cache[reference_key]
        grid = (triton.cdiv(m, config.kwargs['BM']), triton.cdiv(n, config.kwargs['BN']))
        tuned.fn[grid](a, w, sa, sw, output, m, n, k, DEVICE=device, **config.all_kwargs())
        return
    preferred = _preferred_config(tuned, a, w, sa, sw, output)
    if preferred is not None:
        tuned.cache[key] = preferred
    config = tuned.cache.get(key)
    def grid(meta):
        rows, cols = triton.cdiv(m, meta['BM']), triton.cdiv(n, meta['BN'])
        return (rows * cols,) if getattr(tuned.fn, '__name__', None) == '_block_mm_prefill' else (rows, cols)
    if config is None and not torch.cuda.is_current_stream_capturing():
        tuned[grid](a, w, sa, sw, output, m, n, k, DEVICE=device)
    else:
        # Cold capture cannot benchmark. Warm captures keep the chosen tile.
        config = config or GEMM_CONFIGS[1]
        tuned.fn[grid(config.kwargs)](a, w, sa, sw, output, m, n, k,
                       DEVICE=device, **config.all_kwargs())
