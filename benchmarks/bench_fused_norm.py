#!/usr/bin/env python3
"""Compare reference/fused norm replay latency without a model or profiler."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="reports/fused_norm_micro.json")
    p.add_argument("--iters", type=int, default=1000)
    a = p.parse_args()
    if a.iters < 1:
        p.error("iters must be positive")
    import torch
    from nano_qwen.layers import fused_norm as backend
    from nano_qwen.layers.layernorm import RMSNorm, GemmaRMSNorm, RMSNormGated
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    def latency(graph):
        for _ in range(10):
            graph.replay()
        times = []
        for _ in range(3):
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(a.iters):
                graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end)*1000/a.iters)
        return dict(median_us=statistics.median(times), rounds_us=times)

    result = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        dtype="bfloat16", mode="CUDA Graph replay, no profiler", cases=[])
    previous = backend.ENABLED
    try:
        with torch.inference_mode():
            torch.manual_seed(42)
            for kind, rows, n in [("rms", 8, 2560), ("gemma", 8, 2560),
                    ("gemma_residual", 8, 2560), ("head_strided", 128, 128), ("gated", 256, 128)]:
                layer = (RMSNormGated(n) if kind == "gated" else RMSNorm(n) if kind == "rms"
                         else GemmaRMSNorm(n)).cuda().bfloat16()
                layer.weight.uniform_(0.5, 1.5) if kind in ("rms", "gated") else layer.weight.uniform_(-0.2, 0.2)
                if kind == "gated":
                    layer.weight.data = layer.weight.float()
                x = torch.randn(rows, n*(2 if kind == "head_strided" else 1),
                    dtype=torch.bfloat16, device="cuda")[:, :n]
                aux = torch.randn_like(x)
                def call():
                    return layer(x, aux) if kind in ("gated", "gemma_residual") else layer(x)
                graphs, outputs, numbers = {}, {}, {}
                for name, enabled in (("reference", False), ("fused", True)):
                    backend.ENABLED = enabled
                    assert not enabled or backend.can_fuse(x, layer.weight,
                        aux if kind == "gemma_residual" else None, aux if kind == "gated" else None)
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(3):
                            call()
                    torch.cuda.current_stream().wait_stream(side)
                    graphs[name] = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graphs[name]):
                        outputs[name] = call()
                # Alternate measurement order for successive cases.
                order = ["reference", "fused"] if len(result["cases"]) % 2 == 0 else ["fused", "reference"]
                for name in order:
                    numbers[name] = latency(graphs[name])
                def tensors(y):
                    return y if isinstance(y, tuple) else (y,)
                errors = []
                for f, ref in zip(tensors(outputs["fused"]), tensors(outputs["reference"])):
                    torch.testing.assert_close(f, ref, atol=0.016, rtol=0.012)
                    errors.append(dict(max_abs=(f.float()-ref.float()).abs().max().item(),
                        rms=(f.float()-ref.float()).square().mean().sqrt().item()))
                row = dict(kind=kind, rows=rows, hidden=n, errors=errors, **numbers,
                    speedup=numbers["reference"]["median_us"]/numbers["fused"]["median_us"])
                result["cases"].append(row)
                print(f"{kind:16s}: {row['speedup']:.3f}x", flush=True)
                del graphs, outputs
    finally:
        backend.ENABLED = previous
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+"\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
