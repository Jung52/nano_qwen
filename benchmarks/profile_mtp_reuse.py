# ruff: noqa: E402
"""Short diagnostic only: profiler + captured timing events, never formal timing."""

import argparse
import json
import sys
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import torch

sys.path[:0] = [str(Path.cwd() / "src"), str(Path.cwd() / "benchmarks")]
from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.engine.sequence import Sequence
from nano_qwen.sampling_params import SamplingParams
from bench_qwen3_5_mtp_production import make_prompt, release


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mtp-model", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    report = {}
    for eager in (False, True):
        engine = LLMEngine(
            a.model,
            max_num_seqs=1,
            max_num_batched_tokens=1024,
            max_model_len=800,
            gpu_memory_utilization=0.85,
            enforce_eager=eager,
            use_prefill_cudagraph=False,
            enable_mtp=True,
            mtp_model=a.mtp_model,
            mtp_verify_mode="reuse",
        )
        seq = Sequence(
            make_prompt(a.model, 64),
            SamplingParams(temperature=1e-6, max_tokens=512, ignore_eos=True),
        )
        try:
            r = engine.model_runner
            captured = {}
            rounds = []

            def tagged(fn, name):
                def wrapped(*args, **kwargs):
                    label = name(*args, **kwargs) if callable(name) else name
                    timed_capture = (
                        torch.cuda.is_current_stream_capturing()
                        and label in ("phase.save_speculative_state", "phase.lm_head")
                    )
                    if timed_capture:
                        start = torch.cuda.Event(enable_timing=True, external=True)
                        end = torch.cuda.Event(enable_timing=True, external=True)
                        captured.setdefault(label, []).append((start, end))
                        start.record()
                    with torch.profiler.record_function(label):
                        value = fn(*args, **kwargs)
                    if timed_capture:
                        end.record()
                    return value

                return wrapped

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        r, "run_model", tagged(r.run_model, "phase.target_verify")
                    )
                )
                stack.enter_context(
                    patch.object(
                        r,
                        "_run_mtp_decode",
                        tagged(
                            r._run_mtp_decode,
                            lambda *args, **kw: "phase.mtp_cache_only"
                            if kw.get("cache_only")
                            else "phase.mtp_draft",
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(
                        r.model,
                        "compute_logits",
                        tagged(r.model.compute_logits, "phase.lm_head"),
                    )
                )
                for layer in r.gdn_layers:
                    for method in (
                        "save_speculative_state",
                        "rollback_speculative_state",
                    ):
                        stack.enter_context(
                            patch.object(
                                layer,
                                method,
                                tagged(getattr(layer, method), "phase." + method),
                            )
                        )
                if not eager:
                    # Empty engine only: recapture adds event nodes without losing active state.
                    r.cuda_graphs._capture_verify()
                    r.cuda_graphs._capture_mtp_decode()
                engine.scheduler.add(seq)
                for _ in range(9):
                    engine.step()
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    for _ in range(12):
                        before = dict(r.mtp_stats)
                        prior = seq.num_completion_tokens
                        engine.step()
                        row = dict(
                            committed=seq.num_completion_tokens - prior,
                            accepted=r.mtp_stats["accepted_tokens"]
                            - before["accepted_tokens"],
                        )
                        for label, events in captured.items():
                            row[label + "_gpu_us"] = sum(
                                s.elapsed_time(e) * 1000 for s, e in events
                            )
                        rounds.append(row)
                    torch.cuda.synchronize()
            mode = "eager" if eager else "graph"
            trace = a.output.with_suffix("." + mode + ".trace.json")
            prof.export_chrome_trace(str(trace))
            report[mode] = {
                "trace": str(trace),
                "rounds": rounds,
                "events": [
                    {
                        "name": x.key,
                        "count": x.count,
                        "cpu_us": x.cpu_time_total,
                        "self_cpu_us": x.self_cpu_time_total,
                        "gpu_us": x.device_time_total,
                        "self_gpu_us": x.self_device_time_total,
                    }
                    for x in prof.key_averages()
                    if x.key.startswith("phase.")
                    or x.key
                    in (
                        "aten::item",
                        "cudaDeviceSynchronize",
                        "cudaStreamSynchronize",
                        "cudaGraphLaunch",
                    )
                ],
            }
            a.output.write_text(json.dumps(report, indent=2) + "\n")
            print(mode, report[mode]["events"], flush=True)
        finally:
            release(engine, seq)
            engine.exit()


if __name__ == "__main__":
    main()
