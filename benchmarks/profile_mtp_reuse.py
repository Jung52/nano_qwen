# ruff: noqa: E402,F401
"""Separate production timing, CUDA-event stages and untouched B2/B4 timelines."""

import argparse
from collections import defaultdict
from contextlib import nullcontext as _null
from time import perf_counter_ns
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]
from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.engine.sequence import Sequence
from nano_qwen.layers.sampler import Sampler
from nano_qwen.sampling_params import SamplingParams
from bench_mtp_batch2 import run_batch
from bench_mtp_interleaved import execution_mode
from bench_mtp_small_batch import make_cases
from bench_qwen3_5_mtp_production import release
from mtp_profile_analysis import summarize_trace, GraphEventBank
from mtp_profile_instrumentation import PhaseRecorder
from transformers import AutoTokenizer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--mtp-model", required=True)
    p.add_argument("--batches", nargs="+", type=int, default=[2, 4])
    p.add_argument("--execution", choices=["graph", "eager"], default="graph")
    p.add_argument("--timing-rounds", type=int, choices=[0, 2], default=2)
    p.add_argument("--trace-start-tokens", type=int, default=48)
    p.add_argument("--trace-end-tokens", type=int, default=80)
    p.add_argument("--no-dynamic-tail", action="store_true")
    p.add_argument("--passes", nargs="+", choices=["events", "timeline", "host"], default=["events", "timeline", "host"])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if any(not 1 <= b <= 8 for b in a.batches):
        p.error("batch capacities must be in1..8")
    if not 1 <= a.trace_start_tokens < a.trace_end_tokens < 116:
        p.error("trace token window must fit before the dynamic tail")
    return a


def source():
    paths = [*ROOT.joinpath("src/nano_qwen").rglob("*.py"),
             *(ROOT / "benchmarks" / n for n in ("profile_mtp_reuse.py", "mtp_profile_analysis.py",
                                                 "mtp_profile_instrumentation.py", "bench_mtp_batch2.py"))]
    return dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def tuning_choices():
    from nano_qwen.quantization.fp8 import _per_channel_mm_tuned
    from nano_qwen.quantization.fp8_block import _block_mm_tuned
    return {name: [dict(key=list(key), config=config.all_kwargs()) for key, config in tuner.cache.items() if key[0] <= 8]
            for name, tuner in (("per_channel", _per_channel_mm_tuned), ("block_wise", _block_mm_tuned))}


def cohort(engine, prompts, budgets):
    if not engine.is_finished() or engine.model_runner.input_batch.seq_id_to_slot:
        raise ValueError("Diagnostic cohort requires an idle engine")
    seqs = [Sequence(p, SamplingParams(temperature=1e-6, max_tokens=n, ignore_eos=True)) for p, n in zip(prompts, budgets)]
    for seq in seqs:
        engine.scheduler.add(seq)
    engine.step()
    assert all(s.num_completion_tokens == 1 for s in seqs)
    return seqs


def progress(engine, seqs):
    runner = engine.model_runner
    live = [s for s in seqs if not s.is_finished]
    return dict(active=len(live), seq_ids=[s.seq_id for s in live],
                slots=[runner.input_batch.seq_id_to_slot.get(s.seq_id) for s in live],
                completion=[s.num_completion_tokens for s in seqs],
                cached=[s.num_cached_tokens for s in seqs])


def recorded_step(engine, seqs, recorder, index):
    before = progress(engine, seqs)
    prior = dict(engine.model_runner.mtp_stats)
    recorder.begin()
    start = perf_counter_ns()
    with torch.profiler.record_function(f"mtp.round/{index}") if recorder.kind == "trace" else _null():
        engine.step()
    wall_ms = (perf_counter_ns()-start)/1e6
    timers = recorder.finish()
    after = progress(engine, seqs)
    delta = {k: v-prior[k] for k, v in engine.model_runner.mtp_stats.items()}
    return dict(index=index, wall_ms=wall_ms, before=before, after=after,
                committed=sum(after["completion"])-sum(before["completion"]), **delta, **timers)





def event_pass(engine, prompts, expected, recorder):
    seqs = cohort(engine, prompts, [128]*len(prompts))
    rows = []
    try:
        while not all(s.is_finished for s in seqs):
            rows.append(recorded_step(engine, seqs, recorder, len(rows)))
        completions = [s.completion_token_ids for s in seqs]
        if completions != expected:
            raise RuntimeError("Stage-event pass changed strict completion")
        totals = defaultdict(float)
        kind = "host_ms" if recorder.kind == "host" else "gpu_ms"
        for row in rows:
            for label, ms in row[kind].items():
                totals[label] += ms
        return dict(rows=rows, **{kind: dict(totals)}, committed=sum(r["committed"] for r in rows),
                    completion=completions, strict_pass=True,
                    note=("Host-only counters; no profiler, device events or additional synchronization" if recorder.kind == "host" else
                          "Per-step device synchronization and event reads; host gaps from this pass are not production gaps"))
    finally:
        for seq in seqs:
            release(engine, seq)


def trace_pass(engine, prompts, budgets, expected, recorder, path, start, end):
    seqs = cohort(engine, prompts, budgets)
    rows = []
    try:
        while min(s.num_completion_tokens for s in seqs) < start:
            engine.step()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("mtp.window"):
                while not all(s.is_finished for s in seqs) and (end is None or min(s.num_completion_tokens for s in seqs) < end):
                    rows.append(recorded_step(engine, seqs, recorder, len(rows)))
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(path))
        summary = summarize_trace(json.loads(path.read_text())["traceEvents"])
        summary.update(trace=str(path), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       trace_bytes=path.stat().st_size, rows=rows, committed=sum(r["committed"] for r in rows))
        if not summary["physical_device_events"]:
            raise RuntimeError("No physical GPU events; cannot infer device gaps")
        while not all(s.is_finished for s in seqs):
            engine.step()
        completions = [s.completion_token_ids for s in seqs]
        if expected is not None and completions != expected:
            raise RuntimeError("Trace pass changed strict completion")
        summary.update(completion=completions, strict_pass=expected is None or completions == expected)
        return summary
    finally:
        for seq in seqs:
            release(engine, seq)


@torch.inference_mode()
def measure(a):
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    prompt = next(c["prompt"] for c in make_cases(tokenizer) if c["name"] == "synthetic-64-128")
    report = dict(model=a.model, mtp_model=a.mtp_model, batches=a.batches,
                  execution=a.execution, input=64, output=128, source=source(),
                  environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda),
                  protocol="timing before instrumentation; full-cohort CUDA-event stages; separate untouched-graph short timelines",
                  timing=[], profiles={})
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")

    for batch in a.batches:
        engine = LLMEngine(a.model, max_num_seqs=batch, max_num_batched_tokens=1024,
                           max_model_len=1024, gpu_memory_utilization=.85, enforce_eager=a.execution == "eager",
                           use_prefill_cudagraph=False, enable_mtp=True, mtp_model=a.mtp_model, mtp_verify_mode="reuse")
        prompts = [prompt]*batch
        runner = engine.model_runner
        reference = None
        try:
            for mode in ("off", "reuse"):
                with execution_mode(engine, mode == "reuse", Sampler()):
                    run_batch(engine, prompts, 4)
            for repeat in range(1, a.timing_rounds+1):
                for mode in ("off", "reuse"):
                    with execution_mode(engine, mode == "reuse", Sampler()):
                        metrics, tokens = run_batch(engine, prompts, 128)
                    reference = tokens if reference is None else reference
                    strict = tokens == reference
                    report["timing"].append(dict(batch=batch, repeat=repeat, mode=mode, metrics=metrics, completion=tokens, strict_pass=strict))
                    save()
                    print(f"B{batch} timing {mode}{repeat}: {metrics['tokens_per_s']:.2f} strict={strict}", flush=True)
                    if not strict:
                        raise RuntimeError("Unprofiled strict completion mismatch")
            if reference is None:
                with execution_mode(engine, False, Sampler()):
                    _, reference = run_batch(engine, prompts, 128)  # Correctness-only reference, not a timing sample.
            result = report["profiles"][str(batch)] = dict(tuning=tuning_choices(), events={}, timeline={}, dynamic_tail={}, host={})
            for kind in ("host", "events"):
                if kind not in a.passes:
                    continue
                with PhaseRecorder(runner, kind).install() as recorder:
                    for mode in ("off", "reuse"):
                        with execution_mode(engine, mode == "reuse", Sampler()):
                            result[kind][mode] = event_pass(engine, prompts, reference, recorder)
                        save()
                        print(f"B{batch} {kind} stages {mode}: strict=True", flush=True)
            # Event recorder has restored original graphs before this pass.
            if "timeline" not in a.passes:
                continue
            with PhaseRecorder(runner, "trace").install() as recorder:
                for mode in ("off", "reuse"):
                    path = a.output.with_suffix(f".B{batch}.{mode}.trace.json")
                    with execution_mode(engine, mode == "reuse", Sampler()):
                        result["timeline"][mode] = trace_pass(engine, prompts, [128]*batch, reference, recorder,
                                                              path, a.trace_start_tokens, a.trace_end_tokens)
                    save()
                    print(f"B{batch} timeline {mode}: strict=True", flush=True)
                if not a.no_dynamic_tail:
                    dynamic_reference = None
                    budgets = [128-4*i for i in range(batch)]
                    for mode in ("off", "reuse"):
                        path = a.output.with_suffix(f".B{batch}.{mode}.dynamic.trace.json")
                        with execution_mode(engine, mode == "reuse", Sampler()):
                            value = trace_pass(engine, prompts, budgets, dynamic_reference, recorder, path, min(110, min(budgets)-6), None)
                        dynamic_reference = value["completion"] if dynamic_reference is None else dynamic_reference
                        result["dynamic_tail"][mode] = value
                        save()
                        print(f"B{batch} dynamic {mode}: strict=True", flush=True)
        finally:
            engine.exit()
    return report


if __name__ == "__main__":
    measure(parse_args())
