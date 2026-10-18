"""Two-repeat, same-engine rollback A/B timing with a separate host-only probe."""

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]

from bench_mtp_batch2 import run_batch  # noqa: E402
from bench_mtp_interleaved import execution_mode  # noqa: E402
from bench_mtp_small_batch import make_cases  # noqa: E402
from mtp_rollback_reference import REFERENCE, legacy_rollback, reference_methods  # noqa: E402
from nano_qwen.engine.llm_engine import LLMEngine  # noqa: E402
from nano_qwen.layers.attention import Attention  # noqa: E402
from nano_qwen.layers.sampler import Sampler  # noqa: E402
from profile_mtp_reuse import tuning_choices  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def checked_prefixes(engine, prompts):
    """Untimed: copy state at the first accepted and rejected verify rounds."""
    runner, scheduler = engine.model_runner, engine.scheduler
    postprocess = scheduler.postprocess
    checkpoints = {}

    def valid_rows(cache, blocks, count):
        positions = [blocks[p // runner.block_size]*runner.block_size + p % runner.block_size
                     for p in range(count)]
        indices = torch.tensor(positions, device=cache.device, dtype=torch.int64)
        return cache.flatten(0, 1).index_select(0, indices).cpu()

    def post(rows, outputs, prefill):
        if rows and all(s.is_speculative for s in rows):
            # same64 requests share decisions; keep the first full state of
            # each actual outcome, rather than only their released zero pools.
            kind = "accepted" if outputs[0].accepted_count else "rejected"
            if kind not in checkpoints:
                tensors, metadata = {}, []
                for n, layer in enumerate(runner.gdn_layers):
                    tensors[f"gdn{n}.conv"] = layer.conv_states.cpu()
                    tensors[f"gdn{n}.recurrent"] = layer.recurrent_states.cpu()
                for i, (seq, output) in enumerate(zip(rows, outputs)):
                    slot = runner.input_batch.slots_for([seq])[0]
                    prefix = seq.num_cached_tokens + output.cache_advance
                    count = runner._mtp_cache_lens[slot]
                    metadata.append(dict(slot=slot, target_prefix=prefix, mtp_prefix=count, submitted=len(output.committed_tokens)))
                    for n, module in enumerate(m for m in runner.model.modules() if isinstance(m, Attention)):
                        for label, cache in (("k", module.k_cache), ("v", module.v_cache)):
                            tensors[f"request{i}.target{n}.{label}"] = valid_rows(cache, seq.block_table, prefix)
                    for label, cache in zip(("k", "v"), runner.mtp_kv_cache):
                        tensors[f"request{i}.mtp.{label}"] = valid_rows(cache, seq.block_table, count)
                checkpoints[kind] = dict(metadata=metadata, tensors=tensors)
        return postprocess(rows, outputs, prefill)

    with patch.object(scheduler, "postprocess", post):
        _, tokens = run_batch(engine, prompts, 128)
    if checkpoints.keys() != {"accepted", "rejected"}:
        raise RuntimeError("State gate did not exercise both decisions")
    return tokens, checkpoints


def compare_prefixes(reference, actual):
    result = {}
    if reference.keys() != actual.keys():
        raise RuntimeError("State checkpoint outcomes differ")
    for kind, expected in reference.items():
        got = actual[kind]
        if expected["metadata"] != got["metadata"] or expected["tensors"].keys() != got["tensors"].keys():
            raise RuntimeError("State prefixes or tensor sets differ")
        checked_bytes, hashes = 0, {}
        for name, tensor in got["tensors"].items():
            if not torch.isfinite(tensor).all() or not torch.equal(tensor.view(torch.uint8), expected["tensors"][name].view(torch.uint8)):
                raise RuntimeError(f"Nonfinite or changed state: {kind}/{name}")
            raw = tensor.view(torch.uint8).numpy().tobytes()
            hashes[name] = hashlib.sha256(raw).hexdigest()
            checked_bytes += len(raw)
        result[kind] = dict(metadata=got["metadata"], tensors=len(hashes), checked_bytes=checked_bytes,
                            sha256=hashes, bitwise_pass=True, finite_pass=True)
    return result


@contextmanager
def host_probe(engine, legacy):
    """CPU timestamps only: no device reads, timing events or synchronization."""
    runner = engine.model_runner
    rows, restores = [], []
    name = "rollback_speculative_state" if legacy else "rollback_speculative_states"

    def timed_restore(fn):
        def call(slots):
            start = perf_counter_ns()
            try:
                return fn(slots)
            finally:
                restores.append(dict(ms=(perf_counter_ns()-start)/1e6,
                                     slot_count=1 if legacy else slots.numel(),
                                     index_ptr=None if legacy else slots.data_ptr()))
        return call

    step = engine.step

    def measured_step():
        restores.clear()
        prior = dict(runner.mtp_stats)
        active = len(engine.scheduler.running)
        start = perf_counter_ns()
        result = step()
        delta = {key: value-prior[key] for key, value in runner.mtp_stats.items()}
        if delta["verify_steps"]:
            if not legacy and restores:
                if len(restores) != len(runner.gdn_layers) or len({r["index_ptr"] for r in restores}) != 1:
                    raise RuntimeError("Each layer must restore once with the shared index")
            rows.append(dict(wall_ms=(perf_counter_ns()-start)/1e6, active_before=active,
                             **delta, rollback_calls=len(restores),
                             rollback_ms=sum(r["ms"] for r in restores), restores=list(restores)))
        return result

    with ExitStack() as stack:
        for layer in runner.gdn_layers:
            stack.enter_context(patch.object(layer, name, timed_restore(getattr(layer, name))))
        stack.enter_context(patch.object(engine, "step", measured_step))
        yield rows


@torch.inference_mode()
def measure(args):
    methods, reference_hashes = reference_methods(ROOT, args.reference)
    prompt = next(c["prompt"] for c in make_cases(AutoTokenizer.from_pretrained(args.model))
                  if c["name"] == "synthetic-64-128")
    paths = [*ROOT.joinpath("src/nano_qwen").rglob("*.py"), Path(__file__).resolve(),
             ROOT / "benchmarks/mtp_rollback_reference.py"]
    report = dict(model=args.model, mtp_model=args.mtp_model, input=64, output=128, repeats=2,
                  prefill_graph=False, decode_graph=True, batches=args.batches,
                  protocol="same engine; off1/legacy1/batched1/off2/legacy2/batched2; host-only diagnostic afterwards",
                  environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda),
                  reference=dict(commit=args.reference, sha256=reference_hashes),
                  source=dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                              sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),
                  timing=[], host=[], state_gates={}, tuning={})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")

    for batch in args.batches:
        engine = LLMEngine(args.model, max_num_seqs=batch, max_num_batched_tokens=1024,
                           max_model_len=1024, gpu_memory_utilization=.85, enforce_eager=False,
                           use_prefill_cudagraph=False, enable_mtp=True, mtp_model=args.mtp_model,
                           mtp_verify_mode="reuse")
        prompts, expected = [prompt]*batch, None
        runner = engine.model_runner
        try:
            # Compile all restore sizes without touching a live request.
            for count in range(1, batch+1):
                indices = torch.arange(count, device="cuda", dtype=torch.int64)
                for layer in runner.gdn_layers:
                    layer.rollback_speculative_states(indices)
            for mode in ("off", "legacy", "batched"):
                with execution_mode(engine, mode != "off", Sampler()), \
                     legacy_rollback(runner, methods) if mode == "legacy" else nullcontext():
                    run_batch(engine, prompts, 8)
            frozen = tuning_choices()
            report["tuning"][str(batch)] = dict(before=frozen, same_engine=True)
            for repeat in (1, 2):
                for mode in ("off", "legacy", "batched"):
                    with execution_mode(engine, mode != "off", Sampler()), \
                         legacy_rollback(runner, methods) if mode == "legacy" else nullcontext():
                        metrics, tokens = run_batch(engine, prompts, 128)
                    expected = tokens if expected is None else expected
                    strict = tokens == expected and all(len(t) == 128 for t in tokens)
                    report["timing"].append(dict(batch=batch, mode=mode, repeat=repeat,
                                                 metrics=metrics, completion=tokens, strict_pass=strict))
                    save()
                    print(f"B{batch} {mode}{repeat}: {metrics['tokens_per_s']:.2f} tok/s strict={strict}", flush=True)
                    if not strict:
                        raise RuntimeError("Strict completion failed; timings retained for diagnosis")
                    if tuning_choices() != frozen:
                        raise RuntimeError("Tuning configuration changed during the paired experiment")
            for mode in ("legacy", "batched"):
                with legacy_rollback(runner, methods) if mode == "legacy" else nullcontext(), \
                     host_probe(engine, mode == "legacy") as rows:
                    _, tokens = run_batch(engine, prompts, 128)
                if tokens != expected:
                    raise RuntimeError("Host probe changed completion")
                report["host"].append(dict(batch=batch, mode=mode, rows=rows, completion=tokens,
                                           strict_pass=True, note="Diagnostic cohort; excluded from formal timing means"))
                save()
                rejects = [r for r in rows if r["rejected_tokens"]]
                print(f"B{batch} {mode} host: {len(rejects)} reject rounds, "
                      f"{sum(r['rollback_calls'] for r in rows)} restore calls, "
                      f"{sum(r['rollback_ms'] for r in rows):.3f}ms host", flush=True)
            # Separate correctness-only cohorts: CPU state copies must never
            # perturb either the formal samples or the host-only diagnostic.
            with legacy_rollback(runner, methods):
                old_tokens, old_states = checked_prefixes(engine, prompts)
            new_tokens, new_states = checked_prefixes(engine, prompts)
            if old_tokens != new_tokens or new_tokens != expected:
                raise RuntimeError("State-gate completion mismatch")
            report["state_gates"][str(batch)] = compare_prefixes(old_states, new_states)
            del old_states, new_states
            print(f"B{batch} state gates: accepted/rejected completion, GDN and valid target/MTP KV bitwise=True", flush=True)
            report["tuning"][str(batch)].update(after=tuning_choices(), unchanged=tuning_choices() == frozen)
            save()
        finally:
            engine.exit()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mtp-model", required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if any(not 1 <= b <= 8 for b in args.batches):
        parser.error("batch must be in1..8")
    measure(args)
