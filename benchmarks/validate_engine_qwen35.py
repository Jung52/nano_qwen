"""Engine-level correctness validation for the Qwen3.5 inference chain.

Validates the full production path (Scheduler -> MRV2 batch queue ->
persistent InputBatch -> prefill prepare -> packed-varlen GDN prefill ->
persistent state -> sample -> batched decode -> CUDA Graph -> async D2H ->
postprocess) against a conservative eager baseline, using a deterministic
benchmark-only ArgmaxSampler. Near-tie argmax changes within bf16 tolerance
are reported separately from clear token mismatches.

Kernel-level GDN correctness/perf stays in ``bench_gdn_flashqla.py``; this
script only asks: does the kernel stay correct once wired into the engine,
and do the engine paths agree with each other?

Usage::

    python benchmarks/validate_engine_qwen35.py --model /path/to/model
    python benchmarks/validate_engine_qwen35.py --model /path/to/model --mode C

Each mode runs in its own subprocess (dist.init_process_group can only be
initialized once per process).  ``--mode parent`` (default) spawns
BASELINE/A/B/C sequentially and prints the comparison matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from typing import Any, Callable

import torch

from engine_bench_utils import (
    MODES,
    ArgmaxSampler,
    ModeConfig,
    env_info,
    make_const_prompt,
    make_engine,
    make_params,
    make_random_prompt,
    repo_log_path,
    run_until_idle,
    save_json,
)

try:
    from nano_qwen.engine.sequence import Sequence
except ImportError:  # pragma: no cover
    Sequence = None  # imported again inside cases (path set up by utils)


# ---------------------------------------------------------------------------
# Case implementations
# ---------------------------------------------------------------------------

def case_w1_single_request(engine) -> dict[str, Any]:
    """12 prompt lengths around the GDN CHUNK_SIZE=64 boundary, 1 request each."""
    vocab = engine.config.hf_config.vocab_size
    lengths = [1, 31, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511]
    signature: list[list[int]] = []
    count_ok = True
    for length in lengths:
        seq = Sequence(make_random_prompt(length, seed=length, vocab_size=vocab), make_params(8))
        engine.scheduler.add(seq)
        res = run_until_idle(engine, [seq])
        comp = res.completions[seq.seq_id]
        signature.append(comp)
        count_ok &= len(comp) == 8
    return {
        "pass": count_ok,
        "detail": "" if count_ok else "completion count != max_tokens",
        "signature": signature,
    }


def case_w2_varlen_prefill(engine, mode: ModeConfig) -> dict[str, Any]:
    """Packed-varlen prefill batches; same-total layouts must not cross-talk."""
    vocab = engine.config.hf_config.vocab_size
    layouts = [
        [64, 64],          # captures the 128-token piecewise bucket first
        [32, 96],          # must REUSE that bucket without request cross-talk
        [63, 65],
        [1, 127],
        [17, 31, 80],
        [31, 33, 64],
        [65, 127, 129],
    ]
    signature: list[list[int]] = []
    count_ok = True
    bucket_note = ""
    if mode.name == "C":
        runner = engine.model_runner
        bucket_note = f"buckets_before={sorted(runner.prefill_piecewise_graphs)}"
        before = set(runner.prefill_piecewise_graphs)

    for i, layout in enumerate(layouts):
        seqs = [
            Sequence(make_random_prompt(L, seed=1000 + i * 10 + j, vocab_size=vocab), make_params(8))
            for j, L in enumerate(layout)
        ]
        for s in seqs:
            engine.scheduler.add(s)
        run_until_idle(engine, seqs)
        for s in seqs:
            comp = s.completion_token_ids
            signature.append(comp)
            count_ok &= len(comp) == 8

        if mode.name == "C" and layout == [64, 64]:
            after = set(engine.model_runner.prefill_piecewise_graphs)
            if 128 not in after:
                return {"pass": False, "detail": "bucket 128 not captured", "signature": signature}
            before = after  # [32,96] must not capture anything new
        if mode.name == "C" and layout == [32, 96]:
            after = set(engine.model_runner.prefill_piecewise_graphs)
            if after != before:
                bucket_note += f"; unexpected new buckets after [32,96]: {sorted(after - before)}"
                return {"pass": False, "detail": bucket_note, "signature": signature}

    return {
        "pass": count_ok,
        "detail": bucket_note,
        "signature": signature,
    }


def case_w3_conv_boundary(engine) -> dict[str, Any]:
    """Two disjoint-content requests must not pollute each other's causal conv."""
    vocab = engine.config.hf_config.vocab_size
    token_a, token_b = 123, 4567
    layouts = [(63, 65), (64, 64)]
    signature: list[list[int]] = []
    isolation_ok = True
    for la, lb in layouts:
        pa = make_const_prompt(la, token_a)
        pb = make_const_prompt(lb, token_b)
        # solo runs
        a_solo = Sequence(pa, make_params(8))
        engine.scheduler.add(a_solo)
        run_until_idle(engine, [a_solo])
        b_solo = Sequence(pb, make_params(8))
        engine.scheduler.add(b_solo)
        run_until_idle(engine, [b_solo])
        # packed run
        a_packed = Sequence(pa, make_params(8))
        b_packed = Sequence(pb, make_params(8))
        for s in (a_packed, b_packed):
            engine.scheduler.add(s)
        run_until_idle(engine, [a_packed, b_packed])
        a_out, b_out = a_packed.completion_token_ids, b_packed.completion_token_ids
        isolation_ok &= a_out == a_solo.completion_token_ids
        isolation_ok &= b_out == b_solo.completion_token_ids
        signature.extend([a_solo.completion_token_ids, b_solo.completion_token_ids, a_out, b_out])
    return {
        "pass": isolation_ok,
        "detail": "" if isolation_ok else "packed output != solo output (conv boundary pollution)",
        "signature": signature,
    }


def _gdn_checksum(layers, slot: int) -> float:
    total = 0.0
    for layer in layers:
        total += layer.conv_states[slot].abs().sum().item()
        total += layer.recurrent_states[slot].abs().sum().item()
    return total


def _gdn_finite(layers, slot: int) -> bool:
    return all(
        torch.isfinite(layer.conv_states[slot]).all().item()
        and torch.isfinite(layer.recurrent_states[slot]).all().item()
        for layer in layers
    )


def case_w4_state_lifecycle(model: str, mode: ModeConfig) -> dict[str, Any]:
    """GDN persistent state: nonzero+finite after prefill, mutated by decode."""
    engine = make_engine(model, mode, max_num_seqs=1, sampler=ArgmaxSampler(), reset_cuda_stats=True)
    try:
        vocab = engine.config.hf_config.vocab_size
        seq = Sequence(make_random_prompt(100, seed=7, vocab_size=vocab), make_params(8))
        engine.scheduler.add(seq)
        engine.step()  # prefill + first sample (queue=1: consumed synchronously)
        assert seq.num_completion_tokens == 1, f"expected 1 completion after prefill step, got {seq.num_completion_tokens}"
        slot = engine.model_runner.input_batch.seq_id_to_slot[seq.seq_id]
        gdn = engine.model_runner.gdn_layers
        layers = [gdn[0], gdn[len(gdn) // 2], gdn[-1]]
        finite_pre = _gdn_finite(layers, slot)
        sum_pre = _gdn_checksum(layers, slot)
        engine.step()  # one decode step
        finite_post = _gdn_finite(layers, slot)
        sum_post = _gdn_checksum(layers, slot)
        run_until_idle(engine, [seq])  # drain the rest

        ok = finite_pre and finite_post and sum_pre > 0 and sum_post != sum_pre
        detail = (
            f"slot={slot} sum_prefill={sum_pre:.4g} sum_decode={sum_post:.4g} "
            f"finite_pre={finite_pre} finite_post={finite_post}"
        )
        if not ok:
            detail = "FAIL: " + detail
        return {
            "pass": ok,
            "detail": detail,
            "signature": [seq.completion_token_ids],
        }
    finally:
        engine.exit()


def case_w5_slot_reuse(model: str, mode: ModeConfig, args) -> dict[str, Any]:
    """GDN slot reuse: request B after request A must equal B on a fresh engine."""
    def run_b(engine, prompt_b):
        seq = Sequence(prompt_b, make_params(12))
        engine.scheduler.add(seq)
        engine.step()
        slot = engine.model_runner.input_batch.seq_id_to_slot.get(seq.seq_id)
        run_until_idle(engine, [seq])
        return seq.completion_token_ids, slot

    eng1 = make_engine(model, mode, max_num_seqs=1, sampler=ArgmaxSampler(), reset_cuda_stats=True)
    try:
        vocab = eng1.config.hf_config.vocab_size
        prompt_a = make_const_prompt(100, 111)
        prompt_b = make_random_prompt(80, seed=9, vocab_size=vocab)
        a = Sequence(prompt_a, make_params(16))
        eng1.scheduler.add(a)
        run_until_idle(eng1, [a])
        b_reused, slot_reused = run_b(eng1, prompt_b)
        a_gone = a.seq_id not in eng1.model_runner.input_batch.seq_id_to_slot
        b_reused_slot = slot_reused == 0 and a_gone
    finally:
        eng1.exit()

    eng2 = make_engine(model, mode, max_num_seqs=1, sampler=ArgmaxSampler(), reset_cuda_stats=True)
    try:
        b_fresh, slot_fresh = run_b(eng2, prompt_b)
    finally:
        eng2.exit()

    ok = b_reused == b_fresh and b_reused_slot and slot_fresh == 0
    return {
        "pass": ok,
        "detail": f"slot_reused={slot_reused} slot_fresh={slot_fresh} "
                  f"reused==fresh: {b_reused == b_fresh} (len_reused={len(b_reused)}, len_fresh={len(b_fresh)})",
        "signature": [b_reused, b_fresh],
    }


def case_w6_batched_decode(engine) -> dict[str, Any]:
    """Batch sizes 1/2/3/4/5/8 (graph hits 1/2/4/8, eager fallback 3/5)."""
    vocab = engine.config.hf_config.vocab_size
    prompts = [make_random_prompt(128, seed=i, vocab_size=vocab) for i in range(8)]

    solo: list[list[int]] = []
    for p in prompts:
        seq = Sequence(p, make_params(16))
        engine.scheduler.add(seq)
        run_until_idle(engine, [seq])
        solo.append(seq.completion_token_ids)

    per_bs: dict[int, list[list[int]]] = {}
    for bs in (1, 2, 3, 4, 5, 8):
        seqs = [Sequence(prompts[i], make_params(16)) for i in range(bs)]
        for s in seqs:
            engine.scheduler.add(s)
        run_until_idle(engine, seqs)
        per_bs[bs] = [s.completion_token_ids for s in seqs]

    mismatches = []
    for bs in (1, 2, 3, 4, 5, 8):
        for i in range(bs):
            b, s = per_bs[bs][i], solo[i]
            if b != s:
                pos = next((j for j, (x, y) in enumerate(zip(b, s)) if x != y), min(len(b), len(s)))
                mismatches.append(f"bs{bs}.seq{i}@pos{pos}: solo={s[pos:pos + 4]} batched={b[pos:pos + 4]}")
    ok = not mismatches
    detail = "" if ok else "mismatches: " + "; ".join(mismatches[:8])
    return {
        "pass": ok,
        "detail": detail,
        "signature": solo,
        "per_bs": {str(k): v for k, v in per_bs.items()},
    }


def case_w7_depth2_queue(engine) -> dict[str, Any]:
    """queue=1 vs queue=2 must agree; scheduler sets never overlap."""
    vocab = engine.config.hf_config.vocab_size
    lengths = [32, 64, 96, 128, 160, 192, 224, 256]

    def run_q(queue: int):
        engine.max_concurrent_batches = queue
        seqs = [Sequence(make_random_prompt(L, seed=L, vocab_size=vocab), make_params(16)) for L in lengths]
        for s in seqs:
            engine.scheduler.add(s)
        violations: list[str] = []

        def check_invariants() -> str | None:
            sched = engine.scheduler
            w = {s.seq_id for s in sched.waiting}
            r = {s.seq_id for s in sched.running}
            if w & r:
                violations.append("seq_id in both waiting and running")
            if (w | r) & sched.in_flight:
                violations.append("in-flight seq_id also in waiting/running")

        run_until_idle(engine, seqs, after_step=check_invariants)
        final_ok = (
            engine.scheduler.is_finished()
            and not engine.batch_queue
            and not engine.scheduler.in_flight
        )
        outs = [s.completion_token_ids for s in seqs]
        return outs, violations, final_ok

    q1, v1, f1 = run_q(1)
    q2, v2, f2 = run_q(2)
    ok = q1 == q2 and not v1 and not v2 and f1 and f2
    detail = f"violations={v1 + v2} final_ok=({f1},{f2})"
    return {
        "pass": ok,
        "detail": "" if ok else "queue mismatch or scheduler invariant violated: " + detail,
        "signature": q2,
    }


def case_w8_staggered_completion(engine) -> dict[str, Any]:
    """Staggered max_tokens: early finisher frees its slot, a late request enters."""
    vocab = engine.config.hf_config.vocab_size
    max_toks = [1, 2, 4, 8, 16]

    def run_q(queue: int):
        engine.max_concurrent_batches = queue
        seqs = [Sequence(make_random_prompt(64, seed=100 + t, vocab_size=vocab), make_params(t)) for t in max_toks]
        late_holder: list[Sequence] = []

        def submit_late() -> str | None:
            if not late_holder and seqs[0].num_completion_tokens >= 1:
                late = Sequence(make_random_prompt(32, seed=999, vocab_size=vocab), make_params(8))
                engine.scheduler.add(late)
                late_holder.append(late)
            return None

        for s in seqs:
            engine.scheduler.add(s)
        run_until_idle(engine, seqs, after_step=submit_late)
        outs = [s.completion_token_ids for s in seqs]
        outs.append(late_holder[0].completion_token_ids if late_holder else [])
        return outs

    q1 = run_q(1)
    q2 = run_q(2)
    mismatches = []
    for i, (a, b) in enumerate(zip(q1, q2)):
        if a != b:
            pos = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            mismatches.append(f"req{i}@pos{pos}: q1={a[pos:pos + 4]} q2={b[pos:pos + 4]}")
    ok = not mismatches and len(q1[-1]) == 8
    return {
        "pass": ok,
        "detail": "" if ok else "mismatches: " + "; ".join(mismatches[:8]),
        "signature": q2,
        "queue1": q1,
    }


def case_w9_async_d2h(engine) -> dict[str, Any]:
    """sync D2H vs async D2H (double-buffered copy stream) must agree."""
    vocab = engine.config.hf_config.vocab_size
    lengths = [32, 64, 96, 128, 160, 192, 224, 256]

    def run_d2h(async_flag: bool):
        engine.model_runner.async_output = async_flag
        seqs = [Sequence(make_random_prompt(L, seed=200 + L, vocab_size=vocab), make_params(32)) for L in lengths]
        for s in seqs:
            engine.scheduler.add(s)
        run_until_idle(engine, seqs)
        return [s.completion_token_ids for s in seqs]

    sync_out = run_d2h(False)
    async_out = run_d2h(True)
    ok = sync_out == async_out
    return {
        "pass": ok,
        "detail": "" if ok else "async D2H output != sync D2H output (buffer race)",
        "signature": async_out,
    }


def case_w10_long_decode(engine) -> dict[str, Any]:
    """128 decode steps, batch 1 and 4: finite logits/states, exact counts."""
    vocab = engine.config.hf_config.vocab_size
    gdn = engine.model_runner.gdn_layers
    layers = [gdn[0], gdn[len(gdn) // 2], gdn[-1]]
    signature: list[list[int]] = []
    state_ok = True
    for bs in (1, 4):
        seqs = [Sequence(make_random_prompt(128, seed=300 + i, vocab_size=vocab), make_params(128)) for i in range(bs)]
        for s in seqs:
            engine.scheduler.add(s)
        run_until_idle(engine, seqs)  # ArgmaxSampler asserts logits finiteness
        for s in seqs:
            signature.append(s.completion_token_ids)
        for layer in layers:
            state_ok &= torch.isfinite(layer.conv_states).all().item()
            state_ok &= torch.isfinite(layer.recurrent_states).all().item()
    count_ok = all(len(c) == 128 for c in signature)
    return {
        "pass": count_ok and state_ok,
        "detail": "" if count_ok and state_ok else f"count_ok={count_ok} state_finite={state_ok}",
        "signature": signature,
    }


def case_known_limitation_chunked_prefill(model: str, mode: ModeConfig) -> dict[str, Any]:
    """KNOWN LIMITATION: chunked prefill raises in prepare_prefill."""
    engine = make_engine(
        model, mode,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        sampler=ArgmaxSampler(),
        reset_cuda_stats=True,
    )
    try:
        vocab = engine.config.hf_config.vocab_size
        seq = Sequence(make_random_prompt(512, seed=1, vocab_size=vocab), make_params(4))
        engine.scheduler.add(seq)
        observed = "NO ERROR (unexpected: chunked prefill completed)"
        try:
            run_until_idle(engine, [seq])
        except RuntimeError as e:
            observed = str(e)
        expected_phrase = "prefix-cache prefill is not supported by this runner"
        is_known_limitation = expected_phrase in observed
        return {
            "pass": is_known_limitation,
            "detail": f"observed: {observed!r}",
            "signature": [observed],
        }
    finally:
        engine.exit()


# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------

COMPARE_CASES: list[tuple[str, Callable[..., dict[str, Any]], str]] = [
    ("w1_single_request", lambda eng, mode: case_w1_single_request(eng),
     "12 prompt lengths (64-chunk boundary), 1 request"),
    ("w2_varlen_prefill", lambda eng, mode: case_w2_varlen_prefill(eng, mode),
     "packed-varlen prefill layouts, same-bucket graph reuse"),
    ("w3_conv_boundary", lambda eng, mode: case_w3_conv_boundary(eng),
     "causal conv request-boundary isolation (A/B patterns)"),
    ("w6_batched_decode", lambda eng, mode: case_w6_batched_decode(eng),
     "batched decode isolation, bs 1/2/3/4/5/8"),
    ("w7_depth2_queue", lambda eng, mode: case_w7_depth2_queue(eng),
     "queue=1 vs queue=2 + scheduler invariants"),
    ("w8_staggered_completion", lambda eng, mode: case_w8_staggered_completion(eng),
     "staggered completion with late request into freed slot"),
    ("w9_async_d2h", lambda eng, mode: case_w9_async_d2h(eng),
     "sync vs async D2H double-buffered copy"),
    ("w10_long_decode", lambda eng, mode: case_w10_long_decode(eng),
     "128-step decode, batch 1/4, finite logits/state"),
]

LOCAL_CASES: list[tuple[str, Callable[..., dict[str, Any]], str]] = [
    ("w4_state_lifecycle", lambda model, mode, args: case_w4_state_lifecycle(model, mode),
     "GDN state lifecycle (nonzero after prefill, mutated by decode)"),
    ("w5_slot_reuse", lambda model, mode, args: case_w5_slot_reuse(model, mode, args),
     "GDN slot reuse vs fresh engine"),
    ("known_limit_chunked_prefill", lambda model, mode, args: case_known_limitation_chunked_prefill(model, mode),
     "KNOWN LIMITATION: chunked prefill"),
]


def run_child(args) -> None:
    mode = next(m for m in MODES if m.name == args.mode)
    results: dict[str, Any] = {
        "meta": {
            **env_info(),
            "mode": mode.name,
            "model": args.model,
            "config": {
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            },
        },
        "compare_cases": {},
        "local_cases": {},
    }
    print(f"[{mode.name}] building main engine (max_num_seqs={args.max_num_seqs})...")
    sampler = ArgmaxSampler()
    engine = make_engine(
        args.model, mode,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        sampler=sampler,
    )
    try:
        for case_id, fn, desc in COMPARE_CASES:
            if args.cases and not any(case_id.startswith(c) for c in args.cases):
                continue
            print(f"[{mode.name}] {case_id}: {desc} ...", flush=True)
            try:
                sampler.reset_stats()
                out = fn(engine, mode)
                out["ambiguous_argmax_rows"] = sampler.ambiguous_rows
                out["argmax_rows"] = sampler.total_rows
                out["description"] = desc
                results["compare_cases"][case_id] = out
                print(f"[{mode.name}]   -> {'PASS' if out['pass'] else 'FAIL'} ({out['detail'][:120]})", flush=True)
            except Exception:
                results["compare_cases"][case_id] = {
                    "pass": False,
                    "detail": "exception",
                    "error": traceback.format_exc(),
                    "description": desc,
                }
                print(f"[{mode.name}]   -> EXCEPTION\n{traceback.format_exc()}", flush=True)
    finally:
        engine.exit()

    # Local cases (state lifecycle / slot reuse / known limitation) are
    # mode-independent; run them in the BASELINE child only.
    if mode.name == "BASELINE":
        for case_id, fn, desc in LOCAL_CASES:
            if args.cases and not any(case_id.startswith(c) for c in args.cases):
                continue
            print(f"[{mode.name}] {case_id}: {desc} ...", flush=True)
            try:
                out = fn(args.model, mode, args)
                out["description"] = desc
                results["local_cases"][case_id] = out
                print(f"[{mode.name}]   -> {'PASS' if out['pass'] else 'FAIL'} ({out['detail'][:120]})", flush=True)
            except Exception:
                results["local_cases"][case_id] = {
                    "pass": False,
                    "detail": "exception",
                    "error": traceback.format_exc(),
                    "description": desc,
                }
                print(f"[{mode.name}]   -> EXCEPTION\n{traceback.format_exc()}", flush=True)

    out_path = args.json_out or repo_log_path(f"validate_engine_qwen35_{mode.name}")
    save_json(out_path, results)
    print(f"[{mode.name}] results -> {out_path}")


def run_parent(args) -> None:
    mode_names = ["BASELINE", "A", "B", "C"]
    per_mode: dict[str, dict[str, Any]] = {}
    base = args.json_out or None
    for name in mode_names:
        out_path = (base + f".{name}.json") if base else repo_log_path(f"validate_engine_qwen35_{name}")
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--model", args.model,
            "--mode", name,
            "--json-out", out_path,
            "--max-num-seqs", str(args.max_num_seqs),
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            "--max-model-len", str(args.max_model_len),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        ]
        if args.cases:
            cmd += ["--cases", args.cases]
        print(f"=== spawning mode {name}: {' '.join(cmd)}")
        completed = subprocess.run(cmd, text=True)
        if completed.returncode != 0:
            print(f"!!! mode {name} subprocess failed (rc={completed.returncode})")
            per_mode[name] = {"child_rc": completed.returncode}
            continue
        with open(out_path, encoding="utf-8") as f:
            per_mode[name] = json.load(f)

    baseline = per_mode.get("BASELINE", {})
    compare = baseline.get("compare_cases", {})

    print("\n" + "=" * 100)
    print("ENGINE CORRECTNESS MATRIX (signature equality vs BASELINE)")
    print("=" * 100)
    header = f"{'case':<26}" + "".join(f"{m:>10}" for m in mode_names)
    print(header)
    print("-" * 66)
    all_pass = True
    for case_id in compare:
        row = [f"{case_id:<26}"]
        bcase = compare[case_id]
        row.append(f"{'PASS' if bcase.get('pass') else 'FAIL':>10}")
        for m in ("A", "B", "C"):
            c = per_mode.get(m, {}).get("compare_cases", {}).get(case_id)
            if c is None:
                status = "N/A"
            elif c.get("error"):
                status = "ERROR"
            elif not c.get("pass"):
                status = "FAIL"
            else:
                same = c.get("signature") == bcase.get("signature")
                ambiguous = bool(
                    bcase.get("ambiguous_argmax_rows", 0)
                    or c.get("ambiguous_argmax_rows", 0)
                )
                status = "PASS" if same else ("NUMERIC_TIE" if ambiguous else "MISMATCH")
            row.append(f"{status:>10}")
            if status not in ("PASS", "N/A", "NUMERIC_TIE"):
                all_pass = False
        print("".join(row))
        if bcase.get("detail"):
            print(f"  {'':26}baseline detail: {bcase['detail'][:200]}")

    print("\nLOCAL CASES (BASELINE child)")
    for case_id, c in baseline.get("local_cases", {}).items():
        status = "PASS" if c.get("pass") else "FAIL"
        if case_id == "known_limit_chunked_prefill":
            status = "KNOWN_LIMITATION" if c.get("pass") else "UNEXPECTED"
        print(f"  {case_id:<30} {status:<20} {c.get('detail', '')[:160]}")
        if case_id == "known_limit_chunked_prefill" and not c.get("pass"):
            all_pass = False

    aggregate = {
        "meta": env_info(),
        "matrix": {
            case_id: {
                m: (
                    {"pass": True, "matches_baseline": True}
                    if m == "BASELINE"
                    else _match_status(per_mode, m, case_id, compare[case_id])
                )
                for m in mode_names
            }
            for case_id in compare
        },
        "local_cases": baseline.get("local_cases", {}),
    }
    agg_path = save_json(base or repo_log_path("validate_engine_qwen35_aggregate"), aggregate)
    print(f"\naggregate JSON: {agg_path}")
    print("OVERALL:", "ALL PASS" if all_pass else "FAILURES PRESENT")


def _match_status(per_mode: dict, m: str, case_id: str, bcase: dict) -> dict:
    c = per_mode.get(m, {}).get("compare_cases", {}).get(case_id)
    if c is None:
        return {"pass": False, "matches_baseline": False, "note": "child missing"}
    return {
        "pass": bool(c.get("pass")),
        "matches_baseline": c.get("signature") == bcase.get("signature"),
        "numeric_tie": bool(
            (bcase.get("ambiguous_argmax_rows", 0) or 0)
            or (c.get("ambiguous_argmax_rows", 0) or 0)
        ),
        "error": c.get("error", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Engine-level correctness validation (Qwen3.5)")
    parser.add_argument("--model", default=os.environ.get("NANO_QWEN_MODEL", "/home/wei/code/models/qwen"))
    parser.add_argument("--mode", default="parent", choices=["parent"] + [m.name for m in MODES])
    parser.add_argument("--json-out", default="")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--cases", default="", help="comma-separated case ids to run (default: all)")
    args = parser.parse_args()
    args.cases = set(c for c in args.cases.split(",") if c)

    if args.mode == "parent":
        run_parent(args)
    else:
        run_child(args)


if __name__ == "__main__":
    main()
