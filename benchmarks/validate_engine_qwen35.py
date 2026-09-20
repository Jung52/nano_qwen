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
    python benchmarks/validate_engine_qwen35.py --model /path/to/model --cases w11,w12

W11/W12 verify chunk-path correctness, finite GDN/KV states, bounded state
drift, and exact tokens (or a first mismatch classified as a bounded bf16
near-tie). One-shot and chunked/mixed batches intentionally use different
GEMM/kernel shapes, so elementwise allclose, RMSE, and max_abs are reported
as drift diagnostics rather than pass/fail gates. State snapshots transfer
to CPU and synchronize; do not use their runtimes as performance
measurements. Test-only runtime budgets change on one idle engine so model
weights need not be reloaded for every chunk size.

Each mode runs in its own subprocess (dist.init_process_group can only be
initialized once per process).  ``--mode parent`` (default) spawns
BASELINE/A/B/C sequentially and prints the comparison matrix.
"""

from __future__ import annotations

import argparse
import json
import math
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
    validation_settings,
    drive_probe,
    compare_probed_request,
    compare_probed_request_drift,
    competing_tokens_are_tied,
    sampling_diagnostic_summary,
    STATE_DRIFT_RMSE_LIMITS,
    tensor_snapshot_diff,
    token_mismatch_records,
)

try:
    from nano_qwen.engine.sequence import Sequence
except ImportError:  # pragma: no cover
    Sequence = None  # imported again inside cases (path set up by utils)


# ---------------------------------------------------------------------------
# Case implementations
# ---------------------------------------------------------------------------

def _state_fingerprint_diff(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    *,
    atol: float = 5e-4,
    rtol: float = 5e-3,
) -> dict[str, Any]:
    """Compare compact per-layer GDN state summaries at one token."""
    if not baseline or not candidate:
        return {"available": False}
    lhs = {item["layer_idx"]: item for item in baseline.get("gdn_state", [])}
    rhs = {item["layer_idx"]: item for item in candidate.get("gdn_state", [])}
    common = sorted(lhs.keys() & rhs.keys())
    first_bad_layer = None
    largest: dict[str, Any] = {
        "abs_delta": 0.0,
        "layer_idx": None,
        "state": None,
        "metric": None,
    }
    for layer_idx in common:
        layer_bad = False
        for state_name in ("conv", "recurrent"):
            for metric in ("mean", "abs_mean", "rms", "max_abs"):
                a = float(lhs[layer_idx][state_name][metric])
                b = float(rhs[layer_idx][state_name][metric])
                delta = abs(a - b)
                if delta > largest["abs_delta"]:
                    largest = {
                        "abs_delta": delta,
                        "layer_idx": layer_idx,
                        "state": state_name,
                        "metric": metric,
                        "baseline": a,
                        "candidate": b,
                    }
                layer_bad |= delta > atol + rtol * max(abs(a), abs(b))
        if layer_bad and first_bad_layer is None:
            first_bad_layer = layer_idx
    return {
        "available": bool(common),
        "compared_layers": len(common),
        "first_layer_outside_tolerance": first_bad_layer,
        "largest_metric_delta": largest,
        "atol": atol,
        "rtol": rtol,
    }


def _first_signature_mismatch(
    baseline: Any,
    candidate: Any,
    path: tuple[int, ...] = (),
) -> dict[str, Any] | None:
    if isinstance(baseline, list) and isinstance(candidate, list):
        for index, (lhs, rhs) in enumerate(zip(baseline, candidate)):
            mismatch = _first_signature_mismatch(lhs, rhs, path + (index,))
            if mismatch is not None:
                return mismatch
        if len(baseline) != len(candidate):
            index = min(len(baseline), len(candidate))
            return {
                "path": list(path + (index,)),
                "baseline_token": baseline[index] if index < len(baseline) else None,
                "candidate_token": candidate[index] if index < len(candidate) else None,
                "length_mismatch": [len(baseline), len(candidate)],
            }
        return None
    if baseline != candidate:
        return {
            "path": list(path),
            "baseline_token": baseline,
            "candidate_token": candidate,
        }
    return None


def _nested_item(value: Any, path: list[int]) -> Any:
    try:
        for index in path:
            value = value[index]
        return value
    except (IndexError, KeyError, TypeError):
        return None


def _logits_summary(diagnostic: dict[str, Any] | None) -> dict[str, Any] | None:
    if not diagnostic:
        return None
    return {
        key: diagnostic.get(key)
        for key in (
            "top_tokens",
            "top_logits",
            "margin",
            "effective_tolerance",
            "near_tie",
            "logits_dtype",
        )
    }


def _signature_comparison(
    baseline_case: dict[str, Any],
    candidate_case: dict[str, Any],
) -> dict[str, Any]:
    mismatch = _first_signature_mismatch(
        baseline_case.get("signature"),
        candidate_case.get("signature"),
    )
    if mismatch is None:
        return {"matches_baseline": True, "numeric_tie": False}

    path = mismatch["path"]
    baseline_diag = _nested_item(
        baseline_case.get("signature_diagnostics"), path,
    )
    candidate_diag = _nested_item(
        candidate_case.get("signature_diagnostics"), path,
    )
    mismatch["baseline_logits"] = _logits_summary(baseline_diag)
    mismatch["candidate_logits"] = _logits_summary(candidate_diag)
    mismatch["gdn_state_diff"] = _state_fingerprint_diff(
        baseline_diag, candidate_diag,
    )
    return {
        "matches_baseline": False,
        "numeric_tie": competing_tokens_are_tied(
            baseline_diag,
            candidate_diag,
            mismatch.get("baseline_token"),
            mismatch.get("candidate_token"),
        ),
        "first_mismatch": mismatch,
    }


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
    sampler = engine.model_runner.sampler
    diagnostics_available = all(
        hasattr(sampler, name)
        for name in ("enable_diagnostics", "disable_diagnostics", "get_diagnostic")
    )
    if diagnostics_available:
        sampler.enable_diagnostics(capture_state=True)

    try:
        solo: list[list[int]] = []
        for i, prompt in enumerate(prompts):
            seq = Sequence(prompt, make_params(16))
            seq._diagnostic_key = f"w6.solo.{i}"
            engine.scheduler.add(seq)
            run_until_idle(engine, [seq])
            solo.append(seq.completion_token_ids)

        per_bs: dict[int, list[list[int]]] = {}
        for bs in (1, 2, 3, 4, 5, 8):
            seqs = [Sequence(prompts[i], make_params(16)) for i in range(bs)]
            for i, seq in enumerate(seqs):
                seq._diagnostic_key = f"w6.bs{bs}.{i}"
                engine.scheduler.add(seq)
            run_until_idle(engine, seqs)
            per_bs[bs] = [seq.completion_token_ids for seq in seqs]

        mismatches: list[dict[str, Any]] = []
        for bs in (1, 2, 3, 4, 5, 8):
            for request_index in range(bs):
                mismatch = _first_signature_mismatch(
                    solo[request_index], per_bs[bs][request_index],
                )
                if mismatch is None:
                    continue

                position = mismatch["path"][0]
                solo_diag = (
                    sampler.get_diagnostic(
                        f"w6.solo.{request_index}", position,
                    )
                    if diagnostics_available else None
                )
                batched_diag = (
                    sampler.get_diagnostic(
                        f"w6.bs{bs}.{request_index}", position,
                    )
                    if diagnostics_available else None
                )
                previous_position = position - 1 if position > 0 else None
                previous_solo_diag = (
                    sampler.get_diagnostic(
                        f"w6.solo.{request_index}", previous_position,
                    )
                    if diagnostics_available and previous_position is not None
                    else None
                )
                previous_batched_diag = (
                    sampler.get_diagnostic(
                        f"w6.bs{bs}.{request_index}", previous_position,
                    )
                    if diagnostics_available and previous_position is not None
                    else None
                )

                mismatch.update({
                    "batch_size": bs,
                    "request_index": request_index,
                    "position": position,
                    "solo_logits": _logits_summary(solo_diag),
                    "batched_logits": _logits_summary(batched_diag),
                    "numeric_tie": competing_tokens_are_tied(
                        solo_diag,
                        batched_diag,
                        mismatch.get("baseline_token"),
                        mismatch.get("candidate_token"),
                    ),
                    "gdn_state_diff": _state_fingerprint_diff(
                        solo_diag, batched_diag,
                    ),
                    "previous_position": previous_position,
                    "previous_solo_logits": _logits_summary(previous_solo_diag),
                    "previous_batched_logits": _logits_summary(
                        previous_batched_diag,
                    ),
                    "previous_gdn_state_diff": _state_fingerprint_diff(
                        previous_solo_diag, previous_batched_diag,
                    ),
                })
                mismatches.append(mismatch)

        signature_diagnostics = [
            [
                sampler.get_diagnostic(f"w6.solo.{request_index}", position)
                if diagnostics_available else None
                for position in range(len(tokens))
            ]
            for request_index, tokens in enumerate(solo)
        ]
        exact_pass = not mismatches
        all_numeric_ties = bool(mismatches) and all(
            mismatch["numeric_tie"] for mismatch in mismatches
        )
        first_mismatch = mismatches[0] if mismatches else None
        first_non_numeric_mismatch = next(
            (
                mismatch for mismatch in mismatches
                if not mismatch["numeric_tie"]
            ),
            None,
        )
        if first_mismatch is None:
            detail = ""
        else:
            detail = (
                f"first mismatch: bs{first_mismatch['batch_size']}."
                f"seq{first_mismatch['request_index']}@pos"
                f"{first_mismatch['position']}; "
                f"numeric_tie={first_mismatch['numeric_tie']}"
            )
            if first_non_numeric_mismatch is not None:
                detail += (
                    "; first non-numeric mismatch: "
                    f"bs{first_non_numeric_mismatch['batch_size']}."
                    f"seq{first_non_numeric_mismatch['request_index']}@pos"
                    f"{first_non_numeric_mismatch['position']}"
                )
        return {
            "pass": exact_pass,
            "accepted": exact_pass or all_numeric_ties,
            "detail": detail,
            "signature": solo,
            "signature_diagnostics": signature_diagnostics,
            "per_bs": {str(k): v for k, v in per_bs.items()},
            "first_mismatch": first_mismatch,
            "first_non_numeric_mismatch": first_non_numeric_mismatch,
            "mismatches": mismatches,
            "mismatch_count": len(mismatches),
            "numeric_tie_count": sum(
                int(mismatch["numeric_tie"]) for mismatch in mismatches
            ),
        }
    finally:
        if diagnostics_available:
            sampler.disable_diagnostics()


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


def _new_case_config(engine):
    args = engine.validation_args
    output_tokens = args.regression_decode_steps + 1  # token #0 comes from prefill
    if engine.config.max_model_len < 1024 + output_tokens:
        raise AssertionError("w11/w12 need --max-model-len >= 1024 + regression-decode-steps + 1")
    if engine.config.max_num_seqs < 2:
        raise AssertionError("w12 needs --max-num-seqs >= 2")
    return output_tokens, args.state_atol, args.state_rtol


def _probe_single(engine, label, prompt, output_tokens, budget, *, reference):
    with validation_settings(engine, budget=budget, reference=reference):
        seq = Sequence(prompt, make_params(output_tokens))
        engine.scheduler.add(seq)
        return drive_probe(engine, {label: seq})


def case_w11_chunked_prefill(engine, mode: ModeConfig) -> dict[str, Any]:
    """One-shot eager reference vs actual scheduler chunks, then free decode.

    Acceptance: observed chunk path, finite states, per-group relative RMSE
    drift bounds, and exact tokens (or a bounded near-tie first flip).
    Elementwise metrics stay diagnostic-only for cross-shape bf16 noise.
    """
    output_tokens, atol, rtol = _new_case_config(engine)
    vocab = engine.config.hf_config.vocab_size
    lengths = [63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1024]
    chunks = [64, 128, 256]
    runs, signature = [], []
    signature_diagnostics = []
    for length in lengths:
        prompt = make_random_prompt(length, seed=11000 + length, vocab_size=vocab)
        reference = _probe_single(engine, "Q", prompt, output_tokens, length, reference=True)
        ref_prefills = [r["q_len"] for batch in reference.batches for r in batch["rows"] if r["prefill"]]
        if ref_prefills != [length]:
            raise AssertionError(f"Reference was not a single prefill: {ref_prefills}")
        for chunk in chunks:
            try:
                candidate = _probe_single(engine, "Q", prompt, output_tokens, chunk, reference=False)
            except Exception as exc:
                raise RuntimeError(f"w11 failed during execution: prompt={length}, chunk={chunk}") from exc
            observed = [r["q_len"] for batch in candidate.batches for r in batch["rows"] if r["prefill"]]
            expected = [min(chunk, length - start) for start in range(0, length, chunk)]
            comparison = compare_probed_request_drift(
                reference, candidate, "Q", output_tokens=output_tokens,
                atol=atol, rtol=rtol,
            )
            path_ok = observed == expected
            entry = {"prompt_len": length, "chunk": chunk, "reference_prefill": ref_prefills,
                     "expected_chunks": expected, "observed_chunks": observed,
                     "actually_chunked": len(expected) > 1, "chunk_path_pass": path_ok,
                     "pass": path_ok and comparison["pass"], "comparison": comparison,
                     "batches": candidate.batches}
            runs.append(entry)
            signature.append(list(candidate.seqs["Q"].completion_token_ids))
            signature_diagnostics.append([
                sampling_diagnostic_summary(
                    candidate.diagnostics.get("Q", {}).get(position)
                )
                for position in range(output_tokens)
            ])
            print(f"[{mode.name}] w11 prompt={length} chunk={chunk}: "
                  f"{'PASS' if entry['pass'] else 'FAIL'} chunks={observed}", flush=True)
            del candidate
        del reference
    passed = all(r["pass"] for r in runs)
    first = next((r for r in runs if not r["pass"]), None)
    return {"pass": passed, "accepted": passed, "signature": signature,
            "signature_diagnostics": signature_diagnostics,
            "detail": f"{sum(r['pass'] for r in runs)}/{len(runs)} chunk-path/bounded-drift/token comparisons",
            "state_atol": atol, "state_rtol": rtol, "output_tokens": output_tokens,
            "runs": runs, "first_mismatch": None if first is None else {
                "prompt_len": first["prompt_len"], "chunk": first["chunk"],
                "chunk_path_pass": first["chunk_path_pass"],
                "token_index": first["comparison"]["first_token_mismatch"],
                "state": first["comparison"]["first_state_mismatch"]}}


def case_w12_mixed_prefill_decode(engine, mode: ModeConfig) -> dict[str, Any]:
    """Late B arrives after A has already executed one decode forward.

    Scheduler must dispatch B's 128-token prefill and A's one-token decode in
    ONE batch. Both natural free-running trajectories are compared with A and
    B executed completely independently on eager/depth1 reference paths.
    A lengths 254/255/256 put the mixed decode at positions 255/256/257.
    Two B prompts exercise isolation against different competing contents.
    """
    output_tokens, atol, rtol = _new_case_config(engine)
    if output_tokens < 4:
        raise AssertionError("w12 requires --regression-decode-steps >= 3")
    vocab = engine.config.hf_config.vocab_size
    runs, signature = [], []
    signature_diagnostics = []
    for a_len in (254, 255, 256):
        prompt_a = make_random_prompt(a_len, seed=12000 + a_len, vocab_size=vocab)
        reference_a = _probe_single(engine, "A", prompt_a, output_tokens, a_len, reference=True)
        # Different B inputs must not alter A's state or output trajectory.
        for b_seed in (12001, 12099):
            prompt_b = make_random_prompt(128, seed=b_seed, vocab_size=vocab)
            reference_b = _probe_single(engine, "B", prompt_b, output_tokens, 128, reference=True)
            with validation_settings(engine, budget=max(a_len, 129), reference=False):
                a = Sequence(prompt_a, make_params(output_tokens))
                b = Sequence(prompt_b, make_params(output_tokens))
                engine.scheduler.add(a)
                injected = False
                def late_arrival(probe):
                    nonlocal injected
                    if not injected and a.num_completion_tokens == 2:
                        # One prefill + one decode consumed; A is schedulable.
                        if engine.batch_queue or a.seq_id in engine.scheduler.in_flight:
                            raise AssertionError("A still in flight at controlled B arrival")
                        if a not in engine.scheduler.running:
                            raise AssertionError("A not in scheduler.running before B arrival")
                        engine.config.max_num_batched_tokens = 129
                        engine.scheduler.max_num_batched_tokens = 129
                        engine.scheduler.add(b)
                        injected = True
                try:
                    candidate = drive_probe(engine, {"A": a, "B": b}, after_step=late_arrival)
                except Exception as exc:
                    raise RuntimeError(f"w12 failed during execution: A={a_len}, B=128, B_seed={b_seed}") from exc
            mixed = [batch for batch in candidate.batches if any(r["prefill"] for r in batch["rows"])
                     and any(not r["prefill"] for r in batch["rows"])]
            target = [batch for batch in mixed if batch["any_prefill"]
                      and {r["label"] for r in batch["rows"]} == {"A", "B"}
                      and any(r["label"] == "A" and not r["prefill"] and r["q_len"] == 1
                              and r["completion_index"] == 2 for r in batch["rows"])
                      and any(r["label"] == "B" and r["prefill"] and r["q_len"] == 128
                              and r["start"] == 0 for r in batch["rows"])]
            path_ok = injected and len(target) == 1 and len(mixed) == 1
            # In the real scheduler order B comes first while A owns slot 0.
            # Require that nonidentity indirection was actually exercised.
            if target:
                path_ok &= ([r["label"] for r in target[0]["rows"]] == ["B", "A"]
                            and target[0]["metadata"]["state_slots"] == [1, 0]
                            and target[0]["metadata"]["paged"])
            ca = compare_probed_request_drift(
                reference_a, candidate, "A", output_tokens=output_tokens,
                atol=atol, rtol=rtol,
            )
            cb = compare_probed_request_drift(
                reference_b, candidate, "B", output_tokens=output_tokens,
                atol=atol, rtol=rtol,
            )
            entry = {"a_prompt_len": a_len, "b_prompt_len": 128, "b_seed": b_seed,
                     "pass": bool(path_ok and ca["pass"] and cb["pass"]),
                     "mixed_path_pass": bool(path_ok), "mixed_batches": mixed,
                     "A": ca, "B": cb, "batches": candidate.batches,
                     "unscheduled_isolation_checks": candidate.isolation_checks}
            runs.append(entry)
            signature.extend([list(a.completion_token_ids), list(b.completion_token_ids)])
            signature_diagnostics.extend([
                [
                    sampling_diagnostic_summary(
                        candidate.diagnostics.get("A", {}).get(position)
                    )
                    for position in range(output_tokens)
                ],
                [
                    sampling_diagnostic_summary(
                        candidate.diagnostics.get("B", {}).get(position)
                    )
                    for position in range(output_tokens)
                ],
            ])
            print(f"[{mode.name}] w12 A={a_len} B=128 seed={b_seed}: "
                  f"{'PASS' if entry['pass'] else 'FAIL'} mixed_batches={len(mixed)}", flush=True)
            del candidate, reference_b
        del reference_a
    passed = all(r["pass"] for r in runs)
    first = next((r for r in runs if not r["pass"]), None)
    return {"pass": passed, "accepted": passed, "signature": signature,
            "signature_diagnostics": signature_diagnostics,
            "detail": f"{sum(r['pass'] for r in runs)}/{len(runs)} mixed-path/bounded-drift/token comparisons",
            "state_atol": atol, "state_rtol": rtol, "output_tokens": output_tokens,
            "runs": runs, "first_mismatch": None if first is None else {
                "a_prompt_len": first["a_prompt_len"], "b_seed": first["b_seed"],
                "mixed_path_pass": first["mixed_path_pass"],
                "A_token_index": first["A"]["first_token_mismatch"],
                "B_token_index": first["B"]["first_token_mismatch"],
                "A_state": first["A"]["first_state_mismatch"],
                "B_state": first["B"]["first_state_mismatch"]}}


STABILITY_SCALES: dict[str, dict[str, Any]] = {
    "smoke": {"prompts": [1019], "chunks": [256], "batches": [1, 2],
              "seeds": [13001], "decode": 32},
    "default": {"prompts": [1019, 2043], "chunks": [128, 256], "batches": [1, 4],
                "seeds": [13001, 13099]},
    "full": {"prompts": [1019, 2043, 4091, 8189], "chunks": [128, 256, 512, 1024],
             "batches": [1, 4, 8], "seeds": [13001, 13099]},
}


def _state_summary_is_finite(summary: Any) -> bool:
    if not summary:
        return True
    layers = summary.get("gdn_state") if isinstance(summary, dict) else None
    if not isinstance(layers, list):
        return True
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        for state in ("conv", "recurrent"):
            metrics = layer.get(state)
            if not isinstance(metrics, dict):
                continue
            for value in metrics.values():
                try:
                    if not math.isfinite(float(value)):
                        return False
                except (TypeError, ValueError):
                    return False
    return True


def case_w13_chunked_mixed_stability(engine, mode: ModeConfig) -> dict[str, Any]:
    """Long chunked/mixed stability; explicit selection required (--cases w13).

    Covers ~1K..8K prompts, multiple chunk sizes, batches, seeds, and long
    decode. Checks chunk paths, per-step finite GDN summaries, finite terminal
    snapshots, bounded drift versus one-shot (when it fits the built token
    budget), near-tie-classified token divergence, idle-request isolation,
    and slot cleanup. Runtime is dominated by decode length; use the smoke
    preset while iterating.
    """
    args = engine.validation_args
    preset = STABILITY_SCALES[args.stability_scale]
    decode_tokens = int(preset.get("decode") or args.stability_decode_tokens)
    max_model_len = engine.config.max_model_len
    max_num_seqs = engine.config.max_num_seqs
    built_budget = engine.config.max_num_batched_tokens
    vocab = engine.config.hf_config.vocab_size
    limits = dict(STATE_DRIFT_RMSE_LIMITS)
    atol, rtol = args.state_atol, args.state_rtol
    snapshot_positions = {0, decode_tokens - 1}
    runs: list[dict[str, Any]] = []

    def sparse_probe(prompts, budget, *, reference, snapshot_labels=None):
        with validation_settings(engine, budget=budget, reference=reference):
            labeled = {}
            for index, prompt in enumerate(prompts):
                seq = Sequence(prompt, make_params(decode_tokens))
                labeled[f"r{index}"] = seq
                engine.scheduler.add(seq)
            return drive_probe(
                engine, labeled,
                snapshot_positions=snapshot_positions,
                snapshot_labels=snapshot_labels,
                capture_state=True, capture_logits=False,
            )

    def drift_checks(reference, candidate, label):
        ref_tokens = list(reference.seqs[label].completion_token_ids)
        got_tokens = list(candidate.seqs[label].completion_token_ids)
        first_mismatch = next(
            (
                index for index, (lhs, rhs) in enumerate(zip(ref_tokens, got_tokens))
                if lhs != rhs
            ),
            None,
        )
        positions = sorted(
            set(reference.snapshots[label]) & set(candidate.snapshots[label])
        )
        # Only checkpoints whose producing forward consumed the same sampled
        # token history can gate drift. Positions after the first flip (and
        # the flip position itself for later state groups) remain diagnostic.
        positions = [
            position for position in positions
            if first_mismatch is None or position <= first_mismatch
        ]
        checks = []
        for position in positions:
            diff = tensor_snapshot_diff(
                reference.snapshots[label][position],
                candidate.snapshots[label][position],
                atol=atol, rtol=rtol,
            )
            drift_pass = all(
                group.get("nonfinite", 1) == 0
                and group.get("relative_rmse", float("inf"))
                <= limits.get(name, float("inf"))
                for name, group in diff["groups"].items()
            )
            checks.append({
                "position": position,
                "drift_pass": drift_pass,
                "groups": {
                    name: {
                        "rmse": group["rmse"],
                        "relative_rmse": group["relative_rmse"],
                        "max_abs": group["max_abs"],
                    } for name, group in diff["groups"].items()
                },
            })
        return checks

    for prompt_len in preset["prompts"]:
        if prompt_len + decode_tokens > max_model_len:
            runs.append({"prompt_len": prompt_len, "skipped": True,
                         "reason": f"needs max_model_len >= {prompt_len + decode_tokens}"})
            continue
        for seed in preset["seeds"]:
            prompt = make_random_prompt(prompt_len, seed=seed, vocab_size=vocab)
            reference = None
            if prompt_len <= built_budget:
                reference = sparse_probe([prompt], prompt_len, reference=True)
            for chunk in preset["chunks"]:
                if chunk > min(prompt_len, built_budget):
                    continue
                expected_chunks = [
                    min(chunk, prompt_len - start)
                    for start in range(0, prompt_len, chunk)
                ]
                solo = sparse_probe([prompt], chunk, reference=False)
                for batch_size in preset["batches"]:
                    if batch_size == 1:
                        candidate, labels = solo, ["r0"]
                    else:
                        if batch_size > max_num_seqs:
                            continue
                        prompts = [
                            make_random_prompt(
                                prompt_len, seed=seed + 101 * index, vocab_size=vocab,
                            )
                            for index in range(batch_size)
                        ]
                        candidate = sparse_probe(
                            prompts, chunk, reference=False,
                            snapshot_labels={"r0"},
                        )
                        labels = [f"r{index}" for index in range(batch_size)]
                    path_ok = True
                    counts_ok = True
                    finite_ok = True
                    drift_ok = True
                    token_ok = True
                    for label in labels:
                        rows = [
                            r["q_len"] for batch in candidate.batches
                            for r in batch["rows"]
                            if r["prefill"] and r["label"] == label
                        ]
                        path_ok &= rows == expected_chunks
                        seq = candidate.seqs[label]
                        counts_ok &= len(seq.completion_token_ids) == decode_tokens
                        diagnostics = candidate.diagnostics.get(label, {})
                        finite_ok &= all(
                            _state_summary_is_finite(diagnostics.get(position))
                            for position in range(decode_tokens)
                        )
                        if (reference is not None
                                and reference.snapshots.get(label)
                                and candidate.snapshots.get(label)):
                            drift_ok &= all(
                                check["drift_pass"]
                                for check in drift_checks(reference, candidate, label)
                            )
                        solo_tokens = list(solo.seqs["r0"].completion_token_ids)
                        got_tokens = list(seq.completion_token_ids)
                        if label == "r0" and got_tokens != solo_tokens:
                            records = token_mismatch_records(
                                solo_tokens, got_tokens,
                                solo.diagnostics.get("r0", {}),
                                candidate.diagnostics.get(label, {}),
                            )
                            token_ok &= bool(records) and records[0].get("near_tie", False)
                    runs.append({
                        "prompt_len": prompt_len, "chunk": chunk,
                        "batch_size": len(labels), "seed": seed,
                        "num_chunks": len(expected_chunks),
                        "expected_chunks": expected_chunks,
                        "path_ok": bool(path_ok), "counts_ok": bool(counts_ok),
                        "finite_ok": bool(finite_ok),
                        "drift_available": reference is not None,
                        "drift_ok": bool(drift_ok),
                        "token_ok": bool(token_ok),
                        "isolation_checks": candidate.isolation_checks,
                        "drift": (
                            drift_checks(reference, candidate, "r0")
                            if reference is not None else None
                        ),
                        "pass": bool(path_ok and counts_ok and finite_ok
                                     and drift_ok and token_ok),
                    })
                    print(f"[{mode.name}] w13 P={prompt_len} C={chunk} "
                          f"B={len(labels)} seed={seed}: "
                          f"{'PASS' if runs[-1]['pass'] else 'FAIL'}", flush=True)
                del solo
            del reference

    slot_reuse_ok = False
    try:
        short = Sequence(
            make_random_prompt(64, seed=13900, vocab_size=vocab),
            make_params(4),
        )
        engine.scheduler.add(short)
        run_until_idle(engine, [short])
        slot_reuse_ok = (
            short.is_finished
            and len(short.completion_token_ids) == 4
            and not engine.model_runner.input_batch.seq_id_to_slot
        )
    except Exception:
        slot_reuse_ok = False

    executed = [r for r in runs if not r.get("skipped")]
    passed = bool(executed) and all(r["pass"] for r in executed) and slot_reuse_ok
    max_relative = {}
    for run in executed:
        if run.get("drift"):
            for check in run["drift"]:
                for name, group in check["groups"].items():
                    max_relative[name] = max(
                        max_relative.get(name, 0.0), group["relative_rmse"]
                    )
    return {
        "pass": passed, "accepted": passed, "signature": [],
        "detail": f"{sum(r['pass'] for r in executed)}/{len(executed)} stable runs; "
                  f"slot_reuse={slot_reuse_ok}; skipped={len(runs) - len(executed)}",
        "scale": args.stability_scale, "decode_tokens": decode_tokens,
        "drift_limits": limits, "max_relative_rmse": max_relative,
        "runs": runs, "slot_reuse_ok": slot_reuse_ok,
    }


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
    ("w11_chunked_prefill", case_w11_chunked_prefill,
     "one-shot vs chunks 64/128/256; full GDN/KV snapshots and free decode"),
    ("w12_mixed_prefill_decode", case_w12_mixed_prefill_decode,
     "real B-prefill128 + A-decode1 batch vs independent eager requests"),
    ("w13_chunked_mixed_stability", case_w13_chunked_mixed_stability,
     "opt-in long chunked/mixed stability (~1K..8K prompts; --cases w13)"),
]

LOCAL_CASES: list[tuple[str, Callable[..., dict[str, Any]], str]] = [
    ("w4_state_lifecycle", lambda model, mode, args: case_w4_state_lifecycle(model, mode),
     "GDN state lifecycle (nonzero after prefill, mutated by decode)"),
    ("w5_slot_reuse", lambda model, mode, args: case_w5_slot_reuse(model, mode, args),
     "GDN slot reuse vs fresh engine"),
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
    engine.validation_args = args
    try:
        for case_id, fn, desc in COMPARE_CASES:
            if not args.cases and case_id == "w13_chunked_mixed_stability":
                continue
            if args.cases and not any((case_id == c or case_id.startswith(c + "_")) for c in args.cases):
                continue
            print(f"[{mode.name}] {case_id}: {desc} ...", flush=True)
            try:
                sampler.reset_stats()
                out = fn(engine, mode)
                out["ambiguous_argmax_rows"] = sampler.ambiguous_rows
                out["argmax_rows"] = sampler.total_rows
                out["description"] = desc
                results["compare_cases"][case_id] = out
                status = (
                    "PASS" if out["pass"]
                    else "NUMERIC_TIE" if out.get("accepted")
                    else "FAIL"
                )
                print(f"[{mode.name}]   -> {status} ({out['detail'][:120]})", flush=True)
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

    # Local cases (state lifecycle / slot reuse) are
    # mode-independent; run them in the BASELINE child only.
    if mode.name == "BASELINE":
        for case_id, fn, desc in LOCAL_CASES:
            if args.cases and not any((case_id == c or case_id.startswith(c + "_")) for c in args.cases):
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
    selected_compare = [
        case_id for case_id, _, _ in COMPARE_CASES
        if (args.cases or case_id != "w13_chunked_mixed_stability")
        and (not args.cases or any((case_id == c or case_id.startswith(c + "_")) for c in args.cases))
    ]
    selected_local = [
        case_id for case_id, _, _ in LOCAL_CASES
        if not args.cases or any((case_id == c or case_id.startswith(c + "_")) for c in args.cases)
    ]
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
            "--state-atol", str(args.state_atol),
            "--state-rtol", str(args.state_rtol),
            "--regression-decode-steps", str(args.regression_decode_steps),
            "--stability-scale", args.stability_scale,
            "--stability-decode-tokens", str(args.stability_decode_tokens),
        ]
        if args.cases:
            cmd += ["--cases", ",".join(sorted(args.cases))]
        print(f"=== spawning mode {name}: {' '.join(cmd)}")
        completed = subprocess.run(cmd, text=True)
        if completed.returncode != 0:
            print(f"!!! mode {name} subprocess failed (rc={completed.returncode})")
            per_mode[name] = {"child_rc": completed.returncode}
            continue
        try:
            with open(out_path, encoding="utf-8") as f:
                per_mode[name] = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"!!! mode {name} output unreadable: {exc}")
            per_mode[name] = {"child_error": str(exc)}

    baseline = per_mode.get("BASELINE", {})
    compare = baseline.get("compare_cases", {})
    children: dict[str, dict[str, Any]] = {}
    for name in mode_names:
        child = per_mode.get(name, {})
        expected = selected_compare + (selected_local if name == "BASELINE" else [])
        present = set(child.get("compare_cases", {})) | set(child.get("local_cases", {}))
        missing = [case_id for case_id in expected if case_id not in present]
        child_error = child.get("child_error", "")
        returncode = child.get("child_rc", 0)
        children[name] = {
            "completed": returncode == 0 and not child_error and not missing,
            "returncode": returncode,
            "error": child_error,
            "missing_cases": missing,
        }

    print("\n" + "=" * 100)
    print("ENGINE CORRECTNESS MATRIX (signature equality vs BASELINE)")
    print("=" * 100)
    header = f"{'case':<26}" + "".join(f"{m:>14}" for m in mode_names)
    print(header)
    print("-" * 82)
    matrix: dict[str, dict[str, Any]] = {}
    for case_id in selected_compare:
        row = [f"{case_id:<26}"]
        bcase = compare.get(case_id)
        matrix[case_id] = {
            "BASELINE": _baseline_status(per_mode, case_id),
        }
        row.append(f"{matrix[case_id]['BASELINE']['status']:>14}")
        for m in ("A", "B", "C"):
            matrix[case_id][m] = _match_status(
                per_mode, m, case_id, bcase,
            )
            row.append(f"{matrix[case_id][m]['status']:>14}")
        print("".join(row))
        if bcase and bcase.get("detail"):
            print(f"  {'':26}baseline detail: {bcase['detail'][:200]}")
        for mode_name, result in matrix[case_id].items():
            mismatch = result.get("first_mismatch")
            if mismatch:
                print(
                    f"  {'':26}{mode_name} first mismatch: "
                    f"{json.dumps(mismatch, ensure_ascii=False)[:500]}"
                )

    print("\nLOCAL CASES (BASELINE child)")
    local_case_status: dict[str, dict[str, Any]] = {}
    local_cases = baseline.get("local_cases", {})
    for case_id in selected_local:
        case = local_cases.get(case_id)
        if case is None:
            result = {"status": "MISSING", "pass": False, "accepted": False}
            detail = "child result missing"
        elif case.get("error"):
            result = {
                "status": "ERROR",
                "pass": False,
                "accepted": False,
                "error": case["error"],
            }
            detail = case.get("detail", "")
        else:
            exact_pass = bool(case.get("pass"))
            status = "PASS" if exact_pass else "FAIL"
            result = {
                "status": status,
                "pass": exact_pass,
                "accepted": exact_pass,
            }
            detail = case.get("detail", "")
        local_case_status[case_id] = result
        print(f"  {case_id:<30} {result['status']:<20} {detail[:160]}")

    all_cells = [
        result
        for case_results in matrix.values()
        for result in case_results.values()
    ] + list(local_case_status.values())
    overall_pass = (
        bool(selected_compare or selected_local)
        and all(child["completed"] for child in children.values())
        and all(result.get("accepted", False) for result in all_cells)
    )
    overall_exact_pass = overall_pass and all(
        result.get("status") == "PASS"
        for result in all_cells
    )

    aggregate = {
        "meta": env_info(),
        "overall_pass": overall_pass,
        "overall_exact_pass": overall_exact_pass,
        "children": children,
        "matrix": matrix,
        "local_cases": local_cases,
        "local_case_status": local_case_status,
    }
    agg_path = save_json(base or repo_log_path("validate_engine_qwen35_aggregate"), aggregate)
    print(f"\naggregate JSON: {agg_path}")
    print("OVERALL:", "ALL PASS" if overall_pass else "FAILURES PRESENT")


def _baseline_status(per_mode: dict, case_id: str) -> dict[str, Any]:
    child = per_mode.get("BASELINE", {})
    if child.get("child_rc") or child.get("child_error"):
        return {
            "status": "CHILD_ERROR",
            "pass": False,
            "matches_baseline": False,
            "numeric_tie": False,
            "accepted": False,
            "error": child.get("child_error", ""),
        }
    case = child.get("compare_cases", {}).get(case_id)
    if case is None:
        return {
            "status": "MISSING",
            "pass": False,
            "matches_baseline": False,
            "numeric_tie": False,
            "accepted": False,
        }
    if case.get("error"):
        return {
            "status": "ERROR",
            "pass": False,
            "matches_baseline": False,
            "numeric_tie": False,
            "accepted": False,
            "error": case["error"],
        }
    exact_pass = bool(case.get("pass"))
    accepted = bool(case.get("accepted", exact_pass))
    status = "PASS" if exact_pass else "NUMERIC_TIE" if accepted else "FAIL"
    return {
        "status": status,
        "pass": exact_pass,
        "matches_baseline": True,
        "numeric_tie": status == "NUMERIC_TIE",
        "accepted": accepted,
        "error": "",
        "first_mismatch": case.get("first_mismatch"),
        "first_non_numeric_mismatch": case.get(
            "first_non_numeric_mismatch"
        ),
    }


def _match_status(
    per_mode: dict,
    m: str,
    case_id: str,
    bcase: dict | None,
) -> dict[str, Any]:
    child = per_mode.get(m, {})
    if child.get("child_rc") or child.get("child_error"):
        return {
            "status": "CHILD_ERROR",
            "pass": False,
            "matches_baseline": False,
            "numeric_tie": False,
            "accepted": False,
            "error": child.get("child_error", ""),
        }
    c = per_mode.get(m, {}).get("compare_cases", {}).get(case_id)
    if c is None:
        return {
            "status": "MISSING",
            "pass": False,
            "matches_baseline": False,
            "numeric_tie": False,
            "accepted": False,
        }
    if c.get("error"):
        return {
            "status": "ERROR",
            "pass": False,
            "matches_baseline": False,
            "numeric_tie": False,
            "accepted": False,
            "error": c["error"],
        }
    baseline_accepted = bool(
        bcase
        and not bcase.get("error")
        and bcase.get("accepted", bcase.get("pass", False))
    )
    if not baseline_accepted:
        return {
            "status": "BASELINE_FAIL",
            "pass": bool(c.get("pass")),
            "matches_baseline": False,
            "numeric_tie": False,
            "accepted": False,
            "error": "baseline missing or failed",
            "case_first_mismatch": c.get("first_mismatch"),
            "case_first_non_numeric_mismatch": c.get(
                "first_non_numeric_mismatch"
            ),
        }

    exact_pass = bool(c.get("pass"))
    case_accepted = bool(c.get("accepted", exact_pass))
    if not case_accepted:
        return {
            "status": "FAIL",
            "pass": exact_pass,
            "matches_baseline": False,
            "numeric_tie": False,
            "accepted": False,
            "error": "",
            "first_mismatch": c.get("first_mismatch"),
            "first_non_numeric_mismatch": c.get(
                "first_non_numeric_mismatch"
            ),
        }

    comparison = _signature_comparison(bcase, c)
    matches_baseline = comparison["matches_baseline"]
    numeric_tie = bool(comparison["numeric_tie"] or not exact_pass)
    if matches_baseline:
        status = "PASS" if exact_pass else "NUMERIC_TIE"
    elif comparison["numeric_tie"]:
        status = "NUMERIC_TIE"
    else:
        status = "MISMATCH"
    accepted = status in ("PASS", "NUMERIC_TIE")
    return {
        "status": status,
        "pass": exact_pass,
        "matches_baseline": matches_baseline,
        "numeric_tie": numeric_tie,
        "accepted": accepted,
        "error": "",
        "first_mismatch": (
            comparison.get("first_mismatch") or c.get("first_mismatch")
        ),
        "case_first_mismatch": c.get("first_mismatch"),
        "case_first_non_numeric_mismatch": c.get(
            "first_non_numeric_mismatch"
        ),
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
    parser.add_argument("--state-atol", type=float, default=5e-4,
                        help="W11/W12 full GDN and KV elementwise absolute tolerance")
    parser.add_argument("--state-rtol", type=float, default=5e-3,
                        help="W11/W12 full GDN and KV elementwise relative tolerance")
    parser.add_argument("--regression-decode-steps", type=int, default=8,
                        help="W11/W12 decode forwards after prefill (>=3)")
    parser.add_argument("--stability-scale", choices=("smoke", "default", "full"),
                        default="default",
                        help="W13 workload preset (case must be selected explicitly)")
    parser.add_argument("--stability-decode-tokens", type=int, default=128,
                        help="W13 decode tokens unless the preset overrides them")
    args = parser.parse_args()
    args.cases = set(c for c in args.cases.split(",") if c)
    if (not math.isfinite(args.state_atol) or not math.isfinite(args.state_rtol)
            or args.state_atol < 0 or args.state_rtol < 0 or args.regression_decode_steps < 3):
        parser.error("state tolerances must be nonnegative and regression-decode-steps >=3")
    if args.stability_decode_tokens < 4:
        parser.error("stability-decode-tokens must be >= 4")

    if args.mode == "parent":
        run_parent(args)
    else:
        run_child(args)


if __name__ == "__main__":
    main()
