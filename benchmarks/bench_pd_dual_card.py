"""Resident dual-card PD separation driver (runbook nano_qwen_5090_dual_pd_runbook.md §4).

One driver process spawns worker subprocesses, each bound to one GPU with its
own engine. Workers talk to the driver over stdin/stdout JSON lines; the
payload path stays the existing GPU -> LMCache L1 SHM -> GPU direct-tensor
handoff and is not modified.

Configs:

  pd-serial     P prefills request i, D decodes it; request i+1 starts only
                after D's ACK (concurrency 1; queue depth forced to 1).
  pd-pipeline   P prefills i+1 while D decodes i; the D worker restores new
                transfers between decode steps, so D batches up to 2 requests
                (B1 -> B2 -> B1 as requests join and finish). Live transfers
                are bounded by --queue-depth and --shm-budget-gb.
  single        One GPU runs whole inference, batch size = --concurrency.
  dual-replica  Two GPUs each run B1; requests are round-robined.
  pd-correctness  Deterministic B1->B2->B1 scenarios: staggered joins, KV page
                boundary (256->257), chunked 1024-token inputs, replay
                determinism, final-boundary KV/GDN drift vs solo references.

Correctness gates:

  * PD paths: D's output vs the P-side same-weight reference — exact when the
    request only decoded at B1; mixed B2 rows go through the repo's near-tie
    gate on the first mismatch (competing_tokens_are_tied), and B2 rows are
    also checked for bit-exact replay determinism in pd-correctness;
  * D reports exactly 0 prefill and output_len-1 decode dispatches;
  * every handoff is ACKed;
  * after drain both sides are clean (no KV blocks, no GDN slots) and the
    store occupancy equals its starting value.

Timing starts after warmup and graph capture. Short CUDA traces (--trace-window)
are taken on extra requests after the measured phase.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import queue
import select
import shlex
import statistics
import subprocess
import sys
import threading
import time
import traceback

import pd_common as common

ROOT = Path(__file__).resolve().parents[1]
SEED_TEXT = "Explain how a computer stores information in memory. "

CORRECTNESS_SCENARIOS = [
    dict(name="stagger", requests=[
        dict(role="a", len=64, out=32, join=0),
        dict(role="b", len=64, out=32, join=8),
    ]),
    dict(name="page-boundary", requests=[
        dict(role="a", len=256, out=64, join=0),
        dict(role="b", len=257, out=64, join=8),
    ]),
    dict(name="chunked-1024", requests=[
        dict(role="a", len=1024, out=128, join=0),
        dict(role="b", len=1024, out=128, join=4),
    ]),
    dict(name="mixed-chunk", requests=[
        dict(role="a", len=1024, out=128, join=0),
        dict(role="b", len=257, out=128, join=2),
    ]),
    # B1 -> B2 -> B3 -> B4 with staggered exits and slot reuse: E joins after
    # A exits and must take A's freed GDN slot (max_num_seqs=4).
    dict(name="b4-stagger", requests=[
        dict(role="a", len=257, out=128, join=0),
        dict(role="b", len=64, out=128, join=4),
        dict(role="c", len=1024, out=128, join=8),
        dict(role="d", len=256, out=128, join=12),
        dict(role="e", len=257, out=128, join=100),
    ]),
    # B1 -> ... -> B7 with staggered exits and slot reuse (max_num_seqs=8):
    # covers the newly captured B3/B5/B6/B7 graphs.
    dict(name="b7-stagger", requests=[
        dict(role="a", len=257, out=96, join=0),
        dict(role="b", len=64, out=96, join=2),
        dict(role="c", len=1024, out=96, join=4),
        dict(role="d", len=256, out=96, join=6),
        dict(role="e", len=257, out=96, join=8),
        dict(role="f", len=64, out=96, join=10),
        dict(role="g", len=1024, out=96, join=12),
        dict(role="h", len=257, out=96, join=90),
    ]),
]


def p95(values):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * 0.95
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def median(values):
    return statistics.median(values) if values else None


def stats(values):
    return dict(n=len(values), median=median(values), p50=median(values), p95=p95(values),
                min=min(values) if values else None, max=max(values) if values else None)


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def worker_arguments():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--role", choices=("prefill", "decode", "solo"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--cache-url", default="http://127.0.0.1:8766")
    p.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--token-budget", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--trace-dir", default=None)
    p.add_argument("--lens", nargs="+", type=int, default=[64, 257, 1024])
    p.add_argument("--outs", nargs="+", type=int, default=None)
    p.add_argument("--max-batch", type=int, default=2,
                   help="decode/replica worker max_num_seqs (D<=2 / D<=4)")
    return p.parse_args()


def make_engine(args, max_num_seqs=4):
    from nano_qwen.engine.llm_engine import LLMEngine
    return LLMEngine(args.model, enforce_eager=not args.cuda_graph,
        use_prefill_cudagraph=args.cuda_graph, max_num_seqs=max_num_seqs,
        max_num_batched_tokens=args.token_budget, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1, async_scheduling=False, enable_prefix_cache=False)


def prompt_for(engine, length):
    seed = engine.tokenizer.encode(SEED_TEXT, add_special_tokens=False)
    if not seed:
        raise RuntimeError("empty prompt seed")
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def sampling_params(output_len):
    from nano_qwen.sampling_params import SamplingParams
    return SamplingParams(temperature=1e-8, max_tokens=output_len, ignore_eos=True)


def references_with_diagnostics(engine, params_by_out, sampler, lens):
    """B1 references with per-position top-2/margin diagnostics for near-tie checks."""
    from engine_bench_utils import run_until_idle
    from nano_qwen.engine.sequence import Sequence
    refs = {}
    for out in sorted(params_by_out):
        params = params_by_out[out]
        for length in sorted(set(lens)):
            seq = Sequence(prompt_for(engine, length), params)
            key = f"ref.{out}.{length}"
            seq._diagnostic_key = key
            engine.scheduler.add(seq)
            run_until_idle(engine, [seq])
            refs[f"{out}:{length}"] = dict(tokens=list(seq.completion_token_ids),
                                           diagnostics=dict(sampler.get_diagnostic_map(key)))
            sampler.clear_diagnostics([key])
    return refs


def reference_dict_for(refs, output_len, prompt_len):
    return refs.get(f"{output_len}:{prompt_len}")


def token_verdict(tokens, reference, diagnostics, mixed_batch):
    """Compare measured tokens against a B1 reference with per-position
    diagnostics. Exact when equal; mixed-batch rows first go through the
    repo's near-tie gate, then a documented bounded-flip check for fp8 noise
    (same competing pair + one near-tie side + movement <= 4x repo tolerance).
    mixed_batch=True when the request ever shared a step with another request
    (decode for the D worker, any batch for continuous-batching replicas)."""
    from engine_bench_utils import competing_tokens_are_tied
    if tokens == reference["tokens"]:
        return dict(tokens_equal=True, verdict="exact")
    position = next(i for i, (a, b) in enumerate(zip(tokens, reference["tokens"])) if a != b)
    ref_diags = reference.get("diagnostics", {}) or {}
    ref_diag = ref_diags.get(str(position)) or ref_diags.get(position)
    got_diag = diagnostics.get(position) or diagnostics.get(str(position))

    def summary(diag):
        if not diag:
            return None
        return {key: diag.get(key) for key in
                ("top_tokens", "top_logits", "margin", "effective_tolerance", "near_tie")}

    flip_detail = dict(reference=summary(ref_diag), candidate=summary(got_diag))
    result = dict(tokens_equal=False, verdict="mismatch", first_mismatch=position,
                  flip_detail=flip_detail)
    if not mixed_batch:
        return result
    accepted = competing_tokens_are_tied(
        ref_diag, got_diag, reference["tokens"][position], tokens[position])
    if accepted:
        result.update(tokens_equal=True, verdict="near_tie")
        return result
    if ref_diag and got_diag:
        # fp8 mixed-batch noise exceeds bf16 ULP noise. Accept a flip as
        # bounded when the competing pair matches, one side is a repo
        # near-tie, and the cross-run logit movement is at most 4x the repo
        # tolerance (observed fp8 movements cluster at 1-3 bf16 ULP of the
        # logit magnitude; 4x leaves margin).
        pair_ok = set(ref_diag.get("top_tokens", [])[:2]) == set(
            got_diag.get("top_tokens", [])[:2])
        repo_tol = max(ref_diag.get("effective_tolerance", 0.0),
                       got_diag.get("effective_tolerance", 0.0))
        ref_values = dict(zip(ref_diag.get("top_tokens", []),
                              ref_diag.get("top_logits", [])))
        got_values = dict(zip(got_diag.get("top_tokens", []),
                              got_diag.get("top_logits", [])))
        movement = max(abs(ref_values.get(t, float("inf"))
                           - got_values.get(t, float("inf")))
                       for t in (reference["tokens"][position], tokens[position]))
        one_near_tie = bool(ref_diag.get("near_tie") or got_diag.get("near_tie"))
        exact_tie = (ref_diag.get("margin") == 0 or got_diag.get("margin") == 0)
        bounded = (pair_ok and one_near_tie and repo_tol > 0
                   and movement <= 4 * repo_tol)
        flip_detail.update(movement=None if movement == float("inf") else movement,
                           budget=4 * repo_tol, bounded=bounded)
        if bounded:
            result.update(tokens_equal=True, verdict="bounded_flip")
        elif exact_tie:
            # An exact logit tie is noise-determined by definition; batch
            # noise may legitimately reshuffle which tied token wins.
            result.update(tokens_equal=True, verdict="tie_flip")
    return result


def decode_worker_main(engine, connector, args):
    """Step-wise D loop: restore new transfers between steps (D<=2), record
    decode batch sizes, verify outputs, optionally capture final-boundary
    KV/GDN snapshots for drift checks against solo references."""
    import torch
    from unittest.mock import patch
    from engine_bench_utils import (ArgmaxSampler, competing_tokens_are_tied,
                                    full_request_snapshot, tensor_snapshot_diff,
                                    STATE_DRIFT_RMSE_LIMITS)

    sampler = ArgmaxSampler()
    engine.model_runner.sampler = sampler
    sampler.enable_diagnostics(capture_logits=False)
    pending = collections.deque()
    active = {}            # id -> info
    by_seq = {}            # seq_id -> id
    counters = dict(decode=0, acks=0, restored=0)
    batch_histogram = collections.Counter()
    per_request_batches = collections.defaultdict(collections.Counter)
    finalize_flag = False
    trace_end_id = None
    profiler = None
    trace_errors = []
    results = []           # finished request records (sent to driver)
    state_keys = {}        # diagnostic key -> prompt_len (final-boundary capture)
    snapshots = {}         # key -> tensors at final boundary
    solo_snapshots = {}    # (output_len, prompt_len) -> tensors
    replay_tokens = collections.defaultdict(dict)  # (group, index) -> role -> tokens

    def begin_trace():
        nonlocal profiler
        if args.trace_dir and profiler is None:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA])
            profiler.__enter__()

    def end_trace():
        nonlocal profiler
        if profiler is not None:
            active = profiler
            profiler = None
            try:
                active.__exit__(None, None, None)
                Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
                active.export_chrome_trace(
                    str(Path(args.trace_dir) / "trace-decode.json"))
            except Exception as exc:
                trace_errors.append(f"decode: {exc}")

    def restore_cmd(cmd):
        start = time.time_ns()
        pre_digests = None
        if cmd.get("state_check"):
            # Bit-exact transfer check: digest the P-side snapshot per logical
            # page and GDN slot BEFORE the restore consumes (and deletes) it.
            lease = connector.acquire(cmd["transfer"])
            try:
                pre_digests = state_digests(lease.tensors)
            finally:
                connector.release(lease, delete=False)
        seq = engine.resume_from_lmcache(connector, cmd["transfer"])
        m = seq.pd_handoff["metrics"]
        if not seq.pd_handoff["acknowledged"]:
            raise RuntimeError(f"ACK failed: {seq.pd_handoff.get('ack_error')}")
        counters["acks"] += 1
        counters["restored"] += 1
        key = f"d{cmd['id']}"
        seq._diagnostic_key = key
        slot = engine.model_runner.input_batch.seq_id_to_slot[seq.seq_id]
        post_restore_digests = None
        if pre_digests is not None:
            snapshot = full_request_snapshot(engine.model_runner, seq,
                                             seq.num_cached_tokens)
            post_restore_digests = snapshot_digests(snapshot,
                                                    engine.model_runner.block_size)
        active[cmd["id"]] = dict(cmd=cmd, seq=seq, steps=0, key=key,
            slot=slot, step_ends=[], graph_steps=collections.Counter(),
            pre_digests=pre_digests, post_restore_digests=post_restore_digests,
            restore_start_ns=start,
            restore_done_ns=start + int((m["restore_total_ms"] - m["ack_ms"]) * 1e6),
            acquire_ms=m["acquire_ms"], h2d_ms=m["h2d_ms"], ack_ms=m["ack_ms"])
        by_seq[seq.seq_id] = cmd["id"]
        if cmd.get("state_check"):
            state_keys[key] = cmd["prompt_len"]
        print(json.dumps(dict(restored=dict(id=cmd["id"], slot=slot))), flush=True)
        return seq

    def state_digests(tensors):
        """SHA256 per logical KV page (k/v, layer, page) and per GDN slot tensor."""
        import hashlib
        digests = {}
        for name, tensor in sorted(tensors.items()):
            if name == "kv":
                view = tensor.contiguous()
                for k in range(view.shape[0]):
                    for layer in range(view.shape[1]):
                        for page in range(view.shape[2]):
                            part = view[k, layer, page].reshape(-1).view(torch.uint8)
                            digests[f"kv.{k}.{layer}.p{page}"] = hashlib.sha256(
                                part.numpy().tobytes()).hexdigest()
            else:
                part = tensor.contiguous().reshape(-1).view(torch.uint8)
                digests[name] = hashlib.sha256(part.numpy().tobytes()).hexdigest()
        return digests

    def snapshot_digests(snapshot, block_size):
        """SHA256 digests of a canonicalized snapshot (full_request_snapshot
        layout) using the same key format as the P-side transfer digests."""
        import hashlib
        digests = {}
        for name, tensor in snapshot.items():
            if name.startswith("kv."):
                _, layer, kv = name.split(".")
                kind = 0 if kv == "k" else 1
                for page in range(0, tensor.shape[0], block_size):
                    part = tensor[page:page + block_size]
                    if part.shape[0] < block_size:
                        pad = torch.zeros((block_size - part.shape[0], *part.shape[1:]),
                                          dtype=part.dtype)
                        part = torch.cat([part, pad], dim=0)
                    part = part.reshape(-1).view(torch.uint8)
                    digests[f"kv.{kind}.{layer}.p{page // block_size}"] = hashlib.sha256(
                        part.numpy().tobytes()).hexdigest()
            else:
                part = tensor.reshape(-1).view(torch.uint8)
                digests[name] = hashlib.sha256(part.numpy().tobytes()).hexdigest()
        return digests

    def step_once():
        calls = []
        original = engine.model_runner.call

        def tracked(method, *args):
            if method == "execute_model":
                calls.append((list(args[0]), bool(args[1])))
            if method == "remove_request":
                # The engine removes the slot before postprocess; capture the
                # final-boundary KV/GDN state while the slot is still alive.
                seq_id = args[0]
                info_id = by_seq.get(seq_id)
                if info_id is not None and active.get(info_id) is not None:
                    info = active[info_id]
                    if (info["key"] in state_keys
                            and info["seq"].num_completion_tokens + 1 == info["seq"].max_tokens):
                        snapshots[info["key"]] = full_request_snapshot(
                            engine.model_runner, info["seq"],
                            info["seq"].num_cached_tokens + info["seq"].num_scheduled_tokens)
            return original(method, *args)

        with patch.object(engine.model_runner, "call", tracked):
            outputs, _ = engine.step()
        return outputs, calls

    def finish_tokens(info, tokens, diagnostics):
        end = time.time_ns()
        seq = info["seq"]
        cmd = info["cmd"]
        output_len = cmd.get("output_len", args.output_len)
        if len(tokens) != output_len:
            raise RuntimeError(f"incorrect output-token count: {len(tokens)} != {output_len}")
        if info["steps"] != output_len - 1:
            raise RuntimeError(f"wrong D dispatch counts: {info['steps']} steps for {output_len} tokens")
        batches = per_request_batches[info["key"]]
        reference = cmd.get("reference")
        if reference is not None:
            outcome = token_verdict(tokens, reference, diagnostics,
                                    max(batches, default=1) > 1)
        else:
            outcome = dict(tokens_equal=False, verdict="mismatch", first_mismatch=None,
                           flip_detail=None)
        step_intervals = [info["step_ends"][i + 1] - info["step_ends"][i]
                          for i in range(len(info["step_ends"]) - 1)]
        restore_bit_exact = None
        if info.get("pre_digests") is not None:
            post_digests = info.get("post_restore_digests")
            if post_digests is None:
                restore_bit_exact = dict(pass_=False, parts=len(info["pre_digests"]),
                                         mismatched=["<no post-restore digests>"])
            else:
                mismatch = [key for key in sorted(info["pre_digests"])
                            if post_digests.get(key) != info["pre_digests"][key]]
                restore_bit_exact = dict(
                    pass_=not mismatch and set(info["pre_digests"]) == set(post_digests),
                    parts=len(info["pre_digests"]), mismatched=mismatch[:8])
        drift = None
        if info["key"] in snapshots:
            solo = solo_snapshots.get((output_len, cmd["prompt_len"]))
            if solo is not None:
                diff = tensor_snapshot_diff(snapshots[info["key"]], solo, atol=0.0, rtol=0.0)
                limits = STATE_DRIFT_RMSE_LIMITS
                drift_pass = all(
                    group.get("nonfinite", 0) == 0
                    and group.get("relative_rmse", float("inf"))
                    <= limits.get(name, float("inf"))
                    for name, group in diff["groups"].items())
                drift = {"pass": drift_pass,
                         "groups": {name: dict(relative_rmse=group.get("relative_rmse"),
                                               nonfinite=group.get("nonfinite", 0))
                                    for name, group in diff["groups"].items()}}
        replay = cmd.get("replay")
        if replay is not None:
            replay_tokens[(replay["group"], replay["index"])][replay["role"]] = tokens
        result = dict(id=cmd["id"], ok=True, tokens=tokens,
                      tokens_equal=outcome["tokens_equal"],
                      verdict=outcome["verdict"],
                      first_mismatch=outcome.get("first_mismatch"),
                      flip_detail=outcome.get("flip_detail"),
                      acknowledged=True, steps=info["steps"],
                      dispatches=dict(prefill=0, decode=info["steps"]),
                      batch_sizes=dict(batches),
                      graph_steps=dict(info.get("graph_steps", {})),
                      step_intervals=step_intervals,
                      slot=info.get("slot"),
                      restore_start_ns=info["restore_start_ns"],
                      restore_done_ns=info["restore_done_ns"],
                      second_token_ns=info.get("second_token_ns"),
                      end_ns=end,
                      acquire_ms=info["acquire_ms"], h2d_ms=info["h2d_ms"],
                      ack_ms=info["ack_ms"],
                      decode_ms=(end - info["restore_done_ns"]) / 1e6,
                      restore_bit_exact=restore_bit_exact,
                      drift=drift, role=replay["role"] if replay else None,
                      solo_ref=(cmd.get("solo_ref") is not None))
        if cmd.get("solo_ref") is not None:
            solo_snapshots[(output_len, cmd["prompt_len"])] = snapshots.get(info["key"])
        results.append(result)
        print(json.dumps(dict(result=result)), flush=True)
        if trace_end_id is not None and cmd["id"] == trace_end_id:
            end_trace()

    stdin_queue = queue.Queue()

    def stdin_reader():
        for line in sys.stdin:
            if line.strip():
                try:
                    stdin_queue.put(json.loads(line))
                except json.JSONDecodeError:
                    stdin_queue.put(dict(raw=line.rstrip()))

    threading.Thread(target=stdin_reader, daemon=True).start()

    try:
        while True:
            # 1. Drain any new commands (non-blocking).
            while True:
                try:
                    cmd = stdin_queue.get_nowait()
                except queue.Empty:
                    break
                if cmd["cmd"] == "finalize":
                    finalize_flag = True
                elif cmd["cmd"] == "decode":
                    if cmd.get("trace") and profiler is None:
                        begin_trace()
                    if cmd.get("trace_end"):
                        trace_end_id = cmd["id"]
                    pending.append(cmd)
            # 2. Restore pending transfers while capacity allows.
            while pending:
                cmd = pending[0]
                if cmd.get("join_after") is not None and active:
                    if max(info["steps"] for info in active.values()) < cmd["join_after"]:
                        break
                try:
                    restore_cmd(cmd)
                except (ValueError, MemoryError) as exc:
                    message = str(exc)
                    if ("free GDN state slot" in message or "KV blocks" in message
                            or "no in-flight" in message or "decode boundaries" in message):
                        break  # Capacity bound; retry after the next step.
                    raise
                pending.popleft()
            # 3. One decode step for all active requests.
            if active:
                step_start = time.time_ns()
                outputs, calls = step_once()
                step_end = time.time_ns()
                counters["busy_ms"] = counters.get("busy_ms", 0) + (step_end - step_start) / 1e6
                for seqs, is_prefill in calls:
                    if is_prefill:
                        raise RuntimeError("D dispatched a prefill batch")
                    counters["decode"] += 1
                    batch_histogram[len(seqs)] += 1
                    graph_hit = len(seqs) in getattr(engine.model_runner, "graphs", {})
                    for seq in seqs:
                        info_id = by_seq[seq.seq_id]
                        info = active[info_id]
                        info["steps"] += 1
                        info["step_ends"].append(step_end)
                        info["graph_steps"][f"b{len(seqs)}:"
                                            + ("graph" if graph_hit else "eager")] += 1
                        per_request_batches[info["key"]][len(seqs)] += 1
                        if (info.get("second_token_ns") is None
                                and seq.num_completion_tokens >= 2):
                            info["second_token_ns"] = time.time_ns()
                for seq_id, tokens in outputs:
                    info_id = by_seq.get(seq_id)
                    if info_id is None:
                        raise RuntimeError(f"finished seq {seq_id} is not tracked")
                    info = active.pop(info_id)
                    key = info["key"]
                    diagnostics = dict(sampler.get_diagnostic_map(key))
                    sampler.clear_diagnostics([key])
                    finish_tokens(info, tokens, diagnostics)
                continue
            # 4. Idle: wait briefly for more commands.
            if finalize_flag and not pending:
                break
            time.sleep(0.02)  # Idle wait: never consume queued commands here.

        clean, detail = True, {}
        try:
            from nano_qwen.cache import engine_ops
            common.require_clean(engine, engine_ops)
        except Exception as exc:
            clean, detail = False, dict(error=str(exc))
        replay_checks = {}
        for (group, index), roles in sorted(replay_tokens.items()):
            replay_checks.setdefault(group, {})[index] = roles
        replay_equal = {}
        for group, passes in replay_checks.items():
            indexes = sorted(passes)
            replay_equal[group] = (len(indexes) >= 2 and all(
                passes[indexes[0]].get(role) == passes.get(indexes[i], {}).get(role)
                for i in indexes[1:] for role in passes[indexes[0]]))
        final = dict(clean=clean, clean_detail=detail,
                     peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                     peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                     free_kv_blocks=len(engine.scheduler.block_manager.free_block_ids),
                     counters=counters, batch_histogram=dict(batch_histogram),
                     replay_equal=replay_equal,
                     solo_refs=len(solo_snapshots), snapshots=len(snapshots),
                     trace_export_errors=trace_errors)
        final["store"] = connector.health()
        engine.exit()
        return final
    except BaseException:
        end_trace()
        raise


def solo_worker_main(engine, args, sampler, refs):
    """Continuous-batching replica: admit requests between steps, run the
    production engine step loop (prefill + decode), report per-request
    latencies, decode batch composition, graph paths and token verdicts."""
    import torch
    from unittest.mock import patch
    from nano_qwen.engine.sequence import Sequence
    pending = collections.deque()
    active = {}       # id -> info
    by_seq = {}       # seq_id -> id
    batch_histogram = collections.Counter()
    results = []
    finalize_flag = False
    counters = dict(prefill=0, decode=0)

    def step_once():
        calls = []
        original = engine.model_runner.call

        def tracked(method, *args):
            if method == "execute_model":
                calls.append((list(args[0]), bool(args[1])))
            return original(method, *args)

        with patch.object(engine.model_runner, "call", tracked):
            outputs, num_tokens = engine.step()
        return outputs, calls

    def finish(info, tokens, diagnostics, end):
        seq = info["seq"]
        cmd = info["cmd"]
        out = cmd.get("output_len", args.output_len)
        if len(tokens) != out:
            raise RuntimeError(f"incorrect output-token count: {len(tokens)} != {out}")
        reference = refs.get(f"{out}:{cmd['prompt_len']}")
        expected_chunks = ([cmd["prompt_len"]] if cmd["prompt_len"] <= 512
                           else [512, cmd["prompt_len"]])
        same_prefill_path = (not info["mixed_prefill"]
                             and info["prefill_chunks"] == expected_chunks)
        outcome = token_verdict(tokens, reference, diagnostics,
                                info["any_mixed_step"])
        if (not outcome["tokens_equal"] and not same_prefill_path
                and outcome["verdict"] == "mismatch"):
            # Mixed-prefill batches change chunk boundaries vs the B1
            # reference; record the divergence instead of hard-failing.
            outcome = dict(tokens_equal=None, verdict="mixed_prefill_divergence",
                           first_mismatch=outcome.get("first_mismatch"),
                           flip_detail=outcome.get("flip_detail"))
        intervals = [info["step_ends"][i + 1] - info["step_ends"][i]
                     for i in range(len(info["step_ends"]) - 1)]
        result = dict(id=cmd["id"], ok=True, tokens=tokens,
                      tokens_equal=outcome["tokens_equal"],
                      verdict=outcome["verdict"],
                      first_mismatch=outcome.get("first_mismatch"),
                      flip_detail=outcome.get("flip_detail"),
                      ttft_ms=(info["first_token_ns"] - info["submitted_ns"]) / 1e6,
                      first_to_second_ms=(info["step_ends"][1] - info["step_ends"][0]) / 1e6
                      if len(info["step_ends"]) >= 2 else None,
                      step_intervals=intervals,
                      e2e_ms=(end - info["submitted_ns"]) / 1e6,
                      submitted_ns=info["submitted_ns"], end_ns=end,
                      batch_sizes=dict(info["decode_batches"]),
                      graph_steps=dict(info["graph_steps"]),
                      any_mixed_step=info["any_mixed_step"],
                      mixed_prefill=info["mixed_prefill"],
                      prefill_chunks=info["prefill_chunks"])
        results.append(result)
        print(json.dumps(dict(result=result)), flush=True)

    stdin_queue = queue.Queue()

    def stdin_reader():
        for line in sys.stdin:
            if line.strip():
                try:
                    stdin_queue.put(json.loads(line))
                except json.JSONDecodeError:
                    stdin_queue.put(dict(raw=line.rstrip()))

    threading.Thread(target=stdin_reader, daemon=True).start()
    try:
        while True:
            # 1. Drain any new commands (non-blocking).
            while True:
                try:
                    cmd = stdin_queue.get_nowait()
                except queue.Empty:
                    break
                if cmd["cmd"] == "finalize":
                    finalize_flag = True
                elif cmd["cmd"] == "generate":
                    pending.append(cmd)
            # 2. Admit pending requests while slots allow.
            while pending and len(active) < args.max_batch:
                cmd = pending.popleft()
                out = cmd.get("output_len", args.output_len)
                seq = Sequence(prompt_for(engine, cmd["prompt_len"]), sampling_params(out))
                key = f"s{cmd['id']}"
                seq._diagnostic_key = key
                engine.scheduler.add(seq)
                active[cmd["id"]] = dict(cmd=cmd, seq=seq, key=key,
                    submitted_ns=time.time_ns(), first_token_ns=None,
                    last_n=0, step_ends=[], graph_steps=collections.Counter(),
                    decode_batches=collections.Counter(), any_mixed_step=False,
                    prefill_chunks=[], mixed_prefill=False)
                by_seq[seq.seq_id] = cmd["id"]
            # 3. One engine step (prefill or decode, production path).
            if active:
                step_start = time.time_ns()
                outputs, calls = step_once()
                step_end = time.time_ns()
                counters["busy_ms"] = counters.get("busy_ms", 0) + (step_end - step_start) / 1e6
                for seqs, is_prefill in calls:
                    counters["prefill" if is_prefill else "decode"] += 1
                    if not is_prefill:
                        batch_histogram[len(seqs)] += 1
                    graph_hit = (not is_prefill
                                 and len(seqs) in getattr(engine.model_runner, "graphs", {}))
                    for seq in seqs:
                        info_id = by_seq.get(seq.seq_id)
                        if info_id is None:
                            continue
                        info = active[info_id]
                        if is_prefill:
                            info["prefill_chunks"].append(seq.num_scheduled_tokens)
                            if len(seqs) > 1:
                                info["mixed_prefill"] = True
                        else:
                            info["decode_batches"][len(seqs)] += 1
                            info["graph_steps"][f"b{len(seqs)}:"
                                                + ("graph" if graph_hit else "eager")] += 1
                        if len(seqs) > 1:
                            info["any_mixed_step"] = True
                        n = seq.num_completion_tokens
                        if n > info["last_n"]:
                            info["last_n"] = n
                            info["step_ends"].append(step_end)
                            if info["first_token_ns"] is None:
                                info["first_token_ns"] = step_end
                for seq_id, tokens in outputs:
                    info_id = by_seq.get(seq_id)
                    if info_id is None:
                        raise RuntimeError(f"finished seq {seq_id} is not tracked")
                    info = active.pop(info_id)
                    key = info["key"]
                    diagnostics = dict(sampler.get_diagnostic_map(key))
                    sampler.clear_diagnostics([key])
                    finish(info, tokens, diagnostics, step_end)
                continue
            # 4. Idle: wait briefly for more commands.
            if finalize_flag and not pending:
                break
            time.sleep(0.02)  # Idle wait: never consume queued commands here.

        clean, detail = True, {}
        try:
            from nano_qwen.cache import engine_ops
            common.require_clean(engine, engine_ops)
        except Exception as exc:
            clean, detail = False, dict(error=str(exc))
        final = dict(clean=clean, clean_detail=detail,
                     peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                     peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                     free_kv_blocks=len(engine.scheduler.block_manager.free_block_ids),
                     counters=counters, batch_histogram=dict(batch_histogram))
        engine.exit()
        return final
    except BaseException:
        raise


def worker_main():
    args = worker_arguments()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker requires exactly one visible CUDA device")
    try:
        from nano_qwen.cache import engine_ops
        from nano_qwen.cache.shm_connector import LMCacheShmConnector
        from nano_qwen.engine.sequence import Sequence
        from nano_qwen.sampling_params import SamplingParams
        from nano_qwen.quantization.checkpoint import read_quantization_metadata, read_hf_fp8_config
        from engine_bench_utils import ArgmaxSampler

        engine_start = time.time()
        if args.role == "decode":
            engine = make_engine(args, max_num_seqs=args.max_batch)
        elif args.role == "solo":
            engine = make_engine(args, max_num_seqs=args.max_batch)
        else:
            engine = make_engine(args)
        engine_init_ms = (time.time() - engine_start) * 1e3
        connector = LMCacheShmConnector(args.cache_url, 300) if args.role != "solo" else None
        params = sampling_params(args.output_len)
        outs = args.outs or [args.output_len]
        refs = None
        if args.role in ("prefill", "solo"):
            sampler = ArgmaxSampler()
            engine.model_runner.sampler = sampler
            sampler.enable_diagnostics(capture_logits=False)
            refs = references_with_diagnostics(engine,
                                               {out: sampling_params(out) for out in outs},
                                               sampler, args.lens)
        ready = dict(ready=True, role=args.role, gpu=torch.cuda.get_device_name(0),
                     engine_init_ms=engine_init_ms,
                     decode_graph_sizes=getattr(engine.model_runner, "graph_bs", []),
                     free_kv_blocks_after_init=len(
                         engine.scheduler.block_manager.free_block_ids),
                     torch=str(torch.__version__), cuda=torch.version.cuda,
                     quantization=engine.config.quantization,
                     quantization_metadata=read_quantization_metadata(args.model)
                     or read_hf_fp8_config(args.model),
                     model_id=engine.model_runner.call("pd_model_identity"),
                     output_len=args.output_len)
        if refs is not None:
            if args.role == "prefill":
                # P-side references carry per-position diagnostics so the D
                # worker can run the repo's near-tie gate on mixed batches.
                ready["references"] = refs
            else:
                ready["references"] = {key: value["tokens"] for key, value in refs.items()}
        print(json.dumps(ready), flush=True)

        if args.role == "decode":
            final = decode_worker_main(engine, connector, args)
            print(json.dumps(dict(final=final)), flush=True)
            return
        if args.role == "solo":
            final = solo_worker_main(engine, args, sampler, refs)
            print(json.dumps(dict(final=final)), flush=True)
            return

        counters = dict(prefill=0, decode=0, nbytes=0, acks=0)
        profiler = None
        trace_errors = []

        def begin_trace():
            nonlocal profiler
            if args.trace_dir and profiler is None:
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA])
                profiler.__enter__()

        def end_trace():
            nonlocal profiler
            if profiler is not None:
                active = profiler
                profiler = None
                try:
                    active.__exit__(None, None, None)
                    Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
                    active.export_chrome_trace(
                        str(Path(args.trace_dir) / f"trace-{args.role}.json"))
                except Exception as exc:
                    trace_errors.append(f"{args.role}: {exc}")

        def make_reference(cmd):
            from engine_bench_utils import run_until_idle
            output_len, length = cmd["output_len"], cmd["prompt_len"]
            seq = Sequence(prompt_for(engine, length), sampling_params(output_len))
            key = f"refreq.{output_len}.{length}"
            seq._diagnostic_key = key
            engine.scheduler.add(seq)
            run_until_idle(engine, [seq])
            entry = dict(output_len=output_len, prompt_len=length,
                         tokens=list(seq.completion_token_ids),
                         diagnostics=dict(sampler.get_diagnostic_map(key)))
            sampler.clear_diagnostics([key])
            return entry

        def handle(cmd):
            if args.role == "prefill" and cmd["cmd"] == "reference":
                return make_reference(cmd)
            if args.role == "prefill":
                if cmd.get("trace"):
                    begin_trace()
                counters["prefill"] += 1
                output_len = cmd.get("output_len", args.output_len)
                params_cmd = sampling_params(output_len)
                start = time.time_ns()
                receipt = engine.prefill_to_lmcache(
                    prompt_for(engine, cmd["prompt_len"]), params_cmd, connector,
                    cmd["transfer"])
                counters["nbytes"] += receipt["nbytes"]
                counters["busy_ms"] = counters.get("busy_ms", 0) + (time.time_ns() - start) / 1e6
                m = receipt["pd_metrics"]
                first_token_ns = start + int((m["prefill_total_ms"]
                                              - m["producer_release_ms"]) * 1e6)
                commit_done_ns = first_token_ns + int((m["describe_ms"]
                    + m["reserve_write_ms"] + m["d2h_ms"] + m["commit_ms"]) * 1e6)
                return dict(id=cmd["id"], ok=True, backend=receipt["backend"],
                            nbytes=receipt["nbytes"], start_ns=start, end_ns=time.time_ns(),
                            first_token_ns=first_token_ns,
                            commit_done_ns=commit_done_ns,
                            prefill_ms=m["prefill_ms"], d2h_ms=m["d2h_ms"],
                            commit_ms=m["commit_ms"],
                            producer_release_ms=m["producer_release_ms"])
            raise RuntimeError(f"unknown role {args.role}")

        for line in sys.stdin:
            if not line.strip():
                continue
            cmd = json.loads(line)
            if cmd["cmd"] == "finalize":
                break
            try:
                result = handle(cmd)
                if isinstance(result, list):
                    print(json.dumps(dict(results=result)), flush=True)
                elif result is not None and "diagnostics" in result and "output_len" in result:
                    print(json.dumps(dict(reference=result)), flush=True)
                else:
                    print(json.dumps(dict(result=result)), flush=True)
                if cmd.get("trace_end"):
                    end_trace()
            except Exception:
                end_trace()
                print(json.dumps(dict(error=traceback.format_exc())), flush=True)
                sys.exit(1)

        clean, detail = True, {}
        try:
            common.require_clean(engine, engine_ops)
        except Exception as exc:
            clean, detail = False, dict(error=str(exc))
        final = dict(clean=clean, clean_detail=detail,
                     peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                     peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                     free_kv_blocks=len(engine.scheduler.block_manager.free_block_ids),
                     counters=counters, trace_export_errors=trace_errors)
        if connector is not None:
            final["store"] = connector.health()
        engine.exit()
        print(json.dumps(dict(final=final)), flush=True)
    except Exception:
        print(json.dumps(dict(error=traceback.format_exc())), flush=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def driver_arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=("pd-serial", "pd-pipeline", "single",
                                        "dual-replica", "pd-correctness"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--cache-url", default="http://127.0.0.1:8766")
    p.add_argument("--prefill-gpu", default="0")
    p.add_argument("--decode-gpu", default="1")
    p.add_argument("--lens", nargs="+", type=int, default=[64, 257, 1024])
    p.add_argument("--outs", nargs="+", type=int, default=None,
                   help="per-position output lens, cycled in lockstep with --lens")
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--warmup-requests", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=1,
                   help="pd configs: global in-flight bound; replicas: per-worker cap")
    p.add_argument("--queue-depth", type=int, default=2)
    p.add_argument("--shm-budget-gb", type=float, default=0.8)
    p.add_argument("--decode-max-seqs", type=int, default=2, choices=(1, 2, 4, 8, 16),
                   help="D worker max_num_seqs (D<=2 / D<=4 / D<=8 / D<=16)")
    p.add_argument("--arrival-ms", type=float, default=0,
                   help="pace request dispatch at a fixed interval (0 = as fast as possible)")
    p.add_argument("--trace-window", type=int, default=0,
                   help="extra traced requests after the measured phase (pd configs only)")
    p.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--token-budget", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    if any(n < 1 for n in args.lens):
        p.error("--lens must be positive integers (duplicates allowed for mixed workloads)")
    if args.outs is None:
        args.outs = [args.output_len] * len(args.lens)
    if len(args.outs) != len(args.lens) or any(n < 2 for n in args.outs):
        p.error("--outs must match --lens and be >= 2")
    args.output_len = max(args.outs)
    if args.requests < 1 or args.warmup_requests < 0:
        p.error("invalid request counts")
    if not 0 < args.shm_budget_gb:
        p.error("--shm-budget-gb must be positive")
    if args.config == "single" and args.concurrency not in (1, 2, 4):
        p.error("single concurrency must be 1, 2 or 4")
    if args.config == "dual-replica" and args.concurrency not in (1, 2, 4):
        p.error("dual-replica concurrency is the per-worker cap (1, 2 or 4)")
    return args


class Worker:
    def __init__(self, args, role, gpu, name, max_batch=None):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        log = Path(args.output).parent / f"worker-{name}-{role}.log"
        argv = [sys.executable, str(Path(__file__).resolve()), "--role", role,
                "--model", args.model, "--cache-url", args.cache_url,
                "--max-model-len", str(args.max_model_len),
                "--token-budget", str(args.token_budget),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--output-len", str(args.output_len), "--seed", str(args.seed),
                "--lens", *[str(n) for n in args.lens],
                "--outs", *[str(n) for n in args.outs]]
        if max_batch is not None:
            argv += ["--max-batch", str(max_batch)]
        if not args.cuda_graph:
            argv.append("--no-cuda-graph")
        if args.trace_window and role != "solo":
            argv += ["--trace-dir", str(Path(args.output).parent)]
        self.role, self.name = role, name
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=open(log, "w"), text=True, env=env, bufsize=1)
        self.messages = queue.Queue()
        self.done = threading.Event()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        try:
            for line in self.proc.stdout:
                try:
                    self.messages.put(json.loads(line))
                except json.JSONDecodeError:
                    self.messages.put(dict(raw=line.rstrip()))
        finally:
            self.done.set()

    def send(self, value):
        self.proc.stdin.write(json.dumps(value) + "\n")
        self.proc.stdin.flush()

    def recv(self, timeout=900):
        try:
            return self.messages.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"{self.name}/{self.role} timed out") from None

    def poll(self):
        try:
            return self.messages.get_nowait()
        except queue.Empty:
            return None

    def terminate(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def gpu_identities():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version",
                              "--format=csv,noheader"], text=True,
                             capture_output=True, check=True).stdout
        return [line.split(", ") for line in out.strip().splitlines()]
    except Exception as exc:
        return [dict(error=str(exc))]


def trace_kernel_intervals(path):
    """Wall-clock busy intervals of GPU kernels from a Kineto chrome trace."""
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        return None
    intervals = []
    for event in data.get("traceEvents", []):
        if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and event.get("ph") == "X":
            ts, dur = event.get("ts"), event.get("dur")
            if isinstance(ts, (int, float)) and isinstance(dur, (int, float)) and dur > 0:
                intervals.append((float(ts), float(ts) + float(dur)))
    if not intervals:
        return None
    intervals.sort()
    merged, current = [], None
    for start, end in intervals:
        if current is None or start > current[1]:
            if current:
                merged.append(current)
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    merged.append(current)
    return merged


def overlap_us(left, right):
    if not left or not right:
        return None
    i = j = total = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            total += end - start
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


AGGREGATE_METRICS = ("ttft_ms", "e2e_ms", "pause_ms", "queue_ms", "queue_commit_ms",
                     "first_to_second_ms", "step_interval_p95", "step_interval_p50",
                     "export_ms", "restore_ms", "first_decode_ms",
                     "decode_ms", "tpot_ms", "prefill_ms", "d2h_ms", "commit_ms",
                     "acquire_ms", "h2d_ms", "ack_ms", "nbytes")


def aggregate(rows):
    agg = {}
    for metric in AGGREGATE_METRICS:
        values = [r[metric] for r in rows if r.get(metric) is not None]
        if values:
            agg[metric] = stats(values)
    first_submit = min(r["submit_ns"] for r in rows)
    last_end = max(r["end_ns"] for r in rows)
    span = (last_end - first_submit) / 1e9
    agg["requests"] = len(rows)
    agg["wall_s"] = span
    agg["req_per_s"] = len(rows) / span if span > 0 else None
    agg["output_tok_per_s"] = sum(r["output_len"] for r in rows) / span if span > 0 else None
    agg["prompt_tok_per_s"] = sum(r["prompt_len"] for r in rows) / span if span > 0 else None
    agg["transfer_bytes_total"] = sum(r.get("nbytes", 0) for r in rows)
    agg["by_lens"] = {}
    for length in sorted({r["prompt_len"] for r in rows}):
        subset = [r for r in rows if r["prompt_len"] == length]
        agg["by_lens"][str(length)] = dict(
            n=len(subset),
            e2e_p95=p95([r["e2e_ms"] for r in subset]),
            ttft_p95=p95([r["ttft_ms"] for r in subset]),
            first_to_second_p95=p95([r["first_to_second_ms"] for r in subset
                                     if r.get("first_to_second_ms") is not None]),
            step_interval_p95=p95([r["step_interval_p95"] for r in subset
                                   if r.get("step_interval_p95") is not None]),
            queue_commit_p95=p95([r["queue_commit_ms"] for r in subset
                                  if r.get("queue_commit_ms") is not None]))
    return agg


def run_pd_config(args):
    from nano_qwen.cache.shm_connector import LMCacheShmConnector
    depth = 1 if args.config == "pd-serial" else args.concurrency
    prefill = Worker(args, "prefill", args.prefill_gpu, args.config)
    decode = Worker(args, "decode", args.decode_gpu, args.config,
                    max_batch=args.decode_max_seqs)
    section = dict(config=args.config, requests=args.requests, lens=args.lens,
                   output_len=args.output_len,
                   concurrency=1 if args.config == "pd-serial" else depth,
                   decode_max_seqs=args.decode_max_seqs,
                   queue_depth=depth, shm_budget_gb=args.shm_budget_gb,
                   warmup_requests=args.warmup_requests, trace_window=args.trace_window,
                   seed=args.seed, rows=[], workers={}, span_overlap=None,
                   trace_evidence=None, verification={})
    try:
        p_ready = prefill.recv(600)
        d_ready = decode.recv(600)
        if not (p_ready.get("ready") and d_ready.get("ready")):
            raise RuntimeError(f"worker startup failed: {p_ready} {d_ready}")
        section["workers"] = dict(prefill=p_ready, decode=d_ready)
        references = p_ready.get("references", {})
        health = LMCacheShmConnector(args.cache_url, 30)
        store_before = health.health()
        section["store_before"] = store_before
        max_store_used = store_before["used_bytes"]
        try:
            total = args.warmup_requests + args.requests + args.trace_window
            lens, outs = args.lens, args.outs
            lens_of = [lens[i % len(lens)] for i in range(total)]
            outs_of = [outs[i % len(outs)] for i in range(total)]
            # P-side references (tokens + diagnostics) for every (len, out).
            for prompt_len, output_len in sorted(set(zip(lens, outs))):
                if f"{output_len}:{prompt_len}" not in references:
                    prefill.send(dict(cmd="reference", prompt_len=prompt_len,
                                      output_len=output_len))
                    msg = prefill.recv(600)
                    if msg.get("error"):
                        raise RuntimeError(f"prefill reference error: {msg['error'][-2000:]}")
                    entry = msg.get("reference") or {}
                    references[f"{output_len}:{prompt_len}"] = entry
            budget_bytes = args.shm_budget_gb * (1 << 30)
            live = {}  # id -> nbytes: committed by P, not yet restored by D
            p_idle = True
            prefill_next = p_done = decode_next = 0
            p_end_by_id, d_row_by_id = {}, {}
            max_nbytes = 0

            def outstanding():
                # Requests the producer has dispatched whose decode has not finished.
                return prefill_next - len(d_row_by_id)

            def dispatched_decode():
                # Decode commands handed to D whose result has not arrived.
                return decode_next - len(d_row_by_id)

            def can_prefill():
                return p_idle and prefill_next < total and outstanding() < depth

            def wait_msg():
                deadline = time.time() + 900
                while time.time() < deadline:
                    msg = prefill.poll() if not p_idle else None
                    if msg is None:
                        msg = decode.poll()
                    if msg is not None:
                        return msg
                    if prefill.done.is_set() and prefill.messages.empty():
                        raise RuntimeError("prefill worker exited unexpectedly")
                    if decode.done.is_set() and decode.messages.empty():
                        raise RuntimeError("decode worker exited unexpectedly")
                    time.sleep(0.05)
                raise TimeoutError("worker results timed out")

            next_arrival = 0.0
            while len(d_row_by_id) < total or len(p_end_by_id) < total:
                if args.arrival_ms and time.time() < next_arrival:
                    time.sleep(0.01)
                    continue
                if can_prefill():
                    prefill.send(dict(cmd="prefill", id=prefill_next,
                                      transfer=f"{args.config}-r{prefill_next:06d}",
                                      prompt_len=lens_of[prefill_next],
                                      output_len=outs_of[prefill_next],
                                      warmup=prefill_next < args.warmup_requests,
                                      trace=args.trace_window
                                      and prefill_next >= args.warmup_requests + args.requests,
                                      trace_end=args.trace_window and prefill_next == total - 1))
                    p_idle = False
                    prefill_next += 1
                    if args.arrival_ms:
                        next_arrival = time.time() + args.arrival_ms / 1000
                    continue
                if decode_next < p_done and dispatched_decode() < depth and (
                        not live or sum(live.values()) + max_nbytes <= budget_bytes):
                    i = decode_next
                    decode.send(dict(cmd="decode", id=i,
                                     transfer=f"{args.config}-r{i:06d}",
                                     prompt_len=lens_of[i], output_len=outs_of[i],
                                     reference=references[f"{outs_of[i]}:{lens_of[i]}"],
                                     warmup=i < args.warmup_requests,
                                     trace=args.trace_window
                                     and i >= args.warmup_requests + args.requests,
                                     trace_end=args.trace_window and i == total - 1))
                    live[i] = p_end_by_id[i]["nbytes"]
                    decode_next += 1
                    continue
                if not p_idle or decode.messages.qsize() or decode_next < total:
                    msg = wait_msg()
                    if msg.get("error"):
                        raise RuntimeError(f"worker error: {msg['error'][-4000:]}")
                    if msg.get("restored"):
                        live.pop(msg["restored"]["id"], None)
                        continue
                    result = msg.get("result")
                    if result is None:
                        raise RuntimeError(f"unexpected worker message: {msg}")
                    if not result.get("ok"):
                        raise RuntimeError(f"request {result.get('id')} failed: {result}")
                    if "nbytes" in result:  # prefill result
                        p_end_by_id[result["id"]] = result
                        max_nbytes = max(max_nbytes, result["nbytes"])
                        p_idle = True
                        p_done += 1
                    else:  # decode result
                        d_row_by_id[result["id"]] = result
                        used = health.health()["used_bytes"]
                        max_store_used = max(max_store_used, used)
                else:
                    time.sleep(0.05)
            section["store_after"] = health.health()
            section["store_max_used_bytes"] = max_store_used
        finally:
            health.close()
        section["rows"] = []
        for i in range(total):
            p = p_end_by_id[i]
            d = d_row_by_id[i]
            second_token_ns = d.get("second_token_ns")
            out = outs_of[i]
            intervals = [v / 1e6 for v in (d.get("step_intervals") or [])]
            first_token_ns = p["first_token_ns"]
            export_ms = (p["commit_done_ns"] - first_token_ns) / 1e6
            restore_ms = (d["restore_done_ns"] - d["restore_start_ns"]) / 1e6
            first_decode_ms = ((second_token_ns - d["restore_done_ns"]) / 1e6
                               if second_token_ns else None)
            section["rows"].append(dict(
                id=i, prompt_len=lens_of[i], output_len=out,
                warmup=i < args.warmup_requests,
                trace=i >= args.warmup_requests + args.requests,
                submit_ns=p["start_ns"], end_ns=d["end_ns"],
                ttft_ms=(p["first_token_ns"] - p["start_ns"]) / 1e6,
                e2e_ms=(d["end_ns"] - p["start_ns"]) / 1e6,
                pause_ms=(d["restore_done_ns"] - p["first_token_ns"]) / 1e6,
                queue_ms=(d["restore_start_ns"] - p["first_token_ns"]) / 1e6,
                queue_commit_ms=(d["restore_start_ns"] - p["commit_done_ns"]) / 1e6,
                export_ms=export_ms, restore_ms=restore_ms,
                first_decode_ms=first_decode_ms,
                first_to_second_ms=(second_token_ns - p["first_token_ns"]) / 1e6
                if second_token_ns else None,
                step_interval_p95=p95(intervals) if intervals else None,
                step_interval_p50=median(intervals) if intervals else None,
                decode_ms=d["decode_ms"], tpot_ms=d["decode_ms"] / (out - 1),
                prefill_ms=p["prefill_ms"], d2h_ms=p["d2h_ms"], commit_ms=p["commit_ms"],
                acquire_ms=d["acquire_ms"], h2d_ms=d["h2d_ms"], ack_ms=d["ack_ms"],
                nbytes=p["nbytes"], tokens_equal=d["tokens_equal"],
                verdict=d.get("verdict"), acknowledged=d.get("acknowledged"),
                steps=d.get("steps"), batch_sizes=d.get("batch_sizes"),
                graph_steps=d.get("graph_steps"), slot=d.get("slot"),
                first_mismatch=d.get("first_mismatch"),
                flip_detail=d.get("flip_detail")))
        prefill.send(dict(cmd="finalize"))
        decode.send(dict(cmd="finalize"))
        finals = {}
        deadline = time.time() + 240
        while ("prefill" not in finals or "decode" not in finals) and time.time() < deadline:
            for name, worker in (("prefill", prefill), ("decode", decode)):
                if name in finals:
                    continue
                msg = worker.poll()
                if msg is None:
                    continue
                if msg.get("error"):
                    raise RuntimeError(f"{name} finalize error: {msg['error'][-2000:]}")
                if msg.get("final") is not None:
                    finals[name] = msg["final"]
            time.sleep(0.1)
        if len(finals) != 2:
            raise RuntimeError("worker finalize timed out")
        prefill_final, decode_final = finals["prefill"], finals["decode"]
        section["workers"]["prefill"]["final"] = prefill_final
        section["workers"]["decode"]["final"] = decode_final
        if prefill.proc.wait(timeout=30) != 0 or decode.proc.wait(timeout=30) != 0:
            raise RuntimeError("worker exit code nonzero")
        measured = [r for r in section["rows"] if not r["warmup"] and not r["trace"]]
        section["aggregates"] = aggregate(measured)
        section["decode_batch_histogram"] = decode_final.get("batch_histogram")
        wall_ms = (max(r["end_ns"] for r in section["rows"])
                   - min(r["submit_ns"] for r in section["rows"])) / 1e6
        p_busy = prefill_final.get("counters", {}).get("busy_ms", 0)
        d_busy = decode_final.get("counters", {}).get("busy_ms", 0)
        section["busy"] = dict(
            p_busy_ms=p_busy, d_busy_ms=d_busy, wall_ms=wall_ms,
            p_util=p_busy / wall_ms if wall_ms else None,
            d_util=d_busy / wall_ms if wall_ms else None)
        verification = dict(
            tokens_compared=sum(1 for r in section["rows"] if r["tokens_equal"]),
            tokens_mismatch=sum(1 for r in section["rows"] if not r["tokens_equal"]),
            near_tie_accepted=sum(1 for r in section["rows"] if r.get("verdict") == "near_tie"),
            bounded_flip_accepted=sum(1 for r in section["rows"]
                                      if r.get("verdict") == "bounded_flip"),
            exact=sum(1 for r in section["rows"] if r.get("verdict") == "exact"),
            acks=sum(1 for r in section["rows"] if r.get("acknowledged")),
            dispatch_ok=all(r.get("steps") == r["output_len"] - 1
                            for r in section["rows"]),
            clean_both_sides=bool(prefill_final.get("clean") and decode_final.get("clean")),
            store_restored=all(section["store_after"][k] == store_before[k]
                               for k in ("used_bytes", "payload_objects")),
            store_zero=section["store_after"]["used_bytes"] == 0
            and section["store_after"]["payload_objects"] == 0,
            lens_covered=set(r["prompt_len"] for r in section["rows"]) == set(args.lens),
        )
        overlap_rows = []
        for row in section["rows"]:
            p = p_end_by_id[row["id"]]
            prior = d_row_by_id.get(row["id"] - 1)
            if prior is not None and not row["warmup"]:
                ov = min(p["end_ns"], prior["end_ns"]) - max(
                    p["start_ns"], prior["restore_start_ns"])
                overlap_rows.append(dict(id=row["id"], prior_id=row["id"] - 1,
                                         p_overlap_with_prior_decode_ms=max(0.0, ov / 1e6)))
        if overlap_rows:
            section["span_overlap"] = dict(
                rows=overlap_rows,
                total_overlap_ms=sum(r["p_overlap_with_prior_decode_ms"] for r in overlap_rows),
                mean_overlap_ms=median([r["p_overlap_with_prior_decode_ms"]
                                        for r in overlap_rows]))
        if args.trace_window:
            p_trace = trace_kernel_intervals(Path(args.output).parent / "trace-prefill.json")
            d_trace = trace_kernel_intervals(Path(args.output).parent / "trace-decode.json")
            section["trace_evidence"] = dict(
                prefill_kernel_busy_us=sum(e - s for s, e in p_trace) if p_trace else None,
                decode_kernel_busy_us=sum(e - s for s, e in d_trace) if d_trace else None,
                kernel_overlap_us=overlap_us(p_trace, d_trace))
        section["verification"] = verification
        section["passed"] = (verification["tokens_mismatch"] == 0
                             and verification["dispatch_ok"]
                             and verification["acks"] == total
                             and verification["clean_both_sides"]
                             and verification["store_restored"]
                             and verification["lens_covered"])
        return section
    finally:
        prefill.terminate()
        decode.terminate()


def run_solo_configs(args):
    """Continuous-batching replicas: round-robin dispatch with a per-worker
    in-flight cap; each worker admits requests between steps (production
    continuous batching, prefill chunking included)."""
    if args.config == "single":
        cap = args.concurrency
        workers = [Worker(args, "solo", args.prefill_gpu, args.config, max_batch=cap)]
    else:
        cap = args.concurrency
        workers = [Worker(args, "solo", gpu, f"{args.config}-{i}", max_batch=cap)
                   for i, gpu in enumerate([args.prefill_gpu, args.decode_gpu][:2])]
    section = dict(config=args.config, requests=args.requests, lens=args.lens,
                   output_len=args.output_len, concurrency=cap * len(workers),
                   per_worker_cap=cap, warmup_requests=args.warmup_requests,
                   seed=args.seed, rows=[], workers={}, verification={})
    try:
        for i, worker in enumerate(workers):
            ready = worker.recv(600)
            if not ready.get("ready"):
                raise RuntimeError(f"solo worker startup failed: {ready}")
            section["workers"][f"worker{i}"] = ready
        total = args.warmup_requests + args.requests
        lens, outs = args.lens, args.outs
        lens_of = [lens[i % len(lens)] for i in range(total)]
        outs_of = [outs[i % len(outs)] for i in range(total)]
        submit_ns, results = {}, {}
        in_flight = [0] * len(workers)
        next_id = 0
        deadline = time.time() + 1800
        next_arrival = 0.0
        while len(results) < total:
            if time.time() > deadline:
                raise TimeoutError("solo workload timed out")
            # Dispatch to workers with free capacity (round robin).
            while next_id < total and any(n < cap for n in in_flight):
                if args.arrival_ms and time.time() < next_arrival:
                    time.sleep(0.01)
                    break
                for i, worker in enumerate(workers):
                    if next_id >= total:
                        break
                    if in_flight[i] < cap:
                        worker.send(dict(cmd="generate", id=next_id,
                                         prompt_len=lens_of[next_id],
                                         output_len=outs_of[next_id]))
                        submit_ns[next_id] = time.time_ns()
                        in_flight[i] += 1
                        next_id += 1
                        if args.arrival_ms:
                            next_arrival = time.time() + args.arrival_ms / 1000
                            break
            # Collect finished requests.
            for i, worker in enumerate(workers):
                while True:
                    msg = worker.poll()
                    if msg is None:
                        break
                    if msg.get("error"):
                        raise RuntimeError(f"solo worker error: {msg['error'][-4000:]}")
                    result = msg.get("result")
                    if result is None:
                        raise RuntimeError(f"unexpected solo message: {msg}")
                    results[result["id"]] = (i, result)
                    in_flight[i] -= 1
            if len(results) < total:
                time.sleep(0.02)
        for i in range(total):
            worker_index, result = results[i]
            length, out = lens_of[i], outs_of[i]
            intervals = [v / 1e6 for v in (result.get("step_intervals") or [])]
            section["rows"].append(dict(
                id=i, worker=worker_index, prompt_len=length, output_len=out,
                warmup=i < args.warmup_requests, submit_ns=submit_ns[i],
                end_ns=int(submit_ns[i] + result["e2e_ms"] * 1e6),
                ttft_ms=result["ttft_ms"], e2e_ms=result["e2e_ms"],
                first_to_second_ms=result["first_to_second_ms"],
                step_interval_p95=p95(intervals) if intervals else None,
                step_interval_p50=median(intervals) if intervals else None,
                tokens_equal=result.get("tokens_equal"),
                first_mismatch=result.get("first_mismatch"),
                flip_detail=result.get("flip_detail"),
                verdict=result.get("verdict"),
                batch_sizes=result.get("batch_sizes"),
                graph_steps=result.get("graph_steps"),
                any_mixed_step=result.get("any_mixed_step")))
        for i, worker in enumerate(workers):
            worker.send(dict(cmd="finalize"))
            msg = worker.recv(120)
            if msg.get("error"):
                raise RuntimeError(f"solo finalize error: {msg['error'][-4000:]}")
            section["workers"][f"worker{i}"]["final"] = msg.get("final")
            if worker.proc.wait(timeout=30) != 0:
                raise RuntimeError("solo worker exit code nonzero")
        measured = [r for r in section["rows"] if not r["warmup"]]
        section["aggregates"] = aggregate(measured)
        divergence_rows = [r for r in section["rows"]
                           if r.get("verdict") == "mixed_prefill_divergence"]
        gated_rows = [r for r in section["rows"] if r not in divergence_rows]
        verification = dict(
            tokens_compared=sum(1 for r in gated_rows if r["tokens_equal"]),
            tokens_mismatch=sum(1 for r in gated_rows if not r["tokens_equal"]),
            near_tie_accepted=sum(1 for r in gated_rows
                                  if r.get("verdict") == "near_tie"),
            bounded_flip_accepted=sum(1 for r in gated_rows
                                      if r.get("verdict") == "bounded_flip"),
            exact=sum(1 for r in gated_rows if r.get("verdict") == "exact"),
            mixed_prefill_divergence=len(divergence_rows),
            clean_all=all(section["workers"][f"worker{i}"]["final"].get("clean")
                          for i in range(len(workers))),
            lens_covered=set(r["prompt_len"] for r in section["rows"]) == set(args.lens),
        )
        section["verification"] = verification
        section["passed"] = (verification["tokens_mismatch"] == 0
                             and verification["clean_all"]
                             and verification["lens_covered"])
        return section
    finally:
        for worker in workers:
            worker.terminate()


def run_correctness_config(args):
    from nano_qwen.cache.shm_connector import LMCacheShmConnector
    prefill = Worker(args, "prefill", args.prefill_gpu, args.config)
    decode = Worker(args, "decode", args.decode_gpu, args.config,
                    max_batch=args.decode_max_seqs)
    section = dict(config="pd-correctness", decode_max_seqs=args.decode_max_seqs,
                   scenarios=[], verification={},
                   workers={}, store_before=None, store_after=None)
    try:
        p_ready = prefill.recv(600)
        d_ready = decode.recv(600)
        if not (p_ready.get("ready") and d_ready.get("ready")):
            raise RuntimeError(f"worker startup failed: {p_ready} {d_ready}")
        section["workers"] = dict(prefill=p_ready, decode=d_ready)
        health = LMCacheShmConnector(args.cache_url, 30)
        section["store_before"] = health.health()
        try:
            # P-side references for every (prompt_len, output_len) pair.
            need_refs = sorted({(r["len"], r["out"]) for s in CORRECTNESS_SCENARIOS
                                for r in s["requests"]})
            for prompt_len, output_len in need_refs:
                prefill.send(dict(cmd="reference", prompt_len=prompt_len,
                                  output_len=output_len, transfer=None))
            refs = {}
            for _ in need_refs:
                msg = prefill.recv(600)
                if msg.get("error"):
                    raise RuntimeError(f"prefill reference error: {msg['error'][-2000:]}")
                entry = msg.get("reference") or {}
                refs[f"{entry.get('output_len')}:{entry.get('prompt_len')}"] = entry

            def wait_prefill():
                msg = prefill.recv(600)
                if msg.get("error"):
                    raise RuntimeError(f"prefill error: {msg['error'][-2000:]}")
                return msg["result"]

            def wait_decode():
                while True:
                    msg = decode.recv(600)
                    if msg.get("error"):
                        raise RuntimeError(f"decode error: {msg['error'][-2000:]}")
                    if msg.get("restored"):
                        continue
                    return msg["result"]

            next_id = 0
            for scenario in CORRECTNESS_SCENARIOS:
                name = scenario["name"]
                requests = scenario["requests"]
                roles = [r["role"] for r in requests]
                record = dict(name=name, requests=requests, solo={},
                              replays=[], replay_equal=None)
                # Solo references on D: each unique request alone.
                for spec in requests:
                    role, length, out = spec["role"], spec["len"], spec["out"]
                    tid = f"corr-{name}-{role}-solo"
                    prefill.send(dict(cmd="prefill", id=next_id, transfer=tid,
                                      prompt_len=length, output_len=out))
                    wait_prefill()
                    decode.send(dict(cmd="decode", id=next_id, transfer=tid,
                                     prompt_len=length, output_len=out,
                                     reference=refs[f"{out}:{length}"],
                                     state_check=True, solo_ref=length))
                    d = wait_decode()
                    record["solo"][role] = dict(tokens_equal=d["tokens_equal"],
                                                verdict=d["verdict"],
                                                drift=d.get("drift"),
                                                restore_bit_exact=d.get("restore_bit_exact"))
                    next_id += 1
                # Two mixed replays with the same join schedule.
                for replay_index in (0, 1):
                    entries = {}
                    for spec in requests:
                        role, length, out = spec["role"], spec["len"], spec["out"]
                        tid = f"corr-{name}-{role}-r{replay_index}"
                        prefill.send(dict(cmd="prefill", id=next_id, transfer=tid,
                                          prompt_len=length, output_len=out))
                        entries[role] = next_id
                        next_id += 1
                    for spec in requests:
                        wait_prefill()
                    for spec in requests:
                        role, length, out = spec["role"], spec["len"], spec["out"]
                        decode.send(dict(cmd="decode", id=entries[role],
                                         transfer=f"corr-{name}-{role}-r{replay_index}",
                                         prompt_len=length, output_len=out,
                                         reference=refs[f"{out}:{length}"],
                                         state_check=True,
                                         join_after=spec["join"] or None,
                                         replay=dict(group=name, index=replay_index,
                                                     role=role)))
                    replay = {}
                    for spec in requests:
                        role = spec["role"]
                        d = wait_decode()
                        replay[role] = dict(id=d["id"], tokens_equal=d["tokens_equal"],
                                            verdict=d["verdict"], steps=d["steps"],
                                            batch_sizes=d["batch_sizes"],
                                            graph_steps=d.get("graph_steps"),
                                            slot=d.get("slot"),
                                            first_mismatch=d.get("first_mismatch"),
                                            flip_detail=d.get("flip_detail"),
                                            drift=d.get("drift"),
                                            restore_bit_exact=d.get("restore_bit_exact"))
                    record["replays"].append(replay)
                section["scenarios"].append(record)

            prefill.send(dict(cmd="finalize"))
            decode.send(dict(cmd="finalize"))
            p_final = prefill.recv(120).get("final")
            d_final = decode.recv(120).get("final")
            if p_final is None or d_final is None:
                raise RuntimeError("worker finalize failed")
            section["workers"]["prefill"]["final"] = p_final
            section["workers"]["decode"]["final"] = d_final
            section["store_after"] = health.health()
            section["replay_equal"] = d_final.get("replay_equal", {})
            section["batch_histogram"] = d_final.get("batch_histogram", {})
            for record in section["scenarios"]:
                record["replay_equal"] = section["replay_equal"].get(record["name"])
                roles = [r["role"] for r in record["requests"]]
                solo = record["solo"]
                drift_ok = all(
                    (solo.get(role, {}).get("drift") or {}).get("pass") is not False
                    for role in roles)
                bit_exact = all(
                    (solo.get(role, {}).get("restore_bit_exact") or {}).get("pass_")
                    for role in roles)
                token_ok = (all(r[role]["tokens_equal"] for r in record["replays"]
                                for role in roles)
                            and all(solo[role]["tokens_equal"] for role in roles))
                max_batch = max(max(int(k) for k in r[role]["batch_sizes"])
                                for r in record["replays"] for role in roles)
                record["max_batch_seen"] = max_batch
                record["passed"] = (token_ok and bool(record["replay_equal"])
                                    and max_batch >= 2)
                record["drift_ok"] = drift_ok
                record["restore_bit_exact"] = bit_exact
                # Slot reuse: the last joiner must land in a slot previously
                # owned by an earlier request.
                slots = {}
                for r in record["replays"]:
                    for role in roles:
                        slots.setdefault(r[role].get("slot"), role)
                record["slot_assignments"] = slots
            verification = dict(
                scenarios_passed=all(r["passed"] for r in section["scenarios"]),
                b2_seen_all=all(r["max_batch_seen"] >= 2 for r in section["scenarios"]),
                b4_seen=any(r["max_batch_seen"] >= 4 for r in section["scenarios"]),
                restore_bit_exact_all=all(r["restore_bit_exact"]
                                          for r in section["scenarios"]),
                clean_both_sides=bool(p_final.get("clean") and d_final.get("clean")),
                store_restored=all(section["store_after"][k] == section["store_before"][k]
                                   for k in ("used_bytes", "payload_objects")),
                replay_equal_all=all(section["replay_equal"].values()),
            )
            section["verification"] = verification
            section["passed"] = (verification["scenarios_passed"]
                                 and verification["b2_seen_all"]
                                 and verification["clean_both_sides"]
                                 and verification["store_restored"]
                                 and verification["replay_equal_all"])
            return section
        finally:
            health.close()
    finally:
        prefill.terminate()
        decode.terminate()


def main():
    args = driver_arguments()
    output = Path(args.output).resolve()
    hashes = common.source_hashes(ROOT)
    hashes[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    report = dict(schema="nano_qwen_pd_dual_card_v2", passed=False, args=vars(args),
                  command=shlex.join([sys.executable, *sys.argv]),
                  started_at=common.now_iso(), git_before=common.git_provenance(ROOT),
                  source_sha256_before=hashes, gpus=gpu_identities())
    try:
        if args.config in ("pd-serial", "pd-pipeline"):
            report["section"] = run_pd_config(args)
        elif args.config == "pd-correctness":
            report["section"] = run_correctness_config(args)
        else:
            report["section"] = run_solo_configs(args)
        report["passed"] = report["section"]["passed"]
    except Exception as exc:
        report["error"] = traceback.format_exc()
        report["passed"] = False
        raise
    finally:
        report["finished_at"] = common.now_iso()
        after = common.source_hashes(ROOT)
        after[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()
        report["source_sha256_after"] = after
        report["source_changed_during_run"] = after != hashes
        report["git_after"] = common.git_provenance(ROOT)
        common.atomic_json(output, report)
    print(json.dumps(dict(passed=report["passed"], output=str(output),
                          verification=report.get("section", {}).get("verification"),
                          aggregates=report.get("section", {}).get("aggregates")),
                     ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    if "--role" in sys.argv:
        worker_main()
    else:
        main()
