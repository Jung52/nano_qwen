"""Validate the pinned BF16 Qwen3.5 MoE checkpoint on the real engine.

Generate --reference-dir with export_qwen35_moe_reference.py. The report
includes greedy outputs, teacher-forced logits and GDN/KV state drift.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
import transformers
import triton
from safetensors import safe_open
from safetensors.torch import load_file

from engine_bench_utils import (
    ArgmaxSampler,
    LLMEngine,
    SamplingParams,
    Sequence,
    compare_probed_request_drift,
    drive_probe,
    env_info,
    validation_settings,
)
from export_qwen35_moe_reference import CASES, LFS_SHA256, REVISION, verify_revision, write_json


CHUNK_BUDGET = 64
BOUNDARY_LENGTHS = (63, 64, 65, 255, 256, 257)
STATE_ATOL = 5e-4
STATE_RTOL = 5e-3


class TeacherForcedSampler(ArgmaxSampler):
    """Observe the real sampler input before forcing the next token by request ID."""

    def __init__(self, continuations):
        super().__init__()
        self.continuations = continuations
        self.sampled_requests = []

    def set_batch_context(self, seqs, is_prefill):
        super().set_batch_context(seqs, is_prefill)
        self.sampled_requests = list(seqs)

    def forward(self, logits, temperatures, all_greedy=False):
        assert len(self.sampled_requests) == logits.size(0)
        super().forward(logits, temperatures, all_greedy)
        return torch.tensor(
            [self.continuations[seq.seq_id][seq.num_completion_tokens]
             for seq in self.sampled_requests],
            device=logits.device, dtype=torch.long,
        )


def check_drained(engine):
    runner = engine.model_runner
    blocks = engine.scheduler.block_manager
    assert engine.is_finished() and runner._pending is None
    assert not runner.input_batch.seq_id_to_slot and all(s is None for s in runner.input_batch.seqs)
    assert not blocks.used_block_ids and len(blocks.free_block_ids) == len(blocks.blocks)


def probe(engine, requests, budget, *, check_states=False, late=None, after_step=None):
    """Each request provides its prompt and the sequence of token IDs to emit."""
    seqs = {label: Sequence(prompt, SamplingParams(temperature=1e-6, max_tokens=len(tokens), ignore_eos=True))
            for label, (prompt, tokens) in requests.items()}
    continuations = {seq.seq_id: requests[label][1] for label, seq in seqs.items()}
    sampler = TeacherForcedSampler(continuations)
    with validation_settings(engine, budget=budget), patch.object(engine.model_runner, "sampler", sampler):
        for label, seq in seqs.items():
            if label != late:
                engine.scheduler.add(seq)
        observed = drive_probe(engine, seqs, after_step=after_step,
                               snapshot_positions=None if check_states else set(), step_cap=32)
    check_drained(engine)
    for label, seq in seqs.items():
        assert seq.completion_token_ids == continuations[seq.seq_id]
        assert set(observed.diagnostics[label]) == set(range(seq.max_tokens))
    return observed


def logit_rows(observed, label):
    return torch.stack([observed.diagnostics[label][i]["logits"]
                        for i in range(observed.seqs[label].max_tokens)])


def compare_logits(reference, actual):
    assert reference.shape == actual.shape
    a, b = reference.float(), actual.float()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    ids_a, ids_b = a.argmax(dim=-1).tolist(), b.argmax(dim=-1).tolist()
    values_a, top_a = a.topk(2)
    values_b, top_b = b.topk(2)
    delta = b - a
    return {
        "rows": len(a),
        "argmax_matches": sum(i == j for i, j in zip(ids_a, ids_b)),
        "reference_argmax": ids_a,
        "candidate_argmax": ids_b,
        "reference_top2_ids": top_a.tolist(),
        "candidate_top2_ids": top_b.tolist(),
        "reference_top1_margin": (values_a[:, 0] - values_a[:, 1]).tolist(),
        "candidate_top1_margin": (values_b[:, 0] - values_b[:, 1]).tolist(),
        "relative_rmse": (delta.norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-12)).tolist(),
        "max_abs": delta.abs().amax(dim=-1).tolist(),
        "cosine_similarity": F.cosine_similarity(a, b).tolist(),
    }


def greedy(engine, requests, output_tokens, budget):
    with validation_settings(engine, budget=budget):
        params = SamplingParams(temperature=1e-6, max_tokens=output_tokens, ignore_eos=True)
        outputs = engine.generate([prompt for prompt, _ in requests.values()], params, use_tqdm=False)
    check_drained(engine)
    assert len(outputs) == len(requests)
    assert all(len(result["token_ids"]) == output_tokens for result in outputs)
    return dict(zip(requests, outputs))


def validate_batches(engine, reference, reference_logits):
    requests = {}
    for label, _, _ in CASES:
        case = reference["cases"][label]
        prompt = case["prompt_token_ids"]
        assert engine.tokenizer.encode(case["prompt"], add_special_tokens=False) == prompt
        fixed = case["fixed_decode_token_ids"]
        requests[label] = (prompt, [*fixed, fixed[-1]])
    # Distinct prompts and decode histories make swapped rows visible at B4.
    for label, text in (("arithmetic", "The sum of two and five is"),
                        ("chinese_question", "天空为什么是蓝色的？")):
        prompt = engine.tokenizer.encode(text, add_special_tokens=False)
        requests[label] = (prompt, (prompt * 9)[:9])

    b1 = {}
    b1_greedy = {}
    for label, request in requests.items():
        observed = probe(engine, {label: request}, CHUNK_BUDGET)
        b1[label] = logit_rows(observed, label)
        b1_greedy.update(greedy(engine, {label: request}, 8, CHUNK_BUDGET))

    hf = {}
    for label, case in reference["cases"].items():
        hf[label] = compare_logits(reference_logits[label], b1[label])
        hf[label]["greedy"] = b1_greedy[label]
        hf[label]["reference_greedy_ids"] = case["greedy_token_ids"]
        hf[label]["greedy_match"] = b1_greedy[label]["token_ids"] == case["greedy_token_ids"]
        print(f"Transformers {label}: argmax {hf[label]['argmax_matches']}/{hf[label]['rows']}, "
              f"greedy_match={hf[label]['greedy_match']}", flush=True)

    batches = {}
    for width in (2, 4):
        batch = dict(list(requests.items())[:width])
        observed = probe(engine, batch, CHUNK_BUDGET)
        assert len(observed.batches) == 9 and all(len(step["rows"]) == width for step in observed.batches)
        assert all(r["q_len"] == len(batch[r["label"]][0]) for r in observed.batches[0]["rows"])
        assert all(not dispatch["any_prefill"] for dispatch in observed.batches[1:])
        outputs = greedy(engine, batch, 8, CHUNK_BUDGET)
        results = {label: compare_logits(b1[label], logit_rows(observed, label)) for label in batch}
        for label, result in results.items():
            result["greedy"] = outputs[label]
            result["greedy_match"] = outputs[label]["token_ids"] == b1_greedy[label]["token_ids"]
        batches[f"B{width}"] = {"dispatches": observed.batches, "requests": results}
        print(f"B{width} vs B1: argmax {sum(r['argmax_matches'] for r in results.values())}/{width * 9}, "
              f"greedy matches {sum(r['greedy_match'] for r in results.values())}/{width}", flush=True)
    return requests, b1, {"B1_vs_transformers": hf, "B1_greedy": b1_greedy, **batches}


def validate_decode_graph_matches_eager(engine, requests):
    runner = engine.model_runner
    comparisons = {}
    for width in (1, 2, 4):
        batch = dict(list(requests.items())[:width])
        runner.enforce_eager = True
        eager_observed = probe(engine, batch, CHUNK_BUDGET)
        eager_logits = {label: logit_rows(eager_observed, label) for label in batch}
        eager_greedy = greedy(engine, batch, 8, CHUNK_BUDGET)

        runner.cuda_graphs.reset_dispatch_stats()
        runner.enforce_eager = False
        graph_observed = probe(engine, batch, CHUNK_BUDGET)
        graph_greedy = greedy(engine, batch, 8, CHUNK_BUDGET)
        dispatch = runner.cuda_graphs.dispatch_stats().get("decode", {})
        assert dispatch.get("graph_calls") == 15, (width, dispatch)
        assert dispatch.get("eager_calls", 0) == 0, (width, dispatch)

        results = {}
        for label in batch:
            logits = compare_logits(eager_logits[label], logit_rows(graph_observed, label))
            greedy_match = eager_greedy[label]["token_ids"] == graph_greedy[label]["token_ids"]
            assert logits["argmax_matches"] == logits["rows"], (width, label, logits)
            assert greedy_match, (width, label, eager_greedy[label], graph_greedy[label])
            results[label] = {"logits": logits, "greedy_match": greedy_match,
                              "eager_greedy": eager_greedy[label],
                              "graph_greedy": graph_greedy[label]}
        comparisons[f"B{width}"] = {"requests": results, "dispatch": dispatch}
        print(f"B{width} captured decode replay: eager and Graph outputs match", flush=True)
    runner.enforce_eager = False
    return comparisons


def compare_probe_states(reference, candidate, label):
    return compare_probed_request_drift(
        reference, candidate, label, output_tokens=candidate.seqs[label].max_tokens,
        atol=STATE_ATOL, rtol=STATE_RTOL,
    )


def validate_piecewise_prefill(engine, boundaries):
    runner = engine.model_runner
    report = {}
    try:
        for case in boundaries:
            length = case["prompt_tokens"]
            request = {"Q": (case["prompt_ids"], case["forced_output_ids"])}
            for budget in sorted({length, CHUNK_BUDGET}):
                runner.use_prefill_cudagraph = False
                expected = probe(engine, request, budget, check_states=True)
                runner.use_prefill_cudagraph = True
                runner.cuda_graphs.reset_dispatch_stats()
                actual = probe(engine, request, budget, check_states=True)
                dispatch = runner.cuda_graphs.dispatch_stats()["prefill"]
                buckets = runner.prefill_piecewise_graphs
                expected_hits = sum(
                    count for tokens, count in dispatch["token_counts"].items() if tokens in buckets
                )
                assert dispatch["graph_calls"] == expected_hits, dispatch
                label = f"{length}_tokens_budget_{budget}"
                expected_states = expected.snapshots["Q"]
                actual_states = actual.snapshots["Q"]
                assert expected_states.keys() == actual_states.keys(), label
                for index, tensors in expected_states.items():
                    assert tensors.keys() == actual_states[index].keys(), (label, index)
                    for name, tensor in tensors.items():
                        assert torch.equal(tensor, actual_states[index][name]), (label, index, name)
                assert torch.equal(logit_rows(expected, "Q"), logit_rows(actual, "Q")), label
                report[label] = {"dispatch": dispatch, "logits_and_states_exact": True}
    finally:
        runner.use_prefill_cudagraph = False
    assert any(case["dispatch"]["graph_calls"] for case in report.values())
    assert any(case["dispatch"]["eager_calls"] for case in report.values())
    return report


def validate_async_lookahead(engine, requests):
    runner = engine.model_runner
    original_depth = engine.max_concurrent_batches
    original_async = engine.config.async_scheduling
    engine.max_concurrent_batches = 2
    report = {}
    try:
        for width in (1, 2, 4):
            batch = dict(list(requests.items())[:width])
            engine.config.async_scheduling = False
            expected = greedy(engine, batch, 16, CHUNK_BUDGET)
            engine.config.async_scheduling = True
            lookahead_dispatches = []
            call = runner.call

            def record_dispatch(name, *args):
                if name == "execute_model" and not args[1]:
                    seqs = args[0]
                    if all(getattr(seq, "_async_decode_view", False) for seq in seqs):
                        lookahead_dispatches.append(len(seqs))
                return call(name, *args)

            runner.cuda_graphs.reset_dispatch_stats()
            with patch.object(runner, "call", record_dispatch):
                actual = greedy(engine, batch, 16, CHUNK_BUDGET)
            dispatch = runner.cuda_graphs.dispatch_stats()["decode"]
            assert lookahead_dispatches == [width] * 15, (width, lookahead_dispatches)
            assert dispatch["graph_calls"] == 15 and dispatch["eager_calls"] == 0, dispatch
            assert actual == expected, width
            report[f"B{width}"] = {"greedy_match": True, "dispatch": dispatch,
                                  "lookahead_dispatches": len(lookahead_dispatches)}
    finally:
        engine.config.async_scheduling = original_async
        engine.max_concurrent_batches = original_depth
    return report


def validate_chunked_prefill(engine):
    tokenizer = engine.tokenizer
    paragraph = tokenizer.encode(
        "Paris and Rome are European capitals. Each city has a long history, distinct architecture, "
        "and public museums. Visitors often arrive by train and explore the old streets on foot. ",
        add_special_tokens=False,
    )
    tail = tokenizer.encode("\nQuestion: What is the capital of France?\nAnswer:", add_special_tokens=False)
    forced = tokenizer.encode(" Paris is the capital of France.", add_special_tokens=False)[:4]
    results = []
    for length in BOUNDARY_LENGTHS:
        prefix = (paragraph * length)[:length - len(tail)]
        request = {"Q": (prefix + tail, forced)}
        full = probe(engine, request, length, check_states=True)
        chunked = probe(engine, request, CHUNK_BUDGET, check_states=True)
        expected = [min(CHUNK_BUDGET, length - start) for start in range(0, length, CHUNK_BUDGET)]
        prefills = [dispatch for dispatch in chunked.batches if dispatch["any_prefill"]]
        chunks = [dispatch["rows"][0]["q_len"] for dispatch in prefills]
        assert [dispatch["rows"][0]["q_len"] for dispatch in full.batches if dispatch["any_prefill"]] == [length]
        assert chunks == expected, (length, chunks)
        assert [dispatch["metadata"]["paged"] for dispatch in prefills] == [False] + [True] * (len(chunks) - 1)
        assert all(dispatch["rows"][0]["completion_index"] == 0 for dispatch in prefills)
        assert len({tuple(dispatch["metadata"]["state_slots"]) for dispatch in chunked.batches}) == 1
        logits = compare_logits(logit_rows(full, "Q"), logit_rows(chunked, "Q"))
        states = compare_probe_states(full, chunked, "Q")
        full_greedy = greedy(engine, request, len(forced), length)["Q"]
        chunked_greedy = greedy(engine, request, len(forced), CHUNK_BUDGET)["Q"]
        greedy_match = full_greedy["token_ids"] == chunked_greedy["token_ids"]
        print(f"Prefill {length}: chunks={chunks}, argmax {logits['argmax_matches']}/{logits['rows']}, "
              f"greedy_match={greedy_match}, bounded state drift={states['pass']}", flush=True)
        greedy_divergence = None
        if not greedy_match:
            one_shot_ids = full_greedy["token_ids"]
            chunked_ids = chunked_greedy["token_ids"]
            index = next(i for i, (a, b) in enumerate(zip(one_shot_ids, chunked_ids)) if a != b)
            shared_history = {"Q": (request["Q"][0], one_shot_ids[:index + 1])}
            full_path = logit_rows(probe(engine, shared_history, length), "Q")
            chunked_path = logit_rows(probe(engine, shared_history, CHUNK_BUDGET), "Q")
            greedy_divergence = {
                "output_index": index, "one_shot_token_id": one_shot_ids[index],
                "chunked_token_id": chunked_ids[index],
                "logits": compare_logits(full_path[index:index + 1], chunked_path[index:index + 1]),
            }
        results.append({"prompt_tokens": length, "prompt_ids": request["Q"][0],
                        "forced_output_ids": forced, "chunks": chunks,
                        "dispatches": chunked.batches, "logits": logits, "states": states,
                        "one_shot_greedy": full_greedy, "chunked_greedy": chunked_greedy,
                        "greedy_match": greedy_match, "greedy_divergence": greedy_divergence})
    return results


def run_staggered(engine, requests, early, late, *, clear_released_slot):
    runner = engine.model_runner
    finished_at = {}
    initial_slots = {}
    remaining_slots = {}
    stale_layers = []
    early_blocks = []
    step = 0

    def enqueue_after_release(observed):
        nonlocal step
        step += 1
        slots = runner.input_batch.seq_id_to_slot
        live = {label: slots[seq.seq_id] for label, seq in observed.seqs.items() if seq.seq_id in slots}
        if step == 1:
            initial_slots.update(live)
            early_blocks.extend(observed.seqs[early].block_table)
        for label, seq in observed.seqs.items():
            if seq.is_finished and label not in finished_at:
                assert seq.seq_id not in slots and not seq.block_table
                finished_at[label] = step
                if label == early:
                    assert set(live) == set(initial_slots) - {early}
                    remaining_slots.update(live)
                    manager = engine.scheduler.block_manager
                    assert early_blocks and not manager.used_block_ids.intersection(early_blocks)
                    slot = initial_slots[early]
                    for layer in runner.gdn_layers:
                        if torch.count_nonzero(layer.recurrent_states[slot]).item():
                            stale_layers.append(layer.layer_idx)
                        if clear_released_slot:
                            layer.reset_state(slot)
                    assert stale_layers, "The reuse test needs a nonzero released state"
                    engine.scheduler.add(observed.seqs[late])

    observed = probe(engine, requests, CHUNK_BUDGET, check_states=True,
                     late=late, after_step=enqueue_after_release)
    assert finished_at[early] < min(finished_at[label] for label in finished_at if label != early)
    owners = {}
    for dispatch in observed.batches:
        for row, slot in zip(dispatch["rows"], dispatch["metadata"]["state_slots"]):
            assert owners.setdefault(row["label"], slot) == slot
    assert len(set(initial_slots.values())) == 4 and owners[early] == owners[late]
    mixed = next(dispatch for dispatch in observed.batches if any(r["label"] == late for r in dispatch["rows"]))
    assert len(mixed["rows"]) == 4 and mixed["metadata"]["paged"]
    assert {row["label"] for row in mixed["rows"] if row["prefill"]} == {late}
    return observed, {"finish_steps": finished_at, "state_slots": owners,
                      "live_slots_after_release": remaining_slots,
                      "released_slot_nonzero_recurrent_layers": stale_layers}


def validate_slot_reuse(engine, requests, b1):
    output_lengths = dict(zip(requests, (2, 4, 5, 6)))
    early = next(iter(output_lengths))
    late = "late_arrival"
    prompt = engine.tokenizer.encode("Language models predict the next token.", add_special_tokens=False)
    late_request = {late: (prompt, prompt[:4])}
    baseline = probe(engine, late_request, CHUNK_BUDGET, check_states=True)
    staggered = {label: (ids, tokens[:output_lengths[label]]) for label, (ids, tokens) in requests.items()}
    staggered.update(late_request)
    observed, lifecycle = run_staggered(engine, staggered, early, late, clear_released_slot=False)
    clean, clean_lifecycle = run_staggered(engine, staggered, early, late, clear_released_slot=True)
    assert lifecycle == clean_lifecycle
    assert [b["rows"] for b in observed.batches] == [b["rows"] for b in clean.batches]
    assert [b["metadata"]["state_slots"] for b in observed.batches] == [b["metadata"]["state_slots"] for b in clean.batches]
    for label in staggered:
        assert torch.equal(logit_rows(observed, label), logit_rows(clean, label)), label
        assert observed.snapshots[label].keys() == clean.snapshots[label].keys()
        assert len(observed.snapshots[label]) == observed.seqs[label].max_tokens
        for index, snapshot in observed.snapshots[label].items():
            other = clean.snapshots[label][index]
            assert snapshot.keys() == other.keys()
            for name, tensor in snapshot.items():
                assert torch.equal(tensor, other[name]), (label, index, name)

    states = compare_probe_states(baseline, observed, late)
    logits = {label: compare_logits(b1[label][:n], logit_rows(observed, label))
              for label, n in output_lengths.items()}
    logits[late] = compare_logits(logit_rows(baseline, late), logit_rows(observed, late))
    print(f"Slot reuse: {early} -> {late} at slot {lifecycle['state_slots'][late]}, "
          f"same mixed layout after explicit clear: exact; B1 drift within bounds={states['pass']}", flush=True)
    return {**lifecycle, "same_mixed_layout_after_explicit_clear_exact": True,
            "dispatches": observed.batches, "logits": logits, "late_arrival_states": states}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/qwen35-moe-engine-validation.json"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    args = parser.parse_args()

    verify_revision(args.checkpoint)
    reference = json.loads((args.reference_dir / "reference.json").read_text(encoding="utf-8"))
    logits_path = args.reference_dir / "reference_logits.safetensors"
    with safe_open(logits_path, framework="pt") as tensors:
        assert reference["revision"] == tensors.metadata()["revision"] == REVISION
    reference_logits = load_file(logits_path)
    torch.set_num_threads(min(torch.get_num_threads(), 4))

    settings = dict(tensor_parallel_size=1, max_num_seqs=4, max_model_len=512,
                    max_num_batched_tokens=max(BOUNDARY_LENGTHS), async_scheduling=False,
                    gpu_memory_utilization=args.gpu_memory_utilization)
    print("Loading nano_qwen MoE decode Graph engine", flush=True)
    with torch.inference_mode():
        engine = LLMEngine(str(args.checkpoint), **settings, use_prefill_cudagraph=True)
        try:
            runner = engine.model_runner
            assert not runner.enforce_eager
            assert set(runner.graphs) == {1, 2, 4}
            assert sorted(runner.prefill_piecewise_graphs) == [64, 128, 256, 257]
            runner.use_prefill_cudagraph = False
            engine.max_concurrent_batches = 1
            requests, b1, batches = validate_batches(engine, reference, reference_logits)
            graph_parity = validate_decode_graph_matches_eager(engine, requests)
            boundaries = validate_chunked_prefill(engine)
            reuse = validate_slot_reuse(engine, requests, b1)
            piecewise_parity = validate_piecewise_prefill(engine, boundaries)
            runner.use_prefill_cudagraph = True
            async_parity = validate_async_lookahead(engine, requests)
            output_comparisons = [
                *batches["B1_vs_transformers"].values(),
                *batches["B2"]["requests"].values(),
                *batches["B4"]["requests"].values(),
                *boundaries,
            ]
            checks = {
                "greedy_outputs_match": all(case["greedy_match"] for case in output_comparisons),
                "decode_graph_matches_eager": all(
                    result["greedy_match"] and
                    result["logits"]["argmax_matches"] == result["logits"]["rows"]
                    for batch in graph_parity.values()
                    for result in batch["requests"].values()
                ),
                "slot_reuse_matches_explicit_clear": reuse["same_mixed_layout_after_explicit_clear_exact"],
                "prefill_lengths_exceeding_state_drift_bounds": [
                    case["prompt_tokens"] for case in boundaries if not case["states"]["pass"]
                ],
                "mixed_vs_b1_state_drift_within_bounds": reuse["late_arrival_states"]["pass"],
            }
            report = {
                "revision": REVISION,
                "checkpoint_sha256": LFS_SHA256["model.safetensors"],
                "validation_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "environment": {**env_info(), "transformers_version": transformers.__version__,
                                "triton_version": triton.__version__},
                "reference_environment": {k: v for k, v in reference.items() if k != "cases"},
                "engine": {**settings, "queue_depth": engine.max_concurrent_batches,
                           "enforce_eager": runner.enforce_eager, "async_d2h": runner.async_output,
                           "prefill_chunk_budget": CHUNK_BUDGET, "kv_block_size": runner.block_size,
                           "kv_blocks": engine.config.num_kvcache_blocks,
                           "prefill_graph_buckets": sorted(runner.prefill_piecewise_graphs)},
                "tokenizer": {"class": type(engine.tokenizer).__name__, "eos_id": engine.tokenizer.eos_token_id,
                              "sha256": LFS_SHA256["tokenizer.json"], "add_special_tokens": False},
                "short_requests": {label: {"prompt_ids": prompt, "forced_output_ids": tokens}
                                   for label, (prompt, tokens) in requests.items()},
                "checks": checks, "batches": batches,
                "decode_graph_vs_eager": graph_parity,
                "piecewise_prefill_vs_eager": piecewise_parity,
                "async_lookahead_vs_serial": async_parity,
                "prefill_boundaries": boundaries, "slot_reuse": reuse,
            }
        finally:
            engine.exit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    report["source_sha256"] = {
        str(path.relative_to(repo)): sha256(path.read_bytes()).hexdigest()
        for path in sorted((repo / "src/nano_qwen").rglob("*.py"))
    }
    write_json(args.output, report)
    if checks["prefill_lengths_exceeding_state_drift_bounds"] or not checks["mixed_vs_b1_state_drift_within_bounds"]:
        print("Cross-shape state drift exceeded the Qwen3.5 suite's diagnostic bounds. See report.", flush=True)
    if not checks["greedy_outputs_match"]:
        raise SystemExit("Greedy output differed from the reference. See report.")
    if not checks["decode_graph_matches_eager"]:
        raise SystemExit("MoE decode Graph differed from eager execution. See report.")


if __name__ == "__main__":
    main()
