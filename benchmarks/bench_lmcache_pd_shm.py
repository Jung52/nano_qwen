"""Direct GPU <-> LMCache L1 shared tensors. HTTP carries metadata only."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import sys
from time import perf_counter_ns
import traceback
from unittest.mock import patch
import uuid

import pd_common as common

ROOT = Path(__file__).resolve().parents[1]
ATOMIC_STAGES = ("prefill_ms", "describe_ms", "reserve_write_ms", "d2h_ms", "commit_ms",
                 "producer_release_ms", "health_ms", "identity_ms", "acquire_ms",
                 "validate_ms", "allocate_gpu_ms", "h2d_ms", "ack_ms", "decode_ms")


def no_payload_codec(*args, **kwargs):
    raise RuntimeError("direct tensor path unexpectedly called the payload codec")


def codec_guard(ops, state):
    stack = ExitStack()
    for module in (ops, state):
        for name in ("pack", "unpack"):
            if hasattr(module, name):
                raise RuntimeError(f"payload codec {module}.{name} still present")
            stack.enter_context(patch.object(module, name, no_payload_codec, create=True))
    return stack


def complete(engine, seq, synchronize):
    calls = dict(prefill=0, decode=0)
    original = engine.model_runner.call
    def tracked(method, *args):
        if method == "execute_model":
            calls["prefill" if args[1] else "decode"] += 1
        return original(method, *args)
    with patch.object(engine.model_runner, "call", tracked):
        started = perf_counter_ns()
        common.complete(engine, seq)
        synchronize()
        elapsed = (perf_counter_ns() - started) / 1e6
    if calls != dict(prefill=0, decode=seq.max_tokens - 1):
        raise RuntimeError(f"D recomputed prefill or wrong dispatch count: {calls}")
    tokens = list(seq.completion_token_ids)
    if len(tokens) != seq.max_tokens:
        raise RuntimeError("incorrect output-token count")
    return tokens, elapsed, calls


def reserve_guard(engine, sequence_type, params):
    guard = sequence_type([3], params)
    engine.scheduler.block_manager.allocate(guard, 0)
    engine.model_runner.input_batch.update([guard])
    return guard


def release_guard(engine, guard):
    engine.model_runner.call("remove_request", guard.seq_id)
    engine.scheduler.block_manager.deallocate(guard)


def handoff(engine, tokens, params, connector, ops, state, sequence_type,
            synchronize, verify_state=False):
    common.require_clean(engine, ops)
    key, guard = "shm-" + uuid.uuid4().hex, None
    committed = False
    before_sent, before_received = connector.control_sent_bytes, connector.control_received_bytes
    with codec_guard(ops, state):
        synchronize()
        start = perf_counter_ns()
        try:
            receipt = engine.prefill_to_lmcache(tokens, params, connector, key)
            committed = True
            expected_state, expected_metadata = None, None
            if verify_state:
                # Correctness-only copies. This mode reports no valid E2E/perf sample.
                lease = connector.acquire(key)
                try:
                    expected_metadata = dict(lease.metadata)
                    expected_state = {name: value.clone() for name, value in lease.tensors.items()}
                finally:
                    connector.release(lease, delete=False)
                guard = reserve_guard(engine, sequence_type, params)
            seq = engine.resume_from_lmcache(connector, key)
            restored_at = perf_counter_ns()
            slot = engine.model_runner.input_batch.seq_id_to_slot[seq.seq_id]
            blocks = list(seq.block_table)
            if verify_state:
                restored = state.capture(engine.model_runner, seq, receipt["model_id"])
                if restored.metadata != expected_metadata:
                    raise RuntimeError("request metadata differs after restore")
                for name, source in expected_state.items():
                    import torch
                    if not torch.equal(source.reshape(-1).view(torch.uint8),
                                       restored.tensors[name].reshape(-1).view(torch.uint8)):
                        raise RuntimeError(f"state bytes differ: {name}")
            output, decode_ms, calls = complete(engine, seq, synchronize)
            end = perf_counter_ns()
            if not seq.pd_handoff["acknowledged"]:
                lease = seq.pd_handoff.get("lease")
                if lease is not None:
                    connector.release(lease, delete=True)
                raise RuntimeError(f"ACK failed during measured restore: {seq.pd_handoff.get('ack_error')}")
            committed = False  # Successful ACK already deleted the resident object.
            metrics = dict(receipt["pd_metrics"])
            for name, value in seq.pd_handoff["metrics"].items():
                metrics[name] = metrics.get(name, 0.0) + value
            metrics.update(decode_ms=decode_ms, tpot_ms=decode_ms / (params.max_tokens - 1))
            if not verify_state:
                metrics["e2e_ms"] = (end - start) / 1e6
                metrics["handoff_pause_ms"] = (restored_at - receipt["first_token_at_ns"]) / 1e6
                residual = metrics["e2e_ms"] - sum(metrics.get(k, 0.0) for k in ATOMIC_STAGES)
                if residual < -0.01:
                    raise RuntimeError("overlapping native tensor stage timers")
                metrics["other_ms"] = max(0.0, residual)
            result = dict(metrics=metrics, backend=receipt["backend"], nbytes=receipt["nbytes"],
                transfer_id=key, committed=True, acknowledged=True, destination_slot=slot,
                destination_blocks=blocks, source_slot=receipt["source_slot"], source_blocks=receipt["source_blocks"],
                state_bytes_equal=True if verify_state else None, dispatches=calls,
                performance_valid=not verify_state, payload_pack_calls=0, payload_unpack_calls=0,
                payload_http_bytes=0, payload_codec_guard=True,
                control_sent_bytes=connector.control_sent_bytes - before_sent,
                control_received_bytes=connector.control_received_bytes - before_received)
            return result, output
        finally:
            if guard is not None:
                release_guard(engine, guard)
            # Native API itself preserves live GPU requests on store/restore failure.
            # Delete only READY entries; live reader/writer leases are never recycled here.
            if committed:
                try:
                    connector.delete(key)
                except Exception:
                    pass


def run_pair(engine, tokens, params, connector, ops, state, sequence_type,
             synchronize, order, verify_state=False, save_tokens=False):
    entries, outputs = {}, {}
    for name in order:
        if name == "baseline":
            entries[name], outputs[name] = common.baseline(engine, tokens, params, sequence_type, ops, synchronize)
        else:
            entries[name], outputs[name] = handoff(engine, tokens, params, connector, ops, state,
                                                  sequence_type, synchronize, verify_state)
    if outputs["baseline"] != outputs["handoff"]:
        raise RuntimeError("native handoff output differs from uninterrupted baseline")
    common.require_clean(engine, ops)
    row = dict(prompt_tokens=len(tokens), output_len=params.max_tokens, passed=True,
               tokens_equal=True, order=list(order), **entries)
    row["output_sha256"] = hashlib.sha256(json.dumps(outputs["baseline"]).encode()).hexdigest()
    if save_tokens:
        row["output_tokens"] = outputs["baseline"]
    return row


def write_summary(path, summary):
    lines = ["# LMCache direct shared-tensor PD", "",
        "B=1; same resident engine; no tensor payload serialization/HTTP upload/download.", "",
        "| Input | Output | N | Baseline E2E | Direct E2E | Paired extra E2E | Baseline TPOT | Direct TPOT | Pause |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        b, h = s["baseline"], s["handoff"]
        lines.append(f"| {s['prompt_tokens']} | {s['output_len']} | {s['pairs']} | "
            f"{b['e2e_ms']['median']:.3f} | {h['e2e_ms']['median']:.3f} | "
            f"{s['paired_e2e_delta_ms']['median']:.3f} | {b['tpot_ms']['median']:.3f} | "
            f"{h['tpot_ms']['median']:.3f} | {h['handoff_pause_ms']['median']:.3f} |")
    lines += ["", "| Input | Output | Reserve/map | D2H/export | Commit | Acquire/map | H2D/restore | ACK |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        h = s["handoff"]
        keys = ("reserve_write_ms", "d2h_ms", "commit_ms", "acquire_ms", "h2d_ms", "ack_ms")
        lines.append(f"| {s['prompt_tokens']} | {s['output_len']} | " +
                     " | ".join(f"{h[k]['median']:.3f}" for k in keys) + " |")
    lines += ["", "All times in ms; each cell is a median. Metadata/control JSON remains.",
              "D2H includes GPU page gathering/tail clearing; H2D includes GPU slot setup/scatter/synchronization.",
              "Pageable mapped CPU memory is supported with blocking copies; this is not a pure PCIe bandwidth test.", ""]
    Path(path).write_text("\n".join(lines))


def smoke(connector):
    import torch
    initial = connector.health()
    key = "smoke-" + uuid.uuid4().hex
    lease = connector.allocate(key, dict(test="direct_l1"),
                               [dict(name="kv", shape=[1 << 20], dtype="uint8")])
    source = torch.arange(256, dtype=torch.uint8).repeat(4096)
    lease.tensors["kv"].copy_(source)
    allocated = connector.health()
    receipt = connector.commit(lease)
    reader = connector.acquire(key)
    try:
        if not torch.equal(reader.tensors["kv"], source):
            raise RuntimeError("shared L1 tensor data changed")
    finally:
        connector.release(reader, delete=True)
    final = connector.health()
    if not allocated["used_bytes"] > initial["used_bytes"] or final["used_bytes"] != initial["used_bytes"]:
        raise RuntimeError("L1 allocation/cleanup failed")
    return dict(receipt, passed=True, real_l1_allocation=True, payload_http_bytes=0,
                payload_pack_calls=0, payload_unpack_calls=0,
                control_sent_bytes=connector.control_sent_bytes,
                control_received_bytes=connector.control_received_bytes)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["store-smoke", "roundtrip", "perf", "prefill", "decode"], default="perf")
    p.add_argument("--model")
    p.add_argument("--cache-url", default="http://127.0.0.1:8766")
    p.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prompt-lens", nargs="+", type=int, default=[64, 256, 257, 512, 1024])
    p.add_argument("--output-lens", nargs="+", type=int, default=[128, 512])
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--token-budget", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--transfer-id")
    p.add_argument("--with-reference", action="store_true")
    p.add_argument("--reference-json")
    p.add_argument("--save-tokens", action="store_true")
    p.add_argument("--output", default="logs/lmcache_pd/perf_shm_graph.json")
    a = p.parse_args(argv)
    if (a.warmup_rounds < 1 or a.repeats < 1 or a.token_budget < 1 or a.timeout <= 0
            or any(n < 1 for n in a.prompt_lens) or any(n < 2 for n in a.output_lens)
            or len(set(a.prompt_lens)) != len(a.prompt_lens) or len(set(a.output_lens)) != len(a.output_lens)
            or not 0 < a.gpu_memory_utilization < 1 or Path(a.output).suffix != ".json"):
        p.error("invalid lengths/repetitions/budget/memory/timeout/output")
    if a.mode not in ("decode", "store-smoke") and max(a.prompt_lens) + max(a.output_lens) > a.max_model_len:
        p.error("prompt+output exceeds max-model-len")
    if a.mode == "prefill" and (len(a.prompt_lens) != 1 or len(a.output_lens) != 1 or not a.transfer_id):
        p.error("prefill requires one input length, one output length, and --transfer-id")
    if a.mode == "decode" and not a.transfer_id:
        p.error("decode requires --transfer-id")
    return a


def run(args, result, checkpoint):
    from nano_qwen.cache.shm_connector import LMCacheShmConnector
    connector = LMCacheShmConnector(args.cache_url, args.timeout)
    engine = None
    try:
        result["cache_before"] = connector.health()
        if args.mode == "store-smoke":
            result["cases"].append(smoke(connector))
            result["cache_after"] = connector.health()
            return
        import torch
        if not args.model or not torch.cuda.is_available():
            raise RuntimeError("GPU/model required; store-smoke and CPU regression need no GPU")
        from nano_qwen.cache import engine_ops, state
        from nano_qwen.engine.llm_engine import LLMEngine
        from nano_qwen.engine.sequence import Sequence
        from nano_qwen.sampling_params import SamplingParams
        from nano_qwen.quantization.checkpoint import read_quantization_metadata, read_hf_fp8_config
        result["module_origins"] = common.verify_origins([
            engine_ops, state, importlib.import_module("nano_qwen.cache.shm_connector"),
            importlib.import_module("nano_qwen.cache.tensor_state"),
            importlib.import_module("nano_qwen.cache.tensor_ops"),
            importlib.import_module(LLMEngine.__module__)], ROOT)
        engine = LLMEngine(args.model, enforce_eager=not args.cuda_graph,
            use_prefill_cudagraph=args.cuda_graph, max_num_seqs=4,
            max_num_batched_tokens=args.token_budget, max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=1, async_scheduling=False, enable_prefix_cache=False)
        if args.mode != "decode" and max(args.prompt_lens) + max(args.output_lens) > engine.config.max_model_len:
            raise ValueError("model's effective max length is too small")
        identity = engine.model_runner.call("pd_model_identity")
        result["environment"] = dict(torch=str(torch.__version__), torch_cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), model=str(Path(args.model).resolve()), model_id=identity,
            quantization=engine.config.quantization,
            quantization_metadata=read_quantization_metadata(args.model) or read_hf_fp8_config(args.model),
            cuda_graph=args.cuda_graph, kv_cache_dtype=str(engine.model_runner.kv_cache.dtype),
            cpu_mapping_pinned=False, copy_mode="blocking; no assumed CUDA registration in inference process")
        seed = engine.tokenizer.encode("Explain how a computer stores information in memory. ", add_special_tokens=False)
        if not seed:
            raise RuntimeError("empty prompt seed")
        prompts = {n: (seed * ((n + len(seed) - 1) // len(seed)))[:n] for n in args.prompt_lens}
        if args.mode == "perf":
            for warm, repetition, length, output, order in common.schedule(
                    args.prompt_lens, args.output_lens, args.warmup_rounds, args.repeats, args.seed):
                row = dict(prompt_tokens=length, output_len=output, warmup=warm, repetition=repetition, passed=False)
                result["warmup" if warm else "samples"].append(row)
                params = SamplingParams(temperature=1e-8, max_tokens=output, ignore_eos=True)
                row.update(run_pair(engine, prompts[length], params, connector, engine_ops, state, Sequence,
                                    torch.cuda.synchronize, order, save_tokens=args.save_tokens))
                checkpoint()
                print(json.dumps(dict(warmup=warm, repeat=repetition, input=length, output=output,
                    baseline_e2e_ms=row["baseline"]["metrics"]["e2e_ms"],
                    direct_e2e_ms=row["handoff"]["metrics"]["e2e_ms"])), flush=True)
        elif args.mode == "roundtrip":
            for length in args.prompt_lens:
                for output in args.output_lens:
                    params = SamplingParams(temperature=1e-8, max_tokens=output, ignore_eos=True)
                    result["cases"].append(run_pair(engine, prompts[length], params, connector, engine_ops,
                        state, Sequence, torch.cuda.synchronize, ("baseline", "handoff"),
                        verify_state=True, save_tokens=True))
                    checkpoint()
        elif args.mode == "prefill":
            length, output = args.prompt_lens[0], args.output_lens[0]
            params = SamplingParams(temperature=1e-8, max_tokens=output, ignore_eos=True)
            row = dict(transfer_id=args.transfer_id)
            if args.with_reference:
                row["reference_tokens"] = engine.generate([prompts[length]], params, use_tqdm=False)[0]["token_ids"]
            with codec_guard(engine_ops, state):
                row.update(engine.prefill_to_lmcache(prompts[length], params, connector, args.transfer_id))
            row.update(passed=True, performance_valid=False)
            result["cases"].append(row)
        else:
            with codec_guard(engine_ops, state):
                seq = engine.resume_from_lmcache(connector, args.transfer_id)
                slot = engine.model_runner.input_batch.seq_id_to_slot[seq.seq_id]
                output, elapsed, calls = complete(engine, seq, torch.cuda.synchronize)
            row = dict(transfer_id=args.transfer_id, output_tokens=output, decode_ms=elapsed,
                destination_slot=slot, decode_dispatches=calls["decode"], prefill_dispatches=calls["prefill"],
                pd_handoff={k: v for k, v in seq.pd_handoff.items() if k != "lease"},
                passed=bool(seq.pd_handoff["acknowledged"]), performance_valid=False)
            if args.reference_json:
                ref = json.loads(Path(args.reference_json).read_text())
                expected = next(x["reference_tokens"] for x in ref["cases"] if x["transfer_id"] == args.transfer_id)
                row["tokens_equal"] = output == expected
                if not row["tokens_equal"]:
                    raise RuntimeError("separate-process output mismatch")
            if not row["passed"]:
                raise RuntimeError("decode restore ACK failed")
            result["cases"].append(row)
        result["cache_after"] = connector.health()
        if args.mode in ("perf", "roundtrip") and any(
                result["cache_after"][key] != result["cache_before"][key] for key in ("used_bytes", "payload_objects")):
            raise RuntimeError("cache occupancy changed; use a dedicated service and check leases")
    finally:
        active_error = sys.exc_info()[0] is not None
        if engine is not None:
            engine.exit()
        try:
            connector.close()
        except RuntimeError as exc:
            if not active_error:
                raise
            result["cleanup_error"] = str(exc)


def main(argv=None):
    args = arguments(argv)
    output = Path(args.output).resolve()
    hashes = common.source_hashes(ROOT)
    hashes[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = dict(schema="nano_qwen_lmcache_pd_shm_v3", mode=args.mode, passed=False,
        args=vars(args), started_at=common.now_iso(), command=shlex.join([sys.executable, *sys.argv]),
        git_before=common.git_provenance(ROOT), source_sha256_before=hashes,
        topology="same-host P/D through typed buffers in LMCache-owned POSIX SHM; JSON control only",
        payload_pack_calls=0, payload_unpack_calls=0, payload_http_bytes=0,
        performance_valid=args.mode == "perf", warmup=[], samples=[], cases=[])
    def checkpoint():
        result["summary"] = common.summarize(result["samples"])
        common.atomic_json(output, result)
    try:
        run(args, result, checkpoint)
        result["passed"] = True
    except (Exception, KeyboardInterrupt):
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["finished_at"] = common.now_iso()
        after = common.source_hashes(ROOT)
        after[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["source_sha256_after"] = after
        result["source_changed_during_run"] = after != hashes
        result["git_after"] = common.git_provenance(ROOT)
        if after != hashes:
            result["passed"] = False
        checkpoint()
        write_summary(output.with_suffix(".md"), result["summary"])
        print(json.dumps(dict(passed=result["passed"], output=str(output))), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
