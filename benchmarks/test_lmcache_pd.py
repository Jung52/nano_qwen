"""Real-model validation of LMCache CPU-RAM request handoff.

Run --mode store-smoke first (no GPU/model required), then roundtrip on one
GPU. The prefill/decode modes can run sequentially on one GPU or on different
GPUs as independent processes connected to one cache service.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
import traceback
import uuid

from nano_qwen.cache.connector import LMCacheConnector

BASE_COMMIT = "7049b3b5f54f9a13ceab2bc26914a4cf792854c8"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def complete(engine, sequence):
    for _ in range(sequence.max_tokens + 2):
        if engine.is_finished():
            return list(sequence.completion_token_ids)
        engine.step()
    raise RuntimeError("decode did not finish within the token bound")


def reserve_guard(engine, params):
    from nano_qwen.engine.sequence import Sequence, SequenceStatus
    guard = Sequence([3], params)
    guard.status = SequenceStatus.RUNNING
    engine.scheduler.block_manager.allocate(guard, 0)
    engine.model_runner.input_batch.remove_finished()
    engine.model_runner.input_batch.update([guard])
    return guard


def release_guard(engine, guard):
    engine.model_runner.call("remove_request", guard.seq_id)
    engine.scheduler.block_manager.deallocate(guard)


def prompt_tokens(engine, length):
    seed = engine.tokenizer.encode("Explain how a computer stores information in memory. ",
                                   add_special_tokens=False)
    if not seed:
        raise RuntimeError("tokenizer produced no prompt tokens")
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def run(args, result):
    connector = LMCacheConnector(args.cache_url, timeout=args.timeout)
    result["cache"] = connector.health()
    if args.mode == "store-smoke":
        key = args.transfer_id or "smoke-" + uuid.uuid4().hex
        payload = bytes(range(256)) * 4096
        before = connector.health()
        receipt = connector.put(key, payload)
        after = connector.health()
        assert connector.get(key) == payload, "cache bytes changed"
        connector.delete(key)
        assert after["used_bytes"] > before["used_bytes"], "no real L1 allocation"
        result["cases"].append(dict(receipt, passed=True, real_l1_allocation=True))
        return
    if not args.model:
        raise ValueError("--model is required for real-model tests")
    import torch
    from nano_qwen.engine.llm_engine import LLMEngine
    from nano_qwen.sampling_params import SamplingParams
    from nano_qwen.cache.state import capture, pack, unpack

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the real-model modes")
    engine = LLMEngine(args.model, enforce_eager=not args.cuda_graph,
        use_prefill_cudagraph=args.cuda_graph, max_num_seqs=4,
        max_num_batched_tokens=args.token_budget, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1, async_scheduling=False, enable_prefix_cache=False)
    params = SamplingParams(temperature=1e-8, max_tokens=args.output_len, ignore_eos=True)
    try:
        identity = engine.model_runner.call("pd_model_identity")
        result["environment"] = dict(torch=str(torch.__version__),
            gpu=torch.cuda.get_device_name(), model_id=identity,
            cuda_graph=args.cuda_graph, base_commit=BASE_COMMIT)
        if args.mode == "decode":
            if not args.transfer_id:
                raise ValueError("decode requires --transfer-id")
            keys = [(args.transfer_id, None)]
        else:
            if args.mode == "prefill" and len(args.prompt_lens) != 1:
                raise ValueError("prefill mode requires exactly one --prompt-lens value")
            if args.transfer_id and len(args.prompt_lens) != 1:
                raise ValueError("an explicit transfer ID requires one prompt length")
            keys = [(args.transfer_id or "pd-" + uuid.uuid4().hex, length)
                    for length in args.prompt_lens]
        for key, length in keys:
            row = dict(transfer_id=key, prompt_tokens=length, passed=False)
            result["cases"].append(row)
            expected = None
            if args.mode != "decode":
                tokens = prompt_tokens(engine, length)
                if args.mode == "roundtrip" or args.with_reference:
                    t = perf_counter()
                    expected = engine.generate([tokens], params, use_tqdm=False)[0]["token_ids"]
                    row["reference_e2e_ms"] = (perf_counter() - t) * 1000
                    row["reference_tokens"] = list(expected)
                t = perf_counter()
                receipt = engine.prefill_to_lmcache(tokens, params, connector, key)
                row.update(receipt)
                row["prefill_store_ms"] = (perf_counter() - t) * 1000
            if args.mode == "prefill":
                row["passed"] = True
                continue
            raw = connector.get(key)
            snapshot = unpack(raw)
            row["prompt_tokens"] = snapshot.metadata["prompt_len"]
            # Keep one local block/slot busy to exercise relocation.
            guard = reserve_guard(engine, params)
            try:
                t = perf_counter()
                seq = engine.resume_from_lmcache(connector, key, delete_after_restore=False)
                row["restore_ms"] = (perf_counter() - t) * 1000
                slot = engine.model_runner.input_batch.seq_id_to_slot[seq.seq_id]
                row.update(destination_slot=slot, destination_blocks=list(seq.block_table))
                restored = capture(engine.model_runner, seq, identity)
                assert pack(restored) == raw, "KV/GDN/progress changed during roundtrip"
                row["state_bytes_equal"] = True
                connector.delete(key)
                decode_stats = dict(prefill_dispatches=0, decode_dispatches=0)
                original_execute = engine.model_runner.execute_model
                def tracked(seqs, is_prefill):
                    decode_stats["prefill_dispatches" if is_prefill else "decode_dispatches"] += 1
                    return original_execute(seqs, is_prefill)
                engine.model_runner.execute_model = tracked
                try:
                    t = perf_counter()
                    actual = complete(engine, seq)
                    row["decode_ms"] = (perf_counter() - t) * 1000
                finally:
                    engine.model_runner.execute_model = original_execute
                row.update(decode_stats)
                assert row["prefill_dispatches"] == 0, "decoder recomputed the prompt"
                assert row["decode_dispatches"] == seq.max_tokens - 1
                row["output_tokens"] = actual
                row["tpot_ms"] = row["decode_ms"] / max(1, len(actual) - 1)
                if args.reference_json:
                    ref = json.loads(Path(args.reference_json).read_text())
                    expected = next(c["reference_tokens"] for c in ref["cases"]
                                    if c["transfer_id"] == key)
                if expected is not None:
                    row["tokens_equal"] = actual == expected
                    assert row["tokens_equal"], "greedy output differs from uninterrupted reference"
                row["passed"] = True
            finally:
                release_guard(engine, guard)
    finally:
        engine.exit()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["store-smoke", "roundtrip", "prefill", "decode"], default="roundtrip")
    p.add_argument("--model")
    p.add_argument("--cache-url", default="http://127.0.0.1:8765")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--prompt-lens", nargs="+", type=int, default=[64, 256, 257, 512])
    p.add_argument("--output-len", type=int, default=32)
    p.add_argument("--token-budget", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--transfer-id")
    p.add_argument("--with-reference", action="store_true")
    p.add_argument("--reference-json")
    p.add_argument("--output", default="logs/lmcache_pd/results.json")
    args = p.parse_args()
    if args.output_len < 2 or any(n < 1 for n in args.prompt_lens):
        p.error("output-len must be >=2 and prompt lengths must be positive")
    result = dict(mode=args.mode, cases=[], passed=False,
                  transport="binary HTTP via CPU RAM; not RDMA/zero-copy")
    try:
        run(args, result)
        result["passed"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output, result)
        print(json.dumps(dict(passed=result["passed"], output=args.output,
                              cases=len(result["cases"]))), flush=True)


if __name__ == "__main__":
    main()
