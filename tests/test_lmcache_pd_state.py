"""CPU tests of production state/engine/scheduler code; no model/CUDA required.

Original classes are AST-loaded to avoid engine/__init__.py eagerly importing
FlashInfer/Triton. Their method bodies are unmodified; only model execution is
replaced by deterministic CPU tensors.
"""
import ast
from collections import deque
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xxhash

from nano_qwen.cache import engine_ops, state
from nano_qwen.sampling_params import SamplingParams

ROOT = Path(__file__).resolve().parents[1]


def load_classes(path, names, env):
    source = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    nodes = [n for n in source.body if isinstance(n, ast.ClassDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), path, "exec"), env)
    return [env[n] for n in names]


spec = importlib.util.spec_from_file_location("pd_test_sequence", ROOT / "src/nano_qwen/engine/sequence.py")
seq_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seq_module)
Sequence, Status = seq_module.Sequence, seq_module.SequenceStatus
env = dict(Sequence=Sequence, SequenceStatus=Status, deque=deque, np=np,
           xxhash=xxhash, SamplingParams=SamplingParams, torch=torch,
           trace_event=lambda *a, **kw: nullcontext())
Block, BlockManager = load_classes("src/nano_qwen/engine/block_manager.py", ["Block", "BlockManager"], env)
InputBatch, = load_classes("src/nano_qwen/engine/model_runner.py", ["InputBatch"], env)
Scheduler, = load_classes("src/nano_qwen/scheduler/scheduler.py", ["Scheduler"], env)
Engine, = load_classes("src/nano_qwen/engine/llm_engine.py", ["LLMEngine"], env)


class Runner:
    world_size, block_size, async_output = 1, 256, True

    def __init__(self, config):
        self.config = config
        self.kv_cache = torch.zeros(2, 2, config.num_kvcache_blocks, 256, 1, 4)
        self.gdn_layers = [SimpleNamespace(conv_states=torch.zeros(8, 2, 3),
                            recurrent_states=torch.zeros(8, 2, 2, 2)) for _ in range(2)]
        self.input_batch = InputBatch(8)
        self.sampled_token_ids_gpu = torch.zeros(8, dtype=torch.int64)
        self._pending, self.prefill_calls, self.decode_calls = None, 0, 0

    def call(self, method, *args):
        if method == "pd_model_identity":
            return "test-model"
        if method == "validate_request_state":
            return state.validate(self, *args)
        if method == "export_request_state":
            return state.capture(self, *args)
        if method == "restore_request_state":
            return state.restore(self, *args)
        if method == "remove_request":
            return self.input_batch.remove(args[0])
        if method == "execute_model":
            seqs, is_prefill = args
            self.prefill_calls += int(is_prefill)
            self.decode_calls += int(not is_prefill)
            _, fresh = self.input_batch.update(seqs)
            for seq, slot in fresh:
                self.sampled_token_ids_gpu[slot] = seq.last_token
                for layer in self.gdn_layers:
                    layer.conv_states[slot].zero_()
                    layer.recurrent_states[slot].zero_()
            outputs = []
            for seq in seqs:
                slot = self.input_batch.seq_id_to_slot[seq.seq_id]
                for pos in range(seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens):
                    token = seq.token_ids[pos] if seq.is_prefill else int(self.sampled_token_ids_gpu[slot])
                    block = seq.block_table[pos // self.block_size]
                    self.kv_cache[:, :, block, pos % self.block_size].fill_(token + pos)
                    for layer in self.gdn_layers:
                        layer.recurrent_states[slot].add_(token + pos)
                        layer.conv_states[slot] = torch.roll(layer.conv_states[slot], -1, -1)
                        layer.conv_states[slot, :, -1] = token
                token = 3 + int(self.gdn_layers[0].recurrent_states[slot, 0, 0, 0]) % 251
                if not seq.is_prefill or seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq):
                    self.sampled_token_ids_gpu[slot] = token
                outputs.append(token)
            self._pending = outputs
            return
        if method == "sample_tokens":
            output, self._pending = self._pending, None
            return SimpleNamespace(get_output=lambda: output)
        raise AssertionError(method)


class Store:
    """Explicit transport fixture only; real LMCache tests are in test_lmcache_pd_store."""
    def __init__(self):
        self.data = {}

    def health(self):
        return {"backend": "test-fixture"}

    def put(self, key, payload):
        self.data[key] = payload
        return dict(committed=True, sha256=hashlib.sha256(payload).hexdigest())

    def get(self, key):
        return self.data[key]

    def delete(self, key):
        del self.data[key]


def engine(blocks=16, budget=128):
    cfg = SimpleNamespace(tensor_parallel_size=1, async_scheduling=False,
        enable_prefix_cache=False, num_kvcache_blocks=blocks, kvcache_block_size=256,
        max_num_batched_tokens=budget, max_num_seqs=8, max_model_len=4096, eos=1)
    e = Engine.__new__(Engine)
    e.config, e.scheduler, e.model_runner = cfg, Scheduler(cfg), Runner(cfg)
    e.batch_queue, e.max_concurrent_batches = deque(), 2
    e._coalesce_armed = False
    e.coalesce_stats = dict(coalesce_attempts=0, coalesce_successes=0,
        merged_batches=0, merged_requests=0, batch_size_before={}, batch_size_after={})
    return e


def finish(e):
    outputs = []
    for _ in range(5000):
        if e.is_finished():
            return outputs
        values, _ = e.step()
        outputs.extend(values)
    raise AssertionError("engine did not drain")


def prefill(e, store, length=257, key="x"):
    params = SamplingParams(temperature=1e-8, max_tokens=6, ignore_eos=True)
    tokens = [3 + i % 17 for i in range(length)]
    engine_ops.prefill_to_cache(e, tokens, params, store, key, Sequence)
    return tokens, params


@pytest.mark.parametrize("length", [1, 127, 255, 256, 257, 511, 512, 513])
def test_handoff_relocates_slots_and_blocks_and_skips_prefill(length):
    p, d, reference, store = engine(), engine(), engine(), Store()
    tokens, params = prefill(p, store, length)
    ref = Sequence(tokens, params)
    reference.scheduler.add(ref)
    expected = finish(reference)[0][1]
    assert p.is_finished() and not p.scheduler.block_manager.used_block_ids
    # Force a different local block and GDN slot without scheduling the guard.
    guard = Sequence([9], params)
    d.scheduler.block_manager.allocate(guard, 0)
    d.model_runner.input_batch.update([guard])
    original_payload = store.get("x")
    seq = engine_ops.resume_from_cache(d, store, "x", Sequence, SamplingParams, Status.RUNNING)
    assert seq.block_table[0] != 0
    assert d.model_runner.input_batch.seq_id_to_slot[seq.seq_id] == 1
    assert seq.num_cached_tokens == length
    assert len(seq.block_table) == (length + 255) // 256
    assert seq.completion_token_ids == expected[:1]
    assert d.model_runner.sampled_token_ids_gpu[1].item() == expected[0]
    assert state.pack(state.capture(d.model_runner, seq, "test-model")) == original_payload
    actual = finish(d)[0][1]
    assert actual == expected
    assert d.model_runner.prefill_calls == 0
    assert d.model_runner.decode_calls == params.max_tokens - 1
    assert not store.data
    assert d.scheduler.block_manager.used_block_ids == set(guard.block_table)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int64])
def test_codec_preserves_exact_tensor_bytes(dtype):
    s = state.Snapshot({"hello": "world"}, {"a": torch.arange(24).reshape(2, 3, 4).to(dtype)})
    restored = state.unpack(state.pack(s))
    assert restored.metadata == s.metadata
    assert torch.equal(restored.tensors["a"], s.tensors["a"])
    assert restored.tensors["a"].dtype == dtype


def test_corruption_is_rejected():
    raw = bytearray(state.pack(state.Snapshot({}, {"a": torch.ones(2)})))
    raw[-40] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        state.unpack(bytes(raw))


@pytest.mark.parametrize("change", ["model", "progress", "dtype", "gdn"])
def test_invalid_snapshot_leaves_destination_unmodified(change):
    p, d, store = engine(), engine(), Store()
    prefill(p, store)
    s = state.unpack(store.data["x"])
    if change == "model": s.metadata["model_id"] = "other"
    if change == "progress": s.metadata["cached_tokens"] -= 1
    if change == "dtype": s.tensors["kv"] = s.tensors["kv"].to(torch.float16)
    if change == "gdn": del s.tensors["gdn.0.recurrent"]
    store.data["x"] = state.pack(s)
    before = d.model_runner.kv_cache.clone()
    with pytest.raises(ValueError):
        engine_ops.resume_from_cache(d, store, "x", Sequence, SamplingParams, Status.RUNNING)
    assert torch.equal(before, d.model_runner.kv_cache)
    assert not d.scheduler.block_manager.used_block_ids
    assert not d.model_runner.input_batch.seq_id_to_slot
    assert "x" in store.data


def test_no_capacity_and_restore_failure_do_not_ack_snapshot():
    p, store = engine(), Store()
    prefill(p, store, length=513)
    d = engine(blocks=1)
    with pytest.raises(MemoryError):
        engine_ops.resume_from_cache(d, store, "x", Sequence, SamplingParams, Status.RUNNING)
    assert "x" in store.data and not d.scheduler.running


def test_store_failure_preserves_producer_request():
    p, store = engine(), Store()
    def fail(*_): raise MemoryError("full")
    store.put = fail
    with pytest.raises(MemoryError): prefill(p, store)
    assert len(p.scheduler.running) == 1
    assert p.scheduler.block_manager.used_block_ids
    assert p.model_runner.input_batch.seq_id_to_slot
    assert len(finish(p)[0][1]) == 6


def test_idle_and_async_guards():
    e = engine()
    e.config.async_scheduling = True
    with pytest.raises(ValueError, match="async_scheduling"):
        prefill(e, Store())
    e.config.async_scheduling = False
    e.scheduler.add(Sequence([3], SamplingParams()))
    with pytest.raises(ValueError, match="idle"):
        prefill(e, Store())


def test_tail_bytes_are_zeroed():
    p, store = engine(), Store()
    p.model_runner.kv_cache.fill_(12345)
    prefill(p, store, length=257)
    s = state.unpack(store.data["x"])
    assert not torch.count_nonzero(s.tensors["kv"][:, :, -1, 1:])


def test_failed_delete_keeps_restored_request_runnable():
    p, d, store = engine(), engine(), Store()
    prefill(p, store)
    def fail(*_): raise RuntimeError("ACK unavailable")
    store.delete = fail
    seq = engine_ops.resume_from_cache(d, store, "x", Sequence, SamplingParams, Status.RUNNING)
    assert seq.pd_handoff["restored"] and not seq.pd_handoff["acknowledged"]
    assert "ack_error" in seq.pd_handoff
    assert len(finish(d)[0][1]) == 6


def test_checkpoint_identity_checks_weight_contents(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"weights-A")
    def runner():
        return SimpleNamespace(config=SimpleNamespace(model=str(tmp_path), quantization=None,
            hf_config=SimpleNamespace(to_dict=lambda: {"hidden_size": 4})))
    first = state.model_identity(runner())
    path.write_bytes(b"weights-B")
    assert state.model_identity(runner()) != first


def test_destination_context_limit_covers_remaining_generation():
    p, d, store = engine(), engine(), Store()
    prefill(p, store, length=257)
    d.config.max_model_len = 260
    with pytest.raises(ValueError, match="max_model_len"):
        engine_ops.resume_from_cache(d, store, "x", Sequence, SamplingParams, Status.RUNNING)
    assert not d.scheduler.block_manager.used_block_ids


def test_restore_dispatch_failure_reclaims_allocated_blocks_and_slot():
    p, d, store = engine(), engine(), Store()
    prefill(p, store)
    original = d.model_runner.call
    def fail_after_copy(method, *args):
        result = original(method, *args)
        if method == "restore_request_state":
            raise RuntimeError("simulated dispatch failure")
        return result
    d.model_runner.call = fail_after_copy
    with pytest.raises(RuntimeError, match="dispatch failure"):
        engine_ops.resume_from_cache(d, store, "x", Sequence, SamplingParams, Status.RUNNING)
    assert not d.scheduler.block_manager.used_block_ids
    assert not d.model_runner.input_batch.seq_id_to_slot
    assert "x" in store.data and d.is_finished()
