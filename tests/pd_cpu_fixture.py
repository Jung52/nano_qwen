"""CPU engine fixture for PD handoff tests; no model/CUDA required.

Original classes are AST-loaded to avoid engine/__init__.py eagerly importing
FlashInfer/Triton. Their method bodies are unmodified; only model execution is
replaced by deterministic CPU tensors.
"""
import ast
from collections import deque
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import xxhash

from nano_qwen.cache import state
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
