# ruff: noqa: F811
"""All live/subset shapes through B8, using production cores and real graph banks."""

from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

import test_mtp_cache_integration as lifecycle
from test_mtp_batch2 import compare_modes
from test_mtp_verify_reuse import tiny_checkpoint, fp8_checkpoint  # noqa: F401
from test_mtp_weights import checkpoint_pair
from nano_qwen.config import Config
from nano_qwen.engine.cuda_graph import decode_capture_sizes
from nano_qwen.engine.sequence import Sequence
from nano_qwen.utils.context import get_context


@pytest.mark.parametrize("batch", range(1, 9))
def test_every_small_batch_capacity_has_fp8_reference_and_exact_decode_graph(tmp_path, batch):
    target, source = checkpoint_pair(tmp_path)
    config = Config(str(target), max_num_seqs=batch, enable_mtp=True, mtp_model=str(source), mtp_verify_mode="reuse")
    assert config.max_num_seqs == batch
    assert decode_capture_sizes(batch) == list(range(1, batch + 1))


@pytest.fixture(scope="module", params=[4, 8], ids=["B4", "B8"])
def capacity(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False], ids=["eager", "graph"])
def engine(request, capacity, tiny_checkpoint, fp8_checkpoint):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    with lifecycle.hybrid_engine(fp8_checkpoint, request.param, mtp_model=str(tiny_checkpoint),
                                 mtp_verify_mode="reuse", max_num_seqs=capacity, cache_blocks=32) as value:
        yield value


@pytest.mark.parametrize("accepted_count", range(9))
@torch.inference_mode()
def test_all_accept_subset_sizes_keep_serial_prefix_state(engine, accepted_count):
    batch = engine.config.max_num_seqs
    if accepted_count > batch:
        pytest.skip("subset exceeds capacity")
    # Multiplication by3 permutes power-of-two slots, exercising noncontiguous masks.
    mask = tuple((i * 3 % batch) < accepted_count for i in range(batch))
    calls = []
    runner = engine.model_runner
    original = runner._run_mtp_decode

    def decode(seqs, *args, **kwargs):
        calls.append((len(seqs), kwargs.get("cache_only", False)))
        return original(seqs, *args, **kwargs)

    with patch.object(runner, "_run_mtp_decode", decode):
        result = compare_modes(engine, [mask], lengths=(63,) * batch, budgets=(7,) * batch, eos=(None,) * batch)
    assert batch in result["batches"]
    if accepted_count:
        assert (accepted_count, True) in calls
    else:
        assert all(not cache for size, cache in calls)
    assert (batch, False) in calls


@torch.inference_mode()
def test_dynamic_endings_eos_new_arrivals_and_slot_reuse(engine):
    batch = engine.config.max_num_seqs
    lengths = tuple(127 + i for i in range(batch)) + (129,)
    budgets = tuple(2 + 2*i for i in range(batch)) + (5,)
    eos = (None,) * (batch - 1) + (3,) + (None,)
    masks = [tuple(i % 2 == 0 for i in range(batch)), tuple(i % 2 != 0 for i in range(batch))]
    result = compare_modes(engine, masks, lengths=lengths, budgets=budgets, eos=eos, late=True)
    assert 1 in result["batches"] and batch in result["batches"]


@torch.inference_mode()
def test_chunked_prefill_with_queued_request_and_slot_reuse(engine):
    batch = engine.config.max_num_seqs
    compare_modes(engine, [tuple(i % 2 == 0 for i in range(batch))],
                  lengths=tuple(129 if i % 2 == 0 else 257 for i in range(batch)) + (127,),
                  budgets=(4,) * batch + (5,), eos=(None,) * (batch + 1),
                  chunk=128, queued=True)


@torch.inference_mode()
def test_every_runtime_live_size_uses_exact_b_math(engine):
    batch = engine.config.max_num_seqs
    seen = {"verify": set(), "dense": set(), "norm": set(), "draft": set()}
    runner = engine.model_runner
    forward = runner.forward_verify_reuse
    decode = runner._run_mtp_decode

    def verify(ids, positions, **kwargs):
        seen["verify"].add(ids.numel() // 2)
        return forward(ids, positions, **kwargs)

    def draft(seqs, *args, **kwargs):
        if not kwargs.get("cache_only"):
            seen["draft"].add(len(seqs))
        return decode(seqs, *args, **kwargs)

    def rows(kind):
        def hook(module, args):
            reference = get_context().gemm_reference_m
            if reference:
                seen[kind].add((reference, args[0].shape[0]))
        return hook

    with ExitStack() as stack:
        stack.enter_context(patch.object(runner, "forward_verify_reuse", verify))
        stack.enter_context(patch.object(runner, "_run_mtp_decode", draft))
        stack.enter_context(runner.model.model.layers[0].mlp.gate_up_proj.register_forward_pre_hook(rows("dense")))
        stack.enter_context(runner.model.model.layers[0].input_layernorm.register_forward_pre_hook(rows("norm")))
        result = compare_modes(engine, [(False,) * batch], lengths=(64,) * batch,
                               budgets=tuple(3 + i*2 for i in range(batch)), eos=(None,) * batch)
    assert set(range(1, batch + 1)).issubset(result["batches"])
    assert set(range(1, batch + 1)).issubset(seen["draft"])
    assert {(b, 2*b) for b in range(1, batch + 1)}.issubset(seen["dense"])
    assert {(b, b) for b in range(1, batch + 1)}.issubset(seen["norm"])
    if not runner.enforce_eager:
        expected = set(range(1, batch + 1))
        assert set(runner.cuda_graphs.verify_graphs) == expected
        assert set(runner.cuda_graphs.mtp_decode_graphs) == expected
        assert set(runner.cuda_graphs.mtp_cache_graphs) == expected


@pytest.mark.parametrize("count", range(1, 9))
@torch.inference_mode()
def test_every_mtp_subset_full_and_cache_only_write_identical_kv(engine, count):
    batch = engine.config.max_num_seqs
    if count > batch:
        pytest.skip("subset exceeds capacity")
    runner = engine.model_runner
    seqs = [Sequence([2] * 140) for _ in range(batch)]
    for i, seq in enumerate(seqs):
        seq.block_table = [2*i + 1]
    runner.input_batch.update(seqs)
    chosen = seqs[::-1][:count]
    slots = runner.input_batch.slots_for(chosen)
    positions = [129 + i for i in range(count)]
    hidden = torch.randn(count, 128, dtype=torch.bfloat16, device="cuda")
    ids = torch.full((count,), 3, dtype=torch.int64, device="cuda")
    runner.mtp_kv_cache.normal_(std=.01)
    before = runner.mtp_kv_cache.clone()
    runner._mtp_cache_lens.update(zip(slots, [p-1 for p in positions]))
    try:
        runner._run_mtp_decode(chosen, hidden, ids, positions)
        full = runner.mtp_kv_cache.clone()
        runner.mtp_kv_cache.copy_(before)
        runner._mtp_cache_lens.update(zip(slots, [p-1 for p in positions]))
        runner._run_mtp_decode(chosen, hidden, ids, positions, cache_only=True)
        assert torch.equal(full, runner.mtp_kv_cache)
    finally:
        for seq in seqs:
            runner.remove_request(seq.seq_id)
