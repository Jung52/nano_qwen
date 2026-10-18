# ruff: noqa: F811
"""Batched restores must be bitwise, slot-local and equivalent over a lifecycle."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

import test_mtp_cache_integration as lifecycle
from test_mtp_batch2 import batch_case
from test_mtp_verify_reuse import fp8_checkpoint  # noqa: F401
from mtp_rollback_reference import legacy_rollback, reference_methods
from nano_qwen.layers.gated_delta_net import GatedDeltaNet


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def state_layer(conv_shape, recurrent_shape, conv_dtype, recurrent_dtype):
    layer = GatedDeltaNet.__new__(GatedDeltaNet)
    nn.Module.__init__(layer)
    layer.conv_states = torch.randn(conv_shape, device="cuda", dtype=conv_dtype)
    layer.recurrent_states = torch.randn(recurrent_shape, device="cuda", dtype=recurrent_dtype)
    layer.allocate_speculative_snapshot()
    layer._speculative_conv_state.normal_()
    layer._speculative_recurrent_state.normal_()
    # Include signed zero: equality of values alone would miss a lossy copy.
    layer._speculative_conv_state.flatten()[0] = -0.0
    layer._speculative_recurrent_state.flatten()[0] = -0.0
    return layer


def bits(tensor):
    return tensor.view(torch.uint8)


@pytest.mark.parametrize("slots", [[], [0], list(range(8)), [6, 1, 4]])
@pytest.mark.parametrize("conv_dtype,recurrent_dtype", [
    (torch.bfloat16, torch.bfloat16), (torch.float16, torch.float32),
])
@pytest.mark.parametrize("large", [False, True], ids=["masked-tail", "4B-layout"])
@torch.inference_mode()
def test_restore_only_selected_slots_without_gather_temporaries(slots, conv_dtype, recurrent_dtype, large):
    conv_shape = (8, 8192, 3) if large else (8, 7, 3)
    rec_shape = (8, 32, 128, 128) if large else (8, 2, 37, 19)
    layer = state_layer(conv_shape, rec_shape, conv_dtype, recurrent_dtype)
    expected = [layer.conv_states.clone(), layer.recurrent_states.clone()]
    saved = [layer._speculative_conv_state.clone(), layer._speculative_recurrent_state.clone()]
    for slot in slots:
        for dst, src in zip(expected, saved):
            dst[slot].copy_(src[slot])
    indices = torch.tensor(slots, device="cuda", dtype=torch.int64)
    with patch.object(torch.Tensor, "index_select", side_effect=AssertionError("rollback must copy directly by slot")):
        layer.rollback_speculative_states(indices)
    for actual, want in zip((layer.conv_states, layer.recurrent_states), expected):
        assert torch.equal(bits(actual), bits(want))
    for actual, want in zip((layer._speculative_conv_state, layer._speculative_recurrent_state), saved):
        assert torch.equal(bits(actual), bits(want)), "rollback overwrote its snapshot"


@torch.inference_mode()
def test_graph_replay_reads_current_slot_indices_and_snapshot():
    layer = state_layer((8, 7, 3), (8, 2, 37, 19), torch.bfloat16, torch.float32)
    indices = torch.tensor([6, 1, 4], device="cuda")
    layer.rollback_speculative_states(indices)  # JIT before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer.rollback_speculative_states(indices)
    for slots in ([6, 1, 4], [7, 0, 2]):
        layer.conv_states.fill_(3)
        layer.recurrent_states.fill_(5)
        layer._speculative_conv_state.normal_()
        layer._speculative_recurrent_state.normal_()
        expected = [layer.conv_states.clone(), layer.recurrent_states.clone()]
        for slot in slots:
            expected[0][slot].copy_(layer._speculative_conv_state[slot])
            expected[1][slot].copy_(layer._speculative_recurrent_state[slot])
        indices.copy_(torch.tensor(slots, device="cuda"))
        graph.replay()
        assert torch.equal(bits(layer.conv_states), bits(expected[0]))
        assert torch.equal(bits(layer.recurrent_states), bits(expected[1]))


@pytest.fixture(scope="module", params=[True, False], ids=["eager", "graph"])
def rollback_engine(request, tiny_checkpoint, fp8_checkpoint):
    with lifecycle.hybrid_engine(fp8_checkpoint, request.param, mtp_model=str(tiny_checkpoint),
                                 mtp_verify_mode="reuse", max_num_seqs=4, cache_blocks=32) as engine:
        yield engine


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    return lifecycle.build_checkpoint(tmp_path_factory.mktemp("mtp-rollback"), head_dim=128,
                                      layer_types=["linear_attention", "linear_attention", "full_attention"])


@pytest.fixture(scope="module")
def old_methods():
    return reference_methods(Path(__file__).resolve().parents[1])[0]


@pytest.mark.parametrize("case", ["reject-all", "accept-all", "mixed", "eos-reuse"])
@torch.inference_mode()
def test_old_and_batched_lifecycles_are_bitwise_identical(rollback_engine, old_methods, case):
    runner = rollback_engine.model_runner
    masks = {
        "reject-all": [(False,) * 4],
        "accept-all": [(True,) * 4],
        "mixed": [(True, False, True, False), (False, True, False, True)],
        "eos-reuse": [(True, False, True, False), (False, True, False, True)],
    }[case]
    kwargs = dict(lengths=(127, 129, 255, 257), budgets=(7,) * 4, eos=(None,) * 4)
    if case == "eos-reuse":
        kwargs = dict(lengths=(127, 129, 255, 257, 128), budgets=(9, 9, 9, 9, 5),
                      eos=(1, None, 2, None, None), late=True)
    results = []
    for legacy in (True, False):
        with legacy_rollback(runner, old_methods) if legacy else nullcontext():
            results.append(batch_case(rollback_engine, masks, **kwargs))
    old, new = results
    assert old["tokens"] == new["tokens"]
    assert old["decisions"] == new["decisions"]
    assert old["batches"] == new["batches"]
    assert old["snapshots"].keys() == new["snapshots"].keys()
    for key, (count, tensors) in old["snapshots"].items():
        new_count, actual = new["snapshots"][key]
        assert new_count == count
        for expected, got in zip(tensors, actual):
            assert torch.isfinite(got).all(), (case, key, "non-finite state/cache")
            assert torch.equal(bits(expected), bits(got)), key
    for kind in ("hidden", "logits"):
        assert old[kind].keys() == new[kind].keys()
        for key in old[kind]:
            assert torch.isfinite(new[kind][key]).all(), (case, kind, key)
            assert torch.equal(bits(old[kind][key]), bits(new[kind][key])), (kind, key)


@torch.inference_mode()
def test_runner_shares_one_slot_index_across_layers_and_includes_accepted_eos(rollback_engine):
    runner = rollback_engine.model_runner
    assert len(runner.gdn_layers) == 2
    observed = []
    original = GatedDeltaNet.rollback_speculative_states

    def restore(layer, indices):
        observed.append((layer.layer_idx, indices.data_ptr(), tuple(indices.tolist())))
        return original(layer, indices)

    # A accepts EOS, B rejects, C accepts a continuing draft, D accepts bonus EOS.
    with patch.object(GatedDeltaNet, "rollback_speculative_states", restore), \
         patch.object(GatedDeltaNet, "rollback_speculative_state", side_effect=AssertionError("per-request rollback")):
        batch_case(rollback_engine, [(True, False, True, True)], lengths=(64,) * 4,
                   budgets=(5,) * 4, eos=(1, None, None, 2))
    first = observed[:len(runner.gdn_layers)]
    assert first[0][2] == (0, 1), "accepted draft EOS and rejection must share a restore; bonus EOS needs no restore"
    assert len({row[1] for row in first}) == 1
    assert len(first) == len(runner.gdn_layers)
