"""Indexed recurrent updates preserve dense math and avoid state copies."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoConfig

from nano_qwen.engine.cuda_graph import CudaGraphManager
from nano_qwen.layers.gated_delta_net import GatedDeltaNet, decode_gated_delta_rule
from nano_qwen.utils.context import get_context, reset_context, set_context


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("batch", [8, 64])
@torch.inference_mode()
def test_indexed_decode_graph_matches_dense_with_changing_sparse_slots(batch):
    torch.manual_seed(195)
    heads, dim, pool_size = 32, 128, batch * 2 + 3
    q, k, v = [torch.randn((batch, 1, heads, dim), device="cuda", dtype=torch.bfloat16) * .1
               for _ in range(3)]
    a, b = [torch.randn((batch, 1, heads), device="cuda", dtype=torch.bfloat16) * .1
            for _ in range(2)]
    a_log = torch.zeros(heads, device="cuda", dtype=torch.float32)
    bias = torch.zeros(heads, device="cuda", dtype=torch.bfloat16)
    pool = torch.randn((pool_size, heads, dim, dim), device="cuda", dtype=torch.bfloat16) * .01
    initial = pool.clone()
    reference = pool.clone()
    slots = torch.arange(0, batch * 2, 2, device="cuda", dtype=torch.int64).flip(0)

    def indexed():
        return decode_gated_delta_rule(q, k, v, a, b, a_log, bias, pool, state_indices=slots)

    # Warm the exact indexed backend before graph capture.
    _, returned = indexed()
    assert returned.data_ptr() == pool.data_ptr()
    pool.copy_(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, returned = indexed()
    pool.copy_(initial)
    for step in range(4):
        # Includes slot zero and slots beyond B; change indices without recapture.
        current = torch.arange(step % 2, batch * 2, 2, device="cuda", dtype=torch.int64).roll(step)
        slots.copy_(current)
        dense = reference.index_select(0, current)
        expected, updated = decode_gated_delta_rule(q, k, v, a, b, a_log, bias, dense)
        reference.index_copy_(0, current, updated)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Compare the entire pool: untouched slots must also stay identical.
        torch.testing.assert_close(pool, reference, rtol=0, atol=0)


@torch.inference_mode()
def test_gdn_forward_does_not_gather_or_scatter_recurrent_pool():
    config = AutoConfig.for_model("qwen3_5_text")
    config.hidden_size = 128
    config.linear_num_key_heads = 1
    config.linear_num_value_heads = 2
    config.linear_key_head_dim = config.linear_value_head_dim = 128
    with patch.object(dist, "get_rank", return_value=0), patch.object(dist, "get_world_size", return_value=1):
        layer = GatedDeltaNet(config, 0).to(device="cuda", dtype=torch.bfloat16)
    for parameter in layer.parameters():
        parameter.normal_(std=.02)
    layer.allocate_state_pool(7)
    layer.recurrent_states.normal_(std=.01)
    initial = layer.recurrent_states.clone()
    slots = torch.tensor([5, 0], device="cuda")
    x = torch.randn((2, 1, 128), device="cuda", dtype=torch.bfloat16)

    class NoRecurrentCopies(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.index_select.default, torch.ops.aten.index_copy_.default):
                assert args[0] is not layer.recurrent_states, "Recurrent state must be accessed inside the GDN kernel"
            return func(*args, **(kwargs or {}))

    try:
        set_context(False, state_indices=slots)
        with NoRecurrentCopies():
            result = layer(x)
        assert torch.isfinite(result).all()
        torch.testing.assert_close(layer.recurrent_states[[1, 2, 3, 4, 6]], initial[[1, 2, 3, 4, 6]], rtol=0, atol=0)
        assert not torch.equal(layer.recurrent_states[slots], initial[slots])
        other_slot = layer.recurrent_states[0].clone()
        layer.reset_state([5])
        assert torch.count_nonzero(layer.recurrent_states[5]) == 0
        torch.testing.assert_close(layer.recurrent_states[0], other_slot, rtol=0, atol=0)
    finally:
        reset_context()


@torch.inference_mode()
def test_decode_capture_warms_independent_state_slots():
    class SlotAccumulator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.state = torch.zeros(16, device="cuda")

        def forward(self, ids, positions):
            slots = get_context().state_indices
            self.state.index_add_(0, slots, torch.ones_like(ids, dtype=torch.float32))
            return self.state.index_select(0, slots).unsqueeze(1)

    model = SlotAccumulator()
    runner = SimpleNamespace(model=model, block_size=256,
        config=SimpleNamespace(max_num_seqs=16, max_model_len=256,
                               hf_config=SimpleNamespace(hidden_size=1)))
    manager = CudaGraphManager(runner)
    try:
        manager._capture_decode()
        # The largest graph's eager warmups must touch every slot, rather than
        # dispatch every request to slot zero before an in-place state update.
        assert torch.all(model.state > 0)
    finally:
        manager.clear()
        reset_context()
