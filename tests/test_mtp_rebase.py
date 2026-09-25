"""MTP verify retains structured dispatch and indexed GDN rollback semantics."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig

from nano_qwen.engine.model_runner import ModelRunner
from nano_qwen.engine.sequence import Sequence
from nano_qwen.layers.gated_delta_net import GatedDeltaNet
from nano_qwen.quantization.fp8 import Fp8Config
from nano_qwen.quantization.fp8_block import Fp8BlockConfig
from nano_qwen.utils.context import BatchDescriptor, get_context, reset_context, set_context


def test_verify_context_retains_prefill_metadata_and_resets():
    descriptor = BatchDescriptor(mode="spec_decode", uniform_token_count=2)
    try:
        set_context(True, batch_descriptor=descriptor, cudagraph_mode="full",
                    prefill_decode_rows=[1], is_verify=True)
        context = get_context()
        assert context.batch_descriptor is descriptor
        assert context.cudagraph_mode == "full"
        assert context.prefill_decode_rows == [1]
        assert context.is_verify
    finally:
        reset_context()
    assert not get_context().is_verify
    assert get_context().prefill_decode_rows == []


def test_spec_descriptor_dispatches_verify_before_prefill_graph():
    expected = object()
    runner = ModelRunner.__new__(ModelRunner)
    runner.enforce_eager = False
    runner.use_prefill_cudagraph = True
    runner.cuda_graphs = SimpleNamespace(decode_graphs={1: object()}, verify_graph=object(),
                                        run_verify=lambda ids, positions: expected)
    try:
        set_context(True, batch_descriptor=BatchDescriptor(mode="spec_decode"), is_verify=True)
        assert runner.run_model(torch.tensor([3, 4]), torch.tensor([8, 9]), True, True) is expected
    finally:
        reset_context()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_prepare_verify_preserves_structured_batch_and_slot_mapping():
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = 256
    runner.batch_slots_gpu = torch.tensor([2], device="cuda")
    seq = Sequence([3, 4, 5])
    seq.num_cached_tokens = 1
    seq.num_scheduled_tokens = 2
    seq.is_speculative = True
    seq.block_table = [0]
    try:
        ids, positions = runner.prepare_prefill([seq])
        context = get_context()
        assert ids.tolist() == [4, 5] and positions.tolist() == [1, 2]
        assert context.is_verify
        assert context.state_indices.tolist() == [2]
        assert context.context_lens.tolist() == [3]
        assert context.batch_descriptor == BatchDescriptor(
            mode="spec_decode", num_tokens=2, num_reqs=1,
            uniform_token_count=2, max_query_len=2,
        )
    finally:
        reset_context()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("quant_config", [None, Fp8Config(), Fp8BlockConfig({})],
                         ids=["bf16", "per-channel", "block-wise"])
@torch.inference_mode()
def test_verify_graph_snapshot_and_rollback_match_two_decode_steps(quant_config):
    torch.manual_seed(157)
    config = AutoConfig.for_model("qwen3_5_text")
    config.hidden_size = 128
    config.linear_num_key_heads = 1
    config.linear_num_value_heads = 2
    config.linear_key_head_dim = config.linear_value_head_dim = 128
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"), patch.object(dist, "get_rank", return_value=0), \
                patch.object(dist, "get_world_size", return_value=1):
            layer = GatedDeltaNet(config, 0, quant_config=quant_config)
        for name, parameter in layer.named_parameters():
            if parameter.dtype == torch.float8_e4m3fn:
                parameter.copy_(torch.randn(parameter.shape, device="cuda", dtype=torch.float32).to(parameter.dtype))
            elif "weight_scale" in name:
                parameter.fill_(.02)
            else:
                parameter.normal_(std=.02)
        layer.A_log.zero_()
        layer.dt_bias.zero_()
        layer.allocate_state_pool(3)
        layer.allocate_speculative_snapshot()
        layer.conv_states.normal_(std=.01)
        layer.recurrent_states.normal_(std=.01)
        initial = (layer.conv_states.clone(), layer.recurrent_states.clone())
        slots = torch.tensor([2], device="cuda")
        hidden = torch.randn(1, 2, 128, device="cuda")
        set_context(False, state_indices=slots)
        first = layer(hidden[:, :1]).clone()
        after_first = (layer.conv_states.clone(), layer.recurrent_states.clone())
        second = layer(hidden[:, 1:]).clone()
        after_second = (layer.conv_states.clone(), layer.recurrent_states.clone())
        expected = torch.cat((first, second), dim=1)

        def restore_initial():
            layer.conv_states.copy_(initial[0])
            layer.recurrent_states.copy_(initial[1])

        cu = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
        set_context(True, cu_seqlens_q=cu, state_indices=slots,
                    prefill_slices=[(0, 2)], is_verify=True)
        layer(hidden)  # Warm the exact verify path before capture.
        restore_initial()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = layer(hidden)
        for _ in range(2):
            restore_initial()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(layer.conv_states, after_second[0], rtol=0, atol=0)
            torch.testing.assert_close(layer.recurrent_states, after_second[1], rtol=0, atol=0)
            layer.rollback_speculative_state(2)
            torch.testing.assert_close(layer.conv_states, after_first[0], rtol=0, atol=0)
            torch.testing.assert_close(layer.recurrent_states, after_first[1], rtol=0, atol=0)
    finally:
        reset_context()
        torch.set_default_dtype(previous_dtype)
