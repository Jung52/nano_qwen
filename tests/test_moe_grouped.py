import pytest
import torch

from nano_qwen.layers.moe import Qwen3_5RoutedExperts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires CUDA")
@pytest.mark.parametrize("tokens", [1, 4, 65])
@torch.inference_mode()
def test_grouped_experts_replays_with_changed_routes(tokens):
    experts = Qwen3_5RoutedExperts(4, 64, 32).to(device="cuda", dtype=torch.bfloat16)
    generator = torch.Generator(device="cuda").manual_seed(26)
    experts.gate_up_proj.normal_(std=0.04, generator=generator)
    experts.down_proj.normal_(std=0.04, generator=generator)
    hidden = torch.randn(tokens, 64, device="cuda", dtype=torch.bfloat16, generator=generator)
    ids = torch.tensor([2, 0], device="cuda").repeat(tokens, 1)
    weights = torch.tensor([0.25, 0.75], device="cuda", dtype=torch.bfloat16).repeat(tokens, 1)
    new_ids = torch.tensor([3, 1], device="cuda").repeat(tokens, 1)
    new_ids[1::2, 0] = 2  # Changes both the expert IDs and the group sizes.
    changed_weights = weights.flip(1)
    expected_first = experts.forward_eager(hidden, ids, weights)
    expected_second = experts.forward_eager(-hidden, new_ids, changed_weights)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        eager = experts(hidden, ids, weights)
    torch.cuda.current_stream().wait_stream(warmup)

    # BF16 rounding can differ from the per-expert PyTorch GEMM.
    torch.testing.assert_close(eager, expected_first, atol=4e-5, rtol=0.02)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = experts(hidden, ids, weights)
    graph.replay()
    torch.testing.assert_close(actual, eager, atol=0, rtol=0)

    hidden.neg_()
    ids.copy_(new_ids)
    weights.copy_(changed_weights)
    eager = experts(hidden, ids, weights)
    torch.testing.assert_close(eager, expected_second, atol=4e-5, rtol=0.02)
    graph.replay()
    torch.testing.assert_close(actual, eager, atol=0, rtol=0)
