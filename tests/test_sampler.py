"""Greedy requests must break tied logits identically to MTP verification."""

import torch

from nano_qwen.layers.sampler import Sampler


def test_greedy_ties_use_argmax_while_other_requests_keep_sampling():
    logits = torch.ones(128, 2)
    temperatures = torch.tensor([1e-6] * 64 + [1.0] * 64)
    with torch.compiler.set_stance("force_eager"):
        torch.manual_seed(7)
        tokens = Sampler()(logits, temperatures)
    assert torch.equal(tokens[:64], torch.zeros(64, dtype=torch.long))
    assert set(tokens[64:].tolist()) == {0, 1}
