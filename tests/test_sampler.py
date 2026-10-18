"""Greedy requests use deterministic argmax, including tied logits."""

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


def test_all_greedy_hint_skips_softmax_rng_and_scalar_reads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("all-greedy must not sample or synchronize scalar values")

    logits = torch.tensor([[3., 3., 1.], [0., -1., 4.]])
    saved = logits.clone()
    rng = torch.random.get_rng_state().clone()
    monkeypatch.setattr(torch, "softmax", forbidden)
    monkeypatch.setattr(torch.Tensor, "exponential_", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    with torch.compiler.set_stance("force_eager"):
        result = Sampler()(logits, torch.tensor([1e-6, 1e-6]), all_greedy=True)
    assert torch.equal(result, torch.tensor([0, 2]))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(logits, saved)
