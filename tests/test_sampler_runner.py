"""CPU request temperatures choose production sampling without device reads."""

from types import SimpleNamespace

import pytest
import torch

from nano_qwen.engine.model_runner import ModelRunner
from nano_qwen.layers.sampler import Sampler
from nano_qwen.utils.context import reset_context


@pytest.mark.parametrize("prefill", [False, True])
def test_cpu_greedy_routing_keeps_rng_and_samples_correct_tail(prefill, monkeypatch):
    runner = ModelRunner.__new__(ModelRunner)
    runner.rank = 0
    runner.sampler = Sampler()
    runner.batch_slots_gpu = torch.tensor([1, 0])
    runner.sampled_token_ids_gpu = torch.zeros(2, dtype=torch.long)
    runner.gdn_layers = []
    runner.async_output = False
    runner._output_buf_idx = 0
    runner._output_pin_bufs = [torch.empty(2, dtype=torch.long) for _ in range(2)]
    event = SimpleNamespace(record=lambda *a: None, synchronize=lambda: None)
    runner._output_events = [event, event]
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    seqs = [SimpleNamespace(temperature=1e-6, num_scheduled_tokens=2 if prefill else 1,
                            num_cached_tokens=0, num_tokens=2, is_prefill=prefill),
            SimpleNamespace(temperature=1e-6, num_scheduled_tokens=1,
                            num_cached_tokens=0, num_tokens=1, is_prefill=prefill)]
    logits = torch.tensor([[0., 9., 1.], [8., 8., 1.], [0., 1., 9.]]) if prefill else torch.tensor([[8., 8., 1.], [0., 1., 9.]])
    runner._pending = (logits, torch.tensor([1e-6, 1e-6]), seqs, prefill)
    rng = torch.random.get_rng_state().clone()
    try:
        with torch.compiler.set_stance("force_eager"):
            actual = runner.sample_tokens().get_output()
        assert actual == [0, 2]
        assert runner.sampled_token_ids_gpu.tolist() == [2, 0]
        assert torch.equal(rng, torch.random.get_rng_state())
    finally:
        reset_context()
