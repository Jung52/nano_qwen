"""Standalone vocabulary-row planning, independent of speculative decoding."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from nano_qwen.engine.execution import SamplingPlan
from nano_qwen.engine.model_runner import ModelRunner
from nano_qwen.engine.sequence import Sequence
from nano_qwen.utils.context import reset_context


def request(length, cached, scheduled, prefill=True):
    seq = Sequence([2]*length)
    seq.num_cached_tokens, seq.num_scheduled_tokens, seq.is_prefill = cached, scheduled, prefill
    return seq


def test_plan_selects_decode_and_final_prefill_and_skips_partial():
    seqs = [request(6, 5, 1, False), request(8, 0, 3), request(9, 5, 4)]
    assert SamplingPlan.for_requests(seqs) == SamplingPlan((0, 7), (0, 2))


@pytest.mark.parametrize('final', [False, True])
def test_projection_preserves_hidden_and_omits_nonfinal_head(final):
    runner = ModelRunner.__new__(ModelRunner)
    seq = request(3 if final else 5, 0, 3)
    runner._sampling_plan = SamplingPlan.for_requests([seq])
    head = Mock(side_effect=lambda x: x+1)
    runner.model = SimpleNamespace(compute_logits=head, lm_head=SimpleNamespace(num_embeddings=4))
    hidden = torch.arange(12).reshape(3, 4).float()
    logits = runner.project_target_hidden(hidden)
    assert runner._target_hidden is hidden
    assert head.call_count == int(final)
    assert logits.shape == (int(final), 4)
    if final:
        assert torch.equal(logits, hidden[-1:]+1)


@pytest.mark.parametrize('partial_only', [False, True])
def test_sampler_keeps_unprojected_slots_untouched(partial_only, monkeypatch):
    runner = ModelRunner.__new__(ModelRunner)
    a, b = request(5, 0, 3), request(2, 0, 2)
    seqs = [a] if partial_only else [a, b]
    runner.rank, runner.gdn_layers, runner.async_output = 0, [], False
    runner.batch_slots_gpu = torch.tensor([1, 0])
    runner.sampled_token_ids_gpu = torch.tensor([7, 8])
    runner._output_buf_idx = 0
    runner._output_pin_bufs = [torch.empty(2, dtype=torch.long) for _ in range(2)]
    event = SimpleNamespace(record=lambda *a: None, synchronize=lambda: None)
    runner._output_events = [event, event]
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: None)
    seen = {}
    class Sampler:
        def set_batch_context(self, rows, prefill):
            seen['requests'] = rows
        def observe_gdn_state(self, layers, slots):
            seen['slots'] = slots.tolist()
        def __call__(self, logits, temperatures, all_greedy=False):
            seen['sampled'] = len(logits)
            return logits.argmax(-1)
    runner.sampler = Sampler()
    logits = torch.empty(0, 3) if partial_only else torch.tensor([[0., 2., 1.]])
    runner._sampling_plan = SamplingPlan.for_requests(seqs)
    runner._pending = logits, torch.ones(len(seqs)), seqs, True
    result = runner.sample_tokens().get_output()
    assert result == ([-1] if partial_only else [-1, 1])
    assert runner.sampled_token_ids_gpu.tolist() == ([7, 8] if partial_only else [1, 8])
    assert seen.get('sampled', 0) == int(not partial_only)
    if not partial_only:
        assert seen['requests'] == [b] and seen['slots'] == [0]
    reset_context()
