# ruff: noqa: F811
"""CPU tests for selected vocabulary rows and authoritative token commits."""

from types import SimpleNamespace

import pytest
import torch

from nano_qwen.engine import execution
from nano_qwen.engine.sequence import Sequence, SequenceStatus
from nano_qwen.sampling_params import SamplingParams
from nano_qwen.scheduler.scheduler import Scheduler
from test_mtp_cache_lifecycle import runner  # noqa: F401


def request(prompt, cached, count, prefill=True):
    seq = Sequence([2]*prompt, SamplingParams(max_tokens=8, ignore_eos=True))
    seq.num_cached_tokens, seq.num_scheduled_tokens = cached, count
    seq.is_prefill = prefill
    return seq


def test_sampling_plan_skips_partial_prompts_and_preserves_mixed_row_order():
    seqs = [request(5, 4, 1, False), request(8, 0, 3), request(4, 2, 2)]
    plan = execution.SamplingPlan.for_requests(seqs)
    assert plan.row_indices == (0, 5)
    assert plan.request_indices == (0, 2)


def test_verify_plan_uses_both_rows_without_committing_drafts():
    seqs = [request(5, 4, 2), request(7, 6, 2)]
    for seq in seqs:
        seq.is_speculative, seq.draft_token = True, 9
    plan = execution.SamplingPlan.for_requests(seqs)
    assert plan.row_indices == (0, 1, 2, 3)
    assert plan.request_indices == (0, 0, 1, 1)
    assert all(seq.token_ids[-1] == 2 for seq in seqs)


def test_target_projection_keeps_full_hidden_but_selects_only_requested_rows(runner):
    seqs = [request(8, 0, 3), request(4, 2, 2)]
    runner._sampling_plan = execution.SamplingPlan.for_requests(seqs)
    seen = []
    runner.model.compute_logits = lambda rows: seen.append(rows.clone()) or rows*2
    hidden = torch.arange(20).reshape(5,4).float()
    logits = runner.project_target_hidden(hidden)
    assert torch.equal(runner._target_hidden, hidden)
    assert len(seen) == 1 and torch.equal(seen[0], hidden[4:])
    assert torch.equal(logits, hidden[4:]*2)


def test_nonfinal_chunk_never_projects_or_samples(runner):
    seq = request(8, 0, 3)
    runner.input_batch.update([seq])
    runner._sampling_plan = execution.SamplingPlan.for_requests([seq])
    runner.model.lm_head = SimpleNamespace(num_embeddings=16)
    def forbidden(*args, **kwargs):
        raise AssertionError("Nonfinal prompt chunk must not project or sample")
    runner.model.compute_logits = forbidden
    runner.sampler = forbidden
    runner.mtp = None
    logits = runner.project_target_hidden(torch.zeros(3,4))
    assert logits.shape == (0,16)
    runner._pending = (logits, torch.ones(1), [seq], True)
    result = runner.sample_tokens().get_results()[0]
    assert result.committed_tokens == () and result.next_draft is None
    assert result.cache_advance == 3


def scheduled_verify(*, max_tokens=8, eos=False):
    config = SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=4, eos=1,
                             kvcache_block_size=256, enable_prefix_cache=False,
                             enable_mtp=True, num_kvcache_blocks=8)
    scheduler = Scheduler(config)
    seq = Sequence([2]*4, SamplingParams(temperature=1e-6, max_tokens=max_tokens, ignore_eos=not eos))
    scheduler.block_manager.allocate(seq, 0)
    seq.num_cached_tokens = 4
    seq.append_token(6)
    seq.draft_token, seq.status = 9, SequenceStatus.RUNNING
    scheduler.running.append(seq)
    return scheduler, seq


def test_scheduler_keeps_unverified_draft_out_of_authoritative_history():
    scheduler, seq = scheduled_verify()
    before = list(seq.token_ids)
    rows, prefill = scheduler.schedule()
    assert rows == [seq] and prefill and seq.is_speculative
    assert seq.token_ids == before and seq.num_tokens == 5 and seq.last_token == 6
    assert seq.execution_tokens(4, 6) == [6, 9]


@pytest.mark.parametrize("tokens,accepted,advance,finished", [
    ((9, 10), 1, 2, False), ((8,), 0, 1, False),
    ((1,), 1, 1, True), ((9, 1), 1, 2, True),
])
def test_explicit_commit_advances_only_consumed_state(tokens, accepted, advance, finished):
    scheduler, seq = scheduled_verify(eos=True)
    if accepted:
        seq.draft_token = tokens[0]
    scheduler.schedule()
    seen = []
    deallocate = scheduler.block_manager.deallocate
    def release(item):
        seen.append(item.num_cached_tokens)
        deallocate(item)
    scheduler.block_manager.deallocate = release
    result = execution.RequestOutput(tokens, accepted, 12 if not finished else None, advance)
    scheduler.postprocess([seq], [result], True)
    assert seq.completion_token_ids == [6, *tokens]
    assert seq.is_finished == finished and not seq.is_speculative
    assert (seen[0] if finished else seq.num_cached_tokens) == 4+advance
    assert seq.draft_token == (None if finished else 12)
    assert not scheduler.in_flight


def test_rejected_result_cannot_commit_second_step_state():
    scheduler, seq = scheduled_verify()
    scheduler.schedule()
    with pytest.raises(ValueError, match="cache|advance|commit"):
        scheduler.postprocess([seq], [execution.RequestOutput((8,), 0, None, 2)], True)
    assert seq.completion_token_ids == [6]


def test_accepted_non_eos_cannot_discard_bonus_state():
    scheduler, seq = scheduled_verify()
    scheduler.schedule()
    with pytest.raises(ValueError, match='commit|advance'):
        scheduler.postprocess([seq], [execution.RequestOutput((9,), 1, None, 1)], True)
    assert seq.completion_token_ids == [6] and seq.seq_id in scheduler.in_flight


def test_duplicate_request_commit_is_rejected_before_mutation():
    scheduler, seq = scheduled_verify()
    scheduler.schedule()
    output = execution.RequestOutput((8,), 0, None, 1)
    with pytest.raises(ValueError, match='duplicate|Duplicate'):
        scheduler.postprocess([seq, seq], [output, output], True)
    assert seq.completion_token_ids == [6] and seq.seq_id in scheduler.in_flight


def test_sampling_observers_follow_only_projected_requests(runner):
    from nano_qwen.engine.model_runner import InputBatch
    a, b = Sequence([2]*5), Sequence([3]*2)
    a.num_scheduled_tokens, b.num_scheduled_tokens = 3, 2
    runner.mtp = None
    runner.input_batch = InputBatch(2)
    runner.input_batch.update([a, b])
    runner.batch_slots_gpu = torch.tensor([0, 1])
    runner.sampled_token_ids_gpu = torch.tensor([7, 8])
    runner._output_pin_bufs = [torch.empty(2, 3, dtype=torch.long) for _ in range(2)]
    seen = {}
    class ObservedSampler:
        def set_batch_context(self, seqs, prefill):
            seen['seqs'] = seqs
        def observe_gdn_state(self, layers, slots):
            seen['slots'] = slots.tolist()
        def __call__(self, logits, temperatures, all_greedy=False):
            assert len(seen['seqs']) == len(logits) == 1
            return logits.argmax(-1)
    runner.sampler = ObservedSampler()
    logits = torch.zeros(1, 16)
    logits[0, 9] = 10
    runner._sampling_plan = execution.SamplingPlan.for_requests([a, b])
    runner._pending = (logits, torch.ones(2), [a, b], True)
    outputs = runner.sample_tokens().get_results()
    assert seen == dict(seqs=[b], slots=[1])
    assert outputs[0].committed_tokens == () and outputs[1].committed_tokens == (9,)
    assert runner.sampled_token_ids_gpu.tolist() == [7, 9]


def test_preempt_refuses_to_release_inflight_state():
    scheduler, seq = scheduled_verify()
    scheduler.schedule()
    before = (list(seq.token_ids), list(seq.block_table), seq.num_cached_tokens)
    with pytest.raises(ValueError, match='flight'):
        scheduler.preempt(seq)
    assert (seq.token_ids, seq.block_table, seq.num_cached_tokens) == before
    assert seq.seq_id in scheduler.in_flight
