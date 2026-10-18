"""Cache coordinates and lifecycle through the real MTP runner methods."""

from types import SimpleNamespace

import pytest
import torch

from nano_qwen.engine.model_runner import InputBatch, ModelRunner
from nano_qwen.engine.sequence import Sequence, SequenceStatus
from nano_qwen.sampling_params import SamplingParams
from nano_qwen.scheduler.scheduler import Scheduler
from nano_qwen.utils.context import get_context, reset_context


PROMPTS = [127, 128, 129, 255, 256, 257, 511, 512, 513]


class RecordedMTP:
    def __init__(self):
        self.calls = []

    def __call__(self, ids, positions, hidden, embedding):
        c = get_context()
        self.calls.append(dict(ids=ids.tolist(), positions=positions.tolist(),
            writes=c.slot_mapping.tolist(), hidden=hidden.clone(),
            q=None if c.cu_seqlens_q is None else c.cu_seqlens_q.tolist(),
            k=None if c.cu_seqlens_k is None else c.cu_seqlens_k.tolist(),
            context=None if c.context_lens is None else c.context_lens.tolist(),
            max_k=c.max_seqlen_k))
        return torch.zeros(len(ids), 4)

    def cache_only(self, ids, positions, hidden, embedding):
        self(ids, positions, hidden, embedding)


@pytest.fixture
def runner(monkeypatch):
    # Replace only CUDA staging/events; exercise the real cache and sampler logic.
    original_tensor = torch.tensor
    def host_tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_tensor(*args, **kwargs)
    monkeypatch.setattr(torch, "tensor", host_tensor)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    r = ModelRunner.__new__(ModelRunner)
    r.block_size = 256
    r.mtp = RecordedMTP()
    r.model = SimpleNamespace(model=SimpleNamespace(embed_tokens=None),
                              compute_logits=lambda hidden: torch.zeros(len(hidden), 16))
    r._mtp_last_hidden = {}
    r._mtp_cache_lens = {}
    r.cuda_graphs = SimpleNamespace(mtp_decode_graph=None, mtp_cache_graph=None, record_dispatch=lambda *args: None)
    r.input_batch = InputBatch(1)
    r.rank = 0
    r.batch_slots_gpu = torch.tensor([0])
    r.sampled_token_ids_gpu = torch.zeros(1, dtype=torch.long)
    r.gdn_layers = []
    r.sampler = lambda logits, temperatures, all_greedy=False: logits.argmax(dim=-1)
    r.config = SimpleNamespace(eos=1)
    r.mtp_stats = dict(draft_tokens=0, accepted_tokens=0, rejected_tokens=0, verify_steps=0)
    r.async_output = False
    r._output_buf_idx = 0
    r._output_pin_bufs = [torch.empty(1, 3, dtype=torch.long) for _ in range(2)]
    event = SimpleNamespace(record=lambda *a: None, synchronize=lambda: None)
    r._output_events = [event, event]
    yield r
    reset_context()


def sequence(length, blocks=None, max_tokens=12):
    seq = Sequence([2] * length, SamplingParams(temperature=1e-6,
                                               max_tokens=max_tokens, ignore_eos=True))
    seq.block_table = ([7, 3, 9][: (length + 255) // 256] if blocks is None else blocks)
    return seq


def physical(seq, rows):
    return [seq.block_table[p // 256] * 256 + p % 256 for p in rows]


@pytest.mark.parametrize("length", PROMPTS)
def test_prefill_writes_zero_based_cache_rows_at_prompt_boundaries(runner, length):
    seq = sequence(length)
    seq.num_scheduled_tokens = length
    runner._run_mtp_prefill(seq, 0, torch.zeros(length, 4), torch.tensor(4), True)
    call = runner.mtp.calls[-1]
    assert call["positions"] == list(range(1, length + 1))
    assert call["writes"] == physical(seq, range(length))
    assert call["k"] == [0, length]


@pytest.mark.parametrize("chunks", [[128, 1], [1, 128], [1, 1, 127], [128, 128, 1]])
def test_chunked_prefill_keeps_cumulative_kv_and_previous_hidden(runner, chunks):
    seq = sequence(sum(chunks))
    consumed = 0
    for i, count in enumerate(chunks):
        seq.num_cached_tokens = consumed
        seq.num_scheduled_tokens = count
        hidden = torch.arange(consumed, consumed + count).float().unsqueeze(1).expand(-1, 4)
        final = i == len(chunks) - 1
        runner._run_mtp_prefill(seq, 0, hidden, torch.tensor(4), final)
        consumed += count
        assert runner._mtp_last_hidden[0][0].item() == consumed - 1
        if consumed == 1 and not final:
            assert runner.mtp.calls == []
            continue
        call = runner.mtp.calls[-1]
        valid = consumed if final else consumed - 1
        assert call["k"] == [0, valid]
        assert call["max_k"] == valid
    assert [p for call in runner.mtp.calls for p in call["positions"]] == list(range(1, consumed + 1))
    assert [p for call in runner.mtp.calls for p in call["writes"]] == physical(seq, range(consumed))


def test_decode_cache_row_is_independent_of_rope_position(runner):
    seq = sequence(129)
    runner.input_batch.update([seq])
    runner._mtp_cache_lens[0] = 129
    runner._run_mtp_decode([seq], torch.zeros(1, 4), torch.tensor([5]), [130])
    call = runner.mtp.calls[-1]
    assert call["positions"] == [130]
    assert call["writes"] == physical(seq, [129])
    assert call["context"] == [130]


@pytest.mark.parametrize("accepted", [True, False])
def test_sampler_appends_accepted_draft_before_bonus(runner, accepted):
    seq = sequence(129)
    seq.append_token(4)
    seq.draft_token = 5
    seq.is_speculative = True
    seq.num_cached_tokens = 129
    seq.num_scheduled_tokens = 2
    runner.input_batch.update([seq])
    runner._mtp_cache_lens[0] = 129
    runner._target_hidden = torch.tensor([[10., 10., 10., 10.], [20., 20., 20., 20.]])
    logits = torch.zeros(2, 16)
    logits[0, 5 if accepted else 9] = 1
    logits[1, 6] = 1
    runner._pending = (logits, torch.ones(1), [seq], True)
    result = runner.sample_tokens().get_output()
    assert result[0][:2] == ([5, 6] if accepted else [9, -1])
    calls = runner.mtp.calls
    assert [p for c in calls for p in c["ids"]] == ([5, 6] if accepted else [9])
    assert [p for c in calls for p in c["positions"]] == ([130, 131] if accepted else [130])
    assert [p for c in calls for p in c["writes"]] == physical(seq, [129, 130] if accepted else [129])
    assert [c["hidden"][0, 0].item() for c in calls] == ([10., 20.] if accepted else [10.])


def test_verify_fitting_last_block_does_not_reserve_phantom_row():
    config = SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=2, eos=1,
        kvcache_block_size=256, enable_prefix_cache=False, enable_mtp=True,
        num_kvcache_blocks=1)
    scheduler = Scheduler(config)
    seq = sequence(254, blocks=[])
    scheduler.block_manager.allocate(seq, 0)
    seq.num_cached_tokens = 254
    seq.append_token(4)
    seq.draft_token = 5
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)
    batch, prefill = scheduler.schedule()
    assert batch == [seq] and prefill
    assert seq.is_speculative and seq.num_scheduled_tokens == 2
    assert seq.block_table == [0]


def run_lifecycle(runner, length, pattern, chunk, max_tokens=9, eos_at=None):
    config = SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=chunk, eos=1,
        kvcache_block_size=256, enable_prefix_cache=False, enable_mtp=True,
        num_kvcache_blocks=8)
    scheduler = Scheduler(config)
    seq = sequence(length, blocks=[], max_tokens=max_tokens)
    seq.ignore_eos = eos_at is None
    runner.input_batch.update([seq])
    scheduler.add(seq)
    verify = 0

    def token(position):
        return 1 if position - length == eos_at else 2 + position % 10

    for _ in range(length + max_tokens + 10):
        if scheduler.running and seq.draft_token is not None:
            accept = pattern[verify % len(pattern)]
            correct = token(seq.num_tokens)
            seq.draft_token = correct if accept else (correct + 1) % 16
        batch, prefill = scheduler.schedule()
        assert batch == [seq]
        if seq.is_speculative:
            verify += 1
        positions = list(range(seq.num_cached_tokens,
                               seq.num_cached_tokens + seq.num_scheduled_tokens))
        runner._target_hidden = torch.tensor(positions).float().unsqueeze(1).expand(-1, 4)
        logits = torch.zeros(len(positions), 16)
        for i, position in enumerate(positions):
            logits[i, token(position + 1)] = 1
        runner._pending = (logits, torch.ones(1), [seq], prefill)
        results = runner.sample_tokens().get_results()
        scheduler.postprocess([seq], results, prefill)
        calls = runner.mtp.calls
        ropes = [p for call in calls for p in call["positions"]]
        assert ropes == list(range(1, len(ropes) + 1))
        assert [p for call in calls for p in call["writes"]] == list(range(len(ropes)))
        assert runner._mtp_cache_lens[0] == len(ropes)
        for call in calls:
            assert call["hidden"][:, 0].tolist() == [p - 1 for p in call["positions"]]
            assert call["ids"] == [2 if p < length else token(p) for p in call["positions"]]
        if seq.is_finished:
            break
    else:
        pytest.fail("MTP lifecycle did not finish within its token budget")
    expected_length = max_tokens if eos_at is None else min(max_tokens, eos_at + 1)
    assert seq.completion_token_ids == [token(p) for p in range(length, length + expected_length)]
    assert scheduler.is_finished()
    runner.remove_request(seq.seq_id)
    assert runner._mtp_last_hidden == {} and runner._mtp_cache_lens == {}
    return seq.completion_token_ids


@pytest.mark.parametrize("length", PROMPTS)
@pytest.mark.parametrize("pattern", [[True], [False], [True, False], [False, True]])
@pytest.mark.parametrize("chunk", [1024, 128])
def test_full_sampler_scheduler_lifecycle_has_no_mtp_cache_holes(runner, length, pattern, chunk):
    run_lifecycle(runner, length, pattern, chunk)


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 4])
@pytest.mark.parametrize("eos_at", [None, 0, 1, 2])
def test_output_budget_eos_and_finished_slot_reuse(runner, max_tokens, eos_at):
    run_lifecycle(runner, 256, [True, False], 128, max_tokens, eos_at)
    runner.mtp.calls.clear()
    run_lifecycle(runner, 129, [False, True], 128, 4)
