# ruff: noqa: F811
from types import SimpleNamespace

import torch

from test_mtp_cache_lifecycle import runner, sequence  # noqa: F401
from nano_qwen.scheduler.scheduler import Scheduler
from nano_qwen.engine.sequence import SequenceStatus
from nano_qwen.utils.context import get_context


def test_verify_lengths_are_per_request_and_slots_remain_request_major(runner):
    a, b = sequence(130, [4]), sequence(258, [1, 6])
    for seq in (a, b):
        seq.num_cached_tokens = len(seq) - 2
        seq.num_scheduled_tokens = 2
        seq.is_speculative = True
    runner.batch_slots_gpu = torch.tensor([1, 0])
    _, positions = runner.prepare_prefill([a, b])
    context = get_context()
    assert context.context_lens.tolist() == [130, 258]
    assert context.state_indices.tolist() == [1, 0]
    assert positions.tolist() == [128, 129, 256, 257]
    assert context.slot_mapping.tolist() == [
        4 * 256 + 128,
        4 * 256 + 129,
        6 * 256,
        6 * 256 + 1,
    ]


def test_scheduler_keeps_verification_separate_from_final_ordinary_row():
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos=1,
        kvcache_block_size=256,
        enable_prefix_cache=False,
        enable_mtp=True,
        num_kvcache_blocks=8,
    )
    scheduler = Scheduler(config)
    a, b = sequence(129, [], max_tokens=8), sequence(129, [], max_tokens=2)
    for seq in (a, b):
        scheduler.block_manager.allocate(seq, 0)
        seq.num_cached_tokens = 129
        seq.append_token(4)
        seq.draft_token = 5
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)
    rows, _ = scheduler.schedule()
    assert rows == [a] and a.is_speculative
    rows, prefill = scheduler.schedule()
    assert rows == [b] and not prefill and not b.is_speculative


def test_kv_pressure_waits_for_inflight_verify_before_readmission():
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos=1,
        kvcache_block_size=256,
        enable_prefix_cache=False,
        enable_mtp=True,
        num_kvcache_blocks=2,
    )
    scheduler = Scheduler(config)
    a, b = sequence(255, [], max_tokens=3), sequence(255, [], max_tokens=3)
    for seq in (a, b):
        scheduler.block_manager.allocate(seq, 0)
        seq.num_cached_tokens = 255
        seq.append_token(4)
        seq.draft_token = 5
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)
    rows, _ = scheduler.schedule()
    assert rows == [b] and list(scheduler.waiting) == [a]
    assert scheduler.schedule()[0] == []  # B may still finish and free both blocks.
    scheduler.postprocess([b], [[5, 6]], True)
    assert b.is_finished
    assert scheduler.schedule()[0] == [a]


def test_new_admission_waits_for_a_state_slot_including_chunked_requests():
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=64,
        eos=1,
        kvcache_block_size=256,
        enable_prefix_cache=False,
        enable_mtp=True,
        num_kvcache_blocks=8,
    )
    scheduler = Scheduler(config)
    a, b, c = [sequence(128, [], max_tokens=1) for _ in range(3)]
    for seq in (a, b, c):
        scheduler.add(seq)
    assert scheduler.schedule()[0] == [a]
    scheduler.postprocess([a], [[4]], True)  # A chunk needs its reserved slot later.
    assert scheduler.schedule()[0] == [b]
    scheduler.postprocess([b], [[4]], True)
    # C cannot steal the slots held by partial A/B, and cannot block their progress.
    assert scheduler.schedule()[0] == [a]
    assert scheduler.schedule()[0] == [b]
    assert scheduler.schedule()[0] == []
    scheduler.postprocess([a], [[4]], True)
    assert a.is_finished
    assert scheduler.schedule()[0] == [c]


def test_preempted_replay_does_not_block_a_chunk_that_can_release_kv():
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=64,
        eos=1,
        kvcache_block_size=256,
        enable_prefix_cache=False,
        enable_mtp=True,
        num_kvcache_blocks=2,
    )
    scheduler = Scheduler(config)
    a, b = sequence(256, [], max_tokens=4), sequence(128, [], max_tokens=1)
    scheduler.block_manager.allocate(a, 0)
    a.num_cached_tokens = 256
    a.append_token(4)
    a.draft_token = 5
    a.status = SequenceStatus.RUNNING
    scheduler.running.append(a)
    scheduler.add(b)
    assert scheduler.schedule()[0] == [b]
    assert scheduler.schedule()[0] == []  # A preempts while B's chunk is in flight.
    scheduler.postprocess([b], [[4]], True)
    assert list(scheduler.waiting) == [a, b]
    assert scheduler.schedule()[0] == [b]  # B's last chunk can release its block.
    scheduler.postprocess([b], [[4]], True)
    assert b.is_finished
    assert scheduler.schedule()[0] == [a]
