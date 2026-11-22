"""GDN prefill chunk alignment: non-final chunks land on 64-token boundaries;
the final chunk keeps its real length; prompts defer when the remaining
budget fits no aligned chunk (decode keeps progressing)."""
import pytest

from pd_cpu_fixture import engine, finish, Sequence, SamplingParams


def step_chunks(e, seq, until_cached):
    chunks = []
    while seq.num_cached_tokens < until_cached:
        before = seq.num_cached_tokens
        e.step()
        if seq.num_cached_tokens > before:
            chunks.append(seq.num_cached_tokens - before)
    return chunks


def test_non_final_chunks_align_to_64():
    e = engine(blocks=64, budget=511)
    params = SamplingParams(temperature=1e-8, max_tokens=4, ignore_eos=True)
    seq = Sequence([3] * 1024, params)
    e.scheduler.add(seq)
    chunks = step_chunks(e, seq, 1024)
    assert chunks == [448, 448, 128], chunks
    finish(e)
    assert not e.scheduler.block_manager.used_block_ids


def test_final_chunk_keeps_real_length():
    e = engine(blocks=64, budget=512)
    params = SamplingParams(temperature=1e-8, max_tokens=4, ignore_eos=True)
    seq = Sequence([3] * 257, params)  # 257 = 4*64 + 1, single chunk = final
    e.scheduler.add(seq)
    chunks = step_chunks(e, seq, 257)
    assert chunks == [257], chunks
    finish(e)


def test_budget_below_aligned_chunk_raises_when_idle():
    e = engine(blocks=32, budget=40)
    e.scheduler.add(Sequence([3] * 200, SamplingParams(max_tokens=4)))
    with pytest.raises(RuntimeError, match="aligned prefill chunk"):
        e.step()


def test_deferred_prefill_waits_then_progresses_after_decode():
    e = engine(blocks=64, budget=64)
    params = SamplingParams(temperature=1e-8, max_tokens=6, ignore_eos=True)
    dseq = Sequence([3] * 17, params)
    e.scheduler.add(dseq)
    while dseq.num_completion_tokens < 1:
        e.step()  # decode request at the committed boundary
    pseq = Sequence([3] * 200, params)
    e.scheduler.add(pseq)
    # While decode continues (budget 64 -> 63 remaining), the prompt defers.
    while dseq.num_completion_tokens < params.max_tokens:
        before = pseq.num_cached_tokens
        e.step()
        assert pseq.num_cached_tokens == before, "prompt must defer under decode"
    # After the decode drains, the prompt proceeds in aligned chunks.
    chunks = step_chunks(e, pseq, 200)
    assert chunks == [64, 64, 64, 8], chunks  # final chunk keeps real length
    finish(e)
    assert not e.scheduler.block_manager.used_block_ids
