# ruff: noqa: F811
"""B2 uses real dense/state/cache/graph paths; only decisions are forced."""

from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

import test_mtp_cache_integration as lifecycle
from test_mtp_verify_reuse import tiny_checkpoint, fp8_checkpoint  # noqa: F401
from nano_qwen.config import Config
from nano_qwen.engine.sequence import Sequence
from nano_qwen.layers.attention import Attention
from nano_qwen.sampling_params import SamplingParams
from nano_qwen.utils.context import get_context, reset_context
from test_mtp_weights import checkpoint_pair


def test_b2_reuse_config_is_supported_while_larger_batches_remain_guarded(tmp_path):
    target, source = checkpoint_pair(tmp_path)
    config = Config(
        str(target),
        enable_mtp=True,
        max_num_seqs=2,
        mtp_model=str(source),
        mtp_verify_mode="reuse",
    )
    assert config.max_num_seqs == 2
    with pytest.raises((AssertionError, ValueError), match="max_num_seqs"):
        Config(str(target), enable_mtp=True, max_num_seqs=16, mtp_verify_mode="reuse")


@pytest.fixture(scope="module", params=[True, False], ids=["eager", "graph"])
def engine(request, tiny_checkpoint, fp8_checkpoint):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    with lifecycle.hybrid_engine(
        fp8_checkpoint,
        request.param,
        mtp_model=str(tiny_checkpoint),
        mtp_verify_mode="reuse",
        max_num_seqs=2,
    ) as value:
        yield value


@torch.inference_mode()
def batch_case(
    engine,
    masks,
    lengths=(129, 257),
    budgets=(9, 6),
    eos=(None, None),
    chunk=1024,
    late=False,
    queued=False,
):
    runner = engine.model_runner
    engine.scheduler.max_num_batched_tokens = chunk
    seqs = [
        Sequence(
            [2 + i] * n,
            SamplingParams(temperature=1e-6, max_tokens=b, ignore_eos=e is None),
        )
        for i, (n, b, e) in enumerate(zip(lengths, budgets, eos))
    ]
    snapshots, hidden, actual_logits, decisions, batch_sizes = {}, {}, {}, [[] for _ in seqs], []

    def token(i, position):
        return 1 if position - lengths[i] == eos[i] else 2 + i * 20 + position % 10

    run_model, schedule, postprocess = (
        runner.run_model,
        engine.scheduler.schedule,
        engine.scheduler.postprocess,
    )

    def run(ids, positions, prefill, verify=False):
        context = get_context()
        slots = context.state_indices.tolist()
        sizes = (
            (context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]).tolist()
            if prefill
            else [1] * len(slots)
        )
        inactive = [j for j in range(engine.config.max_num_seqs) if j not in slots]
        untouched = [
            (m, j, m.conv_states[j].clone(), m.recurrent_states[j].clone())
            for m in runner.gdn_layers
            for j in inactive
        ]
        logits = run_model(ids, positions, prefill, verify)
        for m, j, c, r in untouched:
            assert torch.equal(c, m.conv_states[j]) and torch.equal(
                r, m.recurrent_states[j]
            ), "inactive graph row wrote state"
        result = torch.zeros_like(logits)
        selected = {row: i for i, row in enumerate(runner._sampling_plan.row_indices)}
        offset = 0
        for slot, size in zip(slots, sizes):
            request = runner.input_batch.seqs[slot]
            i = seqs.index(request)
            for row in range(offset, offset + size):
                p = int(positions[row])
                hidden[i, p] = runner._target_hidden[row].clone()
                if row in selected:
                    actual_logits[i, p] = logits[selected[row]].clone()
                    result[selected[row], token(i, p + 1)] = 10
            offset += size
        return result

    def scheduled():
        for i, seq in enumerate(seqs):
            if seq in engine.scheduler.running and seq.draft_token is not None:
                mask = masks[len(decisions[i]) % len(masks)]
                accept = mask[i % len(mask)]
                seq.draft_token = token(i, seq.num_tokens) if accept else 200 + i
        rows, prefill = schedule()
        batch_sizes.append(len(rows))
        for seq in rows:
            if seq.is_speculative:
                i = seqs.index(seq)
                mask = masks[len(decisions[i]) % len(masks)]
                decisions[i].append(mask[i % len(mask)])
        assert not any(s.is_speculative for s in rows) or all(
            s.is_speculative for s in rows
        )
        return rows, prefill

    def post(rows, outputs, prefill):
        for seq, output in zip(rows, outputs):
            i = seqs.index(seq)
            slot = runner.input_batch.seq_id_to_slot[seq.seq_id]
            prefix = seq.num_cached_tokens + output.cache_advance
            count = runner._mtp_cache_lens.get(slot, 0)
            tensors = []
            for m in runner.gdn_layers:
                tensors.extend(
                    [m.conv_states[slot].clone(), m.recurrent_states[slot].clone()]
                )
            for m in runner.model.modules():
                if isinstance(m, Attention):
                    tensors.extend(
                        lifecycle.valid_cache(t, seq.block_table, prefix)
                        for t in (m.k_cache, m.v_cache)
                    )
            tensors.extend(
                lifecycle.valid_cache(t, seq.block_table, count)
                for t in runner.mtp_kv_cache
            )
            snapshots[i, prefix] = count, tensors
        postprocess(rows, outputs, prefill)

    with ExitStack() as stack:
        stack.enter_context(patch.object(runner, "run_model", run))
        stack.enter_context(patch.object(engine.scheduler, "schedule", scheduled))
        stack.enter_context(patch.object(engine.scheduler, "postprocess", post))
        capacity = engine.config.max_num_seqs
        for seq in seqs if queued else seqs[:capacity]:
            engine.scheduler.add(seq)
        admitted_late = False
        for _ in range(100):
            engine.step()
            if late and seqs[0].is_finished and not admitted_late:
                assert not seqs[1].is_finished
                engine.scheduler.add(seqs[capacity])
                admitted_late = True
            if all(s.is_finished for s in seqs):
                break
    assert all(s.is_finished for s in seqs)
    assert not runner._mtp_cache_lens and not runner._mtp_last_hidden
    return dict(
        tokens=[s.completion_token_ids for s in seqs],
        decisions=decisions,
        snapshots=snapshots,
        hidden=hidden,
        logits=actual_logits,
        batches=batch_sizes,
    )


@pytest.mark.parametrize(
    "mask", [(True, True), (True, False), (False, True), (False, False)]
)
@torch.inference_mode()
def test_independent_accept_masks_preserve_serial_states_and_caches(engine, mask):
    compare_modes(engine, [mask])


@pytest.mark.parametrize(
    "budgets,eos,chunk",
    [
        ((1, 2), (None, None), 1024),
        ((9, 9), (1, None), 1024),
        ((9, 9), (2, None), 1024),
        ((9, 9), (None, 1), 128),
        ((3, 9), (None, None), 128),
    ],
)
@torch.inference_mode()
def test_eos_budgets_chunked_prefill_and_slot_reuse(engine, budgets, eos, chunk):
    compare_modes(
        engine, [(True, False), (False, True)], budgets=budgets, eos=eos, chunk=chunk
    )


@torch.inference_mode()
def test_finished_slot_is_reused_while_other_request_stays_live(engine):
    compare_modes(
        engine,
        [(True, False), (False, True)],
        lengths=(129, 257, 127),
        budgets=(3, 12, 5),
        eos=(None, None, None),
        late=True,
    )


@torch.inference_mode()
def test_three_queued_requests_respect_two_slots_during_chunked_prefill(engine):
    compare_modes(
        engine,
        [(True, False), (False, True)],
        lengths=(129, 257, 127),
        budgets=(4, 6, 5),
        eos=(None, None, None),
        chunk=128,
        queued=True,
    )


def compare_modes(engine, masks, **kwargs):
    runner = engine.model_runner
    try:
        results = []
        for mode in ("serial", "reuse"):
            runner.config.mtp_verify_mode = mode
            if not runner.enforce_eager:
                runner.cuda_graphs._capture_verify()
            results.append(batch_case(engine, masks, **kwargs))
        reference, actual = results
        assert actual["tokens"] == reference["tokens"]
        assert actual["decisions"] == reference["decisions"]
        assert actual["snapshots"].keys() == reference["snapshots"].keys()
        for key, (count, tensors) in reference["snapshots"].items():
            other_count, others = actual["snapshots"][key]
            assert count == other_count
            for a, b in zip(tensors, others):
                assert torch.equal(a, b), (key, (a - b).abs().max())
        for key, h in reference["hidden"].items():
            assert torch.equal(h, actual["hidden"][key]), key
        assert actual["logits"].keys() == reference["logits"].keys()
        for key, logits in reference["logits"].items():
            assert torch.equal(logits, actual["logits"][key]), key
        return actual
    finally:
        runner.config.mtp_verify_mode = "reuse"
        if not runner.enforce_eager and runner.cuda_graphs.verify_graph_mode != "reuse":
            runner.cuda_graphs._capture_verify()
        reset_context()
