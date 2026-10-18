# ruff: noqa: F811
"""Prefill head pruning must retain MTP hidden/KV and lifecycle semantics."""

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

from mtp_prefill_reference import all_row_head, prefill_reference
from nano_qwen.engine.sequence import Sequence
from nano_qwen.utils.context import reset_context
from test_mtp_batch2 import batch_case
from test_mtp_rollback import rollback_engine as engine, tiny_checkpoint, fp8_checkpoint, bits  # noqa: F401


@pytest.mark.parametrize("chunked", [False, True])
@torch.inference_mode()
def test_prefill_projects_only_final_tail_and_keeps_full_hidden_kv(engine, chunked):
    runner = engine.model_runner
    seq = Sequence([2] * 129)
    seq.block_table = [2]
    runner.input_batch.update([seq])
    hidden = torch.randn(129, 128, device="cuda", dtype=torch.bfloat16)
    token = torch.tensor(3, device="cuda")
    reference = prefill_reference()
    results = []
    try:
        for old in (True, False):
            runner._mtp_last_hidden.clear()
            runner._mtp_cache_lens.clear()
            runner.mtp_kv_cache.zero_()
            calls, projected, drafts = [], [], []
            mtp_forward, compute = runner.mtp.forward, runner.model.compute_logits

            def forward(*args):
                output = mtp_forward(*args)
                calls.append(output.clone())
                return output

            def head(rows):
                logits = compute(rows)
                projected.append((rows.shape[0], logits[-1:].clone()))
                return logits

            chunks = [(0, 128, False), (128, 1, True)] if chunked else [(0, 129, True)]
            with all_row_head(runner, reference) if old else nullcontext(), \
                 patch.object(runner.mtp, "forward", forward), patch.object(runner.model, "compute_logits", head):
                for start, count, final in chunks:
                    seq.num_cached_tokens, seq.num_scheduled_tokens = start, count
                    value = runner._run_mtp_prefill(seq, 0, hidden[start:start+count], token, final)
                    drafts.append(None if value is None else int(value))
            results.append(dict(hidden=calls, head=projected, drafts=drafts,
                                kv=runner.mtp_kv_cache.clone(), cache=runner._mtp_cache_lens[0],
                                last=runner._mtp_last_hidden[0].clone()))
        old, new = results
        assert [n for n, logits in new["head"]] == [1], "nonfinal chunks must skip LM head; final projects one row"
        assert [n for n, logits in old["head"]] == ([127, 2] if chunked else [129])
        assert old["drafts"] == new["drafts"]
        assert torch.isfinite(new["head"][0][1]).all()
        assert old["head"][-1][1].argmax() == new["head"][0][1].argmax()
        for a, b in zip(old["hidden"], new["hidden"]):
            assert torch.isfinite(b).all() and torch.equal(bits(a), bits(b))
        assert old["cache"] == new["cache"] == 129
        assert torch.equal(bits(old["kv"]), bits(new["kv"]))
        assert torch.equal(bits(old["last"]), bits(new["last"]))
    finally:
        runner.remove_request(seq.seq_id)
        reset_context()


@torch.inference_mode()
def test_tail_head_matches_old_completion_states_eos_and_reused_slots(engine):
    runner = engine.model_runner
    reference = prefill_reference()
    kwargs = dict(lengths=(127, 129, 255, 257, 128), budgets=(9, 9, 9, 9, 5),
                  eos=(1, None, 2, None, None), late=True, chunk=128)
    with all_row_head(runner, reference):
        old = batch_case(engine, [(True, False, True, False), (False, True, False, True)], **kwargs)
    new = batch_case(engine, [(True, False, True, False), (False, True, False, True)], **kwargs)
    assert old["tokens"] == new["tokens"]
    assert old["decisions"] == new["decisions"] and old["batches"] == new["batches"]
    assert old["snapshots"].keys() == new["snapshots"].keys()
    for key, (count, tensors) in old["snapshots"].items():
        new_count, actual = new["snapshots"][key]
        assert count == new_count
        for a, b in zip(tensors, actual):
            assert torch.isfinite(b).all() and torch.equal(bits(a), bits(b)), key
    for kind in ("hidden", "logits"):
        assert old[kind].keys() == new[kind].keys()
        for key in old[kind]:
            assert torch.isfinite(new[kind][key]).all()
            assert torch.equal(bits(old[kind][key]), bits(new[kind][key])), (kind, key)
