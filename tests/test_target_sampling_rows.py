# ruff: noqa: F811
"""GPU acceptance: row selection preserves hidden/KV under decode-first mixing."""
from unittest.mock import patch

import pytest
import torch

from bench_mtp_interleaved import execution_mode
from nano_qwen.layers.sampler import Sampler
from test_mtp_batch2 import batch_case
from test_mtp_rollback import rollback_engine as engine, tiny_checkpoint, fp8_checkpoint  # noqa: F401


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_selected_target_rows_preserve_mixed_partial_prefill_lifecycle(engine):
    runner = engine.model_runner
    observed = []
    postprocess = engine.scheduler.postprocess
    def post(seqs, results, prefill):
        observed.append(tuple(s.is_prefill for s in seqs))
        return postprocess(seqs, results, prefill)
    def full_projection(hidden):
        runner._target_hidden = hidden
        logits = runner.model.compute_logits(hidden)
        rows = torch.tensor(runner._sampling_plan.row_indices, device=hidden.device, dtype=torch.int64)
        return logits.index_select(0, rows)
    kwargs = dict(lengths=(63, 257), budgets=(8, 8), eos=(None, None), chunk=128)
    with execution_mode(engine, False, Sampler()), patch.object(engine.scheduler, 'postprocess', post):
        with patch.object(runner, 'project_target_hidden', full_projection):
            old = batch_case(engine, [(False, False)], **kwargs)
        new = batch_case(engine, [(False, False)], **kwargs)
    assert (False, True) in observed
    assert old['tokens'] == new['tokens'] and old['batches'] == new['batches']
    assert old['snapshots'].keys() == new['snapshots'].keys()
    for key, (count, tensors) in old['snapshots'].items():
        new_count, actual = new['snapshots'][key]
        assert count == new_count == 0
        for a, b in zip(tensors, actual):
            assert torch.isfinite(b).all() and torch.equal(a.view(torch.uint8), b.view(torch.uint8)), key
    for key, hidden in old['hidden'].items():
        assert torch.equal(hidden.view(torch.uint8), new['hidden'][key].view(torch.uint8)), key
    assert old['logits'].keys() == new['logits'].keys()
    for key, logits in old['logits'].items():
        actual = new['logits'][key]
        assert torch.isfinite(actual).all() and logits.argmax() == actual.argmax(), key
