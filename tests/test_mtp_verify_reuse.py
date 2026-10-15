"""The experimental backbone must pass strict state gates, including graph replay."""

import torch
import pytest
from unittest.mock import patch

import test_mtp_cache_integration as lifecycle
from nano_qwen.engine.sequence import Sequence
from nano_qwen.quantization.convert import convert_checkpoint
from nano_qwen.sampling_params import SamplingParams
from nano_qwen.utils.context import reset_context


tiny_checkpoint = lifecycle.tiny_checkpoint


@pytest.fixture(scope="module")
def fp8_checkpoint(tiny_checkpoint, tmp_path_factory):
    target = tmp_path_factory.mktemp("mtp-reuse-fp8")
    convert_checkpoint(tiny_checkpoint, target, scheme="per_channel")
    return target


@pytest.fixture(scope="module", params=[True, False], ids=["eager", "graph"])
def engine(request, tiny_checkpoint, fp8_checkpoint):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    with lifecycle.hybrid_engine(
        fp8_checkpoint, request.param, mtp_model=str(tiny_checkpoint)
    ) as value:
        yield value


def test_strict_gate_rejects_matching_nonfinite_values():
    from bench_mtp_verify_reuse import difference, first_failure

    finite = difference(torch.tensor([1.0]), torch.tensor([1.0]))
    same_inf = difference(torch.tensor([float("inf")]), torch.tensor([float("inf")]))
    assert first_failure([finite]) is None
    assert same_inf["equal"]
    assert first_failure([finite, same_inf]) is same_inf


@torch.inference_mode()
def test_reused_verify_matches_serial_layers_logits_kv_and_snapshots(engine):
    from bench_mtp_verify_reuse import diagnose

    runner = engine.model_runner
    seq = Sequence(
        [2] * 129, SamplingParams(temperature=1e-6, max_tokens=8, ignore_eos=True)
    )
    engine.scheduler.add(seq)
    engine.step()
    seq.append_token(seq.draft_token)
    seq.is_speculative = True
    seq.num_scheduled_tokens = 2
    norm_rows = []
    norms = [runner.model.model.norm]
    for layer in runner.model.model.layers:
        norms.extend([layer.input_layernorm, layer.post_attention_layernorm])
    handles = [
        module.register_forward_pre_hook(
            lambda m, args: norm_rows.append(args[0].shape[0])
        )
        for module in norms
    ]
    try:
        report = diagnose(runner, seq, pin_tiles=True, graph=not runner.enforce_eager)
        assert report["strict_pass"], report["first_difference"]
        assert set(norm_rows) == {1}
    finally:
        for handle in handles:
            handle.remove()
        engine.scheduler.block_manager.deallocate(seq)
        engine.scheduler.running.clear()
        engine.scheduler.in_flight.clear()
        runner.remove_request(seq.seq_id)
        reset_context()


@pytest.mark.parametrize(
    "pattern", [(True,), (False,), (True, True, False), (True, False)]
)
@torch.inference_mode()
def test_reused_verify_preserves_forced_decision_lifecycle(engine, pattern):
    from bench_mtp_verify_reuse import b1_tiles, forward_reused

    runner = engine.model_runner
    reference = lifecycle.run_case(engine, 129, 128, pattern=pattern)
    try:
        with (
            patch.object(
                runner,
                "forward_verify",
                lambda ids, pos: forward_reused(runner, ids, pos),
            ),
            b1_tiles(),
        ):
            if not runner.enforce_eager:
                runner.cuda_graphs._capture_verify()
            actual = lifecycle.run_case(engine, 129, 128, pattern=pattern)
        assert actual.tokens == reference.tokens
        assert actual.decisions == reference.decisions
        assert actual.snapshots.keys() == reference.snapshots.keys()
        for prefix, state in reference.snapshots.items():
            other = actual.snapshots[prefix]
            assert state["mtp_count"] == other["mtp_count"]
            for kind in ("gdn", "target", "mtp"):
                pairs = zip(state[kind], other[kind])
                for left, right in pairs:
                    if isinstance(left, torch.Tensor):
                        assert torch.equal(left, right), (prefix, kind)
                    else:
                        for a, b in zip(left, right):
                            assert torch.equal(a, b), (prefix, kind)
    finally:
        if not runner.enforce_eager:
            runner.cuda_graphs._capture_verify()
        reset_context()
