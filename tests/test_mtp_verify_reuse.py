"""The experimental backbone must pass strict state gates, including graph replay."""

import torch
import pytest

import test_mtp_cache_integration as lifecycle
from nano_qwen.engine.sequence import Sequence
from nano_qwen.quantization.convert import convert_checkpoint
from nano_qwen.sampling_params import SamplingParams
from nano_qwen.utils.context import get_context, reset_context


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    # All attention projections are block128 compatible for both FP8 schemes.
    return lifecycle.build_checkpoint(
        tmp_path_factory.mktemp("mtp-reuse-source"), head_dim=128
    )


@pytest.fixture(scope="module", params=["per_channel", "block_wise"])
def fp8_checkpoint(request, tiny_checkpoint, tmp_path_factory):
    target = tmp_path_factory.mktemp("mtp-reuse-fp8")
    convert_checkpoint(tiny_checkpoint, target, scheme=request.param)
    return target


@pytest.fixture(scope="module", params=[True, False], ids=["eager", "graph"])
def engine(request, tiny_checkpoint, fp8_checkpoint):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    with lifecycle.hybrid_engine(
        fp8_checkpoint,
        request.param,
        mtp_model=str(tiny_checkpoint),
        mtp_verify_mode="reuse",
    ) as value:
        yield value


@torch.inference_mode()
def test_configured_reuse_batches_dense_work_without_changing_norm_rows(engine):
    runner = engine.model_runner
    seq = Sequence(
        [2] * 129, SamplingParams(temperature=1e-6, max_tokens=8, ignore_eos=True)
    )
    engine.scheduler.add(seq)
    engine.step()
    seq.is_speculative = True
    seq.num_scheduled_tokens = 2
    ids, positions = runner.prepare_prefill([seq])
    rows = []
    handles = [
        layer.mlp.gate_up_proj.register_forward_pre_hook(
            lambda m, args: rows.append(args[0].shape[0])
        )
        for layer in runner.model.model.layers
    ]
    try:
        runner.forward_verify(ids, positions)
        assert rows == [2, 2], (
            "Configured reuse must batch each layer's MLP projections"
        )
        assert get_context().gemm_reference_m is None and get_context().is_verify
    finally:
        for handle in handles:
            handle.remove()
        engine.scheduler.block_manager.deallocate(seq)
        engine.scheduler.running.clear()
        engine.scheduler.in_flight.clear()
        runner.remove_request(seq.seq_id)
        reset_context()


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
        report = diagnose(runner, seq, graph=not runner.enforce_eager)
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
    "length,chunk,pattern,max_tokens,eos_at",
    [
        pytest.param(129, 128, (True,), 9, None, id="accept"),
        pytest.param(129, 128, (False,), 9, None, id="reject"),
        pytest.param(129, 128, (True, True, False), 9, None, id="consecutive"),
        pytest.param(129, 128, (True, False), 9, None, id="alternating-boundary129"),
        *[
            pytest.param(n, 128, (True, False), 9, None, id=f"boundary{n}")
            for n in (127, 128, 255, 256, 257, 511, 512, 513)
        ],
        pytest.param(129, 128, (True,), 9, 1, id="eos-accepted-draft"),
        pytest.param(129, 128, (True,), 9, 2, id="eos-bonus"),
        pytest.param(129, 128, (False,), 9, 1, id="eos-rejected-draft"),
        *[pytest.param(129, 128, (True,), n, None, id=f"budget{n}") for n in (1, 2, 3)],
    ],
)
@torch.inference_mode()
def test_reused_verify_preserves_forced_decision_lifecycle(
    engine, length, chunk, pattern, max_tokens, eos_at
):
    runner = engine.model_runner

    def select(mode):
        runner.config.mtp_verify_mode = mode
        if not runner.enforce_eager:
            runner.cuda_graphs._capture_verify()

    select("serial")
    reference = lifecycle.run_case(
        engine, length, chunk, pattern=pattern, max_tokens=max_tokens, eos_at=eos_at
    )
    try:
        select("reuse")
        actual = lifecycle.run_case(
            engine, length, chunk, pattern=pattern, max_tokens=max_tokens, eos_at=eos_at
        )
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
        select("reuse")
        reset_context()
