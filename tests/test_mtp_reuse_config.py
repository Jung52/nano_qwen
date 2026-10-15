"""An opt-in reuse mode must reject unsupported or misspelled settings."""

import pytest

from nano_qwen.config import Config
from test_mtp_weights import checkpoint_pair


def test_reuse_is_explicit_and_serial_remains_default(tmp_path):
    target, source = checkpoint_pair(tmp_path)
    serial = Config(str(target), enable_mtp=True, max_num_seqs=1, mtp_model=str(source))
    reuse = Config(
        str(target),
        enable_mtp=True,
        max_num_seqs=1,
        mtp_model=str(source),
        mtp_verify_mode="reuse",
    )
    assert serial.mtp_verify_mode == "serial"
    assert reuse.mtp_verify_mode == "reuse"


@pytest.mark.parametrize("mode", ["auto", "resue", "", None])
def test_unknown_verify_mode_is_not_silently_ignored(tmp_path, mode):
    target, _ = checkpoint_pair(tmp_path)
    with pytest.raises(ValueError, match="mtp_verify_mode"):
        Config(str(target), mtp_verify_mode=mode)


def test_reuse_requires_mtp_enabled(tmp_path):
    target, _ = checkpoint_pair(tmp_path)
    with pytest.raises(ValueError, match="enable_mtp"):
        Config(str(target), max_num_seqs=1, mtp_verify_mode="reuse")


def test_reuse_requires_fp8_target(tmp_path):
    _, source = checkpoint_pair(tmp_path)
    with pytest.raises(ValueError, match="FP8"):
        Config(str(source), enable_mtp=True, max_num_seqs=1, mtp_verify_mode="reuse")


@pytest.mark.parametrize("kwargs", [{"max_num_seqs": 16}, {"tensor_parallel_size": 2}])
def test_reuse_preserves_small_batch_tp1_boundary(tmp_path, kwargs):
    target, _ = checkpoint_pair(tmp_path)
    options = dict(enable_mtp=True, max_num_seqs=1, mtp_verify_mode="reuse")
    options.update(kwargs)
    with pytest.raises(
        (ValueError, AssertionError), match="(max_num_seqs|tensor_parallel_size)"
    ):
        Config(str(target), **options)


def test_production_cli_selects_reuse_without_changing_off_mode(monkeypatch, capsys):
    import sys
    import bench_qwen3_5_mtp_production as bench

    monkeypatch.setattr(bench, "make_prompt", lambda *args: [2, 3])
    calls = []

    def run(model, enabled, *args, **kwargs):
        calls.append((enabled, kwargs.get("mtp_verify_mode", "serial")))
        return dict(
            tokens=2,
            elapsed_s=1.0,
            tokens_per_s=2.0,
            stop_reason="max_tokens",
            windows=[],
            accepted_tokens=1,
            verify_steps=1,
        ), [2, 3, 4]

    monkeypatch.setattr(bench, "run_mode", run)
    monkeypatch.setattr(
        sys, "argv", ["bench", "--model", "unused", "--mtp-verify-mode", "reuse"]
    )
    bench.main()
    assert calls == [(False, "serial"), (True, "reuse")]
    assert "RESULT speedup=" in capsys.readouterr().out


def test_changing_mode_does_not_silently_replay_a_different_verify_graph():
    from types import SimpleNamespace
    from nano_qwen.engine.cuda_graph import CudaGraphManager

    runner = SimpleNamespace(config=SimpleNamespace(mtp_verify_mode="serial"))
    graphs = CudaGraphManager(runner)
    graphs.verify_graph_mode = "reuse"
    with pytest.raises(RuntimeError, match="(mode|recapture)"):
        graphs.run_verify(None, None)
