"""CLI and admission checks for larger performance workloads (no GPU needed)."""

from types import SimpleNamespace

import pytest

import validate_fp8_model as benchmark


def performance_args(*extra):
    return benchmark.parse_args([
        "--model", "/unused/model", "--suite", "performance",
        "--json-out", "/unused/report.json", *extra,
    ])


def test_default_performance_workloads_stay_compatible():
    args = performance_args()
    assert args.workloads == [(128, 1), (128, 4), (128, 8), (512, 1), (1024, 1), (512, 4)]
    assert (args.max_num_seqs, args.max_model_len, args.max_num_batched_tokens) == (8, 2048, 512)
    assert args.gpu_memory_utilization == 0.65
    assert not args.chunked_prefill


def test_stress_workloads_size_engine_for_prompt_and_all_completion_tokens():
    args = performance_args(
        "--workloads", "4096:32", "8192:16", "--decode-tokens", "256",
        "--chunked-prefill", "--gpu-memory-utilization", "0.85",
    )
    assert args.workloads == [(4096, 32), (8192, 16)]
    assert args.max_num_seqs == 32
    assert args.max_model_len >= 8192 + 257
    assert args.chunked_prefill


@pytest.mark.parametrize("extra", [
    ["--workloads", "4096"],
    ["--workloads", "0:8"],
    ["--workloads", "2048:-1"],
    ["--workloads", "4096:16", "--max-num-seqs", "8"],
    ["--workloads", "4096:8", "--max-model-len", "4096"],
    ["--gpu-memory-utilization", "nan"],
    ["--gpu-memory-utilization", "1.1"],
    ["--max-num-batched-tokens", "0"],
])
def test_invalid_stress_settings_fail_before_loading_model(extra):
    with pytest.raises(SystemExit) as caught:
        performance_args(*extra)
    assert caught.value.code == 2


def test_kv_shortfall_explains_capacity_before_enqueuing_requests():
    engine = SimpleNamespace(config=SimpleNamespace(
        kvcache_block_size=256, num_kvcache_blocks=575,
        max_num_seqs=32, max_model_len=8192,
    ), is_finished=lambda: True)
    with pytest.raises(ValueError, match=r"need 576.*have 575"):
        benchmark.fixed_batch(engine, [4]*4096, 32, 257)


def test_checkpoint_context_limit_is_checked_after_engine_config_clamps_it():
    engine = SimpleNamespace(config=SimpleNamespace(
        kvcache_block_size=256, num_kvcache_blocks=1024,
        max_num_seqs=32, max_model_len=4096,
    ), is_finished=lambda: True)
    with pytest.raises(ValueError, match="max_model_len"):
        benchmark.fixed_batch(engine, [4]*4096, 8, 257)


def test_remote_perf_cases_alias_keeps_profile_selection():
    args = performance_args('--perf-cases', '512:4', '1024:2')
    assert args.workloads == [(512, 4), (1024, 2)]
    assert args.perf_cases == args.workloads
    assert performance_args('--perf-profile', 'batch').max_num_seqs == 64


def test_chunked_performance_uses_scheduler_window_and_labels_variable_batch(monkeypatch, tmp_path):
    args = performance_args('--workloads', '8:2', '--chunked-prefill', '--decode-tokens', '2')
    args.json_out, args.warmup_rounds, args.rounds = str(tmp_path/'report.json'), 1, 2
    engine = SimpleNamespace(config=SimpleNamespace(kvcache_block_size=256, num_kvcache_blocks=2,
                          max_num_seqs=2, max_model_len=256, enable_prefix_cache=False),
                          tokenizer=SimpleNamespace(encode=lambda *a, **kw: [4, 5]))
    calls = []
    def chunked(engine, prompts, tokens):
        calls.append((prompts, tokens))
        return dict(outputs=['done', 'done'], ttft_ms=2., decode_tokens_per_s=100., e2e_ms=5.)
    def forbid_fixed(*args):
        raise AssertionError('Chunked prefill must use production scheduling')
    monkeypatch.setattr(benchmark, 'batch', chunked)
    monkeypatch.setattr(benchmark, 'fixed_batch', forbid_fixed)
    monkeypatch.setattr(benchmark.torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(benchmark.torch.cuda, 'max_memory_allocated', lambda: 2**20)
    monkeypatch.setattr(benchmark.torch.cuda, 'max_memory_reserved', lambda: 2**20)
    report = {}
    result = benchmark.evaluate_performance(engine, args, report)
    assert len(calls) == 3 and len(result['workloads'][0]['runs']) == 2
    assert 'active decode batch may vary' in result['protocol']
    assert result['prefill_mode'] == 'chunked'
    assert result['workloads'][0]['decode_step_cv_percent'] is None
