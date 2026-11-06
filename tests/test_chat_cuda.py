"""Real tiny hybrid kernels: streaming async/serial equality and Graph dispatch.

Requires the repository's normal CUDA environment; no full model download.
The checkpoint fixture is shared with the existing main lifecycle tests.
"""
from unittest.mock import patch

import pytest
import torch

from test_main_lifecycle import source_checkpoint  # noqa: F401
from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.engine.model_runner import ModelRunner
from nano_qwen.layers.attention import Attention
from nano_qwen.sampling_params import SamplingParams


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def stream_engine(source_checkpoint):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    def small_cache(runner):
        config = runner.config
        config.num_kvcache_blocks = 8
        attentions = [m for m in runner.model.modules() if isinstance(m, Attention)]
        runner.kv_cache = torch.empty(
            2, len(attentions), 8, 256, 1, 128, dtype=torch.bfloat16, device="cuda",
        )
        for i, attention in enumerate(attentions):
            attention.k_cache = runner.kv_cache[0, i]
            attention.v_cache = runner.kv_cache[1, i]

    with torch.inference_mode(), patch.object(ModelRunner, "allocate_kv_cache", small_cache):
        engine = LLMEngine(
            str(source_checkpoint), max_num_seqs=1, max_num_batched_tokens=64,
            max_model_len=1024, enforce_eager=False, async_scheduling=True,
            use_prefill_cudagraph=False,
        )
    try:
        yield engine
    finally:
        engine.exit()


@pytest.mark.parametrize("prompt_length", [63, 255, 257])
@torch.inference_mode()
def test_real_graph_stream_async_matches_serial_tokens_and_gdn_state(stream_engine, prompt_length):
    engine = stream_engine
    runner = engine.model_runner
    assert set(runner.cuda_graphs.decode_graphs) == {1}
    results = []
    for enabled in (False, True):
        engine.config.async_scheduling = enabled
        runner.cuda_graphs.reset_dispatch_stats()
        chunks = list(engine.generate_stream(
            [2] * prompt_length,
            SamplingParams(temperature=1e-6, max_tokens=9, ignore_eos=True),
        ))
        tokens = [t for c in chunks for t in c.token_ids]
        assert len(tokens) == 9 and chunks[-1].finish_reason == "length"
        assert sum(c.finished for c in chunks) == 1
        assert engine.is_finished() and not engine.batch_queue
        assert engine.scheduler.async_decode is None
        assert not runner.input_batch.seq_id_to_slot
        assert len(engine.scheduler.block_manager.free_block_ids) == 8
        dispatch = runner.cuda_graphs.dispatch_stats()["decode"]
        assert dispatch["graph_calls"] == 8 and dispatch["eager_calls"] == 0
        state = [t.clone() for layer in runner.gdn_layers
                 for t in (layer.conv_states, layer.recurrent_states)]
        results.append((tokens, state))
    assert results[0][0] == results[1][0]
    for a, b in zip(results[0][1], results[1][1]):
        assert torch.isfinite(b).all()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
