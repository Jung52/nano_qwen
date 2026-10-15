"""Real hybrid target/MTP kernels, with controlled logits to force decisions.

These tests validate caches and state, not model quality or performance.
Only the target logits and proposed tokens are controlled. Model forwards,
attention, GDN, scheduler, sampling, rollback and graph replay are real.
"""

from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoConfig, PreTrainedTokenizerFast

from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.engine.model_runner import ModelRunner
from nano_qwen.engine.sequence import Sequence
from nano_qwen.layers.attention import Attention
from nano_qwen.models.qwen3_5 import Qwen3_5ForCausalLM
from nano_qwen.models.qwen3_5_mtp import Qwen3_5MTP
from nano_qwen.sampling_params import SamplingParams
from nano_qwen.utils.context import reset_context, set_context


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("mtp-hybrid")
    config = AutoConfig.for_model("qwen3_5_text")
    for name, value in dict(hidden_size=128, intermediate_size=256, vocab_size=256,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=64, linear_num_key_heads=1, linear_num_value_heads=2,
            linear_key_head_dim=128, linear_value_head_dim=128,
            max_position_embeddings=1024, mtp_num_hidden_layers=1,
            layer_types=["linear_attention", "full_attention"],
            dtype=torch.bfloat16).items():
        setattr(config, name, value)
    config.save_pretrained(path)
    torch.manual_seed(614)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with patch.object(dist, "get_rank", return_value=0), \
             patch.object(dist, "get_world_size", return_value=1), \
             patch.object(dist, "is_initialized", return_value=True):
            target = Qwen3_5ForCausalLM(config)
            mtp = Qwen3_5MTP(config)
        tensors = {}
        for prefix, module in [("", target), ("mtp.", mtp)]:
            for name, parameter in module.named_parameters():
                parameter.data.normal_(std=.02)
                name = prefix + name
                if ".gate_up_proj." in name:
                    a, b = parameter.detach().chunk(2, dim=0)
                    tensors[name.replace("gate_up_proj", "gate_proj")] = a.clone()
                    tensors[name.replace("gate_up_proj", "up_proj")] = b.clone()
                else:
                    tensors[name] = parameter.detach().clone()
        save_file(tensors, str(path / "model.safetensors"))
    finally:
        torch.set_default_dtype(old_dtype)
    vocab = {"<unk>": 0, "<eos>": 1, **{f"t{i}": i for i in range(2, 256)}}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
                                        unk_token="<unk>", eos_token="<eos>")
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="module", params=[True, False], ids=["eager", "graph"])
@torch.inference_mode()
def engine(request, tiny_checkpoint):
    observed = SimpleNamespace(latest=None, captured=None)
    original_capture = ModelRunner.capture_cudagraph

    def capture(runner):
        handle = runner.mtp.register_forward_hook(lambda m, args, output: setattr(observed, "latest", output))
        try:
            original_capture(runner)
            observed.captured = observed.latest
        finally:
            handle.remove()

    def small_cache(runner):
        # Replace only allocation sizing, keeping the real paged cache format.
        config = runner.config
        config.num_kvcache_blocks = 8
        attentions = [m for m in runner.model.modules() if isinstance(m, Attention)]
        shape = (8, 256, config.hf_config.num_key_value_heads, config.hf_config.head_dim)
        runner.kv_cache = torch.empty(2, len(attentions), *shape, device="cuda", dtype=torch.bfloat16)
        for i, attention in enumerate(attentions):
            attention.k_cache, attention.v_cache = runner.kv_cache[0, i], runner.kv_cache[1, i]
        runner.mtp_kv_cache = torch.empty(2, *shape, device="cuda", dtype=torch.bfloat16)
        attention = runner.mtp.layers[0].self_attn.attn
        attention.k_cache, attention.v_cache = runner.mtp_kv_cache

    with torch.inference_mode(), patch.object(ModelRunner, "allocate_kv_cache", small_cache), \
         patch.object(ModelRunner, "capture_cudagraph", capture):
        value = LLMEngine(str(tiny_checkpoint), max_num_seqs=1, max_num_batched_tokens=1024,
                          max_model_len=1024, enable_mtp=True, enforce_eager=request.param,
                          use_prefill_cudagraph=False)
    handle = value.model_runner.mtp.register_forward_hook(
        lambda m, args, output: setattr(observed, "latest", output))
    value._mtp_test_observed = observed
    try:
        yield value
    finally:
        handle.remove()
        value.exit()


def valid_cache(cache, blocks, count):
    indices = torch.tensor([blocks[p // 256] * 256 + p % 256 for p in range(count)], device="cuda", dtype=torch.long)
    return cache.flatten(0, 1).index_select(0, indices).clone()


def close_state(actual, expected):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.003)


@torch.inference_mode()
def run_case(engine, length, chunk, pattern=None, max_tokens=9, eos_at=None):
    runner = engine.model_runner
    original_mtp = runner.mtp
    runner.mtp = original_mtp if pattern is not None else None
    engine.scheduler.enable_mtp = pattern is not None
    engine.scheduler.max_num_batched_tokens = chunk
    runner.mtp_kv_cache.fill_(float("nan"))
    prompt = [2] * length
    seq = Sequence(prompt, SamplingParams(temperature=1e-6, max_tokens=max_tokens,
                                          ignore_eos=eos_at is None))
    result = SimpleNamespace(hidden={}, snapshots={}, proposals=[], tokens=None, decisions=[])

    def token(position):
        return 1 if position - length == eos_at else 2 + position % 10

    original_run = runner.run_model
    original_post = engine.scheduler.postprocess
    original_schedule = engine.scheduler.schedule
    original_prefill = runner._run_mtp_prefill
    original_decode = runner._run_mtp_decode

    def run(ids, positions, prefill, verify=False):
        logits = original_run(ids, positions, prefill, verify)
        for position, hidden in zip(positions.tolist(), runner._target_hidden):
            result.hidden[position] = hidden.clone()
        # Control only the branch decision; all hidden/cache computations above are real.
        controlled = torch.zeros_like(logits)
        predicted = torch.tensor([token(p + 1) for p in positions.tolist()], device=logits.device)
        controlled.scatter_(1, predicted.unsqueeze(1), 10)
        return controlled

    def prefill(*args):
        value = original_prefill(*args)
        if value is not None:
            result.proposals.append((runner._mtp_cache_lens[0], engine._mtp_test_observed.latest[-1:].clone()))
        return value

    def decode(*args, **kwargs):
        value = original_decode(*args, **kwargs)
        if not kwargs.get("cache_only", False):
            observed = engine._mtp_test_observed
            hidden = observed.captured if runner.cuda_graphs.mtp_decode_graph is not None else observed.latest
            result.proposals.append((runner._mtp_cache_lens[0], hidden.clone()))
        return value

    def schedule():
        if pattern is not None and engine.scheduler.running and seq.draft_token is not None:
            accept = pattern[len(result.decisions) % len(pattern)]
            expected = token(seq.num_tokens)
            seq.draft_token = expected if accept else (expected + 1) % 256
        rows, prefill = original_schedule()
        if rows and seq.is_speculative:
            result.decisions.append(accept)
        return rows, prefill

    def post(rows, outputs, prefill):
        prefix = seq.num_cached_tokens + seq.num_scheduled_tokens
        if seq.is_speculative and len(outputs[0]) == 1:
            prefix -= 1
        blocks = list(seq.block_table)
        mtp_count = runner._mtp_cache_lens.get(0, 0)
        result.snapshots[prefix] = dict(
            gdn=[(m.conv_states[0].clone(), m.recurrent_states[0].clone()) for m in runner.gdn_layers],
            target=[(valid_cache(m.k_cache, blocks, prefix), valid_cache(m.v_cache, blocks, prefix))
                    for m in runner.model.modules() if isinstance(m, Attention)],
            mtp_count=mtp_count,
            mtp=(valid_cache(runner.mtp_kv_cache[0], blocks, mtp_count),
                 valid_cache(runner.mtp_kv_cache[1], blocks, mtp_count)))
        original_post(rows, outputs, prefill)

    runner.run_model, runner._run_mtp_prefill, runner._run_mtp_decode = run, prefill, decode
    engine.scheduler.schedule, engine.scheduler.postprocess = schedule, post
    engine.scheduler.add(seq)
    try:
        for _ in range(length + max_tokens + 10):
            engine.step()
            if seq.is_finished:
                break
        assert seq.is_finished
        result.tokens = list(seq.completion_token_ids)
        assert runner._mtp_cache_lens == {} and runner._mtp_last_hidden == {}
        return result
    finally:
        runner.run_model = original_run
        runner._run_mtp_prefill, runner._run_mtp_decode = original_prefill, original_decode
        engine.scheduler.schedule, engine.scheduler.postprocess = original_schedule, original_post
        runner.mtp = original_mtp


@torch.inference_mode()
def compare_reference(engine, baseline, actual, length):
    assert actual.tokens == baseline.tokens
    for prefix, snapshot in actual.snapshots.items():
        reference = baseline.snapshots[prefix]
        for kind in ("gdn", "target"):
            for aa, bb in zip(snapshot[kind], reference[kind]):
                for a, b in zip(aa, bb):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
    runner = engine.model_runner
    count = max(s["mtp_count"] for s in actual.snapshots.values())
    attention = runner.mtp.layers[0].self_attn.attn
    caches = attention.k_cache, attention.v_cache
    scratch = torch.full_like(runner.mtp_kv_cache, float("nan"))
    attention.k_cache, attention.v_cache = scratch
    cu = torch.tensor([0, count], dtype=torch.int32, device="cuda")
    try:
        set_context(True, cu_seqlens_q=cu, cu_seqlens_k=cu,
                    max_seqlen_q=count, max_seqlen_k=count,
                    slot_mapping=torch.arange(count, dtype=torch.int32, device="cuda"),
                    prefill_slices=[(0, count)])
        tokens = [2] * length + baseline.tokens
        hidden = runner.mtp(torch.tensor(tokens[1:count + 1], device="cuda"),
                            torch.arange(1, count + 1, device="cuda"),
                            torch.stack([actual.hidden[p] for p in range(count)]),
                            runner.model.model.embed_tokens)
        for prefix, observed in actual.proposals:
            close_state(observed, hidden[prefix - 1:prefix])
        for snapshot in actual.snapshots.values():
            for i, observed in enumerate(snapshot["mtp"]):
                close_state(observed, scratch[i].flatten(0, 1)[:snapshot["mtp_count"]])
    finally:
        attention.k_cache, attention.v_cache = caches
        reset_context()


@pytest.mark.parametrize("length", [127, 128, 129, 255, 256, 257, 511, 512, 513])
def test_real_cache_lifecycle_full_chunked_and_forced_decisions(engine, length):
    observed = {}
    for chunk in (1024, 128):
        baseline = run_case(engine, length, chunk)
        for pattern in ([True], [False], [True, False], [False, True]):
            actual = run_case(engine, length, chunk, pattern)
            compare_reference(engine, baseline, actual, length)
            observed[chunk, tuple(pattern)] = actual
    for pattern in ([True], [False], [True, False], [False, True]):
        full = observed[1024, tuple(pattern)]
        chunked = observed[128, tuple(pattern)]
        assert full.tokens == chunked.tokens
        for prefix in full.snapshots.keys() & chunked.snapshots.keys():
            a, b = full.snapshots[prefix], chunked.snapshots[prefix]
            assert a["mtp_count"] == b["mtp_count"]
            for kind in ("gdn", "target", "mtp"):
                aa = a[kind] if kind != "mtp" else [a[kind]]
                bb = b[kind] if kind != "mtp" else [b[kind]]
                for rows_a, rows_b in zip(aa, bb):
                    for x, y in zip(rows_a, rows_b):
                        # Different prefill GEMM/chunk shapes have BF16
                        # rounding differences. Compare their normalized
                        # state error; same-shape decode above is bitwise.
                        assert torch.isfinite(x).all() and torch.isfinite(y).all()
                        error = (x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-8)
                        assert error < .01, (kind, prefix, error.item())


@pytest.mark.parametrize("max_tokens,eos_at", [(1, None), (2, None), (4, 0), (4, 1), (4, 2), (4, 3)])
def test_real_eos_budget_and_slot_reuse(engine, max_tokens, eos_at):
    baseline = run_case(engine, 129, 128, max_tokens=max_tokens, eos_at=eos_at)
    actual = run_case(engine, 129, 128, [True, False], max_tokens, eos_at)
    compare_reference(engine, baseline, actual, 129)


def test_real_one_token_prefill_chunks_keep_hidden_and_cache(engine):
    baseline = run_case(engine, 129, 1, max_tokens=4)
    actual = run_case(engine, 129, 1, [True], max_tokens=4)
    compare_reference(engine, baseline, actual, 129)


@pytest.mark.parametrize("position", [1, 128, 129, 256, 257, 512, 513])
@torch.inference_mode()
def test_cache_only_matches_full_kv_and_next_prediction_without_unused_work(engine, position):
    runner = engine.model_runner
    seq = Sequence([2] * (position + 1))
    seq.block_table = [6, 2, 5]
    runner.input_batch.update([seq])
    torch.manual_seed(position)
    runner.mtp_kv_cache.normal_(std=.02)
    initial = runner.mtp_kv_cache.clone()
    hidden = torch.randn(2, 128, dtype=torch.bfloat16, device="cuda")
    tokens = torch.tensor([2, 3], device="cuda")
    runner._mtp_cache_lens[0] = position - 1
    runner._run_mtp_decode([seq], hidden[:1], tokens[:1], [position])
    expected_cache = runner.mtp_kv_cache.clone()
    expected_draft = runner._run_mtp_decode([seq], hidden[1:], tokens[1:], [position + 1]).clone()
    runner.mtp_kv_cache.copy_(initial)
    runner._mtp_cache_lens[0] = position - 1
    layer = runner.mtp.layers[0]
    forbidden = [layer.self_attn.q_proj, layer.self_attn.q_norm,
                 layer.self_attn.attn, layer.self_attn.o_proj, layer.mlp, runner.mtp.norm]
    try:
        with ExitStack() as stack:
            for module in forbidden:
                stack.enter_context(patch.object(module, "forward", side_effect=AssertionError("unused MTP work")))
            stack.enter_context(patch.object(runner.model, "compute_logits", side_effect=AssertionError("unused LM head")))
            stack.enter_context(patch.object(runner.cuda_graphs, "run_mtp_decode", side_effect=AssertionError("full MTP graph")))
            result = runner._run_mtp_decode([seq], hidden[:1], tokens[:1], [position], cache_only=True)
        assert result is None
        assert runner._mtp_cache_lens[0] == position
        torch.testing.assert_close(runner.mtp_kv_cache, expected_cache, rtol=0, atol=0)
        actual = runner._run_mtp_decode([seq], hidden[1:], tokens[1:], [position + 1])
        torch.testing.assert_close(actual, expected_draft, rtol=0, atol=0)
    finally:
        runner.remove_request(seq.seq_id)
        reset_context()


@torch.inference_mode()
def test_preemption_replay_matches_fresh_request_state(engine):
    runner, scheduler = engine.model_runner, engine.scheduler
    scheduler.max_num_batched_tokens = 1024
    original_post = scheduler.postprocess
    snapshots = []

    def post(rows, outputs, prefill):
        seq = rows[0]
        count = seq.num_cached_tokens + seq.num_scheduled_tokens
        slot = runner.input_batch.slots_for(rows)[0]
        mtp_count = runner._mtp_cache_lens[slot]
        snapshots.append((list(outputs[0]),
            [(m.conv_states[slot].clone(), m.recurrent_states[slot].clone()) for m in runner.gdn_layers],
            [(valid_cache(m.k_cache, seq.block_table, count), valid_cache(m.v_cache, seq.block_table, count))
             for m in runner.model.modules() if isinstance(m, Attention)],
            [valid_cache(cache, seq.block_table, mtp_count) for cache in runner.mtp_kv_cache]))
        original_post(rows, outputs, prefill)

    seq = Sequence([2] * 129, SamplingParams(temperature=1e-6, max_tokens=8, ignore_eos=True))
    scheduler.postprocess = post
    try:
        scheduler.add(seq)
        engine.step()
        history = list(seq.token_ids)
        old_slot = runner.input_batch.slots_for([seq])[0]
        scheduler.running.remove(seq)
        scheduler.preempt(seq)
        assert seq.num_cached_tokens == 0
        engine.step()
        assert runner.input_batch.slots_for([seq])[0] == old_slot
        replayed = snapshots[-1]
        scheduler.running.remove(seq)
        scheduler.block_manager.deallocate(seq)
        runner.remove_request(seq.seq_id)
        fresh = Sequence(history, SamplingParams(temperature=1e-6, max_tokens=1, ignore_eos=True))
        scheduler.add(fresh)
        engine.step()
        reference = snapshots[-1]
        assert replayed[0] == reference[0]
        for actual, expected in zip(replayed[1:], reference[1:]):
            for aa, bb in zip(actual, expected):
                if isinstance(aa, tuple):
                    for a, b in zip(aa, bb):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                else:
                    torch.testing.assert_close(aa, bb, rtol=0, atol=0)
    finally:
        scheduler.postprocess = original_post
        for active in list(scheduler.running):
            scheduler.block_manager.deallocate(active)
            runner.remove_request(active.seq_id)
        scheduler.running.clear()
        scheduler.in_flight.clear()


@torch.inference_mode()
def test_verify_logits_and_state_match_two_ordinary_decode_steps_exactly(engine):
    runner = engine.model_runner
    seq = Sequence([2] * 129, SamplingParams(temperature=1e-6, max_tokens=8, ignore_eos=True))
    engine.scheduler.add(seq)
    engine.step()
    # A realistic projection width exercises the M=1 GEMV / M=2 GEMM
    # arithmetic difference that the tiny fixture's 128-wide head hides.
    original_logits = runner.model.compute_logits
    torch.manual_seed(5)
    weight = torch.randn(4096, 2560, device="cuda", dtype=torch.bfloat16) * .02
    runner.model.compute_logits = lambda hidden: torch.nn.functional.linear(hidden.repeat(1, 20), weight)
    seq.append_token(42)
    seq.is_speculative = True
    seq.num_scheduled_tokens = 2
    kv = runner.kv_cache.clone()
    states = [(m.conv_states.clone(), m.recurrent_states.clone()) for m in runner.gdn_layers]
    ids, positions = runner.prepare_prefill([seq])
    from nano_qwen.utils.context import get_context
    context = get_context()
    try:
        actual = runner.run_model(ids, positions, True, True).clone()
        actual_states = [(m.conv_states.clone(), m.recurrent_states.clone()) for m in runner.gdn_layers]
        actual_kv = runner.kv_cache.clone()
        runner.kv_cache.copy_(kv)
        for m, (conv, recurrent) in zip(runner.gdn_layers, states):
            m.conv_states.copy_(conv)
            m.recurrent_states.copy_(recurrent)
        expected = []
        for row in range(2):
            set_context(False, slot_mapping=context.slot_mapping[row:row + 1],
                        context_lens=context.cu_seqlens_k[1:] - (1 - row),
                        block_tables=context.block_tables, state_indices=context.state_indices)
            expected.append(runner.run_model(ids[row:row + 1], positions[row:row + 1], False).clone())
        torch.testing.assert_close(actual, torch.cat(expected), rtol=0, atol=0)
        for m, (conv, recurrent) in zip(runner.gdn_layers, actual_states):
            torch.testing.assert_close(m.conv_states, conv, rtol=0, atol=0)
            torch.testing.assert_close(m.recurrent_states, recurrent, rtol=0, atol=0)
        torch.testing.assert_close(runner.kv_cache, actual_kv, rtol=0, atol=0, equal_nan=True)
    finally:
        runner.model.compute_logits = original_logits
        engine.scheduler.block_manager.deallocate(seq)
        engine.scheduler.running.clear()
        engine.scheduler.in_flight.clear()
        runner.remove_request(seq.seq_id)
        reset_context()
