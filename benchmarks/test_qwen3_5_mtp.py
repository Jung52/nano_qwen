"""Real 9B MTP smoke/correctness test (TP=1, CUDA).

Install qwen3_5_mtp.py as src/nano_qwen/models/qwen3_5_mtp.py, then run:
    python benchmarks/test_qwen3_5_mtp.py --model /path/to/Qwen3.5-9B

Requires the same CUDA/FlashAttention stack as nano_qwen and enough VRAM for
the 9B target, its MTP layer and the target's KV allocation.
"""

import argparse
import sys
from time import perf_counter
from pathlib import Path

import torch

# Make `from nano_qwen...` work when run as `python benchmarks/<script>.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nano_qwen.engine.llm_engine import LLMEngine  # noqa: E402
from nano_qwen.layers.attention import Attention  # noqa: E402
from nano_qwen.models.qwen3_5_mtp import (  # noqa: E402
    Qwen3_5MTP,
    load_mtp_weights,
)
from nano_qwen.engine.sequence import Sequence  # noqa: E402
from nano_qwen.sampling_params import SamplingParams  # noqa: E402
from nano_qwen.utils.context import reset_context, set_context  # noqa: E402


def prefill_context(n: int, device: torch.device, slot: torch.Tensor | None = None):
    boundaries = torch.tensor([0, n], dtype=torch.int32, device=device)
    set_context(
        True,
        cu_seqlens_q=boundaries,
        cu_seqlens_k=boundaries,
        max_seqlen_q=n,
        max_seqlen_k=n,
        state_indices=slot,
        prefill_slices=[(0, n)],
        prefill_chunk_indices=torch.tensor(
            [(0, i) for i in range((n + 63) // 64)],
            dtype=torch.int32,
            device=device,
        ),
    )


@torch.inference_mode()
def evaluate_mtp2(
    target,
    runner,
    mtp,
    ids: torch.Tensor,
    positions: torch.Tensor,
    target_hidden: torch.Tensor,
    first_token: torch.Tensor,
    embedding,
) -> dict[str, float]:
    # Reference rows are token 0..N. MTP row i consumes target hidden i and
    # token i+1, then predicts token i+2. Compare it with target logits at
    # row i+1, which predict the same token from the full target context.
    reference_ids = torch.cat((ids, first_token))
    reference_positions = torch.arange(
        reference_ids.numel(), dtype=torch.long, device=ids.device
    )
    attentions = [module for module in target.modules() if isinstance(module, Attention)]
    caches = [(module, module.k_cache, module.v_cache) for module in attentions]
    for module, _, _ in caches:
        module.k_cache = torch.empty(0, device=ids.device)
        module.v_cache = torch.empty(0, device=ids.device)
    for layer in runner.gdn_layers:
        layer.reset_state([0])
    try:
        prefill_context(
            reference_ids.numel(),
            ids.device,
            torch.tensor([0], device=ids.device),
        )
        reference_hidden = target(reference_ids, reference_positions)
    finally:
        reset_context()
        for module, k_cache, v_cache in caches:
            module.k_cache = k_cache
            module.v_cache = v_cache
    reference_logits = target.compute_logits(reference_hidden)
    reference_next = reference_logits[1:].argmax(dim=-1)

    draft_ids = reference_ids[1:]
    draft_positions = reference_positions[1:]
    draft_hidden = reference_hidden[:-1]
    prefill_context(draft_ids.numel(), ids.device)
    try:
        mtp_hidden = mtp(draft_ids, draft_positions, draft_hidden, embedding)
        draft_logits = target.compute_logits(mtp_hidden)
    finally:
        reset_context()

    draft_next = draft_logits.argmax(dim=-1)
    top5 = draft_logits.topk(5, dim=-1).indices
    return {
        "samples": int(draft_next.numel()),
        "top1_accuracy": (draft_next == reference_next).float().mean().item(),
        "target_top5_recall": (
            (top5 == reference_next.unsqueeze(-1)).any(dim=-1).float().mean().item()
        ),
    }


def release_sequence(engine, seq: Sequence) -> None:
    if seq.block_table:
        engine.scheduler.block_manager.deallocate(seq)
    engine.scheduler.waiting.clear()
    engine.scheduler.running.clear()
    engine.scheduler.in_flight.clear()
    engine.model_runner.input_batch.remove(seq.seq_id)


@torch.inference_mode()
def benchmark_baseline(engine, prompt: list[int], steps: int, duration: float):
    seq = Sequence(
        prompt,
        SamplingParams(
            temperature=1e-6,
            max_tokens=steps + 1,
            ignore_eos=True,
        ),
    )
    engine.scheduler.add(seq)

    count = 0
    start = deadline = None
    while not seq.is_finished:
        if start is not None and count >= steps:
            break
        if start is not None and duration > 0 and perf_counter() >= deadline:
            break

        t0 = perf_counter()
        _, num_tokens = engine.step()
        elapsed = perf_counter() - t0

        if num_tokens > 0:
            # Exclude prompt prefill and its first sampled token. Throughput
            # below measures only one-token decode calls, matching production.
            start = perf_counter()
            deadline = start + duration
            continue
        if num_tokens < 0:
            count += 1

    if start is None or count == 0:
        raise RuntimeError("benchmark ended before a decode step ran")
    elapsed = perf_counter() - start
    release_sequence(engine, seq)
    torch.cuda.synchronize()
    return {
        "steps": count,
        "elapsed_s": elapsed,
        "tokens_per_s": count / elapsed,
    }


@torch.inference_mode()
def benchmark_mtp2(
    engine,
    target,
    mtp,
    embedding,
    prompt: list[int],
    steps: int,
    duration: float,
    mtp_k_cache: torch.Tensor,
    mtp_v_cache: torch.Tensor,
    mtp_block_tables: torch.Tensor,
):
    runner = engine.model_runner
    original_compute_logits = target.compute_logits

    def capture_hidden_and_compute_logits(hidden):
        runner._benchmark_target_hidden = hidden
        return original_compute_logits(hidden)

    target.compute_logits = capture_hidden_and_compute_logits
    seq = Sequence(
        prompt,
        SamplingParams(
            temperature=1e-6,
            max_tokens=steps + 2,
            ignore_eos=True,
        ),
    )
    engine.scheduler.add(seq)

    mtp_cache_len = 0
    previous_draft = None
    accepted = torch.zeros((), dtype=torch.long, device=mtp_k_cache.device)
    verified = 0
    count = 0
    start = deadline = None
    try:
        while not seq.is_finished:
            if start is not None and count >= steps:
                break
            if start is not None and duration > 0 and perf_counter() >= deadline:
                break

            engine.step()
            if seq.is_prefill:
                pass
            else:
                count += 1
                slot = runner.input_batch.seq_id_to_slot[seq.seq_id]
                current_token = runner.sampled_token_ids_gpu[slot:slot + 1]
                if previous_draft is not None:
                    accepted += (previous_draft == current_token).sum()
                    verified += 1

            if seq.is_prefill:
                start = perf_counter()
                deadline = start + duration

            # Do not draft after the final requested decode: that token has no
            # subsequent verification step in this finite benchmark window.
            if start is None or count >= steps:
                continue

            slot = runner.input_batch.seq_id_to_slot[seq.seq_id]
            input_token = runner.sampled_token_ids_gpu[slot:slot + 1]
            position = seq.num_tokens - 1
            physical_slot = mtp_block_tables[0, position // mtp_k_cache.size(1)]
            physical_slot = physical_slot * mtp_k_cache.size(1) + position % mtp_k_cache.size(1)
            set_context(
                False,
                slot_mapping=physical_slot.view(1),
                context_lens=torch.tensor(
                    [mtp_cache_len + 1], dtype=torch.int32, device=input_token.device
                ),
                block_tables=mtp_block_tables,
                state_indices=None,
            )
            try:
                draft_hidden = mtp(
                    input_token,
                    torch.tensor([position], dtype=torch.long, device=input_token.device),
                    runner._benchmark_target_hidden[-1:],
                    embedding,
                )
                previous_draft = original_compute_logits(draft_hidden).argmax(-1)
                mtp_cache_len += 1
            finally:
                reset_context()
    finally:
        target.compute_logits = original_compute_logits
        reset_context()

    if count == 0:
        raise RuntimeError("MTP benchmark ended before a decode step ran")
    elapsed = perf_counter() - start
    release_sequence(engine, seq)
    accepted_count = int(accepted.item())
    torch.cuda.synchronize()
    return {
        "steps": count,
        "elapsed_s": elapsed,
        "acceptance": accepted_count / max(verified, 1),
        "accepted": accepted_count,
        "verified": verified,
        "target_tokens_per_s": count / elapsed,
        "projected_tokens_per_s": (count + accepted_count) / elapsed,
    }


@torch.inference_mode()
def test_mtp(
    model_path: str,
    prompt_tokens: int,
    benchmark_steps: int,
    benchmark_duration: float,
    enforce_eager: bool,
):
    engine = LLMEngine(
        model_path,
        tensor_parallel_size=1,
        max_num_seqs=1,
        max_num_batched_tokens=max(512, prompt_tokens + 1),
        max_model_len=prompt_tokens + benchmark_steps + 2,
        gpu_memory_utilization=0.85,
        enforce_eager=enforce_eager,
        use_prefill_cudagraph=False,
    )
    try:
        runner = engine.model_runner
        target = runner.model.eval()
        device = target.lm_head.weight.device
        dtype = target.lm_head.weight.dtype
        mtp = Qwen3_5MTP(engine.config.hf_config).to(device=device, dtype=dtype).eval()
        load_mtp_weights(mtp, model_path)  # strict: every `mtp.*` tensor

        seed = engine.tokenizer.encode("Qwen3.5多步预测模型验证。", add_special_tokens=False)
        if not seed:
            raise RuntimeError("Tokenizer returned an empty prompt")
        prompt = (seed * ((prompt_tokens + len(seed) - 1) // len(seed)))[:prompt_tokens]
        ids = torch.tensor(prompt, dtype=torch.long, device=device)
        positions = torch.arange(prompt_tokens, dtype=torch.long, device=device)

        # Runner's Attention owns a paged target KV cache. This test runs a
        # fresh packed prefill, so temporarily turn off its cache writes.
        attentions = [m for m in target.modules() if isinstance(m, Attention)]
        caches = [(m, m.k_cache, m.v_cache) for m in attentions]
        try:
            for m, _, _ in caches:
                m.k_cache = torch.empty(0, device=device)
                m.v_cache = torch.empty(0, device=device)
            for gdn in runner.gdn_layers:
                gdn.reset_state([0])
            prefill_context(prompt_tokens, device, torch.tensor([0], device=device))
            target_hidden = target(ids, positions)
            first_token = target.compute_logits(target_hidden[-1:]).argmax(-1)
        finally:
            reset_context()
            for m, k, v in caches:
                m.k_cache, m.v_cache = k, v

        # Alignment: target_hidden[i] (token i) + input_ids[i+1] predicts i+2.
        shifted_ids = torch.cat((ids[1:], first_token))
        shifted_positions = positions + 1
        assert shifted_ids.shape == ids.shape
        assert shifted_ids[0].item() == prompt[1]
        assert shifted_positions[0].item() == 1
        assert shifted_ids[-1].item() == first_token.item()

        embedding = target.model.embed_tokens
        prefill_context(prompt_tokens, device)
        try:
            all_hidden = mtp(shifted_ids, shifted_positions, target_hidden, embedding)
        finally:
            reset_context()
        draft_token = target.compute_logits(all_hidden[-1:]).argmax(-1).item()

        # A causal MTP prefill and separate prefix forwards must agree at the
        # same position, including both sides of the 64-token boundary.
        lengths = sorted({1, 2, 63, 64, 65, prompt_tokens} & set(range(1, prompt_tokens + 1)))
        for n in lengths:
            prefill_context(n, device)
            try:
                prefix_last = mtp(
                    shifted_ids[:n], shifted_positions[:n],
                    target_hidden[:n], embedding,
                )[-1:]
            finally:
                reset_context()

            expected = all_hidden[n - 1:n].float()
            relative_l2 = (prefix_last.float() - expected).norm(dim=-1)
            relative_l2 = relative_l2 / expected.norm(dim=-1).clamp_min(1e-6)
            # Different prefill lengths can select different bf16 cuBLAS and
            # Triton kernels. An elementwise residual comparison conflates that
            # launch-dependent quantization with a causal semantics error.
            torch.testing.assert_close(
                relative_l2,
                torch.zeros_like(relative_l2),
                atol=2e-2,
                rtol=0.0,
                msg=f"MTP causal prefix mismatch at length {n}",
            )
            a = target.compute_logits(prefix_last).argmax(-1).item()
            b = target.compute_logits(all_hidden[n - 1:n]).argmax(-1).item()
            assert a == b, f"MTP next token mismatch at length {n}: {a} vs {b}"
            print(f"PASS prefix={n} token={a}")

        print(f"PASS weights=complete target_first={first_token.item()} mtp_draft={draft_token}")

        metrics = evaluate_mtp2(
            target,
            runner,
            mtp,
            ids,
            positions,
            target_hidden,
            first_token,
            embedding,
        )
        print(
            "PASS MTP2 alignment "
            f"samples={metrics['samples']} "
            f"top1={metrics['top1_accuracy']:.4%} "
            f"target_top5={metrics['target_top5_recall']:.4%}"
        )

        block_size = engine.config.kvcache_block_size
        num_kv_heads = engine.config.hf_config.num_key_value_heads
        head_dim = engine.config.hf_config.head_dim
        cache_blocks = (
            (prompt_tokens + benchmark_steps + block_size - 1) // block_size
        )
        mtp_k_cache = torch.empty(
            cache_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        mtp_v_cache = torch.empty_like(mtp_k_cache)
        mtp_attention = mtp.layers[0].self_attn.attn
        mtp_attention.k_cache = mtp_k_cache
        mtp_attention.v_cache = mtp_v_cache
        mtp_block_tables = torch.arange(
            cache_blocks, dtype=torch.int32, device=device
        ).unsqueeze(0)

        baseline = benchmark_baseline(engine, prompt, benchmark_steps, benchmark_duration)
        mtp_result = benchmark_mtp2(
            engine,
            target,
            mtp,
            embedding,
            prompt,
            benchmark_steps,
            benchmark_duration,
            mtp_k_cache,
            mtp_v_cache,
            mtp_block_tables,
        )
        projected_speedup = (
            mtp_result["projected_tokens_per_s"] / baseline["tokens_per_s"]
        )
        print(
            "BASELINE decode "
            f"steps={baseline['steps']} "
            f"elapsed={baseline['elapsed_s']:.3f}s "
            f"throughput={baseline['tokens_per_s']:.2f}tok/s"
        )
        print(
            "MTP2 stress (target plus real draft head; scheduler integration pending) "
            f"steps={mtp_result['steps']} "
            f"elapsed={mtp_result['elapsed_s']:.3f}s "
            f"acceptance={mtp_result['acceptance']:.4%} "
            f"accepted={mtp_result['accepted']}/{mtp_result['verified']} "
            f"target_throughput={mtp_result['target_tokens_per_s']:.2f}tok/s "
            f"projected_speculative_throughput="
            f"{mtp_result['projected_tokens_per_s']:.2f}tok/s"
        )
        print(
            "RESULT projected_mtp2_speedup="
            f"{projected_speedup:.3f}x "
            f"draft_overhead={1 - mtp_result['target_tokens_per_s'] / baseline['tokens_per_s']:.4%}"
        )
    finally:
        reset_context()
        engine.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local Qwen3.5-9B HF directory")
    parser.add_argument("--prompt-tokens", type=int, default=129)
    parser.add_argument(
        "--steps", type=int, default=10240, help="maximum decode steps per mode"
    )
    parser.add_argument(
        "--duration", type=float, default=120.0,
        help="seconds per mode; use 0 to run only the step limit",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    if args.prompt_tokens < 2:
        parser.error("--prompt-tokens must be >= 2")
    if args.steps < 2:
        parser.error("--steps must be >= 2")
    if args.duration < 0:
        parser.error("--duration must be >= 0")
    test_mtp(
        args.model,
        args.prompt_tokens,
        args.steps,
        args.duration,
        args.enforce_eager,
    )
