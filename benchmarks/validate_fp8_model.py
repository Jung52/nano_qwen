"""Complete-model FP8 smoke checks and BF16/FP8 quality/performance evaluation.

Run each model/mode separately. Saved common-input logits allow
BF16/FP8 comparison without keeping two complete models in GPU memory.

Use --suite quality_performance --quality-text <UTF8 corpus> --hidden-out
<reference.pt> for BF16, then --reference-hidden <reference.pt> for FP8.
Reports retain token counts, protocol, full-vocabulary metrics and raw timings.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engine_bench_utils import run_until_idle
from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.engine.sequence import Sequence
from nano_qwen.layers.attention import Attention
from nano_qwen.sampling_params import SamplingParams
from nano_qwen.utils.context import set_context, reset_context


TEXTS = [
    "The sky appears blue because sunlight scatters in Earth's atmosphere. Shorter wavelengths scatter more strongly, so blue light reaches our eyes from many directions.",
    "量化通过减少模型权重和激活使用的位数，降低显存占用。缩放系数用于在低精度数值与原始数值之间转换。评估量化需要同时检查准确率和推理速度。",
    "def add(a, b):\n    return a + b\n\nassert add(2, 3) == 5\n# Adding two integers returns their sum.\n",
]
QUESTIONS = ["Explain briefly why the sky appears blue.", "什么是模型量化？用两句话说明。"]

PERFORMANCE_PROFILES = {
    "legacy": [(128, 1), (128, 4), (128, 8), (512, 1), (1024, 1), (512, 4)],
    "batch": [(128, b) for b in (1, 4, 8, 16, 32, 64)],
    "context": [(512, 1), (512, 4), (512, 8), (1024, 1), (1024, 4),
                (1024, 8), (2048, 1), (2048, 4), (4096, 1)],
}


def parse_perf_case(value):
    try:
        length, size = map(int, value.split(":"))
        if length < 1 or size < 1:
            raise ValueError
        return length, size
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use positive PROMPT:BATCH, e.g. 128:32") from exc


def resolve_perf_cases(args):
    cases = args.perf_cases or PERFORMANCE_PROFILES[args.perf_profile]
    return list(dict.fromkeys(cases))


def capacity_check(engine, length, size, completion_tokens):
    block = engine.config.kvcache_block_size
    needed = size * math.ceil((length + completion_tokens) / block)
    available = engine.config.num_kvcache_blocks
    info = {"required_kv_blocks": needed, "available_kv_blocks": available,
            "kv_block_size": block}
    reasons = []
    if size > engine.config.max_num_seqs:
        reasons.append("batch exceeds max_num_seqs")
    if length + completion_tokens > engine.config.max_model_len:
        reasons.append("prompt plus completion exceeds max_model_len")
    if needed > available:
        reasons.append("insufficient KV capacity without prefix sharing")
    return info, "; ".join(reasons)


def save_report(path, report):
    # Checkpoints retain completed cases if a subsequent case fails.
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)

# Supplemental fixtures are smoke inputs, not Chinese/code benchmark datasets.
QUALITY_TEXTS = {
    "chinese_sample": "\n\n".join([
        TEXTS[1],
        "春天的清晨，河边的柳树长出了新叶。小林骑车去图书馆，借了一本关于天文学的书。晚上，他把书里的星座画在笔记本上。",
        "如果每个盒子装六个苹果，四个盒子一共装二十四个苹果。送给朋友五个以后，还剩十九个。计算时需要先乘法，再减法。",
        "推理系统先读取输入文本，再把文本转换成词元。预填充阶段处理已有上下文，解码阶段逐个生成新的词元。不同阶段的计算特点不同。",
    ]),
    "code_sample": "\n\n".join([
        TEXTS[2],
        "def factorial(n):\n    if n < 0:\n        raise ValueError('negative input')\n    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result\n\nassert factorial(5) == 120\n",
        "def binary_search(values, target):\n    left, right = 0, len(values) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if values[mid] == target:\n            return mid\n        if values[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n",
    ]),
}


def tensor_digest(tensor):
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


def quality_windows(ids, context):
    """Independent windows: first token has no target, every later token is scored."""
    for start in range(0, len(ids), context):
        window = ids[start:start + context]
        if len(window) > 1:
            yield window


def distribution_totals(logits, targets, reference_logits=None):
    """Sum per-target statistics; caller divides by the actual scored-token count."""
    current = logits.float()
    if not torch.isfinite(current).all():
        raise AssertionError("Non-finite quality logits")
    logp = current.log_softmax(-1)
    nll = -logp.gather(1, targets[:, None]).sum(dtype=torch.float64)
    result = {"tokens": targets.numel(), "nll_sum": float(nll)}
    if reference_logits is not None:
        teacher = reference_logits.float()
        if not torch.isfinite(teacher).all():
            raise AssertionError("Non-finite reference logits")
        ref_logp = teacher.log_softmax(-1)
        ref_nll = -ref_logp.gather(1, targets[:, None]).sum(dtype=torch.float64)
        result.update(
            reference_nll_sum=float(ref_nll),
            # Full vocabulary KL(BF16 || candidate), not a top-k approximation.
            kl_sum=float((ref_logp.exp() * (ref_logp - logp)).sum(dtype=torch.float64)),
            top1_matches=int((current.argmax(-1) == teacher.argmax(-1)).sum()),
            logit_error_sq=float((current - teacher).square().sum(dtype=torch.float64)),
            reference_logit_sq=float(teacher.square().sum(dtype=torch.float64)),
        )
    return result


@contextmanager
def standalone_prefill(runner):
    attentions = [m for m in runner.model.modules() if isinstance(m, Attention)]
    caches = [(m, m.k_cache, m.v_cache) for m in attentions]
    try:
        for m, _, _ in caches:
            m.k_cache = m.v_cache = torch.empty(0, device="cuda")
        yield
    finally:
        reset_context()
        for m, k, v in caches:
            m.k_cache, m.v_cache = k, v
        for layer in runner.gdn_layers:
            layer.reset_state([0])


def quality_forward(runner, window):
    ids = torch.tensor(window, device="cuda", dtype=torch.int64)
    for layer in runner.gdn_layers:
        layer.reset_state([0])
    boundaries = torch.tensor([0, len(window)], device="cuda", dtype=torch.int32)
    set_context(True, cu_seqlens_q=boundaries, cu_seqlens_k=boundaries,
                max_seqlen_q=len(window), max_seqlen_k=len(window),
                state_indices=torch.zeros(1, device="cuda", dtype=torch.int64),
                prefill_slices=[(0, len(window))])
    hidden = runner.model(ids, torch.arange(len(window), device="cuda"))
    if not torch.isfinite(hidden).all():
        raise AssertionError("Non-finite quality hidden states")
    return ids, hidden


def evaluate_quality(engine, args):
    runner = engine.model_runner
    raw = Path(args.quality_text).read_bytes()
    corpus_ids = engine.tokenizer.encode(raw.decode("utf-8"), add_special_tokens=False)
    if len(corpus_ids) < 2:
        raise ValueError("Quality text must contain at least two tokens")
    texts = {"wikitext2_test": corpus_ids[:args.quality_tokens]}
    texts.update({name: engine.tokenizer.encode(text, add_special_tokens=False)
                  for name, text in QUALITY_TEXTS.items()})
    metadata = {
        "text_sha256": hashlib.sha256(raw).hexdigest(),
        "head_sha256": tensor_digest(runner.model.lm_head.weight),
        "context_tokens": args.quality_context,
        "token_limit": args.quality_tokens,
        "tokenization": "raw text, add_special_tokens=False, no chat template",
        "protocol": "non-overlapping independent windows; reset KV/GDN/positions; score ids[1:] with hidden[:-1]",
        "forward_path": "standalone eager prefill, irrespective of generation mode",
        "full_corpus_tokens": len(corpus_ids),
        "subset": len(corpus_ids) > args.quality_tokens,
    }
    reference = None
    if args.reference_hidden:
        reference = torch.load(args.reference_hidden, map_location="cpu", weights_only=True)
        if reference["metadata"] != metadata or reference["token_ids"] != texts:
            raise ValueError("Reference head, corpus, tokenization or evaluation protocol differs")
        if reference["quantization"] is not None:
            raise ValueError("Quality reference must be the unquantized BF16 model")
    saved = {"metadata": metadata, "token_ids": texts,
             "quantization": engine.config.quantization, "hidden": {}}
    reports = {}
    with standalone_prefill(runner):
        for domain, tokens in texts.items():
            totals = {}
            saved["hidden"][domain] = []
            windows = list(quality_windows(tokens, args.quality_context))
            if reference and len(reference["hidden"][domain]) != len(windows):
                raise ValueError("Reference window count differs")
            for index, window in enumerate(windows):
                ids, hidden = quality_forward(runner, window)
                ref_hidden = reference["hidden"][domain][index] if reference else None
                if ref_hidden is not None and (ref_hidden.shape != hidden.shape or ref_hidden.dtype != hidden.dtype):
                    raise ValueError("Reference hidden layout differs")
                if args.hidden_out:
                    saved["hidden"][domain].append(hidden.cpu())
                # The unchanged BF16 head projects both hidden states with the
                # same row shape; never store the full corpus x vocab matrix.
                for start in range(0, len(window) - 1, args.logit_block):
                    stop = min(start + args.logit_block, len(window) - 1)
                    current = runner.model.compute_logits(hidden[start:stop])
                    teacher = runner.model.compute_logits(ref_hidden[start:stop].to("cuda")) if reference else None
                    values = distribution_totals(current, ids[start + 1:stop + 1], teacher)
                    for key, value in values.items():
                        totals[key] = totals.get(key, 0) + value
                if (index + 1) % 8 == 0 or index + 1 == len(windows):
                    print(f"QUALITY {domain} windows={index + 1}/{len(windows)}", flush=True)
            count = totals["tokens"]
            nll = totals["nll_sum"] / count
            result = {"input_tokens": len(tokens), "scored_tokens": count,
                      "windows": len(windows), "mean_nll": nll, "perplexity": math.exp(nll)}
            if reference:
                ref_nll = totals["reference_nll_sum"] / count
                result.update(reference_perplexity=math.exp(ref_nll),
                              delta_nll=nll - ref_nll, perplexity_ratio=math.exp(nll - ref_nll),
                              mean_kl_bf16_to_candidate=totals["kl_sum"] / count,
                              top1_agreement=totals["top1_matches"] / count,
                              relative_l2_logits=math.sqrt(totals["logit_error_sq"] / totals["reference_logit_sq"]))
            reports[domain] = result
    if args.hidden_out:
        hidden_path = Path(args.hidden_out)
        hidden_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, hidden_path)
    return {"metadata": metadata, "domains": reports,
            "supplemental_fixtures": QUALITY_TEXTS}


def fixed_batch(engine, prompt, size, tokens):
    """Whole-batch prefill followed by exactly tokens-1 production decode steps."""
    if not engine.is_finished():
        raise AssertionError("Previous fixed batch did not drain")
    seqs = [Sequence(prompt, SamplingParams(temperature=1.0, max_tokens=tokens, ignore_eos=True))
            for _ in range(size)]
    _, reason = capacity_check(engine, len(prompt), size, tokens)
    if reason:
        raise ValueError(reason)
    # Startup warmup stays small on 12GB; timing workload gets a whole-prefill
    # budget. Decode-graph mode has no static prefill buffers to resize.
    previous = engine.scheduler.max_num_batched_tokens
    engine.scheduler.max_num_batched_tokens = len(prompt) * size
    try:
        for seq in seqs:
            engine.scheduler.add(seq)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        engine.step()
        prefill_submit_end = time.perf_counter()
        # A single boundary sync prevents pending prefill GPU work from being
        # charged to decode. This is device-ready latency, NOT streaming TTFT.
        torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        if any(seq.num_completion_tokens != 1 for seq in seqs):
            raise AssertionError("Prefill was chunked or the fixed batch was split")
        for _ in range(tokens - 1):
            engine.step()
        torch.cuda.synchronize()
        end = time.perf_counter()
        if not engine.is_finished() or any(seq.num_completion_tokens != tokens for seq in seqs):
            raise AssertionError("Fixed batch did not finish in the expected step count")
        if engine.model_runner.input_batch.seq_id_to_slot:
            raise AssertionError("Finished batch leaked input slots")
        return {"prefill_submit_ms": (prefill_submit_end - start) * 1000,
                "prefill_ready_ms": (prefill_end - start) * 1000,
                "decode_ms": (end - prefill_end) * 1000,
                "decode_step_ms": (end - prefill_end) * 1000 / (tokens - 1),
                "decode_tokens_per_s": size * (tokens - 1) / (end - prefill_end),
                "per_request_decode_tokens_per_s": (tokens - 1) / (end - prefill_end),
                "e2e_ms": (end - start) * 1000,
                "e2e_completion_tokens_per_s": size * tokens / (end - start),
                "prefill_input_tokens_per_s": size * len(prompt) / (prefill_end - start),
                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    finally:
        engine.scheduler.max_num_batched_tokens = previous


def evaluate_performance(engine, args, report):
    cases = resolve_perf_cases(args)
    if args.mode == "piecewise_graph":
        raise ValueError("Controlled performance suite currently uses eager or decode_graph; use smoke for piecewise checks")
    if engine.config.enable_prefix_cache:
        raise ValueError("Controlled performance requires prefix_cache=False")
    reports = []
    performance = {
        "profile": args.perf_profile, "warmup_rounds": args.warmup_rounds,
        "rounds": args.rounds, "requested_cases": [list(c) for c in cases],
        "timing_version": 2,
        "protocol": "real engine.step, actual queue/output configuration recorded in config; "
                    "production sampler; no profiler; one CUDA sync after prefill, "
                    "none between decode steps, one at end; prefill emits token1",
        "prefill_latency_definition": "prefill_ready_ms measures device-ready boundary; "
                                      "not client-visible first-token delivery",
        "workloads": reports,
    }
    report["performance"] = performance
    base = engine.tokenizer.encode(TEXTS[0], add_special_tokens=False)
    if not base:
        raise ValueError("Empty performance prompt tokenization")
    for length, size in cases:
        capacity, reason = capacity_check(engine, length, size, args.decode_tokens + 1)
        case = {"prompt_tokens": length, "batch": size,
                "decode_forwards": args.decode_tokens,
                "completion_tokens": args.decode_tokens + 1,
                "prefill_total_tokens": length * size,
                "prompt_sha256": None, "runs": [], **capacity}
        reports.append(case)
        report["status"] = "evaluation_in_progress"
        if reason:
            case.update(status="skipped_capacity", reason=reason)
            print(f"SKIP prompt={length} bs={size}: {reason}", flush=True)
            save_report(args.json_out, report)
            continue
        prompt = (base * math.ceil(length / len(base)))[:length]
        case["prompt_sha256"] = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
        runs = case["runs"]
        case["status"] = "running"
        save_report(args.json_out, report)
        try:
            for index in range(args.warmup_rounds + args.rounds):
                torch.manual_seed(20261006 + index)
                result = fixed_batch(engine, prompt, size, args.decode_tokens + 1)
                if index >= args.warmup_rounds:
                    runs.append(result)
                    save_report(args.json_out, report)
        except Exception as exc:
            case.update(status="failed_oom" if isinstance(exc, torch.OutOfMemoryError)
                        else "failed", error_type=type(exc).__name__, error=str(exc))
            report["status"] = "evaluation_failed"
            save_report(args.json_out, report)
            # Do not reuse an engine whose queue/state may be partially mutated.
            raise
        medians = {key: statistics.median(run[key] for run in runs) for key in runs[0]}
        case.update(status="completed", **medians)
        case["decode_step_cv_percent"] = (
            statistics.stdev(r["decode_step_ms"] for r in runs)
            / statistics.mean(r["decode_step_ms"] for r in runs) * 100
            if len(runs) > 1 else None)
        save_report(args.json_out, report)
        print(f"PERF prompt={length} bs={size} prefill_ready={medians['prefill_ready_ms']:.2f}ms "
              f"decode={medians['decode_tokens_per_s']:.2f}t/s", flush=True)
    return performance


def storage_bytes(tensors):
    storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                for t in tensors}
    return sum(storages.values())


class GreedySampler(torch.nn.Module):
    def forward(self, logits, temperatures):
        assert torch.isfinite(logits).all(), "Non-finite generation logits"
        return logits.argmax(dim=-1)


class GraphCounter:
    def __init__(self, runner):
        self.manager = runner.cuda_graphs
        self.hits = {"decode": 0, "prefill": 0}

    def __enter__(self):
        self.originals = {}
        for kind in self.hits:
            name = f"run_{kind}"
            original = getattr(self.manager, name)
            self.originals[name] = original

            def counted(*args, _kind=kind, _original=original, **kwargs):
                self.hits[_kind] += 1
                return _original(*args, **kwargs)

            setattr(self.manager, name, counted)
        return self

    def __exit__(self, *exc):
        for name, original in self.originals.items():
            setattr(self.manager, name, original)


def batch(engine, prompts, tokens, *, ignore_eos=True):
    seqs = [Sequence(p, SamplingParams(temperature=1.0, max_tokens=tokens, ignore_eos=ignore_eos))
            for p in prompts]
    for seq in seqs:
        engine.scheduler.add(seq)
    torch.cuda.synchronize()
    result = run_until_idle(engine, seqs)
    torch.cuda.synchronize()
    assert all(s.is_finished and 0 < len(s.completion_token_ids) <= tokens for s in seqs)
    if ignore_eos:
        assert all(len(s.completion_token_ids) == tokens for s in seqs)
    assert not engine.model_runner.input_batch.seq_id_to_slot
    window = max(result.e2e_s.values()) - min(result.ttft_s.values())
    decode_tokens = sum(len(s.completion_token_ids) - 1 for s in seqs)
    return {"outputs": result.outputs, "ttft_ms": statistics.median(result.ttft_s.values()) * 1000,
            "decode_tokens_per_s": decode_tokens / window if decode_tokens and window > 0 else None}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("eager", "decode_graph", "piecewise_graph"), default="eager")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--suite", choices=("smoke", "quality", "performance", "quality_performance"), default="smoke")
    parser.add_argument("--quality-text", help="UTF8 WikiText-2 raw test text (rows joined with two newlines)")
    parser.add_argument("--quality-context", type=int, default=512)
    parser.add_argument("--quality-tokens", type=int, default=32768)
    parser.add_argument("--logit-block", type=int, default=32)
    parser.add_argument("--hidden-out", help="Save compact hidden states for a later candidate comparison")
    parser.add_argument("--reference-hidden", help="BF16 hidden states from the exact same head/corpus/protocol")
    parser.add_argument("--warmup-rounds", type=int, default=5)
    parser.add_argument("--decode-tokens", type=int, default=256, help="Number of decode forwards after prefill")
    parser.add_argument("--perf-profile", choices=tuple(PERFORMANCE_PROFILES), default="batch")
    parser.add_argument("--perf-cases", nargs="+", type=parse_perf_case,
                        help="Override profile with PROMPT:BATCH cases, e.g. 128:16 512:8")
    parser.add_argument("--max-num-seqs", type=int, help="Default: largest requested performance batch")
    parser.add_argument("--max-model-len", type=int, help="Default: fit prompt + completion, at least 2048")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    args = parser.parse_args()
    for name in ("rounds", "warmup_rounds", "decode_tokens", "logit_block"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.quality_context < 2 or args.quality_tokens < 2:
        parser.error("Quality context and token limit must be >=2")
    if "quality" in args.suite and not args.quality_text:
        parser.error("Quality evaluation requires --quality-text")
    if args.quality_context > 2048:
        parser.error("Quality context limit: <=2048")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if args.max_num_seqs is not None and args.max_num_seqs < 1:
        parser.error("--max-num-seqs must be positive")
    if args.max_model_len is not None and args.max_model_len < 2:
        parser.error("--max-model-len must be >=2")
    if "performance" in args.suite and args.mode == "piecewise_graph":
        parser.error("Controlled performance uses --mode eager or decode_graph")
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    perf_cases = resolve_perf_cases(args) if "performance" in args.suite else []
    max_seqs = args.max_num_seqs or (max(b for _, b in perf_cases) if perf_cases
                                  else 8 if args.suite != "smoke" else 4)
    max_length = args.max_model_len or (max(2048, max(p for p, _ in perf_cases)
                                          + args.decode_tokens + 1) if perf_cases
                                      else 2048 if args.suite != "smoke" else 1024)
    print(f"ENGINE_CONFIG max_num_seqs={max_seqs} max_model_len={max_length} "
          f"gpu_memory_utilization={args.gpu_memory_utilization}", flush=True)
    try:
        engine = LLMEngine(
            args.model, enforce_eager=args.mode == "eager",
            use_prefill_cudagraph=args.mode == "piecewise_graph",
            tensor_parallel_size=1, max_num_seqs=max_seqs,
            max_model_len=max_length, max_num_batched_tokens=512,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    except Exception as exc:
        save_report(output, {"model": args.model, "mode": args.mode, "suite": args.suite,
                             "status": "initialization_failed", "error_type": type(exc).__name__,
                             "error": str(exc), "requested_config": {
                                 "max_num_seqs": max_seqs, "max_model_len": max_length,
                                 "gpu_memory_utilization": args.gpu_memory_utilization}})
        raise
    runner = engine.model_runner
    print(f"MODEL_READY mode={args.mode} quantization={engine.config.quantization}", flush=True)
    report = {"model": args.model, "mode": args.mode, "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "quantization": engine.config.quantization,
              "parameters_mib": storage_bytes(runner.model.parameters()) / 2**20,
              "gdn_state_mib": storage_bytes(t for l in runner.gdn_layers
                                            for t in (l.conv_states, l.recurrent_states)) / 2**20,
              "kv_mib": runner.kv_cache.numel() * runner.kv_cache.element_size() / 2**20,
              "generation": [], "timings": []}
    saved_logits = []
    try:
        if args.suite != "smoke":
            report.update(suite=args.suite, cuda=torch.version.cuda,
                          config={"max_num_seqs": engine.config.max_num_seqs,
                                  "max_model_len": engine.config.max_model_len,
                                  "gpu_memory_utilization": engine.config.gpu_memory_utilization,
                                  "prefix_cache": engine.config.enable_prefix_cache,
                                  "kv_dtype": str(runner.kv_cache.dtype),
                                  "num_kvcache_blocks": engine.config.num_kvcache_blocks,
                                  "kvcache_block_size": engine.config.kvcache_block_size,
                                  "queue_depth": engine.max_concurrent_batches,
                                  "async_output": runner.async_output})
            if "quality" in args.suite:
                report["quality"] = evaluate_quality(engine, args)
                production_sampler = runner.sampler
                runner.sampler = GreedySampler()
                try:
                    questions = [*QUESTIONS, "Write a Python function that returns the factorial of a non-negative integer."]
                    for question in questions:
                        prompt = engine.tokenizer.encode(engine.tokenizer.apply_chat_template(
                            [{"role": "user", "content": question}], tokenize=False,
                            add_generation_prompt=True, enable_thinking=False,
                        ))
                        result = batch(engine, [prompt], 256, ignore_eos=False)
                        ids = result["outputs"][0]
                        report["generation"].append({"question": question, "ids": result["outputs"][0],
                                                     "text": engine.tokenizer.decode(ids, skip_special_tokens=True),
                                                     "stopped_at_eos": ids[-1] == engine.config.eos,
                                                     "max_tokens": 256, "ignore_eos": False})
                finally:
                    runner.sampler = production_sampler
                # Keep completed quality results if a later timing workload fails.
                output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            if "performance" in args.suite:
                report["performance"] = evaluate_performance(engine, args, report)
            skipped = sum(w["status"] == "skipped_capacity"
                          for w in report.get("performance", {}).get("workloads", []))
            report["status"] = "evaluation_completed_with_skips" if skipped else "evaluation_completed"
            save_report(output, report)
            print(f"REPORT {output}", flush=True)
            return
        # Same tokens in all model variants. Disable paged KV writes for this
        # standalone full-sequence prefill; the GDN state is reset per text.
        attentions = [m for m in runner.model.modules() if isinstance(m, Attention)]
        caches = [(m, m.k_cache, m.v_cache) for m in attentions]
        for m, _, _ in caches:
            m.k_cache = m.v_cache = torch.empty(0, device="cuda")
        try:
            for text in TEXTS:
                ids = torch.tensor(engine.tokenizer.encode(text), device="cuda", dtype=torch.int64)
                for layer in runner.gdn_layers:
                    layer.reset_state([0])
                boundaries = torch.tensor([0, ids.numel()], device="cuda", dtype=torch.int32)
                set_context(True, cu_seqlens_q=boundaries, cu_seqlens_k=boundaries,
                            max_seqlen_q=ids.numel(), max_seqlen_k=ids.numel(),
                            state_indices=torch.zeros(1, device="cuda", dtype=torch.int64),
                            prefill_slices=[(0, ids.numel())])
                hidden = runner.model(ids, torch.arange(ids.numel(), device="cuda"))
                logits = runner.model.compute_logits(hidden[-8:])
                assert torch.isfinite(logits).all()
                saved_logits.append({"ids": ids.cpu(), "logits": logits.float().cpu()})
                if args.mode == "piecewise_graph":
                    expected = logits.clone()
                    for layer in runner.gdn_layers:
                        layer.reset_state([0])
                    graph_logits = runner.cuda_graphs.run_prefill(
                        ids, torch.arange(ids.numel(), device="cuda"),
                    )[-8:]
                    assert torch.isfinite(graph_logits).all()
                    report.setdefault("prefill_graph_vs_eager", []).append({
                        "tokens": ids.numel(),
                        "relative_l2": float((graph_logits.float() - expected.float()).norm()
                                             / expected.float().norm()),
                        "top1_agreement": float((graph_logits.argmax(-1) == expected.argmax(-1))
                                                 .float().mean()),
                        "max_abs": float((graph_logits.float() - expected.float()).abs().max()),
                    })
                for layer in runner.gdn_layers:
                    assert torch.isfinite(layer.conv_states).all()
                    assert torch.isfinite(layer.recurrent_states).all()
        finally:
            reset_context()
            for m, k, v in caches:
                m.k_cache, m.v_cache = k, v
            for layer in runner.gdn_layers:
                layer.reset_state([0])
        torch.save(saved_logits, str(output) + ".logits.pt")
        production_sampler = runner.sampler
        runner.sampler = GreedySampler()
        for question in QUESTIONS:
            prompt = engine.tokenizer.encode(engine.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            ))
            with GraphCounter(runner) as counter:
                result = batch(engine, [prompt], 48)
            report["generation"].append({"question": question, "ids": result["outputs"][0],
                                         "text": engine.tokenizer.decode(result["outputs"][0]),
                                         "paths": dict(counter.hits)})
        runner.sampler = production_sampler
        # Warm timings with production async sampling; no per-token diagnostics.
        for length, size, tokens in [(128, 1, 32), (128, 4, 32), (512, 1, 1)]:
            prompt = (engine.tokenizer.encode(TEXTS[0]) * 20)[:length]
            runs = []
            for i in range(args.rounds + 1):
                torch.manual_seed(20261005 + i)
                result = batch(engine, [prompt] * size, tokens)
                if i:
                    runs.append(result)
            report["timings"].append({"prompt_tokens": length, "batch": size, "completion_tokens": tokens,
                "ttft_ms": statistics.median(r["ttft_ms"] for r in runs),
                "decode_tokens_per_s": statistics.median(r["decode_tokens_per_s"] for r in runs)
                    if tokens > 1 else None})
        report["status"] = "runtime_checks_passed"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        if report.get("status") != "evaluation_failed":
            report.update(status="evaluation_failed", error_type=type(exc).__name__, error=str(exc))
            save_report(output, report)
        raise
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
