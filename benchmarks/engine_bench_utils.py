"""Shared helpers for the engine-level validation / benchmark scripts.

Both ``validate_engine_qwen35.py`` and ``bench_engine_qwen35.py`` build
engines through :func:`make_engine` and drive them through
:func:`run_until_idle`, which uses the real production path
(``engine.scheduler.add`` + ``engine.step()``) while keeping ``Sequence``
references for per-request latency measurements.

The engine modes live in :data:`MODES` / :data:`PERF_MODES`.  Each mode is
run in a *separate subprocess* (parent-child pattern) because
``dist.init_process_group`` can only be initialized once per process and
the scripts compare several differently-configured engines.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn

# Make `from nano_qwen...` work when run as `python benchmarks/<script>.py`.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nano_qwen.engine.llm_engine import LLMEngine  # noqa: E402
from nano_qwen.engine.sequence import Sequence  # noqa: E402
from nano_qwen.sampling_params import SamplingParams  # noqa: E402


DEFAULT_MODEL = os.environ.get("NANO_QWEN_MODEL", "/home/wei/code/models/qwen")


# ---------------------------------------------------------------------------
# Environment / misc
# ---------------------------------------------------------------------------

def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        )
        return out.strip()
    except Exception:
        return "unknown"


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "git_commit": git_commit(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tensor_parallel_size": 1,
    }
    for mod, key in (("flashinfer", "flashinfer_version"), ("flash_attn", "flash_attn_version")):
        try:
            m = __import__(mod)
            info[key] = getattr(m, "__version__", "unknown")
        except Exception:
            info[key] = "not installed"
    return info


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def save_json(path: str, data: Any) -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def repo_log_path(name: str, subdir: str = "validate") -> str:
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    log_dir = os.path.join(root, "logs", subdir)
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"{name}_{stamp}.json")


# ---------------------------------------------------------------------------
# Engine modes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModeConfig:
    name: str
    enforce_eager: bool
    queue_depth: int
    async_output: bool
    prefill_cudagraph: bool


# Correctness matrix (validate_engine_qwen35.py)
MODES: list[ModeConfig] = [
    ModeConfig("BASELINE", enforce_eager=True, queue_depth=1, async_output=False, prefill_cudagraph=False),
    ModeConfig("A", enforce_eager=True, queue_depth=2, async_output=True, prefill_cudagraph=False),
    ModeConfig("B", enforce_eager=False, queue_depth=2, async_output=True, prefill_cudagraph=False),
    ModeConfig("C", enforce_eager=False, queue_depth=2, async_output=True, prefill_cudagraph=True),
]

# Performance matrix (bench_engine_qwen35.py)
PERF_MODES: list[ModeConfig] = [
    ModeConfig("P0", enforce_eager=True, queue_depth=1, async_output=False, prefill_cudagraph=False),
    ModeConfig("P1", enforce_eager=True, queue_depth=2, async_output=True, prefill_cudagraph=False),
    ModeConfig("P2", enforce_eager=False, queue_depth=2, async_output=True, prefill_cudagraph=False),
    ModeConfig("P3", enforce_eager=False, queue_depth=2, async_output=True, prefill_cudagraph=True),
]

MODE_NAMES = {m.name: m for m in MODES}
PERF_MODE_NAMES = {m.name: m for m in PERF_MODES}


class ArgmaxSampler(nn.Module):
    """Benchmark-only deterministic sampler with bf16 tie diagnostics.

    Patched onto ``engine.model_runner.sampler`` in validation scripts only;
    the production Sampler is untouched. CUDA Graph and eager paths can differ
    by a bf16 ULP, so rows whose top-two margin is below ``tie_tolerance`` are
    counted as numerical ties instead of silently changing the argmax rule.
    """

    def __init__(self, tie_tolerance: float = 1e-2) -> None:
        super().__init__()
        self.tie_tolerance = tie_tolerance
        self.ambiguous_rows = 0
        self.total_rows = 0

    def reset_stats(self) -> None:
        self.ambiguous_rows = 0
        self.total_rows = 0

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        assert torch.isfinite(logits).all(), "non-finite logits observed"
        values = logits.float().topk(2, dim=-1).values
        self.total_rows += logits.shape[0]
        self.ambiguous_rows += int(
            (values[:, 0] - values[:, 1]).abs().le(self.tie_tolerance).sum().item()
        )
        return logits.argmax(dim=-1)


def make_engine(
    model: str,
    mode: ModeConfig,
    *,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 2048,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.90,
    sampler: nn.Module | None = None,
    reset_cuda_stats: bool = False,
) -> LLMEngine:
    """Build an engine for ``mode`` and apply the runtime toggles.

    ``enforce_eager`` must be set at construction time (it decides whether
    CUDA graphs are captured); queue depth / async output / prefill graphs
    are runtime attributes applied afterwards.

    ``reset_cuda_stats`` releases the caching allocator and zeroes the
    process-wide CUDA memory counters first. ``LLMEngine.exit`` now performs
    the ownership teardown itself; this reset remains defensive because
    ``allocate_kv_cache`` derives its budget from process-wide memory stats.
    """
    if reset_cuda_stats:
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    engine = LLMEngine(
        model,
        enforce_eager=mode.enforce_eager,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    engine.max_concurrent_batches = mode.queue_depth
    engine.model_runner.async_output = mode.async_output
    engine.model_runner.use_prefill_cudagraph = mode.prefill_cudagraph
    if sampler is not None:
        engine.model_runner.sampler = sampler
    return engine


# ---------------------------------------------------------------------------
# Workload building blocks
# ---------------------------------------------------------------------------

def make_random_prompt(length: int, seed: int, vocab_size: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (length,), generator=g).tolist()


def make_const_prompt(length: int, token: int) -> list[int]:
    return [token] * length


def make_params(max_tokens: int) -> SamplingParams:
    return SamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=True)


@dataclass
class RunResult:
    completions: dict[int, list[int]]
    ttft_s: dict[int, float]
    itl_s: dict[int, list[float]]
    e2e_s: dict[int, float]
    steps: int = 0
    outputs: list[list[int]] = field(default_factory=list)  # per seq_id, in submission order


def run_until_idle(
    engine: LLMEngine,
    seqs: list[Sequence],
    after_step: Callable[[], str | None] | None = None,
) -> RunResult:
    """Drive the production engine step loop until idle; collect metrics.

    ``after_step`` runs after every step and may return an error string to
    abort (used for scheduler-invariant checks and late submissions).
    """
    submitted = {s.seq_id: time.perf_counter() for s in seqs}
    completions: dict[int, list[int]] = {s.seq_id: [] for s in seqs}
    ttft: dict[int, float] = {s.seq_id: float("nan") for s in seqs}
    itl: dict[int, list[float]] = {s.seq_id: [] for s in seqs}
    last_seen = {s.seq_id: 0 for s in seqs}
    last_ts = dict(submitted)

    # Safety cap: each seq needs ~1 prefill step + max_tokens decode steps.
    cap = sum(1 + s.max_tokens for s in seqs) * 2 + 200
    steps = 0
    while not engine.is_finished():
        engine.step()
        steps += 1
        if steps > cap:
            raise RuntimeError(
                f"step cap ({cap}) exceeded; engine appears stuck "
                f"(waiting={len(engine.scheduler.waiting)} "
                f"running={len(engine.scheduler.running)} "
                f"in_flight={len(engine.scheduler.in_flight)})"
            )
        now = time.perf_counter()
        for s in seqs:
            n = s.num_completion_tokens
            if n > last_seen[s.seq_id]:
                if last_seen[s.seq_id] == 0:
                    ttft[s.seq_id] = now - submitted[s.seq_id]
                else:
                    itl[s.seq_id].append(now - last_ts[s.seq_id])
                last_ts[s.seq_id] = now
                last_seen[s.seq_id] = n
        if after_step is not None:
            err = after_step()
            if err:
                raise AssertionError(f"after_step check failed: {err}")

    e2e = {s.seq_id: time.perf_counter() - submitted[s.seq_id] for s in seqs}
    return RunResult(
        completions={s.seq_id: s.completion_token_ids for s in seqs},
        ttft_s=ttft,
        itl_s=itl,
        e2e_s=e2e,
        steps=steps,
        outputs=[s.completion_token_ids for s in seqs],
    )


class GraphPathCounter:
    """Counts which ``run_model`` path each batch took (mirrors run_model's logic).

    Wrap with ``with GraphPathCounter(runner) as counter:`` — a lightweight
    benchmark-only wrapper, no production-code change.
    """

    def __init__(self, runner):
        self.runner = runner
        self.decode_graph_hits = 0
        self.decode_eager_fallbacks = 0
        self.prefill_graph_hits = 0
        self.prefill_eager_fallbacks = 0
        self._orig = runner.run_model

    def __enter__(self) -> "GraphPathCounter":
        self.runner.run_model = self._wrapped  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc) -> None:
        self.runner.run_model = self._orig  # type: ignore[method-assign]

    def _wrapped(self, input_ids, positions, is_prefill):
        eager_branch = (
            self.runner.enforce_eager
            or not hasattr(self.runner, "graphs")
            or input_ids.size(0) > 512
        )
        if not is_prefill:
            if not eager_branch and input_ids.size(0) in self.runner.graphs:
                self.decode_graph_hits += 1
            else:
                self.decode_eager_fallbacks += 1
        else:
            if (not eager_branch) and self.runner.use_prefill_cudagraph:
                self.prefill_graph_hits += 1
            else:
                self.prefill_eager_fallbacks += 1
        return self._orig(input_ids, positions, is_prefill)

    def as_dict(self) -> dict[str, int]:
        return {
            "decode_graph_hits": self.decode_graph_hits,
            "decode_eager_fallbacks": self.decode_eager_fallbacks,
            "prefill_graph_hits": self.prefill_graph_hits,
            "prefill_eager_fallbacks": self.prefill_eager_fallbacks,
        }
