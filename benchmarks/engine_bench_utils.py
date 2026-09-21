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
import itertools
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any, Callable

import torch
from torch import nn

# Make `from nano_qwen...` work when run as `python benchmarks/<script>.py`.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nano_qwen.engine.llm_engine import LLMEngine  # noqa: E402
from nano_qwen.engine.sequence import Sequence  # noqa: E402
from nano_qwen.sampling_params import SamplingParams  # noqa: E402


DEFAULT_MODEL = os.environ.get("NANO_QWEN_MODEL", "/home/wei/code/models/qwen")
_PROBE_KEY_COUNTER = itertools.count()


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
        self._diagnostics_enabled = False
        self._capture_state = False
        self._capture_logits = False
        self._batch_context: list[tuple[str | None, int | None]] | None = None
        self._pending_state_stats: torch.Tensor | None = None
        self._pending_layer_ids: list[int] = []
        self._diagnostics: dict[str, dict[int, dict[str, Any]]] = {}

    def reset_stats(self) -> None:
        self.ambiguous_rows = 0
        self.total_rows = 0
        self._batch_context = None
        self._pending_state_stats = None
        self._pending_layer_ids = []
        self._diagnostics.clear()

    def enable_diagnostics(
        self, *, capture_state: bool = False, capture_logits: bool = False
    ) -> None:
        self._diagnostics_enabled = True
        self._capture_state = capture_state
        self._capture_logits = capture_logits

    def disable_diagnostics(self) -> None:
        self._diagnostics_enabled = False
        self._capture_state = False
        self._capture_logits = False
        self._batch_context = None
        self._pending_state_stats = None
        self._pending_layer_ids = []

    def set_batch_context(self, seqs, is_prefill: bool) -> None:
        """Bind each sampler row to a stable request key/token position."""
        if not self._diagnostics_enabled:
            return
        context: list[tuple[str | None, int | None]] = []
        for seq in seqs:
            produces_token = (
                not is_prefill
                or seq.num_cached_tokens + seq.num_scheduled_tokens
                >= seq.num_tokens
            )
            key = getattr(seq, "_diagnostic_key", None)
            position = seq.num_completion_tokens if produces_token else None
            context.append((key, position))
        self._batch_context = context

    def observe_gdn_state(self, layers, slots: torch.Tensor) -> None:
        """Capture compact post-forward GDN state summaries on the GPU."""
        if not self._diagnostics_enabled or not self._capture_state:
            return

        def stats(tensor: torch.Tensor) -> torch.Tensor:
            value = tensor.float()
            dims = tuple(range(1, value.ndim))
            return torch.stack(
                (
                    value.mean(dim=dims),
                    value.abs().mean(dim=dims),
                    value.square().mean(dim=dims).sqrt(),
                    value.abs().amax(dim=dims),
                ),
                dim=-1,
            )

        per_layer = []
        layer_ids = []
        for layer in layers:
            conv = layer.conv_states.index_select(0, slots)
            recurrent = layer.recurrent_states.index_select(0, slots)
            per_layer.append(torch.cat((stats(conv), stats(recurrent)), dim=-1))
            layer_ids.append(layer.layer_idx)
        self._pending_state_stats = (
            torch.stack(per_layer, dim=1) if per_layer else None
        )
        self._pending_layer_ids = layer_ids

    def get_diagnostic(self, key: str, position: int) -> dict[str, Any] | None:
        return self._diagnostics.get(key, {}).get(position)

    def get_diagnostic_map(self, key: str) -> dict[int, dict[str, Any]]:
        return self._diagnostics.get(key, {})

    def clear_diagnostics(self, keys: list[str] | None = None) -> None:
        if keys is None:
            self._diagnostics.clear()
            return
        for key in keys:
            self._diagnostics.pop(key, None)

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        assert torch.isfinite(logits).all(), "non-finite logits observed"
        values, indices = logits.float().topk(2, dim=-1)
        margins = (values[:, 0] - values[:, 1]).abs()
        row_scales = values.abs().amax(dim=-1)
        dtype_eps = torch.finfo(logits.dtype).eps
        effective_tolerances = torch.maximum(
            margins.new_full(margins.shape, self.tie_tolerance),
            row_scales * dtype_eps,
        )
        self.total_rows += logits.shape[0]
        self.ambiguous_rows += int(
            margins.le(effective_tolerances).sum().item()
        )

        if (
            self._diagnostics_enabled
            and self._batch_context is not None
            and len(self._batch_context) == logits.shape[0]
        ):
            top_values = values.detach().cpu().tolist()
            top_indices = indices.detach().cpu().tolist()
            margin_values = margins.detach().cpu().tolist()
            tolerance_values = effective_tolerances.detach().cpu().tolist()
            state_stats = (
                self._pending_state_stats.detach().cpu().tolist()
                if self._pending_state_stats is not None
                else None
            )
            metric_names = ("mean", "abs_mean", "rms", "max_abs")
            for row, (key, position) in enumerate(self._batch_context):
                if key is None or position is None:
                    continue
                diagnostic: dict[str, Any] = {
                    "top_tokens": top_indices[row],
                    "top_logits": top_values[row],
                    "margin": margin_values[row],
                    "base_tolerance": self.tie_tolerance,
                    "effective_tolerance": tolerance_values[row],
                    "near_tie": margin_values[row] <= tolerance_values[row],
                    "logits_dtype": str(logits.dtype).removeprefix("torch."),
                    "state_stage": "post_forward",
                }
                if self._capture_logits:
                    diagnostic["logits"] = (
                        logits[row].detach().float().cpu()
                    )
                if state_stats is not None:
                    diagnostic["gdn_state"] = [
                        {
                            "layer_idx": layer_idx,
                            "conv": dict(zip(metric_names, metrics[:4])),
                            "recurrent": dict(zip(metric_names, metrics[4:])),
                        }
                        for layer_idx, metrics in zip(
                            self._pending_layer_ids,
                            state_stats[row],
                        )
                    ]
                self._diagnostics.setdefault(key, {})[position] = diagnostic

        self._batch_context = None
        self._pending_state_stats = None
        self._pending_layer_ids = []
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
        self.prefill_piecewise_hits = 0
        self.prefill_piecewise_mixed_fallbacks = 0
        self.prefill_piecewise_large_fallbacks = 0
        self.decode_batch_sizes: dict[int, int] = {}
        self.prefill_token_counts: dict[int, int] = {}
        self._orig = runner.run_model

    def __enter__(self) -> "GraphPathCounter":
        self.runner.run_model = self._wrapped  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc) -> None:
        self.runner.run_model = self._orig  # type: ignore[method-assign]

    def _wrapped(self, input_ids, positions, is_prefill, pure_prefill=False):
        num_rows = input_ids.size(0)
        eager_branch = (
            self.runner.enforce_eager
            or not hasattr(self.runner, "graphs")
            or num_rows > 512
        )
        if not is_prefill:
            self.decode_batch_sizes[num_rows] = (
                self.decode_batch_sizes.get(num_rows, 0) + 1
            )
            if not eager_branch and num_rows in self.runner.graphs:
                self.decode_graph_hits += 1
            else:
                self.decode_eager_fallbacks += 1
        else:
            self.prefill_token_counts[num_rows] = (
                self.prefill_token_counts.get(num_rows, 0) + 1
            )
            piecewise_configured = (
                not self.runner.enforce_eager
                and hasattr(self.runner, "graphs")
                and self.runner.use_prefill_cudagraph
            )
            graph_sizes = getattr(self.runner, "prefill_graph_sizes", [])
            piecewise_fits = any(size >= num_rows for size in graph_sizes)
            if (
                piecewise_configured
                and piecewise_fits
                and pure_prefill
            ):
                self.prefill_graph_hits += 1
                self.prefill_piecewise_hits += 1
            else:
                self.prefill_eager_fallbacks += 1
            if piecewise_configured and not pure_prefill:
                self.prefill_piecewise_mixed_fallbacks += 1
            elif piecewise_configured and not piecewise_fits:
                self.prefill_piecewise_large_fallbacks += 1
        return self._orig(
            input_ids,
            positions,
            is_prefill,
            pure_prefill=pure_prefill,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "decode_graph_hits": self.decode_graph_hits,
            "decode_eager_fallbacks": self.decode_eager_fallbacks,
            "prefill_graph_hits": self.prefill_graph_hits,
            "prefill_eager_fallbacks": self.prefill_eager_fallbacks,
            "prefill_piecewise_hits": self.prefill_piecewise_hits,
            "prefill_piecewise_mixed_fallbacks": (
                self.prefill_piecewise_mixed_fallbacks
            ),
            "prefill_piecewise_large_fallbacks": (
                self.prefill_piecewise_large_fallbacks
            ),
            "decode_batch_sizes": dict(sorted(self.decode_batch_sizes.items())),
            "prefill_token_counts": dict(sorted(self.prefill_token_counts.items())),
        }


# ---------------------------------------------------------------------------
# W11 / W12: full state snapshots and real scheduler/runner path assertions.
# These deliberately synchronize/copy to CPU: correctness, NOT performance.
# Checked against Jung52/nano_qwen 0fa22d75ca3d0fd7477e003595ed938e50aaa254.
# ---------------------------------------------------------------------------

@contextmanager
def validation_settings(engine, *, budget: int, reference: bool = False):
    """Change the actual scheduler budget, retaining allocated buffer capacity.

    Reference is eager + depth1 + sync D2H; candidate retains the selected
    BASELINE/A/B/C flags. Restore flags only after each workload has drained.
    Prefix caching is disabled because KV-only reuse cannot restore GDN state.
    """
    if not engine.is_finished():
        raise AssertionError("validation_settings requires an idle engine")
    runner = engine.model_runner
    updates = [(engine.config, "max_num_batched_tokens", budget),
               (engine.scheduler, "max_num_batched_tokens", budget),
               (engine.config, "enable_prefix_cache", False),
               (engine.scheduler, "enable_prefix_cache", False)]
    if reference:
        updates += [(engine, "max_concurrent_batches", 1),
                    (runner, "enforce_eager", True),
                    (runner, "async_output", False),
                    (runner, "use_prefill_cudagraph", False)]
    old = [(obj, key, getattr(obj, key)) for obj, key, _ in updates]
    try:
        for obj, key, value in updates:
            setattr(obj, key, value)
        yield
    finally:
        for obj, key, value in reversed(old):
            setattr(obj, key, value)


def full_request_snapshot(runner, seq, processed: int) -> dict[str, torch.Tensor]:
    """Copy ALL layers, canonicalizing physical KV blocks to token order.

    Called immediately after forward, before sample/remove_request/postprocess.
    Never compare unused/uninitialized cache tail or physical block IDs across
    separate runs. Completion token just sampled has no KV until next forward.
    """
    slot = runner.input_batch.seq_id_to_slot[seq.seq_id]
    result = {}
    if not runner.gdn_layers:
        raise AssertionError("W11/W12 require hybrid Qwen3.5 with GDN layers")
    for i, layer in enumerate(runner.gdn_layers):
        result[f"gdn.{i}.conv"] = layer.conv_states[slot].detach().cpu().clone()
        result[f"gdn.{i}.recurrent"] = layer.recurrent_states[slot].detach().cpu().clone()
    cache = runner.kv_cache  # [2, attention_layers, blocks, block_size, heads, dim]
    if cache is None or cache.ndim != 6 or cache.size(0) != 2:
        raise AssertionError("Unexpected runner.kv_cache layout")
    bs = runner.block_size
    # Transfer one physical block at a time to avoid a large GPU gather temp.
    for i in range(cache.size(1)):
        for k, name in enumerate(("k", "v")):
            parts = []
            for offset in range(0, processed, bs):
                bid = seq.block_table[offset // bs]
                parts.append(cache[k, i, bid, :min(bs, processed - offset)].detach().cpu())
            result[f"kv.{i}.{name}"] = torch.cat(parts, dim=0)
    if any(not torch.isfinite(t).all().item() for t in result.values()):
        raise AssertionError(f"Nonfinite GDN/KV snapshot for request {seq.seq_id}")
    return result


def tensor_snapshot_diff(reference, candidate, *, atol: float, rtol: float):
    """Elementwise |candidate-reference| <= atol + rtol*|reference|.

    Report full-tensor metrics; checksums/statistical fingerprints never decide
    PASS. Zero tolerance is also used for untouched request isolation checks.
    """
    if set(reference) != set(candidate) or not reference:
        return {"pass": False, "reason": "snapshot keys missing/different"}
    groups = {}
    failures = []
    for key in sorted(reference):
        a, b = reference[key].float(), candidate[key].float()
        group = "kv" if key.startswith("kv.") else key.rsplit(".", 1)[1]
        g = groups.setdefault(group, {"elements": 0, "outside_tolerance": 0,
                                     "max_abs": 0.0, "squared_error": 0.0,
                                     "reference_squared": 0.0, "nonfinite": 0})
        if a.shape != b.shape:
            failures.append({"tensor": key, "reason": "shape", "reference": list(a.shape), "candidate": list(b.shape)})
            continue
        diff = (b - a).abs()
        nonfinite = int(((~torch.isfinite(a)) | (~torch.isfinite(b))).sum().item())
        bad = (~torch.isfinite(a)) | (~torch.isfinite(b)) | (diff > atol + rtol * a.abs())
        n_bad = int(bad.sum().item())
        g["elements"] += a.numel()
        g["outside_tolerance"] += n_bad
        g["nonfinite"] += nonfinite
        finite_diff = torch.nan_to_num(diff, nan=float("inf"), posinf=float("inf"))
        finite_ref = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        g["max_abs"] = max(g["max_abs"], float(finite_diff.max().item()))
        g["squared_error"] += float(finite_diff.double().square().sum().item())
        g["reference_squared"] += float(finite_ref.double().square().sum().item())
        if n_bad:
            index = int(bad.flatten().nonzero()[0].item())
            failures.append({"tensor": key, "outside_tolerance": n_bad,
                             "flat_index": index, "reference": float(a.flatten()[index]),
                             "candidate": float(b.flatten()[index]), "max_abs": float(finite_diff.max())})
    for g in groups.values():
        g["rmse"] = (g.pop("squared_error") / max(g["elements"], 1)) ** 0.5
        g["reference_rms"] = (g.pop("reference_squared") / max(g["elements"], 1)) ** 0.5
        g["relative_rmse"] = g["rmse"] / max(g["reference_rms"], 1e-12)
    return {"pass": not failures, "atol": atol, "rtol": rtol,
            "groups": groups, "failing_tensors": len(failures), "first_failures": failures[:8]}


# Bounded-drift acceptance for cross-shape bf16 comparisons. Elementwise
# allclose is diagnostic-only in w11/w12: one-shot and chunked/mixed batches
# legitimately differ by a few bf16 ULPs that amplify across decoder layers.
STATE_DRIFT_RMSE_LIMITS = {
    "conv": 0.05,
    "recurrent": 0.02,
    "kv": 0.05,
}


def competing_tokens_are_tied(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    baseline_token: Any,
    candidate_token: Any,
) -> bool:
    """Accept a winner flip explainable by bounded bf16 perturbations."""
    if (
        not baseline
        or not candidate
        or not isinstance(baseline_token, int)
        or not isinstance(candidate_token, int)
        or baseline_token == candidate_token
    ):
        return False
    rows = []
    competing_tokens = {baseline_token, candidate_token}
    for diagnostic in (baseline, candidate):
        tokens = diagnostic.get("top_tokens", [])
        values = diagnostic.get("top_logits", [])
        if len(tokens) < 2 or len(values) < 2:
            return False
        if set(tokens[:2]) != competing_tokens:
            return False
        value_by_token = dict(zip(tokens[:2], values))
        gap = abs(
            float(value_by_token[baseline_token])
            - float(value_by_token[candidate_token])
        )
        tolerance = float(diagnostic.get("effective_tolerance", 0.0))
        rows.append((value_by_token, gap, tolerance))

    if all(gap <= tolerance for _, gap, tolerance in rows):
        return True

    baseline_values, _, baseline_tolerance = rows[0]
    candidate_values, _, candidate_tolerance = rows[1]
    cross_run_tolerance = max(baseline_tolerance, candidate_tolerance)
    return all(
        abs(float(baseline_values[token]) - float(candidate_values[token]))
        <= cross_run_tolerance
        for token in competing_tokens
    )


def sampling_diagnostic_summary(
    diagnostic: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not diagnostic:
        return None
    return {
        key: diagnostic.get(key)
        for key in (
            "top_tokens",
            "top_logits",
            "margin",
            "effective_tolerance",
            "near_tie",
            "logits_dtype",
        )
    }


def logits_diff_summary(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Full-logit diff stats; used as diagnostics, not as the tie gate."""
    left = baseline.get("logits") if baseline else None
    right = candidate.get("logits") if candidate else None
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return None
    if left.shape != right.shape:
        return {"shape_mismatch": [list(left.shape), list(right.shape)]}
    a, b = left.float(), right.float()
    diff = b - a
    denominator = float(a.norm() * b.norm())
    cosine = (
        float((a * b).sum() / denominator)
        if denominator > 0 and torch.isfinite(diff).all()
        else float("nan")
    )
    return {
        "max_abs": float(diff.abs().max()),
        "mean_abs": float(diff.abs().mean()),
        "cosine": cosine,
    }


def token_mismatch_records(
    reference_tokens: list[int],
    candidate_tokens: list[int],
    reference_diagnostics: dict[int, dict[str, Any]],
    candidate_diagnostics: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for position, (reference_token, candidate_token) in enumerate(
        zip(reference_tokens, candidate_tokens)
    ):
        if reference_token == candidate_token:
            continue
        reference_diag = reference_diagnostics.get(position)
        candidate_diag = candidate_diagnostics.get(position)
        records.append({
            "position": position,
            "reference_token": reference_token,
            "candidate_token": candidate_token,
            "near_tie": competing_tokens_are_tied(
                reference_diag, candidate_diag,
                reference_token, candidate_token,
            ),
            "reference_logits": sampling_diagnostic_summary(reference_diag),
            "candidate_logits": sampling_diagnostic_summary(candidate_diag),
            "logits_diff": logits_diff_summary(reference_diag, candidate_diag),
        })
    if len(reference_tokens) != len(candidate_tokens):
        records.append({
            "position": min(len(reference_tokens), len(candidate_tokens)),
            "length_mismatch": [len(reference_tokens), len(candidate_tokens)],
            "near_tie": False,
        })
    return records


def compare_probed_request_drift(
    reference_probe,
    candidate_probe,
    label: str,
    *,
    output_tokens: int,
    atol: float,
    rtol: float,
    drift_limits: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Bounded-drift acceptance for one-shot vs chunked/mixed batches.

    PASS requires: full checkpoint coverage, finite states, per-group relative
    RMSE below ``drift_limits``, and either exact tokens or a first mismatch
    classified as a bounded bf16 near-tie. Elementwise RMSE/max_abs and the
    requested logits comparison remain diagnostics.
    """
    limits = dict(STATE_DRIFT_RMSE_LIMITS if drift_limits is None else drift_limits)
    a, b = reference_probe.seqs[label], candidate_probe.seqs[label]
    ref_tokens, got_tokens = list(a.completion_token_ids), list(b.completion_token_ids)
    token_records = token_mismatch_records(
        ref_tokens, got_tokens,
        reference_probe.diagnostics.get(label, {}),
        candidate_probe.diagnostics.get(label, {}),
    )
    tokens_exact = not token_records and len(ref_tokens) == len(got_tokens)
    first_token = token_records[0]["position"] if token_records else None
    token_pass = tokens_exact or (
        len(ref_tokens) == len(got_tokens)
        and bool(token_records[0].get("near_tie"))
    )

    left, right = reference_probe.snapshots[label], candidate_probe.snapshots[label]
    coverage_ok = set(left) == set(right) == set(range(output_tokens))
    checks = []
    for index in sorted(set(left) & set(right)):
        diff = tensor_snapshot_diff(left[index], right[index], atol=atol, rtol=rtol)
        drift_pass = all(
            group.get("nonfinite", 0) == 0
            and group.get("relative_rmse", float("inf")) <= limits.get(name, float("inf"))
            for name, group in diff["groups"].items()
        )
        checks.append({"completion_index": index,
                       "stage": "prefill_end" if index == 0 else f"decode_{index}",
                       "same_input_history": ref_tokens[:index] == got_tokens[:index],
                       "drift_pass": drift_pass, **diff})
    # Once a near-tie flip changes the sampled token, every later forward
    # consumes a different input history. Those checkpoints stay in the report
    # for diagnostics but must not gate cross-run state equality.
    gating_checks = [
        check for check in checks if check["same_input_history"]
    ]
    state_pass = coverage_ok and all(check["drift_pass"] for check in gating_checks)
    return {
        "pass": token_pass and state_pass,
        "tokens_exact": tokens_exact,
        "token_pass": token_pass,
        "reference_tokens": ref_tokens,
        "candidate_tokens": got_tokens,
        "first_token_mismatch": first_token,
        "token_mismatches": token_records,
        "first_token_mismatch_detail": token_records[0] if token_records else None,
        "checkpoint_coverage": coverage_ok,
        "state_pass": state_pass,
        "state_checks": checks,
        "first_state_mismatch": next(
            (check for check in gating_checks if not check["drift_pass"]), None
        ),
        "post_divergence_check_count": len(checks) - len(gating_checks),
        "drift_limits": limits,
    }


class FullStateProbe:
    """Observe production dispatches; validate metadata and capture checkpoints.

    No scheduler replacement, no fabricated mixed batch, no teacher forcing.
    The sampler stays the suite's deterministic ArgmaxSampler. A token mismatch
    is recorded with top-2/margin/full-logit diagnostics; w11/w12 acceptance
    treats a bounded first-flip near-tie as numeric noise, not a logic error.
    """

    def __init__(
        self,
        engine,
        labeled_seqs,
        snapshot_positions: set[int] | None = None,
        snapshot_labels: set[str] | None = None,
    ):
        self.engine = engine
        self.runner = engine.model_runner
        self.seqs = dict(labeled_seqs)
        self.labels = {seq.seq_id: label for label, seq in self.seqs.items()}
        self.snapshots = {label: {} for label in self.seqs}
        self.snapshot_positions = (
            None if snapshot_positions is None else set(snapshot_positions)
        )
        self.snapshot_labels = (
            None if snapshot_labels is None else set(snapshot_labels)
        )
        self.diagnostics: dict[str, dict[int, dict[str, Any]]] = {}
        self.batches = []
        self.current = None
        self.isolation_checks = 0

    def __enter__(self):
        from contextlib import ExitStack
        self.stack = ExitStack()
        def patch(obj, name, value):
            owned = name in vars(obj)
            original = getattr(obj, name)
            setattr(obj, name, value)
            if owned:
                self.stack.callback(setattr, obj, name, original)
            else:
                self.stack.callback(delattr, obj, name)
            return original
        self.original_execute = patch(self.runner, "execute_model", self.execute)
        self.original_prepare = patch(self.runner, "prepare_inputs", self.prepare)
        self.original_post = patch(self.engine.scheduler, "postprocess", self.postprocess)
        patch(self.engine.scheduler, "preempt", self.reject_preempt)
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def reject_preempt(self, seq):
        raise AssertionError(f"Unexpected preemption of {seq.seq_id}; workload needs more KV capacity")

    def execute(self, seqs, is_prefill):
        self.current = list(seqs)
        rows = [{"label": self.labels[s.seq_id], "prefill": bool(s.is_prefill),
                 "start": s.num_cached_tokens, "q_len": s.num_scheduled_tokens,
                 "completion_index": s.num_completion_tokens} for s in seqs]
        if len({s.seq_id for s in seqs}) != len(seqs):
            raise AssertionError("Duplicate request in dispatched batch")
        if sum(r["q_len"] for r in rows) > self.engine.scheduler.max_num_batched_tokens:
            raise AssertionError("Scheduler exceeded token budget")
        if bool(is_prefill) != any(r["prefill"] for r in rows):
            raise AssertionError("Batch any_prefill disagrees with request flags")
        active_ids = {s.seq_id for s in seqs}
        idle = [(label, s) for label, s in self.seqs.items()
                if s.seq_id not in active_ids and not s.is_finished
                and s.seq_id in self.runner.input_batch.seq_id_to_slot and s.num_cached_tokens > 0]
        before = {label: full_request_snapshot(self.runner, s, s.num_cached_tokens) for label, s in idle}
        self.original_execute(seqs, is_prefill)
        for label, s in idle:
            after = full_request_snapshot(self.runner, s, s.num_cached_tokens)
            diff = tensor_snapshot_diff(before[label], after, atol=0, rtol=0)
            if not diff["pass"]:
                raise AssertionError(f"Unscheduled request {label} state/KV changed: {diff}")
            self.isolation_checks += 1
        for s in seqs:
            end = s.num_cached_tokens + s.num_scheduled_tokens
            # Ignore intermediate prefill logits: they must not become output.
            if end == s.num_tokens:
                label = self.labels[s.seq_id]
                index = s.num_completion_tokens
                if ((self.snapshot_positions is None or index in self.snapshot_positions)
                        and (self.snapshot_labels is None or label in self.snapshot_labels)):
                    if index in self.snapshots[label]:
                        raise AssertionError("Duplicate completion checkpoint")
                    self.snapshots[label][index] = full_request_snapshot(self.runner, s, end)
        self.batches.append({"any_prefill": bool(is_prefill), "rows": rows,
                             "metadata": self.last_metadata})
        self.current = None

    def prepare(self, seqs, is_prefill):
        from nano_qwen.utils.context import get_context
        result = self.original_prepare(seqs, is_prefill)
        ctx = get_context()
        input_ids, positions, _ = result
        slots = [self.runner.input_batch.seq_id_to_slot[s.seq_id] for s in seqs]
        if len(set(slots)) != len(slots):
            raise AssertionError("GDN persistent slots alias between requests")
        def as_list(t):
            return t.detach().cpu().tolist() if t is not None else None
        def equal(actual, expected, what):
            if actual != expected:
                raise AssertionError(f"{what}: actual={actual!r}, expected={expected!r}")
        equal(as_list(ctx.state_indices), slots, "state_indices")
        equal(as_list(self.runner.batch_slots_gpu[:len(seqs)]), slots, "batch_slots_gpu")
        expected_ids, expected_pos, expected_mapping = [], [], []
        cuq, cuk = [0], [0]
        block_sets = []
        for s in seqs:
            start, qlen = s.num_cached_tokens, s.num_scheduled_tokens
            end = start + qlen
            if qlen <= 0 or end > s.num_tokens:
                raise AssertionError("Invalid scheduled query length")
            expected_ids.extend(s.token_ids[start:end])
            expected_pos.extend(range(start, end))
            expected_mapping.extend(s.block_table[p // self.runner.block_size] * self.runner.block_size
                                    + p % self.runner.block_size for p in range(start, end))
            cuq.append(cuq[-1] + qlen)
            cuk.append(cuk[-1] + end)
            block_sets.append(set(s.block_table))
        if any(a & b for i, a in enumerate(block_sets) for b in block_sets[i + 1:]):
            raise AssertionError("Requests alias KV blocks with prefix cache disabled")
        equal(as_list(input_ids), expected_ids, "input_ids / sampled-token feedback")
        equal(as_list(positions), expected_pos, "positions")
        equal(as_list(ctx.slot_mapping), expected_mapping, "KV slot_mapping")
        equal(ctx.is_prefill, bool(is_prefill), "Context.is_prefill")
        if is_prefill:
            equal(as_list(ctx.cu_seqlens_q), cuq, "cu_seqlens_q")
            equal(as_list(ctx.cu_seqlens_k), cuk, "cu_seqlens_k")
            equal(ctx.prefill_slices, list(zip(cuq[:-1], cuq[1:])), "prefill_slices")
            chunks = [[i, j] for i, s in enumerate(seqs) for j in range((s.num_scheduled_tokens + 63) // 64)]
            equal(as_list(ctx.prefill_chunk_indices), chunks, "GDN CHUNK_SIZE=64 indices")
        else:
            equal(as_list(ctx.context_lens), [s.num_cached_tokens + 1 for s in seqs], "context_lens")
        needs_paged = not is_prefill or any(s.num_cached_tokens > 0 for s in seqs)
        if needs_paged and ctx.block_tables is None:
            raise AssertionError("Continuation/mixed/decode must use paged KV metadata")
        if ctx.block_tables is not None:
            for actual, s in zip(as_list(ctx.block_tables), seqs):
                equal(actual[:len(s.block_table)], s.block_table, "block_tables")
        self.last_metadata = {"state_slots": slots, "q_lens": [s.num_scheduled_tokens for s in seqs],
                              "paged": ctx.block_tables is not None,
                              "positions": expected_pos, "slot_mapping": expected_mapping}
        return result

    def postprocess(self, seqs, token_ids, is_prefill):
        before = [(s.num_cached_tokens, s.num_scheduled_tokens, s.num_tokens, s.num_completion_tokens)
                  for s in seqs]
        if len(token_ids) != len(seqs):
            raise AssertionError("Sample count differs from number of scheduled requests")
        result = self.original_post(seqs, token_ids, is_prefill)
        for s, (cached, qlen, ntokens, completed) in zip(seqs, before):
            expected = completed + int(cached + qlen == ntokens)
            if s.num_completion_tokens != expected:
                raise AssertionError("Intermediate chunk emitted token, or final/decode token was lost")
            # BlockManager.deallocate resets num_cached_tokens on completion.
            expected_cached = 0 if s.is_finished else cached + qlen
            if s.num_cached_tokens != expected_cached:
                raise AssertionError("postprocess num_cached_tokens drift")
        return result


def compare_probed_request(reference_probe, candidate_probe, label, *, output_tokens, atol, rtol):
    a, b = reference_probe.seqs[label], candidate_probe.seqs[label]
    ref_tokens, got_tokens = list(a.completion_token_ids), list(b.completion_token_ids)
    token_ok = ref_tokens == got_tokens and len(ref_tokens) == output_tokens
    first_token = next((i for i, (x, y) in enumerate(zip(ref_tokens, got_tokens)) if x != y), None)
    if first_token is None and len(ref_tokens) != len(got_tokens):
        first_token = min(len(ref_tokens), len(got_tokens))
    left, right = reference_probe.snapshots[label], candidate_probe.snapshots[label]
    coverage_ok = set(left) == set(right) == set(range(output_tokens))
    checks = []
    for index in sorted(set(left) & set(right)):
        diff = tensor_snapshot_diff(left[index], right[index], atol=atol, rtol=rtol)
        checks.append({"completion_index": index,
                       "stage": "prefill_end" if index == 0 else f"decode_{index}",
                       "same_input_history": ref_tokens[:index] == got_tokens[:index], **diff})
    return {"pass": token_ok and coverage_ok and all(c["pass"] for c in checks),
            "tokens_exact": token_ok, "reference_tokens": ref_tokens, "candidate_tokens": got_tokens,
            "first_token_mismatch": first_token, "checkpoint_coverage": coverage_ok,
            "state_checks": checks,
            "first_state_mismatch": next((c for c in checks if not c["pass"]), None)}


def drive_probe(
    engine,
    labeled_seqs,
    *,
    after_step=None,
    step_cap=4096,
    snapshot_positions: set[int] | None = None,
    snapshot_labels: set[str] | None = None,
    capture_state: bool = False,
    capture_logits: bool = True,
):
    """Requests already enqueued; allow a late-arrival callback between steps."""
    sampler = engine.model_runner.sampler
    diagnostics_available = all(
        hasattr(sampler, name)
        for name in (
            "enable_diagnostics", "disable_diagnostics",
            "get_diagnostic_map", "clear_diagnostics",
        )
    )
    diagnostic_keys: dict[str, str] = {}
    probe = None
    if diagnostics_available:
        sampler.enable_diagnostics(
            capture_state=capture_state, capture_logits=capture_logits,
        )
        for label, seq in labeled_seqs.items():
            key = f"probe.{next(_PROBE_KEY_COUNTER)}.{label}"
            seq._diagnostic_key = key
            diagnostic_keys[label] = key
    try:
        with FullStateProbe(
            engine, labeled_seqs, snapshot_positions=snapshot_positions,
            snapshot_labels=snapshot_labels,
        ) as probe, torch.inference_mode():
            steps = 0
            while not engine.is_finished():
                engine.step()
                steps += 1
                if after_step is not None:
                    after_step(probe)
                if steps > step_cap:
                    raise AssertionError("W11/W12 scheduler did not drain before step cap")
    finally:
        if diagnostics_available:
            if probe is not None:
                for label, key in diagnostic_keys.items():
                    probe.diagnostics[label] = dict(sampler.get_diagnostic_map(key))
            sampler.clear_diagnostics(list(diagnostic_keys.values()))
            sampler.disable_diagnostics()
    if any(not s.is_finished for s in labeled_seqs.values()):
        raise AssertionError("A required request was never submitted or did not finish")
    if engine.model_runner.input_batch.seq_id_to_slot:
        raise AssertionError("Request slot leak after workload drained")
    return probe
