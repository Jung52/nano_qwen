"""Shared baseline/timing/statistics helpers for the PD handoff benchmarks."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
from time import perf_counter_ns
from unittest.mock import patch
from contextlib import ExitStack


STAGES = ("prefill_ms", "export_gpu_to_cpu_ms", "validate_ms",
          "restore_cpu_to_gpu_ms", "ack_ms", "health_ms", "identity_ms", "decode_ms")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class Recorder:
    def __init__(self, clock=perf_counter_ns):
        self.clock = clock
        self.ms = defaultdict(float)
        self.calls = defaultdict(int)
        self.dispatches = defaultdict(lambda: dict(prefill=0, decode=0))
        self.phase = "outside"
        self.last_prefill_end = None

    @contextmanager
    def span(self, name):
        start = self.clock()
        try:
            yield
        finally:
            self.ms[name] += (self.clock() - start) / 1e6
            self.calls[name] += 1

    @contextmanager
    def in_phase(self, name):
        previous, self.phase = self.phase, name
        try:
            yield
        finally:
            self.phase = previous

    @contextmanager
    def instrument(self, engine, engine_ops):
        original_step, original_call = engine.step, engine.model_runner.call
        method_stages = {"export_request_state": "export_gpu_to_cpu_ms",
                         "restore_request_state": "restore_cpu_to_gpu_ms",
                         "validate_request_state": "validate_ms",
                         "pd_model_identity": "identity_ms"}

        def step(*args, **kwargs):
            if self.phase != "prefill":
                return original_step(*args, **kwargs)
            with self.span("prefill_ms"):
                result = original_step(*args, **kwargs)
            self.last_prefill_end = self.clock()
            return result

        def call(method, *args):
            if method == "execute_model":
                self.dispatches[self.phase]["prefill" if args[1] else "decode"] += 1
            stage = method_stages.get(method)
            if stage is None:
                return original_call(method, *args)
            with self.span(stage):
                return original_call(method, *args)

        with ExitStack() as stack:
            for obj, name, replacement in ((engine, "step", step),
                    (engine.model_runner, "call", call)):
                stack.enter_context(patch.object(obj, name, replacement))
            yield

    def finish(self, e2e_ms):
        values = {name: self.ms[name] for name in STAGES}
        residual = e2e_ms - sum(values.values())
        if residual < -0.01:
            raise RuntimeError(f"overlapping stage timers: residual={residual} ms")
        values.update(e2e_ms=e2e_ms, other_ms=max(0.0, residual))
        return values


def require_clean(engine, ops):
    ops.require_idle(engine)
    if engine.scheduler.block_manager.used_block_ids:
        raise RuntimeError("request leaked KV blocks")
    if engine.model_runner.input_batch.seq_id_to_slot:
        raise RuntimeError("request leaked state slots")


def complete(engine, seq):
    for _ in range(seq.max_tokens + 2):
        if engine.is_finished():
            return
        engine.step()
    raise RuntimeError("decode exceeded the output-token bound")


def baseline(engine, tokens, params, sequence_type, ops, synchronize):
    require_clean(engine, ops)
    rec = Recorder()
    with rec.instrument(engine, ops):
        synchronize()
        start = rec.clock()
        seq = sequence_type(tokens, params)
        engine.scheduler.add(seq)
        with rec.in_phase("prefill"):
            for _ in range(len(tokens) + 2):
                engine.step()
                if seq.num_cached_tokens == len(tokens) and seq.num_completion_tokens == 1:
                    break
            else:
                raise RuntimeError("baseline did not reach the first-token boundary")
        first = rec.clock()
        with rec.in_phase("decode"), rec.span("decode_ms"):
            complete(engine, seq)
            synchronize()
        end = rec.clock()
    values = rec.finish((end - start) / 1e6)
    values["baseline_ttft_ms"] = (first - start) / 1e6
    values["tpot_ms"] = values["decode_ms"] / (params.max_tokens - 1)
    output = list(seq.completion_token_ids)
    if len(output) != params.max_tokens:
        raise RuntimeError("baseline produced an unexpected number of tokens")
    if rec.dispatches["decode"] != dict(prefill=0, decode=params.max_tokens - 1):
        raise RuntimeError(f"unexpected baseline decode dispatches: {dict(rec.dispatches)}")
    require_clean(engine, ops)
    return dict(metrics=values, calls=dict(rec.calls), dispatches=dict(rec.dispatches)), output


def stats(values):
    return dict(n=len(values), median=statistics.median(values), min=min(values),
                max=max(values), stdev=statistics.stdev(values) if len(values) > 1 else 0.0)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("passed") and not row.get("warmup", False):
            groups[row["prompt_tokens"], row["output_len"]].append(row)
    result = []
    for (length, output), group in sorted(groups.items()):
        item = dict(prompt_tokens=length, output_len=output, pairs=len(group))
        for name in ("baseline", "handoff"):
            item[name] = {metric: stats([row[name]["metrics"][metric] for row in group])
                          for metric in group[0][name]["metrics"]}
        item["paired_e2e_delta_ms"] = stats([
            r["handoff"]["metrics"]["e2e_ms"] - r["baseline"]["metrics"]["e2e_ms"] for r in group])
        item["paired_e2e_ratio_handoff_over_baseline"] = stats([
            r["handoff"]["metrics"]["e2e_ms"] / r["baseline"]["metrics"]["e2e_ms"] for r in group])
        item["paired_tpot_ratio_handoff_over_baseline"] = stats([
            r["handoff"]["metrics"]["tpot_ms"] / r["baseline"]["metrics"]["tpot_ms"] for r in group])
        result.append(item)
    return result


def git_provenance(root):
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], text=True,
                              capture_output=True, check=True, timeout=15).stdout.rstrip("\n")
    try:
        status = git("status", "--porcelain=v1", "--untracked-files=all")
        return dict(head=git("rev-parse", "HEAD"), dirty=bool(status), status=status,
                    branch=git("rev-parse", "--abbrev-ref", "HEAD"))
    except (OSError, subprocess.SubprocessError) as exc:
        return dict(head=None, dirty=None, error=str(exc))


def source_hashes(root):
    paths = sorted((root / "src/nano_qwen").rglob("*.py"))
    paths += [Path(__file__).resolve()]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def verify_origins(modules, root):
    origins = {module.__name__: str(Path(module.__file__).resolve()) for module in modules}
    for name, path in origins.items():
        if not Path(path).is_relative_to(root / "src/nano_qwen"):
            raise RuntimeError(f"wrong checkout imported: {name} from {path}; expected {root / 'src/nano_qwen'}")
    return origins


def schedule(prompt_lens, output_lens, warmup_rounds, repeats, seed):
    matrix = [(n, o) for n in prompt_lens for o in output_lens]
    rng, measured_index = random.Random(seed), 0
    for warmup, rounds in ((True, warmup_rounds), (False, repeats)):
        for repetition in range(rounds):
            cases = list(matrix)
            if not warmup:
                rng.shuffle(cases)
            for length, output in cases:
                order = ("baseline", "handoff") if warmup or measured_index % 2 == 0 else ("handoff", "baseline")
                yield warmup, repetition, length, output, order
                if not warmup:
                    measured_index += 1
