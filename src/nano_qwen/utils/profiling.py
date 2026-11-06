"""Optional host annotations and trace export; no analysis or GPU work added."""
from collections import deque
from contextlib import ExitStack, contextmanager, suppress
import json
from pathlib import Path
import time


def init_kineto(torch):
    """Initialize the profiler before model CUDA Graph capture, if requested."""
    try:
        from torch.profiler._utils import _init_for_cuda_graphs
    except ImportError:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]):
            torch.empty(1, device="cuda").zero_()
            torch.cuda.synchronize()
    else:
        _init_for_cuda_graphs()


class Labels:
    """Attribute actual dispatches, including two dispatches in one CPU step.

    Only the supplied backbone Graph objects are tagged as ``launch``.
    FP8 micro-graphs inherit their enclosing head/sample stage instead.
    """
    def __init__(self, engine, torch, graphs, *, allow_prefill=False):
        self.engine, self.torch, self.graphs = engine, torch, graphs
        self.runner = engine.model_runner
        self.allow_prefill = allow_prefill
        self.undo = ExitStack()
        self.next_id = 0
        self.active = self.last_waited = None
        self.executed = deque()
        self.batches = []

    def mark(self, stage, bid):
        return self.torch.profiler.record_function(f"nq/{stage}/b{bid:05d}")

    def patch(self, obj, attr, value):
        own, old = attr in vars(obj), getattr(obj, attr)
        setattr(obj, attr, value)
        self.undo.callback(setattr if own else delattr, obj, attr, *([old] if own else []))

    @contextmanager
    def activate(self, bid):
        old, self.active = self.active, bid
        try:
            yield
        finally:
            self.active = old

    def __enter__(self):
        try:
            self._install()
        except BaseException:
            self.undo.close()
            raise
        return self

    def _install(self):
        r = self.runner
        original_execute = r.execute_model
        def execute(seqs, is_prefill):
            if is_prefill and not self.allow_prefill:
                raise RuntimeError("Prefill entered the decode-only diagnostic window")
            bid = self.next_id
            self.next_id += 1
            self.batches.append(dict(batch_id=bid, batch_size=len(seqs),
                                     is_prefill=bool(is_prefill), seq_ids=[s.seq_id for s in seqs]))
            with self.activate(bid), self.mark("execute", bid):
                result = original_execute(seqs, is_prefill)
            self.executed.append(bid)
            return result
        self.patch(r, "execute_model", execute)

        original_plan = self.engine.scheduler.schedule
        def plan(*args, **kwargs):
            with self.mark("schedule", self.next_id):
                return original_plan(*args, **kwargs)
        self.patch(self.engine.scheduler, "schedule", plan)

        for graph in self.graphs:
            original_replay = graph.replay
            def replay(*args, _original=original_replay, **kwargs):
                if self.active is None:
                    return _original(*args, **kwargs)
                with self.mark("launch", self.active):
                    return _original(*args, **kwargs)
            self.patch(graph, "replay", replay)

        original_logits = r.model.compute_logits
        def logits(*args, **kwargs):
            if self.active is None:
                return original_logits(*args, **kwargs)
            with self.mark("lm_head", self.active):
                return original_logits(*args, **kwargs)
        self.patch(r.model, "compute_logits", logits)

        original_sample = r.sample_tokens
        def sample(*args, **kwargs):
            bid = self.executed.popleft()
            with self.activate(bid), self.mark("sample_call", bid):
                output = original_sample(*args, **kwargs)
            if output is not None:
                original_get = output.get_output
                def get_output(*args, **kwargs):
                    with self.mark("result_wait", bid):
                        result = original_get(*args, **kwargs)
                    self.last_waited = bid
                    return result
                self.patch(output, "get_output", get_output)
            return output
        self.patch(r, "sample_tokens", sample)

        original_commit = self.engine.scheduler.postprocess
        def commit(*args, **kwargs):
            if self.last_waited is None:
                return original_commit(*args, **kwargs)
            bid, self.last_waited = self.last_waited, None
            with self.mark("commit", bid):
                return original_commit(*args, **kwargs)
        self.patch(self.engine.scheduler, "postprocess", commit)

    def __exit__(self, *exc):
        self.undo.close()


class ProfileSession:
    """Best-effort collection; failure restores wrappers and keeps chat usable."""
    def __init__(self, torch, engine, directory, turn_no):
        self.torch, self.engine = torch, engine
        self.directory, self.turn_no = Path(directory), turn_no
        self.prof = self.labels = None
        self.active = self.stopped = False
        self.message = None

    def start(self):
        try:
            graphs = list(self.engine.model_runner.cuda_graphs.decode_graphs.values())
            self.labels = Labels(self.engine, self.torch, graphs, allow_prefill=True)
            self.labels.__enter__()
            self.prof = self.torch.profiler.profile(activities=[
                self.torch.profiler.ProfilerActivity.CPU,
                self.torch.profiler.ProfilerActivity.CUDA])
            self.active = True
            self.prof.start()
        except Exception as exc:
            self.stop()
            if self.labels is not None:
                with suppress(Exception):
                    self.labels.__exit__()
            self.message = f"[trace] capture disabled: {exc!r}"
        return self

    def stop(self):
        if not self.active:
            return
        self.active = False  # Never retry stop() after an exception.
        try:
            self.labels.__exit__()
        except Exception as exc:
            self.message = f"[trace] annotation cleanup failed: {exc!r}"
        try:
            self.prof.stop()
            self.stopped = True
        except Exception as exc:
            self.message = f"[trace] capture stop failed: {exc!r}"

    def finish(self):
        """Called once after generation; export is kept out of the token loop."""
        self.stop()
        if self.message or not self.stopped:
            return self.message
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"turn{self.turn_no:03d}_{time.time_ns()}.trace.json.gz"
            self.prof.export_chrome_trace(str(path))
            path.with_name(path.name + ".metadata.json").write_text(json.dumps({
                "schema": "nq-dispatch-v1", "batches": self.labels.batches,
                "async_scheduling": self.engine.config.async_scheduling,
                "diagnostic_only": True,
            }, indent=2), encoding="utf-8")
            self.message = f"[trace] exported {path}; analyze with benchmarks/bench_one_batch_trace.py --analyze"
        except Exception as exc:
            self.message = f"[trace] export failed: {exc!r}"
        return self.message
