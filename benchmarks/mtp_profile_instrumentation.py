"""Benchmark-only scopes; no production methods or numerical dispatch changes."""

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from unittest.mock import patch
from time import perf_counter_ns

import torch

from mtp_profile_analysis import GraphEventBank


class PhaseRecorder:
    def __init__(self, runner, kind):
        if kind not in ("events", "trace", "host"):
            raise ValueError("kind must be events, trace or host")
        self.runner, self.kind = runner, kind
        self.bank = GraphEventBank()
        self.capture_key = None
        self.enabled = False
        self.verify = False
        self.mtp_owner = None
        self.in_sample = False
        self.runtime = []
        self.replays = []
        self.host = defaultdict(list)

    @contextmanager
    def scope(self, label, capture_rows=None):
        if self.kind == "host":
            if self.enabled:
                start = perf_counter_ns()
                try:
                    yield
                finally:
                    self.host[label].append((perf_counter_ns()-start)/1e6)
            else:
                yield
            return
        capturing = self.capture_key is not None and torch.cuda.is_current_stream_capturing()
        if self.kind == "trace":
            if self.enabled:
                with torch.profiler.record_function("mtp." + label):
                    yield
            else:
                yield
            return
        if not (capturing or self.enabled) or label == "decision_wait":
            yield
            return
        start = torch.cuda.Event(enable_timing=True, external=capturing)
        end = torch.cuda.Event(enable_timing=True, external=capturing)
        if capturing:
            key = self.capture_key if capture_rows is None else (self.capture_key[0], capture_rows)
            self.bank.add(key, label, start, end)
        else:
            self.runtime.append((label, start, end))
        start.record()
        try:
            yield
        finally:
            end.record()

    def begin(self):
        self.runtime, self.replays = [], []
        self.host = defaultdict(list)
        self.enabled = True

    def finish(self):
        self.enabled = False
        if self.kind == "host":
            return dict(host_ms={k: sum(v) for k, v in self.host.items()},
                        host_calls={k: len(v) for k, v in self.host.items()})
        if self.kind == "trace":
            return {}
        # Stage events are a separate diagnostic pass. Synchronization/event
        # queries inflate its host gaps; use the untouched timeline for gaps.
        torch.cuda.synchronize()
        result = defaultdict(float, self.bank.collect(self.replays))
        for label, start, end in self.runtime:
            result[label] += start.elapsed_time(end)
        return dict(gpu_ms=dict(result), graph_replays=list(self.replays))

    def recapture(self):
        if self.runner.input_batch.seq_id_to_slot:
            raise ValueError("Recapture requires an idle engine")
        if not self.runner.enforce_eager:
            # Ordinary decode's output allocation uses the constructor's
            # temporary model dtype. Recreate that context, then restore it.
            prior = torch.get_default_dtype()
            torch.set_default_dtype(self.runner.model.model.embed_tokens.weight.dtype)
            try:
                self.runner.cuda_graphs.capture()
            finally:
                torch.set_default_dtype(prior)

    @contextmanager
    def install(self):
        r = self.runner
        graphs = r.cuda_graphs

        def capture(fn, bank):
            def call(*args, **kwargs):
                prior = self.capture_key
                size = args[0] if bank != "decode" else 0
                key = (bank, size)
                if bank == "decode":
                    for existing in list(self.bank.events):
                        if existing[0] == bank:
                            self.bank.clear(existing)
                else:
                    self.bank.clear(key)
                self.capture_key = key
                try:
                    return fn(*args, **kwargs)
                finally:
                    self.capture_key = prior
            return call

        def mtp_capture(fn):
            def call(batch, cache_only):
                return capture(lambda *unused: fn(batch, cache_only), "cache" if cache_only else "draft")(batch)
            return call

        def replay(fn, bank, divisor=1):
            def call(ids, *args, **kwargs):
                if self.enabled and self.kind == "events":
                    self.replays.append((bank, ids.numel()//divisor))
                return fn(ids, *args, **kwargs)
            return call

        def tagged(fn, label):
            def call(*args, **kwargs):
                with self.scope(label):
                    return fn(*args, **kwargs)
            return call

        def target(fn):
            def call(ids, positions, prefill, is_verify=False):
                prior = self.verify
                self.verify = is_verify
                try:
                    with self.scope("target"):
                        return fn(ids, positions, prefill, is_verify)
                finally:
                    self.verify = prior
            return call

        def target_body(fn):
            def call(ids, *args, **kwargs):
                # Verify body has its own wrapper; ordinary target forward
                # occurs during decode graph capture or eager execution.
                if self.verify or (self.capture_key and self.capture_key[0] != "decode"):
                    return fn(ids, *args, **kwargs)
                with self.scope("target_body", capture_rows=ids.shape[0]):
                    return fn(ids, *args, **kwargs)
            return call

        def head(fn):
            def call(*args, **kwargs):
                draft = self.mtp_owner == "draft" or (self.capture_key and self.capture_key[0] == "draft")
                with self.scope("draft_head" if draft else "target_head"):
                    return fn(*args, **kwargs)
            return call

        def mtp(fn):
            def call(*args, **kwargs):
                prior = self.mtp_owner
                self.mtp_owner = "cache_only" if kwargs.get("cache_only") else "draft"
                try:
                    with self.scope(self.mtp_owner):
                        return fn(*args, **kwargs)
                finally:
                    self.mtp_owner = prior
            return call

        def sample(fn):
            def call(*args, **kwargs):
                self.in_sample = True
                try:
                    return fn(*args, **kwargs)
                finally:
                    self.in_sample = False
            return call

        tolist = torch.Tensor.tolist

        def decisions(tensor):
            # Unique GPU B×3 int64 table in sample_tokens; CPU AsyncModelOutput
            # conversion and unrelated tensor inspection retain their paths.
            if self.enabled and self.in_sample and tensor.is_cuda and tensor.ndim == 2 and tensor.shape[1] == 3 and tensor.dtype == torch.int64:
                with self.scope("decision_wait"):
                    return tolist(tensor)
            return tolist(tensor)

        try:
            with ExitStack() as stack:
                def replace(obj, name, fn):
                    stack.enter_context(patch.object(obj, name, fn))
                replace(graphs, "_capture_decode", capture(graphs._capture_decode, "decode"))
                replace(graphs, "_capture_verify_batch", capture(graphs._capture_verify_batch, "verify"))
                replace(graphs, "_capture_mtp_decode_batch", mtp_capture(graphs._capture_mtp_decode_batch))
                for name, bank, divisor in (("run_decode", "decode", 1), ("run_verify", "verify", 2),
                                             ("run_mtp_decode", "draft", 1), ("run_mtp_cache_only", "cache", 1)):
                    replace(graphs, name, replay(getattr(graphs, name), bank, divisor))
                replace(r, "run_model", target(r.run_model))
                replace(r.model.model, "forward", target_body(r.model.model.forward))
                replace(r, "forward_verify", tagged(r.forward_verify, "target_body"))
                replace(r.model, "compute_logits", head(r.model.compute_logits))
                replace(r, "_run_mtp_decode", mtp(r._run_mtp_decode))
                replace(r.mtp, "forward", tagged(r.mtp.forward, "draft_body"))
                replace(r.mtp, "cache_only", tagged(r.mtp.cache_only, "cache_body"))
                replace(r, "sample_tokens", sample(r.sample_tokens))
                replace(torch.Tensor, "tolist", decisions)
                for layer in r.gdn_layers:
                    for name, label in (("save_speculative_state", "snapshot"), ("rollback_speculative_states", "rollback")):
                        replace(layer, name, tagged(getattr(layer, name), label))
                if self.kind == "events":
                    self.recapture()
                yield self
        finally:
            self.enabled = False
            if self.kind == "events" and not r.input_batch.seq_id_to_slot:
                self.recapture()  # Original methods, no event nodes, for the timeline pass.
