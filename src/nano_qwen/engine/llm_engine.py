import atexit
import gc
from collections import deque
from dataclasses import dataclass, fields
from time import perf_counter
from typing import Iterator

import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nano_qwen.config import Config
from nano_qwen.engine.sequence import Sequence, SequenceStatus
from nano_qwen.sampling_params import SamplingParams
from nano_qwen.utils.trace import trace_event

from .model_runner import ModelRunner, find_free_port
from ..scheduler import Scheduler


@dataclass(frozen=True)
class StreamChunk:
    """Newly committed token IDs, including EOS when it terminates a request."""

    request_id: int
    token_ids: list[int]
    finished: bool
    finish_reason: str | None = None


class LLMEngine:
    """Thin control plane for scheduling and model execution.

    The engine intentionally does not prepare model inputs or own GPU state.
    Those responsibilities belong to ModelRunner.  Keeping the boundary small
    makes the execution path compatible with the ModelRunnerV2 direction:

        schedule -> execute_model -> sample_tokens -> postprocess

    With async_scheduling=True, stable decode can enqueue one batch ahead.
    CPU token history is committed in order; GPU tokens feed the next decode.
    """

    def __init__(self, model: str, use_prefill_cudagraph: bool = True, **kwargs):
        self._exited = False
        config_fields = {field.name for field in fields(Config) if field.init}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)
        Sequence.block_size = self.config.kvcache_block_size

        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        # Pick the NCCL rendezvous port once so every rank joins the same
        # store; an ephemeral port avoids EADDRINUSE from the previous run's
        # socket lingering in TIME_WAIT.
        dist_port = find_free_port()
        for rank in range(1, self.config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(
                target=ModelRunner,
                args=(self.config, rank, event, dist_port, use_prefill_cudagraph),
            )
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.model_runner = ModelRunner(
            self.config,
            0,
            self.events,
            dist_port,
            use_prefill_cudagraph,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            use_fast=True,
        )
        self.config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(self.config)
        # MRV2 async batch queue: up to max_concurrent_batches batches in
        # flight, CPU runs ahead of GPU by N-1 steps (core.py:622-736).
        self.max_concurrent_batches = 2
        self.batch_queue: deque[tuple[list[Sequence], bool, int, object]] = deque()
        self.coalesce_stats = {
            "coalesce_attempts": 0,
            "coalesce_successes": 0,
            "merged_batches": 0,
            "merged_requests": 0,
            "batch_size_before": {},
            "batch_size_after": {},
        }
        self._coalesce_armed = False
        self._coalesce_before = 0
        self._coalesce_source_ids: frozenset[int] = frozenset()
        # Keep the exact callback so manual exit can unregister it. Leaving a
        # bound method in atexit keeps the whole engine alive until interpreter
        # shutdown, including CUDA graph/model references.
        self._atexit_callback = self.exit
        atexit.register(self._atexit_callback)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        callback = getattr(self, "_atexit_callback", None)
        if callback is not None:
            atexit.unregister(callback)
            self._atexit_callback = None

        runner = getattr(self, "model_runner", None)
        if runner is not None:
            try:
                runner.call("exit")
            finally:
                self.model_runner = None
        for process in getattr(self, "ps", []):
            process.join()

        # AsyncModelOutput retains both device and pinned host tensors until
        # consumed. Drop all engine-owned queues/references before collecting.
        self.batch_queue.clear()
        self.ps.clear()
        self.events.clear()
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None:
            scheduler.waiting.clear()
            scheduler.running.clear()
            scheduler.in_flight.clear()
            scheduler.async_decode = None
            self.scheduler = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams,
    ) -> None:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        self.scheduler.add(Sequence(prompt, sampling_params))

    def step(self):
        """Use lookahead only when the two-buffer output queue can support it."""
        active = getattr(self.scheduler, "async_decode", None)
        enabled = (getattr(self.config, "async_scheduling", False)
                   and self.max_concurrent_batches == 2
                   and getattr(self.model_runner, "async_output", True))
        if active is not None or (enabled and not self.batch_queue
                                  and self.scheduler.begin_async_decode()):
            return self._step_async_decode()
        return self._step_serial()

    def _step_async_decode(self):
        cohort = self.scheduler.async_decode
        # New arrivals need the normal admission/preemption policy. Retire all
        # queued work before returning owners to that policy.
        if self.scheduler.waiting:
            cohort.draining = True
        while len(self.batch_queue) < self.max_concurrent_batches and not cohort.draining:
            with trace_event("schedule", "engine"):
                seqs, is_prefill = self.scheduler.schedule()
            if not seqs:
                break
            assert not is_prefill
            with trace_event("execute_model", "engine", args={"prefill": False, "bs": len(seqs)}):
                self.model_runner.call("execute_model", seqs, False)
            with trace_event("sample_tokens", "engine"):
                async_output = self.model_runner.call("sample_tokens")
            assert async_output is not None
            self.batch_queue.append((seqs, False, -len(seqs), async_output))

        if not self.batch_queue:
            self.scheduler.end_async_decode()
            return self._step_serial()
        seqs, is_prefill, num_tokens, async_output = self.batch_queue.popleft()
        with trace_event("d2h_wait", "engine", args={"prefill": False, "bs": len(seqs)}):
            token_ids = async_output.get_output()
        finished, released = self.scheduler.postprocess(seqs, token_ids, is_prefill)
        for seq_id in released:
            self.model_runner.call("remove_request", seq_id)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in finished]
        if not self.batch_queue:
            self.scheduler.end_async_decode()
        return outputs, num_tokens

    def _step_serial(self):
        """MRV2 async scheduling step with batch queue (core.py:622-736).

        Phase 1 (dispatch): while the queue has room, schedule a batch and
        enqueue execute_model + sample_tokens back-to-back (both non-blocking).
        Phase 2 (consume): pop the oldest batch and block only on its D2H
        event, then postprocess. Zombie seqs are handled by Scheduler.in_flight.

        Decode coalescing: dispatching a decode batch while another decode
        batch is still in flight permanently splits the ready set into
        cohorts (e.g. 11/21 instead of 32), because the in-flight rows are
        not in ``running`` when the next batch is formed. Once the queued
        work contains decode rows, wait for it to drain before scheduling
        more work. Ready decode receives the next token budget first;
        waiting prompt chunks use its remainder.
        """
        # Phase 1: fill the queue (never blocks).
        while len(self.batch_queue) < self.max_concurrent_batches:
            decode_batches = [
                batch_seqs for batch_seqs, _, _, _ in self.batch_queue
                if any(not seq.is_prefill for seq in batch_seqs)
            ]
            if decode_batches:
                if self.scheduler.running and not self._coalesce_armed:
                    source_seqs = [
                        seq for batch_seqs in decode_batches for seq in batch_seqs
                        if not seq.is_prefill
                    ]
                    self._coalesce_armed = True
                    self._coalesce_before = len(source_seqs)
                    self._coalesce_source_ids = frozenset(
                        seq.seq_id for seq in source_seqs
                    )
                    self.coalesce_stats["coalesce_attempts"] += 1
                    self.coalesce_stats["batch_size_before"][
                        self._coalesce_before
                    ] = (
                        self.coalesce_stats["batch_size_before"].get(
                            self._coalesce_before, 0
                        )
                        + 1
                    )
                break
            with trace_event("schedule", "engine"):
                seqs, is_prefill = self.scheduler.schedule()
            if not seqs:
                break
            batch_ids = [seq.seq_id for seq in seqs]
            assert len(set(batch_ids)) == len(batch_ids), (
                "engine produced a duplicate request in one dispatch batch"
            )
            queued_ids = {
                seq.seq_id
                for queued_seqs, _, _, _ in self.batch_queue
                for seq in queued_seqs
            }
            assert not (set(batch_ids) & queued_ids), (
                "engine dispatched a request that is already in flight"
            )
            if self._coalesce_armed:
                source_still_pending = bool(
                    self._coalesce_source_ids & queued_ids
                )
                if not is_prefill:
                    batch_size_after = len(seqs)
                    coalesced = (
                        batch_size_after > self._coalesce_before
                        and bool(self._coalesce_source_ids & set(batch_ids))
                    )
                    if coalesced:
                        self.coalesce_stats["coalesce_successes"] += 1
                        self.coalesce_stats["merged_batches"] += 1
                        self.coalesce_stats["merged_requests"] += batch_size_after
                    self.coalesce_stats["batch_size_after"][batch_size_after] = (
                        self.coalesce_stats["batch_size_after"].get(
                            batch_size_after, 0
                        )
                        + 1
                    )
                    self._coalesce_armed = False
                elif not source_still_pending and not (
                    self._coalesce_source_ids
                    & {seq.seq_id for seq in seqs}
                ):
                    self._coalesce_armed = False
            num_tokens = (
                sum(seq.num_scheduled_tokens for seq in seqs)
                if is_prefill
                else -len(seqs)
            )
            with trace_event(
                "execute_model", "engine",
                args={"prefill": is_prefill, "bs": len(seqs)},
            ):
                self.model_runner.call("execute_model", seqs, is_prefill)
            with trace_event("sample_tokens", "engine"):
                async_output = self.model_runner.call("sample_tokens")
            assert async_output is not None
            self.batch_queue.append((seqs, is_prefill, num_tokens, async_output))

        # Phase 2: consume the oldest batch.
        if not self.batch_queue:
            return [], 0
        seqs, is_prefill, num_tokens, async_output = self.batch_queue.popleft()
        with trace_event(
            "d2h_wait", "engine",
            args={"prefill": is_prefill, "bs": len(seqs)},
        ):
            token_ids = async_output.get_output()

        for seq, token_id in zip(seqs, token_ids):
            if (
                is_prefill
                and seq.num_cached_tokens + seq.num_scheduled_tokens
                < seq.num_tokens
            ):
                continue
            if (
                (not seq.ignore_eos and token_id == self.config.eos)
                or seq.num_completion_tokens + 1 == seq.max_tokens
            ):
                self.model_runner.call("remove_request", seq.seq_id)

        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs
            if seq.is_finished
        ]
        return outputs, num_tokens

    def prefill_to_lmcache(self, prompt, sampling_params, connector, transfer_id):
        """Prefill one request, commit its full KV/GDN state, release GPU state."""
        from nano_qwen.cache.engine_ops import prefill_to_cache
        return prefill_to_cache(self, prompt, sampling_params, connector, transfer_id, Sequence)

    def resume_from_lmcache(self, connector, transfer_id, delete_after_restore=True):
        """Restore a request into local blocks/slots; subsequent step() decodes."""
        from nano_qwen.cache.engine_ops import resume_from_cache
        from nano_qwen.engine.sequence import SequenceStatus
        from nano_qwen.sampling_params import SamplingParams
        return resume_from_cache(self, connector, transfer_id, Sequence, SamplingParams,
                                 SequenceStatus.RUNNING, delete_after_restore)

    def is_finished(self) -> bool:
        return self.scheduler.is_finished() and not self.batch_queue

    def generate_stream(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> Iterator[StreamChunk]:
        """Stream one request on an idle engine without changing scheduling.

        Read the authoritative owner only after step() commits a result.
        Async execution views may already be one token ahead and must never
        be exposed to the caller. The iterator exclusively owns the engine;
        exhaust or close it before adding another request. Closing early
        cancels at a yield boundary and retires queued work before slot reuse.
        """
        if not self.is_finished():
            raise RuntimeError("generate_stream requires an idle engine")
        if not isinstance(sampling_params.max_tokens, int) or sampling_params.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        tokens = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not tokens:
            raise ValueError("prompt must contain at least one token")
        if len(tokens) + sampling_params.max_tokens > self.config.max_model_len:
            raise ValueError(
                f"Prompt ({len(tokens)}) + max_tokens ({sampling_params.max_tokens}) "
                f"exceeds max_model_len ({self.config.max_model_len}); "
                "reduce the output budget or reset/shorten the conversation"
            )

        seq = Sequence(tokens, sampling_params)
        self.scheduler.add(seq)
        emitted = 0
        safe_to_retire = True
        try:
            while not seq.is_finished:
                # Do not hold a thread-local inference-mode context across a
                # yield: caller-side work should retain its original grad mode.
                safe_to_retire = False
                with torch.inference_mode():
                    self.step()
                safe_to_retire = True
                committed = seq.num_completion_tokens
                if committed == emitted:
                    continue  # An intermediate prefill chunk has no output.
                delta = seq.token_ids[seq.num_prompt_tokens + emitted:
                                      seq.num_prompt_tokens + committed]
                emitted = committed
                reason = None
                if seq.is_finished:
                    reason = (
                        "eos" if not seq.ignore_eos and seq.last_token == self.config.eos
                        else "length"
                    )
                yield StreamChunk(seq.seq_id, delta, seq.is_finished, reason)
        finally:
            # An execution failure/interrupt can occur between reservation
            # and enqueue. The caller must exit the engine in that case;
            # attempting normal retirement could mask the original error.
            if safe_to_retire:
                self._retire_stream_request(seq)

    def _retire_stream_request(self, seq: Sequence) -> None:
        """Cancel/finish an exclusive stream, draining before releasing state."""
        if getattr(self, "_exited", False):
            return
        # Finished owners discard lookahead results in AsyncDecodeCohort.
        # Marking an interrupted owner finished also prevents extra commits.
        seq.status = SequenceStatus.FINISHED
        cohort = self.scheduler.async_decode
        if cohort is not None:
            cohort.draining = True
        while self.batch_queue:
            with torch.inference_mode():
                self.step()
        if self.scheduler.async_decode is not None:
            self.scheduler.end_async_decode()

        for queue in (self.scheduler.waiting, self.scheduler.running):
            for owner in list(queue):
                if owner.seq_id == seq.seq_id:
                    queue.remove(owner)
        # Normal completion and async retirement already release their owner.
        # An early close immediately after prefill still owns its block/slot.
        if seq.block_table:
            self.scheduler.block_manager.deallocate(seq)
            self.model_runner.call("remove_request", seq.seq_id)
        self.scheduler.in_flight.discard(seq.seq_id)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(
            total=len(prompts),
            desc="Generating",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()

            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)

            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)

        pbar.close()

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids,
            }
            for token_ids in outputs
        ]
        return outputs
