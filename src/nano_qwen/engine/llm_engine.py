import atexit
import gc
from collections import deque
from dataclasses import fields
from time import perf_counter

import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nano_qwen.config import Config
from nano_qwen.engine.sequence import Sequence
from nano_qwen.sampling_params import SamplingParams
from nano_qwen.utils.trace import trace_event

from .model_runner import ModelRunner, find_free_port
from ..scheduler import Scheduler


class LLMEngine:
    """Thin control plane for scheduling and model execution.

    The engine intentionally does not prepare model inputs or own GPU state.
    Those responsibilities belong to ModelRunner.  Keeping the boundary small
    makes the execution path compatible with the ModelRunnerV2 direction:

        schedule -> execute_model -> sample_tokens -> postprocess

    Each step consumes its sampled tokens before scheduling the next step, so
    Sequence, KV-cache and scheduler state stay consistent.
    """

    def __init__(self, model: str, use_prefill_cudagraph: bool = True, **kwargs):
        self._exited = False
        config_fields = {field.name for field in fields(Config)}
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
        more decode; new prefill work is still allowed to dispatch so
        late arrivals keep making progress.
        """
        # Phase 1: fill the queue (never blocks).
        while len(self.batch_queue) < self.max_concurrent_batches:
            decode_batches = [
                batch_seqs for batch_seqs, _, _, _ in self.batch_queue
                if any(not seq.is_prefill or seq.is_speculative for seq in batch_seqs)
            ]
            if decode_batches and not self.scheduler.waiting:
                if self.scheduler.running and not self._coalesce_armed:
                    source_seqs = [
                        seq for batch_seqs in decode_batches for seq in batch_seqs
                        if not seq.is_prefill or seq.is_speculative
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
            if is_prefill and any(seq.is_speculative for seq in seqs):
                num_tokens = -sum(
                    seq.num_scheduled_tokens for seq in seqs if seq.is_speculative
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

        for seq, row in zip(seqs, token_ids):
            if (
                is_prefill
                and seq.num_cached_tokens + seq.num_scheduled_tokens
                < seq.num_tokens
            ):
                continue
            seq.draft_token = row[2] if row[2] != -1 else None

        self.scheduler.postprocess(
            seqs,
            [
                [token for token in row[:2] if token != -1]
                for row in token_ids
            ],
            is_prefill,
        )
        for seq in seqs:
            if seq.is_finished:
                self.model_runner.call("remove_request", seq.seq_id)
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs
            if seq.is_finished
        ]
        return outputs, num_tokens

    def is_finished(self) -> bool:
        return self.scheduler.is_finished() and not self.batch_queue

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
