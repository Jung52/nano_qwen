import atexit
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class LLMEngine:
    """Thin control plane for scheduling and model execution.

    The engine intentionally does not prepare model inputs or own GPU state.
    Those responsibilities belong to ModelRunner.  Keeping the boundary small
    makes the execution path compatible with the ModelRunnerV2 direction:

        schedule -> execute_model -> sample_tokens -> postprocess

    Consumption is deferred by one step: ``step()`` only launches the current
    forward and returns a handle to its sampled tokens.  ``generate()`` holds
    the previous handle and consumes it *after* launching the next forward, so
    the CPU-side wait in ``get_output()`` overlaps with the in-flight forward
    (CPU processes step N-1 while GPU computes step N).
    """

    def __init__(self, model: str, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)
        Sequence.block_size = self.config.kvcache_block_size

        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for rank in range(1, self.config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(
                target=ModelRunner,
                args=(self.config, rank, event),
            )
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.model_runner = ModelRunner(self.config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            use_fast=True,
        )
        self.config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(self.config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for process in self.ps:
            process.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams,
    ) -> None:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        self.scheduler.add(Sequence(prompt, sampling_params))

    def step(self):
        """Schedule and launch one forward, returning (async_output, num_tokens).

        The returned handle must be consumed via ``consume_output()`` on the
        following iteration, so the D2H wait overlaps with the next forward.
        """
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = (
            sum(seq.num_scheduled_tokens for seq in seqs)
            if is_prefill
            else -len(seqs)
        )

        async_output = None
        if seqs:
            self.model_runner.call("execute_model", seqs, is_prefill)
            async_output = self.model_runner.call("sample_tokens")
        return async_output, num_tokens

    def consume_output(self, async_output) -> list[tuple[int, list[int]]]:
        """Block on a previously returned handle and run scheduler postprocess."""
        if async_output is None:
            return []

        token_ids = async_output.get_output()
        seqs = async_output.seqs
        is_prefill = async_output.is_prefill

        if token_ids is not None:
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
        return [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs
            if seq.is_finished
        ]

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

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

        prev_output = None
        while not self.is_finished():
            t = perf_counter()
            # Launch forward N first, then consume forward N-1 so the CPU wait
            # in get_output() overlaps with the GPU work of forward N.
            async_output, num_tokens = self.step()
            output = self.consume_output(prev_output)
            prev_output = async_output

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

        # Flush the last handle: there is no next forward to overlap with.
        output = self.consume_output(prev_output)
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
