import atexit
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

from .model_runner import ModelRunner
from .scheduler import Scheduler


class LLMEngine:
    """Thin control plane for scheduling and model execution.

    The engine intentionally does not prepare model inputs or own GPU state.
    Those responsibilities belong to ModelRunner.  Keeping the boundary small
    makes the execution path compatible with the ModelRunnerV2 direction:

        schedule -> execute_model -> sample_tokens -> postprocess

    Each step consumes its sampled tokens before scheduling the next step, so
    Sequence, KV-cache and scheduler state stay consistent.
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
        """Run one complete schedule, execute, sample and postprocess step."""
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = (
            sum(seq.num_scheduled_tokens for seq in seqs)
            if is_prefill
            else -len(seqs)
        )

        self.model_runner.call("execute_model", seqs, is_prefill)
        async_output = self.model_runner.call("sample_tokens")
        assert async_output is not None
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