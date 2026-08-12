import torch


class AsyncModelOutput:
    """Handle for an in-flight sampled token batch.

    ``get_output()`` blocks only until the runner's device-to-host copy
    finishes.  The engine may hold this handle across a ``step()`` call and
    consume it after the next forward has been launched, overlapping the
    CPU-side wait with the in-flight forward.
    """

    def __init__(
        self,
        seqs: list,
        is_prefill: bool,
        token_ids_cpu: torch.Tensor,
        ready_event: torch.cuda.Event,
    ) -> None:
        self.seqs = seqs
        self.is_prefill = is_prefill
        self._token_ids_cpu = token_ids_cpu
        self._ready_event = ready_event
        self._result: list[int] | None = None

    def get_output(self) -> list[int]:
        if self._result is None:
            self._ready_event.synchronize()
            self._result = self._token_ids_cpu.tolist()
            self._token_ids_cpu = None
        return self._result
