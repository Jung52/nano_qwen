import torch


class AsyncModelOutput:
    """Single-slot asynchronous D2H output."""

    def __init__(
        self,
        token_ids_gpu: torch.Tensor,
        token_ids_cpu: torch.Tensor,
        ready_event: torch.cuda.Event,
    ) -> None:
        self._token_ids_gpu = token_ids_gpu
        self._token_ids_cpu = token_ids_cpu
        self._ready_event = ready_event
        self._result: list[int] | None = None

    def get_output(self) -> list[int]:
        if self._result is None:
            self._ready_event.synchronize()
            self._result = self._token_ids_cpu.tolist()
            self._token_ids_gpu = None
        return self._result

