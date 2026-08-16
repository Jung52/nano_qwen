import torch


class AsyncModelOutput:
    """Handle for an asynchronous D2H token copy."""

    def __init__(
        self,
        token_ids_gpu: torch.Tensor,
        token_ids_cpu: torch.Tensor,
        ready_event: torch.cuda.Event,
    ) -> None:
        # 保留GPU tensor，避免异步D2H完成前显存被复用
        self._token_ids_gpu = token_ids_gpu
        self._token_ids_cpu = token_ids_cpu
        self._ready_event = ready_event
        self._result: list[int] | None = None

    def get_output(self) -> list[int]:
        if self._result is None:
            self._ready_event.synchronize()
            self._result = self._token_ids_cpu.tolist()
            self._token_ids_gpu = None
            self._token_ids_cpu = None
        return self._result