import torch

from .execution import OutputSpec, RequestOutput


class AsyncModelOutput:
    """Handle for an asynchronous D2H token copy."""

    def __init__(
        self,
        token_ids_gpu: torch.Tensor,
        token_ids_cpu: torch.Tensor,
        ready_event: torch.cuda.Event,
        output_specs: list[OutputSpec] | None = None,
    ) -> None:
        # 保留GPU tensor，避免异步D2H完成前显存被复用
        self._token_ids_gpu = token_ids_gpu
        self._token_ids_cpu = token_ids_cpu
        self._ready_event = ready_event
        self._result: list[list[int]] | None = None
        self._output_specs = output_specs
        self._commits: list[RequestOutput] | None = None

    def get_output(self) -> list[list[int]]:
        if self._result is None:
            self._ready_event.synchronize()
            self._result = self._token_ids_cpu.tolist()
            self._token_ids_gpu = None
            self._token_ids_cpu = None
        return self._result

    def get_results(self) -> list[RequestOutput]:
        """Consume the private token wire once and expose named commit results."""
        if self._commits is None:
            rows = self.get_output()
            if self._output_specs is None or len(rows) != len(self._output_specs):
                raise RuntimeError("Missing explicit execution output metadata")
            commits = []
            for row, spec in zip(rows, self._output_specs):
                tokens = tuple(row[:spec.num_committed])
                if len(row) != 3 or spec.num_committed not in (0, 1, 2) or any(t < 0 for t in tokens):
                    raise RuntimeError("Invalid execution token payload")
                commits.append(RequestOutput(tokens, spec.accepted_count,
                                             None if row[2] == -1 else row[2], spec.cache_advance))
            self._commits = commits
        return self._commits
