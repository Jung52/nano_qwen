"""Explicit benchmark policies; importable without the CUDA/model stack."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModeConfig:
    name: str
    enforce_eager: bool
    queue_depth: int
    async_output: bool
    prefill_cudagraph: bool
    async_scheduling: bool = False

    def __post_init__(self):
        if self.queue_depth not in (1, 2):
            raise ValueError("queue_depth must be 1 or 2")
        if self.async_scheduling and (self.queue_depth != 2 or not self.async_output):
            raise ValueError("async_scheduling requires queue_depth=2 and async_output=True")

    def apply(self, engine):
        engine.max_concurrent_batches = self.queue_depth
        engine.model_runner.async_output = self.async_output
        engine.model_runner.use_prefill_cudagraph = self.prefill_cudagraph
        if getattr(engine.config, "async_scheduling", None) != self.async_scheduling:
            raise RuntimeError("Actual async_scheduling differs from requested benchmark policy")


# v2: A/P1 and later explicitly enable lookahead. Historic queue-only results
# remain historic; do not combine their measurements with this protocol.
PROTOCOL = "explicit-async-v2"
MODES = [
    ModeConfig("BASELINE", True, 1, False, False, False),
    ModeConfig("A", True, 2, True, False, True),
    ModeConfig("B", False, 2, True, False, True),
    ModeConfig("C", False, 2, True, True, True),
]
PERF_MODES = [
    ModeConfig("P0", True, 1, False, False, False),
    ModeConfig("P1", True, 2, True, False, True),
    ModeConfig("P2", False, 2, True, False, True),
    ModeConfig("P3", False, 2, True, True, True),
]
MODE_NAMES = {mode.name: mode for mode in MODES}
PERF_MODE_NAMES = {mode.name: mode for mode in PERF_MODES}
