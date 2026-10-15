# ruff: noqa: F811
import torch

from nano_qwen.layers.sampler import Sampler
from test_mtp_cache_lifecycle import runner, sequence  # noqa: F401


def test_runner_routes_greedy_from_cpu_requests_without_advancing_rng(runner):
    seq = sequence(129)
    seq.temperature = 1e-6
    runner.input_batch.update([seq])
    runner.sampler = Sampler()
    logits = torch.zeros(1, 16)
    logits[0, 9] = 3
    runner._target_hidden = torch.zeros(1, 4)
    runner._pending = (logits, torch.tensor([1e-6]), [seq], False)
    rng = torch.random.get_rng_state().clone()
    with torch.compiler.set_stance("force_eager"):
        output = runner.sample_tokens().get_output()
    assert output[0][0] == 9
    assert torch.equal(rng, torch.random.get_rng_state())
