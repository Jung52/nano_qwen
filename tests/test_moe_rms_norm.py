import pytest
import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm

from nano_qwen.layers.layernorm import Qwen3_5MoERMSNorm


@pytest.mark.parametrize("with_residual", [False, True])
def test_moe_rms_norm_matches_transformers(with_residual):
    generator = torch.Generator().manual_seed(26)
    x = torch.randn(4, 64, generator=generator).bfloat16()
    weight = torch.randn(64, generator=generator).bfloat16()
    norm = Qwen3_5MoERMSNorm(64).bfloat16()
    reference = Qwen3_5MoeRMSNorm(64).bfloat16()
    with torch.inference_mode():
        norm.weight.copy_(weight)
        reference.weight.copy_(weight)
        if with_residual:
            residual = torch.randn(4, 64, generator=generator).bfloat16()
            actual, summed = norm(x, residual)
            assert torch.equal(summed, x + residual)
            expected = reference(x + residual)
        else:
            actual = norm(x)
            expected = reference(x)
        assert torch.equal(actual, expected)
