import torch
from torch import nn
import torch.nn.functional as F

from .base import LinearMethodBase


class UnquantizedLinearMethod(LinearMethodBase):
    def create_weights(self, layer, input_size, output_size, params_dtype):
        layer.weight = nn.Parameter(torch.empty(
            output_size, input_size, dtype=params_dtype,
        ))
        layer.weight.weight_loader = layer.weight_loader

    def apply(self, layer, x, bias=None):
        # utils.loader imports this class while initializing the package.
        from nano_qwen.utils.context import get_context

        if get_context().gemm_reference_m == 1 and x.numel() // x.shape[-1] == 2:
            flat = x.reshape(2, x.shape[-1])
            return torch.cat([F.linear(flat[i:i+1], layer.weight, bias) for i in range(2)]).reshape(
                *x.shape[:-1], layer.weight.shape[0])
        return F.linear(x, layer.weight, bias)
