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
        return F.linear(x, layer.weight, bias)
