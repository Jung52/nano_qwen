from abc import ABC, abstractmethod

import torch
from torch import nn


class LinearMethodBase(ABC):
    """Own parameter storage and local computation; the layer owns TP."""

    @abstractmethod
    def create_weights(
        self, layer: nn.Module, input_size: int, output_size: int,
        params_dtype: torch.dtype,
    ) -> None:
        raise NotImplementedError

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        pass

    @abstractmethod
    def apply(
        self, layer: nn.Module, x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class QuantizationConfig(ABC):
    @abstractmethod
    def get_quant_method(self, layer: nn.Module, prefix: str) -> LinearMethodBase:
        raise NotImplementedError
