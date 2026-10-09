from .base import LinearMethodBase, QuantizationConfig
from .unquantized import UnquantizedLinearMethod


def get_quantization_config(name: str | None, checkpoint_config=None) -> QuantizationConfig | None:
    if name is None:
        return None
    if name == "fp8":
        if checkpoint_config is not None:
            from .fp8_block import Fp8BlockConfig

            return Fp8BlockConfig(checkpoint_config)
        from .fp8 import Fp8Config

        return Fp8Config()
    raise ValueError(f"Unsupported quantization method: {name!r}; expected 'fp8' or None")


__all__ = [
    "LinearMethodBase", "QuantizationConfig", "UnquantizedLinearMethod",
    "get_quantization_config",
]
