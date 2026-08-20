# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
# -*- coding: utf-8 -*-
# Minimal inference-only subset: autotune-cache plumbing, platform probing,
# shared-memory checks, and the tensor-cache decorator.  All training-only
# helpers (input_guard, autocast wrappers, assert_close, ...) are removed.

import functools
import inspect
import os
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import torch
import triton

FLA_CACHE_RESULTS = os.getenv("FLA_CACHE_RESULTS", "1") == "1"

SUPPORTS_AUTOTUNE_CACHE = (
    "cache_results" in inspect.signature(triton.autotune).parameters
)

autotune_cache_kwargs = (
    {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}
)


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Caches the most recent results of a function with tensor inputs."""

    cache_entries: Tuple[Optional[Tuple], Optional[Dict], Any] = []
    cache_size = 4

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries, cache_size
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                ):
                    cache_entries = (
                        cache_entries[:i]
                        + cache_entries[i + 1 :]
                        + [(args, kwargs, last_result)]
                    )
                    return last_result

        result = fn(*args, **kwargs)

        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


def _cpu_device_warning():
    import warnings

    warnings.warn(
        ("Triton is not supported on current platform, roll back to CPU."), stacklevel=1
    )


@lru_cache(maxsize=None)
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        _cpu_device_warning()
        return "cpu"


@lru_cache(maxsize=None)
def _check_platform() -> Literal["nvidia", "amd", "intel", "musa"]:
    device = get_available_device()
    if device == "cuda":
        return "nvidia"
    elif device == "hip":
        return "amd"
    elif device == "xpu":
        return "intel"
    else:
        return device


# For AMD GPUs the triton backend is 'hip' while the torch backend is 'cuda'.
device = get_available_device() if get_available_device() != "hip" else "cuda"
device_torch_lib = getattr(torch, device)
device_platform = _check_platform()

is_nvidia = device_platform == "nvidia"
is_nvidia_hopper = is_nvidia and (
    "NVIDIA H" in torch.cuda.get_device_name(0)
    or torch.cuda.get_device_capability()[0] >= 9
)

# Nvidia Ampere or newer.
is_tf32_supported = is_nvidia and torch.cuda.get_device_capability(0)[0] >= 8
is_gather_supported = hasattr(triton.language, "gather")


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)[
                "max_shared_mem"
            ]
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        _cpu_device_warning()
        return [-1]


class Backend(Enum):
    ADA = 101376  # RTX 4090
    AMPERE = 166912  # A100
    HOPPER = 232448  # H100
    DEFAULT = 102400  # Default

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@lru_cache(maxsize=None)
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False
