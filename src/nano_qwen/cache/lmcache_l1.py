"""Transfer-id validation shared by the SHM cache modules."""
from __future__ import annotations

import re


def validate_key(key: str) -> str:
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", key):
        raise ValueError("transfer_id must contain 1-128 ASCII letters/digits/_.-")
    return key
