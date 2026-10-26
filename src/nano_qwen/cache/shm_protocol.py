"""Small control descriptors only. Tensor payloads never enter this protocol."""
import math
import re

CONTROL_LIMIT = 1 << 20
ITEM_SIZES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8,
              "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1, "bool": 1}


def validate_layout(layout, maximum):
    if not isinstance(layout, list) or not 1 <= len(layout) <= 512:
        raise ValueError("invalid tensor layout")
    names, total = set(), 0
    result = []
    for item in layout:
        if not isinstance(item, dict):
            raise ValueError("invalid tensor layout entry")
        name, shape, dtype = item.get("name"), item.get("shape"), item.get("dtype")
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name)
                or name in names or dtype not in ITEM_SIZES
                or not isinstance(shape, list) or not 1 <= len(shape) <= 8
                or any(type(d) is not int or d < 1 for d in shape)):
            raise ValueError("invalid tensor name/shape/dtype")
        size = math.prod(shape) * ITEM_SIZES[dtype]
        if total % ITEM_SIZES[dtype] or size > maximum - total:
            raise ValueError("unaligned tensor group or layout exceeds capacity")
        names.add(name)
        total += size
        result.append(dict(name=name, shape=list(shape), dtype=dtype, nbytes=size))
    return result, total
