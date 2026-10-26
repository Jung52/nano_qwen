"""Portable, pickle-free KV/GDN snapshots at a committed decode boundary."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import struct

import torch

MAGIC = b"NQPD0001"
MAX_HEADER = 4 << 20
DTYPES = {str(d).removeprefix("torch."): d for d in (
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
    torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool,
)}


@dataclass
class Snapshot:
    metadata: dict
    tensors: dict[str, torch.Tensor]


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid {name}")
    return value


def pack(snapshot: Snapshot) -> bytes:
    entries, pieces, offset = [], [], 0
    for name, value in sorted(snapshot.tensors.items()):
        t = value.detach().to("cpu").contiguous()
        dtype = str(t.dtype).removeprefix("torch.")
        if dtype not in DTYPES:
            raise ValueError(f"unsupported tensor dtype {dtype}")
        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
        entries.append(dict(name=name, shape=list(t.shape), dtype=dtype,
                            offset=offset, nbytes=len(raw)))
        pieces.append(raw)
        offset += len(raw)
    header = json.dumps(dict(metadata=snapshot.metadata, tensors=entries),
                        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(header) > MAX_HEADER:
        raise ValueError("snapshot header too large")
    body = struct.pack("<8sQ", MAGIC, len(header)) + header + b"".join(pieces)
    return body + hashlib.sha256(body).digest()


def unpack(payload: bytes) -> Snapshot:
    if len(payload) < 48:
        raise ValueError("truncated snapshot")
    body, digest = payload[:-32], payload[-32:]
    if hashlib.sha256(body).digest() != digest:
        raise ValueError("snapshot checksum mismatch")
    magic, length = struct.unpack("<8sQ", body[:16])
    if magic != MAGIC or length > MAX_HEADER or 16 + length > len(body):
        raise ValueError("invalid snapshot header")
    header = json.loads(body[16:16 + length])
    if not isinstance(header, dict) or not isinstance(header.get("metadata"), dict):
        raise ValueError("invalid snapshot metadata")
    if not isinstance(header.get("tensors"), list):
        raise ValueError("invalid tensor table")
    raw, tensors, expected_offset = body[16 + length:], {}, 0
    for item in header["tensors"]:
        name, shape, dtype = item["name"], item["shape"], item["dtype"]
        if not isinstance(name, str) or not name or name in tensors:
            raise ValueError("duplicate/invalid tensor name")
        if dtype not in DTYPES or not isinstance(shape, list) or len(shape) > 8:
            raise ValueError("invalid tensor shape/dtype")
        for dim in shape:
            _integer(dim, "shape dimension", 1)
        nbytes = math.prod(shape) * DTYPES[dtype].itemsize
        if (item["offset"] != expected_offset or item["nbytes"] != nbytes
                or nbytes > len(raw) - expected_offset):
            raise ValueError("invalid tensor extent")
        data = bytearray(raw[expected_offset:expected_offset + nbytes])
        tensors[name] = torch.frombuffer(data, dtype=DTYPES[dtype]).reshape(shape).clone()
        expected_offset += nbytes
    if expected_offset != len(raw):
        raise ValueError("unreferenced snapshot bytes")
    return Snapshot(header["metadata"], tensors)


def model_identity(runner) -> str:
    """Hash checkpoint contents once; paths/mtimes are not model identities."""
    cached = getattr(runner, "_pd_model_identity", None)
    if cached is not None:
        return cached
    root = Path(runner.config.model)
    files = sorted([*root.glob("*.safetensors"), *root.glob("pytorch_model*.bin")])
    if not files:
        raise ValueError("PD identity requires checkpoint weight files")
    h = hashlib.sha256()
    for path in files:
        h.update(path.name.encode())
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
    cfg = runner.config.hf_config.to_dict().copy()
    for key in ("_name_or_path", "transformers_version"):
        cfg.pop(key, None)
    versions = {"torch": str(torch.__version__)}
    for name in ("flashinfer-python", "triton"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    h.update(json.dumps(dict(config=cfg, quantization=runner.config.quantization,
                             versions=versions), sort_keys=True, default=str).encode())
    runner._pd_model_identity = h.hexdigest()
    return runner._pd_model_identity


def capture(runner, seq, identity: str) -> Snapshot:
    if runner.world_size != 1 or getattr(runner, "_pending", None) is not None:
        raise ValueError("snapshot requires a completed single-GPU sampling step")
    if (seq.is_finished or seq.is_prefill or seq.num_scheduled_tokens != 0
            or seq.num_cached_tokens != len(seq.token_ids) - 1
            or seq.num_cached_tokens < 1):
        raise ValueError("request is not at a committed decode boundary")
    n = seq.num_cached_tokens
    block_size = runner.block_size
    count = (n + block_size - 1) // block_size
    if len(seq.block_table) != count:
        raise ValueError("unexpected source KV block count")
    slot = runner.input_batch.seq_id_to_slot[seq.seq_id]
    if runner.kv_cache.device.type == "cuda":
        torch.cuda.synchronize(runner.kv_cache.device)
    ids = torch.tensor(seq.block_table, dtype=torch.int64, device=runner.kv_cache.device)
    kv = runner.kv_cache.index_select(2, ids).detach().cpu().clone()
    # Never publish uninitialized tokens in the tail of an allocated KV page.
    if n % block_size:
        kv[:, :, -1, n % block_size:].zero_()
    tensors = {"kv": kv}
    for i, layer in enumerate(runner.gdn_layers):
        tensors[f"gdn.{i}.conv"] = layer.conv_states[slot].detach().cpu().clone()
        tensors[f"gdn.{i}.recurrent"] = layer.recurrent_states[slot].detach().cpu().clone()
    metadata = dict(schema=1, model_id=identity, block_size=block_size,
                    tokens=list(seq.token_ids), prompt_len=seq.num_prompt_tokens,
                    cached_tokens=n, temperature=seq.temperature,
                    max_tokens=seq.max_tokens, ignore_eos=seq.ignore_eos,
                    gdn_layers=len(runner.gdn_layers))
    return Snapshot(metadata, tensors)


def validate(runner, snapshot: Snapshot, identity: str) -> None:
    m, tensors = snapshot.metadata, snapshot.tensors
    if m.get("schema") != 1 or m.get("model_id") != identity:
        raise ValueError("snapshot schema/model identity mismatch")
    if m.get("block_size") != runner.block_size or runner.world_size != 1:
        raise ValueError("snapshot block size/parallelism mismatch")
    tokens = m.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("invalid token history")
    for t in tokens:
        _integer(t, "token")
    n = _integer(m.get("cached_tokens"), "cached_tokens", 1)
    prompt = _integer(m.get("prompt_len"), "prompt_len", 1)
    maximum = _integer(m.get("max_tokens"), "max_tokens", 2)
    if not (prompt <= n == len(tokens) - 1 and len(tokens) - prompt < maximum):
        raise ValueError("invalid cached/completion token progress")
    if prompt + maximum > runner.config.max_model_len:
        raise ValueError("snapshot exceeds destination max_model_len")
    temp = m.get("temperature")
    if (isinstance(temp, bool) or not isinstance(temp, (float, int))
            or not math.isfinite(temp) or temp <= 1e-10
            or type(m.get("ignore_eos")) is not bool):
        raise ValueError("invalid sampling metadata")
    if m.get("gdn_layers") != len(runner.gdn_layers):
        raise ValueError("GDN layer count mismatch")
    expected = {"kv"}
    for i in range(len(runner.gdn_layers)):
        expected.update((f"gdn.{i}.conv", f"gdn.{i}.recurrent"))
    if set(tensors) != expected:
        raise ValueError("missing or unexpected KV/GDN tensors")
    count = (n + runner.block_size - 1) // runner.block_size
    shape = list(runner.kv_cache.shape)
    shape[2] = count
    checks = [(tensors["kv"], shape, runner.kv_cache.dtype)]
    for i, layer in enumerate(runner.gdn_layers):
        checks.extend((tensors[f"gdn.{i}.{name}"], list(pool.shape[1:]), pool.dtype)
                      for name, pool in (("conv", layer.conv_states),
                                         ("recurrent", layer.recurrent_states)))
    for tensor, shape, dtype in checks:
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or list(tensor.shape) != shape or tensor.dtype != dtype):
            raise ValueError("KV/GDN tensor shape/dtype/device mismatch")


def restore(runner, seq, snapshot: Snapshot, identity: str) -> int:
    """Blocks are allocated by engine_ops; slot assignment happens only here."""
    validate(runner, snapshot, identity)
    m = snapshot.metadata
    if (seq.token_ids != m["tokens"] or seq.num_cached_tokens != m["cached_tokens"]
            or seq.num_prompt_tokens != m["prompt_len"] or seq.is_prefill):
        raise ValueError("destination request progress does not match snapshot")
    if getattr(runner, "_pending", None) is not None:
        raise ValueError("destination has pending sampling work")
    if seq.seq_id in runner.input_batch.seq_id_to_slot:
        raise ValueError("destination request already owns a slot")
    if len(seq.block_table) != snapshot.tensors["kv"].shape[2]:
        raise ValueError("destination KV allocation does not match snapshot")
    if not any(item is None for item in runner.input_batch.seqs):
        raise ValueError("no free destination state slot")
    _, entries = runner.input_batch.update([seq])
    slot = entries[0][1]
    try:
        ids = torch.tensor(seq.block_table, dtype=torch.int64, device=runner.kv_cache.device)
        runner.kv_cache.index_copy_(2, ids, snapshot.tensors["kv"].to(runner.kv_cache.device))
        for i, layer in enumerate(runner.gdn_layers):
            layer.conv_states[slot].copy_(snapshot.tensors[f"gdn.{i}.conv"])
            layer.recurrent_states[slot].copy_(snapshot.tensors[f"gdn.{i}.recurrent"])
        runner.sampled_token_ids_gpu[slot] = seq.last_token
        if runner.kv_cache.device.type == "cuda":
            torch.cuda.synchronize(runner.kv_cache.device)
    except BaseException:
        runner.input_batch.remove(seq.seq_id)
        raise
    return slot
