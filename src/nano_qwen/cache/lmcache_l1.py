"""Opaque request snapshots in LMCache 0.5.6's real CPU-DRAM L1 pool.

This uses the MP storage core, not the vLLM-specific MP connector. No dict or
file fallback is provided. Control metadata is small; all payload bytes live
in LMCache MemoryObjs. Pending handoffs are retained until ACK or TTL expiry.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import re
import threading
import time
import uuid


def validate_key(key: str) -> str:
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", key):
        raise ValueError("transfer_id must contain 1-128 ASCII letters/digits/_.-")
    return key


class LMCacheL1Store:
    backend_name = "lmcache_l1"

    def __init__(self, size_bytes: int, ttl_seconds: float = 900):
        if type(size_bytes) is not int or size_bytes < 4096 or ttl_seconds <= 0:
            raise ValueError("invalid L1 capacity/TTL")
        version = importlib.metadata.version("lmcache")
        if version != "0.5.6":
            raise RuntimeError(f"expected lmcache==0.5.6, found {version}")
        # Lazy imports isolate the LMCache/PyTorch ABI from inference workers.
        import torch
        from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
        from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
        from lmcache.v1.distributed.error import L1Error
        from lmcache.v1.distributed.l1_manager import L1Manager

        self.torch, self.Layout, self.Key, self.Error = torch, MemoryLayoutDesc, ObjectKey, L1Error
        config = L1ManagerConfig(L1MemoryManagerConfig(
            size_in_bytes=size_bytes, use_lazy=False, shm_name=""))
        self.manager = L1Manager(config)
        self.capacity, self.ttl = size_bytes, float(ttl_seconds)
        self._catalog = {}
        self._lock = threading.RLock()

    def _key(self, key):
        return self.Key(chunk_hash=hashlib.sha256(key.encode()).digest(),
                        model_name="nano_qwen_pd_v1", kv_rank=0)

    def _expire(self):
        now = time.monotonic()
        for key, info in list(self._catalog.items()):
            if now - info["touched"] > self.ttl:
                self._delete(key)

    def _delete(self, key):
        k = self._key(key)
        error = self.manager.delete([k])[k]
        if error not in (self.Error.SUCCESS, self.Error.KEY_NOT_EXIST):
            raise RuntimeError(f"LMCache delete failed: {error}")
        self._catalog.pop(key, None)

    def put(self, key: str, payload: bytes) -> dict:
        validate_key(key)
        if not payload:
            raise ValueError("empty snapshot")
        sha = hashlib.sha256(payload).hexdigest()
        with self._lock:
            self._expire()
            previous = self._catalog.get(key)
            if previous:
                if previous["sha256"] != sha:
                    raise FileExistsError("transfer_id already contains different data")
                previous["touched"] = time.monotonic()
                return self._receipt(key, previous)
            k, tag = self._key(key), uuid.uuid4().hex
            layout = self.Layout(shapes=[self.torch.Size([len(payload)])],
                                 dtypes=[self.torch.uint8])
            error, obj = self.manager.reserve_write([k], [False], layout, tag=tag)[k]
            if error == self.Error.OUT_OF_MEMORY:
                raise MemoryError("LMCache CPU RAM pool is full; ACK/delete pending handoffs")
            if error != self.Error.SUCCESS or obj is None:
                raise RuntimeError(f"LMCache reserve_write failed: {error}")
            try:
                target = obj.get_tensor(0)
                if target is None:
                    raise RuntimeError("LMCache returned no CPU tensor")
                target.reshape(-1).copy_(self.torch.frombuffer(bytearray(payload),
                                                              dtype=self.torch.uint8))
                error = self.manager.finish_write([k], tag=tag)[k]
                if error != self.Error.SUCCESS:
                    raise RuntimeError(f"LMCache finish_write failed: {error}")
            except BaseException:
                self.manager.finish_write_and_delete([k], tag=tag)
                raise
            info = dict(nbytes=len(payload), sha256=sha, touched=time.monotonic())
            self._catalog[key] = info
            return self._receipt(key, info)

    @staticmethod
    def _receipt(key, info):
        return dict(transfer_id=key, nbytes=info["nbytes"], sha256=info["sha256"],
                    backend="lmcache_l1", committed=True)

    def get(self, key: str) -> bytes:
        validate_key(key)
        with self._lock:
            self._expire()
            if key not in self._catalog:
                raise KeyError(key)
            k = self._key(key)
            error, obj = self.manager.reserve_read([k])[k]
            if error != self.Error.SUCCESS or obj is None:
                raise RuntimeError(f"LMCache reserve_read failed: {error}")
            try:
                tensor = obj.get_tensor(0)
                data = tensor.reshape(-1).numpy().tobytes()
            finally:
                error = self.manager.finish_read([k])[k]
                if error != self.Error.SUCCESS:
                    raise RuntimeError(f"LMCache finish_read failed: {error}")
            info = self._catalog[key]
            if len(data) != info["nbytes"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
                raise RuntimeError("LMCache payload checksum mismatch")
            info["touched"] = time.monotonic()
            return data

    def delete(self, key: str):
        validate_key(key)
        with self._lock:
            self._delete(key)

    def status(self):
        with self._lock:
            self._expire()
            used, total = self.manager.get_memory_usage()
            return dict(backend=self.backend_name, lmcache_version="0.5.6",
                        payload_objects=len(self._catalog), used_bytes=used,
                        capacity_bytes=total)

    def close(self):
        with self._lock:
            self.manager.close()
            self._catalog.clear()
