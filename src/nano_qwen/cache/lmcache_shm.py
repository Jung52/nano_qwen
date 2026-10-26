"""Real LMCache L1 named arena with exclusive, process-owned tensor leases.

Only descriptors live in this catalog. Allocation/commit/read/free are handled
by LMCache. pidfds prevent a live worker's mapped memory being recycled by TTL.
"""
from collections import OrderedDict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import select
import threading
import time
import uuid

from .lmcache_l1 import validate_key
from .shm_protocol import CONTROL_LIMIT, validate_layout


class LMCacheShmStore:
    def __init__(self, size_bytes, ttl_seconds=900):
        if importlib.metadata.version("lmcache") != "0.5.6":
            raise RuntimeError("this adapter requires lmcache==0.5.6")
        if not hasattr(os, "pidfd_open") or not Path("/dev/shm").is_dir():
            raise RuntimeError("Linux pidfd and POSIX /dev/shm are required")
        if type(size_bytes) is not int or size_bytes < 4096 or ttl_seconds <= 0:
            raise ValueError("invalid L1 capacity/TTL")
        free = os.statvfs("/dev/shm")
        if size_bytes > free.f_bavail * free.f_frsize:
            raise MemoryError("/dev/shm has insufficient space; increase container --shm-size or reduce --size-gb")
        import torch
        from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
        from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
        from lmcache.v1.distributed.error import L1Error
        from lmcache.v1.distributed.l1_manager import L1Manager
        self.torch, self.Layout, self.Key, self.Error = torch, MemoryLayoutDesc, ObjectKey, L1Error
        self.session = uuid.uuid4().hex
        self.shm_name = "lmcache_l1_pool_nanoqwen_" + self.session
        config = L1ManagerConfig(L1MemoryManagerConfig(size_bytes, False, shm_name=self.shm_name),
                                 write_ttl_seconds=(1 << 31) - 1, read_ttl_seconds=(1 << 31) - 1)
        self.manager = L1Manager(config)
        self.arena = self.manager.get_l1_memory_desc()
        info = (Path("/dev/shm") / self.shm_name).stat()
        self.pool = dict(name=self.shm_name, size=size_bytes, device=info.st_dev,
                         inode=info.st_ino, session=self.session)
        self.ttl, self.entries = float(ttl_seconds), {}
        self.completed = OrderedDict()
        self.lock = threading.RLock()

    def _key(self, key):
        return self.Key(chunk_hash=hashlib.sha256(key.encode()).digest(),
                        model_name="nano_qwen_pd_tensor_v3", kv_rank=0)

    def _success(self, error, operation):
        if error != self.Error.SUCCESS:
            raise RuntimeError(f"LMCache {operation}: {error}")

    def _owner(self, pid):
        if type(pid) is not int or pid < 1:
            raise ValueError("invalid owner PID")
        try:
            fd = os.pidfd_open(pid, 0)
        except ProcessLookupError as exc:
            raise ValueError("owner process is not alive / not in the same PID namespace") from exc
        if select.select([fd], [], [], 0)[0]:
            os.close(fd)
            raise ValueError("owner process has exited")
        return fd

    def _drop_owner(self, entry):
        fd = entry.pop("pidfd", None)
        if fd is not None:
            os.close(fd)

    def _delete(self, key):
        entry = self.entries[key]
        self._success(self.manager.delete([entry["key"]])[entry["key"]], "delete")
        self._drop_owner(entry)
        del self.entries[key]

    def _reap(self):
        now = time.monotonic()
        for key, entry in list(self.entries.items()):
            state = entry["state"]
            if state in ("writing", "reading"):
                if not select.select([entry["pidfd"]], [], [], 0)[0]:
                    continue  # An elapsed TTL never invalidates a live mapping.
                k = entry["key"]
                if state == "writing":
                    self._success(self.manager.finish_write_and_delete([k], tag=entry["write_token"])[k], "abort dead writer")
                    self._drop_owner(entry)
                    del self.entries[key]
                else:
                    self._success(self.manager.finish_read([k])[k], "release dead reader")
                    self._drop_owner(entry)
                    entry.update(state="ready", touched=now)
                    entry.pop("read_token", None)
            elif now - entry["touched"] > self.ttl:
                self._delete(key)

    def _descriptor(self, key, entry, token):
        return dict(transfer_id=key, token=token, pool=dict(self.pool),
                    metadata=entry["metadata"], tensors=entry["tensors"], nbytes=entry["nbytes"],
                    backend="lmcache_l1_shm", payload_transport="shared_memory")

    def allocate(self, key, metadata, layout, pid):
        validate_key(key)
        if not isinstance(metadata, dict) or len(json.dumps(metadata, allow_nan=False).encode()) > CONTROL_LIMIT // 2:
            raise ValueError("invalid metadata")
        layout, nbytes = validate_layout(layout, self.pool["size"])
        with self.lock:
            self._reap()
            if key in self.entries:
                raise FileExistsError("transfer ID already exists")
            owner = self._owner(pid)
            token, k = uuid.uuid4().hex, self._key(key)
            desc = self.Layout(shapes=[self.torch.Size(x["shape"]) for x in layout],
                               dtypes=[getattr(self.torch, x["dtype"]) for x in layout])
            error, obj = self.manager.reserve_write([k], [False], desc, tag=token)[k]
            if error != self.Error.SUCCESS or obj is None:
                os.close(owner)
                if error == self.Error.OUT_OF_MEMORY:
                    raise MemoryError("LMCache shared L1 is full")
                raise RuntimeError(f"LMCache reserve_write: {error}")
            try:
                tensors = []
                for i, spec in enumerate(layout):
                    tensor = obj.get_tensor(i)
                    offset = tensor.data_ptr() - self.arena.ptr
                    if tensor.device.type != "cpu" or not 0 <= offset <= self.pool["size"] - spec["nbytes"]:
                        raise RuntimeError("LMCache tensor is not backed by the named L1 arena")
                    tensors.append(dict(spec, offset=offset))
                entry = dict(key=k, state="writing", write_token=token, pidfd=owner,
                             metadata=metadata, tensors=tensors, nbytes=nbytes, touched=time.monotonic())
                self.entries[key] = entry
                return self._descriptor(key, entry, token)
            except BaseException:
                self.manager.finish_write_and_delete([k], tag=token)
                os.close(owner)
                raise

    def commit(self, key, token):
        with self.lock:
            self._reap()
            entry = self.entries[key]
            if token != entry["write_token"]:
                raise PermissionError("incorrect writer lease")
            if entry["state"] == "writing":
                k = entry["key"]
                self._success(self.manager.finish_write([k], tag=token)[k], "commit")
                self._drop_owner(entry)
                entry.update(state="ready", touched=time.monotonic())
            return dict(transfer_id=key, nbytes=entry["nbytes"], committed=True,
                        backend="lmcache_l1_shm", payload_transport="shared_memory")

    def abort(self, key, token):
        with self.lock:
            self._reap()
            if key not in self.entries:
                return dict(aborted=True)
            entry = self.entries[key]
            if entry["state"] != "writing" or entry["write_token"] != token:
                raise PermissionError("not an active writer lease")
            k = entry["key"]
            self._success(self.manager.finish_write_and_delete([k], tag=token)[k], "abort")
            self._drop_owner(entry)
            del self.entries[key]
            return dict(aborted=True)

    def acquire(self, key, pid):
        with self.lock:
            self._reap()
            entry = self.entries[key]
            if entry["state"] != "ready":
                raise FileExistsError("snapshot is writing or already leased by a reader")
            owner = self._owner(pid)
            k = entry["key"]
            error, obj = self.manager.reserve_read([k])[k]
            if error != self.Error.SUCCESS or obj is None:
                os.close(owner)
                raise RuntimeError(f"LMCache reserve_read: {error}")
            token = uuid.uuid4().hex
            entry.update(state="reading", read_token=token, pidfd=owner)
            return self._descriptor(key, entry, token)

    def release(self, key, token, delete=False):
        if type(delete) is not bool:
            raise ValueError("delete must be boolean")
        with self.lock:
            self._reap()
            done_key = (key, token)
            if done_key in self.completed:
                return self.completed[done_key]
            entry = self.entries[key]
            if entry["state"] != "reading" or entry.get("read_token") != token:
                raise PermissionError("incorrect reader lease")
            k = entry["key"]
            self._success(self.manager.finish_read([k])[k], "finish read")
            self._drop_owner(entry)
            entry.update(state="ready", touched=time.monotonic())
            entry.pop("read_token", None)
            if delete:
                self._delete(key)
            receipt = dict(released=True, deleted=delete, transfer_id=key)
            self.completed[done_key] = receipt
            while len(self.completed) > 1024:
                self.completed.popitem(last=False)
            return receipt

    def delete(self, key):
        with self.lock:
            self._reap()
            if key not in self.entries:
                return dict(deleted=True)
            if self.entries[key]["state"] != "ready":
                raise FileExistsError("cannot delete a live writer/reader mapping")
            self._delete(key)
            return dict(deleted=True)

    def status(self):
        with self.lock:
            self._reap()
            used, total = self.manager.get_memory_usage()
            return dict(backend="lmcache_l1_shm", lmcache_version="0.5.6", pool=dict(self.pool),
                        used_bytes=used, capacity_bytes=total, payload_objects=len(self.entries),
                        states={s: sum(e["state"] == s for e in self.entries.values())
                                for s in ("writing", "ready", "reading")},
                        payload_http_bytes=0, payload_pack_calls=0, payload_unpack_calls=0)

    def close(self):
        with self.lock:
            self.manager.close()
            for entry in self.entries.values():
                self._drop_owner(entry)
            self.entries.clear()
