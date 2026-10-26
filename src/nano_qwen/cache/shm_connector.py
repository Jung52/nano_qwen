"""Inference client: map the real LMCache arena, exchange JSON descriptors only.

Does not import LMCache. mmap avoids multiprocessing's resource tracker, so a
prefill process exiting cannot unlink the service-owned shared-memory pool.
"""
import json
import mmap
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .lmcache_l1 import validate_key
from .shm_protocol import CONTROL_LIMIT, ITEM_SIZES, validate_layout


class TensorLease:
    def __init__(self, client, descriptor, writing):
        self.client, self.descriptor, self.writing = client, descriptor, writing
        self.key, self.token = descriptor["transfer_id"], descriptor["token"]
        self.metadata, self.tensors = descriptor["metadata"], {}
        self.closed = False
        import torch
        mapping = client._map_pool(descriptor["pool"])
        layout, total = validate_layout(descriptor["tensors"], descriptor["pool"]["size"])
        if total != descriptor["nbytes"]:
            raise ValueError("incorrect tensor byte count")
        ranges = []
        for raw, spec in zip(descriptor["tensors"], layout):
            offset = raw.get("offset")
            if (type(offset) is not int or offset < 0
                    or offset % ITEM_SIZES[spec["dtype"]]
                    or offset + spec["nbytes"] > len(mapping)):
                raise ValueError("invalid shared tensor offset")
            ranges.append((offset, offset + spec["nbytes"]))
            self.tensors[spec["name"]] = torch.frombuffer(mapping,
                dtype=getattr(torch, spec["dtype"]),
                count=spec["nbytes"] // ITEM_SIZES[spec["dtype"]], offset=offset).reshape(spec["shape"])
        if any(a[1] > b[0] for a, b in zip(sorted(ranges), sorted(ranges)[1:])):
            raise ValueError("overlapping shared tensor extents")
        client._leases[self.token] = self

    def drop_views(self):
        # Call only after all GPU work using these CPU buffers is synchronized.
        # Tensor aliases must not escape the lease lifetime.
        self.tensors.clear()


class LMCacheShmConnector:
    direct_tensors = True

    def __init__(self, url="http://127.0.0.1:8766", timeout=120):
        parts = urlsplit(url)
        if (parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost")
                or parts.username or parts.password or parts.path not in ("", "/") or timeout <= 0):
            raise ValueError("SHM connector requires an HTTP loopback URL")
        self.url, self.timeout = url.rstrip("/"), timeout
        self.opener = build_opener(ProxyHandler({}))
        self._mapping, self._pool = None, None
        self._leases = {}
        self.control_sent_bytes = self.control_received_bytes = 0

    def _request(self, path, args=None):
        payload = None if args is None else json.dumps(args, separators=(",", ":"), allow_nan=False).encode()
        if payload is not None and len(payload) > CONTROL_LIMIT:
            raise ValueError("control metadata exceeds limit")
        request = Request(self.url + path, data=payload,
                          headers={"Content-Type": "application/json"},
                          method="GET" if args is None else "POST")
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                size = int(response.headers.get("Content-Length", "0"))
                if not 0 < size <= CONTROL_LIMIT:
                    raise RuntimeError("invalid control response length")
                body = response.read(CONTROL_LIMIT + 1)
                if len(body) != size:
                    raise RuntimeError("truncated control response")
        except HTTPError as exc:
            error = exc.read(CONTROL_LIMIT).decode(errors="replace")
            kinds = {404: KeyError, 409: FileExistsError, 507: MemoryError, 403: PermissionError, 400: ValueError}
            raise kinds.get(exc.code, RuntimeError)(error) from exc
        self.control_sent_bytes += len(payload or b"")
        self.control_received_bytes += len(body)
        return json.loads(body)

    def health(self):
        result = self._request("/health")
        if result.get("backend") != "lmcache_l1_shm":
            raise RuntimeError("expected the LMCache shared-tensor service on port 8766")
        return result

    def _map_pool(self, pool):
        if self._pool is not None:
            if pool != self._pool:
                raise RuntimeError("cache service restarted / shared arena changed; create a new connector")
            return self._mapping
        name, size = pool.get("name"), pool.get("size")
        if (not isinstance(name, str) or not name.startswith("lmcache_l1_pool_nanoqwen_")
                or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name)
                or type(size) is not int or size < 4096):
            raise ValueError("invalid L1 pool descriptor")
        path = Path("/dev/shm") / name
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if (info.st_size != size or info.st_dev != pool["device"] or info.st_ino != pool["inode"]):
                raise RuntimeError("L1 pool identity/size mismatch; processes must share the IPC namespace")
            self._mapping = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
        finally:
            os.close(fd)
        self._pool = dict(pool)
        return self._mapping

    def allocate(self, key, metadata, layout):
        validate_key(key)
        descriptor = self._request("/allocate", dict(key=key, metadata=metadata, layout=layout, pid=os.getpid()))
        try:
            return TensorLease(self, descriptor, writing=True)
        except BaseException:
            self._request("/abort", dict(key=key, token=descriptor["token"]))
            raise

    def commit(self, lease):
        if lease.client is not self or not lease.writing:
            raise ValueError("not this connector's writer lease")
        lease.drop_views()
        receipt = self._request("/commit", dict(key=lease.key, token=lease.token))
        if receipt.get("committed") is not True:
            raise RuntimeError("cache did not commit the tensor state")
        lease.closed = True
        self._leases.pop(lease.token, None)
        return receipt

    def abort(self, lease):
        lease.drop_views()
        receipt = self._request("/abort", dict(key=lease.key, token=lease.token))
        lease.closed = True
        self._leases.pop(lease.token, None)
        return receipt

    def acquire(self, key):
        validate_key(key)
        descriptor = self._request("/acquire", dict(key=key, pid=os.getpid()))
        try:
            return TensorLease(self, descriptor, writing=False)
        except BaseException:
            self._request("/release", dict(key=key, token=descriptor["token"], delete=False))
            raise

    def release(self, lease, delete=False):
        if lease.client is not self or lease.writing:
            raise ValueError("not this connector's reader lease")
        lease.drop_views()
        receipt = self._request("/release", dict(key=lease.key, token=lease.token, delete=delete))
        lease.closed = True
        self._leases.pop(lease.token, None)
        return receipt

    def delete(self, key):
        return self._request("/delete", dict(key=validate_key(key)))

    def close(self):
        if self._leases:
            raise RuntimeError("active tensor leases remain; synchronize then commit/abort/release first")
        if self._mapping is not None:
            self._mapping.close()
            self._mapping, self._pool = None, None
