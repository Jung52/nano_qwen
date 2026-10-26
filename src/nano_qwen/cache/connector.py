"""Inference-side client; no LMCache import or dependency on its torch ABI."""
from __future__ import annotations

import hashlib
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .lmcache_l1 import validate_key


class LMCacheConnector:
    def __init__(self, url="http://127.0.0.1:8765", timeout=120, max_payload=512 << 20):
        if not url.startswith(("http://", "https://")) or timeout <= 0:
            raise ValueError("invalid cache URL/timeout")
        self.url, self.timeout, self.max_payload = url.rstrip("/"), timeout, max_payload

    def _request(self, method, path, data=None):
        headers = {}
        if data is not None:
            if not 0 < len(data) <= self.max_payload:
                raise ValueError("invalid snapshot size")
            headers = {"Content-Type": "application/octet-stream",
                       "X-Content-SHA256": hashlib.sha256(data).hexdigest()}
        request = Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                size = int(response.headers.get("Content-Length", "0"))
                if not 0 <= size <= self.max_payload:
                    raise RuntimeError("oversized cache response")
                body = response.read(self.max_payload + 1)
                if len(body) != size or len(body) > self.max_payload:
                    raise RuntimeError("truncated/oversized cache response")
                if hashlib.sha256(body).hexdigest() != response.headers.get("X-Content-SHA256"):
                    raise RuntimeError("cache response checksum mismatch")
                return body
        except HTTPError as exc:
            message = exc.read(8192).decode(errors="replace")
            if exc.code == 404:
                raise KeyError(path) from exc
            if exc.code == 409:
                raise FileExistsError(message) from exc
            if exc.code == 507:
                raise MemoryError(message) from exc
            raise RuntimeError(f"cache HTTP {exc.code}: {message}") from exc

    def health(self):
        result = json.loads(self._request("GET", "/health"))
        if result.get("backend") != "lmcache_l1":
            raise RuntimeError("server is not using the LMCache L1 backend")
        return result

    def put(self, transfer_id, payload):
        result = json.loads(self._request("PUT", "/snapshots/" + validate_key(transfer_id), payload))
        if (result.get("sha256") != hashlib.sha256(payload).hexdigest()
                or result.get("committed") is not True or result.get("backend") != "lmcache_l1"):
            raise RuntimeError("cache did not acknowledge a committed LMCache write")
        return result

    def get(self, transfer_id):
        return self._request("GET", "/snapshots/" + validate_key(transfer_id))

    def delete(self, transfer_id):
        return json.loads(self._request("DELETE", "/snapshots/" + validate_key(transfer_id)))
