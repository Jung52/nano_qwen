"""Single-node prototype HTTP control/data plane over LMCache's CPU L1.

This is not the official LMCache MP wire protocol, zero-copy SHM, or RDMA.
The simple binary transport makes the state-handoff contract testable first.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import urlsplit

from .lmcache_l1 import LMCacheL1Store, validate_key


def make_server(host, port, store, max_payload=512 << 20):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, body, binary=False):
            if not binary:
                body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/octet-stream" if binary else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-SHA256", hashlib.sha256(body).hexdigest())
            self.end_headers()
            self.wfile.write(body)

        def route(self):
            self.connection.settimeout(60)
            path = urlsplit(self.path).path
            if self.command == "GET" and path == "/health":
                return self.reply(200, store.status())
            if not path.startswith("/snapshots/"):
                return self.reply(404, {"error": "unknown endpoint"})
            key = validate_key(path[len("/snapshots/"):])
            if self.command == "PUT":
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= max_payload:
                    return self.reply(413, {"error": "invalid payload size"})
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise ValueError("truncated upload")
                sha = hashlib.sha256(payload).hexdigest()
                if self.headers.get("X-Content-SHA256") != sha:
                    raise ValueError("upload checksum mismatch")
                return self.reply(200, store.put(key, payload))
            if self.command == "GET":
                return self.reply(200, store.get(key), binary=True)
            if self.command == "DELETE":
                store.delete(key)
                return self.reply(200, {"deleted": key})
            return self.reply(405, {"error": "unsupported method"})

        def dispatch(self):
            try:
                self.route()
            except KeyError:
                self.reply(404, {"error": "snapshot not found or expired"})
            except FileExistsError as exc:
                self.reply(409, {"error": str(exc)})
            except MemoryError as exc:
                self.reply(507, {"error": str(exc)})
            except (ValueError, TypeError) as exc:
                self.reply(400, {"error": str(exc)})
            except Exception as exc:
                self.reply(500, {"error": str(exc)})

        do_PUT = do_GET = do_DELETE = dispatch

    return ThreadingHTTPServer((host, port), Handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--size-gb", type=float, default=1.0)
    p.add_argument("--ttl-seconds", type=float, default=900)
    p.add_argument("--max-payload-mb", type=int, default=512)
    args = p.parse_args()
    store = LMCacheL1Store(int(args.size_gb * (1 << 30)), args.ttl_seconds)
    server = make_server(args.host, args.port, store, args.max_payload_mb << 20)
    print(json.dumps(dict(address=server.server_address, **store.status())), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()


if __name__ == "__main__":
    main()
