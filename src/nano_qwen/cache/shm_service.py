"""Loopback-only JSON control service; all tensors stay in LMCache L1 SHM."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json

from .lmcache_shm import LMCacheShmStore
from .shm_protocol import CONTROL_LIMIT


def make_server(host, port, store):
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("SHM service requires loopback and a shared IPC/PID namespace")
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, value):
            body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
            if len(body) > CONTROL_LIMIT:
                raise ValueError("control response too large")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch()

        def dispatch(self):
            try:
                self.connection.settimeout(30)
                if self.command == "GET" and self.path == "/health":
                    return self.reply(200, store.status())
                if self.command != "POST":
                    return self.reply(404, dict(error="unknown control operation"))
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= CONTROL_LIMIT:
                    return self.reply(413, dict(error="control message exceeds limit"))
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("truncated control message")
                args = json.loads(body)
                if not isinstance(args, dict):
                    raise ValueError("expected a JSON object")
                methods = {"/allocate": store.allocate, "/commit": store.commit,
                           "/abort": store.abort, "/acquire": store.acquire,
                           "/release": store.release, "/delete": store.delete}
                if self.path not in methods:
                    return self.reply(404, dict(error="unknown control operation"))
                self.reply(200, methods[self.path](**args))
            except KeyError as exc:
                self.reply(404, dict(error=str(exc)))
            except FileExistsError as exc:
                self.reply(409, dict(error=str(exc)))
            except MemoryError as exc:
                self.reply(507, dict(error=str(exc)))
            except PermissionError as exc:
                self.reply(403, dict(error=str(exc)))
            except (ValueError, TypeError) as exc:
                self.reply(400, dict(error=str(exc)))
            except Exception as exc:
                self.reply(500, dict(error=str(exc)))
    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--size-gb", type=float, default=1)
    parser.add_argument("--ttl-seconds", type=float, default=900)
    args = parser.parse_args()
    store = LMCacheShmStore(int(args.size_gb * (1 << 30)), args.ttl_seconds)
    server = make_server(args.host, args.port, store)
    print(json.dumps(store.status()), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()


if __name__ == "__main__":
    main()
