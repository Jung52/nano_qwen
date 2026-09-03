"""Minimal Chrome-trace (Perfetto-compatible) event recorder.

Records CPU-side durations with ``time.perf_counter_ns()`` and writes the
Chrome JSON format that ui.perfetto.dev opens directly (Open trace file).

The collector is a per-process global.  The engine is single-process at
TP=1; multi-rank setups would each hold their own collector, so only the
process calling :func:`save_trace` writes a file.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

_COLLECTOR: TraceCollector | None = None


class TraceCollector:
    """Collects complete (ph=X) and instant (ph=i) events in microseconds."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._pid = os.getpid()

    def record(
        self,
        name: str,
        cat: str,
        ts_ns: int,
        dur_ns: int,
        tid: int = 0,
        args: dict[str, Any] | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "name": name,
            "cat": cat,
            "ph": "X",
            "ts": ts_ns // 1000,
            "dur": dur_ns // 1000,
            "pid": self._pid,
            "tid": tid,
        }
        if args:
            event["args"] = args
        self._events.append(event)

    def instant(
        self,
        name: str,
        cat: str,
        tid: int = 0,
        args: dict[str, Any] | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "name": name,
            "cat": cat,
            "ph": "i",
            "s": "p",
            "ts": time.perf_counter_ns() // 1000,
            "pid": self._pid,
            "tid": tid,
        }
        if args:
            event["args"] = args
        self._events.append(event)

    def save(self, path: str) -> str:
        meta = {
            "name": "process_name",
            "ph": "M",
            "pid": self._pid,
            "tid": 0,
            "args": {"name": "nano_qwen"},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"traceEvents": [meta, *self._events], "displayTimeUnit": "ms"},
                f,
            )
        return path


def enable_tracing() -> None:
    global _COLLECTOR
    if _COLLECTOR is None:
        _COLLECTOR = TraceCollector()


def save_trace(path: str) -> str | None:
    if _COLLECTOR is None:
        return None
    return _COLLECTOR.save(path)


@contextmanager
def trace_event(
    name: str,
    cat: str = "bench",
    args: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Record the duration of the wrapped block; no-op unless tracing."""
    collector = _COLLECTOR
    ts = time.perf_counter_ns() if collector is not None else 0
    try:
        yield
    finally:
        if collector is not None:
            collector.record(name, cat, ts, time.perf_counter_ns() - ts, args=args)
