"""Per-job event bus powering the live cockpit (SSE).

Nodes call `emit(...)` unconditionally; it is a no-op unless the current thread of
execution was bound to a JobStream via `bind(...)` (done by the API's job runner).
Tests and CLI runs therefore need no special handling.
"""

from __future__ import annotations

import contextvars
import threading
import time
from typing import Any

_current: contextvars.ContextVar["JobStream | None"] = contextvars.ContextVar(
    "datamedic_job_stream", default=None
)


class JobStream:
    """Append-only event log for one job; SSE subscribers replay then follow."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.closed = False
        self._lock = threading.Lock()

    def emit(self, kind: str, **data: Any) -> None:
        with self._lock:
            self.events.append({"kind": kind, "t": round(time.time(), 3), **data})

    def close(self) -> None:
        with self._lock:
            self.closed = True


STREAMS: dict[str, JobStream] = {}


def create_stream(job_id: str) -> JobStream:
    stream = JobStream()
    STREAMS[job_id] = stream
    return stream


def bind(stream: JobStream | None) -> None:
    """Bind a stream to the current execution context (the job-runner thread)."""
    _current.set(stream)


def emit(kind: str, **data: Any) -> None:
    stream = _current.get()
    if stream is not None:
        stream.emit(kind, **data)


def current_stream() -> JobStream | None:
    """The stream bound to this context, for handing to worker threads explicitly."""
    return _current.get()
