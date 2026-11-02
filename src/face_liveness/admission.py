"""Resource guards: bounded concurrency with a bounded wait queue, and a per-request deadline.

ONNX inference is CPU-bound and not preemptible, so overload protection happens at
admission: at most ``slots`` checks run at once, at most ``max_waiting`` wait for a slot,
and everything beyond that is rejected immediately with a retryable ``BUSY``. The
deadline is checked at every phase boundary so the service never returns a decision
after its budget has elapsed (the caller may already have given up on it).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from .errors import ErrorCode, Guard, LivenessError


class AdmissionController:
    def __init__(self, slots: int, max_waiting: int) -> None:
        self.slots = slots
        self.max_waiting = max_waiting
        self._sem = threading.BoundedSemaphore(slots)
        self._lock = threading.Lock()
        self._waiting = 0
        self._in_flight = 0

    @property
    def waiting(self) -> int:
        return self._waiting

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @contextmanager
    def slot(self, timeout: float) -> Iterator[None]:
        if not self._sem.acquire(blocking=False):
            with self._lock:
                if self._waiting >= self.max_waiting:
                    raise LivenessError(
                        ErrorCode.BUSY, "service at capacity; retry shortly", Guard.QUEUE_FULL
                    )
                self._waiting += 1
            try:
                acquired = self._sem.acquire(timeout=max(timeout, 0.0))
            finally:
                with self._lock:
                    self._waiting -= 1
            if not acquired:
                raise LivenessError(
                    ErrorCode.BUSY, "service at capacity; retry shortly", Guard.QUEUE_TIMEOUT
                )
        with self._lock:
            self._in_flight += 1
        try:
            yield
        finally:
            with self._lock:
                self._in_flight -= 1
            self._sem.release()


class Deadline:
    def __init__(
        self,
        budget_seconds: float,
        started_at: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._expires = (clock() if started_at is None else started_at) + budget_seconds

    def remaining(self) -> float:
        return max(0.0, self._expires - self._clock())

    def check(self, phase: str) -> None:
        if self._clock() >= self._expires:
            raise LivenessError(
                ErrorCode.DEADLINE_EXCEEDED,
                f"request time budget exceeded ({phase})",
                Guard.DEADLINE,
            )
