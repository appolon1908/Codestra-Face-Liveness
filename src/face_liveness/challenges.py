"""Short-lived, single-use challenges (issue / verify contract).

Today a challenge is a freshness and replay-prevention nonce only: Middleware V3 issues
one, the capture is submitted against it, and the challenge is consumed exactly once.
It carries no instructions and the passive model's decision remains authoritative. The
contract exists so that a future *active* liveness model can attach instructions (e.g.
head turn) without changing the API shape; until such a model is installed,
``active_liveness`` is reported as false everywhere.

Challenge IDs are 256-bit random opaque tokens. Only their SHA-256 is held, in process
memory, bounded in count, and forgotten once expired. A consumed challenge is remembered
until its expiry so replays are reported as ``CHALLENGE_ALREADY_USED``. The store is
per-process: with several replicas the verify call must reach the replica that issued
the challenge (session affinity in Middleware V3) or it fails closed as not found.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .errors import ErrorCode, LivenessError

CHALLENGE_PREFIX = "chl_"
_ID_RE = re.compile(r"^chl_[A-Za-z0-9_\-]{43}$")


@dataclass(frozen=True, slots=True)
class IssuedChallenge:
    challenge_id: str
    issued_at: datetime
    expires_at: datetime
    ttl_seconds: int


@dataclass(slots=True)
class _Entry:
    expires_at: float  # monotonic
    consumed: bool = False


class ChallengeStore:
    def __init__(
        self,
        ttl_seconds: int,
        capacity: int,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.capacity = capacity
        self.clock = clock
        self.wall_clock = wall_clock
        self._lock = threading.Lock()
        # Insertion order == expiry order (constant TTL), so purging pops from the front.
        self._entries: OrderedDict[bytes, _Entry] = OrderedDict()

    @staticmethod
    def _key(challenge_id: str) -> bytes:
        return hashlib.sha256(challenge_id.encode()).digest()

    def _purge(self, now: float) -> None:
        while self._entries:
            key, entry = next(iter(self._entries.items()))
            if entry.expires_at > now:
                break
            del self._entries[key]

    def outstanding(self) -> int:
        with self._lock:
            self._purge(self.clock())
            return len(self._entries)

    def issue(self) -> IssuedChallenge:
        challenge_id = CHALLENGE_PREFIX + secrets.token_urlsafe(32)
        with self._lock:
            now = self.clock()
            self._purge(now)
            if len(self._entries) >= self.capacity:
                raise LivenessError(ErrorCode.BUSY, "too many outstanding challenges")
            self._entries[self._key(challenge_id)] = _Entry(expires_at=now + self.ttl_seconds)
        issued = datetime.fromtimestamp(self.wall_clock(), UTC)
        return IssuedChallenge(
            challenge_id=challenge_id,
            issued_at=issued,
            expires_at=issued + timedelta(seconds=self.ttl_seconds),
            ttl_seconds=self.ttl_seconds,
        )

    def consume(self, challenge_id: str) -> None:
        """Atomically consume a challenge. Raises unless it is known, unexpired and unused."""
        if not _ID_RE.match(challenge_id):
            raise LivenessError(ErrorCode.CHALLENGE_NOT_FOUND, "unknown challenge")
        key = self._key(challenge_id)
        with self._lock:
            now = self.clock()
            entry = self._entries.get(key)
            if entry is not None and entry.expires_at <= now:
                raise LivenessError(ErrorCode.CHALLENGE_EXPIRED, "challenge expired")
            self._purge(now)
            if entry is None:
                raise LivenessError(ErrorCode.CHALLENGE_NOT_FOUND, "unknown challenge")
            if entry.consumed:
                raise LivenessError(ErrorCode.CHALLENGE_ALREADY_USED, "challenge already used")
            entry.consumed = True
