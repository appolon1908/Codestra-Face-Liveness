"""Pluggable active (challenge-response) liveness provider interface.

No active provider ships with this service. The interface exists so that a real,
independently tested provider (e.g. a head-turn or blink model with its own PAD
evaluation) can be plugged in without changing the API shape. Until then:

* ``active_liveness`` is false in capabilities, challenges and evidence;
* challenges carry no instructions;
* a fusion policy that requires active evidence fails closed (``EVIDENCE_UNAVAILABLE``).

There are deliberately no heuristic providers (frame differencing, eye-aspect-ratio
blink counting, landmark head-pose checks, ...): they are trivially defeated by replayed
video and must never be presented as production liveness.

A provider is only reported as available when it is ready *and* names the validation
report that tested it (``validation_id``). Providers are selected by id from
``KNOWN_PROVIDERS``, which is empty; any configured id therefore fails closed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .config import Settings
from .imaging import DecodedImage

ACTIVE_CONTRACT_VERSION = "active-provider.v1"


class ActiveOutcome(StrEnum):
    PASS = "pass"  # noqa: S105 - outcome label, not a credential
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    provider_id: str
    version: str
    contract_version: str
    # Identifier of the PAD evaluation (ISO/IEC 30107-3 style) that tested this exact
    # provider version. Empty means untested: the provider is never reported available.
    validation_id: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class ChallengeInstruction:
    """One user action the capture client must perform (e.g. ``turn_head_left``)."""

    action: str
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class ActiveResult:
    provider_id: str
    provider_version: str
    outcome: ActiveOutcome
    score: float | None  # provider confidence in [0, 1] when it has one


class ActiveLivenessProvider(Protocol):
    """What a provider must implement. All methods must be thread-safe."""

    @property
    def descriptor(self) -> ProviderDescriptor: ...

    def ready(self) -> bool: ...

    def instructions(self, challenge_id: str) -> list[ChallengeInstruction]:
        """Instructions bound to a freshly issued challenge."""
        ...

    def evaluate(self, challenge_id: str, frames: Sequence[DecodedImage]) -> ActiveResult:
        """Evaluate the capture made in response to ``challenge_id``.

        Must raise ``LivenessError`` rather than return a result it cannot stand behind.
        """
        ...


# provider id -> factory. Empty on purpose: no tested active provider exists.
KNOWN_PROVIDERS: dict[str, Callable[[Settings], ActiveLivenessProvider]] = {}


@dataclass(frozen=True, slots=True)
class ActiveSlot:
    """The (at most one) configured active provider and why it is or is not available."""

    provider: ActiveLivenessProvider | None
    configured_id: str | None
    error: str | None

    @property
    def available(self) -> bool:
        p = self.provider
        return (
            self.error is None
            and p is not None
            and p.descriptor.contract_version == ACTIVE_CONTRACT_VERSION
            and bool(p.descriptor.validation_id)
            and p.ready()
        )

    @property
    def reason(self) -> str | None:
        if self.available:
            return None
        if self.error is not None:
            return self.error
        p = self.provider
        if p is None:
            return "no tested active liveness provider installed"
        if p.descriptor.contract_version != ACTIVE_CONTRACT_VERSION:
            return f"provider contract {p.descriptor.contract_version} is not supported"
        if not p.descriptor.validation_id:
            return "provider has no validation report; untested providers are never used"
        return "provider not ready"


def resolve_active_provider(
    settings: Settings, injected: ActiveLivenessProvider | None = None
) -> ActiveSlot:
    """Pick the configured provider. Never raises; unknown ids fail closed."""
    if injected is not None:
        return ActiveSlot(injected, injected.descriptor.provider_id, None)
    configured = settings.active_provider or None
    if configured is None:
        return ActiveSlot(None, None, None)
    factory = KNOWN_PROVIDERS.get(configured)
    if factory is None:
        return ActiveSlot(None, configured, f"unknown active liveness provider: {configured}")
    try:
        return ActiveSlot(factory(settings), configured, None)
    except Exception as exc:
        return ActiveSlot(None, configured, f"active provider failed to load: {type(exc).__name__}")
