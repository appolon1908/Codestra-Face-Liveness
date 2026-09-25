"""Score fusion policy: how passive and (future) active evidence combine into one decision.

A policy is selected by an explicit, versioned id (``LIVENESS_FUSION_POLICY_ID``, e.g.
``passive-only.v1``). Policies are immutable: changing behaviour means a new version,
so every decision's ``evidence.policy_id`` identifies exactly the rule that produced it.

Fail closed: a policy that requires evidence the service cannot produce (today: any
active evidence) keeps the service not-ready and every assessment returns
``EVIDENCE_UNAVAILABLE``; the service never silently falls back to a weaker policy.

Only conjunctive fusion is offered: each required evidence source must pass on its own.
Weighted score fusion would need a jointly calibrated threshold, which cannot exist
before a tested active provider does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from .active import ActiveOutcome, ActiveResult, ActiveSlot
from .engine import Decision, DecisionReason, LivenessResult
from .errors import ErrorCode, LivenessError

POLICY_ID_RE = re.compile(r"^[a-z][a-z0-9-]{0,47}\.v[1-9][0-9]{0,3}$")


class EvidenceKind(StrEnum):
    PASSIVE = "passive"
    ACTIVE = "active"


class FusionStrategy(StrEnum):
    PASSIVE_ONLY = "passive_only"
    ALL_REQUIRED = "all_required"


@dataclass(frozen=True, slots=True)
class FusionPolicy:
    policy_id: str
    strategy: FusionStrategy
    required: tuple[EvidenceKind, ...]
    description: str
    # ALL_REQUIRED only: minimum provider score, when the provider reports one.
    active_min_score: float | None = None

    @property
    def requires_active(self) -> bool:
        return EvidenceKind.ACTIVE in self.required


PASSIVE_ONLY_V1 = FusionPolicy(
    policy_id="passive-only.v1",
    strategy=FusionStrategy.PASSIVE_ONLY,
    required=(EvidenceKind.PASSIVE,),
    description="Decision = passive decision (live iff live_score >= threshold).",
)
PASSIVE_AND_ACTIVE_V1 = FusionPolicy(
    policy_id="passive-and-active.v1",
    strategy=FusionStrategy.ALL_REQUIRED,
    required=(EvidenceKind.PASSIVE, EvidenceKind.ACTIVE),
    description=(
        "Live only if the passive decision is live AND a tested active provider passes "
        "(score >= 0.5 when it reports one). Missing active evidence fails closed."
    ),
    active_min_score=0.5,
)
POLICIES: dict[str, FusionPolicy] = {
    p.policy_id: p for p in (PASSIVE_ONLY_V1, PASSIVE_AND_ACTIVE_V1)
}
DEFAULT_POLICY_ID = PASSIVE_ONLY_V1.policy_id


@dataclass(frozen=True, slots=True)
class PolicyState:
    """The configured policy (if known) and whether its required evidence is available."""

    configured_id: str
    policy: FusionPolicy | None
    error: str | None

    @property
    def ready(self) -> bool:
        return self.policy is not None and self.error is None


def resolve_policy(policy_id: str, active: ActiveSlot) -> PolicyState:
    if not POLICY_ID_RE.match(policy_id):
        return PolicyState(
            policy_id, None, "fusion policy id must be explicit and versioned (<name>.v<N>)"
        )
    policy = POLICIES.get(policy_id)
    if policy is None:
        return PolicyState(policy_id, None, f"unknown fusion policy: {policy_id}")
    if policy.requires_active and not active.available:
        return PolicyState(
            policy_id,
            policy,
            f"fusion policy {policy_id} requires active evidence: {active.reason}",
        )
    return PolicyState(policy_id, policy, None)


@dataclass(frozen=True, slots=True)
class FusedDecision:
    decision: Decision
    reason: DecisionReason
    policy_id: str
    evidence_used: tuple[EvidenceKind, ...]
    active: ActiveResult | None


def require_ready(state: PolicyState) -> FusionPolicy:
    """The policy, or EVIDENCE_UNAVAILABLE before any work is spent on the request."""
    if state.policy is None or state.error is not None:
        raise LivenessError(
            ErrorCode.EVIDENCE_UNAVAILABLE,
            state.error or f"fusion policy unavailable: {state.configured_id}",
        )
    return state.policy


def fuse(
    policy: FusionPolicy, passive: LivenessResult, active: ActiveResult | None
) -> FusedDecision:
    """Combine evidence under ``policy``. Raises EVIDENCE_UNAVAILABLE if any is missing."""
    if policy.strategy is FusionStrategy.PASSIVE_ONLY:
        return FusedDecision(
            passive.decision, passive.reason, policy.policy_id, (EvidenceKind.PASSIVE,), None
        )

    if policy.requires_active and active is None:
        raise LivenessError(
            ErrorCode.EVIDENCE_UNAVAILABLE,
            f"fusion policy {policy.policy_id} requires active evidence; none was produced",
        )
    used = (EvidenceKind.PASSIVE, EvidenceKind.ACTIVE) if active else (EvidenceKind.PASSIVE,)
    if passive.decision is not Decision.LIVE:
        return FusedDecision(Decision.SPOOF, passive.reason, policy.policy_id, used, active)
    if active is not None:
        if active.outcome is not ActiveOutcome.PASS:
            return FusedDecision(
                Decision.SPOOF,
                DecisionReason.ACTIVE_CHALLENGE_FAILED,
                policy.policy_id,
                used,
                active,
            )
        if (
            policy.active_min_score is not None
            and active.score is not None
            and active.score < policy.active_min_score
        ):
            return FusedDecision(
                Decision.SPOOF,
                DecisionReason.ACTIVE_SCORE_BELOW_MINIMUM,
                policy.policy_id,
                used,
                active,
            )
    return FusedDecision(
        Decision.LIVE, DecisionReason.ALL_REQUIRED_EVIDENCE_PASSED, policy.policy_id, used, active
    )
