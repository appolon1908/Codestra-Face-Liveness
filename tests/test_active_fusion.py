"""Active-provider interface (mission 6) and versioned score-fusion policy (mission 7)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from face_liveness import active as active_mod
from face_liveness.active import (
    ACTIVE_CONTRACT_VERSION,
    KNOWN_PROVIDERS,
    ActiveOutcome,
    ActiveResult,
    ChallengeInstruction,
    ProviderDescriptor,
    resolve_active_provider,
)
from face_liveness.api import create_app
from face_liveness.config import Environment, Settings
from face_liveness.engine import Decision, DecisionReason, LivenessResult
from face_liveness.errors import ErrorCode, LivenessError
from face_liveness.fusion import (
    PASSIVE_AND_ACTIVE_V1,
    PASSIVE_ONLY_V1,
    POLICIES,
    EvidenceKind,
    fuse,
    resolve_policy,
)
from face_liveness.imaging import DecodedImage
from face_liveness.inference import FaceBox

from .conftest import b64, encode_image, fake_runtime

CHECK = "/v1/liveness/check"
ISSUE = "/v1/liveness/challenges"


def _verify(cid: str) -> str:
    return f"/v1/liveness/challenges/{cid}/verify"


@dataclass
class FakeProvider:
    """Test double only. Nothing like it ships in src/: no heuristic active liveness."""

    validation_id: str = "pad-eval-test-1"
    contract_version: str = ACTIVE_CONTRACT_VERSION
    outcome: ActiveOutcome = ActiveOutcome.PASS
    score: float | None = 0.9
    is_ready: bool = True
    seen: list[tuple[str, int]] = field(default_factory=list)

    @property
    def descriptor(self) -> ProviderDescriptor:
        return ProviderDescriptor(
            provider_id="fake-head-turn",
            version="0.0.1",
            contract_version=self.contract_version,
            validation_id=self.validation_id,
        )

    def ready(self) -> bool:
        return self.is_ready

    def instructions(self, challenge_id: str) -> list[ChallengeInstruction]:
        return [ChallengeInstruction(action="turn_head_left", timeout_seconds=5.0)]

    def evaluate(self, challenge_id: str, frames: Sequence[DecodedImage]) -> ActiveResult:
        self.seen.append((challenge_id, len(frames)))
        return ActiveResult("fake-head-turn", "0.0.1", self.outcome, self.score)


def _passive(decision: Decision, score: float = 0.9) -> LivenessResult:
    reason = (
        DecisionReason.SCORE_AT_OR_ABOVE_THRESHOLD
        if decision is Decision.LIVE
        else DecisionReason.SCORE_BELOW_THRESHOLD
    )
    return LivenessResult(
        decision=decision,
        reason=reason,
        live_score=score,
        threshold=0.85,
        face=FaceBox(0, 0, 100, 100, 0.99),
        components=[],
        faces_detected=1,
        inference_seconds=0.01,
    )


def _settings(**kw: object) -> Settings:
    return Settings(env=Environment.TEST, **kw)  # type: ignore[arg-type]


# --- mission 6: provider slot -----------------------------------------------------------------


def test_no_provider_ships():
    assert KNOWN_PROVIDERS == {}
    slot = resolve_active_provider(_settings())
    assert slot.provider is None and slot.configured_id is None
    assert not slot.available
    assert slot.reason == "no tested active liveness provider installed"


def test_no_heuristic_provider_code_in_package():
    # Guard against "blink / head-turn" heuristics creeping in as production liveness.
    assert active_mod.__file__ is not None
    assert "cv2" not in Path(active_mod.__file__).read_text(encoding="utf-8")
    for path in Path(active_mod.__file__).parent.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for forbidden in ("eye_aspect_ratio", "opticalflow", "optical_flow", "absdiff"):
            assert forbidden not in text, (path.name, forbidden)


def test_configured_unknown_provider_fails_closed(make_client):
    client = make_client(settings_=_settings(active_provider="blinky"))
    ready = client.get("/readyz")
    assert ready.status_code == 503
    assert "unknown active liveness provider: blinky" in ready.json()["checks"]["active_provider"]
    caps = client.get("/v1/capabilities").json()
    assert caps["active_liveness"] is False
    assert caps["active"]["configured_provider_id"] == "blinky"


@pytest.mark.parametrize(
    ("provider", "reason"),
    [
        (FakeProvider(validation_id=""), "untested providers are never used"),
        (FakeProvider(contract_version="active-provider.v0"), "not supported"),
        (FakeProvider(is_ready=False), "provider not ready"),
    ],
)
def test_untested_or_incompatible_provider_is_not_available(provider, reason):
    slot = resolve_active_provider(_settings(), provider)
    assert not slot.available
    assert reason in (slot.reason or "")


def test_default_contract_has_no_active_claims(client):
    caps = client.get("/v1/capabilities").json()
    assert caps["active"] == {
        "contract_version": "active-provider.v1",
        "available": False,
        "configured_provider_id": None,
        "provider": None,
        "max_frames": 8,
        "reason": "no tested active liveness provider installed",
    }
    assert "active_challenge" in caps["unsupported"]
    assert caps["challenge"]["challenge_type"] == "freshness_nonce"


def test_active_frames_without_provider_fail_closed_and_keep_challenge(client, image_b64):
    cid = client.post(ISSUE).json()["challenge_id"]
    body = {"image_base64": image_b64, "active_frames_base64": [image_b64]}
    resp = client.post(_verify(cid), json=body)
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "EVIDENCE_UNAVAILABLE"
    # Rejected before the challenge was spent: a request without frames still works.
    ok = client.post(_verify(cid), json={"image_base64": image_b64})
    assert ok.status_code == 200


def test_active_frames_rejected_on_plain_check(client, image_b64):
    resp = client.post(CHECK, json={"image_base64": image_b64, "active_frames_base64": ["x"]})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_pluggable_provider_end_to_end(detector, classifier, image_b64):
    """A tested provider can be plugged in without changing the API shape."""
    provider = FakeProvider()
    app = create_app(
        _settings(fusion_policy_id="passive-and-active.v1"),
        fake_runtime(detector, classifier),
        active_provider=provider,
    )
    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200
        caps = client.get("/v1/capabilities").json()
        assert caps["active_liveness"] is True
        assert caps["active"]["provider"]["validation_id"] == "pad-eval-test-1"
        assert caps["fusion"]["required_evidence"] == ["passive", "active"]
        issued = client.post(ISSUE).json()
        assert issued["challenge_type"] == "active_challenge"
        assert issued["instructions"] == [{"action": "turn_head_left", "timeout_seconds": 5.0}]

        frames = [b64(encode_image(seed=i)) for i in range(3)]
        body = {"image_base64": image_b64, "active_frames_base64": frames}
        resp = client.post(_verify(issued["challenge_id"]), json=body)
        assert resp.status_code == 200, resp.text
        ev = resp.json()["evidence"]
        assert provider.seen == [(issued["challenge_id"], 3)]
        assert ev["decision"] == "live"
        assert ev["decision_reason"] == "all_required_evidence_passed"
        assert ev["policy_id"] == "passive-and-active.v1"
        assert ev["evidence_used"] == ["passive", "active"]
        assert ev["active_liveness_evaluated"] is True
        assert ev["challenge"]["active_liveness_evaluated"] is True
        assert ev["active"] == {
            "provider_id": "fake-head-turn",
            "provider_version": "0.0.1",
            "outcome": "pass",
            "score": 0.9,
        }

        # Passive-only request under a policy that requires active evidence: fail closed.
        cid = client.post(ISSUE).json()["challenge_id"]
        missing = client.post(_verify(cid), json={"image_base64": image_b64})
        assert missing.status_code == 503
        assert missing.json()["error"]["code"] == "EVIDENCE_UNAVAILABLE"
        assert client.app.state.challenges.outstanding() == 2  # not spent
        assert client.post(CHECK, json={"image_base64": image_b64}).status_code == 503

        provider.outcome = ActiveOutcome.FAIL
        cid = client.post(ISSUE).json()["challenge_id"]
        failed = client.post(_verify(cid), json=body).json()
        assert failed["decision"] == "spoof"
        assert failed["evidence"]["decision_reason"] == "active_challenge_failed"

        too_many = {"image_base64": image_b64, "active_frames_base64": [image_b64] * 9}
        cid = client.post(ISSUE).json()["challenge_id"]
        assert client.post(_verify(cid), json=too_many).status_code == 400


# --- mission 7: fusion policy ------------------------------------------------------------------


def test_policy_ids_are_explicit_and_versioned():
    for pid, policy in POLICIES.items():
        assert pid == policy.policy_id
        assert pid.rsplit(".", 1)[1].startswith("v")
    slot = resolve_active_provider(_settings())
    for bad in ("passive-only", "latest", "PASSIVE-ONLY.V1", ""):
        state = resolve_policy(bad, slot)
        assert not state.ready
        assert "versioned" in (state.error or "")
    unknown = resolve_policy("passive-only.v9", slot)
    assert not unknown.ready and "unknown fusion policy" in (unknown.error or "")


@pytest.mark.parametrize("decision", [Decision.LIVE, Decision.SPOOF])
def test_passive_only_equals_passive_decision(decision):
    passive = _passive(decision)
    fused = fuse(PASSIVE_ONLY_V1, passive, None)
    assert fused.decision is decision
    assert fused.reason is passive.reason
    assert fused.policy_id == "passive-only.v1"
    assert fused.evidence_used == (EvidenceKind.PASSIVE,)


def test_passive_only_ignores_nothing_it_did_not_ask_for():
    # Even if active evidence were present, passive-only decides on passive alone.
    active = ActiveResult("p", "1", ActiveOutcome.FAIL, 0.0)
    assert fuse(PASSIVE_ONLY_V1, _passive(Decision.LIVE), active).decision is Decision.LIVE


def test_required_active_missing_fails_closed():
    with pytest.raises(LivenessError) as exc:
        fuse(PASSIVE_AND_ACTIVE_V1, _passive(Decision.LIVE), None)
    assert exc.value.code is ErrorCode.EVIDENCE_UNAVAILABLE
    assert exc.value.retryable is True


@pytest.mark.parametrize(
    ("passive", "outcome", "score", "decision", "reason"),
    [
        (Decision.LIVE, ActiveOutcome.PASS, 0.9, Decision.LIVE, "all_required_evidence_passed"),
        (Decision.LIVE, ActiveOutcome.PASS, None, Decision.LIVE, "all_required_evidence_passed"),
        (Decision.LIVE, ActiveOutcome.FAIL, 0.9, Decision.SPOOF, "active_challenge_failed"),
        (Decision.LIVE, ActiveOutcome.PASS, 0.4, Decision.SPOOF, "active_score_below_minimum"),
        (Decision.SPOOF, ActiveOutcome.PASS, 1.0, Decision.SPOOF, "score_below_threshold"),
    ],
)
def test_all_required_is_conjunctive(passive, outcome, score, decision, reason):
    active = ActiveResult("p", "1", outcome, score)
    fused = fuse(PASSIVE_AND_ACTIVE_V1, _passive(passive), active)
    assert fused.decision is decision
    assert fused.reason.value == reason
    assert fused.evidence_used == (EvidenceKind.PASSIVE, EvidenceKind.ACTIVE)


def test_future_provider_unavailable_fails_closed_end_to_end(make_client, image_b64):
    client = make_client(settings_=_settings(fusion_policy_id="passive-and-active.v1"))
    ready = client.get("/readyz")
    assert ready.status_code == 503
    assert "requires active evidence" in ready.json()["checks"]["fusion_policy"]
    caps = client.get("/v1/capabilities").json()
    assert caps["fusion"]["ready"] is False
    assert caps["fusion"]["policy_id"] == "passive-and-active.v1"

    resp = client.post(CHECK, json={"image_base64": image_b64})
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "EVIDENCE_UNAVAILABLE"
    assert "decision" not in resp.text

    # The challenge is not spent by a request the policy could never satisfy.
    cid = client.post(ISSUE).json()["challenge_id"]
    assert client.post(_verify(cid), json={"image_base64": image_b64}).status_code == 503
    assert client.app.state.challenges.outstanding() == 1
    metrics = client.get("/metrics").text
    assert 'liveness_fusion_policy_ready{policy_id="passive-and-active.v1"} 0.0' in metrics
    assert 'code="EVIDENCE_UNAVAILABLE",route="/v1/liveness/check",stage="policy"' in metrics


def test_unknown_policy_is_not_ready_and_never_falls_back(make_client, image_b64):
    client = make_client(settings_=_settings(fusion_policy_id="weighted.v1"))
    assert client.get("/readyz").status_code == 503
    assert client.post(CHECK, json={"image_base64": image_b64}).status_code == 503
    assert 'liveness_fusion_policy_ready{policy_id="unknown"} 0.0' in client.get("/metrics").text


def test_passive_only_capabilities(client):
    fusion = client.get("/v1/capabilities").json()["fusion"]
    assert fusion == {
        "policy_id": "passive-only.v1",
        "strategy": "passive_only",
        "required_evidence": ["passive"],
        "description": PASSIVE_ONLY_V1.description,
        "ready": True,
        "reason": None,
    }
