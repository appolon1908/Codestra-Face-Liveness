"""Decision evidence: bounded, fixed-shape, attributable; never raw bytes or tensors."""

from __future__ import annotations

import json

from face_liveness.config import Environment, Settings
from face_liveness.registry import builtin_manifest

CHECK = "/v1/liveness/check"
EVIDENCE_KEYS = {
    "model_id",
    "model_version",
    "model_digest",
    "detector_id",
    "liveness_type",
    "active_liveness_evaluated",
    "score",
    "score_aggregation",
    "threshold",
    "threshold_calibrated",
    "calibration_id",
    "margin",
    "policy_id",
    "evidence_used",
    "decision",
    "decision_reason",
    "challenge",
    "active",
}


def test_live_evidence(client, image_b64):
    body = client.post(CHECK, json={"image_base64": image_b64}).json()
    ev = body["evidence"]
    assert set(ev) == EVIDENCE_KEYS
    manifest = builtin_manifest(Settings(env=Environment.TEST))
    assert ev["model_id"] == manifest.model_id
    assert ev["model_version"] == "1.0.0"
    assert ev["model_digest"] == manifest.digest
    assert ev["detector_id"] == "yunet-2023mar"
    assert ev["liveness_type"] == "passive"
    assert ev["active_liveness_evaluated"] is False
    assert ev["score"] == body["live_score"] == 0.96
    assert ev["threshold"] == 0.85
    assert ev["margin"] == 0.11
    assert ev["decision"] == "live"
    assert ev["decision_reason"] == "score_at_or_above_threshold"
    assert ev["threshold_calibrated"] is False
    assert ev["calibration_id"] is None
    assert ev["challenge"] is None
    assert ev["policy_id"] == "passive-only.v1"
    assert ev["evidence_used"] == ["passive"]
    assert ev["active"] is None


def test_spoof_evidence(client, classifier, image_b64):
    classifier.scores = (0.5, 0.3)
    ev = client.post(CHECK, json={"image_base64": image_b64}).json()["evidence"]
    assert ev["decision"] == "spoof"
    assert ev["decision_reason"] == "score_below_threshold"
    assert ev["margin"] == -0.45


def test_evidence_reports_calibration(make_client, image_b64):
    s = Settings(env=Environment.TEST, live_threshold=0.9, threshold_calibration_id="cal-x")
    ev = make_client(settings_=s).post(CHECK, json={"image_base64": image_b64}).json()["evidence"]
    assert ev["threshold"] == 0.9
    assert ev["threshold_calibrated"] is True
    assert ev["calibration_id"] == "cal-x"


def test_response_is_bounded_and_has_no_image_data(client, image_b64):
    resp = client.post(CHECK, json={"image_base64": image_b64})
    text = resp.text
    assert len(text) < 4096
    assert image_b64[:64] not in text

    def walk(value: object) -> None:
        # Only scalars, short strings and small containers: no arrays of numbers, no blobs.
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            assert len(value) <= 8
            for v in value:
                walk(v)
        elif isinstance(value, str):
            assert len(value) <= 128

    walk(json.loads(text))


def test_errors_carry_no_evidence(client, detector, image_b64):
    detector.faces = []
    assert "evidence" not in client.post(CHECK, json={"image_base64": image_b64}).text
