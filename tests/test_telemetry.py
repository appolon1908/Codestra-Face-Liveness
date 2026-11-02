"""Privacy-safe telemetry (mission 10): bounded labels, decision and guard counters."""

from __future__ import annotations

import re
import time

from prometheus_client.parser import text_string_to_metric_families

from face_liveness.config import Environment, Settings
from face_liveness.errors import Guard
from face_liveness.metrics import ALLOWED_LABEL_NAMES

from .conftest import FakeClassifier, b64, fake_runtime
from .test_guards import png_bomb

CHECK = "/v1/liveness/check"
ISSUE = "/v1/liveness/challenges"
_FLOAT = re.compile(r"^-?\d+\.\d+$")
_HEX = re.compile(r"^[0-9a-f]{32,}$")


def _samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    return [
        (s.name, s.labels, s.value)
        for family in text_string_to_metric_families(text)
        for s in family.samples
    ]


def _value(text: str, name: str, **labels: str) -> float:
    for n, lbls, value in _samples(text):
        if n == name and all(lbls.get(k) == v for k, v in labels.items()):
            return value
    return 0.0


def test_decision_counters_by_outcome_reason_model_and_policy(client, classifier, image_b64):
    client.post(CHECK, json={"image_base64": image_b64})
    client.post(CHECK, json={"image_base64": image_b64})
    classifier.scores = (0.1, 0.2)
    client.post(CHECK, json={"image_base64": image_b64})
    text = client.get("/metrics").text
    base = {"model_version": "1.0.0", "policy_id": "passive-only.v1"}
    assert (
        _value(
            text,
            "liveness_decisions_total",
            decision="live",
            reason="score_at_or_above_threshold",
            **base,
        )
        == 2
    )
    assert (
        _value(
            text,
            "liveness_decisions_total",
            decision="spoof",
            reason="score_below_threshold",
            **base,
        )
        == 1
    )
    count = "liveness_decision_duration_seconds_count"
    assert _value(text, count, decision="live", model_version="1.0.0") == 2
    assert _value(text, "liveness_live_score_count", model_version="1.0.0") == 3


def test_guard_rejections_for_size(make_client):
    s = Settings(env=Environment.TEST, max_image_bytes=2048, max_image_side_px=512)
    client = make_client(settings_=s)
    client.post(CHECK, content=b"x" * 100_000, headers={"Content-Type": "application/json"})
    client.post(CHECK, json={"image_base64": b64(b"\x00" * 2100)})
    client.post(CHECK, json={"image_base64": b64(png_bomb(1000, 200, full=False))})
    s2 = Settings(env=Environment.TEST, max_image_pixels=10_000)
    client2 = make_client(settings_=s2)
    client2.post(CHECK, json={"image_base64": b64(png_bomb(200, 200, full=False))})
    s3 = Settings(env=Environment.TEST, max_decoded_bytes=64 * 1024)
    client3 = make_client(settings_=s3)
    client3.post(CHECK, json={"image_base64": b64(png_bomb(200, 200, full=False))})

    text = client.get("/metrics").text
    for guard in (Guard.REQUEST_BODY, Guard.IMAGE_BYTES, Guard.IMAGE_DIMENSIONS):
        assert _value(text, "liveness_guard_rejections_total", guard=guard.value) == 1, guard
    assert (
        _value(
            client2.get("/metrics").text, "liveness_guard_rejections_total", guard="image_pixels"
        )
        == 1
    )
    assert (
        _value(
            client3.get("/metrics").text, "liveness_guard_rejections_total", guard="decoded_bytes"
        )
        == 1
    )


def test_guard_rejections_for_concurrency_and_timeouts(make_client, detector, image_b64):
    class Slow(FakeClassifier):
        def score(self, bgr, face):  # type: ignore[no-untyped-def]
            time.sleep(0.2)
            return super().score(bgr, face)

    client = make_client(
        runtime=fake_runtime(detector, Slow()),
        settings_=Settings(env=Environment.TEST, request_timeout_seconds=0.05),
    )
    assert client.post(CHECK, json={"image_base64": image_b64}).status_code == 503
    text = client.get("/metrics").text
    assert _value(text, "liveness_guard_rejections_total", guard="deadline") == 1

    busy = make_client(
        settings_=Settings(env=Environment.TEST, max_outstanding_challenges=1),
    )
    busy.post(ISSUE)
    assert busy.post(ISSUE).status_code == 503
    text = busy.get("/metrics").text
    assert _value(text, "liveness_guard_rejections_total", guard="challenge_capacity") == 1


def test_queue_guards_are_distinguished():
    import pytest

    from face_liveness.admission import AdmissionController
    from face_liveness.errors import LivenessError

    ctl = AdmissionController(slots=1, max_waiting=0)
    with ctl.slot(timeout=0), pytest.raises(LivenessError) as full, ctl.slot(timeout=0):
        pass
    assert full.value.guard is Guard.QUEUE_FULL
    ctl = AdmissionController(slots=1, max_waiting=1)
    with ctl.slot(timeout=0), pytest.raises(LivenessError) as waited, ctl.slot(timeout=0.01):
        pass
    assert waited.value.guard is Guard.QUEUE_TIMEOUT


def test_labels_are_bounded_and_carry_no_request_data(make_client, classifier, image_b64):
    client = make_client(settings_=Settings(env=Environment.TEST, max_image_bytes=4096))
    rid = "req-4f1c2b-unique-per-request"
    headers = {"X-Request-ID": rid}
    client.post(CHECK, json={"image_base64": image_b64}, headers=headers)
    classifier.scores = (0.123456, 0.654321)
    client.post(CHECK, json={"image_base64": b64(b"\x00" * 5000)}, headers=headers)
    cid = client.post(ISSUE).json()["challenge_id"]
    client.post(f"/v1/liveness/challenges/{cid}/verify", json={"image_base64": "!!"})
    client.post(f"/v1/liveness/challenges/{cid}/verify", json={"image_base64": image_b64})
    client.get("/v1/models/some-unknown-version-xyz")
    client.get("/no/such/path/with-id-12345")
    text = client.get("/metrics").text

    seen_names: set[str] = set()
    for name, labels, _ in _samples(text):
        seen_names |= set(labels)
        for key, value in labels.items():
            if key in ("le", "quantile"):
                continue
            assert rid not in value
            assert cid not in value and "chl_" not in value
            assert "some-unknown-version" not in value and "12345" not in value
            if key != "manifest_sha256":
                assert not _HEX.match(value), (name, key, value)
            assert not _FLOAT.match(value), (name, key, value)
    assert seen_names - {"le"} <= ALLOWED_LABEL_NAMES
    for forbidden in ("request_id", "challenge_id", "subject_id", "user", "name", "image_sha256"):
        assert forbidden not in seen_names


def test_rejection_route_label_uses_templates(client, image_b64):
    client.post(
        "/v1/liveness/challenges/chl_" + "a" * 43 + "/verify", json={"image_base64": image_b64}
    )
    text = client.get("/metrics").text
    assert 'route="/v1/liveness/challenges/{challenge_id}/verify"' in text
    assert "chl_aaaa" not in text
