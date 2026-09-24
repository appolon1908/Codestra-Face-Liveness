from __future__ import annotations

from face_liveness.config import Environment, Settings
from face_liveness.inference import FaceBox
from face_liveness.runtime import ModelRuntime

from .conftest import b64, encode_image

CHECK = "/v1/liveness/check"


def _err(resp):  # type: ignore[no-untyped-def]
    body = resp.json()
    assert set(body) == {"error"}
    return body["error"]


# --- happy path -------------------------------------------------------------------------


def test_live_decision(client, image_b64):
    resp = client.post(CHECK, json={"image_base64": image_b64}, headers={"X-Request-ID": "req-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "live"
    assert body["is_live"] is True
    assert body["live_score"] == 0.96
    assert body["threshold"] == 0.85
    assert body["method"] == "passive_single_image"
    assert body["request_id"] == "req-1"
    assert resp.headers["X-Request-ID"] == "req-1"
    assert body["faces_detected"] == 1
    assert body["face"] == {"x": 60, "y": 50, "width": 120, "height": 140, "detection_score": 0.98}
    assert [c["name"] for c in body["components"]] == ["fake_0", "fake_1"]
    assert body["image"] == {"width": 240, "height": 240, "media_type": "image/png"}
    assert body["image_persisted"] is False
    assert body["model"]["threshold_calibrated"] is False


def test_spoof_score_below_threshold_is_spoof(client, classifier, image_b64):
    classifier.scores = (0.9, 0.7)  # mean 0.80 < 0.85
    resp = client.post(CHECK, json={"image_base64": image_b64})
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "spoof"
    assert body["is_live"] is False
    assert body["live_score"] == 0.8


def test_score_equal_to_threshold_is_live(client, classifier, image_b64):
    classifier.scores = (0.85,)
    assert client.post(CHECK, json={"image_base64": image_b64}).json()["decision"] == "live"


def test_one_weak_component_can_pull_decision_to_spoof(client, classifier, image_b64):
    classifier.scores = (0.99, 0.5)
    assert client.post(CHECK, json={"image_base64": image_b64}).json()["decision"] == "spoof"


def test_generated_request_id(client, image_b64):
    resp = client.post(CHECK, json={"image_base64": image_b64}, headers={"X-Request-ID": "bad id!"})
    rid = resp.json()["request_id"]
    assert rid != "bad id!" and len(rid) == 36
    assert resp.headers["X-Request-ID"] == rid


def test_data_url_prefix_and_other_formats(client):
    for fmt, media in (("JPEG", "image/jpeg"), ("WEBP", "image/webp")):
        data = f"data:{media};base64," + b64(encode_image(fmt))
        resp = client.post(CHECK, json={"image_base64": data})
        assert resp.status_code == 200, resp.text
        assert resp.json()["image"]["media_type"] == media


# --- negative: faces --------------------------------------------------------------------


def test_no_face(client, detector, classifier, image_b64):
    detector.faces = []
    resp = client.post(CHECK, json={"image_base64": image_b64})
    assert resp.status_code == 422
    err = _err(resp)
    assert err["code"] == "NO_FACE"
    assert err["retryable"] is False
    assert classifier.calls == 0


def test_multiple_faces(client, detector, classifier, image_b64):
    detector.faces = [
        FaceBox(10, 10, 100, 100, 0.99),
        FaceBox(130, 10, 90, 90, 0.95),
    ]
    resp = client.post(CHECK, json={"image_base64": image_b64})
    assert resp.status_code == 422
    assert _err(resp)["code"] == "MULTIPLE_FACES"
    assert classifier.calls == 0


def test_multiple_faces_even_if_tiny_by_default(client, detector, image_b64):
    detector.faces = [FaceBox(10, 10, 150, 150, 0.99), FaceBox(200, 200, 10, 10, 0.9)]
    assert _err(client.post(CHECK, json={"image_base64": image_b64}))["code"] == "MULTIPLE_FACES"


def test_secondary_face_tolerance(make_client, detector, image_b64):
    s = Settings(env=Environment.TEST, secondary_face_area_ratio=0.1)
    client = make_client(settings_=s)
    detector.faces = [FaceBox(10, 10, 150, 150, 0.99), FaceBox(200, 200, 10, 10, 0.9)]
    resp = client.post(CHECK, json={"image_base64": image_b64})
    assert resp.status_code == 200
    assert resp.json()["faces_detected"] == 2


def test_face_too_small(client, detector, image_b64):
    detector.faces = [FaceBox(10, 10, 40, 40, 0.99)]
    assert _err(client.post(CHECK, json={"image_base64": image_b64}))["code"] == "FACE_TOO_SMALL"


# --- negative: input --------------------------------------------------------------------


def test_invalid_base64(client):
    resp = client.post(CHECK, json={"image_base64": "not base64!!"})
    assert resp.status_code == 400
    assert _err(resp)["code"] == "INVALID_REQUEST"


def test_invalid_image_bytes(client, detector):
    resp = client.post(CHECK, json={"image_base64": b64(b"\x89PNG\r\n\x1a\n garbage" * 10)})
    assert resp.status_code == 422
    assert _err(resp)["code"] == "INVALID_IMAGE"
    assert detector.calls == 0


def test_truncated_jpeg(client):
    data = encode_image("JPEG")
    resp = client.post(CHECK, json={"image_base64": b64(data[: len(data) // 2])})
    assert resp.status_code == 422
    assert _err(resp)["code"] == "INVALID_IMAGE"


def test_unsupported_format(client):
    resp = client.post(CHECK, json={"image_base64": b64(encode_image("BMP"))})
    assert resp.status_code == 415
    assert _err(resp)["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_image_too_small(client):
    resp = client.post(CHECK, json={"image_base64": b64(encode_image(size=(64, 64)))})
    assert _err(resp)["code"] == "IMAGE_TOO_SMALL"


def test_image_too_many_pixels(make_client):
    client = make_client(settings_=Settings(env=Environment.TEST, max_image_pixels=40_000))
    resp = client.post(CHECK, json={"image_base64": b64(encode_image(size=(300, 300)))})
    assert resp.status_code == 413
    assert _err(resp)["code"] == "PAYLOAD_TOO_LARGE"


def test_image_too_many_bytes(make_client):
    client = make_client(settings_=Settings(env=Environment.TEST, max_image_bytes=2048))
    resp = client.post(CHECK, json={"image_base64": b64(encode_image())})
    assert resp.status_code == 413
    assert _err(resp)["code"] == "PAYLOAD_TOO_LARGE"


def test_request_body_limit_without_parsing(make_client):
    client = make_client(settings_=Settings(env=Environment.TEST, max_image_bytes=1024))
    resp = client.post(CHECK, content=b"x" * 100_000, headers={"Content-Type": "application/json"})
    assert resp.status_code == 413
    assert _err(resp)["code"] == "PAYLOAD_TOO_LARGE"


def test_missing_field_and_unknown_field(client, image_b64):
    for payload in ({}, {"image_base64": image_b64, "subject_id": "x"}, {"image_base64": ""}):
        resp = client.post(CHECK, json=payload)
        assert resp.status_code == 400
        assert _err(resp)["code"] == "INVALID_REQUEST"


def test_unsupported_mode(client, image_b64):
    resp = client.post(CHECK, json={"image_base64": image_b64, "mode": "active_challenge"})
    assert _err(resp)["code"] == "INVALID_REQUEST"


def test_validation_error_does_not_echo_input(client):
    secret_payload = "A" * 50 + "SENSITIVE"
    resp = client.post(CHECK, json={"image_base64": 12345, "extra": secret_payload})
    assert resp.status_code == 400
    assert "SENSITIVE" not in resp.text


def test_not_json(client):
    resp = client.post(CHECK, content=b"{", headers={"Content-Type": "application/json"})
    assert resp.status_code == 400


# --- negative: model / runtime (fail closed) ---------------------------------------------


def test_model_unavailable_fails_closed(make_client, image_b64):
    client = make_client(runtime=ModelRuntime.unavailable("model artifact missing"))
    resp = client.post(CHECK, json={"image_base64": image_b64})
    assert resp.status_code == 503
    err = _err(resp)
    assert err["code"] == "MODEL_UNAVAILABLE"
    assert err["retryable"] is True
    assert "decision" not in resp.text


def test_model_unavailable_even_for_invalid_image(make_client):
    client = make_client(runtime=ModelRuntime.unavailable("x"))
    resp = client.post(CHECK, json={"image_base64": b64(b"garbage")})
    assert _err(resp)["code"] == "MODEL_UNAVAILABLE"


def test_inference_failure_fails_closed(client, classifier, image_b64):
    classifier.fail = True
    resp = client.post(CHECK, json={"image_base64": image_b64})
    assert resp.status_code == 503
    assert _err(resp)["code"] == "INFERENCE_FAILED"


def test_detector_failure_fails_closed(client, detector, image_b64):
    detector.fail = True
    resp = client.post(CHECK, json={"image_base64": image_b64})
    assert resp.status_code == 503
    assert _err(resp)["code"] == "INFERENCE_FAILED"


def test_out_of_range_score_fails_closed(client, classifier, image_b64):
    classifier.scores = (1.5,)
    assert _err(client.post(CHECK, json={"image_base64": image_b64}))["code"] == "INFERENCE_FAILED"


def test_nan_score_fails_closed(client, classifier, image_b64):
    classifier.scores = (float("nan"),)
    assert _err(client.post(CHECK, json={"image_base64": image_b64}))["code"] == "INFERENCE_FAILED"


def test_busy_returns_retryable_503(make_client, image_b64):
    s = Settings(env=Environment.TEST, max_concurrent_checks=1, busy_timeout_seconds=0)
    client = make_client(settings_=s)
    slots = client.app.state.check_slots  # hold the only slot from outside the request
    slots.acquire()
    try:
        resp = client.post(CHECK, json={"image_base64": image_b64})
    finally:
        slots.release()
    assert resp.status_code == 503
    assert _err(resp)["code"] == "BUSY"
    assert resp.headers["Retry-After"] == "1"


# --- auth -------------------------------------------------------------------------------


def test_auth_required_rejects_missing_and_wrong_token(make_client, image_b64, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("s3cr3t-token\n")
    s = Settings(env=Environment.TEST, require_auth=True, api_token_file=token_file)
    client = make_client(settings_=s)
    assert client.post(CHECK, json={"image_base64": image_b64}).status_code == 401
    bad = client.post(
        CHECK, json={"image_base64": image_b64}, headers={"Authorization": "Bearer nope"}
    )
    assert _err(bad)["code"] == "UNAUTHORIZED"
    ok = client.post(
        CHECK, json={"image_base64": image_b64}, headers={"Authorization": "Bearer s3cr3t-token"}
    )
    assert ok.status_code == 200
    assert client.get("/v1/capabilities").status_code == 401
    assert client.get("/healthz").status_code == 200


def test_auth_required_without_token_fails_closed(make_client, image_b64):
    client = make_client(settings_=Settings(env=Environment.TEST, require_auth=True))
    resp = client.post(
        CHECK, json={"image_base64": image_b64}, headers={"Authorization": "Bearer "}
    )
    assert resp.status_code == 401
    ready = client.get("/readyz")
    assert ready.status_code == 503
    assert ready.json()["checks"]["auth"].startswith("fail")
