from __future__ import annotations

from face_liveness import config
from face_liveness.config import Environment, Settings
from face_liveness.runtime import ModelRuntime


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz_ready(client):
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "checks": {"model": "ok", "fusion_policy": "ok"}}


def test_readyz_not_ready_when_model_unavailable(make_client):
    client = make_client(runtime=ModelRuntime.unavailable("model artifact missing: detector"))
    resp = client.get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert "model artifact missing" in body["checks"]["model"]
    # healthz stays up so the orchestrator does not restart-loop on a config problem.
    assert client.get("/healthz").status_code == 200


def test_capabilities(client):
    body = client.get("/v1/capabilities").json()
    assert body["ready"] is True
    assert [m["id"] for m in body["methods"]] == ["passive_single_image"]
    assert "face_matching" in body["unsupported"]
    assert body["face_matching"] is False
    assert body["image_persistence"] == "none"
    assert body["decision"]["threshold"] == 0.85
    assert body["decision"]["threshold_calibrated"] is False
    assert body["input"]["faces_required"] == 1
    assert body["model"]["artifacts"][0]["sha256"] == config.YUNET_SHA256
    assert body["model"]["version"] == "1.0.0"
    assert len(body["model"]["manifest_sha256"]) == 64
    # Challenges exist as a contract, but no active liveness model is installed.
    assert body["active_liveness"] is False
    assert body["passive_liveness"] is True
    assert body["challenge"]["active_liveness"] is False
    assert body["challenge"]["single_use"] is True
    assert "active_challenge" in body["unsupported"]
    assert body["limits"]["max_queued_checks"] == 8
    assert body["input"]["max_image_side_px"] == 8192


def test_capabilities_reports_calibration(make_client):
    s = Settings(env=Environment.TEST, live_threshold=0.9, threshold_calibration_id="cal-2026-09")
    body = make_client(settings_=s).get("/v1/capabilities").json()
    assert body["decision"] == {
        "score": body["decision"]["score"],
        "threshold": 0.9,
        "threshold_calibrated": True,
        "calibration_id": "cal-2026-09",
    }


def test_status(make_client):
    body = make_client(runtime=ModelRuntime.unavailable("boom")).get("/v1/liveness/status").json()
    assert body["ready"] is False
    assert "boom" in body["reason"]
    assert body["environment"] == "test"


def test_unknown_route_uses_error_envelope(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"
    resp = client.get("/v1/liveness/check")
    assert resp.status_code == 405


def test_metrics(client, classifier, image_b64):
    client.post("/v1/liveness/check", json={"image_base64": image_b64})
    classifier.scores = (0.1, 0.2)
    client.post("/v1/liveness/check", json={"image_base64": image_b64})
    client.post("/v1/liveness/check", json={"image_base64": "!!"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    text = resp.text
    assert 'liveness_checks_total{outcome="live"} 1.0' in text
    assert 'liveness_checks_total{outcome="spoof"} 1.0' in text
    assert 'liveness_checks_total{outcome="INVALID_REQUEST"} 1.0' in text
    assert "liveness_model_ready 1.0" in text
    assert "liveness_threshold 0.85" in text
    http = 'liveness_http_requests_total{method="POST",route="/v1/liveness/check",status="200"}'
    assert f"{http} 2.0" in text
    assert "liveness_live_score_bucket" in text
    assert "liveness_inference_duration_seconds_bucket" in text


def test_metrics_model_not_ready(make_client):
    text = make_client(runtime=ModelRuntime.unavailable("x")).get("/metrics").text
    assert "liveness_model_ready 0.0" in text


def test_structured_access_log_has_no_image_data(client, image_b64, capsys):
    import logging

    from face_liveness.logging_config import configure_logging

    configure_logging("INFO")
    try:
        client.post(
            "/v1/liveness/check",
            json={"image_base64": image_b64},
            headers={"X-Request-ID": "log-1"},
        )
        out = capsys.readouterr().out
    finally:
        logging.getLogger().handlers.clear()
    import json

    records = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    access = [r for r in records if r["msg"] == "request completed"]
    assert access, out
    rec = access[-1]
    assert rec["request_id"] == "log-1"
    assert rec["route"] == "/v1/liveness/check"
    assert rec["status"] == 200
    assert rec["outcome"] == "live"
    assert image_b64[:40] not in out
