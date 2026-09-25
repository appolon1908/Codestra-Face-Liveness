"""Challenge issue/verify contract: short-lived, opaque, single-use; never active liveness."""

from __future__ import annotations

from datetime import datetime

import pytest

from face_liveness.challenges import ChallengeStore
from face_liveness.config import Environment, Settings
from face_liveness.errors import ErrorCode, LivenessError

ISSUE = "/v1/liveness/challenges"


def verify_url(challenge_id: str) -> str:
    return f"/v1/liveness/challenges/{challenge_id}/verify"


def _issue(client) -> str:  # type: ignore[no-untyped-def]
    resp = client.post(ISSUE)
    assert resp.status_code == 201, resp.text
    return resp.json()["challenge_id"]


def test_issue_contract(client):
    resp = client.post(ISSUE)
    assert resp.status_code == 201
    body = resp.json()
    assert body["challenge_id"].startswith("chl_")
    assert len(body["challenge_id"]) == 47
    assert body["challenge_type"] == "freshness_nonce"
    assert body["active_liveness"] is False
    assert body["instructions"] == []
    assert body["authoritative_method"] == "passive_single_image"
    assert body["ttl_seconds"] == 120
    issued = datetime.fromisoformat(body["issued_at"])
    expires = datetime.fromisoformat(body["expires_at"])
    assert (expires - issued).total_seconds() == 120


def test_ids_are_unique_and_opaque(client):
    ids = {_issue(client) for _ in range(50)}
    assert len(ids) == 50


def test_verify_runs_passive_check_and_reports_challenge(client, image_b64):
    cid = _issue(client)
    resp = client.post(verify_url(cid), json={"image_base64": image_b64})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["decision"] == "live"
    assert body["method"] == "passive_single_image"
    assert body["evidence"]["active_liveness_evaluated"] is False
    assert body["evidence"]["challenge"] == {
        "challenge_id": cid,
        "status": "consumed",
        "active_liveness_evaluated": False,
    }


def test_verify_spoof_is_still_spoof(client, classifier, image_b64):
    classifier.scores = (0.1, 0.2)
    cid = _issue(client)
    body = client.post(verify_url(cid), json={"image_base64": image_b64}).json()
    assert body["decision"] == "spoof"
    assert body["evidence"]["challenge"]["status"] == "consumed"


def test_replay_rejected(client, classifier, image_b64):
    cid = _issue(client)
    assert client.post(verify_url(cid), json={"image_base64": image_b64}).status_code == 200
    resp = client.post(verify_url(cid), json={"image_base64": image_b64})
    assert resp.status_code == 409
    err = resp.json()["error"]
    assert err["code"] == "CHALLENGE_ALREADY_USED"
    assert err["retryable"] is False
    assert "decision" not in resp.text
    assert classifier.calls == 1


def test_failed_verify_still_consumes(client, detector, image_b64):
    detector.faces = []
    cid = _issue(client)
    first = client.post(verify_url(cid), json={"image_base64": image_b64})
    assert first.json()["error"]["code"] == "NO_FACE"
    second = client.post(verify_url(cid), json={"image_base64": image_b64})
    assert second.json()["error"]["code"] == "CHALLENGE_ALREADY_USED"


def test_invalid_body_does_not_consume(client, image_b64):
    cid = _issue(client)
    assert client.post(verify_url(cid), json={}).status_code == 400
    assert client.post(verify_url(cid), json={"image_base64": image_b64}).status_code == 200


@pytest.mark.parametrize(
    "cid",
    ["chl_" + "A" * 43, "not-a-challenge", "chl_short", "x" * 128],
)
def test_unknown_or_malformed_challenge(client, classifier, image_b64, cid):
    resp = client.post(verify_url(cid), json={"image_base64": image_b64})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "CHALLENGE_NOT_FOUND"
    assert classifier.calls == 0


def test_overlong_challenge_id_rejected(client, image_b64):
    resp = client.post(verify_url("x" * 500), json={"image_base64": image_b64})
    assert resp.status_code == 400


def test_expired_challenge(client, image_b64):
    store: ChallengeStore = client.app.state.challenges
    now = [1000.0]
    store.clock = lambda: now[0]
    cid = _issue(client)
    now[0] += 121
    resp = client.post(verify_url(cid), json={"image_base64": image_b64})
    assert resp.status_code == 410
    assert resp.json()["error"]["code"] == "CHALLENGE_EXPIRED"


def test_capacity_bound(make_client):
    client = make_client(settings_=Settings(env=Environment.TEST, max_outstanding_challenges=2))
    _issue(client)
    _issue(client)
    resp = client.post(ISSUE)
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "BUSY"


def test_challenge_endpoints_require_auth(make_client, image_b64, tmp_path):
    token = tmp_path / "t"
    token.write_text("tok")
    client = make_client(
        settings_=Settings(env=Environment.TEST, require_auth=True, api_token_file=token)
    )
    assert client.post(ISSUE).status_code == 401
    auth = {"Authorization": "Bearer tok"}
    cid = client.post(ISSUE, headers=auth).json()["challenge_id"]
    assert client.post(verify_url(cid), json={"image_base64": image_b64}).status_code == 401
    ok = client.post(verify_url(cid), json={"image_base64": image_b64}, headers=auth)
    assert ok.status_code == 200


def test_challenge_metrics(client, image_b64):
    cid = _issue(client)
    client.post(verify_url(cid), json={"image_base64": image_b64})
    client.post(verify_url(cid), json={"image_base64": image_b64})
    text = client.get("/metrics").text
    assert 'liveness_challenges_total{event="issued"} 1.0' in text
    assert 'liveness_challenges_total{event="consumed"} 1.0' in text
    assert 'liveness_challenges_total{event="CHALLENGE_ALREADY_USED"} 1.0' in text
    assert "liveness_challenges_outstanding 1.0" in text


# --- store unit tests -------------------------------------------------------------------


def _store(now: list[float], capacity: int = 10) -> ChallengeStore:
    return ChallengeStore(ttl_seconds=10, capacity=capacity, clock=lambda: now[0])


def test_store_purges_expired_and_frees_capacity():
    now = [0.0]
    store = _store(now, capacity=2)
    store.issue()
    store.issue()
    with pytest.raises(LivenessError):
        store.issue()
    now[0] = 10.0
    assert store.outstanding() == 0
    store.issue()


def test_store_consumed_remembered_until_expiry_then_forgotten():
    now = [0.0]
    store = _store(now)
    cid = store.issue().challenge_id
    store.consume(cid)
    now[0] = 5.0
    with pytest.raises(LivenessError) as exc:
        store.consume(cid)
    assert exc.value.code is ErrorCode.CHALLENGE_ALREADY_USED
    now[0] = 11.0
    store.issue()  # purges the expired entry
    with pytest.raises(LivenessError) as exc:
        store.consume(cid)
    assert exc.value.code is ErrorCode.CHALLENGE_NOT_FOUND


def test_store_holds_only_hashes():
    now = [0.0]
    store = _store(now)
    cid = store.issue().challenge_id
    assert all(cid.encode() not in key for key in store._entries)
