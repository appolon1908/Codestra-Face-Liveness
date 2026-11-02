from __future__ import annotations


def _body(image_b64: str) -> dict[str, str]:
    return {"image_base64": image_b64}


def test_evidence_ref_is_tenant_bound_and_reference_only(client, image_b64):
    response = client.post(
        "/v1/liveness/check",
        headers={"X-Tenant-ID": "tenant-a"},
        json=_body(image_b64),
    )
    assert response.status_code == 200, response.text
    evidence_ref = response.json()["evidence_ref"]
    assert evidence_ref.startswith("lev_")

    got = client.get(
        f"/v1/liveness/evidence/{evidence_ref}",
        headers={"X-Tenant-ID": "tenant-a"},
    )
    assert got.status_code == 200
    payload = got.json()
    assert payload["evidence_ref"] == evidence_ref
    assert payload["identity_assertion"] is False
    assert payload["image_persisted"] is False
    serialized = str(payload).lower()
    assert "image_base64" not in serialized
    assert "embedding" not in serialized
    assert "subject_id" not in serialized

    other = client.get(
        f"/v1/liveness/evidence/{evidence_ref}",
        headers={"X-Tenant-ID": "tenant-b"},
    )
    assert other.status_code == 404


def test_evidence_list_is_bounded_and_tenant_scoped(client, image_b64):
    for tenant in ("tenant-a", "tenant-b", "tenant-a"):
        response = client.post(
            "/v1/liveness/check",
            headers={"X-Tenant-ID": tenant},
            json=_body(image_b64),
        )
        assert response.status_code == 200, response.text

    page = client.get(
        "/v1/liveness/evidence?limit=1&offset=0",
        headers={"X-Tenant-ID": "tenant-a"},
    )
    assert page.status_code == 200
    payload = page.json()
    assert payload["limit"] == 1
    assert payload["returned"] == 1
    assert payload["total"] >= 2


def test_check_without_tenant_remains_backward_compatible(client, image_b64):
    response = client.post("/v1/liveness/check", json=_body(image_b64))
    assert response.status_code == 200
    assert response.json()["evidence_ref"] is None
