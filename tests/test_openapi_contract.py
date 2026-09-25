"""openapi.yaml and schemas/*.json are published contracts; they must match exactly."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from face_liveness.api import create_app
from face_liveness.config import Environment, Settings
from face_liveness.contracts import SCHEMA_DIR, json_schemas
from face_liveness.runtime import ModelRuntime

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "openapi.yaml"


def test_openapi_yaml_matches_implementation():
    app = create_app(Settings(env=Environment.TEST), ModelRuntime.unavailable("contract test"))
    generated = json.loads(json.dumps(app.openapi()))
    committed = yaml.safe_load(SPEC.read_text())
    assert committed == generated, "openapi.yaml is stale: run `make openapi`"


def test_json_schemas_match_implementation():
    schemas = json_schemas()
    assert set(schemas) == {p.name for p in (ROOT / SCHEMA_DIR).glob("*.json")}
    for name, schema in schemas.items():
        committed = json.loads((ROOT / SCHEMA_DIR / name).read_text())
        assert committed == schema, f"{SCHEMA_DIR}/{name} is stale: run `make openapi`"


def test_contract_covers_required_endpoints():
    paths = yaml.safe_load(SPEC.read_text())["paths"]
    for path in (
        "/healthz",
        "/readyz",
        "/metrics",
        "/v1/capabilities",
        "/v1/liveness/check",
        "/v1/liveness/challenges",
        "/v1/liveness/challenges/{challenge_id}/verify",
        "/v1/models",
        "/v1/models/{version}",
    ):
        assert path in paths, path
    check = paths["/v1/liveness/check"]["post"]
    for status in ("200", "400", "401", "413", "415", "422", "500", "503"):
        assert status in check["responses"], status
    verify = paths["/v1/liveness/challenges/{challenge_id}/verify"]["post"]
    for status in ("200", "404", "409", "410", "503"):
        assert status in verify["responses"], status
    assert set(paths["/v1/models"]) == {"get"}
    assert set(paths["/v1/models/{version}"]) == {"get"}


def test_contract_never_claims_active_liveness():
    schemas = yaml.safe_load(SPEC.read_text())["components"]["schemas"]
    assert schemas["ChallengeIssueResponse"]["properties"]["active_liveness"]["const"] is False
    assert schemas["ChallengePolicy"]["properties"]["active_liveness"]["const"] is False
    evidence = schemas["DecisionEvidence"]["properties"]
    assert evidence["active_liveness_evaluated"]["const"] is False
    assert evidence["liveness_type"]["const"] == "passive"
