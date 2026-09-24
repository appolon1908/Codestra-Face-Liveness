"""openapi.yaml is the published contract; it must match the implementation exactly."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from face_liveness.api import create_app
from face_liveness.config import Environment, Settings
from face_liveness.runtime import ModelRuntime

SPEC = Path(__file__).resolve().parent.parent / "openapi.yaml"


def test_openapi_yaml_matches_implementation():
    app = create_app(Settings(env=Environment.TEST), ModelRuntime.unavailable("contract test"))
    generated = json.loads(json.dumps(app.openapi()))
    committed = yaml.safe_load(SPEC.read_text())
    assert committed == generated, "openapi.yaml is stale: run `make openapi`"


def test_contract_covers_required_endpoints():
    paths = yaml.safe_load(SPEC.read_text())["paths"]
    for path in ("/healthz", "/readyz", "/metrics", "/v1/capabilities", "/v1/liveness/check"):
        assert path in paths
    check = paths["/v1/liveness/check"]["post"]
    for status in ("200", "400", "401", "413", "415", "422", "500", "503"):
        assert status in check["responses"], status
