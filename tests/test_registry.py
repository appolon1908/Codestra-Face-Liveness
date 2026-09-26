"""Model registry: manifest validation, multiple installed versions, one active, fail closed."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from face_liveness.config import Environment, Settings
from face_liveness.registry import (
    MAX_MANIFEST_BYTES,
    EntryStatus,
    ModelRegistry,
    build_registry,
    builtin_manifest,
    resolve_artifact,
)
from face_liveness.runtime import ModelRuntime, load_runtime

from .test_runtime import FILES, REAL_MODEL_DIR, have_models


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(version: str, files: dict[str, bytes], **overrides: object) -> dict[str, object]:
    det, *cls = files
    doc: dict[str, object] = {
        "schema_version": 1,
        "model_id": "test-model",
        "version": version,
        "liveness_type": "passive",
        "score_aggregation": "mean_live_probability",
        "detector": {
            "name": "det",
            "architecture": "yunet",
            "file": det,
            "sha256": _sha(files[det]),
        },
        "classifiers": [
            {
                "name": f"cls{i}",
                "architecture": "minifasnet",
                "file": f,
                "sha256": _sha(files[f]),
                "crop_scale": 2.7,
            }
            for i, f in enumerate(cls)
        ],
        "license": "Apache-2.0",
    }
    doc.update(overrides)
    return doc


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    (d / "registry").mkdir(parents=True)
    (d / "v2").mkdir()
    return d


FAKE = {"v2/det.onnx": b"detector-bytes", "v2/cls.onnx": b"classifier-bytes"}


def _install(model_dir: Path, name: str, doc: dict[str, object], files=FAKE) -> None:  # type: ignore[no-untyped-def]
    for f, data in files.items():
        (model_dir / f).parent.mkdir(parents=True, exist_ok=True)
        (model_dir / f).write_bytes(data)
    (model_dir / "registry" / name).write_text(json.dumps(doc))


def _settings(model_dir: Path, **kw: object) -> Settings:
    return Settings(env=Environment.TEST, model_dir=model_dir, **kw)  # type: ignore[arg-type]


def test_builtin_is_default_active_version(model_dir):
    reg = build_registry(_settings(model_dir))
    assert reg.active_version == "1.0.0"
    [builtin] = reg.entries
    assert builtin.source == "builtin"
    # No artifacts on disk: invalid, and the service fails closed.
    assert builtin.status is EntryStatus.INVALID
    assert "missing" in (reg.active_error or "")


def test_multiple_installed_one_active(model_dir):
    _install(model_dir, "a.json", _manifest("2.0.0", FAKE))
    _install(model_dir, "b.json", _manifest("2.1.0", FAKE))
    reg = build_registry(_settings(model_dir, active_model_version="2.1.0"))
    by_version = {e.version: e for e in reg.entries}
    assert by_version["2.1.0"].status is EntryStatus.ACTIVE
    assert by_version["2.0.0"].status is EntryStatus.INSTALLED
    assert by_version["2.0.0"].digests_verified is True
    assert [e.version for e in reg.entries if e.status is EntryStatus.ACTIVE] == ["2.1.0"]
    assert reg.active is by_version["2.1.0"]
    assert reg.active_error is None


def test_active_digest_mismatch_fails_closed(model_dir):
    doc = _manifest("2.0.0", FAKE)
    _install(model_dir, "a.json", doc)
    (model_dir / "v2/cls.onnx").write_bytes(b"tampered")
    rt = load_runtime(_settings(model_dir, active_model_version="2.0.0"))
    assert not rt.ready
    assert "digest mismatch" in (rt.error or "")
    assert rt.registry.get("2.0.0").status is EntryStatus.INVALID  # type: ignore[union-attr]


def test_inactive_invalid_version_reported_but_not_fatal(model_dir):
    _install(model_dir, "good.json", _manifest("2.0.0", FAKE))
    bad = _manifest("3.0.0", FAKE)
    bad["detector"] = {**bad["detector"], "sha256": "0" * 64}  # type: ignore[dict-item]
    _install(model_dir, "bad.json", bad)
    reg = build_registry(_settings(model_dir, active_model_version="2.0.0"))
    assert reg.active is not None and reg.active.version == "2.0.0"
    assert reg.get("3.0.0").status is EntryStatus.INVALID  # type: ignore[union-attr]


def test_active_version_not_installed(model_dir):
    rt = load_runtime(_settings(model_dir, active_model_version="9.9.9"))
    assert not rt.ready
    assert rt.error == "active model version not installed: 9.9.9"


def test_duplicate_versions_are_ambiguous(model_dir):
    _install(model_dir, "a.json", _manifest("2.0.0", FAKE))
    _install(model_dir, "b.json", _manifest("2.0.0", FAKE, model_id="other"))
    reg = build_registry(_settings(model_dir, active_model_version="2.0.0"))
    assert reg.active is None
    assert "duplicate model version" in (reg.active_error or "")


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda d: d.update(liveness_type="active"), "liveness_type"),
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d.update(unexpected="x"), "unexpected"),
        (lambda d: d.update(classifiers=[]), "classifiers"),
        (lambda d: d["detector"].update(file="../../etc/passwd"), "detector.file"),
        (lambda d: d["detector"].update(file="/abs/path.onnx"), "detector.file"),
        (lambda d: d["detector"].update(sha256="XYZ"), "detector.sha256"),
        (lambda d: d["classifiers"][0].update(crop_scale=100), "crop_scale"),
        (lambda d: d["classifiers"][0].update(name="det"), "manifest"),
        (lambda d: d.update(version="has space"), "version"),
    ],
)
def test_manifest_validation_rejects(model_dir, mutate, expected):
    doc = _manifest("2.0.0", FAKE)
    mutate(doc)
    _install(model_dir, "m.json", doc)
    reg = build_registry(_settings(model_dir))
    [entry] = [e for e in reg.entries if e.source == "registry/m.json"]
    assert entry.status is EntryStatus.INVALID
    assert entry.manifest is None
    assert "manifest schema invalid" in (entry.error or "")
    assert expected in (entry.error or "")


def test_malformed_and_oversized_manifests(model_dir):
    (model_dir / "registry" / "junk.json").write_text("{not json")
    (model_dir / "registry" / "big.json").write_text(" " * (MAX_MANIFEST_BYTES + 1))
    reg = build_registry(_settings(model_dir))
    errors = {e.source: e.error for e in reg.entries}
    assert errors["registry/junk.json"] == "manifest is not valid JSON"
    assert "exceeds" in (errors["registry/big.json"] or "")


def test_symlink_escape_rejected(model_dir, tmp_path):
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"x")
    (model_dir / "link.onnx").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        resolve_artifact(model_dir, "link.onnx")


def test_unverified_digests_only_when_verification_disabled(model_dir):
    doc = _manifest("2.0.0", FAKE)
    _install(model_dir, "a.json", doc)
    (model_dir / "v2/cls.onnx").write_bytes(b"changed")
    reg = build_registry(
        _settings(model_dir, active_model_version="2.0.0", verify_model_digests=False)
    )
    assert reg.active is not None
    assert reg.active.digests_verified is False
    # Production forces verification back on.
    prod = Settings(env=Environment.PRODUCTION, verify_model_digests=False, model_dir=model_dir)
    assert prod.verify_model_digests is True


def test_manifest_digest_is_stable_and_content_sensitive():
    s = Settings(env=Environment.TEST)
    a, b = builtin_manifest(s), builtin_manifest(s)
    assert a.digest == b.digest
    c = builtin_manifest(Settings(env=Environment.TEST, detector_sha256="1" * 64))
    assert c.digest != a.digest


# --- API --------------------------------------------------------------------------------


def test_models_endpoints(client):
    body = client.get("/v1/models").json()
    assert body["active_version"] == "1.0.0"
    assert body["active_ready"] is True
    [m] = body["models"]
    assert m["status"] == "active"
    assert m["liveness_type"] == "passive"
    assert len(m["manifest_sha256"]) == 64
    assert client.get("/v1/models/1.0.0").json() == m
    missing = client.get("/v1/models/0.0.1")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "NOT_FOUND"


def test_models_endpoint_reports_fail_closed_registry(make_client, model_dir):
    _install(model_dir, "a.json", _manifest("2.0.0", FAKE))
    (model_dir / "v2/cls.onnx").write_bytes(b"tampered")
    rt = load_runtime(_settings(model_dir, active_model_version="2.0.0"))
    client = make_client(runtime=rt)
    body = client.get("/v1/models").json()
    assert body["active_ready"] is False
    assert "digest mismatch" in body["active_error"]
    statuses = {m["source"]: m["status"] for m in body["models"]}
    assert statuses == {"builtin": "invalid", "registry/a.json": "invalid"}
    assert client.get("/readyz").status_code == 503


def test_models_endpoints_are_read_only(client):
    for method in ("post", "put", "delete"):
        assert getattr(client, method)("/v1/models/1.0.0").status_code == 405


def test_runtime_without_manifest_is_not_ready(detector, classifier):
    assert not ModelRuntime(detector=detector, classifier=classifier).ready
    m = builtin_manifest(Settings(env=Environment.TEST))
    rt = ModelRuntime(
        detector=detector, classifier=classifier, manifest=m, registry=ModelRegistry.single(m)
    )
    assert rt.ready


@pytest.mark.models
@pytest.mark.skipif(not have_models, reason="real model artifacts not present")
def test_real_models_second_version_activated(tmp_path):
    for f in FILES:
        shutil.copy(REAL_MODEL_DIR / f, tmp_path / f)
    files = {f: (tmp_path / f).read_bytes() for f in FILES}
    doc = _manifest("2026.10.0", files)
    (tmp_path / "registry").mkdir()
    (tmp_path / "registry" / "next.json").write_text(json.dumps(doc))
    rt = load_runtime(_settings(tmp_path, active_model_version="2026.10.0"))
    assert rt.ready, rt.error
    assert rt.manifest is not None and rt.manifest.version == "2026.10.0"
    assert rt.registry.get("1.0.0").status is EntryStatus.INSTALLED  # type: ignore[union-attr]
