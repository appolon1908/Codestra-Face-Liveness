"""Safe model promotion (mission 8): installed -> candidate -> active, local and fail closed."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from face_liveness import promotion
from face_liveness.calibration import build_report
from face_liveness.config import Environment, Settings
from face_liveness.inference import ComponentScore
from face_liveness.registry import EntryStatus, ModelRegistry, build_registry
from face_liveness.runtime import ModelRuntime, load_runtime
from face_liveness.validation_record import ValidationRecord, load_record, record_path

from .conftest import FakeClassifier, FakeDetector
from .test_calibration import ATTACKS, BONA, POLICY, _samples
from .test_registry import FAKE, _install, _manifest
from .test_runtime import REAL_MODEL_DIR, have_models

VERSION = "2.0.0"
NOW = datetime(2026, 9, 24, 12, tzinfo=UTC)


def _settings(model_dir: Path, **kw: object) -> Settings:
    return Settings(env=Environment.TEST, model_dir=model_dir, **kw)  # type: ignore[arg-type]


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    (d / "registry").mkdir(parents=True)
    _install(d, "next.json", _manifest(VERSION, FAKE))
    return d


@pytest.fixture
def fake_smoke(monkeypatch):  # type: ignore[no-untyped-def]
    """Smoke-test the candidate with fake inference (its artifacts are not real ONNX)."""
    classifier = FakeClassifier(scores=(0.4,))

    def load(settings: Settings) -> ModelRuntime:
        registry = build_registry(settings)
        entry = registry.get(settings.active_model_version)
        assert entry is not None and entry.manifest is not None
        return ModelRuntime(
            detector=FakeDetector(),
            classifier=classifier,
            manifest=entry.manifest,
            registry=ModelRegistry.single(entry.manifest),
        )

    monkeypatch.setattr(promotion, "load_runtime", load)
    return classifier


def _report(model_dir: Path, settings: Settings, path: Path, **policy: object) -> Path:
    manifest = build_registry(_settings(model_dir)).get(VERSION).manifest  # type: ignore[union-attr]
    assert manifest is not None
    pol = POLICY.model_copy(update=policy) if policy else POLICY
    report = build_report(_samples(BONA, ATTACKS), manifest, settings, pol, NOW)
    path.write_text(report.model_dump_json())
    return path


def _validate(model_dir: Path, tmp_path: Path, **settings_kw: object) -> ValidationRecord:
    s = _settings(model_dir, **settings_kw)
    report = _report(model_dir, _settings(model_dir), tmp_path / "report.json")
    record = promotion.validate_version(s, VERSION, report, NOW)
    path = record_path(s.resolved_model_registry_dir, VERSION)
    path.parent.mkdir(exist_ok=True)
    path.write_text(record.model_dump_json())
    return record


# --- validation checks ------------------------------------------------------------------


def test_validation_passes_and_makes_candidate(model_dir, tmp_path, fake_smoke):
    record = _validate(model_dir, tmp_path)
    assert record.passed, record
    assert record.digests.passed and record.digests.artifacts_verified == 2
    assert record.smoke.passed and record.smoke.deterministic and record.smoke.images == 3
    assert record.calibration.passed
    assert record.calibration.calibration_id is not None
    assert record.activation.auto_activated is False
    env = record.activation.env
    assert env["LIVENESS_ACTIVE_MODEL_VERSION"] == VERSION
    assert env["LIVENESS_THRESHOLD_CALIBRATION_ID"] == record.calibration.calibration_id
    assert float(env["LIVENESS_LIVE_THRESHOLD"]) == record.calibration.threshold

    entry = build_registry(_settings(model_dir)).get(VERSION)
    assert entry is not None
    assert entry.status is EntryStatus.CANDIDATE
    assert entry.validation_error is None


def test_installed_without_record(model_dir):
    entry = build_registry(_settings(model_dir)).get(VERSION)
    assert entry is not None
    assert entry.status is EntryStatus.INSTALLED
    assert entry.validation_error == "no validation record"


def test_report_for_another_manifest_fails(model_dir, tmp_path, fake_smoke):
    s = _settings(model_dir)
    other = build_report(
        _samples(BONA, ATTACKS),
        build_registry(s).entries[0].manifest,  # built-in, not the candidate
        s,
        POLICY,
        NOW,
    )
    path = tmp_path / "other.json"
    path.write_text(other.model_dump_json())
    record = promotion.validate_version(s, VERSION, path, NOW)
    assert not record.passed
    assert "different model version or manifest" in record.calibration.detail
    assert record.activation.env == {}


def test_report_with_other_pipeline_settings_fails(model_dir, tmp_path, fake_smoke):
    record = _validate(model_dir, tmp_path, min_face_size_px=96)
    assert not record.calibration.passed
    assert "pipeline settings differ" in record.calibration.detail


def test_ineligible_report_fails(model_dir, tmp_path, fake_smoke):
    s = _settings(model_dir)
    path = _report(model_dir, s, tmp_path / "r.json", min_bona_fide=10_000)
    record = promotion.validate_version(s, VERSION, path, NOW)
    assert not record.calibration.passed
    assert "sample size below policy minimums" in record.calibration.detail


def test_missing_or_garbage_report_fails(model_dir, tmp_path, fake_smoke):
    s = _settings(model_dir)
    garbage = tmp_path / "g.json"
    garbage.write_text('{"auto_promoted": true}')
    for path, detail in ((None, "no calibration report"), (garbage, "does not match schema")):
        record = promotion.validate_version(s, VERSION, path, NOW)
        assert not record.passed
        assert detail in record.calibration.detail


def test_tampered_artifact_fails_digest_and_skips_smoke(model_dir, tmp_path, fake_smoke):
    (model_dir / "v2/cls.onnx").write_bytes(b"tampered")
    record = _validate(model_dir, tmp_path)
    assert not record.digests.passed
    assert "digest mismatch" in record.digests.detail
    assert record.smoke.detail == "skipped: digest check failed"


def test_smoke_rejects_bad_scores(model_dir, tmp_path, fake_smoke):
    fake_smoke.scores = (1.5,)
    record = _validate(model_dir, tmp_path)
    assert not record.smoke.passed
    assert "score out of range" in record.smoke.detail


def test_smoke_rejects_wrong_component_count(model_dir, tmp_path, fake_smoke):
    fake_smoke.scores = (0.5, 0.5)
    assert "expected 1 component" in _validate(model_dir, tmp_path).smoke.detail


def test_smoke_rejects_nondeterminism(model_dir, tmp_path, fake_smoke, monkeypatch):
    calls = iter(range(1000))

    def jitter(bgr, face):  # type: ignore[no-untyped-def]
        return [ComponentScore("c", 0.1 + next(calls) * 0.01)]

    monkeypatch.setattr(fake_smoke, "score", jitter)
    smoke = _validate(model_dir, tmp_path).smoke
    assert not smoke.passed and not smoke.deterministic


def test_record_for_other_manifest_does_not_make_candidate(model_dir, tmp_path, fake_smoke):
    _validate(model_dir, tmp_path)
    # Re-install the same version with different artifacts: the old record no longer applies.
    files = {**FAKE, "v2/cls.onnx": b"new"}
    _install(model_dir, "next.json", _manifest(VERSION, files), files)
    entry = build_registry(_settings(model_dir)).get(VERSION)
    assert entry is not None
    assert entry.status is EntryStatus.INSTALLED
    assert entry.validation_error == "validation record is for a different manifest"


def test_manifest_cannot_reference_a_remote_url(model_dir):
    doc = _manifest("3.0.0", FAKE)
    doc["detector"]["file"] = "https://models.example.com/det.onnx"  # type: ignore[index]
    _install(model_dir, "remote.json", doc)
    [entry] = [e for e in build_registry(_settings(model_dir)).entries if e.version is None]
    assert entry.status is EntryStatus.INVALID
    assert "detector.file" in (entry.error or "")


# --- activation gate (service side) ------------------------------------------------------


def test_production_always_requires_validation(tmp_path):
    s = Settings(env=Environment.PRODUCTION, require_model_validation=False, model_dir=tmp_path)
    assert s.model_validation_required is True
    assert _settings(tmp_path).model_validation_required is False


def test_unvalidated_version_cannot_be_active(model_dir):
    s = _settings(model_dir, active_model_version=VERSION, require_model_validation=True)
    reg = build_registry(s)
    assert reg.active is None
    assert reg.active_error == (
        "active model version not validated for activation: no validation record"
    )
    rt = load_runtime(s)
    assert not rt.ready and "not validated" in (rt.error or "")


def test_validated_version_activates_only_with_its_calibration(model_dir, tmp_path, fake_smoke):
    record = _validate(model_dir, tmp_path)
    env = record.activation.env
    threshold = float(env["LIVENESS_LIVE_THRESHOLD"])
    cal_id = env["LIVENESS_THRESHOLD_CALIBRATION_ID"]

    def registry(**kw: object) -> ModelRegistry:
        return build_registry(
            _settings(model_dir, active_model_version=VERSION, require_model_validation=True, **kw)
        )

    ok = registry(live_threshold=threshold, threshold_calibration_id=cal_id)
    assert ok.active is not None and ok.active.version == VERSION and ok.active_error is None

    stricter = registry(live_threshold=min(threshold + 0.05, 0.99), threshold_calibration_id=cal_id)
    assert stricter.active is not None

    for kw, reason in (
        ({"live_threshold": threshold}, "CALIBRATION_ID does not match"),
        (
            {"live_threshold": threshold, "threshold_calibration_id": "cal-other"},
            "CALIBRATION_ID does not match",
        ),
        (
            {"live_threshold": threshold - 0.05, "threshold_calibration_id": cal_id},
            "below the validated threshold",
        ),
    ):
        reg = registry(**kw)
        assert reg.active is None, kw
        assert reason in (reg.active_error or "")
        assert reg.get(VERSION).status is EntryStatus.CANDIDATE  # type: ignore[union-attr]

    (model_dir / "v2/det.onnx").write_bytes(b"swapped after validation")
    tampered = registry(live_threshold=threshold, threshold_calibration_id=cal_id)
    assert tampered.active is None
    assert "digest mismatch" in (tampered.active_error or "")


@pytest.mark.skipif(not have_models, reason="real model artifacts not present")
def test_builtin_version_is_exempt_from_validation():
    s = _settings(REAL_MODEL_DIR, require_model_validation=True)
    reg = build_registry(s)
    assert reg.active is not None and reg.active.source == "builtin"


def test_models_api_reports_validation(make_client, model_dir, tmp_path, fake_smoke):
    record = _validate(model_dir, tmp_path)
    rt = ModelRuntime.unavailable("x", build_registry(_settings(model_dir)))
    body = make_client(runtime=rt).get(f"/v1/models/{VERSION}").json()
    assert body["status"] == "candidate"
    assert body["validation"]["passed"] is True
    assert body["validation"]["calibration_id"] == record.calibration.calibration_id
    assert body["validation_error"] is None


# --- CLI ----------------------------------------------------------------------------------


def test_cli_validate_then_activate(model_dir, tmp_path, fake_smoke, capsys):
    report = _report(model_dir, _settings(model_dir), tmp_path / "report.json")
    env_before = dict(os.environ)
    base = ["--model-dir", str(model_dir)]

    assert promotion.main([*base, "activate", VERSION]) == 1
    assert "no validation record" in capsys.readouterr().err

    assert promotion.main([*base, "validate", VERSION, "--calibration-report", str(report)]) == 0
    record = load_record(record_path(model_dir / "registry", VERSION))
    assert record.passed

    capsys.readouterr()
    assert promotion.main([*base, "list"]) == 0
    states = {e["version"]: e["status"] for e in json.loads(capsys.readouterr().out)}
    assert states == {"1.0.0": "invalid", VERSION: "candidate"}

    env_file = tmp_path / "deploy.env"
    env_file.write_text("OTHER=keep\nLIVENESS_LIVE_THRESHOLD=0.85\n")
    assert promotion.main([*base, "activate", VERSION, "--env-file", str(env_file)]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed == [f"{k}={v}" for k, v in record.activation.env.items()]
    lines = env_file.read_text().splitlines()
    assert lines[0] == "OTHER=keep"
    assert set(lines[1:]) == set(printed)
    # Local action only: the process environment is untouched.
    assert dict(os.environ) == env_before


def test_cli_validate_failure_exit_code(model_dir, tmp_path, fake_smoke, capsys):
    garbage = tmp_path / "g.json"
    garbage.write_text("{}")
    code = promotion.main(
        ["--model-dir", str(model_dir), "validate", VERSION, "--calibration-report", str(garbage)]
    )
    assert code == 1
    assert "NOT a candidate: calibration" in capsys.readouterr().err
    assert promotion.main(["--model-dir", str(model_dir), "activate", VERSION]) == 1


@pytest.mark.parametrize("version", ["https://evil.example/m.json", "../1.0.0", "a b"])
def test_cli_rejects_urls_and_paths(model_dir, version):
    with pytest.raises(SystemExit) as exc:
        promotion.main(["--model-dir", str(model_dir), "activate", version])
    assert exc.value.code == 2


@pytest.mark.models
@pytest.mark.skipif(not have_models, reason="real model artifacts not present")
def test_real_models_pass_smoke():
    rt = load_runtime(_settings(REAL_MODEL_DIR))
    assert rt.ready, rt.error
    smoke = promotion.check_smoke(rt)
    assert smoke.passed, smoke.detail
    assert smoke.components == 2 and smoke.deterministic


def test_smoke_on_unready_runtime():
    smoke = promotion.check_smoke(ModelRuntime.unavailable("nope"))
    assert not smoke.passed and "nope" in smoke.detail
