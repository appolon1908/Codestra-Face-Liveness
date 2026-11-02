"""Offline calibration reports: metrics, fingerprint, schema, and no auto-promotion."""

from __future__ import annotations

import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from face_liveness import calibration
from face_liveness.calibration import (
    CalibrationPolicy,
    CalibrationReport,
    Label,
    ScoredSample,
    build_report,
    dataset_fingerprint,
)
from face_liveness.config import Environment, Settings
from face_liveness.contracts import json_schemas
from face_liveness.inference import ComponentScore, FaceBox
from face_liveness.registry import ModelRegistry, builtin_manifest
from face_liveness.runtime import ModelRuntime

from .conftest import FakeDetector

SETTINGS = Settings(env=Environment.TEST)
MANIFEST = builtin_manifest(SETTINGS)
POLICY = CalibrationPolicy(
    target_apcer=0.1, min_threshold=0.5, threshold_step=0.01, min_bona_fide=5, min_per_species=5
)


def _samples(bona: list[float], attacks: dict[str, list[float]]) -> list[ScoredSample]:
    out = [ScoredSample(Label.BONA_FIDE, None, f"b{i}", s) for i, s in enumerate(bona)]
    for sp, scores in attacks.items():
        out += [ScoredSample(Label.ATTACK, sp, f"{sp}{i}", s) for i, s in enumerate(scores)]
    return out


BONA = [0.95, 0.9, 0.85, 0.8, 0.7, 0.99, 0.97, 0.92, 0.88, 0.6]
ATTACKS = {
    "print": [0.1, 0.2, 0.3, 0.4, 0.5, 0.05, 0.15, 0.25, 0.35, 0.72],
    "replay": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.05, 0.02, 0.01],
}


def test_metrics_and_candidate():
    report = build_report(_samples(BONA, ATTACKS), MANIFEST, SETTINGS, POLICY)
    cand = report.threshold_candidate
    assert cand is not None
    # Worst species must be <= 10%: print has one attack at 0.72, so t=0.6 leaves
    # print at 0.1 (only 0.72 >= t) and replay at 0.2 (0.6, 0.65) -> first fit is 0.61.
    assert cand.threshold == 0.61
    assert cand.apcer_by_species == {"print": 0.1, "replay": 0.1}
    assert cand.apcer_max == 0.1
    assert cand.bpcer == 0.1  # 0.6 is rejected
    points = {op.threshold: op for op in report.operating_points}
    assert points[0.85].bpcer == 0.3  # 0.8, 0.7, 0.6 < 0.85
    assert points[0.85].apcer_max == 0.0
    assert 0.0 <= report.equal_error_rate.threshold < 1.0
    assert report.score_summary[Label.BONA_FIDE].count == 10
    assert report.dataset.attack_scored == {"print": 10, "replay": 10}
    assert report.sufficient_sample_size is True
    assert report.promotion.eligible_for_review is True
    assert report.promotion.auto_promoted is False
    assert report.model.version == "1.0.0"
    assert report.model.manifest_sha256 == MANIFEST.digest
    assert report.calibration_id.endswith(MANIFEST.digest[:8])


def test_candidate_respects_min_threshold():
    bona = [0.9] * 10
    attacks = {"print": [0.01] * 10}
    report = build_report(_samples(bona, attacks), MANIFEST, SETTINGS, POLICY)
    assert report.threshold_candidate is not None
    assert report.threshold_candidate.threshold == 0.5


def test_no_candidate_blocks_review():
    attacks = {"print": [0.999] * 10}
    report = build_report(_samples(BONA, attacks), MANIFEST, SETTINGS, POLICY)
    assert report.threshold_candidate is None
    assert report.promotion.eligible_for_review is False
    assert "no threshold meets the APCER target" in report.promotion.blockers
    assert report.promotion.auto_promoted is False


def test_insufficient_samples_block_review():
    report = build_report(_samples(BONA[:3], {"print": [0.1]}), MANIFEST, SETTINGS, POLICY)
    assert report.sufficient_sample_size is False
    assert "sample size below policy minimums" in report.promotion.blockers


def test_rejections_counted_separately():
    samples = [
        *_samples(BONA, ATTACKS),
        ScoredSample(Label.BONA_FIDE, None, "r1", None, "NO_FACE"),
        ScoredSample(Label.ATTACK, "print", "r2", None, "MULTIPLE_FACES"),
    ]
    report = build_report(samples, MANIFEST, SETTINGS, POLICY)
    assert report.dataset.images_total == 32
    assert report.dataset.bona_fide_scored == 10
    assert report.dataset.rejected_before_scoring == {
        "bona_fide:NO_FACE": 1,
        "print:MULTIPLE_FACES": 1,
    }


def test_needs_both_classes():
    with pytest.raises(ValueError, match="at least one"):
        build_report(_samples(BONA, {}), MANIFEST, SETTINGS, POLICY)


def test_fingerprint_order_independent_and_label_sensitive():
    samples = _samples(BONA, ATTACKS)
    assert dataset_fingerprint(samples) == dataset_fingerprint(list(reversed(samples)))
    relabelled = [
        ScoredSample(Label.ATTACK, "print", s.sha256, s.score) if i == 0 else s
        for i, s in enumerate(samples)
    ]
    assert dataset_fingerprint(relabelled) != dataset_fingerprint(samples)


def test_report_roundtrips_and_schema_forbids_auto_promotion():
    created = datetime(2026, 9, 24, tzinfo=UTC)
    report = build_report(_samples(BONA, ATTACKS), MANIFEST, SETTINGS, POLICY, created)
    again = CalibrationReport.model_validate_json(report.model_dump_json())
    assert again == report
    tampered = json.loads(report.model_dump_json())
    tampered["promotion"]["auto_promoted"] = True
    with pytest.raises(ValueError):
        CalibrationReport.model_validate(tampered)
    schema = json_schemas()["calibration-report.v1.schema.json"]
    assert schema["$defs"]["Promotion"]["properties"]["auto_promoted"]["const"] is False


# --- CLI end to end (production pipeline, fake model) -------------------------------------


class BrightnessClassifier:
    """Live score = mean brightness; lets the test control scores through pixel values."""

    @property
    def component_names(self) -> list[str]:
        return ["brightness"]

    def score(self, bgr: np.ndarray, face: FaceBox) -> list[ComponentScore]:
        return [ComponentScore("brightness", float(bgr.mean()) / 255.0)]


def _write(path: Path, value: int, size: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (value, value, value)).save(buf, format="PNG")
    path.write_bytes(buf.getvalue())


def test_cli_end_to_end_never_touches_configuration(tmp_path, monkeypatch, capsys):
    data = tmp_path / "calib"
    for i in range(6):
        _write(data / "bona_fide" / f"b{i}.png", 240 - i)
        _write(data / "attack" / "print" / f"p{i}.png", 20 + i)
        _write(data / "attack" / "replay" / f"r{i}.png", 40 + i)
    _write(data / "bona_fide" / "tiny.png", 250, size=50)  # rejected: IMAGE_TOO_SMALL
    (data / "bona_fide" / "notes.txt").write_text("ignored")

    manifest = builtin_manifest(Settings(env=Environment.TEST))
    runtime = ModelRuntime(
        detector=FakeDetector(),
        classifier=BrightnessClassifier(),
        manifest=manifest,
        registry=ModelRegistry.single(manifest),
    )
    monkeypatch.setattr(calibration, "load_runtime", lambda settings: runtime)
    env_before = dict(os.environ)
    before = {p for p in tmp_path.rglob("*")}
    out = tmp_path / "report.json"

    code = calibration.main(
        [
            "--data",
            str(data),
            "--out",
            str(out),
            "--min-bona-fide",
            "5",
            "--min-per-species",
            "5",
        ]
    )
    assert code == 0, capsys.readouterr().err
    report = CalibrationReport.model_validate_json(out.read_text())
    assert report.dataset.bona_fide_scored == 6
    assert report.dataset.attack_scored == {"print": 6, "replay": 6}
    assert report.dataset.rejected_before_scoring == {"bona_fide:IMAGE_TOO_SMALL": 1}
    assert report.threshold_candidate is not None
    assert report.threshold_candidate.apcer_max == 0.0
    assert report.promotion.auto_promoted is False
    # Only the report was written; no env or config was changed.
    assert {p for p in tmp_path.rglob("*")} - before == {out}
    assert dict(os.environ) == env_before


def test_cli_exit_codes(tmp_path, monkeypatch):
    assert calibration.main(["--data", str(tmp_path), "--target-apcer", "2"]) == 2
    monkeypatch.setattr(
        calibration, "load_runtime", lambda settings: ModelRuntime.unavailable("no models")
    )
    assert calibration.main(["--data", str(tmp_path)]) == 2
