"""Model runtime loading: fail closed on missing, tampered, or broken artifacts."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pytest

from face_liveness.config import Environment, Settings
from face_liveness.inference import FaceBox, crop_box
from face_liveness.runtime import load_runtime

REAL_MODEL_DIR = Path(os.environ.get("LIVENESS_TEST_MODEL_DIR", "models"))
FILES = (
    "face_detection_yunet_2023mar.onnx",
    "minifasnet_v2_2.7_80x80.onnx",
    "minifasnet_v1se_4.0_80x80.onnx",
)
have_models = all((REAL_MODEL_DIR / f).is_file() for f in FILES)


def _settings(model_dir: Path, **kw: object) -> Settings:
    return Settings(env=Environment.TEST, model_dir=model_dir, **kw)  # type: ignore[arg-type]


def test_missing_artifacts_not_ready(tmp_path):
    rt = load_runtime(_settings(tmp_path))
    assert not rt.ready
    assert rt.error is not None and "missing" in rt.error


def test_digest_mismatch_not_ready(tmp_path):
    for f in FILES:
        (tmp_path / f).write_bytes(b"not a model")
    rt = load_runtime(_settings(tmp_path))
    assert not rt.ready
    assert rt.error is not None and "digest mismatch" in rt.error


def test_corrupt_model_not_ready_even_without_digest_check(tmp_path):
    for f in FILES:
        (tmp_path / f).write_bytes(b"not a model")
    rt = load_runtime(_settings(tmp_path, verify_model_digests=False))
    assert not rt.ready
    assert rt.error is not None and "failed to load" in rt.error


def test_production_forces_digest_verification(tmp_path):
    s = Settings(env=Environment.PRODUCTION, verify_model_digests=False, model_dir=tmp_path)
    assert s.verify_model_digests is True
    assert s.auth_required is True


def test_production_rejects_inline_token():
    with pytest.raises(ValueError, match="development-only"):
        Settings(env=Environment.PRODUCTION, api_token="x")  # type: ignore[arg-type]


def test_token_file_unreadable_means_no_token(tmp_path):
    s = Settings(env=Environment.TEST, api_token_file=tmp_path / "missing")
    assert s.resolved_api_token is None


def test_crop_box_matches_upstream_semantics():
    # Centred face, crop fits: 2.7x box around the centre.
    assert crop_box(1000, 1000, FaceBox(450, 450, 100, 100, 1), 2.7) == (365, 365, 635, 635)
    # Face at the corner: crop is shifted inside the image, not shrunk.
    left, top, right, bottom = crop_box(1000, 1000, FaceBox(0, 0, 100, 100, 1), 2.7)
    assert (left, top) == (0, 0)
    assert (right, bottom) == (270, 270)
    # Scale clamps to what fits.
    assert crop_box(200, 200, FaceBox(50, 50, 100, 100, 1), 4.0) == (0, 0, 199, 199)


@pytest.mark.models
@pytest.mark.skipif(not have_models, reason="real model artifacts not present")
class TestRealModels:
    def test_loads_and_verifies(self):
        rt = load_runtime(_settings(REAL_MODEL_DIR))
        assert rt.ready, rt.error
        for art in rt.artifacts:
            digest = hashlib.sha256((REAL_MODEL_DIR / art.file).read_bytes()).hexdigest()
            assert digest == art.sha256

    def test_tampered_copy_rejected(self, tmp_path):
        for f in FILES:
            shutil.copy(REAL_MODEL_DIR / f, tmp_path / f)
        with (tmp_path / FILES[1]).open("ab") as fh:
            fh.write(b"\0")
        rt = load_runtime(_settings(tmp_path))
        assert not rt.ready

    def test_no_face_on_noise_and_blank(self):
        import numpy as np

        rt = load_runtime(_settings(REAL_MODEL_DIR))
        assert rt.detector is not None
        rng = np.random.default_rng(1)
        noise = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
        assert rt.detector.detect(noise) == []
        assert rt.detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []

    def test_classifier_scores_in_range(self):
        import numpy as np

        rt = load_runtime(_settings(REAL_MODEL_DIR))
        assert rt.classifier is not None
        img = np.full((300, 300, 3), 128, dtype=np.uint8)
        scores = rt.classifier.score(img, FaceBox(100, 100, 100, 100, 0.9))
        assert [s.name for s in scores] == ["minifasnet_v2", "minifasnet_v1se"]
        assert all(0.0 <= s.live_score <= 1.0 for s in scores)


SAMPLES = os.environ.get("LIVENESS_TEST_SAMPLES_DIR")


@pytest.mark.models
@pytest.mark.skipif(not (have_models and SAMPLES), reason="LIVENESS_TEST_SAMPLES_DIR not set")
def test_labelled_samples_end_to_end(make_client):
    """Samples named live_*.{jpg,png} / spoof_*.{jpg,png}; asserts the decision per file."""
    import base64

    client = make_client(
        runtime=load_runtime(_settings(REAL_MODEL_DIR)), settings_=_settings(REAL_MODEL_DIR)
    )
    files = sorted(
        p for p in Path(SAMPLES or ".").iterdir() if p.suffix.lower() in {".jpg", ".png"}
    )
    assert files
    for path in files:
        expected = "live" if path.name.startswith("live_") else "spoof"
        body = {"image_base64": base64.b64encode(path.read_bytes()).decode()}
        resp = client.post("/v1/liveness/check", json=body)
        assert resp.status_code == 200, (path.name, resp.text)
        assert resp.json()["decision"] == expected, (path.name, resp.json()["live_score"])


def test_config_digests_match_build_manifest():
    from face_liveness import config

    manifest = Path(__file__).resolve().parent.parent / "tools" / "model_digests.sha256"
    pinned = dict(reversed(line.split()) for line in manifest.read_text().splitlines())
    s = Settings(env=Environment.TEST)
    assert pinned == {
        s.detector_file: config.YUNET_SHA256,
        s.minifasnet_v2_file: config.MINIFASNET_V2_SHA256,
        s.minifasnet_v1se_file: config.MINIFASNET_V1SE_SHA256,
    }
