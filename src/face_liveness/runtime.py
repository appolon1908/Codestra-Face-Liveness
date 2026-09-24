"""Model runtime lifecycle: digest verification, loading, readiness (fail closed)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .inference import (
    FaceDetector,
    MiniFASNetEnsemble,
    MiniFASNetMember,
    SpoofClassifier,
    YuNetDetector,
)

log = logging.getLogger(__name__)

MODEL_ID = "minifasnet-v2+v1se-ensemble"
DETECTOR_ID = "yunet-2023mar"


@dataclass(frozen=True, slots=True)
class ArtifactInfo:
    name: str
    file: str
    sha256: str


@dataclass(slots=True)
class ModelRuntime:
    detector: FaceDetector | None = None
    classifier: SpoofClassifier | None = None
    error: str | None = None
    artifacts: list[ArtifactInfo] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.error is None and self.detector is not None and self.classifier is not None

    @classmethod
    def unavailable(cls, reason: str) -> ModelRuntime:
        return cls(error=reason)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runtime(settings: Settings) -> ModelRuntime:
    """Load and verify all model artifacts. Never raises: failures yield a not-ready runtime."""
    specs = [
        ("detector", settings.detector_file, settings.detector_sha256),
        ("minifasnet_v2", settings.minifasnet_v2_file, settings.minifasnet_v2_sha256),
        ("minifasnet_v1se", settings.minifasnet_v1se_file, settings.minifasnet_v1se_sha256),
    ]
    artifacts: list[ArtifactInfo] = []
    for name, file, expected in specs:
        path = settings.model_dir / file
        if not path.is_file():
            return ModelRuntime.unavailable(f"model artifact missing: {name} ({file})")
        actual = _sha256(path)
        if settings.verify_model_digests and actual != expected.lower():
            return ModelRuntime.unavailable(f"model artifact digest mismatch: {name} ({file})")
        artifacts.append(ArtifactInfo(name=name, file=file, sha256=actual))

    try:
        detector = YuNetDetector(
            settings.model_dir / settings.detector_file,
            score_threshold=settings.detector_score_threshold,
            max_side=settings.detector_max_side,
        )
        classifier = MiniFASNetEnsemble(
            [
                MiniFASNetMember(
                    "minifasnet_v2", settings.model_dir / settings.minifasnet_v2_file, 2.7
                ),
                MiniFASNetMember(
                    "minifasnet_v1se", settings.model_dir / settings.minifasnet_v1se_file, 4.0
                ),
            ],
            intra_op_threads=settings.onnx_intra_op_threads,
        )
    except Exception as exc:  # any runtime/driver failure must leave us not-ready
        log.exception("model runtime failed to load")
        return ModelRuntime.unavailable(f"model runtime failed to load: {type(exc).__name__}")

    return ModelRuntime(detector=detector, classifier=classifier, artifacts=artifacts)
