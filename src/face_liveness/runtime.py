"""Model runtime lifecycle: registry, digest verification, loading, readiness (fail closed)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import Settings
from .inference import (
    FaceDetector,
    MiniFASNetEnsemble,
    MiniFASNetMember,
    SpoofClassifier,
    YuNetDetector,
)
from .registry import ArtifactSpec, ModelManifest, ModelRegistry, build_registry, resolve_artifact

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelRuntime:
    detector: FaceDetector | None = None
    classifier: SpoofClassifier | None = None
    error: str | None = None
    # The active, verified model version. Scoring is refused without one, so every
    # decision can be attributed to an exact model version and digest.
    manifest: ModelManifest | None = None
    registry: ModelRegistry = field(
        default_factory=lambda: ModelRegistry(entries=(), active_version="")
    )

    @property
    def ready(self) -> bool:
        return (
            self.error is None
            and self.detector is not None
            and self.classifier is not None
            and self.manifest is not None
        )

    @property
    def artifacts(self) -> list[ArtifactSpec]:
        return self.manifest.artifacts if self.manifest is not None else []

    @property
    def manifest_sha256(self) -> str | None:
        return self.manifest.digest if self.manifest is not None else None

    @classmethod
    def unavailable(cls, reason: str, registry: ModelRegistry | None = None) -> ModelRuntime:
        rt = cls(error=reason)
        if registry is not None:
            rt.registry = registry
        return rt


def load_runtime(settings: Settings) -> ModelRuntime:
    """Load the active model version. Never raises: failures yield a not-ready runtime."""
    registry = build_registry(settings)
    active = registry.active
    if active is None or active.manifest is None:
        reason = registry.active_error or "no active model version"
        return ModelRuntime.unavailable(reason, registry)
    manifest = active.manifest

    try:
        detector = YuNetDetector(
            resolve_artifact(settings.model_dir, manifest.detector.file),
            score_threshold=settings.detector_score_threshold,
            max_side=settings.detector_max_side,
        )
        classifier = MiniFASNetEnsemble(
            [
                MiniFASNetMember(c.name, resolve_artifact(settings.model_dir, c.file), c.crop_scale)
                for c in manifest.classifiers
            ],
            intra_op_threads=settings.onnx_intra_op_threads,
        )
    except Exception as exc:  # any runtime/driver failure must leave us not-ready
        log.exception("model runtime failed to load")
        return ModelRuntime.unavailable(
            f"model runtime failed to load: {type(exc).__name__}", registry
        )

    return ModelRuntime(
        detector=detector, classifier=classifier, manifest=manifest, registry=registry
    )
