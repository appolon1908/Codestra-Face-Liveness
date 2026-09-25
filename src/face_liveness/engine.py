"""Liveness decision pipeline: detect exactly one face, score it, apply the threshold."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import StrEnum

from .config import Settings
from .errors import ErrorCode, LivenessError
from .imaging import DecodedImage
from .inference import ComponentScore, FaceBox
from .runtime import ModelRuntime

log = logging.getLogger(__name__)


class Decision(StrEnum):
    LIVE = "live"
    SPOOF = "spoof"


class DecisionReason(StrEnum):
    SCORE_AT_OR_ABOVE_THRESHOLD = "score_at_or_above_threshold"
    SCORE_BELOW_THRESHOLD = "score_below_threshold"
    # Fusion policies that combine passive and active evidence (see fusion.py).
    ALL_REQUIRED_EVIDENCE_PASSED = "all_required_evidence_passed"
    ACTIVE_CHALLENGE_FAILED = "active_challenge_failed"
    ACTIVE_SCORE_BELOW_MINIMUM = "active_score_below_minimum"


@dataclass(frozen=True, slots=True)
class LivenessResult:
    decision: Decision
    reason: DecisionReason
    live_score: float
    threshold: float
    face: FaceBox
    components: list[ComponentScore]
    faces_detected: int
    inference_seconds: float


class LivenessEngine:
    def __init__(self, settings: Settings, runtime: ModelRuntime) -> None:
        self.settings = settings
        self.runtime = runtime

    def _select_face(self, faces: list[FaceBox]) -> FaceBox:
        if not faces:
            raise LivenessError(ErrorCode.NO_FACE, "no face detected")
        faces = sorted(faces, key=lambda f: f.area, reverse=True)
        primary = faces[0]
        min_area = primary.area * self.settings.secondary_face_area_ratio
        if any(f.area >= min_area for f in faces[1:]):
            raise LivenessError(
                ErrorCode.MULTIPLE_FACES, "more than one face detected; exactly one is required"
            )
        if min(primary.w, primary.h) < self.settings.min_face_size_px:
            raise LivenessError(
                ErrorCode.FACE_TOO_SMALL,
                f"face must be at least {self.settings.min_face_size_px}px",
            )
        return primary

    def check(self, image: DecodedImage) -> LivenessResult:
        runtime = self.runtime
        if not runtime.ready or runtime.detector is None or runtime.classifier is None:
            raise LivenessError(ErrorCode.MODEL_UNAVAILABLE, "liveness model is not available")

        started = time.perf_counter()
        try:
            faces = runtime.detector.detect(image.bgr)
        except Exception as exc:
            log.exception("face detection failed")
            raise LivenessError(ErrorCode.INFERENCE_FAILED, "face detection failed") from exc
        face = self._select_face(faces)

        try:
            components = runtime.classifier.score(image.bgr, face)
        except Exception as exc:
            log.exception("anti-spoofing inference failed")
            raise LivenessError(ErrorCode.INFERENCE_FAILED, "liveness inference failed") from exc
        if not components:
            raise LivenessError(ErrorCode.INFERENCE_FAILED, "liveness inference returned no score")

        live_score = sum(c.live_score for c in components) / len(components)
        if not 0.0 <= live_score <= 1.0:
            raise LivenessError(ErrorCode.INFERENCE_FAILED, "liveness score out of range")
        threshold = self.settings.live_threshold
        if live_score >= threshold:
            decision, reason = Decision.LIVE, DecisionReason.SCORE_AT_OR_ABOVE_THRESHOLD
        else:
            decision, reason = Decision.SPOOF, DecisionReason.SCORE_BELOW_THRESHOLD
        return LivenessResult(
            decision=decision,
            reason=reason,
            live_score=live_score,
            threshold=threshold,
            face=face,
            components=components,
            faces_detected=len(faces),
            inference_seconds=time.perf_counter() - started,
        )
