"""Public API contract (request/response models). Source of truth for openapi.yaml."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .engine import Decision

PASSIVE_SINGLE_IMAGE: Literal["passive_single_image"] = "passive_single_image"


class LivenessCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_base64: str = Field(
        min_length=1,
        description=(
            "Standard base64 of a JPEG, PNG or WebP image containing exactly one face. "
            "A `data:image/...;base64,` prefix is tolerated. The image is processed in "
            "memory only and is never persisted."
        ),
    )
    mode: Literal["passive_single_image"] = Field(
        default=PASSIVE_SINGLE_IMAGE,
        description="Liveness method. Only passive single-image PAD is implemented.",
    )


class FaceRegion(BaseModel):
    x: int
    y: int
    width: int
    height: int
    detection_score: float = Field(ge=0.0, le=1.0)


class ComponentScoreOut(BaseModel):
    name: str
    live_score: float = Field(ge=0.0, le=1.0)


class ModelRef(BaseModel):
    id: str
    detector: str
    threshold_calibrated: bool
    calibration_id: str | None


class ImageInfo(BaseModel):
    width: int
    height: int
    media_type: str


class LivenessCheckResponse(BaseModel):
    request_id: str
    decision: Decision = Field(
        description="`live` only if live_score >= threshold; otherwise `spoof`."
    )
    is_live: bool
    live_score: float = Field(
        ge=0.0, le=1.0, description="Ensemble-mean probability of a bona fide (live) face."
    )
    threshold: float
    method: Literal["passive_single_image"]
    face: FaceRegion
    faces_detected: int
    components: list[ComponentScoreOut]
    model: ModelRef
    image: ImageInfo
    processing_ms: float
    image_persisted: Literal[False] = False


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str
    retryable: bool
    request_id: str | None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, str]


class MethodInfo(BaseModel):
    id: str
    description: str


class InputLimits(BaseModel):
    formats: list[str]
    max_image_bytes: int
    max_image_pixels: int
    min_image_side_px: int
    min_face_size_px: int
    faces_required: Literal[1] = 1


class DecisionPolicy(BaseModel):
    score: str
    threshold: float
    threshold_calibrated: bool
    calibration_id: str | None


class ArtifactOut(BaseModel):
    name: str
    file: str
    sha256: str


class ModelInfo(BaseModel):
    id: str
    detector: str
    license: str
    artifacts: list[ArtifactOut]


class CapabilitiesResponse(BaseModel):
    service: str
    version: str
    api_version: Literal["v1"] = "v1"
    ready: bool
    methods: list[MethodInfo]
    unsupported: list[str]
    input: InputLimits
    decision: DecisionPolicy
    model: ModelInfo
    face_matching: Literal[False] = False
    image_persistence: Literal["none"] = "none"


class StatusResponse(BaseModel):
    service: str
    version: str
    environment: str
    ready: bool
    reason: str | None
    uptime_seconds: float
