"""Public API contract (request/response models). Source of truth for openapi.yaml."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .engine import Decision, DecisionReason
from .registry import EntryStatus

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


class ChallengeEvidence(BaseModel):
    challenge_id: str
    status: Literal["consumed"] = Field(
        description="The challenge was valid, unexpired and unused, and is now spent."
    )
    active_liveness_evaluated: Literal[False] = Field(
        default=False,
        description=(
            "Always false: the challenge proves freshness / single use only. No active "
            "liveness model is installed, so no challenge response was evaluated."
        ),
    )


class DecisionEvidence(BaseModel):
    """Bounded, fixed-shape explanation of how the decision was reached.

    Contains identifiers and scalars only: never image bytes, crops, embeddings or tensors.
    """

    model_id: str
    model_version: str
    model_digest: str = Field(description="SHA-256 of the active model manifest.")
    detector_id: str
    liveness_type: Literal["passive"]
    active_liveness_evaluated: Literal[False] = False
    score: float = Field(ge=0.0, le=1.0, description="Ensemble live score that was thresholded.")
    score_aggregation: Literal["mean_live_probability"]
    threshold: float
    threshold_calibrated: bool
    calibration_id: str | None
    margin: float = Field(description="score - threshold.")
    decision: Decision
    decision_reason: DecisionReason
    challenge: ChallengeEvidence | None = None


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
    evidence: DecisionEvidence


class ChallengeIssueResponse(BaseModel):
    challenge_id: str = Field(
        description="Opaque, single-use identifier. Treat as a secret; do not log in full."
    )
    challenge_type: Literal["freshness_nonce"] = Field(
        description="Only freshness / replay prevention is implemented today."
    )
    issued_at: datetime
    expires_at: datetime
    ttl_seconds: int
    instructions: list[str] = Field(
        description="User actions to perform. Always empty until an active model exists."
    )
    active_liveness: Literal[False] = False
    authoritative_method: Literal["passive_single_image"]


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
    max_image_side_px: int
    max_decoded_bytes: int
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
    id: str | None
    version: str | None
    manifest_sha256: str | None
    detector: str | None
    license: str | None
    artifacts: list[ArtifactOut]


class ChallengePolicy(BaseModel):
    supported: Literal[True] = True
    challenge_type: Literal["freshness_nonce"] = "freshness_nonce"
    ttl_seconds: int
    single_use: Literal[True] = True
    store: Literal["in_process"] = Field(
        default="in_process",
        description="Verify must reach the replica that issued the challenge.",
    )
    active_liveness: Literal[False] = False


class ResourceLimits(BaseModel):
    max_concurrent_checks: int
    max_queued_checks: int
    busy_timeout_seconds: float
    request_timeout_seconds: float
    max_request_bytes: int


class CapabilitiesResponse(BaseModel):
    service: str
    version: str
    api_version: Literal["v1"] = "v1"
    ready: bool
    methods: list[MethodInfo]
    unsupported: list[str]
    passive_liveness: bool
    active_liveness: bool = Field(
        description="True only if the active model version performs active liveness. "
        "No such model exists today, so this is always false."
    )
    challenge: ChallengePolicy
    input: InputLimits
    limits: ResourceLimits
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
    active_model_version: str | None


class ModelVersionOut(BaseModel):
    model_id: str | None
    version: str | None
    status: EntryStatus
    source: str = Field(description="`builtin` or `registry/<file>.json`.")
    liveness_type: Literal["passive"] | None
    manifest_sha256: str | None
    digests_verified: bool
    error: str | None
    license: str | None
    artifacts: list[ArtifactOut]


class ModelRegistryResponse(BaseModel):
    active_version: str = Field(description="Configured active version.")
    active_ready: bool
    active_error: str | None
    models: list[ModelVersionOut]
