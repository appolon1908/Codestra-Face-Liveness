"""Public API contract (request/response models). Source of truth for openapi.yaml."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .active import ActiveOutcome
from .engine import Decision, DecisionReason
from .fusion import EvidenceKind, FusionStrategy
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
    active_frames_base64: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=32,
        description=(
            "Active-liveness capture: frames recorded while the user followed the "
            "challenge instructions (same encoding and limits as `image_base64`, at most "
            "`capabilities.active.max_frames`). Only accepted on challenge verify, and "
            "only when `capabilities.active.available` is true; otherwise the request "
            "fails closed with `EVIDENCE_UNAVAILABLE`. No active provider is installed "
            "today, so this must be omitted."
        ),
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
    active_liveness_evaluated: bool = Field(
        description=(
            "True only if a tested active provider evaluated the challenge response. No "
            "such provider is installed today, so this is false: the challenge proves "
            "freshness / single use only."
        ),
    )


class ActiveEvidence(BaseModel):
    provider_id: str
    provider_version: str
    outcome: ActiveOutcome
    score: float | None = Field(ge=0.0, le=1.0)


class DecisionEvidence(BaseModel):
    """Bounded, fixed-shape explanation of how the decision was reached.

    Contains identifiers and scalars only: never image bytes, crops, embeddings or tensors.
    """

    model_id: str
    model_version: str
    model_digest: str = Field(description="SHA-256 of the active model manifest.")
    detector_id: str
    liveness_type: Literal["passive"] = Field(description="Type of the scoring model.")
    active_liveness_evaluated: bool = Field(
        description="True only if active evidence contributed to the decision."
    )
    score: float = Field(ge=0.0, le=1.0, description="Ensemble live score that was thresholded.")
    score_aggregation: Literal["mean_live_probability"]
    threshold: float
    threshold_calibrated: bool
    calibration_id: str | None
    margin: float = Field(description="score - threshold.")
    policy_id: str = Field(description="Versioned fusion policy that produced `decision`.")
    evidence_used: list[EvidenceKind]
    decision: Decision
    decision_reason: DecisionReason
    challenge: ChallengeEvidence | None = None
    active: ActiveEvidence | None = Field(
        default=None, description="Active provider result; null when none was evaluated."
    )


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


class ChallengeInstructionOut(BaseModel):
    action: str = Field(description="Provider-defined user action, e.g. `turn_head_left`.")
    timeout_seconds: float


class ChallengeIssueResponse(BaseModel):
    challenge_id: str = Field(
        description="Opaque, single-use identifier. Treat as a secret; do not log in full."
    )
    challenge_type: Literal["freshness_nonce", "active_challenge"] = Field(
        description="`active_challenge` only when a tested active provider is available; "
        "today always `freshness_nonce` (freshness / replay prevention only)."
    )
    issued_at: datetime
    expires_at: datetime
    ttl_seconds: int
    instructions: list[ChallengeInstructionOut] = Field(
        description="User actions to perform. Empty unless an active provider is available."
    )
    active_liveness: bool = Field(
        description="True only if a tested active provider issued instructions. False today."
    )
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
    challenge_type: Literal["freshness_nonce", "active_challenge"]
    ttl_seconds: int
    single_use: Literal[True] = True
    store: Literal["in_process"] = Field(
        default="in_process",
        description="Verify must reach the replica that issued the challenge.",
    )
    active_liveness: bool


class ActiveProviderOut(BaseModel):
    provider_id: str
    version: str
    contract_version: str
    validation_id: str | None = Field(description="PAD evaluation that tested this provider.")


class ActiveLivenessInfo(BaseModel):
    """Pluggable active (challenge-response) provider slot. Empty today."""

    contract_version: Literal["active-provider.v1"] = "active-provider.v1"
    available: bool = Field(
        description="True only if a provider is configured, ready, and names the "
        "validation report that tested it. No provider ships with this service."
    )
    configured_provider_id: str | None
    provider: ActiveProviderOut | None
    max_frames: int = Field(description="Upper bound on `active_frames_base64` entries.")
    reason: str | None = Field(description="Why active liveness is unavailable.")


class FusionPolicyOut(BaseModel):
    policy_id: str = Field(description="Explicit, versioned policy id, e.g. `passive-only.v1`.")
    strategy: FusionStrategy | None
    required_evidence: list[EvidenceKind]
    description: str | None
    ready: bool = Field(description="False means every assessment fails closed.")
    reason: str | None


class CapacityProfileRef(BaseModel):
    profile_id: str | None = Field(
        description="`profile_id` of the benchmark report (face-liveness-benchmark) the "
        "concurrency and timeout settings were sized from; null if none was recorded."
    )
    tested: bool = Field(description="True if a benchmark profile is referenced.")
    auto_tuned: Literal[False] = Field(
        default=False, description="Always false: the service never tunes itself."
    )


class ResourceLimits(BaseModel):
    max_concurrent_checks: int
    max_queued_checks: int
    busy_timeout_seconds: float
    request_timeout_seconds: float
    max_request_bytes: int
    capacity_profile: CapacityProfileRef


class CapabilitiesResponse(BaseModel):
    service: str
    version: str
    api_version: Literal["v1"] = "v1"
    ready: bool
    methods: list[MethodInfo]
    unsupported: list[str]
    passive_liveness: bool
    active_liveness: bool = Field(
        description="True only if a tested active provider is available "
        "(`active.available`). No such provider exists today, so this is false."
    )
    active: ActiveLivenessInfo
    fusion: FusionPolicyOut
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


class ValidationOut(BaseModel):
    passed: bool
    validated_at: datetime
    calibration_id: str | None
    threshold: float | None
    digests_passed: bool
    smoke_passed: bool
    calibration_passed: bool


class ModelVersionOut(BaseModel):
    model_id: str | None
    version: str | None
    status: EntryStatus = Field(
        description="`installed` (digests verified) -> `candidate` (passing validation "
        "record) -> `active` (selected by deployment); `invalid` never loads."
    )
    source: str = Field(description="`builtin` or `registry/<file>.json`.")
    liveness_type: Literal["passive"] | None
    manifest_sha256: str | None
    digests_verified: bool
    error: str | None
    license: str | None
    artifacts: list[ArtifactOut]
    validation: ValidationOut | None = Field(
        description="Validation record for this exact manifest, if any."
    )
    validation_error: str | None = Field(description="Why this version is not (yet) a candidate.")


class ModelRegistryResponse(BaseModel):
    active_version: str = Field(description="Configured active version.")
    active_ready: bool
    active_error: str | None
    models: list[ModelVersionOut]
