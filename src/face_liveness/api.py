"""FastAPI application factory: routes, auth, error envelope, metrics, access logs."""

from __future__ import annotations

import hmac
import logging
import re
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from fastapi import Depends, FastAPI, Header, Path, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import SERVICE_NAME, __version__
from .active import ActiveLivenessProvider, ActiveResult, resolve_active_provider
from .admission import AdmissionController, Deadline
from .challenges import ChallengeStore
from .config import Settings
from .engine import Decision, LivenessEngine
from .errors import HTTP_STATUS, STAGE, ErrorCode, Guard, LivenessError
from .fusion import POLICIES, FusionPolicy, fuse, require_ready, resolve_policy
from .imaging import ALLOWED_FORMATS, DecodedImage, decode_base64, decode_image
from .logging_config import request_id_var
from .metrics import Metrics
from .registry import RegistryEntry
from .runtime import ModelRuntime, load_runtime
from .schemas import (
    PASSIVE_SINGLE_IMAGE,
    ActiveEvidence,
    ActiveLivenessInfo,
    ActiveProviderOut,
    ArtifactOut,
    CapabilitiesResponse,
    CapacityProfileRef,
    ChallengeEvidence,
    ChallengeInstructionOut,
    ChallengeIssueResponse,
    ChallengePolicy,
    ComponentScoreOut,
    DecisionEvidence,
    DecisionPolicy,
    ErrorResponse,
    FaceRegion,
    FusionPolicyOut,
    HealthResponse,
    ImageInfo,
    InputLimits,
    LivenessCheckRequest,
    LivenessCheckResponse,
    LivenessEvidencePage,
    LivenessEvidenceSummary,
    MethodInfo,
    ModelInfo,
    ModelRef,
    ModelRegistryResponse,
    ModelVersionOut,
    ReadinessResponse,
    ResourceLimits,
    StatusResponse,
    ValidationOut,
)

log = logging.getLogger("face_liveness.api")

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")
_TENANT_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")
_EVIDENCE_STORE_LIMIT = 1000


def _error_response(code: ErrorCode, message: str) -> JSONResponse:
    body = ErrorResponse.model_validate(
        {
            "error": {
                "code": code.value,
                "message": message,
                "retryable": LivenessError(code, message).retryable,
                "request_id": request_id_var.get(),
            }
        }
    )
    headers = {"Retry-After": "1"} if code is ErrorCode.BUSY else None
    return JSONResponse(body.model_dump(), status_code=HTTP_STATUS[code], headers=headers)


def _error_responses(*codes: ErrorCode) -> dict[int | str, dict[str, Any]]:
    by_status: dict[int, list[str]] = {}
    for code in codes:
        by_status.setdefault(HTTP_STATUS[code], []).append(code.value)
    return {
        status: {"model": ErrorResponse, "description": "Error codes: " + ", ".join(names)}
        for status, names in sorted(by_status.items())
    }


# Errors any image-assessing endpoint can return.
_ASSESSMENT_ERRORS = (
    ErrorCode.INVALID_REQUEST,
    ErrorCode.UNAUTHORIZED,
    ErrorCode.PAYLOAD_TOO_LARGE,
    ErrorCode.UNSUPPORTED_MEDIA_TYPE,
    ErrorCode.INVALID_IMAGE,
    ErrorCode.IMAGE_TOO_SMALL,
    ErrorCode.NO_FACE,
    ErrorCode.MULTIPLE_FACES,
    ErrorCode.FACE_TOO_SMALL,
    ErrorCode.INTERNAL_ERROR,
    ErrorCode.MODEL_UNAVAILABLE,
    ErrorCode.BUSY,
    ErrorCode.DEADLINE_EXCEEDED,
    ErrorCode.EVIDENCE_UNAVAILABLE,
    ErrorCode.INFERENCE_FAILED,
)


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Rejects request bodies above a byte limit, with or without Content-Length."""

    def __init__(
        self, app: ASGIApp, max_bytes: int, on_reject: Callable[[], None] | None = None
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.on_reject = on_reject

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    too_large = int(value) > self.max_bytes
                except ValueError:
                    too_large = False
                if too_large:
                    await self._reject(scope, receive, send)
                    return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not response_started:
                await self._reject(scope, receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.on_reject is not None:
            self.on_reject()
        response = _error_response(ErrorCode.PAYLOAD_TOO_LARGE, "request body too large")
        await response(scope, receive, send)


def _route_template(request: Request) -> str:
    return getattr(request.scope.get("route"), "path", "unmatched")


def _model_version_out(entry: RegistryEntry) -> ModelVersionOut:
    m = entry.manifest
    rec = entry.validation
    return ModelVersionOut(
        model_id=m.model_id if m else None,
        version=m.version if m else None,
        status=entry.status,
        source=entry.source,
        liveness_type=m.liveness_type if m else None,
        manifest_sha256=entry.manifest_sha256,
        digests_verified=entry.digests_verified,
        error=entry.error,
        license=m.license if m else None,
        artifacts=[
            ArtifactOut(name=a.name, file=a.file, sha256=a.sha256)
            for a in (m.artifacts if m else [])
        ],
        validation=(
            ValidationOut(
                passed=rec.passed,
                validated_at=rec.validated_at,
                calibration_id=rec.calibration.calibration_id,
                threshold=rec.calibration.threshold,
                digests_passed=rec.digests.passed,
                smoke_passed=rec.smoke.passed,
                calibration_passed=rec.calibration.passed,
            )
            if rec is not None
            else None
        ),
        validation_error=entry.validation_error,
    )


def _decode(data: str, settings: Settings) -> DecodedImage:
    raw = decode_base64(data, settings.max_image_bytes)
    return decode_image(
        raw,
        max_pixels=settings.max_image_pixels,
        min_side=settings.min_image_side_px,
        max_side=settings.max_image_side_px,
        max_decoded_bytes=settings.max_decoded_bytes,
    )


def create_app(
    settings: Settings | None = None,
    runtime: ModelRuntime | None = None,
    active_provider: ActiveLivenessProvider | None = None,
) -> FastAPI:
    settings = settings or Settings()
    metrics = Metrics()
    active = resolve_active_provider(settings, active_provider)
    policy_state = resolve_policy(settings.fusion_policy_id, active)
    # Active captures carry several frames; the body limit grows only when they are usable.
    body_limit = settings.max_request_bytes * (
        1 + settings.max_active_frames if active.available else 1
    )
    started_at = time.monotonic()
    admission = AdmissionController(settings.max_concurrent_checks, settings.max_queued_checks)
    challenges = ChallengeStore(
        ttl_seconds=settings.challenge_ttl_seconds,
        capacity=settings.max_outstanding_challenges,
    )
    evidence_store: deque[tuple[str, LivenessEvidenceSummary]] = deque(
        maxlen=_EVIDENCE_STORE_LIMIT
    )
    evidence_lock = Lock()
    metrics.checks_in_flight.set_function(lambda: admission.in_flight)
    metrics.checks_waiting.set_function(lambda: admission.waiting)
    metrics.challenges_outstanding.set_function(lambda: challenges.outstanding())
    metrics.active_provider_available.set(1 if active.available else 0)
    metrics.fusion_policy_ready.labels(
        policy_state.configured_id if policy_state.configured_id in POLICIES else "unknown"
    ).set(1 if policy_state.ready else 0)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.runtime is None:
            app.state.runtime = load_runtime(settings)
        rt: ModelRuntime = app.state.runtime
        app.state.engine = LivenessEngine(settings, rt)
        metrics.model_ready.set(1 if rt.ready else 0)
        metrics.threshold.set(settings.live_threshold)
        manifest = rt.manifest
        model_id = manifest.model_id if manifest else "none"
        detector_id = manifest.detector.name if manifest else "none"
        metrics.build.info({"version": __version__, "model": model_id, "detector": detector_id})
        metrics.active_model.info(
            {
                "version": manifest.version if manifest else "none",
                "manifest_sha256": rt.manifest_sha256 or "none",
            }
        )
        for entry in rt.registry.entries:
            if entry.error is not None:
                log.warning(
                    "model registry entry invalid",
                    extra={"source": entry.source, "reason": entry.error},
                )
        if rt.ready and manifest is not None:
            log.info(
                "model runtime ready",
                extra={
                    "model": manifest.model_id,
                    "model_version": manifest.version,
                    "manifest_sha256": rt.manifest_sha256,
                },
            )
        else:
            log.error("model runtime NOT ready; failing closed", extra={"reason": rt.error})
        if not policy_state.ready:
            log.error(
                "fusion policy NOT ready; failing closed", extra={"reason": policy_state.error}
            )
        if not settings.threshold_calibration_id:
            log.warning(
                "live_threshold is the provisional default (uncalibrated)",
                extra={"threshold": settings.live_threshold},
            )
        yield

    app = FastAPI(
        title="Codestra Face Liveness API",
        version=__version__,
        description=(
            "Standalone passive presentation-attack-detection (PAD) service for Codestra "
            "FACE-ID. It answers one question: is the single face in this image a bona fide "
            "live capture or a presentation attack (print, screen replay, mask)? It does "
            "NOT perform face matching or identity verification, and it does NOT perform "
            "active (challenge-response) liveness: challenges are freshness / replay "
            "nonces only and the passive model decision is authoritative. The only "
            "supported caller is Middleware V3, which owns authentication normalisation, "
            "policy, orchestration and cross-system audit. Images are processed in memory "
            "and never persisted."
        ),
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        servers=[{"url": "http://localhost:8080", "description": "Local development"}],
    )
    app.state.runtime = runtime
    app.state.settings = settings
    app.state.metrics = metrics
    app.state.admission = admission
    app.state.challenges = challenges
    app.state.evidence_store = evidence_store

    def _tenant(value: str | None, *, required: bool) -> str | None:
        if value is None:
            if required:
                raise LivenessError(ErrorCode.INVALID_REQUEST, "X-Tenant-ID header is required")
            return None
        value = value.strip()
        if not _TENANT_ID_RE.fullmatch(value):
            raise LivenessError(ErrorCode.INVALID_REQUEST, "invalid X-Tenant-ID header")
        return value

    def _store_evidence(tenant_id: str, response: LivenessCheckResponse) -> str:
        evidence_ref = "lev_" + uuid.uuid4().hex
        summary = LivenessEvidenceSummary(
            evidence_ref=evidence_ref,
            request_id=response.request_id,
            created_at=datetime.now(timezone.utc),
            decision=response.decision,
            is_live=response.is_live,
            live_score=response.live_score,
            threshold=response.threshold,
            method=response.method,
            model_id=response.evidence.model_id,
            model_version=response.evidence.model_version,
            model_digest=response.evidence.model_digest,
            policy_id=response.evidence.policy_id,
            decision_reason=response.evidence.decision_reason,
            active_liveness_evaluated=response.evidence.active_liveness_evaluated,
        )
        with evidence_lock:
            evidence_store.append((tenant_id, summary))
        return evidence_ref

    def record_rejection(
        request: Request | None, code: ErrorCode, guard: Guard | None = None
    ) -> None:
        route = _route_template(request) if request is not None else "pre_routing"
        metrics.rejections.labels(route, STAGE[code], code.value).inc()
        if guard is not None:
            metrics.guard_rejections.labels(guard.value).inc()

    # --- error envelope -----------------------------------------------------------------

    @app.exception_handler(LivenessError)
    async def _liveness_error(request: Request, exc: LivenessError) -> JSONResponse:
        request.state.outcome = exc.code.value
        record_rejection(request, exc.code, exc.guard)
        return _error_response(exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Never echo the submitted input (it may contain image data).
        fields = sorted(
            {".".join(str(p) for p in e.get("loc", ())[1:]) or "body" for e in exc.errors()}
        )
        request.state.outcome = ErrorCode.INVALID_REQUEST.value
        record_rejection(request, ErrorCode.INVALID_REQUEST)
        return _error_response(ErrorCode.INVALID_REQUEST, "invalid request: " + ", ".join(fields))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: ErrorCode.NOT_FOUND, 405: ErrorCode.METHOD_NOT_ALLOWED}.get(
            exc.status_code, ErrorCode.INVALID_REQUEST
        )
        record_rejection(request, code)
        return _error_response(code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        record_rejection(request, ErrorCode.INTERNAL_ERROR)
        return _error_response(ErrorCode.INTERNAL_ERROR, "internal error")

    # --- request context, access log, HTTP metrics ---------------------------------------

    @app.middleware("http")
    async def _observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.received_at = time.monotonic()
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else str(uuid.uuid4())
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - start
            template = _route_template(request)
            metrics.http_requests.labels(template, request.method, str(status)).inc()
            metrics.http_duration.labels(template, request.method).observe(elapsed)
            if template not in ("/healthz", "/readyz", "/metrics"):
                extra: dict[str, Any] = {
                    "method": request.method,
                    "route": template,
                    "status": status,
                    "duration_ms": round(elapsed * 1000, 2),
                }
                for key in ("outcome", "live_score", "model_version", "policy_id"):
                    if hasattr(request.state, key):
                        extra[key] = getattr(request.state, key)
                log.info("request completed", extra=extra)
            request_id_var.reset(token)

    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=body_limit,
        on_reject=lambda: record_rejection(None, ErrorCode.PAYLOAD_TOO_LARGE, Guard.REQUEST_BODY),
    )

    # --- auth ----------------------------------------------------------------------------

    def require_caller(request: Request) -> None:
        if not settings.auth_required:
            return
        expected = settings.resolved_api_token
        header = request.headers.get("Authorization", "")
        scheme, _, presented = header.partition(" ")
        if (
            expected is None
            or scheme.lower() != "bearer"
            or not hmac.compare_digest(presented.encode(), expected.encode())
        ):
            raise LivenessError(ErrorCode.UNAUTHORIZED, "missing or invalid bearer token")

    caller = [Depends(require_caller)]

    def _runtime() -> ModelRuntime:
        rt: ModelRuntime | None = app.state.runtime
        return rt if rt is not None else ModelRuntime.unavailable("model runtime not initialised")

    def _readiness_checks() -> dict[str, str]:
        rt = _runtime()
        checks = {"model": "ok" if rt.ready else f"fail: {rt.error}"}
        checks["fusion_policy"] = "ok" if policy_state.ready else f"fail: {policy_state.error}"
        if active.configured_id is not None:
            # An operator asked for an active provider: its absence is a failure, not a
            # silent downgrade to passive-only.
            checks["active_provider"] = "ok" if active.available else f"fail: {active.reason}"
        if settings.auth_required:
            checks["auth"] = (
                "ok" if settings.resolved_api_token else "fail: api token not configured"
            )
        return checks

    # --- operational endpoints ------------------------------------------------------------

    @app.get("/healthz", response_model=HealthResponse, tags=["operations"], operation_id="health")
    def healthz() -> HealthResponse:
        """Liveness of the process itself. Does not check the model."""
        return HealthResponse(status="ok")

    @app.get(
        "/health/ready",
        include_in_schema=True,
        tags=["operations"],
        operation_id="healthReadyAlias",
    )
    @app.get(
        "/readyz",
        response_model=ReadinessResponse,
        tags=["operations"],
        operation_id="ready",
        responses={503: {"model": ReadinessResponse, "description": "Not ready (fail closed)"}},
    )
    def readyz(response: Response) -> ReadinessResponse:
        """Ready only when models are loaded + digest-verified and auth is configured."""
        checks = _readiness_checks()
        ok = all(v == "ok" for v in checks.values())
        if not ok:
            response.status_code = 503
        return ReadinessResponse(status="ready" if ok else "not_ready", checks=checks)

    @app.get(
        "/metrics",
        tags=["operations"],
        operation_id="metrics",
        response_class=Response,
        responses={200: {"content": {"text/plain": {}}, "description": "Prometheus exposition"}},
    )
    def prometheus_metrics() -> Response:
        """Prometheus metrics in text exposition format."""
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get(
        "/v1/capabilities",
        response_model=CapabilitiesResponse,
        tags=["liveness"],
        operation_id="capabilities",
        dependencies=caller,
        responses=_error_responses(ErrorCode.UNAUTHORIZED),
    )
    def capabilities() -> CapabilitiesResponse:
        """What this service can and cannot do, its limits, and its decision policy."""
        rt = _runtime()
        manifest = rt.manifest
        calibration_id = settings.threshold_calibration_id or None
        policy = policy_state.policy
        desc = active.provider.descriptor if active.provider is not None else None
        unsupported = ["video_sequence", "depth_or_ir", "face_matching", "identity_verification"]
        if not active.available:
            unsupported.insert(0, "active_challenge")
        return CapabilitiesResponse(
            service=SERVICE_NAME,
            version=__version__,
            ready=rt.ready,
            methods=[
                MethodInfo(
                    id=PASSIVE_SINGLE_IMAGE,
                    description=(
                        "Passive presentation-attack detection on one still image "
                        "(MiniFASNet ensemble). No user interaction required."
                    ),
                )
            ],
            unsupported=unsupported,
            passive_liveness=True,
            # No provider ships with the service (active.KNOWN_PROVIDERS is empty), so this
            # is false unless a tested provider is explicitly plugged in.
            active_liveness=active.available,
            active=ActiveLivenessInfo(
                available=active.available,
                configured_provider_id=active.configured_id,
                provider=(
                    ActiveProviderOut(
                        provider_id=desc.provider_id,
                        version=desc.version,
                        contract_version=desc.contract_version,
                        validation_id=desc.validation_id or None,
                    )
                    if desc is not None
                    else None
                ),
                max_frames=settings.max_active_frames,
                reason=active.reason,
            ),
            fusion=FusionPolicyOut(
                policy_id=policy_state.configured_id,
                strategy=policy.strategy if policy else None,
                required_evidence=list(policy.required) if policy else [],
                description=policy.description if policy else None,
                ready=policy_state.ready,
                reason=policy_state.error,
            ),
            challenge=ChallengePolicy(
                ttl_seconds=settings.challenge_ttl_seconds,
                challenge_type="active_challenge" if active.available else "freshness_nonce",
                active_liveness=active.available,
            ),
            input=InputLimits(
                formats=sorted(ALLOWED_FORMATS.values()),
                max_image_bytes=settings.max_image_bytes,
                max_image_pixels=settings.max_image_pixels,
                max_image_side_px=settings.max_image_side_px,
                max_decoded_bytes=settings.max_decoded_bytes,
                min_image_side_px=settings.min_image_side_px,
                min_face_size_px=settings.min_face_size_px,
            ),
            limits=ResourceLimits(
                max_concurrent_checks=settings.max_concurrent_checks,
                max_queued_checks=settings.max_queued_checks,
                busy_timeout_seconds=settings.busy_timeout_seconds,
                request_timeout_seconds=settings.request_timeout_seconds,
                max_request_bytes=body_limit,
                capacity_profile=CapacityProfileRef(
                    profile_id=settings.capacity_profile_id or None,
                    tested=bool(settings.capacity_profile_id),
                ),
            ),
            decision=DecisionPolicy(
                score="mean of MiniFASNet softmax[live] over the ensemble",
                threshold=settings.live_threshold,
                threshold_calibrated=calibration_id is not None,
                calibration_id=calibration_id,
            ),
            model=ModelInfo(
                id=manifest.model_id if manifest else None,
                version=manifest.version if manifest else None,
                manifest_sha256=rt.manifest_sha256,
                detector=manifest.detector.name if manifest else None,
                license=manifest.license if manifest else None,
                artifacts=[
                    ArtifactOut(name=a.name, file=a.file, sha256=a.sha256) for a in rt.artifacts
                ],
            ),
        )

    @app.get(
        "/v1/liveness/status",
        response_model=StatusResponse,
        tags=["liveness"],
        operation_id="serviceStatus",
        dependencies=caller,
        responses=_error_responses(ErrorCode.UNAUTHORIZED),
    )
    def status() -> StatusResponse:
        """Service-local status summary."""
        checks = _readiness_checks()
        failures = [f"{k}: {v}" for k, v in checks.items() if v != "ok"]
        manifest = _runtime().manifest
        return StatusResponse(
            service=SERVICE_NAME,
            version=__version__,
            environment=settings.env.value,
            ready=not failures,
            reason="; ".join(failures) or None,
            uptime_seconds=round(time.monotonic() - started_at, 3),
            active_model_version=manifest.version if manifest else None,
        )

    # --- model registry (read-only) --------------------------------------------------------

    @app.get(
        "/v1/models",
        response_model=ModelRegistryResponse,
        tags=["models"],
        operation_id="listModels",
        dependencies=caller,
        responses=_error_responses(ErrorCode.UNAUTHORIZED),
    )
    def list_models() -> ModelRegistryResponse:
        """Installed model versions, their validation status, and which one is active.

        Read-only: the registry is built and verified at startup. Changing the active
        version is a deployment action (`LIVENESS_ACTIVE_MODEL_VERSION`), not an API call.
        """
        rt = _runtime()
        return ModelRegistryResponse(
            active_version=rt.registry.active_version,
            active_ready=rt.ready,
            active_error=None if rt.ready else rt.error,
            models=[_model_version_out(e) for e in rt.registry.entries],
        )

    @app.get(
        "/v1/models/{version}",
        response_model=ModelVersionOut,
        tags=["models"],
        operation_id="getModel",
        dependencies=caller,
        responses=_error_responses(ErrorCode.UNAUTHORIZED, ErrorCode.NOT_FOUND),
    )
    def get_model(version: str = Path(max_length=64)) -> ModelVersionOut:
        """One installed model version."""
        entry = _runtime().registry.get(version)
        if entry is None:
            raise LivenessError(ErrorCode.NOT_FOUND, "unknown model version")
        return _model_version_out(entry)

    # --- challenges ---------------------------------------------------------------------------

    @app.post(
        "/v1/liveness/challenges",
        response_model=ChallengeIssueResponse,
        status_code=201,
        tags=["liveness"],
        operation_id="issueChallenge",
        dependencies=caller,
        responses=_error_responses(ErrorCode.UNAUTHORIZED, ErrorCode.BUSY),
    )
    def issue_challenge() -> ChallengeIssueResponse:
        """Issue a short-lived, single-use challenge.

        Today this is a freshness / replay-prevention nonce only: no tested active
        provider is installed, so it has no instructions and does not make the check
        *active* liveness. Verify it with `POST /v1/liveness/challenges/{challenge_id}/verify`;
        the configured fusion policy (default `passive-only.v1`) decides.
        """
        try:
            issued = challenges.issue()
        except LivenessError as exc:
            metrics.challenges.labels(exc.code.value).inc()
            raise
        metrics.challenges.labels("issued").inc()
        instructions: list[ChallengeInstructionOut] = []
        if active.available and active.provider is not None:
            instructions = [
                ChallengeInstructionOut(action=i.action, timeout_seconds=i.timeout_seconds)
                for i in active.provider.instructions(issued.challenge_id)
            ]
        return ChallengeIssueResponse(
            challenge_id=issued.challenge_id,
            challenge_type="active_challenge" if instructions else "freshness_nonce",
            issued_at=issued.issued_at,
            expires_at=issued.expires_at,
            ttl_seconds=issued.ttl_seconds,
            instructions=instructions,
            active_liveness=bool(instructions),
            authoritative_method=PASSIVE_SINGLE_IMAGE,
        )

    # --- liveness -------------------------------------------------------------------------

    def evaluate_active(
        frames_b64: list[str], challenge_id: str, deadline: Deadline
    ) -> ActiveResult:
        provider = active.provider
        assert provider is not None  # checked by check_active_request
        frames = [_decode(f, settings) for f in frames_b64]
        deadline.check("active")
        try:
            result = provider.evaluate(challenge_id, frames)
        except LivenessError:
            raise
        except Exception as exc:
            log.exception("active provider evaluation failed")
            raise LivenessError(ErrorCode.INFERENCE_FAILED, "active evaluation failed") from exc
        desc = provider.descriptor
        if (result.provider_id, result.provider_version) != (desc.provider_id, desc.version):
            raise LivenessError(ErrorCode.INFERENCE_FAILED, "active result from wrong provider")
        if result.score is not None and not 0.0 <= result.score <= 1.0:
            raise LivenessError(ErrorCode.INFERENCE_FAILED, "active score out of range")
        return result

    def check_active_request(
        body: LivenessCheckRequest, challenge_id: str | None, policy: FusionPolicy
    ) -> None:
        frames = body.active_frames_base64
        if frames is None:
            if policy.requires_active:
                raise LivenessError(
                    ErrorCode.EVIDENCE_UNAVAILABLE,
                    f"fusion policy {policy.policy_id} requires an active capture "
                    "(challenge verify with active_frames_base64)",
                )
            return
        if challenge_id is None:
            raise LivenessError(
                ErrorCode.INVALID_REQUEST, "active frames are only accepted on challenge verify"
            )
        if not active.available:
            raise LivenessError(
                ErrorCode.EVIDENCE_UNAVAILABLE, f"active liveness unavailable: {active.reason}"
            )
        if len(frames) > settings.max_active_frames:
            raise LivenessError(
                ErrorCode.INVALID_REQUEST,
                f"at most {settings.max_active_frames} active frames are accepted",
            )

    def assess(
        body: LivenessCheckRequest,
        request: Request,
        challenge_id: str | None = None,
        tenant_id: str | None = None,
    ) -> LivenessCheckResponse:
        deadline = Deadline(
            settings.request_timeout_seconds,
            started_at=getattr(request.state, "received_at", None),
        )
        started = time.perf_counter()
        try:
            outcome = "error"
            # A policy that cannot be satisfied fails closed before any work is done and
            # before a challenge is spent.
            policy = require_ready(policy_state)
            check_active_request(body, challenge_id, policy)
            if challenge_id is not None:
                # Consumed first and unconditionally: any verify attempt spends the
                # challenge, so a replay can never obtain a second assessment.
                try:
                    challenges.consume(challenge_id)
                except LivenessError as exc:
                    metrics.challenges.labels(exc.code.value).inc()
                    raise
                metrics.challenges.labels("consumed").inc()
            engine: LivenessEngine = app.state.engine
            rt = engine.runtime
            if not rt.ready or rt.manifest is None:
                raise LivenessError(ErrorCode.MODEL_UNAVAILABLE, "liveness model is not available")
            raw = decode_base64(body.image_base64, settings.max_image_bytes)
            active_result: ActiveResult | None = None
            with admission.slot(timeout=min(settings.busy_timeout_seconds, deadline.remaining())):
                deadline.check("decode")
                image = decode_image(
                    raw,
                    max_pixels=settings.max_image_pixels,
                    min_side=settings.min_image_side_px,
                    max_side=settings.max_image_side_px,
                    max_decoded_bytes=settings.max_decoded_bytes,
                )
                del raw
                deadline.check("inference")
                result = engine.check(image)
                if body.active_frames_base64 is not None and challenge_id is not None:
                    active_result = evaluate_active(
                        body.active_frames_base64, challenge_id, deadline
                    )
            fused = fuse(policy, result, active_result)
            deadline.check("response")
            outcome = fused.decision.value
        except LivenessError as exc:
            outcome = exc.code.value
            raise
        finally:
            metrics.checks.labels(outcome).inc()
            request.state.outcome = outcome

        manifest = rt.manifest
        received = getattr(request.state, "received_at", None)
        metrics.inference_duration.observe(result.inference_seconds)
        metrics.live_score.labels(manifest.version).observe(result.live_score)
        metrics.decisions.labels(
            fused.decision.value, fused.reason.value, manifest.version, fused.policy_id
        ).inc()
        metrics.decision_duration.labels(fused.decision.value, manifest.version).observe(
            time.monotonic() - received if received is not None else time.perf_counter() - started
        )
        request.state.live_score = round(result.live_score, 4)
        request.state.model_version = manifest.version
        request.state.policy_id = fused.policy_id
        calibration_id = settings.threshold_calibration_id or None
        face = result.face
        live_score = round(result.live_score, 6)
        response = LivenessCheckResponse(
            request_id=request_id_var.get() or "",
            decision=fused.decision,
            is_live=fused.decision is Decision.LIVE,
            live_score=live_score,
            threshold=result.threshold,
            method=PASSIVE_SINGLE_IMAGE,
            face=FaceRegion(
                x=max(0, round(face.x)),
                y=max(0, round(face.y)),
                width=round(face.w),
                height=round(face.h),
                detection_score=round(min(max(face.score, 0.0), 1.0), 6),
            ),
            faces_detected=result.faces_detected,
            components=[
                ComponentScoreOut(name=c.name, live_score=round(c.live_score, 6))
                for c in result.components
            ],
            model=ModelRef(
                id=manifest.model_id,
                detector=manifest.detector.name,
                threshold_calibrated=calibration_id is not None,
                calibration_id=calibration_id,
            ),
            image=ImageInfo(width=image.width, height=image.height, media_type=image.media_type),
            processing_ms=round((time.perf_counter() - started) * 1000, 2),
            evidence=DecisionEvidence(
                model_id=manifest.model_id,
                model_version=manifest.version,
                model_digest=manifest.digest,
                detector_id=manifest.detector.name,
                liveness_type=manifest.liveness_type,
                active_liveness_evaluated=active_result is not None,
                score=live_score,
                score_aggregation=manifest.score_aggregation,
                threshold=result.threshold,
                threshold_calibrated=calibration_id is not None,
                calibration_id=calibration_id,
                margin=round(result.live_score - result.threshold, 6),
                policy_id=fused.policy_id,
                evidence_used=list(fused.evidence_used),
                decision=fused.decision,
                decision_reason=fused.reason,
                challenge=(
                    ChallengeEvidence(
                        challenge_id=challenge_id,
                        status="consumed",
                        active_liveness_evaluated=active_result is not None,
                    )
                    if challenge_id is not None
                    else None
                ),
                active=(
                    ActiveEvidence(
                        provider_id=active_result.provider_id,
                        provider_version=active_result.provider_version,
                        outcome=active_result.outcome,
                        score=active_result.score,
                    )
                    if active_result is not None
                    else None
                ),
            ),
        )
        tenant_id = _tenant(tenant_id, required=False)
        if tenant_id is not None:
            response.evidence_ref = _store_evidence(tenant_id, response)
        return response

    @app.post(
        "/v1/liveness/check",
        response_model=LivenessCheckResponse,
        tags=["liveness"],
        operation_id="checkLiveness",
        dependencies=caller,
        responses=_error_responses(*_ASSESSMENT_ERRORS),
    )
    def check_liveness(
        body: LivenessCheckRequest,
        request: Request,
        tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    ) -> LivenessCheckResponse:
        """Passive single-image liveness check.

        A `200` response is a completed assessment: `decision` is `live` or `spoof`.
        Any non-2xx response means **no assessment was made** and the caller must treat
        the subject as not verified (fail closed). `spoof` is a normal 200 outcome, not
        an error.
        """
        return assess(body, request, tenant_id=tenant_id)

    @app.post(
        "/v1/liveness/challenges/{challenge_id}/verify",
        response_model=LivenessCheckResponse,
        tags=["liveness"],
        operation_id="verifyChallenge",
        dependencies=caller,
        responses=_error_responses(
            *_ASSESSMENT_ERRORS,
            ErrorCode.CHALLENGE_NOT_FOUND,
            ErrorCode.CHALLENGE_EXPIRED,
            ErrorCode.CHALLENGE_ALREADY_USED,
        ),
    )
    def verify_challenge(
        body: LivenessCheckRequest,
        request: Request,
        challenge_id: str = Path(max_length=128),
        tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    ) -> LivenessCheckResponse:
        """Consume a challenge and run the passive check on the submitted capture.

        The challenge is spent by the first verify attempt whatever its outcome (issue a
        new one to retry). Unknown, expired or already-used challenges fail closed with
        no assessment. A `200` carries the same passive decision as `/v1/liveness/check`
        plus `evidence.challenge`. `active_frames_base64` is only accepted when a tested
        active provider is available (none is installed today, so it fails closed with
        `EVIDENCE_UNAVAILABLE`); `active_liveness_evaluated` is false without it.
        """
        return assess(body, request, challenge_id=challenge_id, tenant_id=tenant_id)

    @app.get(
        "/v1/liveness/evidence",
        response_model=LivenessEvidencePage,
        tags=["liveness"],
        operation_id="listLivenessEvidence",
        dependencies=caller,
        responses=_error_responses(ErrorCode.UNAUTHORIZED, ErrorCode.INVALID_REQUEST),
    )
    def list_liveness_evidence(
        tenant_id: str = Header(alias="X-Tenant-ID"),
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> LivenessEvidencePage:
        """Bounded, tenant-scoped readback of reference-only assessment summaries."""
        tenant = _tenant(tenant_id, required=True)
        assert tenant is not None
        with evidence_lock:
            matching = [summary for stored_tenant, summary in evidence_store if stored_tenant == tenant]
        matching.reverse()
        items = matching[offset : offset + limit]
        return LivenessEvidencePage(
            items=items, limit=limit, offset=offset, returned=len(items), total=len(matching)
        )

    @app.get(
        "/v1/liveness/evidence/{evidence_ref}",
        response_model=LivenessEvidenceSummary,
        tags=["liveness"],
        operation_id="getLivenessEvidence",
        dependencies=caller,
        responses=_error_responses(
            ErrorCode.UNAUTHORIZED, ErrorCode.INVALID_REQUEST, ErrorCode.NOT_FOUND
        ),
    )
    def get_liveness_evidence(
        evidence_ref: str = Path(pattern=r"^lev_[0-9a-f]{32}$"),
        tenant_id: str = Header(alias="X-Tenant-ID"),
    ) -> LivenessEvidenceSummary:
        """Read one assessment summary only when the tenant binding matches."""
        tenant = _tenant(tenant_id, required=True)
        assert tenant is not None
        with evidence_lock:
            for stored_tenant, summary in reversed(evidence_store):
                if summary.evidence_ref == evidence_ref and stored_tenant == tenant:
                    return summary
        raise LivenessError(ErrorCode.NOT_FOUND, "liveness evidence reference not found")

    return app
