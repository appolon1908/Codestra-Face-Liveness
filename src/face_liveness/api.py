"""FastAPI application factory: routes, auth, error envelope, metrics, access logs."""

from __future__ import annotations

import hmac
import logging
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import SERVICE_NAME, __version__
from .config import Settings
from .engine import Decision, LivenessEngine
from .errors import HTTP_STATUS, ErrorCode, LivenessError
from .imaging import ALLOWED_FORMATS, decode_base64, decode_image
from .logging_config import request_id_var
from .metrics import Metrics
from .runtime import DETECTOR_ID, MODEL_ID, ModelRuntime, load_runtime
from .schemas import (
    PASSIVE_SINGLE_IMAGE,
    ArtifactOut,
    CapabilitiesResponse,
    ComponentScoreOut,
    DecisionPolicy,
    ErrorResponse,
    FaceRegion,
    HealthResponse,
    ImageInfo,
    InputLimits,
    LivenessCheckRequest,
    LivenessCheckResponse,
    MethodInfo,
    ModelInfo,
    ModelRef,
    ReadinessResponse,
    StatusResponse,
)

log = logging.getLogger("face_liveness.api")

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")


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


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Rejects request bodies above a byte limit, with or without Content-Length."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

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
        response = _error_response(ErrorCode.PAYLOAD_TOO_LARGE, "request body too large")
        await response(scope, receive, send)


def create_app(settings: Settings | None = None, runtime: ModelRuntime | None = None) -> FastAPI:
    settings = settings or Settings()
    metrics = Metrics()
    started_at = time.monotonic()
    check_slots = threading.BoundedSemaphore(settings.max_concurrent_checks)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.runtime is None:
            app.state.runtime = load_runtime(settings)
        rt: ModelRuntime = app.state.runtime
        app.state.engine = LivenessEngine(settings, rt)
        metrics.model_ready.set(1 if rt.ready else 0)
        metrics.threshold.set(settings.live_threshold)
        metrics.build.info({"version": __version__, "model": MODEL_ID, "detector": DETECTOR_ID})
        if rt.ready:
            log.info("model runtime ready", extra={"model": MODEL_ID})
        else:
            log.error("model runtime NOT ready; failing closed", extra={"reason": rt.error})
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
            "NOT perform face matching or identity verification. The only supported caller "
            "is Middleware V3, which owns authentication normalisation, policy, orchestration "
            "and cross-system audit. Images are processed in memory and never persisted."
        ),
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        servers=[{"url": "http://localhost:8080", "description": "Local development"}],
    )
    app.state.runtime = runtime
    app.state.settings = settings
    app.state.metrics = metrics
    app.state.check_slots = check_slots

    # --- error envelope -----------------------------------------------------------------

    @app.exception_handler(LivenessError)
    async def _liveness_error(request: Request, exc: LivenessError) -> JSONResponse:
        request.state.outcome = exc.code.value
        return _error_response(exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Never echo the submitted input (it may contain image data).
        fields = sorted(
            {".".join(str(p) for p in e.get("loc", ())[1:]) or "body" for e in exc.errors()}
        )
        request.state.outcome = ErrorCode.INVALID_REQUEST.value
        return _error_response(ErrorCode.INVALID_REQUEST, "invalid request: " + ", ".join(fields))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: ErrorCode.NOT_FOUND, 405: ErrorCode.METHOD_NOT_ALLOWED}.get(
            exc.status_code, ErrorCode.INVALID_REQUEST
        )
        return _error_response(code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return _error_response(ErrorCode.INTERNAL_ERROR, "internal error")

    # --- request context, access log, HTTP metrics ---------------------------------------

    @app.middleware("http")
    async def _observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
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
            route = request.scope.get("route")
            template = getattr(route, "path", "unmatched")
            metrics.http_requests.labels(template, request.method, str(status)).inc()
            metrics.http_duration.labels(template, request.method).observe(elapsed)
            if template not in ("/healthz", "/readyz", "/metrics"):
                extra: dict[str, Any] = {
                    "method": request.method,
                    "route": template,
                    "status": status,
                    "duration_ms": round(elapsed * 1000, 2),
                }
                for key in ("outcome", "live_score"):
                    if hasattr(request.state, key):
                        extra[key] = getattr(request.state, key)
                log.info("request completed", extra=extra)
            request_id_var.reset(token)

    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)

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

    @app.get("/health/ready", include_in_schema=True, tags=["operations"], operation_id="healthReadyAlias")
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
        calibration_id = settings.threshold_calibration_id or None
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
            unsupported=[
                "active_challenge",
                "video_sequence",
                "depth_or_ir",
                "face_matching",
                "identity_verification",
            ],
            input=InputLimits(
                formats=sorted(ALLOWED_FORMATS.values()),
                max_image_bytes=settings.max_image_bytes,
                max_image_pixels=settings.max_image_pixels,
                min_image_side_px=settings.min_image_side_px,
                min_face_size_px=settings.min_face_size_px,
            ),
            decision=DecisionPolicy(
                score="mean of MiniFASNet softmax[live] over the ensemble",
                threshold=settings.live_threshold,
                threshold_calibrated=calibration_id is not None,
                calibration_id=calibration_id,
            ),
            model=ModelInfo(
                id=MODEL_ID,
                detector=DETECTOR_ID,
                license="Apache-2.0 (MiniFASNet weights), MIT (YuNet)",
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
        return StatusResponse(
            service=SERVICE_NAME,
            version=__version__,
            environment=settings.env.value,
            ready=not failures,
            reason="; ".join(failures) or None,
            uptime_seconds=round(time.monotonic() - started_at, 3),
        )

    # --- liveness -------------------------------------------------------------------------

    @app.post(
        "/v1/liveness/check",
        response_model=LivenessCheckResponse,
        tags=["liveness"],
        operation_id="checkLiveness",
        dependencies=caller,
        responses=_error_responses(
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
            ErrorCode.INFERENCE_FAILED,
        ),
    )
    def check_liveness(body: LivenessCheckRequest, request: Request) -> LivenessCheckResponse:
        """Passive single-image liveness check.

        A `200` response is a completed assessment: `decision` is `live` or `spoof`.
        Any non-2xx response means **no assessment was made** and the caller must treat
        the subject as not verified (fail closed). `spoof` is a normal 200 outcome, not
        an error.
        """
        started = time.perf_counter()
        try:
            outcome = "error"
            engine: LivenessEngine = app.state.engine
            if not engine.runtime.ready:
                raise LivenessError(ErrorCode.MODEL_UNAVAILABLE, "liveness model is not available")
            raw = decode_base64(body.image_base64, settings.max_image_bytes)
            image = decode_image(
                raw, max_pixels=settings.max_image_pixels, min_side=settings.min_image_side_px
            )
            del raw
            if not check_slots.acquire(timeout=settings.busy_timeout_seconds):
                raise LivenessError(ErrorCode.BUSY, "service at capacity; retry shortly")
            try:
                result = engine.check(image)
            finally:
                check_slots.release()
            outcome = result.decision.value
        except LivenessError as exc:
            outcome = exc.code.value
            raise
        finally:
            metrics.checks.labels(outcome).inc()
            request.state.outcome = outcome

        metrics.inference_duration.observe(result.inference_seconds)
        metrics.live_score.observe(result.live_score)
        request.state.live_score = round(result.live_score, 4)
        calibration_id = settings.threshold_calibration_id or None
        face = result.face
        return LivenessCheckResponse(
            request_id=request_id_var.get() or "",
            decision=result.decision,
            is_live=result.decision is Decision.LIVE,
            live_score=round(result.live_score, 6),
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
                id=MODEL_ID,
                detector=DETECTOR_ID,
                threshold_calibrated=calibration_id is not None,
                calibration_id=calibration_id,
            ),
            image=ImageInfo(width=image.width, height=image.height, media_type=image.media_type),
            processing_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    return app
