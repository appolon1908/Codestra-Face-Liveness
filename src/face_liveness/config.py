"""Service configuration, sourced exclusively from environment variables.

Secrets are never given defaults and are never committed. The API token is read from a
file reference (``LIVENESS_API_TOKEN_FILE``, e.g. rendered by an OpenBao agent) or, for
local development only, from ``LIVENESS_API_TOKEN``.
"""

from __future__ import annotations

from enum import StrEnum
from functools import cached_property
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Known-good SHA-256 digests of the runtime model artifacts produced by
# tools/fetch_models.sh + tools/convert_minifasnet.py (see docs/MODEL_CARD.md).
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MINIFASNET_V2_SHA256 = "98587ea6a3315f4d0bfcfb9b247ab7f724f3b63ae05f4a0b5a1fd53f414b4571"
MINIFASNET_V1SE_SHA256 = "09e486ea92376a13a50f943ca2a5b41a3c7d4d6b1761c8e87c63eb1205fefb1a"


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIVENESS_", extra="ignore")

    env: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"

    # Model artifacts
    model_dir: Path = Path("/models")
    detector_file: str = "face_detection_yunet_2023mar.onnx"
    detector_sha256: str = YUNET_SHA256
    minifasnet_v2_file: str = "minifasnet_v2_2.7_80x80.onnx"
    minifasnet_v2_sha256: str = MINIFASNET_V2_SHA256
    minifasnet_v1se_file: str = "minifasnet_v1se_4.0_80x80.onnx"
    minifasnet_v1se_sha256: str = MINIFASNET_V1SE_SHA256
    # When true, a digest mismatch keeps the service not-ready (fail closed).
    # Forced on in production.
    verify_model_digests: bool = True
    onnx_intra_op_threads: int = Field(default=2, ge=1, le=64)
    # Concurrent checks per process; excess requests wait up to busy_timeout_seconds
    # and then get a retryable 503 BUSY.
    max_concurrent_checks: int = Field(default=4, ge=1, le=256)
    busy_timeout_seconds: float = Field(default=2.0, ge=0.0, le=60.0)

    # Decision policy
    live_threshold: float = Field(default=0.85, gt=0.0, lt=1.0)
    # Opaque identifier of the calibration run that produced live_threshold.
    # Empty means the threshold is the uncalibrated provisional default.
    threshold_calibration_id: str = ""

    # Face detection / quality gates
    detector_score_threshold: float = Field(default=0.8, gt=0.0, lt=1.0)
    detector_max_side: int = Field(default=640, ge=160, le=4096)
    min_face_size_px: int = Field(default=64, ge=16)
    # Secondary faces smaller than this fraction of the primary face's area are ignored
    # (e.g. a tiny face on a badge in the background). 0 disables tolerance entirely.
    secondary_face_area_ratio: float = Field(default=0.0, ge=0.0, lt=1.0)

    # Input limits
    max_image_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)
    max_image_pixels: int = Field(default=25_000_000, ge=10_000)
    min_image_side_px: int = Field(default=112, ge=32)

    # Caller authentication (the only expected caller is Middleware V3)
    api_token: SecretStr | None = None
    api_token_file: Path | None = None
    require_auth: bool | None = None  # default: True in production, False otherwise

    @model_validator(mode="after")
    def _production_guards(self) -> Settings:
        if self.env is Environment.PRODUCTION:
            self.verify_model_digests = True
            if self.api_token is not None:
                raise ValueError(
                    "LIVENESS_API_TOKEN is development-only; "
                    "use LIVENESS_API_TOKEN_FILE in production"
                )
        return self

    @property
    def auth_required(self) -> bool:
        if self.require_auth is not None:
            return self.require_auth
        return self.env is Environment.PRODUCTION

    @cached_property
    def resolved_api_token(self) -> str | None:
        """Token from file reference (preferred) or env. None if unset/unreadable."""
        if self.api_token_file is not None:
            try:
                token = self.api_token_file.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            return token or None
        if self.api_token is not None:
            return self.api_token.get_secret_value() or None
        return None

    @property
    def max_request_bytes(self) -> int:
        # base64 inflates by 4/3; allow headroom for the JSON envelope.
        return (self.max_image_bytes * 4) // 3 + 16 * 1024
