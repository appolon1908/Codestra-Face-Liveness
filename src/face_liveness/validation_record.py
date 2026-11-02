"""Model validation records: the evidence a candidate version needs before activation.

A record is written by ``face-liveness-models validate`` (see promotion.py) to
``<model_registry_dir>/validations/<version>.json``. It binds one manifest digest to three
checks: artifact digests, a smoke test through the real runtime, and a reference to a
calibration report for exactly that manifest. The service only *reads* records; it never
writes, fetches or activates anything.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

VALIDATION_SCHEMA_VERSION: Literal[1] = 1
MAX_RECORD_BYTES = 64 * 1024
VALIDATIONS_SUBDIR = "validations"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DigestCheck(_Strict):
    passed: bool
    detail: str = Field(max_length=500)
    artifacts_verified: int = Field(ge=0)


class SmokeCheck(_Strict):
    passed: bool
    detail: str = Field(max_length=500)
    images: int = Field(ge=0)
    components: int = Field(ge=0)
    deterministic: bool
    max_latency_ms: float = Field(ge=0.0)


class CalibrationReference(_Strict):
    passed: bool
    detail: str = Field(max_length=500)
    calibration_id: str | None = Field(default=None, max_length=128)
    report_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    threshold: float | None = Field(default=None, gt=0.0, lt=1.0)
    apcer_max: float | None = Field(default=None, ge=0.0, le=1.0)
    bpcer: float | None = Field(default=None, ge=0.0, le=1.0)


class ActivationPlan(_Strict):
    auto_activated: Literal[False] = Field(
        default=False, description="Always false: activation is a local operator action."
    )
    env: dict[str, str] = Field(
        description="Deployment settings that activate this version (empty if not passed)."
    )


class ValidationRecord(_Strict):
    schema_version: Literal[1]
    record_type: Literal["face_liveness_model_validation"]
    model_id: str = Field(max_length=64)
    version: str = Field(max_length=64)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validated_at: datetime
    tool_version: str
    digests: DigestCheck
    smoke: SmokeCheck
    calibration: CalibrationReference
    passed: bool = Field(description="True only if every check passed.")
    activation: ActivationPlan


def record_path(registry_dir: Path, version: str) -> Path:
    return registry_dir / VALIDATIONS_SUBDIR / f"{version}.json"


def load_record(path: Path) -> ValidationRecord:
    """Parse one record. Raises ValueError with a safe message."""
    try:
        if path.stat().st_size > MAX_RECORD_BYTES:
            raise ValueError(f"validation record exceeds {MAX_RECORD_BYTES} bytes")
        data = json.loads(path.read_bytes())
    except FileNotFoundError as exc:
        raise ValueError("no validation record") from exc
    except OSError as exc:
        raise ValueError(f"validation record unreadable: {type(exc).__name__}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("validation record is not valid JSON") from exc
    try:
        return ValidationRecord.model_validate(data)
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) or "record" for e in exc.errors()})
        raise ValueError("validation record schema invalid: " + ", ".join(fields)) from exc


def candidate_error(record: ValidationRecord, version: str, manifest_sha256: str) -> str | None:
    """Why ``record`` does not make this exact manifest a candidate (None if it does)."""
    if record.version != version or record.manifest_sha256 != manifest_sha256:
        return "validation record is for a different manifest"
    if not record.passed:
        return "validation record did not pass"
    if not (record.digests.passed and record.smoke.passed and record.calibration.passed):
        return "validation record has a failed check"
    return None


def activation_error(
    record: ValidationRecord,
    version: str,
    manifest_sha256: str,
    live_threshold: float,
    calibration_id: str,
) -> str | None:
    """Why the deployment may not activate this version with these settings (None if ok)."""
    error = candidate_error(record, version, manifest_sha256)
    if error is not None:
        return error
    cal = record.calibration
    if not calibration_id or cal.calibration_id != calibration_id:
        return "LIVENESS_THRESHOLD_CALIBRATION_ID does not match the validated calibration"
    if cal.threshold is None or live_threshold < cal.threshold:
        return "LIVENESS_LIVE_THRESHOLD is below the validated threshold candidate"
    return None
