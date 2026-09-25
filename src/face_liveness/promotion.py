"""Safe model promotion: installed -> candidate -> active, as local operator actions.

    face-liveness-models list     --model-dir ./models
    face-liveness-models validate 1.1.0 --calibration-report report.json --model-dir ./models
    face-liveness-models activate 1.1.0 --model-dir ./models [--env-file deploy.env]

``validate`` runs three checks against one installed version and writes a validation
record (``<registry dir>/validations/<version>.json``):

1. digests    every artifact exists under the model dir and matches its manifest SHA-256
              (digest verification cannot be disabled for validation);
2. smoke      the version loads through the production runtime, detection runs, every
              classifier returns a finite score in [0, 1], and scoring is deterministic;
3. calibration  a calibration report (face-liveness-calibrate) exists for exactly this
              manifest digest and the current pipeline settings, is eligible for review,
              and has a threshold candidate.

A version with a passing record is a *candidate*. ``activate`` re-verifies the record
against the installed manifest and prints (or writes to a local env file) the three
deployment settings that activate it. Nothing here downloads anything, talks to the
service, or changes a running deployment: activation takes effect only when the
operator deploys those settings. Artifacts must already be local files; there is no
model URL anywhere in the manifest schema or this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from . import __version__
from .calibration import CalibrationReport, PipelineSettings
from .config import Environment, Settings
from .inference import FaceBox
from .registry import EntryStatus, ModelManifest, ModelRegistry, RegistryEntry, build_registry
from .runtime import ModelRuntime, load_runtime
from .validation_record import (
    VALIDATION_SCHEMA_VERSION,
    ActivationPlan,
    CalibrationReference,
    DigestCheck,
    SmokeCheck,
    ValidationRecord,
    activation_error,
    load_record,
    record_path,
)

MAX_REPORT_BYTES = 4 * 1024 * 1024
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-]{0,63}$")
ACTIVATION_KEYS = (
    "LIVENESS_ACTIVE_MODEL_VERSION",
    "LIVENESS_LIVE_THRESHOLD",
    "LIVENESS_THRESHOLD_CALIBRATION_ID",
)


# --- checks ---------------------------------------------------------------------------------


def check_digests(entry: RegistryEntry | None, settings: Settings) -> DigestCheck:
    if entry is None or entry.manifest is None:
        return DigestCheck(passed=False, detail="version not installed", artifacts_verified=0)
    if entry.error is not None:
        return DigestCheck(passed=False, detail=entry.error, artifacts_verified=0)
    if not settings.verify_model_digests or not entry.digests_verified:
        return DigestCheck(passed=False, detail="digests not verified", artifacts_verified=0)
    n = len(entry.manifest.artifacts)
    return DigestCheck(passed=True, detail=f"{n} artifacts match", artifacts_verified=n)


def _smoke_images() -> list[np.ndarray]:
    rng = np.random.default_rng(20260924)
    blank = np.full((240, 320, 3), 128, dtype=np.uint8)
    noise = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    ramp = np.broadcast_to(
        np.linspace(0, 255, 400, dtype=np.uint8)[np.newaxis, :, np.newaxis], (300, 400, 3)
    ).copy()
    return [blank, noise, ramp]


def check_smoke(runtime: ModelRuntime) -> SmokeCheck:
    """Exercise the loaded runtime on synthetic images. Never raises."""
    if not runtime.ready or runtime.detector is None or runtime.classifier is None:
        return SmokeCheck(
            passed=False,
            detail=f"runtime not ready: {runtime.error}",
            images=0,
            components=0,
            deterministic=False,
            max_latency_ms=0.0,
        )
    expected = len(runtime.manifest.classifiers) if runtime.manifest else 0
    images = _smoke_images()
    worst = 0.0
    deterministic = True
    components = 0
    try:
        for bgr in images:
            h, w = bgr.shape[:2]
            started = time.perf_counter()
            runtime.detector.detect(bgr)
            # Score a centred box whether or not a face was found: the classifier must
            # behave on any crop the detector could hand it.
            side = min(h, w) / 2
            box = FaceBox(x=(w - side) / 2, y=(h - side) / 2, w=side, h=side, score=1.0)
            first = runtime.classifier.score(bgr, box)
            worst = max(worst, (time.perf_counter() - started) * 1000)
            second = runtime.classifier.score(bgr, box)
            components = len(first)
            if components != expected:
                raise ValueError(f"expected {expected} component scores, got {components}")
            for a, b in zip(first, second, strict=True):
                if not (math.isfinite(a.live_score) and 0.0 <= a.live_score <= 1.0):
                    raise ValueError(f"{a.name}: score out of range")
                deterministic &= abs(a.live_score - b.live_score) <= 1e-6
    except Exception as exc:
        return SmokeCheck(
            passed=False,
            detail=f"smoke inference failed: {type(exc).__name__}: {exc}"[:500],
            images=len(images),
            components=components,
            deterministic=False,
            max_latency_ms=round(worst, 3),
        )
    return SmokeCheck(
        passed=deterministic,
        detail="ok" if deterministic else "non-deterministic scores",
        images=len(images),
        components=components,
        deterministic=deterministic,
        max_latency_ms=round(worst, 3),
    )


def _pipeline(settings: Settings) -> PipelineSettings:
    return PipelineSettings(
        detector_score_threshold=settings.detector_score_threshold,
        min_face_size_px=settings.min_face_size_px,
        min_image_side_px=settings.min_image_side_px,
        secondary_face_area_ratio=settings.secondary_face_area_ratio,
    )


def check_calibration(
    report_path: Path | None, manifest: ModelManifest | None, settings: Settings
) -> CalibrationReference:
    def fail(detail: str, **kw: object) -> CalibrationReference:
        return CalibrationReference(passed=False, detail=detail, **kw)  # type: ignore[arg-type]

    if report_path is None:
        return fail("no calibration report given")
    if manifest is None:
        return fail("version not installed")
    try:
        if report_path.stat().st_size > MAX_REPORT_BYTES:
            return fail(f"calibration report exceeds {MAX_REPORT_BYTES} bytes")
        raw = report_path.read_bytes()
        report = CalibrationReport.model_validate_json(raw)
    except OSError as exc:
        return fail(f"calibration report unreadable: {type(exc).__name__}")
    except ValidationError:
        return fail("calibration report does not match schema v1")
    sha = hashlib.sha256(raw).hexdigest()
    ref = {"report_sha256": sha, "calibration_id": report.calibration_id}
    if report.model.version != manifest.version or report.model.manifest_sha256 != manifest.digest:
        return fail("calibration report is for a different model version or manifest", **ref)
    if report.pipeline != _pipeline(settings):
        return fail("calibration report pipeline settings differ from current settings", **ref)
    candidate = report.threshold_candidate
    if not report.promotion.eligible_for_review or candidate is None:
        blockers = "; ".join(report.promotion.blockers) or "no threshold candidate"
        return fail(f"calibration report not eligible: {blockers}", **ref)
    return CalibrationReference(
        passed=True,
        detail="ok",
        threshold=candidate.threshold,
        apcer_max=candidate.apcer_max,
        bpcer=candidate.bpcer,
        **ref,
    )


def _smoke_settings(settings: Settings, version: str) -> Settings:
    # Load the candidate itself, with digest verification on and without the activation
    # gate (which it cannot pass before it has been validated).
    return settings.model_copy(
        update={
            "active_model_version": version,
            "require_model_validation": False,
            "verify_model_digests": True,
            "env": Environment.TEST,
        }
    )


def validate_version(
    settings: Settings,
    version: str,
    report_path: Path | None,
    now: datetime | None = None,
) -> ValidationRecord:
    registry = build_registry(_smoke_settings(settings, version))
    entry = registry.get(version)
    manifest = entry.manifest if entry is not None else None
    digests = check_digests(entry, settings)
    if digests.passed:
        smoke = check_smoke(load_runtime(_smoke_settings(settings, version)))
    else:
        smoke = SmokeCheck(
            passed=False,
            detail="skipped: digest check failed",
            images=0,
            components=0,
            deterministic=False,
            max_latency_ms=0.0,
        )
    calibration = check_calibration(report_path, manifest, settings)
    passed = digests.passed and smoke.passed and calibration.passed
    env: dict[str, str] = {}
    if passed and calibration.calibration_id and calibration.threshold is not None:
        env = dict(
            zip(
                ACTIVATION_KEYS,
                (version, repr(calibration.threshold), calibration.calibration_id),
                strict=True,
            )
        )
    return ValidationRecord(
        schema_version=VALIDATION_SCHEMA_VERSION,
        record_type="face_liveness_model_validation",
        model_id=manifest.model_id if manifest else "unknown",
        version=version,
        manifest_sha256=manifest.digest if manifest else "0" * 64,
        validated_at=now or datetime.now(UTC),
        tool_version=__version__,
        digests=digests,
        smoke=smoke,
        calibration=calibration,
        passed=passed,
        activation=ActivationPlan(env=env),
    )


def activation_env(settings: Settings, version: str) -> dict[str, str]:
    """Deployment settings for a validated candidate. Raises ValueError if not allowed."""
    registry = build_registry(_smoke_settings(settings, version))
    entry = registry.get(version)
    if entry is None or entry.manifest is None or entry.error is not None:
        raise ValueError(f"version not installed or invalid: {version}")
    record = load_record(record_path(settings.resolved_model_registry_dir, version))
    env = record.activation.env
    threshold = float(env.get("LIVENESS_LIVE_THRESHOLD", "nan"))
    error = activation_error(
        record,
        version,
        entry.manifest.digest,
        threshold,
        env.get("LIVENESS_THRESHOLD_CALIBRATION_ID", ""),
    )
    if error is None and not entry.digests_verified:
        error = "artifact digests not verified"
    if error is not None:
        raise ValueError(error)
    return {k: env[k] for k in ACTIVATION_KEYS}


def update_env_file(path: Path, values: dict[str, str]) -> None:
    """Set ``values`` in a local KEY=VALUE file, keeping every other line as is."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    out = [ln for ln in lines if ln.partition("=")[0].strip() not in values]
    out += [f"{k}={v}" for k, v in values.items()]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def registry_summary(registry: ModelRegistry) -> list[dict[str, object]]:
    return [
        {
            "version": e.version,
            "source": e.source,
            "status": e.status.value,
            "manifest_sha256": e.manifest_sha256,
            "digests_verified": e.digests_verified,
            "error": e.error,
            "validation_error": e.validation_error if e.status is not EntryStatus.INVALID else None,
        }
        for e in registry.entries
    ]


# --- CLI --------------------------------------------------------------------------------


def _version(value: str) -> str:
    if not _VERSION_RE.match(value):
        raise argparse.ArgumentTypeError("not an installed version id (URLs are not accepted)")
    return value


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="face-liveness-models",
        description="Local model promotion: list, validate a candidate, print activation.",
    )
    ap.add_argument("--model-dir", type=Path, default=Path("models"))
    ap.add_argument("--registry-dir", type=Path, default=None)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="installed versions and their promotion state")
    v = sub.add_parser("validate", help="digest + smoke + calibration checks; writes a record")
    v.add_argument("version", type=_version)
    v.add_argument("--calibration-report", type=Path, required=True)
    v.add_argument("--out", type=Path, default=None, help="default: <registry>/validations/")
    a = sub.add_parser("activate", help="print the deployment settings for a candidate")
    a.add_argument("version", type=_version)
    a.add_argument("--env-file", type=Path, default=None, help="also write them to this file")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Same pipeline settings as the service (LIVENESS_* env), pointed at the given models.
    settings = Settings(
        env=Environment.TEST,
        model_dir=args.model_dir,
        model_registry_dir=args.registry_dir,
        verify_model_digests=True,
    )

    if args.command == "list":
        print(json.dumps(registry_summary(build_registry(settings)), indent=2))
        return 0

    if args.command == "validate":
        record = validate_version(settings, args.version, args.calibration_report)
        out = args.out or record_path(settings.resolved_model_registry_dir, args.version)
        out.parent.mkdir(parents=True, exist_ok=True)
        text = record.model_dump_json(indent=2)
        out.write_text(text + "\n", encoding="utf-8")
        print(text)
        for name in ("digests", "smoke", "calibration"):
            check = getattr(record, name)
            if not check.passed:
                print(f"NOT a candidate: {name}: {check.detail}", file=sys.stderr)
        return 0 if record.passed else 1

    try:
        env = activation_env(settings, args.version)
    except ValueError as exc:
        print(f"refusing to activate {args.version}: {exc}", file=sys.stderr)
        return 1
    for key, value in env.items():
        print(f"{key}={value}")
    if args.env_file is not None:
        update_env_file(args.env_file, env)
        print(f"wrote {args.env_file} (takes effect on the next deployment)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
