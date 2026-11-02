"""Offline threshold calibration against a labelled, locally held dataset.

    face-liveness-calibrate --data ./calib --model-dir ./models --out report.json

Dataset layout (images are read in place, never copied or uploaded):

    calib/bona_fide/**/*.jpg          genuine live captures from the target camera(s)
    calib/attack/<species>/**/*.jpg   presentation attacks grouped by attack species
                                      (e.g. print, replay_phone, replay_monitor, mask)

Every image goes through exactly the production pipeline (same input limits, detector,
face gates and active model version). Images rejected before scoring (no face, multiple
faces, too small, ...) are counted separately: production would never score them, so
they count neither as live nor as spoof.

Metrics follow ISO/IEC 30107-3:
    APCER  attack presentations accepted as bona fide (per species; worst reported).
           The FAR-like "spoof accepted" rate.
    BPCER  bona fide presentations rejected as attacks. The FRR-like "live rejected" rate.

The threshold candidate is the smallest threshold >= ``--min-threshold`` whose worst-species
APCER <= ``--target-apcer``. The report is evidence for a human release decision only:
this tool never changes service configuration, and the service never reads reports.
``promotion.auto_promoted`` is always false by schema.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from . import __version__
from .config import Environment, Settings
from .engine import LivenessEngine
from .errors import ErrorCode, LivenessError
from .imaging import decode_image
from .registry import ModelManifest
from .runtime import load_runtime

REPORT_SCHEMA_VERSION: Literal[1] = 1
REFERENCE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_SPECIES_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,63}$")


class Label(StrEnum):
    BONA_FIDE = "bona_fide"
    ATTACK = "attack"


@dataclass(frozen=True, slots=True)
class DatasetItem:
    path: Path
    label: Label
    species: str | None


@dataclass(frozen=True, slots=True)
class ScoredSample:
    label: Label
    species: str | None
    sha256: str
    score: float | None  # None when rejected before scoring
    rejection: str | None = None


# --- report schema --------------------------------------------------------------------------


class OperatingPoint(BaseModel):
    threshold: float
    apcer_max: float = Field(
        ge=0.0, le=1.0, description="Worst-species APCER: FAR-like spoof acceptance rate."
    )
    apcer_by_species: dict[str, float]
    bpcer: float = Field(ge=0.0, le=1.0, description="BPCER: FRR-like live rejection rate.")


class ScoreSummary(BaseModel):
    count: int
    min: float | None
    p05: float | None
    median: float | None
    p95: float | None
    max: float | None
    mean: float | None


class ModelIdentity(BaseModel):
    model_id: str
    version: str
    manifest_sha256: str
    detector_id: str


class PipelineSettings(BaseModel):
    """Settings that change which images are scored; must match the target deployment."""

    detector_score_threshold: float
    min_face_size_px: int
    min_image_side_px: int
    secondary_face_area_ratio: float


class DatasetInfo(BaseModel):
    fingerprint_sha256: str = Field(
        description="SHA-256 over the sorted (label, species, image SHA-256) triples. "
        "Independent of file names and order; changes if any image or label changes."
    )
    images_total: int
    bona_fide_scored: int
    attack_scored: dict[str, int]
    rejected_before_scoring: dict[str, int] = Field(
        description="Counts keyed by `<label or species>:<error code>`."
    )


class CalibrationPolicy(BaseModel):
    target_apcer: float
    min_threshold: float
    threshold_step: float
    min_bona_fide: int
    min_per_species: int


class Promotion(BaseModel):
    auto_promoted: Literal[False] = Field(
        default=False,
        description="Always false: reports are never applied automatically.",
    )
    status: Literal["candidate_only"] = "candidate_only"
    eligible_for_review: bool
    blockers: list[str]
    note: str


class CalibrationReport(BaseModel):
    schema_version: Literal[1]
    report_type: Literal["face_liveness_calibration"]
    calibration_id: str
    created_at: datetime
    tool_version: str
    model: ModelIdentity
    pipeline: PipelineSettings
    dataset: DatasetInfo
    policy: CalibrationPolicy
    score_summary: dict[Label, ScoreSummary]
    equal_error_rate: OperatingPoint = Field(
        description="Grid point where worst-species APCER and BPCER are closest (approx EER)."
    )
    operating_points: list[OperatingPoint]
    threshold_candidate: OperatingPoint | None = Field(
        description="Smallest threshold meeting the APCER target; null if none does."
    )
    sufficient_sample_size: bool
    promotion: Promotion


# --- dataset ----------------------------------------------------------------------------


def _images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def discover_dataset(root: Path) -> list[DatasetItem]:
    bona_dir, attack_dir = root / Label.BONA_FIDE.value, root / Label.ATTACK.value
    if not bona_dir.is_dir() or not attack_dir.is_dir():
        raise ValueError(f"dataset must contain {bona_dir.name}/ and {attack_dir.name}/<species>/")
    items = [DatasetItem(p, Label.BONA_FIDE, None) for p in _images(bona_dir)]
    for species_dir in sorted(d for d in attack_dir.iterdir() if d.is_dir()):
        if not _SPECIES_RE.match(species_dir.name):
            raise ValueError(f"invalid attack species directory name: {species_dir.name!r}")
        items += [DatasetItem(p, Label.ATTACK, species_dir.name) for p in _images(species_dir)]
    return items


def score_dataset(
    engine: LivenessEngine, settings: Settings, items: Sequence[DatasetItem]
) -> list[ScoredSample]:
    """Score every image through the production decode + detection + scoring path."""
    out: list[ScoredSample] = []
    for item in items:
        if item.path.stat().st_size > settings.max_image_bytes:
            with item.path.open("rb") as fh:
                digest = hashlib.file_digest(fh, "sha256").hexdigest()
            code = ErrorCode.PAYLOAD_TOO_LARGE.value
            out.append(ScoredSample(item.label, item.species, digest, None, code))
            continue
        raw = item.path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        try:
            image = decode_image(
                raw,
                max_pixels=settings.max_image_pixels,
                min_side=settings.min_image_side_px,
                max_side=settings.max_image_side_px,
                max_decoded_bytes=settings.max_decoded_bytes,
            )
            score = engine.check(image).live_score
        except LivenessError as exc:
            out.append(ScoredSample(item.label, item.species, sha, None, exc.code.value))
            continue
        out.append(ScoredSample(item.label, item.species, sha, score))
    return out


def dataset_fingerprint(samples: Sequence[ScoredSample]) -> str:
    lines = sorted(f"{s.label.value}\t{s.species or ''}\t{s.sha256}\n" for s in samples)
    return hashlib.sha256("".join(lines).encode()).hexdigest()


# --- metrics ----------------------------------------------------------------------------


class _Rates:
    def __init__(self, bona_fide: list[float], attacks: dict[str, list[float]]) -> None:
        self._bona = np.sort(np.asarray(bona_fide, dtype=np.float64))
        self._attacks = {k: np.sort(np.asarray(v, dtype=np.float64)) for k, v in attacks.items()}

    def at(self, threshold: float) -> OperatingPoint:
        # Service rule: live iff score >= threshold.
        bpcer = float(np.searchsorted(self._bona, threshold, side="left")) / len(self._bona)
        apcer = {
            k: float(len(v) - np.searchsorted(v, threshold, side="left")) / len(v)
            for k, v in sorted(self._attacks.items())
        }
        return OperatingPoint(
            threshold=threshold,
            apcer_max=max(apcer.values()),
            apcer_by_species={k: round(v, 6) for k, v in apcer.items()},
            bpcer=round(bpcer, 6),
        )


def _grid(lo: float, step: float) -> list[float]:
    n = round((1.0 - lo) / step)
    return [t for t in (round(lo + i * step, 6) for i in range(n + 1)) if t < 1.0]


def _summary(scores: Sequence[float]) -> ScoreSummary:
    if not scores:
        return ScoreSummary(count=0, min=None, p05=None, median=None, p95=None, max=None, mean=None)
    arr = np.asarray(scores, dtype=np.float64)
    q = np.quantile(arr, [0.05, 0.5, 0.95])
    return ScoreSummary(
        count=len(scores),
        min=round(float(arr.min()), 6),
        p05=round(float(q[0]), 6),
        median=round(float(q[1]), 6),
        p95=round(float(q[2]), 6),
        max=round(float(arr.max()), 6),
        mean=round(float(arr.mean()), 6),
    )


def build_report(
    samples: Sequence[ScoredSample],
    manifest: ModelManifest,
    settings: Settings,
    policy: CalibrationPolicy,
    created_at: datetime | None = None,
) -> CalibrationReport:
    bona = [s.score for s in samples if s.label is Label.BONA_FIDE and s.score is not None]
    attacks: dict[str, list[float]] = {}
    for s in samples:
        if s.label is Label.ATTACK and s.score is not None:
            attacks.setdefault(s.species or "unspecified", []).append(s.score)
    if not bona or not attacks:
        raise ValueError("need at least one scored bona fide image and one scored attack species")

    rates = _Rates(bona, attacks)
    candidate = next(
        (
            op
            for op in (rates.at(t) for t in _grid(policy.min_threshold, policy.threshold_step))
            if op.apcer_max <= policy.target_apcer
        ),
        None,
    )
    eer = min(
        (rates.at(t) for t in _grid(0.0, policy.threshold_step)),
        key=lambda op: (abs(op.apcer_max - op.bpcer), op.threshold),
    )
    sufficient = len(bona) >= policy.min_bona_fide and all(
        len(v) >= policy.min_per_species for v in attacks.values()
    )
    blockers = []
    if candidate is None:
        blockers.append("no threshold meets the APCER target")
    if not sufficient:
        blockers.append("sample size below policy minimums")

    rejected = Counter(
        f"{s.species or s.label.value}:{s.rejection}" for s in samples if s.rejection is not None
    )
    fingerprint = dataset_fingerprint(samples)
    created = created_at or datetime.now(UTC)
    return CalibrationReport(
        schema_version=REPORT_SCHEMA_VERSION,
        report_type="face_liveness_calibration",
        calibration_id=(f"cal-{created:%Y%m%dT%H%M%SZ}-{fingerprint[:12]}-{manifest.digest[:8]}"),
        created_at=created,
        tool_version=__version__,
        model=ModelIdentity(
            model_id=manifest.model_id,
            version=manifest.version,
            manifest_sha256=manifest.digest,
            detector_id=manifest.detector.name,
        ),
        pipeline=PipelineSettings(
            detector_score_threshold=settings.detector_score_threshold,
            min_face_size_px=settings.min_face_size_px,
            min_image_side_px=settings.min_image_side_px,
            secondary_face_area_ratio=settings.secondary_face_area_ratio,
        ),
        dataset=DatasetInfo(
            fingerprint_sha256=fingerprint,
            images_total=len(samples),
            bona_fide_scored=len(bona),
            attack_scored={k: len(v) for k, v in sorted(attacks.items())},
            rejected_before_scoring=dict(sorted(rejected.items())),
        ),
        policy=policy,
        score_summary={
            Label.BONA_FIDE: _summary(bona),
            Label.ATTACK: _summary([x for v in attacks.values() for x in v]),
        },
        equal_error_rate=eer,
        operating_points=[rates.at(t) for t in REFERENCE_THRESHOLDS],
        threshold_candidate=candidate,
        sufficient_sample_size=sufficient,
        promotion=Promotion(
            eligible_for_review=not blockers,
            blockers=blockers,
            note=(
                "Never applied automatically. To adopt the candidate, a reviewer sets "
                "LIVENESS_LIVE_THRESHOLD and LIVENESS_THRESHOLD_CALIBRATION_ID through the "
                "release process, for this exact model version and manifest digest."
            ),
        ),
    )


# --- CLI --------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="face-liveness-calibrate",
        description="Produce an offline calibration report (never changes configuration).",
    )
    ap.add_argument("--data", type=Path, required=True, help="labelled dataset root")
    ap.add_argument("--model-dir", type=Path, default=Path("models"))
    ap.add_argument("--model-version", default="", help="installed version (default: built-in)")
    ap.add_argument("--target-apcer", type=float, default=0.01)
    ap.add_argument(
        "--min-threshold",
        type=float,
        default=0.5,
        help="never propose a threshold below this, however clean the dataset looks",
    )
    ap.add_argument("--threshold-step", type=float, default=0.001)
    ap.add_argument("--min-bona-fide", type=int, default=300)
    ap.add_argument("--min-per-species", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not (0.0 <= args.target_apcer < 1.0 and 0.0 < args.min_threshold < 1.0):
        print("--target-apcer must be in [0,1) and --min-threshold in (0,1)", file=sys.stderr)
        return 2
    if not 0.0 < args.threshold_step <= 0.1:
        print("--threshold-step must be in (0, 0.1]", file=sys.stderr)
        return 2
    # Same pipeline settings as the service (LIVENESS_* env), pointed at the given models.
    settings = Settings(
        env=Environment.TEST, model_dir=args.model_dir, active_model_version=args.model_version
    )
    runtime = load_runtime(settings)
    if not runtime.ready or runtime.manifest is None:
        print(f"model runtime not ready: {runtime.error}", file=sys.stderr)
        return 2
    try:
        items = discover_dataset(args.data)
        if not items:
            raise ValueError("no images found")
        samples = score_dataset(LivenessEngine(settings, runtime), settings, items)
        report = build_report(
            samples,
            runtime.manifest,
            settings,
            CalibrationPolicy(
                target_apcer=args.target_apcer,
                min_threshold=args.min_threshold,
                threshold_step=args.threshold_step,
                min_bona_fide=args.min_bona_fide,
                min_per_species=args.min_per_species,
            ),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    text = report.model_dump_json(indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    for blocker in report.promotion.blockers:
        print(f"NOT eligible for review: {blocker}", file=sys.stderr)
    return 0 if report.promotion.eligible_for_review else 1


if __name__ == "__main__":
    sys.exit(main())
