"""Offline threshold calibration against a labelled, locally held dataset.

    python tools/calibrate.py --data ./calib --model-dir ./models --target-apcer 0.01

Dataset layout (images are read, never copied or uploaded):

    calib/bona_fide/*.jpg            genuine live captures from the target camera(s)
    calib/attack/<species>/*.jpg     presentation attacks, grouped by attack species
                                     (e.g. print, replay_phone, replay_monitor, mask)

Metrics follow ISO/IEC 30107-3 terminology:
    APCER  attack presentations wrongly classified as bona fide (per species; worst reported)
    BPCER  bona fide presentations wrongly classified as attacks

The chosen threshold is the smallest one whose worst-species APCER <= target. Images that
fail detection (no face / multiple faces / too small) are reported separately; they would be
rejected before scoring in production, so they count neither as live nor as spoof here.
Outputs a JSON report; set LIVENESS_LIVE_THRESHOLD and LIVENESS_THRESHOLD_CALIBRATION_ID
from it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from face_liveness.config import Environment, Settings
from face_liveness.engine import LivenessEngine
from face_liveness.errors import LivenessError
from face_liveness.imaging import decode_image
from face_liveness.runtime import load_runtime

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate the live-score threshold")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, default=Path("models"))
    ap.add_argument("--target-apcer", type=float, default=0.01)
    ap.add_argument(
        "--min-threshold",
        type=float,
        default=0.5,
        help="never recommend a threshold below this, however clean the dataset looks",
    )
    ap.add_argument("--min-bona-fide", type=int, default=300)
    ap.add_argument("--min-per-species", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    settings = Settings(env=Environment.TEST, model_dir=args.model_dir)
    runtime = load_runtime(settings)
    if not runtime.ready:
        print(f"model runtime not ready: {runtime.error}", file=sys.stderr)
        return 2
    engine = LivenessEngine(settings, runtime)

    bona_fide: list[float] = []
    attacks: dict[str, list[float]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    digest = hashlib.sha256()

    samples = [(p, None) for p in _images(args.data / "bona_fide")]
    for species_dir in sorted(d for d in (args.data / "attack").iterdir() if d.is_dir()):
        samples += [(p, species_dir.name) for p in _images(species_dir)]
    if not samples:
        print("no images found", file=sys.stderr)
        return 2

    for path, species in samples:
        raw = path.read_bytes()
        digest.update(hashlib.sha256(raw).digest())
        try:
            image = decode_image(raw, max_pixels=settings.max_image_pixels, min_side=1)
            score = engine.check(image).live_score
        except LivenessError as exc:
            rejected[f"{species or 'bona_fide'}:{exc.code.value}"] += 1
            continue
        (bona_fide if species is None else attacks[species]).append(score)

    if not bona_fide or not attacks:
        print("need at least one scored bona fide image and one attack species", file=sys.stderr)
        return 2

    def rates(t: float) -> tuple[float, dict[str, float]]:
        bpcer = sum(s < t for s in bona_fide) / len(bona_fide)
        apcer = {k: sum(s >= t for s in v) / len(v) for k, v in attacks.items()}
        return bpcer, apcer

    lo = round(args.min_threshold * 1000)
    candidates = [i / 1000 for i in range(lo, 1000)]
    chosen = None
    table = []
    for t in candidates:
        bpcer, apcer = rates(t)
        worst = max(apcer.values())
        if t in (0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99):
            table.append({"threshold": t, "bpcer": bpcer, "apcer_max": worst, "apcer": apcer})
        if chosen is None and worst <= args.target_apcer:
            chosen = {"threshold": t, "bpcer": bpcer, "apcer_max": worst, "apcer": apcer}

    sufficient = len(bona_fide) >= args.min_bona_fide and all(
        len(v) >= args.min_per_species for v in attacks.values()
    )
    report = {
        "calibration_id": f"cal-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{digest.hexdigest()[:12]}",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": digest.hexdigest(),
        "model_artifacts": {a.file: a.sha256 for a in runtime.artifacts},
        "counts": {
            "bona_fide": len(bona_fide),
            "attack": {k: len(v) for k, v in attacks.items()},
            "rejected_before_scoring": dict(rejected),
        },
        "sufficient_sample_size": sufficient,
        "target_apcer": args.target_apcer,
        "recommended": chosen,
        "reference_points": table,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    if chosen is None:
        print("no threshold meets the APCER target; do not deploy", file=sys.stderr)
        return 1
    if not sufficient:
        print("sample size below minimums; result is NOT a valid calibration", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
