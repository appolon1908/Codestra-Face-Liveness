"""Offline performance / capacity benchmark of the production scoring pipeline.

    face-liveness-benchmark --model-dir ./models --concurrency 1,2,4 --requests 200 \\
        --out bench.json [--images ./faces]

For every image size and concurrency level, requests run through exactly the per-request
work of ``POST /v1/liveness/check`` after the HTTP layer: base64 decode, header-checked
image decode, face detection, and anti-spoofing inference. Inference always runs, on
the detected face or on a centred box when there is none, so synthetic images measure
the full cost. HTTP/JSON parsing, auth and network time are *not* included.

The report gives p50/p95/p99 latency, throughput, process CPU and memory, the image
dimensions used, and the host and settings it was measured with. Its ``profile_id``
may be recorded in ``LIVENESS_CAPACITY_PROFILE_ID`` so capabilities can reference the
profile a deployment was sized from. The report is evidence for a human sizing
decision only: the service never reads it and never tunes itself.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
import platform
import resource
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from . import __version__
from .admission import AdmissionController
from .calibration import IMAGE_SUFFIXES
from .config import Environment, Settings
from .errors import LivenessError
from .imaging import decode_base64, decode_image
from .inference import FaceBox
from .runtime import ModelRuntime, load_runtime

REPORT_SCHEMA_VERSION: Literal[1] = 1
DEFAULT_SIZES = ((640, 480), (1280, 720), (1920, 1080))


class LatencySummary(BaseModel):
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float


class RunResult(BaseModel):
    image: str = Field(description="`<width>x<height>` or `mixed` for a --images corpus.")
    width: int | None
    height: int | None
    payload_bytes: int
    concurrency: int
    requests: int
    errors: dict[str, int] = Field(description="Requests that failed, by error code.")
    faces_detected_rate: float
    latency: LatencySummary
    throughput_rps: float
    cpu_seconds: float
    cpu_utilisation: float = Field(description="Process CPU seconds / wall seconds (cores).")
    rss_peak_mib: float = Field(description="Process peak resident set size so far.")


class HostInfo(BaseModel):
    platform: str
    machine: str
    python: str
    cpu_count: int | None
    cpu_affinity: int | None
    onnxruntime: str


class BenchmarkSettings(BaseModel):
    onnx_intra_op_threads: int
    max_concurrent_checks: int
    detector_max_side: int
    max_image_bytes: int
    max_decoded_bytes: int


class ModelIdentity(BaseModel):
    version: str
    manifest_sha256: str


class BenchmarkReport(BaseModel):
    schema_version: Literal[1]
    report_type: Literal["face_liveness_benchmark"]
    profile_id: str
    created_at: datetime
    tool_version: str
    scope: Literal["in_process_pipeline"] = Field(
        default="in_process_pipeline",
        description="base64 + decode + detect + classify; excludes HTTP, auth and network.",
    )
    model: ModelIdentity
    host: HostInfo
    settings: BenchmarkSettings
    warmup_requests: int
    runs: list[RunResult]
    auto_tuned: Literal[False] = Field(
        default=False, description="Always false: the service never applies benchmark results."
    )


# --- workload ---------------------------------------------------------------------------


def synthetic_jpeg(width: int, height: int, seed: int = 0) -> bytes:
    """Smooth, JPEG-compressible content with realistic payload sizes."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(max(height // 16, 2), max(width // 16, 2), 3))
    img = Image.fromarray(small.astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def load_corpus(root: Path) -> list[bytes]:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise ValueError(f"no images under {root}")
    return [p.read_bytes() for p in files]


def _one(settings: Settings, runtime: ModelRuntime, payload: str) -> bool:
    """One request's pipeline work. Returns whether a face was detected."""
    assert runtime.detector is not None and runtime.classifier is not None
    raw = decode_base64(payload, settings.max_image_bytes)
    image = decode_image(
        raw,
        max_pixels=settings.max_image_pixels,
        min_side=settings.min_image_side_px,
        max_side=settings.max_image_side_px,
        max_decoded_bytes=settings.max_decoded_bytes,
    )
    faces = runtime.detector.detect(image.bgr)
    if faces:
        face = max(faces, key=lambda f: f.area)
    else:
        side = min(image.width, image.height) / 2
        face = FaceBox((image.width - side) / 2, (image.height - side) / 2, side, side, 1.0)
    runtime.classifier.score(image.bgr, face)
    return bool(faces)


def _rss_peak_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)


def _summary(latencies_ms: Sequence[float]) -> LatencySummary:
    if not latencies_ms:
        return LatencySummary(p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, mean_ms=0.0, max_ms=0.0)
    arr = np.asarray(latencies_ms, dtype=np.float64)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return LatencySummary(
        p50_ms=round(float(p50), 3),
        p95_ms=round(float(p95), 3),
        p99_ms=round(float(p99), 3),
        mean_ms=round(float(arr.mean()), 3),
        max_ms=round(float(arr.max()), 3),
    )


def run_level(
    settings: Settings,
    runtime: ModelRuntime,
    payloads: Sequence[bytes],
    label: str,
    size: tuple[int, int] | None,
    concurrency: int,
    requests: int,
) -> RunResult:
    """Run ``requests`` checks with ``concurrency`` client threads through admission."""
    encoded = [base64.b64encode(p).decode() for p in payloads]
    # The service's own admission bound applies, exactly as in production.
    admission = AdmissionController(settings.max_concurrent_checks, max_waiting=requests)
    latencies: list[float] = []
    errors: dict[str, int] = {}
    faces = 0
    lock = threading.Lock()

    def task(i: int) -> None:
        nonlocal faces
        started = time.perf_counter()
        try:
            with admission.slot(timeout=3600.0):
                found = _one(settings, runtime, encoded[i % len(encoded)])
        except LivenessError as exc:
            with lock:
                errors[exc.code.value] = errors.get(exc.code.value, 0) + 1
            return
        elapsed = (time.perf_counter() - started) * 1000
        with lock:
            latencies.append(elapsed)
            faces += found

    cpu0, wall0 = time.process_time(), time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(task, range(requests)))
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    ok = len(latencies)
    return RunResult(
        image=label,
        width=size[0] if size else None,
        height=size[1] if size else None,
        payload_bytes=round(sum(len(p) for p in payloads) / len(payloads)),
        concurrency=concurrency,
        requests=requests,
        errors=dict(sorted(errors.items())),
        faces_detected_rate=round(faces / ok, 4) if ok else 0.0,
        latency=_summary(latencies),
        throughput_rps=round(ok / wall, 3) if wall > 0 else 0.0,
        cpu_seconds=round(cpu, 3),
        cpu_utilisation=round(cpu / wall, 3) if wall > 0 else 0.0,
        rss_peak_mib=_rss_peak_mib(),
    )


def _host() -> HostInfo:
    import onnxruntime

    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return HostInfo(
        platform=platform.platform(),
        machine=platform.machine(),
        python=platform.python_version(),
        cpu_count=os.cpu_count(),
        cpu_affinity=affinity,
        onnxruntime=str(onnxruntime.__version__),
    )


def run_benchmark(
    settings: Settings,
    runtime: ModelRuntime,
    concurrency: Sequence[int],
    requests: int,
    warmup: int,
    sizes: Sequence[tuple[int, int]] = DEFAULT_SIZES,
    corpus: Sequence[bytes] | None = None,
    host: HostInfo | None = None,
    created_at: datetime | None = None,
) -> BenchmarkReport:
    if not runtime.ready or runtime.manifest is None:
        raise ValueError(f"model runtime not ready: {runtime.error}")
    workloads: list[tuple[str, tuple[int, int] | None, list[bytes]]]
    if corpus:
        workloads = [("mixed", None, list(corpus))]
    else:
        workloads = [(f"{w}x{h}", (w, h), [synthetic_jpeg(w, h)]) for w, h in sizes]

    runs: list[RunResult] = []
    for label, size, payloads in workloads:
        if warmup:
            run_level(settings, runtime, payloads, label, size, 1, warmup)
        for c in concurrency:
            runs.append(run_level(settings, runtime, payloads, label, size, c, requests))

    created = created_at or datetime.now(UTC)
    host = host or _host()
    bench_settings = BenchmarkSettings(
        onnx_intra_op_threads=settings.onnx_intra_op_threads,
        max_concurrent_checks=settings.max_concurrent_checks,
        detector_max_side=settings.detector_max_side,
        max_image_bytes=settings.max_image_bytes,
        max_decoded_bytes=settings.max_decoded_bytes,
    )
    fingerprint = hashlib.sha256(
        (host.model_dump_json() + bench_settings.model_dump_json()).encode()
    ).hexdigest()
    return BenchmarkReport(
        schema_version=REPORT_SCHEMA_VERSION,
        report_type="face_liveness_benchmark",
        profile_id=(
            f"cap-{created:%Y%m%dT%H%M%SZ}-{fingerprint[:8]}-{runtime.manifest.digest[:8]}"
        ),
        created_at=created,
        tool_version=__version__,
        model=ModelIdentity(
            version=runtime.manifest.version, manifest_sha256=runtime.manifest.digest
        ),
        host=host,
        settings=bench_settings,
        warmup_requests=warmup,
        runs=runs,
    )


# --- CLI --------------------------------------------------------------------------------


def _ints(value: str) -> list[int]:
    try:
        out = [int(v) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not out or any(not 1 <= v <= 256 for v in out):
        raise argparse.ArgumentTypeError("values must be in 1..256")
    return out


def _sizes(value: str) -> list[tuple[int, int]]:
    out = []
    for item in value.split(","):
        w, _, h = item.strip().partition("x")
        if not (w.isdigit() and h.isdigit()):
            raise argparse.ArgumentTypeError("expected sizes like 640x480,1920x1080")
        out.append((int(w), int(h)))
    return out


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="face-liveness-benchmark",
        description="Measure pipeline latency/throughput/CPU/memory (never tunes the service).",
    )
    ap.add_argument("--model-dir", type=Path, default=Path("models"))
    ap.add_argument("--model-version", default="", help="installed version (default: active)")
    ap.add_argument("--concurrency", type=_ints, default=[1, 2, 4])
    ap.add_argument("--requests", type=int, default=100, help="requests per level")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--sizes", type=_sizes, default=list(DEFAULT_SIZES))
    ap.add_argument("--images", type=Path, default=None, help="use this corpus instead")
    ap.add_argument("--out", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.requests <= 1_000_000 or not 0 <= args.warmup <= 10_000:
        print("--requests must be in 1..1e6 and --warmup in 0..1e4", file=sys.stderr)
        return 2
    settings = Settings(
        env=Environment.TEST, model_dir=args.model_dir, active_model_version=args.model_version
    )
    runtime = load_runtime(settings)
    try:
        corpus = load_corpus(args.images) if args.images else None
        report = run_benchmark(
            settings, runtime, args.concurrency, args.requests, args.warmup, args.sizes, corpus
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    text = report.model_dump_json(indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
