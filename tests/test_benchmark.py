"""Performance / capacity benchmark (mission 9): report only, never auto-tuning."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from face_liveness import benchmark
from face_liveness.benchmark import BenchmarkReport, HostInfo, run_benchmark, synthetic_jpeg
from face_liveness.config import Environment, Settings
from face_liveness.contracts import json_schemas
from face_liveness.runtime import ModelRuntime, load_runtime

from .conftest import FakeClassifier, FakeDetector, encode_image, fake_runtime
from .test_runtime import REAL_MODEL_DIR, have_models

SETTINGS = Settings(env=Environment.TEST)
HOST = HostInfo(
    platform="test",
    machine="x86_64",
    python="3",
    cpu_count=4,
    cpu_affinity=4,
    onnxruntime="1",
)


def test_report_shape_and_percentiles():
    rt = fake_runtime(FakeDetector(), FakeClassifier())
    report = run_benchmark(
        SETTINGS,
        rt,
        concurrency=[1, 3],
        requests=12,
        warmup=1,
        sizes=[(320, 240), (640, 480)],
        host=HOST,
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
    )
    assert report.auto_tuned is False
    assert report.scope == "in_process_pipeline"
    assert report.profile_id.startswith("cap-20260924T000000Z-")
    assert report.model.version == "1.0.0"
    assert [(r.image, r.concurrency) for r in report.runs] == [
        ("320x240", 1),
        ("320x240", 3),
        ("640x480", 1),
        ("640x480", 3),
    ]
    for run in report.runs:
        assert run.requests == 12 and run.errors == {}
        assert (run.width, run.height) == tuple(int(v) for v in run.image.split("x"))
        lat = run.latency
        assert 0 < lat.p50_ms <= lat.p95_ms <= lat.p99_ms <= lat.max_ms
        assert run.throughput_rps > 0
        assert run.cpu_seconds >= 0 and run.cpu_utilisation >= 0
        assert run.rss_peak_mib > 0
        assert run.faces_detected_rate == 1.0
        assert run.payload_bytes > 0
    assert BenchmarkReport.model_validate_json(report.model_dump_json()) == report


def test_profile_id_depends_on_host_and_settings():
    rt = fake_runtime(FakeDetector(), FakeClassifier())
    at = datetime(2026, 9, 24, tzinfo=UTC)

    def profile(settings: Settings, host: HostInfo) -> str:
        return run_benchmark(
            settings, rt, [1], 2, 0, sizes=[(200, 200)], host=host, created_at=at
        ).profile_id

    a = profile(SETTINGS, HOST)
    b = profile(SETTINGS, HOST.model_copy(update={"cpu_count": 64}))
    c = profile(Settings(env=Environment.TEST, onnx_intra_op_threads=4), HOST)
    assert len({a, b, c}) == 3


def test_errors_are_counted_not_timed():
    rt = fake_runtime(FakeDetector(), FakeClassifier())
    s = Settings(env=Environment.TEST, max_image_side_px=256)
    report = run_benchmark(s, rt, [1], 3, 0, sizes=[(640, 480)], host=HOST)
    [run] = report.runs
    assert run.errors == {"PAYLOAD_TOO_LARGE": 3}
    assert run.throughput_rps == 0.0


def test_no_face_still_measures_inference():
    classifier = FakeClassifier()
    rt = fake_runtime(FakeDetector(faces=[]), classifier)
    [run] = run_benchmark(SETTINGS, rt, [1], 4, 0, sizes=[(320, 240)], host=HOST).runs
    assert run.faces_detected_rate == 0.0
    assert classifier.calls == 4


def test_corpus_mode():
    rt = fake_runtime(FakeDetector(), FakeClassifier())
    corpus = [encode_image("JPEG", (300, 200), seed=i) for i in range(3)]
    [run] = run_benchmark(SETTINGS, rt, [2], 6, 0, corpus=corpus, host=HOST).runs
    assert run.image == "mixed" and run.width is None


def test_unready_runtime_refused():
    with pytest.raises(ValueError, match="not ready"):
        run_benchmark(SETTINGS, ModelRuntime.unavailable("x"), [1], 1, 0, host=HOST)


def test_synthetic_images_are_realistic_jpegs():
    data = synthetic_jpeg(1920, 1080)
    assert data[:2] == b"\xff\xd8"
    assert 50_000 < len(data) < 2_000_000


def test_cli_writes_report_only(tmp_path, monkeypatch, capsys):
    rt = fake_runtime(FakeDetector(), FakeClassifier())
    monkeypatch.setattr(benchmark, "load_runtime", lambda settings: rt)
    out = tmp_path / "bench.json"
    env_before = dict(os.environ)
    before = set(tmp_path.iterdir())
    code = benchmark.main(
        [
            *("--requests", "3", "--warmup", "0", "--concurrency", "1,2"),
            *("--sizes", "320x240", "--out", str(out)),
        ]
    )
    assert code == 0, capsys.readouterr().err
    report = BenchmarkReport.model_validate_json(out.read_text())
    assert len(report.runs) == 2
    assert set(tmp_path.iterdir()) - before == {out}
    assert dict(os.environ) == env_before


def test_cli_rejects_bad_args(monkeypatch):
    monkeypatch.setattr(benchmark, "load_runtime", lambda s: ModelRuntime.unavailable("none"))
    assert benchmark.main(["--requests", "0"]) == 2
    assert benchmark.main(["--requests", "1"]) == 2  # runtime not ready
    with pytest.raises(SystemExit):
        benchmark.main(["--concurrency", "0"])
    with pytest.raises(SystemExit):
        benchmark.main(["--sizes", "big"])


def test_schema_published_and_forbids_auto_tuning():
    schema = json_schemas()["benchmark-report.v1.schema.json"]
    assert schema["properties"]["auto_tuned"]["const"] is False


def test_capabilities_reference_capacity_profile(make_client):
    body = make_client().get("/v1/capabilities").json()
    assert body["limits"]["capacity_profile"] == {
        "profile_id": None,
        "tested": False,
        "auto_tuned": False,
    }
    s = Settings(env=Environment.TEST, capacity_profile_id="cap-20260924T000000Z-abcd1234-a950c97c")
    body = make_client(settings_=s).get("/v1/capabilities").json()
    profile = body["limits"]["capacity_profile"]
    assert profile["profile_id"] == "cap-20260924T000000Z-abcd1234-a950c97c"
    assert profile["tested"] is True
    # Referencing a profile changes no limit: the service does not tune itself.
    assert body["limits"]["max_concurrent_checks"] == 4


@pytest.mark.models
@pytest.mark.skipif(not have_models, reason="real model artifacts not present")
def test_real_models_benchmark_smoke():
    s = Settings(env=Environment.TEST, model_dir=REAL_MODEL_DIR)
    rt = load_runtime(s)
    assert rt.ready, rt.error
    [run] = run_benchmark(s, rt, [1], 3, 1, sizes=[(640, 480)]).runs
    assert run.errors == {} and run.latency.p99_ms > 0
