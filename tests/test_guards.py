"""Abuse / resource guards: decoded size, decompression bombs, concurrency, time budget."""

from __future__ import annotations

import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import ImageFile

from face_liveness.admission import AdmissionController, Deadline
from face_liveness.config import Environment, Settings
from face_liveness.errors import ErrorCode, LivenessError
from face_liveness.imaging import decode_image

from .conftest import FakeClassifier, b64

CHECK = "/v1/liveness/check"


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png_bomb(width: int, height: int, *, full: bool) -> bytes:
    """8-bit grayscale PNG of zeros. ``full`` streams real (highly compressible) scanlines;
    otherwise only the header claims the size."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    comp = zlib.compressobj(9)
    if full:
        row = b"\x00" * (width + 1)
        idat = b"".join(comp.compress(row) for _ in range(height)) + comp.flush()
    else:
        idat = comp.compress(b"\x00" * 64) + comp.flush()
    return (
        b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    )


@pytest.fixture
def no_pixel_decode(monkeypatch):
    """Fail the test if any pixel data is decompressed."""

    def boom(self):  # type: ignore[no-untyped-def]
        raise AssertionError("pixel data was decoded")

    monkeypatch.setattr(ImageFile.ImageFile, "load", boom)


# --- decoded size / bombs -----------------------------------------------------------------


def test_small_file_large_decoded_size_rejected_before_decode(no_pixel_decode):
    data = png_bomb(4000, 4000, full=True)
    assert len(data) < 64 * 1024  # tiny on the wire, 16 MP / 64 MB peak when decoded
    with pytest.raises(LivenessError) as exc:
        decode_image(data, max_pixels=10**8, min_side=1, max_decoded_bytes=16 * 1024 * 1024)
    assert exc.value.code is ErrorCode.PAYLOAD_TOO_LARGE
    assert "decoded image" in exc.value.message


def test_oversized_side_rejected_before_decode(no_pixel_decode):
    with pytest.raises(LivenessError) as exc:
        decode_image(png_bomb(20_000, 200, full=False), max_pixels=10**9, min_side=1, max_side=8192)
    assert exc.value.code is ErrorCode.PAYLOAD_TOO_LARGE
    assert "side" in exc.value.message


def test_pixel_count_bomb_rejected_before_decode(no_pixel_decode):
    with pytest.raises(LivenessError) as exc:
        decode_image(png_bomb(8000, 8000, full=False), max_pixels=25_000_000, min_side=1)
    assert exc.value.code is ErrorCode.PAYLOAD_TOO_LARGE


def test_pillow_bomb_guard_is_mapped(no_pixel_decode):
    # 60000 x 60000 exceeds Pillow's own DecompressionBombError limit at open().
    with pytest.raises(LivenessError) as exc:
        decode_image(
            png_bomb(60_000, 60_000, full=False), max_pixels=10**10, min_side=1, max_side=65_535
        )
    assert exc.value.code is ErrorCode.PAYLOAD_TOO_LARGE


def test_api_rejects_bomb_without_running_models(make_client, detector, classifier):
    client = make_client(settings_=Settings(env=Environment.TEST, max_decoded_bytes=8 << 20))
    resp = client.post(CHECK, json={"image_base64": b64(png_bomb(3000, 3000, full=True))})
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"
    assert detector.calls == 0 and classifier.calls == 0
    text = client.get("/metrics").text
    assert (
        'liveness_rejections_total{code="PAYLOAD_TOO_LARGE",route="/v1/liveness/check",'
        'stage="input"} 1.0'
    ) in text


def test_api_rejects_oversized_dimensions(make_client):
    client = make_client(settings_=Settings(env=Environment.TEST, max_image_side_px=1024))
    resp = client.post(CHECK, json={"image_base64": b64(png_bomb(2000, 200, full=False))})
    assert resp.status_code == 413


def test_body_limit_rejection_is_counted(make_client):
    client = make_client(settings_=Settings(env=Environment.TEST, max_image_bytes=1024))
    client.post(CHECK, content=b"x" * 100_000, headers={"Content-Type": "application/json"})
    text = client.get("/metrics").text
    assert (
        'liveness_rejections_total{code="PAYLOAD_TOO_LARGE",route="pre_routing",stage="input"} 1.0'
    ) in text


def test_auth_rejection_is_counted(make_client, image_b64):
    client = make_client(settings_=Settings(env=Environment.TEST, require_auth=True))
    client.post(CHECK, json={"image_base64": image_b64})
    assert 'code="UNAUTHORIZED",route="/v1/liveness/check",stage="auth"} 1.0' in (
        client.get("/metrics").text
    )


# --- concurrency saturation ---------------------------------------------------------------


class BlockingClassifier(FakeClassifier):
    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.entered = threading.Semaphore(0)

    def score(self, bgr, face):  # type: ignore[no-untyped-def]
        self.entered.release()
        assert self.release.wait(10)
        return super().score(bgr, face)


def _wait_for(predicate, timeout: float = 5.0) -> None:  # type: ignore[no-untyped-def]
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.005)


def test_saturation_bounded_queue_rejects_immediately(make_client, detector, image_b64):
    from .conftest import fake_runtime

    classifier = BlockingClassifier()
    s = Settings(
        env=Environment.TEST, max_concurrent_checks=1, max_queued_checks=1, busy_timeout_seconds=10
    )
    client = make_client(runtime=fake_runtime(detector, classifier), settings_=s)
    admission = client.app.state.admission
    body = {"image_base64": image_b64}

    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(client.post, CHECK, json=body)
        assert classifier.entered.acquire(timeout=5)  # holds the only slot
        queued = pool.submit(client.post, CHECK, json=body)
        _wait_for(lambda: admission.waiting == 1)

        started = time.monotonic()
        rejected = client.post(CHECK, json=body)  # slot busy + queue full
        assert time.monotonic() - started < 2.0  # rejected without waiting busy_timeout
        assert rejected.status_code == 503
        assert rejected.json()["error"]["code"] == "BUSY"
        assert rejected.headers["Retry-After"] == "1"

        text = client.get("/metrics").text
        assert "liveness_checks_in_flight 1.0" in text
        assert "liveness_checks_waiting 1.0" in text

        classifier.release.set()
        assert running.result(timeout=10).status_code == 200
        assert queued.result(timeout=10).status_code == 200

    assert admission.in_flight == 0 and admission.waiting == 0
    text = client.get("/metrics").text
    assert (
        'liveness_rejections_total{code="BUSY",route="/v1/liveness/check",stage="admission"}'
        in (text)
    )


def test_saturation_many_parallel_requests_never_exceed_slots(make_client, detector, image_b64):
    from .conftest import fake_runtime

    peak = 0
    lock = threading.Lock()
    active = 0

    class Counting(FakeClassifier):
        def score(self, bgr, face):  # type: ignore[no-untyped-def]
            nonlocal peak, active
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return super().score(bgr, face)

    s = Settings(
        env=Environment.TEST, max_concurrent_checks=2, max_queued_checks=2, busy_timeout_seconds=5
    )
    client = make_client(runtime=fake_runtime(detector, Counting()), settings_=s)
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(
            pool.map(lambda _: client.post(CHECK, json={"image_base64": image_b64}), range(24))
        )
    codes = [r.status_code for r in results]
    assert set(codes) <= {200, 503}
    assert codes.count(200) >= 2
    assert peak <= 2
    assert all(r.json()["error"]["code"] == "BUSY" for r in results if r.status_code == 503)


def test_admission_timeout_when_queue_allowed():
    ctl = AdmissionController(slots=1, max_waiting=1)
    with ctl.slot(timeout=0):
        started = time.monotonic()
        with pytest.raises(LivenessError) as exc, ctl.slot(timeout=0.05):
            pass
        assert time.monotonic() - started < 1.0
    assert exc.value.code is ErrorCode.BUSY
    assert ctl.waiting == 0 and ctl.in_flight == 0


# --- time budget ------------------------------------------------------------------------


def test_deadline_exceeded_returns_no_late_decision(make_client, detector, image_b64):
    from .conftest import fake_runtime

    class Slow(FakeClassifier):
        def score(self, bgr, face):  # type: ignore[no-untyped-def]
            time.sleep(0.2)
            return super().score(bgr, face)

    s = Settings(env=Environment.TEST, request_timeout_seconds=0.05)
    client = make_client(runtime=fake_runtime(detector, Slow()), settings_=s)
    resp = client.post(CHECK, json={"image_base64": image_b64})
    assert resp.status_code == 503
    err = resp.json()["error"]
    assert err["code"] == "DEADLINE_EXCEEDED"
    assert err["retryable"] is True
    assert "decision" not in resp.text
    assert 'stage="deadline"' in client.get("/metrics").text


def test_deadline_bounds_queue_wait():
    now = [0.0]
    d = Deadline(1.0, clock=lambda: now[0])
    assert d.remaining() == 1.0
    now[0] = 0.4
    assert d.remaining() == pytest.approx(0.6)
    d.check("ok")
    now[0] = 1.0
    with pytest.raises(LivenessError) as exc:
        d.check("inference")
    assert exc.value.code is ErrorCode.DEADLINE_EXCEEDED
    assert d.remaining() == 0.0


def test_large_but_legal_image_still_accepted(client):
    from .conftest import encode_image

    arr_img = encode_image("JPEG", size=(1600, 1200))
    resp = client.post(CHECK, json={"image_base64": b64(arr_img)})
    assert resp.status_code == 200
    assert resp.json()["image"] == {"width": 1600, "height": 1200, "media_type": "image/jpeg"}
    assert np.isfinite(resp.json()["live_score"])
