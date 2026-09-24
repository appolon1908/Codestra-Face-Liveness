from __future__ import annotations

import base64
import io
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from face_liveness.api import create_app
from face_liveness.config import Environment, Settings
from face_liveness.inference import ComponentScore, FaceBox
from face_liveness.runtime import ModelRuntime


@dataclass
class FakeDetector:
    faces: list[FaceBox] = field(
        default_factory=lambda: [FaceBox(x=60, y=50, w=120, h=140, score=0.98)]
    )
    fail: bool = False
    calls: int = 0

    def detect(self, bgr: np.ndarray) -> list[FaceBox]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("detector exploded")
        return list(self.faces)


@dataclass
class FakeClassifier:
    scores: tuple[float, ...] = (0.97, 0.95)
    fail: bool = False
    calls: int = 0

    @property
    def component_names(self) -> list[str]:
        return [f"fake_{i}" for i in range(len(self.scores))]

    def score(self, bgr: np.ndarray, face: FaceBox) -> list[ComponentScore]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("onnx session died")
        return [
            ComponentScore(name=n, live_score=s)
            for n, s in zip(self.component_names, self.scores, strict=True)
        ]


def encode_image(fmt: str = "PNG", size: tuple[int, int] = (240, 240), seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@pytest.fixture
def image_b64() -> str:
    return b64(encode_image())


@pytest.fixture
def detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture
def classifier() -> FakeClassifier:
    return FakeClassifier()


@pytest.fixture
def settings() -> Settings:
    return Settings(env=Environment.TEST, live_threshold=0.85)


@pytest.fixture
def make_client(
    settings: Settings, detector: FakeDetector, classifier: FakeClassifier
) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def _make(runtime: ModelRuntime | None = None, settings_: Settings | None = None) -> TestClient:
        rt = runtime or ModelRuntime(detector=detector, classifier=classifier)
        client = TestClient(create_app(settings_ or settings, rt))
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()
