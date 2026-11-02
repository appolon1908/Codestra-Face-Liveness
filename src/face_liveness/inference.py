"""Model wrappers: YuNet face detection and the MiniFASNet anti-spoofing ensemble.

Only these two classes touch OpenCV DNN / ONNX Runtime. Everything else depends on the
``FaceDetector`` / ``SpoofClassifier`` protocols so tests can inject fakes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

MINIFASNET_INPUT = 80
MINIFASNET_LIVE_CLASS = 1  # upstream labels: 1 = real face, 0 / 2 = attack types


@dataclass(frozen=True, slots=True)
class FaceBox:
    x: float
    y: float
    w: float
    h: float
    score: float

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass(frozen=True, slots=True)
class ComponentScore:
    name: str
    live_score: float


class FaceDetector(Protocol):
    def detect(self, bgr: NDArray[np.uint8]) -> list[FaceBox]: ...


class SpoofClassifier(Protocol):
    @property
    def component_names(self) -> list[str]: ...

    def score(self, bgr: NDArray[np.uint8], face: FaceBox) -> list[ComponentScore]: ...


class YuNetDetector:
    """OpenCV YuNet detector. Large images are downscaled for detection only."""

    def __init__(self, model_path: Path, score_threshold: float, max_side: int) -> None:
        self._max_side = max_side
        self._net = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 5000
        )
        # FaceDetectorYN keeps per-call input-size state and is not thread-safe.
        self._lock = threading.Lock()

    def detect(self, bgr: NDArray[np.uint8]) -> list[FaceBox]:
        h, w = bgr.shape[:2]
        scale = min(1.0, self._max_side / max(h, w))
        img: NDArray[Any] = bgr
        if scale < 1.0:
            img = cv2.resize(
                bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
            )
        ih, iw = img.shape[:2]
        with self._lock:
            self._net.setInputSize((iw, ih))
            _, faces = self._net.detect(img)
        if faces is None:
            return []
        return [
            FaceBox(
                x=float(f[0]) / scale,
                y=float(f[1]) / scale,
                w=float(f[2]) / scale,
                h=float(f[3]) / scale,
                score=float(f[14]),
            )
            for f in faces
        ]


def crop_box(src_w: int, src_h: int, face: FaceBox, scale: float) -> tuple[int, int, int, int]:
    """Context crop around the face, matching upstream Silent-Face-Anti-Spoofing CropImage.

    The crop is ``scale`` times the face box, clamped to fit the image, and shifted (not
    shrunk) when it would cross an image border.
    """
    scale = min((src_h - 1) / face.h, (src_w - 1) / face.w, scale)
    new_w, new_h = face.w * scale, face.h * scale
    cx, cy = face.x + face.w / 2, face.y + face.h / 2
    left, top = cx - new_w / 2, cy - new_h / 2
    right, bottom = cx + new_w / 2, cy + new_h / 2
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > src_w - 1:
        left -= right - src_w + 1
        right = src_w - 1
    if bottom > src_h - 1:
        top -= bottom - src_h + 1
        bottom = src_h - 1
    return int(max(left, 0)), int(max(top, 0)), int(right), int(bottom)


def _softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    shifted = np.exp(logits - logits.max())
    out: NDArray[np.float32] = shifted / shifted.sum()
    return out


@dataclass(frozen=True, slots=True)
class MiniFASNetMember:
    name: str
    path: Path
    scale: float


class MiniFASNetEnsemble:
    """Ensemble of MiniFASNet ONNX models, each seeing a different context crop.

    Input: BGR, float32, un-normalised 0..255, NCHW 1x3x80x80 (as in upstream training).
    Output per model: 3 logits; softmax[1] is the live probability.
    """

    def __init__(self, members: list[MiniFASNetMember], intra_op_threads: int) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = 1
        self._members = members
        self._sessions: list[tuple[MiniFASNetMember, ort.InferenceSession, str]] = []
        for member in members:
            session = ort.InferenceSession(
                str(member.path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self._sessions.append((member, session, session.get_inputs()[0].name))
        self._self_test()

    @property
    def component_names(self) -> list[str]:
        return [m.name for m in self._members]

    def _self_test(self) -> None:
        dummy = np.zeros((1, 3, MINIFASNET_INPUT, MINIFASNET_INPUT), dtype=np.float32)
        for member, session, input_name in self._sessions:
            out = session.run(None, {input_name: dummy})[0]
            if out.shape != (1, 3) or not np.all(np.isfinite(out)):
                raise RuntimeError(f"{member.name}: unexpected output {out.shape}")

    def score(self, bgr: NDArray[np.uint8], face: FaceBox) -> list[ComponentScore]:
        h, w = bgr.shape[:2]
        results: list[ComponentScore] = []
        for member, session, input_name in self._sessions:
            left, top, right, bottom = crop_box(w, h, face, member.scale)
            patch = cv2.resize(
                bgr[top : bottom + 1, left : right + 1], (MINIFASNET_INPUT, MINIFASNET_INPUT)
            )
            tensor = patch.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
            logits = session.run(None, {input_name: tensor})[0][0]
            probs = _softmax(logits)
            if not np.all(np.isfinite(probs)):
                raise RuntimeError(f"{member.name}: non-finite output")
            results.append(
                ComponentScore(name=member.name, live_score=float(probs[MINIFASNET_LIVE_CLASS]))
            )
        return results
