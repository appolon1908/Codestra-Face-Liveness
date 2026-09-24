"""Safe, in-memory image decoding. Images are never written to disk."""

from __future__ import annotations

import base64
import binascii
import io
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ErrorCode, LivenessError

ALLOWED_FORMATS: dict[str, str] = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


@dataclass(frozen=True, slots=True)
class DecodedImage:
    bgr: NDArray[np.uint8]  # HxWx3, BGR channel order (OpenCV convention)
    media_type: str

    @property
    def width(self) -> int:
        return int(self.bgr.shape[1])

    @property
    def height(self) -> int:
        return int(self.bgr.shape[0])


def decode_base64(data: str, max_bytes: int) -> bytes:
    # Tolerate a data URL prefix ("data:image/jpeg;base64,...").
    if data.startswith("data:"):
        _, _, data = data.partition(",")
    # Cheap pre-check before allocating the decoded buffer.
    if len(data) * 3 // 4 > max_bytes + 3:
        raise LivenessError(ErrorCode.PAYLOAD_TOO_LARGE, f"image exceeds {max_bytes} bytes")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise LivenessError(ErrorCode.INVALID_REQUEST, "image_base64 is not valid base64") from exc
    if not raw:
        raise LivenessError(ErrorCode.INVALID_IMAGE, "image is empty")
    if len(raw) > max_bytes:
        raise LivenessError(ErrorCode.PAYLOAD_TOO_LARGE, f"image exceeds {max_bytes} bytes")
    return raw


def decode_image(raw: bytes, *, max_pixels: int, min_side: int) -> DecodedImage:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(raw))
            fmt = img.format or ""
            if fmt not in ALLOWED_FORMATS:
                raise LivenessError(
                    ErrorCode.UNSUPPORTED_MEDIA_TYPE,
                    f"unsupported image format {fmt or 'unknown'}; allowed: JPEG, PNG, WEBP",
                )
            width, height = img.size
            if width * height > max_pixels:
                raise LivenessError(
                    ErrorCode.PAYLOAD_TOO_LARGE, f"image exceeds {max_pixels} pixels"
                )
            if getattr(img, "n_frames", 1) > 1:
                raise LivenessError(
                    ErrorCode.INVALID_IMAGE, "animated/multi-frame images are not accepted"
                )
            img.load()
            rgb = ImageOps.exif_transpose(img).convert("RGB")
    except LivenessError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise LivenessError(ErrorCode.INVALID_IMAGE, "image could not be decoded") from exc
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise LivenessError(ErrorCode.PAYLOAD_TOO_LARGE, "image exceeds pixel limit") from exc

    if min(rgb.size) < min_side:
        raise LivenessError(
            ErrorCode.IMAGE_TOO_SMALL, f"image must be at least {min_side}px on each side"
        )
    arr = np.asarray(rgb, dtype=np.uint8)[:, :, ::-1]
    return DecodedImage(bgr=np.ascontiguousarray(arr), media_type=ALLOWED_FORMATS[fmt])
