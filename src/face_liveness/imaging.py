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

from .errors import ErrorCode, Guard, LivenessError

ALLOWED_FORMATS: dict[str, str] = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}

# Bytes per pixel of the decoded bitmap for the modes JPEG/PNG/WebP decode to. Unknown
# modes are assumed to be the widest (32-bit single band or 4 x 8-bit).
_MODE_BYTES: dict[str, int] = {
    "1": 1,
    "L": 1,
    "P": 1,
    "LA": 2,
    "PA": 2,
    "I;16": 2,
    "I;16B": 2,
    "RGB": 3,
    "YCbCr": 3,
    "RGBA": 4,
    "RGBX": 4,
    "CMYK": 4,
    "I": 4,
    "F": 4,
}


def decoded_size_bytes(width: int, height: int, mode: str) -> int:
    """Peak bitmap memory: decoded frame + RGB conversion + two 3-byte numpy copies."""
    return width * height * (_MODE_BYTES.get(mode, 4) + 9)


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
        raise LivenessError(
            ErrorCode.PAYLOAD_TOO_LARGE, f"image exceeds {max_bytes} bytes", Guard.IMAGE_BYTES
        )
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise LivenessError(ErrorCode.INVALID_REQUEST, "image_base64 is not valid base64") from exc
    if not raw:
        raise LivenessError(ErrorCode.INVALID_IMAGE, "image is empty")
    if len(raw) > max_bytes:
        raise LivenessError(
            ErrorCode.PAYLOAD_TOO_LARGE, f"image exceeds {max_bytes} bytes", Guard.IMAGE_BYTES
        )
    return raw


def decode_image(
    raw: bytes,
    *,
    max_pixels: int,
    min_side: int,
    max_side: int = 65_535,
    max_decoded_bytes: int | None = None,
) -> DecodedImage:
    """Decode an image entirely in memory.

    Dimension, pixel-count and decoded-size limits are enforced from the header, before
    any pixel data is decompressed, so small "decompression bomb" inputs are rejected
    without allocating their bitmap.
    """
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
            if max(width, height) > max_side:
                raise LivenessError(
                    ErrorCode.PAYLOAD_TOO_LARGE,
                    f"image side exceeds {max_side}px",
                    Guard.IMAGE_DIMENSIONS,
                )
            if width * height > max_pixels:
                raise LivenessError(
                    ErrorCode.PAYLOAD_TOO_LARGE,
                    f"image exceeds {max_pixels} pixels",
                    Guard.IMAGE_PIXELS,
                )
            if (
                max_decoded_bytes is not None
                and decoded_size_bytes(width, height, img.mode) > max_decoded_bytes
            ):
                raise LivenessError(
                    ErrorCode.PAYLOAD_TOO_LARGE,
                    f"decoded image would exceed {max_decoded_bytes} bytes",
                    Guard.DECODED_BYTES,
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
        raise LivenessError(
            ErrorCode.PAYLOAD_TOO_LARGE, "image exceeds pixel limit", Guard.IMAGE_PIXELS
        ) from exc

    if min(rgb.size) < min_side:
        raise LivenessError(
            ErrorCode.IMAGE_TOO_SMALL, f"image must be at least {min_side}px on each side"
        )
    arr = np.asarray(rgb, dtype=np.uint8)[:, :, ::-1]
    return DecodedImage(bgr=np.ascontiguousarray(arr), media_type=ALLOWED_FORMATS[fmt])
