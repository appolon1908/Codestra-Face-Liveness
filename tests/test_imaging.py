from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from face_liveness.errors import ErrorCode, LivenessError
from face_liveness.imaging import decode_base64, decode_image

from .conftest import b64, encode_image


def test_decodes_to_bgr():
    arr = np.zeros((150, 200, 3), dtype=np.uint8)
    arr[:, :, 0] = 255  # pure red in RGB
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    img = decode_image(buf.getvalue(), max_pixels=10**7, min_side=100)
    assert (img.width, img.height) == (200, 150)
    assert tuple(img.bgr[0, 0]) == (0, 0, 255)


def test_exif_orientation_applied():
    buf = io.BytesIO()
    im = Image.fromarray(np.zeros((150, 300, 3), dtype=np.uint8))
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90 CW
    im.save(buf, format="JPEG", exif=exif)
    img = decode_image(buf.getvalue(), max_pixels=10**7, min_side=100)
    assert (img.width, img.height) == (150, 300)


def test_grayscale_and_alpha_converted():
    for mode in ("L", "RGBA", "P"):
        buf = io.BytesIO()
        Image.new(mode, (160, 160)).save(buf, format="PNG")
        img = decode_image(buf.getvalue(), max_pixels=10**7, min_side=100)
        assert img.bgr.shape == (160, 160, 3)


def test_animated_rejected():
    frames = [Image.new("RGB", (160, 160), c) for c in ("red", "blue")]
    buf = io.BytesIO()
    frames[0].save(buf, format="PNG", save_all=True, append_images=frames[1:])
    with pytest.raises(LivenessError) as exc:
        decode_image(buf.getvalue(), max_pixels=10**7, min_side=100)
    assert exc.value.code is ErrorCode.INVALID_IMAGE


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ("%%%", ErrorCode.INVALID_REQUEST),
        ("", ErrorCode.INVALID_IMAGE),
        ("QUJD\nREVG", ErrorCode.INVALID_REQUEST),
    ],
)
def test_bad_base64(payload, code):
    with pytest.raises(LivenessError) as exc:
        decode_base64(payload, max_bytes=10**6)
    assert exc.value.code is code


def test_base64_size_precheck():
    with pytest.raises(LivenessError) as exc:
        decode_base64(b64(encode_image()), max_bytes=1000)
    assert exc.value.code is ErrorCode.PAYLOAD_TOO_LARGE
