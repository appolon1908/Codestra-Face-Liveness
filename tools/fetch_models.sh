#!/usr/bin/env bash
# Fetch pinned upstream model artifacts and verify their SHA-256 digests.
#
# Downloads:
#   - YuNet face detector (ONNX, MIT)                 -> $MODEL_DIR/face_detection_yunet_2023mar.onnx
#   - MiniFASNet anti-spoofing weights (PyTorch, Apache-2.0)
#                                                    -> $MODEL_DIR/upstream/*.pth
#   - MiniFASNet model definition source (Apache-2.0) -> $MODEL_DIR/upstream/MiniFASNet.py
#
# The .pth weights must then be converted to ONNX with tools/convert_minifasnet.py
# (see docs/MODEL_CARD.md). Nothing here is committed to git.
set -euo pipefail

MODEL_DIR="${1:-${LIVENESS_MODEL_DIR:-./models}}"
SFAS_COMMIT="b6d5f04ad78778917853b25c778acef6d5626d15"
ZOO_COMMIT="47534e27c9851bb1128ccc0102f1145e27f23f98"
SFAS="https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/${SFAS_COMMIT}"
ZOO="https://media.githubusercontent.com/media/opencv/opencv_zoo/${ZOO_COMMIT}"

mkdir -p "${MODEL_DIR}/upstream"

fetch() {
  local url="$1" dest="$2" sha="$3"
  if [[ -f "${dest}" ]] && echo "${sha}  ${dest}" | sha256sum -c --status; then
    echo "ok (cached) ${dest}"
    return
  fi
  curl -fsSL --retry 3 -o "${dest}.tmp" "${url}"
  if ! echo "${sha}  ${dest}.tmp" | sha256sum -c --status; then
    echo "SHA-256 mismatch for ${url}" >&2
    rm -f "${dest}.tmp"
    exit 1
  fi
  mv "${dest}.tmp" "${dest}"
  echo "ok ${dest}"
}

fetch "${ZOO}/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
  "${MODEL_DIR}/face_detection_yunet_2023mar.onnx" \
  "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
fetch "${SFAS}/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth" \
  "${MODEL_DIR}/upstream/2.7_80x80_MiniFASNetV2.pth" \
  "a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0"
fetch "${SFAS}/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth" \
  "${MODEL_DIR}/upstream/4_0_0_80x80_MiniFASNetV1SE.pth" \
  "84ee1d37d96894d5e82de5a57df044ef80a58be2b218b5ed7cdfd875ec2f5990"
fetch "${SFAS}/LICENSE" "${MODEL_DIR}/upstream/LICENSE.Silent-Face-Anti-Spoofing" \
  "daf94bf1dc9cc5700fe5af2c7c0cbd1836e70d509ed78fd0bebef7432edee6fb"
fetch "${SFAS}/src/model_lib/MiniFASNet.py" "${MODEL_DIR}/upstream/MiniFASNet.py" \
  "e498c4ec5e1ddfaba62b941a126c19d65aa564999f3309661fe43ee8bf38acd7"
fetch "https://raw.githubusercontent.com/opencv/opencv_zoo/${ZOO_COMMIT}/models/face_detection_yunet/LICENSE" \
  "${MODEL_DIR}/upstream/LICENSE.YuNet" \
  "c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958"
