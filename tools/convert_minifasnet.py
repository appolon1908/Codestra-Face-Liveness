"""Convert the upstream MiniFASNet PyTorch weights to ONNX.

Build-time tooling only: requires ``torch`` + ``onnx`` (tools/requirements-convert.txt),
which are never installed in the runtime image. Run after tools/fetch_models.sh:

    python tools/convert_minifasnet.py --model-dir ./models

Produces:
    <model-dir>/minifasnet_v2_2.7_80x80.onnx
    <model-dir>/minifasnet_v1se_4.0_80x80.onnx
    <model-dir>/manifest.json   (SHA-256 of every runtime artifact)

The model definition (MiniFASNet.py) is imported from the pinned, digest-verified
upstream download, not vendored into this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch

# (upstream weights file, factory name, output file, crop scale)
MODELS = [
    ("2.7_80x80_MiniFASNetV2.pth", "MiniFASNetV2", "minifasnet_v2_2.7_80x80.onnx", 2.7),
    ("4_0_0_80x80_MiniFASNetV1SE.pth", "MiniFASNetV1SE", "minifasnet_v1se_4.0_80x80.onnx", 4.0),
]
INPUT_H = INPUT_W = 80
OPSET = 17


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_definition(upstream: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("MiniFASNet", upstream / "MiniFASNet.py")
    if spec is None or spec.loader is None:
        raise SystemExit("MiniFASNet.py not found; run tools/fetch_models.sh first")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    model_dir: Path = args.model_dir
    upstream = model_dir / "upstream"
    definition = _load_definition(upstream)
    kernel = ((INPUT_H + 15) // 16, (INPUT_W + 15) // 16)

    torch.manual_seed(0)
    for weights, factory, output, _scale in MODELS:
        model = getattr(definition, factory)(conv6_kernel=kernel)
        state = torch.load(upstream / weights, map_location="cpu", weights_only=True)
        state = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())
        model.load_state_dict(state)
        model.eval()
        dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)
        torch.onnx.export(
            model,
            (dummy,),
            str(model_dir / output),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=OPSET,
            dynamo=False,
        )
        print(f"wrote {model_dir / output}")

    manifest = {
        "detector": {
            "file": "face_detection_yunet_2023mar.onnx",
            "sha256": _sha256(model_dir / "face_detection_yunet_2023mar.onnx"),
        },
        "antispoof": [
            {
                "file": output,
                "sha256": _sha256(model_dir / output),
                "source_weights": weights,
                "source_weights_sha256": _sha256(upstream / weights),
                "scale": scale,
            }
            for weights, _factory, output, scale in MODELS
        ],
        "torch_version": torch.__version__,
        "opset": OPSET,
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
