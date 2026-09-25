"""Regenerate published contracts from the implementation: ``python tools/export_openapi.py``.

Writes openapi.yaml and the JSON Schemas for model manifests and calibration reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from face_liveness.api import create_app
from face_liveness.config import Environment, Settings
from face_liveness.contracts import SCHEMA_DIR, json_schemas
from face_liveness.runtime import ModelRuntime

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    app = create_app(Settings(env=Environment.TEST), ModelRuntime.unavailable("export"))
    spec = json.loads(json.dumps(app.openapi()))
    out = ROOT / "openapi.yaml"
    out.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=100))
    print(f"wrote {out}")
    schema_dir = ROOT / SCHEMA_DIR
    schema_dir.mkdir(exist_ok=True)
    for name, schema in json_schemas().items():
        path = schema_dir / name
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
