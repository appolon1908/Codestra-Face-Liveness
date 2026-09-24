"""Regenerate openapi.yaml from the implementation: ``python tools/export_openapi.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from face_liveness.api import create_app
from face_liveness.config import Environment, Settings
from face_liveness.runtime import ModelRuntime


def main() -> None:
    app = create_app(Settings(env=Environment.TEST), ModelRuntime.unavailable("export"))
    spec = json.loads(json.dumps(app.openapi()))
    out = Path(__file__).resolve().parent.parent / "openapi.yaml"
    out.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=100))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
