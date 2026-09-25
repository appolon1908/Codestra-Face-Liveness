"""Published JSON Schemas for file formats owned by this service (not HTTP payloads)."""

from __future__ import annotations

from typing import Any

from .calibration import CalibrationReport
from .registry import ModelManifest

SCHEMA_DIR = "schemas"


def json_schemas() -> dict[str, dict[str, Any]]:
    return {
        "model-manifest.v1.schema.json": ModelManifest.model_json_schema(),
        "calibration-report.v1.schema.json": CalibrationReport.model_json_schema(),
    }
