"""Published JSON Schemas for file formats owned by this service (not HTTP payloads)."""

from __future__ import annotations

from typing import Any

from .benchmark import BenchmarkReport
from .calibration import CalibrationReport
from .registry import ModelManifest
from .validation_record import ValidationRecord

SCHEMA_DIR = "schemas"


def json_schemas() -> dict[str, dict[str, Any]]:
    return {
        "model-manifest.v1.schema.json": ModelManifest.model_json_schema(),
        "calibration-report.v1.schema.json": CalibrationReport.model_json_schema(),
        "model-validation.v1.schema.json": ValidationRecord.model_json_schema(),
        "benchmark-report.v1.schema.json": BenchmarkReport.model_json_schema(),
    }
