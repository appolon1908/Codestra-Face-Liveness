"""Model registry: installed model versions, manifest validation, active-version selection.

Every model version is described by a manifest naming its artifacts and their SHA-256
digests. The built-in version is synthesised from the pinned settings; further versions
may be installed as ``<model_registry_dir>/*.json``. Exactly one version is active
(``LIVENESS_ACTIVE_MODEL_VERSION``, default: built-in). The registry is built once at
startup and is read-only afterwards.

Lifecycle: ``installed`` (manifest valid, digests verified) -> ``candidate`` (a passing
validation record exists for this exact manifest digest; see validation_record.py) ->
``active`` (selected by deployment). When model validation is required (always in
production), a non-built-in version can only be active if it is a candidate *and* the
deployed threshold/calibration id match its validation record.

Fail closed: if the active version's manifest is invalid, ambiguous (duplicate version),
unvalidated when validation is required, or any of its artifacts is missing or does not
match its digest, no model is loaded and the service stays not-ready. Invalid *inactive*
versions are reported but do not affect readiness.

Artifacts are only ever local files under the model dir. There is no download path: a
manifest ``file`` is a relative path, so a URL is rejected by the schema.

Manifests can only describe passive single-image PAD models. There is no manifest type
for active liveness; adding one requires a real active model and a new schema version.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .config import Settings
from .validation_record import (
    ValidationRecord,
    activation_error,
    candidate_error,
    load_record,
    record_path,
)

log = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION: Literal[1] = 1
MAX_MANIFEST_BYTES = 64 * 1024
MAX_REGISTRY_MANIFESTS = 32

BUILTIN_MODEL_ID = "minifasnet-v2+v1se-ensemble"
BUILTIN_MODEL_VERSION = "1.0.0"
BUILTIN_SOURCE = "builtin"

_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._+\-]{0,63}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
# Relative path whose every segment starts with an alphanumeric: no absolute paths,
# no "." / ".." segments, no hidden files.
_FILE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._\-]*(/[A-Za-z0-9][A-Za-z0-9._\-]*)*$"


class ArtifactSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=_ID_PATTERN)
    file: str = Field(
        pattern=_FILE_PATTERN, max_length=255, description="Path relative to the model dir."
    )
    sha256: str = Field(pattern=_SHA256_PATTERN, description="Lower-case hex SHA-256.")


class DetectorSpec(ArtifactSpec):
    architecture: Literal["yunet"]


class ClassifierSpec(ArtifactSpec):
    architecture: Literal["minifasnet"]
    crop_scale: float = Field(ge=1.0, le=8.0, description="Context crop scale around the face.")


class ModelManifest(BaseModel):
    """A single installed model version (schema version 1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    model_id: str = Field(pattern=_ID_PATTERN)
    version: str = Field(pattern=_ID_PATTERN)
    liveness_type: Literal["passive"] = Field(
        description="Only passive single-image PAD is supported by this schema version."
    )
    score_aggregation: Literal["mean_live_probability"]
    detector: DetectorSpec
    classifiers: list[ClassifierSpec] = Field(min_length=1, max_length=8)
    license: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _unique_names(self) -> ModelManifest:
        names = [self.detector.name, *(c.name for c in self.classifiers)]
        if len(set(names)) != len(names):
            raise ValueError("artifact names must be unique")
        return self

    @property
    def artifacts(self) -> list[ArtifactSpec]:
        return [self.detector, *self.classifiers]

    @cached_property
    def digest(self) -> str:
        """SHA-256 over the canonical JSON manifest; covers every artifact digest."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def builtin_manifest(settings: Settings) -> ModelManifest:
    """The built-in version, described by the pinned artifact settings."""
    return ModelManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        model_id=BUILTIN_MODEL_ID,
        version=BUILTIN_MODEL_VERSION,
        liveness_type="passive",
        score_aggregation="mean_live_probability",
        detector=DetectorSpec(
            name="yunet-2023mar",
            architecture="yunet",
            file=settings.detector_file,
            sha256=settings.detector_sha256.lower(),
        ),
        classifiers=[
            ClassifierSpec(
                name="minifasnet_v2",
                architecture="minifasnet",
                file=settings.minifasnet_v2_file,
                sha256=settings.minifasnet_v2_sha256.lower(),
                crop_scale=2.7,
            ),
            ClassifierSpec(
                name="minifasnet_v1se",
                architecture="minifasnet",
                file=settings.minifasnet_v1se_file,
                sha256=settings.minifasnet_v1se_sha256.lower(),
                crop_scale=4.0,
            ),
        ],
        license="Apache-2.0 (MiniFASNet weights), MIT (YuNet)",
        description="Silent-Face-Anti-Spoofing MiniFASNetV2 + V1SE ensemble, YuNet detector.",
    )


class EntryStatus(StrEnum):
    ACTIVE = "active"
    CANDIDATE = "candidate"
    INSTALLED = "installed"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    source: str
    status: EntryStatus
    manifest: ModelManifest | None
    manifest_sha256: str | None
    digests_verified: bool
    error: str | None
    validation: ValidationRecord | None = None
    validation_error: str | None = None

    @property
    def version(self) -> str | None:
        return self.manifest.version if self.manifest is not None else None


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    entries: tuple[RegistryEntry, ...]
    active_version: str
    active_error: str | None = None

    @property
    def active(self) -> RegistryEntry | None:
        return next((e for e in self.entries if e.status is EntryStatus.ACTIVE), None)

    def get(self, version: str) -> RegistryEntry | None:
        return next((e for e in self.entries if e.version == version), None)

    @classmethod
    def single(cls, manifest: ModelManifest) -> ModelRegistry:
        """In-memory registry with one pre-verified active version (tests, tooling)."""
        entry = RegistryEntry(
            source=BUILTIN_SOURCE,
            status=EntryStatus.ACTIVE,
            manifest=manifest,
            manifest_sha256=manifest.digest,
            digests_verified=True,
            error=None,
        )
        return cls(entries=(entry,), active_version=manifest.version)


def parse_manifest(path: Path) -> ModelManifest:
    """Parse and validate one manifest file. Raises ValueError with a safe message."""
    try:
        size = path.stat().st_size
        if size > MAX_MANIFEST_BYTES:
            raise ValueError(f"manifest exceeds {MAX_MANIFEST_BYTES} bytes")
        data = json.loads(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"manifest unreadable: {type(exc).__name__}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("manifest is not valid JSON") from exc
    try:
        return ModelManifest.model_validate(data)
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) or "manifest" for e in exc.errors()})
        raise ValueError("manifest schema invalid: " + ", ".join(fields)) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(model_dir: Path, file: str) -> Path:
    """Resolve an artifact path, refusing anything that escapes the model dir."""
    root = model_dir.resolve()
    path = (root / file).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"artifact path escapes model dir: {file}")
    return path


def _verify_artifacts(
    manifest: ModelManifest, model_dir: Path, enforce: bool, cache: dict[Path, str]
) -> tuple[bool, str | None]:
    """Return (all digests matched, fatal error). Mismatch is fatal only when enforced."""
    all_match = True
    for art in manifest.artifacts:
        try:
            path = resolve_artifact(model_dir, art.file)
        except ValueError as exc:
            return False, str(exc)
        if not path.is_file():
            return False, f"model artifact missing: {art.name} ({art.file})"
        if path not in cache:
            cache[path] = _sha256(path)
        if cache[path] != art.sha256:
            all_match = False
            if enforce:
                return False, f"model artifact digest mismatch: {art.name} ({art.file})"
            log.warning("model artifact digest mismatch (not enforced)", extra={"file": art.file})
    return all_match, None


def build_registry(settings: Settings) -> ModelRegistry:
    """Discover, validate and verify every installed version. Never raises."""
    candidates: list[tuple[str, ModelManifest | None, str | None]] = [
        (BUILTIN_SOURCE, builtin_manifest(settings), None)
    ]
    registry_dir = settings.resolved_model_registry_dir
    if registry_dir.is_dir():
        files = sorted(registry_dir.glob("*.json"))
        if len(files) > MAX_REGISTRY_MANIFESTS:
            log.warning(
                "too many model manifests; ignoring extras",
                extra={"found": len(files), "limit": MAX_REGISTRY_MANIFESTS},
            )
        for path in files[:MAX_REGISTRY_MANIFESTS]:
            source = f"registry/{path.name}"
            try:
                candidates.append((source, parse_manifest(path), None))
            except ValueError as exc:
                candidates.append((source, None, str(exc)))

    versions = [m.version for _, m, _ in candidates if m is not None]
    duplicates = {v for v in versions if versions.count(v) > 1}
    active_version = settings.active_model_version or BUILTIN_MODEL_VERSION
    hash_cache: dict[Path, str] = {}
    entries: list[RegistryEntry] = []
    active_error = None
    for source, manifest, error in candidates:
        verified = False
        record: ValidationRecord | None = None
        record_error: str | None = None
        if manifest is not None and error is None:
            if manifest.version in duplicates:
                error = f"duplicate model version: {manifest.version}"
            else:
                verified, error = _verify_artifacts(
                    manifest, settings.model_dir, settings.verify_model_digests, hash_cache
                )
        if manifest is not None and error is None:
            record, record_error = _validation(registry_dir, manifest, verified)
        if error is not None or manifest is None:
            status = EntryStatus.INVALID
        elif manifest.version == active_version:
            status = EntryStatus.ACTIVE
            if source != BUILTIN_SOURCE and settings.model_validation_required:
                if record is None or record_error is not None:
                    gate: str | None = record_error or "no validation record"
                else:
                    gate = activation_error(
                        record,
                        manifest.version,
                        manifest.digest,
                        settings.live_threshold,
                        settings.threshold_calibration_id,
                    )
                if gate is not None:
                    status = EntryStatus.INSTALLED if record_error else EntryStatus.CANDIDATE
                    active_error = f"active model version not validated for activation: {gate}"
        elif record_error is None:
            status = EntryStatus.CANDIDATE
        else:
            status = EntryStatus.INSTALLED
        entries.append(
            RegistryEntry(
                source=source,
                status=status,
                manifest=manifest,
                manifest_sha256=manifest.digest if manifest is not None else None,
                digests_verified=verified,
                error=error,
                validation=record,
                validation_error=record_error,
            )
        )

    if active_error is None and not any(e.status is EntryStatus.ACTIVE for e in entries):
        matching = [e for e in entries if e.version == active_version]
        active_error = (
            matching[0].error
            if matching and matching[0].error
            else f"active model version not installed: {active_version}"
        )
    return ModelRegistry(
        entries=tuple(entries), active_version=active_version, active_error=active_error
    )


def _validation(
    registry_dir: Path, manifest: ModelManifest, digests_verified: bool
) -> tuple[ValidationRecord | None, str | None]:
    """The version's validation record and why it does not make it a candidate."""
    try:
        record = load_record(record_path(registry_dir, manifest.version))
    except ValueError as exc:
        return None, str(exc)
    error = candidate_error(record, manifest.version, manifest.digest)
    if error is None and not digests_verified:
        error = "artifact digests not verified"
    return record, error
