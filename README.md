# Codestra Face Liveness

Standalone passive face-liveness and presentation-attack-detection API for Codestra FACE-ID.

## Authority

This repository owns **liveness / anti-spoof decisions only**. It does not identify people, enroll faces, control cameras, store business records, or authorize access. Cross-system orchestration is owned by Middleware V3.

Canonical integration:

```
Caddy -> Kong -> Middleware V3 :8095 -> Codestra Face Liveness API
```

No caller should bypass Middleware for cross-system effects. The service can still be run and tested independently.

## API

Operational endpoints:

- `GET /healthz` — process liveness
- `GET /readyz` — model/auth readiness
- `GET /health/ready` — Middleware V3 readiness alias
- `GET /metrics` — private Prometheus metrics
- `GET /v1/capabilities` — methods, thresholds, limits, model/calibration metadata
- `GET /v1/liveness/status` — local runtime/model status

Liveness:

- `POST /v1/liveness/check` — passive single-image liveness check
- `POST /v1/liveness/challenges` — issue a short-lived, single-use challenge (freshness nonce)
- `POST /v1/liveness/challenges/{challenge_id}/verify` — consume the challenge and run the passive check

Model registry (read-only):

- `GET /v1/models` — installed model versions, validation status, the single active version
- `GET /v1/models/{version}` — one installed version

The committed `openapi.yaml` and `schemas/*.json` (model manifest, calibration report) are
generated from the implementation (`make openapi`) and are tested for exact equality.

## What the liveness result does and does not mean

- The **passive** MiniFASNet ensemble is the only liveness model. Its decision is authoritative.
- `active_liveness` is `false` in capabilities, challenges and evidence. Challenges prove
  freshness and single use only; they carry no instructions and nothing about a user's
  response to a challenge is evaluated. The flag can only become true when a real active
  model exists, which needs a new manifest schema version. Schema v1 accepts
  `liveness_type: passive` only.
- Every `200` response includes `evidence`: model id/version, manifest digest, detector,
  threshold, whether the threshold is calibrated, calibration id, score, margin, and a fixed
  decision reason (`score_at_or_above_threshold` / `score_below_threshold`). Evidence has a
  fixed shape and holds only identifiers and scalars. It never holds image bytes, crops,
  embeddings or tensors.
- Any non-2xx response means no assessment was made. Callers must fail closed.

## Model registry

The built-in version `1.0.0` is described by the pinned digests. Additional versions are
installed as manifests in `<model_dir>/registry/*.json` (schema: `schemas/model-manifest.v1.schema.json`)
and selected with `LIVENESS_ACTIVE_MODEL_VERSION`. Only one version is active. If the active
manifest is invalid or duplicated, or if any of its artifacts is missing, outside the model dir,
or fails its SHA-256 digest, the service stays not-ready (fail closed). Invalid inactive
versions are reported by `/v1/models` but do not affect readiness.

## Calibration

```bash
face-liveness-calibrate --data ./calib --model-dir ./models --out report.json
# or: python tools/calibrate.py ...
```

The dataset layout is `bona_fide/` plus `attack/<species>/`. The report
(`schemas/calibration-report.v1.schema.json`) contains:

- APCER per species and worst-species APCER (the FAR-like rate: spoofs accepted)
- BPCER (the FRR-like rate: live faces rejected)
- an approximate EER
- a threshold candidate
- a dataset fingerprint
- the model version and manifest digest

The report is only a candidate. `promotion.auto_promoted` is always `false`. The service
never reads reports, and the tool never changes configuration. A reviewer adopts a threshold
by setting `LIVENESS_LIVE_THRESHOLD` and `LIVENESS_THRESHOLD_CALIBRATION_ID` through the
release process.

## Resource guards

- Image limits: compressed bytes, longest side, pixel count, and decoded bitmap bytes. All of
  them are checked from the image header before any pixel data is decompressed, so
  decompression bombs are rejected cheaply with `413 PAYLOAD_TOO_LARGE`.
- Bounded concurrency (`MAX_CONCURRENT_CHECKS`) with a bounded wait queue (`MAX_QUEUED_CHECKS`).
  Requests beyond the queue get `503 BUSY` with `Retry-After` immediately.
- A per-request time budget (`REQUEST_TIMEOUT_SECONDS`). The service never returns a
  decision after the budget has elapsed. It returns a retryable `503 DEADLINE_EXCEEDED`
  instead.
- `liveness_rejections_total{route,stage,code}` and in-flight/waiting gauges record rejections
  in structured metrics.

See [docs/OPERATIONS.md](docs/OPERATIONS.md) for configuration, installing model versions,
the Middleware V3 challenge flow, calibration procedure and metrics.

## Security and privacy

- fail closed when the model is unavailable or digest verification fails
- no raw image persistence by default
- no biometric identity matching in this service
- service auth is environment/secret-reference based
- metrics and internal operational data must not be routed publicly
- model provenance, license, digest, threshold, and calibration identifier are exposed as operational metadata
- production thresholds require environment-specific calibration evidence; calibration reports are never auto-promoted
- challenge IDs are 256-bit random tokens held only as SHA-256 hashes in process memory

A normal face match is **not** liveness. This service only reports the anti-spoof result produced by its configured liveness model.

## Local development

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pip install --no-deps -e .
make check
make openapi
```

To fetch and convert pinned model artifacts:

```bash
make models
```

The model fetch/convert path verifies committed SHA-256 digests and preserves upstream license/provenance information.

## Container

```bash
docker build -t codestra/face-liveness:dev .
docker compose config
docker compose up --build
```

The container runs as a non-root user, uses a read-only filesystem, and expects runtime secrets through configured secret files/references.
The image contains the built-in model version and the `face-liveness-calibrate` CLI. To install
additional model versions, mount their manifests and artifacts read-only under `/models`
(see `docker-compose.yml` and docs/OPERATIONS.md).

## Promotion

Code promotes through repository-owned environment branches:

```
development -> testing -> staging -> production
```

The API environment name used in contracts is `test` while the Git branch is `testing`.

Production activation remains a separate release decision and must not be inferred from code existing on an integration branch.
