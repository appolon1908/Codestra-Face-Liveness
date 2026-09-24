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
- `POST /v1/liveness/check` — passive single-image liveness check

The committed `openapi.yaml` is generated from the runtime and is tested for exact equality.

## Security and privacy

- fail closed when the model is unavailable or digest verification fails
- no raw image persistence by default
- no biometric identity matching in this service
- service auth is environment/secret-reference based
- metrics and internal operational data must not be routed publicly
- model provenance, license, digest, threshold, and calibration identifier are exposed as operational metadata
- production thresholds require environment-specific calibration evidence

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

## Promotion

Code promotes through repository-owned environment branches:

```
development -> testing -> staging -> production
```

The API environment name used in contracts is `test` while the Git branch is `testing`.

Production activation remains a separate release decision and must not be inferred from code existing on an integration branch.
