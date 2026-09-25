# Operations and runtime guide

This service is standalone. Middleware V3 is its only intended caller and remains the
integration authority for caller policy, orchestration, retries and cross-system audit.

## Configuration reference

All settings are `LIVENESS_*` environment variables (see `.env.example`).

| Setting | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `/models` | Root for all model artifacts. |
| `MODEL_REGISTRY_DIR` | `<MODEL_DIR>/registry` | Manifests of additional installed versions. |
| `ACTIVE_MODEL_VERSION` | *(empty = `1.0.0`, built-in)* | The single active version. |
| `VERIFY_MODEL_DIGESTS` | `true` (forced in production) | Fail closed on digest mismatch. |
| `LIVE_THRESHOLD` / `THRESHOLD_CALIBRATION_ID` | `0.85` / empty | Decision policy. Set both from a reviewed report. |
| `MAX_IMAGE_BYTES` | 5 MiB | Compressed image size. |
| `MAX_IMAGE_SIDE_PX` | 8192 | Longest side, checked from the header. |
| `MAX_IMAGE_PIXELS` | 25,000,000 | Pixel count, checked from the header. |
| `MAX_DECODED_BYTES` | 192 MiB | Peak decode memory (≈ 12 bytes/px for RGB), checked from the header. About 16 MP RGB. |
| `MAX_CONCURRENT_CHECKS` | 4 | Checks running at once per process. |
| `MAX_QUEUED_CHECKS` | 8 | Checks waiting for a slot. Beyond this: immediate `BUSY`. |
| `BUSY_TIMEOUT_SECONDS` | 2 | Maximum time spent waiting in the queue. |
| `REQUEST_TIMEOUT_SECONDS` | 10 | Total server-side budget per check. |
| `CHALLENGE_TTL_SECONDS` | 120 | Challenge lifetime (10–900). |
| `MAX_OUTSTANDING_CHALLENGES` | 10000 | Memory bound for the challenge store. |

Set `REQUEST_TIMEOUT_SECONDS` below Middleware V3's upstream timeout for this service. Then
the service gives up first and Middleware always gets a definite answer.

Memory sizing: peak memory per check is at most about `MAX_DECODED_BYTES`. Budget the
container for `MAX_CONCURRENT_CHECKS × MAX_DECODED_BYTES`, plus the runtime and models
(about 150 MB), plus the compressed request bodies of queued checks. The defaults
(4 × 192 MiB) fit within the 1 GiB compose limit only when images are that large. Middleware
V3 should downscale captures, because faces do not need more than about 2 MP.

## Challenge flow (Middleware V3)

```
Middleware V3                         Face Liveness
  POST /v1/liveness/challenges   ->   201 {challenge_id, expires_at, instructions: [], active_liveness: false}
  (client captures one image)
  POST /v1/liveness/challenges/{id}/verify {image_base64}
                                 ->   200 {decision, evidence.challenge.status: "consumed", ...}
                                 or   404 CHALLENGE_NOT_FOUND | 410 CHALLENGE_EXPIRED | 409 CHALLENGE_ALREADY_USED
```

- A challenge is spent by the first verify request whose body is valid, whatever the
  outcome (`NO_FACE`, `BUSY`, ...). To retry, issue a new challenge.
- The store is in process memory and holds only SHA-256 hashes of the IDs. With more than
  one replica, verify must reach the replica that issued the challenge (session affinity).
  Otherwise the verify fails closed with `CHALLENGE_NOT_FOUND`. A restart forgets all
  challenges, which also fails closed.
- A spent challenge is remembered until it expires, so a replay returns `409`. After expiry
  it is forgotten and returns `404` (or `410` if it has not been purged yet). Every one of
  these codes means no assessment was made.
- The challenge does **not** make the result active liveness. The passive decision is
  authoritative, and `evidence.active_liveness_evaluated` is always `false`.

## Installing another model version

1. Put the artifacts under the model dir, for example `/models/v2/…`.
2. Write a manifest `/models/registry/<name>.json` that validates against
   `schemas/model-manifest.v1.schema.json`:

   ```json
   {
     "schema_version": 1,
     "model_id": "minifasnet-v2+v1se-ensemble",
     "version": "1.1.0",
     "liveness_type": "passive",
     "score_aggregation": "mean_live_probability",
     "detector": {"name": "yunet-2023mar", "architecture": "yunet",
                  "file": "v2/face_detection_yunet_2023mar.onnx", "sha256": "…"},
     "classifiers": [
       {"name": "minifasnet_v2", "architecture": "minifasnet",
        "file": "v2/minifasnet_v2_2.7_80x80.onnx", "sha256": "…", "crop_scale": 2.7}
     ],
     "license": "Apache-2.0"
   }
   ```

3. Deploy with the current active version. Check `GET /v1/models`: the new version must
   show `status: installed` and `digests_verified: true`.
4. Calibrate the new version (below). A threshold calibrated for one version does not
   carry over to another.
5. Activate it with a deployment: set `LIVENESS_ACTIVE_MODEL_VERSION=1.1.0` together with the
   threshold and calibration ID from the reviewed report for that version.

The registry is read-only at runtime. Nothing in the API installs, activates or deletes a
version. Validation rules (all fail closed for the active version):

- The manifest is strict JSON, at most 64 KiB, and unknown fields are rejected.
- `schema_version` must be 1 and `liveness_type` must be `passive`.
- Artifact paths must be relative. They cannot contain `.`/`..` segments and cannot resolve
  (including through symlinks) outside the model dir.
- Every artifact must exist and match its SHA-256 digest.
- Versions must be unique across all manifests. A duplicated version is ambiguous and is
  treated as invalid.

The **manifest digest** (`model_digest` in evidence, `manifest_sha256` elsewhere) is the
SHA-256 of the canonical manifest JSON. It covers every artifact digest.

## Calibration

```bash
face-liveness-calibrate --data ./calib --model-dir ./models [--model-version 1.1.0] \
  --target-apcer 0.01 --out report.json
```

- Images go through exactly the production pipeline, using the same `LIVENESS_*` input limits and face gates.
  Images rejected before scoring are counted separately by `label:error_code`.
- Metrics: APCER per attack species and worst-species `apcer_max` (FAR-like), and `bpcer`
  (FRR-like). These are reported at reference thresholds and at an approximate EER.
- `threshold_candidate`: the smallest threshold ≥ `--min-threshold` (default 0.5) that
  meets the APCER target.
- `dataset.fingerprint_sha256`: the SHA-256 of the sorted (label, species, image SHA-256)
  triples. It does not depend on file names or ordering.
- `calibration_id`: `cal-<UTC time>-<fingerprint[:12]>-<manifest digest[:8]>`.
- Exit status: 0 = eligible for review, 1 = report written but blocked (see
  `promotion.blockers`), 2 = usage or runtime error.

**No automatic promotion.** `promotion.auto_promoted` is the constant `false` in the schema,
the tool writes only the `--out` file, and the service never reads reports. A human reviewer
adopts a candidate through the normal release process.

## Metrics

| Metric | Meaning |
|---|---|
| `liveness_checks_total{outcome}` | `live`, `spoof`, or an error code. |
| `liveness_rejections_total{route,stage,code}` | Every request rejected without an assessment. `stage` is one of `auth`, `request`, `input`, `quality`, `admission`, `deadline`, `challenge`, `model`, `routing`, `internal`. Body-size rejections before routing have `route="pre_routing"`. |
| `liveness_checks_in_flight`, `liveness_checks_waiting` | Admission slot usage and queue depth. |
| `liveness_challenges_total{event}` | `issued`, `consumed`, or a challenge error code. |
| `liveness_challenges_outstanding` | Unexpired challenges held in memory. |
| `liveness_active_model_info{version,manifest_sha256}` | Identity of the active model version. |
| `liveness_model_ready`, `liveness_threshold`, `liveness_live_score`, `liveness_inference_duration_seconds` | As before. |

Suggested alerts:

- `liveness_model_ready == 0`
- a sustained rate of `liveness_rejections_total{stage="admission"}` or `{stage="deadline"}`,
  which means the service needs capacity
- a spike in `{stage="input",code="PAYLOAD_TOO_LARGE"}`, which suggests abusive inputs
- `liveness_challenges_outstanding` approaching `MAX_OUTSTANDING_CHALLENGES`
