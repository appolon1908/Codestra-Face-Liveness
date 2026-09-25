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
| `REQUIRE_MODEL_VALIDATION` | `false` (forced `true` in production) | Non-built-in active version needs a passing validation record matching the deployed threshold/calibration id. |
| `FUSION_POLICY_ID` | `passive-only.v1` | Explicit, versioned score-fusion policy. Unknown / unversioned ids fail closed. |
| `ACTIVE_PROVIDER` | *(empty)* | Active liveness provider id. None ships; any value fails closed. |
| `MAX_ACTIVE_FRAMES` | 8 | Frames per active capture (only used when a provider is available). |
| `CAPACITY_PROFILE_ID` | *(empty)* | `profile_id` of the benchmark report the capacity settings came from (informational). |
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
- The challenge does **not** make the result active liveness. With no active provider
  installed, the passive decision under `passive-only.v1` is authoritative, and
  `evidence.active_liveness_evaluated` is `false`. Sending `active_frames_base64` fails
  closed with `503 EVIDENCE_UNAVAILABLE` *before* the challenge is spent.

## Active liveness provider slot

`active.py` defines the `ActiveLivenessProvider` interface (contract `active-provider.v1`).
It has four members: `descriptor`, `ready()`, `instructions(challenge_id)`, and
`evaluate(challenge_id, frames)`. The HTTP contract is already in place:
`capabilities.active`, typed challenge `instructions`, `active_frames_base64` on verify,
and `evidence.active`. **No provider ships**, and there are deliberately no heuristic
providers (blink counting, frame differencing, landmark head pose). They are defeated by
replayed video and must never be presented as production liveness. A provider counts as
available only when it is registered in `KNOWN_PROVIDERS`, is ready, implements the
supported contract version, and names the PAD evaluation that tested it
(`validation_id`). Until then `active_liveness` is `false` everywhere. Setting
`LIVENESS_ACTIVE_PROVIDER` to any id makes `/readyz` fail
(`checks.active_provider`) instead of silently running passive-only.

## Score fusion policy

`LIVENESS_FUSION_POLICY_ID` selects one immutable, versioned policy. Every decision
records the policy in `evidence.policy_id`, and `evidence.evidence_used` lists what
contributed.

| Policy | Rule | Today |
|---|---|---|
| `passive-only.v1` (default) | The decision is the passive decision (`live` iff score ≥ threshold). | Ready. |
| `passive-and-active.v1` | `live` only if passive is `live` **and** a tested active provider returns `pass` (and score ≥ 0.5 if it reports one). | Not ready: no provider. `/readyz` 503, every check `503 EVIDENCE_UNAVAILABLE`. |

Fail-closed rules:

- An unknown id or an id without a version (`passive-only`, `latest`) keeps the service
  not-ready.
- If required evidence is unavailable, the service never falls back to a weaker policy.
- Under a policy that requires active evidence, a request without an active capture
  returns `EVIDENCE_UNAVAILABLE` before any work is done and before the challenge is
  spent.

Only conjunctive fusion exists. A weighted score would need a jointly calibrated
threshold, and that needs a real active provider first.

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

3. Check `face-liveness-models --model-dir /models list` (or `GET /v1/models` on a running
   instance): the new version must show `status: installed` and `digests_verified: true`.
4. Calibrate the new version (below). A threshold calibrated for one version does not
   carry over to another.
5. Validate it: `face-liveness-models --model-dir /models validate 1.1.0
   --calibration-report report.json`. This runs the digest, smoke and calibration-reference
   checks and writes `/models/registry/validations/1.1.0.json`. The version is now a
   `candidate`.
6. Activate it: `face-liveness-models --model-dir /models activate 1.1.0 --env-file deploy.env`
   prints and writes `LIVENESS_ACTIVE_MODEL_VERSION`, `LIVENESS_LIVE_THRESHOLD` and
   `LIVENESS_THRESHOLD_CALIBRATION_ID`. Deploy with those settings. In production, the
   service refuses to load a non-built-in version unless all three match a passing record.

See [MODEL_CARD.md](MODEL_CARD.md#lifecycle-installed--candidate--active) for the exact
checks. The registry is read-only at runtime. Nothing in the API installs, activates or
deletes a version, and no model is ever fetched from a URL. Manifest validation rules
(all fail closed for the active version):

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

## Capacity benchmark

```bash
face-liveness-benchmark --model-dir ./models --concurrency 1,2,4,8 --requests 200 \
  --sizes 640x480,1280x720,1920x1080 --out bench.json      # or --images ./faces
```

- The benchmark measures the in-process pipeline: base64 decode, header-checked image
  decode, detection, and inference. HTTP, auth and network time are not included.
  Inference always runs (on a centred box when no face is found), so synthetic images
  measure the full cost.
- For each image size and concurrency level, the report gives p50/p95/p99/mean/max
  latency, throughput, process CPU seconds and utilisation, peak RSS, error counts, and
  face-detection rate. It also records the host (CPU count and affinity, platform,
  onnxruntime) and the settings that affect cost.
- Run it on hardware and CPU limits that match production, with the production
  `LIVENESS_ONNX_INTRA_OP_THREADS`. Size `MAX_CONCURRENT_CHECKS` from the concurrency
  level where throughput stops increasing, and size `REQUEST_TIMEOUT_SECONDS` from the
  p99 at that level plus the queue wait. Then set
  `LIVENESS_CAPACITY_PROFILE_ID=<profile_id>`. Capabilities report it under
  `limits.capacity_profile`.
- **No auto-tuning.** `auto_tuned` is the constant `false` in the schema. The service
  never reads the report, and referencing a profile changes no limit.

## Metrics

See [TELEMETRY.md](TELEMETRY.md) for every metric, the cardinality / privacy policy, the
resource-guard counter (`liveness_guard_rejections_total{guard}`) and suggested alerts.
