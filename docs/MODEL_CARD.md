# Model card and model lifecycle

## Built-in version `1.0.0`: `minifasnet-v2+v1se-ensemble`

| | |
|---|---|
| Task | Passive, single-image presentation-attack detection (PAD): bona fide vs. print / screen replay / mask. |
| Not for | Face matching, identity verification, or age, gender and emotion estimation. It is also not active (challenge-response) liveness. |
| Detector | YuNet `face_detection_yunet_2023mar.onnx` (OpenCV Zoo `47534e27`, MIT). |
| Classifiers | MiniFASNetV2 (context crop 2.7×) and MiniFASNetV1SE (4.0×), 80×80 BGR input, from Silent-Face-Anti-Spoofing `b6d5f04a` (Apache-2.0), converted to ONNX by `tools/convert_minifasnet.py`. |
| Score | Mean over the ensemble of softmax[live]. The decision is `live` iff score ≥ `LIVENESS_LIVE_THRESHOLD`. |
| Provenance | `tools/fetch_models.sh` downloads pinned upstream commits and verifies SHA-256 for every file. The converted artifacts are pinned in `config.py` and `tools/model_digests.sha256`. Nothing is downloaded at runtime. |
| Threshold | `0.85` is a provisional, **uncalibrated** default. Production needs a calibration report on data from the target cameras (`face-liveness-calibrate`). |
| Known limits | Trained on upstream data of unknown demographic balance. Sensitive to capture conditions (blur, extreme pose, low light). Not evaluated here against 3D masks or high-quality replays. Report APCER per attack species and do not rely on one aggregate number. |

## Lifecycle: installed → candidate → active

```
 manifest in <registry>/*.json         validation record in             deployment sets
 + artifacts match their digests       <registry>/validations/<v>.json   LIVENESS_ACTIVE_MODEL_VERSION
        installed  ──── face-liveness-models validate ────▶  candidate  ──── activate ────▶  active
                   (digest + smoke + calibration reference)            (local operator action)
```

| State | Meaning | Shown by |
|---|---|---|
| `invalid` | Manifest invalid or duplicated, or an artifact is missing, escapes the model dir, or fails its digest. It never loads. | `GET /v1/models`, `face-liveness-models list` |
| `installed` | Manifest valid and digests verified, but no passing validation record for *this* manifest digest. | same |
| `candidate` | A passing validation record binds this exact manifest digest to a smoke test and a calibration report. | same (+ `validation`) |
| `active` | The single version selected by the deployment. | same, `evidence.model_version` |

### Validation (`face-liveness-models validate <version> --calibration-report report.json`)

It writes `<registry>/validations/<version>.json` (schema:
`schemas/model-validation.v1.schema.json`). The record passes only if all three checks
pass:

1. **Digests**: every artifact exists under the model dir and matches its manifest
   SHA-256. Digest verification cannot be switched off for validation.
2. **Smoke**: the version loads through the production runtime. That includes the ONNX
   self-test and a check that each classifier returns `(1, 3)` finite logits. Detection
   then runs on three synthetic images. Every classifier returns a finite score in
   `[0, 1]` for a centred crop, the number of component scores matches the manifest,
   and repeated scoring is deterministic.
3. **Calibration reference**: a `face-liveness-calibrate` report exists for exactly this
   model version and manifest digest. Its pipeline settings (detector threshold, minimum
   face and image size, secondary face ratio) are equal to the current settings. It is
   `eligible_for_review` and has a threshold candidate. The record stores the report's
   SHA-256, calibration id, threshold, APCER and BPCER.

Exit status: 0 = candidate, 1 = record written but not a candidate, 2 = usage error.

### Activation (`face-liveness-models activate <version> [--env-file deploy.env]`)

Activation re-checks the record against the installed manifest and its digests. It then
prints the three deployment settings: `LIVENESS_ACTIVE_MODEL_VERSION`,
`LIVENESS_LIVE_THRESHOLD`, and `LIVENESS_THRESHOLD_CALIBRATION_ID`. With `--env-file` it
also writes them into that local file and leaves every other line unchanged. It never
contacts the service, never changes a running process, and never downloads anything.
The version becomes active at the next deployment.

### Enforcement in the service

When `LIVENESS_REQUIRE_MODEL_VALIDATION` is on (**always in production**), a non-built-in
active version is loaded only if all of these hold:

- it is a candidate (a passing record for its exact manifest digest)
- `LIVENESS_THRESHOLD_CALIBRATION_ID` equals the record's calibration id
- `LIVENESS_LIVE_THRESHOLD` is ≥ the record's threshold candidate. A stricter threshold
  is allowed; a looser one is not.

If any check fails, the service stays not-ready with
`active model version not validated for activation: …`. The service fails closed and
never falls back to another version. Replacing an artifact after validation changes its
digest, so the entry becomes `invalid`.

The built-in version is exempt from the record requirement. It is pinned in code, and
its threshold is reported as uncalibrated until a calibration id is configured.

### No remote models

Manifest `file` entries must be relative paths under the model dir (schema pattern, plus
a resolve-and-contain check that also catches symlink escapes). A URL cannot be
expressed. The promotion CLI accepts only installed version ids. Installing a version
means placing files and a manifest on disk (or mounting them read-only in the
container) through your normal artifact pipeline.

## Capacity profile

`face-liveness-benchmark` (schema: `schemas/benchmark-report.v1.schema.json`) measures
one model version on one host with the production decode, detection and inference path.
It reports p50/p95/p99 latency, throughput, CPU and peak RSS for each image size and
concurrency level. Record the report's `profile_id` in `LIVENESS_CAPACITY_PROFILE_ID`
once you have set `MAX_CONCURRENT_CHECKS`, `MAX_QUEUED_CHECKS` and
`REQUEST_TIMEOUT_SECONDS` from it. Capabilities then show which tested profile the
deployment was sized from. The service never reads benchmark reports and never tunes
itself.
