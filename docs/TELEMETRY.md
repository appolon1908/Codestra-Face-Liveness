# Telemetry and cardinality policy

`GET /metrics` is private (never routed through Caddy/Kong). All metrics live on a
service-private Prometheus registry (`src/face_liveness/metrics.py`).

## Policy

Telemetry describes the **service's behaviour**, never a **person or a request**.

1. **Label values come from closed sets only.** A value is either a member of an enum
   in the code (decision, decision reason, error code, stage, guard, challenge event), a
   route *template* (`/v1/liveness/challenges/{challenge_id}/verify`, never the concrete
   path), or one of two identifiers fixed at startup: the active model version and the
   fusion policy id.
2. **Never used as a label, in any metric:**
   - request ids, challenge ids, or any other per-request token
   - subject, user, person, badge, or enrolment ids, and names
   - image bytes, image hashes, face crops, embeddings, or any image-derived value
   - raw or rounded scores or margins (scores are *observed* into fixed histogram buckets)
   - client IP addresses, user agents, or caller-supplied headers
   - unknown/unmatched paths (they are reported as `route="unmatched"`)
3. **Allowed label names** are listed in `metrics.ALLOWED_LABEL_NAMES`.
   `tests/test_telemetry.py` exercises every endpoint, including error paths with
   request ids, challenge ids and unknown paths. It then fails if any exposed label name
   is not on the list, or if any label value looks like a score (a float) or a hash
   (32 or more hex characters). The only exception is `manifest_sha256` on the
   single-series `liveness_active_model_info`.
4. **Bounded series.** Worst case per process:
   `decisions_total` ≤ 2 decisions × 5 reasons × 1 model version × 1 policy = 10 series.
   Rejections ≤ routes × error codes (both closed). Guard rejections = 9 series. The
   model version label stays bounded because only the *active* version is ever emitted,
   and it can change only on a restart.
5. **New metrics** must follow the same rules. Adding a label name means editing
   `ALLOWED_LABEL_NAMES` in code review, together with a justification in this file.

Structured access logs follow the same rules for identifiers. They contain the request
id (needed for correlation with Middleware V3) and the rounded live score, but never
image data, challenge ids, or subject information.

## Metrics

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `liveness_decisions_total` | counter | `decision`, `reason`, `model_version`, `policy_id` | Completed assessments. `reason` is the fixed decision reason. |
| `liveness_decision_duration_seconds` | histogram | `decision`, `model_version` | Receipt → completed assessment, server side. |
| `liveness_live_score` | histogram | `model_version` | Passive score distribution (drift / calibration). |
| `liveness_checks_total` | counter | `outcome` | `live`, `spoof`, or an error code. |
| `liveness_rejections_total` | counter | `route`, `stage`, `code` | Every request rejected without an assessment. |
| `liveness_guard_rejections_total` | counter | `guard` | Resource guard rejections (below). |
| `liveness_checks_in_flight` / `_waiting` | gauge | — | Admission slot use / queue depth. |
| `liveness_challenges_total` | counter | `event` | `issued`, `consumed`, or a challenge error code. |
| `liveness_challenges_outstanding` | gauge | — | Unexpired challenges in memory. |
| `liveness_fusion_policy_ready` | gauge | `policy_id` | 0 = every assessment fails closed. Unknown ids are reported as `unknown`. |
| `liveness_active_provider_available` | gauge | — | 1 only when a tested active provider is plugged in (0 today). |
| `liveness_model_ready`, `liveness_threshold` | gauge | — | Readiness and configured threshold. |
| `liveness_inference_duration_seconds` | histogram | — | Detection + anti-spoof inference. |
| `liveness_http_requests_total`, `liveness_http_request_duration_seconds` | counter / histogram | `route`, `method`, `status` | HTTP layer. |
| `liveness_active_model_info`, `liveness_build_info` | info | `version`, `manifest_sha256`, `model`, `detector` | Identity (one series each). |

### Guard values

| `guard` | Class | Trigger |
|---|---|---|
| `request_body` | size | HTTP body above the limit (before routing) |
| `image_bytes` | size | Compressed image above `MAX_IMAGE_BYTES` |
| `image_dimensions` | size | Longest side above `MAX_IMAGE_SIDE_PX` (header check) |
| `image_pixels` | size | Pixel count above `MAX_IMAGE_PIXELS` (header check / bomb guard) |
| `decoded_bytes` | size | Decoded bitmap above `MAX_DECODED_BYTES` (header check) |
| `queue_full` | concurrency | All slots busy and the wait queue is full: immediate `BUSY` |
| `queue_timeout` | concurrency | Waited `BUSY_TIMEOUT_SECONDS` without getting a slot |
| `challenge_capacity` | concurrency | `MAX_OUTSTANDING_CHALLENGES` reached |
| `deadline` | timeout | `REQUEST_TIMEOUT_SECONDS` exceeded: no late decision |

## Suggested alerts

- `liveness_model_ready == 0` or `liveness_fusion_policy_ready == 0`: the service fails
  closed and every check returns 503.
- `rate(liveness_guard_rejections_total{guard=~"queue_.*|deadline"}[5m]) > 0` sustained:
  the service is out of capacity. Compare with the benchmark profile referenced in
  `capabilities.limits.capacity_profile`.
- A spike in `guard=~"image_.*|decoded_bytes|request_body"`: abusive or misconfigured
  clients.
- A shift in the `liveness_live_score` distribution or in the
  `liveness_decisions_total{decision="spoof"}` ratio for a fixed `model_version`:
  possible input drift. Recalibrate before you change the threshold.
