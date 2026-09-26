"""Prometheus metrics on a service-private registry.

Cardinality / privacy policy (docs/TELEMETRY.md): every label value comes from a closed
enum (decision, reason, error code, stage, guard, event), a route template, or the one
active model version / fusion policy id fixed at startup. Request data never becomes a
label: no request or challenge ids, subject ids or names, image hashes, client
addresses, or scores (scores are only observed into fixed histogram buckets).
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info

# The only label names any metric may use (enforced by tests/test_telemetry.py).
ALLOWED_LABEL_NAMES = frozenset(
    {
        "route",
        "method",
        "status",
        "outcome",
        "stage",
        "code",
        "event",
        "decision",
        "reason",
        "model_version",
        "policy_id",
        "guard",
        # Info metrics: one series each, identity of the running build/model.
        "version",
        "manifest_sha256",
        "model",
        "detector",
    }
)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.http_requests = Counter(
            "liveness_http_requests_total",
            "HTTP requests by route template, method and status code.",
            ["route", "method", "status"],
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "liveness_http_request_duration_seconds",
            "HTTP request latency by route template.",
            ["route", "method"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )
        self.checks = Counter(
            "liveness_checks_total",
            "Liveness checks by outcome (live, spoof, or an error code).",
            ["outcome"],
            registry=self.registry,
        )
        self.inference_duration = Histogram(
            "liveness_inference_duration_seconds",
            "Face detection + anti-spoofing inference time.",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            registry=self.registry,
        )
        self.live_score = Histogram(
            "liveness_live_score",
            "Distribution of passive live scores by model version (drift / calibration "
            "monitoring). Scores are bucketed, never used as label values.",
            ["model_version"],
            buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0),
            registry=self.registry,
        )
        self.decisions = Counter(
            "liveness_decisions_total",
            "Completed assessments by final decision, decision reason, model version and "
            "fusion policy.",
            ["decision", "reason", "model_version", "policy_id"],
            registry=self.registry,
        )
        self.decision_duration = Histogram(
            "liveness_decision_duration_seconds",
            "Server-side time from request receipt to a completed assessment.",
            ["decision", "model_version"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
            registry=self.registry,
        )
        self.guard_rejections = Counter(
            "liveness_guard_rejections_total",
            "Requests rejected by a resource guard: request_body, image_bytes, "
            "image_dimensions, image_pixels, decoded_bytes (size); queue_full, "
            "queue_timeout, challenge_capacity (concurrency); deadline (timeout).",
            ["guard"],
            registry=self.registry,
        )
        self.fusion_policy_ready = Gauge(
            "liveness_fusion_policy_ready",
            "1 if the configured fusion policy has all the evidence it requires, else 0.",
            ["policy_id"],
            registry=self.registry,
        )
        self.active_provider_available = Gauge(
            "liveness_active_provider_available",
            "1 if a tested active liveness provider is available, else 0.",
            registry=self.registry,
        )
        self.model_ready = Gauge(
            "liveness_model_ready",
            "1 if the model runtime is loaded and verified, else 0.",
            registry=self.registry,
        )
        self.threshold = Gauge(
            "liveness_threshold", "Configured live-score threshold.", registry=self.registry
        )
        self.rejections = Counter(
            "liveness_rejections_total",
            "Requests rejected without an assessment, by route template, stage and error code.",
            ["route", "stage", "code"],
            registry=self.registry,
        )
        self.challenges = Counter(
            "liveness_challenges_total",
            "Challenge lifecycle events (issued, consumed, or a rejection error code).",
            ["event"],
            registry=self.registry,
        )
        self.challenges_outstanding = Gauge(
            "liveness_challenges_outstanding",
            "Issued, unexpired challenges held in memory (consumed ones included until expiry).",
            registry=self.registry,
        )
        self.checks_in_flight = Gauge(
            "liveness_checks_in_flight",
            "Checks currently holding an admission slot.",
            registry=self.registry,
        )
        self.checks_waiting = Gauge(
            "liveness_checks_waiting",
            "Checks currently waiting for an admission slot.",
            registry=self.registry,
        )
        self.active_model = Info(
            "liveness_active_model",
            "Active model version and manifest digest.",
            registry=self.registry,
        )
        self.build = Info("liveness_build", "Service and model identity.", registry=self.registry)
