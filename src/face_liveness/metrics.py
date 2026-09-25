"""Prometheus metrics on a service-private registry."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info


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
            "Distribution of ensemble live scores (for drift and calibration monitoring).",
            buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0),
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
