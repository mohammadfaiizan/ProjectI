"""
METRICS AND PROMETHEUS
======================

Problem Statement:
Systems must expose quantitative measurements of their behavior. Prometheus
is the de-facto open-source metrics collection system; understanding its
data model, scrape model, and PromQL is essential for SRE work.

Prometheus Data Model:
  Every metric is a time series identified by:
    <metric_name>{<label1>="<val1>", <label2>="<val2>", ...}

  Metric types:
    Counter:   Monotonically increasing. Resets on restart.
               rate(http_requests_total[5m]) → per-second rate.
    Gauge:     Up/down value. Memory usage, queue depth.
    Histogram: Samples observations into buckets + sum + count.
               histogram_quantile(0.99, rate(...))
    Summary:   Pre-computed quantiles on client side.

Scrape Model:
  Prometheus pulls (scrapes) /metrics endpoints every scrape_interval.
  Target discovery: static config, Kubernetes SD, Consul SD, EC2 SD.
  Pushgateway: for short-lived jobs that can't be scraped.

PromQL (Prometheus Query Language):
  Instant vector:  http_requests_total{job="api"}
  Range vector:    http_requests_total[5m]
  rate():          per-second average over range
  irate():         instantaneous rate (last 2 samples, spiky)
  increase():      total increase over range
  histogram_quantile(0.99, rate(latency_bucket[5m]))
  sum by (pod):    aggregation
  topk(5, ...):    top 5 series

Recording Rules:
  Pre-compute expensive queries; stored as new metrics.
  Example: record: job:http_requests:rate5m
           expr:   rate(http_requests_total[5m])

Alerting Rules:
  ALERTS{alertname, severity, ...} fire when expr is true for `for` duration.
  Routed via Alertmanager: grouping, inhibition, silences, receivers (PD/Slack).

Cardinality:
  HIGH CARDINALITY KILLS PROMETHEUS. Avoid labels with unbounded values
  (user_id, request_id, IP address). Each unique label combination = 1 series.
  10k series is fine. 10M series = OOM.

Remote Write / Thanos / Cortex:
  Prometheus is single-node. For HA + long-term retention:
  Thanos: sidecar reads Prometheus blocks, uploads to object storage.
  Cortex/Mimir: horizontally scalable Prometheus-compatible TSDB.

Exemplars:
  Link metrics to traces. Histogram observation can carry trace_id exemplar.
  Enables "click on a high-latency bucket → jump to trace".
"""

from __future__ import annotations

import math
import time
import random
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# LABEL SET (immutable dict for hashing)
# ─────────────────────────────────────────────

class LabelSet:
    """Frozen label key=value pairs; used as dict key."""

    def __init__(self, labels: Dict[str, str]):
        self._labels = dict(sorted(labels.items()))

    def __eq__(self, other):
        return isinstance(other, LabelSet) and self._labels == other._labels

    def __hash__(self):
        return hash(tuple(self._labels.items()))

    def __repr__(self):
        pairs = ",".join(f'{k}="{v}"' for k, v in self._labels.items())
        return "{" + pairs + "}"

    def get(self, key: str) -> Optional[str]:
        return self._labels.get(key)

    def merge(self, extra: Dict[str, str]) -> "LabelSet":
        merged = {**self._labels, **extra}
        return LabelSet(merged)


# ─────────────────────────────────────────────
# TIME SERIES SAMPLE
# ─────────────────────────────────────────────

@dataclass
class Sample:
    timestamp: float
    value:     float


# ─────────────────────────────────────────────
# COUNTER
# ─────────────────────────────────────────────

class Counter:
    """
    Monotonically increasing metric.
    Tracks total count; use rate() in PromQL for per-second rate.
    """

    def __init__(self, name: str, help_text: str, label_names: List[str]):
        self.name        = name
        self.help        = help_text
        self.label_names = label_names
        self._series: Dict[LabelSet, float] = defaultdict(float)
        self._lock = threading.Lock()

    def labels(self, **kwargs) -> "_CounterChild":
        ls = LabelSet(kwargs)
        return _CounterChild(self, ls)

    def inc(self, labels: LabelSet, amount: float = 1.0):
        if amount < 0:
            raise ValueError("Counter can only increase")
        with self._lock:
            self._series[labels] += amount

    def value(self, labels: LabelSet) -> float:
        return self._series.get(labels, 0.0)

    def collect(self) -> List[Tuple[LabelSet, float]]:
        with self._lock:
            return list(self._series.items())

    def rate(self, labels: LabelSet, samples: List[Sample]) -> float:
        """Simulated rate() — increase per second over sample window."""
        if len(samples) < 2:
            return 0.0
        delta_v = samples[-1].value - samples[0].value
        delta_t = samples[-1].timestamp - samples[0].timestamp
        return delta_v / delta_t if delta_t > 0 else 0.0


class _CounterChild:
    def __init__(self, counter: Counter, labels: LabelSet):
        self._c = counter
        self._ls = labels

    def inc(self, amount: float = 1.0):
        self._c.inc(self._ls, amount)

    def get(self) -> float:
        return self._c.value(self._ls)


# ─────────────────────────────────────────────
# GAUGE
# ─────────────────────────────────────────────

class Gauge:
    """Arbitrary up/down numeric value."""

    def __init__(self, name: str, help_text: str, label_names: List[str]):
        self.name        = name
        self.help        = help_text
        self.label_names = label_names
        self._series: Dict[LabelSet, float] = {}
        self._lock = threading.Lock()

    def labels(self, **kwargs) -> "_GaugeChild":
        return _GaugeChild(self, LabelSet(kwargs))

    def set(self, labels: LabelSet, value: float):
        with self._lock:
            self._series[labels] = value

    def inc(self, labels: LabelSet, amount: float = 1.0):
        with self._lock:
            self._series[labels] = self._series.get(labels, 0.0) + amount

    def dec(self, labels: LabelSet, amount: float = 1.0):
        with self._lock:
            self._series[labels] = self._series.get(labels, 0.0) - amount

    def value(self, labels: LabelSet) -> float:
        return self._series.get(labels, 0.0)

    def collect(self) -> List[Tuple[LabelSet, float]]:
        with self._lock:
            return list(self._series.items())


class _GaugeChild:
    def __init__(self, gauge: Gauge, labels: LabelSet):
        self._g = gauge
        self._ls = labels

    def set(self, v: float):   self._g.set(self._ls, v)
    def inc(self, a: float = 1.0): self._g.inc(self._ls, a)
    def dec(self, a: float = 1.0): self._g.dec(self._ls, a)
    def get(self) -> float:    return self._g.value(self._ls)


# ─────────────────────────────────────────────
# HISTOGRAM
# ─────────────────────────────────────────────

class Histogram:
    """
    Samples observations into configurable buckets.
    Also tracks _sum and _count for average.
    PromQL: histogram_quantile(0.99, rate(name_bucket[5m]))
    """

    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(self, name: str, help_text: str, label_names: List[str],
                 buckets: Optional[List[float]] = None):
        self.name        = name
        self.help        = help_text
        self.label_names = label_names
        self.buckets     = sorted(buckets or self.DEFAULT_BUCKETS) + [math.inf]
        # Per label-set: {bucket_upper: count}, _sum, _count
        self._data: Dict[LabelSet, Dict] = {}
        self._lock = threading.Lock()

    def _init_labels(self, ls: LabelSet):
        if ls not in self._data:
            self._data[ls] = {
                "buckets": {b: 0 for b in self.buckets},
                "sum":     0.0,
                "count":   0,
            }

    def observe(self, labels: LabelSet, value: float):
        with self._lock:
            self._init_labels(labels)
            d = self._data[labels]
            d["sum"]   += value
            d["count"] += 1
            for b in self.buckets:
                if value <= b:
                    d["buckets"][b] += 1

    def quantile(self, labels: LabelSet, q: float) -> float:
        """
        Linear interpolation quantile from bucket counts.
        Approximates histogram_quantile() in PromQL.
        """
        with self._lock:
            if labels not in self._data:
                return 0.0
            d       = self._data[labels]
            total   = d["count"]
            if total == 0:
                return 0.0
            target  = q * total
            prev_b  = 0.0
            prev_c  = 0
            for b, cnt in sorted(d["buckets"].items()):
                if b == math.inf:
                    break
                if cnt >= target:
                    # Interpolate within [prev_b, b]
                    span = b - prev_b
                    frac = (target - prev_c) / max(cnt - prev_c, 1)
                    return prev_b + span * frac
                prev_b = b
                prev_c = cnt
            return prev_b

    def avg(self, labels: LabelSet) -> float:
        d = self._data.get(labels)
        if not d or d["count"] == 0:
            return 0.0
        return d["sum"] / d["count"]

    def labels(self, **kwargs) -> "_HistogramChild":
        return _HistogramChild(self, LabelSet(kwargs))


class _HistogramChild:
    def __init__(self, hist: Histogram, labels: LabelSet):
        self._h  = hist
        self._ls = labels

    def observe(self, value: float):
        self._h.observe(self._ls, value)

    def p99(self)  -> float: return self._h.quantile(self._ls, 0.99)
    def p95(self)  -> float: return self._h.quantile(self._ls, 0.95)
    def p50(self)  -> float: return self._h.quantile(self._ls, 0.50)
    def avg(self)  -> float: return self._h.avg(self._ls)


# ─────────────────────────────────────────────
# PROMETHEUS REGISTRY
# ─────────────────────────────────────────────

class PrometheusRegistry:
    """
    Central registry for all metrics.
    Mimics prometheus_client.CollectorRegistry.
    Produces /metrics text format output.
    """

    def __init__(self):
        self._metrics: Dict[str, object] = {}

    def register_counter(self, name: str, help_text: str,
                         label_names: List[str]) -> Counter:
        m = Counter(name, help_text, label_names)
        self._metrics[name] = m
        return m

    def register_gauge(self, name: str, help_text: str,
                       label_names: List[str]) -> Gauge:
        m = Gauge(name, help_text, label_names)
        self._metrics[name] = m
        return m

    def register_histogram(self, name: str, help_text: str,
                           label_names: List[str],
                           buckets: Optional[List[float]] = None) -> Histogram:
        m = Histogram(name, help_text, label_names, buckets)
        self._metrics[name] = m
        return m

    def text_format(self) -> str:
        """Produce Prometheus text exposition format."""
        lines = []
        for name, metric in self._metrics.items():
            if isinstance(metric, Counter):
                lines.append(f"# HELP {name} {metric.help}")
                lines.append(f"# TYPE {name} counter")
                for ls, v in metric.collect():
                    lines.append(f"{name}{ls} {v}")
            elif isinstance(metric, Gauge):
                lines.append(f"# HELP {name} {metric.help}")
                lines.append(f"# TYPE {name} gauge")
                for ls, v in metric.collect():
                    lines.append(f"{name}{ls} {v}")
            elif isinstance(metric, Histogram):
                lines.append(f"# HELP {name} {metric.help}")
                lines.append(f"# TYPE {name} histogram")
                for ls, d in metric._data.items():
                    for b, cnt in d["buckets"].items():
                        bu = "+Inf" if b == math.inf else str(b)
                        lines.append(f'{name}_bucket{ls.merge({"le": bu})} {cnt}')
                    lines.append(f"{name}_sum{ls} {d['sum']:.4f}")
                    lines.append(f"{name}_count{ls} {d['count']}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# ALERTING RULE ENGINE
# ─────────────────────────────────────────────

@dataclass
class AlertRule:
    name:       str
    expr_fn:    object          # callable() → float; fires when > 0
    for_secs:   float           # must be true for this long
    severity:   str
    summary:    str

@dataclass
class FiringAlert:
    rule_name:  str
    severity:   str
    summary:    str
    started_at: float
    value:      float


class AlertManager:
    """Evaluates alerting rules; routes to receivers."""

    def __init__(self):
        self._rules:   List[AlertRule]  = []
        self._pending: Dict[str, float] = {}   # rule_name → first_true_ts
        self._firing:  Dict[str, FiringAlert] = {}

    def add_rule(self, rule: AlertRule):
        self._rules.append(rule)

    def evaluate(self) -> List[FiringAlert]:
        now    = time.time()
        newly  = []
        for rule in self._rules:
            try:
                val = rule.expr_fn()
            except Exception:
                val = 0.0

            if val > 0:
                if rule.name not in self._pending:
                    self._pending[rule.name] = now
                elapsed = now - self._pending[rule.name]
                if elapsed >= rule.for_secs and rule.name not in self._firing:
                    alert = FiringAlert(rule.name, rule.severity,
                                        rule.summary, now, val)
                    self._firing[rule.name] = alert
                    newly.append(alert)
            else:
                self._pending.pop(rule.name, None)
                self._firing.pop(rule.name, None)
        return newly

    def firing(self) -> List[FiringAlert]:
        return list(self._firing.values())


# ─────────────────────────────────────────────
# RECORDING RULE
# ─────────────────────────────────────────────

@dataclass
class RecordingRule:
    """Pre-compute expensive queries into a new metric."""
    record:  str          # name of derived metric
    expr_fn: object       # callable() → float
    labels:  Dict[str, str] = field(default_factory=dict)


class RuleEngine:
    def __init__(self, registry: PrometheusRegistry):
        self._registry = registry
        self._rules: List[RecordingRule] = []
        self._gauge  = registry.register_gauge(
            "recorded_metric", "Pre-computed recording rules", ["rule"])

    def add_recording_rule(self, rule: RecordingRule):
        self._rules.append(rule)

    def evaluate(self):
        for rule in self._rules:
            try:
                val = rule.expr_fn()
                ls  = LabelSet({"rule": rule.record})
                self._gauge.set(ls, val)
            except Exception:
                pass


# ─────────────────────────────────────────────
# SIMULATED SERVICE
# ─────────────────────────────────────────────

class APIService:
    """
    Simulated HTTP API service that instruments itself with Prometheus metrics.
    """

    def __init__(self, registry: PrometheusRegistry):
        self.requests = registry.register_counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"])

        self.latency = registry.register_histogram(
            "http_request_duration_seconds",
            "HTTP request latency",
            ["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])

        self.in_flight = registry.register_gauge(
            "http_requests_in_flight",
            "In-flight HTTP requests",
            ["endpoint"])

        self._request_samples: List[Sample] = []

    def handle(self, method: str, endpoint: str,
               latency_s: float, status: int):
        ls_req  = LabelSet({"method": method, "endpoint": endpoint,
                            "status_code": str(status)})
        ls_lat  = LabelSet({"method": method, "endpoint": endpoint})
        ls_inf  = LabelSet({"endpoint": endpoint})

        self.in_flight.inc(ls_inf)
        self.latency.observe(ls_lat, latency_s)
        self.requests.inc(ls_req)
        self._request_samples.append(Sample(time.time(), self.requests.value(ls_req)))
        self.in_flight.dec(ls_inf)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_prometheus():
    print("=" * 65)
    print("METRICS AND PROMETHEUS")
    print("=" * 65)

    registry = PrometheusRegistry()
    svc      = APIService(registry)

    # ── Simulate Traffic ──────────────────────
    print("\n[1] SIMULATING TRAFFIC")
    print("─" * 55)

    random.seed(42)
    endpoints = ["/api/users", "/api/orders", "/api/search"]
    methods   = ["GET", "POST"]

    for _ in range(200):
        ep     = random.choice(endpoints)
        method = "GET" if ep == "/api/search" else random.choice(methods)
        # Simulate latency: search is slower
        if ep == "/api/search":
            lat = random.expovariate(1 / 0.8)   # avg 800ms
        else:
            lat = random.expovariate(1 / 0.05)  # avg 50ms
        status = 500 if random.random() < 0.03 else 200
        svc.handle(method, ep, lat, status)

    print("  Simulated 200 requests across 3 endpoints")

    # ── Latency Quantiles ─────────────────────
    print("\n[2] LATENCY QUANTILES (histogram_quantile)")
    print("─" * 55)

    for ep in endpoints:
        h_child = svc.latency.labels(method="GET", endpoint=ep)
        print(f"  {ep}")
        print(f"    p50={h_child.p50()*1000:.1f}ms  "
              f"p95={h_child.p95()*1000:.1f}ms  "
              f"p99={h_child.p99()*1000:.1f}ms  "
              f"avg={h_child.avg()*1000:.1f}ms")

    # ── Request Counts ────────────────────────
    print("\n[3] REQUEST COUNTERS")
    print("─" * 55)

    for ep in endpoints:
        ok  = svc.requests.value(LabelSet({"method": "GET", "endpoint": ep, "status_code": "200"}))
        err = svc.requests.value(LabelSet({"method": "GET", "endpoint": ep, "status_code": "500"}))
        print(f"  {ep}: 200={int(ok)}  500={int(err)}")

    # ── Alerting Rules ────────────────────────
    print("\n[4] ALERTING RULES")
    print("─" * 55)

    am = AlertManager()

    # Alert if p99 latency > 2s on search
    search_hist = svc.latency.labels(method="GET", endpoint="/api/search")
    am.add_rule(AlertRule(
        "HighSearchLatency",
        expr_fn=lambda: max(0, search_hist.p99() - 2.0),
        for_secs=0,   # fire immediately for demo
        severity="warning",
        summary="Search p99 latency > 2s",
    ))

    # Alert if error counter > 5
    def error_count():
        total = 0
        for ep in endpoints:
            for m in methods:
                total += svc.requests.value(
                    LabelSet({"method": m, "endpoint": ep, "status_code": "500"}))
        return total

    am.add_rule(AlertRule(
        "HighErrorRate",
        expr_fn=lambda: max(0, error_count() - 5),
        for_secs=0,
        severity="critical",
        summary="More than 5 HTTP 500 errors",
    ))

    fired = am.evaluate()
    if fired:
        for alert in fired:
            print(f"  FIRING [{alert.severity.upper()}] {alert.rule_name}: {alert.summary}")
            print(f"    value={alert.value:.3f}")
    else:
        print("  No alerts firing")

    all_firing = am.firing()
    print(f"  Total firing alerts: {len(all_firing)}")

    # ── Recording Rules ───────────────────────
    print("\n[5] RECORDING RULES (pre-computed metrics)")
    print("─" * 55)

    rule_engine = RuleEngine(registry)
    rule_engine.add_recording_rule(RecordingRule(
        record="job:http_request_duration_p99:search",
        expr_fn=lambda: search_hist.p99(),
    ))
    rule_engine.add_recording_rule(RecordingRule(
        record="job:http_errors_total",
        expr_fn=error_count,
    ))
    rule_engine.evaluate()

    gauge = registry._metrics["recorded_metric"]
    for ls, v in gauge.collect():
        print(f"  {ls} = {v:.4f}")

    # ── Cardinality Warning ───────────────────
    print("\n[6] CARDINALITY GUIDELINES")
    print("─" * 55)

    guidelines = [
        ("GOOD label",  "status_code",  "~5 values (200/404/500/503/429)"),
        ("GOOD label",  "endpoint",     "~10-50 stable routes"),
        ("BAD label",   "user_id",      "millions of users → OOM"),
        ("BAD label",   "request_id",   "unique per request → millions of series"),
        ("BAD label",   "ip_address",   "unbounded in production"),
        ("OK label",    "pod",          "~100 pods; manageable"),
        ("OK label",    "region",       "~5-20 regions"),
    ]
    for quality, label, note in guidelines:
        print(f"  [{quality:<12}] {label:<15} — {note}")

    # ── /metrics Output (excerpt) ─────────────
    print("\n[7] /metrics TEXT FORMAT (excerpt)")
    print("─" * 55)

    text = registry.text_format()
    lines = text.split("\n")
    print("\n".join(lines[:20]))
    print(f"  ... ({len(lines)} total lines)")

    # ── Prometheus vs Alternatives ────────────
    print("\n[8] PROMETHEUS VS ALTERNATIVES")
    print("─" * 55)

    comparison = [
        ("Prometheus",  "Pull model, PromQL, local TSDB, 15s resolution, no HA"),
        ("Thanos",      "Prometheus + object storage for long retention + HA"),
        ("Cortex/Mimir","Horizontally scalable Prometheus; multi-tenant; S3 backend"),
        ("Datadog",     "SaaS, agent push, 15 months retention, expensive at scale"),
        ("InfluxDB",    "Push model, Flux/InfluxQL, good for IoT/high-rate writes"),
        ("VictoriaMetrics","Prometheus compat, faster, lower RAM, no PromQL edge cases"),
        ("OpenMetrics", "Standardizes Prometheus text format as IETF RFC"),
    ]
    for tool, desc in comparison:
        print(f"  {tool:<20} {desc}")


if __name__ == "__main__":
    demonstrate_prometheus()
