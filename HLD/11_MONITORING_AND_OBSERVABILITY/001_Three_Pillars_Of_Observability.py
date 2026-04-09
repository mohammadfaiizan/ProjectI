"""
THREE PILLARS OF OBSERVABILITY
================================

Problem Statement:
In distributed systems, things go wrong in unexpected ways.
Observability is the ability to understand the internal state of a system
by examining its outputs — without needing to add new instrumentation for
every new question.

Three Pillars:
  1. Metrics:  Numeric time-series data. "What is happening?"
               Count, gauge, histogram. Aggregated. Low cardinality.
               Examples: request_count, latency_p99, error_rate, cpu_percent.
               Tools: Prometheus, Datadog, InfluxDB, CloudWatch.

  2. Logs:     Discrete events with context. "What happened?"
               Structured (JSON) or unstructured. High cardinality OK.
               Examples: HTTP access logs, error logs, audit logs.
               Tools: ELK Stack, Splunk, Loki, CloudWatch Logs.

  3. Traces:   End-to-end journey of a request across services. "Where did it go slow?"
               Spans: start/end/name/metadata. Parent-child relationships.
               TraceID: follows request across all services.
               Tools: Jaeger, Zipkin, AWS X-Ray, OpenTelemetry.

Why Three Pillars Together:
  Metrics → detect an anomaly (latency spiked).
  Logs    → understand what happened (error messages).
  Traces  → pinpoint where it happened (which service, which DB query).

The Observability Maturity Model:
  Level 1: Basic logging (application errors).
  Level 2: Structured logs + metric alerts.
  Level 3: Distributed tracing.
  Level 4: Correlated pillars (trace_id in logs and metrics).
  Level 5: Automated root cause analysis.

Golden Signals (Google SRE):
  Latency:   Time to serve a request (p50, p95, p99).
  Traffic:   Requests per second.
  Errors:    Error rate (5xx / total).
  Saturation: How "full" is the service? CPU%, queue depth.

RED Method (for services):
  Rate:    Requests per second.
  Errors:  Error rate.
  Duration: Latency distribution.

USE Method (for resources):
  Utilization: % of time resource is busy.
  Saturation:  Amount of extra work queued.
  Errors:      Error count.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import time
import uuid
import threading
import statistics
from collections import defaultdict, deque


# ─────────────────────────────────────────────
# METRIC TYPES
# ─────────────────────────────────────────────

class MetricType(Enum):
    COUNTER   = "counter"    # monotonically increasing (e.g., requests_total)
    GAUGE     = "gauge"      # can go up/down (e.g., active_connections)
    HISTOGRAM = "histogram"  # distribution of values (e.g., request_duration)


@dataclass
class MetricSample:
    name      : str
    value     : float
    labels    : Dict[str, str]
    timestamp : float = field(default_factory=time.time)

    def label_key(self) -> str:
        return ",".join(f"{k}={v}" for k, v in sorted(self.labels.items()))


class MetricsRegistry:
    """Simple in-memory metrics registry (Prometheus-like)."""

    def __init__(self):
        self._counters  : Dict[str, float] = {}
        self._gauges    : Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._samples   : List[MetricSample] = []
        self._lock      = threading.Lock()

    def counter_inc(self, name: str, labels: Dict = None, value: float = 1.0):
        key = f"{name}{{{self._label_str(labels)}}}"
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value
            self._samples.append(MetricSample(name, self._counters[key], labels or {}))

    def gauge_set(self, name: str, value: float, labels: Dict = None):
        key = f"{name}{{{self._label_str(labels)}}}"
        with self._lock:
            self._gauges[key] = value
            self._samples.append(MetricSample(name, value, labels or {}))

    def histogram_observe(self, name: str, value: float, labels: Dict = None):
        key = f"{name}{{{self._label_str(labels)}}}"
        with self._lock:
            self._histograms[key].append(value)
            self._samples.append(MetricSample(name, value, labels or {}))

    def percentile(self, name: str, labels: Dict, p: float) -> Optional[float]:
        key = f"{name}{{{self._label_str(labels)}}}"
        vals = sorted(self._histograms.get(key, []))
        if not vals:
            return None
        idx = int(len(vals) * p / 100)
        return vals[min(idx, len(vals) - 1)]

    def get_gauge(self, name: str, labels: Dict = None) -> Optional[float]:
        key = f"{name}{{{self._label_str(labels)}}}"
        return self._gauges.get(key)

    def get_counter(self, name: str, labels: Dict = None) -> float:
        key = f"{name}{{{self._label_str(labels)}}}"
        return self._counters.get(key, 0)

    def _label_str(self, labels: Dict = None) -> str:
        if not labels:
            return ""
        return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))

    def snapshot(self) -> List[Dict]:
        return [{"name": s.name, "value": s.value, "labels": s.labels,
                  "ts": s.timestamp} for s in self._samples[-20:]]


# ─────────────────────────────────────────────
# STRUCTURED LOGGER
# ─────────────────────────────────────────────

class LogLevel(Enum):
    DEBUG = 10
    INFO  = 20
    WARN  = 30
    ERROR = 40
    FATAL = 50


@dataclass
class LogEntry:
    level       : LogLevel
    message     : str
    service     : str
    timestamp   : float
    trace_id    : Optional[str]
    span_id     : Optional[str]
    fields      : Dict[str, Any]

    def to_json(self) -> str:
        import json
        return json.dumps({
            "level"    : self.level.name,
            "message"  : self.message,
            "service"  : self.service,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)),
            "trace_id" : self.trace_id,
            "span_id"  : self.span_id,
            **self.fields,
        })


class StructuredLogger:
    """JSON structured logger with correlation IDs."""

    def __init__(self, service: str, min_level: LogLevel = LogLevel.INFO):
        self.service   = service
        self.min_level = min_level
        self._entries  : List[LogEntry] = []
        self._trace_id : Optional[str] = None
        self._span_id  : Optional[str] = None

    def bind(self, trace_id: str, span_id: str = None) -> "StructuredLogger":
        """Return logger bound to a trace context."""
        logger = StructuredLogger(self.service, self.min_level)
        logger._trace_id = trace_id
        logger._span_id  = span_id
        logger._entries  = self._entries   # shared
        return logger

    def _log(self, level: LogLevel, message: str, **fields):
        if level.value < self.min_level.value:
            return
        entry = LogEntry(
            level=level, message=message, service=self.service,
            timestamp=time.time(), trace_id=self._trace_id,
            span_id=self._span_id, fields=fields,
        )
        self._entries.append(entry)

    def debug(self, msg: str, **kw): self._log(LogLevel.DEBUG, msg, **kw)
    def info (self, msg: str, **kw): self._log(LogLevel.INFO,  msg, **kw)
    def warn (self, msg: str, **kw): self._log(LogLevel.WARN,  msg, **kw)
    def error(self, msg: str, **kw): self._log(LogLevel.ERROR, msg, **kw)

    def recent(self, n: int = 10) -> List[LogEntry]:
        return self._entries[-n:]


# ─────────────────────────────────────────────
# DISTRIBUTED TRACE
# ─────────────────────────────────────────────

@dataclass
class Span:
    span_id    : str
    trace_id   : str
    parent_id  : Optional[str]
    name       : str
    service    : str
    start_time : float
    end_time   : Optional[float] = None
    tags       : Dict[str, Any]  = field(default_factory=dict)
    logs       : List[Dict]      = field(default_factory=list)
    error      : bool = False

    def finish(self, error: bool = False):
        self.end_time = time.time()
        self.error    = error

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def log_event(self, event: str, **fields):
        self.logs.append({"ts": time.time(), "event": event, **fields})


class Tracer:
    """OpenTelemetry-inspired distributed tracer."""

    def __init__(self, service: str):
        self.service  = service
        self._spans   : Dict[str, Span] = {}
        self._traces  : Dict[str, List[Span]] = defaultdict(list)
        self._lock    = threading.Lock()

    def start_span(self, name: str, trace_id: str = None,
                   parent_id: str = None, tags: Dict = None) -> Span:
        trace_id = trace_id or uuid.uuid4().hex
        span_id  = uuid.uuid4().hex[:16]
        span     = Span(
            span_id=span_id, trace_id=trace_id, parent_id=parent_id,
            name=name, service=self.service, start_time=time.time(),
            tags=tags or {},
        )
        with self._lock:
            self._spans[span_id] = span
            self._traces[trace_id].append(span)
        return span

    def finish_span(self, span: Span, error: bool = False):
        span.finish(error=error)

    def get_trace(self, trace_id: str) -> List[Span]:
        return self._traces.get(trace_id, [])

    def print_trace(self, trace_id: str):
        spans = sorted(self._traces.get(trace_id, []), key=lambda s: s.start_time)
        if not spans:
            return
        t0 = spans[0].start_time
        for span in spans:
            offset = (span.start_time - t0) * 1000
            dur    = span.duration_ms or 0
            indent = "  " if span.parent_id else ""
            status = "ERR" if span.error else "OK"
            print(f"    {indent}{span.name:<30} [{status}] "
                  f"+{offset:.0f}ms  dur={dur:.1f}ms  svc={span.service}")


# ─────────────────────────────────────────────
# GOLDEN SIGNALS CALCULATOR
# ─────────────────────────────────────────────

class GoldenSignals:
    """Compute Google SRE Golden Signals from a MetricsRegistry."""

    def __init__(self, registry: MetricsRegistry):
        self._reg = registry

    def latency_p99(self, service: str) -> Optional[float]:
        return self._reg.percentile("request_duration_ms",
                                     {"service": service}, 99)

    def error_rate(self, service: str) -> float:
        total  = self._reg.get_counter("requests_total", {"service": service})
        errors = self._reg.get_counter("requests_total",
                                        {"service": service, "status": "error"})
        if not total:
            return 0.0
        return errors / total

    def rps(self, service: str) -> float:
        # Simplified: just return current counter
        return self._reg.get_counter("requests_total", {"service": service})

    def saturation(self, service: str) -> Optional[float]:
        return self._reg.get_gauge("cpu_utilization_pct", {"service": service})

    def summary(self, service: str) -> Dict:
        return {
            "latency_p99_ms": self.latency_p99(service),
            "error_rate"    : self.error_rate(service),
            "rps"           : self.rps(service),
            "saturation_pct": self.saturation(service),
        }


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_observability():
    print("=" * 65)
    print("THREE PILLARS OF OBSERVABILITY")
    print("=" * 65)

    import random
    random.seed(42)

    # ── Pillar 1: Metrics ─────────────────────────
    print("\n[1] PILLAR 1: METRICS")
    print("─" * 55)

    registry = MetricsRegistry()

    # Simulate traffic
    services = ["user-service", "order-service", "payment-service"]
    for _ in range(100):
        svc = random.choice(services)
        lat = random.gauss(50, 20)
        err = random.random() < 0.05
        registry.counter_inc("requests_total", {"service": svc})
        if err:
            registry.counter_inc("requests_total", {"service": svc, "status": "error"})
        registry.histogram_observe("request_duration_ms", lat, {"service": svc})

    for svc in services:
        registry.gauge_set("cpu_utilization_pct",
                            random.uniform(20, 80), {"service": svc})

    gs = GoldenSignals(registry)
    print(f"  {'Service':<20} {'Req/s':>7} {'Error%':>8} {'P99ms':>8} {'CPU%':>7}")
    print(f"  {'─'*52}")
    for svc in services:
        s = gs.summary(svc)
        print(f"  {svc:<20} {s['rps']:>7.0f} "
              f"{s['error_rate']:>7.1%} "
              f"{s['latency_p99_ms'] or 0:>7.1f}ms "
              f"{s['saturation_pct'] or 0:>6.1f}%")

    # ── Pillar 2: Logs ────────────────────────────
    print("\n\n[2] PILLAR 2: STRUCTURED LOGS")
    print("─" * 55)

    logger    = StructuredLogger("user-service")
    trace_id  = uuid.uuid4().hex
    span_id   = uuid.uuid4().hex[:16]
    bound_log = logger.bind(trace_id, span_id)

    bound_log.info("User login successful", user_id="alice", ip="10.0.0.1",
                    method="webauthn")
    bound_log.warn("Rate limit approaching", user_id="bob", current=95, limit=100)
    bound_log.error("DB connection failed", db_host="pg-primary", retry=3,
                     error="connection_refused")

    print(f"  Sample log entries (JSON):")
    for entry in logger.recent(3):
        print(f"  {entry.to_json()}")

    # ── Pillar 3: Traces ──────────────────────────
    print("\n\n[3] PILLAR 3: DISTRIBUTED TRACES")
    print("─" * 55)

    api_tracer     = Tracer("api-gateway")
    user_tracer    = Tracer("user-service")
    db_tracer      = Tracer("database")

    # Simulate a request flowing through services
    trace_id = uuid.uuid4().hex
    root_span = api_tracer.start_span("POST /api/orders", trace_id=trace_id,
                                       tags={"http.method": "POST", "http.path": "/api/orders"})

    user_span = user_tracer.start_span("validateUser", trace_id=trace_id,
                                        parent_id=root_span.span_id,
                                        tags={"user_id": "alice"})
    db_span   = db_tracer.start_span("SELECT users", trace_id=trace_id,
                                      parent_id=user_span.span_id,
                                      tags={"db.type": "postgresql"})

    time.sleep(0.005)
    db_span.log_event("query_executed", rows=1)
    db_tracer.finish_span(db_span)

    time.sleep(0.003)
    user_tracer.finish_span(user_span)

    pay_span = api_tracer.start_span("callPaymentService", trace_id=trace_id,
                                      parent_id=root_span.span_id)
    time.sleep(0.020)
    api_tracer.finish_span(pay_span)
    api_tracer.finish_span(root_span)

    # Collect all spans across tracers
    all_spans = (api_tracer.get_trace(trace_id) +
                 user_tracer.get_trace(trace_id) +
                 db_tracer.get_trace(trace_id))

    spans_sorted = sorted(all_spans, key=lambda s: s.start_time)
    t0 = spans_sorted[0].start_time
    print(f"  Trace {trace_id[:16]}...:")
    for s in spans_sorted:
        offset = (s.start_time - t0) * 1000
        indent = "    " if s.parent_id else "  "
        print(f"{indent}{s.name:<32} {s.service:<20} "
              f"+{offset:.0f}ms  {s.duration_ms:.1f}ms")

    total_dur = (spans_sorted[-1].end_time - spans_sorted[0].start_time) * 1000
    print(f"  Total request duration: {total_dur:.1f}ms")

    # ── Correlated Pillars ────────────────────────
    print("\n\n[4] CORRELATED OBSERVABILITY — trace_id IN ALL PILLARS")
    print("─" * 55)

    print(f"  trace_id: {trace_id[:24]}")
    print(f"  Metric:  request_duration_ms{{trace_id=...}} included in histogram")
    print(f"  Log:     {{trace_id: \"{trace_id[:16]}...\", level: ERROR}}")
    print(f"  Trace:   {len(all_spans)} spans across {len({s.service for s in all_spans})} services")
    print(f"\n  → Click trace_id in log → open trace → see exact slow DB query")

    # ── Method Summary ────────────────────────────
    print("\n\n[5] OBSERVABILITY METHODS")
    print("─" * 55)

    methods = [
        ("Golden Signals", "Latency, Traffic, Errors, Saturation (Google SRE)"),
        ("RED Method",     "Rate, Errors, Duration (services)"),
        ("USE Method",     "Utilization, Saturation, Errors (resources/infrastructure)"),
        ("SLI",            "Service Level Indicator: actual measured metric"),
        ("SLO",            "Service Level Objective: target (e.g., 99.9% success)"),
        ("Error Budget",   "1 - SLO: allowed failure time. Burn rate alerts."),
    ]
    for method, desc in methods:
        print(f"  {method:<18} {desc}")


if __name__ == "__main__":
    demonstrate_observability()
