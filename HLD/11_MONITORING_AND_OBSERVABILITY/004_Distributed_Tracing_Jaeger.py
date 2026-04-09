"""
DISTRIBUTED TRACING AND JAEGER
================================

Problem Statement:
A single user request may fan out across 10+ microservices. When latency
spikes, which service is slow? Distributed tracing reconstructs the full
call graph from spans emitted by each service.

Core Concepts:
  Trace:    A complete request journey. Has a unique trace_id (128-bit).
  Span:     One unit of work in a trace. Has span_id, parent_span_id,
            service name, operation name, start/end timestamps, status, tags, logs.
  Context Propagation: trace_id + span_id passed in HTTP headers or gRPC metadata.
            W3C Trace Context: traceparent: 00-{trace_id}-{parent_id}-{flags}
            B3: X-B3-TraceId, X-B3-SpanId, X-B3-ParentSpanId

Span Relationships:
  ChildOf:    Parent waits for child to complete (synchronous).
  FollowsFrom: Parent does not wait (async fan-out, fire-and-forget).

Jaeger Architecture:
  Agent:      UDP sidecar that receives spans from app via Thrift/UDP.
  Collector:  Validates, indexes, stores spans into Cassandra/ES/BadgerDB.
  Query:      API + UI for searching and visualizing traces.
  Ingester:   Reads from Kafka; used for high-volume deployments.

Sampling Strategies:
  Head-based: Decision made at trace start. Fast, predictable.
    - Constant:     Sample everything (100%) or nothing.
    - Probabilistic: Sample p% (e.g., 0.1%) of traces.
    - Rate-limiting: N traces/sec regardless of traffic.
  Tail-based: Buffer full trace; decide after seeing all spans.
    - Sample slow traces (p99 > threshold) or error traces.
    - Implemented in OpenTelemetry Collector tail sampler.
    - More accurate but requires memory buffering.

OpenTelemetry (OTel):
  CNCF standard that unifies tracing, metrics, and logs.
  SDK → OTLP (OpenTelemetry Protocol) → OTel Collector → Jaeger/Zipkin/Datadog.
  Replaces vendor-specific SDKs; one instrumentation for all backends.

Trace Analysis Patterns:
  Critical path:  Longest sequential chain of spans → bottleneck.
  Fan-out:        Parent calls N children in parallel; total = max(children).
  Span events:    Structured log within a span (exception stack trace).
  Span attributes: Key-value metadata (http.url, db.statement, user_id).
"""

from __future__ import annotations

import time
import uuid
import json
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


# ─────────────────────────────────────────────
# SPAN STATUS
# ─────────────────────────────────────────────

class SpanStatus(Enum):
    UNSET = "UNSET"
    OK    = "OK"
    ERROR = "ERROR"


# ─────────────────────────────────────────────
# SPAN
# ─────────────────────────────────────────────

@dataclass
class SpanEvent:
    name:       str
    timestamp:  float
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    trace_id:       str
    span_id:        str
    parent_span_id: Optional[str]
    service:        str
    operation:      str
    start_time:     float
    end_time:       Optional[float]    = None
    status:         SpanStatus         = SpanStatus.UNSET
    tags:           Dict[str, Any]     = field(default_factory=dict)
    events:         List[SpanEvent]    = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def finish(self, status: SpanStatus = SpanStatus.OK):
        self.end_time = time.time()
        self.status   = status

    def set_tag(self, key: str, value: Any):
        self.tags[key] = value

    def add_event(self, name: str, **attributes):
        self.events.append(SpanEvent(name, time.time(), attributes))

    def set_error(self, exc: Exception):
        self.status = SpanStatus.ERROR
        self.add_event("exception",
                       exception_type=type(exc).__name__,
                       exception_message=str(exc))

    def to_dict(self) -> Dict:
        return {
            "trace_id":       self.trace_id,
            "span_id":        self.span_id,
            "parent_span_id": self.parent_span_id,
            "service":        self.service,
            "operation":      self.operation,
            "start_ms":       round(self.start_time * 1000),
            "duration_ms":    round(self.duration_ms or 0, 2),
            "status":         self.status.value,
            "tags":           self.tags,
        }


# ─────────────────────────────────────────────
# SPAN CONTEXT (propagated across services)
# ─────────────────────────────────────────────

@dataclass
class SpanContext:
    trace_id:    str
    span_id:     str
    sampled:     bool = True

    def to_w3c_header(self) -> str:
        """W3C traceparent header."""
        flags = "01" if self.sampled else "00"
        return f"00-{self.trace_id}-{self.span_id}-{flags}"

    @classmethod
    def from_w3c_header(cls, header: str) -> Optional["SpanContext"]:
        parts = header.split("-")
        if len(parts) != 4:
            return None
        _, trace_id, span_id, flags = parts
        return cls(trace_id, span_id, flags == "01")

    def to_b3_headers(self) -> Dict[str, str]:
        return {
            "X-B3-TraceId":    self.trace_id,
            "X-B3-SpanId":     self.span_id,
            "X-B3-Sampled":    "1" if self.sampled else "0",
        }


# ─────────────────────────────────────────────
# TRACER (per-service)
# ─────────────────────────────────────────────

class Tracer:
    """
    OpenTelemetry-inspired tracer for a single service.
    Stores completed spans in-memory (in prod: exports to Jaeger Collector).
    """

    _local = threading.local()

    def __init__(self, service: str, collector: "TraceCollector",
                 sample_rate: float = 1.0):
        self._service    = service
        self._collector  = collector
        self._sample_rate = sample_rate

    def _should_sample(self, trace_id: str) -> bool:
        # Deterministic: same trace_id always gives same decision
        h = int(trace_id[:8], 16) / 0xFFFFFFFF
        return h < self._sample_rate

    def start_span(self, operation: str,
                   parent_context: Optional[SpanContext] = None,
                   tags: Optional[Dict] = None) -> "SpanHandle":
        if parent_context:
            trace_id = parent_context.trace_id
            parent_id = parent_context.span_id
            sampled   = parent_context.sampled
        else:
            trace_id  = uuid.uuid4().hex + uuid.uuid4().hex[:16]  # 128-bit
            parent_id = None
            sampled   = self._should_sample(trace_id)

        span_id = uuid.uuid4().hex[:16]

        span = Span(
            trace_id       = trace_id,
            span_id        = span_id,
            parent_span_id = parent_id,
            service        = self._service,
            operation      = operation,
            start_time     = time.time(),
        )
        if tags:
            span.tags.update(tags)

        ctx = SpanContext(trace_id, span_id, sampled)
        return SpanHandle(span, ctx, self._collector if sampled else None)


class SpanHandle:
    """Context manager for a span."""

    def __init__(self, span: Span, ctx: SpanContext,
                 collector: Optional["TraceCollector"]):
        self.span = span
        self.ctx  = ctx
        self._collector = collector

    def __enter__(self) -> "SpanHandle":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.span.set_error(exc_val)
        else:
            self.span.finish(SpanStatus.OK)
        if self._collector:
            self._collector.receive(self.span)
        return False   # don't suppress exceptions


# ─────────────────────────────────────────────
# TRACE COLLECTOR / STORE
# ─────────────────────────────────────────────

class TraceCollector:
    """
    In-memory Jaeger-like trace store.
    Receives individual spans; reconstructs traces on query.
    """

    def __init__(self):
        self._spans: List[Span]             = []
        self._by_trace: Dict[str, List[Span]] = {}
        self._lock = threading.Lock()

    def receive(self, span: Span):
        with self._lock:
            self._spans.append(span)
            self._by_trace.setdefault(span.trace_id, []).append(span)

    def get_trace(self, trace_id: str) -> List[Span]:
        return self._by_trace.get(trace_id, [])

    def search(self, service: Optional[str] = None,
               operation: Optional[str] = None,
               min_duration_ms: Optional[float] = None,
               status: Optional[SpanStatus] = None,
               limit: int = 20) -> List[str]:
        """Return trace_ids matching criteria (root spans only)."""
        results = []
        seen: set = set()
        with self._lock:
            for span in reversed(self._spans):
                if span.trace_id in seen:
                    continue
                if service and span.service != service:
                    continue
                if operation and span.operation != operation:
                    continue
                if status and span.status != status:
                    continue
                trace = self._by_trace.get(span.trace_id, [])
                if min_duration_ms is not None:
                    total = self.trace_duration_ms(span.trace_id)
                    if total < min_duration_ms:
                        continue
                seen.add(span.trace_id)
                results.append(span.trace_id)
                if len(results) >= limit:
                    break
        return results

    def trace_duration_ms(self, trace_id: str) -> float:
        spans = self.get_trace(trace_id)
        if not spans:
            return 0.0
        start = min(s.start_time for s in spans)
        end   = max((s.end_time or s.start_time) for s in spans)
        return (end - start) * 1000

    def critical_path(self, trace_id: str) -> List[Span]:
        """
        Find the critical (longest sequential) path in a trace.
        Simple approach: DFS from root following longest child.
        """
        spans  = self.get_trace(trace_id)
        by_id  = {s.span_id: s for s in spans}
        children: Dict[Optional[str], List[Span]] = {}
        for s in spans:
            children.setdefault(s.parent_span_id, []).append(s)

        root_spans = children.get(None, [])
        if not root_spans:
            return []

        path: List[Span] = []

        def dfs(span: Span):
            path.append(span)
            kids = children.get(span.span_id, [])
            if not kids:
                return
            # Pick child with longest duration
            best = max(kids, key=lambda s: s.duration_ms or 0)
            dfs(best)

        dfs(root_spans[0])
        return path

    def stats(self) -> Dict:
        with self._lock:
            services: Dict[str, int] = {}
            errors = 0
            for s in self._spans:
                services[s.service] = services.get(s.service, 0) + 1
                if s.status == SpanStatus.ERROR:
                    errors += 1
            return {
                "total_spans":  len(self._spans),
                "total_traces": len(self._by_trace),
                "error_spans":  errors,
                "services":     services,
            }


# ─────────────────────────────────────────────
# SAMPLING STRATEGIES
# ─────────────────────────────────────────────

class SamplingStrategy(Enum):
    ALWAYS_ON    = "always_on"
    PROBABILISTIC= "probabilistic"
    RATE_LIMITING= "rate_limiting"
    TAIL_BASED   = "tail_based"

@dataclass
class SamplerConfig:
    strategy: SamplingStrategy
    param:    float = 1.0   # rate for probabilistic; rps for rate-limiting

    def description(self) -> str:
        if self.strategy == SamplingStrategy.ALWAYS_ON:
            return "Sample 100% — best for debugging, too expensive at scale"
        if self.strategy == SamplingStrategy.PROBABILISTIC:
            return f"Sample {self.param*100:.1f}% randomly — low overhead, misses rare errors"
        if self.strategy == SamplingStrategy.RATE_LIMITING:
            return f"Sample {self.param:.0f} traces/sec — predictable volume regardless of load"
        if self.strategy == SamplingStrategy.TAIL_BASED:
            return "Buffer traces, sample slow/error ones — most accurate, higher memory"
        return ""


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_tracing():
    print("=" * 65)
    print("DISTRIBUTED TRACING AND JAEGER")
    print("=" * 65)

    collector = TraceCollector()

    # Create tracers for each service
    tracers = {
        svc: Tracer(svc, collector, sample_rate=1.0)
        for svc in ["api-gateway", "user-service", "order-service",
                    "payment-service", "notification-service"]
    }

    # ── Simulate a distributed request ────────
    print("\n[1] SIMULATING DISTRIBUTED REQUEST")
    print("─" * 55)

    def simulate_request(user_id: str, order_amount: float):
        # API Gateway receives request
        with tracers["api-gateway"].start_span(
            "POST /api/orders",
            tags={"http.method": "POST", "http.url": "/api/orders",
                  "user_id": user_id}
        ) as gw:
            time.sleep(0.002)  # gateway overhead

            # Validate user (synchronous child span)
            user_ctx = gw.ctx
            with tracers["user-service"].start_span(
                "validate_user", parent_context=user_ctx,
                tags={"user_id": user_id}
            ) as us:
                time.sleep(0.005)
                us.span.set_tag("user.valid", True)

            # Create order
            with tracers["order-service"].start_span(
                "create_order", parent_context=gw.ctx,
                tags={"amount": order_amount}
            ) as os_span:
                time.sleep(0.010)
                order_id = f"ord-{uuid.uuid4().hex[:8]}"
                os_span.span.set_tag("order.id", order_id)

                # Process payment (nested under order)
                with tracers["payment-service"].start_span(
                    "charge_card", parent_context=os_span.ctx,
                    tags={"amount": order_amount, "currency": "USD"}
                ) as ps:
                    time.sleep(0.050)   # payment is slow
                    ps.span.add_event("payment_authorized",
                                      auth_code="AUTH-123")

            # Send notification (fire-and-forget / async)
            with tracers["notification-service"].start_span(
                "send_email", parent_context=gw.ctx,
                tags={"recipient": user_id, "type": "order_confirm"}
            ) as ns:
                time.sleep(0.003)

            gw.span.set_tag("http.status_code", 201)
            gw.span.set_tag("order.id", order_id)

        return gw.span.trace_id

    trace_id = simulate_request("user-42", 99.99)
    print(f"  trace_id: {trace_id[:16]}...")

    # Show all spans
    spans = collector.get_trace(trace_id)
    spans.sort(key=lambda s: s.start_time)
    print(f"\n  {'Service':<22} {'Operation':<20} {'Duration':>10}  {'Status'}")
    print("  " + "─" * 65)
    for span in spans:
        indent = "    " if span.parent_span_id else ""
        print(f"  {indent}{span.service:<20} {span.operation:<20} "
              f"{span.duration_ms:>8.1f}ms  {span.status.value}")

    total_ms = collector.trace_duration_ms(trace_id)
    print(f"\n  Total trace duration: {total_ms:.1f}ms")

    # ── Critical Path ─────────────────────────
    print("\n[2] CRITICAL PATH ANALYSIS")
    print("─" * 55)

    path = collector.critical_path(trace_id)
    print("  Critical path (bottleneck chain):")
    for i, span in enumerate(path):
        prefix = "  → " if i > 0 else "  ↳ "
        print(f"  {prefix}{span.service}.{span.operation}  {span.duration_ms:.1f}ms")

    # ── Error Trace ───────────────────────────
    print("\n[3] ERROR TRACE")
    print("─" * 55)

    with tracers["api-gateway"].start_span(
        "GET /api/orders/999",
        tags={"http.method": "GET"}
    ) as gw2:
        with tracers["order-service"].start_span(
            "get_order", parent_context=gw2.ctx,
            tags={"order_id": "999"}
        ) as os2:
            try:
                raise ValueError("Order 999 not found")
            except ValueError as e:
                os2.span.set_error(e)
                os2.span.finish(SpanStatus.ERROR)
                gw2.span.set_tag("http.status_code", 404)

    error_trace = collector.get_trace(gw2.span.trace_id)
    for span in error_trace:
        print(f"  {span.service}.{span.operation}: {span.status.value}")
        if span.events:
            for ev in span.events:
                print(f"    event={ev.name} {ev.attributes}")

    # ── Trace Search ──────────────────────────
    print("\n[4] TRACE SEARCH (Jaeger UI query)")
    print("─" * 55)

    # Simulate more traces
    for _ in range(10):
        simulate_request(f"user-{random.randint(1,100)}", random.uniform(10, 500))

    slow_traces = collector.search(min_duration_ms=50)
    error_traces= collector.search(status=SpanStatus.ERROR)
    svc_traces  = collector.search(service="payment-service")

    print(f"  Slow traces (>50ms):       {len(slow_traces)}")
    print(f"  Error traces:              {len(error_traces)}")
    print(f"  payment-service traces:    {len(svc_traces)}")

    # ── W3C Context Propagation ───────────────
    print("\n[5] CONTEXT PROPAGATION HEADERS")
    print("─" * 55)

    ctx = SpanContext("4bf92f3577b34da6a3ce929d0e0e4736",
                      "00f067aa0ba902b7", True)
    print(f"  W3C traceparent: {ctx.to_w3c_header()}")
    b3 = ctx.to_b3_headers()
    for k, v in b3.items():
        print(f"  B3 {k}: {v}")

    # Reconstruct from header
    parsed = SpanContext.from_w3c_header(ctx.to_w3c_header())
    print(f"\n  Parsed trace_id:  {parsed.trace_id}")
    print(f"  Parsed span_id:   {parsed.span_id}")
    print(f"  Sampled:          {parsed.sampled}")

    # ── Sampling Strategies ───────────────────
    print("\n[6] SAMPLING STRATEGIES")
    print("─" * 55)

    for strategy, param in [
        (SamplingStrategy.ALWAYS_ON,    1.0),
        (SamplingStrategy.PROBABILISTIC, 0.01),
        (SamplingStrategy.RATE_LIMITING, 100.0),
        (SamplingStrategy.TAIL_BASED,    1.0),
    ]:
        cfg = SamplerConfig(strategy, param)
        print(f"  {strategy.value:<20} {cfg.description()}")

    # ── Collector Stats ───────────────────────
    print("\n[7] COLLECTOR STATISTICS")
    print("─" * 55)

    stats = collector.stats()
    print(f"  Total spans:   {stats['total_spans']}")
    print(f"  Total traces:  {stats['total_traces']}")
    print(f"  Error spans:   {stats['error_spans']}")
    print("  Spans by service:")
    for svc, count in sorted(stats["services"].items()):
        print(f"    {svc:<25} {count}")


if __name__ == "__main__":
    demonstrate_tracing()
