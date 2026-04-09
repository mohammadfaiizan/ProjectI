"""
DISTRIBUTED TRACING DESIGN
=============================

Problem Statement:
In a microservices system, a single user request touches 5-10 services.
When something is slow or wrong, which service is responsible?
Logs per service don't give you the full picture across services.
Distributed tracing stitches together the full journey of a request.

Core Concepts:

  Trace:
    Represents the entire end-to-end journey of ONE request.
    Has a globally unique Trace ID shared by all services in that request.

  Span:
    Represents a single unit of work within a trace.
    Each service creates a span when it handles a request.
    Span has: span_id, parent_span_id, start_time, end_time, service_name,
              operation_name, tags, logs, status.

  Parent-Child Span Hierarchy:
    Gateway creates ROOT span (no parent).
    Gateway calls Order Service → Order Service creates CHILD span.
    Order Service calls Inventory → Inventory creates CHILD of Order's span.
    Visualized as a waterfall diagram.

  Context Propagation:
    Trace ID and Span ID are passed via HTTP headers:
      X-Trace-Id:       the global trace ID (same across all services)
      X-Span-Id:        the current span ID
      X-Parent-Span-Id: parent span ID (so we know the hierarchy)
    Every service reads these headers and creates its child span.

  Sampling:
    Can't trace 100% of requests in production (too much overhead).
    Head-based sampling: decide at the start (gateway) whether to trace.
    Tail-based sampling: buffer and decide after seeing the full trace
                         (trace errors/slow requests 100%, normal 1%).

OpenTelemetry (OTel):
  Vendor-neutral standard for traces, metrics, logs.
  SDK instruments your code; exporter sends to Jaeger/Zipkin/DataDog.
  Key benefit: switch backends without changing application code.

Latency Breakdown:
  Total trace duration ≠ sum of span durations (spans overlap if parallel).
  Total = critical path duration.
  Per-service contribution = span duration / total trace duration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import uuid
import threading


# ─────────────────────────────────────────────
# SPAN
# ─────────────────────────────────────────────

class SpanStatus:
    OK    = "ok"
    ERROR = "error"


@dataclass
class SpanLog:
    timestamp : float
    message   : str
    level     : str = "info"   # info / warn / error


@dataclass
class Span:
    trace_id        : str
    span_id         : str
    parent_span_id  : Optional[str]
    service         : str
    operation       : str
    start_time      : float = field(default_factory=time.time)
    end_time        : Optional[float] = None
    status          : str = SpanStatus.OK
    tags            : Dict[str, str] = field(default_factory=dict)
    logs            : List[SpanLog] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def finish(self, status: str = SpanStatus.OK):
        self.end_time = time.time()
        self.status   = status

    def log(self, message: str, level: str = "info"):
        self.logs.append(SpanLog(time.time(), message, level))

    def set_tag(self, key: str, value: str):
        self.tags[key] = value

    def is_root(self) -> bool:
        return self.parent_span_id is None


# ─────────────────────────────────────────────
# TRACE CONTEXT (propagated via headers)
# ─────────────────────────────────────────────

@dataclass
class TraceContext:
    trace_id      : str
    span_id       : str
    parent_span_id: Optional[str] = None
    sampled       : bool = True

    def to_headers(self) -> Dict[str, str]:
        headers = {
            "X-Trace-Id" : self.trace_id,
            "X-Span-Id"  : self.span_id,
            "X-Sampled"  : "1" if self.sampled else "0",
        }
        if self.parent_span_id:
            headers["X-Parent-Span-Id"] = self.parent_span_id
        return headers

    @staticmethod
    def from_headers(headers: Dict[str, str]) -> Optional["TraceContext"]:
        trace_id = headers.get("X-Trace-Id")
        span_id  = headers.get("X-Span-Id")
        if not trace_id or not span_id:
            return None
        return TraceContext(
            trace_id       = trace_id,
            span_id        = span_id,
            parent_span_id = headers.get("X-Parent-Span-Id"),
            sampled        = headers.get("X-Sampled", "1") == "1",
        )


# ─────────────────────────────────────────────
# TRACER
# ─────────────────────────────────────────────

class Tracer:
    """
    OpenTelemetry-style tracer. Creates and manages spans.
    In production: SDK sends completed spans to a collector (Jaeger/Zipkin).
    """

    def __init__(self, service_name: str, collector: "TraceCollector"):
        self.service_name = service_name
        self._collector   = collector

    def start_span(self, operation: str,
                   parent_context: Optional[TraceContext] = None) -> "SpanContext":
        if parent_context:
            trace_id       = parent_context.trace_id
            parent_span_id = parent_context.span_id
        else:
            trace_id       = str(uuid.uuid4()).replace("-", "")[:16]
            parent_span_id = None

        span = Span(
            trace_id       = trace_id,
            span_id        = str(uuid.uuid4()).replace("-", "")[:8],
            parent_span_id = parent_span_id,
            service        = self.service_name,
            operation      = operation,
        )
        return SpanContext(span, self._collector, self.service_name)


class SpanContext:
    """Context manager for a span. Auto-finishes on exit."""

    def __init__(self, span: Span, collector: "TraceCollector", service: str):
        self.span      = span
        self._collector= collector
        self._service  = service

    def child_context(self) -> TraceContext:
        """Propagate to downstream service."""
        return TraceContext(
            trace_id       = self.span.trace_id,
            span_id        = self.span.span_id,
            parent_span_id = self.span.parent_span_id,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.span.status = SpanStatus.ERROR
            self.span.log(str(exc_val), level="error")
        self.span.finish()
        self._collector.ingest(self.span)
        return False


# ─────────────────────────────────────────────
# TRACE COLLECTOR / BACKEND (Jaeger-like)
# ─────────────────────────────────────────────

class TraceCollector:
    """Receives completed spans and assembles them into traces."""

    def __init__(self):
        self._spans  : Dict[str, List[Span]] = {}  # trace_id → [spans]
        self._lock   = threading.Lock()

    def ingest(self, span: Span):
        with self._lock:
            self._spans.setdefault(span.trace_id, []).append(span)

    def get_trace(self, trace_id: str) -> List[Span]:
        return list(self._spans.get(trace_id, []))

    def trace_summary(self, trace_id: str) -> Dict:
        spans = self.get_trace(trace_id)
        if not spans:
            return {}

        root = next((s for s in spans if s.is_root()), spans[0])
        # Build parent→children map
        children: Dict[Optional[str], List[Span]] = {}
        for s in spans:
            children.setdefault(s.parent_span_id, []).append(s)

        finished = [s for s in spans if s.end_time is not None]
        total_ms = root.duration_ms if root.end_time else 0

        return {
            "trace_id"   : trace_id,
            "total_ms"   : round(total_ms, 2),
            "span_count" : len(spans),
            "services"   : list({s.service for s in spans}),
            "root_op"    : root.operation,
            "spans"      : [
                {
                    "service"    : s.service,
                    "operation"  : s.operation,
                    "span_id"    : s.span_id,
                    "parent_id"  : s.parent_span_id,
                    "duration_ms": round(s.duration_ms, 2),
                    "status"     : s.status,
                    "pct_of_total": round(s.duration_ms / max(total_ms, 0.001) * 100, 1),
                }
                for s in sorted(spans, key=lambda x: x.start_time)
            ],
        }

    def all_traces(self) -> List[str]:
        return list(self._spans.keys())


# ─────────────────────────────────────────────
# SIMULATED MICROSERVICES (with tracing)
# ─────────────────────────────────────────────

def make_traced_services(collector: TraceCollector):
    return {
        name: Tracer(name, collector)
        for name in ["gateway", "order-service",
                     "inventory-service", "payment-service"]
    }


def simulate_request_chain(tracers: Dict[str, Tracer],
                            collector: TraceCollector) -> str:
    """Simulate: gateway → order → inventory + payment (parallel) → gateway."""

    # Gateway creates root span
    with tracers["gateway"].start_span("POST /api/checkout") as gw_ctx:
        gw_ctx.span.set_tag("http.method", "POST")
        gw_ctx.span.set_tag("http.url", "/api/checkout")
        time.sleep(0.005)

        # Pass context to order-service
        downstream_ctx = TraceContext(gw_ctx.span.trace_id,
                                      gw_ctx.span.span_id)

        with tracers["order-service"].start_span("create_order", downstream_ctx) as ord_ctx:
            ord_ctx.span.set_tag("order.id", "ord-001")
            time.sleep(0.012)

            # Order service fans out to inventory + payment in parallel
            inv_ctx_in  = TraceContext(ord_ctx.span.trace_id, ord_ctx.span.span_id)
            pay_ctx_in  = TraceContext(ord_ctx.span.trace_id, ord_ctx.span.span_id)

            inv_result = {}
            pay_result = {}

            def call_inventory():
                with tracers["inventory-service"].start_span("reserve_stock", inv_ctx_in) as ctx:
                    ctx.span.set_tag("sku", "SKU-A1")
                    time.sleep(0.018)
                    ctx.span.log("Reserved 2 units of SKU-A1")

            def call_payment():
                with tracers["payment-service"].start_span("charge_card", pay_ctx_in) as ctx:
                    ctx.span.set_tag("amount", "99.99")
                    time.sleep(0.025)  # payment is slower
                    ctx.span.log("Card charged successfully")

            t1 = threading.Thread(target=call_inventory)
            t2 = threading.Thread(target=call_payment)
            t1.start(); t2.start()
            t1.join();  t2.join()

            ord_ctx.span.log("Order confirmed after inventory + payment")

        gw_ctx.span.set_tag("http.status", "200")

    return gw_ctx.span.trace_id


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_distributed_tracing():
    print("=" * 65)
    print("DISTRIBUTED TRACING DESIGN")
    print("=" * 65)

    collector = TraceCollector()
    tracers   = make_traced_services(collector)

    # ── 1. Single traced request ──────────────────
    print("\n[1] SINGLE REQUEST TRACE: gateway → order → inventory + payment")
    print("─" * 55)
    trace_id = simulate_request_chain(tracers, collector)
    print(f"  Trace ID: {trace_id}")

    summary = collector.trace_summary(trace_id)
    print(f"  Total latency:  {summary['total_ms']}ms")
    print(f"  Services hit:   {', '.join(summary['services'])}")
    print(f"  Span count:     {summary['span_count']}")

    # ── 2. Span waterfall ─────────────────────────
    print("\n\n[2] SPAN WATERFALL (latency breakdown)")
    print("─" * 55)
    print(f"  {'Service':<22} {'Operation':<22} {'Duration':<12} {'% of Total':<12} {'Status'}")
    print(f"  {'─'*70}")
    for span in summary["spans"]:
        parent_marker = "" if span["parent_id"] is None else "  └─ "
        print(f"  {parent_marker}{span['service']:<20} {span['operation']:<22} "
              f"{span['duration_ms']:.1f}ms      {span['pct_of_total']}%       "
              f"{span['status']}")

    # ── 3. Header propagation ─────────────────────
    print("\n\n[3] TRACE CONTEXT PROPAGATION VIA HTTP HEADERS")
    print("─" * 55)
    root_ctx = TraceContext(
        trace_id       = "abc123def456",
        span_id        = "root-span-01",
        parent_span_id = None,
        sampled        = True,
    )
    headers = root_ctx.to_headers()
    print(f"  Gateway injects headers:")
    for k, v in headers.items():
        print(f"    {k}: {v}")

    # Downstream service receives and creates child context
    received = TraceContext.from_headers(headers)
    print(f"\n  Downstream service reads headers:")
    print(f"    trace_id={received.trace_id}  span_id={received.span_id}")
    print(f"    sampled={received.sampled}")
    print(f"  Creates child span with parent_span_id={received.span_id}")

    # ── 4. Multiple traces, summary ───────────────
    print("\n\n[4] MULTIPLE REQUESTS — COLLECTOR SUMMARY")
    print("─" * 55)
    for _ in range(3):
        simulate_request_chain(tracers, collector)

    all_trace_ids = collector.all_traces()
    print(f"  Total traces collected: {len(all_trace_ids)}")
    for tid in all_trace_ids:
        s = collector.trace_summary(tid)
        print(f"  trace={tid[:12]}... "
              f"total={s['total_ms']}ms  spans={s['span_count']}")

    # ── 5. Error trace ────────────────────────────
    print("\n\n[5] ERROR SPAN — TRACING FAILURES")
    print("─" * 55)
    error_collector = TraceCollector()
    error_tracer    = Tracer("payment-service", error_collector)

    try:
        with error_tracer.start_span("charge_card") as ctx:
            ctx.span.set_tag("amount", "49.99")
            time.sleep(0.003)
            raise RuntimeError("Card declined: insufficient funds")
    except RuntimeError:
        pass  # span auto-finishes with ERROR status

    trace_ids = error_collector.all_traces()
    if trace_ids:
        err_summary = error_collector.trace_summary(trace_ids[0])
        span_info = err_summary["spans"][0]
        print(f"  Span status:  {span_info['status']}")
        print(f"  Operation:    {span_info['operation']}")
        print(f"  Duration:     {span_info['duration_ms']}ms")
        print(f"  → Error captured automatically in span. Searchable in Jaeger.")

    # ── 6. Sampling strategies ────────────────────
    print("\n\n[6] SAMPLING STRATEGIES")
    print("─" * 55)
    strategies = [
        ("No sampling (100%)",   "Capture every trace. Dev/staging only. Too expensive in prod."),
        ("Rate-based (1%)",      "Sample 1 in 100 requests. Low overhead. Misses rare errors."),
        ("Head-based",           "Decide at gateway. Fast but may drop interesting traces."),
        ("Tail-based",           "Buffer full trace, decide after. Catches all errors/slow reqs."),
        ("Adaptive",             "Auto-adjust rate based on traffic. Balance cost vs visibility."),
    ]
    for name, desc in strategies:
        print(f"  {name:<22} {desc}")

    # ── 7. OTel key concepts ──────────────────────
    print("\n\n[7] OPENTELEMETRY KEY CONCEPTS")
    print("─" * 55)
    concepts = [
        ("Trace",          "Full journey of one request; identified by Trace ID"),
        ("Span",           "One unit of work; has start/end, tags, logs, parent"),
        ("Context prop.",  "X-Trace-Id / X-Span-Id headers carry context between services"),
        ("Collector",      "Receives spans; assembles traces; forwards to backend"),
        ("Backend",        "Jaeger, Zipkin, DataDog, Tempo — stores and visualizes traces"),
        ("Instrumentation","Auto (agent) or manual (SDK). OTel = vendor-neutral SDK"),
        ("Sampling",       "Control what % of requests are traced to manage cost"),
    ]
    for name, desc in concepts:
        print(f"  {name:<18} {desc}")


if __name__ == "__main__":
    demonstrate_distributed_tracing()
