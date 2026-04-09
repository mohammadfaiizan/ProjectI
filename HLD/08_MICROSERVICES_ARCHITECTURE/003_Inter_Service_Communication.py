"""
INTER-SERVICE COMMUNICATION
==============================

Problem Statement:
Services must talk to each other. How they talk determines latency, coupling,
availability, and complexity. Wrong choice = cascading failures or lost messages.

Two Fundamental Models:

  1. Synchronous (Request-Response):
     Caller waits for a response.
     Protocols: REST over HTTP, gRPC (protobuf over HTTP/2).
     Use when: you need the result NOW to proceed.
     Risk: latency amplification, cascading failures.

     Latency Amplification (chain A → B → C → D):
       Total latency = latency(A→B) + latency(B→C) + latency(C→D) + processing
       Four-hop chain at 50ms each = 200ms minimum. One slow service kills all.

  2. Asynchronous (Events / Message Queues):
     Caller fires a message and continues. Consumer processes when ready.
     Protocols: Kafka, RabbitMQ, SQS, EventBridge.
     Use when: result not needed immediately, decoupling matters more than latency.
     Benefit: temporal decoupling — producer and consumer don't need to be up simultaneously.
     Risk: eventual consistency, harder to debug, message ordering challenges.

  REST vs gRPC:
    REST:  Human-readable, JSON, easy tooling, HTTP/1.1 or 2, loose typing.
    gRPC:  Binary protobuf, strongly typed, HTTP/2 multiplexing, ~7x smaller payload.
    Use gRPC for: internal high-throughput service-to-service calls.
    Use REST for: public APIs, browser clients, simpler integrations.

  Circuit Breaker Pattern:
    CLOSED → normal operation; failures tracked.
    OPEN   → failure threshold exceeded; calls blocked immediately (fast fail).
    HALF_OPEN → after timeout, allow probe request; if success → CLOSED, else → OPEN.
    Prevents cascading failures when a downstream service is degraded.

Communication Pattern Selection:
  | Scenario                        | Pattern           |
  |---------------------------------|-------------------|
  | Real-time payment confirmation  | Sync (REST/gRPC)  |
  | Order placed → notify warehouse | Async (event)     |
  | User query requiring DB lookup  | Sync (gRPC)       |
  | Audit log on every action       | Async (fire+forget)|
  | Cross-service saga step         | Async (choreography)|
  | Request needs immediate result  | Sync              |
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import time
import threading
import uuid
import random


# ─────────────────────────────────────────────
# SIMULATED HTTP TRANSPORT
# ─────────────────────────────────────────────

class SimulatedTimeout(Exception):
    pass

class SimulatedServiceError(Exception):
    pass


def simulate_http_call(service: str, endpoint: str,
                       base_latency_ms: float = 30,
                       error_rate: float = 0.0) -> Dict:
    """Simulate an HTTP call with realistic latency and occasional errors."""
    jitter  = random.uniform(0, base_latency_ms * 0.3)
    elapsed = base_latency_ms + jitter
    time.sleep(elapsed / 1000)

    if random.random() < error_rate:
        raise SimulatedServiceError(f"{service} returned 500")

    return {"service": service, "endpoint": endpoint,
            "status": 200, "latency_ms": elapsed}


# ─────────────────────────────────────────────
# SYNCHRONOUS CALL CHAIN (latency amplification)
# ─────────────────────────────────────────────

@dataclass
class ServiceCall:
    service       : str
    endpoint      : str
    latency_ms    : float
    error_rate    : float = 0.0


class SyncCallChain:
    """
    Simulates A → B → C → D synchronous chain.
    Total latency = sum of each hop. One failure breaks the chain.
    """

    def __init__(self, chain: List[ServiceCall]):
        self.chain = chain

    def execute(self, correlation_id: str) -> Dict:
        total_start = time.time()
        results     = []
        for call in self.chain:
            hop_start = time.time()
            try:
                result = simulate_http_call(
                    call.service, call.endpoint,
                    call.latency_ms, call.error_rate)
                result["hop_ms"] = (time.time() - hop_start) * 1000
                results.append(result)
            except SimulatedServiceError as e:
                total_ms = (time.time() - total_start) * 1000
                return {
                    "success"      : False,
                    "failed_at"    : call.service,
                    "error"        : str(e),
                    "total_ms"     : total_ms,
                    "completed_hops": results,
                }

        total_ms = (time.time() - total_start) * 1000
        return {
            "success"  : True,
            "total_ms" : total_ms,
            "hops"     : results,
        }


# ─────────────────────────────────────────────
# ASYNC EVENT BUS
# ─────────────────────────────────────────────

@dataclass
class Event:
    event_type    : str
    payload       : Dict
    correlation_id: str
    timestamp     : float = field(default_factory=time.time)
    event_id      : str   = field(default_factory=lambda: str(uuid.uuid4())[:8])


class AsyncEventBus:
    """
    In-memory async event bus. Publisher fires and forgets.
    Consumers run in background threads.
    """

    def __init__(self):
        self._handlers     : Dict[str, List[Callable]] = {}
        self._event_log    : List[Event] = []
        self._lock         = threading.Lock()

    def subscribe(self, event_type: str, handler: Callable):
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: Event):
        """Fire and forget — returns immediately."""
        with self._lock:
            self._event_log.append(event)
            handlers = list(self._handlers.get(event.event_type, []))

        def _dispatch():
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    print(f"    [EventBus] Handler error for {event.event_type}: {e}")

        t = threading.Thread(target=_dispatch, daemon=True)
        t.start()
        return event.event_id

    def event_count(self) -> int:
        return len(self._event_log)


# ─────────────────────────────────────────────
# CIRCUIT BREAKER
# ─────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED    = "CLOSED"
    OPEN      = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """
    CLOSED: calls pass through; failures tracked.
    OPEN:   after failure_threshold failures, block all calls (fast fail).
    HALF_OPEN: after reset_timeout_s, allow one probe call.
               Success → CLOSED. Failure → OPEN again.
    """

    def __init__(self, name: str,
                 failure_threshold: int = 3,
                 success_threshold: int = 2,
                 reset_timeout_s: float = 2.0):
        self.name              = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.reset_timeout_s   = reset_timeout_s
        self._state            = CircuitState.CLOSED
        self._failure_count    = 0
        self._success_count    = 0
        self._opened_at        : Optional[float] = None
        self._call_log         : List[Dict] = []

    @property
    def state(self) -> CircuitState:
        if (self._state == CircuitState.OPEN and
                self._opened_at is not None and
                time.time() - self._opened_at >= self.reset_timeout_s):
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
        return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            self._call_log.append({"state": "OPEN", "result": "blocked"})
            raise SimulatedServiceError(
                f"Circuit '{self.name}' is OPEN — fast fail")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._call_log.append({"state": "CLOSED", "result": "recovered"})
        else:
            self._failure_count = max(0, self._failure_count - 1)
        self._call_log.append({"state": self._state.value, "result": "success"})

    def _on_failure(self):
        self._failure_count += 1
        self._call_log.append({"state": self._state.value, "result": "failure"})
        if (self._state == CircuitState.CLOSED and
                self._failure_count >= self.failure_threshold):
            self._state     = CircuitState.OPEN
            self._opened_at = time.time()
            self._call_log.append({"state": "OPEN", "result": "tripped"})
        elif self._state == CircuitState.HALF_OPEN:
            self._state     = CircuitState.OPEN
            self._opened_at = time.time()

    def summary(self) -> Dict:
        return {
            "circuit"       : self.name,
            "state"         : self.state.value,
            "failures"      : self._failure_count,
            "total_calls"   : len(self._call_log),
        }


# ─────────────────────────────────────────────
# COMMUNICATION PATTERN SELECTOR
# ─────────────────────────────────────────────

def select_communication_pattern(needs_immediate_result: bool,
                                 caller_can_retry: bool,
                                 data_volume: str,       # low/medium/high
                                 latency_sensitive: bool) -> str:
    if needs_immediate_result and latency_sensitive:
        if data_volume == "high":
            return "gRPC (binary, low overhead, HTTP/2 streaming)"
        return "REST (simple, widely supported)"
    if not needs_immediate_result:
        if caller_can_retry:
            return "Async event (Kafka/SQS) with at-least-once delivery"
        return "Async event with idempotency key for exactly-once semantics"
    return "REST with synchronous response + async follow-up event"


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_inter_service_communication():
    print("=" * 65)
    print("INTER-SERVICE COMMUNICATION")
    print("=" * 65)

    # ── 1. Sync call chain — latency amplification ─
    print("\n[1] SYNCHRONOUS CHAIN — LATENCY AMPLIFICATION")
    print("─" * 55)
    print("  Request: API Gateway → Order → Inventory → Payment → Notification")

    chain = SyncCallChain([
        ServiceCall("order-service",       "/orders",        40),
        ServiceCall("inventory-service",   "/reserve",       35),
        ServiceCall("payment-service",     "/charge",        60),
        ServiceCall("notification-service","/send",          25),
    ])
    corr = str(uuid.uuid4())[:8]
    result = chain.execute(corr)

    if result["success"]:
        print(f"\n  Chain result: SUCCESS  total={result['total_ms']:.1f}ms")
        ideal = sum(c.latency_ms for c in chain.chain)
        print(f"  Sum of hop latencies (ideal): {ideal}ms")
        print(f"  Measured (includes jitter):   {result['total_ms']:.1f}ms")
        print(f"\n  Per-hop breakdown:")
        for hop in result["hops"]:
            print(f"    {hop['service']:<26} {hop['hop_ms']:.1f}ms")
    print(f"\n  Key insight: 4 services × 40ms avg = 160ms MINIMUM.")
    print(f"  One slow service blocks all callers upstream.")

    # ── 2. Async event bus ──────────────────────────
    print("\n\n[2] ASYNC EVENT BUS — TEMPORAL DECOUPLING")
    print("─" * 55)

    bus = AsyncEventBus()
    received_events = []

    def inventory_handler(event: Event):
        received_events.append(f"[inventory] received {event.event_type}")

    def notification_handler(event: Event):
        received_events.append(f"[notification] received {event.event_type}")

    def billing_handler(event: Event):
        received_events.append(f"[billing] received {event.event_type}")

    bus.subscribe("OrderPlaced", inventory_handler)
    bus.subscribe("OrderPlaced", notification_handler)
    bus.subscribe("OrderPlaced", billing_handler)

    print("  Publishing OrderPlaced event (fire-and-forget)...")
    t0 = time.time()
    eid = bus.publish(Event("OrderPlaced",
                            {"order_id": "ord-001", "total": 99.99},
                            corr))
    publish_ms = (time.time() - t0) * 1000
    print(f"  Publish returned in {publish_ms:.2f}ms (non-blocking)")
    print(f"  Event ID: {eid}")

    time.sleep(0.05)  # let handlers run
    print(f"  Handlers fired asynchronously:")
    for msg in received_events:
        print(f"    {msg}")
    print(f"  Total events published: {bus.event_count()}")

    # ── 3. Circuit breaker ──────────────────────────
    print("\n\n[3] CIRCUIT BREAKER — OPEN / HALF_OPEN / CLOSED")
    print("─" * 55)

    cb = CircuitBreaker("payment-service", failure_threshold=3,
                        success_threshold=2, reset_timeout_s=0.5)

    def flaky_call(succeed: bool):
        if not succeed:
            raise SimulatedServiceError("payment-service unavailable")
        return {"status": "ok"}

    print("  Sending calls; first 3 will fail → trip the breaker")
    for i in range(7):
        succeed = i >= 5   # fail first 5, succeed after
        try:
            cb.call(flaky_call, succeed)
            print(f"  Call {i+1}: SUCCESS  state={cb.state.value}")
        except SimulatedServiceError as e:
            print(f"  Call {i+1}: FAILED   state={cb.state.value}  ({e})")

        if i == 4:
            print(f"  [Waiting {cb.reset_timeout_s}s for HALF_OPEN probe window...]")
            time.sleep(cb.reset_timeout_s + 0.05)

    s = cb.summary()
    print(f"\n  Final circuit state: {s['state']}")
    print(f"  Total calls tracked: {s['total_calls']}")

    # ── 4. REST vs gRPC comparison ──────────────────
    print("\n\n[4] REST vs gRPC COMPARISON")
    print("─" * 55)
    comparisons = [
        ("Protocol",      "HTTP/1.1 or 2, JSON",     "HTTP/2, Protobuf binary"),
        ("Payload size",  "~1x (verbose JSON)",       "~7x smaller (binary)"),
        ("Typing",        "Loose (OpenAPI optional)", "Strong (proto schema)"),
        ("Streaming",     "Limited (SSE/WS needed)",  "Native bi-directional"),
        ("Browser support","Native",                  "Needs grpc-web proxy"),
        ("Debugging",     "Easy (human-readable)",    "Harder (binary encoded)"),
        ("Use case",      "Public APIs, simple calls","Internal, high-throughput"),
    ]
    print(f"  {'Attribute':<18} {'REST':<30} {'gRPC'}")
    print(f"  {'─'*70}")
    for attr, rest, grpc in comparisons:
        print(f"  {attr:<18} {rest:<30} {grpc}")

    # ── 5. Pattern selection guide ──────────────────
    print("\n\n[5] COMMUNICATION PATTERN SELECTION GUIDE")
    print("─" * 55)
    scenarios = [
        (True,  True,  "low",    True),
        (True,  True,  "high",   True),
        (False, True,  "medium", False),
        (False, False, "low",    False),
        (True,  False, "medium", False),
    ]
    labels = [
        "User login (needs JWT now)",
        "Internal data sync, high volume",
        "Order placed → warehouse notify",
        "Exactly-once payment event",
        "Fetch user data, not time-critical",
    ]
    print(f"  {'Scenario':<38} {'Recommended Pattern'}")
    print(f"  {'─'*72}")
    for label, (imm, retry, vol, latency) in zip(labels, scenarios):
        pattern = select_communication_pattern(imm, retry, vol, latency)
        print(f"  {label:<38} {pattern}")


if __name__ == "__main__":
    demonstrate_inter_service_communication()
