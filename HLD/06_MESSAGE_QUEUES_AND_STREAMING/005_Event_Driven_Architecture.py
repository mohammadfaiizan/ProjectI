"""
EVENT-DRIVEN ARCHITECTURE (EDA)
================================

Problem Statement:
In traditional request/response (synchronous) systems, services are tightly coupled:
Service A calls Service B directly, waits for a response, then continues.
This creates latency chains, cascading failures, and brittle dependencies.
EDA decouples producers from consumers via events — no direct calls.

Core Concepts:
  Event: immutable record that something happened ("OrderPlaced", "PaymentProcessed").
  Producer: emits events without knowing who will consume them.
  Consumer: reacts to events independently, at their own pace.
  Event Bus / Broker: routes events from producers to consumers.

EDA Patterns:
  1. Event Notification: lightweight trigger — "something happened, go look it up".
     Payload is minimal (just IDs). Consumer fetches details from source if needed.
     Risk: extra HTTP calls; source may have changed by the time consumer fetches.

  2. Event-Carried State Transfer (ECST): event carries full state snapshot.
     Consumer needs no extra calls — it has everything in the event.
     Risk: large payloads; consumer must handle schema evolution.

  3. Event Sourcing: system state IS the event log. Current state = replay all events.
     Audit log is free. Time-travel queries possible.
     Risk: replay complexity; snapshot management.

  4. Command Sourcing: similar but commands (intents) are stored, not outcomes.

Choreography vs Orchestration:
  Choreography: services react to events autonomously. No central coordinator.
    Pro: loosely coupled, resilient.
    Con: hard to trace end-to-end flow; distributed logic.

  Orchestration: a central Saga/Workflow directs services via commands.
    Pro: explicit flow, easy to trace.
    Con: central component becomes a bottleneck / single point of failure.

Event Schema Best Practices:
  - Include: event_id, event_type, version, timestamp, correlation_id, producer.
  - Immutable: never update a published event.
  - Versioning: use schema registry or semver in event_type ("order.placed.v2").

Pitfalls:
  ✗ Fat events (too much state) create coupling through payload structure.
  ✗ Missing correlation_id makes distributed tracing impossible.
  ✗ No idempotency leads to duplicate processing.
  ✗ Implicit ordering assumptions across event types.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from collections import defaultdict
from enum import Enum
import time
import uuid
import threading
import random


# ─────────────────────────────────────────────
# EVENT MODEL
# ─────────────────────────────────────────────

@dataclass
class DomainEvent:
    event_id      : str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type    : str   = ""
    version       : str   = "1.0"
    timestamp     : float = field(default_factory=time.time)
    producer      : str   = ""
    correlation_id: str   = ""       # ties events across services (same original request)
    causation_id  : str   = ""       # direct parent event_id that caused this one
    payload       : Any   = None

    def derive(self, event_type: str, producer: str, payload: Any) -> "DomainEvent":
        """Create a child event caused by this event (preserves correlation chain)."""
        return DomainEvent(
            event_type     = event_type,
            producer       = producer,
            correlation_id = self.correlation_id or self.event_id,
            causation_id   = self.event_id,
            payload        = payload,
        )


# ─────────────────────────────────────────────
# EVENT BUS (In-Process)
# ─────────────────────────────────────────────

class EventBus:
    """
    Simple synchronous in-process event bus.
    In production, replace the dispatch call with publishing to Kafka/SNS/RabbitMQ.
    """

    def __init__(self):
        self._handlers   : Dict[str, List[Callable]] = defaultdict(list)
        self._middlewares: List[Callable] = []
        self._published  : List[DomainEvent] = []
        self._lock       = threading.Lock()

    def subscribe(self, event_type: str, handler: Callable):
        with self._lock:
            self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: Callable):
        """Catch-all handler (useful for logging/audit)."""
        self.subscribe("*", handler)

    def use(self, middleware: Callable):
        """Middleware: fn(event, next_fn) — called before handlers."""
        self._middlewares.append(middleware)

    def publish(self, event: DomainEvent):
        with self._lock:
            self._published.append(event)
            handlers = list(self._handlers.get(event.event_type, []))
            handlers += list(self._handlers.get("*", []))

        chain = self._build_chain(handlers, event)
        chain()

    def _build_chain(self, handlers: List[Callable], event: DomainEvent) -> Callable:
        def dispatch():
            for h in handlers:
                try:
                    h(event)
                except Exception as exc:
                    print(f"  [EventBus] Handler error for {event.event_type}: {exc}")

        # Wrap with middlewares in reverse order
        fn = dispatch
        for mw in reversed(self._middlewares):
            prev = fn
            fn = lambda mw=mw, prev=prev: mw(event, prev)
        return fn

    @property
    def event_count(self) -> int:
        return len(self._published)


# ─────────────────────────────────────────────
# PATTERN 1: EVENT NOTIFICATION
# ─────────────────────────────────────────────

class OrderDatabase:
    """Simulates the source-of-truth store that consumers query after notification."""

    def __init__(self):
        self._orders: Dict[str, Dict] = {}

    def save(self, order_id: str, data: Dict):
        self._orders[order_id] = data

    def find(self, order_id: str) -> Optional[Dict]:
        return self._orders.get(order_id)


class EventNotificationDemo:
    """
    Order service emits a thin "OrderPlaced" event (just order_id).
    Inventory service reacts by fetching full order details from Order service.
    Demonstrates: lightweight notification → consumer fetches state.
    """

    def __init__(self, bus: EventBus):
        self.bus      = bus
        self.db       = OrderDatabase()
        self.fetches  = 0
        bus.subscribe("order.placed.v1", self._on_order_placed)

    def place_order(self, order_id: str, items: List[str], amount: float):
        # Save full state in own store
        self.db.save(order_id, {"items": items, "amount": amount, "status": "placed"})
        # Publish thin notification (just the ID)
        event = DomainEvent(
            event_type     = "order.placed.v1",
            producer       = "order-service",
            correlation_id = order_id,
            payload        = {"order_id": order_id},   # minimal payload
        )
        self.bus.publish(event)
        return event

    def _on_order_placed(self, event: DomainEvent):
        """Inventory reacts: fetches full order to check stock."""
        order_id = event.payload["order_id"]
        order    = self.db.find(order_id)           # extra round-trip to source
        self.fetches += 1
        # (inventory logic would go here)


# ─────────────────────────────────────────────
# PATTERN 2: EVENT-CARRIED STATE TRANSFER
# ─────────────────────────────────────────────

class EcstOrderService:
    """
    Order service embeds full state in the event payload.
    Consumers need no extra calls — they have everything.
    """

    def __init__(self, bus: EventBus):
        self.bus = bus

    def place_order(self, order_id: str, items: List[str], amount: float,
                    customer: Dict) -> DomainEvent:
        event = DomainEvent(
            event_type     = "order.placed.v2",
            producer       = "order-service",
            correlation_id = order_id,
            payload        = {                          # full state snapshot
                "order_id" : order_id,
                "items"    : items,
                "amount"   : amount,
                "customer" : customer,
                "status"   : "placed",
            },
        )
        self.bus.publish(event)
        return event


class InventoryServiceEcst:
    def __init__(self, bus: EventBus):
        self.processed: List[str] = []
        bus.subscribe("order.placed.v2", self._on_order)

    def _on_order(self, event: DomainEvent):
        """No extra calls needed — items are in the payload."""
        items = event.payload["items"]
        self.processed.append(event.payload["order_id"])
        # reserve items directly from payload


class BillingServiceEcst:
    def __init__(self, bus: EventBus):
        self.charges: List[Dict] = []
        bus.subscribe("order.placed.v2", self._on_order)

    def _on_order(self, event: DomainEvent):
        self.charges.append({
            "order_id": event.payload["order_id"],
            "amount"  : event.payload["amount"],
            "customer": event.payload["customer"]["id"],
        })


# ─────────────────────────────────────────────
# CHOREOGRAPHY vs ORCHESTRATION
# ─────────────────────────────────────────────

class ChoreographyOrder:
    """
    Each service reacts to events and emits new events.
    No central coordinator — chain propagates autonomously.
    Order → inventory.reserved → payment.charged → notification.sent
    """

    def __init__(self, bus: EventBus):
        self.bus   = bus
        self.trail : List[str] = []   # tracks event chain for demo

        bus.subscribe("order.placed.v3",        self._inventory_reserve)
        bus.subscribe("inventory.reserved",     self._payment_charge)
        bus.subscribe("payment.charged",        self._notification_send)

    def place_order(self, order_id: str, amount: float) -> DomainEvent:
        evt = DomainEvent(
            event_type     = "order.placed.v3",
            producer       = "order-service",
            correlation_id = order_id,
            payload        = {"order_id": order_id, "amount": amount},
        )
        self.bus.publish(evt)
        return evt

    def _inventory_reserve(self, event: DomainEvent):
        self.trail.append(f"inventory reserved for {event.payload['order_id']}")
        next_event = event.derive("inventory.reserved", "inventory-service",
                                   {"order_id": event.payload["order_id"]})
        self.bus.publish(next_event)

    def _payment_charge(self, event: DomainEvent):
        self.trail.append(f"payment charged for {event.payload['order_id']}")
        next_event = event.derive("payment.charged", "payment-service",
                                   {"order_id": event.payload["order_id"]})
        self.bus.publish(next_event)

    def _notification_send(self, event: DomainEvent):
        self.trail.append(f"notification sent for {event.payload['order_id']}")


class OrchestratorOrder:
    """
    Central orchestrator directs each step via commands, waits for results.
    Explicit flow — easy to read and trace. Tighter coupling to each service.
    """

    def __init__(self):
        self.steps_log: List[str] = []

    def _call_inventory(self, order_id: str) -> bool:
        self.steps_log.append(f"  → [orchestrator] command: reserve inventory for {order_id}")
        time.sleep(0.001)  # simulate call
        return True

    def _call_payment(self, order_id: str, amount: float) -> bool:
        self.steps_log.append(f"  → [orchestrator] command: charge payment for {order_id}")
        time.sleep(0.001)
        return True

    def _call_notification(self, order_id: str) -> bool:
        self.steps_log.append(f"  → [orchestrator] command: send notification for {order_id}")
        time.sleep(0.001)
        return True

    def process_order(self, order_id: str, amount: float) -> bool:
        self.steps_log.append(f"[orchestrator] starting order {order_id}")
        if not self._call_inventory(order_id): return False
        if not self._call_payment(order_id, amount): return False
        if not self._call_notification(order_id): return False
        self.steps_log.append(f"[orchestrator] order {order_id} complete")
        return True


# ─────────────────────────────────────────────
# MIDDLEWARE: LOGGING + CORRELATION ENFORCEMENT
# ─────────────────────────────────────────────

def logging_middleware(event: DomainEvent, next_fn: Callable):
    print(f"  [bus] {event.event_type} | corr={event.correlation_id} | "
          f"cause={event.causation_id or 'root'}")
    next_fn()


def correlation_enforcer(event: DomainEvent, next_fn: Callable):
    if not event.correlation_id:
        print(f"  [WARN] Event {event.event_type} missing correlation_id!")
    next_fn()


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_eda():
    print("=" * 65)
    print("EVENT-DRIVEN ARCHITECTURE")
    print("=" * 65)

    # ── Pattern 1: Event Notification ────────────
    print("\n[1] EVENT NOTIFICATION — thin event, consumer fetches state")
    print("─" * 55)
    bus1  = EventBus()
    demo1 = EventNotificationDemo(bus1)
    for i in range(4):
        demo1.place_order(f"ORD-{i:03d}", [f"item-{i}"], random.uniform(20, 200))
    print(f"  Orders placed: 4  |  Extra DB fetches by inventory: {demo1.fetches}")
    print(f"  → Thin payload keeps event small, but consumer makes extra call")

    # ── Pattern 2: ECST ───────────────────────────
    print("\n\n[2] EVENT-CARRIED STATE TRANSFER — full state in payload")
    print("─" * 55)
    bus2    = EventBus()
    orders2 = EcstOrderService(bus2)
    inv2    = InventoryServiceEcst(bus2)
    bill2   = BillingServiceEcst(bus2)

    amounts = [150.0, 75.0, 320.0, 45.0, 210.0]
    for i, amt in enumerate(amounts):
        orders2.place_order(
            order_id = f"ORD-{i:03d}",
            items    = [f"sku-{j}" for j in range(random.randint(1, 3))],
            amount   = amt,
            customer = {"id": f"cust-{i}", "email": f"user{i}@example.com"},
        )

    print(f"  Orders published: 5")
    print(f"  Inventory processed: {len(inv2.processed)} orders (no extra fetches)")
    print(f"  Billing charges recorded: {len(bill2.charges)}")
    for c in bill2.charges[:3]:
        print(f"    order={c['order_id']} amount=${c['amount']:.0f} cust={c['customer']}")

    # ── Choreography ─────────────────────────────
    print("\n\n[3] CHOREOGRAPHY — services react autonomously")
    print("─" * 55)
    bus3  = EventBus()
    bus3.use(logging_middleware)
    choreo = ChoreographyOrder(bus3)

    print(f"  Processing order ORD-001:")
    choreo.place_order("ORD-001", 99.0)
    print(f"\n  Event chain triggered (each service reacted independently):")
    for step in choreo.trail:
        print(f"    ✓ {step}")

    # ── Orchestration ─────────────────────────────
    print("\n\n[4] ORCHESTRATION — central coordinator directs steps")
    print("─" * 55)
    orch = OrchestratorOrder()
    orch.process_order("ORD-002", 150.0)
    for step in orch.steps_log:
        print(f"  {step}")

    # ── Comparison ────────────────────────────────
    print("\n\n[5] CHOREOGRAPHY vs ORCHESTRATION")
    print("─" * 55)
    rows = [
        ("Coupling",       "Loose — via events",         "Tighter — direct calls"),
        ("Traceability",   "Harder (distributed logic)",  "Easy (central log)"),
        ("Resilience",     "No single point of failure",  "Orchestrator is SPOF"),
        ("Adding steps",   "New service just subscribes", "Update orchestrator"),
        ("Testing",        "Must trace event chain",      "Test orchestrator class"),
        ("Best for",       "Simple, parallel workflows",  "Complex, sequential flows"),
    ]
    print(f"  {'Aspect':<18} {'Choreography':<32} {'Orchestration'}")
    print(f"  {'─'*70}")
    for aspect, choreo_val, orch_val in rows:
        print(f"  {aspect:<18} {choreo_val:<32} {orch_val}")

    # ── EDA Pitfalls ──────────────────────────────
    print("\n\n[6] EDA PITFALLS & MITIGATIONS")
    print("─" * 55)
    pitfalls = [
        ("Missing correlation_id",  "Distributed tracing fails",         "Always set corr_id from request"),
        ("No idempotency",          "Duplicate events = duplicate work",  "Check event_id before processing"),
        ("Fat events",              "Schema coupling via payload",        "Keep payload minimal or versioned"),
        ("Ordering assumptions",    "Event B arrives before event A",     "Use sequence numbers or vector clocks"),
        ("No DLQ",                  "Failed events silently dropped",     "Route failures to dead-letter topic"),
    ]
    print(f"  {'Pitfall':<25} {'Risk':<38} {'Mitigation'}")
    print(f"  {'─'*85}")
    for p, r, m in pitfalls:
        print(f"  {p:<25} {r:<38} {m}")


if __name__ == "__main__":
    demonstrate_eda()
