"""
CQRS — COMMAND QUERY RESPONSIBILITY SEGREGATION
=================================================

Problem Statement:
A single data model used for both writes and reads creates tensions:
- Write model optimised for consistency, validation, business logic.
- Read model optimised for projections, joins, denormalised views.
Trying to satisfy both with one model leads to compromises in both directions.

CQRS splits the model into two:
  Command side: accepts write intents (CreateOrder, CancelOrder).
                Validates, applies business rules, persists to write store.
                Emits domain events.
  Query side:   read-only projections built from domain events.
                Optimised for specific read patterns (denormalized, pre-joined).
                Eventually consistent with the write side.

Key Insight:
  Commands change state. Queries read state.
  They have completely different concerns — give them different models.

Why it helps:
  - Write store (normalized) stays small, clean, easy to validate.
  - Read store (denormalized) is optimized per query pattern.
  - Read side can be rebuilt at any time by replaying events.
  - Read side can scale independently (read-heavy workloads common).
  - Multiple read models for the same data (dashboard, API, search).

Trade-offs:
  ✗ Eventual consistency: read model is behind the write model by some lag.
  ✗ Complexity: two codepaths, two stores, synchronization needed.
  ✗ Overkill for simple CRUD apps with few query patterns.

When to use CQRS:
  ✓ Different scale requirements for reads vs writes.
  ✓ Complex domain with many read projections.
  ✓ Event-sourced systems (CQRS + ES pair naturally).
  ✓ Reporting/analytics queries that don't fit the normalized write model.
  ✗ Simple CRUD, small teams, uniform access patterns → don't add complexity.

CQRS + Event Sourcing:
  Write side appends events. Read side builds projections from events.
  This is the most common pairing in DDD/microservices.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict
from enum import Enum
import time
import uuid
import threading


# ─────────────────────────────────────────────
# COMMAND SIDE
# ─────────────────────────────────────────────

@dataclass
class Command:
    command_id : str  = field(default_factory=lambda: str(uuid.uuid4())[:8])
    command_type: str = ""
    payload    : Any  = None
    issued_by  : str  = ""
    issued_at  : float = field(default_factory=time.time)


@dataclass
class DomainEvent:
    event_id   : str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type : str   = ""
    aggregate_id: str  = ""
    payload    : Any   = None
    version    : int   = 0
    timestamp  : float = field(default_factory=time.time)


class CommandValidationError(Exception):
    pass


class OrderAggregate:
    """
    Write-side aggregate. Enforces business rules.
    State built from event log. Emits events on mutation.
    """

    def __init__(self, order_id: str):
        self.order_id  = order_id
        self.status    = "pending"
        self.items     : List[Dict] = []
        self.total     : float = 0.0
        self.version   : int   = 0
        self._events   : List[DomainEvent] = []   # uncommitted events

    # ── Mutators (apply event, no validation here) ──
    def _apply_order_created(self, evt: DomainEvent):
        p = evt.payload
        self.status = "created"
        self.items  = p["items"]
        self.total  = p["total"]

    def _apply_order_confirmed(self, evt: DomainEvent):
        self.status = "confirmed"

    def _apply_order_cancelled(self, evt: DomainEvent):
        self.status = "cancelled"

    def _apply_item_added(self, evt: DomainEvent):
        p = evt.payload
        self.items.append(p["item"])
        self.total += p["item"]["price"]

    def apply(self, evt: DomainEvent):
        handlers = {
            "order.created"  : self._apply_order_created,
            "order.confirmed": self._apply_order_confirmed,
            "order.cancelled": self._apply_order_cancelled,
            "item.added"     : self._apply_item_added,
        }
        if evt.event_type in handlers:
            handlers[evt.event_type](evt)
        self.version += 1

    def _emit(self, event_type: str, payload: Any):
        evt = DomainEvent(
            event_type    = event_type,
            aggregate_id  = self.order_id,
            payload       = payload,
            version       = self.version + 1,
        )
        self._events.append(evt)
        self.apply(evt)
        return evt

    # ── Command Handlers (validate + emit) ─────────
    def handle_create(self, cmd: Command):
        if self.status != "pending":
            raise CommandValidationError(f"Order already exists: {self.order_id}")
        p = cmd.payload
        if not p.get("items"):
            raise CommandValidationError("Order must have at least one item")
        total = sum(i["price"] for i in p["items"])
        self._emit("order.created", {"items": p["items"], "total": total,
                                      "customer_id": p["customer_id"]})

    def handle_confirm(self, cmd: Command):
        if self.status != "created":
            raise CommandValidationError(f"Cannot confirm order in status: {self.status}")
        self._emit("order.confirmed", {"confirmed_at": time.time()})

    def handle_cancel(self, cmd: Command):
        if self.status == "cancelled":
            raise CommandValidationError("Order already cancelled")
        if self.status == "shipped":
            raise CommandValidationError("Cannot cancel shipped order")
        self._emit("order.cancelled", {"reason": cmd.payload.get("reason", "")})

    def handle_add_item(self, cmd: Command):
        if self.status not in ("created", "pending"):
            raise CommandValidationError(f"Cannot add items in status: {self.status}")
        item = cmd.payload["item"]
        if item.get("price", 0) <= 0:
            raise CommandValidationError("Item price must be positive")
        self._emit("item.added", {"item": item})

    def pop_events(self) -> List[DomainEvent]:
        events = self._events[:]
        self._events.clear()
        return events


# ─────────────────────────────────────────────
# WRITE STORE (Event Store / Command side DB)
# ─────────────────────────────────────────────

class WriteStore:
    """Persists domain events per aggregate. Source of truth for command side."""

    def __init__(self):
        self._events: Dict[str, List[DomainEvent]] = defaultdict(list)

    def append(self, events: List[DomainEvent]):
        for evt in events:
            self._events[evt.aggregate_id].append(evt)

    def load(self, aggregate_id: str) -> List[DomainEvent]:
        return list(self._events.get(aggregate_id, []))

    def all_events(self) -> List[DomainEvent]:
        result = []
        for evts in self._events.values():
            result.extend(evts)
        result.sort(key=lambda e: e.timestamp)
        return result


# ─────────────────────────────────────────────
# READ MODELS (Query side projections)
# ─────────────────────────────────────────────

@dataclass
class OrderSummaryView:
    order_id   : str
    status     : str
    item_count : int
    total      : float
    customer_id: str = ""


@dataclass
class CustomerOrdersView:
    customer_id: str
    orders     : List[Dict] = field(default_factory=list)

    @property
    def total_spent(self) -> float:
        return sum(o["total"] for o in self.orders if o["status"] != "cancelled")


class OrderReadModel:
    """
    Query-side read model. Subscribes to domain events and builds
    denormalized projections optimized for query patterns.
    """

    def __init__(self):
        self._summaries  : Dict[str, OrderSummaryView]   = {}
        self._by_customer: Dict[str, CustomerOrdersView] = defaultdict(
            lambda: CustomerOrdersView(customer_id=""))
        self._lock       = threading.Lock()
        self.events_applied = 0

    def apply(self, evt: DomainEvent):
        with self._lock:
            handler = {
                "order.created"  : self._on_created,
                "order.confirmed": self._on_confirmed,
                "order.cancelled": self._on_cancelled,
                "item.added"     : self._on_item_added,
            }.get(evt.event_type)
            if handler:
                handler(evt)
                self.events_applied += 1

    def _on_created(self, evt: DomainEvent):
        p   = evt.payload
        cid = p.get("customer_id", "")
        self._summaries[evt.aggregate_id] = OrderSummaryView(
            order_id    = evt.aggregate_id,
            status      = "created",
            item_count  = len(p["items"]),
            total       = p["total"],
            customer_id = cid,
        )
        if not self._by_customer[cid].customer_id:
            self._by_customer[cid].customer_id = cid
        self._by_customer[cid].orders.append({
            "order_id": evt.aggregate_id,
            "total"   : p["total"],
            "status"  : "created",
        })

    def _on_confirmed(self, evt: DomainEvent):
        if evt.aggregate_id in self._summaries:
            self._summaries[evt.aggregate_id].status = "confirmed"
            self._update_customer_order_status(evt.aggregate_id, "confirmed")

    def _on_cancelled(self, evt: DomainEvent):
        if evt.aggregate_id in self._summaries:
            self._summaries[evt.aggregate_id].status = "cancelled"
            self._update_customer_order_status(evt.aggregate_id, "cancelled")

    def _on_item_added(self, evt: DomainEvent):
        if evt.aggregate_id in self._summaries:
            summary = self._summaries[evt.aggregate_id]
            summary.item_count += 1
            summary.total      += evt.payload["item"]["price"]

    def _update_customer_order_status(self, order_id: str, status: str):
        for cv in self._by_customer.values():
            for o in cv.orders:
                if o["order_id"] == order_id:
                    o["status"] = status
                    return

    # ── Query Methods ──────────────────────────────
    def get_order(self, order_id: str) -> Optional[OrderSummaryView]:
        return self._summaries.get(order_id)

    def get_customer_orders(self, customer_id: str) -> Optional[CustomerOrdersView]:
        return self._by_customer.get(customer_id)

    def orders_by_status(self, status: str) -> List[OrderSummaryView]:
        return [v for v in self._summaries.values() if v.status == status]

    def rebuild(self, events: List[DomainEvent]):
        """Rebuild projection from scratch by replaying all events."""
        with self._lock:
            self._summaries.clear()
            self._by_customer.clear()
            self.events_applied = 0
        for evt in sorted(events, key=lambda e: e.timestamp):
            self.apply(evt)


# ─────────────────────────────────────────────
# COMMAND BUS (routes commands to aggregates)
# ─────────────────────────────────────────────

class CommandBus:
    def __init__(self, write_store: WriteStore, read_model: OrderReadModel):
        self.write_store = write_store
        self.read_model  = read_model
        self.commands_ok  = 0
        self.commands_err = 0

    def _load_aggregate(self, order_id: str) -> OrderAggregate:
        agg    = OrderAggregate(order_id)
        events = self.write_store.load(order_id)
        for evt in events:
            agg.apply(evt)
        return agg

    def handle(self, cmd: Command) -> bool:
        order_id = cmd.payload.get("order_id", cmd.command_id)
        agg = self._load_aggregate(order_id)
        try:
            handler_map = {
                "CreateOrder" : agg.handle_create,
                "ConfirmOrder": agg.handle_confirm,
                "CancelOrder" : agg.handle_cancel,
                "AddItem"     : agg.handle_add_item,
            }
            handler_map[cmd.command_type](cmd)
            new_events = agg.pop_events()
            self.write_store.append(new_events)
            for evt in new_events:
                self.read_model.apply(evt)   # synchronous projection update
            self.commands_ok += 1
            return True
        except (CommandValidationError, KeyError) as e:
            self.commands_err += 1
            print(f"  [CommandBus] REJECTED {cmd.command_type}: {e}")
            return False


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_cqrs():
    print("=" * 65)
    print("CQRS — COMMAND QUERY RESPONSIBILITY SEGREGATION")
    print("=" * 65)

    write_store = WriteStore()
    read_model  = OrderReadModel()
    bus         = CommandBus(write_store, read_model)

    # ── Command Side: Write Operations ───────────
    print("\n[1] COMMAND SIDE — WRITING")
    print("─" * 55)

    # Create orders
    orders = [
        ("ORD-001", "cust-A", [{"sku": "book", "price": 29.99}, {"sku": "pen", "price": 3.49}]),
        ("ORD-002", "cust-B", [{"sku": "laptop", "price": 999.0}]),
        ("ORD-003", "cust-A", [{"sku": "notebook", "price": 8.99}]),
    ]
    for order_id, customer_id, items in orders:
        cmd = Command(command_type="CreateOrder",
                      payload={"order_id": order_id, "customer_id": customer_id,
                               "items": items})
        ok = bus.handle(cmd)
        print(f"  CreateOrder {order_id}: {'✓' if ok else '✗'}")

    # Confirm one, cancel one, add item to another
    bus.handle(Command(command_type="ConfirmOrder",
                       payload={"order_id": "ORD-001"}))
    print(f"  ConfirmOrder ORD-001: ✓")

    bus.handle(Command(command_type="AddItem",
                       payload={"order_id": "ORD-003",
                                "item": {"sku": "stapler", "price": 12.50}}))
    print(f"  AddItem ORD-003: ✓")

    bus.handle(Command(command_type="CancelOrder",
                       payload={"order_id": "ORD-002", "reason": "customer request"}))
    print(f"  CancelOrder ORD-002: ✓")

    # Test validation rejection
    bus.handle(Command(command_type="ConfirmOrder",
                       payload={"order_id": "ORD-002"}))   # already cancelled

    print(f"\n  Commands OK: {bus.commands_ok}  Rejected: {bus.commands_err}")

    # ── Query Side: Read Projections ─────────────
    print("\n\n[2] QUERY SIDE — READING PROJECTIONS")
    print("─" * 55)

    print(f"  All orders by status:")
    for status in ["created", "confirmed", "cancelled"]:
        orders_in_status = read_model.orders_by_status(status)
        ids = [o.order_id for o in orders_in_status]
        print(f"    {status:<12}: {ids}")

    print(f"\n  Order detail (ORD-003 — had item added):")
    o3 = read_model.get_order("ORD-003")
    if o3:
        print(f"    order_id={o3.order_id} status={o3.status} "
              f"items={o3.item_count} total=${o3.total:.2f}")

    print(f"\n  Customer 'cust-A' view:")
    cv = read_model.get_customer_orders("cust-A")
    if cv:
        for o in cv.orders:
            print(f"    order={o['order_id']} status={o['status']} total=${o['total']:.2f}")
        print(f"    Total spent (non-cancelled): ${cv.total_spent:.2f}")

    # ── Projection Rebuild ────────────────────────
    print("\n\n[3] PROJECTION REBUILD FROM EVENT LOG")
    print("─" * 55)
    all_events = write_store.all_events()
    print(f"  Event log size: {len(all_events)} events")
    for evt in all_events:
        print(f"    [{evt.event_type:<20}] agg={evt.aggregate_id} v={evt.version}")

    print(f"\n  Rebuilding read model from scratch...")
    read_model.rebuild(all_events)
    print(f"  Applied {read_model.events_applied} events")
    o1_after = read_model.get_order("ORD-001")
    print(f"  ORD-001 after rebuild: status={o1_after.status} total=${o1_after.total:.2f}")

    # ── Summary ───────────────────────────────────
    print("\n\n[4] CQRS DESIGN GUIDE")
    print("─" * 55)
    rows = [
        ("Single model",  "Normalized, consistent",   "CRUD, simple apps"),
        ("CQRS",          "Separate W/R models",       "Complex domain, many read patterns"),
        ("CQRS + ES",     "Events = write store",      "Full audit, time-travel, event-driven"),
    ]
    print(f"  {'Approach':<14} {'Characteristic':<28} {'Best for'}")
    print(f"  {'─'*65}")
    for approach, char, best in rows:
        print(f"  {approach:<14} {char:<28} {best}")

    print()
    tradeoffs = [
        ("✓", "Read model can be rebuilt — zero data loss on schema change"),
        ("✓", "Read model tuned per query (denormalized, pre-joined, indexed)"),
        ("✓", "Read and write sides scale independently"),
        ("✗", "Eventual consistency — reads may lag behind writes"),
        ("✗", "More code: two models, two stores, sync mechanism"),
        ("✗", "Not worth it for simple CRUD with 1-2 query patterns"),
    ]
    for symbol, note in tradeoffs:
        print(f"  {symbol} {note}")


if __name__ == "__main__":
    demonstrate_cqrs()
