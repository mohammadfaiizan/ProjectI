"""
READ REPLICAS AND CQRS
========================

Problem Statement:
Most applications have many more reads than writes (read-heavy workloads).
Vertical scaling the primary DB is expensive and has limits. Two complementary
patterns solve this: Read Replicas (scale reads by adding replicas) and CQRS
(separate read and write data models for optimal query performance).

Read Replicas:
  - Primary handles all writes
  - Replicas handle read traffic
  - Accept eventual consistency for replica reads
  - Typical setup: 1 primary + 2-4 replicas
  - AWS RDS: up to 5 replicas per primary

CQRS (Command Query Responsibility Segregation):
  - Commands: write operations that change state
  - Queries: read operations that return data
  - Separate models for reads and writes
  - Command Model: optimized for writes (normalized, ACID)
  - Query Model: optimized for reads (denormalized, pre-joined, indexed)
  - Often paired with Event Sourcing

Why CQRS?
  - Read and write have different performance characteristics
  - Read model can be materialized view, optimized per-use-case
  - Scale reads and writes independently
  - Different consistency requirements
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import time
import random
import uuid


class ModelType(Enum):
    COMMAND = "command"   # write side
    QUERY   = "query"     # read side


@dataclass
class DomainEvent:
    event_id  : str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type: str = ""
    payload   : Dict = field(default_factory=dict)
    timestamp : float = field(default_factory=time.time)
    version   : int = 1


# ─────────────────────────────────────────────
# READ REPLICA ROUTER
# ─────────────────────────────────────────────

class ReadReplicaRouter:
    """
    Routes writes to primary, reads to replicas (round-robin).
    Supports read-your-writes (route to primary within window).
    """

    def __init__(self, primary_latency_ms: float = 5.0,
                 replica_latency_ms: float = 3.0):
        self._primary_data: Dict[str, Any] = {}
        self._replica_data: List[Dict[str, Any]] = []
        self._n_replicas   = 0
        self._rr_idx       = 0
        self._primary_lat  = primary_latency_ms
        self._replica_lat  = replica_latency_ms
        self._repl_lag_ms  = 50.0   # simulated replication lag

        self.primary_writes  = 0
        self.replica_reads   = 0
        self.primary_reads   = 0   # forced reads from primary
        self._user_write_ts  : Dict[str, float] = {}

    def add_replica(self):
        self._replica_data.append(dict(self._primary_data))
        self._n_replicas += 1

    def write(self, user_id: str, key: str, value: Any):
        self.primary_writes += 1
        self._primary_data[key] = value
        self._user_write_ts[user_id] = time.time()
        # Async propagation to replicas (with simulated lag)
        for rd in self._replica_data:
            rd[key] = value   # in reality, this happens ~50ms later

    def read(self, user_id: str, key: str,
             force_primary: bool = False) -> Tuple[Any, str]:
        # Read-your-writes: if user wrote recently, route to primary
        last_write = self._user_write_ts.get(user_id, 0)
        if force_primary or (time.time() - last_write < 1.0):
            self.primary_reads += 1
            return self._primary_data.get(key), "primary"

        if not self._replica_data:
            return self._primary_data.get(key), "primary"

        self.replica_reads += 1
        replica = self._replica_data[self._rr_idx % self._n_replicas]
        self._rr_idx += 1
        return replica.get(key), f"replica-{self._rr_idx % self._n_replicas}"

    def report(self):
        total_reads = self.primary_reads + self.replica_reads
        offload_pct = self.replica_reads / max(1, total_reads) * 100
        print(f"\n  Read Replica Router:")
        print(f"    Writes to primary  : {self.primary_writes}")
        print(f"    Reads from primary : {self.primary_reads}")
        print(f"    Reads from replicas: {self.replica_reads}")
        print(f"    Primary offload    : {offload_pct:.0f}%")


# ─────────────────────────────────────────────
# CQRS — COMMAND SIDE
# ─────────────────────────────────────────────

class OrderCommandModel:
    """
    Command side — normalized write model.
    Emits domain events on state changes.
    """

    def __init__(self, event_bus: List[DomainEvent]):
        self._orders   : Dict[str, Dict] = {}
        self._items    : Dict[str, List] = {}   # order_id → [items]
        self._event_bus = event_bus

    def place_order(self, user_id: str, items: List[Dict],
                     shipping_addr: str) -> str:
        order_id = str(uuid.uuid4())[:8]
        self._orders[order_id] = {
            "order_id": order_id, "user_id": user_id,
            "status": "placed", "shipping_addr": shipping_addr,
            "total_usd": sum(i["price"] * i["qty"] for i in items),
            "created_at": time.time()
        }
        self._items[order_id] = items

        event = DomainEvent(
            event_type="OrderPlaced",
            payload={"order_id": order_id, "user_id": user_id,
                      "total_usd": self._orders[order_id]["total_usd"],
                      "items": items, "shipping_addr": shipping_addr}
        )
        self._event_bus.append(event)
        return order_id

    def ship_order(self, order_id: str, tracking_num: str) -> bool:
        order = self._orders.get(order_id)
        if not order:
            return False
        order["status"]      = "shipped"
        order["tracking_num"]= tracking_num
        order["shipped_at"]  = time.time()
        self._event_bus.append(DomainEvent(
            event_type="OrderShipped",
            payload={"order_id": order_id, "tracking_num": tracking_num}
        ))
        return True

    def cancel_order(self, order_id: str, reason: str) -> bool:
        order = self._orders.get(order_id)
        if not order or order["status"] == "shipped":
            return False
        order["status"] = "cancelled"
        self._event_bus.append(DomainEvent(
            event_type="OrderCancelled",
            payload={"order_id": order_id, "reason": reason}
        ))
        return True


# ─────────────────────────────────────────────
# CQRS — QUERY SIDE (Materialized Views)
# ─────────────────────────────────────────────

class OrderQueryModel:
    """
    Query side — denormalized, pre-joined, optimized for specific queries.
    Updated by consuming domain events (eventually consistent with command).
    """

    def __init__(self):
        # Multiple denormalized projections for different access patterns
        self._by_user  : Dict[str, List[Dict]] = {}   # user_id → [order summaries]
        self._by_status: Dict[str, List[Dict]] = {}   # status  → [order summaries]
        self._detail   : Dict[str, Dict] = {}          # order_id → full detail

    def handle_event(self, event: DomainEvent):
        """Project events into read models."""
        if event.event_type == "OrderPlaced":
            p = event.payload
            summary = {
                "order_id": p["order_id"], "status": "placed",
                "total_usd": p["total_usd"], "items_count": len(p["items"]),
                "created_at": event.timestamp
            }
            self._by_user.setdefault(p["user_id"], []).append(summary)
            self._by_status.setdefault("placed", []).append(summary)
            # Denormalized detail — joins items + address inline
            self._detail[p["order_id"]] = {**summary, "items": p["items"],
                                             "shipping_addr": p["shipping_addr"]}

        elif event.event_type == "OrderShipped":
            p       = event.payload
            oid     = p["order_id"]
            detail  = self._detail.get(oid, {})
            # Update projections in-place
            detail.update({"status": "shipped", "tracking_num": p["tracking_num"]})
            # Update user summary
            uid = detail.get("user_id")
            if uid and uid in self._by_user:
                for s in self._by_user[uid]:
                    if s["order_id"] == oid:
                        s["status"]       = "shipped"
                        s["tracking_num"] = p["tracking_num"]

        elif event.event_type == "OrderCancelled":
            p   = event.payload
            oid = p["order_id"]
            if oid in self._detail:
                self._detail[oid]["status"] = "cancelled"

    def get_user_orders(self, user_id: str) -> List[Dict]:
        """O(1) — pre-indexed by user."""
        return self._by_user.get(user_id, [])

    def get_order_detail(self, order_id: str) -> Optional[Dict]:
        """O(1) — pre-indexed by order_id; includes items (no join needed)."""
        return self._detail.get(order_id)

    def get_by_status(self, status: str) -> List[Dict]:
        """O(1) — pre-indexed by status."""
        return self._by_status.get(status, [])


# ─────────────────────────────────────────────
# EVENT PROJECTOR (synchronizes command→query)
# ─────────────────────────────────────────────

class EventProjector:
    """Consumes events from bus and updates query models."""

    def __init__(self, query_model: OrderQueryModel):
        self.query_model = query_model
        self._processed  = 0
        self._last_event_id: Optional[str] = None

    def process_events(self, events: List[DomainEvent]):
        for event in events:
            if event.event_id != self._last_event_id:
                self.query_model.handle_event(event)
                self._last_event_id = event.event_id
                self._processed += 1

    @property
    def processed_count(self) -> int:
        return self._processed


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_read_replicas_cqrs():
    print("=" * 65)
    print("READ REPLICAS AND CQRS")
    print("=" * 65)

    random.seed(42)

    # ── Read Replicas ─────────────────────────
    print("\n[1] READ REPLICA SCALING")
    print("─" * 55)
    router = ReadReplicaRouter()
    for _ in range(3):
        router.add_replica()

    # Simulate write-heavy then read-heavy workload
    print("  Writes (to primary):")
    users = [f"user_{i}" for i in range(5)]
    for i, user in enumerate(users):
        router.write(user, f"profile:{user}", {"name": user, "level": i})
        print(f"    WRITE profile:{user} → primary")

    print("\n  Reads (to replicas, except recent writers):")
    for i in range(12):
        user  = random.choice(users)
        key   = f"profile:{user}"
        val, source = router.read(user, key)
        print(f"    READ {key:<20} ← {source}")

    router.report()

    # ── CQRS ──────────────────────────────────
    print("\n\n[2] CQRS — SEPARATE READ AND WRITE MODELS")
    print("─" * 55)
    event_bus = []
    command   = OrderCommandModel(event_bus)
    query     = OrderQueryModel()
    projector = EventProjector(query)

    # Place orders (command side)
    print("  Commands:")
    o1 = command.place_order("alice", [
        {"product": "Laptop", "price": 999.99, "qty": 1},
        {"product": "Mouse",  "price": 29.99,  "qty": 2},
    ], "123 Main St, NYC")
    print(f"    PlaceOrder(alice) → order_id={o1}")

    o2 = command.place_order("bob", [
        {"product": "Keyboard", "price": 79.99, "qty": 1},
    ], "456 Oak Ave, LA")
    print(f"    PlaceOrder(bob)   → order_id={o2}")

    o3 = command.place_order("alice", [
        {"product": "Monitor", "price": 399.99, "qty": 1},
    ], "123 Main St, NYC")
    print(f"    PlaceOrder(alice) → order_id={o3}")

    command.ship_order(o1, "TRACK-001")
    print(f"    ShipOrder({o1}) tracking=TRACK-001")

    command.cancel_order(o2, "out of stock")
    print(f"    CancelOrder({o2}) reason='out of stock'")

    # Project events to query model
    projector.process_events(event_bus)
    print(f"\n  Projected {projector.processed_count} events to query model")

    # Query side — no joins, pre-materialized
    print("\n  Queries (zero joins needed):")
    alice_orders = query.get_user_orders("alice")
    print(f"  Alice's orders ({len(alice_orders)}):")
    for o in alice_orders:
        print(f"    {o['order_id']}: status={o['status']}  total=${o.get('total_usd', 0):.2f}  "
              f"tracking={o.get('tracking_num', 'N/A')}")

    detail = query.get_order_detail(o1)
    print(f"\n  Order detail for {o1} (pre-joined, no SELECT JOIN needed):")
    if detail:
        print(f"    status={detail['status']}  items={detail['items']}  "
              f"tracking={detail.get('tracking_num')}")

    by_status = query.get_by_status("placed")
    print(f"\n  Orders with status='placed': {len(by_status)}")

    # ── Benefits ──────────────────────────────
    print("\n\n[3] CQRS BENEFITS SUMMARY")
    print("─" * 55)
    benefits = [
        ("Independent scaling", "Scale read/write services separately"),
        ("Optimized models",    "Command: normalized; Query: denormalized per use case"),
        ("No joins on reads",   "Pre-materialized views → O(1) queries"),
        ("Event history",       "Complete audit trail via event bus"),
        ("Multiple projections","Same events → user view, admin view, analytics"),
        ("Cache friendly",      "Query model is read-only → easy to cache"),
    ]
    for benefit, detail in benefits:
        print(f"  ✅ {benefit:<25} {detail}")

    print("\n\n[4] WHEN TO USE EACH PATTERN")
    print("─" * 55)
    rows = [
        ("Read Replicas",   "Simple, many reads, standard app",    "Moderate"),
        ("CQRS",            "Complex reads, different models needed","High"),
        ("CQRS + EventSrc", "Full audit, time-travel, projections", "Highest"),
        ("None needed",     "< 1K users, simple CRUD app",          "Lowest"),
    ]
    print(f"  {'Pattern':<22} {'Use When':<40} {'Complexity'}")
    print(f"  {'─'*75}")
    for pattern, use_when, complexity in rows:
        print(f"  {pattern:<22} {use_when:<40} {complexity}")


from typing import Tuple  # ensure Tuple is imported for use in ReadReplicaRouter

if __name__ == "__main__":
    demonstrate_read_replicas_cqrs()
