"""
DISTRIBUTED TRANSACTIONS AND SAGA PATTERN
===========================================

Problem Statement:
A single business operation (e.g., "place order") may span multiple services:
  1. Reserve inventory   (Inventory Service)
  2. Charge credit card  (Payment Service)
  3. Create shipment     (Shipping Service)

With a monolith + single DB, a SQL transaction ensures atomicity.
Across microservices with separate DBs, you cannot use a single transaction.

Two-Phase Commit (2PC):
  Phase 1 (Prepare): Coordinator asks all participants to prepare.
          Each participant writes to a durable log, replies YES/NO.
  Phase 2 (Commit/Abort): If ALL said YES → commit. Any NO → abort.
  Problem: coordinator is a SPOF; participants block while coordinator is down.
  Use: XA transactions, distributed DBs (Spanner, CockroachDB)

Saga Pattern:
  Break the long-running transaction into a sequence of local transactions.
  Each step publishes an event or calls the next service.
  If a step fails, execute compensating transactions in reverse.

  Orchestration Saga:
    Central orchestrator (workflow engine) calls each service in sequence.
    Easier to track, centralized error handling. (AWS Step Functions)

  Choreography Saga:
    Each service listens for events and reacts.
    Decoupled, but hard to visualize the overall flow.

Compensation vs Rollback:
  Rollback: undo all changes atomically (only possible within single DB).
  Compensation: execute a semantic undo (e.g., "refund payment").
  Compensations are not instant — they themselves may fail.

Idempotency:
  Every step must be idempotent — safe to retry on failure.
  Use idempotency keys to detect and skip duplicate requests.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import time
import uuid
import random
import threading
from collections import defaultdict


class SagaStatus(Enum):
    PENDING     = "pending"
    RUNNING     = "running"
    COMPLETED   = "completed"
    COMPENSATING= "compensating"
    FAILED      = "failed"


class StepStatus(Enum):
    PENDING   = "pending"
    SUCCEEDED = "succeeded"
    FAILED    = "failed"
    COMPENSATED = "compensated"


class TwoPhaseStatus(Enum):
    INIT    = "init"
    PREPARED= "prepared"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass
class SagaStep:
    name         : str
    execute_fn   : Callable
    compensate_fn: Callable
    status       : StepStatus = StepStatus.PENDING
    result       : Any = None
    error        : str = None


@dataclass
class SagaExecution:
    saga_id    : str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    steps      : List[SagaStep] = field(default_factory=list)
    status     : SagaStatus = SagaStatus.PENDING
    completed_steps: List[str] = field(default_factory=list)
    context    : Dict[str, Any] = field(default_factory=dict)
    created_at : float = field(default_factory=time.time)
    events     : List[str] = field(default_factory=list)

    def log(self, msg: str):
        self.events.append(f"[{time.time():.3f}] {msg}")


# ─────────────────────────────────────────────
# TWO-PHASE COMMIT
# ─────────────────────────────────────────────

@dataclass
class XAParticipant:
    """Simulates a database/service that supports XA transactions."""
    name    : str
    fail_on : str = None   # simulate failure: "prepare" or "commit"

    def __post_init__(self):
        self._prepared_txns : Dict[str, Any] = {}
        self._committed     : Dict[str, Any] = {}
        self.prepare_calls  = 0
        self.commit_calls   = 0
        self.abort_calls    = 0

    def prepare(self, txn_id: str, data: Any) -> bool:
        self.prepare_calls += 1
        if self.fail_on == "prepare":
            return False
        # Write to durable log (WAL) — can survive crash
        self._prepared_txns[txn_id] = data
        return True

    def commit(self, txn_id: str) -> bool:
        self.commit_calls += 1
        if self.fail_on == "commit":
            return False
        data = self._prepared_txns.pop(txn_id, None)
        if data:
            self._committed[txn_id] = data
        return True

    def abort(self, txn_id: str):
        self.abort_calls += 1
        self._prepared_txns.pop(txn_id, None)


class TwoPhaseCommitCoordinator:
    """
    2PC Coordinator: runs prepare phase then commit/abort phase.
    Single point of failure — if coordinator crashes between phases,
    participants are blocked until recovery.
    """

    def __init__(self, participants: List[XAParticipant]):
        self.participants = participants
        self.txn_log      : Dict[str, TwoPhaseStatus] = {}
        self.commits      = 0
        self.aborts       = 0

    def execute(self, txn_id: str, payloads: Dict[str, Any]) -> bool:
        self.txn_log[txn_id] = TwoPhaseStatus.INIT
        print(f"  2PC txn={txn_id}")

        # ── Phase 1: Prepare ─────────────────
        print(f"    Phase 1 (Prepare): asking {len(self.participants)} participants...")
        votes = {}
        for p in self.participants:
            payload = payloads.get(p.name, {})
            vote    = p.prepare(txn_id, payload)
            votes[p.name] = vote
            status  = "YES" if vote else "NO"
            print(f"      {p.name}: {status}")

        all_yes = all(votes.values())
        self.txn_log[txn_id] = TwoPhaseStatus.PREPARED

        # ── Phase 2: Commit or Abort ──────────
        if all_yes:
            print(f"    Phase 2 (Commit): all voted YES → committing...")
            for p in self.participants:
                ok = p.commit(txn_id)
                print(f"      {p.name}: {'OK' if ok else 'FAILED'}")
            self.txn_log[txn_id] = TwoPhaseStatus.COMMITTED
            self.commits += 1
            return True
        else:
            print(f"    Phase 2 (Abort): some voted NO → aborting all...")
            for p in self.participants:
                p.abort(txn_id)
                print(f"      {p.name}: aborted")
            self.txn_log[txn_id] = TwoPhaseStatus.ABORTED
            self.aborts += 1
            return False


# ─────────────────────────────────────────────
# SAGA ORCHESTRATOR
# ─────────────────────────────────────────────

class SagaOrchestrator:
    """
    Orchestration-style Saga: central coordinator calls each step.
    On failure, calls compensating transactions in reverse order.
    """

    def __init__(self):
        self.sagas     : Dict[str, SagaExecution] = {}
        self.completed = 0
        self.failed    = 0

    def create_saga(self, steps: List[SagaStep], context: Dict = None) -> SagaExecution:
        saga = SagaExecution(steps=steps, context=context or {})
        self.sagas[saga.saga_id] = saga
        return saga

    def execute(self, saga: SagaExecution) -> bool:
        saga.status = SagaStatus.RUNNING
        saga.log(f"Saga {saga.saga_id} started with {len(saga.steps)} steps")

        # Forward execution
        for step in saga.steps:
            saga.log(f"Executing step: {step.name}")
            try:
                result = step.execute_fn(saga.context)
                step.status = StepStatus.SUCCEEDED
                step.result = result
                saga.completed_steps.append(step.name)
                saga.log(f"Step {step.name} succeeded: {result}")
            except Exception as e:
                step.status = StepStatus.FAILED
                step.error  = str(e)
                saga.log(f"Step {step.name} FAILED: {e}")

                # Trigger compensation
                self._compensate(saga)
                saga.status = SagaStatus.FAILED
                self.failed += 1
                return False

        saga.status = SagaStatus.COMPLETED
        self.completed += 1
        saga.log(f"Saga {saga.saga_id} completed successfully")
        return True

    def _compensate(self, saga: SagaExecution):
        saga.status = SagaStatus.COMPENSATING
        saga.log("Starting compensation (reverse order)...")
        # Compensate completed steps in reverse
        for step in reversed(saga.steps):
            if step.status == StepStatus.SUCCEEDED:
                saga.log(f"Compensating step: {step.name}")
                try:
                    step.compensate_fn(saga.context)
                    step.status = StepStatus.COMPENSATED
                    saga.log(f"Compensation for {step.name} succeeded")
                except Exception as e:
                    saga.log(f"Compensation for {step.name} FAILED: {e} — requires manual intervention")


# ─────────────────────────────────────────────
# CHOREOGRAPHY SAGA (event-based)
# ─────────────────────────────────────────────

@dataclass
class DomainEvent:
    event_id   : str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type : str = ""
    saga_id    : str = ""
    payload    : Dict[str, Any] = field(default_factory=dict)
    timestamp  : float = field(default_factory=time.time)


class EventBus:
    """Simple in-process event bus for choreography demo."""

    def __init__(self):
        self._subscribers : Dict[str, List[Callable]] = defaultdict(list)
        self._events      : List[DomainEvent] = []

    def subscribe(self, event_type: str, handler: Callable):
        self._subscribers[event_type].append(handler)

    def publish(self, event: DomainEvent):
        self._events.append(event)
        for handler in self._subscribers.get(event.event_type, []):
            handler(event)

    @property
    def event_count(self) -> int:
        return len(self._events)


class ChoreographyService:
    """Base class for choreography saga participants."""

    def __init__(self, name: str, bus: EventBus):
        self.name   = name
        self.bus    = bus
        self.log    : List[str] = []

    def emit(self, event_type: str, saga_id: str, payload: Dict):
        event = DomainEvent(event_type=event_type, saga_id=saga_id, payload=payload)
        self.log.append(f"EMIT {event_type}")
        self.bus.publish(event)


# ─────────────────────────────────────────────
# ORDER SAGA SIMULATION
# ─────────────────────────────────────────────

class InventoryService:
    def __init__(self):
        self._inventory = {"laptop": 10, "mouse": 50, "keyboard": 5}
        self._reservations : Dict[str, Dict] = {}

    def reserve(self, context: Dict) -> Dict:
        item     = context.get("item")
        quantity = context.get("quantity", 1)
        if self._inventory.get(item, 0) < quantity:
            raise Exception(f"Insufficient stock for {item}")
        self._inventory[item] -= quantity
        reservation_id = str(uuid.uuid4())[:8]
        self._reservations[reservation_id] = {"item": item, "quantity": quantity}
        context["reservation_id"] = reservation_id
        return {"reservation_id": reservation_id, "item": item}

    def release(self, context: Dict):
        reservation_id = context.get("reservation_id")
        if reservation_id and reservation_id in self._reservations:
            res = self._reservations.pop(reservation_id)
            self._inventory[res["item"]] += res["quantity"]


class PaymentService:
    def __init__(self):
        self._balances = {"user-1": 500.0, "user-2": 50.0}
        self._charges  : Dict[str, float] = {}

    def charge(self, context: Dict) -> Dict:
        user_id = context.get("user_id")
        amount  = context.get("amount", 0)
        balance = self._balances.get(user_id, 0)
        if balance < amount:
            raise Exception(f"Insufficient funds for {user_id} (balance={balance}, need={amount})")
        self._balances[user_id] -= amount
        charge_id = str(uuid.uuid4())[:8]
        self._charges[charge_id] = amount
        context["charge_id"] = charge_id
        return {"charge_id": charge_id, "amount": amount}

    def refund(self, context: Dict):
        charge_id = context.get("charge_id")
        user_id   = context.get("user_id")
        if charge_id and charge_id in self._charges:
            amount = self._charges.pop(charge_id)
            self._balances[user_id] = self._balances.get(user_id, 0) + amount


class ShippingService:
    def __init__(self):
        self._shipments : Dict[str, Dict] = {}

    def create_shipment(self, context: Dict) -> Dict:
        shipment_id = str(uuid.uuid4())[:8]
        self._shipments[shipment_id] = {
            "item": context.get("item"),
            "user": context.get("user_id"),
            "address": context.get("address", "123 Main St"),
        }
        context["shipment_id"] = shipment_id
        return {"shipment_id": shipment_id}

    def cancel_shipment(self, context: Dict):
        shipment_id = context.get("shipment_id")
        if shipment_id:
            self._shipments.pop(shipment_id, None)


# ─────────────────────────────────────────────
# IDEMPOTENCY KEY STORE
# ─────────────────────────────────────────────

class IdempotencyStore:
    """
    Stores results of completed operations keyed by idempotency key.
    On duplicate request, returns cached result instead of re-executing.
    """

    def __init__(self):
        self._store    : Dict[str, Any] = {}
        self.cache_hits = 0
        self.executions = 0

    def get_or_execute(self, key: str, fn: Callable) -> Any:
        if key in self._store:
            self.cache_hits += 1
            return self._store[key]
        result = fn()
        self._store[key] = result
        self.executions += 1
        return result


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_distributed_transactions():
    print("=" * 65)
    print("DISTRIBUTED TRANSACTIONS AND SAGA PATTERN")
    print("=" * 65)

    random.seed(42)

    # ── Two-Phase Commit ──────────────────────
    print("\n[1] TWO-PHASE COMMIT (2PC)")
    print("─" * 55)

    # Successful 2PC
    inventory_db = XAParticipant("inventory_db")
    payment_db   = XAParticipant("payment_db")
    shipping_db  = XAParticipant("shipping_db")

    coordinator = TwoPhaseCommitCoordinator([inventory_db, payment_db, shipping_db])
    ok = coordinator.execute("txn-001", {
        "inventory_db": {"item": "laptop", "quantity": -1},
        "payment_db":   {"user": "user-1", "amount": 999},
        "shipping_db":  {"address": "123 Main St"},
    })
    print(f"  Result: {'COMMITTED' if ok else 'ABORTED'}\n")

    # 2PC with one participant failing prepare
    payment_fail = XAParticipant("payment_db", fail_on="prepare")
    coordinator2 = TwoPhaseCommitCoordinator([inventory_db, payment_fail, shipping_db])
    ok2 = coordinator2.execute("txn-002", {
        "inventory_db": {"item": "laptop", "quantity": -1},
        "payment_db":   {"user": "user-2", "amount": 999},
        "shipping_db":  {"address": "456 Oak Ave"},
    })
    print(f"  Result: {'COMMITTED' if ok2 else 'ABORTED'}")
    print(f"\n  2PC Limitations:")
    print(f"    • Coordinator SPOF — if coordinator crashes after Phase 1")
    print(f"      but before Phase 2, participants block indefinitely")
    print(f"    • Synchronous blocking — all participants must be available")
    print(f"    • Not suitable for microservices across network partitions")

    # ── Saga Orchestration ─────────────────────
    print("\n\n[2] SAGA ORCHESTRATION PATTERN")
    print("─" * 55)

    inv_svc  = InventoryService()
    pay_svc  = PaymentService()
    ship_svc = ShippingService()
    orch     = SagaOrchestrator()

    # Successful order saga
    print("  Scenario A: successful order (laptop for user-1, $999)")
    saga_a = orch.create_saga(
        steps=[
            SagaStep("reserve_inventory",
                     execute_fn=lambda ctx: inv_svc.reserve(ctx),
                     compensate_fn=lambda ctx: inv_svc.release(ctx)),
            SagaStep("charge_payment",
                     execute_fn=lambda ctx: pay_svc.charge(ctx),
                     compensate_fn=lambda ctx: pay_svc.refund(ctx)),
            SagaStep("create_shipment",
                     execute_fn=lambda ctx: ship_svc.create_shipment(ctx),
                     compensate_fn=lambda ctx: ship_svc.cancel_shipment(ctx)),
        ],
        context={"user_id": "user-1", "item": "laptop", "quantity": 1,
                 "amount": 999.0, "address": "123 Main St"}
    )
    success_a = orch.execute(saga_a)
    print(f"  Status: {saga_a.status.value}")
    for event in saga_a.events:
        print(f"    {event}")

    # Failed order saga (insufficient funds) → compensation
    print(f"\n  Scenario B: failed order (laptop for user-2, $999 — only $50 balance)")
    saga_b = orch.create_saga(
        steps=[
            SagaStep("reserve_inventory",
                     execute_fn=lambda ctx: inv_svc.reserve(ctx),
                     compensate_fn=lambda ctx: inv_svc.release(ctx)),
            SagaStep("charge_payment",
                     execute_fn=lambda ctx: pay_svc.charge(ctx),
                     compensate_fn=lambda ctx: pay_svc.refund(ctx)),
            SagaStep("create_shipment",
                     execute_fn=lambda ctx: ship_svc.create_shipment(ctx),
                     compensate_fn=lambda ctx: ship_svc.cancel_shipment(ctx)),
        ],
        context={"user_id": "user-2", "item": "laptop", "quantity": 1,
                 "amount": 999.0, "address": "456 Oak Ave"}
    )
    success_b = orch.execute(saga_b)
    print(f"  Status: {saga_b.status.value}")
    for event in saga_b.events:
        print(f"    {event}")

    print(f"\n  Orchestrator stats: completed={orch.completed}  failed={orch.failed}")

    # ── Choreography Saga ─────────────────────
    print("\n\n[3] SAGA CHOREOGRAPHY PATTERN (event-driven)")
    print("─" * 55)

    bus = EventBus()

    # Services publish/subscribe to events
    class OrderSvc(ChoreographyService):
        def place_order(self, saga_id, item, user):
            print(f"    OrderService → placing order for {item}")
            self.emit("ORDER_CREATED", saga_id, {"item": item, "user": user})

    class InvSvcChoreo(ChoreographyService):
        def __init__(self, bus):
            super().__init__("InventoryService", bus)
            bus.subscribe("ORDER_CREATED", self.on_order_created)
            bus.subscribe("PAYMENT_FAILED", self.on_payment_failed)

        def on_order_created(self, event):
            print(f"    InventoryService → reserving {event.payload['item']}")
            self.emit("INVENTORY_RESERVED", event.saga_id, event.payload)

        def on_payment_failed(self, event):
            print(f"    InventoryService → releasing reservation (compensation)")

    class PaySvcChoreo(ChoreographyService):
        def __init__(self, bus):
            super().__init__("PaymentService", bus)
            bus.subscribe("INVENTORY_RESERVED", self.on_inventory_reserved)

        def on_inventory_reserved(self, event):
            user = event.payload.get("user", "unknown")
            print(f"    PaymentService → charging {user}")
            # Simulate payment failure for user-2
            if user == "user-2":
                print(f"    PaymentService → charge FAILED (insufficient funds)")
                self.emit("PAYMENT_FAILED", event.saga_id, event.payload)
            else:
                self.emit("PAYMENT_SUCCEEDED", event.saga_id, event.payload)

    class ShipSvcChoreo(ChoreographyService):
        def __init__(self, bus):
            super().__init__("ShippingService", bus)
            bus.subscribe("PAYMENT_SUCCEEDED", self.on_payment_succeeded)

        def on_payment_succeeded(self, event):
            print(f"    ShippingService → creating shipment")
            self.emit("ORDER_COMPLETED", event.saga_id, event.payload)

    order_svc = OrderSvc("OrderService", bus)
    inv_choreo = InvSvcChoreo(bus)
    pay_choreo = PaySvcChoreo(bus)
    ship_choreo = ShipSvcChoreo(bus)

    print("  Scenario A: user-1 orders laptop (successful)")
    order_svc.place_order("saga-choreo-1", "laptop", "user-1")
    print(f"  Events published: {bus.event_count}\n")

    print("  Scenario B: user-2 orders laptop (payment fails → compensation)")
    order_svc.place_order("saga-choreo-2", "laptop", "user-2")
    print(f"  Events published: {bus.event_count}")

    # ── Idempotency ──────────────────────────
    print("\n\n[4] IDEMPOTENCY KEYS (safe retries)")
    print("─" * 55)
    idem_store = IdempotencyStore()

    charge_count = {"n": 0}

    def charge_user():
        charge_count["n"] += 1
        return {"charge_id": "ch_abc123", "amount": 999.0}

    # Simulate 3 retries of the same request (network timeout after success)
    for attempt in range(3):
        result = idem_store.get_or_execute("req-idempotency-key-xyz", charge_user)
        print(f"  Attempt {attempt + 1}: charge_count={charge_count['n']}  result={result}")

    print(f"\n  Cache hits: {idem_store.cache_hits}  Actual executions: {idem_store.executions}")
    print("  → Idempotency key prevented duplicate charges on retry")

    # ── Comparison ────────────────────────────
    print("\n\n[5] 2PC vs SAGA COMPARISON")
    print("─" * 55)
    comparison = [
        ("Consistency",  "Strong (ACID)",    "Eventual (BASE)"),
        ("Blocking",     "Yes — blocks on crash", "No — async compensation"),
        ("Throughput",   "Low (sync locks)", "High (async events)"),
        ("Failure mode", "Coordinator SPOF", "Compensation chain"),
        ("Complexity",   "Low (DB handles)", "High (compensations needed)"),
        ("Use case",     "Same-DB txns, Spanner", "Microservices, long-running"),
    ]
    print(f"  {'Aspect':<16} {'2PC':<28} {'Saga'}")
    print(f"  {'─'*65}")
    for aspect, two_pc, saga in comparison:
        print(f"  {aspect:<16} {two_pc:<28} {saga}")


if __name__ == "__main__":
    demonstrate_distributed_transactions()
