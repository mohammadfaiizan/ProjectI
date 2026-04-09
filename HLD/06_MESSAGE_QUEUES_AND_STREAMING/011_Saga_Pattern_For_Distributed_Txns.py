"""
SAGA PATTERN FOR DISTRIBUTED TRANSACTIONS
==========================================

Problem Statement:
Distributed systems can't use a single database transaction across services.
Two-phase commit (2PC) exists but is slow, blocking, and rarely used at scale.
The Saga pattern provides a way to maintain data consistency across services
without distributed locks, using a sequence of local transactions + compensations.

How it works:
  A Saga is a sequence of local transactions T1, T2, ..., Tn.
  Each Ti is atomic within its own service/database.
  If Ti succeeds but Ti+1 fails: run compensating transactions C(i), C(i-1), ..., C1.
  Compensations undo the effect of already-completed transactions.

Two Saga implementations:
  1. Choreography Saga:
     Services communicate via events. No central coordinator.
     Each service listens for events, performs its transaction, publishes outcome.
     Pro: decoupled. Con: hard to track overall state; complex error paths.

  2. Orchestration Saga:
     A central Saga Orchestrator directs each service step.
     Orchestrator knows the full workflow and handles failures explicitly.
     Pro: easy to understand and debug. Con: orchestrator = central component.

Compensating Transactions:
  Must be idempotent (safe to call multiple times on retry).
  Should be semantic undo, not literal undo:
    T: "reserve inventory" → C: "release reservation" (not "delete row")
  Some compensations are impossible (email sent) → accept + send correction.

Key Failure Modes:
  - Step fails: run compensations for all completed steps in reverse.
  - Compensation fails: retry compensation (idempotent!) until success.
  - Orchestrator crashes: persist Saga state; resume on restart.
  - Partial failure: Saga is "in progress" until all compensations complete.

Saga State Machine:
  PENDING → RUNNING → COMPLETED
                    ↘ COMPENSATING → COMPENSATED (failed saga, rolled back)
                                   ↘ FAILED (compensation failed — human intervention)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import time
import uuid
import threading
import random


# ─────────────────────────────────────────────
# SAGA STATE
# ─────────────────────────────────────────────

class SagaStatus(Enum):
    PENDING       = "pending"
    RUNNING       = "running"
    COMPLETED     = "completed"
    COMPENSATING  = "compensating"
    COMPENSATED   = "compensated"
    FAILED        = "failed"   # compensation also failed


@dataclass
class StepResult:
    step_name : str
    success   : bool
    output    : Any    = None
    error     : str    = ""
    timestamp : float  = field(default_factory=time.time)


@dataclass
class SagaState:
    saga_id       : str          = field(default_factory=lambda: str(uuid.uuid4())[:8])
    saga_type     : str          = ""
    status        : SagaStatus   = SagaStatus.PENDING
    current_step  : int          = 0
    context       : Dict         = field(default_factory=dict)
    step_results  : List[StepResult] = field(default_factory=list)
    created_at    : float        = field(default_factory=time.time)
    completed_at  : Optional[float] = None


# ─────────────────────────────────────────────
# SAGA STEP DEFINITION
# ─────────────────────────────────────────────

@dataclass
class SagaStep:
    name       : str
    execute    : Callable[[Dict], Dict]      # performs the transaction; returns output dict
    compensate : Callable[[Dict], bool]      # undoes the transaction; returns success bool
    retry_on_compensate: bool = True


# ─────────────────────────────────────────────
# SAGA ORCHESTRATOR
# ─────────────────────────────────────────────

class SagaOrchestrator:
    """
    Executes a saga: runs steps in order. On failure, runs compensations in reverse.
    State is persisted to allow resume on crash (simplified: in-memory here).
    """

    def __init__(self):
        self._sagas: Dict[str, SagaState] = {}

    def execute(self, saga_type: str, steps: List[SagaStep],
                context: Dict) -> SagaState:
        state           = SagaState(saga_type=saga_type, context=dict(context))
        state.status    = SagaStatus.RUNNING
        self._sagas[state.saga_id] = state

        print(f"  [Saga {state.saga_id}] Starting '{saga_type}'")

        completed_steps: List[int] = []

        for i, step in enumerate(steps):
            state.current_step = i
            print(f"  [Saga {state.saga_id}] Step {i+1}/{len(steps)}: {step.name}")
            try:
                output = step.execute(state.context)
                if output:
                    state.context.update(output)
                state.step_results.append(StepResult(step.name, success=True, output=output))
                completed_steps.append(i)
                print(f"    → ✓ {step.name} succeeded")
            except Exception as exc:
                state.step_results.append(StepResult(step.name, success=False, error=str(exc)))
                print(f"    → ✗ {step.name} FAILED: {exc}")
                # Compensate in reverse
                self._compensate(state, steps, completed_steps)
                return state

        state.status       = SagaStatus.COMPLETED
        state.completed_at = time.time()
        print(f"  [Saga {state.saga_id}] ✓ COMPLETED")
        return state

    def _compensate(self, state: SagaState, steps: List[SagaStep],
                    completed: List[int]):
        state.status = SagaStatus.COMPENSATING
        print(f"  [Saga {state.saga_id}] Starting compensation (reverse order)")

        for i in reversed(completed):
            step = steps[i]
            print(f"  [Saga {state.saga_id}] Compensating: {step.name}")
            max_attempts = 3 if step.retry_on_compensate else 1
            for attempt in range(max_attempts):
                try:
                    step.compensate(state.context)
                    print(f"    → ✓ {step.name} compensated")
                    break
                except Exception as exc:
                    if attempt < max_attempts - 1:
                        print(f"    → retry compensation for {step.name}: {exc}")
                        time.sleep(0.005)
                    else:
                        print(f"    → ✗ Compensation FAILED for {step.name}: {exc}")
                        state.status = SagaStatus.FAILED
                        return

        if state.status != SagaStatus.FAILED:
            state.status = SagaStatus.COMPENSATED
        print(f"  [Saga {state.saga_id}] Compensation status: {state.status.value}")

    def get(self, saga_id: str) -> Optional[SagaState]:
        return self._sagas.get(saga_id)


# ─────────────────────────────────────────────
# EXAMPLE: ORDER FULFILLMENT SAGA
# Order → reserve inventory → charge payment → ship order
# ─────────────────────────────────────────────

class InventoryService:
    def __init__(self):
        self._stock        = {"SKU-A": 10, "SKU-B": 2}
        self._reservations : Dict[str, str] = {}   # reservation_id → sku

    def reserve(self, ctx: Dict) -> Dict:
        sku = ctx["sku"]
        qty = ctx.get("qty", 1)
        if self._stock.get(sku, 0) < qty:
            raise Exception(f"Insufficient stock for {sku}: {self._stock.get(sku, 0)} < {qty}")
        reservation_id = str(uuid.uuid4())[:8]
        self._stock[sku] -= qty
        self._reservations[reservation_id] = sku
        return {"reservation_id": reservation_id}

    def release(self, ctx: Dict) -> bool:
        res_id = ctx.get("reservation_id")
        if res_id and res_id in self._reservations:
            sku = self._reservations.pop(res_id)
            self._stock[sku] = self._stock.get(sku, 0) + ctx.get("qty", 1)
        return True


class PaymentService:
    def __init__(self, should_fail: bool = False):
        self.should_fail   = should_fail
        self._charges      : Dict[str, float] = {}

    def charge(self, ctx: Dict) -> Dict:
        if self.should_fail:
            raise Exception("Payment gateway timeout")
        charge_id = str(uuid.uuid4())[:8]
        self._charges[charge_id] = ctx["amount"]
        return {"charge_id": charge_id}

    def refund(self, ctx: Dict) -> bool:
        charge_id = ctx.get("charge_id")
        if charge_id and charge_id in self._charges:
            del self._charges[charge_id]
        return True


class ShippingService:
    def __init__(self):
        self._shipments: Dict[str, Dict] = {}

    def create_shipment(self, ctx: Dict) -> Dict:
        shipment_id = str(uuid.uuid4())[:8]
        self._shipments[shipment_id] = {
            "order_id": ctx["order_id"],
            "address" : ctx["address"],
        }
        return {"shipment_id": shipment_id}

    def cancel_shipment(self, ctx: Dict) -> bool:
        s_id = ctx.get("shipment_id")
        if s_id:
            self._shipments.pop(s_id, None)
        return True


# ─────────────────────────────────────────────
# CHOREOGRAPHY SAGA (event-driven)
# ─────────────────────────────────────────────

class ChoreographySaga:
    """
    Order saga via event choreography.
    Services listen for events and emit results (or failure events).
    """

    def __init__(self):
        self._log : List[str] = []
        self._inventory = InventoryService()
        self._payment   = PaymentService(should_fail=False)

    def on_order_created(self, order_id: str, sku: str, amount: float) -> bool:
        """Saga entry point: receives OrderCreated event."""
        self._log.append(f"order.created({order_id})")
        try:
            result = self._inventory.reserve({"sku": sku, "qty": 1})
            reservation_id = result["reservation_id"]
            self._log.append(f"inventory.reserved({reservation_id})")
            return self.on_inventory_reserved(order_id, reservation_id, amount)
        except Exception as e:
            self._log.append(f"order.failed({order_id}: {e})")
            return False

    def on_inventory_reserved(self, order_id: str, reservation_id: str,
                               amount: float) -> bool:
        try:
            result = self._payment.charge({"amount": amount})
            self._log.append(f"payment.charged({result['charge_id']})")
            self._log.append(f"order.completed({order_id})")
            return True
        except Exception as e:
            self._log.append(f"payment.failed — compensating inventory")
            self._inventory.release({"reservation_id": reservation_id, "qty": 1,
                                     "sku": "SKU-A"})
            self._log.append(f"inventory.released({reservation_id})")
            return False


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_saga():
    print("=" * 65)
    print("SAGA PATTERN FOR DISTRIBUTED TRANSACTIONS")
    print("=" * 65)

    random.seed(42)

    # ── Successful Orchestration Saga ─────────────
    print("\n[1] ORCHESTRATION SAGA — SUCCESS PATH")
    print("─" * 55)

    inventory = InventoryService()
    payment   = PaymentService(should_fail=False)
    shipping  = ShippingService()
    orch      = SagaOrchestrator()

    steps_ok = [
        SagaStep(
            name       = "ReserveInventory",
            execute    = lambda ctx: inventory.reserve(ctx),
            compensate = lambda ctx: inventory.release(ctx),
        ),
        SagaStep(
            name       = "ChargePayment",
            execute    = lambda ctx: payment.charge(ctx),
            compensate = lambda ctx: payment.refund(ctx),
        ),
        SagaStep(
            name       = "CreateShipment",
            execute    = lambda ctx: shipping.create_shipment(ctx),
            compensate = lambda ctx: shipping.cancel_shipment(ctx),
        ),
    ]
    ctx = {"order_id": "ORD-001", "sku": "SKU-A", "qty": 1,
           "amount": 99.0, "address": "123 Main St"}
    state = orch.execute("OrderFulfillment", steps_ok, ctx)
    print(f"\n  Final status : {state.status.value}")
    print(f"  Context keys : {list(state.context.keys())}")

    # ── Failed Orchestration Saga (payment fails) ─
    print("\n\n[2] ORCHESTRATION SAGA — PAYMENT FAILS → COMPENSATE")
    print("─" * 55)

    inventory2 = InventoryService()
    payment2   = PaymentService(should_fail=True)    # will fail
    shipping2  = ShippingService()

    steps_fail = [
        SagaStep(
            name       = "ReserveInventory",
            execute    = lambda ctx: inventory2.reserve(ctx),
            compensate = lambda ctx: inventory2.release(ctx),
        ),
        SagaStep(
            name       = "ChargePayment",
            execute    = lambda ctx: payment2.charge(ctx),
            compensate = lambda ctx: payment2.refund(ctx),
        ),
        SagaStep(
            name       = "CreateShipment",
            execute    = lambda ctx: shipping2.create_shipment(ctx),
            compensate = lambda ctx: shipping2.cancel_shipment(ctx),
        ),
    ]
    ctx2  = {"order_id": "ORD-002", "sku": "SKU-A", "qty": 1,
             "amount": 150.0, "address": "456 Oak Ave"}
    state2 = orch.execute("OrderFulfillment", steps_fail, ctx2)
    print(f"\n  Final status : {state2.status.value}")
    print(f"  Inventory stock for SKU-A restored: "
          f"{inventory2._stock.get('SKU-A')} (expected 10)")

    # ── Out-of-stock failure ──────────────────────
    print("\n\n[3] ORCHESTRATION SAGA — OUT OF STOCK")
    print("─" * 55)
    inventory3 = InventoryService()
    payment3   = PaymentService()
    shipping3  = ShippingService()
    steps3 = [
        SagaStep("ReserveInventory", lambda ctx: inventory3.reserve(ctx),
                 lambda ctx: inventory3.release(ctx)),
        SagaStep("ChargePayment",    lambda ctx: payment3.charge(ctx),
                 lambda ctx: payment3.refund(ctx)),
    ]
    ctx3 = {"order_id": "ORD-003", "sku": "SKU-Z", "qty": 1, "amount": 50.0}
    state3 = orch.execute("OrderFulfillment", steps3, ctx3)
    print(f"\n  Final status : {state3.status.value}")

    # ── Choreography Saga ─────────────────────────
    print("\n\n[4] CHOREOGRAPHY SAGA — EVENT-DRIVEN")
    print("─" * 55)
    choreo = ChoreographySaga()
    result = choreo.on_order_created("ORD-004", "SKU-A", 75.0)
    print(f"  Outcome: {'✓ completed' if result else '✗ failed'}")
    print(f"  Event log:")
    for entry in choreo._log:
        print(f"    → {entry}")

    # ── Comparison ────────────────────────────────
    print("\n\n[5] ORCHESTRATION vs CHOREOGRAPHY")
    print("─" * 55)
    rows = [
        ("Visibility",       "Central — easy to trace in orchestrator", "Distributed — trace via events"),
        ("Coupling",         "Services coupled to orchestrator",        "Services coupled via event schema"),
        ("Adding steps",     "Update orchestrator workflow",            "New service subscribes to event"),
        ("Error handling",   "Explicit in orchestrator",                "Implicit — each service handles"),
        ("Testing",          "Test orchestrator class",                 "End-to-end event trace"),
        ("SPOF",             "Orchestrator is SPOF",                    "No SPOF"),
        ("Best for",         "Complex, sequential workflows",           "Simple, parallel workflows"),
    ]
    print(f"  {'Aspect':<18} {'Orchestration':<40} {'Choreography'}")
    print(f"  {'─'*85}")
    for aspect, orch_val, choreo_val in rows:
        print(f"  {aspect:<18} {orch_val:<40} {choreo_val}")

    print("\n\n[6] SAGA DESIGN RULES")
    print("─" * 55)
    rules = [
        "Compensations must be idempotent — retried on failure",
        "Persist Saga state to survive orchestrator crashes",
        "Each step must be atomic within its own service",
        "Avoid tight timing: steps may run seconds/minutes apart",
        "Some compensations are impossible (email sent) — accept + alert",
        "Use correlation_id to link all events in a saga",
        "Monitor saga duration — stuck sagas need alerting",
    ]
    for rule in rules:
        print(f"  • {rule}")


if __name__ == "__main__":
    demonstrate_saga()
