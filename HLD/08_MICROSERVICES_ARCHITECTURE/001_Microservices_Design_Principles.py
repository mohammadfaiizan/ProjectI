"""
MICROSERVICES DESIGN PRINCIPLES
==================================

Problem Statement:
Monolithic applications become hard to maintain, scale, and deploy as they grow.
Microservices decompose the system into small, independently deployable services.
But: microservices introduce distributed systems complexity. Don't decompose prematurely.

Core Principles:

  1. Single Responsibility / Bounded Context:
     Each service owns one business capability (Order management, User profiles).
     Clear boundaries. Service doesn't bleed into another's domain.
     Model after Conway's Law: system structure mirrors team structure.

  2. Loose Coupling:
     Services don't share databases. Each owns its data.
     Communication via well-defined APIs (REST, gRPC, events).
     Change service internals without affecting consumers.

  3. High Cohesion:
     Related functionality grouped together within a service.
     User address + User profile → same service (not split).

  4. Service Autonomy:
     Service can be deployed, scaled, and updated independently.
     No shared deployment pipeline blocking. No shared database.

  5. Resilience / Failure Isolation:
     One service failing doesn't cascade to bring down others.
     Circuit breakers, bulkheads, timeouts, fallbacks.

  6. Observable:
     Every service emits metrics, logs with correlation IDs, distributed traces.
     Understand what's happening without connecting to the server.

  7. Designed for Failure:
     Network calls fail. Dependencies are unreliable.
     Every service assumes its dependencies will be slow/unavailable.
     Design for partial availability, graceful degradation.

When NOT to Use Microservices:
  - Small team (<5 engineers): operational overhead outweighs benefits.
  - Simple domain: few natural boundaries.
  - Early-stage product: premature decomposition kills velocity.
  - Start with a modular monolith, extract services when clear boundaries emerge.

Microservices vs Modular Monolith:
  Monolith:     One deployable unit. Shared DB. Simple ops. Poor isolation.
  Mod Monolith: One deployable, clear internal modules, shared DB. Middle ground.
  Microservices: Independent deployables. Each with own DB. High ops overhead.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum
import time
import uuid
import threading
import random


# ─────────────────────────────────────────────
# SERVICE REGISTRY (simplified)
# ─────────────────────────────────────────────

@dataclass
class ServiceCapability:
    name        : str
    version     : str
    endpoints   : List[str]
    owns_tables : List[str]
    team        : str


# ─────────────────────────────────────────────
# BOUNDED CONTEXT VALIDATOR
# ─────────────────────────────────────────────

class BoundedContextViolation(Exception):
    pass


class BoundedContext:
    """
    Enforces that a service only accesses its own data.
    Raises violation if service tries to read another service's tables.
    """

    def __init__(self):
        self._ownership: Dict[str, str] = {}   # table → service

    def register(self, service: str, tables: List[str]):
        for table in tables:
            if table in self._ownership and self._ownership[table] != service:
                raise BoundedContextViolation(
                    f"Table '{table}' already owned by '{self._ownership[table]}'. "
                    f"Service '{service}' cannot claim it.")
            self._ownership[table] = service

    def check_access(self, service: str, table: str):
        owner = self._ownership.get(table)
        if owner and owner != service:
            raise BoundedContextViolation(
                f"Service '{service}' attempted to access table '{table}' "
                f"owned by '{owner}'. Access denied (bounded context violation).")

    def table_owner(self, table: str) -> Optional[str]:
        return self._ownership.get(table)


# ─────────────────────────────────────────────
# SERVICE SKELETON
# ─────────────────────────────────────────────

class MicroService:
    """
    Base class for a microservice.
    Demonstrates: own data store, health check, metrics, correlation ID propagation.
    """

    def __init__(self, service_name: str, bounded_context: BoundedContext,
                 owned_tables: List[str]):
        self.service_name   = service_name
        self.context        = bounded_context
        self.owned_tables   = owned_tables
        self._store         : Dict[str, Any] = {}
        self._request_count = 0
        self._error_count   = 0
        self._latencies     : List[float] = []

        bounded_context.register(service_name, owned_tables)

    def _store_get(self, table: str, key: str, correlation_id: str) -> Optional[Any]:
        self.context.check_access(self.service_name, table)
        return self._store.get(f"{table}:{key}")

    def _store_put(self, table: str, key: str, value: Any, correlation_id: str):
        self.context.check_access(self.service_name, table)
        self._store[f"{table}:{key}"] = value

    def health_check(self) -> Dict[str, Any]:
        return {
            "service": self.service_name,
            "status" : "healthy",
            "uptime_s": 0,
        }

    def metrics(self) -> Dict[str, float]:
        p99 = sorted(self._latencies)[int(len(self._latencies) * 0.99)] \
              if len(self._latencies) > 1 else 0.0
        return {
            "requests"   : self._request_count,
            "errors"     : self._error_count,
            "error_rate" : self._error_count / max(self._request_count, 1),
            "p99_ms"     : p99,
        }

    def _record(self, latency_ms: float, error: bool = False):
        self._request_count += 1
        self._latencies.append(latency_ms)
        if error:
            self._error_count += 1


# ─────────────────────────────────────────────
# EXAMPLE SERVICES
# ─────────────────────────────────────────────

class OrderService(MicroService):
    def __init__(self, context: BoundedContext):
        super().__init__("order-service", context, ["orders", "order_items"])

    def create_order(self, customer_id: str, items: List[Dict],
                     correlation_id: str) -> Dict:
        t0 = time.time()
        order_id = str(uuid.uuid4())[:8]
        order    = {"order_id": order_id, "customer_id": customer_id,
                    "items": items, "status": "pending"}
        self._store_put("orders", order_id, order, correlation_id)
        self._record((time.time() - t0) * 1000)
        return order

    def get_order(self, order_id: str, correlation_id: str) -> Optional[Dict]:
        t0 = time.time()
        result = self._store_get("orders", order_id, correlation_id)
        self._record((time.time() - t0) * 1000)
        return result


class UserService(MicroService):
    def __init__(self, context: BoundedContext):
        super().__init__("user-service", context, ["users", "user_addresses"])

    def create_user(self, email: str, name: str, correlation_id: str) -> Dict:
        user_id = str(uuid.uuid4())[:8]
        user    = {"user_id": user_id, "email": email, "name": name}
        self._store_put("users", user_id, user, correlation_id)
        return user

    def get_user(self, user_id: str, correlation_id: str) -> Optional[Dict]:
        return self._store_get("users", user_id, correlation_id)


class InventoryService(MicroService):
    def __init__(self, context: BoundedContext):
        super().__init__("inventory-service", context, ["inventory", "reservations"])

    def reserve(self, sku: str, qty: int, correlation_id: str) -> bool:
        t0      = time.time()
        key     = f"inventory:{sku}"
        stock   = self._store.get(key, 10)   # default 10 units
        if stock < qty:
            self._record((time.time() - t0) * 1000, error=True)
            return False
        self._store[key] = stock - qty
        self._record((time.time() - t0) * 1000)
        return True


# ─────────────────────────────────────────────
# ANTI-PATTERN: SHARED DATABASE
# ─────────────────────────────────────────────

def demonstrate_bounded_context_violation(context: BoundedContext):
    """Shows what happens when service crosses bounded context."""
    try:
        # Inventory service tries to read orders table (belongs to order-service)
        inv_service = InventoryService(context)
        inv_service.context.check_access("inventory-service", "orders")
        return False
    except BoundedContextViolation as e:
        return str(e)


# ─────────────────────────────────────────────
# DECOMPOSITION ASSESSMENT
# ─────────────────────────────────────────────

@dataclass
class ServiceCandidate:
    name          : str
    business_cap  : str
    team          : str
    change_freq   : str   # high/medium/low
    scale_needs   : str   # independent/shared
    data_isolation: str   # strict/shared
    score         : int   = 0

    def evaluate(self) -> int:
        """Score: higher = better candidate for extraction."""
        score = 0
        if self.change_freq  == "high"       : score += 3
        if self.scale_needs  == "independent": score += 3
        if self.data_isolation == "strict"   : score += 2
        return score


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_microservices_principles():
    print("=" * 65)
    print("MICROSERVICES DESIGN PRINCIPLES")
    print("=" * 65)

    context = BoundedContext()

    # ── Service Creation ──────────────────────────
    print("\n[1] BOUNDED CONTEXT — EACH SERVICE OWNS ITS DATA")
    print("─" * 55)

    order_svc = OrderService(context)
    user_svc  = UserService(context)
    inv_svc   = InventoryService(context)

    corr_id = str(uuid.uuid4())[:8]
    user    = user_svc.create_user("alice@example.com", "Alice", corr_id)
    order   = order_svc.create_order(user["user_id"],
                                      [{"sku": "A1", "qty": 2}], corr_id)
    reserved = inv_svc.reserve("A1", 2, corr_id)

    print(f"  User created: {user}")
    print(f"  Order created: {order['order_id']} status={order['status']}")
    print(f"  Inventory reserved: {reserved}")
    print(f"\n  Data ownership:")
    for table, owner in context._ownership.items():
        print(f"    '{table}' → {owner}")

    # ── Bounded Context Violation ─────────────────
    print("\n\n[2] BOUNDED CONTEXT VIOLATION — ACCESS DENIED")
    print("─" * 55)

    violation = demonstrate_bounded_context_violation(context)
    if violation:
        print(f"  Caught violation:\n    {violation[:80]}...")
    print(f"  → Services must call APIs to access another service's data")

    # ── Service Metrics ───────────────────────────
    print("\n\n[3] SERVICE METRICS — OBSERVABILITY")
    print("─" * 55)

    for svc in [order_svc, user_svc, inv_svc]:
        m = svc.metrics()
        print(f"  {svc.service_name}: requests={m['requests']} "
              f"errors={m['errors']} error_rate={m['error_rate']:.1%}")

    # ── Decomposition Scoring ─────────────────────
    print("\n\n[4] SERVICE DECOMPOSITION ASSESSMENT")
    print("─" * 55)

    candidates = [
        ServiceCandidate("payment",   "Process payments",   "payments-team",
                         "low", "independent", "strict"),
        ServiceCandidate("inventory", "Stock management",   "catalog-team",
                         "medium", "independent", "strict"),
        ServiceCandidate("email",     "Send emails",        "platform-team",
                         "low", "shared", "shared"),
        ServiceCandidate("orders",    "Order lifecycle",    "orders-team",
                         "high", "independent", "strict"),
        ServiceCandidate("reporting", "Generate reports",   "analytics-team",
                         "low", "shared", "shared"),
    ]

    print(f"  {'Service':<14} {'Change Freq':<13} {'Scale':<15} "
          f"{'Data Iso':<12} {'Score':<8} {'Extract?'}")
    print(f"  {'─'*72}")
    for c in sorted(candidates, key=lambda x: -x.evaluate()):
        score = c.evaluate()
        extract = "✓ Yes" if score >= 5 else ("Consider" if score >= 3 else "✗ No")
        print(f"  {c.name:<14} {c.change_freq:<13} {c.scale_needs:<15} "
              f"{c.data_isolation:<12} {score:<8} {extract}")

    # ── Principles Summary ────────────────────────
    print("\n\n[5] MICROSERVICES DESIGN PRINCIPLES")
    print("─" * 55)
    principles = [
        ("Single Responsibility", "One business capability per service"),
        ("Loose Coupling",        "No shared DB; APIs only; async events"),
        ("High Cohesion",         "Related code/data in same service"),
        ("Autonomy",              "Independent deploy, scale, update"),
        ("Resilience",            "Circuit breakers, fallbacks, timeouts"),
        ("Observability",         "Metrics, logs, traces per service"),
        ("Designed for failure",  "Assume all calls can fail"),
        ("API first",             "Define contract before implementation"),
    ]
    for principle, description in principles:
        print(f"  {principle:<24} {description}")

    # ── When to Use ───────────────────────────────
    print("\n\n[6] MONOLITH vs MOD MONOLITH vs MICROSERVICES")
    print("─" * 55)
    rows = [
        ("Monolith",        "1-5 devs, early stage, fast iteration",
         "Single deploy, hard to scale independently"),
        ("Mod Monolith",    "5-20 devs, mature domain, moderate scale",
         "Internal modules, shared DB, simpler than MS"),
        ("Microservices",   "20+ devs, multiple teams, high scale",
         "Full isolation, independent deploys, high ops cost"),
    ]
    for arch, when, tradeoff in rows:
        print(f"  {arch:<18}")
        print(f"    When: {when}")
        print(f"    Tradeoff: {tradeoff}")


if __name__ == "__main__":
    demonstrate_microservices_principles()
