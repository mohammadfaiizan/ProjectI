"""
SERVICE DECOMPOSITION STRATEGIES
==================================

Problem Statement:
How do you split a monolith into microservices? Cut along the wrong lines and you
create a distributed monolith — all the ops overhead with none of the autonomy.
The key question: what IS a good service boundary?

Domain-Driven Design (DDD) Vocabulary:
  Domain:             The overall problem space (e.g., e-commerce).
  Subdomain:          A part of the domain (Ordering, Payments, Catalog).
    Core subdomain:     Competitive differentiator — build, don't buy.
    Supporting subdomain: Necessary but not differentiating — can outsource.
    Generic subdomain:  Commodity (email, auth) — buy/use SaaS.
  Bounded Context:    The boundary within which a model is consistent and unambiguous.
                      "Order" means something different in Billing vs Fulfillment.
  Ubiquitous Language: A shared vocabulary used by both engineers and domain experts
                       inside a bounded context. No translation needed.

Decomposition Strategies:

  1. By Business Capability:
     Each service implements one business capability (what the business does).
     Capabilities: Order Management, Inventory, Payments, Customer Management.
     Stable — business capabilities change slowly.
     Good starting point when you don't have a deep domain model yet.

  2. By Subdomain (DDD):
     Each service aligns with a bounded context.
     Higher fidelity to the domain model.
     Requires domain experts and event storming sessions.
     Better for complex domains with rich behavior.

  3. By Volatility:
     Group by how frequently things change.
     Frequently-changing code isolated to its own service → faster iteration.

  4. Strangler Fig Pre-Assessment:
     Before migrating, identify which modules are safest to extract first.
     Criteria: clear API boundary, low coupling, independent data, high value.

Cohesion vs Coupling (the litmus test):
  High Cohesion:  Things that change together, stay together.
  Low Coupling:   Services don't know each other's internals.
  If splitting a feature requires changing two services: wrong boundary.

Conway's Law:
  "Organizations design systems that mirror their communication structure."
  Practical inverse: Design team structure to match desired service boundaries.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import uuid


# ─────────────────────────────────────────────
# DDD VOCABULARY TYPES
# ─────────────────────────────────────────────

class SubdomainType(Enum):
    CORE       = "core"       # differentiator — own it
    SUPPORTING = "supporting" # necessary — build simply or outsource
    GENERIC    = "generic"    # commodity — buy/SaaS


@dataclass
class UbiquitousLanguage:
    """Terms within a bounded context. Same word can mean different things
    across contexts — that's intentional, not a bug."""
    context_name : str
    terms        : Dict[str, str]   # term → definition within this context

    def define(self, term: str, definition: str):
        self.terms[term] = definition

    def lookup(self, term: str) -> str:
        return self.terms.get(term, f"[undefined in {self.context_name}]")


@dataclass
class BoundedContext:
    name       : str
    subdomain  : SubdomainType
    language   : UbiquitousLanguage
    owns_data  : List[str]          # tables / collections
    team       : str
    events_published : List[str] = field(default_factory=list)
    events_consumed  : List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# SERVICE CANDIDATE EVALUATION
# ─────────────────────────────────────────────

@dataclass
class ServiceCandidate:
    name              : str
    business_capability: str
    team              : str
    change_freq       : str   # high / medium / low
    scale_independently: bool
    data_isolation    : bool  # True = owns its own data cleanly
    coupling_score    : int   # 1-5; 1=tightly coupled to many, 5=standalone
    team_alignment    : bool  # True = one team owns it entirely

    def extraction_score(self) -> int:
        """Higher = better candidate for microservice extraction."""
        score = 0
        freq_map = {"high": 3, "medium": 2, "low": 1}
        score += freq_map.get(self.change_freq, 0)
        if self.scale_independently : score += 3
        if self.data_isolation      : score += 3
        score += self.coupling_score          # up to 5
        if self.team_alignment      : score += 2
        return score                          # max = 16

    def recommendation(self) -> str:
        s = self.extraction_score()
        if s >= 12 : return "Extract now"
        if s >= 8  : return "Plan extraction"
        if s >= 5  : return "Monitor"
        return "Keep in monolith"


# ─────────────────────────────────────────────
# DECOMPOSITION STRATEGY EVALUATOR
# ─────────────────────────────────────────────

class DecompositionEvaluator:
    """Evaluates two decomposition strategies for a given domain."""

    def __init__(self, domain_name: str):
        self.domain_name       = domain_name
        self.capabilities      : List[Dict] = []    # by-capability decomposition
        self.bounded_contexts  : List[BoundedContext] = []   # by-subdomain

    def add_capability(self, name: str, description: str,
                       team: str, data_stores: List[str]):
        self.capabilities.append({
            "name": name, "description": description,
            "team": team, "data_stores": data_stores,
        })

    def add_bounded_context(self, bc: BoundedContext):
        self.bounded_contexts.append(bc)

    def check_data_overlap(self) -> List[Tuple[str, str, str]]:
        """Find tables claimed by more than one bounded context (bad)."""
        seen : Dict[str, str] = {}
        conflicts = []
        for bc in self.bounded_contexts:
            for table in bc.owns_data:
                if table in seen:
                    conflicts.append((table, seen[table], bc.name))
                else:
                    seen[table] = bc.name
        return conflicts

    def coupling_matrix(self) -> Dict[str, List[str]]:
        """Which contexts consume which events — shows coupling."""
        matrix = {}
        for bc in self.bounded_contexts:
            matrix[bc.name] = bc.events_consumed
        return matrix


# ─────────────────────────────────────────────
# STRANGLER FIG PRE-ASSESSMENT
# ─────────────────────────────────────────────

@dataclass
class MonolithModule:
    name            : str
    lines_of_code   : int
    api_entry_points: int   # clean HTTP/RPC entry points
    db_table_count  : int
    shared_tables   : int   # tables shared with other modules
    test_coverage   : float # 0.0 - 1.0

    def strangler_readiness(self) -> float:
        """
        Score 0-100: higher = easier to extract via Strangler Fig.
        Ideal: few shared tables, clean API boundary, good test coverage.
        """
        sharing_penalty  = self.shared_tables / max(self.db_table_count, 1)
        api_clarity      = min(self.api_entry_points / 5.0, 1.0)
        score = (
            (1 - sharing_penalty) * 40 +   # low sharing = good
            api_clarity          * 30 +    # clear entry points = good
            self.test_coverage   * 30      # tests = safe to migrate
        )
        return round(score, 1)

    def extraction_order_label(self) -> str:
        s = self.strangler_readiness()
        if s >= 70 : return "Extract first"
        if s >= 45 : return "Extract second"
        if s >= 25 : return "Extract last"
        return "Leave in monolith"


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_service_decomposition():
    print("=" * 65)
    print("SERVICE DECOMPOSITION STRATEGIES")
    print("=" * 65)

    # ── 1. DDD Bounded Contexts ───────────────────
    print("\n[1] DDD BOUNDED CONTEXTS — UBIQUITOUS LANGUAGE")
    print("─" * 55)

    order_lang = UbiquitousLanguage("ordering", {})
    order_lang.define("Order",    "A customer request to purchase items, with lifecycle: PENDING→CONFIRMED→SHIPPED")
    order_lang.define("Customer", "The entity placing the order; identified by customer_id")
    order_lang.define("Product",  "A line item within the order (just SKU + qty, no details)")

    billing_lang = UbiquitousLanguage("billing", {})
    billing_lang.define("Order",    "A payment obligation; triggers an Invoice")
    billing_lang.define("Customer", "The payer; has payment methods and billing address")
    billing_lang.define("Product",  "Not used in billing context; replaced by LineItem with price")

    print("  'Order' means different things in different contexts:")
    print(f"  [Ordering]  Order = {order_lang.lookup('Order')[:60]}...")
    print(f"  [Billing]   Order = {billing_lang.lookup('Order')[:60]}...")
    print(f"\n  'Customer' differs too:")
    print(f"  [Ordering]  Customer = {order_lang.lookup('Customer')}")
    print(f"  [Billing]   Customer = {billing_lang.lookup('Customer')}")
    print(f"\n  → Each context owns its own model. No shared 'Order' class.")

    # ── 2. Decomposition by Capability ───────────
    print("\n\n[2] DECOMPOSITION BY BUSINESS CAPABILITY")
    print("─" * 55)

    evaluator = DecompositionEvaluator("E-Commerce Platform")
    capabilities = [
        ("order-service",    "Manage order lifecycle",           "orders-team",  ["orders","order_items"]),
        ("catalog-service",  "Product catalog and search",       "catalog-team", ["products","categories"]),
        ("inventory-service","Stock levels and reservations",    "wh-team",      ["inventory","reservations"]),
        ("payment-service",  "Process and record payments",      "pay-team",     ["payments","refunds"]),
        ("user-service",     "User accounts and authentication", "iam-team",     ["users","sessions"]),
        ("notification-svc", "Email, SMS, push notifications",   "platform-team",["notifications"]),
    ]
    for name, desc, team, data in capabilities:
        evaluator.add_capability(name, desc, team, data)

    print(f"  {'Service':<22} {'Team':<15} {'Data Stores'}")
    print(f"  {'─'*60}")
    for cap in evaluator.capabilities:
        print(f"  {cap['name']:<22} {cap['team']:<15} {', '.join(cap['data_stores'])}")
    print(f"\n  Total services: {len(evaluator.capabilities)}")
    print(f"  Each team deploys independently. No shared tables.")

    # ── 3. Decomposition by Subdomain ─────────────
    print("\n\n[3] DECOMPOSITION BY SUBDOMAIN (DDD)")
    print("─" * 55)

    contexts = [
        BoundedContext("ordering",     SubdomainType.CORE,
                       order_lang,    ["orders","order_items"],
                       "orders-team",
                       events_published=["OrderPlaced","OrderCancelled"],
                       events_consumed=["PaymentConfirmed","StockReserved"]),
        BoundedContext("payments",     SubdomainType.CORE,
                       billing_lang,  ["payments","refunds","invoices"],
                       "payments-team",
                       events_published=["PaymentConfirmed","PaymentFailed"],
                       events_consumed=["OrderPlaced"]),
        BoundedContext("inventory",    SubdomainType.SUPPORTING,
                       UbiquitousLanguage("inventory",{}), ["inventory","reservations"],
                       "warehouse-team",
                       events_published=["StockReserved","StockDepleted"],
                       events_consumed=["OrderPlaced","OrderCancelled"]),
        BoundedContext("notifications",SubdomainType.GENERIC,
                       UbiquitousLanguage("notifications",{}), ["notification_log"],
                       "platform-team",
                       events_published=[],
                       events_consumed=["OrderPlaced","PaymentConfirmed","OrderShipped"]),
    ]
    for bc in contexts:
        evaluator.add_bounded_context(bc)

    print(f"  {'Context':<18} {'Type':<12} {'Team':<16} {'Publishes'}")
    print(f"  {'─'*65}")
    for bc in evaluator.bounded_contexts:
        pub = ", ".join(bc.events_published) if bc.events_published else "—"
        print(f"  {bc.name:<18} {bc.subdomain.value:<12} {bc.team:<16} {pub}")

    conflicts = evaluator.check_data_overlap()
    if conflicts:
        print(f"\n  DATA CONFLICTS DETECTED:")
        for table, owner1, owner2 in conflicts:
            print(f"    '{table}' claimed by both '{owner1}' and '{owner2}' — FIX THIS")
    else:
        print(f"\n  No data ownership conflicts detected.")

    # ── 4. Candidate Scoring ──────────────────────
    print("\n\n[4] SERVICE EXTRACTION SCORING")
    print("─" * 55)

    candidates = [
        ServiceCandidate("payments",    "Process payments",   "pay-team",    "low",    True,  True,  5, True),
        ServiceCandidate("orders",      "Order lifecycle",    "orders-team", "high",   True,  True,  4, True),
        ServiceCandidate("inventory",   "Stock management",   "wh-team",     "medium", True,  True,  4, True),
        ServiceCandidate("email",       "Send emails",        "platform",    "low",    False, False, 2, False),
        ServiceCandidate("reporting",   "Reports/analytics",  "analytics",   "low",    False, False, 1, False),
        ServiceCandidate("user-prefs",  "User preferences",   "ux-team",     "medium", False, True,  3, True),
    ]

    print(f"  {'Service':<14} {'Score':<8} {'Recommendation'}")
    print(f"  {'─'*45}")
    for c in sorted(candidates, key=lambda x: -x.extraction_score()):
        print(f"  {c.name:<14} {c.extraction_score():<8} {c.recommendation()}")

    # ── 5. Strangler Fig Readiness ─────────────────
    print("\n\n[5] STRANGLER FIG PRE-ASSESSMENT")
    print("─" * 55)
    print("  Which monolith modules are safest to extract first?")
    print()

    modules = [
        MonolithModule("PaymentModule",      3200, 8, 4, 0, 0.82),
        MonolithModule("OrderModule",        8500, 12, 7, 1, 0.71),
        MonolithModule("ReportingModule",    5100, 3, 9, 6, 0.30),
        MonolithModule("NotificationModule", 1200, 4, 2, 0, 0.90),
        MonolithModule("UserModule",         4300, 6, 5, 2, 0.65),
        MonolithModule("InventoryModule",    2800, 7, 4, 1, 0.75),
    ]

    modules.sort(key=lambda m: -m.strangler_readiness())
    print(f"  {'Module':<24} {'Readiness':<12} {'Shared Tables':<15} {'Coverage':<10} {'Action'}")
    print(f"  {'─'*72}")
    for m in modules:
        print(f"  {m.name:<24} {m.strangler_readiness():<12} "
              f"{m.shared_tables}/{m.db_table_count} tables   "
              f"{m.test_coverage:.0%}       {m.extraction_order_label()}")

    # ── 6. Anti-patterns ──────────────────────────
    print("\n\n[6] DECOMPOSITION ANTI-PATTERNS")
    print("─" * 55)
    anti_patterns = [
        ("Nanoservices",         "Services too small; one function per service → huge overhead"),
        ("Chatty services",      "Service A calls B 10x per request → latency, coupling"),
        ("Shared database",      "Two services read/write same table → tight coupling"),
        ("Distributed monolith", "Microservices deployed separately but coupled in behavior"),
        ("Wrong boundaries",     "Split by technical layer (all DAOs in one svc) not domain"),
    ]
    for name, desc in anti_patterns:
        print(f"  [AVOID] {name}")
        print(f"          {desc}")


if __name__ == "__main__":
    demonstrate_service_decomposition()
