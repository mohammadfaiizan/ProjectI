"""
MONOLITH VS MICROSERVICES
==========================

Problem Statement:
When designing a system, one of the first architectural decisions is whether
to build a monolith (single deployable unit) or microservices (independently
deployable services). Each has real trade-offs that must be understood before
choosing.

Architecture Diagrams:

  MONOLITH:
  ┌──────────────────────────────────┐
  │         E-Commerce App           │
  │  ┌──────┐ ┌───────┐ ┌────────┐  │
  │  │Users │ │Orders │ │Payment │  │
  │  └──────┘ └───────┘ └────────┘  │
  │       Shared Database            │
  └──────────────────────────────────┘

  MICROSERVICES:
  [User-Svc]──┐
  [Order-Svc]─┼──► [API Gateway] ──► Client
  [Pay-Svc]───┘
     │ each has its own DB

Key Concepts:
- Monolith   : Simple, fast initially; hard to scale individual components
- Microservice: Independent deploy/scale; complex networking and ops
- Bounded Context (DDD): Each service owns its domain data
- Strangler Fig: Incremental migration from monolith to microservices
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class ArchStyle(Enum):
    MONOLITH      = "monolith"
    MODULAR_MONO  = "modular_monolith"
    MICROSERVICES = "microservices"


class CommunicationType(Enum):
    IN_PROCESS = "in-process function call"
    HTTP_REST  = "HTTP REST"
    GRPC       = "gRPC"
    MESSAGE    = "async message queue"


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class Module:
    name       : str
    loc        : int          # lines of code
    team_owner : str
    dependencies: List[str] = field(default_factory=list)


@dataclass
class Microservice:
    name             : str
    responsibility   : str
    database         : str
    team_owner       : str
    language         : str = "Python"
    comm_type        : CommunicationType = CommunicationType.HTTP_REST
    replicas         : int = 2


# ─────────────────────────────────────────────
# MONOLITH
# ─────────────────────────────────────────────

class MonolithApp:
    """
    A monolithic application that bundles all modules into one deployable.
    """

    def __init__(self, name: str, database: str):
        self.name     = name
        self.database = database
        self.modules : List[Module] = []

    def add_module(self, module: Module):
        self.modules.append(module)

    def total_loc(self) -> int:
        return sum(m.loc for m in self.modules)

    def coupling_matrix(self) -> Dict[str, List[str]]:
        """Shows which modules depend on which."""
        return {m.name: m.dependencies for m in self.modules}

    def deploy(self):
        print(f"\n  [DEPLOY] {self.name} (Monolith)")
        print(f"  Deploying ALL {len(self.modules)} modules in one unit…")
        print(f"  Total LOC: {self.total_loc():,}")
        print(f"  Shared DB: {self.database}")
        print(f"  ⚠  Any change requires full redeploy")

    def analyze_coupling(self):
        matrix = self.coupling_matrix()
        print(f"\n  Coupling analysis for {self.name}:")
        for mod, deps in matrix.items():
            dep_str = ", ".join(deps) if deps else "none"
            icon = "⚠ " if len(deps) > 2 else "✅"
            print(f"    {icon} {mod:<18} depends on: {dep_str}")

    def print_pros_cons(self):
        print("\n  MONOLITH:")
        pros = ["Simple to develop initially", "No network latency between modules",
                "Easy to test end-to-end", "Single deployment pipeline"]
        cons = ["Hard to scale individual bottlenecks", "One bug can crash everything",
                "Tech stack locked for all modules", "Large codebase slows dev over time",
                "Full redeploy for every small change"]
        for p in pros: print(f"    ✅ {p}")
        for c in cons: print(f"    ❌ {c}")


# ─────────────────────────────────────────────
# MICROSERVICES
# ─────────────────────────────────────────────

class ServiceRegistry:
    """Service discovery: maps service name → endpoint."""

    def __init__(self):
        self._registry: Dict[str, str] = {}

    def register(self, service_name: str, endpoint: str):
        self._registry[service_name] = endpoint
        print(f"  ✅ Registered: {service_name} → {endpoint}")

    def discover(self, service_name: str) -> Optional[str]:
        return self._registry.get(service_name)

    def list_all(self):
        print("\n  Service Registry:")
        for name, ep in self._registry.items():
            print(f"    {name:<20} → {ep}")


class MicroservicesApp:
    """
    A microservices application: independently deployable services.
    """

    def __init__(self, name: str):
        self.name     = name
        self.services : List[Microservice] = []
        self.registry = ServiceRegistry()

    def add_service(self, svc: Microservice):
        self.services.append(svc)
        endpoint = f"http://{svc.name.lower().replace(' ', '-')}.internal:8080"
        self.registry.register(svc.name, endpoint)

    def deploy_service(self, service_name: str):
        svc = next((s for s in self.services if s.name == service_name), None)
        if svc is None:
            print(f"  ❌ Service '{service_name}' not found")
            return
        print(f"\n  [DEPLOY] {service_name} only (no other services affected)")
        print(f"  Language : {svc.language}")
        print(f"  Database : {svc.database}")
        print(f"  Replicas : {svc.replicas}")
        print(f"  Comm     : {svc.comm_type.value}")
        print(f"  ✅ Deploy in ~30s; zero downtime with rolling update")

    def scale_service(self, service_name: str, new_replicas: int):
        svc = next((s for s in self.services if s.name == service_name), None)
        if svc:
            old = svc.replicas
            svc.replicas = new_replicas
            print(f"  [SCALE] {service_name}: {old} → {new_replicas} replicas (others unchanged)")

    def print_architecture(self):
        print(f"\n  MICROSERVICES ARCHITECTURE: {self.name}")
        print(f"  {'Service':<20} {'DB':<20} {'Lang':<10} {'Comm':<20} {'Replicas'}")
        print(f"  {'─'*80}")
        for svc in self.services:
            print(f"  {svc.name:<20} {svc.database:<20} {svc.language:<10} "
                  f"{svc.comm_type.value:<20} {svc.replicas}")

    def print_pros_cons(self):
        print("\n  MICROSERVICES:")
        pros = ["Scale individual services independently", "Independent deployments",
                "Team autonomy (own language, DB, deploy)", "Fault isolation",
                "Technology diversity per service"]
        cons = ["Network latency between services", "Distributed tracing complexity",
                "Data consistency across services (no shared DB)", "Operational overhead (k8s, service mesh)",
                "Harder to test end-to-end"]
        for p in pros: print(f"    ✅ {p}")
        for c in cons: print(f"    ❌ {c}")


class MigrationAnalyzer:
    """Analyses a monolith and suggests decomposition candidates."""

    def __init__(self, monolith: MonolithApp):
        self.monolith = monolith

    def suggest_decomposition(self) -> List[str]:
        candidates = []
        for module in self.monolith.modules:
            if len(module.dependencies) <= 1:
                candidates.append(module.name)
        return candidates

    def report(self):
        candidates = self.suggest_decomposition()
        print(f"\n  Migration Analysis for '{self.monolith.name}':")
        print(f"  Modules with low coupling (safe to extract first):")
        for c in candidates:
            print(f"    ✅ {c} → good microservice candidate")
        tight = [m.name for m in self.monolith.modules if len(m.dependencies) > 2]
        if tight:
            print(f"  Tightly coupled (extract last or keep in monolith):")
            for t in tight:
                print(f"    ⚠  {t}")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_monolith_vs_microservices():
    print("=" * 65)
    print("MONOLITH VS MICROSERVICES: E-Commerce Platform")
    print("=" * 65)

    # ── Monolith ──────────────────────────────
    print("\n── MONOLITH APPROACH ──────────────────────────")
    mono = MonolithApp("ShopApp", "PostgreSQL (shared)")
    mono.add_module(Module("UserModule",     5_000,  "team-auth",     []))
    mono.add_module(Module("ProductModule",  8_000,  "team-catalog",  ["UserModule"]))
    mono.add_module(Module("OrderModule",    12_000, "team-orders",   ["UserModule", "ProductModule", "PaymentModule"]))
    mono.add_module(Module("PaymentModule",  6_000,  "team-payments", ["UserModule", "OrderModule"]))
    mono.add_module(Module("NotifyModule",   3_000,  "team-comms",    ["UserModule"]))
    mono.add_module(Module("SearchModule",   4_000,  "team-catalog",  ["ProductModule"]))

    mono.deploy()
    mono.analyze_coupling()
    mono.print_pros_cons()

    # ── Migration Analysis ────────────────────
    print("\n── MIGRATION ANALYSIS ──────────────────────────")
    analyzer = MigrationAnalyzer(mono)
    analyzer.report()

    # ── Microservices ─────────────────────────
    print("\n── MICROSERVICES APPROACH ──────────────────────")
    msa = MicroservicesApp("ShopApp MSA")
    msa.add_service(Microservice("user-service",    "User accounts & auth",    "PostgreSQL",  "team-auth",     "Go",      CommunicationType.GRPC,    3))
    msa.add_service(Microservice("product-service", "Catalog & inventory",     "PostgreSQL",  "team-catalog",  "Python",  CommunicationType.HTTP_REST,3))
    msa.add_service(Microservice("order-service",   "Order lifecycle",         "PostgreSQL",  "team-orders",   "Java",    CommunicationType.HTTP_REST,4))
    msa.add_service(Microservice("payment-service", "Payments & refunds",      "PostgreSQL",  "team-payments", "Java",    CommunicationType.GRPC,    2))
    msa.add_service(Microservice("notify-service",  "Email/SMS/push",          "Redis+SQS",   "team-comms",    "Node.js", CommunicationType.MESSAGE,  2))
    msa.add_service(Microservice("search-service",  "Full-text product search","Elasticsearch","team-catalog",  "Python",  CommunicationType.HTTP_REST,3))

    msa.print_architecture()
    msa.print_pros_cons()

    # ── Targeted operations ───────────────────
    print("\n── MICROSERVICES IN ACTION ─────────────────────")
    print("\nScenario: Only payment-service has high load on Black Friday")
    msa.scale_service("payment-service", 20)

    print("\nScenario: Deploy a fix to notify-service only")
    msa.deploy_service("notify-service")

    # ── Decision Guide ────────────────────────
    print("\n── WHEN TO CHOOSE WHAT ─────────────────────────")
    print("  Choose Monolith when:")
    print("    • Early-stage startup (move fast, prove product)")
    print("    • Team < 10 engineers")
    print("    • Domain is not well-understood yet")
    print("    • Low operational maturity")
    print("\n  Choose Microservices when:")
    print("    • Teams > 50 engineers need autonomy")
    print("    • Clear bounded contexts exist")
    print("    • Services have very different scaling needs")
    print("    • You have good observability (tracing, logging, metrics)")


if __name__ == "__main__":
    demonstrate_monolith_vs_microservices()
