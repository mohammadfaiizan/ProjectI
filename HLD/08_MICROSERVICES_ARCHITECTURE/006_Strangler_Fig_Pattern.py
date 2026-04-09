"""
STRANGLER FIG PATTERN
=======================

Problem Statement:
You can't rewrite a working monolith overnight. Big-bang rewrites fail.
The strangler fig pattern migrates functionality from a monolith to
microservices incrementally, feature by feature, with zero downtime.

Named after the strangler fig tree: it grows around its host tree,
gradually replacing it. The host tree (monolith) can eventually be removed.

How It Works:
  1. A proxy (strangler proxy) sits in front of ALL requests.
  2. For each feature/route, the proxy decides:
       → Send to MONOLITH (original code, default at start)
       → Send to NEW MICROSERVICE (migrated code)
  3. Migration is controlled by feature flags (0% → 10% → 50% → 100%).
  4. At 100% and stable, the monolith code path for that feature is deleted.
  5. When all features migrate, the monolith is dead.

Migration Phases for a Single Feature:
  Phase 0 (Shadow):  New service runs but results are discarded.
                     Compare outputs; find bugs without affecting users.
  Phase 1 (Canary):  10% of traffic → new service. Monitor errors.
  Phase 2 (Expand):  50% → new service. Monitor.
  Phase 3 (Full):    100% → new service.
  Phase 4 (Cleanup): Delete monolith code for this feature.

Risk Mitigation:
  - Rollback by setting flag back to 0% (monolith handles all).
  - Shadow mode: test new service without user impact.
  - Per-feature flags: each feature migrates independently.

What Makes a Good First Feature to Migrate:
  - Clear API boundary (HTTP endpoints).
  - Own its data cleanly (no shared tables).
  - Good test coverage in the monolith (can verify parity).
  - Low traffic volume initially (easier to monitor).

Strangler vs Branch by Abstraction:
  Strangler:  Network-level proxy; language-agnostic.
  Branch by Abstraction: In-process interface; same codebase, swap impl.
  Use Strangler when migrating to separate services/processes.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import time
import random
import uuid
import threading


# ─────────────────────────────────────────────
# MIGRATION STATE
# ─────────────────────────────────────────────

class MigrationPhase(Enum):
    MONOLITH  = "monolith"    # 100% monolith
    SHADOW    = "shadow"      # new service runs, result discarded
    CANARY    = "canary"      # X% to new service
    FULL      = "full"        # 100% new service
    COMPLETE  = "complete"    # monolith code deleted


@dataclass
class FeatureMigration:
    feature_name      : str
    phase             : MigrationPhase = MigrationPhase.MONOLITH
    traffic_percentage: float = 0.0      # % going to new service
    rollout_history   : List[Dict] = field(default_factory=list)
    shadow_mismatches : int = 0
    shadow_comparisons: int = 0

    def set_traffic(self, pct: float, actor: str = "ops"):
        old_pct   = self.traffic_percentage
        self.traffic_percentage = max(0.0, min(100.0, pct))
        if pct <= 0:
            self.phase = MigrationPhase.MONOLITH
        elif pct < 100:
            self.phase = MigrationPhase.CANARY
        else:
            self.phase = MigrationPhase.FULL
        self.rollout_history.append({
            "actor" : actor,
            "from"  : old_pct,
            "to"    : self.traffic_percentage,
            "phase" : self.phase.value,
            "ts"    : time.time(),
        })

    def enable_shadow(self):
        self.phase = MigrationPhase.SHADOW
        self.rollout_history.append({"phase": "shadow", "ts": time.time()})

    def routes_to_new(self, request_id: str) -> bool:
        if self.phase == MigrationPhase.FULL:
            return True
        if self.phase == MigrationPhase.MONOLITH:
            return False
        if self.phase == MigrationPhase.COMPLETE:
            return True
        # Canary: deterministic per request_id for consistency
        hash_val = int(request_id.replace("-", ""), 16) % 100
        return hash_val < self.traffic_percentage


# ─────────────────────────────────────────────
# MONOLITH AND MICROSERVICE STUBS
# ─────────────────────────────────────────────

class MonolithHandler:
    """Legacy monolith code. Reliable but slow and hard to change."""

    def __init__(self):
        self._call_count = 0

    def handle(self, route: str, payload: Dict) -> Dict:
        self._call_count += 1
        time.sleep(0.015)   # 15ms — monolith is slower
        return {
            "source"   : "monolith",
            "route"    : route,
            "data"     : f"[monolith] result for {route}",
            "latency"  : "15ms",
            "call_num" : self._call_count,
        }


class NewMicroservice:
    """New microservice. Faster but initially untested in prod."""

    def __init__(self, name: str, error_rate: float = 0.0):
        self.name        = name
        self.error_rate  = error_rate
        self._call_count = 0

    def handle(self, route: str, payload: Dict) -> Dict:
        self._call_count += 1
        time.sleep(0.006)   # 6ms — microservice is faster
        if random.random() < self.error_rate:
            raise RuntimeError(f"{self.name} returned 500")
        return {
            "source"   : self.name,
            "route"    : route,
            "data"     : f"[{self.name}] result for {route}",
            "latency"  : "6ms",
            "call_num" : self._call_count,
        }


# ─────────────────────────────────────────────
# STRANGLER FIG PROXY
# ─────────────────────────────────────────────

class StranglerFigProxy:
    """
    Intercepts all requests. Routes to monolith or new service
    based on per-feature migration state.
    """

    def __init__(self, monolith: MonolithHandler):
        self.monolith         = monolith
        self._migrations      : Dict[str, FeatureMigration] = {}
        self._services        : Dict[str, NewMicroservice]  = {}
        self._request_log     : List[Dict] = []
        self._lock            = threading.Lock()

    def register_migration(self, feature: str, new_service: NewMicroservice):
        self._migrations[feature] = FeatureMigration(feature)
        self._services[feature]   = new_service

    def get_migration(self, feature: str) -> Optional[FeatureMigration]:
        return self._migrations.get(feature)

    def handle(self, feature: str, route: str, payload: Dict) -> Dict:
        request_id = str(uuid.uuid4())[:8]
        migration  = self._migrations.get(feature)

        if migration is None:
            # Feature not registered → always goes to monolith
            result = self.monolith.handle(route, payload)
            self._log(request_id, feature, "monolith", result, None)
            return result

        if migration.phase == MigrationPhase.SHADOW:
            return self._shadow_handle(feature, route, payload, request_id, migration)

        if migration.routes_to_new(request_id):
            try:
                svc    = self._services[feature]
                result = svc.handle(route, payload)
                self._log(request_id, feature, svc.name, result, None)
                return result
            except Exception as e:
                # Fallback to monolith on error
                result = self.monolith.handle(route, payload)
                result["_fallback"] = str(e)
                self._log(request_id, feature, "monolith-fallback", result, str(e))
                return result
        else:
            result = self.monolith.handle(route, payload)
            self._log(request_id, feature, "monolith", result, None)
            return result

    def _shadow_handle(self, feature: str, route: str, payload: Dict,
                       request_id: str, migration: FeatureMigration) -> Dict:
        """Run both; return monolith result; compare outputs asynchronously."""
        mono_result = self.monolith.handle(route, payload)

        def compare():
            try:
                svc    = self._services[feature]
                new_r  = svc.handle(route, payload)
                with self._lock:
                    migration.shadow_comparisons += 1
                    if mono_result.get("data") != new_r.get("data"):
                        migration.shadow_mismatches += 1
            except Exception:
                pass

        threading.Thread(target=compare, daemon=True).start()
        self._log(request_id, feature, "monolith+shadow", mono_result, None)
        return mono_result

    def _log(self, req_id: str, feature: str, routed_to: str,
             result: Dict, error: Optional[str]):
        self._request_log.append({
            "req_id"    : req_id,
            "feature"   : feature,
            "routed_to" : routed_to,
            "error"     : error,
        })

    def routing_stats(self) -> Dict[str, Dict]:
        stats: Dict[str, Dict] = {}
        for entry in self._request_log:
            feat = entry["feature"]
            if feat not in stats:
                stats[feat] = {"monolith": 0, "new_service": 0,
                               "fallback": 0, "shadow": 0, "errors": 0}
            dest = entry["routed_to"]
            if "fallback" in dest:
                stats[feat]["fallback"] += 1
            elif "shadow" in dest:
                stats[feat]["shadow"] += 1
            elif "monolith" in dest:
                stats[feat]["monolith"] += 1
            else:
                stats[feat]["new_service"] += 1
            if entry["error"]:
                stats[feat]["errors"] += 1
        return stats


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_strangler_fig():
    print("=" * 65)
    print("STRANGLER FIG PATTERN")
    print("=" * 65)

    monolith = MonolithHandler()
    proxy    = StranglerFigProxy(monolith)

    order_svc   = NewMicroservice("order-microservice",   error_rate=0.0)
    payment_svc = NewMicroservice("payment-microservice", error_rate=0.05)

    proxy.register_migration("orders",   order_svc)
    proxy.register_migration("payments", payment_svc)

    def send_requests(feature: str, n: int = 10):
        for _ in range(n):
            proxy.handle(feature, f"/api/{feature}", {"data": "test"})

    # ── 1. Phase 0: Everything on monolith ────────
    print("\n[1] PHASE 0 — ALL TRAFFIC ON MONOLITH")
    print("─" * 55)
    send_requests("orders", 10)
    stats = proxy.routing_stats()
    print(f"  Orders routing: {stats.get('orders', {})}")
    print(f"  → 100% monolith. New service not receiving any traffic.")

    # ── 2. Phase 1: Shadow mode ───────────────────
    print("\n\n[2] PHASE 1 — SHADOW MODE (test parity)")
    print("─" * 55)
    proxy.get_migration("orders").enable_shadow()
    send_requests("orders", 10)
    time.sleep(0.1)  # let shadow threads complete
    mig = proxy.get_migration("orders")
    print(f"  Shadow comparisons: {mig.shadow_comparisons}")
    print(f"  Shadow mismatches:  {mig.shadow_mismatches}")
    print(f"  → Monolith result returned to users. New service tested silently.")

    # ── 3. Phase 2: 10% canary ────────────────────
    print("\n\n[3] PHASE 2 — CANARY AT 10%")
    print("─" * 55)
    proxy.get_migration("orders").set_traffic(10.0, actor="ops-eng")
    send_requests("orders", 50)
    stats = proxy.routing_stats()["orders"]
    new_svc_pct = stats["new_service"] / (stats["monolith"] + stats["new_service"] + 1) * 100
    print(f"  After 50 requests: {stats}")
    print(f"  New service received ~{new_svc_pct:.0f}% of traffic")

    # ── 4. Phase 3: 50% rollout ───────────────────
    print("\n\n[4] PHASE 3 — RAMP TO 50%")
    print("─" * 55)
    proxy.get_migration("orders").set_traffic(50.0, actor="ops-eng")
    send_requests("orders", 50)
    stats = proxy.routing_stats()["orders"]
    print(f"  Cumulative: {stats}")

    # ── 5. Phase 4: Full migration ────────────────
    print("\n\n[5] PHASE 4 — FULL MIGRATION (100%)")
    print("─" * 55)
    proxy.get_migration("orders").set_traffic(100.0, actor="ops-eng")
    send_requests("orders", 10)
    stats = proxy.routing_stats()["orders"]
    print(f"  Cumulative routing stats: {stats}")
    print(f"  → All orders traffic on new microservice.")
    print(f"  → Monolith order code can now be deleted.")

    # ── 6. Rollback demonstration ─────────────────
    print("\n\n[6] ROLLBACK — BACK TO MONOLITH INSTANTLY")
    print("─" * 55)
    proxy.get_migration("orders").set_traffic(0.0, actor="incident-on-call")
    result = proxy.handle("orders", "/api/orders", {})
    print(f"  After rollback: source={result['source']}")
    print(f"  → One flag change. Zero downtime. Instant rollback.")

    # ── 7. Payment service with fallback ──────────
    print("\n\n[7] FALLBACK ON ERROR — PAYMENT SERVICE (5% error rate)")
    print("─" * 55)
    proxy.get_migration("payments").set_traffic(100.0, actor="ops")
    fallback_count = 0
    for _ in range(30):
        r = proxy.handle("payments", "/api/payments", {})
        if r.get("_fallback"):
            fallback_count += 1
    print(f"  30 requests at 100% new service (5% error rate)")
    print(f"  Fallbacks to monolith: ~{fallback_count}")
    print(f"  → Errors auto-route to monolith. Users never see failures.")

    # ── 8. Rollout history ────────────────────────
    print("\n\n[8] MIGRATION ROLLOUT HISTORY — ORDERS FEATURE")
    print("─" * 55)
    mig = proxy.get_migration("orders")
    for step in mig.rollout_history:
        phase = step.get("phase", "?")
        frm   = step.get("from", "—")
        to    = step.get("to", "—")
        actor = step.get("actor", "—")
        if "from" in step:
            print(f"  {actor:<15} {frm}% → {to}%  [{phase}]")
        else:
            print(f"  {'auto':<15} — → —  [{phase}]")

    # ── 9. Key principles ─────────────────────────
    print("\n\n[9] STRANGLER FIG KEY PRINCIPLES")
    print("─" * 55)
    principles = [
        ("Incremental",      "Migrate one feature at a time, not all at once"),
        ("No big bang",      "Each step is reversible; rollback is one flag change"),
        ("Shadow first",     "Validate parity before routing real traffic"),
        ("Proxy is neutral", "Proxy routes; it doesn't contain business logic"),
        ("Feature flags",    "Traffic % is a runtime setting, not a deployment"),
        ("Monolith lives",   "Monolith stays alive until 100% migrated and stable"),
        ("Delete after",     "Only delete monolith code after weeks of 100% stability"),
    ]
    for name, desc in principles:
        print(f"  {name:<20} {desc}")


if __name__ == "__main__":
    demonstrate_strangler_fig()
