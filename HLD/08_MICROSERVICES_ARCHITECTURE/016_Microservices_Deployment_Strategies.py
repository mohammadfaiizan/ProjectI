"""
MICROSERVICES DEPLOYMENT STRATEGIES
========================================

Problem Statement:
Deploying a new version of a microservice carries risk:
  - New version has a bug → users hit errors.
  - Deployment takes time → brief downtime if done naively.
  - Rollback must be fast → can't wait for a full redeploy.

Deployment strategies control how traffic shifts from old → new version,
trading off: deployment speed, blast radius, complexity, and rollback speed.

BLUE-GREEN DEPLOYMENT:
  Maintain two identical environments: Blue (current) and Green (new).
  Blue serves 100% of traffic. Deploy new version to Green.
  Test Green. When ready, switch load balancer → Green serves 100%.
  Blue becomes standby for instant rollback.
  Pro:  Instant cutover. Instant rollback (flip LB back to Blue).
  Con:  Double infrastructure cost. Schema migrations are hard.
  Use:  Stable services that need instant rollback capability.

CANARY DEPLOYMENT:
  Route a small % of traffic (1-5%) to the new version.
  Monitor error rate, latency, business metrics.
  Gradually increase %: 5% → 20% → 50% → 100%.
  If metrics degrade → drain canary, route all traffic back to old version.
  Pro:  Limited blast radius. Real-traffic validation.
  Con:  Slower; requires monitoring; some users see new version.
  Use:  High-risk changes, user-facing features.

ROLLING DEPLOYMENT:
  Replace old instances with new ones gradually, one-by-one.
  At any point during deployment, some instances run old, some run new.
  Kubernetes does this by default.
  Pro:  No extra infrastructure. Gradual.
  Con:  Two versions running simultaneously (backward compat required).
        Harder to roll back (must roll forward or explicitly scale back).
  Use:  Stateless services with good health checks.

FEATURE FLAG DEPLOYMENT:
  Deploy code but hide the feature behind a flag.
  Feature is disabled in production. Gradually enable via flag.
  Decouple deployment from feature release.
  Pro:  Instant rollback (flip flag). Dark launch.
  Con:  Flag debt accumulates; must clean up old flags.
  Use:  All changes — combine with any deployment strategy.

Comparison:
  Blue-Green:  Fastest rollback. Most expensive (2x infra).
  Canary:      Safest blast radius. Needs good observability.
  Rolling:     Default for K8s. Needs backward compatibility.
  Feature flag:Fastest "rollback". Works at application level.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import time
import threading
import random
import uuid


# ─────────────────────────────────────────────
# SERVICE INSTANCE
# ─────────────────────────────────────────────

class InstanceState(Enum):
    STARTING = "starting"
    HEALTHY  = "healthy"
    DRAINING = "draining"
    STOPPED  = "stopped"


@dataclass
class ServiceInstance:
    instance_id : str
    version     : str
    color       : str = "blue"   # blue / green / canary
    state       : InstanceState = InstanceState.STARTING
    error_rate  : float = 0.0
    latency_ms  : float = 20.0
    start_time  : float = field(default_factory=time.time)
    request_count: int  = 0
    error_count : int   = 0

    def handle_request(self) -> Dict:
        self.request_count += 1
        time.sleep(self.latency_ms / 1000)
        if random.random() < self.error_rate:
            self.error_count += 1
            raise RuntimeError(f"Instance {self.instance_id} v{self.version} error")
        return {
            "instance"  : self.instance_id,
            "version"   : self.version,
            "color"     : self.color,
        }

    @property
    def is_healthy(self) -> bool:
        return self.state == InstanceState.HEALTHY

    @property
    def observed_error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count


# ─────────────────────────────────────────────
# LOAD BALANCER
# ─────────────────────────────────────────────

class TrafficRouter:
    """Routes traffic to instances based on weights (used by all strategies)."""

    def __init__(self):
        self._pools : Dict[str, List[ServiceInstance]] = {}   # color → instances
        self._weights: Dict[str, float] = {}                  # color → % traffic
        self._lock   = threading.Lock()
        self._log    : List[Dict] = []
        self._rr_counters : Dict[str, int] = {}

    def add_instances(self, color: str, instances: List[ServiceInstance]):
        with self._lock:
            self._pools[color] = instances
            if color not in self._weights:
                self._weights[color] = 0.0

    def set_weights(self, weights: Dict[str, float]):
        """weights: {"blue": 90.0, "green": 10.0}. Must sum to 100."""
        with self._lock:
            self._weights = dict(weights)

    def route(self) -> Optional[ServiceInstance]:
        with self._lock:
            colors    = list(self._weights.keys())
            weights   = [self._weights[c] for c in colors]
            total     = sum(weights)
            if total == 0:
                return None

            r = random.uniform(0, total)
            cumulative = 0.0
            selected_color = colors[-1]
            for color, weight in zip(colors, weights):
                cumulative += weight
                if r <= cumulative:
                    selected_color = color
                    break

            healthy = [i for i in self._pools.get(selected_color, [])
                       if i.is_healthy]
            if not healthy:
                return None

            # Round-robin within color
            idx = self._rr_counters.get(selected_color, 0)
            inst = healthy[idx % len(healthy)]
            self._rr_counters[selected_color] = idx + 1
            return inst

    def routing_stats(self) -> Dict[str, Dict]:
        with self._lock:
            stats = {}
            for color, instances in self._pools.items():
                total    = sum(i.request_count for i in instances)
                errors   = sum(i.error_count   for i in instances)
                versions = list({i.version for i in instances})
                stats[color] = {
                    "requests" : total,
                    "errors"   : errors,
                    "error_pct": round(errors / max(total, 1) * 100, 1),
                    "versions" : versions,
                    "instances": len(instances),
                    "weight_pct": self._weights.get(color, 0),
                }
            return stats


# ─────────────────────────────────────────────
# DEPLOYMENT MANAGER
# ─────────────────────────────────────────────

class DeploymentManager:
    """Orchestrates deployment strategies."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.router       = TrafficRouter()
        self._deploy_log  : List[Dict] = []

    def _log(self, strategy: str, action: str, details: Dict = None):
        self._deploy_log.append({
            "strategy": strategy,
            "action"  : action,
            "details" : details or {},
            "ts"      : time.time(),
        })

    # ── Blue-Green ─────────────────────────────────
    def blue_green_deploy(self, new_version: str,
                          old_error_rate: float = 0.0,
                          new_error_rate: float = 0.0) -> bool:
        """Deploy new version to Green; run smoke tests; switch traffic."""
        self._log("blue-green", "start", {"new_version": new_version})

        # Blue is live
        blue = [ServiceInstance(f"blue-{i}", "v1.0", "blue",
                                InstanceState.HEALTHY, old_error_rate, 20)
                for i in range(3)]
        self.router.add_instances("blue", blue)
        self.router.set_weights({"blue": 100.0})

        # Deploy to Green (idle)
        green = [ServiceInstance(f"green-{i}", new_version, "green",
                                 InstanceState.HEALTHY, new_error_rate, 18)
                 for i in range(3)]
        self.router.add_instances("green", green)
        self._log("blue-green", "green_deployed")

        # Smoke test Green (not yet serving traffic)
        smoke_pass = new_error_rate < 0.5
        if not smoke_pass:
            self._log("blue-green", "smoke_test_failed", {"error_rate": new_error_rate})
            return False
        self._log("blue-green", "smoke_test_passed")

        # Flip traffic to Green
        self.router.set_weights({"blue": 0.0, "green": 100.0})
        self._log("blue-green", "traffic_switched_to_green")

        # Blue stays on standby for rollback
        return True

    def blue_green_rollback(self):
        """Instant rollback: flip traffic back to Blue."""
        self.router.set_weights({"blue": 100.0, "green": 0.0})
        self._log("blue-green", "rollback_to_blue")

    # ── Canary ─────────────────────────────────────
    def canary_deploy(self, new_version: str, stages: List[Tuple[float, int]],
                      new_error_rate: float = 0.0,
                      error_threshold: float = 0.05) -> bool:
        """
        stages: [(canary_pct, n_requests_to_monitor), ...]
        Returns True if all stages passed, False if rolled back.
        """
        # Stable fleet
        stable = [ServiceInstance(f"stable-{i}", "v1.0", "stable",
                                  InstanceState.HEALTHY, 0.01, 20)
                  for i in range(4)]
        self.router.add_instances("stable", stable)

        # Deploy canary instance(s)
        canary = [ServiceInstance(f"canary-0", new_version, "canary",
                                  InstanceState.HEALTHY, new_error_rate, 18)]
        self.router.add_instances("canary", canary)
        self._log("canary", "deployed", {"version": new_version})

        for pct, n_requests in stages:
            self.router.set_weights({"stable": 100.0 - pct, "canary": pct})
            self._log("canary", "traffic_shifted",
                      {"canary_pct": pct, "monitoring_requests": n_requests})

            # Send traffic and monitor
            errors = 0
            for _ in range(n_requests):
                inst = self.router.route()
                if inst:
                    try:
                        inst.handle_request()
                    except RuntimeError:
                        errors += 1

            observed_rate = errors / max(n_requests, 1)
            canary_inst   = canary[0]

            self._log("canary", "stage_metrics", {
                "canary_pct": pct,
                "canary_requests": canary_inst.request_count,
                "canary_errors"  : canary_inst.error_count,
                "canary_error_rate": f"{canary_inst.observed_error_rate:.1%}",
            })

            if canary_inst.observed_error_rate > error_threshold:
                self._log("canary", "rollback",
                          {"reason": f"error_rate={canary_inst.observed_error_rate:.1%} "
                                     f"> threshold={error_threshold:.1%}"})
                self.router.set_weights({"stable": 100.0, "canary": 0.0})
                return False

        # All stages passed → promote canary
        self.router.set_weights({"stable": 0.0, "canary": 100.0})
        self._log("canary", "promoted")
        return True

    # ── Rolling ────────────────────────────────────
    def rolling_deploy(self, new_version: str, instance_count: int = 4,
                       new_error_rate: float = 0.0):
        """Replace instances one-by-one."""
        instances = [
            ServiceInstance(f"inst-{i}", "v1.0", "rolling",
                            InstanceState.HEALTHY, 0.0, 20)
            for i in range(instance_count)
        ]
        self.router.add_instances("rolling", instances)
        self.router.set_weights({"rolling": 100.0})
        self._log("rolling", "start", {"total_instances": instance_count})

        replaced = []
        for i, old_inst in enumerate(instances):
            # Drain old instance
            old_inst.state = InstanceState.DRAINING
            # Start new instance
            new_inst = ServiceInstance(f"inst-{i}-new", new_version, "rolling",
                                       InstanceState.HEALTHY, new_error_rate, 18)
            instances[i] = new_inst
            old_inst.state = InstanceState.STOPPED
            replaced.append(new_inst.instance_id)
            self._log("rolling", "instance_replaced", {
                "old": old_inst.instance_id,
                "new": new_inst.instance_id,
                "progress": f"{i+1}/{instance_count}"
            })

        return replaced


# ─────────────────────────────────────────────
# FEATURE FLAG DEPLOYMENT
# ─────────────────────────────────────────────

class FeatureFlagDeployment:
    """Code is deployed but feature hidden behind a flag."""

    def __init__(self):
        self._flags : Dict[str, Dict] = {}

    def define(self, flag: str, enabled: bool = False, rollout_pct: float = 0.0):
        self._flags[flag] = {"enabled": enabled, "rollout_pct": rollout_pct,
                             "history": []}

    def update(self, flag: str, enabled: bool, rollout_pct: float, actor: str = "ops"):
        if flag in self._flags:
            self._flags[flag]["enabled"]     = enabled
            self._flags[flag]["rollout_pct"] = rollout_pct
            self._flags[flag]["history"].append({
                "actor": actor, "enabled": enabled,
                "rollout_pct": rollout_pct, "ts": time.time()
            })

    def is_enabled(self, flag: str, user_id: str) -> bool:
        f = self._flags.get(flag, {})
        if not f.get("enabled"):
            return False
        pct = f.get("rollout_pct", 0)
        import hashlib
        h = int(hashlib.md5(f"{flag}:{user_id}".encode()).hexdigest(), 16) % 100
        return h < pct


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_deployment_strategies():
    print("=" * 65)
    print("MICROSERVICES DEPLOYMENT STRATEGIES")
    print("=" * 65)

    # ── 1. Blue-Green ─────────────────────────────
    print("\n[1] BLUE-GREEN DEPLOYMENT")
    print("─" * 55)
    dm1 = DeploymentManager("order-service")
    print("  Deploying v2.0 (no errors) to Green:")
    success = dm1.blue_green_deploy("v2.0", new_error_rate=0.0)
    print(f"  Deployment result: {'SUCCESS' if success else 'FAILED'}")
    stats = dm1.router.routing_stats()
    for color, data in stats.items():
        if data["instances"] > 0:
            print(f"  [{color}] weight={data['weight_pct']}%  "
                  f"version={data['versions']}  instances={data['instances']}")

    print(f"\n  Simulating rollback (bug found after switch):")
    dm1.blue_green_rollback()
    stats = dm1.router.routing_stats()
    for color, data in stats.items():
        if data["instances"] > 0 and data["weight_pct"] > 0:
            print(f"  [{color}] weight={data['weight_pct']}% (back to blue)")
    print(f"  Rollback: INSTANT (load balancer flip only)")

    print(f"\n  Blue-Green with broken new version:")
    dm1b = DeploymentManager("order-service-b")
    success = dm1b.blue_green_deploy("v2.1-broken", new_error_rate=0.9)
    print(f"  Deployment result: {'SUCCESS' if success else 'BLOCKED (smoke test failed)'}")

    # ── 2. Canary deployment ──────────────────────
    print("\n\n[2] CANARY DEPLOYMENT — PROGRESSIVE TRAFFIC SHIFT")
    print("─" * 55)

    dm2 = DeploymentManager("payment-service")
    stages = [(5, 30), (20, 50), (50, 50), (100, 0)]

    print("  Deploying v2.0 (1% error rate) with stages: 5%→20%→50%→100%")
    success = dm2.canary_deploy("v2.0", stages,
                                new_error_rate=0.01, error_threshold=0.05)
    print(f"  Canary promotion: {'SUCCESS' if success else 'ROLLED BACK'}")

    stats = dm2.router.routing_stats()
    for pool, data in stats.items():
        if data["instances"] > 0:
            print(f"  [{pool}] weight={data['weight_pct']}%  "
                  f"requests={data['requests']}  errors={data['error_pct']}%")

    print(f"\n  Deploying v2.1-buggy (10% error rate) — should trigger rollback:")
    dm3 = DeploymentManager("payment-service-v2")
    success = dm3.canary_deploy("v2.1-buggy",
                                [(5, 40)],   # single stage is enough to detect
                                new_error_rate=0.15,
                                error_threshold=0.05)
    print(f"  Result: {'SUCCESS' if success else 'ROLLED BACK (error rate too high)'}")
    log_entry = [e for e in dm3._deploy_log if e["action"] == "rollback"]
    if log_entry:
        print(f"  Reason: {log_entry[0]['details']['reason']}")

    # ── 3. Rolling deployment ─────────────────────
    print("\n\n[3] ROLLING DEPLOYMENT — INSTANCE BY INSTANCE")
    print("─" * 55)
    dm4 = DeploymentManager("inventory-service")
    print("  Rolling update: 4 instances, replacing one at a time")
    replaced = dm4.rolling_deploy("v3.0", instance_count=4, new_error_rate=0.0)
    print(f"  Replaced instances: {replaced}")
    log = [e for e in dm4._deploy_log if e["action"] == "instance_replaced"]
    for entry in log:
        print(f"    {entry['details']['old']:<15} → {entry['details']['new']:<15} "
              f"({entry['details']['progress']})")
    print(f"  → At each step, old and new versions serve traffic simultaneously.")
    print(f"    Requires backward compatibility between v_old and v_new.")

    # ── 4. Feature flag deployment ────────────────
    print("\n\n[4] FEATURE FLAG DEPLOYMENT — DECOUPLE CODE FROM FEATURE RELEASE")
    print("─" * 55)
    flags = FeatureFlagDeployment()
    flags.define("new_payment_flow", enabled=False, rollout_pct=0.0)

    print("  Code deployed. Feature disabled. No users affected.")
    users = [f"user-{i:03d}" for i in range(20)]
    enabled_0 = sum(1 for u in users if flags.is_enabled("new_payment_flow", u))
    print(f"  users seeing new flow: {enabled_0}/20")

    flags.update("new_payment_flow", enabled=True, rollout_pct=10.0, actor="product")
    enabled_10 = sum(1 for u in users if flags.is_enabled("new_payment_flow", u))
    print(f"  After 10% rollout:    {enabled_10}/20 users")

    flags.update("new_payment_flow", enabled=True, rollout_pct=50.0, actor="product")
    enabled_50 = sum(1 for u in users if flags.is_enabled("new_payment_flow", u))
    print(f"  After 50% rollout:    {enabled_50}/20 users")

    flags.update("new_payment_flow", enabled=True, rollout_pct=100.0, actor="product")
    enabled_100 = sum(1 for u in users if flags.is_enabled("new_payment_flow", u))
    print(f"  After 100% rollout:   {enabled_100}/20 users")

    flags.update("new_payment_flow", enabled=False, rollout_pct=0.0, actor="on-call")
    disabled = sum(1 for u in users if flags.is_enabled("new_payment_flow", u))
    print(f"  After kill-switch:    {disabled}/20 users (instant rollback)")

    # ── 5. Strategy comparison ────────────────────
    print("\n\n[5] DEPLOYMENT STRATEGY COMPARISON")
    print("─" * 55)
    rows = [
        ("Blue-Green",    "Instant",    "Blue stays on", "2x infra cost",    "Schema migrations"),
        ("Canary",        "Minutes",    "Drain canary",  "Complex monitoring","Two versions live"),
        ("Rolling",       "Gradual",    "Roll forward",  "Backward compat",  "No instant rollback"),
        ("Feature Flag",  "Instant",    "Flip flag off", "Flag debt",        "Code deployed but dark"),
    ]
    print(f"  {'Strategy':<14} {'Cutover':<12} {'Rollback':<16} {'Cost':<20} {'Note'}")
    print(f"  {'─'*78}")
    for name, cutover, rollback, cost, note in rows:
        print(f"  {name:<14} {cutover:<12} {rollback:<16} {cost:<20} {note}")

    # ── 6. Deployment log ─────────────────────────
    print("\n\n[6] DEPLOYMENT LOG — CANARY STAGES")
    print("─" * 55)
    for entry in dm2._deploy_log:
        action  = entry["action"]
        details = entry["details"]
        if action in ("traffic_shifted", "stage_metrics", "promoted", "rollback"):
            print(f"  [{action:<22}] {details}")


if __name__ == "__main__":
    demonstrate_deployment_strategies()
