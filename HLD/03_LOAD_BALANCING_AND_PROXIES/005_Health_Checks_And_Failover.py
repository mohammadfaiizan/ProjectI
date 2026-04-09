"""
HEALTH CHECKS AND FAILOVER
============================

Problem Statement:
Servers fail — hardware crashes, processes hang, memory exhausts. Without
health checking, a load balancer keeps routing traffic to dead servers,
causing errors for users. Health checks detect failure and remove unhealthy
backends; when they recover, they're added back automatically.

Types of Health Checks:
  Passive  : Monitor real traffic responses (504→ mark unhealthy). No probing.
  Active   : LB probes backends on a schedule (/health endpoint, TCP connect)

Check Levels:
  L3/L4: TCP connect on port — is the process up?
  L7:    HTTP GET /health — is the app logic healthy?
  Deep:  GET /health/deep — checks DB/cache connectivity

Failover Strategies:
  LB-level : Remove unhealthy backend from pool (immediate, per-request)
  DNS-level : Point domain to standby IP (slow — TTL propagation)
  Active-Passive: standby takes over when primary fails
  Active-Active : multiple primaries, any can handle traffic

Key Parameters:
  interval_s        : how often to probe (e.g., 5s)
  timeout_s         : how long to wait for probe response (e.g., 2s)
  healthy_threshold : consecutive successes to mark healthy (e.g., 2)
  unhealthy_threshold: consecutive failures to mark unhealthy (e.g., 3)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import time
import random
import threading


class BackendState(Enum):
    HEALTHY   = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING  = "draining"   # graceful shutdown


class CheckType(Enum):
    TCP    = "tcp"
    HTTP   = "http"
    HTTPS  = "https"
    DEEP   = "http_deep"


@dataclass
class HealthCheckConfig:
    check_type          : CheckType = CheckType.HTTP
    path                : str       = "/health"
    interval_s          : float     = 5.0
    timeout_s           : float     = 2.0
    healthy_threshold   : int       = 2
    unhealthy_threshold : int       = 3
    expected_status     : int       = 200


@dataclass
class Backend:
    backend_id   : str
    host         : str
    port         : int
    state        : BackendState = BackendState.HEALTHY
    # Simulate failure scenario
    _fail_after  : int = 999   # fail after N probes
    _recover_after: int = 999  # recover after N consecutive fails

    def __post_init__(self):
        self._probe_count = 0
        self._fail_count  = 0

    def probe_result(self) -> bool:
        """Simulate health check response from this backend."""
        self._probe_count += 1
        if self._probe_count > self._fail_after:
            self._fail_count += 1
            if self._fail_count > self._recover_after:
                # Backend has recovered
                self._fail_count  = 0
                self._probe_count = 0
                return True
            return False
        return True


@dataclass
class ProbeResult:
    backend_id  : str
    success     : bool
    latency_ms  : float
    status_code : int = 200
    check_type  : CheckType = CheckType.HTTP
    timestamp   : float = field(default_factory=time.time)


# ─────────────────────────────────────────────
# HEALTH CHECKER
# ─────────────────────────────────────────────

class HealthChecker:
    def __init__(self, config: HealthCheckConfig):
        self.config          = config
        self._fail_streak    : Dict[str, int] = {}
        self._success_streak : Dict[str, int] = {}
        self.probe_history   : List[ProbeResult] = []
        self.state_changes   : List[str] = []

    def probe(self, backend: Backend) -> ProbeResult:
        start   = time.perf_counter()
        success = backend.probe_result()
        latency = (time.perf_counter() - start) * 1000 + random.uniform(0.5, 3.0)
        return ProbeResult(
            backend_id  = backend.backend_id,
            success     = success,
            latency_ms  = round(latency, 2),
            status_code = 200 if success else 503,
            check_type  = self.config.check_type
        )

    def evaluate(self, backend: Backend, result: ProbeResult):
        bid = backend.backend_id
        if result.success:
            self._fail_streak[bid]    = 0
            self._success_streak[bid] = self._success_streak.get(bid, 0) + 1
            if (backend.state == BackendState.UNHEALTHY and
                    self._success_streak[bid] >= self.config.healthy_threshold):
                backend.state = BackendState.HEALTHY
                msg = f"  ✅ {bid} RECOVERED (successes={self._success_streak[bid]})"
                self.state_changes.append(msg)
                print(msg)
        else:
            self._success_streak[bid] = 0
            self._fail_streak[bid]    = self._fail_streak.get(bid, 0) + 1
            if (backend.state == BackendState.HEALTHY and
                    self._fail_streak[bid] >= self.config.unhealthy_threshold):
                backend.state = BackendState.UNHEALTHY
                msg = f"  ❌ {bid} UNHEALTHY (failures={self._fail_streak[bid]})"
                self.state_changes.append(msg)
                print(msg)

    def run_round(self, backends: List[Backend]):
        for backend in backends:
            result = self.probe(backend)
            self.probe_history.append(result)
            self.evaluate(backend, result)

    def summary(self):
        print(f"\n  Health Check Summary:")
        print(f"    Total probes  : {len(self.probe_history)}")
        print(f"    State changes : {len(self.state_changes)}")


# ─────────────────────────────────────────────
# PASSIVE HEALTH MONITOR
# ─────────────────────────────────────────────

class PassiveHealthMonitor:
    """
    Monitors real traffic responses. No probing.
    Marks backend unhealthy after consecutive 5xx errors.
    """

    def __init__(self, error_threshold: int = 5, window_s: float = 30.0):
        self.error_threshold = error_threshold
        self.window_s        = window_s
        self._errors         : Dict[str, List[float]] = {}
        self.backends_removed: List[str] = []

    def record(self, backend_id: str, status_code: int) -> bool:
        """Returns True if backend should be removed."""
        if status_code >= 500:
            now = time.time()
            self._errors.setdefault(backend_id, []).append(now)
            # Remove old errors outside window
            self._errors[backend_id] = [
                t for t in self._errors[backend_id] if now - t <= self.window_s
            ]
            if len(self._errors[backend_id]) >= self.error_threshold:
                if backend_id not in self.backends_removed:
                    self.backends_removed.append(backend_id)
                    print(f"  Passive: {backend_id} removed — "
                          f"{len(self._errors[backend_id])} errors in {self.window_s}s")
                return True
        return False


# ─────────────────────────────────────────────
# FAILOVER MANAGER
# ─────────────────────────────────────────────

class FailoverManager:
    """
    Manages primary/standby failover (active-passive pattern).
    Monitors primary; promotes standby if primary is down.
    """

    def __init__(self, primary: Backend, standby: Backend):
        self.primary  = primary
        self.standby  = standby
        self._active  = primary
        self.failovers = 0

    @property
    def active(self) -> Backend:
        return self._active

    def check_and_failover(self):
        if (self._active == self.primary and
                self.primary.state == BackendState.UNHEALTHY):
            self._active = self.standby
            self.failovers += 1
            print(f"  FAILOVER: {self.primary.backend_id} → {self.standby.backend_id} "
                  f"(failover #{self.failovers})")
        elif (self._active == self.standby and
              self.primary.state == BackendState.HEALTHY):
            self._active = self.primary
            print(f"  FAILBACK: {self.standby.backend_id} → {self.primary.backend_id} (primary restored)")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_health_checks():
    print("=" * 65)
    print("HEALTH CHECKS AND FAILOVER")
    print("=" * 65)
    random.seed(42)

    # ── Active Health Checks ──────────────────
    print("\n[1] ACTIVE HEALTH CHECKS — BACKEND FAILURE & RECOVERY")
    print("─" * 55)
    config = HealthCheckConfig(
        check_type=CheckType.HTTP,
        path="/health",
        interval_s=5.0,
        healthy_threshold=2,
        unhealthy_threshold=3
    )
    checker = HealthChecker(config)

    backends = [
        Backend("web-1", "10.0.1.1", 8080, _fail_after=2, _recover_after=4),
        Backend("web-2", "10.0.1.2", 8080),   # always healthy
        Backend("web-3", "10.0.1.3", 8080, _fail_after=1, _recover_after=3),
    ]

    print("  web-1: fails after 2 probes, recovers after 4 failures")
    print("  web-3: fails after 1 probe, recovers after 3 failures")
    print()
    for round_num in range(1, 10):
        print(f"  --- Round {round_num} ---")
        checker.run_round(backends)
        for b in backends:
            icon = "✅" if b.state == BackendState.HEALTHY else "❌"
            print(f"    {icon} {b.backend_id}: {b.state.value}")

    checker.summary()

    # ── Passive Health Monitor ────────────────
    print("\n\n[2] PASSIVE HEALTH MONITOR (traffic-based)")
    print("─" * 55)
    passive = PassiveHealthMonitor(error_threshold=5, window_s=30.0)
    responses = [
        ("web-1", 200), ("web-1", 200), ("web-1", 503), ("web-1", 503),
        ("web-1", 503), ("web-1", 503), ("web-1", 503),   # → removed
        ("web-2", 200), ("web-2", 200), ("web-2", 500),
    ]
    print("  Recording real traffic responses:")
    for backend_id, status in responses:
        removed = passive.record(backend_id, status)
        icon = "🚫" if removed else ("❌" if status >= 500 else "✅")
        print(f"    {backend_id}: HTTP {status}  {icon}")

    print(f"\n  Backends removed from pool: {passive.backends_removed}")

    # ── Active-Passive Failover ───────────────
    print("\n\n[3] ACTIVE-PASSIVE FAILOVER")
    print("─" * 55)
    primary = Backend("primary-db", "10.0.5.1", 5432)
    standby = Backend("standby-db", "10.0.5.2", 5432)
    fm = FailoverManager(primary, standby)

    print(f"  Active: {fm.active.backend_id}")

    # Simulate primary failure
    primary.state = BackendState.UNHEALTHY
    print(f"  primary-db marked UNHEALTHY")
    fm.check_and_failover()
    print(f"  Active: {fm.active.backend_id}")

    # Simulate primary recovery
    primary.state = BackendState.HEALTHY
    print(f"  primary-db recovered")
    fm.check_and_failover()
    print(f"  Active: {fm.active.backend_id}")

    # ── Health Check Config Guide ─────────────
    print("\n\n[4] HEALTH CHECK CONFIGURATION GUIDE")
    print("─" * 55)
    print(f"  {'Parameter':<28} {'Aggressive':<15} {'Standard':<15} {'Conservative'}")
    print(f"  {'─'*70}")
    rows = [
        ("interval_s",          "5s",      "10s",    "30s"),
        ("timeout_s",           "1s",      "3s",     "10s"),
        ("unhealthy_threshold", "1",       "2-3",    "5"),
        ("healthy_threshold",   "1",       "2",      "3"),
        ("Time to detect fail", "5-10s",   "20-30s", "2-3 min"),
        ("False positive risk", "High",    "Low",    "Very low"),
        ("Use case",            "Payments","APIs",   "Batch jobs"),
    ]
    for row in rows:
        print(f"  {row[0]:<28} {row[1]:<15} {row[2]:<15} {row[3]}")

    # ── Deep Health Checks ────────────────────
    print("\n\n[5] HEALTH CHECK ENDPOINT DESIGN")
    print("─" * 55)
    print("  GET /health        → shallow: returns 200 if process is up")
    print("  GET /health/ready  → readiness: checks DB/cache/dependencies")
    print("  GET /health/live   → liveness: only fails if process is stuck")
    print()
    print("  Kubernetes probes:")
    print("    livenessProbe  → GET /health/live   (restart if fails)")
    print("    readinessProbe → GET /health/ready  (remove from svc if fails)")
    print("    startupProbe   → GET /health/live   (grace period on startup)")
    print()
    example = {
        "status": "ok",
        "checks": {
            "database": "ok",
            "redis":    "ok",
            "disk":     "ok"
        },
        "version": "v2.3.1",
        "uptime_s": 3600
    }
    import json
    print("  Example /health response:")
    print(f"  {json.dumps(example, indent=4)}")


if __name__ == "__main__":
    demonstrate_health_checks()
