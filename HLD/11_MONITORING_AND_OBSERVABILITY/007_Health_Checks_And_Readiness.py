"""
HEALTH CHECKS AND READINESS PROBES
=====================================

Problem Statement:
Kubernetes and load balancers need to know if a pod can serve traffic.
Health checks prevent routing to broken instances and enable zero-downtime
deploys. Poorly designed health checks cause outages or false positives.

Probe Types (Kubernetes):
  Liveness:    Is the process alive? Unhealthy → kill + restart.
               Check: process loop not deadlocked, no OOM.
               Should NOT check external dependencies (DB). If DB is down,
               restarting the pod won't fix it — causes cascading restarts.

  Readiness:   Is the pod ready to receive traffic? Unhealthy → remove from LB.
               Check: DB connected, cache warm, config loaded.
               A pod can be live (not killed) but not ready (no traffic).

  Startup:     Is the slow-starting container finished initializing?
               Replaces liveness during startup to avoid premature kills.
               Use for containers that take > 30s to start.

Check Mechanisms:
  HTTP GET:    GET /health → 200 OK.
  TCP socket:  TCP connection succeeds.
  Exec:        Run command in container; exit 0 = healthy.

Probe Parameters:
  initialDelaySeconds: Wait before first probe (for slow starts).
  periodSeconds:       How often to probe (default 10s).
  timeoutSeconds:      Probe timeout (default 1s).
  failureThreshold:    Consecutive failures before action (default 3).
  successThreshold:    Consecutive successes to recover (default 1).

Health Check Endpoints:
  /health or /healthz:       Simple liveness. Returns 200 or 503.
  /ready or /readyz:         Readiness. Checks dependencies.
  /startup or /startupz:     Startup check.
  /metrics:                  Prometheus metrics (separate).
  /info or /version:         Build version, git SHA.

Dependency Health:
  Check each dependency independently and return structured JSON:
  {
    "status": "degraded",
    "checks": {
      "database": {"status": "ok",       "latency_ms": 2},
      "redis":    {"status": "degraded", "latency_ms": 250},
      "external_api": {"status": "ok",   "latency_ms": 45}
    }
  }
  Degrade gracefully: if Redis is down but DB is up, return 200 (not 503)
  for readiness if the app can function without Redis (degraded mode).

Circuit Breaker in Health:
  Don't block health check on slow dependency checks.
  Use timeout (100ms) on each dependency check.
  Cache the last good check result; don't hammer broken deps.
"""

from __future__ import annotations

import time
import threading
import random
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum


# ─────────────────────────────────────────────
# CHECK STATUS
# ─────────────────────────────────────────────

class CheckStatus(Enum):
    OK       = "ok"
    DEGRADED = "degraded"
    FAILED   = "failed"
    UNKNOWN  = "unknown"

    def is_healthy(self) -> bool:
        return self in (CheckStatus.OK, CheckStatus.DEGRADED)


# ─────────────────────────────────────────────
# DEPENDENCY CHECK
# ─────────────────────────────────────────────

@dataclass
class CheckResult:
    name:       str
    status:     CheckStatus
    latency_ms: float
    message:    str          = ""
    critical:   bool         = True    # if True, failure makes pod not ready

    def to_dict(self) -> Dict:
        return {
            "status":     self.status.value,
            "latency_ms": round(self.latency_ms, 2),
            "message":    self.message,
            "critical":   self.critical,
        }


class DependencyChecker:
    """
    Checks a single external dependency with timeout and caching.
    """

    def __init__(self, name: str, check_fn: Callable[[], bool],
                 critical: bool = True, timeout_s: float = 0.5,
                 cache_ttl_s: float = 5.0):
        self.name        = name
        self._fn         = check_fn
        self.critical    = critical
        self._timeout_s  = timeout_s
        self._cache_ttl  = cache_ttl_s
        self._last_result: Optional[CheckResult] = None
        self._last_check_ts: float = 0.0
        self._lock       = threading.Lock()

    def check(self, use_cache: bool = True) -> CheckResult:
        now = time.time()
        with self._lock:
            if (use_cache and self._last_result and
                    now - self._last_check_ts < self._cache_ttl):
                return self._last_result

        # Run check with timeout
        result_holder = [None]
        error_holder  = [None]

        def run():
            try:
                start = time.time()
                ok    = self._fn()
                lat   = (time.time() - start) * 1000
                result_holder[0] = (ok, lat)
            except Exception as e:
                error_holder[0] = str(e)

        t = threading.Thread(target=run)
        t.start()
        t.join(self._timeout_s)

        if t.is_alive():
            status = CheckStatus.FAILED
            msg    = f"check timed out after {self._timeout_s*1000:.0f}ms"
            lat    = self._timeout_s * 1000
        elif error_holder[0]:
            status = CheckStatus.FAILED
            msg    = error_holder[0]
            lat    = 0.0
        else:
            ok, lat = result_holder[0]
            status  = CheckStatus.OK if ok else CheckStatus.FAILED
            msg     = "ok" if ok else "check returned false"

        result = CheckResult(self.name, status, lat, msg, self.critical)
        with self._lock:
            self._last_result    = result
            self._last_check_ts  = now
        return result


# ─────────────────────────────────────────────
# HEALTH CHECK AGGREGATOR
# ─────────────────────────────────────────────

@dataclass
class HealthResponse:
    status:      CheckStatus
    checks:      Dict[str, CheckResult]
    version:     str
    uptime_s:    float
    timestamp:   float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "status":    self.status.value,
            "version":   self.version,
            "uptime_s":  round(self.uptime_s, 1),
            "timestamp": self.timestamp,
            "checks":    {k: v.to_dict() for k, v in self.checks.items()},
        }

    def http_status_code(self) -> int:
        """503 for failed, 207 for degraded, 200 for ok."""
        if self.status == CheckStatus.FAILED:
            return 503
        if self.status == CheckStatus.DEGRADED:
            return 207
        return 200


class HealthEndpoint:
    """
    Serves /healthz, /readyz, /startupz endpoints.
    """

    def __init__(self, version: str):
        self._version      = version
        self._start_time   = time.time()
        self._checkers:    Dict[str, DependencyChecker] = {}
        self._is_ready     = False
        self._is_started   = False
        self._is_shutdown  = False

    def add_checker(self, checker: DependencyChecker):
        self._checkers[checker.name] = checker

    def mark_started(self):
        self._is_started = True

    def mark_ready(self, ready: bool = True):
        self._is_ready = ready

    def begin_shutdown(self):
        """Called on SIGTERM; immediately fail readiness so LB removes pod."""
        self._is_ready    = False
        self._is_shutdown = True

    def _aggregate(self, checks: Dict[str, CheckResult]) -> CheckStatus:
        if not checks:
            return CheckStatus.OK
        critical_failed = any(
            r.status == CheckStatus.FAILED and r.critical
            for r in checks.values()
        )
        if critical_failed:
            return CheckStatus.FAILED
        any_degraded = any(r.status == CheckStatus.DEGRADED for r in checks.values())
        non_critical_failed = any(
            r.status == CheckStatus.FAILED and not r.critical
            for r in checks.values()
        )
        if any_degraded or non_critical_failed:
            return CheckStatus.DEGRADED
        return CheckStatus.OK

    def liveness(self) -> HealthResponse:
        """
        /healthz — is the process alive?
        Only basic checks: no deadlock, no OOM.
        Do NOT check external dependencies here.
        """
        uptime  = time.time() - self._start_time
        status  = CheckStatus.FAILED if self._is_shutdown else CheckStatus.OK
        return HealthResponse(status, {}, self._version, uptime)

    def readiness(self) -> HealthResponse:
        """
        /readyz — can we serve traffic?
        Check all dependencies.
        """
        uptime = time.time() - self._start_time

        if not self._is_started or not self._is_ready or self._is_shutdown:
            return HealthResponse(
                CheckStatus.FAILED, {}, self._version, uptime,
            )

        check_results = {name: c.check() for name, c in self._checkers.items()}
        overall       = self._aggregate(check_results)
        return HealthResponse(overall, check_results, self._version, uptime)

    def startup(self) -> HealthResponse:
        """
        /startupz — has the container finished starting?
        """
        uptime = time.time() - self._start_time
        status = CheckStatus.OK if self._is_started else CheckStatus.FAILED
        return HealthResponse(status, {}, self._version, uptime)


# ─────────────────────────────────────────────
# KUBERNETES PROBE SIMULATOR
# ─────────────────────────────────────────────

@dataclass
class ProbeConfig:
    initial_delay_s:   float = 5.0
    period_s:          float = 10.0
    timeout_s:         float = 1.0
    failure_threshold: int   = 3
    success_threshold: int   = 1


class PodProbeSimulator:
    """
    Simulates Kubernetes probing a pod over time.
    Tracks consecutive failures and determines pod state.
    """

    def __init__(self, endpoint: HealthEndpoint, config: ProbeConfig):
        self._ep             = endpoint
        self._cfg            = config
        self._liveness_fails = 0
        self._ready_fails    = 0
        self._ready_successes= 0
        self.pod_alive       = True
        self.pod_ready       = False
        self._started_at     = time.time()

    def probe_liveness(self) -> bool:
        resp = self._ep.liveness()
        ok   = resp.status.is_healthy()
        if ok:
            self._liveness_fails = 0
        else:
            self._liveness_fails += 1
            if self._liveness_fails >= self._cfg.failure_threshold:
                self.pod_alive = False   # Kill pod
        return ok

    def probe_readiness(self) -> bool:
        resp = self._ep.readiness()
        ok   = resp.status.is_healthy()
        if ok:
            self._ready_fails    = 0
            self._ready_successes += 1
            if self._ready_successes >= self._cfg.success_threshold:
                self.pod_ready = True
        else:
            self._ready_fails    += 1
            self._ready_successes = 0
            if self._ready_fails >= self._cfg.failure_threshold:
                self.pod_ready = False   # Remove from LB
        return ok


# ─────────────────────────────────────────────
# GRACEFUL SHUTDOWN SEQUENCER
# ─────────────────────────────────────────────

class GracefulShutdown:
    """
    Implements the recommended graceful shutdown sequence:
    1. Receive SIGTERM
    2. Stop accepting new connections (fail readiness immediately)
    3. Wait for LB to drain (propagation delay ~15s)
    4. Finish in-flight requests
    5. Close DB connections, flush buffers
    6. Exit
    """

    def __init__(self, endpoint: HealthEndpoint,
                 drain_wait_s: float = 15.0, max_inflight_s: float = 30.0):
        self._ep            = endpoint
        self._drain_wait    = drain_wait_s
        self._max_inflight  = max_inflight_s
        self._in_flight     = 0
        self._lock          = threading.Lock()

    def shutdown(self, verbose: bool = True) -> List[str]:
        steps = []

        def log(msg: str):
            steps.append(msg)
            if verbose:
                print(f"    {msg}")

        log("SIGTERM received")
        self._ep.begin_shutdown()
        log(f"Readiness probe will fail → LB draining")
        log(f"Waiting {self._drain_wait}s for LB to propagate")
        # In prod: time.sleep(self._drain_wait)

        start = time.time()
        while self._in_flight > 0 and time.time() - start < self._max_inflight_s:
            pass
        log(f"In-flight requests drained")
        log("Flushing metrics / closing DB connections")
        log("Process exit 0")
        return steps


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_health_checks():
    print("=" * 65)
    print("HEALTH CHECKS AND READINESS PROBES")
    print("=" * 65)

    # ── Setup dependencies ────────────────────
    print("\n[1] DEPENDENCY CHECKERS")
    print("─" * 55)

    db_healthy  = [True]
    redis_healthy = [True]
    ext_healthy   = [True]

    checkers = [
        DependencyChecker("database",     lambda: db_healthy[0],    critical=True,
                          timeout_s=0.1, cache_ttl_s=5.0),
        DependencyChecker("redis",        lambda: redis_healthy[0], critical=False,
                          timeout_s=0.1, cache_ttl_s=5.0),
        DependencyChecker("external_api", lambda: ext_healthy[0],   critical=False,
                          timeout_s=0.1, cache_ttl_s=5.0),
    ]

    endpoint = HealthEndpoint("v2.5.1-abc1234")
    for c in checkers:
        endpoint.add_checker(c)
        print(f"  {c.name:<20} critical={c.critical}  timeout={c._timeout_s*1000:.0f}ms")

    # ── Startup sequence ──────────────────────
    print("\n[2] STARTUP SEQUENCE")
    print("─" * 55)

    # Before mark_started
    resp = endpoint.startup()
    print(f"  startup (before init): status={resp.status.value}  HTTP={resp.http_status_code()}")

    resp = endpoint.readiness()
    print(f"  readiness (not ready): status={resp.status.value}  HTTP={resp.http_status_code()}")

    # Mark as started and ready
    endpoint.mark_started()
    endpoint.mark_ready(True)

    resp = endpoint.startup()
    print(f"  startup (after init):  status={resp.status.value}  HTTP={resp.http_status_code()}")

    # ── Healthy state ─────────────────────────
    print("\n[3] HEALTHY STATE")
    print("─" * 55)

    resp = endpoint.readiness()
    print(f"  /readyz status: {resp.status.value}  HTTP={resp.http_status_code()}")
    print(f"  {json.dumps(resp.to_dict(), indent=4)[:300]}...")

    # ── DB fails (critical) ───────────────────
    print("\n[4] CRITICAL DEPENDENCY FAILS (Database)")
    print("─" * 55)

    db_healthy[0] = False
    # Bypass cache for demo
    for c in checkers:
        c._last_result = None

    resp = endpoint.readiness()
    print(f"  /readyz status: {resp.status.value}  HTTP={resp.http_status_code()}")
    for name, check in resp.checks.items():
        icon = "✓" if check.status == CheckStatus.OK else "✗"
        print(f"    [{icon}] {name:<18}: {check.status.value}")

    # ── Redis fails (non-critical = degraded) ──
    print("\n[5] NON-CRITICAL DEPENDENCY FAILS (Redis)")
    print("─" * 55)

    db_healthy[0]    = True
    redis_healthy[0] = False
    for c in checkers:
        c._last_result = None

    resp = endpoint.readiness()
    print(f"  /readyz status: {resp.status.value}  HTTP={resp.http_status_code()}")
    print("  (Degraded = still serving traffic, but warning)")
    for name, check in resp.checks.items():
        icon = "✓" if check.status == CheckStatus.OK else ("~" if check.critical is False else "✗")
        print(f"    [{icon}] {name:<18}: {check.status.value}  critical={check.critical}")

    # Reset
    redis_healthy[0] = True
    for c in checkers:
        c._last_result = None

    # ── Kubernetes Probe Simulation ────────────
    print("\n[6] KUBERNETES PROBE SIMULATION")
    print("─" * 55)

    cfg      = ProbeConfig(initial_delay_s=0, period_s=1,
                           failure_threshold=3, success_threshold=1)
    sim      = PodProbeSimulator(endpoint, cfg)

    results = []
    for tick in range(8):
        liveness_ok  = sim.probe_liveness()
        readiness_ok = sim.probe_readiness()
        results.append((tick, liveness_ok, readiness_ok,
                        sim.pod_alive, sim.pod_ready))
        if tick == 4:
            db_healthy[0] = False   # DB dies at tick 4
            for c in checkers:
                c._last_result = None

    print(f"  {'Tick':<6} {'Liveness':<12} {'Readiness':<12} {'Alive':<8} {'Ready'}")
    print("  " + "─" * 50)
    for tick, liveness_ok, readiness_ok, alive, ready in results:
        note = " ← DB fails" if tick == 4 else ""
        print(f"  {tick:<6} {str(liveness_ok):<12} {str(readiness_ok):<12} "
              f"{str(alive):<8} {ready}{note}")

    # Reset
    db_healthy[0] = True
    for c in checkers:
        c._last_result = None
    endpoint._is_ready = True

    # ── Graceful Shutdown ─────────────────────
    print("\n[7] GRACEFUL SHUTDOWN SEQUENCE")
    print("─" * 55)

    gs = GracefulShutdown(endpoint, drain_wait_s=15, max_inflight_s=30)
    gs.shutdown(verbose=True)

    # ── Probe Anti-Patterns ───────────────────
    print("\n[8] PROBE DESIGN GUIDELINES")
    print("─" * 55)

    guidelines = [
        ("LIVENESS DO",    "Only check if the process is alive (deadlock detection)"),
        ("LIVENESS DON'T", "Check database in liveness — cascading restart storm"),
        ("READINESS DO",   "Check all critical dependencies with short timeout"),
        ("READINESS DO",   "Fail immediately on SIGTERM to drain LB connections"),
        ("STARTUP DO",     "Use startup probe for slow-init containers"),
        ("TIMEOUT",        "Keep health check total < 100ms; use cached results"),
        ("CIRCUIT BREAK",  "Cache last result 5-15s; don't hammer broken deps"),
        ("HTTP CODE",      "200=ok, 207=degraded, 503=not ready/failed"),
    ]
    for label, desc in guidelines:
        print(f"  [{label:<18}] {desc}")


if __name__ == "__main__":
    demonstrate_health_checks()
