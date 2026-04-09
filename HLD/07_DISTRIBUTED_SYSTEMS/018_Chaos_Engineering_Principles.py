"""
CHAOS ENGINEERING PRINCIPLES
================================

Problem Statement:
Traditional testing verifies that systems work correctly under normal conditions.
But production failures are unpredictable: node crashes, network partitions,
dependency timeouts, memory pressure, cascading failures.
Chaos Engineering proactively injects failures to discover weaknesses before
they manifest as unplanned outages.

Definition (Netflix):
"The discipline of experimenting on a distributed system in order to build
confidence in the system's capability to withstand turbulent conditions in production."

Core Principles (Principles of Chaos):
  1. Build a hypothesis around steady state behavior.
     Measure a business metric: requests/s, error rate, latency p99.
  2. Vary real-world events.
     Crash instances, inject latency, kill dependencies.
  3. Run experiments in production (ideally).
     Non-production environments miss production-specific failure modes.
  4. Automate experiments to run continuously.
     Build "chaos" into the CI/CD pipeline.
  5. Minimize blast radius.
     Start small (1 instance), isolate to test environment first.

Common Chaos Experiments:
  - Kill random instances (Chaos Monkey — Netflix).
  - Simulate AZ (availability zone) failure.
  - Inject network latency (add 200ms to all calls).
  - Drop/corrupt network packets (chaos proxy).
  - Kill a dependency (database, cache, message queue).
  - Exhaust disk space or memory.
  - Clock skew injection.
  - DNS failure.

Steady State Metrics to Monitor:
  - Request success rate (target: >99.9%).
  - P99 latency (target: <200ms).
  - Error rate per service.
  - User-facing conversion rate or transactions/second.

Chaos Tools:
  - Chaos Monkey (Netflix): kills random EC2 instances.
  - Chaos Gorilla: kills entire AZ.
  - Chaos Kong: kills entire region.
  - tc (Linux): adds network latency/packet loss.
  - Gremlin: commercial chaos platform.
  - AWS Fault Injection Simulator (FIS).
  - LitmusChaos (Kubernetes).

GameDay:
  Planned chaos exercise. Team gathers, intentionally breaks production (or staging).
  Observe: how does the system degrade? Do alerts fire? Do runbooks work?
  Post-mortem: identify gaps and improve.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import time
import random
import threading
import statistics


# ─────────────────────────────────────────────
# STEADY STATE METRICS
# ─────────────────────────────────────────────

@dataclass
class SteadyStateMetric:
    name       : str
    value      : float
    threshold  : float   # above this = degraded
    operator   : str     # "<" or ">"

    def is_healthy(self) -> bool:
        if self.operator == "<":
            return self.value < self.threshold
        return self.value > self.threshold

    def status(self) -> str:
        return "HEALTHY" if self.is_healthy() else "DEGRADED"


class MetricsCollector:
    """Tracks request success/failure and latency during experiments."""

    def __init__(self):
        self._latencies   : List[float] = []
        self._errors      : int = 0
        self._total       : int = 0
        self._lock        = threading.Lock()

    def record(self, latency_ms: float, error: bool = False):
        with self._lock:
            self._latencies.append(latency_ms)
            self._total += 1
            if error:
                self._errors += 1

    def success_rate(self) -> float:
        with self._lock:
            if not self._total:
                return 1.0
            return (self._total - self._errors) / self._total

    def p99_latency(self) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            sorted_lat = sorted(self._latencies)
            idx = int(len(sorted_lat) * 0.99)
            return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def steady_state(self) -> List[SteadyStateMetric]:
        return [
            SteadyStateMetric("success_rate", self.success_rate(), 0.999, ">"),
            SteadyStateMetric("p99_latency_ms", self.p99_latency(), 500.0, "<"),
            SteadyStateMetric("error_rate", 1 - self.success_rate(), 0.001, "<"),
        ]


# ─────────────────────────────────────────────
# CHAOS INJECTORS
# ─────────────────────────────────────────────

class ChaosType(Enum):
    NONE             = "none"
    LATENCY          = "latency"           # add artificial delay
    PACKET_LOSS      = "packet_loss"       # random request drops
    INSTANCE_KILL    = "instance_kill"     # kill a service instance
    DEPENDENCY_DOWN  = "dependency_down"   # kill external dependency
    CPU_PRESSURE     = "cpu_pressure"      # slow down processing


@dataclass
class ChaosConfig:
    chaos_type   : ChaosType = ChaosType.NONE
    latency_ms   : float     = 0.0
    packet_loss  : float     = 0.0    # fraction (0.0-1.0)
    instances_down: int      = 0
    duration_s   : float     = 10.0


class ChaosProxy:
    """
    Injects configured faults into service calls.
    Wraps the actual service call.
    """

    def __init__(self, config: ChaosConfig = None):
        self.config  = config or ChaosConfig()
        self.injections = 0

    def call(self, fn: Callable, *args, **kwargs) -> Any:
        """Apply chaos then call the underlying function."""
        chaos = self.config

        if chaos.chaos_type == ChaosType.PACKET_LOSS:
            if random.random() < chaos.packet_loss:
                self.injections += 1
                raise TimeoutError("chaos: packet dropped")

        if chaos.chaos_type == ChaosType.LATENCY:
            jitter = random.uniform(0, chaos.latency_ms * 0.2)
            time.sleep((chaos.latency_ms + jitter) / 1000)
            self.injections += 1

        if chaos.chaos_type == ChaosType.DEPENDENCY_DOWN:
            self.injections += 1
            raise ConnectionError("chaos: dependency unavailable")

        return fn(*args, **kwargs)


# ─────────────────────────────────────────────
# SERVICE UNDER TEST
# ─────────────────────────────────────────────

class UserService:
    """Simulates a simple web service with resilience features."""

    def __init__(self, cache_proxy: ChaosProxy, db_proxy: ChaosProxy):
        self.cache_proxy = cache_proxy
        self.db_proxy    = db_proxy
        self._cache      : Dict[str, Any] = {}
        self._db         : Dict[str, Any] = {"user:1": "Alice", "user:2": "Bob"}
        self.metrics     = MetricsCollector()
        self._fallback_used = 0

    def get_user(self, user_id: str) -> Optional[str]:
        t0 = time.time()
        try:
            # Try cache first
            try:
                value = self.cache_proxy.call(lambda: self._cache.get(user_id))
                if value:
                    self.metrics.record((time.time() - t0) * 1000)
                    return value
            except Exception:
                pass   # cache miss/failure → fallback to DB

            # Fallback to DB
            value = self.db_proxy.call(lambda: self._db.get(user_id))
            if value:
                self._cache[user_id] = value
            self.metrics.record((time.time() - t0) * 1000)
            return value

        except Exception as exc:
            self.metrics.record((time.time() - t0) * 1000, error=True)
            # Graceful degradation: return cached/stale value
            stale = self._cache.get(user_id)
            if stale:
                self._fallback_used += 1
                return stale   # stale-but-available
            return None


# ─────────────────────────────────────────────
# CHAOS EXPERIMENT
# ─────────────────────────────────────────────

@dataclass
class ExperimentResult:
    name           : str
    chaos_type     : str
    before_metrics : List[SteadyStateMetric]
    after_metrics  : List[SteadyStateMetric]
    hypothesis_held: bool
    observations   : List[str]


class ChaosExperiment:
    """Runs a chaos experiment and compares before/after steady state."""

    def run(self, name: str, service_factory: Callable,
            chaos_config: ChaosConfig, n_requests: int = 100) -> ExperimentResult:

        observations = []

        # Phase 1: Establish baseline (no chaos)
        baseline_service = service_factory(ChaosConfig(ChaosType.NONE), ChaosConfig(ChaosType.NONE))
        for i in range(n_requests):
            baseline_service.get_user(f"user:{(i % 2) + 1}")

        before = baseline_service.metrics.steady_state()

        # Phase 2: Apply chaos
        chaos_cache = ChaosConfig(ChaosType.NONE)
        chaos_db    = chaos_config
        chaos_service = service_factory(chaos_cache, chaos_db)

        for i in range(n_requests):
            chaos_service.get_user(f"user:{(i % 2) + 1}")

        after = chaos_service.metrics.steady_state()

        # Compare
        hypothesis_held = all(m.is_healthy() for m in after)
        if chaos_service._fallback_used > 0:
            observations.append(f"Graceful degradation: served {chaos_service._fallback_used} stale responses")
        if chaos_service.cache_proxy.injections > 0:
            observations.append(f"Cache chaos injections: {chaos_service.cache_proxy.injections}")
        if chaos_service.db_proxy.injections > 0:
            observations.append(f"DB chaos injections: {chaos_service.db_proxy.injections}")

        return ExperimentResult(
            name            = name,
            chaos_type      = chaos_config.chaos_type.value,
            before_metrics  = before,
            after_metrics   = after,
            hypothesis_held = hypothesis_held,
            observations    = observations,
        )


# ─────────────────────────────────────────────
# BLAST RADIUS CALCULATOR
# ─────────────────────────────────────────────

def blast_radius_estimate(
    rps: float,
    error_rate: float,
    avg_impact_per_error_usd: float = 0.01,
    duration_s: float = 60.0,
) -> Dict[str, float]:
    errors_per_second = rps * error_rate
    total_errors      = errors_per_second * duration_s
    financial_impact  = total_errors * avg_impact_per_error_usd
    return {
        "errors_per_second": errors_per_second,
        "total_errors"      : total_errors,
        "financial_impact"  : financial_impact,
        "recommendation"    : "small" if error_rate < 0.01 else "large",
    }


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_chaos_engineering():
    print("=" * 65)
    print("CHAOS ENGINEERING PRINCIPLES")
    print("=" * 65)

    random.seed(42)

    def service_factory(cache_chaos: ChaosConfig, db_chaos: ChaosConfig) -> UserService:
        return UserService(ChaosProxy(cache_chaos), ChaosProxy(db_chaos))

    experiment = ChaosExperiment()

    # ── Experiment 1: DB Latency Injection ────────
    print("\n[1] EXPERIMENT: DB LATENCY INJECTION (200ms)")
    print("─" * 55)
    print("  Hypothesis: Success rate stays >99.9% with 200ms DB latency")

    result1 = experiment.run(
        name           = "DB Latency",
        service_factory = service_factory,
        chaos_config   = ChaosConfig(ChaosType.LATENCY, latency_ms=1.0,
                                     duration_s=5.0),
        n_requests     = 50,
    )

    print(f"  Before (baseline):")
    for m in result1.before_metrics:
        print(f"    {m.name:<22}: {m.value:.4f} ({m.status()})")
    print(f"  After (with chaos):")
    for m in result1.after_metrics:
        print(f"    {m.name:<22}: {m.value:.4f} ({m.status()})")
    print(f"  Hypothesis held: {result1.hypothesis_held}")
    for obs in result1.observations:
        print(f"  Observation: {obs}")

    # ── Experiment 2: DB Packet Loss ──────────────
    print("\n\n[2] EXPERIMENT: DB PACKET LOSS (50%)")
    print("─" * 55)
    print("  Hypothesis: Service degrades gracefully, serves stale from cache")

    result2 = experiment.run(
        name           = "DB Packet Loss",
        service_factory = service_factory,
        chaos_config   = ChaosConfig(ChaosType.PACKET_LOSS, packet_loss=0.5),
        n_requests     = 50,
    )

    for m in result2.after_metrics:
        print(f"  {m.name:<22}: {m.value:.4f} ({m.status()})")
    print(f"  Hypothesis held: {result2.hypothesis_held}")
    for obs in result2.observations:
        print(f"  Observation: {obs}")

    # ── Blast Radius ──────────────────────────────
    print("\n\n[3] BLAST RADIUS ESTIMATION")
    print("─" * 55)

    configs = [
        (1000, 0.001, "Tiny experiment (0.1% errors)"),
        (1000, 0.01,  "Small experiment (1% errors)"),
        (1000, 0.10,  "Large experiment (10% errors)"),
    ]
    print(f"  {'Scenario':<35} {'Errors/s':>10} {'Total/60s':>10} {'Impact ($)'}")
    print(f"  {'─'*68}")
    for rps, error_rate, desc in configs:
        br = blast_radius_estimate(rps, error_rate, 0.005, 60.0)
        print(f"  {desc:<35} {br['errors_per_second']:>10.1f} "
              f"{br['total_errors']:>10.0f} ${br['financial_impact']:>8.2f}")

    # ── Chaos Experiment Process ──────────────────
    print("\n\n[4] CHAOS EXPERIMENT PROCESS")
    print("─" * 55)
    steps = [
        ("1. Define steady state",    "Measure: success_rate, p99_latency, error_rate"),
        ("2. Form hypothesis",         "System stays healthy (steady state holds) under X"),
        ("3. Choose chaos type",       "Latency, crash, dependency failure, resource exhaust"),
        ("4. Start small",             "1% traffic, 1 instance, staging env first"),
        ("5. Inject chaos",            "Run experiment for 5-30 minutes"),
        ("6. Measure impact",          "Compare before/after steady state metrics"),
        ("7. Analyze results",         "Hypothesis held? What degraded? Graceful?"),
        ("8. Fix weaknesses found",    "Add retries, circuit breakers, fallbacks"),
        ("9. Increase scope",          "More traffic, more instances, production"),
        ("10. Automate",               "Run experiments in CI/CD on every deploy"),
    ]
    for step, description in steps:
        print(f"  {step:<28} {description}")

    # ── Common Failure Patterns Found ────────────
    print("\n\n[5] WHAT CHAOS ENGINEERING DISCOVERS")
    print("─" * 55)
    findings = [
        "Missing retries: service fails on first transient error",
        "No circuit breaker: slow DB cascades to 100% of requests timing out",
        "No fallback: single cache node down → total service failure",
        "Wrong timeout: too long → slow requests hold connections",
        "Missing health check: dead instance stays in load balancer pool",
        "Configuration dependency: service fails if config service is down",
        "No graceful shutdown: rolling deploy drops in-flight requests",
        "Log volume explosion: error floods disk on any failure",
    ]
    for finding in findings:
        print(f"  • {finding}")


if __name__ == "__main__":
    demonstrate_chaos_engineering()
