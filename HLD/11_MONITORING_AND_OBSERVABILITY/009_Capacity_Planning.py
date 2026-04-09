"""
CAPACITY PLANNING
==================

Problem Statement:
Running out of capacity causes outages. Over-provisioning wastes money.
Capacity planning predicts future resource needs and triggers scaling
decisions before headroom runs out.

Key Resources to Plan:
  Compute: CPU cores, memory, GPU.
  Storage: disk IOPS, throughput, total capacity.
  Network: bandwidth, connection count, NAT table.
  Database: connections, query QPS, replication lag, storage growth.
  Queue:    message backlog, consumer throughput, lag.

Capacity Planning Process:
  1. Baseline:     Measure current utilization (CPU%, RPS, storage GB).
  2. Growth model: Project future demand (linear, exponential, seasonal).
  3. Headroom:     Target utilization = 70%. Trigger scaling at 80%.
  4. Runways:      At current growth rate, how long until we hit capacity?
  5. Rightsizing:  Identify underutilized resources; downsize to save cost.

Load Testing:
  Before scaling, know the system's breaking point.
  Types:
    Soak test:     Sustained load at 100% expected traffic for hours.
    Stress test:   Ramp up until failure; find breaking point.
    Spike test:    Sudden 10x load; measures elasticity.
    Breakpoint:    Find exact RPS where p99 > SLO or errors > threshold.

Little's Law:
  L = λ × W
  L = average number of items in system (queue length / active requests)
  λ = arrival rate (RPS)
  W = average time in system (latency)
  Use: if latency increases, queue grows proportionally.

Amdahl's Law:
  Speedup = 1 / (1 - p + p/n)
  p = parallelizable fraction, n = processors.
  Limits horizontal scaling: 95% parallel → max 20× speedup.

USL (Universal Scalability Law):
  Adds contention (σ) and coherence (κ) terms to Amdahl.
  Models when adding nodes hurts throughput (too much coordination).
"""

from __future__ import annotations

import math
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ─────────────────────────────────────────────
# RESOURCE UTILIZATION SNAPSHOT
# ─────────────────────────────────────────────

@dataclass
class ResourceSnapshot:
    timestamp:    float
    cpu_pct:      float
    memory_pct:   float
    disk_pct:     float
    rps:          float
    p99_latency_ms: float
    active_conns: int


# ─────────────────────────────────────────────
# GROWTH MODEL
# ─────────────────────────────────────────────

class GrowthModel(Enum):
    LINEAR      = "linear"
    EXPONENTIAL = "exponential"
    SEASONAL    = "seasonal"


@dataclass
class GrowthFit:
    model:       GrowthModel
    params:      Dict[str, float]   # model-specific params
    r_squared:   float              # goodness of fit

    def predict(self, future_days: int, base_value: float) -> float:
        if self.model == GrowthModel.LINEAR:
            slope = self.params.get("slope", 0.0)
            return base_value + slope * future_days
        elif self.model == GrowthModel.EXPONENTIAL:
            daily_rate = self.params.get("daily_rate", 0.0)
            return base_value * ((1 + daily_rate) ** future_days)
        elif self.model == GrowthModel.SEASONAL:
            # Base linear + seasonal adjustment
            slope     = self.params.get("slope", 0.0)
            amplitude = self.params.get("amplitude", 0.0)
            period    = self.params.get("period", 365.0)
            trend     = base_value + slope * future_days
            seasonal  = amplitude * math.sin(2 * math.pi * future_days / period)
            return trend + seasonal
        return base_value


class GrowthAnalyzer:
    """Fits growth models to historical metric data."""

    def fit_linear(self, values: List[float]) -> GrowthFit:
        """Simple linear regression: value = a + b * t."""
        n   = len(values)
        if n < 2:
            return GrowthFit(GrowthModel.LINEAR, {"slope": 0.0}, 0.0)
        xs  = list(range(n))
        mx  = sum(xs) / n
        my  = sum(values) / n
        num = sum((xs[i] - mx) * (values[i] - my) for i in range(n))
        den = sum((xs[i] - mx) ** 2 for i in range(n))
        b   = num / den if den > 0 else 0.0
        a   = my - b * mx

        # R²
        ss_res = sum((values[i] - (a + b * xs[i])) ** 2 for i in range(n))
        ss_tot = sum((values[i] - my) ** 2 for i in range(n))
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return GrowthFit(GrowthModel.LINEAR, {"intercept": a, "slope": b}, r2)

    def fit_exponential(self, values: List[float]) -> GrowthFit:
        """Fit y = a * e^(b*t) using linear regression on log(y)."""
        log_vals = []
        for v in values:
            if v > 0:
                log_vals.append(math.log(v))
            else:
                log_vals.append(0.0)
        linear = self.fit_linear(log_vals)
        b      = linear.params["slope"]
        daily_rate = math.exp(b) - 1
        return GrowthFit(GrowthModel.EXPONENTIAL,
                         {"daily_rate": daily_rate, "slope": b},
                         linear.r_squared)

    def runway_days(self, current: float, capacity: float,
                    fit: GrowthFit, max_days: int = 730) -> Optional[int]:
        """How many days until `current` (projected) hits `capacity`?"""
        for d in range(1, max_days + 1):
            projected = fit.predict(d, current)
            if projected >= capacity:
                return d
        return None   # won't hit in max_days


# ─────────────────────────────────────────────
# CAPACITY HEADROOM
# ─────────────────────────────────────────────

@dataclass
class HeadroomStatus:
    resource:      str
    current_pct:   float
    target_pct:    float   # desired utilization
    alert_pct:     float   # trigger scaling action
    runway_days:   Optional[int]
    action:        str

    @property
    def is_critical(self) -> bool:
        return self.current_pct >= self.alert_pct


class CapacityTracker:
    """Tracks headroom for a resource."""

    def __init__(self, resource: str, capacity: float,
                 target_util: float = 0.70, alert_util: float = 0.80):
        self.resource    = resource
        self.capacity    = capacity
        self._target     = target_util
        self._alert      = alert_util
        self._history:   List[float] = []   # historical utilization values
        self._analyzer   = GrowthAnalyzer()

    def record(self, value: float):
        self._history.append(value)

    def status(self) -> HeadroomStatus:
        if not self._history:
            return HeadroomStatus(self.resource, 0.0, self._target,
                                  self._alert, None, "no data")

        current_pct = self._history[-1] / self.capacity
        runway      = None
        action      = "ok"

        if len(self._history) >= 7:
            fit    = self._analyzer.fit_linear(self._history)
            runway = self._analyzer.runway_days(
                self._history[-1], self.capacity * self._alert, fit)

        if current_pct >= self._alert:
            action = "SCALE NOW"
        elif runway is not None and runway < 30:
            action = f"Scale within {runway}d"
        elif current_pct < 0.30 and len(self._history) > 30:
            action = "Downsize (underutilized)"

        return HeadroomStatus(
            self.resource, current_pct, self._target, self._alert,
            runway, action,
        )


# ─────────────────────────────────────────────
# LITTLE'S LAW CALCULATOR
# ─────────────────────────────────────────────

class LittlesLaw:
    """L = λ × W"""

    @staticmethod
    def queue_length(arrival_rate: float, avg_latency_s: float) -> float:
        """Average number of requests in system."""
        return arrival_rate * avg_latency_s

    @staticmethod
    def max_rps_for_latency(target_latency_s: float,
                            max_concurrency: int) -> float:
        """Max arrival rate for a given latency and concurrency limit."""
        return max_concurrency / target_latency_s

    @staticmethod
    def latency_from_load(arrival_rate: float,
                          service_time_s: float,
                          n_workers: int) -> float:
        """
        M/M/c queue: approximate average response time.
        Simplified: service_time / (1 - ρ) where ρ = λ/(μ*c).
        """
        mu  = 1.0 / service_time_s     # service rate per worker
        rho = arrival_rate / (mu * n_workers)   # utilization
        if rho >= 1.0:
            return float("inf")   # queued → unbounded
        return service_time_s / (1 - rho)


# ─────────────────────────────────────────────
# AMDAHL'S LAW
# ─────────────────────────────────────────────

class AmdahlsLaw:
    """
    Theoretical speedup with n parallel processors.
    p = fraction of task that is parallelizable.
    """

    @staticmethod
    def speedup(p: float, n: int) -> float:
        return 1.0 / ((1 - p) + p / n)

    @staticmethod
    def max_speedup(p: float) -> float:
        """Limit as n → ∞."""
        return 1.0 / (1 - p) if p < 1.0 else float("inf")


# ─────────────────────────────────────────────
# UNIVERSAL SCALABILITY LAW (USL)
# ─────────────────────────────────────────────

class USL:
    """
    Throughput(N) = N / (1 + σ(N-1) + κN(N-1))
    σ = contention coefficient
    κ = coherence coefficient
    N = number of nodes
    Normalised: throughput at N=1 = 1.
    """

    def __init__(self, sigma: float, kappa: float):
        self._sigma = sigma
        self._kappa = kappa

    def throughput(self, n: int) -> float:
        return n / (1 + self._sigma * (n - 1) + self._kappa * n * (n - 1))

    def optimal_n(self, max_n: int = 100) -> int:
        """Find N that maximises throughput."""
        best_n = 1
        best_t = self.throughput(1)
        for n in range(2, max_n + 1):
            t = self.throughput(n)
            if t > best_t:
                best_t = t
                best_n = n
            elif t < best_t:
                break
        return best_n


# ─────────────────────────────────────────────
# LOAD TEST SIMULATOR
# ─────────────────────────────────────────────

@dataclass
class LoadTestResult:
    rps:            float
    p50_ms:         float
    p99_ms:         float
    error_rate:     float
    cpu_pct:        float

    @property
    def is_acceptable(self) -> bool:
        return self.p99_ms < 500 and self.error_rate < 0.01


def simulate_load_test(service_time_ms: float,
                       n_workers: int,
                       rps_levels: List[float]) -> List[LoadTestResult]:
    """
    Simulates load test using M/M/c queue model.
    Errors start when CPU > 90% (simplified).
    """
    results = []
    for rps in rps_levels:
        latency_s   = LittlesLaw.latency_from_load(rps, service_time_ms / 1000, n_workers)
        p99_ms      = min(latency_s * 1000 * 3.0, 30000)   # p99 ~ 3x avg (simplified)
        p50_ms      = latency_s * 1000
        cpu         = min(rps * service_time_ms / 1000 / n_workers, 1.0) * 100
        error_rate  = max(0.0, (cpu - 90) / 100) if cpu > 90 else 0.0
        results.append(LoadTestResult(rps, p50_ms, p99_ms, error_rate, cpu))
    return results


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_capacity():
    print("=" * 65)
    print("CAPACITY PLANNING")
    print("=" * 65)

    random.seed(42)

    # ── Historical Growth Data ─────────────────
    print("\n[1] GROWTH ANALYSIS (30-day history)")
    print("─" * 55)

    analyzer = GrowthAnalyzer()

    # Simulate 30 days of daily RPS measurements (growing ~5%/week)
    rps_history = []
    base = 500.0
    for d in range(30):
        weekly_growth = 1.007   # ~5%/week = ~1%/day * 0.7
        value = base * (weekly_growth ** d) + random.gauss(0, 10)
        rps_history.append(max(0, value))

    linear_fit = analyzer.fit_linear(rps_history)
    exp_fit    = analyzer.fit_exponential(rps_history)

    print(f"  30-day RPS: {rps_history[0]:.0f} → {rps_history[-1]:.0f}")
    print(f"  Linear fit:      slope={linear_fit.params['slope']:.2f} RPS/day  "
          f"R²={linear_fit.r_squared:.3f}")
    print(f"  Exponential fit: daily_rate={exp_fit.params['daily_rate']*100:.2f}%  "
          f"R²={exp_fit.r_squared:.3f}")

    # Runway to capacity (2000 RPS)
    capacity = 2000.0
    runway_l = analyzer.runway_days(rps_history[-1], capacity * 0.80, linear_fit)
    runway_e = analyzer.runway_days(rps_history[-1], capacity * 0.80, exp_fit)
    print(f"\n  Capacity = {capacity:.0f} RPS  |  Alert at 80% = {capacity*0.8:.0f} RPS")
    print(f"  Linear  runway: {runway_l}d")
    print(f"  Exp     runway: {runway_e}d")

    # ── Resource Headroom ─────────────────────
    print("\n[2] RESOURCE HEADROOM DASHBOARD")
    print("─" * 55)

    trackers = {
        "cpu_cores":  CapacityTracker("CPU",     100,  0.70, 0.80),
        "memory_gb":  CapacityTracker("Memory",  512,  0.70, 0.80),
        "disk_gb":    CapacityTracker("Disk",    10000,0.70, 0.85),
        "db_conns":   CapacityTracker("DB Conn", 500,  0.60, 0.75),
    }

    # Simulate 30 days of usage
    usage = {
        "cpu_cores": [40 + random.gauss(0, 2) + i * 0.8 for i in range(30)],
        "memory_gb": [180 + random.gauss(0, 5) + i * 1.5 for i in range(30)],
        "disk_gb":   [4000 + i * 60  for i in range(30)],
        "db_conns":  [350 + random.gauss(0, 10) + i * 2 for i in range(30)],
    }

    for key, tracker in trackers.items():
        for v in usage[key]:
            tracker.record(v)

    print(f"  {'Resource':<12} {'Current%':>10} {'Target%':>10} {'Alert%':>10} "
          f"{'Runway':>8}  {'Action'}")
    print("  " + "─" * 70)
    for key, tracker in trackers.items():
        s = tracker.status()
        runway = f"{s.runway_days}d" if s.runway_days else "N/A"
        flag   = " ← WARN" if s.is_critical else ""
        print(f"  {s.resource:<12} {s.current_pct*100:>9.1f}% {s.target_pct*100:>9.0f}% "
              f"{s.alert_pct*100:>9.0f}% {runway:>8}  {s.action}{flag}")

    # ── Little's Law ──────────────────────────
    print("\n[3] LITTLE'S LAW ANALYSIS")
    print("─" * 55)

    print("  L = λ × W  (avg in system = arrival rate × avg latency)")
    for rps, lat_s in [(100, 0.05), (500, 0.05), (500, 0.20), (1000, 0.50)]:
        L = LittlesLaw.queue_length(rps, lat_s)
        print(f"  λ={rps:>5} RPS, W={lat_s*1000:.0f}ms → L={L:.0f} concurrent requests")

    max_rps = LittlesLaw.max_rps_for_latency(0.200, 50)
    print(f"\n  Max RPS to keep latency ≤ 200ms with 50 workers: {max_rps:.0f} RPS")

    # M/M/c queue: latency vs workers
    print("\n  Latency vs workers (arrival=800 RPS, service_time=50ms):")
    for workers in [10, 20, 40, 50, 60]:
        lat = LittlesLaw.latency_from_load(800, 0.050, workers)
        if lat == float("inf"):
            print(f"    workers={workers:>4}: SATURATED")
        else:
            print(f"    workers={workers:>4}: avg_latency={lat*1000:.1f}ms")

    # ── Amdahl's Law ──────────────────────────
    print("\n[4] AMDAHL'S LAW")
    print("─" * 55)

    print(f"  {'p (parallel %)':<20} {'n=2':>6} {'n=4':>6} {'n=8':>6} "
          f"{'n=16':>6} {'∞':>8}")
    print("  " + "─" * 54)
    for p in [0.50, 0.80, 0.90, 0.95, 0.99]:
        speedups = [f"{AmdahlsLaw.speedup(p, n):.2f}×" for n in [2, 4, 8, 16]]
        max_s    = AmdahlsLaw.max_speedup(p)
        print(f"  {p*100:.0f}%{'':<16} "
              + "  ".join(f"{s:>6}" for s in speedups)
              + f"  {max_s:>6.1f}×")

    # ── USL ───────────────────────────────────
    print("\n[5] UNIVERSAL SCALABILITY LAW")
    print("─" * 55)

    scenarios = [
        ("Stateless API",   0.01, 0.0001),
        ("Shared DB lock",  0.10, 0.001),
        ("Heavy coord",     0.20, 0.010),
    ]
    print(f"  {'Scenario':<18} {'σ':>6} {'κ':>8} {'Optimal N':>10} "
          f"{'Throughput at opt':>18}")
    print("  " + "─" * 65)
    for name, sigma, kappa in scenarios:
        usl   = USL(sigma, kappa)
        opt_n = usl.optimal_n()
        opt_t = usl.throughput(opt_n)
        print(f"  {name:<18} {sigma:>6.2f} {kappa:>8.4f} {opt_n:>10}  {opt_t:>16.2f}×")

    # ── Load Test Simulation ──────────────────
    print("\n[6] LOAD TEST RESULTS")
    print("─" * 55)

    rps_levels  = [100, 200, 400, 600, 800, 1000, 1200]
    test_results = simulate_load_test(
        service_time_ms=20,
        n_workers=20,
        rps_levels=rps_levels,
    )

    print(f"  Service: 20ms per request, 20 worker threads")
    print(f"  {'RPS':>6} {'p50 (ms)':>10} {'p99 (ms)':>10} {'Error%':>8} "
          f"{'CPU%':>6}  {'OK?'}")
    print("  " + "─" * 55)
    for r in test_results:
        ok   = "✓" if r.is_acceptable else "✗"
        flag = " ← BREAKPOINT" if not r.is_acceptable and test_results[test_results.index(r)-1].is_acceptable else ""
        print(f"  {r.rps:>6.0f} {r.p50_ms:>10.1f} {r.p99_ms:>10.1f} "
              f"{r.error_rate*100:>7.2f}% {r.cpu_pct:>5.1f}%  {ok}{flag}")

    # ── Capacity Planning Checklist ───────────
    print("\n[7] CAPACITY PLANNING CHECKLIST")
    print("─" * 55)

    checklist = [
        "Baseline current utilization for all 4 golden resources",
        "Fit growth model (linear vs exponential) with R² > 0.9",
        "Compute runway to 80% utilization under each model",
        "Schedule scaling action when runway < 30 days",
        "Run load test to find breaking point before every major launch",
        "Set auto-scaling triggers at 70% CPU (not 90%)",
        "Right-size: downscale resources with < 30% utilization",
        "Plan for 3× peak headroom for traffic spikes",
    ]
    for i, item in enumerate(checklist, 1):
        print(f"  {i}. {item}")


if __name__ == "__main__":
    demonstrate_capacity()
