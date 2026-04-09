"""
SLA / SLO / SLI CONCEPTS
==========================

Problem Statement:
Reliability engineering requires precise language to define, measure, and
enforce service quality. SLIs, SLOs, and SLAs form the vocabulary used by
teams to set expectations and track reliability in production systems.

Definitions:
- SLI (Service Level Indicator) : What you MEASURE  (e.g., "% of requests < 200ms")
- SLO (Service Level Objective) : Your TARGET        (e.g., "99.9% of requests < 200ms")
- SLA (Service Level Agreement) : CONTRACTUAL target  (SLO - buffer; has penalties)
- Error Budget  : 1 - SLO target. How much unreliability you're allowed.
                  If SLO = 99.9%, error budget = 0.1% → 43.8 min/month

Error Budget Usage:
  Error budget rate = (1 - current_availability) / (1 - SLO_target)
  Rate > 1 → burning budget faster than allowed → freeze deployments / investigate
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
import random


class SLIType(Enum):
    AVAILABILITY  = "availability"
    LATENCY       = "latency"
    ERROR_RATE    = "error_rate"
    THROUGHPUT    = "throughput"
    FRESHNESS     = "freshness"


@dataclass
class SLI:
    name           : str
    sli_type       : SLIType
    measurement    : str        # human description of what is measured
    current_value  : float      # measured value (0–100 for %)
    unit           : str = "%"

    def report(self):
        print(f"  SLI [{self.name}]: {self.current_value:.4f}{self.unit}  ({self.measurement})")


@dataclass
class SLO:
    name          : str
    sli           : SLI
    target        : float       # e.g. 99.9 (percent)
    window_days   : int = 30

    @property
    def is_met(self) -> bool:
        return self.sli.current_value >= self.target

    @property
    def error_budget_pct(self) -> float:
        return 100.0 - self.target           # e.g. 0.1 for 99.9%

    @property
    def error_budget_minutes_per_month(self) -> float:
        return self.error_budget_pct / 100.0 * 43_800  # minutes in 30 days

    @property
    def remaining_budget_pct(self) -> float:
        if self.sli.current_value >= 100.0:
            return 100.0
        burned = 100.0 - self.sli.current_value
        allowed = 100.0 - self.target
        if allowed == 0:
            return 0.0
        return max(0.0, 100.0 - (burned / allowed * 100.0))

    def report(self):
        status = "✅ MET" if self.is_met else "❌ BREACHED"
        print(f"\n  SLO [{self.name}]  {status}")
        print(f"    Target  : {self.target}%")
        print(f"    Current : {self.sli.current_value:.4f}%")
        print(f"    Error budget       : {self.error_budget_pct:.3f}%  "
              f"({self.error_budget_minutes_per_month:.1f} min/month)")
        print(f"    Budget remaining   : {self.remaining_budget_pct:.1f}%")


@dataclass
class SLA:
    name          : str
    slos          : List[SLO]
    customer_tier : str
    penalty_clause: str

    def all_met(self) -> bool:
        return all(s.is_met for s in self.slos)

    def report(self):
        status = "✅ SLA COMPLIANT" if self.all_met() else "❌ SLA VIOLATED"
        print(f"\n  SLA [{self.name}] — {self.customer_tier} tier  →  {status}")
        print(f"    Penalty : {self.penalty_clause}")
        for slo in self.slos:
            flag = "✅" if slo.is_met else "❌"
            print(f"    {flag} {slo.name:<30} target={slo.target}%  actual={slo.sli.current_value:.4f}%")


class ErrorBudgetMonitor:
    """
    Tracks error budget burn rate over a rolling window.
    Alerts if budget is burning too fast (fast-burn alert).
    """

    def __init__(self, slo: SLO):
        self.slo      = slo
        self.history  : List[float] = []  # per-tick availability measurements

    def record(self, availability_pct: float):
        self.history.append(availability_pct)

    def burn_rate(self, window: int = 60) -> float:
        """
        Burn rate = (1 - actual_availability) / (1 - SLO_target)
        Rate > 1 means burning budget faster than allowed.
        """
        recent = self.history[-window:] if len(self.history) >= window else self.history
        if not recent:
            return 0.0
        avg_avail  = sum(recent) / len(recent) / 100.0
        slo_target = self.slo.target / 100.0
        if slo_target >= 1.0:
            return 0.0
        return (1 - avg_avail) / (1 - slo_target)

    def alert_status(self, short_window: int = 5, long_window: int = 60) -> str:
        fast = self.burn_rate(short_window)
        slow = self.burn_rate(long_window)
        if fast > 14.4:   # would exhaust 30-day budget in 2 hours
            return f"🔴 CRITICAL fast-burn  (rate={fast:.1f}x) — page NOW"
        if fast > 6.0:    # would exhaust in 5 hours
            return f"🟠 HIGH fast-burn  (rate={fast:.1f}x) — alert team"
        if slow > 3.0:    # consuming budget 3x faster than allowed
            return f"🟡 MODERATE slow-burn  (rate={slow:.1f}x) — ticket + review"
        return f"🟢 Normal burn rate  (fast={fast:.2f}x, slow={slow:.2f}x)"

    def report(self):
        budget_remaining = self.slo.remaining_budget_pct
        print(f"\n  Error Budget Monitor [{self.slo.name}]:")
        print(f"    Measurements : {len(self.history)}")
        if self.history:
            avg = sum(self.history) / len(self.history)
            print(f"    Avg avail    : {avg:.4f}%")
        print(f"    Budget left  : {budget_remaining:.1f}%")
        print(f"    Burn status  : {self.alert_status()}")


def demonstrate_sla_slo_sli_concepts():
    print("=" * 65)
    print("SLA / SLO / SLI CONCEPTS")
    print("System: Payment API Service")
    print("=" * 65)

    # ── Define SLIs ───────────────────────────
    print("\n[1] SERVICE LEVEL INDICATORS (SLIs)")
    print("─" * 50)
    sli_avail    = SLI("availability",  SLIType.AVAILABILITY, "% requests returning 2xx/3xx", 99.94)
    sli_latency  = SLI("p99_latency",   SLIType.LATENCY,      "% requests completing < 200ms", 99.85)
    sli_errors   = SLI("error_rate",    SLIType.ERROR_RATE,   "% requests NOT resulting in 5xx", 99.97)
    sli_fresh     = SLI("data_freshness",SLIType.FRESHNESS,   "% reads returning data < 5s old", 99.80)

    for sli in [sli_avail, sli_latency, sli_errors, sli_fresh]:
        sli.report()

    # ── Define SLOs ───────────────────────────
    print("\n\n[2] SERVICE LEVEL OBJECTIVES (SLOs)")
    print("─" * 50)
    slo_avail   = SLO("Availability SLO",  sli_avail,   target=99.95, window_days=30)
    slo_latency = SLO("Latency SLO",       sli_latency, target=99.9,  window_days=30)
    slo_errors  = SLO("Error Rate SLO",    sli_errors,  target=99.9,  window_days=30)
    slo_fresh   = SLO("Freshness SLO",     sli_fresh,   target=99.5,  window_days=30)

    for slo in [slo_avail, slo_latency, slo_errors, slo_fresh]:
        slo.report()

    # ── Define SLAs ───────────────────────────
    print("\n\n[3] SERVICE LEVEL AGREEMENTS (SLAs)")
    print("─" * 50)
    sla_enterprise = SLA(
        "Enterprise API SLA",
        slos=[slo_avail, slo_latency, slo_errors],
        customer_tier="Enterprise",
        penalty_clause="10% monthly bill credit per violated SLO"
    )
    sla_free = SLA(
        "Free Tier SLA",
        slos=[slo_avail],
        customer_tier="Free",
        penalty_clause="Best effort — no credit"
    )
    sla_enterprise.report()
    sla_free.report()

    # ── Error Budget Monitoring ───────────────
    print("\n\n[4] ERROR BUDGET BURN RATE MONITORING")
    print("─" * 50)
    monitor = ErrorBudgetMonitor(slo_avail)

    # Simulate 30 days: mostly healthy, with a 2-hour incident on day 10
    random.seed(42)
    for day in range(30):
        for _ in range(24):   # 1 measurement per hour
            if day == 9 and _ < 2:
                monitor.record(95.0)   # incident: 5% error rate for 2 hours
            elif day == 20 and _ < 1:
                monitor.record(98.0)   # minor blip
            else:
                monitor.record(99.96 + random.uniform(-0.05, 0.04))

    monitor.report()

    # ── Fast-burn scenario ────────────────────
    print("\n\n[5] FAST-BURN ALERT SIMULATION")
    print("─" * 50)
    fast_burn_monitor = ErrorBudgetMonitor(slo_avail)
    print("  Injecting outage: 15% error rate for 10 minutes…")
    for _ in range(10):
        fast_burn_monitor.record(85.0)
    for _ in range(50):
        fast_burn_monitor.record(99.97)
    fast_burn_monitor.report()

    # ── Error Budget Policy ───────────────────
    print("\n\n[6] ERROR BUDGET POLICY")
    print("─" * 50)
    policy_rows = [
        (">100%", "Budget fully consumed", "🔴 Freeze all releases; only reliability fixes"),
        ("75–100%","Budget nearly gone",   "🟠 No new features; focus on reliability"),
        ("50–75%", "Budget at risk",       "🟡 Review risky deployments; add tests"),
        ("<50%",   "Healthy budget",       "🟢 Normal development velocity"),
    ]
    print(f"  {'Budget Burned':<15} {'Status':<22} Policy")
    print(f"  {'─'*65}")
    for burned, status, policy in policy_rows:
        print(f"  {burned:<15} {status:<22} {policy}")


if __name__ == "__main__":
    demonstrate_sla_slo_sli_concepts()
