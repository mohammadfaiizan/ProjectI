"""
SLI, SLO, AND ERROR BUDGETS
==============================

Problem Statement:
"Is the service up?" is too binary. SLI/SLO/Error Budget frameworks
provide a principled way to define, measure, and act on reliability.
They balance engineering velocity (new features) against reliability.

Definitions:
  SLA (Service Level Agreement):  Legal contract with customers.
                                  Penalty if breached. Business/legal concern.
  SLO (Service Level Objective):  Internal target for a given SLI.
                                  Stricter than SLA: if SLO ≥ SLA, SLA breaches are rare.
  SLI (Service Level Indicator):  Measured metric used to evaluate SLO.
                                  Must be measurable, meaningful, attributable.
  Error Budget:  (1 - SLO) × window. How much unreliability you're allowed.
                 99.9% SLO over 30 days → 43.2 minutes of budget.

Common SLIs:
  Availability: good_requests / total_requests (HTTP 2xx/3xx / all).
  Latency:      fraction of requests faster than threshold.
                e.g., 95% of requests < 200ms.
  Throughput:   requests/sec meeting a minimum rate.
  Freshness:    fraction of data updated within threshold (for pipelines).
  Correctness:  fraction of results with correct output.
  Durability:   fraction of writes that can be read back.

Error Budget Policy:
  If budget > 50%:     Full velocity; deploy freely.
  If budget 10-50%:    Slowdown; requires extra review.
  If budget < 10%:     Feature freeze; reliability work only.
  If budget exhausted:  No deploys until budget replenishes OR SLO revised.

Burn Rate:
  Rate at which error budget is being consumed.
  Burn rate 1 = exactly consuming budget to hit 0 at window end.
  Burn rate 2 = consuming 2× normal; will exhaust budget in half the window.
  Multi-window burn rate alert:
    Fast burn: 14.4x over 1h AND 6h (critical).
    Slow burn: 3x over 6h AND 3d (warning).

SLO Windows:
  Rolling 28 or 30 days: always reflects last month.
  Calendar month: aligns with billing/business cycles.
  Rolling is preferred: no "new month resets budget" cliff.
"""

from __future__ import annotations

import math
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ─────────────────────────────────────────────
# SLI TYPES
# ─────────────────────────────────────────────

class SLIType(Enum):
    AVAILABILITY = "availability"
    LATENCY      = "latency"
    THROUGHPUT   = "throughput"
    FRESHNESS    = "freshness"
    CORRECTNESS  = "correctness"


# ─────────────────────────────────────────────
# SLI MEASUREMENT
# ─────────────────────────────────────────────

@dataclass
class SLIWindow:
    """Aggregated SLI measurement over a time window."""
    sli_type:      SLIType
    good_events:   int
    total_events:  int
    window_start:  float
    window_end:    float

    @property
    def ratio(self) -> float:
        if self.total_events == 0:
            return 1.0
        return self.good_events / self.total_events

    @property
    def bad_events(self) -> int:
        return self.total_events - self.good_events

    @property
    def duration_s(self) -> float:
        return self.window_end - self.window_start


# ─────────────────────────────────────────────
# SLO
# ─────────────────────────────────────────────

@dataclass
class SLO:
    name:           str
    service:        str
    sli_type:       SLIType
    target:         float        # e.g., 0.999
    window_days:    int          # e.g., 30
    description:    str

    @property
    def window_s(self) -> float:
        return self.window_days * 86400

    @property
    def error_budget_ratio(self) -> float:
        """Allowed failure fraction."""
        return 1.0 - self.target

    @property
    def error_budget_minutes(self) -> float:
        return self.error_budget_ratio * self.window_days * 24 * 60

    def burn_rate_threshold(self, exhaustion_hours: float) -> float:
        """
        What burn rate would exhaust the budget in `exhaustion_hours`?
        burn_rate = window_hours / exhaustion_hours
        """
        window_hours = self.window_days * 24
        return window_hours / exhaustion_hours

    def sla_target(self) -> float:
        """SLA is typically 0.1% below SLO target."""
        return max(0.0, self.target - 0.001)


# ─────────────────────────────────────────────
# ERROR BUDGET TRACKER
# ─────────────────────────────────────────────

class BudgetPolicy(Enum):
    FULL_VELOCITY   = "full_velocity"    # > 50% remaining
    SLOWDOWN        = "slowdown"         # 10-50% remaining
    FEATURE_FREEZE  = "feature_freeze"   # < 10% remaining
    EXHAUSTED       = "exhausted"        # 0% remaining


@dataclass
class ErrorBudgetStatus:
    slo:              SLO
    window:           SLIWindow
    remaining_ratio:  float   # 0.0 = exhausted, 1.0 = full
    burn_rate:        float   # current consumption rate vs budget

    @property
    def remaining_minutes(self) -> float:
        return self.remaining_ratio * self.slo.error_budget_minutes

    @property
    def policy(self) -> BudgetPolicy:
        if self.remaining_ratio <= 0.0:
            return BudgetPolicy.EXHAUSTED
        if self.remaining_ratio < 0.10:
            return BudgetPolicy.FEATURE_FREEZE
        if self.remaining_ratio < 0.50:
            return BudgetPolicy.SLOWDOWN
        return BudgetPolicy.FULL_VELOCITY

    @property
    def is_burning_fast(self) -> bool:
        return self.burn_rate > 1.0


class ErrorBudgetTracker:
    def __init__(self, slo: SLO):
        self._slo      = slo
        self._windows: List[SLIWindow] = []

    def record_window(self, window: SLIWindow):
        self._windows.append(window)
        # Keep only windows within SLO window
        cutoff = time.time() - self._slo.window_s
        self._windows = [w for w in self._windows if w.window_end >= cutoff]

    def status(self) -> ErrorBudgetStatus:
        if not self._windows:
            return ErrorBudgetStatus(self._slo, None, 1.0, 0.0)

        total_good  = sum(w.good_events  for w in self._windows)
        total_total = sum(w.total_events for w in self._windows)

        if total_total == 0:
            return ErrorBudgetStatus(self._slo, None, 1.0, 0.0)

        # How much of the error budget was used
        actual_error_ratio = (total_total - total_good) / total_total
        budget_ratio       = self._slo.error_budget_ratio

        if budget_ratio == 0:
            remaining = 1.0 if actual_error_ratio == 0 else 0.0
        else:
            used_fraction = actual_error_ratio / budget_ratio
            remaining     = max(0.0, 1.0 - used_fraction)

        # Burn rate: actual_error / expected_error per unit time
        # If burn_rate=2, consuming 2× what the budget allows
        total_duration = sum(w.duration_s for w in self._windows)
        expected_errors_per_s = budget_ratio / self._slo.window_s
        actual_errors_per_s   = (
            (total_total - total_good) / total_duration
            if total_duration > 0 else 0.0
        )
        burn_rate = (actual_errors_per_s / expected_errors_per_s
                     if expected_errors_per_s > 0 else 0.0)

        # Synthetic window for status object
        agg = SLIWindow(
            self._slo.sli_type,
            total_good, total_total,
            self._windows[0].window_start,
            self._windows[-1].window_end,
        )

        return ErrorBudgetStatus(self._slo, agg, remaining, burn_rate)


# ─────────────────────────────────────────────
# BURN RATE ALERTING
# ─────────────────────────────────────────────

@dataclass
class BurnRateAlert:
    slo_name:     str
    severity:     str
    burn_rate:    float
    short_window: str   # e.g., "1h"
    long_window:  str   # e.g., "6h"
    description:  str


def multi_window_burn_rate_check(
        short_rate: float, long_rate: float,
        slo: SLO, short_win: str, long_win: str) -> Optional[BurnRateAlert]:
    """
    Google SRE multi-window burn rate alerting.
    Fast burn:  14.4× over 1h AND 6h  → critical (1h budget exhaustion possible)
    Slow burn:  3×    over 6h AND 3d  → warning
    """
    critical_threshold = slo.burn_rate_threshold(1.0)   # exhaust in 1h? → 720×
    # Standard thresholds from Google SRE workbook
    fast_threshold = 14.4
    slow_threshold = 3.0

    if short_rate >= fast_threshold and long_rate >= fast_threshold:
        return BurnRateAlert(
            slo.name, "critical", max(short_rate, long_rate),
            short_win, long_win,
            f"Fast burn: {short_rate:.1f}× rate over {short_win} and {long_win}. "
            f"Budget could exhaust in {slo.window_days*24/max(short_rate,1):.1f}h"
        )
    if short_rate >= slow_threshold and long_rate >= slow_threshold:
        return BurnRateAlert(
            slo.name, "warning", max(short_rate, long_rate),
            short_win, long_win,
            f"Slow burn: {short_rate:.1f}× rate over {short_win} and {long_win}."
        )
    return None


# ─────────────────────────────────────────────
# SLO DASHBOARD
# ─────────────────────────────────────────────

class SLODashboard:
    """Aggregates multiple SLO trackers for display."""

    def __init__(self):
        self._trackers: Dict[str, ErrorBudgetTracker] = {}

    def add_slo(self, slo: SLO) -> ErrorBudgetTracker:
        tracker = ErrorBudgetTracker(slo)
        self._trackers[slo.name] = tracker
        return tracker

    def summary(self) -> List[Dict]:
        rows = []
        for name, tracker in self._trackers.items():
            s = tracker.status()
            if s.window:
                rows.append({
                    "slo":           name,
                    "target":        f"{s.slo.target*100:.3f}%",
                    "actual":        f"{s.window.ratio*100:.3f}%",
                    "budget_left":   f"{s.remaining_ratio*100:.1f}%",
                    "budget_min":    f"{s.remaining_minutes:.0f}m",
                    "burn_rate":     f"{s.burn_rate:.2f}×",
                    "policy":        s.policy.value,
                })
        return rows


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_slo():
    print("=" * 65)
    print("SLI, SLO, AND ERROR BUDGETS")
    print("=" * 65)

    random.seed(42)

    # ── Define SLOs ───────────────────────────
    print("\n[1] SLO DEFINITIONS")
    print("─" * 55)

    slos = [
        SLO("api_availability",  "api",     SLIType.AVAILABILITY, 0.999,  30,
            "99.9% of requests return 2xx/3xx"),
        SLO("api_latency",       "api",     SLIType.LATENCY,      0.950,  30,
            "95% of requests < 200ms"),
        SLO("checkout_avail",    "checkout",SLIType.AVAILABILITY, 0.9999, 30,
            "99.99% availability — revenue-critical"),
        SLO("pipeline_freshness","pipeline",SLIType.FRESHNESS,    0.990,  7,
            "99% of data updated within 1h"),
    ]

    print(f"  {'SLO':<25} {'Target':<10} {'Window':<8} {'Budget (min)':>12}")
    print("  " + "─" * 55)
    for slo in slos:
        print(f"  {slo.name:<25} {slo.target*100:.3f}%   {slo.window_days}d"
              f"    {slo.error_budget_minutes:>10.1f}m")

    # ── Simulate Measurements ─────────────────
    print("\n[2] SIMULATING 30 DAYS OF TRAFFIC")
    print("─" * 55)

    dashboard = SLODashboard()
    trackers  = {slo.name: dashboard.add_slo(slo) for slo in slos}

    # Simulate daily windows for 30 days
    now = time.time()
    for day in range(30):
        ts_start = now - (30 - day) * 86400
        ts_end   = ts_start + 86400

        # api_availability: mostly healthy, one bad day
        if day == 12:
            good, total = 8200, 10000   # 82% — bad day
        else:
            total = random.randint(9000, 11000)
            good  = total - random.randint(0, 8)   # 0-8 errors per day
        trackers["api_availability"].record_window(
            SLIWindow(SLIType.AVAILABILITY, good, total, ts_start, ts_end))

        # api_latency: 95th percentile compliance
        total = random.randint(9000, 11000)
        good  = int(total * random.uniform(0.93, 0.98))
        trackers["api_latency"].record_window(
            SLIWindow(SLIType.LATENCY, good, total, ts_start, ts_end))

        # checkout_availability: very tight SLO
        total = random.randint(5000, 6000)
        good  = total - random.randint(0, 1)
        trackers["checkout_avail"].record_window(
            SLIWindow(SLIType.AVAILABILITY, good, total, ts_start, ts_end))

        # pipeline_freshness: weekly window only
        if day >= 23:   # last 7 days
            total = 1440       # one check per minute
            good  = 1440 - random.randint(0, 20)
            trackers["pipeline_freshness"].record_window(
                SLIWindow(SLIType.FRESHNESS, good, total, ts_start, ts_end))

    print("  30 days of SLI measurements recorded")

    # ── SLO Dashboard ─────────────────────────
    print("\n[3] SLO DASHBOARD")
    print("─" * 55)

    summary = dashboard.summary()
    print(f"  {'SLO':<25} {'Target':<10} {'Actual':<10} {'Budget Left':>12} "
          f"{'Burn Rate':>10}  {'Policy'}")
    print("  " + "─" * 85)
    for row in summary:
        policy_icon = {
            "full_velocity":  "✓",
            "slowdown":       "!",
            "feature_freeze": "!!",
            "exhausted":      "✗",
        }.get(row["policy"], "?")
        print(f"  {row['slo']:<25} {row['target']:<10} {row['actual']:<10} "
              f"{row['budget_left']:>10}  {row['burn_rate']:>8}  "
              f"[{policy_icon}] {row['policy']}")

    # ── Error Budget Policy ───────────────────
    print("\n[4] ERROR BUDGET POLICY")
    print("─" * 55)

    policy_table = [
        ("FULL_VELOCITY  > 50%",  "Deploy freely; new features OK"),
        ("SLOWDOWN    10-50%",    "Extra review required; no risky deploys"),
        ("FEATURE_FREEZE < 10%",  "No new features; reliability work only"),
        ("EXHAUSTED      0%",     "No deploys until next window OR SLO revised"),
    ]
    for policy, desc in policy_table:
        print(f"  {policy:<25} {desc}")

    # ── Burn Rate Alerts ──────────────────────
    print("\n[5] MULTI-WINDOW BURN RATE ALERTS")
    print("─" * 55)

    api_slo = slos[0]   # api_availability 99.9%

    scenarios = [
        ("Normal",       1.0,  1.0,  "1h", "6h"),
        ("Slow burn",    4.0,  3.5,  "6h", "3d"),
        ("Fast burn",    20.0, 15.0, "1h", "6h"),
        ("Short spike",  50.0, 1.2,  "1h", "6h"),  # long window OK → no alert
    ]

    for label, short_r, long_r, sw, lw in scenarios:
        alert = multi_window_burn_rate_check(short_r, long_r, api_slo, sw, lw)
        if alert:
            print(f"  [{label:<12}] {alert.severity.upper()}: {alert.description}")
        else:
            print(f"  [{label:<12}] OK — no alert")

    # ── SLO Targets and SLA ───────────────────
    print("\n[6] SLO vs SLA vs NINES")
    print("─" * 55)

    nines_table = [
        (0.90,    "one nine",    30 * 24 * 60 * (1 - 0.90)),
        (0.99,    "two nines",   30 * 24 * 60 * (1 - 0.99)),
        (0.999,   "three nines", 30 * 24 * 60 * (1 - 0.999)),
        (0.9999,  "four nines",  30 * 24 * 60 * (1 - 0.9999)),
        (0.99999, "five nines",  30 * 24 * 60 * (1 - 0.99999)),
    ]
    print(f"  {'Target':<10} {'Nines':<14} {'Budget/30d':>12}  {'Budget/30d (hm)'}")
    print("  " + "─" * 60)
    for target, label, mins in nines_table:
        h  = int(mins // 60)
        m  = int(mins %  60)
        s  = int((mins * 60) % 60)
        print(f"  {target*100:<9.3f}% {label:<14} {mins:>10.1f}m  {h}h {m}m {s}s")

    # ── Key Insight ───────────────────────────
    print("\n[7] KEY INSIGHTS")
    print("─" * 55)

    insights = [
        "SLO must be stricter than SLA — buffer absorbs measurement lag",
        "100% reliability is impossible and wastes budget (deploys need errors)",
        "Burn rate alerting catches slow leaks that raw error counts miss",
        "Error budget aligns product (velocity) and SRE (reliability) goals",
        "SLI must be what the user experiences, not internal proxies",
        "Latency SLI: use percentile (p95/p99), never average",
        "Dashboard should show budget remaining, not just current SLI",
    ]
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")


if __name__ == "__main__":
    demonstrate_slo()
