"""
HORIZONTAL VS VERTICAL SCALING
================================

Problem Statement:
As traffic grows, systems must scale to handle more load. There are two
fundamental strategies: vertical scaling (bigger machine) and horizontal
scaling (more machines). Understanding their limits and trade-offs is
critical for any system design.

Key Concepts:
- Vertical Scaling (Scale Up) : Upgrade CPU/RAM/disk on existing server. Has a hard ceiling.
- Horizontal Scaling (Scale Out): Add more servers behind a load balancer. Near-infinite.
- Stateless Services: Required for horizontal scaling (no server-local state)
- Auto-scaling: Automatically adjusting server count based on load metrics
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict
import random


class ScalingStrategy(Enum):
    VERTICAL   = "vertical"
    HORIZONTAL = "horizontal"
    HYBRID     = "hybrid"


@dataclass
class ServerSpec:
    tier          : str
    cpu_cores     : int
    ram_gb        : int
    cost_per_hour : float
    max_qps       : int

    def __str__(self):
        return (f"{self.tier:<20} CPU:{self.cpu_cores:>4}  "
                f"RAM:{self.ram_gb:>5}GB  "
                f"${self.cost_per_hour:>6.2f}/hr  "
                f"Max QPS:{self.max_qps:>8,}")


@dataclass
class ServerLoad:
    cpu_pct      : float
    memory_pct   : float
    requests_per_sec: float

    @property
    def is_overloaded(self) -> bool:
        return self.cpu_pct > 80 or self.memory_pct > 85


class VerticalScaler:
    """Upgrades a single server to a higher tier."""

    TIERS: List[ServerSpec] = [
        ServerSpec("t3.micro",    2,   1,   0.01,    500),
        ServerSpec("t3.small",    2,   2,   0.02,  1_000),
        ServerSpec("t3.medium",   2,   4,   0.04,  2_000),
        ServerSpec("t3.large",    2,   8,   0.08,  4_000),
        ServerSpec("t3.xlarge",   4,  16,   0.17,  8_000),
        ServerSpec("t3.2xlarge",  8,  32,   0.33, 16_000),
        ServerSpec("m5.4xlarge", 16,  64,   0.77, 30_000),
        ServerSpec("m5.12xlarge",48, 192,   2.30, 80_000),
        ServerSpec("m5.24xlarge",96, 384,   4.61,150_000),
    ]

    def __init__(self):
        self.current_tier_idx = 0

    @property
    def current_spec(self) -> ServerSpec:
        return self.TIERS[self.current_tier_idx]

    def scale_up(self) -> bool:
        if self.current_tier_idx >= len(self.TIERS) - 1:
            print(f"  ⛔ CEILING REACHED at {self.current_spec.tier} — cannot scale up further!")
            return False
        self.current_tier_idx += 1
        print(f"  ↑  Scaled up to {self.current_spec.tier}  (cost: ${self.current_spec.cost_per_hour:.2f}/hr)")
        return True

    def can_handle(self, qps: int) -> bool:
        return self.current_spec.max_qps >= qps

    def report(self):
        spec = self.current_spec
        print(f"  Current spec : {spec}")


class HorizontalScaler:
    """Adds/removes servers behind a load balancer."""

    def __init__(self, base_spec: ServerSpec):
        self.base_spec = base_spec
        self.servers   : List[str] = []
        self._counter  = 0

    def scale_out(self, count: int = 1):
        for _ in range(count):
            self._counter += 1
            server_id = f"app-{self._counter:03d}"
            self.servers.append(server_id)
            print(f"  ↗  Added server {server_id} (total: {len(self.servers)})")

    def scale_in(self, count: int = 1):
        for _ in range(count):
            if len(self.servers) > 1:
                removed = self.servers.pop()
                print(f"  ↙  Removed server {removed} (total: {len(self.servers)})")

    @property
    def total_qps_capacity(self) -> int:
        return len(self.servers) * self.base_spec.max_qps

    @property
    def cost_per_hour(self) -> float:
        return len(self.servers) * self.base_spec.cost_per_hour

    def can_handle(self, qps: int) -> bool:
        return self.total_qps_capacity >= qps

    def report(self):
        print(f"  Servers      : {len(self.servers)}")
        print(f"  Each spec    : {self.base_spec}")
        print(f"  Total QPS    : {self.total_qps_capacity:,}")
        print(f"  Total cost   : ${self.cost_per_hour:.2f}/hr")


class AutoScaler:
    """Automatically scales a horizontal fleet based on load metrics."""

    def __init__(self, scaler: HorizontalScaler,
                 scale_out_threshold: float = 70.0,
                 scale_in_threshold : float = 30.0,
                 cooldown_sec       : int   = 60):
        self.scaler               = scaler
        self.scale_out_threshold  = scale_out_threshold
        self.scale_in_threshold   = scale_in_threshold
        self.cooldown_sec         = cooldown_sec
        self._last_action_at      = -cooldown_sec  # allow first action immediately

    def evaluate(self, current_load_pct: float, tick: int):
        if current_load_pct > self.scale_out_threshold:
            if tick - self._last_action_at >= self.cooldown_sec:
                print(f"  🔴 Load {current_load_pct:.0f}% > {self.scale_out_threshold}% → SCALE OUT")
                self.scaler.scale_out(2)
                self._last_action_at = tick
            else:
                print(f"  ⏳ Load {current_load_pct:.0f}% high but in cooldown")
        elif current_load_pct < self.scale_in_threshold:
            if tick - self._last_action_at >= self.cooldown_sec and len(self.scaler.servers) > 2:
                print(f"  🟢 Load {current_load_pct:.0f}% < {self.scale_in_threshold}% → SCALE IN")
                self.scaler.scale_in(1)
                self._last_action_at = tick
        else:
            print(f"  ✅ Load {current_load_pct:.0f}% — no scaling needed")


def demonstrate_horizontal_vs_vertical_scaling():
    print("=" * 65)
    print("HORIZONTAL VS VERTICAL SCALING")
    print("=" * 65)

    # ── Vertical Scaling Journey ──────────────
    print("\n[VERTICAL SCALING] Starting at t3.micro, growing traffic")
    print("─" * 55)
    vert = VerticalScaler()
    vert.report()

    qps_demands = [400, 900, 3_500, 15_000, 75_000, 140_000, 160_000]
    for qps in qps_demands:
        print(f"\n  Traffic demand: {qps:,} QPS")
        while not vert.can_handle(qps):
            if not vert.scale_up():
                break
        if vert.can_handle(qps):
            print(f"  ✅ Handled by {vert.current_spec.tier}")
        else:
            print(f"  ❌ Vertical scaling exhausted! Need horizontal scaling.")

    # ── Horizontal Scaling Comparison ────────
    print("\n\n[HORIZONTAL SCALING] Same traffic demands")
    print("─" * 55)
    base = ServerSpec("t3.xlarge", 4, 16, 0.17, 8_000)
    horiz = HorizontalScaler(base)
    horiz.scale_out(1)   # start with 1

    for qps in qps_demands:
        print(f"\n  Traffic demand: {qps:,} QPS")
        while not horiz.can_handle(qps):
            horiz.scale_out(2)
        print(f"  ✅ Handled by {len(horiz.servers)} servers (total {horiz.total_qps_capacity:,} QPS capacity, ${horiz.cost_per_hour:.2f}/hr)")

    # ── Auto-Scaler Simulation ────────────────
    print("\n\n[AUTO-SCALING] Simulating variable load over 24 ticks")
    print("─" * 55)
    fleet  = HorizontalScaler(base)
    fleet.scale_out(3)    # start with 3 servers
    auto   = AutoScaler(fleet, scale_out_threshold=70, scale_in_threshold=30, cooldown_sec=3)

    # Simulate morning ramp-up, peak, evening ramp-down
    load_profile = [15, 20, 30, 50, 65, 80, 85, 90, 85, 75, 70, 65,
                    80, 90, 88, 75, 60, 50, 40, 35, 25, 20, 15, 10]
    for tick, load_pct in enumerate(load_profile):
        print(f"\n  Tick {tick:02d} | Load: {load_pct:3d}% | Servers: {len(fleet.servers)}", end="  ")
        auto.evaluate(load_pct, tick)

    print(f"\n\nFinal fleet: {len(fleet.servers)} servers | ${fleet.cost_per_hour:.2f}/hr")

    # ── Trade-off Summary ─────────────────────
    print("\n\nSCALING TRADE-OFF SUMMARY:")
    print(f"  {'Aspect':<25} {'Vertical':<25} {'Horizontal'}")
    print(f"  {'─'*70}")
    rows = [
        ("Max scale",        "Hard ceiling (finite hardware)", "Near-infinite"),
        ("Complexity",       "Simple (one server)",            "Complex (LB, service discovery)"),
        ("Cost efficiency",  "Expensive at high end",          "Pay per unit"),
        ("Downtime",         "Downtime to upgrade",            "Rolling deploy, zero downtime"),
        ("State",            "Easy (local state OK)",          "Must be stateless or external"),
        ("Failure blast",    "Full outage if server dies",     "One server dies, rest serve"),
    ]
    for aspect, vert_v, horiz_v in rows:
        print(f"  {aspect:<25} {vert_v:<25} {horiz_v}")


if __name__ == "__main__":
    demonstrate_horizontal_vs_vertical_scaling()
