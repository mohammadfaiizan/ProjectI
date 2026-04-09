"""
AUTO SCALING WITH LOAD BALANCERS
===================================

Problem Statement:
Traffic is unpredictable — flash sales, viral content, time-of-day patterns.
Manual scaling is too slow and over-provisioning is wasteful. Auto scaling
automatically adds/removes servers based on metrics, integrated tightly
with the load balancer.

Auto Scaling Types:
  Reactive  : scale out when metric threshold exceeded (CPU > 70%)
  Predictive: scale based on historical patterns (ML-based, AWS Predictive)
  Scheduled : known traffic patterns (scale up at 9am, down at 11pm)

Scale Out vs Scale Up:
  Scale Out (horizontal) → add more instances (preferred for stateless)
  Scale Up   (vertical)  → give existing instances more CPU/RAM

Key Metrics for Scaling:
  CPU utilization    → classic, but lags actual request queue
  Request count      → more direct (ALB target tracking)
  Active connections → for long-lived connections (WebSocket)
  Queue depth        → for async workers (SQS depth)
  Custom metrics     → business metrics (orders/sec)

Cool-down Period:
  After scaling action, wait before triggering another.
  Prevents thrashing (scale out → metric drops → scale in → metric spikes).
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
import time
import random
import math


class ScalingAction(Enum):
    SCALE_OUT   = "scale_out"
    SCALE_IN    = "scale_in"
    NO_ACTION   = "no_action"


class ScalingPolicy(Enum):
    STEP_SCALING   = "step_scaling"
    TARGET_TRACKING= "target_tracking"
    SCHEDULED      = "scheduled"
    PREDICTIVE     = "predictive"


@dataclass
class Instance:
    instance_id : str
    launch_time : float = field(default_factory=time.time)
    cpu_pct     : float = 20.0
    req_per_sec : float = 0.0
    status      : str   = "running"   # pending / running / terminating


@dataclass
class ScalingEvent:
    timestamp      : float
    action         : ScalingAction
    reason         : str
    instances_before: int
    instances_after: int
    metric_value   : float


@dataclass
class LoadMetrics:
    cpu_avg_pct      : float
    requests_per_sec : float
    active_conn      : int
    queue_depth      : int = 0

    def __str__(self):
        return (f"cpu={self.cpu_avg_pct:.1f}%  "
                f"rps={self.requests_per_sec:.0f}  "
                f"conn={self.active_conn}")


# ─────────────────────────────────────────────
# TARGET TRACKING POLICY
# ─────────────────────────────────────────────

class TargetTrackingPolicy:
    """
    Maintains a target metric value (e.g., CPU at 60%).
    AWS ALB Request Count Per Target is a common target tracking policy.
    """

    def __init__(self, target_cpu: float = 60.0,
                 target_rps_per_instance: float = 100.0,
                 scale_out_cooldown_s: float = 60.0,
                 scale_in_cooldown_s : float = 300.0):
        self.target_cpu          = target_cpu
        self.target_rps          = target_rps_per_instance
        self.scale_out_cooldown  = scale_out_cooldown_s
        self.scale_in_cooldown   = scale_in_cooldown_s
        self._last_scale_action  : float = 0.0
        self._last_scale_in      : float = 0.0

    def decide(self, metrics: LoadMetrics, current_count: int,
               elapsed_s: float) -> Tuple[ScalingAction, int, str]:
        """Returns (action, desired_count, reason)."""
        in_cooldown    = elapsed_s - self._last_scale_action < self.scale_out_cooldown
        in_in_cooldown = elapsed_s - self._last_scale_in < self.scale_in_cooldown

        # Scale based on RPS target
        if metrics.requests_per_sec > 0:
            desired_by_rps = math.ceil(metrics.requests_per_sec / self.target_rps)
        else:
            desired_by_rps = current_count

        # Scale based on CPU target
        if metrics.cpu_avg_pct > 0 and current_count > 0:
            desired_by_cpu = math.ceil(current_count * metrics.cpu_avg_pct / self.target_cpu)
        else:
            desired_by_cpu = current_count

        desired = max(desired_by_rps, desired_by_cpu, 1)

        if desired > current_count and not in_cooldown:
            self._last_scale_action = elapsed_s
            reason = (f"scale out: rps={metrics.requests_per_sec:.0f} "
                      f"desired={desired} (cpu={metrics.cpu_avg_pct:.0f}%)")
            return ScalingAction.SCALE_OUT, desired, reason

        if desired < current_count and not in_in_cooldown:
            # Scale in conservatively — never below 1
            safe_desired = max(desired, 1)
            if safe_desired < current_count:
                self._last_scale_in = elapsed_s
                reason = (f"scale in: rps={metrics.requests_per_sec:.0f} "
                          f"desired={safe_desired} (cpu={metrics.cpu_avg_pct:.0f}%)")
                return ScalingAction.SCALE_IN, safe_desired, reason

        return ScalingAction.NO_ACTION, current_count, "within target range"


# ─────────────────────────────────────────────
# STEP SCALING POLICY
# ─────────────────────────────────────────────

class StepScalingPolicy:
    """
    Different scaling steps for different metric ranges.
    E.g.: CPU 60-70% → add 1; CPU 70-85% → add 2; CPU >85% → add 4.
    """

    def __init__(self):
        self._steps_out : List[Tuple[float, float, int]] = []   # (lower, upper, add_n)
        self._steps_in  : List[Tuple[float, float, int]] = []   # (lower, upper, remove_n)

    def add_scale_out_step(self, lower: float, upper: float, adjustment: int):
        self._steps_out.append((lower, upper, adjustment))

    def add_scale_in_step(self, lower: float, upper: float, adjustment: int):
        self._steps_in.append((lower, upper, adjustment))

    def decide(self, cpu: float, current_count: int) -> Tuple[int, str]:
        for lower, upper, adj in self._steps_out:
            if lower <= cpu < upper or (upper == float("inf") and cpu >= lower):
                return current_count + adj, f"CPU={cpu:.0f}% → add {adj}"
        for lower, upper, adj in self._steps_in:
            if lower <= cpu < upper:
                return max(1, current_count - adj), f"CPU={cpu:.0f}% → remove {adj}"
        return current_count, "no step matched"


# ─────────────────────────────────────────────
# AUTO SCALING GROUP
# ─────────────────────────────────────────────

class AutoScalingGroup:
    def __init__(self, name: str, min_size: int = 1,
                 max_size: int = 20, desired: int = 2):
        self.name          = name
        self.min_size      = min_size
        self.max_size      = max_size
        self._instances    : List[Instance] = []
        self.scaling_events: List[ScalingEvent] = []
        self._counter      = 0

        for _ in range(desired):
            self._add_instance()

    def _add_instance(self) -> Instance:
        self._counter += 1
        inst = Instance(f"{self.name}-{self._counter:03d}")
        self._instances.append(inst)
        return inst

    def _remove_instance(self) -> Optional[Instance]:
        if self._instances:
            return self._instances.pop()
        return None

    @property
    def count(self) -> int:
        return len(self._instances)

    @property
    def instance_ids(self) -> List[str]:
        return [i.instance_id for i in self._instances]

    def scale_to(self, desired: int, action: ScalingAction,
                 reason: str, metric: float):
        desired  = max(self.min_size, min(self.max_size, desired))
        before   = self.count
        if desired > before:
            for _ in range(desired - before):
                self._add_instance()
        elif desired < before:
            for _ in range(before - desired):
                self._remove_instance()

        if desired != before:
            event = ScalingEvent(
                timestamp=time.time(),
                action=action,
                reason=reason,
                instances_before=before,
                instances_after=self.count,
                metric_value=metric
            )
            self.scaling_events.append(event)
            icon = "↑" if action == ScalingAction.SCALE_OUT else "↓"
            print(f"  {icon} ASG [{self.name}]: {before} → {self.count}  reason: {reason}")

    def report(self):
        print(f"\n  ASG [{self.name}] Summary:")
        print(f"    Current instances : {self.count} ({self.min_size}–{self.max_size})")
        print(f"    Instances         : {self.instance_ids}")
        print(f"    Scaling events    : {len(self.scaling_events)}")
        for e in self.scaling_events:
            icon = "↑" if e.action == ScalingAction.SCALE_OUT else "↓"
            print(f"      {icon} {e.instances_before}→{e.instances_after}  metric={e.metric_value:.1f}  {e.reason}")


# ─────────────────────────────────────────────
# TRAFFIC SIMULATOR
# ─────────────────────────────────────────────

class TrafficSimulator:
    """Generates realistic traffic patterns for scaling demos."""

    @staticmethod
    def day_pattern(hour: int, base_rps: float = 50.0) -> float:
        """Simulate traffic by hour of day."""
        patterns = {
            0:  0.15, 1:  0.10, 2:  0.08, 3:  0.07,
            4:  0.08, 5:  0.12, 6:  0.20, 7:  0.35,
            8:  0.60, 9:  0.85, 10: 0.95, 11: 1.00,
            12: 0.90, 13: 0.95, 14: 0.92, 15: 0.88,
            16: 0.85, 17: 0.80, 18: 0.65, 19: 0.55,
            20: 0.45, 21: 0.38, 22: 0.30, 23: 0.20,
        }
        return base_rps * patterns.get(hour, 0.5)

    @staticmethod
    def flash_sale_pattern(minute: int, base_rps: float = 50.0) -> float:
        """Simulate flash sale traffic spike."""
        if 0 <= minute < 5:
            return base_rps * (1 + minute * 4)   # rapid ramp-up
        if 5 <= minute < 30:
            return base_rps * 20 + random.uniform(-base_rps, base_rps)
        if 30 <= minute < 45:
            return base_rps * max(1, 20 - (minute - 30) * 1.2)
        return base_rps

    @staticmethod
    def rps_to_cpu(rps: float, instances: int,
                   rps_per_instance_at_100pct: float = 200.0) -> float:
        if instances == 0:
            return 100.0
        load_per = rps / instances
        return min(100.0, (load_per / rps_per_instance_at_100pct) * 100)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_auto_scaling():
    print("=" * 65)
    print("AUTO SCALING WITH LOAD BALANCERS")
    print("=" * 65)

    random.seed(42)

    # ── Target Tracking ───────────────────────
    print("\n[1] TARGET TRACKING — DAY-OF-WEEK TRAFFIC PATTERN")
    print("─" * 55)
    asg    = AutoScalingGroup("web-asg", min_size=2, max_size=20, desired=2)
    policy = TargetTrackingPolicy(target_cpu=60.0, target_rps_per_instance=100.0,
                                   scale_out_cooldown_s=2.0, scale_in_cooldown_s=5.0)
    sim    = TrafficSimulator()

    print(f"  Simulating 24 hours (hourly snapshots):")
    print(f"  {'Hour':<6} {'RPS':<8} {'CPU%':<8} {'Instances':<12} {'Action'}")
    print(f"  {'─'*55}")
    for hour in range(0, 24, 2):
        rps = sim.day_pattern(hour, base_rps=200.0)
        cpu = sim.rps_to_cpu(rps, asg.count)
        metrics = LoadMetrics(cpu_avg_pct=cpu, requests_per_sec=rps, active_conn=int(rps))
        action, desired, reason = policy.decide(metrics, asg.count, elapsed_s=hour * 3600.0)
        asg.scale_to(desired, action, reason, cpu)
        print(f"  {hour:02d}:00  {rps:<8.0f} {cpu:<8.1f} {asg.count:<12} "
              f"{'→ ' + reason[:35] if action != ScalingAction.NO_ACTION else 'steady'}")

    asg.report()

    # ── Step Scaling ──────────────────────────
    print("\n\n[2] STEP SCALING POLICY")
    print("─" * 55)
    step_policy = StepScalingPolicy()
    step_policy.add_scale_out_step(60,  70,  1)   # +1 instance
    step_policy.add_scale_out_step(70,  85,  2)   # +2 instances
    step_policy.add_scale_out_step(85, float("inf"), 4)   # +4 instances
    step_policy.add_scale_in_step(20,  40,  1)    # -1 instance

    print("  Steps: CPU 60-70%→+1, 70-85%→+2, >85%→+4, <40%→-1")
    for cpu in [50, 65, 75, 90, 30]:
        current  = 4
        desired, reason = step_policy.decide(cpu, current)
        print(f"  CPU={cpu}%  current={current}  desired={desired}  ({reason})")

    # ── Flash Sale Scenario ───────────────────
    print("\n\n[3] FLASH SALE — RAPID SCALE-OUT")
    print("─" * 55)
    flash_asg    = AutoScalingGroup("flash-asg", min_size=2, max_size=50, desired=2)
    flash_policy = TargetTrackingPolicy(target_cpu=60.0, target_rps_per_instance=50.0,
                                         scale_out_cooldown_s=1.0, scale_in_cooldown_s=2.0)
    print(f"  {'Min':<6} {'RPS':<8} {'CPU%':<8} {'Instances':<12} Note")
    print(f"  {'─'*50}")
    for minute in [0, 2, 5, 10, 20, 30, 40, 50]:
        rps = sim.flash_sale_pattern(minute, base_rps=50.0)
        cpu = sim.rps_to_cpu(rps, flash_asg.count, rps_per_instance_at_100pct=100.0)
        metrics = LoadMetrics(cpu_avg_pct=cpu, requests_per_sec=rps,
                               active_conn=int(rps))
        action, desired, reason = flash_policy.decide(metrics, flash_asg.count,
                                                        elapsed_s=minute * 60.0)
        flash_asg.scale_to(desired, action, reason, cpu)
        note = "🔥 SALE START" if minute == 5 else ("✅ NORMAL" if minute >= 40 else "")
        print(f"  {minute:<6} {rps:<8.0f} {cpu:<8.1f} {flash_asg.count:<12} {note}")

    # ── Scaling Best Practices ────────────────
    print("\n\n[4] AUTO SCALING BEST PRACTICES")
    print("─" * 55)
    practices = [
        ("Use ALB request count",    "More direct than CPU; scales on actual load"),
        ("Set scale-in conservatively","Scale out fast (60s), scale in slow (5min)"),
        ("Pre-warm for known events", "Scheduled scaling before flash sales"),
        ("Min=2 instances",          "Avoid single-instance SPOF even at low load"),
        ("Use lifecycle hooks",       "Run init scripts before instance joins LB"),
        ("Instance warm-up time",    "New instances need 2-3min before taking traffic"),
        ("Capacity buffers",         "Target 60-70% CPU, not 90% — headroom for spikes"),
        ("Test scale events",        "Chaos engineering: kill instances, verify recovery"),
    ]
    for practice, detail in practices:
        print(f"  • {practice:<35} {detail}")

    # ── Topology ─────────────────────────────
    print("\n\n[5] ASG + LB TOPOLOGY")
    print("─" * 55)
    print("  Internet → DNS → ALB (L7) → ASG (2-20 instances)")
    print("                              └─ Auto-deregisters from ALB on scale-in")
    print("                              └─ Connection draining (30s)")
    print("                              └─ Health check before accepting traffic")
    print("  Spot Instances: 70% cheaper, 2min termination notice")
    print("  On-Demand: stable baseline (min capacity)")
    print("  Mix: 70% Spot + 30% On-Demand → cost savings + reliability")


if __name__ == "__main__":
    demonstrate_auto_scaling()
