"""
HEARTBEAT AND FAILURE DETECTION
==================================

Problem Statement:
In distributed systems, nodes fail silently: no crash notification, just silence.
Failure detection answers: "Is this node alive or dead?"
The challenge: distinguishing a crashed node from a slow/partitioned one.
False positives (declaring alive node dead) → unnecessary rebalancing.
False negatives (declaring dead node alive) → serving stale data, missed failover.

Failure Detection Models:

  1. Heartbeat Timeout:
     Node sends periodic heartbeats. If none received within T, declare failed.
     Simple. Problem: timeout T is a trade-off.
     T too small → false positives (node is slow, not dead).
     T too large → slow detection (delay before failover).

  2. Phi Accrual Failure Detector (Cassandra):
     Instead of binary (alive/dead), outputs phi φ — probability it's alive.
     φ increases as heartbeats become overdue based on historical arrival distribution.
     Caller decides threshold: if φ > 8, consider it failed.
     Adaptive: adjusts to network conditions automatically.

  3. SWIM (Scalable Weakly-consistent Infection-style Membership):
     Combines probing with gossip.
     Direct ping: if no response → indirect ping via K random nodes.
     If still no response → mark as suspect.
     If still no response after timeout → declare failed.
     False positive protection: node can refute suspicion if it receives notification.
     Used by: Consul, HashiCorp tools.

  4. Lease-based:
     Node holds a time-bounded lease. If lease expires → considered failed.
     Proactive: requires renewal rather than passive monitoring.
     Used by: leader election, distributed locks.

  5. Quorum Sensing:
     Node is considered failed if it can't reach a quorum of peers.
     Self-removal from cluster if isolated (split-brain protection).
     Used by: Raft when follower misses too many heartbeats.

Timeout Setting:
  target_timeout = expected_RTT + safety_margin
  Too short → false positives. Too long → slow failover.
  Adaptive: sample RTTs, set timeout at 99th percentile + buffer.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque
import time
import threading
import random
import math


# ─────────────────────────────────────────────
# HEARTBEAT MONITOR
# ─────────────────────────────────────────────

class HeartbeatMonitor:
    """
    Simple timeout-based failure detector.
    Declares a node dead if no heartbeat received within `timeout_s`.
    """

    def __init__(self, timeout_s: float = 2.0):
        self.timeout_s   = timeout_s
        self._last_seen  : Dict[str, float] = {}
        self._lock       = threading.Lock()
        self.detections  = 0

    def record(self, node_id: str):
        with self._lock:
            self._last_seen[node_id] = time.time()

    def is_alive(self, node_id: str) -> bool:
        with self._lock:
            last = self._last_seen.get(node_id, 0)
            return (time.time() - last) < self.timeout_s

    def check_all(self) -> Dict[str, str]:
        """Returns {node_id: 'alive' | 'suspected'} for all monitored nodes."""
        result = {}
        with self._lock:
            now = time.time()
            for node_id, last in self._last_seen.items():
                age    = now - last
                status = "alive" if age < self.timeout_s else "suspected"
                if status == "suspected":
                    self.detections += 1
                result[node_id] = status
        return result


# ─────────────────────────────────────────────
# PHI ACCRUAL FAILURE DETECTOR (Cassandra-style)
# ─────────────────────────────────────────────

class PhiAccrualDetector:
    """
    Computes φ (phi) — a continuously increasing suspicion level.
    Based on historical heartbeat inter-arrival times.
    φ = -log(P_late | distribution of past inter-arrivals).
    Threshold: φ > 8 means ~99.97% probability the node is down.
    """

    def __init__(self, window_size: int = 100, min_std_dev_ms: float = 200.0):
        self.window_size   = window_size
        self.min_std_dev_ms = min_std_dev_ms
        self._arrivals     : deque = deque(maxlen=window_size)
        self._last_arrival : float = 0.0
        self._lock         = threading.Lock()

    def heartbeat(self):
        """Record heartbeat arrival."""
        with self._lock:
            now = time.time()
            if self._last_arrival > 0:
                interval_ms = (now - self._last_arrival) * 1000
                self._arrivals.append(interval_ms)
            self._last_arrival = now

    def phi(self) -> float:
        """Compute current phi value."""
        with self._lock:
            if not self._arrivals or self._last_arrival == 0:
                return 0.0
            now    = time.time()
            t_diff = (now - self._last_arrival) * 1000   # ms since last heartbeat

            intervals = list(self._arrivals)
            mean      = sum(intervals) / len(intervals)
            if len(intervals) > 1:
                variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
                std_dev  = max(math.sqrt(variance), self.min_std_dev_ms)
            else:
                std_dev  = self.min_std_dev_ms

            # Cumulative distribution function of exponential distribution
            # P(T > t_diff | mean) ≈ e^(-t_diff/mean)
            p_later = math.exp(-t_diff / mean) if mean > 0 else 0.0
            p_later = max(p_later, 1e-15)   # avoid log(0)
            return -math.log(p_later) / math.log(10)   # log base 10 phi

    def is_alive(self, threshold: float = 8.0) -> bool:
        return self.phi() < threshold


# ─────────────────────────────────────────────
# SWIM PROBER
# ─────────────────────────────────────────────

class SWIMDetector:
    """
    SWIM protocol failure detection.
    Phase 1: Direct ping.
    Phase 2: If no response, k-indirect pings.
    Phase 3: If still no response, suspect → broadcast via gossip.
    Phase 4: If suspected node doesn't refute → declare dead.
    """

    def __init__(self, node_id: str, probe_interval_s: float = 0.2,
                 suspect_timeout_s: float = 0.5):
        self.node_id        = node_id
        self.probe_interval = probe_interval_s
        self.suspect_timeout = suspect_timeout_s
        self._alive_set     : Dict[str, float] = {}   # node_id → last known alive
        self._suspected     : Dict[str, float] = {}   # node_id → suspected_since
        self._dead          : set = set()

    def direct_ping(self, target_id: str, cluster: Dict[str, "TargetNode"]) -> bool:
        node = cluster.get(target_id)
        return node.respond() if node else False

    def indirect_ping(self, target_id: str, cluster: Dict[str, "TargetNode"],
                      k: int = 2) -> bool:
        intermediaries = [n for nid, n in cluster.items()
                          if nid != self.node_id and nid != target_id]
        intermediaries = random.sample(intermediaries, min(k, len(intermediaries)))
        for node in intermediaries:
            if node.relay_ping(target_id, cluster):
                return True
        return False

    def probe_cycle(self, target_id: str, cluster: Dict[str, "TargetNode"]):
        if self.direct_ping(target_id, cluster):
            self._alive_set[target_id] = time.time()
            self._suspected.pop(target_id, None)
        else:
            if self.indirect_ping(target_id, cluster):
                self._alive_set[target_id] = time.time()
                self._suspected.pop(target_id, None)
            else:
                if target_id not in self._suspected:
                    self._suspected[target_id] = time.time()

    def check_suspects(self):
        """Move long-suspected nodes to dead."""
        now = time.time()
        for node_id, suspect_since in list(self._suspected.items()):
            if now - suspect_since > self.suspect_timeout:
                self._dead.add(node_id)
                del self._suspected[node_id]


class TargetNode:
    def __init__(self, node_id: str, alive: bool = True, drop_rate: float = 0.0):
        self.node_id   = node_id
        self._alive    = alive
        self.drop_rate = drop_rate   # simulates packet loss

    def respond(self) -> bool:
        if not self._alive:
            return False
        return random.random() > self.drop_rate

    def relay_ping(self, target_id: str, cluster: Dict[str, "TargetNode"]) -> bool:
        if not self._alive:
            return False
        target = cluster.get(target_id)
        return target.respond() if target else False


# ─────────────────────────────────────────────
# ADAPTIVE TIMEOUT CALCULATOR
# ─────────────────────────────────────────────

class AdaptiveTimeout:
    """
    Calculates failure detection timeout based on observed RTTs.
    Timeout = RTT_99th_percentile + safety_margin.
    """

    def __init__(self, safety_margin_ms: float = 500.0):
        self._rtts         : deque = deque(maxlen=100)
        self.safety_margin = safety_margin_ms

    def record_rtt(self, rtt_ms: float):
        self._rtts.append(rtt_ms)

    def recommended_timeout_ms(self) -> float:
        if not self._rtts:
            return 5000.0
        sorted_rtts = sorted(self._rtts)
        idx = int(len(sorted_rtts) * 0.99)
        p99 = sorted_rtts[min(idx, len(sorted_rtts) - 1)]
        return p99 + self.safety_margin

    @property
    def mean_rtt_ms(self) -> float:
        return sum(self._rtts) / len(self._rtts) if self._rtts else 0.0


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_failure_detection():
    print("=" * 65)
    print("HEARTBEAT AND FAILURE DETECTION")
    print("=" * 65)

    random.seed(7)

    # ── Heartbeat Timeout ─────────────────────────
    print("\n[1] HEARTBEAT TIMEOUT — SIMPLE DETECTION")
    print("─" * 55)

    monitor = HeartbeatMonitor(timeout_s=0.2)
    nodes   = ["N1", "N2", "N3"]
    for n in nodes:
        monitor.record(n)

    print(f"  All nodes sent heartbeats. Status:")
    print(f"  {monitor.check_all()}")

    # Simulate N3 stopping heartbeats
    time.sleep(0.25)
    monitor.record("N1")   # N1 renews
    monitor.record("N2")   # N2 renews
    # N3 does NOT renew

    status = monitor.check_all()
    print(f"\n  After 250ms (timeout=200ms), N3 stopped heartbeating:")
    for n, s in status.items():
        print(f"    {n}: {s}")

    # ── Phi Accrual Detector ──────────────────────
    print("\n\n[2] PHI ACCRUAL DETECTOR — CASSANDRA-STYLE")
    print("─" * 55)

    phi_detector = PhiAccrualDetector(window_size=20)

    # Normal heartbeats every 100ms → phi stays low
    print(f"  Normal heartbeats (every ~100ms):")
    for i in range(10):
        phi_detector.heartbeat()
        time.sleep(0.01)   # 10ms for demo (would be 100ms in prod)
        phi = phi_detector.phi()
        print(f"    beat {i+1}: φ={phi:.2f}  alive={phi_detector.is_alive()}")

    # Stop heartbeats → phi increases
    print(f"\n  Heartbeats stopped. φ growing:")
    for wait_ms in [10, 20, 40, 80, 150]:
        time.sleep(wait_ms / 1000)
        phi = phi_detector.phi()
        alive = phi_detector.is_alive(threshold=3.0)
        print(f"    +{wait_ms}ms overdue: φ={phi:.2f}  alive(threshold=3)={alive}")

    # ── SWIM Failure Detection ────────────────────
    print("\n\n[3] SWIM PROTOCOL — PROBE + INDIRECT PING")
    print("─" * 55)

    cluster = {
        "N1": TargetNode("N1", alive=True),
        "N2": TargetNode("N2", alive=True),
        "N3": TargetNode("N3", alive=False),   # crashed
        "N4": TargetNode("N4", alive=True),
        "N5": TargetNode("N5", alive=True),
    }
    detector = SWIMDetector("N0", suspect_timeout_s=0.1)

    print(f"  Probing each node (N3 is crashed):")
    for target_id in ["N1", "N2", "N3", "N4", "N5"]:
        detector.probe_cycle(target_id, cluster)

    time.sleep(0.15)   # let suspect timeout expire for N3
    detector.check_suspects()

    print(f"  Alive confirmed: {sorted(detector._alive_set.keys())}")
    print(f"  Suspected      : {sorted(detector._suspected.keys())}")
    print(f"  Declared dead  : {sorted(detector._dead)}")

    # ── Adaptive Timeout ──────────────────────────
    print("\n\n[4] ADAPTIVE TIMEOUT CALCULATOR")
    print("─" * 55)

    adaptive = AdaptiveTimeout(safety_margin_ms=200.0)

    # Normal traffic: RTTs 20-80ms
    for _ in range(90):
        adaptive.record_rtt(random.uniform(20, 80))
    # Occasional spikes
    for _ in range(10):
        adaptive.record_rtt(random.uniform(200, 400))

    print(f"  Sampled 100 RTTs (mostly 20-80ms, some spikes up to 400ms)")
    print(f"  Mean RTT: {adaptive.mean_rtt_ms:.1f}ms")
    print(f"  Recommended timeout: {adaptive.recommended_timeout_ms():.0f}ms "
          f"(P99 + 200ms margin)")

    # ── Comparison ────────────────────────────────
    print("\n\n[5] FAILURE DETECTOR COMPARISON")
    print("─" * 55)
    rows = [
        ("Heartbeat timeout","Simple, binary",      "Fixed timeout → FP or slow detect"),
        ("Phi Accrual",      "Probabilistic, adaptive","Complex; tunable threshold"),
        ("SWIM",             "Scalable, low msgs",  "Eventual detection (not instant)"),
        ("Lease",            "Proactive renewal",   "Requires renew logic in service"),
        ("Quorum sensing",   "Self-healing",        "Node removes itself if isolated"),
    ]
    print(f"  {'Detector':<22} {'Benefit':<28} {'Drawback'}")
    print(f"  {'─'*72}")
    for detector, benefit, drawback in rows:
        print(f"  {detector:<22} {benefit:<28} {drawback}")

    print("\n\n[6] FAILURE DETECTION DESIGN TIPS")
    print("─" * 55)
    tips = [
        "Timeout = P99(RTT) + margin. Sample RTTs to set adaptive timeout",
        "Phi accrual: threshold φ=8 means ~1/10000 false positive rate",
        "SWIM indirect probing reduces false positives from network blips",
        "Separate: slow node (reduce traffic) vs dead node (failover)",
        "Alert on detection rate: spikes = network issue, not node failures",
        "Graceful shutdown: deregister explicitly rather than waiting for TTL",
    ]
    for tip in tips:
        print(f"  • {tip}")


if __name__ == "__main__":
    demonstrate_failure_detection()
