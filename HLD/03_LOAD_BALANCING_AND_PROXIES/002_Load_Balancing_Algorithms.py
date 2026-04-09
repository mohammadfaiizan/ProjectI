"""
LOAD BALANCING ALGORITHMS
===========================

Problem Statement:
The choice of LB algorithm determines how evenly requests are distributed,
how well the system handles heterogeneous backends, and whether sessions
are sticky. Wrong algorithm → hot spots, wasted capacity, or dropped sessions.

Algorithms Covered:
  Round Robin        → cycle through backends in order (equal weight assumed)
  Weighted Round Robin → backends get traffic proportional to their weight
  Least Connections  → send to backend with fewest active connections
  Least Response Time → send to fastest responding backend
  IP Hash            → hash client IP → always same backend (sticky)
  Random             → random backend (simple, surprisingly effective)
  Resource-Based     → route based on CPU/memory metrics (adaptive)

When to Use Which:
  Round Robin          : identical backends, stateless requests
  Weighted RR          : heterogeneous backends (2x CPU → weight=2)
  Least Connections    : long-lived connections (WebSocket, file upload)
  Least Response Time  : latency-sensitive workloads, mixed backends
  IP Hash              : when you need session affinity without cookies
  Random               : simplest, scales well, good for short requests
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random
import hashlib
import time


class Algorithm(Enum):
    ROUND_ROBIN          = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS    = "least_connections"
    LEAST_RESPONSE_TIME  = "least_response_time"
    IP_HASH              = "ip_hash"
    RANDOM               = "random"
    RESOURCE_BASED       = "resource_based"


@dataclass
class Server:
    server_id       : str
    weight          : int   = 1
    active_conn     : int   = 0
    total_handled   : int   = 0
    avg_response_ms : float = 10.0
    cpu_pct         : float = 30.0

    @property
    def score(self) -> float:
        """Lower is better for least-response-time."""
        return self.avg_response_ms

    def receive(self):
        self.active_conn += 1
        self.total_handled += 1

    def complete(self, response_ms: float):
        self.active_conn -= 1
        # Exponential moving average
        alpha = 0.2
        self.avg_response_ms = alpha * response_ms + (1 - alpha) * self.avg_response_ms


# ─────────────────────────────────────────────
# ALGORITHMS
# ─────────────────────────────────────────────

class RoundRobin:
    def __init__(self, servers: List[Server]):
        self.servers = servers
        self._index  = 0

    def pick(self, client_ip: str = "") -> Server:
        available = [s for s in self.servers if s.active_conn >= 0]
        s = available[self._index % len(available)]
        self._index += 1
        return s


class WeightedRoundRobin:
    """
    Generates a weighted sequence: weight=2 → appears twice per cycle.
    Uses the GCD-based smooth algorithm (Nginx-style).
    """

    def __init__(self, servers: List[Server]):
        self.servers = servers
        self._current_weights = {s.server_id: 0 for s in servers}

    def pick(self, client_ip: str = "") -> Server:
        # Smooth weighted round-robin: adjust current weight each call
        total_weight = sum(s.weight for s in self.servers)
        for s in self.servers:
            self._current_weights[s.server_id] += s.weight

        best = max(self.servers, key=lambda s: self._current_weights[s.server_id])
        self._current_weights[best.server_id] -= total_weight
        return best


class LeastConnections:
    def __init__(self, servers: List[Server]):
        self.servers = servers

    def pick(self, client_ip: str = "") -> Server:
        return min(self.servers, key=lambda s: s.active_conn)


class LeastResponseTime:
    def __init__(self, servers: List[Server]):
        self.servers = servers

    def pick(self, client_ip: str = "") -> Server:
        # Balance score: response time weighted by active connections
        def score(s: Server) -> float:
            return s.avg_response_ms * (1 + 0.1 * s.active_conn)
        return min(self.servers, key=score)


class IPHash:
    """
    Hash client IP to always route to the same backend.
    Provides session affinity without server-side state.
    """

    def __init__(self, servers: List[Server]):
        self.servers = servers

    def pick(self, client_ip: str) -> Server:
        h = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return self.servers[h % len(self.servers)]


class RandomPick:
    def __init__(self, servers: List[Server]):
        self.servers = servers

    def pick(self, client_ip: str = "") -> Server:
        return random.choice(self.servers)


class ResourceBased:
    """Routes to server with most available capacity (lowest CPU)."""

    def __init__(self, servers: List[Server]):
        self.servers = servers

    def pick(self, client_ip: str = "") -> Server:
        return min(self.servers, key=lambda s: s.cpu_pct)


# ─────────────────────────────────────────────
# SIMULATOR
# ─────────────────────────────────────────────

class AlgorithmSimulator:
    def __init__(self, algorithm_name: Algorithm, algo_instance):
        self.name      = algorithm_name.value
        self.algo      = algo_instance
        self.selections: Dict[str, int] = {}

    def run(self, requests: int, client_ips: List[str]) -> Dict[str, int]:
        self.selections = {}
        for i in range(requests):
            ip     = client_ips[i % len(client_ips)]
            server = self.algo.pick(ip)
            server.receive()
            # Simulate completion
            resp_time = server.avg_response_ms + random.uniform(-2, 4)
            server.complete(resp_time)
            self.selections[server.server_id] = self.selections.get(server.server_id, 0) + 1
        return self.selections

    def distribution_report(self, total: int):
        print(f"\n  [{self.name}]")
        for sid, count in sorted(self.selections.items()):
            pct = count / total * 100
            bar = "█" * (count * 40 // total)
            print(f"    {sid:<12} {count:>5} requests ({pct:5.1f}%)  {bar}")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_lb_algorithms():
    print("=" * 65)
    print("LOAD BALANCING ALGORITHMS")
    print("=" * 65)
    random.seed(42)

    N_REQUESTS = 100
    client_ips = [f"10.{i}.{j}.1" for i in range(5) for j in range(10)]

    # ── Round Robin ───────────────────────────
    print("\n[1] ROUND ROBIN (equal servers)")
    print("─" * 55)
    servers_rr = [Server(f"srv-{i}", weight=1) for i in range(1, 4)]
    sim = AlgorithmSimulator(Algorithm.ROUND_ROBIN, RoundRobin(servers_rr))
    sim.run(N_REQUESTS, client_ips)
    sim.distribution_report(N_REQUESTS)

    # ── Weighted Round Robin ──────────────────
    print("\n\n[2] WEIGHTED ROUND ROBIN (different capacities)")
    print("─" * 55)
    servers_wrr = [
        Server("large-1",  weight=4),   # 4x capacity
        Server("medium-2", weight=2),   # 2x capacity
        Server("small-3",  weight=1),   # baseline
    ]
    print("  Weights: large=4, medium=2, small=1  → expect 57%/29%/14% split")
    sim2 = AlgorithmSimulator(Algorithm.WEIGHTED_ROUND_ROBIN,
                               WeightedRoundRobin(servers_wrr))
    sim2.run(N_REQUESTS, client_ips)
    sim2.distribution_report(N_REQUESTS)

    # ── Least Connections ─────────────────────
    print("\n\n[3] LEAST CONNECTIONS (long-lived connections)")
    print("─" * 55)
    servers_lc = [
        Server("ws-1", active_conn=20),   # pre-loaded
        Server("ws-2", active_conn=5),
        Server("ws-3", active_conn=2),    # fewest — will get most traffic
    ]
    print("  Initial connections: ws-1=20, ws-2=5, ws-3=2")
    sim3 = AlgorithmSimulator(Algorithm.LEAST_CONNECTIONS,
                               LeastConnections(servers_lc))
    sim3.run(N_REQUESTS, client_ips)
    sim3.distribution_report(N_REQUESTS)

    # ── Least Response Time ───────────────────
    print("\n\n[4] LEAST RESPONSE TIME")
    print("─" * 55)
    servers_lrt = [
        Server("fast-1",  avg_response_ms=5.0),
        Server("avg-2",   avg_response_ms=20.0),
        Server("slow-3",  avg_response_ms=80.0),
    ]
    print("  Avg latency: fast=5ms, avg=20ms, slow=80ms")
    sim4 = AlgorithmSimulator(Algorithm.LEAST_RESPONSE_TIME,
                               LeastResponseTime(servers_lrt))
    sim4.run(N_REQUESTS, client_ips)
    sim4.distribution_report(N_REQUESTS)

    # ── IP Hash ───────────────────────────────
    print("\n\n[5] IP HASH (sticky sessions)")
    print("─" * 55)
    servers_ip = [Server(f"app-{i}", weight=1) for i in range(1, 4)]
    ip_hasher  = IPHash(servers_ip)
    print("  Same client IP always routes to same server:")
    test_ips = ["192.168.1.10", "10.0.0.1", "172.16.5.3", "192.168.1.10"]
    for ip in test_ips:
        s = ip_hasher.pick(ip)
        print(f"    {ip:<18} → {s.server_id}")
    sim5 = AlgorithmSimulator(Algorithm.IP_HASH, ip_hasher)
    sim5.run(N_REQUESTS, client_ips)
    sim5.distribution_report(N_REQUESTS)

    # ── Resource Based ────────────────────────
    print("\n\n[6] RESOURCE BASED (CPU-aware)")
    print("─" * 55)
    servers_rb = [
        Server("cpu-1", cpu_pct=80.0),   # overloaded
        Server("cpu-2", cpu_pct=40.0),
        Server("cpu-3", cpu_pct=15.0),   # most capacity
    ]
    print("  CPU: cpu-1=80%, cpu-2=40%, cpu-3=15%")
    sim6 = AlgorithmSimulator(Algorithm.RESOURCE_BASED,
                               ResourceBased(servers_rb))
    sim6.run(N_REQUESTS, client_ips)
    sim6.distribution_report(N_REQUESTS)

    # ── Algorithm Guide ───────────────────────
    print("\n\n[7] ALGORITHM DECISION GUIDE")
    print("─" * 55)
    guide = [
        ("Round Robin",           "Stateless APIs, identical backends, simple setup"),
        ("Weighted Round Robin",  "Heterogeneous backends (mix of instance sizes)"),
        ("Least Connections",     "Long-lived connections: WebSocket, file upload"),
        ("Least Response Time",   "Mixed-performance backends, latency-sensitive"),
        ("IP Hash",               "Session affinity without shared session store"),
        ("Random",                "Simplest; surprisingly good for short requests"),
        ("Resource Based",        "Adaptive routing with real-time CPU/memory data"),
    ]
    print(f"  {'Algorithm':<28} {'Best For'}")
    print(f"  {'─'*70}")
    for algo, use_case in guide:
        print(f"  {algo:<28} {use_case}")


if __name__ == "__main__":
    demonstrate_lb_algorithms()
