"""
LOAD BALANCING FUNDAMENTALS
=============================

Problem Statement:
A single server can handle only so many requests. As traffic grows, you need
to distribute requests across multiple servers to increase capacity and
eliminate the single point of failure. Load balancers are the mechanism
that makes horizontal scaling work.

Load Balancer Responsibilities:
  1. Traffic distribution — spread requests across backend pool
  2. Health checking — remove unhealthy backends automatically
  3. Connection draining — graceful shutdown during deployments
  4. Session persistence — sticky sessions if needed
  5. SSL termination — offload TLS from application servers

Load Balancer Types:
  L4 (Transport): routes based on IP/port (TCP/UDP) — faster, no content
  L7 (Application): routes based on HTTP headers/path/cookies — smarter

Key Algorithms (next file covers these in depth):
  Round Robin, Weighted RR, Least Connections, IP Hash, Random

Deployment Topologies:
  Single LB         → SPOF, not recommended for production
  Active-Passive LB → failover via floating IP / DNS
  Active-Active LB  → anycast or DNS round-robin across LBs
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import random
import uuid


class LBLayer(Enum):
    L4 = "L4_transport"   # TCP/UDP level
    L7 = "L7_application" # HTTP level


class HealthStatus(Enum):
    HEALTHY   = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING  = "draining"   # graceful shutdown


@dataclass
class Backend:
    backend_id   : str
    host         : str
    port         : int
    weight       : int = 1
    status       : HealthStatus = HealthStatus.HEALTHY
    active_conn  : int = 0
    total_requests: int = 0
    failed_requests: int = 0
    response_time_ms: float = 10.0   # simulated avg response time

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def is_available(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    def handle_request(self) -> Tuple[int, float]:
        self.total_requests += 1
        self.active_conn   += 1
        # Simulate occasional failure
        if random.random() < 0.05:   # 5% failure rate
            self.failed_requests += 1
            self.active_conn     -= 1
            return 500, 0.0
        latency = self.response_time_ms + random.uniform(-2, 5)
        self.active_conn -= 1
        return 200, latency


@dataclass
class LBRequest:
    request_id  : str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    client_ip   : str = "0.0.0.0"
    path        : str = "/"
    method      : str = "GET"
    content_type: str = "application/json"


@dataclass
class LBResponse:
    request_id  : str
    status_code : int
    backend_id  : str
    latency_ms  : float
    from_cache  : bool = False


# ─────────────────────────────────────────────
# HEALTH CHECKER
# ─────────────────────────────────────────────

class HealthChecker:
    """
    Periodically probes backends to detect failures.
    Removes unhealthy backends and re-adds when healthy again.
    """

    def __init__(self, interval_s: float = 5.0, threshold_down: int = 3,
                 threshold_up: int = 2):
        self.interval_s      = interval_s
        self.threshold_down  = threshold_down   # consecutive failures to mark down
        self.threshold_up    = threshold_up     # consecutive successes to re-add
        self._fail_counts    : Dict[str, int] = {}
        self._success_counts : Dict[str, int] = {}
        self.events          : List[str] = []

    def probe(self, backend: Backend) -> bool:
        """Returns True if healthy. Simulates HTTP GET /health."""
        # Simulate: backend with "bad" in ID fails health checks
        is_healthy = "bad" not in backend.backend_id
        if not is_healthy:
            self._fail_counts[backend.backend_id] = \
                self._fail_counts.get(backend.backend_id, 0) + 1
            self._success_counts[backend.backend_id] = 0
        else:
            self._success_counts[backend.backend_id] = \
                self._success_counts.get(backend.backend_id, 0) + 1
            self._fail_counts[backend.backend_id] = 0
        return is_healthy

    def run_checks(self, backends: List[Backend]):
        for backend in backends:
            healthy = self.probe(backend)
            fails   = self._fail_counts.get(backend.backend_id, 0)
            success = self._success_counts.get(backend.backend_id, 0)

            if backend.status == HealthStatus.HEALTHY and fails >= self.threshold_down:
                backend.status = HealthStatus.UNHEALTHY
                msg = f"  HealthChecker: {backend.backend_id} marked UNHEALTHY (fails={fails})"
                self.events.append(msg)
                print(msg)
            elif backend.status == HealthStatus.UNHEALTHY and success >= self.threshold_up:
                backend.status = HealthStatus.HEALTHY
                msg = f"  HealthChecker: {backend.backend_id} re-added to pool (successes={success})"
                self.events.append(msg)
                print(msg)
            else:
                status_icon = "✅" if healthy else "❌"
                print(f"  HealthChecker: {status_icon} {backend.backend_id} "
                      f"→ {'ok' if healthy else f'fail #{fails}'}")


# ─────────────────────────────────────────────
# BASE LOAD BALANCER
# ─────────────────────────────────────────────

class LoadBalancer:
    """Base class — subclasses implement specific algorithms."""

    def __init__(self, name: str, layer: LBLayer = LBLayer.L7):
        self.name         = name
        self.layer        = layer
        self.backends     : List[Backend] = []
        self.health_checker = HealthChecker()
        self.total_requests = 0
        self.failed_requests = 0
        self.response_log   : List[LBResponse] = []

    def add_backend(self, backend: Backend):
        self.backends.append(backend)

    def healthy_backends(self) -> List[Backend]:
        return [b for b in self.backends if b.is_available]

    def select_backend(self, req: LBRequest) -> Optional[Backend]:
        raise NotImplementedError

    def handle(self, req: LBRequest) -> LBResponse:
        self.total_requests += 1
        backend = self.select_backend(req)
        if not backend:
            self.failed_requests += 1
            return LBResponse(req.request_id, 503, "none", 0.0)

        status, latency = backend.handle_request()
        if status >= 500:
            self.failed_requests += 1

        resp = LBResponse(req.request_id, status, backend.backend_id, round(latency, 2))
        self.response_log.append(resp)
        return resp

    def run_health_checks(self):
        self.health_checker.run_checks(self.backends)

    def report(self):
        healthy = len(self.healthy_backends())
        print(f"\n  LoadBalancer [{self.name}]:")
        print(f"    Total requests : {self.total_requests}")
        print(f"    Failed         : {self.failed_requests}")
        print(f"    Success rate   : {(1 - self.failed_requests/max(1,self.total_requests)):.1%}")
        print(f"    Healthy backends: {healthy}/{len(self.backends)}")
        print(f"\n  Backend Stats:")
        for b in self.backends:
            icon = "✅" if b.is_available else "❌"
            print(f"    {icon} {b.backend_id:<15} requests={b.total_requests:<6} "
                  f"errors={b.failed_requests:<4} err_rate={b.error_rate:.1%}")


# ─────────────────────────────────────────────
# ROUND ROBIN LB
# ─────────────────────────────────────────────

class RoundRobinLB(LoadBalancer):
    def __init__(self, name: str = "RoundRobin"):
        super().__init__(name)
        self._index = 0

    def select_backend(self, req: LBRequest) -> Optional[Backend]:
        available = self.healthy_backends()
        if not available:
            return None
        backend = available[self._index % len(available)]
        self._index += 1
        return backend


# ─────────────────────────────────────────────
# LOAD BALANCER TOPOLOGY ANALYZER
# ─────────────────────────────────────────────

class TopologyAnalyzer:
    """Calculates availability for different LB topologies."""

    @staticmethod
    def single_lb_availability(lb_uptime: float, server_uptime: float,
                                 n_servers: int) -> float:
        """Single LB is a SPOF — its downtime = system downtime."""
        server_pool = 1 - (1 - server_uptime) ** n_servers
        return lb_uptime * server_pool

    @staticmethod
    def active_passive_availability(lb_uptime: float, server_uptime: float,
                                     n_servers: int) -> float:
        """Two LBs in active-passive — both must fail for outage."""
        lb_pool     = 1 - (1 - lb_uptime) ** 2
        server_pool = 1 - (1 - server_uptime) ** n_servers
        return lb_pool * server_pool

    @staticmethod
    def active_active_availability(lb_uptime: float, server_uptime: float,
                                    n_lbs: int, n_servers: int) -> float:
        """All N LBs must fail simultaneously for outage."""
        lb_pool     = 1 - (1 - lb_uptime) ** n_lbs
        server_pool = 1 - (1 - server_uptime) ** n_servers
        return lb_pool * server_pool

    @staticmethod
    def downtime_minutes_per_year(availability: float) -> float:
        return (1 - availability) * 525600   # minutes in a year


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_lb_fundamentals():
    print("=" * 65)
    print("LOAD BALANCING FUNDAMENTALS")
    print("=" * 65)

    random.seed(42)

    # ── Setup ─────────────────────────────────
    print("\n[1] LOAD BALANCER WITH HEALTH CHECKS")
    print("─" * 55)
    lb = RoundRobinLB("prod-lb")
    backends = [
        Backend("web-1", "10.0.1.1", 8080, response_time_ms=8.0),
        Backend("web-2", "10.0.1.2", 8080, response_time_ms=12.0),
        Backend("bad-3", "10.0.1.3", 8080, response_time_ms=5.0),  # will fail health check
        Backend("web-4", "10.0.1.4", 8080, response_time_ms=10.0),
    ]
    for b in backends:
        lb.add_backend(b)

    print("\n  Running health checks:")
    lb.run_health_checks()
    lb.run_health_checks()
    lb.run_health_checks()

    # ── Traffic Distribution ──────────────────
    print("\n\n  Sending 20 requests:")
    for i in range(20):
        req  = LBRequest(client_ip=f"1.2.3.{i}", path="/api/data")
        resp = lb.handle(req)
        if i < 8:
            print(f"  [{i+1:02d}] → {resp.backend_id:<12} status={resp.status_code}  {resp.latency_ms:.1f}ms")

    lb.report()

    # ── Topology Availability ─────────────────
    print("\n\n[2] TOPOLOGY AVAILABILITY COMPARISON")
    print("─" * 55)
    ta = TopologyAnalyzer()
    lb_uptime  = 0.9999    # 99.99%
    srv_uptime = 0.999     # 99.9%
    n_servers  = 3

    topologies = [
        ("Single LB (SPOF)",
         ta.single_lb_availability(lb_uptime, srv_uptime, n_servers)),
        ("Active-Passive (2 LBs)",
         ta.active_passive_availability(lb_uptime, srv_uptime, n_servers)),
        ("Active-Active (3 LBs)",
         ta.active_active_availability(lb_uptime, srv_uptime, 3, n_servers)),
    ]
    print(f"  (LB uptime={lb_uptime:.2%}, server uptime={srv_uptime:.2%}, {n_servers} servers)")
    print(f"\n  {'Topology':<30} {'Availability':<15} {'Downtime/year'}")
    print(f"  {'─'*60}")
    for name, avail in topologies:
        downtime = ta.downtime_minutes_per_year(avail)
        print(f"  {name:<30} {avail:.6%}   {downtime:.1f} min/year")

    # ── L4 vs L7 ──────────────────────────────
    print("\n\n[3] L4 vs L7 LOAD BALANCER")
    print("─" * 55)
    rows = [
        ("OSI Layer",       "Layer 4 (Transport)",        "Layer 7 (Application)"),
        ("Routing basis",   "IP address + port",          "URL, headers, cookies, body"),
        ("Protocol aware",  "No — just TCP/UDP bytes",    "Yes — understands HTTP/S"),
        ("SSL termination", "No (pass-through only)",     "Yes — full SSL termination"),
        ("Content-based",   "No — can't read HTTP",       "Yes — /api → svc-A, /ui → svc-B"),
        ("Performance",     "Faster — less processing",   "Slightly slower — parses HTTP"),
        ("Sticky sessions", "IP hash only",               "Cookie-based, header-based"),
        ("Use case",        "UDP/TCP gaming, DNS",        "HTTP APIs, microservices"),
        ("Examples",        "AWS NLB, HAProxy L4",        "AWS ALB, Nginx, Traefik"),
    ]
    print(f"  {'Aspect':<20} {'L4':<30} {'L7'}")
    print(f"  {'─'*75}")
    for aspect, l4, l7 in rows:
        print(f"  {aspect:<20} {l4:<30} {l7}")

    # ── Connection Draining ───────────────────
    print("\n\n[4] GRACEFUL SHUTDOWN — CONNECTION DRAINING")
    print("─" * 55)
    drain_steps = [
        ("1", "Mark backend as DRAINING in LB pool"),
        ("2", "Stop routing NEW requests to draining backend"),
        ("3", "Wait for in-flight requests to complete (e.g., 30s timeout)"),
        ("4", "After drain period, safely shut down the instance"),
        ("5", "Health check detects it gone → removes from pool"),
    ]
    for step, desc in drain_steps:
        print(f"  Step {step}: {desc}")


if __name__ == "__main__":
    demonstrate_lb_fundamentals()
