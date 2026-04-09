"""
SERVICE MESH ARCHITECTURE
==========================

Problem Statement:
In a microservices cluster, every service needs to: discover other services,
load balance requests, retry failures, encrypt traffic, and emit telemetry.
Duplicating this logic in every service is impractical. A service mesh moves
these concerns into a sidecar proxy (Envoy) deployed alongside each service.

How It Works:
  Each pod gets a sidecar proxy (e.g., Envoy).
  All traffic flows through sidecars — services talk to localhost.
  The control plane (Istio Pilot) configures all sidecars centrally.

  Service A → Sidecar-A → Sidecar-B → Service B
               (mTLS)      (mTLS)

Key Features:
  - mTLS: automatic mutual TLS between all services
  - Observability: metrics, traces, logs without app changes
  - Traffic Management: retries, timeouts, circuit breaking, canary
  - Service Discovery: automatic — no hardcoded addresses
  - Policy: access control, rate limiting centrally enforced

Popular Implementations:
  Istio (Envoy sidecar), Linkerd, Consul Connect, AWS App Mesh
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import random
import uuid


class TrafficPolicy(Enum):
    ROUND_ROBIN    = "round_robin"
    LEAST_CONN     = "least_conn"
    RANDOM         = "random"
    CANARY         = "canary"


class CircuitState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


@dataclass
class ServiceInstance:
    instance_id  : str
    service_name : str
    host         : str
    port         : int
    healthy      : bool = True
    version      : str  = "v1"
    weight       : int  = 100   # for weighted routing


@dataclass
class MeshRequest:
    request_id   : str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source       : str = ""
    destination  : str = ""
    method       : str = "GET"
    path         : str = "/"
    retries_left : int = 3
    timeout_ms   : float = 1000.0


@dataclass
class MeshResponse:
    status_code : int
    body        : str
    latency_ms  : float = 0.0
    retries_used: int = 0
    encrypted   : bool = True  # always mTLS in mesh
    served_by   : str = ""


# ─────────────────────────────────────────────
# SERVICE REGISTRY (Control Plane data)
# ─────────────────────────────────────────────

class ServiceRegistry:
    """Tracks all service instances in the mesh."""

    def __init__(self):
        self._services: Dict[str, List[ServiceInstance]] = {}

    def register(self, instance: ServiceInstance):
        self._services.setdefault(instance.service_name, []).append(instance)

    def deregister(self, instance_id: str, service_name: str):
        self._services[service_name] = [
            i for i in self._services.get(service_name, [])
            if i.instance_id != instance_id
        ]

    def get_healthy(self, service_name: str) -> List[ServiceInstance]:
        return [i for i in self._services.get(service_name, []) if i.healthy]

    def list_all(self):
        print(f"\n  Service Registry:")
        for svc, instances in self._services.items():
            for inst in instances:
                status = "✅" if inst.healthy else "❌"
                print(f"    {status} {inst.service_name} [{inst.version}] "
                      f"{inst.host}:{inst.port} (weight={inst.weight})")


# ─────────────────────────────────────────────
# CIRCUIT BREAKER (per destination)
# ─────────────────────────────────────────────

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout_s: float = 5.0):
        self.failure_threshold  = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.state              = CircuitState.CLOSED
        self._failures          = 0
        self._last_failure_time = 0.0

    def record_success(self):
        self._failures = 0
        self.state     = CircuitState.CLOSED

    def record_failure(self):
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self._last_failure_time > self.recovery_timeout_s:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True   # HALF_OPEN: allow one probe


# ─────────────────────────────────────────────
# SIDECAR PROXY (Envoy-like)
# ─────────────────────────────────────────────

class SidecarProxy:
    """
    Sidecar deployed alongside each service instance.
    Intercepts all inbound and outbound traffic.
    """

    def __init__(self, service_name: str, registry: ServiceRegistry):
        self.service_name    = service_name
        self.registry        = registry
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._rr_counters     : Dict[str, int] = {}
        self.metrics          = {
            "requests_out": 0, "requests_in": 0,
            "retries": 0, "circuit_opens": 0, "errors": 0
        }
        self.traces: List[Dict] = []

    def _get_cb(self, dest: str) -> CircuitBreaker:
        if dest not in self._circuit_breakers:
            self._circuit_breakers[dest] = CircuitBreaker()
        return self._circuit_breakers[dest]

    def _load_balance(self, instances: List[ServiceInstance],
                       policy: TrafficPolicy) -> Optional[ServiceInstance]:
        if not instances:
            return None
        if policy == TrafficPolicy.ROUND_ROBIN:
            key = instances[0].service_name
            idx = self._rr_counters.get(key, 0) % len(instances)
            self._rr_counters[key] = idx + 1
            return instances[idx]
        if policy == TrafficPolicy.RANDOM:
            return random.choice(instances)
        if policy == TrafficPolicy.LEAST_CONN:
            return instances[0]   # simplified
        return instances[0]

    def send(self, req: MeshRequest,
             policy: TrafficPolicy = TrafficPolicy.ROUND_ROBIN) -> MeshResponse:
        self.metrics["requests_out"] += 1
        start = time.perf_counter()

        cb = self._get_cb(req.destination)
        if not cb.allow_request():
            self.metrics["circuit_opens"] += 1
            return MeshResponse(503, "Circuit open — destination unavailable",
                                 served_by=self.service_name)

        instances = self.registry.get_healthy(req.destination)
        instance  = self._load_balance(instances, policy)
        if not instance:
            return MeshResponse(503, f"No healthy instances for {req.destination}",
                                 served_by=self.service_name)

        # Simulate call with possible failure
        retries_used = 0
        for attempt in range(req.retries_left + 1):
            success = random.random() > 0.2   # 20% failure rate
            if success:
                cb.record_success()
                latency = round((time.perf_counter() - start) * 1000 + 5 + attempt * 10, 2)
                resp = MeshResponse(
                    status_code=200,
                    body=f"OK from {instance.instance_id}",
                    latency_ms=latency,
                    retries_used=retries_used,
                    encrypted=True,
                    served_by=instance.instance_id
                )
                self.traces.append({
                    "req_id": req.request_id, "src": req.source,
                    "dst": req.destination, "status": 200,
                    "latency": latency, "retries": retries_used,
                    "instance": instance.instance_id
                })
                return resp
            else:
                cb.record_failure()
                retries_used += 1
                self.metrics["retries"] += 1
                if attempt < req.retries_left:
                    print(f"    Sidecar [{self.service_name}]: retry {attempt+1} for {req.destination}")

        self.metrics["errors"] += 1
        return MeshResponse(503, "Max retries exceeded",
                             latency_ms=round((time.perf_counter() - start) * 1000, 2),
                             retries_used=retries_used, served_by=instance.instance_id)


# ─────────────────────────────────────────────
# CONTROL PLANE (Istiod-like)
# ─────────────────────────────────────────────

class ControlPlane:
    """
    Configures all sidecar proxies centrally.
    Pushes routing rules, policies, and TLS certs.
    """

    def __init__(self, registry: ServiceRegistry):
        self.registry   = registry
        self._policies  : Dict[str, Dict] = {}

    def apply_traffic_policy(self, service: str, retries: int, timeout_ms: float,
                              canary_version: str = None, canary_weight: int = 0):
        self._policies[service] = {
            "retries": retries, "timeout_ms": timeout_ms,
            "canary_version": canary_version, "canary_weight": canary_weight
        }
        print(f"  ControlPlane: applied policy for {service}: "
              f"retries={retries}, timeout={timeout_ms}ms"
              + (f", canary={canary_version}@{canary_weight}%" if canary_version else ""))

    def apply_canary_weights(self, service_name: str, v1_weight: int, v2_weight: int):
        instances = self.registry._services.get(service_name, [])
        for inst in instances:
            if inst.version == "v1":
                inst.weight = v1_weight
            elif inst.version == "v2":
                inst.weight = v2_weight
        print(f"  ControlPlane: canary split for {service_name}: v1={v1_weight}% v2={v2_weight}%")

    def show_policies(self):
        print(f"\n  Active Traffic Policies:")
        for svc, policy in self._policies.items():
            print(f"    {svc}: {policy}")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_service_mesh():
    print("=" * 65)
    print("SERVICE MESH ARCHITECTURE")
    print("=" * 65)

    # ── Setup Registry ────────────────────────
    print("\n[1] SERVICE REGISTRY & INSTANCES")
    print("─" * 55)
    registry = ServiceRegistry()
    instances = [
        ServiceInstance("user-1",    "user-service",    "10.0.1.1", 8080, version="v1"),
        ServiceInstance("user-2",    "user-service",    "10.0.1.2", 8080, version="v1"),
        ServiceInstance("order-1",   "order-service",   "10.0.2.1", 8080, version="v1"),
        ServiceInstance("order-2",   "order-service",   "10.0.2.2", 8080, version="v2", weight=20),
        ServiceInstance("payment-1", "payment-service", "10.0.3.1", 8080, version="v1"),
        ServiceInstance("notif-1",   "notif-service",   "10.0.4.1", 8080, version="v1", healthy=False),
    ]
    for inst in instances:
        registry.register(inst)
    registry.list_all()

    # ── Control Plane ─────────────────────────
    print("\n\n[2] CONTROL PLANE — TRAFFIC POLICIES")
    print("─" * 55)
    cp = ControlPlane(registry)
    cp.apply_traffic_policy("order-service",   retries=3, timeout_ms=500,
                             canary_version="v2", canary_weight=20)
    cp.apply_traffic_policy("payment-service", retries=2, timeout_ms=2000)
    cp.apply_canary_weights("order-service", v1_weight=80, v2_weight=20)
    cp.show_policies()

    # ── Sidecar Proxy ─────────────────────────
    print("\n\n[3] SIDECAR PROXY — REQUESTS WITH RETRIES + mTLS")
    print("─" * 55)
    random.seed(42)
    checkout_sidecar = SidecarProxy("checkout-service", registry)

    print("\n  Sending 6 requests from checkout → order-service:")
    for i in range(6):
        req = MeshRequest(
            source="checkout-service",
            destination="order-service",
            method="POST",
            path="/orders",
            retries_left=2
        )
        resp = checkout_sidecar.send(req, TrafficPolicy.ROUND_ROBIN)
        print(f"  [{i+1}] status={resp.status_code}  served_by={resp.served_by}  "
              f"retries={resp.retries_used}  mTLS={resp.encrypted}  {resp.latency_ms}ms")

    # ── Circuit Breaker ───────────────────────
    print("\n\n[4] CIRCUIT BREAKER BEHAVIOR")
    print("─" * 55)
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_s=2.0)
    print("  Simulating 5 consecutive failures:")
    for i in range(5):
        allowed = cb.allow_request()
        if allowed:
            cb.record_failure()
            print(f"  Attempt {i+1}: allowed=True  state={cb.state.value}  failures={cb._failures}")
        else:
            print(f"  Attempt {i+1}: allowed=False  state={cb.state.value}  (circuit OPEN)")

    # ── Observability ─────────────────────────
    print("\n\n[5] DISTRIBUTED TRACES (from sidecar)")
    print("─" * 55)
    if checkout_sidecar.traces:
        print(f"  {'ReqID':<10} {'Src':<20} {'Dst':<18} {'Status':<7} {'ms':<8} {'Retries'}")
        print(f"  {'─'*70}")
        for t in checkout_sidecar.traces[:5]:
            print(f"  {t['req_id']:<10} {t['src']:<20} {t['dst']:<18} "
                  f"{t['status']:<7} {t['latency']:<8} {t['retries']}")

    # ── Comparison ────────────────────────────
    print("\n\n[6] SERVICE MESH vs API GATEWAY")
    print("─" * 55)
    rows = [
        ("Traffic direction", "East-West (svc↔svc)", "North-South (client→svc)"),
        ("Location",          "Sidecar per pod",      "Edge / cluster boundary"),
        ("Encryption",        "mTLS auto everywhere", "SSL termination at edge"),
        ("Discovery",         "Built-in (all svcs)",  "Manual route config"),
        ("Retries/CB",        "Per-sidecar",          "Per-route config"),
        ("Observability",     "Full mesh traces",     "Edge access logs"),
        ("Auth",              "SPIFFE/X.509 certs",   "JWT / API keys"),
        ("Use when",          "Many internal svcs",   "External client API"),
    ]
    print(f"  {'Aspect':<22} {'Service Mesh':<28} {'API Gateway'}")
    print(f"  {'─'*75}")
    for aspect, mesh_v, gw_v in rows:
        print(f"  {aspect:<22} {mesh_v:<28} {gw_v}")


if __name__ == "__main__":
    demonstrate_service_mesh()
