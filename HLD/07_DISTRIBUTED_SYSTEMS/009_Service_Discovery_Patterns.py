"""
SERVICE DISCOVERY PATTERNS
============================

Problem Statement:
In a dynamic distributed system, service instances come and go:
auto-scaling adds new pods, deployments roll, instances crash.
Clients must find healthy service endpoints without hardcoded IPs.
Service discovery solves: "Given service name 'payment-svc', what's the endpoint?"

Patterns:

  1. DNS-Based Discovery:
     Each service registered as a DNS SRV record.
     Client resolves DNS → gets IP:port.
     Simple. Relies on DNS TTL (may be stale). Built-in load balancing via DNS round-robin.
     Used by: Kubernetes (ClusterIP + kube-dns), AWS Route53 health checks.

  2. Client-Side Discovery:
     Client queries a registry (Consul, Eureka) for healthy instances.
     Client chooses instance via its own load-balancing logic (round-robin, etc.).
     More flexible (custom LB). Tighter coupling: client needs discovery SDK.
     Used by: Netflix Ribbon + Eureka, Consul + client library.

  3. Server-Side Discovery:
     Client sends request to a router/LB.
     Router queries registry and forwards to healthy instance.
     Client has no discovery logic. Registry hidden behind router.
     Used by: AWS ALB + ECS, Kubernetes kube-proxy, Envoy sidecar.

  4. Service Mesh (Envoy/Istio):
     Sidecar proxy runs alongside each service instance.
     Sidecar intercepts all outbound calls, queries xDS API for endpoints.
     Central control plane (Istio Pilot/Istiod) pushes config to sidecars.
     Transparent to application. Rich features: retries, circuit breaking, mTLS.
     Used by: Istio, Linkerd, Consul Connect.

Registry Options:
  Consul:      HTTP API + DNS, health checks, K/V store, multi-datacenter.
  etcd:        Kubernetes' backing store. Low-level, not purpose-built for discovery.
  ZooKeeper:   Older, complex ops. Still used in legacy systems (Kafka uses it optionally).
  Eureka:      Netflix OSS, Java-centric. Simple, highly available, AP.
  Kubernetes:  Built-in via Endpoints, Service objects, kube-dns.

Health Checks:
  TTL-based:  Service must re-register within TTL or be marked unhealthy.
  HTTP check: Registry polls /health endpoint.
  TCP check:  Registry verifies TCP connection.
  Deregistration: Service sends explicit deregister on graceful shutdown.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from collections import defaultdict
import time
import uuid
import threading
import random


# ─────────────────────────────────────────────
# SERVICE REGISTRATION
# ─────────────────────────────────────────────

@dataclass
class ServiceInstance:
    instance_id : str  = field(default_factory=lambda: str(uuid.uuid4())[:8])
    service_name: str  = ""
    host        : str  = ""
    port        : int  = 0
    metadata    : Dict = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    healthy     : bool = True

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


# ─────────────────────────────────────────────
# SERVICE REGISTRY (Consul/Eureka-like)
# ─────────────────────────────────────────────

class ServiceRegistry:
    """
    Central registry. Services register, send heartbeats, and deregister.
    Clients query for healthy instances of a given service.
    """

    def __init__(self, heartbeat_ttl_s: float = 3.0):
        self._services   : Dict[str, Dict[str, ServiceInstance]] = defaultdict(dict)
        self._lock       = threading.Lock()
        self.heartbeat_ttl_s = heartbeat_ttl_s
        self.registrations   = 0
        self.deregistrations = 0

    def register(self, instance: ServiceInstance):
        with self._lock:
            self._services[instance.service_name][instance.instance_id] = instance
            self.registrations += 1

    def deregister(self, service_name: str, instance_id: str):
        with self._lock:
            self._services[service_name].pop(instance_id, None)
            self.deregistrations += 1

    def heartbeat(self, service_name: str, instance_id: str) -> bool:
        with self._lock:
            instance = self._services.get(service_name, {}).get(instance_id)
            if not instance:
                return False
            instance.last_heartbeat = time.time()
            instance.healthy        = True
            return True

    def mark_unhealthy(self, service_name: str, instance_id: str):
        with self._lock:
            instance = self._services.get(service_name, {}).get(instance_id)
            if instance:
                instance.healthy = False

    def get_healthy(self, service_name: str) -> List[ServiceInstance]:
        with self._lock:
            now       = time.time()
            instances = self._services.get(service_name, {}).values()
            return [
                inst for inst in instances
                if inst.healthy and (now - inst.last_heartbeat) < self.heartbeat_ttl_s
            ]

    def expire_stale(self):
        """Remove instances that haven't sent a heartbeat within TTL."""
        with self._lock:
            now  = time.time()
            to_remove = []
            for svc_name, instances in self._services.items():
                for inst_id, inst in instances.items():
                    if now - inst.last_heartbeat > self.heartbeat_ttl_s:
                        to_remove.append((svc_name, inst_id))
            for svc_name, inst_id in to_remove:
                self._services[svc_name].pop(inst_id, None)
            return len(to_remove)


# ─────────────────────────────────────────────
# CLIENT-SIDE DISCOVERY (with load balancing)
# ─────────────────────────────────────────────

class LoadBalancingStrategy:
    ROUND_ROBIN = "round_robin"
    RANDOM      = "random"
    LEAST_CONN  = "least_connections"


class ClientSideDiscovery:
    """
    Client queries registry, applies its own load-balancing algorithm.
    Caches results for `cache_ttl_s` to reduce registry load.
    """

    def __init__(self, registry: ServiceRegistry, strategy: str = LoadBalancingStrategy.ROUND_ROBIN,
                 cache_ttl_s: float = 5.0):
        self.registry     = registry
        self.strategy     = strategy
        self.cache_ttl_s  = cache_ttl_s
        self._cache       : Dict[str, List[ServiceInstance]] = {}
        self._cache_time  : Dict[str, float]                 = {}
        self._rr_counters : Dict[str, int]                   = defaultdict(int)
        self._conn_counts : Dict[str, int]                   = defaultdict(int)
        self.cache_hits   = 0
        self.registry_queries = 0

    def _refresh(self, service_name: str) -> List[ServiceInstance]:
        self.registry_queries += 1
        instances = self.registry.get_healthy(service_name)
        self._cache[service_name]      = instances
        self._cache_time[service_name] = time.time()
        return instances

    def _get_instances(self, service_name: str) -> List[ServiceInstance]:
        now = time.time()
        if (service_name in self._cache and
                now - self._cache_time.get(service_name, 0) < self.cache_ttl_s):
            self.cache_hits += 1
            return self._cache[service_name]
        return self._refresh(service_name)

    def choose(self, service_name: str) -> Optional[ServiceInstance]:
        instances = self._get_instances(service_name)
        if not instances:
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            idx = self._rr_counters[service_name] % len(instances)
            self._rr_counters[service_name] += 1
            return instances[idx]

        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(instances)

        elif self.strategy == LoadBalancingStrategy.LEAST_CONN:
            return min(instances,
                       key=lambda i: self._conn_counts.get(i.instance_id, 0))

        return instances[0]

    def request(self, service_name: str) -> Optional[str]:
        instance = self.choose(service_name)
        if not instance:
            return None
        self._conn_counts[instance.instance_id] += 1
        result = f"→ {instance.address} (id={instance.instance_id})"
        self._conn_counts[instance.instance_id] -= 1
        return result


# ─────────────────────────────────────────────
# SERVER-SIDE DISCOVERY (Load Balancer + Registry)
# ─────────────────────────────────────────────

class ServerSideLoadBalancer:
    """
    Router queries registry for each request. Client only knows the LB address.
    Implements round-robin across healthy backends.
    """

    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self._counters: Dict[str, int] = defaultdict(int)
        self.requests_routed = 0
        self.requests_failed = 0

    def route(self, service_name: str) -> Optional[str]:
        instances = self.registry.get_healthy(service_name)
        if not instances:
            self.requests_failed += 1
            return None

        idx      = self._counters[service_name] % len(instances)
        self._counters[service_name] += 1
        instance = instances[idx]
        self.requests_routed += 1
        return instance.address


# ─────────────────────────────────────────────
# HEARTBEAT AGENT (runs in service instance)
# ─────────────────────────────────────────────

class HeartbeatAgent:
    """Background agent that renews service registration via heartbeats."""

    def __init__(self, registry: ServiceRegistry, instance: ServiceInstance,
                 interval_s: float = 1.0):
        self.registry    = registry
        self.instance    = instance
        self.interval    = interval_s
        self._running    = False
        self._thread     : Optional[threading.Thread] = None
        self.heartbeats  = 0

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            ok = self.registry.heartbeat(self.instance.service_name,
                                          self.instance.instance_id)
            if ok:
                self.heartbeats += 1
            time.sleep(self.interval)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_service_discovery():
    print("=" * 65)
    print("SERVICE DISCOVERY PATTERNS")
    print("=" * 65)

    registry = ServiceRegistry(heartbeat_ttl_s=0.5)

    # ── Service Registration ──────────────────────
    print("\n[1] SERVICE REGISTRATION AND DISCOVERY")
    print("─" * 55)

    # Register 3 payment-svc instances
    instances = []
    for i in range(3):
        inst = ServiceInstance(
            service_name = "payment-svc",
            host         = f"10.0.0.{i+1}",
            port         = 8080,
            metadata     = {"region": "us-east-1", "version": "1.2"},
        )
        registry.register(inst)
        instances.append(inst)
        print(f"  Registered {inst.service_name} @ {inst.address} (id={inst.instance_id})")

    healthy = registry.get_healthy("payment-svc")
    print(f"  Healthy instances: {len(healthy)}")

    # ── Client-Side Discovery ─────────────────────
    print("\n\n[2] CLIENT-SIDE DISCOVERY — ROUND-ROBIN")
    print("─" * 55)

    client = ClientSideDiscovery(registry, strategy=LoadBalancingStrategy.ROUND_ROBIN)
    print(f"  10 requests with round-robin:")
    for i in range(10):
        result = client.request("payment-svc")
        print(f"    req {i+1}: {result}")

    print(f"\n  Registry queries: {client.registry_queries}  "
          f"Cache hits: {client.cache_hits}")

    # ── Server-Side Discovery ─────────────────────
    print("\n\n[3] SERVER-SIDE DISCOVERY — LOAD BALANCER ROUTES")
    print("─" * 55)

    lb = ServerSideLoadBalancer(registry)
    print(f"  6 requests through load balancer:")
    for i in range(6):
        dest = lb.route("payment-svc")
        print(f"    req {i+1}: routed to {dest}")

    print(f"  Total routed: {lb.requests_routed}")

    # ── Heartbeat + TTL Expiry ────────────────────
    print("\n\n[4] HEARTBEAT FAILURE → INSTANCE EXPIRY")
    print("─" * 55)

    registry2 = ServiceRegistry(heartbeat_ttl_s=0.3)
    inst_a = ServiceInstance(service_name="cache-svc", host="10.1.1.1", port=6379)
    inst_b = ServiceInstance(service_name="cache-svc", host="10.1.1.2", port=6379)
    registry2.register(inst_a)
    registry2.register(inst_b)

    # inst_a sends heartbeats; inst_b does not (crashed)
    agent_a = HeartbeatAgent(registry2, inst_a, interval_s=0.1)
    agent_a.start()

    time.sleep(0.5)   # inst_b's TTL expires (0.3s)
    agent_a.stop()

    expired = registry2.expire_stale()
    healthy_after = registry2.get_healthy("cache-svc")
    print(f"  Instance A (heartbeating): healthy={inst_a.healthy}")
    print(f"  Instance B (no heartbeat): expired after {registry2.heartbeat_ttl_s}s TTL")
    print(f"  Expired instances removed: {expired}")
    print(f"  Healthy after expiry: {[h.address for h in healthy_after]}")
    print(f"  Agent A sent {agent_a.heartbeats} heartbeats")

    # ── Pattern Comparison ────────────────────────
    print("\n\n[5] DISCOVERY PATTERN COMPARISON")
    print("─" * 55)
    patterns = [
        ("DNS-based",       "Simple, built-in", "Stale TTL, no LB logic", "k8s, Route53"),
        ("Client-side",     "Flexible LB",      "Client needs SDK",       "Ribbon+Eureka"),
        ("Server-side",     "Client ignorant",  "LB becomes bottleneck",  "ALB+ECS, k8s"),
        ("Service mesh",    "Transparent, rich","Sidecar complexity",      "Istio, Linkerd"),
    ]
    print(f"  {'Pattern':<16} {'Benefit':<22} {'Drawback':<24} {'Examples'}")
    print(f"  {'─'*78}")
    for pattern, benefit, drawback, examples in patterns:
        print(f"  {pattern:<16} {benefit:<22} {drawback:<24} {examples}")

    print("\n\n[6] HEALTH CHECK TYPES")
    print("─" * 55)
    checks = [
        ("TTL-based",   "Service sends heartbeat before TTL expires"),
        ("HTTP check",  "Registry polls GET /health; expects 200"),
        ("TCP check",   "Registry verifies TCP connection accepted"),
        ("Script check","Registry runs custom script (exit 0 = healthy)"),
        ("gRPC check",  "Registry uses gRPC Health Check Protocol"),
    ]
    for check_type, description in checks:
        print(f"  {check_type:<16} {description}")


if __name__ == "__main__":
    demonstrate_service_discovery()
