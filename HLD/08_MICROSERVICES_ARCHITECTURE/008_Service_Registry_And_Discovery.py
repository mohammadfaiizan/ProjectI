"""
SERVICE REGISTRY AND DISCOVERY (Microservices Context)
=========================================================

Problem Statement:
In microservices, service instances are ephemeral. Containers restart,
scale up/down, move to different hosts. You can't hardcode IPs.
Services must discover each other dynamically.

Core Concepts:
  Registry:   A centralized database of service instances.
              Stores: service name, instance ID, host, port, health status, metadata.
  Registration: Services announce themselves on startup; deregister on shutdown.
  Discovery:  A consumer queries the registry to find healthy instances.

Two Discovery Models:

  1. Client-Side Discovery:
     Consumer queries registry directly, then picks an instance (load balancing
     happens in the client library — e.g., Ribbon/Netflix Eureka).
     Pro:  No extra hop. Client controls load balancing strategy.
     Con:  Every service needs a discovery library. Tightly coupled to registry.

  2. Server-Side Discovery:
     Consumer calls a load balancer or service mesh. LB queries registry.
     (e.g., AWS ALB + ECS, Kubernetes Service + kube-proxy).
     Pro:  Service code is discovery-agnostic.
     Con:  Extra network hop; LB is another component to manage.

Registration Patterns:
  Self-Registration: Service registers itself on startup (simpler, but
                     requires service to know the registry URL).
  Third-Party Registration: Orchestrator (K8s, Consul agent) registers the
                     service (better separation of concerns).

Health-Check Driven Routing:
  Registry only returns instances that pass health checks.
  Health check types: HTTP GET /health, TCP ping, TTL (service sends heartbeat).
  If a service stops sending heartbeats → removed from registry → no traffic.

Microservices-Specific Patterns:
  Consul Connect:   Service mesh with sidecar proxy. Automatic mTLS + discovery.
  Kubernetes DNS:   Every Service gets a stable DNS name (svc-name.namespace.svc.cluster.local).
                    No library needed; just DNS resolution.
  Eureka (Netflix): Client-side discovery. Services register; clients cache registry.
  Service Mesh:     Discovery + load balancing + mTLS in the data plane.
                    Application code just connects to localhost; sidecar handles rest.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import time
import uuid
import threading
import random


# ─────────────────────────────────────────────
# SERVICE INSTANCE
# ─────────────────────────────────────────────

class HealthStatus(Enum):
    PASSING  = "passing"
    WARNING  = "warning"
    CRITICAL = "critical"


@dataclass
class ServiceInstance:
    service_name : str
    instance_id  : str
    host         : str
    port         : int
    version      : str
    tags         : List[str] = field(default_factory=list)
    metadata     : Dict[str, str] = field(default_factory=dict)
    health       : HealthStatus = HealthStatus.PASSING
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    weight       : int = 100   # for weighted load balancing

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def is_healthy(self) -> bool:
        return self.health == HealthStatus.PASSING


# ─────────────────────────────────────────────
# SERVICE REGISTRY (Consul/Eureka style)
# ─────────────────────────────────────────────

class ServiceRegistry:
    """
    Central registry. Tracks all service instances and their health.
    Supports: registration, deregistration, health updates, TTL expiry.
    """

    TTL_SECONDS = 30   # instance removed if no heartbeat within TTL

    def __init__(self):
        self._instances : Dict[str, ServiceInstance] = {}   # instance_id → instance
        self._lock      = threading.Lock()
        self._watchers  : Dict[str, List[callable]] = {}    # service_name → callbacks
        self._event_log : List[Dict] = []

    def register(self, instance: ServiceInstance):
        with self._lock:
            self._instances[instance.instance_id] = instance
            self._log("registered", instance)
            self._notify(instance.service_name, "registered", instance)

    def deregister(self, instance_id: str):
        with self._lock:
            inst = self._instances.pop(instance_id, None)
            if inst:
                self._log("deregistered", inst)
                self._notify(inst.service_name, "deregistered", inst)

    def heartbeat(self, instance_id: str) -> bool:
        with self._lock:
            inst = self._instances.get(instance_id)
            if inst:
                inst.last_heartbeat = time.time()
                inst.health = HealthStatus.PASSING
                return True
            return False

    def update_health(self, instance_id: str, status: HealthStatus):
        with self._lock:
            inst = self._instances.get(instance_id)
            if inst:
                old = inst.health
                inst.health = status
                if old != status:
                    self._log(f"health:{status.value}", inst)
                    self._notify(inst.service_name, f"health:{status.value}", inst)

    def discover(self, service_name: str,
                 tag: Optional[str] = None,
                 healthy_only: bool = True) -> List[ServiceInstance]:
        with self._lock:
            results = [
                inst for inst in self._instances.values()
                if inst.service_name == service_name
                and (not healthy_only or inst.is_healthy)
                and (tag is None or tag in inst.tags)
            ]
            return list(results)

    def evict_stale(self) -> int:
        """Remove instances that haven't sent a heartbeat within TTL."""
        cutoff = time.time() - self.TTL_SECONDS
        to_remove = []
        with self._lock:
            for iid, inst in self._instances.items():
                if inst.last_heartbeat < cutoff:
                    to_remove.append(iid)
            for iid in to_remove:
                inst = self._instances.pop(iid)
                self._log("ttl_expired", inst)
        return len(to_remove)

    def watch(self, service_name: str, callback: callable):
        self._watchers.setdefault(service_name, []).append(callback)

    def _notify(self, service_name: str, event: str, instance: ServiceInstance):
        for cb in self._watchers.get(service_name, []):
            threading.Thread(target=cb,
                             args=(event, instance), daemon=True).start()

    def _log(self, event: str, inst: ServiceInstance):
        self._event_log.append({
            "event"      : event,
            "service"    : inst.service_name,
            "instance_id": inst.instance_id,
            "address"    : inst.address,
            "ts"         : time.time(),
        })

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            by_service: Dict[str, Dict] = {}
            for inst in self._instances.values():
                s = inst.service_name
                if s not in by_service:
                    by_service[s] = {"total": 0, "healthy": 0, "instances": []}
                by_service[s]["total"] += 1
                if inst.is_healthy:
                    by_service[s]["healthy"] += 1
                by_service[s]["instances"].append(inst.address)
            return by_service


# ─────────────────────────────────────────────
# LOAD BALANCING STRATEGIES
# ─────────────────────────────────────────────

class LoadBalancer:
    """Client-side load balancer. Queries registry; picks instance."""

    def __init__(self, registry: ServiceRegistry, strategy: str = "round_robin"):
        self.registry = registry
        self.strategy = strategy
        self._rr_idx  : Dict[str, int] = {}

    def pick(self, service_name: str) -> Optional[ServiceInstance]:
        instances = self.registry.discover(service_name, healthy_only=True)
        if not instances:
            return None

        if self.strategy == "round_robin":
            idx = self._rr_idx.get(service_name, 0)
            inst = instances[idx % len(instances)]
            self._rr_idx[service_name] = idx + 1
            return inst

        if self.strategy == "random":
            return random.choice(instances)

        if self.strategy == "weighted":
            total_weight = sum(i.weight for i in instances)
            r = random.uniform(0, total_weight)
            cumulative = 0
            for inst in instances:
                cumulative += inst.weight
                if r <= cumulative:
                    return inst
            return instances[-1]

        if self.strategy == "least_connections":
            # Simplified: use weight as proxy for load
            return min(instances, key=lambda i: i.weight)

        return instances[0]


# ─────────────────────────────────────────────
# KUBERNETES DNS-STYLE DISCOVERY STUB
# ─────────────────────────────────────────────

class KubernetesDNS:
    """
    Simulates K8s DNS resolution.
    Services get stable names: <service>.<namespace>.svc.cluster.local
    kube-proxy load-balances across healthy pods (iptables/IPVS).
    """

    def __init__(self):
        self._services: Dict[str, List[str]] = {}   # DNS name → [IP:port]

    def create_service(self, name: str, namespace: str, endpoints: List[str]):
        dns = f"{name}.{namespace}.svc.cluster.local"
        self._services[dns] = endpoints

    def resolve(self, dns_name: str) -> Optional[str]:
        """Returns one endpoint (kube-proxy does the actual LB)."""
        endpoints = self._services.get(dns_name, [])
        if not endpoints:
            return None
        return random.choice(endpoints)   # kube-proxy selects


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_service_discovery():
    print("=" * 65)
    print("SERVICE REGISTRY AND DISCOVERY (MICROSERVICES)")
    print("=" * 65)

    registry = ServiceRegistry()

    # ── 1. Registration ───────────────────────────
    print("\n[1] SERVICE REGISTRATION (SELF-REGISTRATION)")
    print("─" * 55)

    instances = [
        ServiceInstance("order-service", "ord-1", "10.0.0.1", 8080, "v2.1",
                        tags=["http", "v2"], weight=100),
        ServiceInstance("order-service", "ord-2", "10.0.0.2", 8080, "v2.1",
                        tags=["http", "v2"], weight=100),
        ServiceInstance("order-service", "ord-3", "10.0.0.3", 8080, "v2.0",
                        tags=["http", "v1"], weight=50),  # old version, lower weight
        ServiceInstance("user-service",  "usr-1", "10.0.1.1", 8081, "v1.5",
                        tags=["http"]),
        ServiceInstance("user-service",  "usr-2", "10.0.1.2", 8081, "v1.5",
                        tags=["http"]),
    ]

    for inst in instances:
        registry.register(inst)

    print(f"  Registered {len(instances)} instances")
    for svc, data in registry.summary().items():
        print(f"  {svc}: {data['healthy']}/{data['total']} healthy "
              f"→ {data['instances']}")

    # ── 2. Health-check driven routing ───────────
    print("\n\n[2] HEALTH-CHECK DRIVEN ROUTING")
    print("─" * 55)

    registry.update_health("ord-2", HealthStatus.CRITICAL)
    registry.update_health("ord-3", HealthStatus.WARNING)

    all_order  = registry.discover("order-service", healthy_only=False)
    only_healthy = registry.discover("order-service", healthy_only=True)

    print(f"  All order instances:            {len(all_order)}")
    print(f"  Healthy instances only:         {len(only_healthy)}")
    print(f"  Healthy addresses:              {[i.address for i in only_healthy]}")
    print(f"  → ord-2 (CRITICAL) removed from rotation.")

    # ── 3. Client-side discovery + load balancing ─
    print("\n\n[3] CLIENT-SIDE DISCOVERY — LOAD BALANCING")
    print("─" * 55)

    lb = LoadBalancer(registry, strategy="round_robin")
    print(f"  Round-robin over healthy instances:")
    for i in range(4):
        inst = lb.pick("order-service")
        if inst:
            print(f"    Request {i+1}: → {inst.address} (weight={inst.weight})")

    lb2 = LoadBalancer(registry, strategy="weighted")
    print(f"\n  Weighted distribution (50 requests):")
    counts: Dict[str, int] = {}
    for _ in range(50):
        inst = lb2.pick("order-service")
        if inst:
            counts[inst.address] = counts.get(inst.address, 0) + 1
    for addr, cnt in sorted(counts.items()):
        print(f"    {addr}: {cnt} requests")

    # ── 4. Watch/callback ─────────────────────────
    print("\n\n[4] REGISTRY WATCH — REACT TO INSTANCE CHANGES")
    print("─" * 55)
    events_seen = []

    def on_order_change(event: str, inst: ServiceInstance):
        events_seen.append(f"{event} {inst.instance_id}@{inst.address}")

    registry.watch("order-service", on_order_change)
    new_inst = ServiceInstance("order-service", "ord-4", "10.0.0.4", 8080, "v2.2",
                               tags=["http", "v2"])
    registry.register(new_inst)
    registry.deregister("ord-1")
    time.sleep(0.05)  # let callbacks run
    print(f"  Events received by watcher:")
    for ev in events_seen:
        print(f"    {ev}")

    # ── 5. TTL and heartbeat ──────────────────────
    print("\n\n[5] TTL-BASED EVICTION (simulated)")
    print("─" * 55)
    stale = ServiceInstance("payment-service", "pay-stale", "10.2.0.1", 9090, "v1.0")
    stale.last_heartbeat = time.time() - (ServiceRegistry.TTL_SECONDS + 5)
    registry.register(stale)

    before = len(registry.discover("payment-service", healthy_only=False))
    evicted = registry.evict_stale()
    after  = len(registry.discover("payment-service", healthy_only=False))

    print(f"  Instances before eviction: {before}")
    print(f"  Evicted (TTL expired):     {evicted}")
    print(f"  Instances after eviction:  {after}")
    print(f"  → Stale instance (no heartbeat for {ServiceRegistry.TTL_SECONDS}s) removed.")

    # ── 6. Kubernetes DNS ─────────────────────────
    print("\n\n[6] KUBERNETES DNS-STYLE DISCOVERY")
    print("─" * 55)
    k8s_dns = KubernetesDNS()
    k8s_dns.create_service("order-service", "default",
                            ["10.244.0.1:8080", "10.244.0.2:8080", "10.244.0.3:8080"])
    k8s_dns.create_service("user-service", "default",
                            ["10.244.1.1:8081", "10.244.1.2:8081"])

    print(f"  Resolved order-service DNS:")
    for _ in range(3):
        addr = k8s_dns.resolve("order-service.default.svc.cluster.local")
        print(f"    order-service.default.svc.cluster.local → {addr}")
    print(f"  No discovery library needed — just DNS. kube-proxy handles LB.")

    # ── 7. Discovery models comparison ───────────
    print("\n\n[7] DISCOVERY PATTERNS COMPARISON")
    print("─" * 55)
    rows = [
        ("Client-side",    "Eureka, Ribbon",     "Client queries registry, picks instance",    "Library coupling"),
        ("Server-side",    "ALB + ECS, K8s Svc", "LB queries registry; service calls LB",      "Extra hop"),
        ("DNS-based",      "K8s DNS",            "DNS lookup → stable VIP; kube-proxy LBs",    "No fine-grained control"),
        ("Service Mesh",   "Consul Connect, Istio","Sidecar handles discovery transparently",  "Ops complexity"),
    ]
    print(f"  {'Model':<16} {'Example':<22} {'How':<42} {'Con'}")
    print(f"  {'─'*90}")
    for model, example, how, con in rows:
        print(f"  {model:<16} {example:<22} {how:<42} {con}")


if __name__ == "__main__":
    demonstrate_service_discovery()
