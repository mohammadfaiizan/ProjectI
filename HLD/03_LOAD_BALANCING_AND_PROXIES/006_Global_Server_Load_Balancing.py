"""
GLOBAL SERVER LOAD BALANCING (GSLB)
======================================

Problem Statement:
A single data center is a geographic SPOF and adds latency for distant users.
Global Server Load Balancing routes users to the nearest (or healthiest)
data center using DNS, Anycast, or GeoDNS — before the request even hits
your infrastructure.

GSLB Strategies:
  1. GeoDNS       → DNS returns different IPs based on client location
  2. Anycast      → same IP announced from multiple PoPs via BGP;
                    network routes to nearest (Cloudflare, Google 8.8.8.8)
  3. Latency-based→ measure RTT to each DC; route to fastest
  4. Failover     → primary DC normally; secondary DC on failure
  5. Weighted     → split traffic (80% DC1, 20% DC2) for migrations

Key Challenges:
  - DNS TTL: lower TTL → faster failover but more DNS queries
  - Split-brain: two DCs may disagree on data (CAP theorem applies)
  - Consistency: user in EU shouldn't see stale data written in US
  - Health propagation: need to detect DC failure fast (<60s)

Popular GSLB Tools:
  AWS Route 53 (latency/health/geo), Cloudflare DNS, Akamai GTM,
  NS1, F5 BIG-IP GTM
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import random
import hashlib


class RoutingPolicy(Enum):
    GEO_PROXIMITY  = "geo_proximity"
    LATENCY_BASED  = "latency_based"
    FAILOVER       = "failover"
    WEIGHTED       = "weighted"
    MULTI_VALUE    = "multi_value"


class DCStatus(Enum):
    HEALTHY    = "healthy"
    DEGRADED   = "degraded"
    UNHEALTHY  = "unhealthy"


@dataclass
class DataCenter:
    dc_id       : str
    region      : str
    ip_address  : str
    latitude    : float
    longitude   : float
    weight      : int   = 100   # for weighted routing
    status      : DCStatus = DCStatus.HEALTHY
    capacity_pct: float = 100.0   # available capacity

    def haversine_km(self, lat2: float, lon2: float) -> float:
        """Approximate distance in km."""
        import math
        R = 6371.0
        d_lat = math.radians(lat2 - self.latitude)
        d_lon = math.radians(lon2 - self.longitude)
        a = (math.sin(d_lat / 2) ** 2 +
             math.cos(math.radians(self.latitude)) *
             math.cos(math.radians(lat2)) *
             math.sin(d_lon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class ClientLocation:
    client_ip : str
    region    : str
    latitude  : float
    longitude : float


@dataclass
class GSLBResponse:
    client_region : str
    selected_dc   : DataCenter
    policy        : RoutingPolicy
    reason        : str
    ttl_s         : int = 60


# ─────────────────────────────────────────────
# GSLB ENGINE
# ─────────────────────────────────────────────

class GSLBEngine:
    def __init__(self, policy: RoutingPolicy = RoutingPolicy.LATENCY_BASED):
        self.policy   = policy
        self.dcs      : List[DataCenter] = []
        self._health_checker_log: List[str] = []
        self.routing_table: Dict[str, str] = {}   # client_region → dc_id

    def add_dc(self, dc: DataCenter):
        self.dcs.append(dc)

    def healthy_dcs(self) -> List[DataCenter]:
        return [dc for dc in self.dcs if dc.status == DCStatus.HEALTHY]

    # ── Geo Proximity ──────────────────────
    def _geo_proximity(self, client: ClientLocation) -> DataCenter:
        healthy = self.healthy_dcs()
        return min(healthy, key=lambda dc: dc.haversine_km(client.latitude, client.longitude))

    # ── Latency Based ──────────────────────
    # Simulated latency table: region pair → ms
    LATENCY_TABLE: Dict[Tuple[str, str], float] = {
        ("us", "us-east-1"): 5,   ("us", "eu-west-1"): 110,  ("us", "ap-south-1"): 200,
        ("eu", "us-east-1"): 110, ("eu", "eu-west-1"): 8,    ("eu", "ap-south-1"): 180,
        ("ap", "us-east-1"): 220, ("ap", "eu-west-1"): 190,  ("ap", "ap-south-1"): 12,
        ("sa", "us-east-1"): 80,  ("sa", "eu-west-1"): 170,  ("sa", "ap-south-1"): 260,
    }

    def _latency_based(self, client: ClientLocation) -> DataCenter:
        healthy = self.healthy_dcs()
        def lat(dc: DataCenter) -> float:
            return self.LATENCY_TABLE.get((client.region, dc.dc_id), 500.0)
        return min(healthy, key=lat)

    # ── Failover ──────────────────────────
    def _failover(self) -> DataCenter:
        """Primary: first DC; fallback: next healthy."""
        for dc in self.dcs:
            if dc.status == DCStatus.HEALTHY:
                return dc
        return self.dcs[0]   # last resort

    # ── Weighted ──────────────────────────
    def _weighted(self, client: ClientLocation) -> DataCenter:
        healthy = self.healthy_dcs()
        total   = sum(dc.weight for dc in healthy)
        h = int(hashlib.md5(client.client_ip.encode()).hexdigest(), 16) % total
        cumulative = 0
        for dc in healthy:
            cumulative += dc.weight
            if h < cumulative:
                return dc
        return healthy[-1]

    def route(self, client: ClientLocation) -> GSLBResponse:
        healthy = self.healthy_dcs()
        if not healthy:
            # Return first DC anyway (emergency)
            dc     = self.dcs[0]
            reason = "ALL DCs unhealthy — emergency routing to first"
            return GSLBResponse(client.region, dc, self.policy, reason, ttl_s=10)

        if self.policy == RoutingPolicy.GEO_PROXIMITY:
            dc     = self._geo_proximity(client)
            dist   = dc.haversine_km(client.latitude, client.longitude)
            reason = f"nearest DC by distance ({dist:.0f} km)"
        elif self.policy == RoutingPolicy.LATENCY_BASED:
            dc     = self._latency_based(client)
            lat    = self.LATENCY_TABLE.get((client.region, dc.dc_id), 999)
            reason = f"lowest latency DC ({lat}ms)"
        elif self.policy == RoutingPolicy.FAILOVER:
            dc     = self._failover()
            reason = f"primary/failover (status={dc.status.value})"
        elif self.policy == RoutingPolicy.WEIGHTED:
            dc     = self._weighted(client)
            reason = f"weighted split (weight={dc.weight})"
        else:
            dc     = random.choice(healthy)
            reason = "random healthy"

        return GSLBResponse(client.region, dc, self.policy, reason)

    def mark_unhealthy(self, dc_id: str):
        for dc in self.dcs:
            if dc.dc_id == dc_id:
                dc.status = DCStatus.UNHEALTHY
                msg = f"  GSLB: {dc_id} marked UNHEALTHY — removed from rotation"
                self._health_checker_log.append(msg)
                print(msg)

    def mark_healthy(self, dc_id: str):
        for dc in self.dcs:
            if dc.dc_id == dc_id:
                dc.status = DCStatus.HEALTHY
                msg = f"  GSLB: {dc_id} restored → back in rotation"
                self._health_checker_log.append(msg)
                print(msg)

    def show_dcs(self):
        print(f"\n  Data Centers:")
        for dc in self.dcs:
            icon = "✅" if dc.status == DCStatus.HEALTHY else "❌"
            print(f"  {icon} {dc.dc_id:<15} {dc.region:<12} {dc.ip_address:<18} "
                  f"weight={dc.weight} capacity={dc.capacity_pct:.0f}%")


# ─────────────────────────────────────────────
# ANYCAST SIMULATOR
# ─────────────────────────────────────────────

class AnycastSimulator:
    """
    Same IP announced from multiple PoPs via BGP.
    Client's ISP routes to topologically nearest PoP.
    Used by: Cloudflare (1.1.1.1), Google (8.8.8.8), all major CDNs.
    """

    def __init__(self, anycast_ip: str = "1.1.1.1"):
        self.anycast_ip = anycast_ip
        self._pops      : List[Dict] = []

    def add_pop(self, city: str, region: str, asn: int, latency_from: Dict[str, float]):
        self._pops.append({"city": city, "region": region, "asn": asn,
                            "latency_from": latency_from})

    def resolve(self, client_region: str) -> Dict:
        """BGP routes client to lowest-latency PoP."""
        best     = min(self._pops,
                       key=lambda p: p["latency_from"].get(client_region, 9999))
        return best

    def show(self):
        print(f"\n  Anycast {self.anycast_ip} PoPs:")
        for pop in self._pops:
            print(f"    {pop['city']:<20} ASN={pop['asn']:<8} "
                  f"latencies={pop['latency_from']}")


# ─────────────────────────────────────────────
# DNS FAILOVER TIMER
# ─────────────────────────────────────────────

class DNSFailoverTimer:
    """Shows how TTL affects failover time."""

    @staticmethod
    def failover_time_s(ttl: int, health_check_interval: int,
                         unhealthy_threshold: int) -> Dict[str, float]:
        detect_time = health_check_interval * unhealthy_threshold
        dns_flush   = ttl   # time for old DNS caches to expire
        total       = detect_time + dns_flush
        return {
            "detect_failure_s": detect_time,
            "dns_propagation_s": dns_flush,
            "total_failover_s": total,
            "total_failover_min": round(total / 60, 1)
        }


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_gslb():
    print("=" * 65)
    print("GLOBAL SERVER LOAD BALANCING (GSLB)")
    print("=" * 65)

    # ── Setup DCs ─────────────────────────────
    dcs = [
        DataCenter("us-east-1",  "us",   "52.1.1.1",   37.9, -77.0,  weight=50),
        DataCenter("eu-west-1",  "eu",   "34.2.2.2",   53.3, -6.3,   weight=30),
        DataCenter("ap-south-1", "asia", "13.3.3.3",   19.1, 72.9,   weight=20),
    ]

    clients = [
        ClientLocation("1.1.1.1", "us",  37.5, -77.0),  # Virginia, US
        ClientLocation("2.2.2.2", "eu",  51.5,  -0.1),  # London, UK
        ClientLocation("3.3.3.3", "ap",  1.35, 103.8),  # Singapore
        ClientLocation("4.4.4.4", "sa", -23.5, -46.6),  # São Paulo
    ]

    # ── Latency-Based Routing ─────────────────
    print("\n[1] LATENCY-BASED ROUTING")
    print("─" * 55)
    gslb_lat = GSLBEngine(RoutingPolicy.LATENCY_BASED)
    for dc in dcs:
        gslb_lat.add_dc(dc)
    gslb_lat.show_dcs()

    print(f"\n  {'Client':<12} {'Region':<8} → {'DC':<15} {'Reason'}")
    print(f"  {'─'*60}")
    for client in clients:
        resp = gslb_lat.route(client)
        print(f"  {client.client_ip:<12} {client.region:<8} → "
              f"{resp.selected_dc.dc_id:<15} {resp.reason}")

    # ── Failover Scenario ─────────────────────
    print("\n\n[2] FAILOVER SCENARIO — DC FAILURE")
    print("─" * 55)
    gslb_fo = GSLBEngine(RoutingPolicy.FAILOVER)
    for dc in dcs:
        gslb_fo.add_dc(dc)

    us_client = ClientLocation("1.1.1.1", "us", 37.5, -77.0)
    resp_before = gslb_fo.route(us_client)
    print(f"  Normal: {us_client.client_ip} → {resp_before.selected_dc.dc_id}")

    gslb_fo.mark_unhealthy("us-east-1")
    resp_after = gslb_fo.route(us_client)
    print(f"  After failure: {us_client.client_ip} → {resp_after.selected_dc.dc_id}")

    gslb_fo.mark_healthy("us-east-1")
    resp_recover = gslb_fo.route(us_client)
    print(f"  After recovery: {us_client.client_ip} → {resp_recover.selected_dc.dc_id}")

    # ── Weighted Routing ──────────────────────
    print("\n\n[3] WEIGHTED ROUTING (traffic migration)")
    print("─" * 55)
    gslb_w = GSLBEngine(RoutingPolicy.WEIGHTED)
    for dc in dcs:
        gslb_w.add_dc(dc)

    print("  Sending 10 requests (weights: us=50, eu=30, ap=20):")
    dist = {}
    for i in range(100):
        cl = ClientLocation(f"10.0.0.{i}", "us", 37.5, -77.0)
        r  = gslb_w.route(cl)
        dist[r.selected_dc.dc_id] = dist.get(r.selected_dc.dc_id, 0) + 1
    for dc_id, cnt in sorted(dist.items()):
        print(f"  {dc_id}: {cnt}% (target: us=50, eu=30, ap=20)")

    # ── Anycast ───────────────────────────────
    print("\n\n[4] ANYCAST — SAME IP, MULTIPLE PoPs")
    print("─" * 55)
    anycast = AnycastSimulator("1.1.1.1")
    anycast.add_pop("Ashburn",   "us",   13335, {"us": 3,   "eu": 80,  "ap": 150})
    anycast.add_pop("Amsterdam", "eu",   13335, {"us": 80,  "eu": 4,   "ap": 120})
    anycast.add_pop("Singapore", "ap",   13335, {"us": 160, "eu": 130, "ap": 5})
    anycast.show()

    print(f"\n  DNS query for 1.1.1.1 from different regions:")
    for region in ["us", "eu", "ap"]:
        pop = anycast.resolve(region)
        print(f"  Client in {region} → routed to {pop['city']} PoP (latency={pop['latency_from'][region]}ms)")

    # ── DNS TTL & Failover Time ───────────────
    print("\n\n[5] DNS TTL vs FAILOVER TIME TRADE-OFF")
    print("─" * 55)
    timer = DNSFailoverTimer()
    configs = [
        (30,  5, 3, "Fast failover (critical)"),
        (60,  10, 3, "Standard API"),
        (300, 10, 3, "Stable service"),
        (3600,30, 3, "Long-lived record"),
    ]
    print(f"  {'TTL':<8} {'Detect':<10} {'Propagation':<15} {'Total':<12} Use case")
    print(f"  {'─'*60}")
    for ttl, interval, threshold, note in configs:
        t = timer.failover_time_s(ttl, interval, threshold)
        print(f"  {ttl:<8}s {t['detect_failure_s']:<10}s {t['dns_propagation_s']:<15}s "
              f"{t['total_failover_min']:.1f}min    {note}")


if __name__ == "__main__":
    demonstrate_gslb()
