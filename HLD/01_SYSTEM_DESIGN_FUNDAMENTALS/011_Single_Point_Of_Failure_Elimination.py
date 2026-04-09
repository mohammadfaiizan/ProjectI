"""
SINGLE POINT OF FAILURE (SPOF) ELIMINATION
============================================

Problem Statement:
A Single Point of Failure (SPOF) is any component whose failure brings down
the entire system. In high-availability architectures, every layer must be
redundant. Identifying and eliminating SPOFs is essential to achieving
99.99%+ uptime.

Architecture Layers with SPOF Risk:
  [DNS] → [Load Balancer] → [App Server] → [Cache] → [Database] → [Message Queue]

Redundancy Types:
  Active-Active  : Multiple instances serve traffic simultaneously
  Active-Passive : Primary serves traffic; secondary is warm standby (failover)
  N+1            : N working instances + 1 spare

Availability Impact (Serial):
  3-tier with one SPOF at 99% → overall = 99% (weakest link)
  Remove all SPOFs (all 99.99%) → overall ≈ 99.97%
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict


class ArchLayer(Enum):
    DNS           = "DNS"
    LOAD_BALANCER = "Load Balancer"
    APP_SERVER    = "App Server"
    CACHE         = "Cache"
    DATABASE      = "Database"
    MESSAGE_QUEUE = "Message Queue"
    OBJECT_STORAGE= "Object Storage"


class RedundancyType(Enum):
    NONE           = "none (SPOF)"
    ACTIVE_ACTIVE  = "active-active"
    ACTIVE_PASSIVE = "active-passive (failover)"
    N_PLUS_ONE     = "N+1"
    GEOGRAPHIC     = "multi-region active-active"


@dataclass
class Component:
    name            : str
    layer           : ArchLayer
    availability_pct: float
    redundancy      : RedundancyType
    instances       : int = 1
    notes           : str = ""

    @property
    def is_spof(self) -> bool:
        return self.redundancy == RedundancyType.NONE or self.instances < 2


class SystemArchitecture:
    """Models a multi-layer system architecture and analyses SPOFs."""

    def __init__(self, name: str):
        self.name       = name
        self.components : List[Component] = []

    def add(self, component: Component):
        self.components.append(component)

    def overall_availability(self) -> float:
        result = 1.0
        for c in self.components:
            result *= (c.availability_pct / 100.0)
        return result * 100.0

    def spofs(self) -> List[Component]:
        return [c for c in self.components if c.is_spof]

    def report(self):
        print(f"\n  Architecture: {self.name}")
        print(f"  {'Layer':<20} {'Component':<25} {'Avail%':<10} {'Redundancy':<25} {'SPOF?'}")
        print(f"  {'─'*90}")
        for c in self.components:
            spof_flag = "⚠  YES" if c.is_spof else "✅ no"
            print(f"  {c.layer.value:<20} {c.name:<25} {c.availability_pct:<10.2f} "
                  f"{c.redundancy.value:<25} {spof_flag}")
        print(f"\n  Overall availability (serial): {self.overall_availability():.4f}%")
        spof_list = self.spofs()
        if spof_list:
            print(f"  ⚠  {len(spof_list)} SPOF(s) found: {', '.join(s.name for s in spof_list)}")
        else:
            print(f"  ✅ No SPOFs found!")


class SPOFEliminator:
    """Suggests redundancy upgrades for each SPOF in an architecture."""

    REMEDIES: Dict[ArchLayer, str] = {
        ArchLayer.DNS:           "Use multiple DNS providers (Route53 + Cloudflare); GeoDNS for failover",
        ArchLayer.LOAD_BALANCER: "Deploy 2+ LBs active-active (AWS ALB handles this internally)",
        ArchLayer.APP_SERVER:    "Minimum 2 instances behind LB; auto-scaling group",
        ArchLayer.CACHE:         "Redis Sentinel (1 master + 2 replicas) or Redis Cluster",
        ArchLayer.DATABASE:      "Primary + Read Replica with automatic failover (RDS Multi-AZ)",
        ArchLayer.MESSAGE_QUEUE: "Kafka with 3+ brokers and replication factor 3",
        ArchLayer.OBJECT_STORAGE:"S3 is inherently HA; for on-prem use erasure coding + geo-replication",
    }

    def remediate(self, arch: SystemArchitecture) -> SystemArchitecture:
        """Returns an improved architecture with SPOFs eliminated."""
        improved = SystemArchitecture(f"{arch.name} (HA)")
        for c in arch.components:
            if c.is_spof:
                remedy = self.REMEDIES.get(c.layer, "Add redundant instance")
                print(f"  🔧 Fixing SPOF [{c.name}]:")
                print(f"     Remedy: {remedy}")
                new_comp = Component(
                    name             = c.name + " (HA)",
                    layer            = c.layer,
                    availability_pct = 99.99,
                    redundancy       = RedundancyType.ACTIVE_ACTIVE
                    if c.layer in (ArchLayer.LOAD_BALANCER, ArchLayer.APP_SERVER, ArchLayer.CACHE)
                    else RedundancyType.ACTIVE_PASSIVE,
                    instances        = 2,
                    notes            = remedy,
                )
                improved.add(new_comp)
            else:
                improved.add(c)
        return improved


def _parallel_availability(single_pct: float, n: int) -> float:
    """Availability of n redundant components (any one surviving = system alive)."""
    return (1.0 - (1.0 - single_pct / 100.0) ** n) * 100.0


def demonstrate_spof_elimination():
    print("=" * 65)
    print("SINGLE POINT OF FAILURE ELIMINATION")
    print("=" * 65)

    # ── Naive Architecture (full of SPOFs) ────
    print("\n[1] NAIVE 3-TIER ARCHITECTURE (before hardening)")
    print("─" * 55)
    naive = SystemArchitecture("Naive E-Commerce")
    naive.add(Component("Route53",        ArchLayer.DNS,           99.99, RedundancyType.ACTIVE_ACTIVE, 1))
    naive.add(Component("HAProxy (x1)",   ArchLayer.LOAD_BALANCER, 99.90, RedundancyType.NONE,          1))  # SPOF
    naive.add(Component("App Server (x1)",ArchLayer.APP_SERVER,    99.50, RedundancyType.NONE,          1))  # SPOF
    naive.add(Component("Redis (x1)",     ArchLayer.CACHE,         99.90, RedundancyType.NONE,          1))  # SPOF
    naive.add(Component("MySQL Primary",  ArchLayer.DATABASE,      99.90, RedundancyType.NONE,          1))  # SPOF
    naive.report()

    # ── Fix SPOFs ─────────────────────────────
    print("\n\n[2] SPOF REMEDIATION")
    print("─" * 55)
    eliminator = SPOFEliminator()
    hardened   = eliminator.remediate(naive)

    print("\n\n[3] HARDENED ARCHITECTURE (after SPOF elimination)")
    print("─" * 55)
    hardened.report()

    # ── Parallel availability math ────────────
    print("\n\n[4] REDUNDANCY MATH (Parallel Availability)")
    print("─" * 55)
    print("  Formula: 1 - (1 - A)^n  where A = single-instance availability\n")
    rows = [
        ("App Server (99.5%)",   99.5,  [2, 3, 5]),
        ("DB Primary (99.9%)",   99.9,  [2, 3]),
        ("Cache Redis (99.9%)",  99.9,  [2, 3]),
        ("LB (99.9%)",           99.9,  [2]),
    ]
    for label, single, ns in rows:
        print(f"  {label}:")
        print(f"    Single instance : {single:.2f}%")
        for n in ns:
            pa = _parallel_availability(single, n)
            print(f"    {n} instances      : {pa:.5f}%")

    # ── Multi-AZ / Multi-Region ───────────────
    print("\n\n[5] MULTI-AZ AND MULTI-REGION PATTERNS")
    print("─" * 55)
    patterns = [
        ("Single AZ",       "Single point of failure at AZ level. Avoid for production."),
        ("Multi-AZ",        "Survives single AZ failure. Standard HA baseline. AWS RDS Multi-AZ example."),
        ("Multi-Region A-P","Primary region serves; secondary on standby. RPO: minutes. RTO: minutes."),
        ("Multi-Region A-A","Both regions serve traffic. No failover needed. Complex: data sync & consistency."),
    ]
    for name, desc in patterns:
        print(f"\n  [{name}]")
        print(f"    {desc}")

    print("\n\n[6] SPOF CHECKLIST FOR INTERVIEWS")
    print("─" * 55)
    checklist = [
        "DNS — use multiple providers or managed DNS with health checks",
        "Load Balancer — managed LBs (AWS ALB) are inherently HA",
        "App servers — min 2 instances; auto-scaling group across AZs",
        "Cache — Redis Sentinel or Cluster (never single Redis)",
        "Database — Multi-AZ primary with automatic failover replica",
        "Message Queue — Kafka 3+ brokers, replication factor=3",
        "Storage — S3 (11 nines durability); on-prem: RAID + replication",
        "Network — dual NICs; redundant switches; BGP multi-homing",
    ]
    for i, item in enumerate(checklist, 1):
        print(f"  {i}. ✅ {item}")


if __name__ == "__main__":
    demonstrate_spof_elimination()
