"""
AVAILABILITY VS CONSISTENCY
=============================

Problem Statement:
In distributed systems, when a network partition occurs, engineers must
choose between keeping the system available (returning potentially stale
data) or keeping it consistent (refusing to serve until data is certain).
This is the core of the CAP theorem.

CAP Theorem:
  In a distributed system, during a Network Partition, you can have:
    C - Consistency   : Every read receives the most recent write or an error
    A - Availability  : Every request receives a response (may not be latest)
  You CANNOT have both C and A during a partition.

Real Examples:
  CP systems: ZooKeeper, etcd, HBase, PostgreSQL
  AP systems: Cassandra, DynamoDB, CouchDB, DNS
  CA systems: Traditional RDBMS (single node — no partition)

Availability Nines:
  99%     → 87.6  hours downtime/year
  99.9%   →  8.76 hours downtime/year
  99.99%  → 52.6  minutes downtime/year
  99.999% →  5.26 minutes downtime/year
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
import random


class CAPChoice(Enum):
    CP = "CP"   # Consistent + Partition-tolerant
    AP = "AP"   # Available  + Partition-tolerant
    CA = "CA"   # Consistent + Available (no partition tolerance)


class ConsistencyLevel(Enum):
    STRONG     = "strong"       # Linearisable reads
    SEQUENTIAL = "sequential"   # All nodes see same order
    CAUSAL     = "causal"       # Causal dependencies respected
    EVENTUAL   = "eventual"     # Converges over time


@dataclass
class SystemProfile:
    name             : str
    cap_choice       : CAPChoice
    consistency_level: ConsistencyLevel
    examples         : List[str]
    use_case         : str


@dataclass
class DataNode:
    node_id  : str
    region   : str
    value    : Optional[str] = None
    version  : int = 0
    reachable: bool = True


class AvailabilityCalculator:
    """Compute downtime from an uptime percentage."""

    @staticmethod
    def downtime_per_year(uptime_pct: float) -> float:
        """Returns downtime in minutes per year."""
        return (1.0 - uptime_pct / 100.0) * 525_600  # minutes in a year

    @staticmethod
    def nines(uptime_pct: float) -> str:
        nines_count = 0
        val = uptime_pct
        while val >= 9:
            val = (val - 9) * 10
            nines_count += 1
        # simpler approach
        s = str(uptime_pct)
        nines = 0
        for ch in s.replace(".", ""):
            if ch == "9":
                nines += 1
            else:
                break
        return f"{nines} nines"

    @classmethod
    def sla_table(cls):
        print("\n  Availability SLA Table:")
        print(f"  {'SLA %':<12} {'Nines':<12} {'Downtime / Year':<20} {'Downtime / Month'}")
        print(f"  {'─'*60}")
        slas = [99.0, 99.9, 99.95, 99.99, 99.999]
        for sla in slas:
            dt_year  = cls.downtime_per_year(sla)
            dt_month = dt_year / 12
            if dt_year >= 60:
                dt_year_str  = f"{dt_year/60:.1f} hrs"
                dt_month_str = f"{dt_month/60:.1f} hrs"
            else:
                dt_year_str  = f"{dt_year:.1f} min"
                dt_month_str = f"{dt_month:.1f} min"
            print(f"  {sla:<12.3f} {'':12} {dt_year_str:<20} {dt_month_str}")


class SerialAvailabilityCalculator:
    """
    When components are in series, overall availability = product of each component's availability.
    Add redundancy (parallel) to improve: 1 - (1 - A)^n
    """

    @staticmethod
    def serial(availabilities: List[float]) -> float:
        result = 1.0
        for a in availabilities:
            result *= a
        return result

    @staticmethod
    def parallel(availability: float, n: int) -> float:
        """n redundant components in parallel."""
        return 1.0 - (1.0 - availability) ** n


class CAPSimulator:
    """
    Simulates a CP vs AP system's behaviour during a network partition.
    """

    def __init__(self, system_name: str, cap_choice: CAPChoice):
        self.system_name = system_name
        self.cap_choice  = cap_choice
        self.nodes       = [
            DataNode("node-1", "us-east"),
            DataNode("node-2", "eu-west"),
            DataNode("node-3", "ap-south"),
        ]
        # Initialise all nodes with same value
        for n in self.nodes:
            n.value   = "initial_value"
            n.version = 1

    def write(self, new_value: str) -> bool:
        leader = self.nodes[0]
        if not leader.reachable:
            if self.cap_choice == CAPChoice.CP:
                print(f"  [{self.system_name}] WRITE REJECTED — leader unreachable (CP: no stale writes)")
                return False
            else:
                # AP: accept write on any reachable node
                for n in self.nodes:
                    if n.reachable:
                        n.value   = new_value
                        n.version += 1
                        print(f"  [{self.system_name}] WRITE accepted on {n.node_id} (AP: availability first)")
                        return True
                return False
        # Leader is reachable — replicate
        leader.value   = new_value
        leader.version += 1
        replicated = 0
        for n in self.nodes[1:]:
            if n.reachable:
                n.value   = new_value
                n.version  = leader.version
                replicated += 1
        print(f"  [{self.system_name}] WRITE ok → replicated to {replicated}/{len(self.nodes)-1} followers")
        return True

    def read(self, node_id: str) -> Optional[str]:
        node = next((n for n in self.nodes if n.node_id == node_id), None)
        if node is None:
            return None
        if not node.reachable:
            if self.cap_choice == CAPChoice.CP:
                print(f"  [{self.system_name}] READ REJECTED on {node_id} — unreachable (CP: no stale reads)")
                return None
            else:
                print(f"  [{self.system_name}] READ returns last-known value (AP: stale but available)")
                return node.value
        return node.value

    def partition(self, node_id: str):
        for n in self.nodes:
            if n.node_id == node_id:
                n.reachable = False
                print(f"  ⚡ PARTITION: {node_id} isolated from cluster")

    def heal(self, node_id: str):
        for n in self.nodes:
            if n.node_id == node_id:
                n.reachable = True
                print(f"  🔄 HEAL: {node_id} rejoined cluster")


def demonstrate_availability_vs_consistency():
    print("=" * 65)
    print("AVAILABILITY VS CONSISTENCY (CAP THEOREM)")
    print("=" * 65)

    # ── Availability SLA Table ────────────────
    AvailabilityCalculator.sla_table()

    # ── Serial Availability ───────────────────
    print("\n\n  [SERIAL & PARALLEL AVAILABILITY]")
    calc = SerialAvailabilityCalculator()
    # Typical 3-tier: LB (99.99%) → App (99.9%) → DB (99.95%)
    components = [99.99, 99.9, 99.95]
    overall    = calc.serial([a/100 for a in components]) * 100
    print(f"\n  3-tier system (LB→App→DB):")
    for c in components:
        print(f"    Component availability: {c}%")
    print(f"    Overall (serial) : {overall:.4f}%  ← weakest link dominates!")

    # Add DB redundancy
    db_redundant = calc.parallel(99.95/100, 2) * 100
    overall_r    = calc.serial([99.99/100, 99.9/100, db_redundant/100]) * 100
    print(f"\n  With 2 DB nodes in parallel: DB availability → {db_redundant:.5f}%")
    print(f"    Overall improved : {overall_r:.4f}%")

    # ── CAP System Catalogue ──────────────────
    print("\n\n  [REAL SYSTEM CAP CLASSIFICATIONS]")
    systems = [
        SystemProfile("ZooKeeper",  CAPChoice.CP, ConsistencyLevel.STRONG,
                      ["ZooKeeper"], "Leader election, distributed locks"),
        SystemProfile("etcd",       CAPChoice.CP, ConsistencyLevel.STRONG,
                      ["Kubernetes config store"], "Service configuration, k8s state"),
        SystemProfile("Cassandra",  CAPChoice.AP, ConsistencyLevel.EVENTUAL,
                      ["Discord, Netflix"], "Write-heavy, multi-region, sensor data"),
        SystemProfile("DynamoDB",   CAPChoice.AP, ConsistencyLevel.EVENTUAL,
                      ["Amazon shopping cart"], "Key-value at massive scale, low latency"),
        SystemProfile("CouchDB",    CAPChoice.AP, ConsistencyLevel.EVENTUAL,
                      ["Offline-first apps"], "Mobile sync, multi-master"),
        SystemProfile("PostgreSQL", CAPChoice.CA, ConsistencyLevel.STRONG,
                      ["OLTP, financial"], "Transactions, strong ACID (single region)"),
    ]
    print(f"\n  {'System':<15} {'CAP':<5} {'Consistency':<15} {'Use Case'}")
    print(f"  {'─'*70}")
    for s in systems:
        print(f"  {s.name:<15} {s.cap_choice.value:<5} {s.consistency_level.value:<15} {s.use_case}")

    # ── CAP Partition Simulation ──────────────
    print("\n\n  [NETWORK PARTITION SIMULATION]")
    cp_sys = CAPSimulator("ZooKeeper-style (CP)", CAPChoice.CP)
    ap_sys = CAPSimulator("Cassandra-style (AP)", CAPChoice.AP)

    print("\n  --- Normal Operation ---")
    cp_sys.write("user_data_v1")
    ap_sys.write("user_data_v1")

    print("\n  --- Network Partition: node-2 isolated ---")
    cp_sys.partition("node-2")
    ap_sys.partition("node-2")

    print("\n  --- Read from partitioned node ---")
    cp_sys.read("node-2")
    ap_sys.read("node-2")

    print("\n  --- Write during partition ---")
    cp_sys.write("user_data_v2")
    ap_sys.write("user_data_v2")

    print("\n  --- Heal partition ---")
    cp_sys.heal("node-2")
    ap_sys.heal("node-2")

    print("\n  KEY TAKEAWAY:")
    print("  • CP (ZooKeeper): Refuses to serve stale data — prefers error over inconsistency")
    print("  • AP (Cassandra): Always responds — accepts temporary inconsistency")
    print("  • Choose CP for: financial transactions, leader election, config stores")
    print("  • Choose AP for: shopping carts, social feeds, DNS, analytics")


if __name__ == "__main__":
    demonstrate_availability_vs_consistency()
