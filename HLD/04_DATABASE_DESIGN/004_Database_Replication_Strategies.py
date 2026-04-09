"""
DATABASE REPLICATION STRATEGIES
==================================

Problem Statement:
A single database node is a SPOF and can't handle high read volume.
Replication copies data to multiple nodes, enabling high availability,
disaster recovery, read scaling, and geographic distribution.

Replication Topologies:
  Primary-Replica (Master-Slave):
    All writes → Primary
    Reads can → Replica (read-only)
    Failover: promote replica to primary

  Multi-Primary (Active-Active):
    Writes → any node
    Conflict resolution required
    Better write availability
    Examples: MySQL Galera, Cassandra multi-DC

  Chain Replication:
    Write to head → propagated down chain → tail answers reads
    Used in CRAQ, some distributed storage

Replication Methods:
  Synchronous: Primary waits for replica ACK before confirming write.
               → No data loss, but higher write latency
  Asynchronous: Primary confirms write immediately; replica catches up.
               → Lower write latency, replication lag possible
  Semi-synchronous: Wait for at least one replica ACK (MySQL default).

Replication Lag Problems:
  Read-your-writes: User writes then reads stale data from replica
  Fix: route writes and immediate reads to primary; use sync replication
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import time
import random
import threading


class ReplicationMode(Enum):
    SYNCHRONOUS  = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    SEMI_SYNC    = "semi_synchronous"


class NodeRole(Enum):
    PRIMARY = "primary"
    REPLICA = "replica"
    STANDBY = "standby"   # warm standby — receives logs, not serving reads


@dataclass
class WALRecord:
    lsn        : int   # Log Sequence Number — monotonically increasing
    operation  : str   # INSERT/UPDATE/DELETE
    table      : str
    key        : str
    value      : object
    committed  : bool = True
    timestamp  : float = field(default_factory=time.time)


@dataclass
class ReplicationLag:
    replica_id  : str
    lag_records : int    # records behind primary
    lag_bytes   : int    # bytes of WAL not yet applied
    lag_seconds : float  # time lag

    @property
    def is_acceptable(self) -> bool:
        return self.lag_seconds < 5.0


# ─────────────────────────────────────────────
# DATABASE NODE
# ─────────────────────────────────────────────

class DBNode:
    def __init__(self, node_id: str, role: NodeRole):
        self.node_id      = node_id
        self.role         = role
        self._data        : Dict[str, object] = {}
        self._wal         : List[WALRecord] = []
        self._applied_lsn = 0
        self.reads        = 0
        self.writes       = 0
        self.healthy      = True

    @property
    def current_lsn(self) -> int:
        return len(self._wal)

    def apply_wal_record(self, record: WALRecord):
        """Apply a WAL record (called on replica)."""
        self._data[f"{record.table}:{record.key}"] = record.value
        self._applied_lsn = record.lsn

    def local_write(self, table: str, key: str, value: object) -> WALRecord:
        """Primary only: write data and generate WAL."""
        self.writes += 1
        full_key = f"{table}:{key}"
        self._data[full_key] = value
        record = WALRecord(
            lsn=self.current_lsn + 1,
            operation="WRITE",
            table=table, key=key, value=value
        )
        self._wal.append(record)
        return record

    def read(self, table: str, key: str) -> Optional[object]:
        self.reads += 1
        return self._data.get(f"{table}:{key}")

    def lag_behind(self, primary: "DBNode") -> ReplicationLag:
        primary_lsn = primary.current_lsn
        behind      = primary_lsn - self._applied_lsn
        lag_seconds = behind * 0.001   # ~1ms per record simulated
        return ReplicationLag(self.node_id, behind, behind * 200, lag_seconds)


# ─────────────────────────────────────────────
# PRIMARY-REPLICA CLUSTER
# ─────────────────────────────────────────────

class PrimaryReplicaCluster:
    """
    Classic primary-replica (master-slave) setup.
    Writes go to primary; reads can go to replicas.
    """

    def __init__(self, mode: ReplicationMode = ReplicationMode.ASYNCHRONOUS):
        self.mode    = mode
        self.primary : Optional[DBNode] = None
        self.replicas: List[DBNode] = []
        self._rr_idx = 0
        self.write_count = 0
        self.read_count  = 0
        self.replication_failures = 0

    def add_primary(self, node: DBNode):
        node.role   = NodeRole.PRIMARY
        self.primary = node

    def add_replica(self, node: DBNode):
        node.role = NodeRole.REPLICA
        self.replicas.append(node)

    def write(self, table: str, key: str, value: object) -> bool:
        if not self.primary or not self.primary.healthy:
            print("  ❌ Primary unavailable — write failed")
            return False

        record = self.primary.local_write(table, key, value)
        self.write_count += 1

        if self.mode == ReplicationMode.SYNCHRONOUS:
            # Wait for ALL replicas to confirm
            for replica in self.replicas:
                if replica.healthy:
                    replica.apply_wal_record(record)   # synchronous
            return True

        elif self.mode == ReplicationMode.SEMI_SYNC:
            # Wait for at least 1 replica
            acked = 0
            for replica in self.replicas:
                if replica.healthy:
                    replica.apply_wal_record(record)
                    acked += 1
                    if acked >= 1:
                        break
            return True

        else:  # ASYNCHRONOUS
            # Fire-and-forget; replica will catch up
            def async_replicate():
                for replica in self.replicas:
                    if replica.healthy:
                        time.sleep(0.001)   # simulate network delay
                        replica.apply_wal_record(record)
            t = threading.Thread(target=async_replicate, daemon=True)
            t.start()
            return True

    def read(self, table: str, key: str,
             read_from_primary: bool = False) -> Optional[object]:
        self.read_count += 1
        if read_from_primary or not self.replicas:
            return self.primary.read(table, key)
        # Round-robin across replicas
        healthy_replicas = [r for r in self.replicas if r.healthy]
        if not healthy_replicas:
            return self.primary.read(table, key)
        replica = healthy_replicas[self._rr_idx % len(healthy_replicas)]
        self._rr_idx += 1
        return replica.read(table, key)

    def failover(self) -> Optional[DBNode]:
        """Promote best replica to primary."""
        if not self.replicas:
            return None
        # Pick replica with highest applied LSN (least lag)
        best = max(self.replicas, key=lambda r: r._applied_lsn)
        print(f"  FAILOVER: promoting {best.node_id} to primary "
              f"(applied_lsn={best._applied_lsn})")
        best.role   = NodeRole.PRIMARY
        old_primary = self.primary
        self.primary = best
        self.replicas = [r for r in self.replicas if r.node_id != best.node_id]
        return best

    def lag_report(self):
        print(f"\n  Replication Lag ({self.mode.value}):")
        for r in self.replicas:
            lag = r.lag_behind(self.primary)
            ok  = "✅" if lag.is_acceptable else "⚠ "
            print(f"    {ok} {r.node_id}: {lag.lag_records} records behind  "
                  f"~{lag.lag_seconds:.3f}s lag")

    def report(self):
        print(f"\n  Cluster [{self.mode.value}]:")
        print(f"    Primary : {self.primary.node_id}  writes={self.primary.writes}")
        for r in self.replicas:
            print(f"    Replica : {r.node_id}  reads={r.reads}  "
                  f"applied_lsn={r._applied_lsn}/{self.primary.current_lsn}")
        print(f"    Cluster writes: {self.write_count}  reads: {self.read_count}")


# ─────────────────────────────────────────────
# READ YOUR WRITES CONSISTENCY
# ─────────────────────────────────────────────

class ReadYourWritesRouter:
    """
    Ensures a user reads their own writes.
    After a write, route reads to primary for a brief window.
    """

    def __init__(self, cluster: PrimaryReplicaCluster,
                 primary_read_window_s: float = 2.0):
        self.cluster = cluster
        self.window  = primary_read_window_s
        self._user_last_write: Dict[str, float] = {}

    def write(self, user_id: str, table: str, key: str, value: object) -> bool:
        result = self.cluster.write(table, key, value)
        self._user_last_write[user_id] = time.time()
        return result

    def read(self, user_id: str, table: str, key: str) -> Optional[object]:
        last_write = self._user_last_write.get(user_id, 0)
        read_primary = (time.time() - last_write) < self.window
        return self.cluster.read(table, key, read_from_primary=read_primary)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_replication():
    print("=" * 65)
    print("DATABASE REPLICATION STRATEGIES")
    print("=" * 65)

    # ── Async Replication ──────────────────────
    print("\n[1] ASYNCHRONOUS REPLICATION")
    print("─" * 55)
    cluster = PrimaryReplicaCluster(ReplicationMode.ASYNCHRONOUS)
    cluster.add_primary(DBNode("primary-1",  NodeRole.PRIMARY))
    cluster.add_replica(DBNode("replica-us", NodeRole.REPLICA))
    cluster.add_replica(DBNode("replica-eu", NodeRole.REPLICA))

    # Write some data
    for i in range(10):
        cluster.write("users", f"user_{i}", {"name": f"User{i}", "age": 20 + i})

    # Give async replication time to propagate
    time.sleep(0.05)

    # Read from replicas
    for key in ["user_0", "user_5", "user_9"]:
        val = cluster.read("users", key)
        print(f"  READ {key} from replica: {val}")

    cluster.lag_report()
    cluster.report()

    # ── Sync vs Async ─────────────────────────
    print("\n\n[2] SYNCHRONOUS REPLICATION (guaranteed no lag)")
    print("─" * 55)
    sync_cluster = PrimaryReplicaCluster(ReplicationMode.SYNCHRONOUS)
    sync_cluster.add_primary(DBNode("pg-primary", NodeRole.PRIMARY))
    sync_cluster.add_replica(DBNode("pg-standby", NodeRole.REPLICA))

    start = time.perf_counter()
    for i in range(5):
        sync_cluster.write("orders", f"order_{i}", {"total": i * 100})
    sync_latency = (time.perf_counter() - start) * 1000

    # Immediately read from replica — no lag
    for key in ["order_0", "order_4"]:
        val = sync_cluster.read("orders", key)
        print(f"  READ {key} from standby: {val}")
    sync_cluster.lag_report()

    # ── Failover ──────────────────────────────
    print("\n\n[3] AUTOMATIC FAILOVER")
    print("─" * 55)
    fo_cluster = PrimaryReplicaCluster(ReplicationMode.SEMI_SYNC)
    fo_cluster.add_primary(DBNode("primary-A", NodeRole.PRIMARY))
    fo_cluster.add_replica(DBNode("replica-B", NodeRole.REPLICA))
    fo_cluster.add_replica(DBNode("replica-C", NodeRole.REPLICA))

    for i in range(20):
        fo_cluster.write("data", f"k{i}", i * 10)

    time.sleep(0.1)
    print(f"  Primary {fo_cluster.primary.node_id} is healthy")
    print(f"  Simulating primary failure...")
    fo_cluster.primary.healthy = False

    new_primary = fo_cluster.failover()
    fo_cluster.write("data", "post_failover", "new data")
    val = fo_cluster.read("data", "post_failover")
    print(f"  Write and read after failover: {val}")

    # ── Read-Your-Writes ──────────────────────
    print("\n\n[4] READ-YOUR-WRITES CONSISTENCY")
    print("─" * 55)
    base_cluster = PrimaryReplicaCluster(ReplicationMode.ASYNCHRONOUS)
    base_cluster.add_primary(DBNode("primary", NodeRole.PRIMARY))
    base_cluster.add_replica(DBNode("replica", NodeRole.REPLICA))

    router = ReadYourWritesRouter(base_cluster, primary_read_window_s=2.0)

    router.write("alice", "profiles", "alice", {"name": "Alice", "bio": "updated"})
    val = router.read("alice", "profiles", "alice")
    print(f"  Alice writes then immediately reads: {val}")
    print(f"  (routed to primary within 2s window — guaranteed read-your-writes)")

    time.sleep(0.01)
    val2 = router.read("bob", "profiles", "alice")
    print(f"  Bob reads alice's profile: {val2}")
    print(f"  (Bob had no recent write → replica read)")

    # ── Comparison ────────────────────────────
    print("\n\n[5] REPLICATION MODE COMPARISON")
    print("─" * 55)
    rows = [
        ("Write latency",   "Baseline (no wait)",     "+30-50ms (waits 1 replica)", "+50-200ms (waits all)"),
        ("Data loss risk",  "Yes — can lose writes",  "Very low — 1 durable copy", "None — all durable"),
        ("Throughput",      "Highest",                "High",                       "Reduced by slowest node"),
        ("Failure recovery","Some data loss",         "Near-zero loss",             "Zero data loss"),
        ("Use case",        "Analytics replicas",     "Production apps",            "Financial, critical data"),
        ("PostgreSQL",      "Default",                "synchronous_standby_names=1","synchronous_standby_names=*"),
    ]
    print(f"  {'Aspect':<22} {'Async':<28} {'Semi-sync':<30} {'Sync'}")
    print(f"  {'─'*90}")
    for row in rows:
        print(f"  {row[0]:<22} {row[1]:<28} {row[2]:<30} {row[3]}")


if __name__ == "__main__":
    demonstrate_replication()
