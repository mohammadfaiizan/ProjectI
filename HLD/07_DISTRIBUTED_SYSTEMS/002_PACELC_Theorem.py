"""
PACELC THEOREM
===============

Problem Statement:
CAP theorem only models the partitioned case. But in normal operation (no partition),
there's still a trade-off: Latency vs Consistency.
PACELC (Abadi, 2010) extends CAP with this insight.

PACELC Full Form:
  If Partition (P):  choose between Availability (A) and Consistency (C)
  ELse (no partition): choose between Latency (L) and Consistency (C)

Why PACELC Matters:
  Network partitions are rare. Most of the time, systems run normally.
  The latency vs consistency trade-off is present EVERY request, not just during faults.
  Example: should a write wait for all replicas before returning? → More consistent but slower.

Latency-Consistency Trade-off (normal operation):
  High Consistency (e.g., sync replication):
    Write must propagate to N replicas before ACK → higher write latency.
    Read always up-to-date.
  Low Latency (e.g., async replication):
    Write returns after primary writes → fast, but replicas may lag.
    Read from replica may be stale.

PACELC Classifications:
  PA/EL: AP during partition, low latency during normal ops.
    → Cassandra, DynamoDB. Fast writes, eventual consistency.
  PC/EC: CP during partition, consistent during normal ops.
    → HBase, BigTable, Spanner. Slower but always consistent.
  PA/EC: AP during partition, consistent during normal ops (unusual).
    → PNUTS (Yahoo).
  PC/EL: CP during partition, low latency normally (unusual).
    → MongoDB (primary only reads).

The Fundamental Tension:
  To achieve strong consistency under NORMAL operation:
    → Synchronous replication → higher write latency (wait for all replicas).
  To achieve low latency under normal operation:
    → Asynchronous replication → reads may be stale on replicas.

Tunable Consistency (Cassandra/DynamoDB):
  Request-level trade-off via consistency level parameter:
  ONE: fastest, lowest consistency.
  QUORUM: R+W > N, balanced.
  ALL: slowest, strongest consistency.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import threading
import random


# ─────────────────────────────────────────────
# REPLICATION MODE
# ─────────────────────────────────────────────

class ReplicationMode:
    SYNC  = "sync"    # wait for all replicas before ACK → strong consistency, high latency
    ASYNC = "async"   # ACK after primary write → low latency, possible staleness


# ─────────────────────────────────────────────
# SIMULATED REPLICA
# ─────────────────────────────────────────────

@dataclass
class Replica:
    replica_id : str
    is_primary : bool = False
    _data      : Dict[str, Any] = field(default_factory=dict)
    _version   : Dict[str, int] = field(default_factory=dict)
    lag_ms     : float = 0.0   # replication lag for async replicas

    def write_local(self, key: str, value: Any, version: int):
        self._data[key]    = value
        self._version[key] = version

    def read(self, key: str) -> Tuple[Optional[Any], int]:
        return self._data.get(key), self._version.get(key, 0)


# ─────────────────────────────────────────────
# PACELC DATABASE SIMULATOR
# ─────────────────────────────────────────────

class PACELCDatabase:
    """
    Simulates a replicated database with tunable consistency.
    Demonstrates latency vs consistency trade-off in normal operation.
    """

    def __init__(self, n_replicas: int = 3, replication_mode: str = ReplicationMode.ASYNC,
                 replica_lag_ms: float = 50.0):
        self.replication_mode = replication_mode
        self.replica_lag_ms   = replica_lag_ms
        self._global_version  = 0
        self._lock            = threading.Lock()

        self.replicas = [Replica(
            replica_id = f"replica-{i}",
            is_primary = (i == 0),
            lag_ms     = 0 if i == 0 else replica_lag_ms,
        ) for i in range(n_replicas)]

        self.primary = self.replicas[0]
        self.write_latencies  : List[float] = []
        self.stale_reads      : int = 0
        self.consistent_reads : int = 0

    def write(self, key: str, value: Any) -> float:
        """Returns write latency in ms."""
        with self._lock:
            self._global_version += 1
            version = self._global_version

        t0 = time.time()
        self.primary.write_local(key, value, version)

        if self.replication_mode == ReplicationMode.SYNC:
            # Wait for ALL replicas to acknowledge
            for replica in self.replicas[1:]:
                time.sleep(replica.lag_ms / 1000)
                replica.write_local(key, value, version)
        else:
            # Async: replicate in background, return immediately
            def replicate_async(r: Replica, k, v, ver):
                time.sleep(r.lag_ms / 1000)
                r.write_local(k, v, ver)

            for replica in self.replicas[1:]:
                t = threading.Thread(target=replicate_async,
                                     args=(replica, key, value, version), daemon=True)
                t.start()

        latency_ms = (time.time() - t0) * 1000
        self.write_latencies.append(latency_ms)
        return latency_ms

    def read(self, key: str, from_replica: int = 0) -> Tuple[Optional[Any], bool, int]:
        """Returns (value, is_stale, replica_version)."""
        replica   = self.replicas[from_replica]
        value, replica_version = replica.read(key)

        with self._lock:
            current_version = self._global_version

        stale = replica_version < current_version and value is not None
        if stale:
            self.stale_reads += 1
        else:
            self.consistent_reads += 1
        return value, stale, replica_version

    def avg_write_latency_ms(self) -> float:
        return sum(self.write_latencies) / len(self.write_latencies) if self.write_latencies else 0


# ─────────────────────────────────────────────
# TUNABLE CONSISTENCY (Cassandra-style)
# ─────────────────────────────────────────────

class ConsistencyLevel(str):
    ONE    = "ONE"
    QUORUM = "QUORUM"
    ALL    = "ALL"


class TunableConsistencyDB:
    """
    Simulates Cassandra-style tunable consistency.
    W + R > N → "quorum" consistency.
    """

    def __init__(self, n: int = 3):
        self.N       = n
        self._nodes  = [{} for _ in range(n)]
        self._versions : Dict[str, int] = {}
        self._global_version = 0

    def _required(self, level: str) -> int:
        if level == ConsistencyLevel.ONE:    return 1
        if level == ConsistencyLevel.QUORUM: return self.N // 2 + 1
        if level == ConsistencyLevel.ALL:    return self.N
        return 1

    def write(self, key: str, value: Any, level: str = ConsistencyLevel.ONE) -> Dict:
        self._global_version += 1
        ver = self._global_version
        required = self._required(level)
        acked = 0
        for node in self._nodes:
            node[key] = (value, ver)
            acked += 1
            if acked >= required:
                break   # return after required acks (simulate partial replication)
        # remaining nodes get async replication (may lag)
        return {"acked": acked, "required": required, "version": ver}

    def read(self, key: str, level: str = ConsistencyLevel.ONE) -> Dict:
        required = self._required(level)
        results  = []
        for node in self._nodes[:required]:
            if key in node:
                results.append(node[key])

        if not results:
            return {"value": None, "consistent": True}

        # Return highest version seen
        best = max(results, key=lambda x: x[1])
        consistent = best[1] == self._global_version
        return {"value": best[0], "version": best[1],
                "consistent": consistent, "acked_from": required}


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_pacelc():
    print("=" * 65)
    print("PACELC THEOREM")
    print("=" * 65)

    # ── Sync vs Async Replication Latency ─────────
    print("\n[1] NORMAL OPERATION: LATENCY vs CONSISTENCY")
    print("─" * 55)

    sync_db  = PACELCDatabase(n_replicas=3, replication_mode=ReplicationMode.SYNC,
                               replica_lag_ms=20.0)
    async_db = PACELCDatabase(n_replicas=3, replication_mode=ReplicationMode.ASYNC,
                               replica_lag_ms=20.0)

    for i in range(5):
        sync_db.write(f"key-{i}", f"value-{i}")
        async_db.write(f"key-{i}", f"value-{i}")

    print(f"  Sync replication (PC/EC):")
    print(f"    Avg write latency: {sync_db.avg_write_latency_ms():.1f}ms "
          f"(waits for all replicas)")
    print(f"  Async replication (PA/EL):")
    print(f"    Avg write latency: {async_db.avg_write_latency_ms():.3f}ms "
          f"(returns after primary write)")

    # ── Stale Read from Async Replica ─────────────
    print("\n\n[2] ASYNC REPLICATION — STALE REPLICA READS")
    print("─" * 55)

    fast_db = PACELCDatabase(n_replicas=3, replication_mode=ReplicationMode.ASYNC,
                              replica_lag_ms=50.0)
    fast_db.write("balance", 1000)

    # Immediately read from replica (before async replication completes)
    val_primary, stale_primary, _ = fast_db.read("balance", from_replica=0)
    val_replica,  stale_replica,  _ = fast_db.read("balance", from_replica=1)

    print(f"  Primary read:  value={val_primary} stale={stale_primary}")
    print(f"  Replica read (before replication): value={val_replica} stale={stale_replica}")

    # Wait for replication to complete
    time.sleep(0.1)   # 100ms > 50ms lag
    fast_db.write("balance", 1000, )   # dummy to reset replica version tracking
    val_after, stale_after, _ = fast_db.read("balance", from_replica=1)
    print(f"  Replica read (after lag resolved): stale={stale_after}")

    # ── Tunable Consistency ───────────────────────
    print("\n\n[3] TUNABLE CONSISTENCY (CASSANDRA-STYLE)")
    print("─" * 55)

    db = TunableConsistencyDB(n=3)
    print(f"  N=3 replicas")

    for write_level in [ConsistencyLevel.ONE, ConsistencyLevel.QUORUM, ConsistencyLevel.ALL]:
        result = db.write("order_count", 42, level=write_level)
        read_result = db.read("order_count", level=write_level)
        print(f"  Write {write_level:<8}: acked={result['acked']}/{db.N}  "
              f"Read {write_level:<8}: value={read_result['value']} "
              f"consistent={read_result['consistent']}")

    print(f"\n  Quorum math (N=3): W=2, R=2 → W+R=4 > N=3 → guaranteed overlap → consistent")
    print(f"  ONE+ONE: W=1, R=1 → W+R=2 < N=3 → may read from node that missed write")

    # ── PACELC Classification Table ───────────────
    print("\n\n[4] PACELC SYSTEM CLASSIFICATIONS")
    print("─" * 55)
    systems = [
        ("Cassandra",       "PA/EL", "AP during partition; async replication by default"),
        ("DynamoDB",        "PA/EL", "AP + eventually consistent reads by default"),
        ("Riak",            "PA/EL", "AP + eventual consistency, CRDT support"),
        ("HBase",           "PC/EC", "CP + synchronous replication"),
        ("BigTable",        "PC/EC", "CP + strong consistency"),
        ("Spanner",         "PC/EC", "CP + TrueTime for global consistency"),
        ("PNUTS (Yahoo)",   "PA/EC", "AP during partition, consistent normally"),
        ("MongoDB",         "PC/EL", "CP + reads from primary (low latency)"),
        ("PostgreSQL",      "PC/EC", "Single node = no partition; sync replication"),
    ]
    print(f"  {'System':<20} {'PACELC':<8} {'Notes'}")
    print(f"  {'─'*72}")
    for system, pacelc, notes in systems:
        print(f"  {system:<20} {pacelc:<8} {notes}")

    # ── Latency vs Consistency Trade-off ──────────
    print("\n\n[5] LATENCY vs CONSISTENCY SPECTRUM")
    print("─" * 55)
    levels = [
        ("Linearizable",       "All replicas ack before return",  "~100-500ms write", "Banks, locks"),
        ("Sequential",         "Primary acks, sync to secondaries","~10-50ms write",  "Financial apps"),
        ("Eventual (quorum)",  "W+R > N replicas",                "~5-20ms write",   "Most web apps"),
        ("Eventual (async)",   "Primary acks immediately",        "<1ms write",       "Metrics, logs"),
        ("Read-your-writes",   "Guarantee own writes are visible", "variable",        "Social media"),
    ]
    print(f"  {'Consistency Level':<22} {'Mechanism':<37} {'Latency':<18} {'Use Case'}")
    print(f"  {'─'*90}")
    for level, mech, latency, use in levels:
        print(f"  {level:<22} {mech:<37} {latency:<18} {use}")


if __name__ == "__main__":
    demonstrate_pacelc()
