"""
DATA REPLICATION STRATEGIES
==============================

Problem Statement:
Store data on more than one node so that failures don't cause data loss or downtime.
Replication enables: fault tolerance, read scaling, geo-locality.
But: replication creates consistency challenges and write amplification.

Replication Topologies:

  1. Single-Leader (Primary-Replica):
     One primary accepts writes. Replicas receive updates asynchronously (or sync).
     Clients read from primary (strong) or replicas (eventual).
     Used by: PostgreSQL, MySQL, MongoDB, Redis Sentinel.
     Pros: simple, no write conflicts.
     Cons: primary is bottleneck; failover needed on primary death.

  2. Multi-Leader (Multi-Primary):
     Multiple nodes accept writes concurrently.
     Leaders sync to each other. Conflicts possible → need resolution.
     Used by: CouchDB, some MySQL configurations, geo-distributed setups.
     Pros: writes can be local (low latency). Better for multi-region.
     Cons: conflict resolution complexity.

  3. Leaderless (Dynamo-style):
     No primary. Writes and reads go to any N nodes.
     W writers + R readers. Quorum: W + R > N → overlap → consistency.
     Used by: Cassandra, Riak, DynamoDB.
     Pros: no leader SPOF. Highly available.
     Cons: read repair, anti-entropy needed. Application handles conflicts.

Synchronous vs Asynchronous:
  Sync:  Primary waits for replica ACK before returning to client.
         Strong durability: data on N nodes before ACK.
         Latency penalty: must wait for slowest replica.
  Async: Primary returns immediately. Replicas lag by milliseconds-seconds.
         Fast writes. Risk: data loss if primary fails before replication.
  Semi-sync: Wait for at least 1 replica (MySQL semi-synchronous). Balance.

Replication Lag:
  The delay between write on primary and availability on replica.
  Sources: network, replica CPU/IO.
  Impact: reads from replica may return stale data.
  Monitoring: replica_lag_seconds; alert if > threshold.

Read Repair:
  On quorum read: if some replicas return stale data, the coordinator
  sends the latest version back to lagging replicas. (Cassandra)

Anti-Entropy:
  Background process comparing data between replicas using Merkle trees.
  Identifies and repairs divergence. See 013_Anti_Entropy_And_Merkle_Trees.py.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import time
import threading
import random
import uuid


# ─────────────────────────────────────────────
# REPLICA NODE
# ─────────────────────────────────────────────

@dataclass
class ReplicaEntry:
    value    : Any
    version  : int
    updated_at: float = field(default_factory=time.time)


class ReplicaNode:
    def __init__(self, node_id: str, replication_lag_ms: float = 0.0):
        self.node_id          = node_id
        self.replication_lag_ms = replication_lag_ms
        self._data            : Dict[str, ReplicaEntry] = {}
        self._version_counter = 0
        self._lock            = threading.Lock()
        self.writes           = 0
        self.reads            = 0
        self.is_primary       = False

    def write(self, key: str, value: Any) -> int:
        with self._lock:
            self._version_counter += 1
            self._data[key] = ReplicaEntry(value=value, version=self._version_counter)
            self.writes += 1
            return self._version_counter

    def read(self, key: str) -> Optional[Tuple[Any, int]]:
        with self._lock:
            self.reads += 1
            entry = self._data.get(key)
            return (entry.value, entry.version) if entry else (None, 0)

    def replicate_from(self, key: str, value: Any, version: int):
        """Apply replication update (only if newer version)."""
        with self._lock:
            existing = self._data.get(key)
            if not existing or version > existing.version:
                self._data[key] = ReplicaEntry(value=value, version=version)

    def lag_behind(self, primary: "ReplicaNode") -> int:
        """How many versions behind compared to primary."""
        primary_max = max((e.version for e in primary._data.values()), default=0)
        my_max      = max((e.version for e in self._data.values()), default=0)
        return max(0, primary_max - my_max)


# ─────────────────────────────────────────────
# SINGLE-LEADER REPLICATION
# ─────────────────────────────────────────────

class SingleLeaderStore:
    """
    Primary-replica setup. Writes go to primary.
    Replication to replicas (sync or async).
    """

    def __init__(self, n_replicas: int = 2, sync_replicas: int = 0,
                 replica_lag_ms: float = 20.0):
        self.primary    = ReplicaNode("primary", replication_lag_ms=0)
        self.primary.is_primary = True
        self.replicas   = [ReplicaNode(f"replica-{i}", replica_lag_ms)
                           for i in range(n_replicas)]
        self.sync_replicas  = sync_replicas   # how many replicas must ack synchronously
        self.async_replicas = n_replicas - sync_replicas
        self._total_writes  = 0

    def write(self, key: str, value: Any) -> Tuple[bool, float]:
        """Returns (success, latency_ms)."""
        t0      = time.time()
        version = self.primary.write(key, value)
        self._total_writes += 1

        # Synchronous replication to sync_replicas
        for replica in self.replicas[:self.sync_replicas]:
            time.sleep(replica.replication_lag_ms / 1000)
            replica.replicate_from(key, value, version)

        latency_ms = (time.time() - t0) * 1000

        # Async replication in background for remaining replicas
        for replica in self.replicas[self.sync_replicas:]:
            t = threading.Thread(
                target=self._async_replicate,
                args=(replica, key, value, version),
                daemon=True,
            )
            t.start()

        return True, latency_ms

    def _async_replicate(self, replica: ReplicaNode, key: str, value: Any, version: int):
        time.sleep(replica.replication_lag_ms / 1000)
        replica.replicate_from(key, value, version)

    def read(self, key: str, from_primary: bool = True) -> Optional[Any]:
        node  = self.primary if from_primary else random.choice(self.replicas)
        result = node.read(key)
        return result[0] if result else None

    def replica_lag(self) -> Dict[str, int]:
        return {r.node_id: r.lag_behind(self.primary) for r in self.replicas}


# ─────────────────────────────────────────────
# LEADERLESS (DYNAMO-STYLE) REPLICATION
# ─────────────────────────────────────────────

class LeaderlessStore:
    """
    Dynamo-style: writes/reads distributed across N nodes.
    Uses quorum (W + R > N) for consistency.
    Read repair: if stale replica detected, update it.
    """

    def __init__(self, n: int = 5, w: int = 3, r: int = 3):
        self.N      = n
        self.W      = w
        self.R      = r
        self.nodes  = [ReplicaNode(f"node-{i}") for i in range(n)]
        self.read_repairs = 0

    def write(self, key: str, value: Any) -> bool:
        """Write to W nodes. Returns True if W acks received."""
        version  = int(time.time() * 1000)   # millisecond timestamp as version
        acks     = 0
        shuffled = random.sample(self.nodes, len(self.nodes))
        for node in shuffled:
            node.replicate_from(key, value, version)
            acks += 1
            if acks >= self.W:
                break
        return acks >= self.W

    def read(self, key: str) -> Optional[Any]:
        """Read from R nodes. Return highest version. Repair stale replicas."""
        responses = []
        shuffled  = random.sample(self.nodes, len(self.nodes))
        for node in shuffled[:self.R]:
            result = node.read(key)
            responses.append((node, result[0], result[1]))

        if not responses:
            return None

        # Find highest version
        best_val, best_ver = None, 0
        for _, val, ver in responses:
            if ver > best_ver:
                best_val, best_ver = val, ver

        # Read repair: update stale nodes
        for node, val, ver in responses:
            if ver < best_ver and best_val is not None:
                node.replicate_from(key, best_val, best_ver)
                self.read_repairs += 1

        return best_val

    def quorum_satisfied(self) -> bool:
        return self.W + self.R > self.N


# ─────────────────────────────────────────────
# MULTI-LEADER CONFLICT RESOLUTION
# ─────────────────────────────────────────────

class MultiLeaderStore:
    """
    Two leaders accept concurrent writes. On sync, resolve via Last-Write-Wins (LWW).
    """

    def __init__(self):
        self.leader1 = ReplicaNode("leader-1")
        self.leader2 = ReplicaNode("leader-2")
        self.conflicts_resolved = 0

    def write_l1(self, key: str, value: Any):
        self.leader1.write(key, value)

    def write_l2(self, key: str, value: Any):
        self.leader2.write(key, value)

    def sync(self):
        """Cross-replicate between leaders. LWW conflict resolution."""
        # L1 → L2
        for key, entry in self.leader1._data.items():
            existing = self.leader2._data.get(key)
            if not existing or entry.version > existing.version:
                self.leader2.replicate_from(key, entry.value, entry.version)
                if existing and entry.version != existing.version:
                    self.conflicts_resolved += 1

        # L2 → L1
        for key, entry in self.leader2._data.items():
            existing = self.leader1._data.get(key)
            if not existing or entry.version > existing.version:
                self.leader1.replicate_from(key, entry.value, entry.version)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_replication():
    print("=" * 65)
    print("DATA REPLICATION STRATEGIES")
    print("=" * 65)

    random.seed(3)

    # ── Single-Leader: Async vs Sync ──────────────
    print("\n[1] SINGLE-LEADER REPLICATION — ASYNC vs SYNC")
    print("─" * 55)

    async_store = SingleLeaderStore(n_replicas=2, sync_replicas=0, replica_lag_ms=30.0)
    sync_store  = SingleLeaderStore(n_replicas=2, sync_replicas=2, replica_lag_ms=30.0)

    # Write and measure latency
    _, async_latency = async_store.write("config", "value-1")
    _, sync_latency  = sync_store.write("config",  "value-1")

    print(f"  Async replication write latency: {async_latency:.2f}ms (returns before replicas)")
    print(f"  Sync  replication write latency: {sync_latency:.2f}ms (waits for replicas)")
    print(f"  Sync is ~{sync_latency/max(async_latency,0.001):.0f}x slower due to replica wait")

    # Check replica lag on async
    time.sleep(0.001)
    lag = async_store.replica_lag()
    print(f"\n  Async replica lag immediately after write: {lag}")
    time.sleep(0.05)   # wait for async replication
    lag_after = async_store.replica_lag()
    print(f"  Async replica lag after 50ms: {lag_after}")

    # ── Leaderless Quorum ─────────────────────────
    print("\n\n[2] LEADERLESS — DYNAMO QUORUM (N=5, W=3, R=3)")
    print("─" * 55)

    lless = LeaderlessStore(n=5, w=3, r=3)
    print(f"  Quorum satisfied (W+R>N): {lless.quorum_satisfied()} "
          f"(W+R={lless.W+lless.R} > N={lless.N})")

    # Write to W=3 nodes
    lless.write("cart", ["item-1", "item-2"])
    lless.write("cart", ["item-1", "item-2", "item-3"])   # update

    # Read from R=3 nodes → highest version wins
    value = lless.read("cart")
    print(f"  Written cart updates: 2  Read result: {value}")
    print(f"  Read repairs performed: {lless.read_repairs}")

    # ── Multi-Leader Conflict ─────────────────────
    print("\n\n[3] MULTI-LEADER — CONFLICT RESOLUTION (LWW)")
    print("─" * 55)

    ml = MultiLeaderStore()

    # Concurrent writes to both leaders
    ml.write_l1("settings", "dark_mode=on")
    time.sleep(0.001)
    ml.write_l2("settings", "dark_mode=off")   # slightly later → higher version

    print(f"  L1 settings: {ml.leader1.read('settings')[0]}")
    print(f"  L2 settings: {ml.leader2.read('settings')[0]}")
    print(f"  (Diverged — concurrent writes)")

    ml.sync()
    print(f"\n  After sync (LWW — latest timestamp wins):")
    print(f"  L1 settings: {ml.leader1.read('settings')[0]}")
    print(f"  L2 settings: {ml.leader2.read('settings')[0]}")
    print(f"  Conflicts resolved: {ml.conflicts_resolved}")

    # ── Primary Failover ──────────────────────────
    print("\n\n[4] PRIMARY FAILOVER SIMULATION")
    print("─" * 55)

    store = SingleLeaderStore(n_replicas=2, sync_replicas=1, replica_lag_ms=5.0)
    for i in range(5):
        store.write(f"key-{i}", f"value-{i}")

    time.sleep(0.05)   # let async replicas catch up

    # "Crash" primary — promote replica-0
    old_primary   = store.primary
    new_primary   = store.replicas[0]
    new_primary.is_primary = True
    store.primary = new_primary
    store.replicas = store.replicas[1:]

    # Check data on new primary
    for i in range(5):
        val = new_primary.read(f"key-{i}")
        print(f"  Failover: new primary has key-{i}={val[0]}")

    # ── Replication Strategy Guide ────────────────
    print("\n\n[5] REPLICATION STRATEGY SELECTION GUIDE")
    print("─" * 55)
    rows = [
        ("Single-leader async", "High write throughput", "Data loss on primary crash"),
        ("Single-leader sync",  "Strong durability",     "Latency penalty; slow replica = slow all"),
        ("Multi-leader",        "Multi-region writes",   "Conflict resolution complexity"),
        ("Leaderless",          "No SPOF, high avail",  "Read repair, eventual consistency"),
    ]
    print(f"  {'Strategy':<24} {'Benefit':<28} {'Risk'}")
    print(f"  {'─'*72}")
    for strategy, benefit, risk in rows:
        print(f"  {strategy:<24} {benefit:<28} {risk}")

    print("\n\n[6] QUORUM TRADE-OFFS (N=5)")
    print("─" * 55)
    configs = [
        (1, 1, "Fastest, weakest"),
        (1, 5, "Read all — strong read, fast write"),
        (5, 1, "Write all — strong write, fast read"),
        (3, 3, "Balanced: W+R=6 > N=5"),
        (2, 4, "Read-heavy: strong reads"),
    ]
    print(f"  {'W':<5} {'R':<5} {'W+R':<6} {'Strong?':<10} {'Notes'}")
    print(f"  {'─'*50}")
    for w, r, notes in configs:
        strong = "Yes" if w + r > 5 else "No"
        print(f"  {w:<5} {r:<5} {w+r:<6} {strong:<10} {notes}")


if __name__ == "__main__":
    demonstrate_replication()
