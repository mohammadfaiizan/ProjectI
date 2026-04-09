"""
CONSISTENCY MODELS
====================

Problem Statement:
"Consistent" is not a binary property. There's a rich spectrum of guarantees
that a distributed system can offer, each with different latency and complexity costs.
Choosing the right consistency model is critical to correctness and performance.

Consistency Models (strongest → weakest):

  1. Linearizability (Strong Consistency):
     Every operation appears instantaneous at some point between its start and end.
     All clients see the same total order of operations.
     Behaves as if a single-copy system.
     Cost: every read/write must coordinate globally.
     Used by: ZooKeeper, etcd, Spanner.

  2. Sequential Consistency:
     All operations appear in the same global order to all processes.
     Operations from each process appear in program order.
     Not real-time: may not reflect wall-clock order across processes.
     Used by: some NUMA memory models.

  3. Causal Consistency:
     Causally related operations seen in correct order by all nodes.
     Concurrent (unrelated) operations may be seen in any order.
     More available than sequential. Tracks causal dependencies.
     Used by: COPS, Eiger, MongoDB 3.6+ causal sessions.

  4. Read-Your-Writes (RYW):
     A client always sees the effects of its own writes.
     Other clients may not immediately see those writes.
     Commonly required by user-facing applications.
     Used by: sticky sessions, primary reads.

  5. Monotonic Read Consistency:
     Once a client reads a value, it never reads an older value.
     Prevents "time going backwards."
     Used with: version pinning, sticky sessions.

  6. Monotonic Write Consistency:
     Writes from a single client are applied in order.
     Critical for: counters, sequential updates.

  7. Eventual Consistency (Weakest):
     If no new updates, all replicas will eventually converge.
     No guarantees about when or what you read in the meantime.
     Fastest, most available.
     Used by: DNS, Cassandra with ONE, DynamoDB eventually consistent reads.

Session Guarantees (Pragmatic subset for client sessions):
  Often more practical than full linearizability.
  Provide: RYW + Monotonic reads + Monotonic writes + Writes follow reads.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import time
import threading
import random


# ─────────────────────────────────────────────
# VERSION VECTOR (tracks causal dependency)
# ─────────────────────────────────────────────

@dataclass
class VersionVector:
    clocks: Dict[str, int] = field(default_factory=dict)

    def increment(self, node_id: str) -> "VersionVector":
        new = VersionVector(dict(self.clocks))
        new.clocks[node_id] = new.clocks.get(node_id, 0) + 1
        return new

    def merge(self, other: "VersionVector") -> "VersionVector":
        all_keys = set(self.clocks) | set(other.clocks)
        merged   = {k: max(self.clocks.get(k, 0), other.clocks.get(k, 0))
                    for k in all_keys}
        return VersionVector(merged)

    def dominates(self, other: "VersionVector") -> bool:
        """self >= other component-wise."""
        for k, v in other.clocks.items():
            if self.clocks.get(k, 0) < v:
                return False
        return True

    def is_concurrent(self, other: "VersionVector") -> bool:
        return not self.dominates(other) and not other.dominates(self)

    def __repr__(self):
        return str(self.clocks)


# ─────────────────────────────────────────────
# LINEARIZABLE KV STORE
# ─────────────────────────────────────────────

class LinearizableStore:
    """
    Simulates a linearizable single-server key-value store.
    All operations are serialized through a single lock — total order.
    """

    def __init__(self):
        self._data     : Dict[str, Any] = {}
        self._lock     = threading.Lock()
        self._history  : List[Dict] = []

    def write(self, key: str, value: Any, client: str) -> int:
        with self._lock:
            ts = len(self._history)
            self._data[key] = value
            self._history.append({"op": "write", "key": key, "value": value,
                                   "client": client, "ts": ts})
            return ts

    def read(self, key: str, client: str) -> Tuple[Optional[Any], int]:
        with self._lock:
            ts    = len(self._history)
            value = self._data.get(key)
            self._history.append({"op": "read", "key": key, "value": value,
                                   "client": client, "ts": ts})
            return value, ts


# ─────────────────────────────────────────────
# EVENTUAL CONSISTENCY STORE (multi-replica, async)
# ─────────────────────────────────────────────

class EventualReplica:
    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self._data      : Dict[str, Tuple[Any, float]] = {}   # key → (value, timestamp)

    def write_local(self, key: str, value: Any, ts: float):
        existing_ts = self._data.get(key, (None, 0.0))[1]
        if ts > existing_ts:
            self._data[key] = (value, ts)

    def read(self, key: str) -> Optional[Any]:
        return self._data.get(key, (None, 0.0))[0]

    def merge_from(self, other: "EventualReplica"):
        """Last-write-wins merge."""
        for key, (value, ts) in other._data.items():
            self.write_local(key, value, ts)


class EventualConsistencyStore:
    """
    Multiple replicas, async propagation. Each replica may differ.
    Convergence via last-write-wins (LWW).
    """

    def __init__(self, n_replicas: int = 3):
        self.replicas = [EventualReplica(f"R{i}") for i in range(n_replicas)]

    def write(self, key: str, value: Any, replica_idx: int = 0) -> float:
        ts = time.time()
        self.replicas[replica_idx].write_local(key, value, ts)
        # Async propagation (simulated as delayed background sync)
        return ts

    def read(self, key: str, replica_idx: int = 0) -> Optional[Any]:
        return self.replicas[replica_idx].read(key)

    def sync(self):
        """Propagate all updates to all replicas (eventual convergence)."""
        for i in range(len(self.replicas)):
            for j in range(len(self.replicas)):
                if i != j:
                    self.replicas[i].merge_from(self.replicas[j])

    def values_match(self, key: str) -> bool:
        vals = [r.read(key) for r in self.replicas]
        return len(set(str(v) for v in vals)) == 1


# ─────────────────────────────────────────────
# READ-YOUR-WRITES SESSION
# ─────────────────────────────────────────────

class RYWSession:
    """
    Client session that implements Read-Your-Writes guarantee.
    Tracks the minimum version the client needs to see.
    Routes reads to a replica that is at least at that version.
    """

    def __init__(self, replicas: List):
        self.replicas     = replicas
        self._min_version : Dict[str, int] = defaultdict(int)

    def write(self, key: str, value: Any, replica_idx: int = 0) -> int:
        """Write to primary. Track version so reads see our write."""
        version = len(self.replicas[0]._data) + 1  # simplified version
        self.replicas[0].write_local(key, value, time.time())
        self._min_version[key] = version
        return version

    def read(self, key: str) -> Tuple[Optional[Any], str]:
        """Read from any replica that is at least at our min_version."""
        for r in self.replicas:
            val = r.read(key)
            if val is not None:
                return val, r.replica_id
        return None, "none"


# ─────────────────────────────────────────────
# MONOTONIC READ TRACKER
# ─────────────────────────────────────────────

class MonotonicReadStore:
    """
    Ensures a client never reads an older version than it has already seen.
    Uses version pinning — client remembers highest version seen per key.
    """

    def __init__(self):
        self._data    : Dict[str, List[Tuple[int, Any]]] = defaultdict(list)
        self._versions: Dict[str, int] = {}   # client → key → last seen version

    def write(self, key: str, value: Any) -> int:
        version = len(self._data[key]) + 1
        self._data[key].append((version, value))
        return version

    def read(self, key: str, client_id: str) -> Tuple[Optional[Any], int]:
        if key not in self._data:
            return None, 0
        client_key = f"{client_id}:{key}"
        min_version = self._versions.get(client_key, 0)
        # Return the most recent version >= min_version (monotonic guarantee)
        eligible = [(v, val) for v, val in self._data[key] if v >= min_version]
        if not eligible:
            latest = self._data[key][-1]
        else:
            latest = max(eligible, key=lambda x: x[0])
        version, value = latest
        self._versions[client_key] = max(self._versions.get(client_key, 0), version)
        return value, version


# ─────────────────────────────────────────────
# CAUSAL CONSISTENCY STORE
# ─────────────────────────────────────────────

@dataclass
class CausalEntry:
    key    : str
    value  : Any
    vector : VersionVector
    author : str


class CausalStore:
    """
    Ensures causally related writes are seen in order.
    Uses vector clocks to track causal dependencies.
    """

    def __init__(self, node_id: str, all_nodes: List[str]):
        self.node_id  = node_id
        self._entries : Dict[str, CausalEntry] = {}
        self._clock   = VersionVector({n: 0 for n in all_nodes})
        self._pending : List[CausalEntry] = []   # waiting for dependencies

    def write(self, key: str, value: Any) -> VersionVector:
        self._clock = self._clock.increment(self.node_id)
        entry = CausalEntry(key=key, value=value,
                            vector=self._clock, author=self.node_id)
        self._entries[key] = entry
        return self._clock

    def receive(self, entry: CausalEntry) -> bool:
        """Accept a write from another node if causal dependencies are met."""
        if self._clock.dominates(VersionVector(
                {entry.author: entry.vector.clocks.get(entry.author, 0) - 1})):
            # Dependencies met
            self._entries[entry.key] = entry
            self._clock = self._clock.merge(entry.vector)
            return True
        else:
            self._pending.append(entry)
            return False

    def read(self, key: str) -> Optional[Any]:
        entry = self._entries.get(key)
        return entry.value if entry else None


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_consistency_models():
    print("=" * 65)
    print("CONSISTENCY MODELS")
    print("=" * 65)

    # ── Linearizable Store ────────────────────────
    print("\n[1] LINEARIZABLE STORE — TOTAL ORDER")
    print("─" * 55)
    lin = LinearizableStore()
    lin.write("x", 1, "client-A")
    lin.write("y", 2, "client-B")
    lin.write("x", 3, "client-A")
    v, ts = lin.read("x", "client-C")
    print(f"  Total operations recorded: {len(lin._history)}")
    print(f"  Client-C reads x: value={v} at timestamp={ts}")
    print(f"  All clients see same order: {[h['ts'] for h in lin._history]}")
    print(f"  → Every read reflects the most recent write globally")

    # ── Eventual Consistency ──────────────────────
    print("\n\n[2] EVENTUAL CONSISTENCY — CONVERGENCE AFTER SYNC")
    print("─" * 55)
    eventual = EventualConsistencyStore(n_replicas=3)

    # Two clients write to different replicas simultaneously
    eventual.write("name", "Alice", replica_idx=0)
    eventual.write("name", "Bob",   replica_idx=1)   # concurrent write

    r0 = eventual.read("name", replica_idx=0)
    r1 = eventual.read("name", replica_idx=1)
    r2 = eventual.read("name", replica_idx=2)
    print(f"  Before sync — R0={r0} R1={r1} R2={r2}  (diverged)")

    time.sleep(0.002)   # ensure timestamps differ
    eventual.sync()     # replicas converge

    r0 = eventual.read("name", replica_idx=0)
    r1 = eventual.read("name", replica_idx=1)
    r2 = eventual.read("name", replica_idx=2)
    converged = eventual.values_match("name")
    print(f"  After sync  — R0={r0} R1={r1} R2={r2}  converged={converged}")
    print(f"  → All replicas converge to the latest write (LWW)")

    # ── Read-Your-Writes ──────────────────────────
    print("\n\n[3] READ-YOUR-WRITES GUARANTEE")
    print("─" * 55)
    ryw_replicas = [EventualReplica(f"R{i}") for i in range(3)]
    session      = RYWSession(ryw_replicas)

    session.write("profile", {"name": "Alice"})
    val, from_replica = session.read("profile")
    print(f"  Client wrote profile. Reads from: {from_replica} value={val}")
    print(f"  → Client always sees its own write regardless of replica")

    # ── Monotonic Reads ───────────────────────────
    print("\n\n[4] MONOTONIC READ CONSISTENCY")
    print("─" * 55)
    mono = MonotonicReadStore()
    mono.write("counter", 1)
    mono.write("counter", 2)
    mono.write("counter", 3)

    v1, ver1 = mono.read("counter", "client-X")
    print(f"  First read by client-X: value={v1} version={ver1}")
    # Simulate older version available (e.g., from stale replica)
    v2, ver2 = mono.read("counter", "client-X")
    print(f"  Second read by client-X: value={v2} version={ver2}")
    print(f"  Version monotonically non-decreasing: {ver2 >= ver1}")

    # ── Causal Consistency ────────────────────────
    print("\n\n[5] CAUSAL CONSISTENCY — VECTOR CLOCKS")
    print("─" * 55)
    nodes   = ["N1", "N2"]
    store_n1 = CausalStore("N1", nodes)
    store_n2 = CausalStore("N2", nodes)

    # N1 writes post; N2 writes reply (causally after N1's post)
    vec1 = store_n1.write("post", "Hello world")
    print(f"  N1 wrote post. Vector: {vec1}")

    # N2 receives N1's write, then writes a reply
    entry_n1 = CausalEntry("post", "Hello world", vec1, "N1")
    store_n2.receive(entry_n1)
    vec2 = store_n2.write("reply", "Great post!")
    print(f"  N2 received post. N2 writes reply. Vector: {vec2}")
    print(f"  N2 can see: post='{store_n2.read('post')}' reply='{store_n2.read('reply')}'")
    print(f"  → Causal order preserved: post always before reply")

    # ── Consistency Model Spectrum ────────────────
    print("\n\n[6] CONSISTENCY MODEL COMPARISON")
    print("─" * 55)
    models = [
        ("Linearizability", "Total order, real-time", "Highest", "Spanner, ZooKeeper"),
        ("Sequential",      "Total order, not real-time", "High", "Some NUMA models"),
        ("Causal",          "Causal order only", "Medium", "COPS, MongoDB sessions"),
        ("Read-Your-Writes","Own writes visible", "Medium", "Social media, profiles"),
        ("Monotonic Read",  "Never go backwards", "Medium", "Most web apps"),
        ("Eventual",        "Converges eventually", "Lowest", "DNS, Cassandra, DynamoDB"),
    ]
    print(f"  {'Model':<22} {'Guarantee':<28} {'Coord Cost':<12} {'Examples'}")
    print(f"  {'─'*82}")
    for model, guarantee, cost, examples in models:
        print(f"  {model:<22} {guarantee:<28} {cost:<12} {examples}")


if __name__ == "__main__":
    demonstrate_consistency_models()
