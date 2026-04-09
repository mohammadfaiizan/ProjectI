"""
CACHE CONSISTENCY IN DISTRIBUTED SYSTEMS
==========================================

Problem Statement:
In a distributed system, caches across multiple nodes can hold different
versions of the same data. User A writes a new profile picture; User B
sees the old one from cache. How do you ensure all caches reflect the
current state within an acceptable staleness window?

Consistency Models for Caches:

  Strong Consistency:
    Every read sees the latest write. Cache is invalidated synchronously
    before the write returns. High consistency, lower performance.
    Hard to achieve in distributed settings (requires global lock or consensus).

  Eventual Consistency:
    Writes propagate to all caches asynchronously.
    All caches converge to the same value given no new writes.
    Most distributed caches are eventually consistent.

  Read-Your-Writes (RYW):
    A user always sees their own writes. Others may see stale data.
    Implementation: after write, route reads to primary for a window (e.g., 2s),
    or invalidate this user's cache entry immediately.

  Monotonic Reads:
    Once a user sees version V, they never see version < V.
    Prevents going back in time when load balancer routes to different caches.

  Bounded Staleness:
    Stale data served, but guaranteed no older than Δ seconds.
    Implementation: TTL guarantees upper bound on staleness.

Cache Invalidation Propagation Patterns:
  1. Synchronous invalidation:
     Write → invalidate all cache nodes → commit write.
     Consistent but slower (blocks on all nodes).

  2. Asynchronous invalidation (pub/sub):
     Write → commit → publish event → caches invalidate async.
     Fast writes, brief inconsistency window (~ms).

  3. Cache-aside with short TTL:
     No explicit invalidation — entries expire after TTL.
     Simple; staleness bounded by TTL.

  4. CRDT-based merge:
     Multiple cache replicas independently accept writes.
     Merge using Conflict-free Replicated Data Types.
     Use: eventually consistent counters, sets.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
import time
import threading
import random
from collections import defaultdict
from enum import Enum


class ConsistencyModel(Enum):
    STRONG   = "strong"
    EVENTUAL = "eventual"
    RYW      = "read_your_writes"
    BOUNDED  = "bounded_staleness"


# ─────────────────────────────────────────────
# VECTOR CLOCK (for ordering)
# ─────────────────────────────────────────────

class VectorClock:
    """Tracks causal ordering of events across nodes."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._clock  : Dict[str, int] = defaultdict(int)

    def tick(self):
        self._clock[self.node_id] += 1

    def update(self, other: "VectorClock"):
        for node, ts in other._clock.items():
            self._clock[node] = max(self._clock[node], ts)
        self._clock[self.node_id] += 1

    def is_concurrent(self, other: "VectorClock") -> bool:
        """Neither clock dominates the other — concurrent writes."""
        a_gt_b = any(self._clock.get(n, 0) > other._clock.get(n, 0) for n in self._clock)
        b_gt_a = any(other._clock.get(n, 0) > self._clock.get(n, 0) for n in other._clock)
        return a_gt_b and b_gt_a

    def dominates(self, other: "VectorClock") -> bool:
        """This clock is strictly after other."""
        any_gt = any(self._clock.get(n, 0) > other._clock.get(n, 0) for n in self._clock)
        all_ge = all(self._clock.get(n, 0) >= other._clock.get(n, 0) for n in other._clock)
        return any_gt and all_ge

    def __str__(self):
        return str(dict(self._clock))


# ─────────────────────────────────────────────
# VERSIONED CACHE ENTRY
# ─────────────────────────────────────────────

@dataclass
class VersionedEntry:
    value     : Any
    version   : int         # monotonic write version
    written_at: float = field(default_factory=time.time)
    node_id   : str = ""


# ─────────────────────────────────────────────
# CACHE NODE (with consistency models)
# ─────────────────────────────────────────────

class ConsistentCacheNode:
    """
    Cache node that tracks versions and supports different consistency models.
    """

    def __init__(self, node_id: str, ttl_s: float = 60.0):
        self.node_id    = node_id
        self.ttl_s      = ttl_s
        self._store     : Dict[str, VersionedEntry] = {}
        self._lock      = threading.Lock()
        self.reads      = 0
        self.stale_reads= 0
        self.writes     = 0

    def get(self, key: str, min_version: int = 0) -> Optional[VersionedEntry]:
        """
        Read value. Returns None if:
        - Not in cache
        - Expired
        - Version < min_version (for monotonic reads)
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.time() - entry.written_at > self.ttl_s:
                del self._store[key]
                return None
            if entry.version < min_version:
                self.stale_reads += 1
                return None   # Would violate monotonic read guarantee
            self.reads += 1
            return entry

    def set(self, key: str, value: Any, version: int):
        """Write value. Only write if newer version (LWW)."""
        with self._lock:
            existing = self._store.get(key)
            if existing and existing.version >= version:
                return False   # Reject old version
            self._store[key] = VersionedEntry(value, version, node_id=self.node_id)
            self.writes += 1
            return True

    def invalidate(self, key: str):
        with self._lock:
            self._store.pop(key, None)

    def size(self) -> int:
        return len(self._store)


# ─────────────────────────────────────────────
# DISTRIBUTED CACHE COORDINATOR
# ─────────────────────────────────────────────

class DistributedCacheCoordinator:
    """
    Manages consistency across multiple cache nodes.
    Supports different consistency models.
    """

    def __init__(self, nodes: List[ConsistentCacheNode],
                 model: ConsistencyModel = ConsistencyModel.EVENTUAL):
        self.nodes        = nodes
        self.model        = model
        self._write_ver   : Dict[str, int] = defaultdict(int)
        self._user_writes : Dict[str, Dict[str, int]] = defaultdict(dict)  # user → {key: version}
        self._lock        = threading.Lock()
        self.writes       = 0
        self.inconsistencies_detected = 0

    def _next_version(self, key: str) -> int:
        with self._lock:
            self._write_ver[key] += 1
            return self._write_ver[key]

    def write(self, key: str, value: Any, user_id: str = None) -> int:
        """Write to DB-equivalent and propagate to caches."""
        version = self._next_version(key)
        self.writes += 1

        if self.model == ConsistencyModel.STRONG:
            # Synchronous: write to all nodes before returning
            for node in self.nodes:
                node.set(key, value, version)

        elif self.model == ConsistencyModel.RYW:
            # Track version for this user (read-your-writes)
            if user_id:
                self._user_writes[user_id][key] = version
            # Async propagation (simplified: write immediately for demo)
            t = threading.Thread(target=self._async_propagate, args=(key, value, version))
            t.start()

        else:  # EVENTUAL / BOUNDED
            # Async propagation with random delay (simulates network)
            t = threading.Thread(target=self._async_propagate,
                                  args=(key, value, version), daemon=True)
            t.start()

        return version

    def _async_propagate(self, key: str, value: Any, version: int):
        """Simulate async replication to all cache nodes."""
        for node in self.nodes:
            delay = random.uniform(0, 50) / 1000   # 0-50ms propagation delay
            time.sleep(delay)
            node.set(key, value, version)

    def read(self, key: str, node_idx: int = 0, user_id: str = None) -> Optional[Any]:
        """Read from a specific cache node."""
        node = self.nodes[node_idx % len(self.nodes)]

        # Read-Your-Writes: require min version if this user wrote it
        min_version = 0
        if self.model == ConsistencyModel.RYW and user_id:
            min_version = self._user_writes.get(user_id, {}).get(key, 0)

        entry = node.get(key, min_version=min_version)
        return entry.value if entry else None

    def check_consistency(self, key: str) -> Dict[str, Any]:
        """Check if all nodes have the same version for a key."""
        versions = {}
        for node in self.nodes:
            entry = node._store.get(key)
            versions[node.node_id] = entry.version if entry else None

        version_set = set(v for v in versions.values() if v is not None)
        is_consistent = len(version_set) <= 1
        if not is_consistent:
            self.inconsistencies_detected += 1
        return {"versions": versions, "consistent": is_consistent}


# ─────────────────────────────────────────────
# READ-YOUR-WRITES SESSION TRACKER
# ─────────────────────────────────────────────

class ReadYourWritesTracker:
    """
    Tracks per-user write timestamps to route reads to primary DB
    within a window after a user's write.
    """

    def __init__(self, primary_window_s: float = 2.0):
        self._user_last_write : Dict[str, float] = {}
        self.primary_window   = primary_window_s

    def record_write(self, user_id: str):
        self._user_last_write[user_id] = time.time()

    def should_read_primary(self, user_id: str) -> bool:
        """Return True if user wrote recently → must read from primary."""
        last_write = self._user_last_write.get(user_id, 0)
        return time.time() - last_write < self.primary_window

    def staleness_window_s(self, user_id: str) -> float:
        """How much longer until replica reads are safe?"""
        last_write = self._user_last_write.get(user_id, 0)
        remaining  = self.primary_window - (time.time() - last_write)
        return max(0.0, remaining)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_cache_consistency():
    print("=" * 65)
    print("CACHE CONSISTENCY IN DISTRIBUTED SYSTEMS")
    print("=" * 65)

    random.seed(42)

    nodes = [
        ConsistentCacheNode(f"cache-node-{i}", ttl_s=300.0)
        for i in range(3)
    ]

    # ── Eventual Consistency ───────────────────
    print("\n[1] EVENTUAL CONSISTENCY — ASYNC PROPAGATION")
    print("─" * 55)
    ev_coord = DistributedCacheCoordinator(nodes, ConsistencyModel.EVENTUAL)

    # Pre-warm all nodes
    for node in nodes:
        node.set("user:1", {"name": "Alice_v0"}, version=1)

    print(f"  Initial state: all nodes have user:1=Alice_v0 (version=1)")

    # Write new value
    ev_coord.write("user:1", {"name": "Alice_v1"}, user_id="alice")
    time.sleep(0.01)   # brief window before propagation

    # Check immediately (nodes may not have propagated)
    print(f"  Immediately after write (async propagation in flight):")
    consistency = ev_coord.check_consistency("user:1")
    for node_id, ver in consistency["versions"].items():
        print(f"    {node_id}: version={ver}")
    print(f"    Consistent: {consistency['consistent']}")

    # Wait for propagation
    time.sleep(0.1)
    print(f"\n  After 100ms (propagation complete):")
    consistency = ev_coord.check_consistency("user:1")
    for node_id, ver in consistency["versions"].items():
        entry = nodes[[n.node_id for n in nodes].index(node_id)]._store.get("user:1")
        val = entry.value.get("name") if entry else None
        print(f"    {node_id}: version={ver} value={val}")
    print(f"    Consistent: {consistency['consistent']}")

    # ── Strong Consistency ─────────────────────
    print("\n\n[2] STRONG CONSISTENCY — SYNCHRONOUS PROPAGATION")
    print("─" * 55)
    nodes2    = [ConsistentCacheNode(f"strong-node-{i}") for i in range(3)]
    st_coord  = DistributedCacheCoordinator(nodes2, ConsistencyModel.STRONG)

    start = time.perf_counter()
    st_coord.write("product:1", {"price": 99.99})
    write_ms = (time.perf_counter() - start) * 1000

    consistency = st_coord.check_consistency("product:1")
    print(f"  Write latency (synchronous to 3 nodes): {write_ms:.1f}ms")
    print(f"  Immediately consistent: {consistency['consistent']}")
    for node_id, ver in consistency["versions"].items():
        print(f"    {node_id}: version={ver} (all updated before write returned)")

    # ── Read-Your-Writes ─────────────────────
    print("\n\n[3] READ-YOUR-WRITES (RYW) GUARANTEE")
    print("─" * 55)
    ryw_tracker = ReadYourWritesTracker(primary_window_s=2.0)

    print("  User Alice updates her profile picture")
    ryw_tracker.record_write("alice")

    # Immediately after write — must read from primary
    for attempt in range(4):
        should_primary = ryw_tracker.should_read_primary("alice")
        remaining      = ryw_tracker.staleness_window_s("alice")
        print(f"  t=0.{attempt}s: should_read_primary={should_primary}  "
              f"window_remaining={remaining:.1f}s")
        time.sleep(0.5)

    print(f"  (After 2s: reads can go to any replica — RYW window closed)")

    # Bob's reads (different user — no special routing needed)
    should_primary_bob = ryw_tracker.should_read_primary("bob")
    print(f"\n  Bob reading Alice's profile: should_primary={should_primary_bob}  (Bob didn't write)")

    # ── Vector Clocks ─────────────────────────
    print("\n\n[4] VECTOR CLOCKS — DETECTING CONCURRENT WRITES")
    print("─" * 55)
    vc_a = VectorClock("node-A")
    vc_b = VectorClock("node-B")

    vc_a.tick()   # A writes
    vc_a.tick()   # A writes again
    print(f"  Node A after 2 writes: {vc_a}")

    # B gets updated from A
    vc_b.update(vc_a)
    print(f"  Node B after syncing from A: {vc_b}")

    # Now both write independently (concurrent)
    vc_a.tick()   # A writes
    vc_b.tick()   # B writes at same time
    vc_b.tick()

    print(f"  After concurrent writes:")
    print(f"    Node A: {vc_a}")
    print(f"    Node B: {vc_b}")
    print(f"    Concurrent (conflict): {vc_a.is_concurrent(vc_b)}")
    print(f"    A dominates B: {vc_a.dominates(vc_b)}")
    print(f"    B dominates A: {vc_b.dominates(vc_a)}")
    print(f"  → Concurrent writes need conflict resolution (LWW, merge, human review)")

    # ── Consistency Models Comparison ──────────
    print("\n\n[5] CONSISTENCY MODELS COMPARISON")
    print("─" * 55)
    models = [
        ("Strong",         "No stale reads",         "High write latency (sync to all)", "Financial, inventory counts"),
        ("Read-Your-Writes","Own writes visible",     "Other users may see stale",        "Profile updates, settings"),
        ("Monotonic Read", "No going backwards",      "Slight routing overhead",          "News feeds, timelines"),
        ("Bounded Stale",  "Stale up to TTL max",     "Simple (just use TTL)",            "Prices, public content"),
        ("Eventual",       "Converges eventually",    "Brief stale window (~ms-s)",       "Likes, view counts, non-critical"),
    ]
    print(f"  {'Model':<22} {'Guarantee':<28} {'Cost':<30} {'Best For'}")
    print(f"  {'─'*90}")
    for model, guarantee, cost, best_for in models:
        print(f"  {model:<22} {guarantee:<28} {cost:<30} {best_for}")

    # ── Practical Guidelines ───────────────────
    print("\n\n[6] PRACTICAL CONSISTENCY GUIDELINES")
    print("─" * 55)
    guidelines = [
        ("User profile/settings",    "Read-your-writes",  "User must see their own changes"),
        ("Shopping cart",            "Strong",            "Item counts must be accurate"),
        ("Product prices",           "Bounded (60s TTL)", "Brief staleness acceptable"),
        ("Inventory stock count",    "Strong",            "Overselling is a business error"),
        ("Like/view counts",         "Eventual",          "Approximate is fine"),
        ("News feed / timeline",     "Eventual + mono",   "No backwards jumps, ~1s lag ok"),
        ("Session (auth state)",     "Strong",            "Logout must be instant everywhere"),
        ("Search index",             "Eventual",          "New content visible in <60s ok"),
        ("Financial transactions",   "Strong",            "ACID required — don't cache"),
    ]
    print(f"  {'Data':<30} {'Model':<20} {'Reason'}")
    print(f"  {'─'*75}")
    for data, model, reason in guidelines:
        print(f"  {data:<30} {model:<20} {reason}")


if __name__ == "__main__":
    demonstrate_cache_consistency()
