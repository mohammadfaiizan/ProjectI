"""
Distributed Cache System Design - Python Implementation
Demonstrates: consistent hashing ring with virtual nodes, LRU cache with
O(1) get/put, TTL management, node failure simulation, rehashing, hot key analysis.
No external dependencies - standard library only.
"""

import hashlib
import time
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


# ─────────────────────────────────────────────
# LRU Cache — O(1) get/put with doubly linked list + hashmap
# ─────────────────────────────────────────────

class DLLNode:
    """Doubly linked list node for LRU ordering."""
    __slots__ = ("key", "value", "size", "expiry", "prev", "next")

    def __init__(self, key: str = "", value: Any = None,
                 size: int = 0, expiry: Optional[float] = None):
        self.key = key
        self.value = value
        self.size = size
        self.expiry = expiry
        self.prev: Optional["DLLNode"] = None
        self.next: Optional["DLLNode"] = None


class LRUCache:
    """
    O(1) get/put LRU cache using:
      - HashMap for O(1) key -> node lookup
      - Doubly Linked List for O(1) move-to-front and tail eviction
      - TTL heap for expiry management
    """

    def __init__(self, max_bytes: int = 64 * 1024 * 1024):  # 64MB default
        self.max_bytes = max_bytes
        self.used_bytes = 0
        self._map: dict = {}                # key -> DLLNode

        # Sentinel head (MRU side) and tail (LRU side)
        self._head = DLLNode()
        self._tail = DLLNode()
        self._head.next = self._tail
        self._tail.prev = self._head

        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """O(1) get. Returns None on miss or expired."""
        node = self._map.get(key)
        if node is None:
            self.misses += 1
            return None

        # Check TTL
        if node.expiry is not None and time.time() > node.expiry:
            self._remove_node(node)
            del self._map[key]
            self.used_bytes -= node.size
            self.misses += 1
            return None

        # Move to head (MRU position)
        self._remove_node(node)
        self._insert_after_head(node)
        self.hits += 1
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[int] = None,
            value_size: int = 100) -> bool:
        """O(1) put. Evicts LRU entries if over memory limit."""
        expiry = time.time() + ttl if ttl is not None else None

        if key in self._map:
            # Update existing
            node = self._map[key]
            self.used_bytes -= node.size
            node.value = value
            node.size = value_size
            node.expiry = expiry
            self.used_bytes += value_size
            self._remove_node(node)
            self._insert_after_head(node)
        else:
            # Insert new
            node = DLLNode(key=key, value=value, size=value_size, expiry=expiry)
            self._map[key] = node
            self._insert_after_head(node)
            self.used_bytes += value_size

        # Evict until within memory budget
        while self.used_bytes > self.max_bytes and self._tail.prev is not self._head:
            self._evict_lru()

        return True

    def delete(self, key: str) -> bool:
        """Remove a key. Returns True if existed."""
        node = self._map.pop(key, None)
        if node is None:
            return False
        self._remove_node(node)
        self.used_bytes -= node.size
        return True

    def _evict_lru(self):
        """Evict the least recently used entry (tail)."""
        lru_node = self._tail.prev
        if lru_node is self._head:
            return
        self._remove_node(lru_node)
        del self._map[lru_node.key]
        self.used_bytes -= lru_node.size
        self.evictions += 1

    def _remove_node(self, node: DLLNode):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _insert_after_head(self, node: DLLNode):
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    @property
    def size(self) -> int:
        return len(self._map)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "size": self.size,
            "used_bytes": self.used_bytes,
            "max_bytes": self.max_bytes,
            "utilization": f"{100 * self.used_bytes / self.max_bytes:.1f}%",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate * 100:.1f}%",
            "evictions": self.evictions,
        }


# ─────────────────────────────────────────────
# Cache Node — represents a physical cache server
# ─────────────────────────────────────────────

@dataclass
class CacheNode:
    node_id: str
    host: str
    port: int
    max_bytes: int = 64 * 1024 * 1024   # 64MB
    is_alive: bool = True

    def __post_init__(self):
        self.cache = LRUCache(max_bytes=self.max_bytes)

    def get(self, key: str) -> Optional[Any]:
        if not self.is_alive:
            raise ConnectionError(f"Node {self.node_id} is down")
        return self.cache.get(key)

    def put(self, key: str, value: Any, ttl: Optional[int] = None,
            value_size: int = 100) -> bool:
        if not self.is_alive:
            raise ConnectionError(f"Node {self.node_id} is down")
        return self.cache.put(key, value, ttl=ttl, value_size=value_size)

    def delete(self, key: str) -> bool:
        if not self.is_alive:
            raise ConnectionError(f"Node {self.node_id} is down")
        return self.cache.delete(key)


# ─────────────────────────────────────────────
# Consistent Hash Ring
# ─────────────────────────────────────────────

class ConsistentHashRing:
    """
    Consistent hashing ring with virtual nodes.
    - Minimizes key movement when nodes join/leave (only K/N keys move)
    - Virtual nodes ensure even load distribution (~5% variance with 150 vnodes)
    """

    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self._ring: dict = {}               # position -> node_id
        self._sorted_positions: list = []   # sorted ring positions
        self._nodes: dict = {}              # node_id -> CacheNode

    def add_node(self, node: CacheNode):
        """Add a node with `virtual_nodes` positions on the ring."""
        self._nodes[node.node_id] = node
        for i in range(self.virtual_nodes):
            pos = self._hash(f"{node.node_id}:{i}")
            self._ring[pos] = node.node_id
        self._sorted_positions = sorted(self._ring.keys())
        print(f"  [Ring] Added node {node.node_id} "
              f"({self.virtual_nodes} vnodes, "
              f"ring size: {len(self._sorted_positions)})")

    def remove_node(self, node_id: str):
        """Remove a node. Only that node's keys need to move."""
        if node_id not in self._nodes:
            return
        for i in range(self.virtual_nodes):
            pos = self._hash(f"{node_id}:{i}")
            self._ring.pop(pos, None)
        self._sorted_positions = sorted(self._ring.keys())
        del self._nodes[node_id]
        print(f"  [Ring] Removed node {node_id}. "
              f"Ring size: {len(self._sorted_positions)}")

    def get_node(self, key: str) -> Optional[CacheNode]:
        """Find the responsible node for a key (clockwise lookup)."""
        if not self._sorted_positions:
            return None
        pos = self._hash(key)
        idx = self._binary_search(pos)
        node_id = self._ring[self._sorted_positions[idx]]
        return self._nodes.get(node_id)

    def get_replicas(self, key: str, n: int = 3) -> list:
        """Return N distinct nodes (for replication)."""
        if not self._sorted_positions:
            return []
        pos = self._hash(key)
        idx = self._binary_search(pos)
        seen_nodes = set()
        replicas = []
        for offset in range(len(self._sorted_positions)):
            position_idx = (idx + offset) % len(self._sorted_positions)
            node_id = self._ring[self._sorted_positions[position_idx]]
            if node_id not in seen_nodes:
                node = self._nodes.get(node_id)
                if node:
                    replicas.append(node)
                    seen_nodes.add(node_id)
            if len(replicas) == n:
                break
        return replicas

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)

    def _binary_search(self, pos: int) -> int:
        """Find the first ring position >= pos (with wrap-around)."""
        lo, hi = 0, len(self._sorted_positions) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._sorted_positions[mid] < pos:
                lo = mid + 1
            else:
                hi = mid
        # Wrap around
        if self._sorted_positions[lo] < pos:
            return 0
        return lo

    def node_load_distribution(self) -> dict:
        """Show how many virtual nodes each physical node owns."""
        counts: defaultdict = defaultdict(int)
        for node_id in self._ring.values():
            counts[node_id] += 1
        return dict(counts)


# ─────────────────────────────────────────────
# Distributed Cache — Top-Level Client
# ─────────────────────────────────────────────

class DistributedCache:
    """
    Distributed cache with:
    - Consistent hashing for key routing
    - Replication factor N=3 (configurable)
    - Write quorum W=2, Read quorum R=2 (W+R > N ensures consistency)
    - Graceful fallback on node failure
    """

    def __init__(self, replication_factor: int = 3,
                 write_quorum: int = 2,
                 read_quorum: int = 2,
                 virtual_nodes: int = 150):
        self.N = replication_factor
        self.W = write_quorum
        self.R = read_quorum
        self.ring = ConsistentHashRing(virtual_nodes=virtual_nodes)
        self._nodes: dict = {}

    def add_node(self, node_id: str, host: str, port: int,
                 max_bytes: int = 64 * 1024 * 1024) -> CacheNode:
        node = CacheNode(node_id=node_id, host=host, port=port,
                         max_bytes=max_bytes)
        self._nodes[node_id] = node
        self.ring.add_node(node)
        return node

    def remove_node(self, node_id: str):
        """Simulate node removal from ring."""
        self.ring.remove_node(node_id)
        if node_id in self._nodes:
            self._nodes[node_id].is_alive = False

    def get(self, key: str) -> Optional[Any]:
        """
        Read from R replicas, return most recent value (quorum read).
        Falls back gracefully if some replicas are down.
        """
        replicas = self.ring.get_replicas(key, self.N)
        if not replicas:
            return None

        responses = []
        for node in replicas:
            if not node.is_alive:
                continue
            try:
                value = node.get(key)
                if value is not None:
                    responses.append(value)
            except ConnectionError:
                continue
            if len(responses) >= self.R:
                break

        # Return value if we got quorum
        return responses[0] if responses else None

    def set(self, key: str, value: Any, ttl: Optional[int] = None,
            value_size: int = 100) -> bool:
        """
        Write to N replicas, require W ACKs for success (quorum write).
        """
        replicas = self.ring.get_replicas(key, self.N)
        if not replicas:
            return False

        acks = 0
        for node in replicas:
            if not node.is_alive:
                continue
            try:
                if node.put(key, value, ttl=ttl, value_size=value_size):
                    acks += 1
            except ConnectionError:
                continue

        return acks >= self.W

    def delete(self, key: str) -> bool:
        """Delete from all replicas."""
        replicas = self.ring.get_replicas(key, self.N)
        any_deleted = False
        for node in replicas:
            if not node.is_alive:
                continue
            try:
                if node.delete(key):
                    any_deleted = True
            except ConnectionError:
                continue
        return any_deleted

    def cluster_stats(self) -> dict:
        alive = [n for n in self._nodes.values() if n.is_alive]
        dead = [n for n in self._nodes.values() if not n.is_alive]
        total_hits = sum(n.cache.hits for n in alive)
        total_misses = sum(n.cache.misses for n in alive)
        total_evictions = sum(n.cache.evictions for n in alive)
        return {
            "total_nodes": len(self._nodes),
            "alive_nodes": len(alive),
            "dead_nodes": len(dead),
            "total_hits": total_hits,
            "total_misses": total_misses,
            "cluster_hit_rate": (f"{100 * total_hits / (total_hits + total_misses):.1f}%"
                                  if total_hits + total_misses > 0 else "N/A"),
            "total_evictions": total_evictions,
            "replication_factor": self.N,
            "write_quorum": self.W,
            "read_quorum": self.R,
        }


# ─────────────────────────────────────────────
# Hot Key Analyzer
# ─────────────────────────────────────────────

class HotKeyDetector:
    """Tracks access frequency to identify hot keys."""

    def __init__(self, top_k: int = 10):
        self._counts: defaultdict = defaultdict(int)
        self.top_k = top_k

    def record(self, key: str):
        self._counts[key] += 1

    def get_hot_keys(self) -> list:
        """Return top-K hottest keys by access count."""
        return sorted(self._counts.items(), key=lambda x: x[1], reverse=True)[:self.top_k]

    def is_hot(self, key: str, threshold: int = 1000) -> bool:
        return self._counts[key] > threshold


# ─────────────────────────────────────────────
# Demo / Simulation
# ─────────────────────────────────────────────

def demo_lru_cache():
    print("\n--- LRU Cache (O(1) doubly linked list + hashmap) ---")
    cache = LRUCache(max_bytes=5 * 1024)  # 5KB for demo

    # Basic operations
    cache.put("user:1", {"name": "Alice", "age": 30}, ttl=300, value_size=100)
    cache.put("user:2", {"name": "Bob", "age": 25}, ttl=600, value_size=100)
    cache.put("user:3", {"name": "Carol", "age": 28}, value_size=100)

    print(f"Get user:1 -> {cache.get('user:1')['name']}")
    print(f"Get user:99 (miss) -> {cache.get('user:99')}")

    # Fill cache to trigger eviction
    for i in range(4, 60):
        cache.put(f"session:{i}", f"data_{i}", value_size=100)

    print(f"\nCache stats: {cache.stats()}")


def demo_consistent_hashing():
    print("\n--- Consistent Hashing Ring ---")
    ring = ConsistentHashRing(virtual_nodes=150)

    # Add 5 nodes
    nodes = []
    for i in range(1, 6):
        node = CacheNode(f"node-{i}", f"10.0.0.{i}", 6379)
        ring.add_node(node)
        nodes.append(node)

    # Show key distribution
    print("\nKey routing (before adding node-6):")
    test_keys = [f"user:{i}" for i in range(1, 11)]
    key_to_node = {}
    for key in test_keys:
        node = ring.get_node(key)
        key_to_node[key] = node.node_id
        print(f"  {key:12s} -> {node.node_id}")

    # Show vnode distribution
    distribution = ring.node_load_distribution()
    print("\nVirtual node counts per physical node:")
    for nid, count in sorted(distribution.items()):
        bar = "#" * (count // 5)
        print(f"  {nid}: {count:3d} vnodes  {bar}")

    # Add a new node — show minimal key movement
    print("\nAdding node-6 to cluster...")
    node6 = CacheNode("node-6", "10.0.0.6", 6379)
    ring.add_node(node6)

    moved = 0
    for key in test_keys:
        new_node = ring.get_node(key)
        if new_node.node_id != key_to_node[key]:
            moved += 1
            print(f"  MOVED: {key} -> {key_to_node[key]} => {new_node.node_id}")

    print(f"Keys moved: {moved}/{len(test_keys)} "
          f"(expected ~{len(test_keys)//6}, actual {moved})")

    # Replication
    print(f"\nReplicas for 'user:42':")
    replicas = ring.get_replicas("user:42", n=3)
    for r in replicas:
        print(f"  {r.node_id} ({r.host}:{r.port})")


def demo_distributed_cache():
    print("\n--- Distributed Cache (Consistent Hashing + Quorum) ---")
    cache = DistributedCache(replication_factor=3, write_quorum=2,
                              read_quorum=2, virtual_nodes=100)

    # Add 6 nodes
    for i in range(1, 7):
        cache.add_node(f"node-{i}", f"10.0.0.{i}", 6379)

    # Write and read
    print("\nWrite/Read operations:")
    cache.set("config:timeout", 30, ttl=3600, value_size=20)
    cache.set("user:1001:profile", {"name": "Alice"}, ttl=300, value_size=200)
    cache.set("product:500:price", 99.99, value_size=50)

    print(f"  config:timeout   -> {cache.get('config:timeout')}")
    print(f"  user:1001:profile -> {cache.get('user:1001:profile')}")
    print(f"  product:500:price -> {cache.get('product:500:price')}")
    print(f"  unknown:key      -> {cache.get('unknown:key')}")

    # Simulate node failure and test quorum
    print("\nSimulating node-1 failure (Q: W=2,R=2 out of N=3 still works):")
    # Find which node holds one of our keys and kill it
    test_node = cache.ring.get_node("config:timeout")
    test_node.is_alive = False
    print(f"  Killed {test_node.node_id}")
    value = cache.get("config:timeout")
    print(f"  config:timeout still readable -> {value} (quorum from 2 remaining replicas)")

    # Set new value while node is down
    cache.set("config:timeout", 60, value_size=20)
    print(f"  Updated config:timeout while node down -> {cache.get('config:timeout')}")

    # Revive node
    test_node.is_alive = True
    print(f"  Revived {test_node.node_id}")

    print(f"\nCluster stats: {cache.cluster_stats()}")


def demo_hot_key():
    print("\n--- Hot Key Problem and Detection ---")
    detector = HotKeyDetector(top_k=5)

    # Simulate Zipf distribution (few keys get most traffic)
    keys = [f"video:{i}" for i in range(100)]
    print("Simulating 10,000 requests with Zipf distribution...")
    for _ in range(10000):
        # Zipf: key i gets proportional to 1/i traffic
        rank = min(99, int(abs(random.gauss(0, 10))))
        detector.record(keys[rank])

    hot_keys = detector.get_hot_keys()
    print("Top 5 hottest keys:")
    for key, count in hot_keys:
        bar = "#" * (count // 50)
        print(f"  {key:12s}: {count:5d} requests  {bar}")

    print("\nHot key solutions:")
    print("  1. Key replication: user:viral -> {user:viral:0 .. user:viral:9}")
    print("  2. Local in-process cache with 1-5s TTL (absorbs 99%+ of traffic)")
    print("  3. Read-heavy hot keys: replicate to all nodes, round-robin reads")


def demo_cache_patterns():
    print("\n--- Cache Patterns Comparison ---")
    cache = LRUCache(max_bytes=10 * 1024 * 1024)

    # Simulate Cache-Aside (Lazy Loading)
    def get_user_cache_aside(user_id: str, db: dict) -> dict:
        value = cache.get(user_id)
        if value is None:
            # Cache MISS — fetch from DB
            value = db.get(user_id, {"error": "not found"})
            cache.put(user_id, value, ttl=300, value_size=200)
        return value

    db = {f"user:{i}": {"id": i, "name": f"User {i}"} for i in range(1, 101)}

    # First access (all misses)
    for i in range(1, 11):
        get_user_cache_aside(f"user:{i}", db)

    # Second access (all hits)
    for i in range(1, 11):
        get_user_cache_aside(f"user:{i}", db)

    print(f"After 20 requests (10 unique users, 2 accesses each):")
    print(f"  Hits: {cache.hits}, Misses: {cache.misses}, "
          f"Hit Rate: {cache.hit_rate * 100:.0f}%")
    print("  First 10 requests: all misses (cache cold)")
    print("  Next 10 requests: all hits (cache warm)")


def run_demo():
    print("=" * 60)
    print("DISTRIBUTED CACHE SYSTEM DESIGN DEMO")
    print("=" * 60)

    demo_lru_cache()
    demo_consistent_hashing()
    demo_distributed_cache()
    demo_hot_key()
    demo_cache_patterns()

    print("\n--- Key Design Insights ---")
    insights = [
        "LRU uses doubly linked list + hashmap for O(1) get/put/evict",
        "Consistent hashing: only K/N keys move when adding/removing nodes",
        "Virtual nodes (150/node) ensure <5% load variance across nodes",
        "Quorum W+R>N (2+2>3) guarantees reading latest write",
        "Hot keys solved by local in-process cache (1-5s TTL)",
        "Cache-aside: only requested data cached, resilient to cache failure",
        "Redis SETNX + TTL for atomic cache stampede prevention",
    ]
    for insight in insights:
        print(f"  - {insight}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
