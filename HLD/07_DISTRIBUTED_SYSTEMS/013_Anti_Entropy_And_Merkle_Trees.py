"""
ANTI-ENTROPY AND MERKLE TREES
================================

Problem Statement:
In an eventually consistent system, replicas may diverge:
- Failed writes: node was down during a write, missed updates.
- Network partitions: temporary isolation, data diverged.
- Silent bit rot: data corruption without detection.
Anti-entropy is the background process that detects and repairs these divergences.

Naive Anti-Entropy:
  Compare all key-value pairs between two nodes.
  Send all keys from N1 to N2 and vice versa. O(N) data transferred.
  Too expensive for large datasets.

Merkle Tree (Hash Tree):
  Tree of hashes. Leaf nodes = hash(data segment).
  Parent = hash(left_child || right_child).
  Root = single hash representing the entire dataset.
  Two nodes with identical root → data is identical (with high probability).
  Two nodes with different root → walk the tree to find diverging subtrees.
  Cost: O(log N) messages to identify diverging leaf segments.
  Used by: Cassandra, DynamoDB, Git, BitTorrent, ZFS, Certificate Transparency.

Anti-Entropy via Merkle Trees:
  1. Both nodes compute Merkle tree over their data (partitioned into segments).
  2. Exchange root hashes. If equal → done.
  3. If not equal → recursively compare children until diverging leaves found.
  4. Exchange only the diverging data segments.
  O(d * log N) messages where d = number of diverging segments.

Merkle Tree in Cassandra:
  Each node computes a Merkle tree over each virtual node (vnode).
  Anti-entropy repair (nodetool repair) exchanges trees with neighbors.
  Cassandra repairs one vnode at a time during "repair" operations.

Read Repair (Complementary):
  Lazy, on read path. Only repairs keys that are actually read.
  Anti-entropy: proactive, repairs all keys including unread ones.
  Both needed for full coverage.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import hashlib
import time
import random


# ─────────────────────────────────────────────
# MERKLE TREE
# ─────────────────────────────────────────────

@dataclass
class MerkleNode:
    hash_val  : str
    is_leaf   : bool   = False
    key_range : Tuple  = (0, 0)   # (start, end) for leaf nodes
    left      : Optional["MerkleNode"] = None
    right     : Optional["MerkleNode"] = None


def _hash(data: str) -> str:
    return hashlib.md5(data.encode()).hexdigest()[:16]


def build_merkle_tree(data_segments: List[Tuple[int, str]]) -> MerkleNode:
    """
    Build a Merkle tree from a list of (segment_id, hash_of_segment).
    """
    if not data_segments:
        return MerkleNode(hash_val=_hash("empty"))

    # Create leaf nodes
    leaves = []
    for seg_id, seg_hash in data_segments:
        node = MerkleNode(
            hash_val  = seg_hash,
            is_leaf   = True,
            key_range = (seg_id, seg_id),
        )
        leaves.append(node)

    # Build up the tree
    current_level = leaves
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            if i + 1 < len(current_level):
                right = current_level[i + 1]
                combined = _hash(left.hash_val + right.hash_val)
                parent = MerkleNode(
                    hash_val  = combined,
                    left      = left,
                    right     = right,
                    key_range = (left.key_range[0], right.key_range[1]),
                )
            else:
                # Odd node: promote with itself
                parent = MerkleNode(
                    hash_val  = left.hash_val,
                    left      = left,
                    right     = None,
                    key_range = left.key_range,
                )
            next_level.append(parent)
        current_level = next_level

    return current_level[0]


def find_diverging_segments(
    node1: Optional[MerkleNode],
    node2: Optional[MerkleNode],
) -> List[Tuple[int, int]]:
    """
    Compare two Merkle trees.
    Returns list of (start, end) key ranges that differ.
    """
    if node1 is None and node2 is None:
        return []
    if node1 is None or node2 is None:
        return [(0, 0)]   # one side is empty

    if node1.hash_val == node2.hash_val:
        return []   # subtrees are identical

    if node1.is_leaf or node2.is_leaf:
        # Leaf differs → this segment needs repair
        return [node1.key_range]

    # Recurse into children
    diff = []
    diff.extend(find_diverging_segments(node1.left,  node2.left))
    diff.extend(find_diverging_segments(node1.right, node2.right))
    return diff


# ─────────────────────────────────────────────
# DATA SEGMENT (partitions the keyspace)
# ─────────────────────────────────────────────

class DataNode:
    """
    A replica that stores data and can compute a Merkle tree over its content.
    Data is partitioned into segments for Merkle tree computation.
    """

    def __init__(self, node_id: str, n_segments: int = 8):
        self.node_id    = node_id
        self.n_segments = n_segments
        self._data      : Dict[str, Any] = {}
        self.sync_ops   = 0

    def put(self, key: str, value: Any):
        self._data[key] = value

    def delete(self, key: str):
        self._data.pop(key, None)

    def get(self, key: str) -> Optional[Any]:
        return self._data.get(key)

    def _segment_for_key(self, key: str) -> int:
        """Assign key to a segment based on hash."""
        return int(_hash(key), 16) % self.n_segments

    def _segment_hash(self, seg_id: int) -> str:
        """Hash of all key-value pairs in a segment (sorted for determinism)."""
        items = sorted(
            (k, v) for k, v in self._data.items()
            if self._segment_for_key(k) == seg_id
        )
        return _hash(str(items))

    def build_merkle_tree(self) -> MerkleNode:
        segments = [(i, self._segment_hash(i)) for i in range(self.n_segments)]
        return build_merkle_tree(segments)

    def get_segment_keys(self, seg_id: int) -> Dict[str, Any]:
        return {k: v for k, v in self._data.items()
                if self._segment_for_key(k) == seg_id}


# ─────────────────────────────────────────────
# ANTI-ENTROPY ENGINE
# ─────────────────────────────────────────────

class AntiEntropyEngine:
    """
    Compares two nodes using Merkle trees.
    Repairs diverging segments by syncing missing/different data.
    """

    def repair(self, node_a: DataNode, node_b: DataNode) -> Dict:
        """
        Run anti-entropy repair between two nodes.
        Returns repair stats.
        """
        tree_a = node_a.build_merkle_tree()
        tree_b = node_b.build_merkle_tree()

        if tree_a.hash_val == tree_b.hash_val:
            return {"messages": 2, "diverging_segments": 0, "keys_synced": 0}

        # Find diverging segments
        diverging = find_diverging_segments(tree_a, tree_b)
        messages  = 2 + len(diverging) * 2   # root exchange + per-segment comparison

        keys_synced = 0
        for seg_start, seg_end in diverging:
            for seg_id in range(seg_start, seg_end + 1):
                # Sync: take union, prefer node_a's version (could be smarter)
                seg_a = node_a.get_segment_keys(seg_id)
                seg_b = node_b.get_segment_keys(seg_id)

                # Keys in A not in B (or different)
                for k, v in seg_a.items():
                    if k not in seg_b or seg_b[k] != v:
                        node_b.put(k, v)
                        keys_synced += 1
                        node_b.sync_ops += 1

                # Keys in B not in A
                for k, v in seg_b.items():
                    if k not in seg_a:
                        node_a.put(k, v)
                        keys_synced += 1
                        node_a.sync_ops += 1

        return {
            "messages"          : messages,
            "diverging_segments": len(diverging),
            "keys_synced"       : keys_synced,
        }

    def are_in_sync(self, node_a: DataNode, node_b: DataNode) -> bool:
        return node_a.build_merkle_tree().hash_val == node_b.build_merkle_tree().hash_val


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_anti_entropy():
    print("=" * 65)
    print("ANTI-ENTROPY AND MERKLE TREES")
    print("=" * 65)

    random.seed(42)

    # ── Merkle Tree Construction ───────────────────
    print("\n[1] MERKLE TREE — CONSTRUCTION")
    print("─" * 55)

    node_a = DataNode("replica-A", n_segments=8)
    node_b = DataNode("replica-B", n_segments=8)

    # Identical initial data
    for i in range(20):
        key = f"user:{i:03d}"
        val = {"name": f"User-{i}", "score": i * 10}
        node_a.put(key, val)
        node_b.put(key, val)

    tree_a = node_a.build_merkle_tree()
    tree_b = node_b.build_merkle_tree()
    print(f"  20 keys, identical on both replicas")
    print(f"  Root A: {tree_a.hash_val}")
    print(f"  Root B: {tree_b.hash_val}")
    print(f"  Roots match: {tree_a.hash_val == tree_b.hash_val}")

    # ── Divergence Detection ──────────────────────
    print("\n\n[2] DIVERGENCE DETECTION — EFFICIENT IDENTIFICATION")
    print("─" * 55)

    # Introduce 3 divergences on node_a (missed writes)
    node_a.put("user:021", {"name": "User-21"})
    node_a.put("user:022", {"name": "User-22"})
    node_a.delete("user:005")   # deleted on A but still on B

    tree_a2 = node_a.build_merkle_tree()
    tree_b2 = node_b.build_merkle_tree()

    diverging = find_diverging_segments(tree_a2, tree_b2)
    total_keys = len(node_a._data) + len(node_b._data)

    print(f"  Introduced 3 divergences (2 new + 1 delete)")
    print(f"  Root A: {tree_a2.hash_val}")
    print(f"  Root B: {tree_b2.hash_val}")
    print(f"  Diverging segments found: {len(diverging)} (out of 8 segments)")
    print(f"  Comparison messages: ~{2 + len(diverging)*2} "
          f"(vs naive: {total_keys} key exchanges)")

    # ── Anti-Entropy Repair ───────────────────────
    print("\n\n[3] ANTI-ENTROPY REPAIR")
    print("─" * 55)

    engine = AntiEntropyEngine()
    stats  = engine.repair(node_a, node_b)

    print(f"  Before repair: same root? {stats['diverging_segments'] == 0}")
    print(f"  Diverging segments: {stats['diverging_segments']}")
    print(f"  Keys synced     : {stats['keys_synced']}")
    print(f"  Messages exchanged: ~{stats['messages']}")
    print(f"\n  After repair: in sync? {engine.are_in_sync(node_a, node_b)}")

    # ── Scale Comparison ──────────────────────────
    print("\n\n[4] EFFICIENCY — MERKLE vs NAIVE SYNC")
    print("─" * 55)

    scenarios = [
        (1_000_000, 10),
        (1_000_000, 100),
        (1_000_000, 1000),
        (10_000_000, 100),
    ]
    print(f"  {'Total keys':<14} {'Diverging':<12} {'Naive msgs':<14} {'Merkle msgs (est)'}")
    print(f"  {'─'*60}")
    import math
    for total_keys, diverging_keys in scenarios:
        n_segments   = 1000
        div_segs     = max(1, int(diverging_keys / (total_keys / n_segments)))
        merkle_msgs  = 2 + int(math.log2(n_segments)) * div_segs * 2
        naive_msgs   = total_keys
        print(f"  {total_keys:<14,} {diverging_keys:<12,} {naive_msgs:<14,} {merkle_msgs}")

    # ── How Cassandra Uses Merkle Trees ───────────
    print("\n\n[5] MERKLE TREES IN CASSANDRA")
    print("─" * 55)
    steps = [
        "1. nodetool repair initiated (or automatic scheduled repair)",
        "2. Each node computes Merkle tree over each token range (vnode)",
        "3. Adjacent replicas exchange root hashes",
        "4. If different: walk tree to find diverging token ranges",
        "5. Stream only diverging data between replicas (not full dataset)",
        "6. Repeat until all vnodes repaired",
    ]
    for step in steps:
        print(f"  {step}")

    print("\n\n[6] ANTI-ENTROPY STRATEGY COMPARISON")
    print("─" * 55)
    rows = [
        ("Read repair",  "Lazy, on read path",     "Only fixes read keys",      "Cassandra, Riak"),
        ("Merkle sync",  "Proactive, background",  "Finds all divergence",      "Cassandra repair"),
        ("Full sync",    "Simple, complete",       "O(N) data transfer",        "Small datasets"),
        ("CRC check",    "Fast range verification","Per-range hash comparison",  "ZFS, BitTorrent"),
    ]
    print(f"  {'Strategy':<16} {'When':<22} {'Limitation':<25} {'Used by'}")
    print(f"  {'─'*75}")
    for strategy, when, limit, used_by in rows:
        print(f"  {strategy:<16} {when:<22} {limit:<25} {used_by}")


if __name__ == "__main__":
    demonstrate_anti_entropy()
