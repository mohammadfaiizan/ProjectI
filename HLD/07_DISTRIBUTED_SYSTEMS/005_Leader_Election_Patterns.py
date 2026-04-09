"""
LEADER ELECTION PATTERNS
==========================

Problem Statement:
Many distributed systems require exactly one node to be "in charge" at a time:
  - Only one node should write to the primary database.
  - Only one scheduler should run a cron job.
  - Only one consumer should hold a partition lock.
Without proper leader election, you get split-brain (two leaders) → data corruption.

Leader Election Requirements:
  Safety:    At most one leader at any time (no split-brain).
  Liveness:  A leader will eventually be elected (system makes progress).
  Stability: Leader doesn't change unnecessarily (avoid flapping).

Patterns:

  1. Bully Algorithm:
     Highest-ID node wins. Node starts election by messaging all higher-ID nodes.
     If no response → it becomes leader and broadcasts victory.
     Simple but: O(n²) messages; highest node always wins even if slow.

  2. Ring Election:
     Nodes arranged in a logical ring. Election token passes clockwise.
     Each node appends its ID. Highest ID when token returns = leader.
     O(n) messages. Requires ring topology.

  3. Consensus-Based (Raft/Paxos):
     Proper distributed consensus. See 004_Distributed_Consensus_Algorithms.py.
     Used by etcd, ZooKeeper, Consul.

  4. Lease-Based (ZooKeeper/etcd):
     Nodes try to create an ephemeral lock node.
     First to create it becomes leader. Node holds a lease (TTL).
     Must renew lease before it expires. On crash: TTL expires → new election.
     Most common pattern in production systems.

  5. Randomized Election (Raft-style):
     Random election timeout. First to time out starts election.
     Reduces collision probability. Used in Raft.

Split-Brain Prevention:
  Fencing token: new leader gets a monotonically increasing token.
  All followers reject requests with old token.
  Ensures old leader's writes are rejected even if it doesn't know it's dethroned.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum
import time
import uuid
import threading
import random


# ─────────────────────────────────────────────
# LEASE-BASED LEADER ELECTION (ZooKeeper/etcd style)
# ─────────────────────────────────────────────

class LeaderStore:
    """
    Simulates etcd/ZooKeeper's atomic compare-and-set (CAS) for leader key.
    Only one node can hold the leader key at a time.
    """

    def __init__(self, lease_ttl_s: float = 2.0):
        self._leader     : Optional[str]  = None
        self._token      : int            = 0    # fencing token
        self._expires_at : float          = 0.0
        self.lease_ttl_s = lease_ttl_s
        self._lock       = threading.Lock()

    def try_acquire(self, node_id: str) -> Optional[int]:
        """
        Atomically acquire leadership if no current leader or lease expired.
        Returns fencing token if successful, None otherwise.
        """
        with self._lock:
            now = time.time()
            if self._leader and now < self._expires_at:
                return None   # still has active leader
            self._leader     = node_id
            self._token     += 1
            self._expires_at = now + self.lease_ttl_s
            return self._token

    def renew(self, node_id: str, token: int) -> bool:
        """Renew lease. Only current leader with correct token can renew."""
        with self._lock:
            if self._leader != node_id or self._token != token:
                return False
            self._expires_at = time.time() + self.lease_ttl_s
            return True

    def release(self, node_id: str, token: int) -> bool:
        with self._lock:
            if self._leader == node_id and self._token == token:
                self._leader     = None
                self._expires_at = 0.0
                return True
            return False

    def current_leader(self) -> Optional[str]:
        with self._lock:
            if self._leader and time.time() < self._expires_at:
                return self._leader
            return None

    def current_token(self) -> int:
        return self._token


class LeaderNode:
    """
    A cluster node that participates in lease-based leader election.
    Runs a background thread to renew its lease while leader.
    """

    def __init__(self, node_id: str, store: LeaderStore,
                 renew_interval_s: float = 0.3):
        self.node_id         = node_id
        self.store           = store
        self.renew_interval  = renew_interval_s
        self._token          : Optional[int] = None
        self._is_leader      = False
        self._running        = False
        self._thread         : Optional[threading.Thread] = None
        self.terms_as_leader : List[Tuple[float, float]] = []   # (start, end)
        self._term_start     : float = 0.0
        self.renewals        = 0
        self.lost_leadership = 0

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            if not self._is_leader:
                token = self.store.try_acquire(self.node_id)
                if token:
                    self._token      = token
                    self._is_leader  = True
                    self._term_start = time.time()
            else:
                if not self.store.renew(self.node_id, self._token):
                    self._is_leader = False
                    self.lost_leadership += 1
                    self.terms_as_leader.append((self._term_start, time.time()))
                else:
                    self.renewals += 1
            time.sleep(self.renew_interval)

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    def fencing_token(self) -> Optional[int]:
        return self._token if self._is_leader else None


# ─────────────────────────────────────────────
# BULLY ALGORITHM (simplified)
# ─────────────────────────────────────────────

class BullyNode:
    """
    Bully algorithm: highest ID wins election.
    Node starts election when it suspects the current leader is down.
    """

    def __init__(self, node_id: int, cluster: "BullyCluster"):
        self.node_id  = node_id
        self.cluster  = cluster
        self.alive    = True
        self.leader   = -1

    def start_election(self):
        """Send election message to all nodes with higher IDs."""
        higher = [n for n in self.cluster.nodes.values()
                  if n.node_id > self.node_id and n.alive]
        if not higher:
            # No higher node → become leader
            self.cluster.broadcast_leader(self.node_id)
            return
        # In full implementation: wait for OK response, timeout → become leader
        # Simplified: trigger election recursively
        highest_alive = max(higher, key=lambda n: n.node_id)
        highest_alive.start_election()

    def receive_victory(self, leader_id: int):
        self.leader = leader_id


class BullyCluster:
    def __init__(self, n_nodes: int):
        self.nodes = {i: BullyNode(i, self) for i in range(n_nodes)}

    def broadcast_leader(self, leader_id: int):
        for node in self.nodes.values():
            if node.alive:
                node.receive_victory(leader_id)

    def kill(self, node_id: int):
        self.nodes[node_id].alive = False
        if all(n.leader == node_id for n in self.nodes.values() if n.alive):
            # Leader died — start election from lowest alive node
            lowest = min(n for n in self.nodes.values() if n.alive and n.node_id != node_id,
                         key=lambda n: n.node_id)
            lowest.start_election()


# ─────────────────────────────────────────────
# FENCING TOKEN USAGE
# ─────────────────────────────────────────────

class FencedResource:
    """
    Resource that rejects requests with stale fencing tokens.
    Protects against split-brain writes from deposed leaders.
    """

    def __init__(self):
        self._highest_token : int = 0
        self._data          : Dict[str, Any] = {}
        self.rejected       : int = 0
        self.accepted       : int = 0

    def write(self, key: str, value: Any, token: int) -> bool:
        if token < self._highest_token:
            self.rejected += 1
            return False
        if token > self._highest_token:
            self._highest_token = token
        self._data[key] = value
        self.accepted += 1
        return True


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_leader_election():
    print("=" * 65)
    print("LEADER ELECTION PATTERNS")
    print("=" * 65)

    # ── Lease-Based Election ──────────────────────
    print("\n[1] LEASE-BASED ELECTION (etcd/ZooKeeper style)")
    print("─" * 55)

    store = LeaderStore(lease_ttl_s=0.5)
    nodes = [LeaderNode(f"node-{i}", store, renew_interval_s=0.1)
             for i in range(5)]
    for n in nodes:
        n.start()

    time.sleep(0.2)

    leader = store.current_leader()
    token  = store.current_token()
    print(f"  Leader elected: {leader} (token={token})")
    for n in nodes:
        print(f"    {n.node_id}: is_leader={n.is_leader}")

    # ── Fencing Token ─────────────────────────────
    print("\n\n[2] FENCING TOKEN — SPLIT-BRAIN PROTECTION")
    print("─" * 55)

    store2     = LeaderStore(lease_ttl_s=0.2)
    resource   = FencedResource()
    node_a     = LeaderNode("nodeA", store2, renew_interval_s=0.05)
    node_b     = LeaderNode("nodeB", store2, renew_interval_s=0.05)

    node_a.start()
    time.sleep(0.05)   # A acquires leadership first

    token_a = node_a.fencing_token()
    print(f"  nodeA is leader, token={token_a}")

    # nodeA writes with its token
    resource.write("config", "version-1", token=token_a)
    print(f"  nodeA writes config: token={token_a} → accepted={resource.accepted}")

    # Simulate nodeA network partition — lease expires
    node_a.stop()
    time.sleep(0.4)   # wait for lease to expire

    node_b.start()
    time.sleep(0.1)

    token_b = node_b.fencing_token()
    print(f"  nodeB is now leader, token={token_b}")
    resource.write("config", "version-2", token=token_b)
    print(f"  nodeB writes config: token={token_b} → accepted")

    # Old nodeA wakes up and tries to write with stale token
    stale_ok = resource.write("config", "stale-version", token=token_a)
    print(f"  nodeA (deposed) tries to write with old token={token_a}: "
          f"accepted={stale_ok}")
    print(f"  Resource: accepted={resource.accepted} rejected={resource.rejected}")

    # ── Bully Algorithm ───────────────────────────
    print("\n\n[3] BULLY ALGORITHM — HIGHEST ID WINS")
    print("─" * 55)

    cluster = BullyCluster(5)
    # Elect initial leader (highest ID = 4)
    cluster.nodes[0].start_election()   # lowest node starts election
    leaders = {n.node_id: n.leader for n in cluster.nodes.values() if n.alive}
    current_leader = leaders.get(0, -1)
    print(f"  Initial leader: Node {max(leaders.values())} (highest ID)")

    # Kill leader → new election
    cluster.kill(4)
    leaders_after = {n.node_id: n.leader for n in cluster.nodes.values() if n.alive}
    print(f"  After killing Node 4: new leader = Node {max(leaders_after.values())}")

    # Stop all nodes
    for n in nodes:
        n.stop()
    node_a.stop()
    node_b.stop()

    # ── Pattern Comparison ────────────────────────
    print("\n\n[4] LEADER ELECTION PATTERNS COMPARISON")
    print("─" * 55)
    patterns = [
        ("Bully",         "O(n²) msgs",    "Simple clusters, small N"),
        ("Ring",          "O(n) msgs",     "Token ring topologies"),
        ("Raft/Paxos",    "Strong safety",  "etcd, CockroachDB, Consul"),
        ("Lease/etcd",    "TTL-based",      "Most production systems"),
        ("Randomized",    "Low collision",  "Raft election timeout"),
    ]
    print(f"  {'Pattern':<16} {'Cost':<16} {'Use Case'}")
    print(f"  {'─'*55}")
    for pattern, cost, use in patterns:
        print(f"  {pattern:<16} {cost:<16} {use}")

    print("\n\n[5] SPLIT-BRAIN PREVENTION CHECKLIST")
    print("─" * 55)
    checklist = [
        "Use fencing tokens — storage layer rejects stale leader writes",
        "Lease TTL must be shorter than 'time before split-brain causes damage'",
        "Leader must renew lease aggressively (renew at TTL/3 interval)",
        "On lease loss → leader must immediately stop writes",
        "Followers should not follow a leader whose token < current max token",
        "Monitor leader churn — frequent re-elections signal instability",
    ]
    for item in checklist:
        print(f"  • {item}")


if __name__ == "__main__":
    demonstrate_leader_election()
