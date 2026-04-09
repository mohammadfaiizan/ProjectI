"""
GOSSIP PROTOCOL DESIGN
========================

Problem Statement:
How does a distributed system propagate information (node state, configuration,
failure detection) to all N nodes without a central coordinator?
Broadcasting to all N nodes: O(N) messages. Gossip achieves O(log N) rounds
to reach all nodes with high probability.

How Gossip Works:
  Each node maintains state (key-value map, e.g., "node-3 is alive").
  Every T seconds (gossip interval): each node picks K random peers,
  sends its entire state (or a digest). Peers merge the incoming state.
  Convergence: after O(log N) rounds, all nodes have seen all updates.

Gossip Variants:
  Push-only:    sender pushes its state. Simple. May send redundant data.
  Pull-only:    receiver requests delta since it last synced. More efficient.
  Push-Pull:    sender pushes its state AND requests receiver's state back.
                Most efficient. Converges fastest.

Anti-Entropy vs Rumor Mongering:
  Anti-Entropy:   periodic full state sync to ensure convergence (correctness).
  Rumor Mongering (epidemic): node gossips about new updates until "bored"
                  (e.g., K neighbors already knew). Faster for hot updates.

Gossip in Production:
  - Cassandra: ring membership, token ownership, failure detection.
  - Redis Cluster: cluster topology, slot assignments.
  - DynamoDB: node membership (simplified ring gossip).
  - Consul: health state, service discovery.
  - SWIM Protocol (Scalable Weakly-consistent Infection-style Membership):
    Combines gossip with failure detection via probing (ping/indirect ping).

Convergence Analysis:
  After r rounds of gossip with fanout K:
  Expected fraction infected = 1 - (1 - 1/N)^(K*r) ≈ 1 - e^(-K*r/N)
  After O(log N) rounds: essentially all nodes informed.
  Each node sends K messages/round → total O(KN log N) messages to converge.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import time
import random
import uuid
import math


# ─────────────────────────────────────────────
# GOSSIP STATE ENTRY
# ─────────────────────────────────────────────

@dataclass
class GossipEntry:
    key       : str
    value     : Any
    version   : int          # monotonically increasing
    origin    : str          # which node produced this entry
    timestamp : float = field(default_factory=time.time)


# ─────────────────────────────────────────────
# GOSSIP NODE
# ─────────────────────────────────────────────

class GossipNode:
    """
    Gossip node with push-pull protocol.
    Maintains a state table: key → GossipEntry.
    On gossip round: picks K random peers, exchanges state.
    """

    def __init__(self, node_id: str, fanout: int = 2):
        self.node_id  = node_id
        self.fanout   = fanout
        self._state   : Dict[str, GossipEntry] = {}
        self.rounds   = 0
        self.messages_sent     = 0
        self.messages_received = 0

        # Initialize node's own entry (alive)
        self.set(f"node:{node_id}:status", "alive", origin=node_id)

    def set(self, key: str, value: Any, origin: str = None):
        origin = origin or self.node_id
        existing = self._state.get(key)
        version  = (existing.version + 1) if existing else 1
        self._state[key] = GossipEntry(key=key, value=value,
                                        version=version, origin=origin)

    def get(self, key: str) -> Optional[Any]:
        entry = self._state.get(key)
        return entry.value if entry else None

    def get_digest(self) -> Dict[str, int]:
        """Digest: {key: version} — used for delta gossip."""
        return {k: e.version for k, e in self._state.items()}

    def full_state(self) -> Dict[str, GossipEntry]:
        return dict(self._state)

    def receive_push(self, remote_state: Dict[str, GossipEntry]):
        """Merge incoming state: accept if version is higher."""
        self.messages_received += 1
        for key, remote_entry in remote_state.items():
            existing = self._state.get(key)
            if not existing or remote_entry.version > existing.version:
                self._state[key] = remote_entry

    def gossip_round(self, all_nodes: Dict[str, "GossipNode"]):
        """Pick K random peers. Push-pull."""
        self.rounds += 1
        peers_pool = [n for nid, n in all_nodes.items() if nid != self.node_id]
        peers      = random.sample(peers_pool, min(self.fanout, len(peers_pool)))

        for peer in peers:
            # Push: send our state
            peer.receive_push(self.full_state())
            self.messages_sent += 1
            # Pull: receive peer's state back
            self.receive_push(peer.full_state())
            self.messages_received += 1

    def state_size(self) -> int:
        return len(self._state)

    def known_nodes(self) -> Set[str]:
        return {e.origin for e in self._state.values()}


# ─────────────────────────────────────────────
# GOSSIP CLUSTER SIMULATOR
# ─────────────────────────────────────────────

class GossipCluster:
    def __init__(self, n_nodes: int, fanout: int = 2):
        self.nodes   = {f"N{i}": GossipNode(f"N{i}", fanout=fanout)
                        for i in range(n_nodes)}
        self.fanout  = fanout

    def run_round(self):
        """One gossip round for all nodes (simulated concurrently, randomized order)."""
        node_list = list(self.nodes.values())
        random.shuffle(node_list)
        for node in node_list:
            node.gossip_round(self.nodes)

    def propagate_update(self, origin_node: str, key: str, value: Any) -> int:
        """
        Originate an update at one node. Run rounds until all nodes know.
        Returns number of rounds needed.
        """
        self.nodes[origin_node].set(key, value)
        rounds = 0
        while True:
            rounds += 1
            self.run_round()
            if all(n.get(key) == value for n in self.nodes.values()):
                break
            if rounds > 50:
                break   # safety stop
        return rounds

    def coverage(self, key: str, expected_value: Any) -> float:
        """Fraction of nodes that have the correct value."""
        known = sum(1 for n in self.nodes.values() if n.get(key) == expected_value)
        return known / len(self.nodes)

    def total_messages(self) -> int:
        return sum(n.messages_sent for n in self.nodes.values())


# ─────────────────────────────────────────────
# FAILURE DETECTION VIA GOSSIP (SWIM-like)
# ─────────────────────────────────────────────

class SWIMNode:
    """
    SWIM-style failure detection:
    1. Probing: each node pings a random peer directly.
    2. Indirect ping: if direct ping fails, ask K other nodes to ping.
    3. Gossip: once confirmed dead (no response from indirect), gossip the update.
    """

    def __init__(self, node_id: str, failure_probability: float = 0.0):
        self.node_id            = node_id
        self.failure_probability = failure_probability
        self._alive             = True
        self._member_list       : Dict[str, str] = {}   # node_id → status

    def crash(self):
        self._alive = False

    def ping(self) -> bool:
        """Returns True if alive and reachable."""
        if not self._alive:
            return False
        return random.random() > self.failure_probability

    def probe(self, target: "SWIMNode", cluster: Dict[str, "SWIMNode"],
              indirect_k: int = 2) -> str:
        """
        Probe target. If no response, use indirect ping via K nodes.
        Returns 'alive' or 'suspected'.
        """
        # Direct ping
        if target.ping():
            return "alive"

        # Indirect ping via K random nodes
        indirect_nodes = [n for nid, n in cluster.items()
                          if nid != self.node_id and nid != target.node_id]
        samplers = random.sample(indirect_nodes, min(indirect_k, len(indirect_nodes)))
        for node in samplers:
            if node.ping() and target.ping():
                return "alive"   # alive through indirect

        # Mark as suspected
        return "suspected"

    def update_member_list(self, node_id: str, status: str):
        self._member_list[node_id] = status

    @property
    def is_alive(self) -> bool:
        return self._alive


# ─────────────────────────────────────────────
# CONVERGENCE ANALYSIS
# ─────────────────────────────────────────────

def theoretical_convergence(n: int, fanout: int, rounds: int) -> float:
    """Expected fraction of nodes informed after `rounds` gossip rounds."""
    p_not_infected_per_round = (1 - 1/n) ** fanout
    p_infected = 1 - p_not_infected_per_round ** rounds
    return min(p_infected, 1.0)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_gossip():
    print("=" * 65)
    print("GOSSIP PROTOCOL DESIGN")
    print("=" * 65)

    random.seed(42)

    # ── Basic Gossip Propagation ──────────────────
    print("\n[1] GOSSIP PROPAGATION — 20 NODES, FANOUT=2")
    print("─" * 55)

    cluster = GossipCluster(n_nodes=20, fanout=2)
    rounds  = cluster.propagate_update("N0", "config:feature_flag", "enabled")

    print(f"  Cluster: 20 nodes, fanout=2")
    print(f"  Update originated at N0")
    print(f"  Rounds to full convergence: {rounds}")
    print(f"  Total messages sent: {cluster.total_messages()}")
    print(f"  Theoretical: ~{math.ceil(math.log(20)/math.log(2))} rounds for log₂(20)≈4.3")

    # ── Round-by-round coverage ───────────────────
    print("\n\n[2] ROUND-BY-ROUND COVERAGE")
    print("─" * 55)

    cluster2 = GossipCluster(n_nodes=50, fanout=3)
    cluster2.nodes["N0"].set("alert", "high-cpu")

    print(f"  Cluster: 50 nodes, fanout=3")
    print(f"  {'Round':<8} {'Coverage':<12} {'Theoretical'}")
    for r in range(1, 9):
        cluster2.run_round()
        coverage = cluster2.coverage("alert", "high-cpu")
        theoretical = theoretical_convergence(50, 3, r)
        bar = "█" * int(coverage * 20)
        print(f"  {r:<8} {coverage*100:>6.1f}%  {bar:<22} "
              f"(theory: {theoretical*100:.1f}%)")
        if coverage >= 1.0:
            break

    # ── Multiple Updates ──────────────────────────
    print("\n\n[3] GOSSIP CONVERGENCE WITH MULTIPLE UPDATES")
    print("─" * 55)

    cluster3 = GossipCluster(n_nodes=10, fanout=2)

    # Different nodes originate different updates
    updates = [
        ("N0", "node:N3:status", "down"),
        ("N5", "config:timeout",  "30s"),
        ("N9", "node:N7:status", "rejoined"),
    ]
    for origin, key, value in updates:
        cluster3.nodes[origin].set(key, value)

    for _ in range(5):
        cluster3.run_round()

    print(f"  After 5 rounds (10 nodes, fanout=2):")
    for origin, key, value in updates:
        coverage = cluster3.coverage(key, value)
        print(f"    '{key}={value}' coverage: {coverage*100:.0f}%")

    # ── SWIM Failure Detection ────────────────────
    print("\n\n[4] SWIM FAILURE DETECTION")
    print("─" * 55)

    swim_nodes = {f"N{i}": SWIMNode(f"N{i}") for i in range(5)}

    # Crash N3
    swim_nodes["N3"].crash()
    print(f"  N3 crashed. N0 probing N3:")

    status = swim_nodes["N0"].probe(swim_nodes["N3"], swim_nodes, indirect_k=2)
    print(f"  Result: {status}")

    # Gossip the failure to all nodes
    for node in swim_nodes.values():
        if node.is_alive:
            node.update_member_list("N3", status)

    known_dead = sum(1 for n in swim_nodes.values()
                     if n.is_alive and n._member_list.get("N3") == "suspected")
    print(f"  Nodes that marked N3 as suspected: {known_dead}/{len(swim_nodes)-1}")

    # ── Gossip Properties Summary ─────────────────
    print("\n\n[5] GOSSIP PROTOCOL PROPERTIES")
    print("─" * 55)
    rows = [
        ("Convergence",    f"O(log N) rounds with fanout K"),
        ("Messages/round", f"O(K*N) total per round"),
        ("Fault tolerance", "No SPOF — any node can originate updates"),
        ("Consistency",    "Eventual — nodes converge over time"),
        ("Node failure",   "Detected via SWIM probing + gossip"),
        ("Network usage",  "Full state push: O(N²) | Delta sync: O(N log N)"),
        ("Duplicate msgs", "Nodes may receive same update many times (idempotent)"),
    ]
    for prop, value in rows:
        print(f"  {prop:<20} {value}")

    print("\n\n[6] GOSSIP USE CASES IN PRODUCTION")
    print("─" * 55)
    uses = [
        ("Cassandra",    "Ring membership, token ownership, failure detection"),
        ("Redis Cluster","Cluster topology, slot assignments, node health"),
        ("Consul",       "Service health, KV store propagation"),
        ("SWIM",         "Membership protocol: ping + indirect ping + gossip"),
        ("Riak",         "Ring state, node join/leave propagation"),
    ]
    for system, use in uses:
        print(f"  {system:<16} {use}")


if __name__ == "__main__":
    demonstrate_gossip()
