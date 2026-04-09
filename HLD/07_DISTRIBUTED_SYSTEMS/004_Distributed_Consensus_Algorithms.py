"""
DISTRIBUTED CONSENSUS ALGORITHMS
===================================

Problem Statement:
Multiple nodes must agree on a single value even if some nodes fail or messages are lost.
This is the consensus problem: fundamental to leader election, distributed locks,
replicated state machines, and any "source of truth" in a distributed system.

FLP Impossibility (Fischer-Lynch-Paterson, 1985):
  In an async network, it's impossible to guarantee consensus
  if even ONE process can fail. This is a theoretical bound.
  Practical algorithms work around it via partial synchrony or timeouts.

Paxos (Lamport, 1989):
  Foundational consensus algorithm. Notoriously hard to understand.
  Phases:
    Phase 1a (Prepare): Proposer sends Prepare(n) to majority of acceptors.
    Phase 1b (Promise): Acceptor promises not to accept proposals < n; returns
                        any previously accepted value.
    Phase 2a (Accept): Proposer sends Accept(n, value) to majority.
    Phase 2b (Accepted): Acceptors accept if they haven't promised a higher n.
  Value chosen when a majority of acceptors have accepted the same (n, value).
  Problem: liveness (dueling proposers), no leader, no log replication built-in.

Raft (Ongaro & Ousterhout, 2014):
  Designed to be understandable. Used in: etcd, CockroachDB, TiKV, Consul.
  Key ideas:
    Strong leader: only the leader handles client requests and replicates log.
    Log replication: leader appends entry, replicates to majority, then commits.
    Leader election: servers start as followers. On timeout → candidate.
                     Candidate requests votes. Wins if majority vote for it.
    Term: logical clock, monotonically increasing. Used to detect stale leaders.
  Safety guarantee: at most one leader per term. Committed entries never lost.

Multi-Raft / Sharding:
  For scale, partition the key space into many Raft groups (like CockroachDB).
  Each shard runs independent Raft consensus.

ZAB (ZooKeeper Atomic Broadcast):
  Used by ZooKeeper. Similar to Raft but with atomic broadcast semantics.
  Primary-backup model. All writes go through single primary.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import time
import uuid
import threading
import random


# ─────────────────────────────────────────────
# RAFT ROLES AND LOG ENTRY
# ─────────────────────────────────────────────

class RaftRole(Enum):
    FOLLOWER  = "follower"
    CANDIDATE = "candidate"
    LEADER    = "leader"


@dataclass
class LogEntry:
    term    : int
    index   : int
    command : Any


# ─────────────────────────────────────────────
# SIMPLIFIED RAFT NODE
# ─────────────────────────────────────────────

class RaftNode:
    """
    Simplified Raft implementation demonstrating core concepts.
    Not production-grade — focuses on illustrating the algorithm.
    """

    ELECTION_TIMEOUT_MS = (150, 300)   # random in range

    def __init__(self, node_id: str, peers: List[str]):
        self.node_id       = node_id
        self.peers         = peers
        self.role          = RaftRole.FOLLOWER
        self.current_term  = 0
        self.voted_for     : Optional[str] = None
        self.log           : List[LogEntry] = []
        self.commit_index  = -1
        self.last_applied  = -1
        # Leader state
        self.next_index    : Dict[str, int] = {}
        self.match_index   : Dict[str, int] = {}
        # Election
        self._last_heartbeat = time.time()
        self._votes_received : Set[str] = set()
        self._lock           = threading.Lock()
        # Stats
        self.elections_started = 0
        self.terms_seen        : Set[int] = set()

    # ── Leader Election ─────────────────────────

    def start_election(self, cluster: "RaftCluster") -> bool:
        """Start an election. Returns True if won."""
        with self._lock:
            self.current_term += 1
            self.role          = RaftRole.CANDIDATE
            self.voted_for     = self.node_id
            self._votes_received = {self.node_id}
            self.elections_started += 1
            term = self.current_term

        votes = 1
        for peer_id in self.peers:
            peer = cluster.nodes.get(peer_id)
            if peer and cluster.can_communicate(self.node_id, peer_id):
                granted = peer.request_vote(self.node_id, term,
                                             len(self.log) - 1,
                                             self.log[-1].term if self.log else 0)
                if granted:
                    votes += 1

        majority = len(self.peers) // 2 + 1 + 1   # total = peers + self
        # Actually: majority = (len(peers) + 1) // 2 + 1 for total cluster
        majority = (len(self.peers) + 1) // 2 + 1

        if votes >= majority:
            with self._lock:
                if self.current_term == term:
                    self.role = RaftRole.LEADER
                    for p in self.peers:
                        self.next_index[p]  = len(self.log)
                        self.match_index[p] = -1
            return True
        with self._lock:
            self.role = RaftRole.FOLLOWER
        return False

    def request_vote(self, candidate_id: str, term: int,
                     last_log_index: int, last_log_term: int) -> bool:
        with self._lock:
            if term < self.current_term:
                return False
            if term > self.current_term:
                self.current_term = term
                self.voted_for    = None
                self.role         = RaftRole.FOLLOWER

            if self.voted_for and self.voted_for != candidate_id:
                return False

            # Log up-to-date check
            my_last_term  = self.log[-1].term  if self.log else 0
            my_last_index = len(self.log) - 1
            if last_log_term < my_last_term:
                return False
            if last_log_term == my_last_term and last_log_index < my_last_index:
                return False

            self.voted_for = candidate_id
            return True

    # ── Log Replication ─────────────────────────

    def append_entry(self, command: Any, cluster: "RaftCluster") -> bool:
        """Leader appends to log and replicates. Returns True if committed."""
        if self.role != RaftRole.LEADER:
            return False

        with self._lock:
            idx   = len(self.log)
            entry = LogEntry(term=self.current_term, index=idx, command=command)
            self.log.append(entry)

        replicated = 1   # self
        for peer_id in self.peers:
            peer = cluster.nodes.get(peer_id)
            if peer and cluster.can_communicate(self.node_id, peer_id):
                if peer.receive_append_entries(self.node_id, self.current_term,
                                               idx - 1,
                                               self.log[idx - 1].term if idx > 0 else 0,
                                               [entry], self.commit_index):
                    replicated += 1
                    self.match_index[peer_id] = idx
                    self.next_index[peer_id]  = idx + 1

        majority = (len(self.peers) + 1) // 2 + 1
        if replicated >= majority:
            with self._lock:
                self.commit_index = idx
                self.last_applied = idx
            return True
        return False

    def receive_append_entries(self, leader_id: str, term: int,
                                prev_index: int, prev_term: int,
                                entries: List[LogEntry], leader_commit: int) -> bool:
        with self._lock:
            if term < self.current_term:
                return False
            if term > self.current_term:
                self.current_term = term
            self.role            = RaftRole.FOLLOWER
            self._last_heartbeat = time.time()
            self.terms_seen.add(term)

            # Consistency check
            if prev_index >= 0:
                if len(self.log) <= prev_index:
                    return False
                if self.log[prev_index].term != prev_term:
                    return False

            # Append entries
            for entry in entries:
                while len(self.log) <= entry.index:
                    self.log.append(None)
                self.log[entry.index] = entry

            if leader_commit > self.commit_index:
                self.commit_index = min(leader_commit, len(self.log) - 1)
            return True

    def get_committed_log(self) -> List[Any]:
        return [e.command for e in self.log[:self.commit_index + 1] if e]


# ─────────────────────────────────────────────
# RAFT CLUSTER
# ─────────────────────────────────────────────

class RaftCluster:
    def __init__(self, node_ids: List[str]):
        self.nodes      = {nid: RaftNode(nid, [n for n in node_ids if n != nid])
                           for nid in node_ids}
        self._partitioned: Set[str] = set()

    def can_communicate(self, src: str, dst: str) -> bool:
        return src not in self._partitioned and dst not in self._partitioned

    def partition(self, node_id: str):
        self._partitioned.add(node_id)

    def heal(self, node_id: str):
        self._partitioned.discard(node_id)

    def elect_leader(self) -> Optional[str]:
        """Trigger election from first node. Returns winner or None."""
        for node_id, node in self.nodes.items():
            if node.start_election(self):
                return node_id
        return None

    def leader(self) -> Optional["RaftNode"]:
        for node in self.nodes.values():
            if node.role == RaftRole.LEADER:
                return node
        return None


# ─────────────────────────────────────────────
# PAXOS (single-decree, simplified)
# ─────────────────────────────────────────────

class PaxosAcceptor:
    def __init__(self, acceptor_id: str):
        self.acceptor_id = acceptor_id
        self.promised_n  = -1
        self.accepted_n  = -1
        self.accepted_v  = None

    def prepare(self, n: int) -> Optional[Tuple[int, Any]]:
        """Phase 1b: Promise if n > promised_n. Return (accepted_n, accepted_v)."""
        if n > self.promised_n:
            self.promised_n = n
            return (self.accepted_n, self.accepted_v)
        return None   # reject

    def accept(self, n: int, value: Any) -> bool:
        """Phase 2b: Accept if n >= promised_n."""
        if n >= self.promised_n:
            self.promised_n  = n
            self.accepted_n  = n
            self.accepted_v  = value
            return True
        return False


class PaxosProposer:
    def __init__(self, proposer_id: int, acceptors: List[PaxosAcceptor]):
        self.proposer_id = proposer_id
        self.acceptors   = acceptors
        self._proposal_n = proposer_id   # unique starting proposal number

    def propose(self, value: Any) -> Optional[Any]:
        """Run full Paxos. Returns chosen value (may differ from proposed)."""
        n       = self._proposal_n
        majority = len(self.acceptors) // 2 + 1

        # Phase 1: Prepare
        promises = []
        for acc in self.acceptors:
            result = acc.prepare(n)
            if result is not None:
                promises.append(result)

        if len(promises) < majority:
            self._proposal_n += len(self.acceptors)   # increment and retry
            return None   # prepare failed

        # Phase 2: Accept — use highest previously accepted value if any
        prev_accepted = [(n_prev, v) for n_prev, v in promises if v is not None]
        if prev_accepted:
            _, value = max(prev_accepted, key=lambda x: x[0])

        accepts = 0
        for acc in self.acceptors:
            if acc.accept(n, value):
                accepts += 1

        if accepts >= majority:
            return value   # consensus reached
        return None


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_consensus():
    print("=" * 65)
    print("DISTRIBUTED CONSENSUS ALGORITHMS")
    print("=" * 65)

    random.seed(42)

    # ── Raft: Leader Election ─────────────────────
    print("\n[1] RAFT — LEADER ELECTION")
    print("─" * 55)

    cluster = RaftCluster(["N1", "N2", "N3", "N4", "N5"])
    leader_id = cluster.elect_leader()

    print(f"  Cluster: 5 nodes")
    for nid, node in cluster.nodes.items():
        print(f"    {nid}: role={node.role.value} term={node.current_term}")
    print(f"  Elected leader: {leader_id}")

    # ── Raft: Log Replication ─────────────────────
    print("\n\n[2] RAFT — LOG REPLICATION")
    print("─" * 55)

    leader = cluster.leader()
    if leader:
        commands = ["SET x 1", "SET y 2", "SET x 3"]
        for cmd in commands:
            ok = leader.append_entry(cmd, cluster)
            print(f"  Append '{cmd}': committed={ok}")

        print(f"\n  Leader committed log: {leader.get_committed_log()}")
        for nid, node in cluster.nodes.items():
            if node.role == RaftRole.FOLLOWER:
                print(f"    {nid} log: {node.get_committed_log()}")

    # ── Raft: Partition + Re-election ─────────────
    print("\n\n[3] RAFT — LEADER PARTITION → RE-ELECTION")
    print("─" * 55)
    old_leader = leader_id
    cluster.partition(old_leader)   # isolate the leader
    print(f"  Partitioned leader: {old_leader}")

    # Another node should win election
    new_leader_id = None
    for nid in cluster.nodes:
        if nid != old_leader:
            node = cluster.nodes[nid]
            if node.start_election(cluster):
                new_leader_id = nid
                break

    print(f"  New leader elected: {new_leader_id}")
    print(f"  Old leader still thinks it's leader: "
          f"{cluster.nodes[old_leader].role.value}")

    # Heal partition
    cluster.heal(old_leader)
    # On receiving heartbeat with higher term, old leader steps down
    if new_leader_id:
        cluster.nodes[old_leader].receive_append_entries(
            new_leader_id, cluster.nodes[new_leader_id].current_term,
            -1, 0, [], cluster.nodes[new_leader_id].commit_index)
    print(f"  After healing — old leader role: {cluster.nodes[old_leader].role.value}")

    # ── Paxos ─────────────────────────────────────
    print("\n\n[4] PAXOS — SINGLE-DECREE CONSENSUS")
    print("─" * 55)
    acceptors = [PaxosAcceptor(f"A{i}") for i in range(5)]
    proposer1 = PaxosProposer(proposer_id=1, acceptors=acceptors)
    proposer2 = PaxosProposer(proposer_id=2, acceptors=acceptors)

    # First proposer proposes "leader=node-1"
    result1 = proposer1.propose("leader=node-1")
    print(f"  Proposer 1 proposes 'leader=node-1': chosen={result1}")

    # Second proposer (concurrent) tries to propose different value
    result2 = proposer2.propose("leader=node-2")
    print(f"  Proposer 2 proposes 'leader=node-2': chosen={result2}")
    print(f"  → Consensus value: {result1 or result2}")

    # ── Algorithm Comparison ──────────────────────
    print("\n\n[5] PAXOS vs RAFT COMPARISON")
    print("─" * 55)
    rows = [
        ("Understandability",  "Very hard",       "Designed to be simple"),
        ("Leader",             "No fixed leader",  "Strong single leader"),
        ("Log replication",    "Not included",     "Built-in (Raft log)"),
        ("Liveness",           "Dueling proposers","Randomized timeouts"),
        ("Reconfiguration",    "Complex",          "Joint consensus"),
        ("Used in",            "Chubby (Google)",  "etcd, CockroachDB, Consul"),
        ("Fault tolerance",    "f of 2f+1",        "f of 2f+1"),
        ("Election timeout",   "N/A",              "150-300ms random"),
    ]
    print(f"  {'Aspect':<22} {'Paxos':<22} {'Raft'}")
    print(f"  {'─'*65}")
    for aspect, paxos_val, raft_val in rows:
        print(f"  {aspect:<22} {paxos_val:<22} {raft_val}")

    print("\n\n[6] CONSENSUS IN PRACTICE")
    print("─" * 55)
    points = [
        "Consensus requires 2f+1 nodes to tolerate f failures (majority quorum)",
        "Leader election uses randomized timeouts to avoid split votes",
        "Every write goes through leader; replicated to majority before commit",
        "Network partition can prevent progress (CP — FLP impossibility)",
        "Raft log: each entry has (term, index, command); once committed, durable",
        "In practice: use etcd/ZooKeeper; don't implement Paxos/Raft yourself",
    ]
    for point in points:
        print(f"  • {point}")


if __name__ == "__main__":
    demonstrate_consensus()
