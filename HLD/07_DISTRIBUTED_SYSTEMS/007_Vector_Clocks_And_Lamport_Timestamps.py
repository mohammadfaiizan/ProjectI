"""
VECTOR CLOCKS AND LAMPORT TIMESTAMPS
=======================================

Problem Statement:
In distributed systems, there is no global clock. Wall-clock time can't be used
to determine the order of events across nodes (NTP skew, clock drift).
We need logical time: a way to reason about causality without real clocks.

Lamport Timestamps (1978):
  Each process maintains a counter L.
  Send: increment L, attach L to message.
  Receive: L = max(L_self, L_message) + 1.
  Guarantees: if A → B (A causally precedes B), then L(A) < L(B).
  Does NOT guarantee: L(A) < L(B) implies A → B. Two events may be concurrent.
  Good for: ordering events in a single log. Insufficient for detecting concurrency.

Vector Clocks (Fidge/Mattern, 1988):
  Each process maintains a vector V of size N (one counter per node).
  Send: increment own counter, attach full vector.
  Receive: component-wise max + increment own counter.
  Guarantees: A → B iff V(A) < V(B) component-wise.
  Can detect concurrent events: if neither V(A) < V(B) nor V(B) < V(A).
  Used in: Riak, DynamoDB (version vectors), Amazon Dynamo paper.
  Cost: vector size grows with N nodes.

Dotted Version Vectors:
  Extension of version vectors for distributed key-value stores.
  Handles the case where multiple clients write the same key concurrently.
  Used in: Riak 2.0+.

Hybrid Logical Clocks (HLC):
  Combines physical time (NTP) + logical component.
  Allows queries like "read all events before real time T".
  Used by: CockroachDB, YugabyteDB.

Causal Consistency via Version Vectors:
  Client tracks the version vector of the latest event it observed.
  When issuing a request: sends its version vector as a "causal token".
  Server waits until its state is at least as current as the token before responding.
  Guarantees: Read-your-writes, monotonic reads across servers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import time
import uuid


# ─────────────────────────────────────────────
# LAMPORT CLOCK
# ─────────────────────────────────────────────

class LamportClock:
    def __init__(self, node_id: str):
        self.node_id  = node_id
        self._counter = 0

    def tick(self) -> int:
        self._counter += 1
        return self._counter

    def send(self) -> int:
        """Returns timestamp to attach to outgoing message."""
        return self.tick()

    def receive(self, msg_timestamp: int) -> int:
        self._counter = max(self._counter, msg_timestamp) + 1
        return self._counter

    @property
    def time(self) -> int:
        return self._counter


# ─────────────────────────────────────────────
# VECTOR CLOCK
# ─────────────────────────────────────────────

class VectorClock:
    def __init__(self, node_id: str, all_nodes: List[str]):
        self.node_id = node_id
        self._clocks  = {n: 0 for n in all_nodes}

    def tick(self) -> Dict[str, int]:
        self._clocks[self.node_id] += 1
        return dict(self._clocks)

    def send(self) -> Dict[str, int]:
        return self.tick()

    def receive(self, remote_vector: Dict[str, int]) -> Dict[str, int]:
        """Merge and increment."""
        for node, ts in remote_vector.items():
            self._clocks[node] = max(self._clocks.get(node, 0), ts)
        self._clocks[self.node_id] += 1
        return dict(self._clocks)

    @property
    def vector(self) -> Dict[str, int]:
        return dict(self._clocks)

    @staticmethod
    def happens_before(v1: Dict[str, int], v2: Dict[str, int]) -> bool:
        """v1 → v2 iff v1 ≤ v2 component-wise and v1 ≠ v2."""
        all_keys = set(v1) | set(v2)
        leq = all(v1.get(k, 0) <= v2.get(k, 0) for k in all_keys)
        neq = any(v1.get(k, 0) != v2.get(k, 0) for k in all_keys)
        return leq and neq

    @staticmethod
    def concurrent(v1: Dict[str, int], v2: Dict[str, int]) -> bool:
        """Neither v1 → v2 nor v2 → v1."""
        return (not VectorClock.happens_before(v1, v2) and
                not VectorClock.happens_before(v2, v1))


# ─────────────────────────────────────────────
# CAUSAL EVENT LOG (using vector clocks)
# ─────────────────────────────────────────────

@dataclass
class CausalEvent:
    event_id : str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    node_id  : str   = ""
    action   : str   = ""
    vector   : Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class CausalEventLog:
    """Records events with vector clocks. Can sort by causal order."""

    def __init__(self):
        self._events: List[CausalEvent] = []

    def record(self, event: CausalEvent):
        self._events.append(event)

    def causal_sort(self) -> List[CausalEvent]:
        """Topological sort by causal order (not unique for concurrent events)."""
        events = list(self._events)
        sorted_events: List[CausalEvent] = []
        remaining = set(range(len(events)))

        while remaining:
            for i in remaining:
                # Event i can be placed if no event in remaining causally precedes it
                can_place = True
                for j in remaining:
                    if i == j:
                        continue
                    if VectorClock.happens_before(events[j].vector, events[i].vector):
                        can_place = False
                        break
                if can_place:
                    sorted_events.append(events[i])
                    remaining.discard(i)
                    break
            else:
                # Concurrent events — pick first remaining
                i = min(remaining)
                sorted_events.append(events[i])
                remaining.discard(i)

        return sorted_events

    def find_concurrent_pairs(self) -> List[Tuple[CausalEvent, CausalEvent]]:
        pairs = []
        for i, e1 in enumerate(self._events):
            for j, e2 in enumerate(self._events):
                if i >= j:
                    continue
                if VectorClock.concurrent(e1.vector, e2.vector):
                    pairs.append((e1, e2))
        return pairs


# ─────────────────────────────────────────────
# DYNAMO-STYLE CONFLICT DETECTION
# ─────────────────────────────────────────────

@dataclass
class DynamoObject:
    key      : str
    value    : Any
    vector   : Dict[str, int]   # version vector


class DynamoNode:
    """
    Simulates DynamoDB/Riak style conflict detection using version vectors.
    On concurrent writes → returns conflict (caller must resolve via LWW, CRDT, etc.)
    """

    def __init__(self, node_id: str, all_nodes: List[str]):
        self.node_id  = node_id
        self._store   : Dict[str, List[DynamoObject]] = defaultdict(list)
        self._vc      = VectorClock(node_id, all_nodes)
        self.conflicts_detected = 0

    def put(self, key: str, value: Any, client_context: Dict[str, int]) -> Dict[str, int]:
        """Write with causal context. Returns new vector."""
        new_vector = self._vc.receive(client_context)
        existing   = self._store.get(key, [])

        # Supersede versions dominated by new write
        surviving = []
        for obj in existing:
            if not VectorClock.happens_before(obj.vector, new_vector):
                surviving.append(obj)   # concurrent version — keep as sibling

        surviving.append(DynamoObject(key=key, value=value, vector=dict(new_vector)))
        self._store[key] = surviving

        if len(surviving) > 1:
            self.conflicts_detected += 1

        return dict(new_vector)

    def get(self, key: str) -> List[DynamoObject]:
        """Returns all versions (siblings if conflict exists)."""
        return self._store.get(key, [])

    def is_conflict(self, key: str) -> bool:
        return len(self._store.get(key, [])) > 1


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_vector_clocks():
    print("=" * 65)
    print("VECTOR CLOCKS AND LAMPORT TIMESTAMPS")
    print("=" * 65)

    nodes = ["N1", "N2", "N3"]

    # ── Lamport Timestamps ────────────────────────
    print("\n[1] LAMPORT TIMESTAMPS — ORDERING EVENTS")
    print("─" * 55)

    lc = {n: LamportClock(n) for n in nodes}

    # N1 sends a message to N2
    ts_n1_send = lc["N1"].send()
    print(f"  N1 sends message:    L(N1)={ts_n1_send}")

    ts_n2_recv = lc["N2"].receive(ts_n1_send)
    print(f"  N2 receives message: L(N2)={ts_n2_recv}  (max(0,{ts_n1_send})+1)")

    # N2 then sends to N3
    ts_n2_send = lc["N2"].send()
    ts_n3_recv = lc["N3"].receive(ts_n2_send)
    print(f"  N2 sends to N3:      L(N2)={ts_n2_send}")
    print(f"  N3 receives:         L(N3)={ts_n3_recv}")

    # N1 event (unrelated, concurrent with N3's receive)
    ts_n1_local = lc["N1"].tick()
    print(f"  N1 local event:      L(N1)={ts_n1_local}")

    print(f"\n  Lamport ordering: N1_send({ts_n1_send}) → N2_recv({ts_n2_recv}) "
          f"→ N2_send({ts_n2_send}) → N3_recv({ts_n3_recv})")
    print(f"  N1_local({ts_n1_local}) concurrent with N3_recv — can't tell from timestamps")

    # ── Vector Clocks ─────────────────────────────
    print("\n\n[2] VECTOR CLOCKS — DETECTING CONCURRENCY")
    print("─" * 55)

    vc = {n: VectorClock(n, nodes) for n in nodes}

    # N1 writes (event E1)
    v_e1 = vc["N1"].tick()
    print(f"  E1 (N1 writes):    {v_e1}")

    # N1 sends to N2 (event E2)
    v_e2_send = vc["N1"].send()
    v_e2_recv = vc["N2"].receive(v_e2_send)
    print(f"  E2 (N2 receives):  {v_e2_recv}")

    # N3 writes independently (event E3 — concurrent with E2)
    v_e3 = vc["N3"].tick()
    print(f"  E3 (N3 writes):    {v_e3}  [concurrent with E2]")

    # N2 sends to N3 (event E4)
    v_e4_send = vc["N2"].send()
    v_e4_recv = vc["N3"].receive(v_e4_send)
    print(f"  E4 (N3 receives):  {v_e4_recv}  [E2 → E4]")

    print(f"\n  Causal relationships:")
    pairs = [
        ("E1", v_e1, "E2", v_e2_recv),
        ("E1", v_e1, "E4", v_e4_recv),
        ("E2", v_e2_recv, "E3", v_e3),
        ("E2", v_e2_recv, "E4", v_e4_recv),
        ("E3", v_e3, "E4", v_e4_recv),
    ]
    for n1, v1, n2, v2 in pairs:
        hb  = VectorClock.happens_before(v1, v2)
        con = VectorClock.concurrent(v1, v2)
        rel = "→" if hb else ("∥" if con else "←")
        print(f"    {n1} {rel} {n2}")

    # ── Dynamo Conflict Detection ─────────────────
    print("\n\n[3] DYNAMO-STYLE CONFLICT DETECTION")
    print("─" * 55)

    all_nodes = ["DynN1", "DynN2"]
    dn1 = DynamoNode("DynN1", all_nodes)
    dn2 = DynamoNode("DynN2", all_nodes)

    # Client A writes to N1
    ctx_a = dn1.put("cart", ["item-1"], client_context={})
    print(f"  Client A writes cart to N1: vector={ctx_a}")

    # Client B (concurrently) writes to N2 without knowing about A's write
    ctx_b = dn2.put("cart", ["item-2"], client_context={})
    print(f"  Client B writes cart to N2: vector={ctx_b}")

    # N1 receives N2's write (replication)
    dn1.put("cart", ["item-2"], client_context=ctx_b)

    # Now N1 has two versions (siblings)
    versions = dn1.get("cart")
    print(f"  N1 after replication: {len(versions)} version(s)")
    for i, v in enumerate(versions):
        print(f"    version {i+1}: value={v.value} vector={v.vector}")
    print(f"  Conflict detected: {dn1.is_conflict('cart')}")
    print(f"  → Application must resolve (LWW, CRDT, user merge)")

    # ── Clock Comparison ──────────────────────────
    print("\n\n[4] LOGICAL CLOCK COMPARISON")
    print("─" * 55)
    rows = [
        ("Lamport",    "Counter per node",   "O(1)",   "A→B implies L(A)<L(B)",        "No (concurrent?)"),
        ("Vector",     "Vector per node",    "O(N)",   "A→B iff V(A)<V(B)",            "Yes"),
        ("HLC",        "Physical + logical", "O(1)",   "A→B + wall-clock ordering",    "Partial"),
        ("True Time",  "GPS + atomic clock", "O(1)",   "Bounded uncertainty ≤6ms",     "Yes (Spanner)"),
    ]
    print(f"  {'Type':<12} {'Structure':<22} {'Size':<7} {'Guarantee':<36} {'Detect Concurrent?'}")
    print(f"  {'─'*95}")
    for clock_type, struct, size, guarantee, detect in rows:
        print(f"  {clock_type:<12} {struct:<22} {size:<7} {guarantee:<36} {detect}")


if __name__ == "__main__":
    demonstrate_vector_clocks()
