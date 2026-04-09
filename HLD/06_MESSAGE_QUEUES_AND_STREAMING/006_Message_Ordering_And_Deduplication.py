"""
MESSAGE ORDERING AND DEDUPLICATION
====================================

Problem Statement:
Distributed systems face two fundamental messaging challenges:
1. Ordering: messages may arrive out of order due to retries, network jitter, or parallel consumers.
2. Deduplication: at-least-once delivery means duplicates are inevitable; processing twice causes bugs.

Ordering Guarantees (weakest → strongest):
  No ordering:       Any consumer gets any message in any order. Highest throughput.
  Partition-ordered: Within a partition/queue, messages are ordered by arrival. Kafka default.
  Key-ordered:       Messages with the same key always go to the same partition → ordered per key.
  Global ordering:   All messages across all partitions are ordered. Only with 1 partition or a sequencer.
                     Near-impossible at scale without sacrificing throughput.

Why ordering breaks:
  - Retries: message 2 fails, message 3 succeeds, message 2 retried → 3 before 2.
  - Parallel consumers: two workers process different messages concurrently.
  - Network: packets from different paths arrive out of order.
  - Producer batching: local reorder before sending.

Deduplication Approaches:
  1. Idempotent Consumer: processing the same message twice produces the same result.
     Use: DB upsert (INSERT ON CONFLICT DO NOTHING), conditional updates, SET operations.
     Best approach — requires no tracking infrastructure.

  2. Exactly-Once via Dedup Store: record processed event_ids in a store (Redis/DB).
     On receive: check if already processed; if yes, skip; if no, process + record.
     Problem: check-then-act is not atomic unless done inside a transaction.

  3. Kafka Idempotent Producer: producer assigns sequence numbers per partition.
     Broker deduplicates within a session window (5 retries, ~5 minutes).
     Does NOT cover: same message from different producer sessions.

  4. Transactional Outbox: write event to same DB transaction as state change.
     Outbox table polled → published. Dedup via unique constraint on event_id.

Sequence Numbers & Gap Detection:
  Each producer increments a per-key sequence number.
  Consumer detects: missing (gap), duplicate (seen seq), or out-of-order.
  Buffer out-of-order messages until the gap is filled or a timeout triggers skip.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import time
import uuid
import threading
import random


# ─────────────────────────────────────────────
# ORDERED MESSAGE WITH SEQUENCE NUMBER
# ─────────────────────────────────────────────

@dataclass
class OrderedMessage:
    msg_id      : str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    partition_key: str  = ""
    seq_num     : int   = 0      # per-partition sequence (monotonic, starts at 1)
    payload     : Any   = None
    timestamp   : float = field(default_factory=time.time)
    producer_id : str   = ""


# ─────────────────────────────────────────────
# ORDERED PRODUCER (sequence numbering)
# ─────────────────────────────────────────────

class SequencedProducer:
    """Assigns monotonically increasing sequence numbers per partition key."""

    def __init__(self, producer_id: str):
        self.producer_id = producer_id
        self._seqs: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def produce(self, key: str, payload: Any) -> OrderedMessage:
        with self._lock:
            self._seqs[key] += 1
            seq = self._seqs[key]
        return OrderedMessage(
            partition_key = key,
            seq_num       = seq,
            payload       = payload,
            producer_id   = self.producer_id,
        )


# ─────────────────────────────────────────────
# OUT-OF-ORDER CONSUMER WITH REORDER BUFFER
# ─────────────────────────────────────────────

class ReorderBuffer:
    """
    Buffers out-of-order messages until the expected sequence arrives.
    Delivers messages in strict order. Skips gaps after a timeout.
    """

    def __init__(self, gap_timeout_s: float = 0.5):
        self._next_expected : Dict[str, int] = defaultdict(lambda: 1)
        self._buffer        : Dict[str, Dict[int, OrderedMessage]] = defaultdict(dict)
        self._delivered     : List[OrderedMessage] = []
        self._gaps_skipped  : int = 0
        self._duplicates    : int = 0
        self._out_of_order  : int = 0
        self._lock          = threading.Lock()
        self.gap_timeout_s  = gap_timeout_s

    def receive(self, msg: OrderedMessage):
        with self._lock:
            key = msg.partition_key
            expected = self._next_expected[key]

            if msg.seq_num < expected:
                self._duplicates += 1
                return   # already delivered

            if msg.seq_num == expected:
                self._deliver_in_order(key, msg)
            else:
                # Future message — buffer it
                self._out_of_order += 1
                self._buffer[key][msg.seq_num] = msg

    def _deliver_in_order(self, key: str, msg: OrderedMessage):
        """Deliver msg, then drain any buffered consecutive messages."""
        self._delivered.append(msg)
        self._next_expected[key] = msg.seq_num + 1

        # Drain buffer for consecutive messages
        while True:
            next_seq = self._next_expected[key]
            if next_seq in self._buffer[key]:
                buffered = self._buffer[key].pop(next_seq)
                self._delivered.append(buffered)
                self._next_expected[key] = next_seq + 1
            else:
                break

    def flush_gaps(self, key: str):
        """Skip buffered gap — call after timeout to unblock the stream."""
        with self._lock:
            if not self._buffer[key]:
                return
            min_buffered = min(self._buffer[key].keys())
            skipped = min_buffered - self._next_expected[key]
            self._gaps_skipped += skipped
            self._next_expected[key] = min_buffered
            self._deliver_in_order(key, self._buffer[key].pop(min_buffered))

    @property
    def stats(self) -> Dict:
        return {
            "delivered"   : len(self._delivered),
            "buffered"    : sum(len(v) for v in self._buffer.values()),
            "duplicates"  : self._duplicates,
            "out_of_order": self._out_of_order,
            "gaps_skipped": self._gaps_skipped,
        }


# ─────────────────────────────────────────────
# DEDUPLICATION STORE (Idempotency Key)
# ─────────────────────────────────────────────

class DeduplicationStore:
    """
    Records processed message IDs with TTL.
    Thread-safe check-and-mark using a lock (simulates Redis SET NX EX).
    """

    def __init__(self, ttl_s: float = 60.0):
        self._seen : Dict[str, float] = {}    # msg_id → expiry time
        self._lock = threading.Lock()
        self.duplicate_count = 0
        self.processed_count = 0
        self.ttl_s = ttl_s

    def is_duplicate(self, msg_id: str) -> bool:
        with self._lock:
            now = time.time()
            # Evict expired entries
            expired = [k for k, exp in self._seen.items() if exp < now]
            for k in expired:
                del self._seen[k]

            if msg_id in self._seen:
                self.duplicate_count += 1
                return True
            # Mark as seen
            self._seen[msg_id] = now + self.ttl_s
            self.processed_count += 1
            return False

    @property
    def store_size(self) -> int:
        return len(self._seen)


# ─────────────────────────────────────────────
# IDEMPOTENT CONSUMER (DB Upsert pattern)
# ─────────────────────────────────────────────

class OrderStateStore:
    """Simulates a DB table with upsert (INSERT ON CONFLICT DO NOTHING)."""

    def __init__(self):
        self._orders: Dict[str, Dict] = {}
        self._upserts = 0
        self._no_ops  = 0

    def upsert_if_newer(self, order_id: str, status: str, version: int) -> bool:
        """Returns True if state was updated, False if ignored (idempotent)."""
        existing = self._orders.get(order_id)
        if existing and existing["version"] >= version:
            self._no_ops += 1
            return False
        self._orders[order_id] = {"status": status, "version": version}
        self._upserts += 1
        return True

    @property
    def stats(self) -> Dict:
        return {"total_orders": len(self._orders),
                "upserts": self._upserts, "no_ops": self._no_ops}


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_ordering_dedup():
    print("=" * 65)
    print("MESSAGE ORDERING AND DEDUPLICATION")
    print("=" * 65)

    random.seed(7)

    # ── Reorder Buffer ───────────────────────────
    print("\n[1] OUT-OF-ORDER DELIVERY — REORDER BUFFER")
    print("─" * 55)

    producer = SequencedProducer("svc-a")
    buffer   = ReorderBuffer(gap_timeout_s=0.1)

    # Produce 10 messages for key "user-42"
    messages = [producer.produce("user-42", {"action": f"step-{i}"}) for i in range(1, 11)]

    # Shuffle to simulate out-of-order network delivery
    shuffled = messages[:]
    random.shuffle(shuffled)

    print(f"  Original order  : {[m.seq_num for m in messages]}")
    print(f"  Delivery order  : {[m.seq_num for m in shuffled]}")

    for msg in shuffled:
        buffer.receive(msg)

    delivered_seqs = [m.seq_num for m in buffer._delivered]
    print(f"  Delivered order : {delivered_seqs}")
    print(f"  In order?       : {delivered_seqs == sorted(delivered_seqs)}")
    s = buffer.stats
    print(f"  Stats: delivered={s['delivered']} out-of-order={s['out_of_order']} "
          f"buffered={s['buffered']}")

    # ── Gap detection ────────────────────────────
    print("\n\n[2] GAP DETECTION — SKIP AFTER TIMEOUT")
    print("─" * 55)

    buf2 = ReorderBuffer(gap_timeout_s=0.05)
    prod2 = SequencedProducer("svc-b")
    msgs2 = [prod2.produce("key-X", {"v": i}) for i in range(1, 8)]

    # Deliver all except seq=3 (simulating a dropped message)
    dropped = 3
    for msg in msgs2:
        if msg.seq_num != dropped:
            buf2.receive(msg)

    print(f"  Delivered seqs 1-7, dropped seq={dropped}")
    print(f"  Buffered (waiting for seq {dropped}): {buf2.stats['buffered']}")
    print(f"  Delivered so far: {[m.seq_num for m in buf2._delivered]}")

    # Simulate gap timeout → flush
    buf2.flush_gaps("key-X")
    print(f"  After flush_gaps(): delivered {[m.seq_num for m in buf2._delivered]}")
    print(f"  Gaps skipped: {buf2.stats['gaps_skipped']}")

    # ── Deduplication Store ──────────────────────
    print("\n\n[3] DEDUPLICATION STORE — AT-LEAST-ONCE DELIVERY")
    print("─" * 55)

    dedup    = DeduplicationStore(ttl_s=5.0)
    producer3 = SequencedProducer("svc-c")

    # Simulate 10 messages, each retried once (duplicate)
    original = [producer3.produce("account-99", {"txn": i}) for i in range(10)]
    delivery = original + original[:4]   # first 4 re-delivered (network retry)
    random.shuffle(delivery)

    processed_ids: List[str] = []
    for msg in delivery:
        if not dedup.is_duplicate(msg.msg_id):
            processed_ids.append(msg.msg_id)   # process once

    print(f"  Total deliveries  : {len(delivery)}")
    print(f"  Unique processed  : {dedup.processed_count}")
    print(f"  Duplicates blocked: {dedup.duplicate_count}")
    print(f"  Dedup store size  : {dedup.store_size} entries")

    # ── Idempotent Consumer ──────────────────────
    print("\n\n[4] IDEMPOTENT CONSUMER — DB UPSERT PATTERN")
    print("─" * 55)

    store = OrderStateStore()

    # Simulate order status updates — some duplicated, some out-of-order
    events = [
        ("ORD-001", "placed",    1),
        ("ORD-001", "confirmed", 2),
        ("ORD-002", "placed",    1),
        ("ORD-001", "placed",    1),   # duplicate — same version, ignored
        ("ORD-001", "shipped",   3),
        ("ORD-002", "placed",    1),   # duplicate
        ("ORD-001", "confirmed", 2),   # late duplicate — ignored (lower version)
        ("ORD-002", "confirmed", 2),
    ]

    for order_id, status, ver in events:
        updated = store.upsert_if_newer(order_id, status, ver)
        symbol  = "✓ updated" if updated else "✗ ignored"
        print(f"  {symbol}: order={order_id} status={status} ver={ver}")

    s = store.stats
    print(f"\n  DB state: {s['total_orders']} orders | "
          f"upserts={s['upserts']} no-ops={s['no_ops']}")

    # ── Summary ──────────────────────────────────
    print("\n\n[5] ORDERING & DEDUPLICATION STRATEGIES")
    print("─" * 55)
    rows = [
        ("Kafka key routing",    "Same key → same partition → ordered"),
        ("Sequence numbers",     "Detect gaps, duplicates, out-of-order"),
        ("Reorder buffer",       "Buffer future msgs, drain on in-order arrival"),
        ("Dedup store (Redis)",  "SET NX EX on event_id — skip if exists"),
        ("Idempotent upsert",    "INSERT ON CONFLICT DO NOTHING in DB"),
        ("Transactional outbox", "Publish in same DB txn as state change"),
        ("Kafka idempotent prod","Broker deduplicates within session window"),
    ]
    for strategy, description in rows:
        print(f"  {strategy:<28} {description}")


if __name__ == "__main__":
    demonstrate_ordering_dedup()
