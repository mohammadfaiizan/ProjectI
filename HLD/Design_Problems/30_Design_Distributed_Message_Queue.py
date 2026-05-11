"""
Problem 30: Design a Distributed Message Queue
================================================
Working simulation of a Kafka-like distributed message queue with:
- Partition: append-only log with WAL-like structure and offset index
- Broker: manages multiple partitions, handles produce/fetch
- ConsumerGroup: partition assignment and per-partition offset tracking
- PartitionLeaderElector: consistent hash-based leader assignment
- MessageProducer: at-least-once delivery with retry
- IdempotentProducer: sequence numbers to detect and drop duplicates
- DeadLetterQueue: failed messages after max retries
- DistributedMQ: top-level orchestrator
"""

import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class DeliverySemantics(Enum):
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Message:
    offset: int
    key: Optional[bytes]
    value: bytes
    headers: dict[str, str]
    timestamp: float
    producer_id: Optional[str] = None
    sequence_num: Optional[int] = None  # For idempotent producer dedup

    def __repr__(self) -> str:
        key_str = self.key.decode("utf-8", errors="replace") if self.key else "None"
        val_str = self.value.decode("utf-8", errors="replace")[:40]
        return f"Message(offset={self.offset}, key={key_str}, value={val_str}...)"


@dataclass
class ProduceResult:
    topic: str
    partition: int
    offset: int
    timestamp: float
    success: bool
    error: str = ""


@dataclass
class ConsumeResult:
    messages: list[Message]
    partition: int
    committed_offset: int


# ─── Partition (Append-Only Log) ─────────────────────────────────────────────

class Partition:
    """
    Append-only log for a single partition.
    Maintains a sparse offset index for O(log n) offset lookups.
    Messages never overwritten — only appended or deleted by retention policy.
    """

    INDEX_INTERVAL = 10  # Index every N messages for sparse indexing

    def __init__(self, topic: str, partition_id: int,
                 retention_ms: int = 7 * 24 * 3600 * 1000):
        self.topic = topic
        self.partition_id = partition_id
        self.retention_ms = retention_ms
        self._log: list[Message] = []           # The actual append-only log
        self._offset_index: dict[int, int] = {} # offset → list index (sparse)
        self._base_offset: int = 0              # Offset of first message in log
        self._high_watermark: int = 0           # Last fully replicated offset
        self._next_offset: int = 0

    def append(self, key: Optional[bytes], value: bytes,
               headers: dict, timestamp: float,
               producer_id: Optional[str] = None,
               sequence_num: Optional[int] = None) -> int:
        """Append a message to the log. Returns assigned offset. O(1) amortized."""
        offset = self._next_offset
        msg = Message(offset=offset, key=key, value=value, headers=headers,
                      timestamp=timestamp, producer_id=producer_id, sequence_num=sequence_num)
        list_idx = len(self._log)
        self._log.append(msg)

        # Sparse index: record every INDEX_INTERVAL messages
        if list_idx % self.INDEX_INTERVAL == 0:
            self._offset_index[offset] = list_idx

        self._next_offset += 1
        self._high_watermark = self._next_offset  # In simulation, no replication lag
        return offset

    def fetch(self, start_offset: int, max_messages: int = 100) -> list[Message]:
        """
        Fetch messages starting from start_offset.
        Uses sparse index for O(log n) initial seek, then sequential scan.
        """
        if start_offset >= self._next_offset or not self._log:
            return []

        # Find nearest index entry at or before start_offset
        list_idx = self._seek_to_offset(start_offset)

        results = []
        for i in range(list_idx, len(self._log)):
            msg = self._log[i]
            if msg.offset >= start_offset:
                results.append(msg)
                if len(results) >= max_messages:
                    break
        return results

    def _seek_to_offset(self, target_offset: int) -> int:
        """Binary search in sparse index to find starting list index. O(log n)."""
        sorted_indexed_offsets = sorted(self._offset_index.keys())
        lo, hi = 0, len(sorted_indexed_offsets) - 1
        best_idx = 0

        while lo <= hi:
            mid = (lo + hi) // 2
            indexed_offset = sorted_indexed_offsets[mid]
            if indexed_offset <= target_offset:
                best_idx = self._offset_index[indexed_offset]
                lo = mid + 1
            else:
                hi = mid - 1
        return best_idx

    def get_high_watermark(self) -> int:
        return self._high_watermark

    def get_earliest_offset(self) -> int:
        return self._log[0].offset if self._log else 0

    def size(self) -> int:
        return len(self._log)

    def apply_retention(self, now: float) -> int:
        """Delete messages older than retention_ms. Returns count deleted."""
        cutoff_ms = now * 1000 - self.retention_ms
        cutoff_ts = cutoff_ms / 1000
        original_size = len(self._log)
        self._log = [m for m in self._log if m.timestamp >= cutoff_ts]
        # Rebuild index
        self._offset_index = {}
        for i, msg in enumerate(self._log):
            if i % self.INDEX_INTERVAL == 0:
                self._offset_index[msg.offset] = i
        if self._log:
            self._base_offset = self._log[0].offset
        return original_size - len(self._log)


# ─── Broker ───────────────────────────────────────────────────────────────────

class Broker:
    """
    A single broker managing multiple topic-partitions.
    In production each broker holds a subset of all partitions
    and acts as leader for some and follower for others.
    """

    def __init__(self, broker_id: int):
        self.broker_id = broker_id
        self._partitions: dict[tuple[str, int], Partition] = {}  # (topic, part_id) → Partition
        self._isr: dict[tuple[str, int], list[int]] = {}        # (topic, part_id) → [broker_ids]
        self._produce_count = 0
        self._fetch_count = 0

    def create_partition(self, topic: str, partition_id: int,
                         retention_ms: int = 7 * 24 * 3600 * 1000) -> Partition:
        key = (topic, partition_id)
        if key not in self._partitions:
            self._partitions[key] = Partition(topic, partition_id, retention_ms)
            self._isr[key] = [self.broker_id]
        return self._partitions[key]

    def produce(self, topic: str, partition_id: int, key: Optional[bytes],
                value: bytes, headers: dict = None, timestamp: float = None,
                producer_id: str = None, sequence_num: int = None) -> ProduceResult:
        key_t = (topic, partition_id)
        if key_t not in self._partitions:
            return ProduceResult(topic, partition_id, -1, 0, False, "Partition not found")

        ts = timestamp or time.time()
        offset = self._partitions[key_t].append(
            key, value, headers or {}, ts, producer_id, sequence_num
        )
        self._produce_count += 1
        return ProduceResult(topic, partition_id, offset, ts, True)

    def fetch(self, topic: str, partition_id: int,
              start_offset: int, max_messages: int = 100) -> list[Message]:
        key_t = (topic, partition_id)
        if key_t not in self._partitions:
            return []
        self._fetch_count += 1
        return self._partitions[key_t].fetch(start_offset, max_messages)

    def get_high_watermark(self, topic: str, partition_id: int) -> int:
        key_t = (topic, partition_id)
        if key_t not in self._partitions:
            return 0
        return self._partitions[key_t].get_high_watermark()

    def get_partition(self, topic: str, partition_id: int) -> Optional[Partition]:
        return self._partitions.get((topic, partition_id))

    def stats(self) -> dict:
        return {
            "broker_id": self.broker_id,
            "partitions": len(self._partitions),
            "total_produced": self._produce_count,
            "total_fetched": self._fetch_count,
            "partition_sizes": {f"{t}:{p}": part.size()
                                for (t, p), part in self._partitions.items()}
        }


# ─── Partition Leader Elector ─────────────────────────────────────────────────

class PartitionLeaderElector:
    """
    Simple consistent hash-based partition leader assignment.
    In production: KRaft (Raft-based) controller handles election.
    """

    def __init__(self, brokers: list[Broker]):
        self.brokers = brokers
        self._leaders: dict[tuple[str, int], int] = {}  # (topic, part) → broker_id

    def elect_leader(self, topic: str, partition_id: int) -> Broker:
        """Assign partition to broker via consistent hash."""
        key = f"{topic}:{partition_id}"
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        broker_idx = h % len(self.brokers)
        self._leaders[(topic, partition_id)] = self.brokers[broker_idx].broker_id
        return self.brokers[broker_idx]

    def get_leader(self, topic: str, partition_id: int) -> Optional[Broker]:
        broker_id = self._leaders.get((topic, partition_id))
        if broker_id is None:
            return None
        return next((b for b in self.brokers if b.broker_id == broker_id), None)

    def handle_broker_failure(self, failed_broker_id: int) -> list[tuple[str, int]]:
        """Reassign partitions led by the failed broker to remaining brokers."""
        reassigned = []
        active_brokers = [b for b in self.brokers if b.broker_id != failed_broker_id]
        if not active_brokers:
            return reassigned

        for (topic, part_id), leader_id in list(self._leaders.items()):
            if leader_id == failed_broker_id:
                # Pick new leader (next in hash ring)
                h = int(hashlib.md5(f"{topic}:{part_id}:failover".encode()).hexdigest(), 16)
                new_broker = active_brokers[h % len(active_brokers)]
                self._leaders[(topic, part_id)] = new_broker.broker_id
                reassigned.append((topic, part_id))
        return reassigned


# ─── Consumer Group ───────────────────────────────────────────────────────────

class ConsumerGroup:
    """
    Manages partition assignment and offset tracking for a consumer group.
    Each partition assigned to exactly one consumer in the group at a time.
    """

    def __init__(self, group_id: str):
        self.group_id = group_id
        self._members: list[str] = []                         # consumer_ids
        self._assignment: dict[str, list[tuple[str, int]]] = {}  # consumer_id → [(topic, part)]
        self._offsets: dict[tuple[str, int], int] = {}        # (topic, part) → committed_offset
        self._generation: int = 0

    def join(self, consumer_id: str) -> None:
        if consumer_id not in self._members:
            self._members.append(consumer_id)
            self._rebalance()

    def leave(self, consumer_id: str) -> None:
        if consumer_id in self._members:
            self._members.remove(consumer_id)
            self._rebalance()

    def _rebalance(self) -> None:
        """Round-robin partition assignment across active consumers."""
        self._generation += 1
        self._assignment = {m: [] for m in self._members}
        all_partitions = list(self._offsets.keys()) + [
            p for p in self._pending_partitions if p not in self._offsets
        ]

        if not self._members or not all_partitions:
            return

        for i, partition in enumerate(all_partitions):
            consumer = self._members[i % len(self._members)]
            self._assignment[consumer].append(partition)

    def register_partitions(self, topic: str, partition_ids: list[int]) -> None:
        """Register topic-partitions to be assigned to this group."""
        for pid in partition_ids:
            key = (topic, pid)
            if key not in self._offsets:
                self._offsets[key] = 0  # Start from beginning
        self._rebalance()

    @property
    def _pending_partitions(self) -> list[tuple[str, int]]:
        return list(self._offsets.keys())

    def get_assignment(self, consumer_id: str) -> list[tuple[str, int]]:
        return self._assignment.get(consumer_id, [])

    def commit_offset(self, topic: str, partition_id: int, offset: int) -> None:
        self._offsets[(topic, partition_id)] = offset

    def get_offset(self, topic: str, partition_id: int) -> int:
        return self._offsets.get((topic, partition_id), 0)

    def get_lag(self, topic: str, partition_id: int, high_watermark: int) -> int:
        committed = self.get_offset(topic, partition_id)
        return max(0, high_watermark - committed)

    def seek_to_beginning(self, topic: str, partition_id: int) -> None:
        self._offsets[(topic, partition_id)] = 0

    def seek_to_offset(self, topic: str, partition_id: int, offset: int) -> None:
        self._offsets[(topic, partition_id)] = offset

    def total_lag(self, mq: 'DistributedMQ') -> dict:
        lag_info = {}
        for (topic, pid), offset in self._offsets.items():
            hwm = mq.get_high_watermark(topic, pid)
            lag_info[f"{topic}:{pid}"] = max(0, hwm - offset)
        return lag_info


# ─── Dead Letter Queue ────────────────────────────────────────────────────────

class DeadLetterQueue:
    """Store for messages that failed processing after max retries."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._queue: list[dict] = []
        self._retry_counts: dict[str, int] = defaultdict(int)  # message_id → retry_count

    def record_failure(self, message: Message, error: str, message_id: str) -> bool:
        """
        Record a processing failure.
        Returns True if message sent to DLQ (max retries exceeded).
        """
        self._retry_counts[message_id] += 1
        if self._retry_counts[message_id] >= self.max_retries:
            self._queue.append({
                "message": message,
                "error": error,
                "retry_count": self._retry_counts[message_id],
                "dlq_timestamp": time.time(),
                "message_id": message_id
            })
            return True
        return False

    def get_dlq_messages(self) -> list[dict]:
        return list(self._queue)

    def size(self) -> int:
        return len(self._queue)


# ─── Message Producer ─────────────────────────────────────────────────────────

class MessageProducer:
    """At-least-once producer with retry on failure."""

    def __init__(self, mq: 'DistributedMQ', max_retries: int = 3,
                 semantics: DeliverySemantics = DeliverySemantics.AT_LEAST_ONCE):
        self.mq = mq
        self.max_retries = max_retries
        self.semantics = semantics
        self._producer_id = str(uuid.uuid4())[:8]
        self._success_count = 0
        self._fail_count = 0

    def produce(self, topic: str, value: bytes, key: Optional[bytes] = None,
                headers: dict = None) -> Optional[ProduceResult]:
        """Produce a message with retry."""
        for attempt in range(self.max_retries + 1):
            result = self.mq.produce(topic, value, key, headers or {})
            if result and result.success:
                self._success_count += 1
                return result
            if attempt < self.max_retries:
                # Exponential backoff (simulated)
                pass
        self._fail_count += 1
        return None

    def produce_batch(self, topic: str, messages: list[tuple[bytes, Optional[bytes]]]) -> list[ProduceResult]:
        """Produce a batch of (value, key) tuples."""
        results = []
        for value, key in messages:
            result = self.produce(topic, value, key)
            if result:
                results.append(result)
        return results

    def stats(self) -> dict:
        return {"producer_id": self._producer_id, "success": self._success_count,
                "failed": self._fail_count, "semantics": self.semantics.value}


# ─── Idempotent Producer ──────────────────────────────────────────────────────

class IdempotentProducer:
    """
    Exactly-once producer using sequence numbers per (producer_id, partition).
    Broker deduplicates retried messages based on (producer_id, sequence_num).
    """

    def __init__(self, mq: 'DistributedMQ'):
        self.mq = mq
        self._producer_id = str(uuid.uuid4())[:8]
        self._sequence_nums: dict[tuple[str, int], int] = defaultdict(int)
        self._acked: dict[tuple[str, int, int], bool] = {}  # (topic, part, seq) → acked
        self._success = 0
        self._duplicates_dropped = 0

    def produce(self, topic: str, value: bytes, key: Optional[bytes] = None) -> Optional[ProduceResult]:
        """Produce with idempotency guarantee."""
        # Determine target partition
        part_id = self.mq._route_to_partition(topic, key)
        seq = self._sequence_nums[(topic, part_id)]
        ack_key = (topic, part_id, seq)

        # Simulate: if already acked, don't re-send
        if ack_key in self._acked:
            self._duplicates_dropped += 1
            return ProduceResult(topic, part_id, -1, time.time(), True)  # Idempotent success

        result = self.mq.produce(topic, value, key, {},
                                  producer_id=self._producer_id, sequence_num=seq)
        if result and result.success:
            self._acked[ack_key] = True
            self._sequence_nums[(topic, part_id)] += 1
            self._success += 1
        return result

    def stats(self) -> dict:
        return {"producer_id": self._producer_id, "success": self._success,
                "duplicates_dropped": self._duplicates_dropped}


# ─── Distributed Message Queue (Orchestrator) ────────────────────────────────

class DistributedMQ:
    """Top-level distributed message queue, orchestrating all components."""

    def __init__(self, num_brokers: int = 3):
        self._brokers = [Broker(i) for i in range(num_brokers)]
        self._leader_elector = PartitionLeaderElector(self._brokers)
        self._topics: dict[str, int] = {}           # topic → num_partitions
        self._consumer_groups: dict[str, ConsumerGroup] = {}
        self._dlq = DeadLetterQueue()
        self._seen_sequences: dict[tuple[str, str, int], int] = {}  # (pid, topic, part) → last_seq

    def create_topic(self, topic: str, num_partitions: int = 3,
                     replication_factor: int = 2,
                     retention_ms: int = 7 * 24 * 3600 * 1000) -> None:
        """Create a topic with N partitions distributed across brokers."""
        if topic in self._topics:
            print(f"  Topic '{topic}' already exists")
            return
        self._topics[topic] = num_partitions
        for pid in range(num_partitions):
            leader = self._leader_elector.elect_leader(topic, pid)
            leader.create_partition(topic, pid, retention_ms)
        print(f"  Created topic '{topic}' with {num_partitions} partitions "
              f"(RF={replication_factor}) across {num_partitions} brokers")

    def produce(self, topic: str, value: bytes, key: Optional[bytes] = None,
                headers: dict = None, producer_id: str = None,
                sequence_num: int = None) -> Optional[ProduceResult]:
        """Route message to appropriate partition and produce."""
        if topic not in self._topics:
            print(f"  [ERROR] Topic '{topic}' does not exist")
            return None

        partition_id = self._route_to_partition(topic, key)
        broker = self._leader_elector.get_leader(topic, partition_id)
        if not broker:
            return None

        # Idempotent dedup check: drop duplicate sequences
        if producer_id and sequence_num is not None:
            dedup_key = (producer_id, topic, partition_id)
            last_seq = self._seen_sequences.get(dedup_key, -1)
            if sequence_num <= last_seq:
                return ProduceResult(topic, partition_id, -1, time.time(), True)  # Deduped
            self._seen_sequences[dedup_key] = sequence_num

        return broker.produce(topic, partition_id, key, value,
                              headers or {}, time.time(), producer_id, sequence_num)

    def consume(self, topic: str, consumer_id: str, group_id: str,
                max_messages: int = 10) -> list[ConsumeResult]:
        """Consume messages for assigned partitions."""
        group = self._consumer_groups.get(group_id)
        if not group:
            group = self._get_or_create_group(group_id, topic)
            group.join(consumer_id)

        results = []
        for (t, pid) in group.get_assignment(consumer_id):
            if t != topic:
                continue
            offset = group.get_offset(t, pid)
            broker = self._leader_elector.get_leader(t, pid)
            if not broker:
                continue
            messages = broker.fetch(t, pid, offset, max_messages)
            if messages:
                results.append(ConsumeResult(messages, pid, offset))
        return results

    def commit_offset(self, topic: str, group_id: str, partition_id: int, offset: int) -> None:
        group = self._consumer_groups.get(group_id)
        if group:
            group.commit_offset(topic, partition_id, offset)

    def seek_to_beginning(self, topic: str, group_id: str) -> None:
        group = self._consumer_groups.get(group_id)
        if group:
            for pid in range(self._topics.get(topic, 0)):
                group.seek_to_beginning(topic, pid)
        print(f"  Group '{group_id}' seeked to beginning for topic '{topic}'")

    def get_high_watermark(self, topic: str, partition_id: int) -> int:
        broker = self._leader_elector.get_leader(topic, partition_id)
        return broker.get_high_watermark(topic, partition_id) if broker else 0

    def get_consumer_lag(self, group_id: str, topic: str) -> dict:
        group = self._consumer_groups.get(group_id)
        if not group:
            return {}
        return {k: v for k, v in group.total_lag(self).items() if topic in k}

    def simulate_broker_failure(self, broker_id: int) -> list:
        """Simulate broker failure and leader re-election."""
        reassigned = self._leader_elector.handle_broker_failure(broker_id)
        print(f"  [FAILOVER] Broker {broker_id} failed; reassigned {len(reassigned)} partitions")
        return reassigned

    def _route_to_partition(self, topic: str, key: Optional[bytes]) -> int:
        """Key-based routing: same key → same partition. Round-robin if no key."""
        num_partitions = self._topics.get(topic, 1)
        if key is None:
            return int(time.time() * 1000) % num_partitions
        h = int(hashlib.md5(key).hexdigest(), 16)
        return h % num_partitions

    def _get_or_create_group(self, group_id: str, topic: str) -> ConsumerGroup:
        if group_id not in self._consumer_groups:
            group = ConsumerGroup(group_id)
            num_partitions = self._topics.get(topic, 1)
            group.register_partitions(topic, list(range(num_partitions)))
            self._consumer_groups[group_id] = group
        return self._consumer_groups[group_id]

    def print_stats(self) -> None:
        print(f"\n--- Cluster Statistics ---")
        for broker in self._brokers:
            s = broker.stats()
            print(f"  Broker {s['broker_id']}: {s['partitions']} partitions, "
                  f"produced={s['total_produced']}, fetched={s['total_fetched']}")
            for part_key, size in s['partition_sizes'].items():
                print(f"    Partition {part_key}: {size} messages")
        print(f"  DLQ size: {self._dlq.size()}")


# ─── Demo / Simulation ────────────────────────────────────────────────────────

def run_simulation():
    print("=" * 65)
    print("DISTRIBUTED MESSAGE QUEUE SIMULATION")
    print("=" * 65)

    mq = DistributedMQ(num_brokers=3)

    # ── Create topics ─────────────────────────────────────────
    print("\n--- Creating Topics ---")
    mq.create_topic("user-events",   num_partitions=3, replication_factor=2)
    mq.create_topic("order-events",  num_partitions=2, replication_factor=3)
    mq.create_topic("audit-log",     num_partitions=1, replication_factor=3)

    # ── Standard Producer (at-least-once) ─────────────────────
    print("\n--- At-Least-Once Producer ---")
    producer = MessageProducer(mq, semantics=DeliverySemantics.AT_LEAST_ONCE)
    events = [
        (b'{"user_id": 1, "action": "login"}',    b"user-1"),
        (b'{"user_id": 2, "action": "purchase"}',  b"user-2"),
        (b'{"user_id": 1, "action": "view"}',      b"user-1"),
        (b'{"user_id": 3, "action": "signup"}',    b"user-3"),
        (b'{"user_id": 2, "action": "logout"}',    b"user-2"),
    ]
    for value, key in events:
        result = producer.produce("user-events", value, key)
        if result:
            print(f"  Produced offset={result.offset} partition={result.partition} "
                  f"key={key.decode()}")

    # ── Idempotent Producer ───────────────────────────────────
    print("\n--- Idempotent Producer (Exactly-Once) ---")
    idem_producer = IdempotentProducer(mq)
    orders = [
        (b'{"order_id": "o1", "amount": 99.99}',   b"order-1"),
        (b'{"order_id": "o2", "amount": 149.50}',  b"order-2"),
        (b'{"order_id": "o1", "amount": 99.99}',   b"order-1"),  # Retry/duplicate
    ]
    for i, (value, key) in enumerate(orders):
        result = idem_producer.produce("order-events", value, key)
        if result:
            is_dup = result.offset == -1
            print(f"  Produce {i+1}: offset={result.offset} | "
                  f"{'DUPLICATE DROPPED' if is_dup else 'OK'}")
    print(f"  Idempotent stats: {idem_producer.stats()}")

    # ── Consumer Group ────────────────────────────────────────
    print("\n--- Consumer Group: billing-service ---")
    group_id = "billing-service"

    # Simulate 2 consumers in the group
    for consumer_id in ["consumer-1", "consumer-2"]:
        mq._get_or_create_group(group_id, "user-events").join(consumer_id)

    group = mq._consumer_groups[group_id]
    print(f"  Assignment: {group._assignment}")

    for consumer_id in ["consumer-1", "consumer-2"]:
        results = mq.consume("user-events", consumer_id, group_id, max_messages=5)
        total_msgs = sum(len(r.messages) for r in results)
        print(f"\n  {consumer_id} fetched {total_msgs} messages:")
        for cr in results:
            for msg in cr.messages:
                print(f"    Partition {cr.partition} offset={msg.offset}: "
                      f"{msg.value.decode()[:50]}")
            # Commit offset after processing
            if cr.messages:
                new_offset = cr.messages[-1].offset + 1
                mq.commit_offset("user-events", group_id, cr.partition, new_offset)

    # ── Consumer Lag ──────────────────────────────────────────
    print("\n--- Consumer Lag Monitoring ---")
    # Produce more messages without consuming
    for i in range(5):
        producer.produce("user-events", f'{{"user_id": {i+10}, "action": "browse"}}'.encode())
    lag = mq.get_consumer_lag(group_id, "user-events")
    print(f"  Lag for group '{group_id}':")
    for part, lag_val in lag.items():
        print(f"    {part}: {lag_val} messages behind")

    # ── Seek to Beginning (Replay) ────────────────────────────
    print("\n--- Replay: Seek to Beginning ---")
    mq.seek_to_beginning("user-events", group_id)
    replay_results = mq.consume("user-events", "consumer-1", group_id, max_messages=20)
    total_replayed = sum(len(r.messages) for r in replay_results)
    print(f"  Replayed {total_replayed} messages from beginning")

    # ── Dead Letter Queue Demo ────────────────────────────────
    print("\n--- Dead Letter Queue Simulation ---")
    dlq = DeadLetterQueue(max_retries=3)
    test_msg = Message(offset=42, key=b"user-99",
                       value=b'{"corrupt": true}', headers={}, timestamp=time.time())
    msg_id = "test-msg-42"
    for attempt in range(4):
        sent_to_dlq = dlq.record_failure(test_msg, f"ProcessingError attempt {attempt+1}", msg_id)
        print(f"  Attempt {attempt+1}: sent_to_dlq={sent_to_dlq}")
    print(f"  DLQ contents:")
    for entry in dlq.get_dlq_messages():
        print(f"    msg_id={entry['message_id']} retries={entry['retry_count']} error={entry['error']}")

    # ── Broker Failure & Failover ─────────────────────────────
    print("\n--- Broker Failure Simulation ---")
    reassigned = mq.simulate_broker_failure(0)
    print(f"  Partitions reassigned: {reassigned}")
    # Verify we can still produce after failover
    result = producer.produce("user-events", b'{"after_failover": true}', b"user-1")
    print(f"  Post-failover produce: success={result.success if result else False}")

    # ── Partition Offset Index Demo ───────────────────────────
    print("\n--- Partition Log & Offset Index Demo ---")
    broker = mq._brokers[0]
    # Find a partition this broker is responsible for
    for (topic, pid), partition in broker._partitions.items():
        if partition.size() > 0:
            print(f"  Partition {topic}:{pid}")
            print(f"    Total messages : {partition.size()}")
            print(f"    High watermark : {partition.get_high_watermark()}")
            print(f"    Index entries  : {len(partition._offset_index)}")
            # Fetch from middle
            mid_offset = partition.get_high_watermark() // 2
            fetched = partition.fetch(mid_offset, max_messages=2)
            print(f"    Fetch from offset {mid_offset}: got {len(fetched)} messages")
            break

    # ── Final Stats ───────────────────────────────────────────
    mq.print_stats()
    print(f"\n  Producer stats : {producer.stats()}")

    print("\n" + "=" * 65)
    print("Simulation complete.")


if __name__ == "__main__":
    run_simulation()
