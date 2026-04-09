"""
DISTRIBUTED MESSAGE QUEUE (KAFKA-LIKE)
========================================

FUNCTIONAL REQUIREMENTS:
- Producers publish messages to topics
- Consumers subscribe and read messages in order (per partition)
- Topics partitioned for parallelism
- Consumer groups: each group gets all messages; within group, each partition assigned to one consumer
- Replay: consumers can seek to any offset
- Retention: configurable (time-based or size-based)

NON-FUNCTIONAL REQUIREMENTS:
- Throughput: 10 M messages/second (Kafka benchmarks)
- End-to-end latency: < 5 ms (p95)
- Durability: replicated to 3 brokers; no message loss with ACK=all
- Ordering: per-partition ordering guaranteed
- Storage: 100 GB/day × 7-day retention = 700 GB (one topic)
- At-least-once delivery (exactly-once with transactions)

ARCHITECTURE:
  Producers ──▶ Partitioner ──▶ Broker(s)
                                    │
                             Partition Log
                             (append-only)
                                    │
                             Consumer Group ──▶ Consumers

KEY DESIGN DECISIONS:
1. PARTITIONING:
   - Key-based: consistent hash of message key → deterministic partition assignment.
     Messages with same key always go to same partition → ordered per key.
   - Round-robin: for keyless messages (load balance).
   Partition count determines max parallelism. Can increase partitions, not decrease.

2. STORAGE — Append-only segment files:
   - Each partition = ordered, append-only log of segments
   - Active segment: newest, receives appends
   - Closed segments: read-only, subject to retention GC
   - Segment index: sparse offset → file position (every 4096 bytes)
   - Sequential I/O: much faster than random writes (500 MB/s vs 50 MB/s HDD)

3. REPLICATION:
   - Leader-follower: one leader per partition; followers replicate.
   - ISR (In-Sync Replicas): followers within max_lag_ms of leader.
   - Producer ACK levels:
     - acks=0: fire-and-forget (fastest, may lose)
     - acks=1: leader ACK only (may lose if leader fails before replication)
     - acks=all: wait for ISR ACK (strongest durability)
   - Leader election: controller broker, ZooKeeper/KRaft metadata.

4. CONSUMER GROUPS:
   - Group coordinator: broker that manages group membership.
   - Group rebalance: triggered on consumer join/leave.
   - Partition assignment: range or round-robin strategy.
   - Offset commit: consumers commit offsets to __consumer_offsets topic.
     Committed offset = "I have processed up to here."

5. CONSUMER SEEK:
   - Seek to beginning/end/timestamp/specific offset
   - Useful for: replay from failure, backfill, debugging

6. EXACTLY-ONCE SEMANTICS (EOS):
   - Producer: idempotent sequences + transactions
   - Consumer: transactional offset commit (atomic with downstream write)

7. LOG COMPACTION:
   - For changelog topics: keep latest value per key (like a KV store)
   - GC thread scans logs, removes superseded records.

KAFKA INTERNALS — WRITE PATH:
  1. Producer batches messages (linger.ms, batch.size)
  2. Partitioner selects partition
  3. Network I/O to leader broker
  4. Leader writes to page cache (OS buffer, async flush)
  5. Followers fetch from leader (pull model)
  6. Leader updates ISR; sends ACK to producer when quorum reached
  7. OS periodically flushes page cache to disk

KAFKA INTERNALS — READ PATH:
  1. Consumer sends fetch request with offset
  2. Broker seeks to offset in segment index
  3. Zero-copy sendfile: segment → NIC directly (skips user space)
  4. Consumer processes and commits offset
"""

from __future__ import annotations
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable
from enum import Enum
from collections import defaultdict
import threading
import math


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class Message:
    key: Optional[bytes]
    value: bytes
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # Set by broker on append
    offset: int = -1
    partition: int = -1
    topic: str = ""

    def __len__(self) -> int:
        return len(self.value) + (len(self.key) if self.key else 0)


class AckLevel(Enum):
    ACK_0 = 0    # Fire and forget
    ACK_1 = 1    # Leader ACK
    ACK_ALL = -1  # All ISR must ACK


class OffsetReset(Enum):
    EARLIEST = "earliest"
    LATEST = "latest"


# ---------------------------------------------------------------------------
# Partition Log (append-only)
# ---------------------------------------------------------------------------

class PartitionLog:
    """
    Simulates an append-only partition log with offset indexing.
    In production: segment files on disk with sparse index.
    """

    def __init__(self, topic: str, partition_id: int,
                 retention_bytes: int = 100 * 1024 * 1024,  # 100 MB
                 retention_ms: float = 7 * 86400 * 1000):   # 7 days
        self.topic = topic
        self.partition_id = partition_id
        self._retention_bytes = retention_bytes
        self._retention_ms = retention_ms
        self._messages: List[Message] = []
        self._base_offset = 0   # Offset of first message (increases as retention deletes)
        self._lock = threading.Lock()
        self._total_bytes = 0

    def append(self, msg: Message) -> int:
        with self._lock:
            offset = self._base_offset + len(self._messages)
            msg.offset = offset
            msg.partition = self.partition_id
            msg.topic = self.topic
            self._messages.append(msg)
            self._total_bytes += len(msg)
            self._gc_if_needed()
            return offset

    def read(self, offset: int, max_messages: int = 100) -> List[Message]:
        with self._lock:
            local_idx = offset - self._base_offset
            if local_idx < 0:
                local_idx = 0  # Seeked before retention window
            end_idx = local_idx + max_messages
            return self._messages[local_idx:end_idx]

    def _gc_if_needed(self) -> None:
        """Remove old messages based on retention policy."""
        while self._messages:
            oldest = self._messages[0]
            age_ms = (time.time() - oldest.timestamp) * 1000
            if age_ms > self._retention_ms:
                self._total_bytes -= len(oldest)
                self._messages.pop(0)
                self._base_offset += 1
            elif self._total_bytes > self._retention_bytes:
                self._total_bytes -= len(oldest)
                self._messages.pop(0)
                self._base_offset += 1
            else:
                break

    @property
    def log_end_offset(self) -> int:
        with self._lock:
            return self._base_offset + len(self._messages)

    @property
    def log_start_offset(self) -> int:
        return self._base_offset

    def size(self) -> int:
        with self._lock:
            return len(self._messages)


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------

class Broker:
    def __init__(self, broker_id: str):
        self.broker_id = broker_id
        self._logs: Dict[str, Dict[int, PartitionLog]] = defaultdict(dict)
        self._lock = threading.Lock()

    def ensure_partition(self, topic: str, partition_id: int) -> PartitionLog:
        with self._lock:
            if partition_id not in self._logs[topic]:
                self._logs[topic][partition_id] = PartitionLog(topic, partition_id)
            return self._logs[topic][partition_id]

    def append(self, topic: str, partition_id: int, msg: Message) -> int:
        log = self.ensure_partition(topic, partition_id)
        return log.append(msg)

    def fetch(self, topic: str, partition_id: int, offset: int,
               max_messages: int = 100) -> List[Message]:
        log = self._logs.get(topic, {}).get(partition_id)
        if not log:
            return []
        return log.read(offset, max_messages)

    def log_end_offset(self, topic: str, partition_id: int) -> int:
        log = self._logs.get(topic, {}).get(partition_id)
        return log.log_end_offset if log else 0

    def log_start_offset(self, topic: str, partition_id: int) -> int:
        log = self._logs.get(topic, {}).get(partition_id)
        return log.log_start_offset if log else 0

    def partition_size(self, topic: str, partition_id: int) -> int:
        log = self._logs.get(topic, {}).get(partition_id)
        return log.size() if log else 0


# ---------------------------------------------------------------------------
# Topic Metadata & Cluster
# ---------------------------------------------------------------------------

@dataclass
class TopicConfig:
    name: str
    num_partitions: int
    replication_factor: int = 3
    retention_ms: float = 7 * 86400 * 1000
    retention_bytes: int = -1  # -1 = no limit


class Cluster:
    """Simulates Kafka cluster with multiple brokers."""

    def __init__(self):
        self._brokers: Dict[str, Broker] = {}
        self._topics: Dict[str, TopicConfig] = {}
        self._leader_map: Dict[str, Dict[int, str]] = defaultdict(dict)  # topic → partition → broker_id

    def add_broker(self, broker: Broker) -> None:
        self._brokers[broker.broker_id] = broker

    def create_topic(self, config: TopicConfig) -> None:
        self._topics[config.name] = config
        broker_ids = list(self._brokers.keys())
        if not broker_ids:
            raise RuntimeError("No brokers available")
        # Assign leaders round-robin
        for pid in range(config.num_partitions):
            leader = broker_ids[pid % len(broker_ids)]
            self._leader_map[config.name][pid] = leader

    def leader_for(self, topic: str, partition_id: int) -> Optional[Broker]:
        leader_id = self._leader_map.get(topic, {}).get(partition_id)
        return self._brokers.get(leader_id) if leader_id else None

    def get_topic(self, topic: str) -> Optional[TopicConfig]:
        return self._topics.get(topic)

    def partition_count(self, topic: str) -> int:
        cfg = self._topics.get(topic)
        return cfg.num_partitions if cfg else 0


# ---------------------------------------------------------------------------
# Partitioner
# ---------------------------------------------------------------------------

class Partitioner:
    @staticmethod
    def partition(key: Optional[bytes], num_partitions: int,
                  counter: List[int]) -> int:
        if key:
            # Consistent hash: same key always → same partition
            h = int(hashlib.md5(key).hexdigest(), 16)
            return h % num_partitions
        else:
            # Round-robin for keyless messages
            p = counter[0] % num_partitions
            counter[0] += 1
            return p


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------

class Producer:
    def __init__(self, cluster: Cluster, ack_level: AckLevel = AckLevel.ACK_1):
        self._cluster = cluster
        self._ack = ack_level
        self._counter = [0]   # For round-robin partition selection

    def produce(self, topic: str, value: bytes, key: Optional[bytes] = None,
                headers: Dict[str, str] = None) -> Optional[int]:
        """Publish a message. Returns offset if ACK=1 or ACK=ALL."""
        config = self._cluster.get_topic(topic)
        if not config:
            return None

        partition_id = Partitioner.partition(key, config.num_partitions, self._counter)
        broker = self._cluster.leader_for(topic, partition_id)
        if not broker:
            return None

        msg = Message(key=key, value=value, headers=headers or {})
        offset = broker.append(topic, partition_id, msg)

        if self._ack == AckLevel.ACK_0:
            return None
        return offset

    def produce_batch(self, topic: str, messages: List[Tuple[bytes, Optional[bytes]]]) -> List[int]:
        """Produce a batch. Returns list of offsets."""
        offsets = []
        for value, key in messages:
            offset = self.produce(topic, value, key)
            if offset is not None:
                offsets.append(offset)
        return offsets


# ---------------------------------------------------------------------------
# Consumer Group
# ---------------------------------------------------------------------------

@dataclass
class ConsumerGroupMember:
    consumer_id: str
    assigned_partitions: List[int] = field(default_factory=list)
    heartbeat_at: float = field(default_factory=time.time)


class ConsumerGroup:
    """
    Manages partition assignment for a consumer group.
    Uses range assignment strategy.
    """

    def __init__(self, group_id: str, topic: str, cluster: Cluster):
        self.group_id = group_id
        self.topic = topic
        self._cluster = cluster
        self._members: Dict[str, ConsumerGroupMember] = {}
        # consumer_id → partition_id → committed_offset
        self._offsets: Dict[str, Dict[int, int]] = defaultdict(dict)
        self._lock = threading.Lock()

    def join(self, consumer_id: str) -> ConsumerGroupMember:
        with self._lock:
            member = ConsumerGroupMember(consumer_id)
            self._members[consumer_id] = member
            self._rebalance()
            return member

    def leave(self, consumer_id: str) -> None:
        with self._lock:
            self._members.pop(consumer_id, None)
            if self._members:
                self._rebalance()

    def _rebalance(self) -> None:
        """Assign partitions to members (range strategy)."""
        num_partitions = self._cluster.partition_count(self.topic)
        member_list = sorted(self._members.keys())
        if not member_list:
            return

        # Reset all assignments
        for m in self._members.values():
            m.assigned_partitions = []

        # Range assignment: divide partitions evenly
        partitions_per_member = num_partitions // len(member_list)
        remainder = num_partitions % len(member_list)
        start = 0
        for i, consumer_id in enumerate(member_list):
            count = partitions_per_member + (1 if i < remainder else 0)
            self._members[consumer_id].assigned_partitions = list(range(start, start + count))
            start += count

    def commit_offset(self, consumer_id: str, partition_id: int, offset: int) -> None:
        with self._lock:
            self._offsets[consumer_id][partition_id] = offset

    def get_offset(self, consumer_id: str, partition_id: int,
                    reset: OffsetReset = OffsetReset.LATEST) -> int:
        committed = self._offsets.get(consumer_id, {}).get(partition_id)
        if committed is not None:
            return committed
        # No committed offset → apply reset policy
        broker = self._cluster.leader_for(self.topic, partition_id)
        if not broker:
            return 0
        if reset == OffsetReset.EARLIEST:
            return broker.log_start_offset(self.topic, partition_id)
        else:
            return broker.log_end_offset(self.topic, partition_id)

    def get_assignments(self, consumer_id: str) -> List[int]:
        with self._lock:
            member = self._members.get(consumer_id)
            return member.assigned_partitions if member else []

    def lag(self, consumer_id: str) -> Dict[int, int]:
        """Compute consumer lag (messages behind) per partition."""
        result = {}
        for pid in self.get_assignments(consumer_id):
            broker = self._cluster.leader_for(self.topic, pid)
            if not broker:
                continue
            committed = self._offsets.get(consumer_id, {}).get(pid, 0)
            leo = broker.log_end_offset(self.topic, pid)
            result[pid] = max(0, leo - committed)
        return result


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------

class Consumer:
    def __init__(self, consumer_id: str, group: ConsumerGroup,
                 cluster: Cluster, offset_reset: OffsetReset = OffsetReset.LATEST):
        self.consumer_id = consumer_id
        self._group = group
        self._cluster = cluster
        self._offset_reset = offset_reset
        self._current_offsets: Dict[int, int] = {}
        group.join(consumer_id)

    def poll(self, max_records: int = 100,
              timeout_ms: float = 100) -> List[Message]:
        """Poll for new messages across assigned partitions."""
        records = []
        assigned = self._group.get_assignments(self.consumer_id)

        for pid in assigned:
            offset = self._current_offsets.get(
                pid, self._group.get_offset(self.consumer_id, pid, self._offset_reset)
            )
            broker = self._cluster.leader_for(self._group.topic, pid)
            if not broker:
                continue
            msgs = broker.fetch(self._group.topic, pid, offset, max_records)
            for msg in msgs:
                records.append(msg)
                self._current_offsets[pid] = msg.offset + 1
            if len(records) >= max_records:
                break

        return records

    def commit(self) -> None:
        """Commit current offsets to the group."""
        for pid, offset in self._current_offsets.items():
            self._group.commit_offset(self.consumer_id, pid, offset)

    def seek(self, partition_id: int, offset: int) -> None:
        """Seek to a specific offset (replay from this point)."""
        self._current_offsets[partition_id] = offset

    def seek_to_beginning(self) -> None:
        for pid in self._group.get_assignments(self.consumer_id):
            broker = self._cluster.leader_for(self._group.topic, pid)
            if broker:
                self._current_offsets[pid] = broker.log_start_offset(
                    self._group.topic, pid
                )

    def seek_to_end(self) -> None:
        for pid in self._group.get_assignments(self.consumer_id):
            broker = self._cluster.leader_for(self._group.topic, pid)
            if broker:
                self._current_offsets[pid] = broker.log_end_offset(
                    self._group.topic, pid
                )

    def close(self) -> None:
        self._group.leave(self.consumer_id)

    @property
    def lag(self) -> Dict[int, int]:
        return self._group.lag(self.consumer_id)


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def build_cluster(num_brokers: int = 3) -> Cluster:
    cluster = Cluster()
    for i in range(1, num_brokers + 1):
        cluster.add_broker(Broker(f"broker-{i}"))
    return cluster


def demonstrate_1_produce_and_consume():
    print("\n=== 1. Basic Produce & Consume ===")
    cluster = build_cluster(3)
    cluster.create_topic(TopicConfig("orders", num_partitions=3))

    producer = Producer(cluster, ack_level=AckLevel.ACK_1)
    group = ConsumerGroup("order-processors", "orders", cluster)
    consumer = Consumer("consumer-1", group, cluster, OffsetReset.EARLIEST)

    # Produce messages
    order_ids = [f"ORD-{i:04d}" for i in range(1, 11)]
    offsets = []
    for oid in order_ids:
        offset = producer.produce("orders", oid.encode(), key=oid.encode())
        offsets.append(offset)

    print(f"Produced {len(order_ids)} orders")
    print(f"Offsets: {offsets[:5]}...")

    # Consume
    records = consumer.poll(max_records=20)
    print(f"Consumed {len(records)} records:")
    for r in records[:5]:
        print(f"  partition={r.partition}, offset={r.offset}, "
              f"value={r.value.decode()}")

    consumer.commit()
    lag = consumer.lag
    print(f"\nConsumer lag after commit: {lag} (0 = fully caught up)")


def demonstrate_2_key_based_partitioning():
    print("\n=== 2. Key-based Partitioning (Same Key → Same Partition) ===")
    cluster = build_cluster(3)
    cluster.create_topic(TopicConfig("events", num_partitions=4))

    producer = Producer(cluster)

    # Messages with same user_id should go to same partition
    events = [
        ("user_alice", "login"), ("user_bob", "purchase"), ("user_alice", "view"),
        ("user_carol", "logout"), ("user_bob", "review"), ("user_alice", "purchase"),
    ]
    partition_map: Dict[str, Set[int]] = defaultdict(set)
    for user_id, event in events:
        msg = Message(key=user_id.encode(), value=event.encode())
        pid = Partitioner.partition(user_id.encode(), 4, [0])
        broker = cluster.leader_for("events", pid)
        if broker:
            offset = broker.append("events", pid, msg)
            partition_map[user_id].add(pid)
        print(f"  {user_id} → partition {pid} ({event})")

    print(f"\nPartition consistency check:")
    for user, partitions in partition_map.items():
        consistent = len(partitions) == 1
        print(f"  {user}: partitions={partitions}, "
              f"consistent={'YES' if consistent else 'NO (BUG!)'}")


def demonstrate_3_consumer_group_rebalance():
    print("\n=== 3. Consumer Group Rebalance ===")
    cluster = build_cluster(3)
    cluster.create_topic(TopicConfig("tasks", num_partitions=6))
    group = ConsumerGroup("task-workers", "tasks", cluster)

    # 3 consumers join
    c1 = Consumer("worker-1", group, cluster, OffsetReset.EARLIEST)
    c2 = Consumer("worker-2", group, cluster, OffsetReset.EARLIEST)
    c3 = Consumer("worker-3", group, cluster, OffsetReset.EARLIEST)

    print(f"After 3 consumers join (6 partitions):")
    for c in [c1, c2, c3]:
        print(f"  {c.consumer_id}: partitions={group.get_assignments(c.consumer_id)}")

    # Consumer 2 leaves — rebalance
    c2.close()
    print(f"\nAfter worker-2 leaves:")
    for c in [c1, c3]:
        print(f"  {c.consumer_id}: partitions={group.get_assignments(c.consumer_id)}")

    # New consumer joins
    c4 = Consumer("worker-4", group, cluster, OffsetReset.EARLIEST)
    print(f"\nAfter worker-4 joins:")
    for c in [c1, c3, c4]:
        print(f"  {c.consumer_id}: partitions={group.get_assignments(c.consumer_id)}")


def demonstrate_4_seek_and_replay():
    print("\n=== 4. Offset Seek & Message Replay ===")
    cluster = build_cluster(2)
    cluster.create_topic(TopicConfig("audit-log", num_partitions=1))

    producer = Producer(cluster)
    group = ConsumerGroup("auditor", "audit-log", cluster)
    consumer = Consumer("auditor-1", group, cluster, OffsetReset.EARLIEST)

    # Produce 10 events
    for i in range(10):
        producer.produce("audit-log", f"event-{i}".encode())

    # Consume all
    records = consumer.poll(max_records=20)
    consumer.commit()
    print(f"Initial consume: {len(records)} records")
    print(f"  Last offset consumed: {records[-1].offset}")

    # Seek back to beginning and replay
    consumer.seek_to_beginning()
    replayed = consumer.poll(max_records=5)  # Read first 5 only
    print(f"\nAfter seek to beginning, replayed: {len(replayed)} records")
    for r in replayed:
        print(f"  offset={r.offset}, value={r.value.decode()}")

    # Seek to specific offset
    consumer.seek(partition_id=0, offset=7)
    from_7 = consumer.poll(max_records=20)
    print(f"\nAfter seek to offset 7:")
    for r in from_7:
        print(f"  offset={r.offset}, value={r.value.decode()}")


def demonstrate_5_consumer_lag():
    print("\n=== 5. Consumer Lag Monitoring ===")
    cluster = build_cluster(2)
    cluster.create_topic(TopicConfig("metrics", num_partitions=3))

    producer = Producer(cluster)
    group = ConsumerGroup("metric-processors", "metrics", cluster)
    consumer = Consumer("processor-1", group, cluster, OffsetReset.EARLIEST)

    # Produce 100 messages
    for i in range(100):
        producer.produce("metrics", f"metric_{i}".encode())

    # Consumer processes only 30
    records = consumer.poll(max_records=30)
    consumer.commit()

    lag = consumer.lag
    total_lag = sum(lag.values())
    print(f"Consumer processed: {len(records)} messages")
    print(f"Consumer lag per partition: {lag}")
    print(f"Total lag: {total_lag} messages behind")

    # Catch up
    while True:
        batch = consumer.poll(max_records=50)
        if not batch:
            break
        consumer.commit()

    lag_after = consumer.lag
    print(f"\nAfter catching up: lag={lag_after} (all zeros = fully caught up)")


def demonstrate_6_log_retention():
    print("\n=== 6. Log Retention & GC ===")
    cluster = build_cluster(1)
    # Short retention for demo: 5 messages max
    config = TopicConfig("temp-events", num_partitions=1)
    cluster.create_topic(config)

    broker = cluster.leader_for("temp-events", 0)
    # Manually set short retention
    log = broker.ensure_partition("temp-events", 0)
    log._retention_bytes = 50   # Very small

    for i in range(10):
        msg = Message(key=None, value=f"msg-{i:02d}".encode())
        offset = broker.append("temp-events", 0, msg)

    print(f"After appending 10 messages:")
    print(f"  Log end offset: {log.log_end_offset}")
    print(f"  Log start offset: {log.log_start_offset}")
    print(f"  Messages retained: {log.size()}")
    print(f"  (Older messages GC'd due to size retention)")


if __name__ == "__main__":
    demonstrate_1_produce_and_consume()
    demonstrate_2_key_based_partitioning()
    demonstrate_3_consumer_group_rebalance()
    demonstrate_4_seek_and_replay()
    demonstrate_5_consumer_lag()
    demonstrate_6_log_retention()
