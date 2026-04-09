"""
KAFKA ARCHITECTURE DEEP DIVE
================================

Problem Statement:
Traditional message queues delete messages after consumption. Kafka keeps
messages for days/weeks, enabling: replay, multiple independent consumers,
event sourcing, and stream processing — all from the same immutable log.

Kafka Core Concepts:

  Topic    : Named log of messages (like a DB table).
  Partition: Ordered, immutable sequence of messages. Topics split into N partitions.
             Each partition has one leader broker + N replicas.
  Offset   : Sequential message ID within a partition. Consumers track their offset.
  Broker   : Kafka server. Cluster has N brokers; each stores some partitions.
  Producer : Publishes to topic, chooses partition (by key hash or round-robin).
  Consumer : Reads from assigned partitions, commits offset to track progress.
  Consumer Group: Set of consumers that share partition assignment.
                  Each partition assigned to exactly one consumer in the group.
                  Scale consumers = scale throughput (add consumers → add partitions).

Why Kafka is Fast:
  Sequential disk writes (log append) → 100MB/s+ write throughput.
  Zero-copy transfer: OS sendfile() from page cache to network.
  Batching: producers batch multiple records; consumers fetch in bulk.
  Compression: LZ4/Snappy per batch.

Kafka vs Traditional Queue:
  Traditional (RabbitMQ): messages consumed → deleted.
  Kafka: messages retained for days/weeks. Many consumer groups read same topic.
  Result: decouple consumers from producers in time.

Kafka Guarantees:
  Within a partition: strict FIFO ordering.
  Across partitions: no global ordering.
  Key-based routing: same key → same partition → ordered per key.
  Replication: N copies across brokers (configurable acks: 0, 1, all).
  Exactly-once: transactions + idempotent producer (Kafka 0.11+).

Use Cases:
  Event log     : "Order placed", "Payment processed" — replay anytime.
  Activity feed : User clicks, page views → Flink/Spark aggregation.
  CDC           : Database Change Data Capture via Debezium.
  Data pipeline : ETL backbone (app → Kafka → data lake).
  Microservice bus: event-driven architecture backbone.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
import time
import random
import threading
import hashlib
from collections import defaultdict, deque
from enum import Enum


class AckMode(Enum):
    NONE   = 0   # fire and forget
    LEADER = 1   # ack from leader only
    ALL    = -1  # ack from all in-sync replicas


# ─────────────────────────────────────────────
# KAFKA RECORD
# ─────────────────────────────────────────────

@dataclass
class KafkaRecord:
    topic       : str
    partition   : int
    offset      : int = 0
    key         : Optional[bytes] = None
    value       : bytes = b""
    timestamp   : float = field(default_factory=time.time)
    headers     : Dict[str, str] = field(default_factory=dict)

    def __str__(self):
        key_str   = self.key.decode() if self.key else "None"
        value_str = self.value.decode()[:40] if self.value else ""
        return f"[{self.topic}:{self.partition}@{self.offset}] key={key_str} value={value_str}"


# ─────────────────────────────────────────────
# KAFKA PARTITION
# ─────────────────────────────────────────────

class KafkaPartition:
    """
    Ordered, append-only log segment.
    Messages are never deleted (retention enforced separately).
    """

    def __init__(self, topic: str, partition_id: int, leader_id: int):
        self.topic        = topic
        self.partition_id = partition_id
        self.leader_id    = leader_id
        self._log         : List[KafkaRecord] = []
        self._lock        = threading.Lock()

    def append(self, key: Optional[bytes], value: bytes,
               headers: Dict[str, str] = None) -> KafkaRecord:
        with self._lock:
            offset = len(self._log)
            record = KafkaRecord(
                topic=self.topic, partition=self.partition_id, offset=offset,
                key=key, value=value, headers=headers or {}
            )
            self._log.append(record)
            return record

    def read(self, offset: int, max_records: int = 100) -> List[KafkaRecord]:
        """Read records starting from offset."""
        with self._lock:
            return self._log[offset:offset + max_records]

    def end_offset(self) -> int:
        return len(self._log)

    def earliest_offset(self) -> int:
        return 0   # simplified: no compaction/retention here


# ─────────────────────────────────────────────
# KAFKA BROKER
# ─────────────────────────────────────────────

class KafkaBroker:
    def __init__(self, broker_id: int):
        self.broker_id  = broker_id
        self._partitions: Dict[Tuple[str, int], KafkaPartition] = {}

    def add_partition(self, partition: KafkaPartition):
        self._partitions[(partition.topic, partition.partition_id)] = partition

    def get_partition(self, topic: str, partition_id: int) -> Optional[KafkaPartition]:
        return self._partitions.get((topic, partition_id))

    def all_partitions(self) -> List[KafkaPartition]:
        return list(self._partitions.values())


# ─────────────────────────────────────────────
# KAFKA CLUSTER
# ─────────────────────────────────────────────

class KafkaCluster:
    """Simulates a Kafka cluster: topics, partitions, partition leadership."""

    def __init__(self, n_brokers: int = 3):
        self._brokers      : List[KafkaBroker] = [KafkaBroker(i) for i in range(n_brokers)]
        self._topics       : Dict[str, Dict]   = {}   # topic → {n_parts, replication}
        self._partitions   : Dict[Tuple[str, int], KafkaPartition] = {}

    def create_topic(self, name: str, n_partitions: int = 3, replication: int = 2):
        self._topics[name] = {"n_partitions": n_partitions, "replication": replication}
        for p in range(n_partitions):
            leader = self._brokers[p % len(self._brokers)]
            partition = KafkaPartition(name, p, leader.broker_id)
            self._partitions[(name, p)] = partition
            leader.add_partition(partition)

    def get_partition(self, topic: str, partition_id: int) -> Optional[KafkaPartition]:
        return self._partitions.get((topic, partition_id))

    def n_partitions(self, topic: str) -> int:
        return self._topics.get(topic, {}).get("n_partitions", 0)

    def list_topics(self) -> List[str]:
        return list(self._topics.keys())


# ─────────────────────────────────────────────
# KAFKA PRODUCER
# ─────────────────────────────────────────────

class KafkaProducer:
    """
    Routes records to partitions by key hash or round-robin.
    Batches records for throughput.
    """

    def __init__(self, cluster: KafkaCluster, acks: AckMode = AckMode.ALL):
        self.cluster   = cluster
        self.acks      = acks
        self.sent      = 0
        self.errors    = 0
        self._rr_counter : Dict[str, int] = defaultdict(int)

    def _choose_partition(self, topic: str, key: Optional[bytes]) -> int:
        n = self.cluster.n_partitions(topic)
        if n == 0:
            raise ValueError(f"Topic '{topic}' not found")
        if key is not None:
            # Consistent hash: same key → same partition
            return int(hashlib.md5(key).hexdigest(), 16) % n
        else:
            # Round-robin for keyless records
            p = self._rr_counter[topic] % n
            self._rr_counter[topic] += 1
            return p

    def send(self, topic: str, value: Any, key: Any = None,
             headers: Dict[str, str] = None) -> Optional[KafkaRecord]:
        key_bytes   = str(key).encode() if key is not None else None
        value_bytes = str(value).encode() if not isinstance(value, bytes) else value

        try:
            partition_id = self._choose_partition(topic, key_bytes)
            partition    = self.cluster.get_partition(topic, partition_id)
            if not partition:
                self.errors += 1
                return None
            record = partition.append(key_bytes, value_bytes, headers)
            self.sent += 1
            return record
        except Exception as e:
            self.errors += 1
            return None

    def send_batch(self, topic: str, records: List[Tuple]) -> int:
        """Send multiple records efficiently."""
        sent = 0
        for item in records:
            key, value = (item[0], item[1]) if len(item) == 2 else (None, item[0])
            if self.send(topic, value, key):
                sent += 1
        return sent


# ─────────────────────────────────────────────
# KAFKA CONSUMER + CONSUMER GROUP
# ─────────────────────────────────────────────

@dataclass
class ConsumerOffset:
    topic      : str
    partition  : int
    offset     : int = 0   # next offset to read


class KafkaConsumer:
    """
    Reads from assigned partitions, tracks committed offsets.
    Supports manual and auto offset commit.
    """

    def __init__(self, consumer_id: str, group_id: str, cluster: KafkaCluster):
        self.consumer_id = consumer_id
        self.group_id    = group_id
        self.cluster     = cluster
        self._offsets    : Dict[Tuple[str, int], int] = {}   # (topic, part) → next offset
        self._assignment : List[Tuple[str, int]] = []
        self.records_consumed = 0
        self.committed_records = 0

    def assign(self, topic_partitions: List[Tuple[str, int]]):
        self._assignment = topic_partitions
        for tp in topic_partitions:
            if tp not in self._offsets:
                self._offsets[tp] = 0

    def poll(self, max_records: int = 10) -> List[KafkaRecord]:
        """Fetch records from all assigned partitions."""
        all_records = []
        for (topic, part_id) in self._assignment:
            partition = self.cluster.get_partition(topic, part_id)
            if not partition:
                continue
            offset  = self._offsets.get((topic, part_id), 0)
            records = partition.read(offset, max_records)
            all_records.extend(records)
        return all_records

    def commit(self, records: List[KafkaRecord]):
        """Commit offsets after processing."""
        for record in records:
            tp = (record.topic, record.partition)
            self._offsets[tp] = record.offset + 1
            self.committed_records += 1
        self.records_consumed += len(records)

    def seek_to_offset(self, topic: str, partition: int, offset: int):
        """Seek to specific offset (for replay)."""
        self._offsets[(topic, partition)] = offset

    def seek_to_beginning(self, topic: str):
        """Reset to earliest offset (replay all messages)."""
        n = self.cluster.n_partitions(topic)
        for p in range(n):
            if (topic, p) in self._offsets:
                self._offsets[(topic, p)] = 0

    def lag(self, topic: str) -> int:
        """Total unread messages across assigned partitions."""
        total_lag = 0
        for (t, p), offset in self._offsets.items():
            if t != topic:
                continue
            partition = self.cluster.get_partition(t, p)
            if partition:
                total_lag += partition.end_offset() - offset
        return total_lag


class ConsumerGroup:
    """
    Assigns partitions to consumers.
    Each partition → exactly one consumer in the group.
    Adding consumers increases parallelism up to n_partitions.
    """

    def __init__(self, group_id: str, cluster: KafkaCluster, topic: str):
        self.group_id   = group_id
        self.cluster    = cluster
        self.topic      = topic
        self.consumers  : List[KafkaConsumer] = []
        self._assignment: Dict[str, List[Tuple[str, int]]] = {}

    def add_consumer(self, consumer: KafkaConsumer):
        self.consumers.append(consumer)
        self._rebalance()

    def remove_consumer(self, consumer_id: str):
        self.consumers = [c for c in self.consumers if c.consumer_id != consumer_id]
        self._rebalance()

    def _rebalance(self):
        """Round-robin partition assignment across consumers."""
        n_parts = self.cluster.n_partitions(self.topic)
        all_parts = [(self.topic, p) for p in range(n_parts)]
        n_consumers = max(1, len(self.consumers))

        self._assignment = defaultdict(list)
        for i, tp in enumerate(all_parts):
            consumer = self.consumers[i % n_consumers]
            self._assignment[consumer.consumer_id].append(tp)

        for consumer in self.consumers:
            consumer.assign(self._assignment[consumer.consumer_id])

    def assignment_str(self) -> str:
        lines = []
        for consumer in self.consumers:
            parts = self._assignment.get(consumer.consumer_id, [])
            lines.append(f"    {consumer.consumer_id}: {[p[1] for p in parts]}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_kafka():
    print("=" * 65)
    print("KAFKA ARCHITECTURE DEEP DIVE")
    print("=" * 65)

    random.seed(42)

    # ── Cluster Setup ─────────────────────────
    print("\n[1] KAFKA CLUSTER SETUP")
    print("─" * 55)
    cluster = KafkaCluster(n_brokers=3)
    cluster.create_topic("orders", n_partitions=4, replication=2)
    cluster.create_topic("events", n_partitions=3, replication=2)
    print(f"  Cluster: 3 brokers, topics: {cluster.list_topics()}")
    print(f"  orders: 4 partitions  events: 3 partitions")

    # ── Producer — Key-Based Routing ──────────
    print("\n\n[2] PRODUCER — KEY-BASED PARTITION ROUTING")
    print("─" * 55)
    producer = KafkaProducer(cluster, acks=AckMode.ALL)

    # Same user_id → always same partition
    users = ["user-1", "user-2", "user-3"]
    for user in users:
        for order_n in range(3):
            record = producer.send("orders",
                                    value={"order": order_n, "user": user, "amt": random.randint(10, 500)},
                                    key=user)
            if record:
                print(f"  user={user} order#{order_n} → partition={record.partition} offset={record.offset}")

    # Keyless → round-robin
    print(f"\n  Keyless records (round-robin):")
    for i in range(4):
        record = producer.send("events", value=f"event-{i}")
        if record:
            print(f"  event-{i} → partition={record.partition}")

    print(f"\n  Producer: sent={producer.sent}  errors={producer.errors}")

    # ── Consumer Group ─────────────────────────
    print("\n\n[3] CONSUMER GROUP — PARTITION ASSIGNMENT")
    print("─" * 55)
    group = ConsumerGroup("order-processors", cluster, "orders")

    # Start with 2 consumers
    c1 = KafkaConsumer("consumer-1", "order-processors", cluster)
    c2 = KafkaConsumer("consumer-2", "order-processors", cluster)
    group.add_consumer(c1)
    group.add_consumer(c2)
    print(f"  2 consumers, 4 partitions:")
    print(group.assignment_str())

    # Add a 3rd consumer → rebalance
    c3 = KafkaConsumer("consumer-3", "order-processors", cluster)
    group.add_consumer(c3)
    print(f"\n  After adding consumer-3 (rebalance):")
    print(group.assignment_str())

    # ── Consuming Messages ─────────────────────
    print("\n\n[4] CONSUMING AND COMMITTING OFFSETS")
    print("─" * 55)
    records = c1.poll(max_records=20)
    print(f"  consumer-1 polled {len(records)} records:")
    for r in records:
        val_str = r.value.decode()[:50]
        print(f"    {r.topic}:{r.partition}@{r.offset} → {val_str}")

    c1.commit(records)
    print(f"\n  Committed {c1.committed_records} offsets")
    print(f"  Lag after commit: {c1.lag('orders')}")

    # ── Replay (Seek to Beginning) ─────────────
    print("\n\n[5] REPLAY — SEEK TO BEGINNING")
    print("─" * 55)
    replayer = KafkaConsumer("replayer-1", "analytics-replayer", cluster)
    replayer.assign([(t, p) for t in ["orders"] for p in range(cluster.n_partitions("orders"))])
    replayer.seek_to_beginning("orders")

    all_records = replayer.poll(max_records=1000)
    print(f"  Replayer reads ALL historical records from 'orders':")
    print(f"  Total replayed: {len(all_records)} records")
    print(f"  (Consumer group 'analytics-replayer' doesn't affect 'order-processors' offsets)")

    # ── Partitioning Strategy ──────────────────
    print("\n\n[6] PARTITIONING STRATEGIES")
    print("─" * 55)
    strategies = [
        ("Key-based (hash)",    "user_id, order_id",   "Per-key ordering",      "Skew if hot keys"),
        ("Round-robin",         "No key",              "Even distribution",     "No ordering"),
        ("Custom partitioner",  "Tenant → shard map",  "Business logic routing","Complex"),
        ("Time-based",          "Hour/day bucket key", "Range queries easier",  "Hot partition for now"),
    ]
    print(f"  {'Strategy':<22} {'Key':<22} {'Pro':<24} {'Con'}")
    print(f"  {'─'*80}")
    for strat, key, pro, con in strategies:
        print(f"  {strat:<22} {key:<22} {pro:<24} {con}")

    # ── Kafka vs Traditional Queue ─────────────
    print("\n\n[7] KAFKA vs TRADITIONAL QUEUE (RABBITMQ/SQS)")
    print("─" * 55)
    comparison = [
        ("Message retention",  "Deleted after ack",        "Kept for days/weeks"),
        ("Multiple consumers", "Competing (one gets msg)", "Each group gets all"),
        ("Ordering",           "FIFO per queue",           "FIFO per partition"),
        ("Throughput",         "100K-500K msg/s",          "Millions/s"),
        ("Replay",             "Not possible",             "Yes — seek to offset"),
        ("Consumer lag",       "Queue depth",              "Lag per partition"),
        ("Protocol",           "AMQP/HTTP",                "Custom TCP protocol"),
        ("Use case",           "Task queues, RPC",         "Event streaming, log"),
    ]
    print(f"  {'Aspect':<22} {'RabbitMQ/SQS':<28} {'Kafka'}")
    print(f"  {'─'*70}")
    for aspect, rabbit, kafka in comparison:
        print(f"  {aspect:<22} {rabbit:<28} {kafka}")

    print(f"\n  Kafka total produced: {producer.sent} records  Consumers: {len(group.consumers)}")


if __name__ == "__main__":
    demonstrate_kafka()
