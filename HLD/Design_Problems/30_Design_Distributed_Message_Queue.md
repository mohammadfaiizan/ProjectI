# Problem 30: Design a Distributed Message Queue

---

## 1. Problem Statement & Clarifying Questions

### Problem Statement
Design a distributed message queue (like Apache Kafka) that supports high-throughput, fault-tolerant, ordered message delivery with consumer groups, configurable delivery semantics, and horizontal scalability to handle millions of messages per second.

### Clarifying Questions
1. **Use case**: Point-to-point (RabbitMQ) or pub-sub log (Kafka)? (Log-based, like Kafka)
2. **Throughput**: Messages per second? Message size? (1M messages/sec, avg 1 KB each)
3. **Ordering**: Strict global ordering, or per-partition ordering? (Per-partition only)
4. **Delivery semantics**: At-most-once, at-least-once, or exactly-once? (All three, configurable)
5. **Retention**: Delete after consumption, or retain for replay? (Retain for configurable period)
6. **Consumer groups**: Multiple independent consumers per topic? (Yes, like Kafka consumer groups)
7. **Message size**: Max message size? (Default 1 MB, configurable up to 100 MB)
8. **Durability**: How many replicas? Can we lose messages during broker failure? (0 message loss)
9. **Latency**: What's acceptable end-to-end latency? (< 10ms P99 for non-batch)
10. **Schema**: Schema enforcement? (Optional Schema Registry integration)

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
- Create topics with configurable partition count and replication factor
- Produce messages with optional key (key-based routing to same partition)
- Consume messages via consumer groups with offset tracking
- Commit offsets (auto or manual) to enable replay
- Seek to any offset (beginning, end, or specific offset)
- Configurable retention (time-based: 7 days; size-based: 1 TB per partition)
- Dead letter queue for messages that fail processing after max retries
- Log compaction (keep only latest value per key in compacted topics)
- At-most-once, at-least-once, and exactly-once delivery semantics
- Topic-level and consumer-group-level lag monitoring

### Non-Functional Requirements
- **Throughput**: 1M messages/sec write; 10M messages/sec read (10× fan-out)
- **Latency**: < 5ms P99 for single producer write; < 10ms end-to-end
- **Durability**: No message loss with replication factor ≥ 2 (0 RPO)
- **Availability**: 99.99% (< 1 hour/year downtime)
- **Scalability**: Add brokers without downtime; rebalance partitions online
- **Ordering**: Strict per-partition ordering guaranteed
- **Replay**: Consumers can re-read any message within retention window

---

## 3. Capacity Estimation

### Write Throughput
- 1M messages/sec × 1 KB/message = 1 GB/sec write throughput
- With replication factor 3: 3 GB/sec total write across cluster
- 10 brokers × 300 MB/sec each = 3 GB/sec (NVMe SSDs handle 1-3 GB/sec write)

### Storage
- 1M messages/sec × 1 KB × 86,400 sec/day = 86.4 TB/day
- With 7-day retention: 86.4 × 7 = 605 TB active storage
- With 3× replication: 1.8 PB total storage
- Per broker (50 brokers): 36 TB (manageable with high-density HDDs or NVMe SSDs)

### Read Throughput
- 10 consumer groups × 1M messages/sec = 10M messages/sec reads
- 10 MB/sec per consumer group × 10 groups = 100 GB/sec total reads
- Zero-copy sendfile: reads don't go through application memory → OS page cache serves reads

### Partition Count
- 1M messages/sec ÷ 100K messages/sec per partition = 10 partitions minimum
- For parallel consumer groups: aim for 100 partitions per high-throughput topic
- Max partitions per broker (practical): ~2,000

---

## 4. High-Level Architecture (ASCII Diagram)

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                           PRODUCERS                                       │
 │   App Servers │ Event Generators │ Change Data Capture (CDC)             │
 │   Producer SDK: batching, compression, retry, idempotent sequence nums   │
 └───────────────┬──────────────────────────────────────────────────────────┘
                 │ Produce(topic, key, value, headers)
 ┌───────────────▼──────────────────────────────────────────────────────────┐
 │                        BROKER CLUSTER                                     │
 │                                                                            │
 │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  │
 │  │  Broker 1   │   │  Broker 2   │   │  Broker 3   │   │  Broker N   │  │
 │  │             │   │             │   │             │   │             │  │
 │  │ Partition 0 │   │ Partition 1 │   │ Partition 2 │   │ Partition K │  │
 │  │ (LEADER)    │   │ (LEADER)    │   │ (LEADER)    │   │ (LEADER)    │  │
 │  │             │   │             │   │             │   │             │  │
 │  │ Partition 1 │   │ Partition 2 │   │ Partition 0 │   │ Partition N │  │
 │  │ (FOLLOWER)  │   │ (FOLLOWER)  │   │ (FOLLOWER)  │   │ (FOLLOWER)  │  │
 │  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘  │
 │                                                                            │
 │  ┌─────────────────────────────────────────────────────────────────────┐  │
 │  │              CONTROLLER (Raft / KRaft)                              │  │
 │  │  Partition leader election │ ISR management │ Broker registration   │  │
 │  └─────────────────────────────────────────────────────────────────────┘  │
 └───────────────────────────────────────────────────────────────────────────┘
                 │
 ┌───────────────▼──────────────────────────────────────────────────────────┐
 │                          CONSUMER GROUPS                                  │
 │                                                                            │
 │  Consumer Group A (Billing Service)   Consumer Group B (Analytics)        │
 │  Consumer 1 → Partition 0,1           Consumer 1 → Partition 0,1,2,3     │
 │  Consumer 2 → Partition 2,3           Consumer 2 → Partition 4,5,6,7     │
 │  Consumer 3 → Partition 4,5           Consumer 3 → Partition 8,9         │
 │                                                                            │
 │  Group Coordinator (Broker): offset storage, rebalancing protocol        │
 └──────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Partition Storage (WAL Structure)
Each partition is an append-only log on disk:
```
/data/topic-name/partition-0/
    00000000000000000000.log      ← segment file: raw message bytes
    00000000000000000000.index    ← sparse index: offset → file position
    00000000000000000000.timeindex ← time index: timestamp → offset
    00000000001000000000.log      ← next segment after 1B messages
    00000000001000000000.index
```

**Segment files**: Each log is split into 1 GB segments. Active segment = append-only; closed segments = immutable.

**Sparse Index**: Maps every Nth message offset to file byte position.
```
Offset 0     → byte 0
Offset 1000  → byte 1,048,576
Offset 2000  → byte 2,097,152
```
Finding offset 1500: binary search in index → find entry for 1000 → scan forward 500 messages.

**Message Format:**
```
[8 bytes: offset] [4 bytes: message_size] [4 bytes: CRC32]
[1 byte: magic] [1 byte: attributes (compression)] [8 bytes: timestamp]
[4 bytes: key_length] [key_bytes] [4 bytes: value_length] [value_bytes]
```

### 5.2 Replication and ISR (In-Sync Replicas)
**Leader-Follower Replication:**
1. Producer sends message to **partition leader**
2. Leader appends to local log; broadcasts to all ISR followers via replication protocol
3. Followers fetch new messages and acknowledge
4. Leader advances **High Watermark (HWM)** when all ISR followers have acknowledged
5. Only messages below HWM are visible to consumers (committed messages)

**ISR Management:**
- ISR = set of replicas within `replica.lag.time.max.ms` (default: 10 seconds) of leader
- If a follower falls behind (network issue, GC pause): removed from ISR
- If `min.insync.replicas=2` and ISR shrinks to 1: producer gets NotEnoughReplicas error
- Follower rejoins ISR once it catches up with leader's log

**Leader Election (KRaft):**
- Kafka 3.x uses Raft-based KRaft (ZooKeeper eliminated)
- Controller quorum (3-5 nodes) maintains cluster metadata log via Raft consensus
- On broker failure: controller detects (missed heartbeats), selects new leader from ISR

### 5.3 Producer: Delivery Semantics

**At-Most-Once:**
```python
producer.send(topic, value=message)  # Fire and forget; no retries
# acks=0: don't wait for broker acknowledgement
```
Use case: metrics, analytics where occasional loss is acceptable.

**At-Least-Once:**
```python
producer = Producer(acks='all', retries=3)  # Wait for all ISR acks; retry on failure
producer.send(topic, value=message)
# Risk: on retry after timeout (broker got msg but response lost) → duplicate message
```
Use case: event logging, audit trails where duplicates can be filtered downstream.

**Exactly-Once (Idempotent Producer):**
```python
producer = Producer(enable_idempotence=True)
# Producer gets PID (Producer ID) from broker
# Each message has (PID, partition, sequence_number)
# Broker deduplicates based on sequence number — drops retries
```

**Transactional Producer (Cross-partition exactly-once):**
```python
producer = Producer(transactional_id='my-transaction')
producer.init_transactions()
producer.begin_transaction()
producer.send('topic-A', 'msg1')
producer.send('topic-B', 'msg2')
producer.commit_transaction()  # Atomic: either both messages visible or neither
```

### 5.4 Consumer Groups and Partition Assignment

**Partition Assignment Strategies:**
- **Range**: Sort partitions and consumers; assign contiguous ranges → uneven if partitions not divisible
- **Round-Robin**: Distribute one partition at a time → most even distribution
- **Sticky**: Like round-robin but minimizes partition movement on rebalance

**Rebalancing Protocol:**
1. Consumer joins/leaves → Group Coordinator detects (missed heartbeat or JoinGroup request)
2. Group Coordinator sends JoinGroup response: one consumer elected as "Group Leader"
3. Group Leader receives member list + their subscriptions; computes partition assignment
4. Group Leader sends SyncGroup request with assignment plan
5. All consumers receive their assignments; resume consuming

**Consumer Lag:**
- `lag = partition_high_watermark - consumer_committed_offset`
- Lag > threshold → alert (consumer falling behind; may need scaling)
- Kafka's `__consumer_offsets` topic: stores committed offsets; replicated like any topic

### 5.5 Zero-Copy Transfer
Traditional read: Disk → Kernel buffer → User space → Socket buffer → NIC
Zero-copy: Disk → Kernel buffer → NIC (via sendfile() syscall)
```
# Linux sendfile syscall avoids two copies through user space
sendfile(socket_fd, file_fd, offset, count)
```
Result: Consumer reads achieve NIC line rate (~10 Gbps) without CPU bottleneck.

### 5.6 Batch Produce and Consume
**Producer batching:**
- Accumulate messages in RecordBatch until `batch.size` (16 KB) or `linger.ms` (5ms)
- Compress batch: LZ4 (fast), Snappy (balanced), GZIP (best compression)
- Send entire batch in one request → amortize network round-trip overhead

**Consumer fetching:**
- `fetch.min.bytes=1MB`: Wait until 1 MB of data available before returning
- `fetch.max.wait.ms=500`: Return after 500ms even if < 1 MB available
- Result: Consumer reads batches of messages → fewer round-trips → higher throughput

### 5.7 Log Compaction
For stateful use cases (e.g., database changelog, user settings):
- **Normal topics**: Delete old segments by time or size
- **Compacted topics**: Keep only the latest message per key; old versions deleted
- **Tombstone**: Message with null value → key eventually deleted from compacted log
- **Use case**: Kafka as a database changelog (Debezium CDC → Kafka compacted topic → downstream state rebuilding)

### 5.8 Dead Letter Queue (DLQ)
- Consumer fails to process a message after `max.retries` attempts
- Instead of blocking the partition, send failed message to `{topic}.DLQ` topic
- DLQ message includes: original message + error details + retry count + timestamp
- DLQ monitor: alerts on high DLQ rate; ops team can inspect and replay after fix

---

## 6. Database Design

### Broker Storage (Partition Log on Disk)
```
# Segment file format (simplified)
[RECORD_BATCH_HEADER]
  base_offset: int64       ← first offset in batch
  batch_length: int32
  partition_leader_epoch: int32
  magic: int8 (= 2)
  crc: int32
  attributes: int16        ← compression type, timestamp type
  last_offset_delta: int32
  first_timestamp: int64
  max_timestamp: int64
  producer_id: int64       ← for idempotent/transactional producers
  producer_epoch: int16
  base_sequence: int32
  num_records: int32
[RECORDS]
  [attributes: int8] [timestamp_delta: varint] [offset_delta: varint]
  [key_length: varint] [key: bytes] [value_length: varint] [value: bytes]
  [headers_count: varint] [headers...]
```

### Consumer Offsets (Internal Kafka Topic: `__consumer_offsets`)
```
# Stored as Kafka messages in a compacted topic
Key:   {group_id, topic, partition}
Value: {offset, metadata, commit_timestamp, leader_epoch}

Example:
Key:   "billing-service|user-events|3"
Value: {offset: 15234567, metadata: "consumer-3", timestamp: 1700000000000}
```

### Cluster Metadata (KRaft Log)
```
# Metadata stored in Raft log, replicated to all controller nodes
Record types:
  REGISTER_BROKER    → broker_id, host, port, rack
  TOPIC_RECORD       → topic_id, topic_name
  PARTITION_RECORD   → topic_id, partition_id, leader, ISR[], replicas[]
  PRODUCER_ID_RECORD → next_producer_id (for idempotent producers)
  CONFIG_RECORD      → config_key, config_value
```

---

## 7. API Design

### Producer API
```python
# Produce message
producer.produce(
    topic="user-events",
    key="user-123",              # Routes to consistent partition
    value=json.dumps(event),
    headers={"version": "1.0"},
    on_delivery=delivery_callback  # Async acknowledgement
)
producer.flush()                 # Wait for all in-flight messages

# Transactional produce
with producer.transaction():
    producer.produce("topic-A", key="k1", value="v1")
    producer.produce("topic-B", key="k2", value="v2")
    producer.commit_consumer_offsets(consumer, offsets)  # Consume-transform-produce
```

### Consumer API
```python
consumer = Consumer({
    'group.id': 'billing-service',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False  # Manual commit for exactly-once
})
consumer.subscribe(['user-events'])

while True:
    msg = consumer.poll(timeout=1.0)
    if msg:
        process(msg.value())
        consumer.commit(asynchronous=False)  # Manual commit after processing
```

### Admin API
```python
# Create topic
admin.create_topics([NewTopic(
    name="user-events",
    num_partitions=100,
    replication_factor=3,
    config={"retention.ms": "604800000",  # 7 days
            "compression.type": "lz4"}
)])

# Get consumer lag
for partition in consumer_group.partitions:
    lag = partition.high_watermark - partition.committed_offset
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: Single Partition Write Throughput
- Each partition is sequential on disk → max ~100 MB/sec write per partition
- **Solution**: More partitions = more parallelism; 100 partitions → 10 GB/sec aggregate write

### Bottleneck 2: Consumer Rebalancing Storm
- All consumers stop during rebalance → processing gap of 10-30 seconds
- **Solution**: Incremental cooperative rebalancing (Kafka 2.4+) — only reassigned partitions pause

### Bottleneck 3: ZooKeeper/KRaft Metadata Bottleneck
- Old Kafka: ZooKeeper handling all metadata for 1000s of partitions
- **Solution**: KRaft (Kafka Raft) — ZooKeeper eliminated; controller scales to 1M partitions

### Bottleneck 4: Consumer Offset Commit Overhead
- 1000 consumers × commit every message × 1000 partitions = huge offset commit load
- **Solution**: Batch offset commits; auto-commit every 5 seconds; use manual commit after processing batch

### Bottleneck 5: Cross-Region Replication Latency
- Synchronous replication across data centers adds 50-100ms RTT
- **Solution**: Async replication for cross-region (MirrorMaker 2); accept eventual consistency across regions

---

## 9. Trade-offs & Design Decisions

### Decision 1: Message Queue vs Message Stream
- **Queue (RabbitMQ)**: Message deleted after consumption; point-to-point; supports complex routing; low latency
- **Stream (Kafka)**: Log retained after consumption; pub-sub; replay capability; higher throughput
- **Choice**: Log-based stream (Kafka model) — enables replay, multiple independent consumers, event sourcing

### Decision 2: Push vs Pull Consumer Model
- **Push**: Broker pushes to consumer; consumer can be overwhelmed; broker manages flow control
- **Pull**: Consumer pulls at its own rate; natural backpressure; consumer controls prefetch size
- **Choice**: Pull — consumers pull batches, which prevents overwhelm and allows higher throughput

### Decision 3: KRaft vs ZooKeeper for Metadata
- **ZooKeeper**: Kafka's original metadata store; external dependency; limits partition count to ~200K
- **KRaft**: Built-in Raft; self-contained; scales to 1M+ partitions; simpler ops
- **Choice**: KRaft (Kafka 3.x default) — ZooKeeper is legacy; simplifies deployment

### Decision 4: Partition Count vs Overhead
- **More partitions**: Higher throughput, more parallelism, finer-grained rebalancing
- **Fewer partitions**: Lower overhead (file handles, replication round-trips), faster leader election
- **Rule of thumb**: Start with max(target_throughput ÷ per-partition_throughput, consumer_count)

### Decision 5: Exactly-Once Complexity vs Simplicity
- **Exactly-once**: Requires idempotent producer + transactional API + consumer reading committed offsets only
- **At-least-once + idempotent consumer**: Simpler; application deduplicates by message ID
- **Choice**: Exactly-once for financial transactions; at-least-once + idempotent consumer for most use cases

---

## 10. Key Interview Talking Points

### 1. Why Kafka Uses Pull Instead of Push
Pull allows each consumer to proceed at its own pace. A slow consumer doesn't cause the broker to buffer messages indefinitely — it just reads later (within retention window). With push, the broker must track each consumer's capacity and throttle sends accordingly — complex and fragile.

### 2. How Ordering Is Guaranteed
Ordering is guaranteed within a single partition. A message with key K always goes to the same partition (consistent hash of K). If you need all events for user-123 to be ordered, use user_id as the message key. Global ordering across partitions is not guaranteed — this is a fundamental trade-off for throughput.

### 3. The ISR Mechanism for Durability
ISR = replicas fully caught up with leader. If you configure `acks=all` and `min.insync.replicas=2`, the producer only gets acknowledged when the leader AND at least 1 follower have written the message. If the leader fails, a new leader is elected from ISR — guaranteed to have the message.

### 4. Consumer Group Rebalancing
Walk through the JoinGroup/SyncGroup protocol. The group coordinator (a broker) manages this. The group leader (first consumer to join) computes the assignment and sends it back. All members then receive their new partition assignments. During rebalance, all consumers pause — this is the "stop-the-world" problem that incremental cooperative rebalancing solves.

### 5. Log Compaction vs Retention
Regular retention: delete data older than 7 days. Log compaction: keep only the latest message per key, regardless of age. Compaction enables Kafka as a changelog store for database snapshots — consumers can rebuild state from compacted topic without needing all historical events.

### 6. Zero-Copy and Why It Matters
At 10M messages/sec × 1 KB = 10 GB/sec. Without zero-copy, data goes: disk → OS page cache → Kafka application → socket buffer → NIC. That's 2 copies through user space, requiring 20 GB/sec memory bandwidth. With sendfile, data goes directly from page cache to NIC — Kafka's application code never touches the data. This enables wire-speed throughput.

### 7. Exactly-Once Processing End-to-End
The hardest part: Kafka gives exactly-once delivery within Kafka (idempotent producer + transactions). But consumer-side processing (e.g., writing to a database) needs separate idempotency. Solution: include message offset in the database write as the idempotency key. This couples Kafka offset to database transaction — powerful but complex.

### 8. Kafka vs RabbitMQ vs SQS
- **Kafka**: Log retention, replay, high throughput (1M+ msg/sec), ordering per partition, complex to operate
- **RabbitMQ**: Message acknowledgement-based deletion, complex routing (exchanges/bindings), lower throughput, simpler for queue patterns
- **SQS**: Fully managed, at-least-once, no ordering (FIFO queue for ordered), limited to 300 TPS (standard) or 3K TPS (FIFO)
