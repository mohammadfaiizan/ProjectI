# 09 — Message Queues and Event Streaming

---

## Table of Contents
1. [Why Message Queues](#1-why-message-queues)
2. [Point-to-Point vs Publish-Subscribe](#2-point-to-point-vs-publish-subscribe)
3. [Queue vs Stream Semantics](#3-queue-vs-stream-semantics)
4. [Kafka Architecture](#4-kafka-architecture)
5. [Kafka Delivery Semantics](#5-kafka-delivery-semantics)
6. [Kafka Use Cases](#6-kafka-use-cases)
7. [Kafka Partitioning Strategy](#7-kafka-partitioning-strategy)
8. [Consumer Lag and Backpressure](#8-consumer-lag-and-backpressure)
9. [RabbitMQ Architecture](#9-rabbitmq-architecture)
10. [RabbitMQ Patterns](#10-rabbitmq-patterns)
11. [SQS vs Kafka vs RabbitMQ](#11-sqs-vs-kafka-vs-rabbitmq)
12. [Event-Driven Architecture](#12-event-driven-architecture)
13. [Event Sourcing Pattern](#13-event-sourcing-pattern)
14. [CQRS with Event Sourcing](#14-cqrs-with-event-sourcing)
15. [Saga Pattern](#15-saga-pattern)
16. [Message Ordering Guarantees](#16-message-ordering-guarantees)
17. [Message Deduplication](#17-message-deduplication)
18. [Dead Letter Queue Design](#18-dead-letter-queue-design)
19. [Backpressure and Flow Control](#19-backpressure-and-flow-control)
20. [Schema Registry and Message Versioning](#20-schema-registry-and-message-versioning)
21. [Stream Processing](#21-stream-processing)
22. [Lambda vs Kappa Architecture](#22-lambda-vs-kappa-architecture)
23. [Quick Reference](#23-quick-reference)

---

## 1. Why Message Queues

Message queues solve fundamental problems in distributed systems by acting as intermediaries between producers and consumers.

### Core Problems Solved

**Decoupling**
- Producer does not need to know about consumers — it just publishes to the queue
- Services can be deployed, scaled, or replaced independently
- Reduces tight coupling that causes cascading failures

```
Without queue:         With queue:
OrderService ──► PaymentService     OrderService ──► Queue ──► PaymentService
                 InventoryService                         └──► InventoryService
                 EmailService                             └──► EmailService
```

**Buffering / Load Leveling**
- Absorbs traffic spikes — queue acts as a buffer between bursty producers and slow consumers
- Example: E-commerce flash sale — 100k orders/second hit the queue, but order processing runs at 10k/second without dropping requests

```
Flash Sale Spike:
Producers: ──────█████████──────────  (spike)
Queue:     ──────█████████████──────  (absorbs spike)
Consumers: ──────█████████████──────  (steady rate)
```

**Asynchronous Processing**
- Request returns immediately; work happens in background
- User gets acknowledgment ("order placed") while actual processing occurs later
- Enables long-running tasks (video transcoding, report generation, email sending)

**Reliability**
- Messages persist on disk — if consumer crashes, messages are not lost
- At-least-once delivery guarantees: retry on failure
- Acknowledgment-based: message stays in queue until consumer explicitly ACKs

**Rate Limiting / Throttling**
- Control how fast consumers pull from the queue
- Protect downstream services from overload

### Trade-offs of Introducing Queues

| Benefit | Trade-off |
|---|---|
| Decoupling | Added operational complexity |
| Async processing | Harder to debug end-to-end flows |
| Spike absorption | Eventual consistency (not real-time) |
| Fault tolerance | Message ordering becomes challenging |
| Scalability | Monitoring consumer lag required |

---

## 2. Point-to-Point vs Publish-Subscribe

### Point-to-Point (P2P) / Queue Model

- One producer, one consumer per message
- Message is consumed exactly once and deleted
- Load balanced naturally across multiple consumers
- Use case: task distribution, work queues

```
Producer ──► [Queue] ──► Consumer A  (or)
                    └──► Consumer B  (competing consumers — only one gets the message)
```

### Publish-Subscribe (Pub/Sub) / Topic Model

- One producer, many consumers (subscribers)
- Each subscriber gets its own copy of the message
- Decouples publishers from subscribers completely
- Use case: event notification, broadcasting, fan-out

```
Publisher ──► [Topic] ──► Subscriber A (gets message)
                     └──► Subscriber B (gets message)
                     └──► Subscriber C (gets message)
```

### Hybrid Model (Kafka)

Kafka combines both: topics with consumer groups achieve both patterns.

```
Topic ──► Consumer Group A (acts as P2P within group — one consumer per partition)
     └──► Consumer Group B (acts as P2P within group)
     └──► Consumer Group C

Each group independently reads all messages (pub/sub between groups)
Within a group, each partition assigned to one consumer (P2P load balancing)
```

### Comparison Table

| Feature | Point-to-Point | Publish-Subscribe |
|---|---|---|
| Message delivery | One consumer only | All subscribers |
| Competing consumers | Yes (load balancing) | No (each gets copy) |
| Message retention | Deleted after consumption | Depends on subscription |
| Use case | Task queues, work distribution | Event notification, fan-out |
| Examples | SQS, RabbitMQ queue | Kafka topics, SNS |

---

## 3. Queue vs Stream Semantics

This is a critical distinction for interviews.

### Queue Semantics

- Message is **deleted** after acknowledgment
- Consumer "pops" message from queue — gone forever
- State is ephemeral: queue depth = pending work
- Cannot replay old messages
- Multiple consumers compete for the same messages
- Examples: SQS, RabbitMQ, ActiveMQ

```
Queue lifecycle:
Message arrives → stored in queue → consumer pulls → consumer ACKs → message deleted
```

### Stream Semantics

- Messages are **retained** in an append-only log
- Consumers track their own **offset** (position in the log)
- Multiple consumer groups read independently — each at their own pace
- **Replay** is possible: reset offset to re-read historical data
- Log compaction can reduce storage while keeping latest state per key
- Examples: Kafka, Kinesis, Pulsar, Redis Streams

```
Stream lifecycle:
Message arrives → appended to log → consumer reads at offset → consumer advances offset
                                  → message stays in log for N days
```

### Key Difference

```
Queue:    [msg1][msg2][msg3] → Consumer reads msg1 → [msg2][msg3]
                                                      msg1 is GONE

Stream:   offset=0  offset=1  offset=2
          [msg1]    [msg2]    [msg3]
          Consumer A reads at offset 1 → advances to offset 2
          Consumer B reads at offset 0 → advances to offset 1
          All messages STILL IN LOG
```

### Comparison Table

| Aspect | Queue | Stream |
|---|---|---|
| Message retention | Deleted after ACK | Retained by time/size policy |
| Multiple consumers | Compete for messages | Independent offsets per group |
| Replay | Not possible | Possible (reset offset) |
| Ordering | Best-effort (FIFO) | Strict per-partition |
| Use case | Task distribution | Event streaming, audit log |
| Storage model | Transient | Persistent log |
| Backpressure | Natural (queue depth) | Consumer lag monitoring |

---

## 4. Kafka Architecture

### Core Concepts

**Topic**
- Named channel for messages
- Divided into one or more partitions
- Producers write to topics; consumers read from topics

**Partition**
- Ordered, immutable sequence of records
- Unit of parallelism in Kafka
- Each partition is a separate log on disk
- Assigned a sequential offset per message

**Offset**
- Unique, sequential ID of a message within a partition
- Consumer tracks its offset independently
- Can commit offset to Kafka or manage externally

**Broker**
- Kafka server that stores and serves messages
- Each broker hosts some partitions
- Cluster of brokers distributes load

**Cluster Topology**

```
Kafka Cluster (3 brokers):

Broker 1:  Topic-A Partition 0 (leader), Topic-A Partition 1 (follower)
Broker 2:  Topic-A Partition 1 (leader), Topic-A Partition 0 (follower)
Broker 3:  Topic-A Partition 0 (follower), Topic-A Partition 1 (follower)
```

**Replication**
- Each partition has one leader and N-1 followers (replicas)
- Producers write to leader; followers replicate
- ISR (In-Sync Replicas): replicas that are caught up with leader
- If leader fails, a follower from ISR becomes new leader

**Producer**
```
Producer settings:
acks=0:   No wait — fire and forget (lowest latency, highest data loss risk)
acks=1:   Wait for leader ACK (moderate durability)
acks=all: Wait for all ISR ACK (highest durability, higher latency)
```

**Consumer Groups**
- Group of consumers that jointly consume a topic
- Each partition assigned to exactly one consumer in the group
- Enables horizontal scaling: add consumers to scale consumption
- Max parallelism = number of partitions

```
Topic with 4 partitions:
Partition 0 → Consumer 1
Partition 1 → Consumer 2
Partition 2 → Consumer 3
Partition 3 → Consumer 4

If 5th consumer joins: one consumer idle (more consumers than partitions)
If only 2 consumers:
  Consumer 1 → Partition 0, Partition 1
  Consumer 2 → Partition 2, Partition 3
```

### ZooKeeper vs KRaft

**ZooKeeper (legacy)**
- External cluster managing Kafka metadata
- Controller election, topic configuration, consumer group offsets
- Operational overhead: separate ZooKeeper cluster needed
- Bottleneck at scale

**KRaft (Kafka Raft Metadata, Kafka 2.8+)**
- Kafka manages its own metadata using Raft consensus
- Eliminates ZooKeeper dependency
- Faster controller failover
- Simpler deployment

```
ZooKeeper mode:
[Kafka Brokers] ←→ [ZooKeeper Ensemble (3+ nodes)]

KRaft mode:
[Kafka Brokers (some act as controllers)] — self-contained
```

### Log Storage

```
Topic: orders  (4 partitions, replication factor 3)

Partition 0 log file:
00000000000000000000.log  → messages 0–999999
00000000000001000000.log  → messages 1000000+
00000000000000000000.index → sparse index for fast offset lookup
```

---

## 5. Kafka Delivery Semantics

### At-Most-Once

- Message delivered 0 or 1 times — never duplicated
- Producer: `acks=0`, no retry
- Consumer: commit offset before processing
- Risk: if processing fails after commit, message is lost
- Use case: metrics, analytics where losing some data is acceptable

```
Sequence:
1. Consumer reads message at offset N
2. Consumer commits offset N+1  ← offset advanced
3. Consumer processes message
4. If step 3 crashes → message lost, offset already advanced
```

### At-Least-Once

- Message delivered 1 or more times — never lost, possibly duplicated
- Producer: `acks=all` + retry on failure
- Consumer: commit offset after processing
- Risk: duplicate processing if crash between process and commit
- Requires idempotent consumers
- Most common default

```
Sequence:
1. Consumer reads message at offset N
2. Consumer processes message
3. Consumer commits offset N+1
4. If step 3 crashes → message re-processed on restart (duplicate)
```

### Exactly-Once (EOS)

- Message delivered exactly once — hardest to achieve
- Kafka 0.11+ supports idempotent producers and transactional APIs
- Producer idempotency: each message has sequence number; broker deduplicates retries
- Transactions: atomic write across multiple partitions

```java
// Kafka EOS producer
producer.initTransactions();
producer.beginTransaction();
producer.send(new ProducerRecord<>("output-topic", key, value));
producer.commitTransaction(); // atomic
```

**Exactly-Once in Kafka Streams**
- `processing.guarantee=exactly_once_v2`
- Reads, processes, and writes atomically

| Semantic | Producer | Consumer | Use Case |
|---|---|---|---|
| At-most-once | acks=0, no retry | Commit before process | Metrics, logs |
| At-least-once | acks=all, retry | Commit after process | Most applications |
| Exactly-once | Idempotent + transactions | Transactional reads | Financial, billing |

---

## 6. Kafka Use Cases

### Event Streaming / Real-Time Pipelines

```
IoT Sensors → Kafka → Stream Processor → Dashboard
                 └──→ Database sink
                 └──→ Alert service
```

### Log Aggregation

- Collect logs from hundreds of services into Kafka topics
- Replace traditional log shippers (Logstash, Fluentd) with Kafka as backbone
- Downstream consumers: Elasticsearch, S3, analytics

```
Service A logs → Kafka (logs topic) → Elasticsearch (search)
Service B logs →                    → S3 (archival)
Service C logs →                    → Splunk (monitoring)
```

### Change Data Capture (CDC)

- Capture every change (INSERT/UPDATE/DELETE) from a database
- Debezium: connects to DB WAL (Write-Ahead Log) and publishes to Kafka
- Use cases: cache invalidation, search index updates, audit logs, data sync

```
PostgreSQL WAL → Debezium → Kafka (users-cdc topic) → Redis cache invalidation
                                                     → Elasticsearch update
                                                     → Analytics DB
```

### Event Sourcing

- Store all events as the source of truth in Kafka
- Rebuild state by replaying events

### Microservices Decoupling

- Services communicate via events without direct API calls
- OrderService publishes `order.placed` → InventoryService, PaymentService react

---

## 7. Kafka Partitioning Strategy

Partitioning determines how messages are distributed across partitions.

### Key-Based Partitioning

- Producer sends message with a key
- Kafka hashes the key: `partition = hash(key) % num_partitions`
- All messages with the same key go to the same partition → guaranteed ordering per key
- Example: all events for user_id=123 in same partition

```python
# Messages with same key → same partition
producer.send("orders", key=b"user123", value=order_event)
producer.send("orders", key=b"user123", value=another_event)
# Both go to same partition → ordered
```

**Risk:** Hot partitions if key distribution is skewed (e.g., one celebrity user generates 90% of events)

### Round-Robin (No Key)

- Messages distributed evenly across partitions
- No ordering guarantees across messages
- Best for even load distribution
- Example: logging, metrics

### Custom Partitioner

```java
public class CustomPartitioner implements Partitioner {
    public int partition(String topic, Object key, byte[] keyBytes,
                         Object value, byte[] valueBytes, Cluster cluster) {
        // Custom logic: route VIP orders to partition 0
        Order order = (Order) value;
        if (order.isVip()) return 0;
        return Math.abs(key.hashCode()) % cluster.partitionCountForTopic(topic);
    }
}
```

### Partition Count Considerations

| Factor | Recommendation |
|---|---|
| Throughput | More partitions = more parallelism |
| Consumer parallelism | Max consumers = num_partitions |
| Broker load | Partitions distributed across brokers |
| Latency | More partitions → slightly higher end-to-end latency |
| Rebalancing | More partitions → longer rebalancing |

Rule of thumb: `num_partitions = target_throughput / throughput_per_consumer`

---

## 8. Consumer Lag and Backpressure

### Consumer Lag

Consumer lag = latest offset in partition - consumer's current offset

```
Partition 0:
  Latest offset: 10000
  Consumer offset: 9500
  LAG: 500 messages

High lag = consumer is falling behind
Zero lag = consumer is caught up
```

**Monitoring lag:**
```bash
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --describe --group my-consumer-group
# Shows: LAG column per partition
```

### Causes of Consumer Lag

1. Slow processing logic
2. Consumer crash/restart (time to rebalance)
3. Traffic spike (burst of messages)
4. Insufficient consumer instances

### Dealing with Lag

1. **Scale consumers**: Add more consumer instances (up to num_partitions)
2. **Optimize processing**: Batch DB writes, async I/O
3. **Increase partitions**: More parallelism (note: cannot decrease partitions)
4. **Separate fast/slow consumers**: Different topics for different processing speeds

### Backpressure in Kafka

Kafka does not push messages to consumers — consumers **pull** on their own schedule. This is inherent backpressure.

```
Consumer controls rate:
max.poll.records = 500       → process 500 messages per poll
fetch.min.bytes = 1024       → wait until 1KB available before fetching
fetch.max.wait.ms = 500      → max 500ms to wait for min bytes
```

### Consumer Group Rebalancing

When consumer joins/leaves/crashes, Kafka reassigns partitions:
1. Group coordinator detects change
2. All consumers stop consuming (stop-the-world)
3. Group coordinator assigns partitions
4. Consumers resume

Mitigating rebalance impact:
- Incremental Cooperative Rebalancing (Kafka 2.4+): only reassign affected partitions
- `session.timeout.ms`: how long before consumer considered dead
- `heartbeat.interval.ms`: how often consumer sends heartbeat

---

## 9. RabbitMQ Architecture

RabbitMQ implements AMQP (Advanced Message Queuing Protocol).

### Core Components

**Exchange**
- Receives messages from producers
- Routes messages to queues based on routing rules
- Does NOT store messages

**Queue**
- Stores messages waiting to be consumed
- Durable queues survive broker restart
- Exclusive queues deleted when connection closes

**Binding**
- Link between exchange and queue
- Routing key used to match messages to queues

**Message Flow**

```
Producer → Exchange → (routing/binding rules) → Queue → Consumer
```

### Exchange Types

**Direct Exchange**
- Routes to queues where binding key = routing key (exact match)
- Use case: task routing by type

```
Message: routing_key="order.process"
Binding: queue="order-processing" → binding_key="order.process"
Result: message goes to "order-processing" queue
```

**Fanout Exchange**
- Broadcasts to ALL bound queues regardless of routing key
- Use case: notifications, event broadcasting

```
Exchange ──► Queue A (all messages)
        └──► Queue B (all messages)
        └──► Queue C (all messages)
```

**Topic Exchange**
- Routes based on wildcard routing key patterns
- `*` matches one word; `#` matches zero or more words
- Use case: selective subscriptions

```
Binding patterns:
  "orders.*"   → matches "orders.created", "orders.shipped"
  "orders.#"   → matches "orders.created.us", "orders.shipped.eu"
  "*.error"    → matches "payments.error", "auth.error"
```

**Headers Exchange**
- Routes based on message header attributes instead of routing key
- More flexible but slower than direct/topic

### Exchange Type Comparison

| Exchange | Routing Logic | Use Case |
|---|---|---|
| Direct | Exact key match | Task queues, RPC |
| Fanout | Broadcast to all | Notifications, events |
| Topic | Wildcard pattern match | Selective subscriptions |
| Headers | Header attributes | Complex routing logic |

---

## 10. RabbitMQ Patterns

### Dead Letter Queue (DLQ)

Messages land in DLQ when:
1. Message rejected (basic.reject / basic.nack)
2. Message TTL expired
3. Queue length limit reached

```
Normal queue: x-dead-letter-exchange → DLX
DLX routes to → Dead Letter Queue

Producer → [main-queue] → Consumer (fails)
                       → [DLQ] → Alert / manual inspection
```

```python
# Python pika example
channel.queue_declare(
    queue='main-queue',
    arguments={
        'x-dead-letter-exchange': 'dlx',
        'x-message-ttl': 60000,  # 60 second TTL
        'x-max-length': 10000
    }
)
```

### Delayed Messages

RabbitMQ does not natively support delayed messages; patterns to achieve this:
1. **TTL + DLQ trick**: Publish to queue with TTL. When expired, dead-lettered to processing queue.
2. **RabbitMQ Delayed Message Plugin**: Direct delay header support.

```
Producer → [delay-queue TTL=30s] → (expires) → DLX → [processing-queue] → Consumer
```

### Priority Queues

- Queues with `x-max-priority` setting (1–255)
- Higher priority messages processed first
- Use case: premium user requests before free-tier

```python
channel.queue_declare(
    queue='priority-queue',
    arguments={'x-max-priority': 10}
)
# Publish with priority
channel.basic_publish(
    exchange='',
    routing_key='priority-queue',
    properties=pika.BasicProperties(priority=8),  # high priority
    body='VIP order'
)
```

### Retry Pattern with RabbitMQ

```
[main-queue] → Consumer fails
            → Reject with requeue=False
            → Message goes to [retry-queue TTL=5s]
            → After 5s, dead-lettered to [main-queue]
            → After N retries, goes to [DLQ]
```

---

## 11. SQS vs Kafka vs RabbitMQ

| Feature | Amazon SQS | Apache Kafka | RabbitMQ |
|---|---|---|---|
| **Type** | Managed queue | Distributed log | Message broker (AMQP) |
| **Message retention** | Up to 14 days | Configurable (forever) | Until consumed |
| **Delivery semantics** | At-least-once | Configurable (EOS available) | At-least-once (with ACK) |
| **Ordering** | Per message group (FIFO) | Per partition | Per queue |
| **Throughput** | High (auto-scaled) | Very high (millions/sec) | High (50-100k/sec) |
| **Replay** | No | Yes | No |
| **Consumer model** | Pull (polling) | Pull (offset-based) | Push (or pull) |
| **Routing** | Simple (queue name) | Topic + partition | Exchanges + bindings |
| **Protocol** | HTTP/HTTPS (AWS SDK) | Kafka protocol | AMQP, MQTT, STOMP |
| **Deployment** | Fully managed (AWS) | Self-hosted or MSK | Self-hosted or CloudAMQP |
| **Max message size** | 256 KB | 1 MB (default) | 128 MB |
| **DLQ** | Native support | Manual | Native support |
| **Geo-replication** | Multi-region (SNS+SQS) | MirrorMaker 2 | Federated exchanges |
| **Best for** | AWS-native async tasks | Event streaming, CDC | Complex routing, RPC |

### When to Choose What

**Choose SQS when:**
- Deep in AWS ecosystem
- Simple task queues without complex routing
- No need for replay or event streaming
- Serverless / Lambda integration

**Choose Kafka when:**
- Event streaming at scale
- Multiple consumers need same events
- Need message replay
- CDC, log aggregation, real-time analytics

**Choose RabbitMQ when:**
- Complex routing logic (topic, header exchanges)
- RPC patterns
- Per-message TTL, priority queues
- Language-agnostic messaging (AMQP standard)

---

## 12. Event-Driven Architecture

### Events vs Commands

| Aspect | Event | Command |
|---|---|---|
| What it represents | Something that happened | Request to do something |
| Tense | Past tense | Imperative |
| Example | `OrderPlaced`, `UserRegistered` | `PlaceOrder`, `RegisterUser` |
| Direction | Broadcast | Targeted |
| Sender knows receiver | No | Yes |
| Receiver obligation | Optional (can ignore) | Must process |
| Coupling | Low | Higher |

### Choreography vs Orchestration

**Choreography**
- Services react to events independently
- No central coordinator
- Services subscribe to events they care about
- Pros: decoupled, resilient
- Cons: hard to trace full flow, logic distributed everywhere

```
OrderService publishes OrderPlaced
  → PaymentService subscribes → charges customer → publishes PaymentProcessed
  → InventoryService subscribes → reserves stock → publishes StockReserved
  → ShippingService subscribes to both → ships → publishes OrderShipped
```

**Orchestration**
- Central orchestrator tells each service what to do
- Orchestrator knows the full workflow
- Pros: easy to trace, central error handling
- Cons: orchestrator becomes bottleneck, higher coupling

```
OrderOrchestrator:
  1. Call PaymentService.charge()
  2. Call InventoryService.reserve()
  3. Call ShippingService.ship()
  4. Publish OrderCompleted
```

### Comparison

| Aspect | Choreography | Orchestration |
|---|---|---|
| Coupling | Low | Higher (to orchestrator) |
| Visibility | Hard to see full flow | Clear workflow |
| Error handling | Distributed, complex | Centralized |
| Scalability | Better | Orchestrator can bottleneck |
| Testing | Harder | Easier |
| Use case | Simple events, microservices | Complex workflows, sagas |

---

## 13. Event Sourcing Pattern

### Core Concept

Instead of storing current state, store every **event** that led to that state.

```
Traditional: users table → {id: 1, balance: 150, name: "Alice"}

Event Sourcing:
events table:
  {type: "AccountCreated", user_id: 1, initial_balance: 100}
  {type: "MoneyDeposited", user_id: 1, amount: 100}
  {type: "MoneyWithdrawn", user_id: 1, amount: 50}

Current state = replay all events: 100 + 100 - 50 = 150
```

### Benefits

1. **Complete audit trail**: Every change is recorded
2. **Temporal queries**: "What was the state at time T?"
3. **Replay**: Reconstruct state from scratch
4. **Event-driven integration**: Other services subscribe to domain events
5. **Debugging**: Reproduce bugs by replaying events

### Challenges

1. **Query complexity**: Getting current state requires replaying events (use snapshots)
2. **Event schema evolution**: Old events must remain compatible
3. **Storage growth**: Events accumulate over time
4. **Learning curve**: Different mental model

### Snapshots

To avoid replaying all events from the beginning:
```
Snapshot at event #1000: {balance: 500}
New events: #1001, #1002, #1003

Current state = snapshot + replay events after snapshot
```

### Event Store Design

```
events:
  id         UUID
  stream_id  VARCHAR  (aggregate ID)
  type       VARCHAR  (event type)
  data       JSONB    (event payload)
  metadata   JSONB    (correlation ID, causation ID, timestamp)
  version    INT      (sequence within stream)
  created_at TIMESTAMP
```

---

## 14. CQRS with Event Sourcing

### CQRS Overview

**Command Query Responsibility Segregation** — separate the read and write models.

```
Write Side:               Read Side:
Command → Aggregate       Query → Read Model (projection)
        → Event Store     
        → Publish events → Update read model
```

### Why CQRS + Event Sourcing?

- Write side (event store) optimized for appending events
- Read side (projections) optimized for specific query patterns
- Multiple read models from same event stream (different views)

```
Event Stream: OrderPlaced, PaymentProcessed, OrderShipped

Read Models:
  OrderSummaryView: {order_id, status, total}
  CustomerOrderHistoryView: {customer_id, orders: [...]}
  FinanceReportView: {date, revenue, transactions}
```

### CQRS Architecture

```
HTTP POST /orders              HTTP GET /orders/123
     │                               │
     ▼                               ▼
[Command Handler]           [Query Handler]
     │                               │
     ▼                               ▼
[Event Store]               [Read DB / Cache]
     │                               ▲
     └──► Event Bus ─────────────────┘
              │          (Projection updater)
              ▼
       Other Services
```

### Trade-offs

| Benefit | Challenge |
|---|---|
| Optimized read/write paths | Eventual consistency between write and read |
| Multiple specialized read models | Increased complexity |
| Independent scaling of reads/writes | More infrastructure |
| Full audit trail | Projection rebuild time |

---

## 15. Saga Pattern

### The Problem

Distributed transactions across microservices: how to maintain consistency without 2PC?

### 2-Phase Commit (2PC) — Why Not Use It

```
Phase 1 (Prepare): Coordinator asks all participants to prepare
Phase 2 (Commit): If all prepared, coordinator tells all to commit

Problems:
- Coordinator is SPOF
- Blocking: participants lock resources during prepare phase
- Poor performance
- Not suitable for microservices across different DBs
```

### Saga Pattern

Break distributed transaction into sequence of local transactions. Each step publishes an event or message triggering the next step. If a step fails, compensating transactions undo previous steps.

### Choreography Saga

```
OrderService: OrderPlaced event
  → PaymentService: PaymentProcessed event
    → InventoryService: StockReserved event
      → ShippingService: OrderFulfilled event

On failure:
InventoryService fails → StockReservationFailed event
  → PaymentService compensates: PaymentRefunded
    → OrderService compensates: OrderCancelled
```

### Orchestration Saga

```
OrderSaga Orchestrator:
  1. → PaymentService: charge($100)
     ← PaymentCompleted
  2. → InventoryService: reserve(item)
     ← StockReserved
  3. → ShippingService: ship()
     ← Shipped
  
On failure at step 2:
  2. ← StockReservationFailed
  → PaymentService: refund($100) [compensating transaction]
  → OrderService: mark order failed
```

### Comparison Table

| Aspect | 2PC | Choreography Saga | Orchestration Saga |
|---|---|---|---|
| Coordination | Central coordinator | None (events) | Central orchestrator |
| Coupling | All participants locked | Low | Medium (to orchestrator) |
| Performance | Poor (blocking) | Good | Good |
| Consistency | Strong | Eventual | Eventual |
| Complexity | High (protocol) | Medium (event tracking) | Lower (visible workflow) |
| SPOF | Coordinator | No | Orchestrator |
| Use case | Avoid in microservices | Simple sagas | Complex sagas |

---

## 16. Message Ordering Guarantees

### Global Ordering

- All messages across ALL partitions/consumers ordered by time
- Very hard to achieve at scale — requires single partition
- Throughput limited to one consumer
- Kafka: use single partition for global order

### Per-Partition Ordering (Kafka)

- Within a partition, messages are strictly ordered
- Across partitions, no ordering guarantee
- Best practice: use keys to route related messages to same partition

```
user_id=Alice → partition 0 → [event1, event2, event3] (ordered)
user_id=Bob   → partition 1 → [event1, event2] (ordered)
No ordering guarantee between Alice's events and Bob's events
```

### Ordering Challenges

**Out-of-order delivery in at-least-once systems:**
- Consumer crashes and reprocesses → old messages re-delivered
- Solution: sequence numbers or timestamps in events

**Network reordering:**
- In-flight messages can arrive out of order
- Solution: per-key sequences + idempotency

### Sequence Numbers

```json
{
  "event_id": "uuid-123",
  "user_id": "alice",
  "sequence": 42,
  "type": "OrderPlaced",
  "timestamp": "2024-01-01T10:00:00Z"
}
```

Consumer rejects event with sequence < expected → detects out-of-order.

---

## 17. Message Deduplication

### The Problem

At-least-once delivery means duplicates. Consumers must handle idempotently.

### Idempotency Keys

Each message carries a unique key. Consumer checks if key was already processed.

```python
def process_message(message):
    idempotency_key = message['event_id']
    
    # Check Redis/DB if already processed
    if redis.exists(f"processed:{idempotency_key}"):
        return  # Duplicate, skip
    
    # Process message
    do_processing(message)
    
    # Mark as processed (with TTL to bound storage)
    redis.setex(f"processed:{idempotency_key}", 86400, "1")
```

### Deduplication Window

- Store processed IDs for a time window (e.g., 24 hours)
- Messages older than window assumed safe (duplicates unlikely)
- SQS FIFO queues: 5-minute deduplication window built-in

### Database Upsert Pattern

```sql
-- Idempotent: INSERT OR UPDATE based on event_id
INSERT INTO order_events (event_id, order_id, status, updated_at)
VALUES ($1, $2, $3, NOW())
ON CONFLICT (event_id) DO NOTHING;
```

### Kafka Idempotent Producer

- `enable.idempotence=true`
- Producer gets a PID (Producer ID) and adds sequence numbers
- Broker deduplicates within a session

---

## 18. Dead Letter Queue Design

### Purpose

- Hold messages that cannot be successfully processed
- Prevent "poison pill" messages from blocking the queue
- Enable investigation and reprocessing

### Dead Letter Triggers

1. Processing error (exception thrown)
2. Max retry attempts exceeded
3. Message TTL expired
4. Message malformed / schema validation failed
5. Consumer rejected message explicitly

### DLQ Architecture

```
[Main Queue] → Consumer → Success: ACK message
                       → Failure: 
                           retry_count < max_retries → back to queue (with backoff)
                           retry_count >= max_retries → publish to [DLQ]

[DLQ] → Alert/PagerDuty
      → Manual inspection dashboard
      → Reprocess job (after fix deployed)
```

### Retry Strategy

**Exponential Backoff:**
```
Attempt 1: wait 1s
Attempt 2: wait 2s
Attempt 3: wait 4s
Attempt 4: wait 8s
Attempt 5: → DLQ
```

**Jitter** (prevents thundering herd):
```
wait = base_delay * 2^attempt + random(0, 1000ms)
```

### DLQ Message Schema

```json
{
  "original_message": { ... },
  "original_queue": "order-processing",
  "failure_reason": "PaymentServiceUnavailable",
  "failure_timestamp": "2024-01-01T10:00:00Z",
  "retry_count": 5,
  "first_failure": "2024-01-01T09:55:00Z",
  "stack_trace": "..."
}
```

### DLQ Reprocessing

After fixing the bug:
1. Inspector examines DLQ messages
2. Filtered subset republished to original queue
3. Monitor for successful processing

---

## 19. Backpressure and Flow Control

### What is Backpressure?

Mechanism to signal upstream producers to slow down when downstream is overloaded.

### Kafka Pull Model (Natural Backpressure)

- Consumers pull at their own rate
- If consumer is slow: lag increases (visible metric)
- Producer keeps writing — queue absorbs burst

```
Fast producer → Kafka (absorbs burst)
Slow consumer → reads at its own pace
               → lag increases
               → scale consumers or optimize
```

### RabbitMQ Flow Control

- RabbitMQ monitors memory and disk
- If memory > threshold: sends `channel.flow(false)` to producers
- Producer must stop sending until `channel.flow(true)` received

### TCP-Level Backpressure

- Receiver's TCP window shrinks → sender slows
- Reactive Streams (Java): `request(n)` — consumer asks for N items

### Reactive Streams Backpressure

```
Subscriber → request(100) → Publisher sends 100 items
Subscriber processes... → request(50) → Publisher sends 50 items
Subscriber slow → request(0) → Publisher pauses
```

### Rate Limiting Patterns

**Token Bucket:** Consumer releases tokens; producer can only send if token available.

**Leaky Bucket:** Messages enter at any rate; exit at fixed rate (leaks).

### Circuit Breaker for Backpressure

```
Consumer detects overload → opens circuit breaker
Producer receives error   → backs off
Consumer recovers         → circuit half-open → test request
Test succeeds             → circuit closed → normal flow
```

---

## 20. Schema Registry and Message Versioning

### Why Schema Registry?

- Producers and consumers need to agree on message format
- Schema changes must be backwards/forwards compatible
- Without registry: tight coupling, brittle pipelines

### Confluent Schema Registry

- Central store for Avro/Protobuf/JSON schemas
- Each topic associated with a schema (key schema + value schema)
- Schemas versioned; compatibility enforced

```
Producer → serializes with schema v1 → Kafka (stores schema ID in message)
Consumer → reads schema ID → fetches schema from registry → deserializes
```

### Avro Schema Example

```json
{
  "type": "record",
  "name": "Order",
  "fields": [
    {"name": "order_id", "type": "string"},
    {"name": "user_id", "type": "string"},
    {"name": "amount", "type": "double"},
    {"name": "currency", "type": {"type": "string", "default": "USD"}}
  ]
}
```

### Compatibility Modes

| Mode | Description | Safe changes |
|---|---|---|
| BACKWARD | New schema can read old data | Add optional fields with default |
| FORWARD | Old schema can read new data | Remove optional fields |
| FULL | Both backward + forward | Add/remove optional fields with default |
| NONE | No compatibility check | Any change |

### Schema Evolution Rules

```
Safe (BACKWARD compatible):
  ✓ Add optional field with default value
  ✓ Remove field that had no default
  ✗ Remove required field
  ✗ Change field type (int → string)
  ✗ Rename field
```

### Protobuf Advantages

- Language-neutral binary format
- Field numbers enable safe schema evolution
- Smaller payload than JSON (3–10x)
- More verbose schema definition but better tooling

---

## 21. Stream Processing

### Apache Flink

- True streaming (event-at-a-time), not micro-batch
- Stateful processing with fault-tolerant checkpoints
- Low latency (milliseconds)
- Exactly-once semantics
- Use case: real-time fraud detection, CEP (complex event processing)

```java
// Flink: count orders per user in 5-minute windows
DataStream<Order> orders = env.addSource(kafkaSource);
orders
  .keyBy(order -> order.getUserId())
  .window(TumblingEventTimeWindows.of(Time.minutes(5)))
  .aggregate(new OrderCounter())
  .addSink(redisSink);
```

### Apache Spark Streaming (Structured Streaming)

- Micro-batch model (treats stream as series of small batches)
- Higher latency than Flink (seconds)
- Excellent for batch + streaming unified code
- Stronger SQL/DataFrame API

### Kafka Streams

- Library (not cluster) — runs inside your application
- Processes Kafka topics directly
- Good for microservices that need simple stream processing
- Exactly-once semantics built-in

```java
// Kafka Streams: word count
KStream<String, String> text = builder.stream("input-topic");
KTable<String, Long> wordCounts = text
  .flatMapValues(v -> Arrays.asList(v.split("\\s+")))
  .groupBy((k, v) -> v)
  .count();
wordCounts.toStream().to("output-topic");
```

### Comparison

| Feature | Flink | Spark Streaming | Kafka Streams |
|---|---|---|---|
| Processing model | True streaming | Micro-batch | True streaming |
| Latency | Milliseconds | Seconds | Milliseconds |
| State management | Built-in (RocksDB) | In-memory / checkpoint | Built-in (RocksDB) |
| Deployment | Separate cluster | Spark cluster | Embedded in app |
| Exactly-once | Yes | Yes | Yes |
| SQL support | Flink SQL | SparkSQL | KSQL |
| Use case | Real-time, CEP | Batch + streaming | Microservices |

---

## 22. Lambda vs Kappa Architecture

### Lambda Architecture

Three layers processing the same data:

```
Data Source → [Batch Layer] → Batch Views  ─┐
           → [Speed Layer] → Real-time Views─┴→ Serving Layer → Query
           → [Serving Layer] merges views
```

- **Batch Layer**: Processes all historical data periodically (MapReduce, Spark)
- **Speed Layer**: Processes recent data in real-time (Flink, Storm)
- **Serving Layer**: Merges batch + speed views for query

**Pros:** Accurate batch processing; speed layer for recency
**Cons:** Two codebases (batch + speed); complex merging logic; reprocessing required for corrections

### Kappa Architecture

Single streaming pipeline handles both real-time and historical:

```
Data Source → Kafka (retention: months) → Stream Processor → Serving Layer → Query

Reprocessing: reset Kafka offset → replay from beginning
```

- One codebase for both real-time and historical
- Kafka as long-term storage + replay mechanism
- Simpler operations

**Pros:** One codebase, simpler, no batch/speed split
**Cons:** Stream reprocessing slower than batch for very large history; Kafka storage costs

### Comparison

| Aspect | Lambda | Kappa |
|---|---|---|
| Complexity | High (two systems) | Lower |
| Latency | Low (speed layer) | Low |
| Historical processing | Efficient (batch) | Kafka replay (slower) |
| Correctness | Strong (batch recompute) | Depends on replay fidelity |
| Operational cost | High | Lower |
| Use case | Large-scale analytics | Event-driven, microservices |

---

## 23. Quick Reference

### Queue vs Stream Comparison

| Feature | Queue (SQS, RabbitMQ) | Stream (Kafka, Kinesis) |
|---|---|---|
| After consumption | Message deleted | Message retained |
| Replay | No | Yes |
| Multiple consumers | Compete (one gets it) | Independent offsets |
| Ordering | Limited (FIFO queues) | Per-partition |
| Storage | Transient | Persistent log |
| Use case | Task distribution | Event streaming, CDC |

### Messaging Pattern Decision Tree

```
Need replay / audit log?
  YES → Kafka / Kinesis
  NO  →
    Complex routing logic?
      YES → RabbitMQ (exchanges/bindings)
      NO  →
        AWS ecosystem?
          YES → SQS (standard or FIFO)
          NO  →
            Simple task queue?
              YES → SQS / RabbitMQ direct exchange
              NO  → Kafka for scale
```

### Delivery Semantics Summary

| Semantic | Lost? | Duplicates? | Implementation |
|---|---|---|---|
| At-most-once | Possible | No | Commit before process |
| At-least-once | No | Possible | Commit after process + idempotent consumer |
| Exactly-once | No | No | Kafka transactions / idempotent producer |

### Interview Cheat Sheet — Top Questions

1. **Why Kafka over RabbitMQ?** → Replay, event log, higher throughput, CDC
2. **How does Kafka guarantee ordering?** → Per-partition only; use keys for related messages
3. **What is consumer lag?** → Gap between latest offset and consumer offset
4. **Exactly-once in Kafka** → Idempotent producer + transactions
5. **Choreography vs orchestration** → Decoupled events vs central coordinator
6. **Event sourcing** → Store events, not state; replay to rebuild
7. **CQRS** → Separate write (command) and read (query) models
8. **Saga vs 2PC** → 2PC is blocking/SPOF; saga is eventual consistency with compensation
9. **Schema registry** → Enforce schema compatibility; decouple producer/consumer schema changes
10. **Dead letter queue** → Hold unprocessable messages; retry with backoff; alert on DLQ growth
```
