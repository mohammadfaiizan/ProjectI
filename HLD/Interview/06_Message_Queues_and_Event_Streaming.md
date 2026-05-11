# Message Queues and Event Streaming — HLD Interview Q&A

---

## Easy (Q1–Q7)

---

### Q1. What is the difference between a message queue and a message stream?

**Answer:**

A **message queue** is a point-to-point, pull-based system where each message is consumed by exactly one consumer and then deleted. The queue holds messages until a consumer acknowledges them. Examples: RabbitMQ, AWS SQS, ActiveMQ.

A **message stream** is a distributed, ordered, append-only log where messages are retained for a configurable period regardless of consumption. Multiple independent consumer groups can read the same stream from any offset. Examples: Apache Kafka, AWS Kinesis, Apache Pulsar.

| Property              | Message Queue              | Message Stream                  |
|-----------------------|----------------------------|---------------------------------|
| Consumption model     | One consumer per message   | Multiple consumer groups        |
| Message retention     | Deleted after ACK          | Retained (time/size-based)      |
| Replay                | Not possible               | Possible (seek to offset)       |
| Ordering              | FIFO within queue          | Ordered within partition        |
| Use case              | Task distribution, RPC     | Event sourcing, analytics       |
| Backpressure          | Queue depth                | Consumer lag / offset lag       |
| Examples              | RabbitMQ, SQS              | Kafka, Kinesis, Pulsar          |

**When to choose a queue:** You need work distribution — a job should be processed exactly once by one worker (e.g., sending an email, processing a payment).

**When to choose a stream:** You need event broadcasting, replay, or multiple independent consumers reading the same data for different purposes (e.g., analytics pipeline + fraud detection reading the same transaction events).

A helpful mental model: a queue is a to-do list (tasks get checked off), while a stream is a ledger (entries accumulate and can be re-read).

---

### Q2. Describe the high-level architecture of Apache Kafka.

**Answer:**

Kafka is a distributed, partitioned, replicated commit log. The main components are:

```
Producers
   |
   v
[Topic: orders]
  Partition 0  -->  [Broker 1 (Leader)] <--> [Broker 2 (Replica)] <--> [Broker 3 (Replica)]
  Partition 1  -->  [Broker 2 (Leader)] <--> [Broker 1 (Replica)] <--> [Broker 3 (Replica)]
  Partition 2  -->  [Broker 3 (Leader)] <--> [Broker 1 (Replica)] <--> [Broker 2 (Replica)]
   |
   v
Consumer Groups
  Group A: [Consumer 1 -> P0], [Consumer 2 -> P1], [Consumer 3 -> P2]
  Group B: [Consumer A -> P0, P1, P2]   (fewer consumers than partitions)
```

**Key components:**

- **Broker:** A Kafka server that stores data and serves reads/writes. A cluster has multiple brokers.
- **Topic:** A named category of messages, split into partitions.
- **Partition:** An ordered, immutable sequence of records. Unit of parallelism. Each partition has one leader broker and N-1 replica followers.
- **Producer:** Publishes records to topics. Routes to partitions using a partition key hash or round-robin.
- **Consumer Group:** A set of consumers that divide up partitions. Each partition is consumed by exactly one consumer per group.
- **ZooKeeper / KRaft:** Originally ZooKeeper stored broker metadata. KRaft (Kafka Raft) replaces ZooKeeper in newer versions (Kafka 3.x+), embedding metadata management into the brokers themselves.
- **Offset:** An integer identifying each record within a partition. Consumers track their own offsets.

**Data flow:** Producer writes to partition leader → leader appends to log and replicates to followers → consumer polls the leader (or follower in newer versions) → consumer commits offset after processing.

---

### Q3. What are the three message delivery guarantees in Kafka, and how are they achieved?

**Answer:**

**1. At-Most-Once (fire and forget)**
- Messages may be lost, never duplicated.
- Producer: `acks=0` (no acknowledgment wait).
- Consumer: commit offset before processing.
- Use case: metrics/logs where occasional loss is acceptable.

**2. At-Least-Once (default)**
- Messages are never lost but may be duplicated.
- Producer: `acks=all` + retries enabled.
- Consumer: commit offset after processing.
- If the consumer crashes after processing but before committing, the message is reprocessed.
- Use case: most production systems; handle duplicates via idempotency.

**3. Exactly-Once (EOS — Exactly Once Semantics)**
- Messages delivered and processed exactly once.
- Requires: idempotent producer + transactional API.
- Producer side: `enable.idempotence=true` assigns each producer a PID; Kafka deduplicates retries using sequence numbers.
- Consumer-to-producer (stream processing): use Kafka Transactions — atomically consume, process, and produce in one transaction.
- Consumer side: read_committed isolation level so only committed messages are visible.

```
# Kafka producer config for exactly-once
props.put("enable.idempotence", "true");
props.put("acks", "all");
props.put("transactional.id", "my-transactional-id");

producer.initTransactions();
producer.beginTransaction();
producer.send(new ProducerRecord<>(outputTopic, key, value));
producer.sendOffsetsToTransaction(offsets, consumerGroupMetadata);
producer.commitTransaction();
```

| Guarantee    | Producer acks | Consumer offset commit | Duplicates | Loss |
|--------------|--------------|------------------------|------------|------|
| At-most-once | 0            | Before processing      | No         | Yes  |
| At-least-once| all          | After processing       | Yes        | No   |
| Exactly-once | all + idempotent | Transactional     | No         | No   |

---

### Q4. How do Kafka consumer groups work, and how are partitions assigned?

**Answer:**

A **consumer group** is a logical group of consumers identified by a `group.id`. Kafka ensures each partition in a topic is assigned to exactly one consumer within a group at any time, enabling parallel consumption.

**Key rules:**
- If consumers < partitions: some consumers handle multiple partitions.
- If consumers = partitions: ideal 1:1 mapping.
- If consumers > partitions: excess consumers are idle.

**Rebalance process:**
When a consumer joins or leaves, Kafka triggers a rebalance via the **Group Coordinator** (a designated broker):
1. All consumers send `JoinGroup` request.
2. One consumer is elected **Group Leader**.
3. Group Leader receives full partition/member list and computes assignment.
4. Leader sends `SyncGroup` with the assignment map.
5. All consumers receive their partition assignment.

```
Topic: payments (6 partitions: P0-P5)
Consumer Group: payment-processors

Before rebalance (3 consumers):
  Consumer-1 -> [P0, P1]
  Consumer-2 -> [P2, P3]
  Consumer-3 -> [P4, P5]

After Consumer-2 crashes:
  Consumer-1 -> [P0, P1, P2]
  Consumer-3 -> [P3, P4, P5]
```

**Partition assignment strategies:**
- **Range:** Assigns contiguous partitions per consumer (default for most topics).
- **RoundRobin:** Distributes partitions evenly across consumers.
- **Sticky:** Like RoundRobin but minimizes partition movement during rebalance (preferred for stateful consumers).
- **CooperativeSticky:** Incremental rebalancing — only moves necessary partitions, avoiding stop-the-world pauses.

**Heartbeat and session timeout:** Consumers send periodic heartbeats. If the broker doesn't receive a heartbeat within `session.timeout.ms` (default 45s), the consumer is considered dead and triggers a rebalance.

---

### Q5. What is a partition key in Kafka, and how should you choose one?

**Answer:**

A **partition key** is a value producers attach to records to control which partition they land in. Kafka hashes the key using `murmur2` and takes modulo of the partition count: `partition = hash(key) % numPartitions`.

**Why it matters:**
- Records with the same key always go to the same partition → guarantees ordering per key.
- Choosing a bad key causes **hot partitions** (uneven load).

**Selection strategy:**

| Goal | Key Choice |
|------|-----------|
| Ordering per user | `user_id` |
| Ordering per order | `order_id` |
| Even distribution | Use high-cardinality field |
| Related events together | Domain entity ID |

**Anti-patterns:**
- Using `null` key → round-robin, no ordering guarantee.
- Using `status` (low cardinality) → hot partitions (e.g., 80% messages have key "PENDING").
- Using timestamp → always new partition, defeats ordering.

**Practical example for an e-commerce system:**
- `order_id` as key ensures all events for an order (created, paid, shipped, delivered) land in the same partition in order.
- This allows a downstream consumer to reconstruct order state sequentially without coordination.

**Repartitioning problem:** If you change the number of partitions, the hash mapping changes. Existing records in old partitions are not moved. Best practice: choose partition count carefully upfront (over-partition initially, e.g., 2x expected throughput).

---

### Q6. What are the four exchange types in RabbitMQ, and when do you use each?

**Answer:**

In RabbitMQ, a producer sends messages to an **exchange**, which routes them to **queues** based on routing rules. The exchange type determines routing logic.

**1. Direct Exchange**
Routes messages to queues whose binding key exactly matches the message's routing key.
```
Producer -> Exchange [routing_key="error"] -> Queue: error-queue
Producer -> Exchange [routing_key="info"]  -> Queue: info-queue
```
Use case: task routing by type (email vs SMS notifications).

**2. Fanout Exchange**
Broadcasts every message to all bound queues, ignoring routing key.
```
Producer -> Fanout Exchange -> Queue A (mobile app)
                            -> Queue B (email service)
                            -> Queue C (analytics)
```
Use case: pub/sub broadcasting (event notification to multiple services).

**3. Topic Exchange**
Routes based on pattern matching with wildcards in binding keys:
- `*` matches exactly one word.
- `#` matches zero or more words.
```
Routing key: "orders.europe.payment"
Binding "#.payment"   -> matches
Binding "orders.*.*"  -> matches
Binding "orders.us.*" -> no match
```
Use case: flexible routing (route European payment events differently than US ones).

**4. Headers Exchange**
Routes based on message header attributes, ignoring routing key entirely.
```
Headers: {type: "report", format: "pdf"} -> PDF-report-queue
Headers: {type: "report", format: "csv"} -> CSV-report-queue
```
Use case: complex routing logic that doesn't map to a simple string key.

| Exchange | Routing Logic | Use Case |
|----------|-------------|----------|
| Direct | Exact key match | Task queues |
| Fanout | Broadcast all | Pub/sub |
| Topic | Wildcard patterns | Log routing |
| Headers | Header attributes | Complex routing |

---

### Q7. When should you use Kafka vs RabbitMQ vs AWS SQS?

**Answer:**

Each system has distinct strengths suited to different use cases.

**Apache Kafka — choose when:**
- You need a durable, replayable event log.
- Multiple independent consumers need to read the same data.
- You're building event sourcing, CQRS, or stream processing pipelines.
- Throughput requirements are very high (millions of msgs/sec).
- You need exactly-once processing semantics.
- Data retention beyond consumption is required (audit trail, replay).

**RabbitMQ — choose when:**
- You need complex routing logic (topic/headers exchanges).
- You need per-message TTL, priority queues, or dead-letter routing.
- Message ordering requirements are strict (single queue, single consumer).
- Low-latency task queues with acknowledgment semantics.
- You need request/reply (RPC over messaging).
- You don't need message replay.

**AWS SQS — choose when:**
- You're already on AWS and want fully managed, zero-ops messaging.
- You need simple FIFO or standard queues.
- **SQS FIFO** for exactly-once + strict ordering within a message group.
- Visibility timeout semantics for at-least-once processing.
- Integration with Lambda, SNS (fan-out pattern).

| Criteria | Kafka | RabbitMQ | SQS |
|----------|-------|----------|-----|
| Throughput | Very High | Medium | High |
| Replay | Yes | No | No |
| Complex routing | No | Yes | No |
| Managed | Self/Confluent | Self/CloudAMQP | Fully managed |
| Exactly-once | Yes (EOS) | No | FIFO queues |
| Message ordering | Per partition | Per queue | FIFO queues |
| Retention | Configurable | Until ACK | Up to 14 days |

**Common pattern:** Use Kafka as the backbone event bus, and RabbitMQ/SQS for specific task queue needs downstream.

---

## Medium (Q8–Q15)

---

### Q8. What is event-driven architecture, and what is the difference between choreography and orchestration?

**Answer:**

**Event-Driven Architecture (EDA)** is a design pattern where services communicate by producing and consuming events rather than making direct calls. Events represent facts that have happened ("OrderPlaced", "PaymentProcessed").

**Choreography** — services react to events independently; no central coordinator.

```
[Order Service] --publishes--> OrderPlaced
                                  |
         +-----------+-----------+----------+
         v           v           v          v
   [Inventory]  [Payment]  [Notification]  [Analytics]
   (reserves)  (charges)    (emails user)   (logs event)
   publishes:   publishes:
   StockReserved PaymentDone
         |           |
         +-----------+
               v
         [Shipping Service]
         (creates shipment)
```

Pros: Loose coupling, high autonomy, easy to add new consumers.
Cons: Hard to trace overall workflow, debugging distributed failures is complex, risk of circular event chains.

**Orchestration** — a central orchestrator tells each service what to do and waits for responses.

```
[Order Orchestrator]
  1. -> calls Inventory Service -> StockReserved
  2. -> calls Payment Service   -> PaymentProcessed
  3. -> calls Shipping Service  -> ShipmentCreated
  4. -> calls Notification      -> EmailSent
```

Pros: Clear visibility of workflow state, easy to add compensating transactions, centralized error handling.
Cons: Orchestrator becomes a coupling point and potential bottleneck.

**When to choose:**

| Criteria | Choreography | Orchestration |
|----------|-------------|---------------|
| Coupling | Lower | Higher |
| Visibility | Harder | Easier |
| Complexity | High (event chains) | Medium (central logic) |
| Failure handling | Distributed | Centralized |
| Best for | Simple fan-out events | Complex multi-step workflows |

In practice, many systems use both: choreography for broadcasting state changes, orchestration (via Saga) for multi-step business processes.

---

### Q9. What is the Event Sourcing pattern, and why would you use it?

**Answer:**

**Event Sourcing** is a persistence pattern where instead of storing the current state of an entity, you store the full sequence of events that led to that state. The current state is derived by replaying events.

```
Traditional (State-based):
  orders table: {id: 123, status: "shipped", total: 99.99, ...}

Event Sourcing (Event-based):
  event_store:
    {streamId: "order-123", seq: 1, event: "OrderCreated", data: {total: 99.99}}
    {streamId: "order-123", seq: 2, event: "PaymentProcessed", data: {method: "card"}}
    {streamId: "order-123", seq: 3, event: "OrderShipped", data: {trackingId: "XYZ"}}
```

**Reconstructing state:**
```python
def load_order(order_id):
    events = event_store.get_events(order_id)
    order = Order()
    for event in events:
        order.apply(event)  # each event mutates state
    return order
```

**Why use it:**

1. **Complete audit trail** — every change is recorded with who/what/when.
2. **Temporal queries** — "What was the state at 3pm yesterday?" (replay up to that timestamp).
3. **Event replay** — rebuild projections, fix bugs by replaying corrected events.
4. **Decoupled projections** — multiple read models built from same event stream.
5. **Debugging** — reproduce production issues by replaying event sequences.

**Challenges:**

- **Snapshots needed** for long-lived aggregates (avoid replaying 10,000 events on every read). Store snapshot every N events.
- **Schema evolution** — events are immutable; handle via upcasting (transform old event format to new on read).
- **Eventual consistency** — read models (projections) may lag behind event store.
- **Increased storage** — events accumulate; use compaction or archiving for old streams.

**Best suited for:** financial systems, e-commerce orders, healthcare records — anywhere audit trail and state history matter.

---

### Q10. How does CQRS work with Event Sourcing, and what problem does it solve?

**Answer:**

**CQRS (Command Query Responsibility Segregation)** separates the write model (commands that change state) from the read model (queries that return data).

```
                    +------------------+
  [Client] -------> | Command Handler  |
  (Write)           | (Write Model)    |
                    | - validates cmd  |
                    | - applies biz    |
                    |   rules          |
                    | - saves events   |
                    +--------+---------+
                             |
                    [Event Store / Kafka]
                             |
              +--------------+--------------+
              v              v              v
      [Order View]   [Analytics View]  [Search Index]
      (Postgres)       (Cassandra)      (Elasticsearch)
              ^              ^              ^
              +---[Projections (event handlers)]---+
                    (Read Models / Query Handlers)

  [Client] -------> | Query Handler   |
  (Read)            | (Read Model)    |
                    | - fast queries  |
                    | - denormalized  |
                    +------------------+
```

**Why combine CQRS with Event Sourcing:**

1. Event Sourcing produces events naturally → those same events update read-model projections.
2. Write model stays clean (event-based aggregates); read model is optimized for query patterns.
3. Multiple specialized read models (orders view, analytics, full-text search) from the same event stream.

**Trade-offs:**

| Aspect | Benefit | Cost |
|--------|---------|------|
| Read performance | Denormalized, fast | Data duplication |
| Write performance | Simple append | Projection lag |
| Flexibility | Multiple views | Increased complexity |
| Eventual consistency | Scalable | Reads may be stale |

**Consistency handling:** Reads from the query side may be stale by milliseconds. For use cases requiring fresh data after a write, either:
- Use a version token (client polls until projection catches up), or
- Read from the write model directly (breaking CQRS for that specific query).

**When to use CQRS alone (without ES):** When read/write have very different scaling needs, e.g., 1000:1 read-to-write ratio. Apply CQRS at the service level, not necessarily with full event sourcing.

---

### Q11. What is the Saga pattern, and how does it handle distributed transactions?

**Answer:**

A **Saga** is a sequence of local transactions, each publishing an event or message to trigger the next step. If a step fails, compensating transactions undo the previous steps.

**Why not use 2PC (Two-Phase Commit)?**
2PC is a distributed transaction protocol that locks resources across services until all agree. Problems: blocking locks, single coordinator failure halts all participants, tight coupling, not suitable for microservices across different databases/services.

**Choreography-based Saga:**
```
[Order Service]    --OrderCreated-->   [Payment Service]
                                            |
                                    PaymentSucceeded
                                            |
                                    [Inventory Service]
                                            |
                                    StockReserved
                                            |
                                    [Shipping Service]
                                            |
                                    ShipmentCreated

Failure: if Inventory fails:
  InventoryFailed  --> [Payment Service compensates: RefundPayment]
  PaymentRefunded  --> [Order Service compensates: CancelOrder]
```

**Orchestration-based Saga:**
```python
class OrderSaga:
    def execute(self, order_id):
        try:
            payment_service.charge(order_id)
            inventory_service.reserve(order_id)
            shipping_service.schedule(order_id)
        except InventoryError:
            payment_service.refund(order_id)
            order_service.cancel(order_id)
            raise
```

**Comparison:**

| Aspect | Choreography | Orchestration | 2PC |
|--------|-------------|---------------|-----|
| Coupling | Low | Medium | High |
| Visibility | Low | High | High |
| Failure handling | Distributed | Centralized | Atomic |
| Blocking | No | No | Yes |
| Consistency | Eventual | Eventual | Strong |

**Key insight:** Sagas achieve **eventual consistency**, not strong consistency. Each local transaction commits immediately; compensation runs asynchronously if needed. This means intermediate inconsistent states exist briefly.

**Idempotency is critical:** Compensating transactions must be idempotent — retrying a refund should not double-refund.

---

### Q12. How does Kafka guarantee message ordering, and what are its limitations?

**Answer:**

**Kafka guarantees ordering within a partition.** Messages written to the same partition are read in the exact order they were written. There is no ordering guarantee across partitions.

```
Topic: orders (3 partitions)

Partition 0: [msg1, msg4, msg7, msg10]  <- ordered within P0
Partition 1: [msg2, msg5, msg8, msg11]  <- ordered within P1
Partition 2: [msg3, msg6, msg9, msg12]  <- ordered within P2

Consumer reading all 3 partitions may see: msg1, msg2, msg3, msg4... (no global order)
```

**Achieving ordering for a domain entity:**
Use a consistent partition key (e.g., `order_id`). All events for `order-123` go to the same partition → processed in order.

**Pitfall — producer retries can break ordering:**
If a producer sends msg1, msg2 and msg1 fails, it retries. Without idempotency, order becomes: msg2, msg1.

**Fix:** Enable idempotent producer:
```
enable.idempotence=true
max.in.flight.requests.per.connection=5  # safe with idempotence
```
With idempotence, Kafka deduplicates retries using sequence numbers, preserving order.

**Limitations of Kafka ordering:**

| Scenario | Ordering Guarantee |
|----------|--------------------|
| Same partition, same producer | Yes |
| Cross-partition | No |
| After repartitioning | Breaks (hash changes) |
| Multiple producers, same partition | No (interleaved) |

**Global ordering** (all messages across all partitions in order): only possible with a single partition topic. This eliminates parallelism — not recommended for high-throughput systems.

**Practical design:** Design around per-entity ordering (partition by entity ID). If global ordering is truly needed, accept the throughput constraint of a single partition.

---

### Q13. How do you handle duplicate messages in a distributed messaging system?

**Answer:**

Duplicates are inevitable in at-least-once delivery systems (consumer crashes after processing but before committing offset). The solution is **idempotent consumers**.

**Idempotency:** Processing the same message N times produces the same result as processing it once.

**Strategies:**

**1. Natural idempotency**
Some operations are inherently idempotent. `SET status = 'paid'` run twice has the same effect as once.
```sql
UPDATE orders SET status = 'paid' WHERE id = ? AND status != 'paid';
-- or use INSERT ... ON CONFLICT DO NOTHING
```

**2. Deduplication table**
Store a unique message ID in a processed-messages table:
```sql
CREATE TABLE processed_messages (
    message_id VARCHAR(64) PRIMARY KEY,
    processed_at TIMESTAMP
);

BEGIN TRANSACTION;
  -- Attempt to insert message_id
  INSERT INTO processed_messages(message_id, processed_at)
  VALUES (?, NOW())
  ON CONFLICT (message_id) DO NOTHING;
  
  IF rows_affected > 0 THEN
    -- Process the business logic
    UPDATE orders SET status = 'paid' WHERE id = ?;
  END IF;
COMMIT;
```

**3. Idempotency key in API calls**
Pass a client-generated UUID with each request. Server stores and returns cached response if key was seen before.
```
POST /payments
Idempotency-Key: 550e8400-e29b-41d4-a716-446655440000
Body: {amount: 100, currency: USD}
```

**4. Conditional writes / optimistic locking**
Use version numbers. Only process if version matches expected:
```python
result = db.update_if_version_matches(
    item_id=order_id,
    expected_version=3,
    new_data={"status": "shipped", "version": 4}
)
if not result:
    # Already processed (version advanced), skip
    pass
```

**5. Event deduplication in Kafka**
Exactly-once semantics (EOS) with idempotent producer + transactional consumer handles this at the infrastructure level.

**Key principle:** Idempotency must be designed at the **business logic level**, not just infrastructure. Always tie deduplication to a meaningful business key, not just a message offset.

---

### Q14. What is a Dead Letter Queue (DLQ), and how should you design one?

**Answer:**

A **Dead Letter Queue (DLQ)** is a holding queue for messages that could not be successfully processed after a configured number of retries. Instead of dropping failed messages, they are moved to the DLQ for inspection and manual reprocessing.

**Why messages end up in DLQ:**
- Deserialization failure (corrupt/invalid message format).
- Business logic exception (e.g., referenced entity doesn't exist).
- Downstream service unavailability (after max retries exhausted).
- Message too large or expired TTL.

**Design pattern:**

```
Normal flow:
  Producer -> [Main Queue] -> Consumer (processes successfully) -> ACK

Failure flow:
  Producer -> [Main Queue] -> Consumer (fails) -> retry 1
                                               -> retry 2
                                               -> retry 3 (max retries)
                                               -> [Dead Letter Queue]
                                                       |
                                               DLQ Processor (alerts, logs, fixes, replays)
```

**Key design decisions:**

1. **Retry count before DLQ:** Configure based on error type. Transient errors (network): high retry count. Permanent errors (invalid data): move to DLQ immediately.

2. **Exponential backoff between retries:**
```
Retry 1: wait 1s
Retry 2: wait 2s
Retry 3: wait 4s
Retry 4: DLQ
```

3. **DLQ message enrichment:** When moving to DLQ, attach metadata:
   - Original queue/topic
   - Error message and stack trace
   - Number of retries attempted
   - Timestamp of first and last failure

4. **DLQ monitoring:** Alert on DLQ depth increase. Track DLQ growth rate as a system health metric.

5. **Replay mechanism:** Ability to fix the root cause and replay DLQ messages back to the main queue. In Kafka, this is a consumer that reads the DLQ topic and re-publishes to the original topic.

6. **Separate DLQ per topic:** Avoid a single DLQ for all topics — makes it hard to diagnose which service is failing.

**In AWS SQS:**
```
aws sqs set-queue-attributes --queue-url <main-queue-url>
  --attributes '{"RedrivePolicy": "{\"deadLetterTargetArn\": \"<dlq-arn>\", \"maxReceiveCount\": \"3\"}"}'
```

---

### Q15. What is backpressure in messaging systems, and how do you handle consumer lag?

**Answer:**

**Backpressure** is the mechanism by which a slow consumer signals upstream producers to slow down, preventing memory exhaustion and system overload.

**Consumer lag** is the difference between the latest message offset in a Kafka partition and the consumer's current committed offset. High lag means the consumer is falling behind.

```
Partition state:
  Latest offset (end): 1,000,000
  Consumer committed:    850,000
  Lag:                   150,000  <- alert if > threshold
```

**Causes of consumer lag:**
- Consumer is too slow (heavy processing, slow DB writes).
- Sudden spike in producer throughput.
- Consumer group rebalance.
- GC pauses or resource contention.

**Handling strategies:**

**1. Scale out consumers (horizontal scaling)**
Add more consumers to the group. Maximum parallelism = number of partitions.
```
3 consumers, 6 partitions -> each consumer handles 2 partitions
6 consumers, 6 partitions -> each consumer handles 1 partition
```

**2. Increase partition count**
More partitions = more parallelism potential. (Note: cannot decrease partition count.)

**3. Async processing with internal queue**
Consumer batch-reads from Kafka, pushes to internal thread pool for async processing:
```python
while True:
    records = consumer.poll(timeout_ms=500)
    futures = [thread_pool.submit(process, r) for r in records]
    wait_for_futures(futures)
    consumer.commit()
```

**4. Batch processing**
Process records in micro-batches rather than one-by-one to amortize overhead.

**5. Separate slow consumers to dedicated topics/consumer groups**
Prevent slow consumers from affecting fast ones. Use separate consumer groups per downstream service.

**6. Circuit breaker on consumer**
If the downstream service (DB, API) is slow, pause consumption temporarily rather than accumulating lag indefinitely.

**Monitoring metrics to track:**
- `kafka_consumer_lag` per partition
- Consumer throughput (msgs/sec)
- Processing time per record
- End-to-end latency (event timestamp to processing timestamp)

**Alert thresholds:** Set lag alerts at 2 levels: warning (lag > X) and critical (lag growing for >N minutes).

---

## Hard (Q16–Q20)

---

### Q16. What is a Schema Registry, and how does Avro/Protobuf enable safe message versioning?

**Answer:**

A **Schema Registry** is a centralized service that stores and enforces message schemas. Producers register schemas before publishing; consumers fetch schemas to deserialize messages. This prevents schema drift and enables backward/forward compatibility.

**Why it's needed:**
Without a schema registry, a producer can silently change message format, breaking all consumers.

**Confluent Schema Registry flow:**
```
Producer:
  1. Serialize data with Avro schema
  2. Register schema with registry (returns schema_id)
  3. Write [magic_byte(1) + schema_id(4 bytes) + avro_bytes] to Kafka

Consumer:
  1. Read message from Kafka
  2. Extract schema_id from first 5 bytes
  3. Fetch schema from registry by ID (cached locally)
  4. Deserialize avro_bytes using schema
```

**Avro compatibility modes:**

| Mode | Description | Producer can | Consumer can |
|------|-------------|-------------|-------------|
| BACKWARD | New schema reads old data | Remove fields, add optional | Old consumers still work |
| FORWARD | Old schema reads new data | Add fields, remove optional | New consumers still work |
| FULL | Both backward + forward | Add/remove optional fields | Both directions safe |
| NONE | No compatibility check | Anything | Must update together |

**Avro schema evolution example:**
```json
// V1 schema
{"type": "record", "name": "Order",
 "fields": [
   {"name": "id", "type": "string"},
   {"name": "amount", "type": "double"}
 ]}

// V2 schema (backward compatible — added optional field with default)
{"type": "record", "name": "Order",
 "fields": [
   {"name": "id", "type": "string"},
   {"name": "amount", "type": "double"},
   {"name": "currency", "type": "string", "default": "USD"}
 ]}
```

**Protobuf advantages over Avro:**
- Field numbers (not names) used for encoding → renaming fields is safe.
- Better language support.
- Forward compatibility by default (unknown fields are ignored).

**Schema versioning best practices:**
1. Never remove a required field in BACKWARD mode.
2. Never change a field's type.
3. Always provide defaults for new fields.
4. Use subject naming strategy aligned with topic (topic + `-value`, topic + `-key`).

---

### Q17. What is the difference between Lambda Architecture and Kappa Architecture?

**Answer:**

Both are data processing architectures for handling large-scale real-time and batch analytics.

**Lambda Architecture** (Nathan Marz, 2011):
Three layers:
```
Raw data
    |
    +-----------> [Batch Layer] -------> Batch Views
    |             (Hadoop/Spark)         (accurate, slow)
    |                                           |
    +-----------> [Speed Layer] -----> Real-time Views
                  (Storm/Flink)        (approximate, fast)
                                               |
                                    [Serving Layer]
                                    (merges batch + speed)
                                               |
                                          [Client]
```

**Batch Layer:** Reprocesses all historical data on a schedule (hourly, daily). Produces accurate results. High latency (hours).
**Speed Layer:** Processes only recent data in real-time. Low latency (seconds). May be approximate.
**Serving Layer:** Merges batch and speed views to answer queries.

**Problems with Lambda:**
- Maintaining two separate codebases (batch + streaming) for the same logic.
- Merging batch and speed views is complex.
- Operationally expensive.

**Kappa Architecture** (Jay Kreps, 2014):
Simplifies Lambda by eliminating the batch layer. Everything is a stream.

```
Raw data
    |
    +-----------> [Kafka (immutable log)] <-- historical replay possible
                           |
                  [Stream Processing]
                  (Flink/Kafka Streams)
                           |
                   [Serving Layer]
                           |
                      [Client]
```

**Reprocessing in Kappa:**
When logic changes, replay from the beginning of the Kafka log with a new consumer group + new output topic. Once caught up, swap the serving layer to point to the new output.

**Comparison:**

| Aspect | Lambda | Kappa |
|--------|--------|-------|
| Complexity | High (two systems) | Lower (one system) |
| Accuracy | Batch = exact | Depends on stream engine |
| Latency | Batch: high, Speed: low | Low for both |
| Reprocessing | Batch recompute | Kafka replay |
| Operational cost | High | Lower |
| Best for | Complex batch + stream | Stream-first architectures |

**When Lambda still makes sense:** When batch jobs use fundamentally different algorithms (e.g., ML training on full dataset) that can't be expressed as streaming operations.

---

### Q18. How does Kafka achieve exactly-once semantics end-to-end?

**Answer:**

Kafka's exactly-once semantics (EOS) requires coordination at both the producer and consumer levels, plus transactional support for stream processing.

**Three components of Kafka EOS:**

**1. Idempotent Producer (deduplication of retries)**
```
Producer assigned: PID (Producer ID) at startup
Each message gets: sequence_number (monotonically increasing per partition)

Broker logic:
  If received (PID, partition, seq_num) already in buffer:
    -> Discard duplicate, return ACK (not an error)
  Else:
    -> Append to partition log
```
Config: `enable.idempotence=true`
This prevents duplicates from producer retries within a single producer session.

**2. Transactions (atomic multi-partition writes)**
```java
producer.initTransactions();  // registers transactional.id with broker

// Transaction 1
producer.beginTransaction();
producer.send(new ProducerRecord<>("topic-A", key, val));
producer.send(new ProducerRecord<>("topic-B", key, val));
producer.commitTransaction();  // atomic: both land or neither

// On failure:
producer.abortTransaction();   // broker discards uncommitted writes
```

**3. Consumer transactional reads (read_committed isolation)**
```
consumer.config("isolation.level", "read_committed");
// Consumer only sees messages from COMMITTED transactions
// Aborted transaction messages are hidden (though present in log)
```

**End-to-end EOS in Kafka Streams (consume-process-produce loop):**
```
Input Topic -> [Consumer] -> [Process] -> [Producer] -> Output Topic
                                 |
                         sendOffsetsToTransaction()
                         // atomically commits:
                         // 1. output messages to output topic
                         // 2. consumer offsets to __consumer_offsets
                         // Ensures: processed = produced, no double-counting
```

**Limitations and caveats:**
- EOS applies within Kafka-to-Kafka pipelines. If processing involves an external system (DB write), Kafka cannot guarantee atomicity with that external write.
- Performance cost: transactions add latency (~10-20ms per transaction commit).
- `transactional.id` must be stable across restarts. If the same ID is reused by a new producer instance, the old session is fenced off (zombie fencing).

**Zombie fencing:**
```
Old producer (crashed, restarting) has epoch=1
New producer starts with same transactional.id -> broker assigns epoch=2
If old producer tries to write: broker rejects (epoch too old) -> EpochFencedException
```

This prevents split-brain scenarios where two producer instances think they're the authoritative writer.

---

### Q19. What is a Kafka compacted topic, and when should you use it?

**Answer:**

A **compacted topic** is a Kafka topic where the broker guarantees that for each message key, at least the most recent value is retained. Older records with the same key are eventually deleted during **log compaction**, but the latest record per key is never deleted (unless its value is a tombstone/null).

**How compaction works:**
```
Before compaction:
  P0: [key=A:v1, key=B:v1, key=A:v2, key=C:v1, key=B:v2, key=A:v3]

After compaction:
  P0: [key=A:v3, key=C:v1, key=B:v2]
  (oldest records per key removed, latest retained)

Tombstone: key=B with value=null -> B will be deleted entirely after tombstone period
```

**Log cleaner process:**
- Runs as a background thread on each broker.
- Divides the log into "clean" (already compacted) and "dirty" (not yet compacted) segments.
- Builds a key-to-latest-offset map for the dirty portion.
- Rewrites dirty log, keeping only latest value per key.

**Configuration:**
```
log.cleanup.policy=compact          # enable compaction (vs delete)
log.compaction.lag.ms=...           # minimum age of a message before eligible for compaction
min.cleanable.dirty.ratio=0.5       # compact when 50% of log is "dirty"
delete.retention.ms=86400000        # how long tombstones are kept
```

**Use cases:**

| Use Case | Why Compacted |
|----------|--------------|
| Consumer group offsets (`__consumer_offsets`) | Store latest offset per group-partition |
| Database change capture (CDC) | Latest row state per primary key |
| Configuration/feature flags | Latest config value per key |
| Cache warming / materialized views | Current state of each entity |
| Kafka Streams state stores (changelogs) | Rebuild state on restart |

**Compacted + delete policy combination:**
```
log.cleanup.policy=compact,delete
```
Combines compaction (keep latest per key) with time-based deletion (remove records older than retention.ms). Useful for use cases that need recent history but also eventual cleanup.

**Key distinction:** A regular topic is a log of events (history). A compacted topic is a log of **current state** (like a distributed key-value store). New consumers joining late will still get the latest value for every key — they see a consistent snapshot.

---

### Q20. What are the windowing types in stream processing, and how do they differ?

**Answer:**

**Windowing** in stream processing groups events into finite buckets (windows) for aggregation. Without windows, you'd aggregate over an infinite, ever-growing stream.

**1. Tumbling Window**
Fixed-size, non-overlapping, contiguous windows.
```
Events: e1(t=1), e2(t=3), e3(t=5), e4(t=7), e5(t=9)
Window size: 4 seconds

Window 1 [0-4):  e1, e2
Window 2 [4-8):  e3, e4
Window 3 [8-12): e5
```
Use case: "Count orders per minute", "Revenue per hour". Each event belongs to exactly one window.

**2. Sliding Window**
Fixed-size windows that advance by a smaller step (hop), causing overlap.
```
Window size: 6s, Slide: 2s

Window 1 [0-6):  e1, e2, e3
Window 2 [2-8):  e2, e3, e4
Window 3 [4-10): e3, e4, e5
```
Events belong to multiple windows. Useful for continuous moving averages.
Use case: "5-minute rolling average CPU", "Sliding window fraud detection (3 transactions in 60s)".

**3. Session Window**
Dynamic-size windows based on activity gaps. A session ends when there's no activity for a gap duration.
```
Gap timeout: 3s
Events: e1(t=1), e2(t=2), [gap=5s], e3(t=8), e4(t=9), e5(t=10)

Session 1 [1-2]: e1, e2  (closed because gap > 3s)
Session 2 [8-10]: e3, e4, e5
```
Use case: "Session-based web analytics", "Group user activity bursts", "IoT sensor data burst analysis".

**4. Global Window (with triggers)**
All events go into a single window, with custom triggers for emission.
```
Global window + trigger: emit every 100 events OR every 10 seconds
```
Use case: batch processing in a streaming context.

**Handling Late Arrivals (Watermarks):**
Out-of-order events are common (network delays). Watermarks define how late data can be and still be included in a window.

```
Watermark = max_event_time_seen - allowed_lateness

If event_time < watermark: event is "late"
Late handling options:
  - Drop late events
  - Include in a late pane (partial update)
  - Side output for separate processing
```

**Flink/Kafka Streams windowing:**
```java
// Tumbling in Kafka Streams
TimeWindows.ofSizeWithNoGrace(Duration.ofMinutes(1));

// Sliding
SlidingWindows.withTimeDifferenceAndGrace(Duration.ofSeconds(30), Duration.ofSeconds(5));

// Session
SessionWindows.ofInactivityGapWithNoGrace(Duration.ofMinutes(5));
```

**Comparison table:**

| Window Type | Size | Overlap | Use Case |
|-------------|------|---------|----------|
| Tumbling | Fixed | None | Hourly/minute aggregates |
| Sliding | Fixed | Yes | Rolling averages |
| Session | Dynamic | None | User behavior sessions |
| Global | Infinite | N/A | Custom trigger logic |

---

## Quick Reference

### Kafka Architecture
| Component | Role |
|-----------|------|
| Broker | Server storing partitions, serving reads/writes |
| Topic | Named category, split into partitions |
| Partition | Ordered immutable log; unit of parallelism |
| Producer | Writes records; routes by key hash |
| Consumer Group | Divides partitions among members |
| Offset | Record position within a partition |
| Replication Factor | Number of copies per partition |

### Delivery Guarantees
| Guarantee | acks | Offset Commit | Duplicates | Loss |
|-----------|------|---------------|------------|------|
| At-most-once | 0 | Before process | No | Yes |
| At-least-once | all | After process | Yes | No |
| Exactly-once | all + idempotent | Transactional | No | No |

### Kafka vs RabbitMQ vs SQS
| Feature | Kafka | RabbitMQ | SQS |
|---------|-------|----------|-----|
| Replay | Yes | No | No |
| Complex routing | No | Yes | No |
| Throughput | Very High | Medium | High |
| Managed | Self/Confluent | Self/Cloud | Fully |

### Exchange Types (RabbitMQ)
| Type | Routing Logic |
|------|-------------|
| Direct | Exact key match |
| Fanout | All bound queues |
| Topic | Wildcard pattern (`*`, `#`) |
| Headers | Header attributes |

### Window Types
| Type | Overlap | Size | Key Use Case |
|------|---------|------|-------------|
| Tumbling | No | Fixed | Per-minute counts |
| Sliding | Yes | Fixed | Rolling averages |
| Session | No | Dynamic | User sessions |

### Schema Compatibility Modes
| Mode | Producer can change | Consumer compatibility |
|------|-------------------|----------------------|
| BACKWARD | Remove fields, add optional | Old consumers work |
| FORWARD | Add fields, remove optional | New consumers work |
| FULL | Add/remove optional only | Both directions |

### Key Kafka Configs
```
# Producer
enable.idempotence=true
acks=all
retries=MAX_INT
transactional.id=<unique-id>

# Consumer
isolation.level=read_committed
auto.offset.reset=earliest
enable.auto.commit=false

# Topic
log.cleanup.policy=compact     # for compacted topics
replication.factor=3
min.insync.replicas=2
```
