# 30 — Event-Driven Architecture

---

## Easy (Q1–Q7)

---

### Q1. What is event-driven architecture and how does it differ from request-response?

**Event-driven architecture (EDA)** is a software design paradigm where services communicate by producing and consuming events — records that something has happened — rather than by calling each other directly.

**Request-response (synchronous):**
```
Client              Order Service         Inventory Service
  │─── POST /orders ───▶│                       │
  │                      │── GET /inventory ────▶│
  │                      │◀── 200 OK (qty: 5) ───│
  │                      │ (processes order)      │
  │◀─── 200 OK ──────────│                       │

Properties:
  - Caller blocks until callee responds
  - Tight coupling: Order Service must know Inventory Service's API
  - If Inventory Service is down, Order Service fails
```

**Event-driven (asynchronous):**
```
Client              Order Service            Event Broker
  │─── POST /orders ───▶│                       │
  │                      │── publish ────────────▶│ "OrderCreated"
  │◀─── 202 Accepted ────│                       │
  │                      │                       │── consume ──▶ Inventory Service
  │                      │                       │── consume ──▶ Email Service
  │                      │                       │── consume ──▶ Analytics Service

Properties:
  - Caller does not block; gets acknowledgement immediately
  - Loose coupling: Order Service doesn't know who consumes events
  - Inventory Service can be down; events queue up and are processed when it recovers
  - New consumers can be added without modifying Order Service
```

**Core comparison:**

| Dimension | Request-Response | Event-Driven |
|---|---|---|
| Coupling | Tight (knows callee) | Loose (knows only event schema) |
| Temporal coupling | Both must be up simultaneously | Producer and consumer can be up at different times |
| Latency | Low (synchronous) | Higher (async processing) |
| Consistency | Immediate | Eventual |
| Debuggability | Simple call stack | Requires event trace correlation |
| Scalability | Callee must scale to match caller | Consumer scales independently |

EDA excels when operations can be asynchronous, when multiple services react to the same business event, and when services need independent scalability and deployment cadences.

---

### Q2. What is the difference between an event, a command, and a query?

These three message types have distinct semantic meanings and different architectural implications. Confusing them leads to poorly designed systems.

**Event:**
- Records that something **has happened** — a fact about the past.
- Named in past tense.
- Published to anyone who cares (fire and forget by publisher).
- Publisher does not know or care who consumes it.
- Immutable fact — should never be modified after publication.

```json
{
  "type": "OrderPlaced",
  "order_id": "ord_123",
  "user_id": "usr_456",
  "total_amount": 5000,
  "occurred_at": "2024-01-15T10:30:00Z"
}
```

**Command:**
- Requests that something **should happen** — an instruction about the future.
- Named in imperative form.
- Directed at a specific service (has exactly one intended recipient).
- May be rejected (the recipient can say no).
- Has a response (success/failure).

```json
{
  "type": "ReserveInventory",
  "order_id": "ord_123",
  "sku": "PROD_789",
  "quantity": 2,
  "requested_at": "2024-01-15T10:30:01Z"
}
```

**Query:**
- Requests **information** — asks for current state.
- No side effects (pure read).
- Synchronous in most designs (caller wants the answer now).
- In CQRS, queries read from dedicated read models.

```json
{
  "type": "GetOrderStatus",
  "order_id": "ord_123"
}
```

**Decision guide:**

| Message Type | Has recipient? | Can be rejected? | Causes side effects? | Naming |
|---|---|---|---|---|
| Event | No (broadcast) | No | N/A (already happened) | Past tense |
| Command | Yes (one) | Yes | Yes | Imperative |
| Query | Yes (one) | No | No | Question form |

**Common mistake:** Naming an event as a command. `UserShouldBeEmailed` is a command disguised as an event. Events describe facts: `OrderConfirmed`. If the email service doesn't exist, the event was still published — it just wasn't consumed yet. This subtle difference is what makes events enable loose coupling.

---

### Q3. What are domain events and how do they enable loose coupling between services?

**Domain events** are events that represent meaningful occurrences in a business domain — not technical events (like "database row updated") but business-level facts (like "customer placed an order").

The concept originates from **Domain-Driven Design (DDD)** by Eric Evans. A domain event captures what happened, why it matters, and the relevant business context.

**Characteristics of a well-designed domain event:**
```json
{
  "event_id": "evt_550e8400-e29b-41d4-a716",
  "event_type": "OrderPlaced",
  "occurred_at": "2024-01-15T10:30:00Z",
  "aggregate_type": "Order",
  "aggregate_id": "ord_123",
  "version": 3,
  "payload": {
    "customer_id": "cust_789",
    "items": [
      { "sku": "BOOK_001", "qty": 1, "price": 2999 }
    ],
    "shipping_address": { "city": "Berlin", "country": "DE" },
    "total": 2999,
    "currency": "EUR"
  }
}
```

**How domain events enable loose coupling:**

```
Before (tight coupling):
  OrderService → calls InventoryService.reserve()
  OrderService → calls EmailService.sendConfirmation()
  OrderService → calls LoyaltyService.awardPoints()
  OrderService → calls AnalyticsService.trackConversion()
  
  Problem: OrderService has 4 dependencies. If any is slow/down, orders fail.
           Adding RecommendationService requires changing OrderService code.

After (loose coupling via domain events):
  OrderService → publishes OrderPlaced event
  
  Consumers (independently):
    InventoryService → consumes OrderPlaced → reserves stock
    EmailService → consumes OrderPlaced → sends email
    LoyaltyService → consumes OrderPlaced → awards points
    AnalyticsService → consumes OrderPlaced → tracks conversion
    
  To add RecommendationService: it subscribes to OrderPlaced — OrderService unchanged.
  If EmailService is down: orders still succeed, emails queued for later.
```

**Bounded context integration:**
Domain events are the primary mechanism for integration between **bounded contexts** in DDD. The Order bounded context publishes `OrderPlaced`; the Inventory bounded context translates this into its own language (`OrderFulfillmentRequested`) without either context depending on the other's model.

---

### Q4. What is the role of an event broker? Compare Kafka, RabbitMQ, and EventBridge.

An **event broker** is infrastructure that receives events from producers, durably stores them, and delivers them to one or more consumers. It provides decoupling, durability, fan-out, and routing capabilities.

**Key broker functions:**
1. **Durability** — events persist even if consumers are down.
2. **Fan-out** — one producer's event can be delivered to multiple independent consumers.
3. **Routing** — filter or route events to specific consumers based on content or topic.
4. **Back-pressure** — consumers can process at their own rate; broker buffers the difference.
5. **Replay** — re-process events from a point in history (Kafka only).

**Kafka:**
```
Philosophy: Distributed commit log
  - Events stored durably for configurable retention (hours to forever)
  - Consumer groups track offsets; events can be replayed from any offset
  - Ordered within a partition (useful for entity-level ordering)
  - High throughput: millions of messages/second
  - Complex ops: requires ZooKeeper/KRaft, tuning for latency vs throughput

Best for: event sourcing, audit logs, stream processing, high-throughput pipelines
```

**RabbitMQ:**
```
Philosophy: Message broker with routing
  - Exchanges route messages to queues based on routing keys, patterns, topics
  - Messages deleted after acknowledgement (not a log)
  - Excellent for task queues, work distribution, complex routing
  - Simpler to operate than Kafka; supports AMQP, MQTT, STOMP

Best for: task queues, RPC over messaging, complex routing, work distribution
```

**AWS EventBridge:**
```
Philosophy: Serverless event bus
  - No infrastructure to manage
  - Rules with content-based filtering route events to Lambda, SQS, SNS, etc.
  - 60+ AWS service integrations (CloudTrail, S3, EC2 state changes)
  - Schema registry for event discovery and validation
  - No replay capability (events not durably stored beyond 24 hours)

Best for: AWS-native event routing, serverless architectures, low/moderate volume
```

**Comparison table:**

| Feature | Kafka | RabbitMQ | EventBridge |
|---|---|---|---|
| Replay history | Yes (configurable retention) | No | No (24h only) |
| Throughput | Very high (MB/s) | High | Moderate |
| Ordering | Per partition | Per queue | No guarantee |
| Routing | Topic-based | Exchange types | Content-based rules |
| Ops complexity | High | Medium | Zero (managed) |
| Consumer groups | Yes | Yes (competing) | Targets |
| Best for | Streaming, event sourcing | Task queues, RPC | AWS-native events |

---

### Q5. What is the difference between choreography and orchestration in event-driven systems?

When multiple services must collaborate to complete a business process, there are two architectural approaches: **choreography** (services react to events independently) and **orchestration** (a central coordinator directs services).

**Choreography:**
```
No central brain — each service listens for events and reacts

OrderService: publishes OrderPlaced
              │
              ▼ (event)
InventoryService: listens for OrderPlaced
                  → reserves stock
                  → publishes InventoryReserved
                               │
                               ▼ (event)
PaymentService: listens for InventoryReserved
                → charges card
                → publishes PaymentProcessed
                               │
                               ▼ (event)
FulfillmentService: listens for PaymentProcessed
                    → ships order
```

**Orchestration:**
```
Central saga orchestrator controls the flow

OrderOrchestrator:
  1. Sends command "ReserveInventory" → waits for reply
  2. Sends command "ChargePayment" → waits for reply
  3. Sends command "ShipOrder" → waits for reply
  4. Handles failures with explicit compensating commands
```

**Comparison:**

| Dimension | Choreography | Orchestration |
|---|---|---|
| Coupling | Services know only event schema | Services know only orchestrator |
| Debuggability | Hard — trace flows across services | Easier — orchestrator holds state |
| Resilience | High — no single point of failure | Lower — orchestrator is SPOF |
| Flexibility | Easy to add new consumers | Must modify orchestrator to add steps |
| Observability | Distributed tracing required | Orchestrator state is the trace |
| Failure handling | Each service handles its own failures | Orchestrator manages compensations |

**Which to choose:**
- **Choreography** is better for loosely coupled, independent workflows where each step doesn't depend on the result of previous steps (e.g., notifications, analytics).
- **Orchestration** is better for complex, stateful workflows where the outcome of each step determines the next action (e.g., payment processing, order fulfillment with compensation).

**Hybrid approach (most common in practice):** Use orchestration within a bounded context (where tight coupling is acceptable) and choreography between bounded contexts (where loose coupling is paramount).

---

### Q6. What is event sourcing? Why use an append-only log as source of truth?

**Event sourcing** is a pattern where instead of storing the **current state** of an entity, you store the **complete history of events** that led to that state. The current state is derived by replaying events from the beginning (or from a snapshot).

**Traditional state storage:**
```sql
-- Only current state stored; history lost
UPDATE orders SET status = 'shipped', shipped_at = NOW() WHERE id = 'ord_123';
-- Previous states (created, paid, etc.) are gone
```

**Event sourcing:**
```
events table (append-only):
  ord_123 | OrderCreated     | { customer_id: 789, total: 5000 }
  ord_123 | PaymentReceived  | { payment_id: pay_456, amount: 5000 }
  ord_123 | ItemsPicked      | { warehouse: WH-01, picker: emp_123 }
  ord_123 | OrderShipped     | { tracking: FEDEX-789, carrier: FedEx }
  ord_123 | OrderDelivered   | { delivered_at: 2024-01-17T14:30:00 }

Current state = replay all events from first to last
```

**Rebuilding state from events:**
```python
def rebuild_order(order_id, db):
    order = Order(id=order_id)
    events = db.query(
        "SELECT event_type, payload FROM events WHERE aggregate_id = %s ORDER BY sequence",
        [order_id]
    )
    for event in events:
        order.apply(event)
    return order

class Order:
    def apply(self, event):
        handlers = {
            "OrderCreated":    self.on_order_created,
            "PaymentReceived": self.on_payment_received,
            "OrderShipped":    self.on_order_shipped,
            "OrderDelivered":  self.on_order_delivered,
        }
        handlers[event.type](event.payload)
    
    def on_order_shipped(self, payload):
        self.status = "shipped"
        self.tracking_number = payload["tracking"]
        self.shipped_at = payload.get("shipped_at")
```

**Why append-only log is powerful:**

| Benefit | Description |
|---|---|
| Complete audit trail | Every state change recorded with reason and timestamp |
| Temporal queries | "What was the order status at 10:30 AM?" — replay up to that timestamp |
| Bug debugging | Reproduce any past state by replaying events |
| Event replay | Build a new read model by replaying all historical events |
| CQRS integration | Events are the natural feed for updating read models |
| Compliance | Immutable audit log satisfies financial/healthcare regulations |

**Trade-off:** Loading an entity requires replaying potentially thousands of events (mitigated by snapshots — see Q8).

---

### Q7. How do you handle eventual consistency in event-driven systems? What is the lag problem?

**Eventual consistency** means that after a write, there is a period (the "lag") during which different parts of the system may return different answers for the same question. All parts will eventually converge to the same answer — but not immediately.

**The lag problem:**
```
User posts a photo:
  1. Photo service stores photo in S3
  2. Photo service publishes PhotoUploaded event to Kafka
  3. Consumer processes event, updates photo feed
  4. Second consumer processes event, updates thumbnail

But at T+0ms (immediately after upload response):
  - User clicks "My Photos" 
  - Photo service (strong consistency): photo exists
  - Feed service (reading from event-driven read model): photo NOT YET visible
  - Lag: 50-500ms until event is consumed and feed is updated

User experience: user refreshes and photo appears — or worse, sees stale state
```

**Handling strategies:**

**1. Read-your-writes consistency (most important):**
```python
def upload_photo(user_id, photo_data):
    photo_id = photo_service.store(photo_data)
    event_bus.publish("PhotoUploaded", {"photo_id": photo_id, "user_id": user_id})
    
    # After write: send user to a "strong consistency read" path for their OWN data
    return {
        "photo_id": photo_id,
        "redirect": f"/photos/{photo_id}?strong_read=true"
    }
    # Other users see eventually consistent feed; uploader sees their own photo immediately
```

**2. Optimistic UI (update locally before confirmation):**
```javascript
// Frontend: show the photo immediately in local state
function uploadPhoto(file) {
    const localId = generateLocalId();
    setPhotos([...photos, { id: localId, src: URL.createObjectURL(file), status: 'uploading' }]);
    
    apiClient.post('/photos', file).then(response => {
        // Replace local placeholder with server-confirmed photo
        setPhotos(photos.map(p => p.id === localId ? response.data : p));
    });
}
```

**3. Bounded staleness (communicate SLA):**
```
Design principle: make the lag visible and bounded, not hidden
  API documentation: "Feed updates within 2 seconds of upload"
  SLA: P99 event processing latency < 2 seconds
  Monitoring: alert if consumer lag > 5 seconds
```

**4. Synchronous path for critical operations:**
- For operations where stale reads are unacceptable (payment confirmation, stock check), use a synchronous call for that specific step.
- Use event-driven for everything else.

**5. Saga correlation IDs:**
```
Track whether all expected events have been processed
  If OrderPlaced correlates to InventoryReserved within 5 seconds → consistent
  If not → flag for manual review or compensating action
```

---

## Medium (Q8–Q15)

---

### Q8. What is event replay and how do snapshots improve performance?

**Event replay** is the ability to re-process historical events — either to rebuild a corrupted read model, hydrate a new service with historical data, or debug a past system state. It is one of event sourcing's most powerful capabilities.

**Event replay use cases:**
```
1. Bug fix: Found a calculation bug in the analytics service
   Solution: Fix the bug, replay all historical events → analytics rebuilt correctly

2. New service: Recommendation service needs order history
   Solution: Replay all OrderPlaced events from the beginning → recommendations bootstrapped

3. Read model migration: Changing the schema of the product search index
   Solution: Replay all ProductCreated/ProductUpdated events → new index built in parallel

4. Audit: Investigate a fraud incident
   Solution: Replay events for a specific user → reconstruct every action in order
```

**Snapshot pattern for performance:**

Without snapshots, loading an entity with 10,000 events requires replaying all 10,000:
```
Read order #123 with 10,000 events → replay all 10,000 → slow
```

With snapshots:
```
Every 100 events, store a snapshot of current state:
  snapshot(order_123, seq=100)  = { status: "created", total: 5000, ... }
  snapshot(order_123, seq=200)  = { status: "paid", total: 5000, ... }
  snapshot(order_123, seq=9900) = { status: "shipped", tracking: "...", ... }
  events(order_123, seq=9901..10000) = last 100 events only

Load order: read snapshot at seq=9900 + replay only 100 remaining events
Speedup: 100x fewer events to process
```

**Implementation:**
```python
def load_order_with_snapshot(order_id, db):
    # Load latest snapshot
    snapshot = db.query("""
        SELECT sequence, state
        FROM order_snapshots
        WHERE order_id = %s
        ORDER BY sequence DESC LIMIT 1
    """, [order_id]).fetchone()
    
    start_sequence = 0
    order = Order(id=order_id)
    
    if snapshot:
        order.restore_from_snapshot(snapshot['state'])
        start_sequence = snapshot['sequence']
    
    # Only replay events after the snapshot
    events = db.query("""
        SELECT event_type, payload, sequence
        FROM events
        WHERE aggregate_id = %s AND sequence > %s
        ORDER BY sequence ASC
    """, [order_id, start_sequence])
    
    for event in events:
        order.apply(event)
        if order.sequence % 100 == 0:
            save_snapshot(order)  # Snapshot every 100 events
    
    return order
```

**Snapshot storage:**
```sql
CREATE TABLE order_snapshots (
    order_id  VARCHAR(255),
    sequence  BIGINT,
    state     JSONB NOT NULL,
    taken_at  TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (order_id, sequence)
);
```

---

### Q9. How do you handle out-of-order events in an event-driven system?

Out-of-order events are common in distributed systems because of network delays, retry storms, and partition rebalancing. An event consumer must be robust to receiving events in a different order than they were produced.

**Why events arrive out of order:**
```
Producer sends:
  Event 1: OrderCreated  (seq=1, ts=10:30:00)
  Event 2: OrderPaid     (seq=2, ts=10:30:01)
  Event 3: OrderShipped  (seq=3, ts=10:30:02)

Consumer may receive:
  Event 3: OrderShipped  ← arrives first (different partition, no ordering guarantee)
  Event 1: OrderCreated
  Event 2: OrderPaid

Problem: how do we process "OrderShipped" before seeing "OrderCreated"?
```

**Solution 1: Kafka partition ordering (most common):**
```python
# Produce all events for the same entity to the SAME partition
# Kafka guarantees ordering within a partition

producer.send(
    topic='orders',
    key='ord_123'.encode(),   # Same order_id → same partition → ordered delivery
    value=event_data
)
# All events for ord_123 go to the same partition → always delivered in order
```

**Solution 2: Sequence number validation with buffering:**
```python
class OrderEventProcessor:
    def __init__(self):
        self.expected_sequences = {}   # order_id → next expected sequence
        self.buffers = {}              # order_id → {seq: event} (out-of-order buffer)
    
    def process(self, event):
        order_id = event['aggregate_id']
        seq = event['sequence']
        expected = self.expected_sequences.get(order_id, 1)
        
        if seq == expected:
            self.apply(event)
            self.expected_sequences[order_id] = seq + 1
            # Check buffer for next events
            self.drain_buffer(order_id)
        elif seq > expected:
            # Buffer for later
            self.buffers.setdefault(order_id, {})[seq] = event
        else:
            # seq < expected: already processed, ignore (idempotent)
            pass
    
    def drain_buffer(self, order_id):
        expected = self.expected_sequences[order_id]
        while expected in self.buffers.get(order_id, {}):
            event = self.buffers[order_id].pop(expected)
            self.apply(event)
            expected += 1
        self.expected_sequences[order_id] = expected
```

**Solution 3: State machine with tolerant transitions:**
```python
class OrderStateMachine:
    VALID_TRANSITIONS = {
        None:        ['OrderCreated'],
        'created':   ['OrderPaid', 'OrderCancelled'],
        'paid':      ['OrderShipped', 'OrderRefunded'],
        'shipped':   ['OrderDelivered'],
    }
    
    def apply(self, event, current_state):
        allowed = self.VALID_TRANSITIONS.get(current_state, [])
        if event.type not in allowed:
            # Out-of-order or invalid: log and discard or buffer
            logger.warning(f"Invalid transition: {current_state} + {event.type}")
            return current_state
        return self.transitions[event.type](current_state, event)
```

**Solution 4: Timestamps with tolerance window:**
- Events with timestamps within a 5-second window are considered potentially out-of-order.
- Buffer events until the window passes before processing.
- Common in stream processing frameworks (Apache Flink, Kafka Streams).

---

### Q10. Explain the dual-write problem and how the outbox pattern solves it.

The **dual-write problem** is one of the most dangerous consistency pitfalls in event-driven systems. It occurs whenever an application writes to two separate systems (e.g., a database AND a message broker) without atomicity guarantees.

**The problem in detail:**
```
Order service: place_order() method

Option A: Write DB first, then publish event
  db.insert_order(order)     ← succeeds
  kafka.publish(OrderPlaced) ← FAILS (broker unavailable, network error)
  
  Result: Order in DB but no event published
  Downstream services (inventory, email, fulfillment) never notified
  System is inconsistent

Option B: Publish event first, then write DB
  kafka.publish(OrderPlaced) ← succeeds
  db.insert_order(order)     ← FAILS (DB constraint violation)
  
  Result: Event published but no order in DB
  Inventory reserved for an order that doesn't exist
  Ghost reservation
```

**The outbox pattern — the correct solution:**
```
Atomic operation: write order AND event in same DB transaction

BEGIN TRANSACTION;
  INSERT INTO orders (id, customer_id, total, status)
  VALUES ('ord_123', 'cust_789', 5000, 'pending');
  
  INSERT INTO outbox (event_type, aggregate_id, payload, published)
  VALUES ('OrderPlaced', 'ord_123', '{"customer_id":"cust_789","total":5000}', false);
COMMIT;
```

Both writes succeed or both fail — atomicity guaranteed by the database's own transaction mechanism. No two-phase commit needed.

**Outbox relay (Transactional Outbox with Debezium CDC):**
```
PostgreSQL WAL (Write-Ahead Log)
         │
         ▼ (CDC — Change Data Capture)
    Debezium Connector
         │ captures INSERT to outbox table
         ▼
    Kafka Topic: order-events
         │
         ▼
    Consumers: InventoryService, EmailService, FulfillmentService
```

Debezium reads the PostgreSQL WAL directly — no polling, near-zero latency, and guaranteed at-least-once delivery. If Debezium crashes, it resumes from the last committed WAL position.

**Alternative: polling relay:**
```python
def outbox_relay():
    while True:
        pending = db.query("""
            SELECT id, event_type, aggregate_id, payload
            FROM outbox
            WHERE published = false
            ORDER BY created_at
            LIMIT 100
        """)
        
        for event in pending:
            kafka.produce(event.event_type, event.payload)
            db.execute("UPDATE outbox SET published=true WHERE id=%s", [event.id])
        
        time.sleep(0.1)  # 100ms polling interval
```

The polling relay is simpler but adds 0–100ms latency and puts load on the database.

---

### Q11. How do you implement a saga with choreography and compensating events?

The **choreography-based saga** implements a distributed transaction by having each service react to domain events and either progress the saga forward (by emitting a success event) or trigger compensation (by emitting a failure event that other services react to).

**Order fulfillment saga — happy path:**
```
OrderService:       OrderPlaced
                         │
InventoryService:   InventoryReserved
                         │
PaymentService:     PaymentProcessed
                         │
FulfillmentService: ShipmentCreated
                         │
OrderService:       OrderConfirmed
```

**Compensation flow when payment fails:**
```
OrderService:       OrderPlaced
                         │
InventoryService:   InventoryReserved
                         │
PaymentService:     PaymentFailed (card declined)
                         │
                    ┌────┘
                    ▼
InventoryService:   listens for PaymentFailed
                    → releases reserved inventory
                    → publishes InventoryReleased
                         │
OrderService:       listens for PaymentFailed
                    → sets order status = 'failed'
                    → publishes OrderCancelled
```

**Implementation (Inventory service as event consumer):**
```python
class InventoryService:
    def handle_order_placed(self, event):
        """Forward step: reserve inventory"""
        try:
            reservation_id = self.db.reserve_stock(
                sku=event.payload['sku'],
                quantity=event.payload['quantity'],
                order_id=event.payload['order_id']
            )
            self.event_bus.publish("InventoryReserved", {
                "order_id": event.payload['order_id'],
                "reservation_id": reservation_id,
                "sku": event.payload['sku'],
                "quantity": event.payload['quantity']
            })
        except InsufficientStockError:
            self.event_bus.publish("InventoryReservationFailed", {
                "order_id": event.payload['order_id'],
                "reason": "insufficient_stock"
            })
    
    def handle_payment_failed(self, event):
        """Compensating step: release reservation"""
        reservation = self.db.get_reservation(order_id=event.payload['order_id'])
        if reservation and reservation.status == 'reserved':
            self.db.release_reservation(reservation.id)
            self.event_bus.publish("InventoryReleased", {
                "order_id": event.payload['order_id'],
                "reservation_id": reservation.id
            })
```

**Critical properties of compensating transactions:**
- Must be **idempotent** — the compensation handler may be called multiple times.
- Must be **always possible** — if you can't compensate, don't use sagas for that step.
- Compensations are **semantic undos**, not technical rollbacks — the original operation happened.
- Use a saga state tracker (or the order's status field) to know which compensations are needed.

---

### Q12. What is a dead letter queue (DLQ)? When should you retry vs DLQ?

A **dead letter queue (DLQ)** is a separate queue where messages that cannot be processed successfully are moved after exhausting retry attempts. It prevents a poison-pill message from blocking the entire queue indefinitely.

**Why DLQs are necessary:**
```
Normal processing:
  Message → Consumer → Process → Acknowledge → Message removed

Failed processing without DLQ:
  Message → Consumer → FAILS → Retry → FAILS → Retry → Retry → ∞
  Queue head is blocked by one bad message
  All subsequent messages pile up unprocessed
  (called "head-of-line blocking")
```

**With DLQ:**
```
Message fails 3 times → moved to DLQ → queue continues with next message
Alert fired: "message moved to DLQ, manual investigation required"
Operations team investigates cause, fixes bug, replays message from DLQ
```

**Retry vs DLQ decision framework:**

| Error Type | Action | Rationale |
|---|---|---|
| Transient (network timeout, DB connection) | Retry with exponential backoff | Likely recovers on retry |
| Resource temporarily unavailable | Retry (few attempts) | Resource may become available |
| Invalid message format | DLQ immediately | Retrying won't fix a malformed message |
| Business logic violation | DLQ immediately | Bug in consumer; retry doesn't help |
| Downstream service permanently down | DLQ after N retries | Service may not recover |
| Dependency version mismatch | DLQ | Schema incompatibility needs engineering fix |

**Implementation (SQS with DLQ):**
```python
# CloudFormation / Terraform:
# Main queue with redrive policy (max 3 retries)
main_queue = sqs.create_queue(
    QueueName='order-events',
    Attributes={
        'RedrivePolicy': json.dumps({
            'deadLetterTargetArn': dlq_arn,
            'maxReceiveCount': '3'  # After 3 failed attempts → DLQ
        }),
        'VisibilityTimeout': '30'   # 30s to process before re-visibility
    }
)

# Consumer with explicit failure handling:
def consume_order_event(message):
    try:
        process_order_event(message)
        sqs.delete_message(...)    # Acknowledge success
    except ValidationError as e:
        # Don't retry — move to DLQ immediately by NOT deleting and letting it expire
        logger.error(f"Invalid message format: {e}, message will DLQ")
        # Optionally: manually move to DLQ for faster resolution
        sqs.send_message(QueueUrl=dlq_url, MessageBody=message.body)
        sqs.delete_message(...)    # Remove from main queue immediately
    except TransientError:
        pass   # Let SQS retry (don't delete)
```

**DLQ monitoring:**
- Alert when DLQ depth > 0 (any message in DLQ needs attention).
- Alert when DLQ depth grows over time (systematic failure, not one-off).
- Dashboard showing DLQ age (how long messages have been waiting).

---

### Q13. How do you handle event schema evolution without breaking consumers?

Event schema evolution is the challenge of changing the structure of events over time while keeping existing consumers working. This is the distributed systems equivalent of database schema migration — but potentially more complex because consumers may be independently deployed and may process historical events.

**Evolution strategies:**

**1. Backward compatibility (new producer, old consumer):**
- Old consumer can read new event format.
- Rule: **Add fields only; never remove or rename; never change field types**.

```json
// V1 event (original):
{ "order_id": "ord_123", "total": 5000 }

// V2 event (new field added):
{ "order_id": "ord_123", "total": 5000, "currency": "USD" }
// Old consumer ignores "currency" → still works
```

**2. Forward compatibility (old producer, new consumer):**
- New consumer can read old event format.
- Rule: **New consumers must handle missing optional fields**.

```python
# New consumer handling V1 events (missing "currency"):
class OrderProcessor:
    def handle_order_placed(self, event):
        total = event['total']
        currency = event.get('currency', 'USD')  # Default for V1 events
        process_payment(total, currency)
```

**3. Schema Registry (Avro/Protobuf with Confluent Schema Registry):**
```
# Avro schema V1:
{
  "type": "record",
  "name": "OrderPlaced",
  "fields": [
    {"name": "order_id", "type": "string"},
    {"name": "total", "type": "long"}
  ]
}

# Avro schema V2 (backward compatible — new field with default):
{
  "type": "record",
  "name": "OrderPlaced",
  "fields": [
    {"name": "order_id", "type": "string"},
    {"name": "total", "type": "long"},
    {"name": "currency", "type": "string", "default": "USD"}  ← has default → backward compat
  ]
}
```

Schema Registry enforces compatibility rules before a new schema version is registered — preventing breaking changes from reaching production.

**4. Event versioning (explicit version in event type):**
```
Event types:
  "OrderPlaced.v1" → old format
  "OrderPlaced.v2" → new format with currency field
  
Consumers:
  Legacy consumer: subscribes to "OrderPlaced.v1" only
  New consumer: subscribes to both "OrderPlaced.v1" and "OrderPlaced.v2"
  
Migration path:
  1. Deploy new producer (emits both v1 and v2)
  2. Deploy new consumer (handles both)
  3. Retire old consumer
  4. Stop emitting v1
```

**What NEVER to do:**
- Remove a field that consumers depend on.
- Change a field's type (e.g., `int` → `string`).
- Rename a field without keeping the old name as an alias.
- Change the semantic meaning of an existing field.

---

### Q14. Explain CQRS with event sourcing. How are read models built as event projections?

**CQRS (Command Query Responsibility Segregation)** separates the write model (commands that change state) from the read model (queries that return data). Combined with event sourcing, the read models are built by **projecting** (subscribing to and processing) events from the event log.

**Architecture:**
```
Write Side (Commands):
  Client ──command──▶ Command Handler
                            │
                            ▼
                      Event Store (append-only)
                      [OrderCreated, ItemAdded, OrderPaid, ...]
                            │
                            ▼
                      Event Bus (Kafka)
                            │
                            ├──▶ Projection 1: Order Summary Read Model (Redis)
                            ├──▶ Projection 2: Order Search Index (Elasticsearch)
                            ├──▶ Projection 3: User Order History (PostgreSQL)
                            └──▶ Projection 4: Analytics (BigQuery)

Read Side (Queries):
  Client ──query──▶ Read Model (optimised for specific query pattern)
                     Returns denormalised, pre-computed view
```

**Building a read model projection:**
```python
class OrderSummaryProjection:
    """
    Listens to order events and maintains a Redis read model
    optimised for "get order summary by order_id" queries
    """
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def handle(self, event):
        handlers = {
            "OrderCreated":   self.on_created,
            "ItemAdded":      self.on_item_added,
            "OrderPaid":      self.on_paid,
            "OrderShipped":   self.on_shipped,
            "OrderDelivered": self.on_delivered,
        }
        handler = handlers.get(event.type)
        if handler:
            handler(event)
    
    def on_created(self, event):
        self.redis.hset(f"order:{event.aggregate_id}", mapping={
            "status": "created",
            "total": event.payload["total"],
            "customer_id": event.payload["customer_id"],
            "item_count": 0,
            "created_at": event.occurred_at
        })
    
    def on_item_added(self, event):
        self.redis.hincrby(f"order:{event.aggregate_id}", "item_count", 1)
        self.redis.hincrbyfloat(f"order:{event.aggregate_id}", "total", 
                                event.payload["price"])
    
    def on_paid(self, event):
        self.redis.hset(f"order:{event.aggregate_id}", mapping={
            "status": "paid",
            "paid_at": event.occurred_at,
            "payment_id": event.payload["payment_id"]
        })
```

**Rebuilding a projection (event replay):**
```python
def rebuild_order_summary_projection(event_store, projection, from_beginning=True):
    """
    Wipe and rebuild the read model from the event log.
    Used when fixing bugs or adding new fields to the read model.
    """
    projection.clear_all()
    
    for event in event_store.stream_all(
        event_types=["OrderCreated", "ItemAdded", "OrderPaid", "OrderShipped"],
        from_sequence=0 if from_beginning else projection.last_processed_sequence
    ):
        projection.handle(event)
        projection.last_processed_sequence = event.sequence
```

**Benefits of multiple projections:**
- Each read model is optimised for a different query pattern — no schema compromises.
- A bug in one projection can be fixed by rebuilding from events — no data loss.
- New query patterns can be served by new projections without touching the write model.

---

### Q15. How do you test event-driven systems? Describe consumer contract tests and event replay testing.

Testing event-driven systems is more complex than testing synchronous systems because the interactions are asynchronous, loosely coupled, and time-dependent.

**1. Unit tests for event handlers:**
```python
def test_inventory_service_handles_order_placed():
    # Arrange
    inventory_repo = FakeInventoryRepository({"SKU_001": 100})
    event_bus = FakeEventBus()
    service = InventoryService(inventory_repo, event_bus)
    
    event = OrderPlacedEvent(order_id="ord_123", sku="SKU_001", quantity=2)
    
    # Act
    service.handle_order_placed(event)
    
    # Assert
    assert inventory_repo.get_quantity("SKU_001") == 98
    assert event_bus.published_count("InventoryReserved") == 1
    published = event_bus.get_published("InventoryReserved")[0]
    assert published.order_id == "ord_123"
```

**2. Consumer contract tests (Pact):**
Contract tests prevent integration failures by verifying that the event format a consumer expects matches what the producer actually publishes.

```python
# Consumer side (Inventory Service) defines its contract:
@pact
def test_order_placed_contract():
    expected_event = {
        "event_type": "OrderPlaced",
        "payload": {
            "order_id": Like("ord_123"),        # Must exist, any string
            "sku": Like("SKU_001"),
            "quantity": Like(2),                # Must be a number
            "customer_id": Like("cust_789")
        }
    }
    
    # Verify our handler works with this event shape
    handler = InventoryService().handle_order_placed
    handler(expected_event)
    # If handler processes it without error → contract is satisfied

# Producer side (Order Service) verifies it publishes the expected format:
# Pact broker compares consumer's expected format against producer's actual output
# CI fails if producer changes format in a way that breaks any consumer's contract
```

**3. Integration tests with an embedded broker:**
```python
import pytest
from testcontainers.kafka import KafkaContainer

@pytest.fixture(scope="session")
def kafka():
    with KafkaContainer() as container:
        yield container.get_bootstrap_server()

def test_order_placement_flow_integration(kafka):
    producer = KafkaProducer(bootstrap_servers=kafka)
    consumer = KafkaConsumer("inventory-events", bootstrap_servers=kafka)
    
    # Publish an order event
    producer.send("order-events", value=b'{"type":"OrderPlaced","order_id":"ord_123",...}')
    producer.flush()
    
    # Start inventory service (connects to test Kafka)
    inventory_service = InventoryService(kafka_server=kafka)
    inventory_service.start()
    
    # Verify inventory event emitted within 5 seconds
    for msg in consumer:
        event = json.loads(msg.value)
        if event["type"] == "InventoryReserved" and event["order_id"] == "ord_123":
            assert event["reservation_id"] is not None
            break
    else:
        pytest.fail("InventoryReserved event not received within timeout")
```

**4. Event replay testing for projections:**
```python
def test_order_projection_replay():
    event_store = InMemoryEventStore()
    
    # Seed historical events
    events = [
        OrderCreatedEvent(order_id="ord_1", total=1000, ts="2024-01-01"),
        PaymentReceivedEvent(order_id="ord_1", amount=1000, ts="2024-01-01"),
        OrderShippedEvent(order_id="ord_1", tracking="TRK001", ts="2024-01-02"),
    ]
    event_store.append_all(events)
    
    # Build projection via replay
    projection = OrderSummaryProjection(InMemoryStore())
    rebuild_projection(event_store, projection)
    
    # Verify final state
    summary = projection.get("ord_1")
    assert summary["status"] == "shipped"
    assert summary["tracking_number"] == "TRK001"
```

---

## Hard (Q16–Q20)

---

### Q16. How do you implement event replay to rebuild views and debug production issues?

Event replay is one of the most powerful capabilities of event-sourced systems — it turns your event store into a time machine. Implementing it correctly requires addressing ordering, idempotency, and performance.

**Production event replay use cases:**

**1. Read model rebuild (most common):**
```python
class ProjectionRebuildOrchestrator:
    def __init__(self, event_store, projection, batch_size=1000):
        self.event_store = event_store
        self.projection = projection
        self.batch_size = batch_size
    
    def rebuild(self, from_timestamp=None):
        print(f"Starting projection rebuild from {from_timestamp or 'beginning'}")
        
        # Step 1: Build new projection in parallel (don't disrupt live service)
        new_projection = self.projection.create_empty_copy()
        
        # Step 2: Replay events in batches (avoid loading all events into memory)
        cursor = None
        total_processed = 0
        
        while True:
            batch = self.event_store.fetch_events(
                after_cursor=cursor,
                limit=self.batch_size,
                from_timestamp=from_timestamp
            )
            if not batch:
                break
            
            for event in batch:
                new_projection.apply(event)
            
            cursor = batch[-1].cursor
            total_processed += len(batch)
            print(f"Processed {total_processed} events...")
        
        # Step 3: Catch up any events that arrived during replay
        self.catchup(new_projection, cursor)
        
        # Step 4: Atomic swap — new projection becomes live
        self.projection.atomic_swap(new_projection)
        print("Projection rebuild complete")
    
    def catchup(self, projection, from_cursor):
        """Process events that arrived during the rebuild"""
        while True:
            new_events = self.event_store.fetch_events(after_cursor=from_cursor, limit=100)
            if not new_events:
                return  # Caught up
            for event in new_events:
                projection.apply(event)
            from_cursor = new_events[-1].cursor
```

**2. Debugging a production incident:**
```python
def debug_order_incident(order_id, time_range):
    """Replay all events for one order to trace what happened"""
    events = event_store.fetch_events(
        aggregate_id=order_id,
        from_time=time_range.start,
        to_time=time_range.end
    )
    
    order = Order(id=order_id)
    for event in events:
        print(f"[{event.occurred_at}] {event.type}: {event.payload}")
        previous_state = copy.deepcopy(order)
        order.apply(event)
        diff = compute_diff(previous_state, order)
        print(f"  State change: {diff}")
```

**3. Testing a bug fix:**
```python
def test_fix_against_production_events():
    """
    Replay 1 week of production events through a fixed version of the code
    to verify the fix resolves the known issue without introducing regressions
    """
    production_events = event_store.fetch_events(
        from_time=datetime.now() - timedelta(days=7),
        to_time=datetime.now()
    )
    
    # Use the fixed code (not production code)
    new_processor = FixedOrderProcessor()
    errors_before = 0
    errors_after = 0
    
    for event in production_events:
        try:
            new_processor.process(event)
        except KnownBugError:
            errors_before += 1
        except Exception as e:
            errors_after += 1  # New errors introduced by fix
    
    assert errors_after == 0, f"Fix introduced {errors_after} new errors"
    print(f"Fixed {errors_before} previously failing events")
```

**Rate control during replay:** Replay can put significant load on downstream systems. Implement rate limiting:
```python
replayer = EventReplayer(
    events_per_second=1000,  # Don't overwhelm downstream
    skip_side_effects=True   # Don't re-send emails during replay
)
```

---

### Q17. What is event storming? How does it help design a large event-driven system?

**Event storming** is a collaborative workshop technique invented by Alberto Brandolini for exploring complex domains by identifying domain events, commands, aggregates, and bounded contexts through a structured facilitated session.

**The workshop format:**
```
Materials needed:
  - Large wall or roll of paper (6+ metres wide)
  - Orange sticky notes  → Domain events
  - Blue sticky notes    → Commands
  - Yellow sticky notes  → Aggregates (entities)
  - Lilac sticky notes   → Policies ("whenever X, do Y")
  - Red sticky notes     → Hotspots (confusion, disagreement)
  - Green sticky notes   → Read models / views
  - Pink sticky notes    → External systems / actors

Participants:
  - Domain experts (product managers, business analysts)
  - Developers (especially backend / integration)
  - Architects
  - UX designers (who understand user workflows)
```

**Phase 1 — Unstructured exploration (chaotic):**
- Everyone simultaneously writes domain events on orange stickies.
- No order, no discussion — just brainstorming.
- Events are phrased in past tense: "OrderPlaced", "PaymentDeclined", "ItemShipped".
- Place all events on the wall in rough chronological order.

**Phase 2 — Enforce timeline:**
- Facilitator helps the group arrange events on a left-to-right timeline.
- Hotspots (red stickies) mark areas of confusion or disagreement.
- Questions like "What must happen before this event can occur?" reveal dependencies.

**Phase 3 — Add commands, aggregates, and policies:**
```
[User clicks "Buy"]             [Policy: When PaymentFailed]
      │                                   │
   (command)                         → CancelOrder
  PlaceOrder                         → NotifyUser
      │
 [Order aggregate]
      │ triggers
      ▼
  OrderPlaced ──────────────────▶ [Policy: When OrderPlaced]
                                       │
                                       └──▶ ReserveInventory (command)
                                                 │
                                            InventoryReserved
                                                 │
                                                 └──▶ ChargePayment (command)
```

**Outcomes:**
1. **Bounded context identification** — clusters of closely related events/aggregates become bounded context candidates.
2. **Integration points** — where events flow between clusters reveals the service-to-service contracts.
3. **Aggregate discovery** — which commands and events belong to the same "thing" that changes together.
4. **Hotspot resolution** — team aligns on disputed areas before writing code.

**Example output for an e-commerce system:**
```
Bounded Context 1: Order Management
  Events: OrderPlaced, OrderModified, OrderCancelled, OrderFulfilled
  Aggregates: Order

Bounded Context 2: Inventory
  Events: StockReserved, StockReleased, StockReplenished, StockDepleted
  Aggregates: StockItem

Bounded Context 3: Payment
  Events: PaymentInitiated, PaymentProcessed, PaymentFailed, RefundIssued
  Aggregates: Payment

Integration events (flow between contexts):
  OrderPlaced → Inventory (triggers StockReserved)
  StockReserved → Payment (triggers PaymentInitiated)
  PaymentFailed → Order (triggers OrderCancelled)
  PaymentFailed → Inventory (triggers StockReleased)
```

---

### Q18. Design an event-driven architecture for a payments system end-to-end.

**System overview:**
A payments system must handle order payment with reliability, exactly-once processing, compliance audit trail, and fault tolerance.

**Event flow:**
```
┌──────────────────────────────────────────────────────────────────┐
│                    Client (Web / Mobile)                          │
│  POST /orders  →  202 Accepted (async)                           │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│               Order Service (writes to outbox)                    │
│  Writes: orders table + outbox table (same transaction)          │
│  Publishes via Debezium CDC: OrderPlaced                         │
└────────────────────────────┬─────────────────────────────────────┘
                             │ OrderPlaced event
                             ▼
                    ┌────────────────┐
                    │  Kafka Topic   │
                    │ order-events   │ (partition by order_id)
                    └───────┬────────┘
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
┌──────────────────┐ ┌───────────────┐ ┌──────────────────┐
│ Inventory Service│ │Payment Service│ │Fraud Check       │
│                  │ │               │ │Service           │
│ reserves stock   │ │               │ │                  │
│ → InventoryReserved│               │ │→ FraudCheckPassed│
│   or             │ │               │ │   or             │
│ → InsufficientStock│              │ │→ FraudCheckFailed│
└─────────┬────────┘ └──────┬────────┘ └──────┬───────────┘
          │                 │                  │
          └─────────────────▼──────────────────┘
                  PaymentOrchestrator
                  (collects all required events)
                  → charges card only when:
                     InventoryReserved AND FraudCheckPassed
                  → aborts if either fails
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
          PaymentProcessed      PaymentFailed
                  │                   │
                  ▼                   ▼
       ┌──────────────────┐  ┌──────────────────┐
       │ Fulfillment      │  │ Compensation Saga│
       │ → creates shipment│ │ → releases stock │
       │ → ShipmentCreated│ │ → cancels order  │
       └──────────────────┘ └──────────────────┘
```

**Exactly-once guarantees at each stage:**

```python
class PaymentService:
    def handle_payment_initiation(self, event):
        """
        Processes payment exactly once using outbox + idempotency key
        """
        order_id = event['order_id']
        
        # Step 1: Check if already processed (inbox deduplication)
        if self.inbox_store.contains(event['event_id']):
            return   # Already processed
        
        with self.db.transaction():
            # Step 2: Attempt payment with idempotency key
            charge_result = self.stripe.charge(
                amount=event['amount'],
                idempotency_key=f"payment-{order_id}",  # Idempotent at Stripe
                card_token=event['card_token']
            )
            
            # Step 3: Record payment
            self.db.insert_payment(order_id, charge_result.charge_id)
            
            # Step 4: Write outbox event (same transaction)
            self.db.insert_outbox(
                event_type="PaymentProcessed",
                payload={"order_id": order_id, "charge_id": charge_result.charge_id}
            )
            
            # Step 5: Mark event as processed (inbox)
            self.db.insert_inbox(event_id=event['event_id'])
        # All 4 writes commit atomically
```

**Audit trail (event sourcing for compliance):**
```
payments_event_store (append-only):
  order_123 | FraudCheckRequested | {ip: "1.2.3.4", device: "mobile"}    | T+0ms
  order_123 | FraudCheckPassed    | {score: 12, provider: "Sift"}        | T+230ms
  order_123 | InventoryReserved   | {sku: "BOOK_001", qty: 1}            | T+150ms
  order_123 | PaymentInitiated    | {amount: 2999, method: "card"}       | T+300ms
  order_123 | PaymentProcessed    | {charge_id: "ch_xyz", amount: 2999}  | T+450ms
  order_123 | OrderFulfilled      | {shipment_id: "ship_abc"}            | T+600ms

Complete audit trail for any dispute or regulatory inquiry
```

---

### Q19. When should you NOT use event-driven architecture?

Understanding when NOT to use a pattern is as important as knowing how to use it. EDA is not universally superior — it has genuine drawbacks that make it the wrong choice in specific scenarios.

**Anti-pattern 1: Simple CRUD applications:**
```
Requirement: Simple blog platform — create/read/update/delete posts
  
EDA approach: PostCreated event → Kafka → consumer updates read DB
  Complexity: event broker, consumer, schema registry, outbox pattern
  Value: none (single service, no other consumers)

Better: Simple REST API → PostgreSQL
  Single transaction, immediate consistency, no infrastructure overhead
  
Rule: If there is only ONE consumer of the data (the same service), use synchronous writes.
```

**Anti-pattern 2: Strong consistency required:**
```
Requirement: Bank transfer — debit account A, credit account B
  Both must happen or neither must happen
  
EDA approach: TransferInitiated → Debit saga → Credit saga
  Problem: during the saga, account A is debited but account B not yet credited
  A user checking their balance mid-saga sees missing money
  Compensation (rollback debit) adds complexity and risk

Better: Two-phase commit within the same database
  Single atomic transaction: debit + credit in one BEGIN/COMMIT
  
Rule: If operations must be atomically consistent (no intermediate state visible), use transactions not sagas.
```

**Anti-pattern 3: Sub-millisecond latency requirements:**
```
Requirement: High-frequency trading — price feed processing < 1ms

EDA approach: event broker adds 1-10ms minimum
  Kafka network round-trip: ~1ms per event
  Serialization/deserialization: additional overhead
  
Better: Shared memory, direct function calls, or UDP multicast
  In-process queues or ring buffers (LMAX Disruptor pattern): nanosecond latency
  
Rule: If latency requirements are sub-millisecond, any network hop (including broker) is too slow.
```

**Anti-pattern 4: Small team / simple domain:**
```
Team: 2 engineers, simple e-commerce site, 1,000 orders/day

EDA approach:
  Infrastructure: Kafka cluster, schema registry, Debezium, multiple consumer services
  Engineering: event schema design, consumer contract tests, DLQ handling
  Operational: monitoring 5+ services instead of 1
  
Better: Monolith with direct function calls
  Deploy a single well-structured application
  Add EDA when you have > 3 teams needing independent deployment cadences

Rule: Don't solve team scaling problems (independent deployability) before you have the team scale problem.
```

**Anti-pattern 5: Frequent event schema changes:**
```
Requirement: Product is in rapid iteration; event schemas change weekly

EDA approach: every schema change requires:
  - Consumer updates
  - Schema compatibility testing
  - Coordinated deployment
  - Historical event handling (old schema events still in broker)

Better: Direct service calls with versioned APIs
  API versioning (v1/v2) is simpler to reason about
  GraphQL lets clients request only fields they need
  
Rule: Event schemas are a public contract — they are expensive to change. Use EDA when schemas are stable.
```

**Summary:**

| Scenario | Use EDA? | Alternative |
|---|---|---|
| Simple CRUD | No | Direct DB writes |
| Strong consistency required | No | DB transactions |
| Sub-millisecond latency | No | In-process queues |
| Small team / simple domain | Not yet | Monolith |
| Frequent schema changes | No | REST API versioning |
| Multiple services react to one event | Yes | EDA shines |
| Independent scalability needed | Yes | EDA shines |
| Temporal decoupling needed | Yes | EDA shines |

---

### Q20. Design a complete event-driven architecture for an e-commerce order system. Include all events, sagas, and failure modes.

This is the canonical system design case study for event-driven architecture. A complete design covers the happy path, all failure modes, and operational concerns.

**All domain events in the system:**
```
Order lifecycle:
  OrderPlaced          → customer submitted order
  OrderConfirmed       → all checks passed, order accepted
  OrderCancelled       → order cancelled (by customer or system)
  OrderFulfilled       → shipped to customer
  OrderDelivered       → delivery confirmed

Inventory:
  InventoryReserved    → stock allocated for order
  InventoryReleased    → reservation cancelled (compensation)
  InventoryDepleted    → stock level reached zero
  InventoryReplenished → restocking completed

Payment:
  PaymentInitiated     → payment attempt started
  PaymentProcessed     → successful charge
  PaymentFailed        → card declined or error
  RefundIssued         → customer refunded

Fulfillment:
  ShipmentCreated      → shipment label generated
  ShipmentPickedUp     → carrier collected package
  ShipmentDelivered    → delivery confirmed by carrier
  ShipmentFailed       → delivery failure (address issue etc.)
```

**Complete system architecture:**
```
┌────────────────────────────────────────────────────────────────────────────┐
│                           Kafka Event Bus                                   │
│  Topics: order-events, inventory-events, payment-events, fulfillment-events │
│  Partitioned by: order_id (ensures ordered delivery per order)              │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼─────────────────────────────────┐
    ▼                               ▼                                  ▼
Order Service               Inventory Service                 Payment Service
(command handler)           (event consumer)                 (event consumer)
    │                               │                                  │
    │ OrderPlaced                   │ reserves stock                   │ charges card
    │ ──────────────────────────────┼──────────────────────────────────┼──▶
    │                               │ InventoryReserved                │
    │                               │ InsufficientStock                │ PaymentProcessed
    │                               ▼                                  │ PaymentFailed
    │                       Fulfillment Service                        │
    │                       (consumes PaymentProcessed)                │
    │                               │ ShipmentCreated                  │
    │                               │ ShipmentDelivered                │
    ▼                               ▼                                  ▼
  Read Models:           Order History Read Model           Payment Audit Log
  (Redis/Postgres)       (CQRS projection)                 (append-only)
```

**Happy path event sequence:**
```
T+0ms:   OrderPlaced { order_id, items, customer_id, total }
T+10ms:  FraudCheckPassed { order_id, risk_score: 5 }
T+50ms:  InventoryReserved { order_id, items: [{sku, qty}] }
T+100ms: PaymentInitiated { order_id, amount, payment_method }
T+350ms: PaymentProcessed { order_id, charge_id, amount }
T+400ms: OrderConfirmed { order_id, estimated_delivery }
T+600ms: ShipmentCreated { order_id, tracking_number, carrier }
T+86400000ms: ShipmentDelivered { order_id, signed_by }
T+86400001ms: OrderFulfilled { order_id }
```

**Failure mode: Payment declined:**
```
PaymentFailed { order_id, reason: "insufficient_funds" }
  │
  ├──▶ InventoryService: listens for PaymentFailed
  │        → releases reservation
  │        → publishes InventoryReleased
  │
  ├──▶ OrderService: listens for PaymentFailed
  │        → sets order status = 'payment_failed'
  │        → publishes OrderCancelled { reason: "payment_failed" }
  │
  └──▶ NotificationService: listens for OrderCancelled
           → sends "payment failed" email to customer
           → suggests retrying with different payment method
```

**Failure mode: Stock unavailable after payment:**
```
PaymentProcessed
InventoryService: can't fulfill (item sold out since reservation)
   → publishes InventoryFulfillmentFailed

Compensation saga:
  PaymentService: listens for InventoryFulfillmentFailed
    → issues refund to customer
    → publishes RefundIssued

  OrderService: listens for InventoryFulfillmentFailed
    → publishes OrderCancelled { reason: "out_of_stock" }

  NotificationService: listens for OrderCancelled + RefundIssued
    → sends "sorry, item unavailable, refund issued" email
```

**Dead letter queue handling:**
```python
class OrderEventConsumer:
    MAX_RETRIES = 3
    
    def process(self, event):
        for attempt in range(self.MAX_RETRIES):
            try:
                self._process_with_inbox(event)
                return
            except TransientError as e:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                # Exhausted retries → DLQ
                self.dlq.publish(event, error=str(e), attempts=attempt+1)
                self.alerts.fire("order_event_dlq", event_id=event.id)
    
    def _process_with_inbox(self, event):
        if self.inbox.contains(event.id):
            return  # Already processed
        with self.db.transaction():
            self.business_logic(event)
            self.inbox.mark_processed(event.id)
```

**Observability:**
```
Distributed tracing:
  Every event carries: trace_id, span_id (propagated from original HTTP request)
  OpenTelemetry spans created for each consumer invocation
  Jaeger/Zipkin shows complete order journey across all services

Key metrics:
  event_processing_latency_p99{service="inventory", event_type="OrderPlaced"}
  event_consumer_lag{topic="order-events", consumer_group="inventory-service"}
  dlq_depth{service="payment"}

SLOs:
  OrderPlaced → PaymentProcessed: P99 < 2 seconds
  PaymentProcessed → ShipmentCreated: P99 < 5 seconds
  Consumer lag: P99 < 10 seconds (events processed within 10s of being published)
```

---

## Quick Reference

| Topic | Key Point |
|---|---|
| EDA vs request-response | EDA: async, loose coupling, eventual consistency; RR: sync, tight coupling, immediate consistency |
| Event vs command vs query | Event: past tense, broadcast; Command: imperative, directed; Query: read-only, sync |
| Domain events | Business-meaningful facts in past tense; enable loose coupling between bounded contexts |
| Kafka vs RabbitMQ vs EventBridge | Kafka: replay, high-throughput; Rabbit: task queues, routing; EventBridge: AWS-native, serverless |
| Choreography | Services react to events independently; resilient; hard to debug |
| Orchestration | Central coordinator; easier to debug; SPOF risk |
| Event sourcing | Append-only event log as source of truth; full audit trail; support replay |
| Snapshots | Cache state every N events; load snapshot + N delta events; 100x+ speedup |
| Eventual consistency | Lag between write and read model update; solve with read-your-writes pattern |
| Out-of-order events | Partition by entity key (Kafka); sequence number buffers; tolerant state machines |
| Dual-write problem | DB + broker can't be written atomically without outbox pattern |
| Outbox pattern | Write event to DB outbox in same transaction; Debezium/relay publishes to broker |
| Saga + choreography | Each service emits success/failure events; others react with compensations |
| Dead letter queue | After N retries, move to DLQ; alert; investigate; replay after fix |
| Schema evolution | Add fields with defaults only; use schema registry; version event types |
| CQRS + event sourcing | Write side → event store; projections consume events → read models |
| Event storming | Collaborative workshop; orange=events, blue=commands; discovers bounded contexts |
| Consumer contract tests | Pact tests; prevent breaking schema changes from reaching production |
| Payments EDA | OrderPlaced → FraudCheck + InventoryReserve + PaymentCharge (all parallel) → all succeed → OrderConfirmed |
| When NOT to use EDA | Simple CRUD, strong consistency, sub-ms latency, small team, frequent schema changes |
