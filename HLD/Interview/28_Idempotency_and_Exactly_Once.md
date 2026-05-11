# 28 — Idempotency and Exactly-Once

---

## Easy (Q1–Q7)

---

### Q1. What is idempotency and why does it matter in distributed systems?

**Idempotency** means that performing an operation multiple times produces the same result as performing it exactly once. The term comes from mathematics: `f(f(x)) = f(x)`.

In distributed systems, idempotency is critical because **failures and retries are not exceptional — they are routine**. Networks drop packets, services restart, timeouts occur before a response arrives. Without idempotency, retrying a failed operation can cause duplicate side effects: double charges, duplicate orders, double email sends.

**Concrete example:**

```
Client sends: POST /payments  { amount: $100, card: 4242... }
  Server receives request → charges card
  Server response: 200 OK { payment_id: abc123 }
  Network failure → client never receives response

Client retry: POST /payments  { amount: $100, card: 4242... }
  Non-idempotent server: charges card AGAIN → $200 charged
  Idempotent server: detects duplicate → returns { payment_id: abc123 }
                     same response, no second charge
```

**The key insight:** In a distributed system, from the client's perspective, a timeout is indistinguishable from a failure. The client cannot know whether the server processed the request. Therefore, **clients must retry**, and **servers must handle retries safely**.

**Where idempotency matters most:**
- Payment processing (double charges are catastrophic)
- Order creation (duplicate orders)
- Email/SMS dispatch (duplicate notifications)
- Database mutations (unintended duplicate records)
- Message consumers (processing a message twice)

Idempotency is achieved through: idempotency keys, conditional writes, deduplication stores, and database constraints. A system that cannot be safely retried is fragile by design.

---

### Q2. Which HTTP methods are idempotent and which are not?

The HTTP specification defines idempotency expectations for each method. Understanding this is foundational for API design.

**Idempotent methods:**

| Method | Idempotent? | Safe? | Description |
|---|---|---|---|
| GET | Yes | Yes | Read-only; same response for same URL |
| HEAD | Yes | Yes | Like GET but no response body |
| PUT | Yes | No | Replace resource entirely; same result if called twice |
| DELETE | Yes | No | Resource gone after first call; second call has no extra effect |
| OPTIONS | Yes | Yes | Metadata query; no state change |

**Non-idempotent methods:**

| Method | Idempotent? | Safe? | Description |
|---|---|---|---|
| POST | No | No | Creates a new resource; calling twice creates two resources |
| PATCH | Depends | No | Partial update; depends on operation semantics |

**PUT vs PATCH idempotency:**
```
PUT /users/123 { name: "Alice", email: "alice@example.com" }
→ Always results in exactly this state, regardless of how many times called
→ IDEMPOTENT

PATCH /users/123 { increment_score: 1 }
→ Each call increments score by 1
→ NOT IDEMPOTENT

PATCH /users/123 { score: 10 }
→ Always sets score to exactly 10
→ IDEMPOTENT (depends on the operation — set vs increment)
```

**Why POST is non-idempotent:**
```
POST /orders { items: [...] }
→ First call: creates order #1001
→ Second call: creates order #1002 (duplicate)
→ Each POST creates a new resource
```

This is why POST endpoints for resource creation require **idempotency keys** — the HTTP spec makes no promise that POST is safe to retry. The application must implement deduplication logic explicitly.

---

### Q3. How do you implement idempotency keys for POST APIs?

An **idempotency key** is a client-generated unique identifier attached to a request that allows the server to detect and deduplicate retries. Stripe pioneered this pattern for payment APIs and it is now industry standard.

**Implementation steps:**

**1. Client generates a UUID and attaches it to the request:**
```http
POST /api/v1/payments
Idempotency-Key: 550e8400-e29b-41d4-a716-446655440000
Content-Type: application/json

{ "amount": 5000, "currency": "USD", "card_token": "tok_visa" }
```

**2. Server checks idempotency store before processing:**
```python
def process_payment(request):
    idempotency_key = request.headers.get("Idempotency-Key")
    if not idempotency_key:
        return 400, "Idempotency-Key header required"
    
    # Check if we've seen this key before
    cached = idempotency_store.get(idempotency_key)
    if cached:
        return cached["status_code"], cached["response_body"]
    
    # Mark as in-flight (prevent concurrent processing)
    idempotency_store.set(idempotency_key, {"status": "processing"}, ttl=60)
    
    # Process the payment
    result = payment_processor.charge(request.body)
    
    # Store the result with longer TTL
    idempotency_store.set(
        idempotency_key,
        {"status_code": 200, "response_body": result},
        ttl=86400 * 7  # 7 days
    )
    
    return 200, result
```

**3. Idempotency store options:**
- **Redis**: `SET idempotency:{key} {response} EX 604800 NX` — atomic, fast, TTL supported.
- **Database table**: `INSERT INTO idempotency_keys (key, response, created_at) ON CONFLICT DO NOTHING`.

**Schema:**
```sql
CREATE TABLE idempotency_keys (
    key           VARCHAR(255) PRIMARY KEY,
    request_hash  VARCHAR(64),     -- SHA256 of request body
    status_code   INT,
    response_body JSONB,
    created_at    TIMESTAMP DEFAULT NOW(),
    expires_at    TIMESTAMP DEFAULT NOW() + INTERVAL '7 days'
);
CREATE INDEX ON idempotency_keys (expires_at);  -- for cleanup job
```

**Handling conflicting requests with same key:**
- If the same key is submitted with different request bodies → `422 Unprocessable Entity` (the key is bound to the first request's parameters).
- Use `request_hash` to detect this case.

---

### Q4. What are at-most-once, at-least-once, and exactly-once delivery semantics?

These three delivery guarantees describe what a messaging system promises about how many times a consumer will process a given message.

**At-most-once:**
- Message is delivered zero or one time — never more than once.
- If the system fails between sending and acknowledging, the message is lost.
- No duplicates, but messages can be dropped.

```
Producer → send message → (don't wait for ack)
Consumer receives → (crash before processing)
Message lost — never reprocessed
```

Use case: Fire-and-forget metrics, log streaming where some loss is acceptable.

**At-least-once:**
- Message is delivered one or more times — never zero times.
- The system retries on failure, but retries can cause duplicates.
- No message loss, but duplicates possible.

```
Producer → send message → wait for ack
Broker crashes after receiving but before sending ack
Producer retries → message delivered twice to consumer
```

Use case: Most production systems; consumers implement idempotency to handle duplicates.

**Exactly-once:**
- Message is delivered exactly one time — no loss, no duplicates.
- Hardest to achieve; requires coordination between producer, broker, and consumer.

```
Kafka exactly-once:
  Producer: sequence numbers prevent duplicate produces
  Broker: deduplicates by producer ID + sequence number
  Consumer: transactional reads + commits ensure processed-once
```

**Comparison table:**

| Semantic | Message Loss | Duplicates | Complexity | Cost |
|---|---|---|---|---|
| At-most-once | Possible | None | Low | Low |
| At-least-once | None | Possible | Medium | Medium |
| Exactly-once | None | None | High | High |

**Practical recommendation:** Design consumers to be idempotent and use at-least-once delivery. True exactly-once at the infrastructure level is expensive and complex; idempotent consumers achieve the same business outcome with at-least-once delivery.

---

### Q5. How does Kafka achieve exactly-once semantics with idempotent producers and transactions?

Kafka provides exactly-once semantics (EOS) through two complementary mechanisms: **idempotent producers** and the **transactional API**.

**Idempotent Producer (exactly-once per partition):**
- Each producer is assigned a `producer_id` by the broker.
- Each message gets a monotonically increasing `sequence_number`.
- Broker deduplicates: if it receives a message with the same `producer_id + sequence_number` it already committed, it discards the duplicate silently.

```
Producer (pid=1): sends {seq=0, msg="A"} → broker stores
Producer (pid=1): retries {seq=0, msg="A"} → broker sees seq=0 already stored → discards
Producer (pid=1): sends {seq=1, msg="B"} → broker stores
```

Enable with: `enable.idempotence=true` in producer config.

**Transactional API (exactly-once across partitions + consumer offset):**
The transactional API ensures that a produce + consumer offset commit is atomic — either both happen or neither does.

```python
producer = KafkaProducer(
    transactional_id='payment-processor-1',
    enable_idempotence=True
)

producer.init_transactions()

for message in consumer:
    try:
        producer.begin_transaction()
        
        # Process and produce result
        result = process_payment(message)
        producer.send('payments-processed', value=result)
        
        # Commit consumer offset AND produced message atomically
        producer.send_offsets_to_transaction(
            {TopicPartition('payments', 0): OffsetAndMetadata(message.offset + 1, '')},
            consumer_group_id='payment-consumers'
        )
        producer.commit_transaction()
        
    except Exception:
        producer.abort_transaction()
```

**Consumer isolation — read_committed:**
```python
consumer = KafkaConsumer(
    isolation_level='read_committed'  # Only see committed messages
)
```

With `read_committed`, consumers never see messages from aborted transactions, preventing processing of "phantom" messages.

**Limitations:**
- Exactly-once is within Kafka only. If your consumer also writes to a database, the DB write is outside the transaction — you need the outbox pattern for true end-to-end exactly-once.
- Transaction overhead: ~20–30% throughput reduction vs at-least-once.

---

### Q6. What is the outbox pattern and how does it solve the dual-write problem?

The **dual-write problem** arises when an application needs to write to two separate systems atomically — typically a database and a message broker. Without coordination, one write can succeed while the other fails, leaving systems in an inconsistent state.

**The problem:**
```
Case 1: DB write succeeds, event publish fails
  → Order created in DB but OrderCreated event never published
  → Inventory service never decremented stock
  → Order is orphaned

Case 2: Event published, DB write fails
  → Inventory service decrements stock for an order that doesn't exist
  → Ghost reservation
```

**The outbox pattern solution:**
Instead of writing to DB and broker separately, write the event to an **outbox table** in the same database transaction as the business data. A separate process (the outbox relay) reads the outbox and publishes to the broker.

```
Step 1: Single atomic DB transaction
┌─────────────────────────────────────────┐
│ BEGIN TRANSACTION                        │
│   INSERT INTO orders (id, ...) VALUES .. │
│   INSERT INTO outbox (event_type,        │
│     payload, created_at, published=false)│
│     VALUES ('OrderCreated', {...}, ...)  │
│ COMMIT                                   │
└─────────────────────────────────────────┘

Step 2: Outbox relay (separate process)
  SELECT * FROM outbox WHERE published = false LIMIT 100
  For each event: publish to Kafka/SQS
  UPDATE outbox SET published = true WHERE id = ?
```

**Outbox table schema:**
```sql
CREATE TABLE outbox (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type    VARCHAR(100) NOT NULL,
    aggregate_id  VARCHAR(255) NOT NULL,
    payload       JSONB NOT NULL,
    created_at    TIMESTAMP DEFAULT NOW(),
    published_at  TIMESTAMP,
    retry_count   INT DEFAULT 0
);
```

**Relay implementations:**
- **Polling**: Simple loop, but adds DB load and latency.
- **Debezium (CDC)**: Captures DB binlog/WAL changes — near-zero latency, no polling overhead. The preferred production approach.

The outbox pattern guarantees **at-least-once** event delivery. Consumers must still be idempotent — the relay may publish the same event more than once if it crashes between publish and marking published.

---

### Q7. Explain at-most-once, at-least-once and the inbox pattern for consumer-side deduplication.

The **inbox pattern** is the consumer-side complement to the outbox pattern. While the outbox ensures reliable event publishing, the inbox ensures reliable event consumption — preventing duplicate processing even when the message broker delivers messages more than once.

**The problem without inbox:**
```
Kafka: delivers OrderCreated message to consumer
Consumer: starts processing → sends email → updates DB
Consumer crashes before committing Kafka offset
Kafka: redelivers OrderCreated to consumer on restart
Consumer: sends email AGAIN → duplicate email sent to user
```

**Inbox pattern solution:**
```python
def consume_order_created(message):
    event_id = message.headers['event_id']
    
    with db.transaction():
        # Check if we've already processed this event
        exists = db.execute(
            "SELECT 1 FROM inbox WHERE event_id = %s",
            [event_id]
        ).fetchone()
        
        if exists:
            # Already processed — idempotent ignore
            kafka_consumer.commit()
            return
        
        # Process the event
        send_order_confirmation_email(message.value)
        update_inventory(message.value)
        
        # Mark as processed in the same transaction
        db.execute(
            "INSERT INTO inbox (event_id, processed_at) VALUES (%s, NOW())",
            [event_id]
        )
    # Commit Kafka offset only after DB transaction commits
    kafka_consumer.commit()
```

**Inbox table schema:**
```sql
CREATE TABLE inbox (
    event_id     VARCHAR(255) PRIMARY KEY,
    event_type   VARCHAR(100),
    processed_at TIMESTAMP DEFAULT NOW()
);

-- Cleanup job: DELETE FROM inbox WHERE processed_at < NOW() - INTERVAL '7 days'
```

**Key properties of inbox pattern:**
- The inbox record insert and business logic are in the same transaction — they commit or rollback together.
- If the transaction commits but Kafka offset commit fails → Kafka redelivers → `SELECT 1` finds existing record → safely ignored.
- This achieves **exactly-once business logic execution** on top of at-least-once message delivery.

**Deduplication window:** Inbox records must be retained long enough to cover the broker's redelivery window. For Kafka with default settings, 7 days is usually sufficient.

---

## Medium (Q8–Q15)

---

### Q8. How do you make a payment API idempotent end-to-end?

Payment idempotency is the most critical form of idempotency — double charges have direct financial and reputational consequences. A production-grade payment API requires multiple layers of idempotency working together.

**Complete flow:**

```
Client                     Payment API               Payment Processor
  │                              │                         │
  │ POST /payments               │                         │
  │ Idempotency-Key: key-abc123  │                         │
  │ { amount: 5000, token: ... } │                         │
  │─────────────────────────────▶│                         │
  │                              │ Check idempotency store │
  │                              │ key-abc123 not found    │
  │                              │                         │
  │                              │ INSERT INTO payments    │
  │                              │ (idempotency_key,       │
  │                              │  status='pending')      │
  │                              │─ charge(token, 5000) ──▶│
  │                              │                         │ charge_id=ch_xyz
  │                              │◀── { charge_id } ───────│
  │                              │                         │
  │                              │ UPDATE payments SET     │
  │                              │ status='success',       │
  │                              │ charge_id='ch_xyz'      │
  │◀─────────────────────────────│                         │
  │ 200 { payment_id, charge_id }│                         │
  │                              │                         │
  │ (Network timeout — client    │                         │
  │  never got response)         │                         │
  │                              │                         │
  │ RETRY: POST /payments        │                         │
  │ Idempotency-Key: key-abc123  │                         │
  │─────────────────────────────▶│                         │
  │                              │ Check idempotency store │
  │                              │ key-abc123 FOUND        │
  │                              │ status='success'        │
  │◀─────────────────────────────│                         │
  │ 200 { payment_id, charge_id }│  (same response, no     │
  │  (cached response returned)  │   second charge)        │
```

**Database schema for idempotent payments:**
```sql
CREATE TABLE payments (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key  VARCHAR(255) UNIQUE NOT NULL,
    amount           BIGINT NOT NULL,
    currency         CHAR(3) NOT NULL,
    status           VARCHAR(20) DEFAULT 'pending',
    charge_id        VARCHAR(255),   -- from payment processor
    request_hash     VARCHAR(64),    -- SHA256(amount + currency + card_token)
    response_body    JSONB,
    created_at       TIMESTAMP DEFAULT NOW(),
    expires_at       TIMESTAMP DEFAULT NOW() + INTERVAL '7 days'
);
```

**Handling race conditions (concurrent retries):**
```sql
-- Atomic insert: only one request wins, others get error
INSERT INTO payments (idempotency_key, amount, request_hash)
VALUES ($1, $2, $3)
ON CONFLICT (idempotency_key) DO NOTHING
RETURNING id;

-- If id is NULL, another request is processing:
SELECT status, response_body FROM payments WHERE idempotency_key = $1;
```

**Edge case: request in 'pending' state (processing started but not yet finished):**
- Return `HTTP 202 Accepted` with a `Retry-After: 5` header.
- Client polls for completion.
- Do not start a second processing attempt for the same key.

---

### Q9. What is optimistic locking and how does it ensure idempotent updates?

**Optimistic locking** assumes that conflicts between concurrent writes are rare. Rather than acquiring a lock before reading, it reads freely but performs a conditional update that fails if the data was modified between the read and write.

This is implemented via a **version field** or **timestamp** on the record, checked at write time using a Compare-And-Swap (CAS) operation.

**Schema:**
```sql
CREATE TABLE user_profiles (
    user_id    UUID PRIMARY KEY,
    name       VARCHAR(255),
    email      VARCHAR(255),
    version    BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMP
);
```

**Read-modify-write cycle:**
```python
def update_user_profile(user_id, new_name, new_email):
    # Step 1: Read current state with version
    user = db.execute(
        "SELECT name, email, version FROM user_profiles WHERE user_id = %s",
        [user_id]
    ).fetchone()
    
    current_version = user['version']
    
    # Step 2: Apply modifications locally
    # ... validation, business logic ...
    
    # Step 3: Conditional update — only succeeds if version unchanged
    rows_updated = db.execute("""
        UPDATE user_profiles
        SET name = %s, email = %s, version = version + 1, updated_at = NOW()
        WHERE user_id = %s AND version = %s
    """, [new_name, new_email, user_id, current_version]).rowcount
    
    if rows_updated == 0:
        raise OptimisticLockException("Concurrent modification detected, retry")
    
    return True
```

**Why this achieves idempotency:**
```
Client A reads version=5, prepares update
Client B reads version=5, prepares different update
Client A writes: WHERE version=5 → succeeds, version becomes 6
Client B writes: WHERE version=5 → FAILS (version is now 6)
Client B retries: reads version=6, recalculates, writes WHERE version=6 → succeeds

Second retry of Client A's same request:
  WHERE version=5 → FAILS (version is now 6+)
  → No double update occurs
```

**Optimistic vs pessimistic locking comparison:**

| Aspect | Optimistic Locking | Pessimistic Locking |
|---|---|---|
| Lock held | No lock held | Row lock held during transaction |
| Conflict handling | Detected at write, retry needed | Prevented at read via lock |
| Performance | High (no lock wait) | Lower (lock contention) |
| Best for | Low conflict, read-heavy | High conflict, write-heavy |
| Risk | Starvation under high contention | Deadlock, lock timeout |

---

### Q10. What is a fencing token and why do distributed locks need them?

A **fencing token** is a monotonically increasing number returned by a distributed lock service that the lock holder must include in any operation protected by the lock. It prevents a slow/paused lock holder from overwriting data written by a newer lock holder.

**The problem without fencing tokens:**
```
Client 1 acquires lock from Redis at time T=0 (TTL = 30s)
Client 1 pauses (GC pause, network delay) for 35 seconds
Redis: lock TTL expires at T=30
Client 2 acquires same lock at T=31
Client 2 writes data to DB: { value: "Client2's data" }
Client 1 resumes at T=35 (still thinks it holds the lock)
Client 1 writes data to DB: { value: "Client1's stale data" }
PROBLEM: Client 2's write was overwritten by Client 1
```

**With fencing tokens:**
```
Lock service issues token #100 to Client 1
Client 1 pauses...
Lock service issues token #101 to Client 2
Client 2 writes to storage: { value: "C2 data", fencing_token: 101 }
  Storage records: "highest seen token = 101"
Client 1 resumes, writes: { value: "C1 data", fencing_token: 100 }
  Storage sees: 100 < 101 → REJECT (stale lock holder)
```

**Implementation in storage layer:**
```python
class FencedStorage:
    def __init__(self):
        self.highest_seen_token = 0
        self.lock = threading.Lock()
    
    def write(self, data, fencing_token):
        with self.lock:
            if fencing_token <= self.highest_seen_token:
                raise StaleTokenException(
                    f"Token {fencing_token} rejected, "
                    f"highest seen: {self.highest_seen_token}"
                )
            self.highest_seen_token = fencing_token
            self.data = data
```

**Important caveat (Martin Kleppmann):** Fencing tokens require that the **storage system** enforces the token check. This is the only reliable way to prevent a slow lock holder from corrupting data. A distributed lock alone (Redis Redlock, ZooKeeper) cannot prevent this problem — the fencing must be enforced at the point of the protected resource, not at the lock service.

---

### Q11. How does the saga pattern handle consistency failures with compensating transactions?

The **saga pattern** manages long-running distributed transactions across multiple services. Instead of a two-phase commit (which requires all services to be available and locked), a saga decomposes a transaction into a sequence of local transactions, each followed by a domain event. If any step fails, compensating transactions are executed in reverse order to undo the completed steps.

**Example: Book travel (flight + hotel + car)**

```
Forward transactions:
  T1: Reserve flight       → emit FlightReserved
  T2: Reserve hotel        → emit HotelReserved
  T3: Charge credit card   → emit PaymentProcessed
  T4: Confirm all bookings → emit BookingConfirmed

Compensating transactions (if T3 fails):
  C2: Cancel hotel reservation
  C1: Cancel flight reservation
  (C3 doesn't run — payment never happened)
```

**Choreography-based saga (event-driven):**
```
FlightService  →  FlightReserved event
                         │
                         ▼
HotelService   →  HotelReserved event
                         │
                         ▼
PaymentService →  (fails) → PaymentFailed event
                         │
                         ▼
HotelService   →  listens for PaymentFailed → cancels hotel
                         │
                         ▼
FlightService  →  listens for HotelCancelled → cancels flight
```

**Properties of compensating transactions:**
- They must be **idempotent** — they may be invoked multiple times if the saga orchestrator crashes.
- They undo the business effect, not necessarily the technical operation.
- They cannot always fully undo: sending an email cannot be unsent; compensating transaction might send a "disregard previous email" instead.
- They must be **always possible** — if you can't compensate, you can't use sagas for that step.

**Saga vs 2PC comparison:**

| Aspect | Saga | 2PC |
|---|---|---|
| Locking | No locks across services | All participants locked during prepare phase |
| Availability | High (no distributed lock) | Lower (any participant failing blocks transaction) |
| Consistency | Eventual (temporary inconsistency during saga) | Immediate (all-or-nothing) |
| Complexity | High (compensating logic needed) | Medium (protocol complexity) |
| Failure handling | Explicit compensating transactions | Coordinator manages rollback |

---

### Q12. How do databases support idempotent operations natively?

Databases provide several built-in mechanisms for idempotent writes, eliminating the need for application-level check-then-insert logic (which is vulnerable to race conditions).

**INSERT ON CONFLICT DO NOTHING (PostgreSQL):**
```sql
-- Safe: if the row already exists, silently ignore
INSERT INTO payments (idempotency_key, amount, status)
VALUES ('key-abc123', 5000, 'pending')
ON CONFLICT (idempotency_key) DO NOTHING;
```

**UPSERT — INSERT ON CONFLICT DO UPDATE:**
```sql
-- Idempotent create-or-update
INSERT INTO user_sessions (session_id, user_id, last_seen)
VALUES ('sess_xyz', 'user_123', NOW())
ON CONFLICT (session_id) DO UPDATE
  SET last_seen = EXCLUDED.last_seen
  WHERE user_sessions.last_seen < EXCLUDED.last_seen;  -- Only update if newer
```

**MySQL INSERT IGNORE:**
```sql
INSERT IGNORE INTO email_sends (idempotency_key, email, sent_at)
VALUES ('key-abc123', 'user@example.com', NOW());
-- Silently ignores duplicate key errors
```

**DynamoDB conditional writes:**
```python
table.put_item(
    Item={'payment_id': 'key-abc123', 'amount': 5000},
    ConditionExpression='attribute_not_exists(payment_id)'
    # Fails with ConditionalCheckFailedException if item exists
    # Client catches this and returns cached response
)
```

**MERGE statement (SQL Server, Oracle):**
```sql
MERGE INTO payments AS target
USING (SELECT 'key-abc123' AS idempotency_key, 5000 AS amount) AS source
ON target.idempotency_key = source.idempotency_key
WHEN NOT MATCHED THEN
    INSERT (idempotency_key, amount) VALUES (source.idempotency_key, source.amount)
WHEN MATCHED THEN
    DO NOTHING;  -- Already exists, don't modify
```

**Why application-level check-then-insert is wrong:**
```python
# WRONG — race condition:
existing = db.query("SELECT * FROM payments WHERE key = %s", [key])
if not existing:
    db.execute("INSERT INTO payments ...")  # Another request can insert here!

# RIGHT — database atomic constraint:
db.execute("INSERT INTO payments ... ON CONFLICT DO NOTHING")
```

---

### Q13. How long should idempotency keys be retained? What factors determine the window?

The **deduplication window** — how long idempotency keys and their cached responses are retained — is a trade-off between storage cost, correctness, and the realistic retry behaviour of clients.

**Factors that determine the retention window:**

**1. Client retry window:**
- A client implementing exponential backoff with a maximum retry time of 48 hours needs the server to remember the key for at least 48 hours.
- Formula: `retention >= max_client_retry_window`

**2. Broker/queue visibility timeout:**
- SQS visibility timeout: up to 12 hours. If a consumer takes longer than 12 hours to process, SQS redelivers.
- For queue-based systems: `retention >= visibility_timeout + processing_time`

**3. Storage cost:**
- Each key record: ~500 bytes.
- At 1M requests/day × 7 days = 7M records × 500 bytes = 3.5 GB — negligible.
- At 1B requests/day × 30 days = 30B records × 500 bytes = 15 TB — significant.

**Industry standards:**

| Company | Retention Period | Reasoning |
|---|---|---|
| Stripe | 24 hours | Payment retries expected within hours |
| Stripe (critical APIs) | 7 days | Covers weekend delays, manual retries |
| Twilio | 4 hours | SMS delivery is fast; long window unnecessary |
| AWS SQS dedup | 5 minutes | Based on visibility timeout window |
| PayPal | 45 days | Matches dispute resolution window |

**Recommended tiered approach:**
```
Transient operations (session actions): 1 hour
Standard API operations: 24 hours
Financial operations (payments): 7 days
Compliance-sensitive operations: 90 days (match dispute window)
```

**Cleanup strategy:**
```sql
-- Scheduled job (run hourly):
DELETE FROM idempotency_keys
WHERE expires_at < NOW()
  AND LIMIT 10000;  -- Batch to avoid table lock
```

**Indefinite retention is wrong:** Storing idempotency keys forever is a GDPR violation if the key or response contains personal data, and is operationally unnecessary. Design a cleanup job from day one.

---

### Q14. How do message brokers handle deduplication? Compare Kafka and SQS.

Message brokers provide different deduplication guarantees, and understanding these determines what application-level deduplication is still required.

**Kafka deduplication:**
- **Producer-side**: `enable.idempotence=true` assigns each producer a `producer_id` and tracks sequence numbers. Duplicate retries with the same sequence number are discarded by the broker.
- **Scope**: Per producer session and per partition. Deduplication does not survive producer restarts (new `producer_id` assigned).
- **Consumer-side**: Kafka does NOT deduplicate for consumers. A consumer may process the same message twice if:
  - It commits offset after processing, and crashes after processing but before commit.
  - It is restarted and reprocesses from the last committed offset.

```
Kafka deduplication boundaries:
  ✓ Duplicate produce retries within same producer session → deduplicated
  ✗ Producer restart → new producer_id → duplicate possible
  ✗ Consumer rebalance → consumer may reprocess uncommitted messages
```

**SQS deduplication:**
- **Standard queues**: At-least-once delivery. Duplicates are possible and expected. Applications must implement inbox pattern.
- **FIFO queues**: Exactly-once within a **5-minute deduplication window**. Uses `MessageDeduplicationId` (or content-based SHA256 hash).

```python
sqs.send_message(
    QueueUrl='https://sqs.us-east-1.amazonaws.com/123/MyQueue.fifo',
    MessageBody='{"order_id": "ord_123", "amount": 5000}',
    MessageGroupId='order-ord_123',
    MessageDeduplicationId='ord_123-payment-attempt-1'
    # Any message with same deduplication ID within 5 minutes is discarded
)
```

**SQS FIFO limitation:** The 5-minute window means that if a processing job takes > 5 minutes and must retry, duplicates become possible again.

**Summary comparison:**

| Feature | Kafka (idempotent) | SQS Standard | SQS FIFO |
|---|---|---|---|
| Producer dedup | Yes (within session) | No | No |
| Consumer dedup | No | No | Yes (5-min window) |
| Ordering | Per partition | No guarantee | Per MessageGroupId |
| Throughput | Very high | Very high | Limited (3,000/s) |
| Application dedup needed? | Yes (consumer side) | Yes | Usually not |

---

### Q15. What are the most common mistakes that break idempotency?

Understanding idempotency failure patterns is as important as knowing how to implement idempotency correctly. Here are the most common mistakes observed in production systems.

**Mistake 1: Generating random IDs inside the handler instead of using request-provided IDs:**
```python
# WRONG: Each retry generates a new order_id
def create_order(request):
    order_id = str(uuid.uuid4())  # New UUID every time!
    db.insert("orders", {"id": order_id, ...})
    return {"order_id": order_id}

# RIGHT: Use idempotency key as the deterministic ID
def create_order(request):
    idempotency_key = request.headers["Idempotency-Key"]
    order_id = idempotency_key  # Or deterministically derived from it
    db.execute("INSERT INTO orders (id, ...) ON CONFLICT DO NOTHING", [order_id, ...])
```

**Mistake 2: Performing non-idempotent side effects before checking idempotency:**
```python
# WRONG: Email sent before idempotency check
def create_order(request):
    send_confirmation_email(request.user_email)  # Side effect first!
    existing = check_idempotency_key(request.headers["key"])
    if existing:
        return existing  # Email already sent twice
```

**Mistake 3: Partial idempotency (protecting the DB write but not external calls):**
```python
# WRONG: DB is idempotent but payment processor is called multiple times
def create_order(request):
    key = request.headers["Idempotency-Key"]
    if not db.check_key(key):
        db.insert_order_with_key(key, ...)
        # NOT IDEMPOTENT: no idempotency key passed to payment processor
        stripe.charge(request.card_token, request.amount)
```

**Mistake 4: Race condition in check-then-act without atomic write:**
```python
# WRONG: Check and insert are not atomic
if not db.exists("SELECT 1 FROM keys WHERE key = %s", [key]):
    # Another thread can insert the same key here!
    db.execute("INSERT INTO keys ...")
```

**Mistake 5: Ignoring different responses for same idempotency key:**
```python
# WRONG: Returns different response each time based on current state
def get_payment_status(request):
    key = request.headers["Idempotency-Key"]
    payment = db.get_payment_by_key(key)
    return payment  # Status might have changed between calls
    # Should always return the SAME response as the original call
```

**Mistake 6: Short deduplication window that expires before the client's retry window:**
- Setting 5-minute idempotency TTL when clients can retry for up to 24 hours due to queue backpressure.

---

## Hard (Q16–Q20)

---

### Q16. Why cannot two-phase commit (2PC) achieve true exactly-once in all failure scenarios?

**Two-phase commit (2PC)** is the classic distributed protocol for achieving atomicity across multiple participants. Despite its widespread use, it has fundamental failure modes that prevent it from guaranteeing exactly-once in all scenarios.

**2PC Protocol:**
```
Phase 1 (Prepare):
  Coordinator → all participants: "Can you commit?"
  Participants → Coordinator: "Yes" or "No"

Phase 2 (Commit):
  If all Yes: Coordinator → all: "Commit"
  If any No:  Coordinator → all: "Abort"
```

**Failure scenario 1: Coordinator fails after Phase 1 but before Phase 2:**
```
Coordinator sends PREPARE to all participants → all respond YES
Coordinator crashes (now holds the only knowledge of the decision)

Participants are now in "prepared" state — they have locks held
They cannot commit (coordinator didn't send COMMIT)
They cannot abort (coordinator might have sent COMMIT to others)
They are STUCK until coordinator recovers
→ System is blocked (availability violated)
```

**Failure scenario 2: Coordinator and one participant fail simultaneously:**
```
Coordinator sends COMMIT to participant A → A commits, coordinator crashes
Coordinator sends COMMIT to participant B → B never receives it (coordinator dead)

Recovery: New coordinator asks participants for their state
  A: "I committed"
  B: "I'm in prepared state"
Decision: Unknown — new coordinator cannot determine if the original coordinator
committed B before crashing
→ B must wait, or system administrator must intervene manually
```

**Failure scenario 3: Network partition during commit:**
```
Coordinator → COMMIT → Participant A (succeeds)
Coordinator → COMMIT → Participant B (network partition, never arrives)

After partition heals:
  A has committed
  B is in prepared state
  B asks coordinator: coordinator may have crashed and been replaced
  
If B decides to abort independently (timeout): inconsistency — A committed, B aborted
If B waits forever: availability violated
```

**The fundamental limitation:** 2PC is a **blocking protocol**. If the coordinator fails after Phase 1, participants must block until it recovers. This sacrifices availability. True exactly-once under all failure scenarios requires either: (a) accepting blocking (2PC), or (b) using a non-blocking protocol like Paxos/Raft-based consensus, which has its own complexity and performance cost.

**Practical conclusion:** 2PC is appropriate for short-duration transactions within a single database or tightly coupled systems (e.g., two databases in the same data centre). For loosely coupled microservices, the saga pattern with idempotent compensating transactions is more reliable despite providing only eventual consistency.

---

### Q17. How do you test idempotency? What does a complete idempotency test suite look like?

Testing idempotency requires verifying not just that the happy path works, but that **side effects occur exactly once** regardless of how many times the operation is attempted.

**Test 1: Basic idempotency — same request twice, single side effect:**
```python
def test_payment_idempotency_basic():
    key = "test-key-001"
    payload = {"amount": 5000, "currency": "USD", "card": "tok_visa"}
    
    # First call
    resp1 = client.post("/payments", json=payload, 
                        headers={"Idempotency-Key": key})
    assert resp1.status_code == 200
    payment_id = resp1.json()["payment_id"]
    
    # Second call (retry simulation)
    resp2 = client.post("/payments", json=payload,
                        headers={"Idempotency-Key": key})
    assert resp2.status_code == 200
    assert resp2.json()["payment_id"] == payment_id  # Same response
    
    # Verify: only ONE charge exists
    charges = stripe_mock.get_charges(card="tok_visa", amount=5000)
    assert len(charges) == 1, f"Expected 1 charge, got {len(charges)}"
```

**Test 2: Three retries, exactly one side effect:**
```python
def test_payment_idempotency_multiple_retries():
    key = "test-key-002"
    responses = []
    
    for i in range(3):
        resp = client.post("/payments", json={"amount": 1000},
                           headers={"Idempotency-Key": key})
        responses.append(resp)
    
    # All responses identical
    assert all(r.status_code == 200 for r in responses)
    assert len(set(r.json()["payment_id"] for r in responses)) == 1
    
    # Exactly one email sent
    emails_sent = email_mock.count_sent_to("user@example.com")
    assert emails_sent == 1
```

**Test 3: Concurrent requests with same key — only one processed:**
```python
def test_payment_idempotency_concurrent():
    key = "test-key-concurrent"
    results = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(create_payment, key) for _ in range(5)]
        results = [f.result() for f in futures]
    
    payment_ids = [r["payment_id"] for r in results if r.get("payment_id")]
    unique_ids = set(payment_ids)
    
    assert len(unique_ids) == 1, "Concurrent requests produced multiple payments"
    assert stripe_mock.charge_count() == 1
```

**Test 4: Different keys produce different results:**
```python
def test_different_keys_produce_different_payments():
    resp1 = client.post("/payments", json={"amount": 5000},
                        headers={"Idempotency-Key": "key-001"})
    resp2 = client.post("/payments", json={"amount": 5000},
                        headers={"Idempotency-Key": "key-002"})
    
    assert resp1.json()["payment_id"] != resp2.json()["payment_id"]
    assert stripe_mock.charge_count() == 2
```

**Test 5: Simulated failure mid-processing — retry completes correctly:**
```python
def test_idempotency_after_partial_failure():
    # First attempt: fail after DB insert but before stripe charge
    with mock.patch('app.stripe.charge', side_effect=Exception("Network error")):
        resp1 = client.post("/payments", json={"amount": 5000},
                            headers={"Idempotency-Key": "key-fail"})
        assert resp1.status_code == 500
    
    # Second attempt: should complete successfully
    resp2 = client.post("/payments", json={"amount": 5000},
                        headers={"Idempotency-Key": "key-fail"})
    assert resp2.status_code == 200
    
    # Verify: exactly one charge
    assert stripe_mock.charge_count() == 1
```

**Test 6: Expired idempotency key creates new payment:**
```python
def test_expired_idempotency_key():
    key = "key-expired"
    client.post("/payments", json={"amount": 5000}, headers={"Idempotency-Key": key})
    
    # Advance time past TTL
    time_machine.advance(days=8)
    
    resp = client.post("/payments", json={"amount": 5000}, headers={"Idempotency-Key": key})
    assert resp.status_code == 200
    # This creates a NEW payment — old key expired
    assert stripe_mock.charge_count() == 2
```

---

### Q18. What is idempotency in batch processing? How does checkpoint-based resume work?

Batch processing jobs — ETL pipelines, nightly reports, data migrations — must be idempotent in a different sense than APIs: if a batch job fails halfway through, rerunning it from the start should not double-process the first half.

**The problem without idempotent batch processing:**
```
Batch job: process 10M orders, compute tax, write to tax_calculations table
Job runs for 4 hours, writes 7M records, then crashes

Naive restart: runs from beginning again
  → 7M records processed AGAIN
  → Tax calculations doubled for those records (if using INSERT not UPSERT)
  → Downstream reports incorrect
```

**Checkpoint-based approach:**
```python
class IdempotentBatchJob:
    def __init__(self, job_name, checkpoint_store):
        self.job_name = job_name
        self.checkpoint_store = checkpoint_store
    
    def run(self, source_records):
        # Load last checkpoint
        last_processed_id = self.checkpoint_store.get(
            f"{self.job_name}:last_id", default=None
        )
        
        # Resume from checkpoint
        if last_processed_id:
            print(f"Resuming from: {last_processed_id}")
        
        batch = []
        for record in source_records.after(last_processed_id):
            batch.append(self.process_record(record))
            
            if len(batch) >= 1000:
                self.flush_batch(batch)
                # Checkpoint: record last successfully processed ID
                self.checkpoint_store.set(
                    f"{self.job_name}:last_id", 
                    batch[-1]["source_id"]
                )
                batch = []
        
        if batch:
            self.flush_batch(batch)
            self.checkpoint_store.set(f"{self.job_name}:last_id", batch[-1]["source_id"])
    
    def flush_batch(self, batch):
        # Use UPSERT to ensure idempotent writes
        db.execute_many("""
            INSERT INTO tax_calculations (order_id, tax_amount, calculated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (order_id) DO UPDATE
              SET tax_amount = EXCLUDED.tax_amount,
                  calculated_at = EXCLUDED.calculated_at
        """, [(r["order_id"], r["tax"], r["timestamp"]) for r in batch])
```

**Date-partitioned idempotency:**
```
For daily batch jobs: use date as the natural idempotency key
  Partition: s3://data/processed/date=2024-01-15/
  If partition exists: skip (already processed)
  If partition missing: process and write

DELETE FROM tax_calculations WHERE calculation_date = '2024-01-15';
INSERT INTO tax_calculations SELECT ... FROM orders WHERE date = '2024-01-15';
-- This is idempotent: always produces identical results for same date
```

**Spark checkpoint idempotency:**
```python
# Spark Structured Streaming: built-in checkpoint support
query = (df
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", "s3://checkpoints/job-name/")
    .start("s3://output/tax-calculations/"))
# Checkpoint tracks exactly which micro-batches have been processed
# Resuming after failure replays only unprocessed batches
```

---

### Q19. What is an idempotent retry mechanism? Combine exponential backoff, idempotency keys, and circuit breakers.

A **safe retry mechanism** combines three complementary patterns: exponential backoff prevents retry storms, idempotency keys prevent duplicate side effects, and circuit breakers prevent retrying a permanently failed service.

**Complete implementation:**

```python
import time
import random
import uuid
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests fast
    HALF_OPEN = "half_open" # Testing if service recovered

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        self.recovery_timeout = recovery_timeout
    
    def call_allowed(self):
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN: allow one test request
    
    def record_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

class SafeRetryClient:
    def __init__(self, service_client):
        self.client = service_client
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
    
    def create_payment(self, amount, currency, card_token,
                       max_retries=3, base_delay=1.0):
        # Generate idempotency key ONCE — reuse across all retries
        idempotency_key = str(uuid.uuid4())
        
        for attempt in range(max_retries + 1):
            # Check circuit breaker before attempting
            if not self.circuit_breaker.call_allowed():
                raise CircuitOpenException("Payment service circuit is OPEN")
            
            try:
                response = self.client.post(
                    "/payments",
                    json={"amount": amount, "currency": currency, "card": card_token},
                    headers={"Idempotency-Key": idempotency_key},
                    timeout=5
                )
                
                if response.status_code in (200, 201):
                    self.circuit_breaker.record_success()
                    return response.json()
                
                # 4xx errors: don't retry (client error, idempotency key mismatch)
                if 400 <= response.status_code < 500:
                    raise ClientError(f"Non-retryable error: {response.status_code}")
                
                # 5xx errors: retry with backoff
                self.circuit_breaker.record_failure()
                
            except (ConnectionError, TimeoutError) as e:
                self.circuit_breaker.record_failure()
                if attempt == max_retries:
                    raise
            
            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                delay = min(delay, 30)  # Cap at 30 seconds
                print(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s "
                      f"(key: {idempotency_key})")
                time.sleep(delay)
        
        raise MaxRetriesExceeded("Payment failed after all retries")
```

**Backoff schedule:**
```
Attempt 0: immediate
Attempt 1: wait 1s + jitter (0.0–1.0s)  → ~1–2s
Attempt 2: wait 2s + jitter              → ~2–3s
Attempt 3: wait 4s + jitter              → ~4–5s
Circuit opens after 5 failures in succession
  → No retries for 60s (recovery timeout)
  → Half-open: one test request allowed
```

**The key property:** The `idempotency_key` is generated **once** before the retry loop. All three retries submit the same key. The server returns the same response after the first successful processing — the client receives a consistent result regardless of which attempt first succeeded.

---

### Q20. Design a safe, idempotent payment processing system end-to-end.

This question integrates all idempotency concepts into a complete system design.

**System requirements:**
- Accept payments from mobile/web clients.
- Exactly-once processing guarantee (no double charges).
- Reliable event publishing for downstream services (inventory, fulfillment, email).
- Handle partial failures gracefully.
- Support payment status queries.

**Complete architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client (Mobile/Web)                       │
│  Generates idempotency_key = UUID on button press               │
│  Stores key locally; reuses on retry                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │ POST /payments
                            │ Idempotency-Key: {uuid}
                            │ Authorization: Bearer {jwt}
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway + WAF                           │
│  Rate limiting: 10 req/s per user (prevents retry storms)       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Payment Service                               │
│                                                                  │
│  1. Validate request (amount, currency, card_token)             │
│  2. Check idempotency store (Redis)                             │
│     → HIT: return cached response                               │
│     → MISS: proceed                                             │
│  3. BEGIN TRANSACTION                                           │
│     a. INSERT INTO payments (key, status='initiated', amount)   │
│        ON CONFLICT DO NOTHING                                   │
│     b. INSERT INTO outbox (event_type='PaymentInitiated', ...)  │
│  4. COMMIT TRANSACTION (atomic DB + outbox)                     │
│  5. Call payment processor (Stripe/Braintree)                   │
│     → Pass idempotency_key to processor API                     │
│  6. BEGIN TRANSACTION                                           │
│     a. UPDATE payments SET status='completed', charge_id=...    │
│     b. INSERT INTO outbox (event_type='PaymentCompleted', ...)  │
│  7. COMMIT TRANSACTION                                          │
│  8. Cache response in Redis (TTL=7 days)                        │
│  9. Return 200 { payment_id, status: 'completed' }             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Outbox Relay (Debezium CDC)                     │
│  Reads outbox table via PostgreSQL WAL / binlog                 │
│  Publishes events to Kafka topic: payments                      │
│  → At-least-once delivery (Debezium may retry)                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│              Kafka Topic: payments (idempotent producers)        │
│              Consumers: Inbox pattern for deduplication          │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Email Service                                              │  │
│  │  Inbox: INSERT INTO email_inbox (event_id) ON CONFLICT ... │  │
│  │  Sends confirmation email exactly once                     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Fulfillment Service                                        │  │
│  │  Inbox: deduplication on event_id                         │  │
│  │  Creates shipment exactly once                             │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**Idempotency at each layer:**

| Layer | Mechanism | Guarantee |
|---|---|---|
| Client | Store key on button press, reuse on retry | Same key for all retries |
| API Gateway | Redis idempotency key check | Early dedup before business logic |
| Payment service DB | `ON CONFLICT DO NOTHING` | Atomic single DB record |
| Payment processor | Pass key to Stripe API | Stripe deduplicates charges |
| Outbox relay | Debezium + Kafka idempotent producer | At-least-once event publish |
| Email consumer | Inbox table + DB transaction | Exactly-once email send |
| Fulfillment consumer | Inbox table + DB transaction | Exactly-once shipment creation |

**Failure handling matrix:**

| Failure Point | What Happens | Recovery |
|---|---|---|
| Client timeout | Client retries with same key | Server returns cached response |
| Payment service crash mid-processing | DB in 'initiated' state | Background job detects stale 'initiated' records, queries processor for status |
| Stripe returns error | Transaction rolls back, client gets 500 | Client retries; Stripe idempotency key prevents double charge |
| Kafka publish fails | Outbox record remains unpublished | Debezium retries from WAL |
| Email consumer crashes | Kafka redelivers; inbox check deduplicates | Email sent exactly once |

---

## Quick Reference

| Topic | Key Point |
|---|---|
| Idempotency | Same operation multiple times = same result; critical for retry safety |
| HTTP idempotency | GET/PUT/DELETE idempotent; POST is not |
| Idempotency key | Client-generated UUID in header; stored in DB with response; TTL 24h–7 days |
| At-most-once | Messages may be lost, never duplicated |
| At-least-once | No message loss, duplicates possible; most practical choice |
| Exactly-once | No loss, no duplicates; expensive; Kafka achieves within broker |
| Kafka EOS | `enable.idempotence=true` (producer) + `isolation_level=read_committed` (consumer) |
| Outbox pattern | Write event to DB in same transaction as business data; relay publishes to broker |
| Inbox pattern | Consumer inserts event_id to DB before processing; deduplicate retries |
| Payment idempotency | Idempotency key + conditional insert + cache response + pass key to Stripe |
| Optimistic locking | Version field; `UPDATE ... WHERE version=N`; retry on 0 rows updated |
| Fencing token | Monotonic number from lock service; storage rejects stale token |
| Saga pattern | Local transactions + compensating transactions; eventual consistency |
| INSERT ON CONFLICT | Atomic database-level idempotency; no check-then-insert race condition |
| Deduplication window | 24h for APIs; 7 days for payments; match client retry window |
| Kafka dedup | Sequence numbers within producer session; consumers still need inbox |
| SQS FIFO dedup | 5-minute deduplication window via MessageDeduplicationId |
| 2PC limitation | Blocking protocol; coordinator failure leaves participants stuck |
| Batch checkpointing | Persist last processed ID; resume from checkpoint on restart; UPSERT writes |
| Safe retry | Exponential backoff + fixed idempotency key + circuit breaker |
