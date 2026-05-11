# HLD Interview Q&A — File 16: Consistency and Transactions

> 20 questions across Easy (Q1–7), Medium (Q8–15), and Hard (Q16–20).
> Each answer is 150–300+ words with diagrams, tables, or code where helpful.

---

## EASY (Q1–Q7)

---

### Q1. What are the four main consistency models, and can you give a real-world example of each?

**Answer:**

Consistency models define how and when writes become visible to readers in a distributed system. From strongest to weakest:

| Model | Guarantee | Real-World Example |
|---|---|---|
| Linearizability | Every operation appears instantaneous; reads always return the latest write | Google Spanner — global financial ledger entries |
| Sequential Consistency | All nodes see operations in the same order, but not necessarily in real time | Multi-player game state across nodes |
| Causal Consistency | Operations that are causally related appear in the correct order | Facebook comments — reply always appears after its parent post |
| Eventual Consistency | Given no new writes, all replicas converge eventually | Amazon S3 — object PUT may not be immediately visible globally |

**Linearizability** is the gold standard but is expensive — it requires coordination (Paxos/Raft quorums). It means if write W completes before read R starts, R must see W.

**Sequential consistency** relaxes real-time ordering but maintains a global order. Think of a bulletin board: everyone sees posts in the same order, but the order might not match wall-clock time.

**Causal consistency** tracks dependencies using vector clocks or logical timestamps. Amazon DynamoDB's conditional writes and Cassandra's lightweight transactions give causal semantics within a partition.

**Eventual consistency** is the most scalable. DNS propagation is a classic example: a new record may take minutes to appear worldwide, but eventually all resolvers agree.

Choosing the right model is about balancing latency, availability, and correctness requirements. Most systems use different models for different data types — user sessions might be eventual, while payment balances are linearizable.

---

### Q2. What are the four SQL isolation levels, and which anomaly does each prevent?

**Answer:**

SQL defines four isolation levels (ANSI/ISO SQL standard) that trade consistency for concurrency:

| Isolation Level | Dirty Read | Non-Repeatable Read | Phantom Read | Write Skew |
|---|---|---|---|---|
| READ UNCOMMITTED | Possible | Possible | Possible | Possible |
| READ COMMITTED | Prevented | Possible | Possible | Possible |
| REPEATABLE READ | Prevented | Prevented | Possible | Possible |
| SERIALIZABLE | Prevented | Prevented | Prevented | Prevented |

**Anomaly definitions:**
- **Dirty Read:** Transaction reads data written by a not-yet-committed transaction.
- **Non-Repeatable Read:** Transaction reads the same row twice and gets different values (another committed transaction updated it between reads).
- **Phantom Read:** Transaction re-executes a query and finds new rows inserted by another committed transaction.
- **Write Skew:** Two transactions read overlapping data, make decisions based on it, and write to disjoint sets, violating an invariant.

**Example of write skew:**
```
T1 reads: doctors_on_call = [Alice, Bob]
T2 reads: doctors_on_call = [Alice, Bob]
T1: Alice requests time off → if count > 1, allow → sets Alice = off_call
T2: Bob requests time off → if count > 1, allow → sets Bob = off_call
Result: 0 doctors on call — invariant violated!
```
Only SERIALIZABLE prevents this. Most databases default to READ COMMITTED. PostgreSQL uses REPEATABLE READ by default in explicit transactions but SERIALIZABLE is available.

---

### Q3. What is MVCC, and how does PostgreSQL implement it?

**Answer:**

**Multi-Version Concurrency Control (MVCC)** allows reads and writes to proceed concurrently without blocking each other by keeping multiple versions of each row.

**Core idea:** Instead of locking rows for reads, the database keeps old versions of rows so readers can access a consistent snapshot while writers create new versions.

**PostgreSQL MVCC implementation:**

Each row has two hidden system columns:
- `xmin` — transaction ID that created this row version
- `xmax` — transaction ID that deleted/updated this row version (0 if current)

```sql
-- You can inspect these directly:
SELECT xmin, xmax, id, name FROM users WHERE id = 1;
```

When a transaction starts, PostgreSQL takes a **snapshot** — the set of active transaction IDs at that moment. A row version is visible to the transaction if:
- `xmin` is committed AND `xmin` < snapshot
- `xmax` is zero OR `xmax` is not yet committed OR `xmax` > snapshot

**Update mechanics:**
```
UPDATE users SET name = 'Bob' WHERE id = 1;
```
PostgreSQL does NOT modify the existing row. It:
1. Marks old row with `xmax = current_txn_id`
2. Inserts a new row with `xmin = current_txn_id`, new value

**Dead tuple cleanup:** Old versions accumulate (called "dead tuples"). The `VACUUM` process cleans them up. `AUTOVACUUM` runs automatically to prevent table bloat.

**Benefits:** Readers never block writers, writers never block readers. Each transaction sees a consistent point-in-time snapshot. This is why PostgreSQL can do long-running analytical queries without blocking OLTP writes.

---

### Q4. What is the difference between optimistic and pessimistic locking, and when should you use each?

**Answer:**

**Pessimistic Locking** assumes conflicts are likely — it locks the resource before reading to prevent concurrent modification.

```sql
-- Pessimistic: Lock the row immediately
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;
-- Row is locked; other transactions must wait
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;
```

**Optimistic Locking** assumes conflicts are rare — it reads without locking, then checks at write time whether anyone else modified the data.

```sql
-- Optimistic: Read with version number
SELECT id, balance, version FROM accounts WHERE id = 1;
-- Application processes, then:
UPDATE accounts
SET balance = balance - 100, version = version + 1
WHERE id = 1 AND version = <read_version>;
-- If 0 rows updated → conflict detected, retry
```

**Decision Matrix:**

| Factor | Pessimistic | Optimistic |
|---|---|---|
| Conflict probability | High | Low |
| Transaction duration | Short | Short to medium |
| Read/write ratio | Write-heavy | Read-heavy |
| Risk of deadlock | Higher | None |
| Throughput under contention | Low (waiting) | High (retry vs wait) |

**When to use pessimistic:** Financial transactions (debit/credit), inventory reservation, booking systems — anywhere double-booking or overdraft is unacceptable and contention is high.

**When to use optimistic:** Profile updates, content edits, shopping cart operations — low-contention scenarios where retrying is cheap and holding locks is wasteful.

**Hybrid approach:** Many ORMs like Hibernate and Rails ActiveRecord support optimistic locking via a `lock_version` column automatically.

---

### Q5. How does Two-Phase Commit (2PC) work, and why can it cause blocking?

**Answer:**

**2PC** is a distributed algorithm ensuring all participants in a distributed transaction either all commit or all abort — achieving atomicity across multiple nodes.

**The two phases:**

```
Phase 1: PREPARE (Voting)
Coordinator → Participant A: "Can you commit?"
Coordinator → Participant B: "Can you commit?"
Participant A → Coordinator: "YES (PREPARED)"
Participant B → Coordinator: "YES (PREPARED)"

Phase 2: COMMIT (Decision)
Coordinator → Participant A: "COMMIT"
Coordinator → Participant B: "COMMIT"
Participant A → Coordinator: "ACK"
Participant B → Coordinator: "ACK"
```

If any participant votes NO in Phase 1, the coordinator sends ABORT to all.

**ASCII Diagram:**
```
Coordinator
     |
     |--PREPARE-->  Node A  (locks row X, responds YES)
     |--PREPARE-->  Node B  (locks row Y, responds YES)
     |
     |<--YES------  Node A
     |<--YES------  Node B
     |
     |--COMMIT--->  Node A
     |--COMMIT--->  Node B
```

**Why it blocks:** During Phase 1, each participant has written to its WAL log and is holding locks, waiting for the coordinator's Phase 2 decision. If the **coordinator crashes** after participants have voted YES but before sending COMMIT, participants are stuck in the "prepared" state — they cannot commit (might be wrong) or abort (coordinator might have decided COMMIT) without hearing from the coordinator. This is the **blocking problem** of 2PC.

Recovery requires the coordinator to come back online or a timeout-based protocol. **3PC** (Three-Phase Commit) addresses this but introduces its own complexity. In practice, most systems use Saga pattern or compensating transactions instead of 2PC for long-running operations.

---

### Q6. What is the Outbox Pattern and why is it used for reliable event publishing?

**Answer:**

The **Outbox Pattern** solves the "dual write" problem: how do you atomically update a database AND publish a message to a queue/event bus? If you write to the DB and then publish to Kafka, a crash between those two steps leaves them inconsistent.

**The Problem:**
```
// WRONG: Two separate writes — not atomic
db.save(order)          // succeeds
kafka.publish(order)    // crash here → event never sent!
```

**The Solution — Outbox Table:**
```sql
-- Outbox table in same database
CREATE TABLE outbox (
    id UUID PRIMARY KEY,
    event_type VARCHAR(100),
    aggregate_id VARCHAR(100),
    payload JSONB,
    created_at TIMESTAMP,
    published_at TIMESTAMP  -- NULL until sent
);
```

```
Application Transaction:
BEGIN;
  INSERT INTO orders (id, ...) VALUES (...);
  INSERT INTO outbox (id, event_type, payload) VALUES (..., 'ORDER_CREATED', ...);
COMMIT;
-- Atomically: either both succeed or both fail
```

A separate **Relay Process** (or Change Data Capture via Debezium) polls the outbox table and publishes unpublished events to Kafka/RabbitMQ, then marks them as published.

```
Outbox Relay:
  SELECT * FROM outbox WHERE published_at IS NULL ORDER BY created_at LIMIT 100;
  → publish each to Kafka
  → UPDATE outbox SET published_at = NOW() WHERE id = ?
```

**Benefits:**
- Exactly-once semantics for database writes
- At-least-once delivery for events (relay retries on failure)
- No distributed transaction needed
- Works with any message broker

**Drawback:** Slight latency between DB commit and event publication. CDC-based approaches (Debezium reading WAL) minimize this to milliseconds.

---

### Q7. What is an idempotency key and how does it prevent double-charging in payment systems?

**Answer:**

An **idempotency key** is a unique client-generated identifier sent with a request that allows the server to recognize and safely replay duplicate requests without executing the operation twice.

**The Problem:**
```
Client → Server: "Charge $100 to card XXXX"
Server charges card → Server crashes before responding
Client (timeout) → Server: "Charge $100 to card XXXX" (retry)
Result: Customer charged $200!
```

**The Solution:**
```http
POST /v1/charges
Idempotency-Key: a8098c1a-f86e-11da-bd1a-00112444be1e
Content-Type: application/json

{ "amount": 100, "currency": "USD", "card": "tok_xxxx" }
```

**Server-side implementation:**
```sql
CREATE TABLE idempotency_keys (
    key UUID PRIMARY KEY,
    request_hash VARCHAR(64),
    response_body JSONB,
    status_code INT,
    created_at TIMESTAMP,
    expires_at TIMESTAMP
);
```

```python
def process_payment(idempotency_key, amount, card):
    existing = db.get(f"idempotency:{idempotency_key}")
    if existing:
        return existing.response  # Return cached response
    
    # Lock to prevent concurrent duplicate
    with redis.lock(f"lock:idempotency:{idempotency_key}"):
        result = charge_card(amount, card)
        db.save(idempotency_key, result, ttl=24*60*60)
        return result
```

**Key properties:**
- Keys should expire after a reasonable window (24 hours for Stripe)
- The key must be tied to the specific request payload (detect payload mismatch)
- Lock during processing to prevent concurrent duplicates
- Must be stored durably — not just in memory

**Stripe's approach:** Any retried request with the same idempotency key within 24 hours returns the original response verbatim, regardless of whether the operation succeeded or failed.

---

## MEDIUM (Q8–Q15)

---

### Q8. What is write skew, and how does SERIALIZABLE isolation prevent it?

**Answer:**

**Write skew** is a subtle anomaly where two concurrent transactions each read a set of rows, make a decision based on what they read, and then write to different rows — individually each transaction is correct, but together they violate an invariant.

**Classic example — Doctor On-Call:**
```
Invariant: At least 1 doctor must be on call at all times.
Initial state: Alice = on_call, Bob = on_call

T1 (Alice requests time off):
  SELECT COUNT(*) FROM doctors WHERE on_call = TRUE  → 2
  Since count > 1, it's safe to take time off
  UPDATE doctors SET on_call = FALSE WHERE name = 'Alice'

T2 (Bob requests time off):
  SELECT COUNT(*) FROM doctors WHERE on_call = TRUE  → 2
  Since count > 1, it's safe to take time off
  UPDATE doctors SET on_call = FALSE WHERE name = 'Bob'

Result: Both execute successfully. 0 doctors on call!
```

Neither T1 nor T2 modified the same row, so row-level locks don't help. **REPEATABLE READ** doesn't prevent this because each transaction only re-reads its own rows.

**Why SERIALIZABLE works:**

PostgreSQL's SERIALIZABLE uses **Serializable Snapshot Isolation (SSI)**, which tracks **read-write dependencies** between transactions:
- T1 reads the doctors count (a predicate read)
- T2 writes to a doctor row that T1 would have seen in its snapshot
- This creates a cycle: T1 → T2 → T1
- SSI detects this cycle and aborts one transaction

```sql
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
BEGIN;
SELECT COUNT(*) FROM doctors WHERE on_call = TRUE;
UPDATE doctors SET on_call = FALSE WHERE name = 'Alice';
COMMIT;  -- May receive: ERROR: could not serialize access due to read/write dependencies
```

**Application must retry** on serialization failures. The throughput cost is typically 10–30% compared to REPEATABLE READ in PostgreSQL's SSI implementation.

**Other write skew examples:**
- Meeting room double booking
- Username uniqueness check-then-insert (without UNIQUE constraint)
- Wallet balance going negative when two concurrent withdrawals read the same balance

---

### Q9. How do CRDTs work, and can you explain a G-Counter (Grow-only Counter)?

**Answer:**

**CRDTs (Conflict-free Replicated Data Types)** are data structures designed so that concurrent updates can be merged automatically without conflicts, always converging to the same result regardless of the order operations are applied.

**Two types:**
- **CvRDT (State-based):** Replicas periodically merge their full state using a merge function
- **CmRDT (Operation-based):** Replicas exchange and apply operations

**G-Counter (Grow-only Counter):**

Each node maintains a vector of counters — one slot per node in the cluster. A node can only increment its own slot.

```python
class GCounter:
    def __init__(self, node_id, num_nodes):
        self.node_id = node_id
        self.vector = [0] * num_nodes  # One counter per node
    
    def increment(self):
        self.vector[self.node_id] += 1
    
    def value(self):
        return sum(self.vector)
    
    def merge(self, other):
        # Take element-wise max — idempotent and commutative
        merged = GCounter(self.node_id, len(self.vector))
        merged.vector = [max(a, b) for a, b in zip(self.vector, other.vector)]
        return merged
```

**Example with 3 nodes:**
```
Node 0: [5, 0, 0]  ← incremented 5 times
Node 1: [0, 3, 0]  ← incremented 3 times
Node 2: [0, 0, 2]  ← incremented 2 times

After merge: [5, 3, 2] → total value = 10
```

**Why it's conflict-free:** The merge function (`max` per element) is:
- **Commutative:** merge(A, B) = merge(B, A)
- **Associative:** merge(merge(A,B), C) = merge(A, merge(B,C))
- **Idempotent:** merge(A, A) = A

**PN-Counter** extends this for increment and decrement by maintaining two G-Counters: one for increments (P) and one for decrements (N). Value = P.value() - N.value().

**Real-world use:** Redis cluster uses CRDTs for its replicated data types in Redis Enterprise. Riak uses CRDTs for counters, sets, and maps. Collaborative text editors use CRDTs (e.g., Automerge, Yjs) for operational data.

---

### Q10. How does distributed locking with Redis work, and what is the Redlock algorithm?

**Answer:**

**Single-node Redis lock (basic):**
```bash
SET lock:resource_name unique_token NX PX 30000
# NX = only set if Not eXists
# PX 30000 = expire after 30 seconds (auto-release if client crashes)
```

Release (must use Lua for atomicity):
```lua
-- Only delete if we own the lock
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
```

**Problem:** Single Redis node is a single point of failure.

**Redlock Algorithm (Martin Kleppmann / Redis Labs):**

Designed for N independent Redis masters (typically 5):

```
1. Get current timestamp T1
2. For each of N Redis nodes:
   - Try to SET lock:X <token> NX PX <lock_timeout>
   - If acquired within a small timeout, count as success
3. Lock acquired if: acquired on majority (N/2 + 1) nodes
   AND total elapsed time < lock_timeout
4. Effective lock time = lock_timeout - elapsed_time
5. If not acquired: send DEL to all N nodes (cleanup)
```

**ASCII Diagram (N=5):**
```
Client → Redis1 (acquired) ✓
Client → Redis2 (acquired) ✓
Client → Redis3 (acquired) ✓   ← 3/5 = majority
Client → Redis4 (timeout)  ✗
Client → Redis5 (acquired) ✓
```

**Controversy:** Martin Kleppmann argued that Redlock has subtle safety issues under certain failure modes (clock jumps, GC pauses). The main issue is that a lock can expire while the holder is paused (GC pause), and another process acquires it — two processes now believe they hold the lock.

**Fencing tokens** (covered in Q19) address this by using a monotonically increasing token from the lock service, verified by the resource being locked.

**When to use Redis distributed locks:**
- Rate limiting (not strictly requiring mutual exclusion)
- Distributed cron — prevent duplicate job execution
- Leader election in low-stakes scenarios

For strict mutual exclusion with correctness guarantees, prefer **ZooKeeper** (ephemeral nodes + sequential nodes) or etcd (which uses Raft for consensus).

---

### Q11. What is the Saga pattern, and how does choreography differ from orchestration?

**Answer:**

**Saga** is a pattern for managing long-running distributed transactions without 2PC. A saga is a sequence of local transactions, each publishing events or messages. If a step fails, compensating transactions undo the previous steps.

**Example: E-commerce Order Saga**
```
Steps:
1. Create Order
2. Reserve Inventory
3. Process Payment
4. Arrange Shipping

Compensating transactions:
1. Cancel Order
2. Release Inventory
3. Refund Payment
4. Cancel Shipping
```

**Choreography (Event-Driven):**

Each service listens for events and decides what to do next. No central coordinator.

```
OrderService →[OrderCreated]→ InventoryService
InventoryService →[InventoryReserved]→ PaymentService
PaymentService →[PaymentFailed]→ InventoryService (releases stock)
InventoryService →[InventoryReleased]→ OrderService (cancels order)
```

**Orchestration (Command-Driven):**

A central Saga Orchestrator sends commands and listens for responses.

```
SagaOrchestrator:
  → COMMAND: ReserveInventory → InventoryService
  ← REPLY: InventoryReserved
  → COMMAND: ProcessPayment → PaymentService
  ← REPLY: PaymentFailed
  → COMMAND: ReleaseInventory → InventoryService (compensate)
  → COMMAND: CancelOrder → OrderService (compensate)
```

**Comparison:**

| Aspect | Choreography | Orchestration |
|---|---|---|
| Coupling | Loose (event-driven) | Tighter (central coordinator) |
| Observability | Hard (flow is implicit) | Easy (state in orchestrator) |
| Complexity | Grows with services | Centralized complexity |
| Single point of failure | No | Orchestrator is SPOF |
| Best for | Simple, linear flows | Complex, branching workflows |
| Testing | Hard to trace | Easier to test orchestrator |

**When choreography breaks down:** With many services, tracing why a saga failed becomes a debugging nightmare — you need distributed tracing (Jaeger/Zipkin) to reconstruct the flow. Netflix uses orchestration with their Conductor framework; AWS Step Functions implements orchestrated sagas.

---

### Q12. How does causal consistency work, and give a real-world use case?

**Answer:**

**Causal consistency** guarantees that operations that are causally related are seen by all nodes in the same causal order. Operations that are not causally related (concurrent) may be seen in different orders by different nodes.

**Causal relationships:**
- If A happened before B and B happened before C, then all nodes see A before B before C
- If two operations are concurrent (neither happened before the other), nodes may see them in any order

**Tracking causality — Vector Clocks:**
```
Node 1: {N1: 1, N2: 0, N3: 0}  [Event A]
Node 2: {N1: 0, N2: 1, N3: 0}  [Event B, concurrent with A]
Node 2 receives A: {N1: 1, N2: 2, N3: 0}  [Event C, causally after A]
```

**Real-world use case — Social Media Comments:**

```
Timeline:
Alice posts: "I got the job!" → Post P1

Bob sees P1, replies: "Congratulations!" → Comment C1
Carol sees P1 and C1, replies: "Amazing news!" → Comment C2
```

Without causal consistency, Carol's reply could appear before Bob's — confusing to readers. Causal consistency ensures:
- C1 appears after P1 (Bob saw the post before commenting)
- C2 appears after P1 and C1 (Carol saw both)

**Implementation — causal tokens:**
```python
# Server returns causal token with each read
response = db.read(post_id)
causal_token = response.version_vector  # e.g., {server1: 5, server2: 3}

# Client includes token with next write
db.write(comment, after=causal_token)
# Server waits until it has caught up to causal_token before serving
```

**Real systems:**
- **Amazon DynamoDB** offers "causal consistency" within a session using consistent reads
- **MongoDB** provides causal consistency via session-level causally consistent reads
- **Cassandra** uses lightweight transactions for causal ordering within a partition

**Trade-off vs linearizability:** Causal consistency is achievable with high availability (unlike linearizability which requires coordination) while still providing meaningful ordering guarantees — it sits between sequential consistency and eventual consistency in the hierarchy.

---

### Q13. How does DynamoDB achieve eventually consistent reads, and when should you use strongly consistent reads?

**Answer:**

**DynamoDB replication model:**

DynamoDB uses a **multi-AZ replication** model. Each item is stored across 3 AZs (Availability Zones). Writes go to a leader node which replicates to 2 replicas asynchronously.

```
Write flow:
Client → DynamoDB Leader Node (AZ-1)
         ↓ Sync
         Replica Node (AZ-2)
         ↓ Async
         Replica Node (AZ-3)
         ↓
Client receives ACK (after 2/3 nodes acknowledge = quorum write)
```

**Eventually Consistent Read (default):**
```python
response = dynamodb.get_item(
    TableName='Orders',
    Key={'order_id': {'S': 'ORD-123'}},
    ConsistentRead=False  # default — routes to any replica
)
```
- Routes to any of the 3 replicas (whichever is closest/fastest)
- The replica may be slightly behind — you might read stale data for a few hundred milliseconds
- Costs 0.5 RCU (Read Capacity Units) per 4KB
- Highest throughput, lowest latency

**Strongly Consistent Read:**
```python
response = dynamodb.get_item(
    TableName='Orders',
    Key={'order_id': {'S': 'ORD-123'}},
    ConsistentRead=True  # routes to leader node
)
```
- Routes only to the leader node
- Always returns the most recent committed write
- Costs 1 RCU per 4KB (2x the cost)
- Higher latency (must hit leader, which is in a specific AZ)

**Decision guide:**

| Use Case | Recommendation |
|---|---|
| Shopping cart add-to-cart confirmation | Strong consistency |
| Reading product catalog | Eventually consistent |
| User session check | Depends on session token freshness requirements |
| Analytics/reporting | Eventually consistent |
| Inventory reservation | Strong consistency |
| Showing post like count | Eventually consistent |

**DAX (DynamoDB Accelerator):** DynamoDB's in-memory cache. DAX only supports eventually consistent reads — strongly consistent reads bypass DAX and go directly to DynamoDB.

---

### Q14. What is the difference between Paxos and Raft?

**Answer:**

Both Paxos and Raft are **consensus algorithms** — protocols allowing a cluster of nodes to agree on a single value even in the presence of failures. Both require a majority (quorum) of nodes to be available.

**Paxos:**

Proposed by Leslie Lamport in 1989. Classic Paxos achieves consensus for a single value. Multi-Paxos extends it for a replicated log (sequence of values).

**Phases:**
```
Phase 1a (Prepare): Proposer → Acceptors: "Prepare(n)" — ballot number n
Phase 1b (Promise): Acceptors → Proposer: "Promise(n, accepted_value)"
Phase 2a (Accept):  Proposer → Acceptors: "Accept(n, value)"
Phase 2b (Accepted): Acceptors → Learner: "Accepted(n, value)"
```

**Raft:**

Designed by Ongaro and Ousterhout in 2014 with the explicit goal of being **understandable**. Raft decomposes consensus into three sub-problems:

1. **Leader election:** One leader per term; candidates request votes
2. **Log replication:** Leader receives client requests, appends to log, replicates to followers
3. **Safety:** Never commit entries that could be lost

```
Term 1: [Leader: Node A] → Receives entries, replicates
Term 2: A fails → [Election] → Node B wins → New leader
Term 3: [Leader: Node B] → Continues
```

**Key differences:**

| Aspect | Paxos | Raft |
|---|---|---|
| Design goal | Correctness proof | Understandability |
| Leader | Optional (leaderless variants exist) | Required (strong leader) |
| Log gaps | Allowed (holes in log) | Not allowed (sequential) |
| Complexity | Very high | Moderate |
| Implementations | Zab (ZooKeeper), Google Chubby | etcd, CockroachDB, TiKV |
| Membership changes | Ad-hoc | Built-in joint consensus |

**Practical outcome:** Raft is the default choice for new systems. etcd (Kubernetes backend), CockroachDB, and Consul all use Raft. The Paxos family (Zab, Multi-Paxos) powers ZooKeeper and Google Spanner.

---

### Q15. What is a compensating transaction, and how does it differ from a rollback?

**Answer:**

A **compensating transaction** is a business-level operation that logically reverses the effects of a previously committed transaction. It is fundamentally different from a database rollback.

**Database Rollback:**
- Undoes changes atomically at the database level
- Operates below the application layer (database handles it)
- Instant — reverts to pre-transaction state
- Only possible for uncommitted transactions (before COMMIT)

**Compensating Transaction:**
- A new forward transaction that negates the business effect
- Applied after the original transaction has already committed
- Creates a new audit trail entry
- Must be explicitly designed and coded
- May have side effects that cannot be truly reversed

**Example — Travel Booking:**
```
Original Saga:
  T1: Book flight (COMMITTED) → sends confirmation email
  T2: Book hotel (COMMITTED) → reserves room
  T3: Charge credit card (FAILED)

Compensating transactions:
  C2: Cancel hotel reservation (new transaction)
  C1: Cancel flight booking (new transaction)
  -- C3: Not needed (payment never succeeded)
```

**Design considerations for compensating transactions:**
```python
class BookFlightCompensation:
    def execute(self, booking_id):
        booking = db.get_booking(booking_id)
        
        if booking.status == 'BOARDED':
            raise ImpossibleCompensation("Cannot unboard passenger")
        
        if booking.status in ['CONFIRMED', 'CHECKED_IN']:
            airline_api.cancel(booking.confirmation_code)
            db.update_booking(booking_id, status='CANCELLED')
            audit_log.record('FLIGHT_CANCELLED', booking_id, reason='saga_compensation')
            
            if within_free_cancellation_window(booking):
                refund_service.issue_refund(booking.payment_id)
```

**Key insight:** Not all operations can be compensated. You cannot un-send an email, un-deliver a package, or un-launch a missile. For such cases, design for **countermeasures** (send "we're sorry" email, issue a return label) rather than true reversal.

---

## HARD (Q16–Q20)

---

### Q16. Is it possible to design a system that needs both strong consistency and high availability? How do you reconcile the CAP theorem?

**Answer:**

The **CAP theorem** states that during a network partition, a distributed system must choose between Consistency (C) and Availability (A). This seems to say C and A are mutually exclusive. But the reality is more nuanced.

**What CAP actually says:**

CAP applies only during a **network partition** (P). When there is no partition (the normal case), you can have both C and A. The choice is: when a partition occurs, do you:
- **CP:** Refuse to serve requests that can't be answered consistently (sacrifice availability)
- **AP:** Serve requests with potentially stale data (sacrifice consistency)

**The PACELC model is more complete:**

```
PACELC:
  If Partition: choose between A and C
  Else (no partition): choose between L(atency) and C(onsistency)
```

**Practical strategies to achieve "good enough" C + A:**

**1. Tunable consistency (Cassandra/DynamoDB):**
```
Write quorum: W nodes must acknowledge
Read quorum: R nodes must respond
For strong consistency: W + R > N (total replicas)
  e.g., N=3, W=2, R=2 → W+R=4 > 3 ✓
For availability: W=1, R=1 (fastest but stale reads possible)
```

**2. Bounded staleness:**
Offer consistency with a defined lag (e.g., "reads are at most 5 seconds stale"). Azure Cosmos DB offers this as an explicit consistency level.

**3. Read-your-writes consistency:**
A weaker guarantee: "you always see your own writes." Implemented by routing reads to the leader for a session, then allowing follower reads after a short time.

**4. Google Spanner's approach:**
Spanner achieves linearizability across globally distributed nodes using **TrueTime** — GPS and atomic clocks give a globally synchronized clock with bounded uncertainty. Spanner commits with a timestamp, then waits out the clock uncertainty window before returning to the client — turning latency into a consistency guarantee.

```
Spanner commit:
  Prepare transaction → get commit timestamp T
  Wait for TrueTime.after(T) == true (typically 7-14ms)
  Return to client
```

**Practical answer:** For most systems, design with:
- Strong consistency for **critical paths** (payments, inventory)
- Eventual consistency for **non-critical reads** (feed, recommendations)
- Explicit consistency SLAs documented per endpoint

True strong consistency + true high availability is impossible during partitions (CAP), but partitions are rare and brief — design for the 99.99% case and handle partition scenarios with graceful degradation.

---

### Q17. What is a fencing token and why does it matter for distributed locks?

**Answer:**

A **fencing token** is a monotonically increasing number issued by the lock service when a lock is granted. It is passed to the resource being locked, which rejects requests with a token lower than the highest it has already seen.

**The Problem Without Fencing Tokens:**

```
Timeline:
t1: Client A acquires Redis lock, receives token (not used)
t2: Client A starts processing, gets a GC pause for 90 seconds
t3: Lock expires (TTL = 30s)
t4: Client B acquires the lock
t5: Client B writes to database
t6: Client A resumes from GC pause
t7: Client A writes to database — CORRUPTS Client B's data!
```

Both A and B believe they hold the lock at t6, because A doesn't know its lock expired.

**The Solution — Fencing Token:**

```
Lock Service issues incrementing tokens:
Client A acquires lock → receives token 33
Client B acquires lock → receives token 34

Client A's request to storage: write(data, fence=33)
Storage: "highest seen token = 34, rejecting token 33!"
Client A's write is rejected.

Client B's request to storage: write(data, fence=34)
Storage: "token 34 >= 34, accepting!"
```

**Implementation:**
```python
# Lock service (ZooKeeper sequential node example)
def acquire_lock(resource):
    zk_path = zk.create(f'/locks/{resource}/lock-', 
                        ephemeral=True, 
                        sequence=True)
    sequence_number = int(zk_path.split('-')[-1])
    return Lock(path=zk_path, token=sequence_number)

# Storage layer
class StorageLayer:
    def __init__(self):
        self.fence_tokens = {}  # resource → max_seen_token
    
    def write(self, resource, data, fence_token):
        max_seen = self.fence_tokens.get(resource, 0)
        if fence_token < max_seen:
            raise StaleWriteError(f"Token {fence_token} < {max_seen}")
        self.fence_tokens[resource] = fence_token
        self._do_write(resource, data)
```

**Why most distributed lock implementations are insufficient without fencing:**
- Redis SETNX doesn't issue monotonic tokens
- Redlock has no fencing mechanism
- Only **ZooKeeper sequential ephemeral nodes** and **etcd** provide the monotonic token property natively

**Key insight from Martin Kleppmann:** "A distributed lock without a fencing mechanism is like a car alarm without a kill switch — it doesn't actually protect you."

---

### Q18. How does the Jepsen framework test distributed system correctness?

**Answer:**

**Jepsen** (by Kyle Kingsbury / "aphyr") is an open-source framework for testing distributed systems for safety violations. It has exposed critical consistency bugs in MongoDB, Redis, Cassandra, Kafka, etcd, and dozens of other systems.

**How Jepsen works:**

```
Architecture:
  Control Node (Clojure client)
       ↓
  5 DB Nodes (typical cluster)
       ↓
  Nemesis (fault injector)
```

**Test lifecycle:**
```
1. Setup: Deploy fresh cluster, install target DB
2. Run:
   - n concurrent clients issue operations (reads/writes)
   - Nemesis concurrently injects faults:
     - Network partitions (iptables)
     - Kill processes (SIGKILL)
     - Clock skew (ntpdate manipulation)
     - Disk full scenarios
3. Collect: Record every operation with [invocation_time, completion_time, value]
4. Analyze: Run a checker against the operation history
```

**The checker — Elle and Knossos:**

```
Operation history example:
  [t=100ms] Client1: write(x, 5) → ok
  [t=110ms] Client2: read(x) → 3   ← should be 5!
  [t=115ms] Client3: read(x) → 5

Knossos linearizability checker:
  Attempts to find a valid sequential ordering of ops
  that is consistent with the concurrent history
  If no valid ordering exists → LINEARIZABILITY VIOLATION
```

**Elle** (newer checker) specifically finds anomalies per isolation level:
- G0: Dirty writes
- G1a: Aborted reads
- G1c: Circular information flow
- G2: Anti-dependency cycles (detecting serializability violations)

**Famous Jepsen findings:**

| System | Finding | Year |
|---|---|---|
| MongoDB 2.4 | Data loss during network partition | 2013 |
| Redis (Sentinel) | Split-brain allows loss of committed writes | 2013 |
| Cassandra | Regularly violated its own stated consistency guarantees | 2013 |
| Kafka 0.9 | Message loss and reordering under partition | 2016 |
| etcd 3.4.3 | Stale reads under specific conditions | 2020 |

**How to apply Jepsen thinking to your design:**
1. Document what consistency guarantees your system claims
2. Identify what happens when each component fails
3. Test with real fault injection — not just happy-path unit tests
4. Chaos engineering (Netflix's Chaos Monkey) is the production-safe cousin of Jepsen

---

### Q19. What is the key difference between linearizability and serializability?

**Answer:**

This is one of the most commonly confused concepts in distributed systems. They sound similar but operate at different levels.

**Serializability** (database concept):
- Applies to **transactions** (groups of operations)
- Guarantees that the result of executing transactions concurrently is equivalent to some **serial** (one-at-a-time) execution
- Does NOT require the serial order to match real-time (wall clock) ordering
- An isolation property of ACID transactions

```
Example — Serializable but not Linearizable:
T1: write(x=1), read(y)   starts and finishes BEFORE T2
T2: write(y=1), read(x)

Valid serializable order: T2 then T1
T2 runs "first" in the serial order, even though T1 started first in real time
```

**Linearizability** (distributed systems concept):
- Applies to **individual operations** (typically single-object, single-operation)
- Requires that operations appear instantaneous at some point between their invocation and completion
- The serial order MUST respect real-time ordering
- A recency/consistency property of replicated objects

```
Example — Linearizable:
t=100ms: Client1: write(x=1) completes
t=200ms: Client2: read(x) starts
Linearizability REQUIRES Client2 to see x=1
(The write completed before the read started)
```

**Combining them — Strict Serializability:**

```
Strict Serializability = Serializability + Linearizability
= Transactions + Real-time ordering
= Strongest possible guarantee
```

Google Spanner provides strict serializability. This is also called "external consistency."

**Comparison Table:**

| Property | Serializability | Linearizability |
|---|---|---|
| Applies to | Transactions (multi-op) | Single operations |
| Ordering | Equivalent serial order (not real-time) | Respects real-time wall clock |
| Level | Database isolation | Distributed systems |
| Cost | High (locking/SSI) | Very high (coordination) |
| Standard | SQL ACID | Distributed systems theory |

**Practical implications:**
- PostgreSQL SERIALIZABLE isolation gives serializability within a single node, not linearizability across replicas
- For linearizability across replicas, you need consensus (Raft/Paxos) — e.g., etcd, ZooKeeper
- Most distributed systems that claim "strong consistency" mean linearizability of single operations, not serializability of transactions

---

### Q20. How do you design a system where DynamoDB transactions and conditional writes work together to prevent race conditions?

**Answer:**

DynamoDB offers several mechanisms to handle concurrent writes safely, from single-item conditions to multi-item ACID transactions.

**Level 1: Conditional Writes (single item)**

```python
# Optimistic locking with version number
try:
    dynamodb.update_item(
        TableName='Accounts',
        Key={'user_id': {'S': 'user123'}},
        UpdateExpression='SET balance = :new_bal, version = :new_ver',
        ConditionExpression='version = :expected_ver AND balance >= :amount',
        ExpressionAttributeValues={
            ':new_bal': {'N': str(current_balance - amount)},
            ':new_ver': {'N': str(current_version + 1)},
            ':expected_ver': {'N': str(current_version)},
            ':amount': {'N': str(amount)}
        }
    )
except dynamodb.exceptions.ConditionalCheckFailedException:
    # Retry with fresh read
    retry_withdrawal(user_id, amount)
```

**Level 2: DynamoDB Transactions (multi-item ACID)**

```python
# Transfer money between two accounts atomically
dynamodb.transact_write(
    TransactItems=[
        {
            'Update': {
                'TableName': 'Accounts',
                'Key': {'user_id': {'S': 'sender'}},
                'UpdateExpression': 'SET balance = balance - :amount',
                'ConditionExpression': 'balance >= :amount',
                'ExpressionAttributeValues': {':amount': {'N': '100'}}
            }
        },
        {
            'Update': {
                'TableName': 'Accounts',
                'Key': {'user_id': {'S': 'receiver'}},
                'UpdateExpression': 'SET balance = balance + :amount',
                'ExpressionAttributeValues': {':amount': {'N': '100'}}
            }
        },
        {
            'Put': {
                'TableName': 'Transactions',
                'Item': {
                    'txn_id': {'S': 'TXN-001'},
                    'amount': {'N': '100'},
                    'status': {'S': 'COMPLETED'}
                },
                'ConditionExpression': 'attribute_not_exists(txn_id)'  # Idempotency
            }
        }
    ]
)
```

**Limitations of DynamoDB Transactions:**
- Up to 25 items per transaction
- 4MB total transaction size
- 2x the cost of individual operations
- Not suitable for cross-table aggregations

**Pattern: Idempotent transactions in DynamoDB:**
```python
def idempotent_transfer(txn_id, sender, receiver, amount):
    # The ConditionExpression on txn_id prevents replay
    # If txn_id already exists → ConditionalCheckFailedException
    # Application can safely retry by catching this specific exception
    try:
        dynamodb.transact_write(...)
        return "SUCCESS"
    except ConditionalCheckFailedException as e:
        existing = dynamodb.get_item('Transactions', txn_id)
        return existing['status']  # Already processed — return original result
```

**Design decision matrix for DynamoDB consistency:**

| Scenario | Mechanism |
|---|---|
| Prevent duplicate order | Conditional write + idempotency key |
| Atomic multi-item update | TransactWrite |
| Inventory reservation | Conditional update with balance check |
| Read-your-writes | ConsistentRead=True |
| High-throughput counter | DAX + eventually consistent |

---

## Quick Reference

### Consistency Model Hierarchy
```
Strongest → Weakest:
Linearizability → Sequential → Causal → Eventual
```

### SQL Isolation Levels vs Anomalies
| Level | Dirty Read | Non-Rep Read | Phantom | Write Skew |
|---|---|---|---|---|
| READ UNCOMMITTED | Y | Y | Y | Y |
| READ COMMITTED | N | Y | Y | Y |
| REPEATABLE READ | N | N | Y | Y |
| SERIALIZABLE | N | N | N | N |

### 2PC States
```
INIT → PREPARED → COMMITTED/ABORTED
         ↑
    (blocking zone: coordinator crash here = stuck)
```

### Locking Decision
- **High contention + short txn** → Pessimistic
- **Low contention + read-heavy** → Optimistic
- **Distributed** → Redlock (low stakes) / ZooKeeper (strict)

### Saga Pattern
- **Choreography** = loose coupling, hard to trace
- **Orchestration** = central control, easier to debug

### CAP in Practice
- CA (no partition tolerance) = single-node systems
- CP = ZooKeeper, etcd, HBase
- AP = Cassandra, DynamoDB (default), CouchDB

### Key Algorithms
| Algorithm | Used In | Key Property |
|---|---|---|
| Paxos | ZooKeeper (Zab), Spanner | Proven correct, complex |
| Raft | etcd, CockroachDB, Consul | Understandable, strong leader |
| MVCC | PostgreSQL, MySQL InnoDB | Readers don't block writers |
| SSI | PostgreSQL SERIALIZABLE | Detects read-write cycles |

### Idempotency Key Flow
```
Client generates UUID → sends with request
Server: check cache → if exists, return cached → if not, process + cache
```

### Outbox Pattern
```
[App DB Transaction]
  ├── Write business entity
  └── Write to outbox table
[Relay Process]
  ├── Poll outbox
  ├── Publish to Kafka
  └── Mark as published
```
