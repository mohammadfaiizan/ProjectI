# 18. Consistency and Distributed Transactions

## Table of Contents
1. [Consistency Spectrum](#1-consistency-spectrum)
2. [ACID vs BASE](#2-acid-vs-base)
3. [Isolation Levels](#3-isolation-levels)
4. [MVCC](#4-mvcc-multi-version-concurrency-control)
5. [Pessimistic vs Optimistic Locking](#5-pessimistic-vs-optimistic-locking)
6. [Two-Phase Commit (2PC)](#6-two-phase-commit-2pc)
7. [Three-Phase Commit (3PC)](#7-three-phase-commit-3pc)
8. [Saga Pattern](#8-saga-pattern)
9. [Outbox Pattern](#9-outbox-pattern)
10. [Idempotency](#10-idempotency)
11. [Conflict Resolution Strategies](#11-conflict-resolution-strategies)
12. [CRDTs](#12-crdts)
13. [Read-Your-Writes and Monotonic Reads](#13-read-your-writes-and-monotonic-reads)
14. [Causal Consistency](#14-causal-consistency)
15. [Write Skew Anomaly](#15-write-skew-anomaly)
16. [Distributed Locking](#16-distributed-locking)
17. [Exactly-Once Delivery](#17-exactly-once-delivery)
18. [Linearizability Testing](#18-linearizability-testing)
19. [Database Choice by Consistency Requirement](#19-database-choice-by-consistency-requirement)
20. [Quick Reference](#20-quick-reference)

---

## 1. Consistency Spectrum

Consistency models form a spectrum from strongest to weakest, with a trade-off against availability and performance:

```
Strongest                                                    Weakest
    |                                                           |
Linearizability → Sequential → Causal → Read-Your-Writes → Eventual
    |                                                           |
  Slowest                                                   Fastest
```

### Linearizability (Strong Consistency)

**Definition:** Every operation appears to take effect instantaneously at some point between its invocation and completion. All clients observe a single, consistent global order of operations.

```
Timeline:
  Client A writes: x = 1    [---W(x=1)---]
  Client B reads:                          [---R(x=?)--->  must return 1

Any read after a write completes MUST return the written value.
Operations can be placed on a single global timeline.
```

**Implementations:**
- Single-leader replication with synchronous writes
- Raft consensus (etcd, ZooKeeper, CockroachDB)
- Google Spanner (TrueTime + two-phase commit across datacenters)

**Cost:** Requires coordination; high latency under network partitions (violates availability per CAP theorem).

### Sequential Consistency

**Definition:** All operations appear to execute in some sequential order, and that order is consistent with the order of operations on each individual process.

```
Timeline (sequential but NOT linearizable):
  Client A: W(x=1)                     W(x=2)
  Client B:          R(x=1)  R(x=2)
  Client C:          R(x=1)  R(x=2)

  All clients see A's writes in order (1 before 2) ← sequential
  But C might see x=2 before A's write visually completes ← not linearizable
```

### Causal Consistency

**Definition:** Operations that are causally related must be seen in the same order by all processes. Concurrent (causally unrelated) operations may be seen in different orders.

```
Event A: User posts "Hello"        → docID=1
Event B: User replies to docID=1   → reply to "Hello"

Causal dependency: B depends on A
Causal consistency guarantees: B is never visible before A

Two users writing concurrently with no dependency:
  User 1 writes: product price = $100
  User 2 writes: product name = "Widget"
  No causal dependency → can be seen in any order
```

### Eventual Consistency

**Definition:** If no new updates are made to an object, eventually all reads will return the last written value. No guarantees about when or intermediate states.

```
DynamoDB default, Cassandra with consistency=ONE:
  Write goes to one node → other nodes updated asynchronously
  
  Client A writes x=5 to replica-1
  Client B reads x from replica-2 → may get x=3 (stale)
  
  After propagation delay (ms to seconds): all replicas converge to x=5
```

---

## 2. ACID vs BASE

### ACID Properties

```
A - Atomicity:   Transaction succeeds completely or fails completely; no partial state
C - Consistency: Database moves from one valid state to another; constraints satisfied
I - Isolation:   Concurrent transactions don't interfere; as if run serially
D - Durability:  Committed data survives crashes; written to durable storage (fsync)
```

**ACID Implementation:**
```sql
BEGIN TRANSACTION;
  UPDATE accounts SET balance = balance - 100 WHERE id = 1;  -- debit
  UPDATE accounts SET balance = balance + 100 WHERE id = 2;  -- credit
COMMIT;  -- Both succeed together
-- OR --
ROLLBACK; -- Neither change persists on error
```

### BASE Properties

```
BA - Basically Available:  System remains available during failures (partial responses OK)
S  - Soft state:           State may change over time without input (convergence happening)
E  - Eventually consistent: System will become consistent given enough time
```

### ACID vs BASE Comparison

| Dimension | ACID | BASE |
|-----------|------|------|
| Consistency | Strong, immediate | Eventual |
| Availability | Lower (coordinator needed) | Higher |
| Latency | Higher (synchronous coordination) | Lower (async replication) |
| Throughput | Lower | Higher |
| Failure model | Fail-stop preferred | Tolerates partial failures |
| Use cases | Banking, inventory, order management | Social feeds, shopping carts, analytics |
| Databases | PostgreSQL, MySQL, Oracle, CockroachDB | DynamoDB, Cassandra, MongoDB (default) |

### When to Choose Each

**ACID (use when):**
- Money transfers, debit/credit operations
- Inventory deductions (avoid overselling)
- User authentication state (session creation)
- Order state machine transitions

**BASE (use when):**
- View/like counts (approximate is fine)
- User activity feeds
- Product catalog reads
- Session caching
- Analytics data collection

---

## 3. Isolation Levels

SQL standard defines four isolation levels, each preventing a different class of anomaly:

### Anomalies Defined

**Dirty Read:** Read uncommitted data from another transaction that later rolls back.
```sql
-- Transaction 1:
BEGIN;
UPDATE accounts SET balance = 1000 WHERE id = 1;
-- NOT committed yet

-- Transaction 2 (READ UNCOMMITTED):
SELECT balance FROM accounts WHERE id = 1;  -- Returns 1000 (dirty!)

-- Transaction 1:
ROLLBACK;  -- 1000 was never real
```

**Non-Repeatable Read:** Same row returns different values within same transaction.
```sql
-- Transaction 1:
BEGIN;
SELECT price FROM products WHERE id = 5;  -- Returns 100

-- Transaction 2 commits:
UPDATE products SET price = 200 WHERE id = 5;
COMMIT;

-- Transaction 1 (same transaction):
SELECT price FROM products WHERE id = 5;  -- Returns 200! (changed)
```

**Phantom Read:** Re-executing a range query returns new rows.
```sql
-- Transaction 1:
BEGIN;
SELECT * FROM orders WHERE amount > 1000;  -- Returns 5 rows

-- Transaction 2 inserts and commits:
INSERT INTO orders (amount) VALUES (1500);
COMMIT;

-- Transaction 1 (same transaction):
SELECT * FROM orders WHERE amount > 1000;  -- Returns 6 rows! (phantom)
```

### Isolation Level Matrix

| Isolation Level | Dirty Read | Non-Repeatable Read | Phantom Read | Performance |
|----------------|-----------|--------------------|-----------|-----------:|
| READ UNCOMMITTED | Possible | Possible | Possible | Fastest |
| READ COMMITTED | Prevented | Possible | Possible | Fast |
| REPEATABLE READ | Prevented | Prevented | Possible* | Medium |
| SERIALIZABLE | Prevented | Prevented | Prevented | Slowest |

*PostgreSQL's REPEATABLE READ also prevents phantom reads via MVCC snapshots.

### Practical Usage

```sql
-- PostgreSQL defaults
SHOW default_transaction_isolation;  -- read committed

-- Set per transaction
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
BEGIN TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;  -- No dirty reads in PostgreSQL even at this level
```

**Interview tip:** READ COMMITTED is the PostgreSQL default and handles 99% of application needs. Use SERIALIZABLE when correctness absolutely requires it (e.g., double-booking prevention). REPEATABLE READ is good for read-intensive reports.

---

## 4. MVCC (Multi-Version Concurrency Control)

### Core Idea

Instead of locking rows for reads, maintain multiple versions of each row. Readers see a consistent snapshot; writers create new versions.

**Benefit:** Readers never block writers; writers never block readers.

### PostgreSQL MVCC Implementation

```
Each row has system columns:
  xmin: transaction ID that created this version
  xmax: transaction ID that deleted/updated this version (0 if current)
  ctid:  physical location (page, offset) of row

Row lifecycle:
  INSERT: creates new row with xmin=current_txn, xmax=0
  UPDATE: marks old row xmax=current_txn, inserts new row with xmin=current_txn
  DELETE: marks row xmax=current_txn

SELECT uses a snapshot:
  - Snapshot taken at transaction start (READ COMMITTED: per statement)
  - Visible row: xmin committed before snapshot AND (xmax=0 OR xmax not committed yet)

Example:
  Row: {id:1, name:"Alice"} xmin=100, xmax=0  (current version)
  
  Transaction 200 runs UPDATE SET name="Bob":
    Old row: xmin=100, xmax=200  (being deleted by txn 200)
    New row: xmin=200, xmax=0    (new version)
    
  Transaction 150 (started before 200):
    Sees old row (xmin=100 committed, xmax=200 not yet committed to snapshot)
    
  Transaction 250 (started after 200 commits):
    Sees new row (xmin=200 committed)
```

### Dead Tuple Cleanup (VACUUM)

```
Problem: Old row versions accumulate, wasting space
Solution: VACUUM scans pages, removes rows where xmax transaction is committed
          and no active snapshot could still see the old version.

autovacuum: PostgreSQL background process
  - Triggers when dead tuple % exceeds threshold (20% default)
  - Also updates table statistics for query planner

Transaction ID wraparound:
  - 32-bit txn IDs wrap around after ~2 billion transactions
  - VACUUM FREEZE: marks old rows with FrozenXID to prevent future invisibility
```

### InnoDB MVCC

```
InnoDB uses undo logs for versioning:
  - Each row has a 6-byte rollback pointer to undo log entry
  - Undo log contains previous version of the row
  - Chain of undo entries = version history

Read view:
  - Created at transaction start
  - Contains: list of active transaction IDs, min/max active txn ID
  - Row is visible if row's txn ID is committed before read view creation

Purge thread: background process that removes undo log entries no longer needed
```

---

## 5. Pessimistic vs Optimistic Locking

### Pessimistic Locking

**Assumption:** Conflicts are common; lock before modifying.

```sql
-- SELECT FOR UPDATE: acquires exclusive lock on selected rows
BEGIN;
SELECT * FROM inventory WHERE product_id = 42 FOR UPDATE;
-- Other transactions trying to SELECT FOR UPDATE on same row will BLOCK

UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 42;
COMMIT;

Variants:
  FOR UPDATE:        exclusive lock (no concurrent reads in some DBs)
  FOR UPDATE SKIP LOCKED: skip locked rows (useful for job queues)
  FOR SHARE:         shared lock (allow other readers, block writers)
  NOWAIT:            fail immediately if lock unavailable
```

**Use cases:**
- High-contention resources (limited inventory, seats)
- Long-running transactions where optimistic retry would be costly

### Optimistic Locking

**Assumption:** Conflicts are rare; detect at commit time.

```sql
-- Version-based optimistic locking
-- Table has a version column

-- Step 1: Read
SELECT id, price, version FROM products WHERE id = 1;
-- Returns: {id:1, price:100, version:5}

-- Step 2: Business logic in application
-- Step 3: Update with version check
UPDATE products 
SET price = 110, version = version + 1
WHERE id = 1 AND version = 5;  -- Fails if someone else updated first

-- Check rows affected:
--   1 row affected → success
--   0 rows affected → conflict! retry or error
```

**Using CAS (Compare-And-Swap) pattern:**
```python
MAX_RETRIES = 3

def update_price(product_id: int, new_price: float):
    for attempt in range(MAX_RETRIES):
        row = db.query("SELECT price, version FROM products WHERE id = ?", product_id)
        
        affected = db.execute(
            "UPDATE products SET price = ?, version = ? WHERE id = ? AND version = ?",
            new_price, row.version + 1, product_id, row.version
        )
        
        if affected == 1:
            return  # Success
        
        time.sleep(backoff(attempt))
    
    raise Exception("Too many conflicts, giving up")
```

### Comparison

| Dimension | Pessimistic | Optimistic |
|-----------|------------|-----------|
| Lock held | Throughout transaction | Not held (version check at commit) |
| Deadlock risk | Yes (with multiple locks) | No (no locks) |
| Throughput | Lower (lock contention) | Higher (no lock waits) |
| Conflict cost | Prevention (wait) | Detection (retry) |
| Best for | High contention | Low contention |
| Risk | Deadlocks, lock timeouts | Livelock if retries don't back off |

---

## 6. Two-Phase Commit (2PC)

### Protocol

2PC coordinates atomic commits across multiple distributed nodes:

```
Participants: Node A (account service), Node B (inventory service)
Coordinator: Transaction Manager (TM)

Phase 1: PREPARE
  TM → A: "Can you commit transaction 123?"
  TM → B: "Can you commit transaction 123?"
  
  A: acquires locks, writes to WAL, responds "YES" (or "NO" if error)
  B: acquires locks, writes to WAL, responds "YES" (or "NO" if error)

Phase 2: COMMIT (or ABORT)
  If all YES:
    TM → A: "COMMIT 123"
    TM → B: "COMMIT 123"
    A, B: commit, release locks, acknowledge
  
  If any NO:
    TM → A: "ABORT 123"
    TM → B: "ABORT 123"
    A, B: rollback, release locks
```

### Failure Scenarios

```
Scenario 1: Participant fails before sending vote
  TM waits for timeout → assumes NO → sends ABORT
  Safe: transaction was not prepared

Scenario 2: Participant fails after voting YES but before commit
  TM sends COMMIT to other participants
  Failed participant: on recovery, reads WAL, sees "prepared 123" → commits
  (WAL ensures durability of prepared state)

Scenario 3: Coordinator fails AFTER sending PREPARE, BEFORE sending COMMIT
  Participants are stuck in PREPARED state holding locks
  Cannot unilaterally commit or abort (don't know if all voted YES)
  Resolution: manual intervention, or cooperative termination protocol
  THIS IS THE BLOCKING PROBLEM OF 2PC

Scenario 4: Coordinator fails AFTER sending some COMMITs
  Some nodes committed, some didn't
  Resolution: new coordinator reads TM log, resends COMMIT to uncommitted nodes
```

### 2PC in Practice

```
Databases using 2PC:
  - PostgreSQL PREPARE TRANSACTION / COMMIT PREPARED
  - MySQL XA transactions
  - Java EE JTA (Java Transaction API)

Cloud-managed 2PC:
  - Google Spanner: 2PC across Paxos groups, uses TrueTime to order commits
  - AWS DynamoDB Transactions: optimistic 2PC-like protocol

Drawbacks:
  - Blocking protocol: coordinator failure causes indefinite lock holding
  - Latency: minimum 2 round trips across network
  - Not partition tolerant: cannot proceed if coordinator unreachable
```

---

## 7. Three-Phase Commit (3PC)

### Added Phase: Pre-Commit

3PC adds a phase to remove the blocking window:

```
Phase 1: PREPARE (same as 2PC)
  TM → Participants: "Can you commit?"
  All reply YES or NO

Phase 2: PRE-COMMIT (new phase)
  TM → Participants: "Everyone said YES, prepare to commit"
  Participants: acknowledge (no locks released yet)
  
  Key insight: if coordinator fails here, participants know all others voted YES
               They can safely commit without coordinator

Phase 3: COMMIT
  TM → Participants: "Commit now"
  Participants: commit, release locks
```

### Why 3PC Still Has Problems

```
Network partition scenario:
  Coordinator sends PRE-COMMIT to A and B
  Network partitions: A receives PRE-COMMIT, B does not
  
  If coordinator fails:
    A assumes all are ready → A commits
    B received no PRE-COMMIT → B aborts (timeout assumption)
    
  Result: A committed, B rolled back → INCONSISTENT STATE

3PC assumes fail-stop model (nodes either work or stop)
In real networks, message delays create ambiguous "did they receive it?" scenarios
```

**Bottom line:** 3PC is rarely used in practice. Raft/Paxos are preferred for fault-tolerant consensus.

---

## 8. Saga Pattern

### Motivation

Long-running distributed transactions spanning multiple services cannot use 2PC efficiently (too much lock holding, too brittle). Sagas break the transaction into local transactions with compensating transactions for rollback.

### Choreography-Based Saga

Each service publishes events and reacts to events from other services:

```
Order Saga (Choreography):

OrderService:           PLACE_ORDER → publishes OrderCreated
InventoryService:       listens OrderCreated → reserves stock → publishes StockReserved
PaymentService:         listens StockReserved → charges card → publishes PaymentDone
ShippingService:        listens PaymentDone → creates shipment → publishes Shipped

Failure path:
PaymentService FAILS:   publishes PaymentFailed
InventoryService:       listens PaymentFailed → releases stock → publishes StockReleased
OrderService:           listens StockReleased → cancels order → publishes OrderCancelled
```

**Compensating transactions:**
```python
# Normal transaction: ReserveInventory
# Compensating transaction: ReleaseInventory

# Normal: ChargePayment
# Compensating: RefundPayment

# Compensations must be idempotent (called multiple times = same result)
# Compensations may not be immediate (partial rollback window exists)
```

### Orchestration-Based Saga

A central orchestrator (saga state machine) drives the workflow:

```python
class OrderSaga:
    def execute(self, order_id: str):
        saga_state = SagaState(order_id=order_id, step=0, compensations=[])
        
        steps = [
            (self.reserve_inventory, self.release_inventory),
            (self.charge_payment,    self.refund_payment),
            (self.create_shipment,   self.cancel_shipment),
        ]
        
        for forward, compensate in steps:
            try:
                forward(order_id)
                saga_state.compensations.append(compensate)
                saga_state.step += 1
                self.save(saga_state)  # Persist progress
            except Exception:
                # Execute compensations in reverse
                for comp in reversed(saga_state.compensations):
                    try:
                        comp(order_id)
                    except Exception as e:
                        self.alert_humans(order_id, e)  # Manual intervention needed
                raise
```

### Saga vs 2PC Comparison

| Dimension | 2PC | Saga |
|-----------|-----|------|
| Locking | Holds locks across all participants | Only local transaction locks |
| Atomicity | Atomic (all-or-nothing) | ACI (no isolation between steps) |
| Isolation | Strong isolation | No cross-service isolation |
| Intermediate state | Not visible | Visible (partially completed state) |
| Rollback | True rollback | Compensating transactions (semantic undo) |
| Failure recovery | Blocking on coordinator failure | Non-blocking (retry compensations) |
| Complexity | Simpler logic, complex failure handling | Complex logic, handles failures gracefully |
| Use case | Same-database multi-table | Cross-service distributed workflows |

### Choreography vs Orchestration

| Dimension | Choreography | Orchestration |
|-----------|-------------|--------------|
| Control | Decentralized (events) | Centralized (orchestrator) |
| Coupling | Loose coupling | Services coupled to orchestrator |
| Visibility | Hard to track overall state | Easy to track (orchestrator state) |
| Testability | Hard to test end-to-end | Easier (orchestrator testable) |
| Failure handling | Each service handles its failures | Orchestrator handles all failures |
| Best for | Simple flows, event-driven architectures | Complex flows, many steps |

---

## 9. Outbox Pattern

### Problem

Writing to a database and publishing an event is not atomic:
```
// ANTI-PATTERN: Race condition
BEGIN TRANSACTION;
  INSERT INTO orders (id, status) VALUES (123, 'created');
COMMIT;
// If app crashes here → event never published
publish_event("order_created", {id: 123})  // Not atomic with DB write
```

### Solution: Transactional Outbox

```
Write both the domain object AND the event in the same DB transaction:

BEGIN TRANSACTION;
  INSERT INTO orders (id, status) VALUES (123, 'created');
  INSERT INTO outbox (event_type, payload, created_at, sent)
    VALUES ('order_created', '{"id":123}', NOW(), false);
COMMIT;

// Outbox poller (separate process):
// 1. SELECT * FROM outbox WHERE sent = false LIMIT 100 FOR UPDATE SKIP LOCKED
// 2. Publish each event to message broker (Kafka, RabbitMQ, SQS)
// 3. UPDATE outbox SET sent = true WHERE id = ?
// 4. Optionally: DELETE FROM outbox WHERE sent = true AND created_at < NOW() - INTERVAL '7 days'
```

### Change Data Capture (CDC) Approach

```
More efficient than polling: use DB's replication log to detect outbox changes

Debezium (PostgreSQL):
  - Reads PostgreSQL WAL (Write-Ahead Log)
  - Captures every INSERT/UPDATE/DELETE on outbox table
  - Publishes to Kafka in near-real-time
  - No polling overhead, millisecond latency

Debezium configuration:
  database.hostname: postgres
  database.port: 5432
  database.dbname: orders
  table.include.list: public.outbox
  transforms: outbox
  transforms.outbox.type: io.debezium.transforms.outbox.EventRouter
```

### Inbox Pattern (Idempotent Consumer Side)

```sql
-- Idempotent message processing
CREATE TABLE processed_messages (
  message_id UUID PRIMARY KEY,
  processed_at TIMESTAMP DEFAULT NOW()
);

-- In consumer:
BEGIN;
  INSERT INTO processed_messages (message_id) VALUES (?)
    ON CONFLICT DO NOTHING;
  
  GET DIAGNOSTICS affected = ROW_COUNT;
  
  IF affected > 0 THEN
    -- Process the message (idempotent business logic)
    process_order_created(payload);
  END IF;
COMMIT;
```

---

## 10. Idempotency

### Idempotency Keys

An operation is idempotent if applying it multiple times produces the same result as applying it once.

```
POST /payments  (NOT idempotent by default)

With idempotency key:
POST /payments
Idempotency-Key: client-generated-uuid-abc123

Server logic:
  1. Hash Idempotency-Key → lookup in cache/DB
  2. If found: return cached response (don't process again)
  3. If not found: process → store result with key → return result
  
Cache TTL: 24 hours (long enough for retries, not forever)
```

### Implementation

```python
class IdempotentPaymentService:
    def __init__(self, redis: Redis, payment_processor):
        self.redis = redis
        self.processor = payment_processor

    def charge(self, idempotency_key: str, amount: float, customer_id: str):
        cache_key = f"idem:{idempotency_key}"
        
        # Check for existing result (with distributed lock to prevent race)
        with self.redis.lock(f"lock:{cache_key}", timeout=30):
            cached = self.redis.get(cache_key)
            if cached:
                return json.loads(cached)  # Return previous result
            
            # Process payment
            result = self.processor.charge(amount, customer_id)
            
            # Cache result for 24 hours
            self.redis.setex(cache_key, 86400, json.dumps(result))
            
            return result
```

### At-Least-Once + Idempotent Consumer

```
Message delivery guarantees:
  Exactly-once: very hard, high cost
  At-least-once: duplicates possible, but common in practice
  At-most-once:  messages may be lost
  
Best practice: at-least-once delivery + idempotent consumers

Idempotent operations:
  - SET x = 5 (idempotent; same result regardless of how many times applied)
  - INSERT ... ON CONFLICT DO NOTHING (idempotent)
  - DELETE WHERE id = X (idempotent after first delete)

Non-idempotent:
  - INCREMENT balance BY 10 (each application adds 10 more)
  - Solution: track message ID, skip if already processed
```

---

## 11. Conflict Resolution Strategies

### Last Write Wins (LWW)

```
Cassandra default conflict resolution:
  - Each write carries a timestamp (client-provided or server-assigned)
  - On conflict: highest timestamp wins

Problems:
  - Clock skew: servers have slightly different clocks
  - Lost writes: concurrent writes → one silently discarded
  - Not causally consistent: later-timestamped write may be causally earlier

Use when: eventual consistency acceptable, data loss tolerable
         (e.g., user profile updates, not financial transactions)
```

### Vector Clocks

```
Vector clock: [NodeA: 3, NodeB: 2, NodeC: 1]
  - Each node tracks how many updates it has seen from every other node

Causality detection:
  V1 = [A:3, B:2] dominates V2 = [A:2, B:2] → V1 is causally later
  V1 = [A:3, B:1] and V2 = [A:2, B:2] → concurrent (neither dominates)
  
  For concurrent versions: need conflict resolution (merge or ask user)
  
Amazon DynamoDB uses vector clocks (older versions; now uses last-write-wins by default)
Amazon Shopping Cart: concurrent adds are merged (union of items)
```

### Conflict-Free Merged State

```python
# Merge function for conflict resolution
def resolve_conflict(v1: dict, v2: dict, clock1: VectorClock, clock2: VectorClock):
    if clock1.dominates(clock2):
        return v1
    elif clock2.dominates(clock1):
        return v2
    else:  # Concurrent - merge
        return merge_semantically(v1, v2)

# Example: Shopping cart merge
def merge_carts(cart1: Set, cart2: Set) -> Set:
    return cart1 | cart2  # Union of items

# Example: Last-writer-wins per field
def merge_profile(p1: dict, p2: dict, t1: int, t2: int) -> dict:
    if t1 > t2:
        return p1
    return p2
```

---

## 12. CRDTs

CRDTs (Conflict-free Replicated Data Types) are data structures where all concurrent updates can be merged without conflicts, automatically converging to a correct state.

### G-Counter (Grow-Only Counter)

```python
class GCounter:
    def __init__(self, node_id: str, all_nodes: List[str]):
        self.node_id = node_id
        self.counts = {node: 0 for node in all_nodes}
    
    def increment(self):
        self.counts[self.node_id] += 1
    
    def value(self) -> int:
        return sum(self.counts.values())
    
    def merge(self, other: 'GCounter'):
        # Take the maximum count from each node
        for node in self.counts:
            self.counts[node] = max(self.counts[node], other.counts[node])

# Concurrent increments on different nodes:
# Node A: {A:5, B:3}  Node B: {A:4, B:7}
# Merge:             → {A:5, B:7}  (take max per node)
# Value: 5 + 7 = 12  ✓
```

### PN-Counter (Positive-Negative Counter)

```python
class PNCounter:
    def __init__(self, node_id: str, all_nodes: List[str]):
        self.positive = GCounter(node_id, all_nodes)  # tracks increments
        self.negative = GCounter(node_id, all_nodes)  # tracks decrements
    
    def increment(self): self.positive.increment()
    def decrement(self): self.negative.increment()
    
    def value(self) -> int:
        return self.positive.value() - self.negative.value()
    
    def merge(self, other: 'PNCounter'):
        self.positive.merge(other.positive)
        self.negative.merge(other.negative)
```

### OR-Set (Observed-Remove Set)

Handles add/remove conflicts (pure G-Set cannot remove elements):

```python
class ORSet:
    def __init__(self):
        self.elements = {}  # {element: {unique_tag, ...}}
    
    def add(self, element):
        tag = generate_uuid()
        self.elements.setdefault(element, set()).add(tag)
    
    def remove(self, element):
        # Remove all currently observed tags for this element
        if element in self.elements:
            del self.elements[element]
    
    def contains(self, element) -> bool:
        return element in self.elements and len(self.elements[element]) > 0
    
    def merge(self, other: 'ORSet'):
        # For each element, take union of tags
        all_elements = set(self.elements) | set(other.elements)
        for elem in all_elements:
            tags_a = self.elements.get(elem, set())
            tags_b = other.elements.get(elem, set())
            merged_tags = tags_a | tags_b
            if merged_tags:
                self.elements[elem] = merged_tags
            else:
                self.elements.pop(elem, None)

# Concurrent add+remove resolution:
# Node A: add("apple"), tag=T1 → {apple: {T1}}
# Node B: remove("apple") removes {T1} → {}
# Node A adds "apple" again with tag T2
# Merge: Node A has {apple: {T2}}, Node B has {} 
# Result: apple is present (Node A's re-add with new tag wins)
```

### CRDT Use Cases

| CRDT | Use Case |
|------|----------|
| G-Counter | Page view counts, download counts |
| PN-Counter | Real-time user count, inventory |
| LWW-Register | User profile last updated |
| OR-Set | Collaborative document tags |
| RGA (Replicated Growable Array) | Collaborative text editing |
| MV-Register | Shopping cart |

---

## 13. Read-Your-Writes and Monotonic Reads

### Read-Your-Writes Consistency

**Guarantee:** After a user writes a value, they will always read their own write (but other users may see stale data).

```
Problem scenario:
  User posts update to replica-1
  User reads from replica-2 (lagging) → doesn't see their own update
  User thinks their update was lost!

Solutions:
1. Sticky sessions (route user to same replica after write):
   - Use IP hash LB or session cookie to route to same server
   - Problem: fails if that server goes down

2. Read from primary after own write:
   - Track: last_write_timestamp per user
   - If current_time - last_write < replication_lag_estimate: read from primary
   - Otherwise: read from replica

3. Read-your-writes via version tracking:
   - After write, remember version = write_timestamp
   - On read: send min_version in request
   - Replica serves only if local version >= min_version
   - Otherwise, route to primary
```

### Monotonic Reads

**Guarantee:** A user will never read older data after having already read newer data.

```
Problem:
  Read 1 from replica-1 (low lag): sees version 5
  Read 2 from replica-2 (high lag): sees version 3 ← going backwards!

Solution: Route user to same replica consistently
  - Session affinity
  - Consistent hashing on user_id → same replica

When to relax: background data loads where monotonicity not critical
```

---

## 14. Causal Consistency

### Causal Dependencies

```
Sequence of operations with causal dependencies:
  1. Alice writes: "I am posting a job offer"
  2. Bob reads (1), then writes: "I am applying for this job"
  3. Carol reads (2), then writes: "Me too!"

Causal ordering: 1 → 2 → 3
Requirement: No node should see 3 without having seen 1 and 2

Concurrent (no causal dependency):
  Alice writes: "It's sunny today"
  Bob writes:   "I like pancakes"
  These are concurrent — no dependency — order can vary per observer
```

### Causal Broadcast

```python
# Each message carries a vector clock representing causal history
class CausalBroadcast:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.vc = VectorClock()
        self.pending = []  # Buffer for out-of-order messages
    
    def send(self, message: str):
        self.vc.increment(self.node_id)
        broadcast(message, clock=self.vc.copy())
    
    def receive(self, message: str, sender_id: str, clock: VectorClock):
        # Buffer message until all causal dependencies satisfied
        self.pending.append((message, sender_id, clock))
        self._deliver_ready()
    
    def _deliver_ready(self):
        # Deliver message if:
        #   clock[sender] == local_vc[sender] + 1 (next expected from sender)
        #   clock[k] <= local_vc[k] for all k != sender (all other deps satisfied)
        for msg, sender, clock in self.pending[:]:
            if self._can_deliver(sender, clock):
                self.pending.remove((msg, sender, clock))
                self.vc.increment(sender)
                self.on_deliver(msg)
                self._deliver_ready()  # Check if now unblocked others
```

---

## 15. Write Skew Anomaly

### Definition

Write skew occurs when two concurrent transactions read overlapping data and make disjoint writes based on those reads, violating a constraint that depends on the combined data.

### Classic Example: On-Call Doctor

```sql
-- Constraint: At least one doctor must always be on call
-- Doctors: Alice (on_call=true), Bob (on_call=true)

-- Transaction 1 (Alice wants to go off call):
SELECT COUNT(*) FROM doctors WHERE on_call = true;  -- Returns 2
-- "2 doctors on call, safe to go off"
UPDATE doctors SET on_call = false WHERE name = 'Alice';

-- Transaction 2 (Bob wants to go off call, CONCURRENT):
SELECT COUNT(*) FROM doctors WHERE on_call = true;  -- Also returns 2!
-- "2 doctors on call, safe to go off"
UPDATE doctors SET on_call = false WHERE name = 'Bob';

-- RESULT: 0 doctors on call! Constraint violated.
-- Neither transaction wrote to the same row (no dirty write)
-- READ COMMITTED and REPEATABLE READ don't prevent this
-- Requires: SERIALIZABLE isolation
```

### Other Write Skew Examples

```sql
-- Room booking collision
-- Two users book same room for overlapping times
-- Each checks count of bookings for that time slot → sees 0 → books

-- Username uniqueness (without DB constraint)
-- Two users claim same username
-- Each checks existence → not found → inserts

-- Inventory overselling
-- 1 item left, 2 orders both read qty=1, both proceed to decrement
```

### Prevention

```sql
-- Option 1: SERIALIZABLE isolation
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SELECT COUNT(*) FROM doctors WHERE on_call = true;
UPDATE doctors SET on_call = false WHERE name = 'Alice';
COMMIT;
-- PostgreSQL uses SSI (Serializable Snapshot Isolation)
-- Detects read-write conflicts, aborts one transaction

-- Option 2: SELECT FOR UPDATE (locks the rows even on reads)
BEGIN;
SELECT COUNT(*) FROM doctors WHERE on_call = true FOR UPDATE;
UPDATE doctors SET on_call = false WHERE name = 'Alice';
COMMIT;

-- Option 3: Materialized conflict (create a lock row)
SELECT * FROM locks WHERE resource = 'oncall_minimum' FOR UPDATE;
```

---

## 16. Distributed Locking

### Redis Redlock Algorithm

```python
# Redlock: acquire lock on majority of N independent Redis nodes

class Redlock:
    def __init__(self, redis_nodes: List[Redis]):
        self.nodes = redis_nodes
        self.quorum = len(redis_nodes) // 2 + 1
    
    def acquire(self, resource: str, ttl_ms: int) -> Optional[str]:
        token = generate_uuid()
        start_time = time_ms()
        acquired = 0
        
        for node in self.nodes:
            if self._acquire_on_node(node, resource, token, ttl_ms):
                acquired += 1
        
        elapsed = time_ms() - start_time
        validity = ttl_ms - elapsed - CLOCK_DRIFT_FACTOR
        
        if acquired >= self.quorum and validity > 0:
            return token  # Lock acquired
        else:
            self.release(resource, token)  # Release partial acquisitions
            return None
    
    def _acquire_on_node(self, node, resource, token, ttl_ms) -> bool:
        return node.set(resource, token, nx=True, px=ttl_ms)
    
    def release(self, resource: str, token: str):
        for node in self.nodes:
            # Only release if we own the lock (compare token)
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            node.eval(lua_script, 1, resource, token)
```

**Redlock controversy (Martin Kleppmann):**
- Clock drift can cause two clients to both think they hold the lock
- GC pause between lock acquisition and use can invalidate the lock
- For true safety: use fencing tokens (monotonically increasing IDs)

### ZooKeeper Distributed Lock

```
ZooKeeper uses ephemeral sequential znodes:

1. Client creates znode: /locks/my_lock_0000000001
2. Client lists children: [0000000001, 0000000002, 0000000003]
3. If lowest ID: client holds lock
4. Else: watch the next-lowest node (0000000001 watches nothing if it IS lowest)
5. When watched node is deleted (holder releases/crashes): client re-evaluates

Benefits over Redlock:
  - Strongly consistent (ZooKeeper uses ZAB consensus protocol)
  - Ephemeral node: lock auto-released if client session expires
  - No clock dependence
```

### Database Advisory Locks (PostgreSQL)

```sql
-- Session-level advisory lock (released when session ends)
SELECT pg_advisory_lock(42);    -- acquire
SELECT pg_advisory_unlock(42);  -- release

-- Transaction-level advisory lock (released at end of transaction)
SELECT pg_advisory_xact_lock(42);

-- Non-blocking (returns false if can't acquire)
SELECT pg_try_advisory_lock(42);

-- Use case: ensure only one instance of cron job runs
DO $$
BEGIN
  IF pg_try_advisory_lock(hashtext('nightly_job')) THEN
    -- Run job
    PERFORM run_nightly_job();
    PERFORM pg_advisory_unlock(hashtext('nightly_job'));
  END IF;
END $$;
```

---

## 17. Exactly-Once Delivery

### Kafka Idempotent Producers

```python
# Kafka producer with idempotency enabled
producer = KafkaProducer(
    bootstrap_servers=['kafka:9092'],
    enable_idempotence=True,         # Assign producer ID + sequence numbers
    acks='all',                       # Wait for all ISR replicas
    retries=2147483647,               # Retry indefinitely
    max_in_flight_requests_per_connection=5  # Ordered delivery with idempotence
)

# Kafka assigns ProducerID (PID) to each producer
# Each message gets sequence number per partition
# Broker rejects duplicate (same PID + seqnum)
# Surviving retries don't produce duplicates
```

### Kafka Transactions (Exactly-Once Across Topics)

```python
producer = KafkaProducer(
    bootstrap_servers=['kafka:9092'],
    transactional_id='order-processor-1'  # Unique per producer instance
)
producer.init_transactions()

try:
    producer.begin_transaction()
    
    # Read from input topic
    # Process
    
    # Write to output topic
    producer.send('processed-orders', value=result)
    
    # Commit offsets atomically with the output (no duplicate processing)
    producer.send_offsets_to_transaction(
        offsets={TopicPartition('orders', 0): 100},
        group_id='order-consumer-group'
    )
    
    producer.commit_transaction()
except Exception:
    producer.abort_transaction()
```

### Transactional Outbox + Kafka (End-to-End Exactly-Once)

```
Full exactly-once pipeline:

[Application] 
  → DB transaction: INSERT domain_object + outbox_event
  → Debezium reads WAL → publishes to Kafka (idempotent producer)
  → Consumer: process + mark offset (Kafka transactions)
  → Consumer: INSERT processed_messages + business update (DB transaction)

At-least-once delivery + idempotent consumer = effectively exactly-once semantics
```

---

## 18. Linearizability Testing

### Jepsen Framework

Jepsen is an automated tool for testing distributed systems under network partitions and clock skew:

```
Jepsen test structure:
  1. nemesis: injects failures (partition network, kill nodes, corrupt clocks)
  2. client: performs concurrent operations (reads, writes, CAS)
  3. checker: verifies operation history satisfies claimed consistency model

History recorded:
  {:type :invoke, :f :write, :value 5, :time 1000ns}
  {:type :ok,     :f :write, :value 5, :time 2000ns}
  {:type :invoke, :f :read,  :value nil, :time 1500ns}
  {:type :ok,     :f :read,  :value 3,   :time 3000ns}

Checker verifies: Can we construct a linearized history?
  - All operations appear to execute atomically
  - Results are consistent with sequential execution
```

### Knossos Checker

```
Knossos is Jepsen's linearizability checker:

Algorithm: WGL (Wing & Gong Linearizability)
  - For each operation: try all possible linearization points
  - Check if resulting sequential history is valid for the data structure model
  - Exponential worst case, but pruning makes it practical for small histories

Output:
  "Valid" → history is linearizable
  "Invalid" → found non-linearizable execution (with counterexample)
```

---

## 19. Database Choice by Consistency Requirement

| Use Case | Consistency Need | Recommended DB | Why |
|----------|-----------------|----------------|-----|
| Bank transfers | Strong (ACID) | PostgreSQL, CockroachDB | ACID with serializable isolation |
| E-commerce orders | Strong per order | PostgreSQL, MySQL | ACID, familiar SQL |
| User profiles | Read-your-writes | DynamoDB (strong reads) | Simple key-value, strong read option |
| Social media feed | Eventual | Cassandra, DynamoDB | High write throughput |
| Inventory management | Strong | PostgreSQL + SELECT FOR UPDATE | Prevents overselling |
| Shopping cart | Eventual + merge | DynamoDB with CRDT logic | Availability over consistency |
| Global config | Strong + distributed | etcd, ZooKeeper | Raft consensus |
| Analytics | Eventual fine | ClickHouse, BigQuery | Batch, not OLTP |
| Session store | Read-your-writes | Redis | Fast, TTL-based |
| Distributed lock | Strong | Redis Redlock, ZooKeeper | Purpose-built |

---

## 20. Quick Reference

### Isolation Level Anomaly Prevention Table

| Level | Dirty Read | Non-Repeatable Read | Phantom Read | Write Skew |
|-------|-----------|--------------------|-----------|---------:|
| READ UNCOMMITTED | May occur | May occur | May occur | May occur |
| READ COMMITTED | Prevented | May occur | May occur | May occur |
| REPEATABLE READ | Prevented | Prevented | Prevented* | May occur |
| SERIALIZABLE | Prevented | Prevented | Prevented | Prevented |

*PostgreSQL REPEATABLE READ prevents phantoms via MVCC snapshots.

### Saga vs 2PC Decision Matrix

| Scenario | Recommendation |
|----------|---------------|
| All services share one database | 2PC (or just one ACID transaction) |
| 2-3 services, fast operations | 2PC (XA transactions) |
| 5+ services, long-running operations | Saga (orchestration preferred) |
| Need strong atomicity, no intermediate visibility | 2PC |
| Can tolerate eventual consistency | Saga |
| Microservices with independent data stores | Saga |

### Consistency Model Hierarchy

```
Strongest consistency
        |
   Linearizability    (all operations appear atomic, globally ordered)
        |
  Sequential Consistency  (per-process order respected globally)
        |
  Causal Consistency  (causal dependencies respected)
        |
  Read-Your-Writes    (own writes always visible to self)
        |
  Monotonic Reads     (reads never go backwards)
        |
  Monotonic Writes    (own writes applied in order)
        |
  Eventual Consistency (convergence given no new writes)
        |
Weakest consistency
```

### Common Interview Q&A

**Q: What is the difference between 2PC and Saga?**
A: 2PC is a blocking protocol requiring all participants to hold locks until coordinator decision; Saga uses local transactions with compensating transactions, has no cross-service locking, tolerates partial failure but allows intermediate states to be visible.

**Q: How does MVCC help performance?**
A: MVCC allows readers and writers to operate concurrently without blocking each other. Readers see a consistent snapshot of the database without taking read locks, allowing high concurrency.

**Q: What is write skew and how do you prevent it?**
A: Write skew occurs when two concurrent transactions read overlapping data and make disjoint writes that together violate a constraint. Prevention: SERIALIZABLE isolation (SSI in PostgreSQL), SELECT FOR UPDATE on the read set, or application-level locks.

**Q: Why is Redlock controversial?**
A: Clock drift on Redis nodes can cause the TTL to expire on some nodes before others, potentially allowing two clients to believe they hold the lock simultaneously. GC pauses can also expire a lock while the holder is still "within" the lock. Fencing tokens are needed for true safety.
