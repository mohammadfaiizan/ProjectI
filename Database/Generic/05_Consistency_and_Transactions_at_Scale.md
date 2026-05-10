# Consistency and Transactions at Scale

## Easy (Q1–Q7)

---

**Q1. What is the difference between strong consistency and eventual consistency in a distributed database?**

**Strong consistency (linearizability):**
Every read reflects the most recent write, regardless of which node serves the request. From the application's perspective, operations appear to execute instantaneously in a global total order.

```
T=0: Write balance=1000 (committed on Node A)
T=1: Read from Node B → returns 1000 (guaranteed to see the write)

Implementation: write must be acknowledged by a quorum before returning to client
Cost: write latency = network RTT to quorum members (e.g., 20–80ms in distributed setup)
```

**Eventual consistency:**
After a write, reads may return stale data for a period. Eventually, when no new writes occur, all nodes converge to the same value.

```
T=0: Write balance=1000 (committed on Node A)
T=0.1s: Read from Node B → returns 900 (replica not yet updated)
T=0.5s: Read from Node B → returns 1000 (replication has propagated)

Implementation: write acknowledged by one node; async replication to others
Cost: no coordination overhead → writes are fast (< 5ms)
Staleness window: typically milliseconds to seconds
```

**When it matters:**
```
Use strong consistency:
  Bank balance (never show stale balance before a transaction)
  Inventory count (avoid oversell)
  Auth tokens (revoked token must be immediately invalid everywhere)

Use eventual consistency:
  Social media like counts (off by a few for a few seconds: acceptable)
  User preference settings (slightly stale is fine)
  Product view counts / analytics events
```

---

**Q2. What is a transaction and what do ACID properties guarantee?**

A transaction is a sequence of database operations executed as a single logical unit — either all operations succeed (commit) or all are undone (rollback).

**ACID properties:**

**Atomicity:** All or nothing. If any operation fails, all changes are rolled back.
```sql
BEGIN;
UPDATE accounts SET balance = balance - 500 WHERE id = 1;  -- debit
UPDATE accounts SET balance = balance + 500 WHERE id = 2;  -- credit
COMMIT;
-- If the second UPDATE fails: ROLLBACK undoes the first UPDATE too
-- Money cannot disappear or appear from nowhere
```

**Consistency:** The database moves from one valid state to another. All constraints, triggers, and rules remain satisfied after the transaction.
```sql
-- balance NOT NULL, CHECK (balance >= 0) constraint
-- A transaction trying to set balance to -100 will fail → database stays consistent
```

**Isolation:** Concurrent transactions do not interfere with each other's intermediate state.
```
T1 and T2 run concurrently
T1 cannot read T2's uncommitted changes
T2 cannot read T1's uncommitted changes
Result is equivalent to T1 and T2 running serially
```

**Durability:** Once committed, data survives system crashes (written to durable storage via WAL).
```
COMMIT returns → data is in WAL on disk → survives immediate power failure
```

---

**Q3. What are the four SQL isolation levels and what anomaly does each prevent?**

| Isolation Level | Dirty Read | Non-Repeatable Read | Phantom Read |
|---|---|---|---|
| READ UNCOMMITTED | Possible | Possible | Possible |
| READ COMMITTED | Prevented | Possible | Possible |
| REPEATABLE READ | Prevented | Prevented | Possible* |
| SERIALIZABLE | Prevented | Prevented | Prevented |

*MySQL InnoDB prevents phantom reads at REPEATABLE READ via gap locks.

**Dirty read:** Reading uncommitted changes from another transaction.
**Non-repeatable read:** Re-reading a row within a transaction returns different values (another transaction updated and committed between reads).
**Phantom read:** Re-running a range query returns different rows (another transaction inserted/deleted rows in that range).

```sql
-- Set isolation level:
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
-- or globally:
SET default_transaction_isolation = 'repeatable read';
```

**Default levels:** PostgreSQL = READ COMMITTED, MySQL InnoDB = REPEATABLE READ.

---

**Q4. What is a deadlock and what are the strategies to prevent them?**

A deadlock is a cycle where two or more transactions each hold a lock that another needs, so none can proceed.

```
T1: Locks Row A → waiting for Row B
T2: Locks Row B → waiting for Row A
Neither can proceed → database detects the cycle → aborts one transaction (the "victim")
```

**Prevention strategies:**

**1. Consistent lock ordering** (most effective):
```sql
-- Always lock rows with lower ID first
BEGIN;
SELECT * FROM accounts WHERE id IN (1, 2) ORDER BY id FOR UPDATE;
-- Locks id=1, then id=2 — always in the same order
-- T1 and T2 both try to lock id=1 first → one blocks, other proceeds → no deadlock
```

**2. Short transactions:**
```
Lock, do minimal work, unlock quickly
Never call external APIs or wait for user input while holding locks
```

**3. Lock all needed rows at once:**
```sql
-- Instead of locking rows one at a time, lock all in one SELECT FOR UPDATE
SELECT * FROM orders WHERE order_id IN (100, 200, 300) FOR UPDATE;
```

**4. Application-level retry:**
```python
import psycopg2
def transfer(conn, from_id, to_id, amount):
    for attempt in range(3):
        try:
            with conn.cursor() as cur:
                cur.execute("BEGIN")
                cur.execute("SELECT * FROM accounts WHERE id = %s FOR UPDATE", [from_id])
                cur.execute("SELECT * FROM accounts WHERE id = %s FOR UPDATE", [to_id])
                # ... perform transfer
                cur.execute("COMMIT")
            return  # success
        except psycopg2.errors.DeadlockDetected:
            conn.rollback()
            time.sleep(0.05 * (2 ** attempt))  # exponential backoff
    raise Exception("Transfer failed after 3 attempts")
```

---

**Q5. What is the difference between optimistic and pessimistic locking?**

**Pessimistic locking:** Assume conflicts will happen. Lock the row when reading to prevent concurrent modifications.

```sql
BEGIN;
SELECT * FROM inventory WHERE sku = 'LAPTOP' FOR UPDATE;  -- exclusive lock
-- Other transactions: blocked on FOR UPDATE for this row
UPDATE inventory SET qty = qty - 1 WHERE sku = 'LAPTOP';
COMMIT;
```

- Good when: conflicts are frequent (popular item, high-contention row)
- Bad when: many concurrent users → long wait queues → poor user experience

**Optimistic locking:** Assume conflicts are rare. Read without locking; verify at write time using a version number or timestamp.

```sql
-- Read (no lock):
SELECT id, qty, version FROM inventory WHERE sku = 'LAPTOP';
-- Returns: qty=5, version=42

-- Write (include version check):
UPDATE inventory
SET qty = qty - 1, version = version + 1
WHERE sku = 'LAPTOP' AND version = 42;
-- If 0 rows updated: someone else modified it → retry
```

- Good when: conflicts are rare (most updates succeed on first try)
- Bad when: high conflict rate → many retries → wasted work

**Rule:** Use pessimistic locking for always-contended resources (flash sales, last seat, shared queue). Use optimistic locking for typical CRUD operations where simultaneous edits are rare.

---

**Q6. What is a two-phase commit (2PC) and what problem does it solve?**

2PC is a distributed transaction protocol that ensures atomicity across multiple database nodes or services.

**The problem:** A bank transfer needs to debit Account A on Database 1 and credit Account B on Database 2. How do you guarantee that either both happen or neither?

**2PC Protocol:**

```
Phase 1 — Prepare:
  Coordinator → DB1: "Can you commit: debit Account A by $100?"
  Coordinator → DB2: "Can you commit: credit Account B by $100?"
  
  DB1: Locks the row, executes the statement, writes to WAL, replies: "PREPARED"
  DB2: Locks the row, executes the statement, writes to WAL, replies: "PREPARED"

Phase 2 — Commit (only if all replied PREPARED):
  Coordinator → DB1: "COMMIT"
  Coordinator → DB2: "COMMIT"
  Both apply the change and release locks.
  
  If any participant replied ABORT in Phase 1:
    Coordinator → DB1: "ROLLBACK"
    Coordinator → DB2: "ROLLBACK"
```

**Critical failure mode:** If the coordinator crashes after Phase 1 but before Phase 2, both participants hold locks indefinitely (in-doubt transactions) until the coordinator recovers with its decision.

---

**Q7. What is the difference between a database transaction and a business transaction (saga)?**

**Database transaction:** A short-lived, atomic unit of work within a single database. Typically milliseconds.
- Uses locks (pessimistic) or version checks (optimistic)
- Rolled back atomically if any step fails
- All operations on the same database connection

**Business transaction (saga):** A long-running, multi-step process that spans multiple services or databases.
- Cannot use a single ACID transaction (multiple services, minutes/hours duration)
- Implemented as a sequence of local database transactions with compensating actions

```
Business transaction: E-commerce order fulfillment (may take minutes)

Step 1: Order Service — INSERT order (local ACID ✓)
Step 2: Payment Service — charge credit card (local ACID ✓)
Step 3: Inventory Service — decrement stock (local ACID ✓)
Step 4: Shipping Service — create shipment (local ACID ✓)

If Step 3 fails (out of stock):
  Compensating actions:
    Payment Service: refund credit card
    Order Service: UPDATE order SET status = 'cancelled'
    
These compensating actions are also local ACID transactions — no global lock held
```

The saga pattern trades atomicity for availability — there is a brief window where some steps have committed but others haven't yet.

---

## Medium (Q8–Q15)

---

**Q8. How does MVCC (Multi-Version Concurrency Control) allow reads and writes to proceed concurrently without blocking each other?**

MVCC keeps multiple versions of each row, allowing readers to see a consistent snapshot of the data at a point in time without blocking writers.

**PostgreSQL MVCC:**
```
Each row has hidden system columns:
  xmin: transaction ID that created this row version
  xmax: transaction ID that deleted/updated this row version (0 = still live)

When T2 reads while T1 is updating:
  
  Before T1's UPDATE:
    Row: {id=1, name="Alice", xmin=100, xmax=0}   ← live row
    
  T1 runs: UPDATE users SET name = "Alicia" WHERE id = 1
    Old row: {id=1, name="Alice",  xmin=100, xmax=200}  ← dead (deleted by T1=200)
    New row: {id=1, name="Alicia", xmin=200, xmax=0}    ← live (created by T1=200)
    (T1 has not committed yet)

  T2 reads: SELECT name FROM users WHERE id = 1
    T2's snapshot: only see rows where xmin ≤ T2's snapshot AND (xmax = 0 OR xmax > T2's snapshot)
    → Sees old row {name="Alice"} because T1=200 is not committed yet
    → T2 is NOT blocked by T1's write
    
  T1 commits:
    New row (xmin=200) becomes visible to new transactions
    Old row (xmax=200) is dead — will be reclaimed by VACUUM
```

**Key benefit:** Readers and writers never block each other (unlike lock-based concurrency where a write locks out all reads on that row).

**Key cost:** Dead row versions accumulate → VACUUM required to reclaim space.

---

**Q9. What is the phantom read anomaly and how does SERIALIZABLE isolation prevent it?**

**Phantom read:** A transaction re-executes a range query and finds new rows have appeared (or disappeared) because another committed transaction inserted/deleted rows matching the range.

```
T1 (REPEATABLE READ):
  SELECT COUNT(*) FROM orders WHERE amount > 1000;  -- returns 5

T2 (concurrent):
  INSERT INTO orders (customer_id, amount) VALUES (7, 1500);  COMMIT;

T1 re-executes:
  SELECT COUNT(*) FROM orders WHERE amount > 1000;  -- returns 6!
  → T1 saw a "phantom" row that didn't exist at the start of its transaction
```

**Why REPEATABLE READ doesn't prevent it:**
- REPEATABLE READ prevents re-reading the same row from changing
- But it does not prevent new rows appearing in range queries
- T1 did not "read" the phantom row initially — it appeared from outside its snapshot

**How SERIALIZABLE prevents it:**

**PostgreSQL SSI (Serializable Snapshot Isolation):**
```
PostgreSQL tracks read-write dependencies between concurrent transactions:
  T1 reads: range [amount > 1000]       → "T1 read a predicate"
  T2 writes: INSERT with amount=1500     → "T2 wrote to T1's predicate range"
  
  Dependency: T2 → T1 (T2's write would affect T1's read)
  T1 reads before T2 commits → T2's write creates a dangerous anti-dependency cycle
  
  SSI detects this dangerous cycle → aborts T1 or T2 (the one that would cause non-serializable behavior)
  Application retries the aborted transaction
```

**Cost of SERIALIZABLE:** More transactions may be aborted and retried. Best for use cases where correctness is critical (financial systems, inventory).

---

**Q10. How do you implement an idempotent database operation and why is it critical in distributed systems?**

An idempotent operation produces the same result whether executed once or multiple times.

**Why it matters in distributed systems:**
```
Payment service sends request to Database:
  → Database processes payment
  → Database sends ACK
  → Network failure: ACK never reaches payment service
  
Payment service: "Did it work? I don't know. I'll retry."
  → If operation is NOT idempotent: payment charged twice!
  → If operation IS idempotent: retry is safe (second execution is no-op)
```

**Implementing idempotency with idempotency keys:**

```sql
-- Store idempotency key per operation:
CREATE TABLE payment_idempotency (
    idempotency_key  UUID    PRIMARY KEY,
    payment_result   JSONB   NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT NOW()
) WITH (fillfactor = 80);

-- Idempotent payment processing:
CREATE OR REPLACE FUNCTION process_payment(
    p_idempotency_key UUID,
    p_amount NUMERIC,
    p_account_id INT
) RETURNS JSONB AS $$
DECLARE
    v_existing JSONB;
    v_result   JSONB;
BEGIN
    -- Check if already processed:
    SELECT payment_result INTO v_existing
    FROM payment_idempotency WHERE idempotency_key = p_idempotency_key;
    
    IF FOUND THEN
        RETURN v_existing;  -- Same result as first execution, no double-charge
    END IF;
    
    -- Process the payment (first time only):
    INSERT INTO payments (account_id, amount) VALUES (p_account_id, p_amount)
    RETURNING jsonb_build_object('payment_id', payment_id, 'status', 'success') INTO v_result;
    
    -- Record the idempotency key with the result:
    INSERT INTO payment_idempotency (idempotency_key, payment_result) VALUES (p_idempotency_key, v_result);
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql;
```

**Using `ON CONFLICT DO NOTHING` for simple idempotency:**
```sql
-- Insert event — safe to retry:
INSERT INTO events (event_id, user_id, type, created_at)
VALUES ('uuid-from-client', 1234, 'click', NOW())
ON CONFLICT (event_id) DO NOTHING;
-- Second call with same event_id: silently ignored → idempotent
```

---

**Q11. What is the outbox pattern and how does it ensure consistency between a database write and an event publication?**

**The dual-write problem:**
```python
# WRONG: two separate writes — one might fail
def place_order(order_data):
    db.execute("INSERT INTO orders ...", order_data)   # DB write succeeds
    kafka.send("order.placed", order_data)             # Kafka publish fails → inconsistency!
    # Order is in DB but event never published → downstream services never process it
```

**The outbox pattern solves this:**
```sql
-- Write to DB and outbox table in ONE local ACID transaction:
BEGIN;
INSERT INTO orders (order_id, customer_id, amount, status)
VALUES (123, 456, 99.99, 'pending');

INSERT INTO outbox (event_id, aggregate_id, event_type, payload, created_at)
VALUES (gen_random_uuid(), 123, 'order.placed',
        '{"order_id":123,"customer_id":456,"amount":99.99}', NOW());
COMMIT;
-- If either fails: both rolled back atomically
-- If both succeed: outbox entry guarantees the event will eventually be published
```

```python
# Outbox relay worker (separate process):
def relay_outbox_events():
    while True:
        events = db.execute("""
            SELECT event_id, event_type, payload
            FROM outbox
            WHERE processed_at IS NULL
            ORDER BY created_at
            LIMIT 100
            FOR UPDATE SKIP LOCKED
        """)
        
        for event in events:
            kafka.send(event.event_type, event.payload)
            db.execute("UPDATE outbox SET processed_at = NOW() WHERE event_id = %s", event.event_id)
        
        time.sleep(0.1)  # poll interval
```

**Guarantee:** The DB write and event publication are atomic (via the outbox table). The relay may publish the event more than once (if it crashes after publishing but before marking `processed_at`), so consumers must be idempotent.

---

**Q12. How does the write skew anomaly differ from dirty reads and lost updates? What isolation level prevents it?**

**Write skew** is the subtlest isolation anomaly: two transactions both read the same set of rows, make a decision based on those reads, then write to different rows — but the combined writes violate a business invariant that neither write alone violates.

**Example: hospital on-call system**
```
Business rule: at least one doctor must be on-call at all times
State: both Dr. A and Dr. B are on-call

T1 (Dr. A going off-call):
  SELECT COUNT(*) FROM on_call WHERE duty = TRUE;  -- sees 2 → OK to go off-call
  UPDATE on_call SET duty = FALSE WHERE doctor_id = 'A';

T2 (Dr. B going off-call, concurrent):
  SELECT COUNT(*) FROM on_call WHERE duty = TRUE;  -- also sees 2 → OK to go off-call
  UPDATE on_call SET duty = FALSE WHERE doctor_id = 'B';

Result: both doctors off-call → invariant violated
Neither wrote to a row the other read (different rows) → no write-write conflict detected at REPEATABLE READ
```

**Why REPEATABLE READ fails:**
```
T1 reads the on_call count (a predicate over the table)
T2's write creates new doctor B off-call state
Neither write modifies what the other READ → REPEATABLE READ doesn't catch it
The problem is a read-write anti-dependency, not a write-write conflict
```

**SERIALIZABLE prevents it:**
```
PostgreSQL SSI tracks that:
  T1 read the on_call count
  T2's write would change the result of T1's read (now count=1 instead of 2)
  T1 also writes (changes duty for A)
  → Circular dependency detected → one transaction aborted → retry → serializable result
```

**Fix without SERIALIZABLE:**
```sql
-- Materialize the conflict: lock the rows being read
BEGIN;
SELECT COUNT(*) FROM on_call WHERE duty = TRUE FOR UPDATE;  -- locks all on_call rows
-- Now T2 blocks until T1 completes → no write skew possible
```

---

**Q13. How does the Saga pattern work in practice and what are its trade-offs vs 2PC?**

**Saga: sequence of local transactions with compensating actions**

Two coordination styles:

**Choreography-based saga** (event-driven):
```
Order Service: INSERT order → publishes OrderCreated event
                                          ↓
Payment Service: charges card → publishes PaymentProcessed event
                                                    ↓
Inventory Service: decrements stock → publishes StockReserved event
                                                           ↓
Shipping Service: creates shipment → publishes ShipmentCreated event

On failure (inventory out of stock):
  Inventory Service: publishes StockReservationFailed event
                                    ↓
  Payment Service: refunds card → publishes PaymentRefunded event
                                          ↓
  Order Service: cancels order

No central coordinator — services react to events
```

**Orchestration-based saga** (central coordinator):
```python
class OrderSaga:
    def execute(self, order_id):
        try:
            payment_id = payment_service.charge(order_id)
            reservation_id = inventory_service.reserve(order_id)
            shipment_id = shipping_service.create(order_id)
            order_service.complete(order_id)
        except PaymentFailed:
            order_service.cancel(order_id)
        except InventoryFailed:
            payment_service.refund(payment_id)
            order_service.cancel(order_id)
        except ShipmentFailed:
            inventory_service.release(reservation_id)
            payment_service.refund(payment_id)
            order_service.cancel(order_id)
```

**Comparison: 2PC vs Saga**

| Aspect | 2PC | Saga |
|---|---|---|
| Consistency | Strong (atomic) | Eventual (steps visible individually) |
| Locks held | During entire protocol | Only during each local step |
| Latency | High (multiple round trips) | Low per step |
| Coordinator failure | In-doubt state (blocking) | Saga can resume from last step |
| Cross-service support | Requires all services to support XA | Any service with local ACID |
| Rollback | True rollback (no partial state visible) | Compensating transactions (partial state briefly visible) |
| Use case | Cross-database atomic writes | Cross-service business processes |

---

**Q14. How does a relational database handle concurrent writes to the same row? Walk through the locking mechanism.**

**Scenario:** Two concurrent transactions both try to update `inventory.qty` for sku='LAPTOP' (qty=5).

```
T1: BEGIN;
    SELECT qty FROM inventory WHERE sku = 'LAPTOP' FOR UPDATE;
    -- T1 acquires exclusive row-level lock on this row

T2: BEGIN;
    SELECT qty FROM inventory WHERE sku = 'LAPTOP' FOR UPDATE;
    -- T2 tries to acquire exclusive lock → BLOCKED (T1 holds it)
    -- T2 waits...

T1: UPDATE inventory SET qty = qty - 1 WHERE sku = 'LAPTOP';
    -- qty = 4
    COMMIT;
    -- T1 releases the lock

T2: (unblocked)
    -- T2's SELECT now executes and reads qty = 4
    UPDATE inventory SET qty = qty - 1 WHERE sku = 'LAPTOP';
    -- qty = 3
    COMMIT;
```

**Without FOR UPDATE (no explicit lock):**
```
T1: SELECT qty FROM inventory WHERE sku = 'LAPTOP';  -- reads 5
T2: SELECT qty FROM inventory WHERE sku = 'LAPTOP';  -- reads 5 (concurrent, no lock)
T1: UPDATE inventory SET qty = 4 WHERE sku = 'LAPTOP';  COMMIT;
T2: UPDATE inventory SET qty = 4 WHERE sku = 'LAPTOP';  COMMIT;
-- Lost update! qty should be 3 but it's 4
```

**PostgreSQL lock escalation for UPDATE:**
```
When UPDATE is issued (without prior FOR UPDATE):
  PostgreSQL first takes a "For Update" lock on the old row version
  If another transaction holds a conflicting lock: current transaction waits
  If row was updated by another committed transaction: re-read, re-evaluate WHERE, re-apply update
  This prevents lost updates automatically (unlike some databases that allow it)
```

---

**Q15. How do you design a database schema for a global distributed application where different regions own different data?**

**Data ownership model:**
```
Each user's data is owned by their home region
The home region is the single authoritative writer for that data
Other regions have read-only async copies of the data (with replication lag)
```

**Schema design for region ownership:**
```sql
-- Global routing metadata (tiny, replicated to all regions):
CREATE TABLE user_region_routing (
    user_id      BIGINT  PRIMARY KEY,
    home_region  TEXT    NOT NULL,  -- 'NA', 'EU', 'APAC'
    shard_id     INT     NOT NULL   -- which shard within the region
);

-- In each region's database, enforce ownership:
CREATE TABLE users (
    user_id     BIGINT    PRIMARY KEY,
    email       TEXT      UNIQUE NOT NULL,
    home_region TEXT      NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Row-level security: EU cluster only accepts EU user writes
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
CREATE POLICY region_ownership ON users
    FOR INSERT WITH CHECK (home_region = current_setting('app.current_region'));
    
-- Application sets the region when connecting:
SET app.current_region = 'EU';
```

**Cross-region data access:**
```
User A (NA) views profile of User B (EU):
  Option 1: Application routes request to EU region API → EU database serves it → 80ms latency
  Option 2: EU database is async-replicated to NA → NA serves from replica (stale by ~100ms)
  
Choose Option 1 for profile pages (freshness matters), Option 2 for cached/static content

Cross-region write (EU user places order involving NA product inventory):
  Order record: written to EU cluster (user's home region)
  NA inventory: decremented via API call to NA service (or Saga: reserve → confirm pattern)
  Accept that the two writes are not atomic (Saga with compensation)
```

**GDPR and data residency enforcement:**
```sql
-- EU data must never leave EU
-- At DB level: EU cluster has no replication to non-EU regions for user_id range X–Y
-- At application level: EU user requests always routed to EU cluster
-- At network level: EU cluster in EU VPC, no cross-region replication configured
-- Audit: pg_audit extension logs all SELECT/INSERT on user tables with region verification
```

---

## Hard (Q16–Q20)

---

**Q16. Design a distributed locking system using a database to coordinate access to a shared resource across multiple application servers.**

**Use case:** A scheduled job should run on exactly one application server at a time, even with 20 app servers running.

**Implementation using PostgreSQL advisory locks:**

```sql
-- Advisory lock: takes a 64-bit integer key
-- Session-level: held until connection closes or explicitly released
-- Transaction-level: held until transaction ends

-- Try to acquire a non-blocking advisory lock:
SELECT pg_try_advisory_lock(12345);  -- 12345 is the job's unique key
-- Returns: true if acquired, false if another session holds it

-- Job runner:
CREATE OR REPLACE FUNCTION run_scheduled_job(job_id BIGINT)
RETURNS VOID AS $$
BEGIN
    IF NOT pg_try_advisory_lock(job_id) THEN
        RAISE NOTICE 'Job % already running on another server', job_id;
        RETURN;
    END IF;
    
    -- Execute job logic here
    PERFORM execute_nightly_report();
    
    PERFORM pg_advisory_unlock(job_id);
END;
$$ LANGUAGE plpgsql;
```

**Implementation using a locks table (more visible, with TTL):**

```sql
CREATE TABLE distributed_locks (
    lock_name    TEXT         PRIMARY KEY,
    holder_id    TEXT         NOT NULL,  -- application server ID
    acquired_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    expires_at   TIMESTAMPTZ  NOT NULL,
    heartbeat_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Acquire lock (atomic INSERT with conflict handling):
INSERT INTO distributed_locks (lock_name, holder_id, expires_at)
VALUES ('nightly_report', 'server-42', NOW() + INTERVAL '5 minutes')
ON CONFLICT (lock_name) DO UPDATE
    SET holder_id = EXCLUDED.holder_id,
        acquired_at = NOW(),
        expires_at = EXCLUDED.expires_at
    WHERE distributed_locks.expires_at < NOW()  -- only steal expired locks
        OR distributed_locks.holder_id = EXCLUDED.holder_id;  -- or renew our own
-- Check if we got it: SELECT holder_id FROM distributed_locks WHERE lock_name = 'nightly_report' AND holder_id = 'server-42'

-- Heartbeat (renew lock to prove we're still alive):
UPDATE distributed_locks
SET heartbeat_at = NOW(), expires_at = NOW() + INTERVAL '5 minutes'
WHERE lock_name = 'nightly_report' AND holder_id = 'server-42';

-- Release:
DELETE FROM distributed_locks WHERE lock_name = 'nightly_report' AND holder_id = 'server-42';

-- Cleanup expired locks (run periodically):
DELETE FROM distributed_locks WHERE expires_at < NOW();
```

**Comparison:**

| Approach | Visibility | Overhead | TTL support | Best for |
|---|---|---|---|---|
| Advisory lock | Low (pg_locks table) | Very low | Session lifetime | Simple single-process exclusion |
| Locks table | High (queryable) | Low | Yes (explicit) | Long-running jobs, audit trail needed |
| SKIP LOCKED | Implicit | Very low | Queue TTL | Job queues, worker pools |

---

**Q17. A payment system needs to guarantee that a customer is never charged twice for the same order. Design the database schema and transaction logic.**

**Schema:**
```sql
CREATE TABLE orders (
    order_id         UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id      BIGINT       NOT NULL,
    amount_cents     INT          NOT NULL,
    status           TEXT         NOT NULL DEFAULT 'pending',
    created_at       TIMESTAMPTZ  DEFAULT NOW(),
    CONSTRAINT chk_status CHECK (status IN ('pending', 'processing', 'paid', 'failed', 'refunded'))
);

CREATE TABLE payment_attempts (
    attempt_id       UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id         UUID         NOT NULL REFERENCES orders(order_id),
    idempotency_key  UUID         NOT NULL UNIQUE,  -- client-generated, prevents duplicate charges
    amount_cents     INT          NOT NULL,
    gateway_txn_id   TEXT         UNIQUE,           -- external payment gateway reference
    status           TEXT         NOT NULL DEFAULT 'initiated',
    created_at       TIMESTAMPTZ  DEFAULT NOW(),
    completed_at     TIMESTAMPTZ,
    CONSTRAINT chk_status CHECK (status IN ('initiated', 'succeeded', 'failed'))
);

-- Index for idempotency lookup (most critical):
CREATE UNIQUE INDEX ON payment_attempts (idempotency_key);
-- Index for per-order query:
CREATE INDEX ON payment_attempts (order_id, status);
```

**Transaction logic:**
```sql
CREATE OR REPLACE FUNCTION charge_order(
    p_order_id       UUID,
    p_idempotency_key UUID,
    p_amount_cents   INT
) RETURNS payment_attempts AS $$
DECLARE
    v_attempt payment_attempts;
    v_order   orders;
BEGIN
    -- Step 1: Check idempotency (fast path if already processed)
    SELECT * INTO v_attempt FROM payment_attempts WHERE idempotency_key = p_idempotency_key;
    IF FOUND THEN
        RETURN v_attempt;  -- Return previous result, no double charge
    END IF;
    
    -- Step 2: Lock the order row to prevent concurrent payment attempts for the same order
    SELECT * INTO v_order FROM orders WHERE order_id = p_order_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Order not found: %', p_order_id;
    END IF;
    
    IF v_order.status != 'pending' THEN
        RAISE EXCEPTION 'Order % is in status %, cannot charge', p_order_id, v_order.status;
    END IF;
    
    -- Step 3: Create payment attempt record
    INSERT INTO payment_attempts (order_id, idempotency_key, amount_cents, status)
    VALUES (p_order_id, p_idempotency_key, p_amount_cents, 'initiated')
    RETURNING * INTO v_attempt;
    
    -- Step 4: Mark order as processing (prevents another concurrent charge)
    UPDATE orders SET status = 'processing' WHERE order_id = p_order_id;
    
    RETURN v_attempt;
    -- Caller does the actual gateway charge AFTER this function returns
    -- Then calls complete_payment() or fail_payment()
END;
$$ LANGUAGE plpgsql;
```

**Double-charge prevention guarantees:**
1. `UNIQUE` on `idempotency_key` → retried requests return the same result
2. `FOR UPDATE` on order + status check → concurrent calls for same order are serialized; only one gets 'pending' state
3. `status = 'processing'` after first call → subsequent calls fail at status check

---

**Q18. How would you implement a global counter (e.g., total likes on a post) that handles 100,000 increments per second with strong consistency?**

**The problem with naive approach:**
```sql
-- 100K concurrent: UPDATE posts SET like_count = like_count + 1 WHERE post_id = X
-- All 100K transactions serialized on the same row lock → bottleneck
-- Max throughput: ~5K/s on typical hardware
```

**Solution 1: Sharded counters (for strong consistency)**
```sql
-- Shard the counter across N rows; reduces hot spot contention by N×
CREATE TABLE post_like_shards (
    post_id    BIGINT NOT NULL,
    shard_id   INT    NOT NULL,   -- 0 to 9 (10 shards)
    like_count BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (post_id, shard_id)
);

-- Increment: pick random shard to distribute lock contention
UPDATE post_like_shards
SET like_count = like_count + 1
WHERE post_id = 1234 AND shard_id = floor(random() * 10)::INT;

-- Read total: aggregate all shards
SELECT SUM(like_count) FROM post_like_shards WHERE post_id = 1234;

-- Throughput: 10 shards → 50K/s; 100 shards → 500K/s
-- Trade-off: reads require aggregation (not critical for like counts)
```

**Solution 2: Redis atomic increment + async DB sync (for very high rates)**
```redis
INCR post:1234:likes
-- Redis INCR is O(1), single-threaded → handles 1M ops/second easily
-- Strongly consistent within Redis (single-node)
-- Durably consistent: async flush to PostgreSQL every 10 seconds
```

```python
# Flush worker (runs every 10 seconds):
def flush_like_counts():
    for post_id in get_active_posts():
        count = redis.get(f"post:{post_id}:likes")
        if count:
            db.execute("""
                INSERT INTO post_likes_aggregate (post_id, total_likes)
                VALUES (%s, %s)
                ON CONFLICT (post_id) DO UPDATE
                SET total_likes = EXCLUDED.total_likes
            """, post_id, int(count))
```

**Trade-off comparison:**

| Approach | Throughput | Consistency | Complexity |
|---|---|---|---|
| Single row UPDATE | ~5K/s | Strong | Minimal |
| Sharded counters | ~50K–500K/s | Strong | Low |
| Redis + async sync | 1M+/s | Strong in Redis, eventual in DB | Medium |
| Cassandra counter | 100K+/s | Strong within Cassandra | Medium |

For 100K/s: sharded counters (10 shards) are the cleanest solution maintaining full SQL + ACID.

---

**Q19. What are the consistency guarantees of Cassandra's tunable consistency model and how do you choose the right level for a financial use case?**

**Cassandra's consistency model:**
Cassandra replicates each row to RF (replication factor) nodes. Read and write consistency levels specify how many nodes must acknowledge before the operation returns.

```
RF = 3 (three copies on three nodes)

Write CL options:
  ONE:        1 node must acknowledge (fastest, can lose data if that node fails before replication)
  QUORUM:     2 nodes must acknowledge (RF/2 + 1 = 2)
  ALL:        3 nodes must acknowledge (safest, fails if any node is down)
  LOCAL_ONE:  1 node in local datacenter
  LOCAL_QUORUM: quorum within local datacenter (good for multi-DC)

Read CL options:
  ONE:        read from 1 node (may return stale data)
  QUORUM:     read from 2 nodes, return most recent
  ALL:        read from all 3 nodes
```

**Achieving strong consistency:**
```
Formula: write_CL + read_CL > RF → guaranteed to read the latest write

RF=3:
  QUORUM write (2 nodes) + QUORUM read (2 nodes) = 4 > 3 ✓ Strong consistency
  ONE write (1 node) + ONE read (1 node) = 2 ≤ 3 ✗ May read stale

For financial use case:
  Write: LOCAL_QUORUM (2 nodes in local DC — ensures write is durable even if 1 node fails)
  Read:  LOCAL_QUORUM (2 nodes in local DC — ensures latest write is always seen)
  
  This gives: strong consistency within the datacenter
              write still proceeds if one local node is down
              reads never stale within the datacenter
```

**Important caveat for Cassandra + finance:**
```
Cassandra uses LAST-WRITE-WINS conflict resolution based on timestamps
  T1: UPDATE balance = 1000 at timestamp 100
  T2: UPDATE balance = 900 at timestamp 99
  → Cassandra keeps 1000 (higher timestamp wins) regardless of order of arrival
  
For debit/credit: Cassandra is generally NOT the right choice
  → Cannot enforce balance >= 0 constraint
  → LWW can cause silent data loss
  
Better: Use Cassandra for immutable event log (append-only transactions)
        Compute current balance from the event log
        Or: use SERIAL/LIGHTWEIGHT TRANSACTIONS for conditional updates (compare-and-set)
```

---

**Q20. Design a database architecture for a real-time inventory system that must never oversell, must handle 10,000 order placements per second, and must show accurate stock counts to users.**

**Requirements breakdown:**
- No oversell: `qty >= 0` must always hold after any order
- 10,000 orders/second → 10,000 `qty - 1` operations/second
- Accurate stock display: users see real-time (or near-real-time) stock counts

**Architecture:**

**Tier 1: Authoritative inventory (PostgreSQL with sharding)**
```sql
-- Shard by sku_id (natural isolation: each product is independent)
-- Each product's inventory on one shard → no cross-shard contention per product

CREATE TABLE inventory (
    sku_id        TEXT          PRIMARY KEY,
    warehouse_id  INT           NOT NULL,
    qty_available INT           NOT NULL CHECK (qty_available >= 0),  -- DB-enforced no-oversell
    qty_reserved  INT           NOT NULL DEFAULT 0 CHECK (qty_reserved >= 0),
    version       BIGINT        NOT NULL DEFAULT 0,
    updated_at    TIMESTAMPTZ   DEFAULT NOW()
);
-- Optimistic locking for normal updates (no lock on read):
-- Reserve on order: UPDATE ... SET qty_reserved = qty_reserved + 1 WHERE sku_id = X AND qty_available > qty_reserved AND version = V

-- Or pessimistic (for flash sales / always-correct):
SELECT qty_available FROM inventory WHERE sku_id = 'LAPTOP-X1' FOR UPDATE;
-- Then: UPDATE inventory SET qty_available = qty_available - 1 WHERE sku_id = 'LAPTOP-X1';
```

**Tier 2: Redis for read acceleration (stock display)**
```redis
-- Cache current stock count per SKU:
SET inventory:LAPTOP-X1 45 EX 30    -- 30-second TTL
-- On each inventory change: update Redis immediately (write-through)
-- Users see at most 30s stale stock count (acceptable for display)
-- Actual order placement ALWAYS goes to PostgreSQL (authoritative)
```

**Tier 3: Request serialization for extreme hotspots (flash sales)**
```python
# For "last N items" scenarios where many concurrent requests hit the same SKU:
# Use a Redis counter to pre-check before hitting PostgreSQL

def try_reserve(sku_id, qty=1):
    # Fast pre-check in Redis (not authoritative, but filters most failures cheaply)
    remaining = redis.decr(f"inventory:{sku_id}")
    if remaining < 0:
        redis.incr(f"inventory:{sku_id}")  # undo the decrement
        return False, "Out of stock"
    
    # Authoritative write to PostgreSQL
    rows = db.execute("""
        UPDATE inventory SET qty_available = qty_available - %s
        WHERE sku_id = %s AND qty_available >= %s
        RETURNING qty_available
    """, qty, sku_id, qty)
    
    if not rows:
        redis.incr(f"inventory:{sku_id}")  # undo Redis decrement (DB says no)
        return False, "Out of stock"
    
    return True, rows[0].qty_available
```

**Throughput analysis:**
```
10,000 orders/second across 10,000 distinct SKUs:
  → 1 order/second per SKU on average → trivial (no contention per SKU)
  
10,000 orders/second for ONE popular SKU (flash sale):
  → 10,000 concurrent FOR UPDATE attempts on same row
  → PostgreSQL serializes → ~5,000–8,000 successful/second on good hardware
  → If more needed: Redis pre-check filters, queue excess requests
  
Shard by SKU:
  Each SKU's inventory on one shard
  10,000 SKUs × 1 order/sec = trivially distributed across shards
  100 SKUs at 100 orders/sec = each hot SKU on its own shard
```

**Guaranteed no-oversell:**
```
CHECK (qty_available >= 0) at DB level → transaction aborts if it would go negative
UPDATE ... WHERE qty_available >= 1 → update only succeeds if stock exists
FOR UPDATE serializes concurrent decrements on same row
These three together make oversell physically impossible in PostgreSQL
```

---

## Quick Reference

```
Consistency levels:
  Strong (linearizable):  every read sees most recent write, any node
  Eventual:               reads converge over time, may be stale
  Causal:                 you see your own writes, causality preserved
  Monotonic read:         you never see an older version after reading a newer one

Transaction anomalies (from weakest to strongest guarantee needed):
  Dirty read         → READ COMMITTED prevents this
  Non-repeatable read → REPEATABLE READ prevents this
  Phantom read       → SERIALIZABLE prevents this
  Write skew         → SERIALIZABLE (SSI) prevents this

Cross-service consistency patterns:
  2PC:      strong consistency, coordinator risk, high latency
  Saga:     eventual consistency, compensating actions, low latency
  Outbox:   DB write + event publication atomically (via local transaction)
  Idempotency key: safe retries without duplicate effects

Locking strategies:
  Pessimistic (FOR UPDATE): good when conflicts are frequent
  Optimistic (version check): good when conflicts are rare
  Advisory locks:            good for distributed job coordination

No-oversell checklist:
  1. CHECK (qty >= 0) constraint on DB
  2. UPDATE ... WHERE qty >= 1 (conditional update)
  3. FOR UPDATE or optimistic version check to serialize concurrent writes
  4. Redis pre-check as fast filter for clearly-failed requests
```
