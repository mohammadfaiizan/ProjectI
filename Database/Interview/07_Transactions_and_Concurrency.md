# Transactions and Concurrency — Interview Questions

## Easy (Q1–Q8)

---

**Q1. What is a transaction in SQL, and why is it important?**

A transaction is a logical unit of work that groups one or more SQL statements together so they either all succeed or all fail — there is no partial success.

```sql
BEGIN;
UPDATE accounts SET balance = balance - 500 WHERE account_id = 1;
UPDATE accounts SET balance = balance + 500 WHERE account_id = 2;
COMMIT;  -- both updates committed together
```

Without transactions, a crash between the two UPDATEs would leave one account debited and the other not credited — money would vanish.

---

**Q2. What are the ACID properties? Explain each briefly.**

| Property | Meaning |
|---|---|
| **Atomicity** | All statements in a transaction commit or all roll back — no partial result |
| **Consistency** | The database moves from one valid state to another; all constraints remain satisfied |
| **Isolation** | Concurrent transactions behave as if they ran serially; one transaction's intermediate state is not visible to others |
| **Durability** | Once committed, data survives system crashes (written to durable storage / WAL) |

---

**Q3. What is the difference between COMMIT and ROLLBACK?**

- `COMMIT` makes all changes in the current transaction permanent and visible to other transactions.
- `ROLLBACK` undoes all changes made since the transaction began (or since the last SAVEPOINT).

```sql
BEGIN;
DELETE FROM orders WHERE order_date < '2020-01-01';
-- Changed mind:
ROLLBACK;  -- orders are restored
```

---

**Q4. What is autocommit, and how does it affect transactions?**

In autocommit mode each statement is automatically wrapped in its own transaction and committed immediately on success. This is the default in MySQL and many other databases.

```sql
-- MySQL: autocommit is ON by default
SET autocommit = 0;   -- disable to write explicit transactions
SET autocommit = 1;   -- re-enable

-- PostgreSQL: autocommit is always on unless you open a transaction block
BEGIN;
-- autocommit suspended until COMMIT/ROLLBACK
```

When autocommit is on, a bare `UPDATE` commits itself. Forgetting to open a `BEGIN` block is a common mistake.

---

**Q5. What is a SAVEPOINT and when would you use it?**

A SAVEPOINT marks a point within a transaction to which you can partially roll back without aborting the entire transaction.

```sql
BEGIN;
INSERT INTO orders (customer_id, total) VALUES (1, 100);
SAVEPOINT after_order;

INSERT INTO order_items (order_id, product_id) VALUES (999, 5);  -- bad order_id
-- Only roll back the bad insert:
ROLLBACK TO SAVEPOINT after_order;

-- The order INSERT is still pending
COMMIT;
```

Use case: batch processing where one bad record should not abort the whole batch.

---

**Q6. What is a dirty read?**

A dirty read occurs when transaction A reads data that transaction B has modified but not yet committed. If B rolls back, A has read data that never officially existed.

```
T1: UPDATE products SET price = 999 WHERE id = 1;  -- not committed
T2: SELECT price FROM products WHERE id = 1;       -- reads 999 (dirty)
T1: ROLLBACK;                                      -- price reverts to original
T2 has used a price that never existed
```

Dirty reads are only possible at the READ UNCOMMITTED isolation level.

---

**Q7. What is the difference between optimistic and pessimistic locking?**

| Aspect | Pessimistic | Optimistic |
|---|---|---|
| Assumption | Conflicts are likely | Conflicts are rare |
| Mechanism | Lock row on read (`SELECT FOR UPDATE`) | Read freely; check at write time |
| Concurrency | Lower (locks block others) | Higher (no locks during read) |
| Failure mode | Waits / deadlock | Write fails; caller must retry |
| Best for | High-contention rows | Low-contention, read-heavy |

```sql
-- Pessimistic
SELECT * FROM seats WHERE seat_id = 42 FOR UPDATE;
UPDATE seats SET status = 'booked' WHERE seat_id = 42;

-- Optimistic (version column)
SELECT id, version, price FROM products WHERE id = 1;  -- version = 7
UPDATE products SET price = 50, version = 8
WHERE id = 1 AND version = 7;  -- 0 rows if someone else updated first
```

---

**Q8. What does SELECT FOR UPDATE do?**

It acquires an exclusive row-level lock on the selected rows, preventing other transactions from locking or modifying them until the current transaction commits or rolls back.

```sql
BEGIN;
SELECT * FROM inventory WHERE product_id = 10 FOR UPDATE;
-- Other sessions trying SELECT FOR UPDATE on the same row will block here
UPDATE inventory SET qty = qty - 1 WHERE product_id = 10;
COMMIT;
```

PostgreSQL extensions:
- `FOR UPDATE NOWAIT` — fail immediately if lock cannot be acquired
- `FOR UPDATE SKIP LOCKED` — skip already-locked rows (ideal for job queues)

---

## Medium (Q9–Q15)

---

**Q9. What are the four SQL transaction isolation levels and what anomalies does each prevent?**

| Isolation Level | Dirty Read | Non-Repeatable Read | Phantom Read |
|---|---|---|---|
| READ UNCOMMITTED | Possible | Possible | Possible |
| READ COMMITTED | Prevented | Possible | Possible |
| REPEATABLE READ | Prevented | Prevented | Possible (prevented in MySQL InnoDB) |
| SERIALIZABLE | Prevented | Prevented | Prevented |

**Definitions:**
- **Dirty read** — reading uncommitted data from another transaction
- **Non-repeatable read** — re-reading a row within a transaction yields different values (another transaction updated and committed between the two reads)
- **Phantom read** — re-running a range query yields different rows (another transaction inserted/deleted rows in that range between the two reads)

```sql
-- Set isolation level (MySQL)
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
BEGIN;
...
```

---

**Q10. Explain a non-repeatable read with a concrete example.**

```
-- Session A (REPEATABLE READ or lower)
BEGIN;
SELECT salary FROM employees WHERE id = 5;
-- Returns 60000

-- Session B (concurrently)
BEGIN;
UPDATE employees SET salary = 70000 WHERE id = 5;
COMMIT;

-- Session A reads again
SELECT salary FROM employees WHERE id = 5;
-- Returns 70000 at READ COMMITTED — the value changed mid-transaction
-- Returns 60000 at REPEATABLE READ — snapshot is frozen
COMMIT;
```

At READ COMMITTED, session A sees the committed change from B on its second read — this is a non-repeatable read. REPEATABLE READ prevents it by taking a snapshot at the start of the transaction.

---

**Q11. What is a phantom read and how does SERIALIZABLE prevent it?**

A phantom read happens when a transaction re-executes a range query and finds new rows have appeared (or disappeared) because another committed transaction inserted or deleted matching rows.

```
-- Session A (REPEATABLE READ)
BEGIN;
SELECT COUNT(*) FROM orders WHERE amount > 1000;  -- returns 5

-- Session B
INSERT INTO orders (customer_id, amount) VALUES (7, 1500);
COMMIT;

-- Session A
SELECT COUNT(*) FROM orders WHERE amount > 1000;  -- returns 6 (phantom)
COMMIT;
```

At SERIALIZABLE, the database detects this conflict (predicate lock or SSI — Serializable Snapshot Isolation in PostgreSQL) and forces the transactions to behave as if they ran one after the other, aborting one of them if necessary.

---

**Q12. What is a deadlock and how can it be prevented?**

A deadlock is a cycle where two or more transactions each hold a lock the other needs, so none can proceed.

```
T1: LOCK row A → waiting for row B
T2: LOCK row B → waiting for row A
-- Neither can proceed
```

**Detection:** Databases detect cycles in the lock wait graph and automatically abort one transaction (the "victim") with an error.

**Prevention strategies:**
1. **Consistent lock ordering** — always acquire locks in the same order (lock row with lower ID first)
2. **Short transactions** — minimize the window between acquiring locks
3. **Lock at once** — use `SELECT ... FOR UPDATE` of all needed rows in a single statement
4. **Retry logic** — application retries the aborted transaction
5. **Lower isolation level** — use READ COMMITTED where full isolation is not needed

```sql
-- Deadlock-prone: each transaction locks in opposite order
-- T1: UPDATE accounts WHERE id = 1; then UPDATE WHERE id = 2;
-- T2: UPDATE accounts WHERE id = 2; then UPDATE WHERE id = 1;

-- Safe: always lock lower id first
BEGIN;
SELECT * FROM accounts WHERE id IN (1, 2) ORDER BY id FOR UPDATE;
-- Both rows locked at once in consistent order
```

---

**Q13. What is MVCC (Multi-Version Concurrency Control)?**

MVCC allows readers to see a consistent snapshot of data without blocking writers and writers to proceed without blocking readers. Instead of locking rows for reads, the database keeps multiple versions of each row.

**How it works (PostgreSQL):**
- Each row has `xmin` (transaction that created it) and `xmax` (transaction that deleted/updated it)
- A transaction sees row versions where `xmin` ≤ its snapshot and `xmax` is either 0 or from a transaction that was not committed at snapshot time
- Writers create a new row version rather than overwriting

```sql
-- PostgreSQL: view row version info
SELECT xmin, xmax, id, salary FROM employees WHERE id = 5;
-- xmin = transaction that inserted the row
-- xmax = 0 means the row is current (not deleted/updated)
```

**Consequences:**
- Dead tuples accumulate and must be reclaimed by VACUUM
- Read queries never block on writes and vice versa
- Long-running transactions hold old snapshots open, preventing VACUUM from reclaiming space

---

**Q14. Explain the difference between row-level and table-level locking.**

| Lock Level | Granularity | Concurrency | Overhead |
|---|---|---|---|
| Table-level | Entire table | Low (only one writer at a time) | Low |
| Page-level | Database page (~8KB) | Medium | Medium |
| Row-level | Individual rows | High (many concurrent writers) | Higher |

```sql
-- Table-level lock (MySQL)
LOCK TABLES employees WRITE;  -- exclusive, no other reads or writes
LOCK TABLES employees READ;   -- shared, allows reads, blocks writes
UNLOCK TABLES;

-- PostgreSQL explicit table lock
LOCK TABLE employees IN EXCLUSIVE MODE;
LOCK TABLE employees IN SHARE MODE;

-- Row-level lock (all databases via DML or SELECT FOR UPDATE)
SELECT * FROM employees WHERE id = 5 FOR UPDATE;  -- row lock only
```

Row-level locking (used by PostgreSQL InnoDB, SQL Server) allows many transactions to work on different rows simultaneously. MyISAM used table-level locks and had poor write concurrency.

---

**Q15. What happens when you use NOT IN with a subquery that returns a NULL?**

The result is always empty — no rows are returned — even rows that should logically match.

```sql
-- employees NOT IN the manager_id list
SELECT name FROM employees
WHERE employee_id NOT IN (SELECT manager_id FROM employees);
-- Returns 0 rows if ANY manager_id is NULL

-- Why: NOT IN is equivalent to:
-- employee_id <> v1 AND employee_id <> v2 AND ... AND employee_id <> NULL
-- Anything compared to NULL yields UNKNOWN, not TRUE
-- AND chain with UNKNOWN = UNKNOWN (not TRUE), so the row is excluded

-- Safe alternative: NOT EXISTS
SELECT e.name FROM employees e
WHERE NOT EXISTS (
    SELECT 1 FROM employees m
    WHERE m.manager_id = e.employee_id
);

-- Or filter NULLs explicitly:
WHERE employee_id NOT IN (
    SELECT manager_id FROM employees WHERE manager_id IS NOT NULL
);
```

---

## Hard (Q16–Q20)

---

**Q16. How does two-phase locking (2PL) work, and what is its relationship to serializability?**

Two-phase locking is a concurrency control protocol that guarantees conflict-serializability by splitting a transaction into two phases:

1. **Growing phase** — the transaction acquires locks and never releases any
2. **Shrinking phase** — the transaction releases locks and never acquires new ones

```
T1: LOCK(A) → LOCK(B) → LOCK(C) | → UNLOCK(A) → UNLOCK(B) → UNLOCK(C)
           Growing phase          |       Shrinking phase
```

**Variants:**
- **Strict 2PL** — all exclusive (write) locks held until commit/rollback. Prevents cascading rollbacks. Most databases use this.
- **Rigorous 2PL** — all locks (read and write) held until commit. Simplifies recovery.

**Relationship to serializability:** Any schedule produced by 2PL is conflict-serializable. However, 2PL can cause deadlocks (two transactions in their growing phase waiting on each other). This is why databases need deadlock detection.

**PostgreSQL uses SSI (Serializable Snapshot Isolation)** for its SERIALIZABLE level — not 2PL — which avoids read locks while still detecting dangerous read-write cycles.

---

**Q17. Design a job queue system using SELECT FOR UPDATE SKIP LOCKED. Explain why SKIP LOCKED is critical.**

```sql
-- Table definition
CREATE TABLE job_queue (
    id          BIGSERIAL PRIMARY KEY,
    payload     JSONB        NOT NULL,
    status      TEXT         NOT NULL DEFAULT 'pending',  -- pending, processing, done, failed
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    started_at  TIMESTAMPTZ,
    worker_id   TEXT
);

CREATE INDEX idx_job_queue_pending ON job_queue (status, created_at)
WHERE status = 'pending';

-- Worker process: claim next available job atomically
BEGIN;
SELECT id, payload
FROM   job_queue
WHERE  status = 'pending'
ORDER  BY created_at
LIMIT  1
FOR UPDATE SKIP LOCKED;         -- <-- key clause

-- If no row returned: no work available, exit
-- If row returned: mark it as processing
UPDATE job_queue
SET    status     = 'processing',
       started_at = NOW(),
       worker_id  = 'worker-42'
WHERE  id = :claimed_id;

COMMIT;
-- Process job here, then:
UPDATE job_queue SET status = 'done' WHERE id = :claimed_id;
```

**Why SKIP LOCKED is critical:**
- Without it: multiple workers all `SELECT FOR UPDATE` the same row → only one gets through; others wait (serialized, not parallel)
- With `SKIP LOCKED`: each worker skips rows locked by other workers and picks the next available → true parallel processing with no contention
- `NOWAIT` is an alternative that fails immediately instead of waiting, but SKIP LOCKED is better for queues

---

**Q18. What is write skew, and why is it not prevented by REPEATABLE READ?**

Write skew is a subtle anomaly where two transactions both read the same set of rows, make a decision based on that read, and write to different rows in a way that violates an invariant — but neither transaction actually modified any row the other transaction read, so no write-write conflict is detected.

**Classic example: on-call doctors**

```sql
-- Business rule: at least one doctor must be on call at all times
-- Currently: doctor A and doctor B are both on call

-- Transaction 1 (doctor A going off call)
BEGIN;
SELECT COUNT(*) FROM on_call WHERE on_duty = TRUE;  -- sees 2, OK to proceed
UPDATE on_call SET on_duty = FALSE WHERE doctor_id = 'A';
COMMIT;

-- Transaction 2 (doctor B going off call, runs concurrently)
BEGIN;
SELECT COUNT(*) FROM on_call WHERE on_duty = TRUE;  -- also sees 2, OK to proceed
UPDATE on_call SET on_duty = FALSE WHERE doctor_id = 'B';
COMMIT;

-- Result: both doctors are off call — invariant violated
-- Neither transaction modified what the other read — no conflict detected at REPEATABLE READ
```

**Why REPEATABLE READ fails:** It prevents non-repeatable reads on rows actually read, but write skew involves writing to different rows than those read. The anti-dependency (T1 reads what T2's write would change) is not captured.

**Fix:** Use SERIALIZABLE isolation (SSI detects this cycle) or use `SELECT FOR UPDATE` to lock all rows that inform the decision:

```sql
SELECT COUNT(*) FROM on_call WHERE on_duty = TRUE FOR UPDATE;
-- Now both transactions try to lock the same rows → one waits → no skew
```

---

**Q19. Explain how PostgreSQL's VACUUM relates to MVCC and long-running transactions.**

Because MVCC creates a new row version on every UPDATE and DELETE (old versions are not overwritten in-place), dead tuples accumulate on disk. VACUUM reclaims this space.

```
Normal row: xmin=100, xmax=0    (live, created by txn 100)
After UPDATE: 
  Old version: xmin=100, xmax=200  (dead, superseded by txn 200)
  New version: xmin=200, xmax=0    (live)
```

**VACUUM process:**
1. Scans table pages for dead tuples (xmax is a committed transaction)
2. Marks dead tuples as free space (does not return to OS)
3. Updates the Free Space Map (FSM) and Visibility Map (VM)
4. Advances `relfrozenxid` to prevent XID wraparound

**Long-running transactions block VACUUM:**
- VACUUM cannot remove a dead tuple if any active transaction has a snapshot older than the dead tuple's `xmax`
- A 30-minute reporting query prevents VACUUM from reclaiming any tuples modified during those 30 minutes
- Check with: `SELECT pid, now() - xact_start, query FROM pg_stat_activity WHERE state = 'active' ORDER BY xact_start;`

**VACUUM vs VACUUM FULL:**
```sql
VACUUM employees;            -- reclaims dead space without table lock, online
VACUUM FULL employees;       -- rewrites entire table, reclaims space to OS, full table lock
VACUUM ANALYZE employees;    -- vacuum + update planner statistics
```

**autovacuum** runs automatically in the background. Tune `autovacuum_vacuum_scale_factor` and `autovacuum_vacuum_threshold` for large frequently-updated tables.

---

**Q20. How would you implement a distributed transaction across two separate databases? What are the trade-offs?**

**Two-Phase Commit (2PC) — the classical solution:**

```
Phase 1 — Prepare:
  Coordinator → Participant A: "Prepare to commit txn X"
  Coordinator → Participant B: "Prepare to commit txn X"
  Participants write to durable log (WAL), reply PREPARED or ABORT

Phase 2 — Commit (only if all replied PREPARED):
  Coordinator → Participant A: "Commit txn X"
  Coordinator → Participant B: "Commit txn X"
  If any replied ABORT → Coordinator sends ROLLBACK to all
```

```sql
-- PostgreSQL prepared transactions
BEGIN;
UPDATE db1.accounts SET balance = balance - 100 WHERE id = 1;
PREPARE TRANSACTION 'txn_transfer_001';  -- phase 1

-- Coordinator checks all participants, then:
COMMIT PREPARED 'txn_transfer_001';     -- phase 2
-- Or:
ROLLBACK PREPARED 'txn_transfer_001';
```

**Trade-offs of 2PC:**

| Concern | Detail |
|---|---|
| Availability | If coordinator crashes between prepare and commit, participants are blocked (holding locks) indefinitely — this is the "in-doubt transaction" problem |
| Latency | Two round trips of network calls instead of one |
| Complexity | Requires all databases to support XA/2PC protocol |
| Blocking | Participants hold locks until coordinator sends decision |

**Modern alternatives:**

1. **Saga pattern** — break distributed transaction into local transactions with compensating actions on failure
   ```
   Order Service: create order → Payment Service: charge card → Inventory Service: deduct stock
   On failure: Inventory Service: restore stock → Payment Service: refund → Order Service: cancel order
   ```

2. **Outbox pattern** — write event + local state change in single local ACID transaction; a relay process publishes the event
   ```sql
   BEGIN;
   INSERT INTO orders ...;
   INSERT INTO outbox (event_type, payload) VALUES ('ORDER_CREATED', '{"order_id":1}');
   COMMIT;
   -- Separate relay reads outbox and publishes to message broker
   ```

3. **Eventual consistency with idempotent operations** — accept that systems will be temporarily inconsistent; design operations to be safe to replay.

Best practice: avoid distributed transactions whenever possible by co-locating related data or accepting eventual consistency.

---

## Quick Reference

```sql
-- Transaction control
BEGIN / START TRANSACTION;
COMMIT;
ROLLBACK;
SAVEPOINT sp1;
ROLLBACK TO SAVEPOINT sp1;
RELEASE SAVEPOINT sp1;

-- Isolation level
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- Locking
SELECT ... FOR UPDATE;
SELECT ... FOR UPDATE NOWAIT;
SELECT ... FOR UPDATE SKIP LOCKED;
SELECT ... FOR SHARE;

-- PostgreSQL prepared transactions
PREPARE TRANSACTION 'txn_id';
COMMIT PREPARED 'txn_id';
ROLLBACK PREPARED 'txn_id';

-- Vacuum
VACUUM table_name;
VACUUM FULL table_name;
VACUUM ANALYZE table_name;

-- View active transactions
SELECT pid, xact_start, state, query FROM pg_stat_activity;
-- View locks
SELECT * FROM pg_locks JOIN pg_stat_activity USING (pid);
```
