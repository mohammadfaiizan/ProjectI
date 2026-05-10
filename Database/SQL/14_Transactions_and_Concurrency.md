# Transactions and Concurrency

## Table of Contents
1. [What is a Transaction?](#1-what-is-a-transaction)
2. [ACID Properties](#2-acid-properties)
3. [Transaction Control Commands](#3-transaction-control-commands)
4. [Isolation Levels](#4-isolation-levels)
5. [Concurrency Problems](#5-concurrency-problems)
6. [Locking](#6-locking)
7. [Deadlocks](#7-deadlocks)
8. [Optimistic vs Pessimistic Locking](#8-optimistic-vs-pessimistic-locking)
9. [MVCC (Multi-Version Concurrency Control)](#9-mvcc-multi-version-concurrency-control)
10. [Practical Transaction Patterns](#10-practical-transaction-patterns)

---

## 1. What is a Transaction?

A transaction is a sequence of SQL operations that are executed as a single logical unit of work. Either ALL operations succeed (commit) or ALL fail (rollback).

### Example: Bank Transfer
```sql
-- Without transaction: dangerous!
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
-- ← What if crash happens here?
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

-- With transaction: safe!
BEGIN;
    UPDATE accounts SET balance = balance - 100 WHERE id = 1;
    UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
-- Either both updates apply, or neither does
```

---

## 2. ACID Properties

### A — Atomicity
All operations in a transaction are treated as a single unit. Either all succeed or all fail.
```sql
BEGIN;
    INSERT INTO orders (customer_id, total) VALUES (1, 500);    -- Step 1
    UPDATE inventory SET stock = stock - 1 WHERE product_id = 42; -- Step 2
    -- If step 2 fails, step 1 is also rolled back
COMMIT;
```

### C — Consistency
A transaction brings the database from one valid state to another. All constraints, rules, and cascades are enforced.
```sql
-- The DB enforces constraints within the transaction
BEGIN;
    INSERT INTO order_items (order_id, product_id, quantity, price)
    VALUES (1, 42, 2, 29.99);
    -- CHECK constraint ensures quantity > 0
    -- FOREIGN KEY ensures order_id exists
COMMIT;
```

### I — Isolation
Concurrent transactions are isolated from each other. Intermediate states are not visible to other transactions (to varying degrees, based on isolation level).

### D — Durability
Once committed, a transaction's changes are permanent, even after system failure. Data is written to disk and logged.

---

## 3. Transaction Control Commands

### BEGIN / START TRANSACTION
```sql
-- MySQL
START TRANSACTION;
BEGIN;

-- PostgreSQL
BEGIN;
BEGIN TRANSACTION;
BEGIN WORK;

-- SQL Server
BEGIN TRANSACTION;
BEGIN TRAN;
```

### COMMIT
```sql
COMMIT;           -- MySQL / PostgreSQL
COMMIT WORK;      -- Standard SQL
COMMIT TRANSACTION; -- SQL Server
```

### ROLLBACK
```sql
ROLLBACK;           -- Undo all changes since BEGIN
ROLLBACK WORK;
ROLLBACK TRANSACTION; -- SQL Server
```

### SAVEPOINT
Savepoints allow partial rollback within a transaction.
```sql
BEGIN;

INSERT INTO orders (customer_id) VALUES (1);
SAVEPOINT sp1;

INSERT INTO order_items (order_id, product_id) VALUES (1, 42);
SAVEPOINT sp2;

-- Oops, wrong product
ROLLBACK TO SAVEPOINT sp2;  -- Only undo the order_items insert

INSERT INTO order_items (order_id, product_id) VALUES (1, 99);

COMMIT;

-- Drop a savepoint (release its resources)
RELEASE SAVEPOINT sp1;
```

### SET AUTOCOMMIT
```sql
-- MySQL: by default, every statement auto-commits
-- Disable for manual transactions:
SET AUTOCOMMIT = 0;
-- ... your statements ...
COMMIT;

-- Re-enable
SET AUTOCOMMIT = 1;

-- PostgreSQL: AUTOCOMMIT is off by default in psql interactive mode
-- Every statement needs explicit BEGIN/COMMIT

-- Check current autocommit setting
SHOW AUTOCOMMIT;      -- MySQL
SHOW autocommit;      -- PostgreSQL (using \echo :AUTOCOMMIT)
```

### Read-Only Transactions
```sql
-- PostgreSQL
BEGIN TRANSACTION READ ONLY;
-- ... SELECT statements only ...
COMMIT;

-- MySQL
SET TRANSACTION READ ONLY;
START TRANSACTION;
```

---

## 4. Isolation Levels

SQL defines 4 isolation levels that control what effects of concurrent transactions are visible.

### SET ISOLATION LEVEL
```sql
-- MySQL
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;   -- Default for MySQL/InnoDB
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- PostgreSQL
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;    -- Default for PostgreSQL
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- For the session (MySQL)
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- For all subsequent transactions
SET GLOBAL TRANSACTION ISOLATION LEVEL READ COMMITTED;
```

### The Four Isolation Levels

#### 1. READ UNCOMMITTED (Lowest)
Can read data from uncommitted transactions ("dirty reads").
```sql
-- Transaction A
BEGIN;
UPDATE employees SET salary = 999999 WHERE id = 1;
-- No commit yet...

-- Transaction B (READ UNCOMMITTED)
BEGIN;
SELECT salary FROM employees WHERE id = 1;
-- Returns 999999 — a dirty read!
-- If Transaction A rolls back, this was wrong data

-- Almost never used in practice
```

#### 2. READ COMMITTED
Only reads committed data. Prevents dirty reads. Default in PostgreSQL, Oracle, SQL Server.
```sql
-- Transaction A
BEGIN;
SELECT salary FROM employees WHERE id = 1;  -- Returns 75000

-- Transaction B commits a change
-- (another session)
UPDATE employees SET salary = 80000 WHERE id = 1;
COMMIT;

-- Transaction A (READ COMMITTED)
SELECT salary FROM employees WHERE id = 1;  -- Returns 80000 now!
-- This is a "non-repeatable read" — same query, different result
```

#### 3. REPEATABLE READ
Same data is returned for repeated reads within a transaction. Default in MySQL/InnoDB.
```sql
-- Transaction A (REPEATABLE READ)
BEGIN;
SELECT salary FROM employees WHERE id = 1;  -- Returns 75000

-- Transaction B commits a change
UPDATE employees SET salary = 80000 WHERE id = 1; COMMIT;

-- Transaction A reads again
SELECT salary FROM employees WHERE id = 1;  -- Still 75000! (snapshot)

-- But phantom reads can still occur (new rows inserted by other transactions)
SELECT COUNT(*) FROM employees WHERE dept_id = 10;  -- Returns 5
-- Transaction B inserts a new employee in dept_id 10 and commits
SELECT COUNT(*) FROM employees WHERE dept_id = 10;  -- May return 6 (phantom)
```

#### 4. SERIALIZABLE (Highest)
Transactions are executed as if they were serial (one at a time). No concurrency anomalies.
```sql
-- All three anomalies (dirty read, non-repeatable read, phantom read) are prevented
-- Highest protection, lowest concurrency
-- Uses range locks or predicate locks

SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
BEGIN;
SELECT COUNT(*) FROM employees WHERE dept_id = 10;  -- Returns 5
-- Another transaction cannot insert/delete rows matching dept_id = 10
-- until this transaction completes
```

### Isolation Level Comparison

| Level | Dirty Read | Non-Repeatable Read | Phantom Read |
|-------|-----------|---------------------|--------------|
| READ UNCOMMITTED | Possible | Possible | Possible |
| READ COMMITTED | Prevented | Possible | Possible |
| REPEATABLE READ | Prevented | Prevented | Possible* |
| SERIALIZABLE | Prevented | Prevented | Prevented |

*MySQL REPEATABLE READ prevents phantom reads via gap locks

---

## 5. Concurrency Problems

### Dirty Read
Reading uncommitted data from another transaction.
```
T1: UPDATE salary = 99999 (not committed)
T2: SELECT salary → 99999 (dirty read)
T1: ROLLBACK
T2 has wrong data!
```

### Non-Repeatable Read
Same row returns different data when read twice within a transaction.
```
T1: SELECT salary → 75000
T2: UPDATE salary = 80000; COMMIT
T1: SELECT salary → 80000 (different!)
```

### Phantom Read
A subsequent query returns rows that were not there before (due to INSERT by another transaction).
```
T1: SELECT COUNT(*) WHERE dept=10 → 5
T2: INSERT INTO employees (dept_id=10); COMMIT
T1: SELECT COUNT(*) WHERE dept=10 → 6 (phantom row!)
```

### Lost Update
Two transactions read the same data and both update it, losing one update.
```
T1: SELECT balance → 100
T2: SELECT balance → 100
T1: UPDATE balance = 100 - 30 = 70; COMMIT
T2: UPDATE balance = 100 + 50 = 150; COMMIT  ← T1's debit is lost!
```

### Prevention
```sql
-- Prevent lost update with SELECT FOR UPDATE
BEGIN;
SELECT balance FROM accounts WHERE id = 1 FOR UPDATE;  -- Locks the row
-- T2 blocks here until T1 commits
UPDATE accounts SET balance = balance - 30 WHERE id = 1;
COMMIT;
```

---

## 6. Locking

### SELECT FOR UPDATE (Exclusive Lock)
Locks selected rows, preventing other transactions from modifying them.
```sql
-- PostgreSQL / MySQL
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;
-- Row is locked; other transactions that try to SELECT FOR UPDATE will block
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;

-- Skip locked rows (PostgreSQL / MySQL 8.0+)
SELECT * FROM job_queue WHERE status = 'pending' FOR UPDATE SKIP LOCKED LIMIT 1;
-- Gets a row that isn't locked by another transaction
-- Useful for task queues
```

### SELECT FOR SHARE (Shared Lock)
Allows other transactions to read but not modify the locked rows.
```sql
-- PostgreSQL
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR SHARE;
-- Other transactions can read but cannot update
COMMIT;

-- MySQL equivalent
SELECT * FROM accounts WHERE id = 1 LOCK IN SHARE MODE;
```

### NOWAIT (Fail instead of block)
```sql
-- PostgreSQL / MySQL 8.0+
SELECT * FROM accounts WHERE id = 1 FOR UPDATE NOWAIT;
-- If row is locked, immediately fail with error instead of waiting
```

### Table-Level Locks
```sql
-- MySQL
LOCK TABLES employees READ;          -- Shared lock
LOCK TABLES employees WRITE;         -- Exclusive lock
LOCK TABLES employees READ, orders WRITE;
UNLOCK TABLES;

-- PostgreSQL
BEGIN;
LOCK TABLE employees IN SHARE MODE;
LOCK TABLE employees IN EXCLUSIVE MODE;
LOCK TABLE employees IN ACCESS EXCLUSIVE MODE;  -- Strongest; blocks all
COMMIT;  -- Lock released at commit
```

---

## 7. Deadlocks

A deadlock occurs when two transactions each hold a lock the other needs.

### Deadlock Example
```
T1: LOCK row A → waiting for row B
T2: LOCK row B → waiting for row A
Both transactions wait forever!
```

### Database Response
Most databases detect deadlocks automatically and roll back one transaction.
```
ERROR:  deadlock detected
DETAIL: Process 12345 waits for ShareLock on transaction 67890
        Process 67890 waits for ShareLock on transaction 12345
HINT:   See server log for query details.
```

### Prevention Strategies
```sql
-- 1. Access resources in consistent order
-- Bad (can deadlock):
-- T1: Lock users table, then orders table
-- T2: Lock orders table, then users table

-- Good: always lock users → orders (consistent order)
BEGIN;
SELECT * FROM users WHERE id = 1 FOR UPDATE;
SELECT * FROM orders WHERE user_id = 1 FOR UPDATE;
COMMIT;

-- 2. Keep transactions short
-- Acquire locks as late as possible, release as early as possible

-- 3. Use SELECT FOR UPDATE SKIP LOCKED for queue-like workloads

-- 4. Retry on deadlock (application level)
-- Catch ERROR 1213 (MySQL) or error code 40P01 (PostgreSQL)
-- and retry the transaction
```

---

## 8. Optimistic vs Pessimistic Locking

### Pessimistic Locking
Assume conflicts will happen; lock rows before reading.
```sql
-- Pessimistic: lock the row immediately
BEGIN;
SELECT balance FROM accounts WHERE id = 1 FOR UPDATE;  -- Lock acquired
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;
-- Good when: high contention, short transactions
```

### Optimistic Locking
Assume conflicts are rare; use version numbers to detect concurrent modifications.
```sql
-- Table has a version column
CREATE TABLE products (
    id      INT PRIMARY KEY,
    name    VARCHAR(100),
    price   DECIMAL(10,2),
    version INT DEFAULT 0
);

-- Read the row
SELECT id, name, price, version FROM products WHERE id = 42;
-- Returns: id=42, name='Widget', price=9.99, version=5

-- Update with version check
UPDATE products
SET price = 12.99, version = version + 1
WHERE id = 42 AND version = 5;  -- Only updates if version hasn't changed

-- Check affected rows
-- If 0 rows updated → someone else modified it → retry or notify user
-- If 1 row updated → success
```

---

## 9. MVCC (Multi-Version Concurrency Control)

MVCC allows readers and writers to not block each other by keeping multiple versions of rows.

### How It Works
```
Time:  T1-start    T2-start    T1-commit   T2-reads
Row A: [v1: 100]   [v1: 100]   [v2: 200]   T2 still sees v1 (100)
                               [v2 added]   (snapshot at T2-start)
```

### PostgreSQL MVCC
```sql
-- Each row has:
-- xmin: transaction ID that created this row version
-- xmax: transaction ID that deleted/updated this row (0 if current)

-- See row versions (advanced)
SELECT xmin, xmax, id, salary FROM employees WHERE id = 1;

-- Transactions see rows where xmin <= their transaction ID AND xmax = 0 (or xmax > their ID)
```

### Vacuum (PostgreSQL) — Clean up old row versions
```sql
-- Dead rows from MVCC accumulate; VACUUM reclaims space
VACUUM employees;                -- Clean dead rows (keeps space for reuse)
VACUUM FULL employees;           -- Compact table (requires exclusive lock)
VACUUM ANALYZE employees;        -- Clean + update statistics
ANALYZE employees;               -- Just update statistics

-- Autovacuum runs automatically; manual vacuum for large batch updates
```

---

## 10. Practical Transaction Patterns

### Pattern 1: Money Transfer
```sql
CREATE PROCEDURE transfer_money(
    p_from INT, p_to INT, p_amount DECIMAL(10,2)
)
BEGIN
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        RESIGNAL;
    END;

    START TRANSACTION;

    -- Check sufficient balance
    IF (SELECT balance FROM accounts WHERE id = p_from FOR UPDATE) < p_amount THEN
        ROLLBACK;
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Insufficient funds';
    END IF;

    UPDATE accounts SET balance = balance - p_amount WHERE id = p_from;
    UPDATE accounts SET balance = balance + p_amount WHERE id = p_to;

    COMMIT;
END;
```

### Pattern 2: Batch Processing with Savepoints
```sql
BEGIN;

SAVEPOINT batch_start;

DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN SELECT id FROM pending_jobs LOOP
        BEGIN
            -- Process each job
            UPDATE jobs SET status = 'done' WHERE id = r.id;
            -- ... complex logic ...
        EXCEPTION
            WHEN OTHERS THEN
                -- Only rollback this one job, not the whole batch
                RAISE NOTICE 'Job % failed: %', r.id, SQLERRM;
        END;
    END LOOP;
END;
$$;

COMMIT;
```

### Pattern 3: Conditional Commit
```sql
BEGIN;

UPDATE inventory SET stock = stock - 1
WHERE product_id = 42 AND stock > 0;

-- Check if update succeeded (stock was available)
SELECT ROW_COUNT() INTO @affected;  -- MySQL

IF @affected > 0 THEN
    INSERT INTO orders (product_id, customer_id) VALUES (42, 1);
    COMMIT;
ELSE
    ROLLBACK;
    SELECT 'Out of stock' AS message;
END IF;
```

### Pattern 4: Long-Running with Progress
```sql
-- Process in chunks to avoid long transactions and reduce lock contention
DELIMITER $$
CREATE PROCEDURE batch_salary_update()
BEGIN
    DECLARE v_batch_size INT DEFAULT 1000;
    DECLARE v_offset INT DEFAULT 0;
    DECLARE v_total INT;

    SELECT COUNT(*) INTO v_total FROM employees WHERE needs_raise = TRUE;

    WHILE v_offset < v_total DO
        START TRANSACTION;
        UPDATE employees SET salary = salary * 1.05
        WHERE needs_raise = TRUE
        ORDER BY id
        LIMIT v_batch_size OFFSET v_offset;
        COMMIT;

        SET v_offset = v_offset + v_batch_size;

        DO SLEEP(0.1);  -- Give other transactions breathing room
    END WHILE;
END$$
DELIMITER ;
```

---

## Quick Reference

```sql
-- Transaction control
BEGIN;  / START TRANSACTION;
COMMIT;
ROLLBACK;
SAVEPOINT sp_name;
ROLLBACK TO SAVEPOINT sp_name;
RELEASE SAVEPOINT sp_name;

-- Isolation levels
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- Locking
SELECT ... FOR UPDATE;                 -- Exclusive row lock
SELECT ... FOR SHARE;                  -- Shared row lock (PostgreSQL)
SELECT ... LOCK IN SHARE MODE;         -- Shared row lock (MySQL)
SELECT ... FOR UPDATE SKIP LOCKED;     -- Skip locked rows (queue pattern)
SELECT ... FOR UPDATE NOWAIT;          -- Fail immediately if locked

-- Concurrency problems by isolation level
READ UNCOMMITTED: dirty reads, non-repeatable, phantom
READ COMMITTED:   non-repeatable reads, phantom reads
REPEATABLE READ:  phantom reads
SERIALIZABLE:     none (but lowest concurrency)

-- MVCC (PostgreSQL)
VACUUM t;            -- Clean dead rows
VACUUM FULL t;       -- Compact (exclusive lock)
VACUUM ANALYZE t;    -- Clean + update stats
```
