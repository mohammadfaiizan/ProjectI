# Performance and Optimization

## Table of Contents
1. [Query Execution Plan (EXPLAIN)](#1-query-execution-plan-explain)
2. [Index Optimization](#2-index-optimization)
3. [Query Rewriting Techniques](#3-query-rewriting-techniques)
4. [JOIN Optimization](#4-join-optimization)
5. [Subquery Optimization](#5-subquery-optimization)
6. [Partitioning](#6-partitioning)
7. [Caching and Statistics](#7-caching-and-statistics)
8. [Schema Optimization](#8-schema-optimization)
9. [Common Performance Anti-Patterns](#9-common-performance-anti-patterns)
10. [Monitoring and Profiling](#10-monitoring-and-profiling)

---

## 1. Query Execution Plan (EXPLAIN)

EXPLAIN shows how the database plans to execute a query, revealing bottlenecks.

### MySQL EXPLAIN
```sql
-- Basic explain
EXPLAIN SELECT * FROM employees WHERE dept_id = 10;

-- Result columns:
-- id: step number
-- select_type: SIMPLE, SUBQUERY, DERIVED, UNION
-- table: which table
-- type: access method (ALL, index, range, ref, eq_ref, const)
-- possible_keys: candidate indexes
-- key: index actually used
-- key_len: index length used
-- ref: columns compared to index
-- rows: estimated rows examined
-- Extra: Using filesort, Using temporary, Using index

-- Extended explain (MySQL 5.7+)
EXPLAIN FORMAT=JSON SELECT * FROM employees WHERE dept_id = 10;
EXPLAIN FORMAT=TREE SELECT * FROM employees WHERE dept_id = 10;

-- EXPLAIN with execution (MySQL 8.0+)
EXPLAIN ANALYZE SELECT * FROM employees WHERE dept_id = 10;
```

### Access Type Rankings (Best to Worst)
```
const     → Reads exactly one row (primary key / unique key = constant)
eq_ref    → One row per join (unique index lookup)
ref       → Multiple rows, non-unique index lookup
range     → Index range scan (BETWEEN, IN, <, >)
index     → Full index scan (better than ALL)
ALL       → Full table scan (worst — often needs an index)
```

### PostgreSQL EXPLAIN
```sql
-- Basic plan
EXPLAIN SELECT * FROM employees WHERE dept_id = 10;

-- With actual timing and row counts
EXPLAIN ANALYZE SELECT * FROM employees WHERE dept_id = 10;

-- Verbose output
EXPLAIN (ANALYZE, VERBOSE, BUFFERS, FORMAT JSON)
SELECT * FROM employees WHERE dept_id = 10;

-- Key plan nodes:
-- Seq Scan       → Full table scan
-- Index Scan     → B-tree scan with table lookup
-- Index Only Scan → Uses index only (covering index)
-- Bitmap Heap Scan → Multiple index entries combined
-- Hash Join      → Hash table join
-- Nested Loop    → Row-by-row join (good for small tables)
-- Merge Join     → Sorted merge join (good for large sorted inputs)
-- Sort           → Explicit sort (look for high cost)
-- Hash Aggregate → GROUP BY via hash
-- cost=0.00..X   → Estimated cost (startup..total)
-- rows=N         → Estimated row count
-- actual rows=N  → Actual rows (with ANALYZE)
-- loops=N        → Times this node was executed
```

### Reading Explain Output
```sql
EXPLAIN ANALYZE
SELECT e.name, d.name
FROM employees e
JOIN departments d ON e.dept_id = d.id
WHERE e.salary > 80000;

-- Sample output:
-- Hash Join  (cost=5.25..25.60 rows=3 width=64) (actual time=0.050..0.080 rows=3 loops=1)
--   Hash Cond: (e.dept_id = d.id)
--   ->  Seq Scan on employees e  (cost=0.00..18.50 rows=3) (actual rows=3)
--         Filter: (salary > 80000)
--         Rows Removed by Filter: 47
--   ->  Hash  (cost=3.10..3.10 rows=10)
--         Buckets: 1024  Batches: 1
--         ->  Seq Scan on departments d  (cost=0.00..3.10 rows=10)
-- Planning Time: 0.150 ms
-- Execution Time: 0.120 ms
```

### SQL Server Execution Plan
```sql
-- Text plan
SET SHOWPLAN_TEXT ON;
SELECT * FROM employees WHERE dept_id = 10;
SET SHOWPLAN_TEXT OFF;

-- XML plan (for SSMS visual display)
SET SHOWPLAN_XML ON;
SELECT * FROM employees WHERE dept_id = 10;
SET SHOWPLAN_XML OFF;

-- Actual execution plan
SET STATISTICS IO ON;
SET STATISTICS TIME ON;
SELECT * FROM employees WHERE dept_id = 10;
SET STATISTICS IO OFF;
SET STATISTICS TIME OFF;
```

---

## 2. Index Optimization

### Identify Missing Indexes
```sql
-- PostgreSQL: queries doing seq scans on large tables
SELECT
    relname AS table_name,
    seq_scan,
    idx_scan,
    n_live_tup AS live_rows
FROM pg_stat_user_tables
WHERE seq_scan > idx_scan AND n_live_tup > 10000
ORDER BY seq_scan DESC;

-- MySQL: slow query log
-- Enable slow query log:
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 1;  -- Queries > 1 second
SET GLOBAL slow_query_log_file = '/var/log/mysql/slow.log';
```

### Index-Friendly Query Patterns
```sql
-- 1. Use indexes: equality and range on indexed columns
WHERE dept_id = 10 AND salary > 50000
-- Best index: (dept_id, salary) — equality first

-- 2. Covering indexes: include all needed columns
SELECT name, email FROM users WHERE status = 'active';
-- CREATE INDEX idx ON users (status) INCLUDE (name, email);

-- 3. Avoid function on indexed column
-- Bad:
WHERE YEAR(hire_date) = 2020                    -- no index
WHERE UPPER(last_name) = 'SMITH'                -- no index

-- Good (compute range instead):
WHERE hire_date BETWEEN '2020-01-01' AND '2020-12-31'  -- index used
WHERE last_name = 'Smith'  -- (store normalized data)

-- 4. Match data types to avoid implicit casting
-- Bad (id is INT, '42' is VARCHAR):
WHERE id = '42'   -- implicit cast prevents index use

-- Good:
WHERE id = 42

-- 5. LIKE with leading literal
-- Uses index:
WHERE last_name LIKE 'Smi%'

-- Does NOT use B-tree index:
WHERE last_name LIKE '%mith'

-- 6. Leading index column must be in query
-- Index: (a, b, c)
WHERE a = 1             -- uses index (a)
WHERE a = 1 AND b = 2   -- uses index (a, b)
WHERE b = 2             -- does NOT use composite index
```

### Index Selectivity
```sql
-- Check index selectivity
SELECT
    COUNT(DISTINCT dept_id) AS distinct_values,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT dept_id) * 1.0 / COUNT(*) AS selectivity
FROM employees;
-- Selectivity near 1.0 = highly selective (good for indexing)
-- Selectivity near 0.0 = low selectivity (index rarely helps)

-- Rule: If selectivity < 5%, index may not be used by optimizer
-- (Full scan is cheaper than index scan for 50% of rows)
```

---

## 3. Query Rewriting Techniques

### Use Covered Query vs. SELECT *
```sql
-- Bad: retrieves all columns, always hits table
SELECT * FROM orders WHERE customer_id = 42;

-- Good: only needed columns, can use covering index
SELECT id, order_date, total FROM orders WHERE customer_id = 42;
```

### EXISTS vs IN vs JOIN for existence check
```sql
-- Fastest: EXISTS (stops at first match)
SELECT c.name FROM customers c
WHERE EXISTS (SELECT 1 FROM orders WHERE customer_id = c.id);

-- Usually optimized similarly by modern engines:
SELECT c.name FROM customers c
WHERE c.id IN (SELECT customer_id FROM orders);

SELECT DISTINCT c.name FROM customers c
JOIN orders o ON c.id = o.customer_id;
```

### Push Filters Down
```sql
-- Bad: filter applied after large join
SELECT *
FROM orders o
JOIN order_items oi ON o.id = oi.order_id
WHERE o.status = 'pending';

-- Good: filter applied before join (usually optimizer does this automatically)
SELECT *
FROM (SELECT * FROM orders WHERE status = 'pending') o
JOIN order_items oi ON o.id = oi.order_id;
```

### Avoid OR with Different Columns (Use UNION ALL)
```sql
-- Bad: OR can prevent index use
SELECT * FROM employees WHERE email = 'a@b.com' OR phone = '555-1234';

-- Good: each part uses its own index
SELECT * FROM employees WHERE email = 'a@b.com'
UNION ALL
SELECT * FROM employees WHERE phone = '555-1234'
  AND email != 'a@b.com';  -- Avoid duplicates
```

### Avoid Correlated Subqueries (Use JOIN or Window Functions)
```sql
-- Slow: correlated subquery runs once per row
SELECT e1.name, e1.salary,
    (SELECT AVG(e2.salary) FROM employees e2 WHERE e2.dept_id = e1.dept_id) AS dept_avg
FROM employees e1;

-- Fast: compute avg once per dept
SELECT e.name, e.salary, d.avg_sal
FROM employees e
JOIN (
    SELECT dept_id, AVG(salary) AS avg_sal FROM employees GROUP BY dept_id
) d ON e.dept_id = d.dept_id;

-- Or with window function:
SELECT name, salary, AVG(salary) OVER (PARTITION BY dept_id) AS dept_avg
FROM employees;
```

### Limit Early
```sql
-- Bad: aggregate then limit
SELECT * FROM (
    SELECT customer_id, SUM(total) AS spent FROM orders GROUP BY customer_id
) t ORDER BY spent DESC LIMIT 10;

-- This is fine — the subquery is needed. But:
-- If just the top-N rows are needed without aggregation:
SELECT * FROM employees
ORDER BY salary DESC
LIMIT 10;
-- Not: SELECT * FROM (SELECT * FROM employees ORDER BY salary) t LIMIT 10;
```

### Avoid DISTINCT When Not Needed
```sql
-- Bad: unnecessary DISTINCT
SELECT DISTINCT dept_id FROM employees;

-- Better: if you just need unique dept_ids
SELECT dept_id FROM employees GROUP BY dept_id;

-- Or if using JOIN and getting duplicates, fix the join instead
```

---

## 4. JOIN Optimization

### Join Order Matters
```sql
-- The optimizer usually chooses the best join order
-- But you can hint or restructure for complex queries

-- Start with the most selective filter
SELECT e.name, d.name, p.title
FROM employees e
JOIN departments d ON e.dept_id = d.id      -- large table
JOIN projects p ON e.current_project = p.id  -- filtered
WHERE p.status = 'active'  -- Most selective: put active projects first
  AND d.location = 'NY';

-- PostgreSQL join hint (not standard, usually not needed)
-- SET enable_hashjoin = OFF;
-- SET enable_mergejoin = OFF;
-- SET join_collapse_limit = 1;  -- Honor FROM clause order
```

### Index on Join Columns
```sql
-- Every column in ON clause should be indexed
CREATE INDEX idx_orders_customer ON orders (customer_id);
CREATE INDEX idx_items_order ON order_items (order_id);

-- Good join uses index lookup, not full scan:
SELECT o.*, oi.product_id
FROM orders o
JOIN order_items oi ON o.id = oi.order_id  -- oi.order_id should be indexed
WHERE o.customer_id = 42;                   -- o.customer_id should be indexed
```

### Reduce Result Set Before Joining
```sql
-- Bad: join full tables, then filter
SELECT e.name, d.name
FROM employees e
JOIN departments d ON e.dept_id = d.id
WHERE d.location = 'NY' AND e.salary > 80000;

-- Good: pre-filter where possible (usually optimizer does this)
SELECT e.name, d.name
FROM (SELECT * FROM employees WHERE salary > 80000) e
JOIN (SELECT * FROM departments WHERE location = 'NY') d
    ON e.dept_id = d.id;
```

---

## 5. Subquery Optimization

### Replace Scalar Subquery with JOIN
```sql
-- Slow: scalar subquery executes N times
SELECT id, name,
    (SELECT name FROM departments WHERE id = e.dept_id) AS dept_name
FROM employees e;

-- Fast: single join
SELECT e.id, e.name, d.name AS dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id;
```

### Replace IN Subquery with JOIN
```sql
-- May be slow for large subquery result
SELECT * FROM orders
WHERE customer_id IN (SELECT id FROM customers WHERE country = 'US');

-- Often faster
SELECT DISTINCT o.*
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE c.country = 'US';
```

### CTE vs Subquery Performance
```sql
-- PostgreSQL: CTEs may be materialized (stored), preventing optimization
-- Use NOT MATERIALIZED when you want the optimizer to inline it
WITH active_emps AS NOT MATERIALIZED (
    SELECT * FROM employees WHERE is_active = TRUE
)
SELECT * FROM active_emps WHERE salary > 80000;

-- MySQL: CTEs are always optimized as derived tables
```

---

## 6. Partitioning

Partitioning splits a large table into smaller physical pieces for better performance.

### Range Partitioning (MySQL)
```sql
-- Partition orders by year
CREATE TABLE orders (
    id         INT,
    order_date DATE,
    total      DECIMAL(10,2),
    PRIMARY KEY (id, order_date)  -- Partition key must be in PK
)
PARTITION BY RANGE (YEAR(order_date)) (
    PARTITION p2021 VALUES LESS THAN (2022),
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- Queries filter by year → only scan relevant partition (partition pruning)
SELECT * FROM orders WHERE order_date BETWEEN '2023-01-01' AND '2023-12-31';
-- Only scans p2023 partition!
```

### Range Partitioning (PostgreSQL)
```sql
-- Parent table
CREATE TABLE orders (
    id         SERIAL,
    order_date DATE NOT NULL,
    total      DECIMAL(10,2)
) PARTITION BY RANGE (order_date);

-- Child partitions
CREATE TABLE orders_2022 PARTITION OF orders
    FOR VALUES FROM ('2022-01-01') TO ('2023-01-01');

CREATE TABLE orders_2023 PARTITION OF orders
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE orders_2024 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- Default partition
CREATE TABLE orders_default PARTITION OF orders DEFAULT;

-- Query auto-routes to correct partition
SELECT * FROM orders WHERE order_date >= '2023-01-01';
-- Queries the orders_2023 partition only
```

### Hash Partitioning (PostgreSQL)
```sql
CREATE TABLE users (
    id   INT,
    name VARCHAR(100)
) PARTITION BY HASH (id);

CREATE TABLE users_0 PARTITION OF users FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE users_1 PARTITION OF users FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE users_2 PARTITION OF users FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE users_3 PARTITION OF users FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

### List Partitioning
```sql
-- MySQL
CREATE TABLE sales (
    id INT,
    region VARCHAR(20),
    amount DECIMAL
)
PARTITION BY LIST COLUMNS(region) (
    PARTITION p_us VALUES IN ('US', 'CA', 'MX'),
    PARTITION p_eu VALUES IN ('UK', 'DE', 'FR', 'ES'),
    PARTITION p_ap VALUES IN ('JP', 'CN', 'IN', 'AU')
);
```

### Partition Maintenance
```sql
-- MySQL: add / drop partitions
ALTER TABLE orders ADD PARTITION (PARTITION p2025 VALUES LESS THAN (2026));
ALTER TABLE orders DROP PARTITION p2021;           -- Deletes data!
ALTER TABLE orders TRUNCATE PARTITION p2021;       -- Delete data, keep partition
ALTER TABLE orders EXCHANGE PARTITION p2021 WITH TABLE orders_archive;

-- PostgreSQL: detach / attach
ALTER TABLE orders DETACH PARTITION orders_2021;  -- Becomes standalone table
ALTER TABLE orders ATTACH PARTITION orders_2021 FOR VALUES FROM ('2021-01-01') TO ('2022-01-01');
```

---

## 7. Caching and Statistics

### Query Cache (MySQL)
```sql
-- MySQL 8.0: query cache REMOVED (deprecated in 5.7)
-- Use external caches (Redis, Memcached) for query result caching

-- Check if query will use cache (MySQL < 8.0)
SHOW STATUS LIKE 'Qcache%';
```

### Buffer Pool (InnoDB)
```sql
-- Check buffer pool hit ratio (should be > 99%)
SHOW STATUS LIKE 'Innodb_buffer_pool%';
-- Innodb_buffer_pool_read_requests / (read_requests + reads) = hit ratio

-- Adjust buffer pool size (my.cnf)
-- innodb_buffer_pool_size = 4G  -- 60-80% of available RAM
```

### PostgreSQL Shared Buffers
```sql
-- Check cache hit ratio
SELECT
    sum(heap_blks_hit) /
    NULLIF(sum(heap_blks_hit) + sum(heap_blks_read), 0) AS cache_hit_ratio
FROM pg_statio_user_tables;

-- postgresql.conf
-- shared_buffers = 256MB    (start here, tune up)
-- effective_cache_size = 4GB  (tell optimizer about OS cache)
-- work_mem = 4MB             (per sort operation)
-- maintenance_work_mem = 1GB  (for VACUUM, CREATE INDEX)
```

### Update Statistics
```sql
-- PostgreSQL: auto-analyze runs periodically; manual if needed
ANALYZE employees;          -- One table
ANALYZE;                    -- All tables
VACUUM ANALYZE employees;   -- Vacuum + analyze

-- MySQL
ANALYZE TABLE employees;

-- SQL Server
UPDATE STATISTICS employees;
UPDATE STATISTICS employees WITH FULLSCAN;   -- Full sample
```

---

## 8. Schema Optimization

### Appropriate Data Types
```sql
-- Bad: using VARCHAR(255) for everything
status    VARCHAR(255)    -- wastes space, slower comparison
dept_id   VARCHAR(255)    -- int comparison is faster

-- Good: use appropriate types
status    VARCHAR(20)     -- or ENUM
dept_id   INT             -- integer

-- Use TINYINT instead of INT when range allows
is_active TINYINT(1)     -- 1 byte vs 4 bytes
score     TINYINT UNSIGNED  -- 0-255 range (1 byte)

-- DECIMAL vs FLOAT for money
price  FLOAT          -- ✗ floating point errors
price  DECIMAL(10,2)  -- ✓ exact
```

### Normalization vs Denormalization
```sql
-- Normalized (3NF): no redundancy, slower complex queries
-- employees → departments (foreign key)

-- Denormalized: redundant data, faster reads
-- employees table has dept_name column (copied from departments)
-- Must maintain consistency with triggers or application code

-- Denormalization examples:
-- Add count column: customers.order_count (maintained by trigger)
-- Add redundant column: orders.customer_name (avoid JOIN for reports)
-- JSON columns for flexible attributes (avoid EAV anti-pattern)
```

### Table Design Checklist
```sql
-- ✓ Use surrogate keys (INT/UUID) as primary keys
-- ✓ Index all foreign keys
-- ✓ Appropriate data types for each column
-- ✓ NOT NULL where applicable
-- ✓ Sensible defaults
-- ✓ created_at/updated_at audit columns
-- ✓ Consider archival strategy for large tables
-- ✓ Partition large tables by date or range
```

---

## 9. Common Performance Anti-Patterns

### Anti-Pattern 1: SELECT *
```sql
-- Bad: fetches all columns, can't use covering index
SELECT * FROM orders WHERE customer_id = 42;

-- Good: select only needed columns
SELECT id, order_date, total FROM orders WHERE customer_id = 42;
```

### Anti-Pattern 2: N+1 Query Problem
```sql
-- Bad: one query per customer (N+1)
SELECT id FROM customers;  -- Returns 1000 customers
-- Then for each customer:
SELECT * FROM orders WHERE customer_id = ?;  -- 1000 queries!

-- Good: single JOIN
SELECT c.id, c.name, o.id AS order_id, o.total
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.id;
```

### Anti-Pattern 3: Using OFFSET for Deep Pagination
```sql
-- Bad: OFFSET 100000 still scans 100000 rows
SELECT * FROM orders ORDER BY id LIMIT 10 OFFSET 100000;

-- Good: keyset pagination
SELECT * FROM orders WHERE id > :last_seen_id ORDER BY id LIMIT 10;
-- Always O(log n) regardless of page number
```

### Anti-Pattern 4: Implicit Type Conversion
```sql
-- Assuming id is INT:
WHERE id = '42'    -- String '42' forces implicit cast, prevents index use
WHERE id = 42      -- Correct
```

### Anti-Pattern 5: NOT IN with Nullable Subquery
```sql
-- If subquery can return NULL, NOT IN returns no rows
SELECT * FROM a WHERE id NOT IN (SELECT b_id FROM b);
-- If b has any NULL b_id → returns 0 rows!

-- Safe alternatives:
SELECT * FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.b_id = a.id);
SELECT * FROM a WHERE id NOT IN (SELECT b_id FROM b WHERE b_id IS NOT NULL);
```

### Anti-Pattern 6: Wildcard LIKE
```sql
-- Can't use B-tree index:
WHERE name LIKE '%smith%'

-- Solutions:
-- Full-text index (MySQL FULLTEXT, PostgreSQL GIN/tsvector)
-- Trigram index (PostgreSQL pg_trgm)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_name_trgm ON employees USING GIN (last_name gin_trgm_ops);
-- Now LIKE '%smith%' uses the trigram index!
```

---

## 10. Monitoring and Profiling

### MySQL Slow Query Log
```sql
SHOW VARIABLES LIKE 'slow_query%';
SHOW VARIABLES LIKE 'long_query_time';
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 2;  -- Log queries > 2 seconds

-- Find slowest queries in slow log
-- Use: pt-query-digest (Percona Toolkit)
```

### MySQL Performance Schema
```sql
-- Enable
UPDATE performance_schema.setup_instruments SET ENABLED = 'YES', TIMED = 'YES'
WHERE NAME LIKE '%statement%';

-- Top slow queries
SELECT
    DIGEST_TEXT,
    COUNT_STAR AS exec_count,
    AVG_TIMER_WAIT / 1000000000 AS avg_seconds,
    SUM_ROWS_EXAMINED AS rows_examined
FROM performance_schema.events_statements_summary_by_digest
ORDER BY SUM_TIMER_WAIT DESC
LIMIT 10;
```

### PostgreSQL pg_stat_statements
```sql
-- Enable in postgresql.conf:
-- shared_preload_libraries = 'pg_stat_statements'
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Top slow queries
SELECT
    query,
    calls,
    total_exec_time / calls AS avg_ms,
    rows / calls AS avg_rows
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;

-- Reset stats
SELECT pg_stat_statements_reset();
```

### SHOW PROCESSLIST (MySQL)
```sql
SHOW PROCESSLIST;
SHOW FULL PROCESSLIST;

-- Kill long-running query
KILL QUERY 12345;   -- Kill just the query
KILL 12345;         -- Kill the connection
```

---

## Performance Quick Reference

```sql
-- EXPLAIN
EXPLAIN SELECT ...;                           -- MySQL / PostgreSQL
EXPLAIN ANALYZE SELECT ...;                   -- PostgreSQL (with actual stats)
EXPLAIN FORMAT=JSON SELECT ...;               -- MySQL / PostgreSQL

-- Key access types (MySQL): ALL > index > range > ref > eq_ref > const
-- Key plan nodes (PostgreSQL): Seq Scan > Bitmap > Index Scan > Index Only Scan

-- Index tips
-- Create on WHERE columns (high selectivity first)
-- Create on JOIN ON columns (foreign keys)
-- Covering index for SELECT columns
-- Partial index with WHERE condition
-- Avoid functions on indexed column in WHERE

-- Query tips
-- SELECT specific columns, not *
-- Push filters down (WHERE before JOIN when possible)
-- Avoid correlated subqueries (use JOIN instead)
-- Use EXISTS instead of IN for large sets
-- Use UNION ALL not UNION when duplicates are ok
-- Keyset pagination over OFFSET for large pages
-- LIMIT early in the query

-- Schema tips
-- Correct data types (INT not VARCHAR for IDs)
-- NOT NULL where appropriate
-- Archive old data (partition or separate table)
-- ANALYZE / VACUUM regularly (PostgreSQL)
-- ANALYZE TABLE (MySQL)
```
