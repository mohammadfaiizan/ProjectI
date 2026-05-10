# Indexes

## Table of Contents
1. [What is an Index?](#1-what-is-an-index)
2. [Index Types](#2-index-types)
3. [Creating and Managing Indexes](#3-creating-and-managing-indexes)
4. [Composite Indexes](#4-composite-indexes)
5. [Partial Indexes](#5-partial-indexes)
6. [Covering Indexes](#6-covering-indexes)
7. [Index Internals](#7-index-internals)
8. [When Indexes Are and Are Not Used](#8-when-indexes-are-and-are-not-used)
9. [Index Maintenance](#9-index-maintenance)
10. [Indexing Strategies](#10-indexing-strategies)

---

## 1. What is an Index?

An index is a separate data structure that improves the speed of data retrieval at the cost of additional storage and write overhead.

### Without Index (Full Table Scan)
```
SELECT * FROM employees WHERE email = 'alice@company.com';
-- Scans every row: O(n) time
```

### With Index (Index Seek)
```
-- B-tree index on email
-- Finds the row directly: O(log n) time
```

### Trade-offs
| Benefit | Cost |
|---------|------|
| Faster SELECT | Slower INSERT/UPDATE/DELETE |
| Faster ORDER BY | Extra disk space |
| Faster JOIN conditions | Memory overhead |
| Faster WHERE filters | Maintenance on schema change |

---

## 2. Index Types

### B-Tree Index (Default)
Used for equality and range queries. The most common type.
```sql
-- Created automatically for PRIMARY KEY and UNIQUE
CREATE INDEX idx_employees_salary ON employees (salary);

-- Supports: =, <, >, <=, >=, BETWEEN, IN, LIKE 'prefix%'
-- Does NOT support: LIKE '%suffix', functions on column
```

### Hash Index
Optimized for exact equality only. Faster than B-tree for equality, useless for ranges.
```sql
-- PostgreSQL
CREATE INDEX idx_employees_email_hash
ON employees USING HASH (email);

-- MySQL: automatically used for MEMORY tables
-- Hash index: O(1) for =, but no range support
```

### GiST Index (Generalized Search Tree) — PostgreSQL
```sql
-- Used for geometric types, full-text, range types
CREATE INDEX idx_locations_coords ON locations USING GIST (coordinates);
CREATE INDEX idx_ranges ON events USING GIST (daterange);
```

### GIN Index (Generalized Inverted Index) — PostgreSQL
```sql
-- Used for full-text search, arrays, JSONB
CREATE INDEX idx_articles_content ON articles USING GIN (to_tsvector('english', content));
CREATE INDEX idx_products_tags ON products USING GIN (tags);  -- array column
CREATE INDEX idx_data_json ON events USING GIN (metadata);    -- JSONB column
```

### BRIN Index (Block Range Index) — PostgreSQL
```sql
-- Very compact; useful for naturally ordered large tables (timestamps, IDs)
CREATE INDEX idx_orders_created ON orders USING BRIN (created_at);
-- Each index entry covers a range of table blocks, not individual rows
-- Much smaller than B-tree, but less precise
```

### Full-Text Index
```sql
-- MySQL
CREATE FULLTEXT INDEX idx_articles_content ON articles (title, body);
SELECT * FROM articles WHERE MATCH(title, body) AGAINST('SQL performance' IN BOOLEAN MODE);

-- PostgreSQL: use GIN with tsvector
CREATE INDEX idx_articles_fts ON articles USING GIN (to_tsvector('english', body));
SELECT * FROM articles WHERE to_tsvector('english', body) @@ to_tsquery('SQL & performance');
```

### Spatial Index
```sql
-- MySQL
CREATE SPATIAL INDEX idx_locations_point ON locations (geo_point);
SELECT * FROM locations WHERE ST_CONTAINS(ST_GEOMFROMTEXT('POLYGON(...)'), geo_point);
```

---

## 3. Creating and Managing Indexes

### Create Index
```sql
-- Basic syntax
CREATE INDEX index_name ON table_name (column1 [, column2, ...]);

-- Examples
CREATE INDEX idx_employees_last_name ON employees (last_name);
CREATE INDEX idx_orders_customer_id  ON orders (customer_id);
CREATE INDEX idx_orders_status       ON orders (status);
```

### Create Unique Index
```sql
CREATE UNIQUE INDEX uq_employees_email ON employees (email);

-- Same as adding a UNIQUE constraint (which creates an index automatically)
ALTER TABLE employees ADD CONSTRAINT uq_email UNIQUE (email);
```

### Create Index Concurrently (PostgreSQL)
```sql
-- Builds index without locking the table (slower but non-blocking)
CREATE INDEX CONCURRENTLY idx_employees_salary ON employees (salary);

-- Drop concurrently
DROP INDEX CONCURRENTLY idx_employees_salary;
```

### Drop Index
```sql
DROP INDEX idx_employees_last_name;                        -- PostgreSQL
DROP INDEX idx_employees_last_name ON employees;           -- MySQL
DROP INDEX idx_employees_last_name ON dbo.employees;       -- SQL Server
```

### View Indexes

```sql
-- MySQL
SHOW INDEX FROM employees;
SHOW CREATE TABLE employees;

-- PostgreSQL
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'employees';

-- Or using \d+ in psql:
\d employees

-- SQL Server
SELECT * FROM sys.indexes WHERE object_id = OBJECT_ID('employees');

-- Generic (information_schema)
SELECT *
FROM information_schema.STATISTICS
WHERE TABLE_NAME = 'employees';    -- MySQL
```

---

## 4. Composite Indexes

A composite (multi-column) index covers multiple columns.

### Creating Composite Indexes
```sql
-- Index on (last_name, first_name)
CREATE INDEX idx_employees_name ON employees (last_name, first_name);

-- Index on (dept_id, salary)
CREATE INDEX idx_employees_dept_salary ON employees (dept_id, salary);
```

### Column Order Matters (Leftmost Prefix Rule)

```sql
-- Index: (last_name, first_name, dept_id)
CREATE INDEX idx ON employees (last_name, first_name, dept_id);

-- Uses the index:
WHERE last_name = 'Smith'                                -- Leftmost prefix
WHERE last_name = 'Smith' AND first_name = 'John'        -- First 2 columns
WHERE last_name = 'Smith' AND first_name = 'John' AND dept_id = 10  -- All 3

-- Does NOT use the index:
WHERE first_name = 'John'                               -- Skips leftmost column
WHERE dept_id = 10                                      -- Skips first two
WHERE first_name = 'John' AND dept_id = 10              -- Skips last_name

-- Uses the index partially:
WHERE last_name = 'Smith' AND dept_id = 10
-- Uses index for last_name = 'Smith', then filters dept_id without index
```

### Choosing Column Order for Composite Index
1. Put columns used in equality conditions first
2. Put columns used in range conditions last
3. Put higher-selectivity columns first (more distinct values)

```sql
-- Query: WHERE dept_id = 10 AND status = 'active' AND hire_date > '2020-01-01'
-- Best index: (dept_id, status, hire_date)
-- - Equality columns (dept_id, status) first
-- - Range column (hire_date) last
CREATE INDEX idx_emp_dept_status_date ON employees (dept_id, status, hire_date);
```

---

## 5. Partial Indexes

Index only a subset of rows (where a condition is true). Smaller and faster than full indexes.

```sql
-- PostgreSQL: partial index
CREATE INDEX idx_orders_pending
ON orders (customer_id, order_date)
WHERE status = 'pending';

-- Only used by queries that include the partial index condition:
SELECT * FROM orders WHERE status = 'pending' AND customer_id = 42;
-- ✓ Uses partial index

SELECT * FROM orders WHERE customer_id = 42;
-- ✗ Does NOT use partial index (no status = 'pending' filter)
```

### Practical Partial Indexes
```sql
-- Index only non-NULL values
CREATE INDEX idx_employees_bonus
ON employees (bonus)
WHERE bonus IS NOT NULL;

-- Index only active records (common soft-delete pattern)
CREATE INDEX idx_users_active
ON users (email, last_login)
WHERE deleted_at IS NULL;

-- Index only recent orders
CREATE INDEX idx_recent_orders
ON orders (customer_id, total)
WHERE order_date >= '2024-01-01';
```

### MySQL Partial Index (via expression index — MySQL 8.0+)
```sql
-- MySQL doesn't support WHERE-based partial indexes natively
-- Workaround: use a generated column
ALTER TABLE orders ADD COLUMN is_pending TINYINT(1)
GENERATED ALWAYS AS (IF(status = 'pending', 1, NULL)) VIRTUAL;

CREATE INDEX idx_pending ON orders (is_pending, customer_id);
```

---

## 6. Covering Indexes

A covering index includes all columns needed by a query, avoiding a table lookup ("index only scan").

```sql
-- Query needs: customer_id, order_date, total, status
SELECT customer_id, order_date, total
FROM orders
WHERE status = 'shipped';

-- Regular index (must go back to table for non-index columns)
CREATE INDEX idx_status ON orders (status);

-- Covering index (all needed columns in index)
CREATE INDEX idx_orders_covering ON orders (status, customer_id, order_date, total);
-- Index satisfies entire query without touching the table
```

### PostgreSQL: INCLUDE clause (non-key columns in index)
```sql
-- Key column: status (for WHERE)
-- Non-key columns: customer_id, order_date, total (for SELECT)
CREATE INDEX idx_orders_covering
ON orders (status)
INCLUDE (customer_id, order_date, total);
-- Non-key columns not used for ordering, just carried along for covering
```

### SQL Server: INCLUDE
```sql
CREATE INDEX idx_orders_covering
ON orders (status)
INCLUDE (customer_id, order_date, total);
```

---

## 7. Index Internals

### B-Tree Structure
```
                    [50]
                   /    \
            [25]           [75]
           /    \          /   \
       [10,20] [30,40] [60,70] [80,90]
       leaf     leaf    leaf    leaf
```
- Root → Branch → Leaf pages
- Leaf pages contain actual row data (or row pointers)
- Each level narrows the search space
- Height ≈ log_B(n) where B is branching factor (typically 100-1000)
- For 1 billion rows: ~3-4 levels deep

### Clustered vs Non-Clustered Index

| Feature | Clustered | Non-Clustered |
|---------|-----------|---------------|
| Row order | Rows stored in index order | Rows stored separately |
| Count per table | 1 (only one physical order) | Many |
| Access | Direct row access | Extra lookup (key → row) |
| MySQL (InnoDB) | PRIMARY KEY is always clustered | Secondary indexes |
| PostgreSQL | CLUSTER command (manual) | All indexes non-clustered by default |

```sql
-- MySQL InnoDB: PRIMARY KEY = clustered index
CREATE TABLE employees (
    id INT PRIMARY KEY,  -- This IS the clustered index
    name VARCHAR(100)
);

-- PostgreSQL: manually cluster a table by an index
CLUSTER employees USING idx_employees_dept_id;
-- This physically reorders the table; index must exist first
-- Note: CLUSTER doesn't stay clustered on future writes

-- SQL Server
CREATE CLUSTERED INDEX idx_clustered ON employees (id);
CREATE NONCLUSTERED INDEX idx_non_clustered ON employees (email);
```

### Index Fill Factor
```sql
-- Leave space in index pages for future insertions (reduces page splits)
CREATE INDEX idx_employees_name ON employees (last_name)
WITH (fillfactor = 80);   -- PostgreSQL: 80% full
-- SQL Server equivalent:
CREATE INDEX idx ON employees (last_name) WITH (FILLFACTOR = 80);
```

---

## 8. When Indexes Are and Are Not Used

### Indexes ARE Used
```sql
-- Equality on indexed column
WHERE email = 'alice@company.com'

-- Range on indexed column
WHERE salary BETWEEN 50000 AND 100000
WHERE hire_date >= '2020-01-01'

-- LIKE with leading literal (prefix search)
WHERE last_name LIKE 'Smi%'

-- IS NULL (if indexed)
WHERE manager_id IS NULL

-- IN with small list
WHERE dept_id IN (10, 20)

-- ORDER BY matches index order
ORDER BY last_name ASC  -- if index is on (last_name ASC)
```

### Indexes Are NOT Used
```sql
-- Function on indexed column
WHERE UPPER(email) = 'ALICE@COMPANY.COM'  -- ✗ index not used
-- Fix: create function-based index
CREATE INDEX idx_email_upper ON employees (UPPER(email));
WHERE UPPER(email) = 'ALICE@COMPANY.COM'  -- ✓ now uses index

-- LIKE with leading wildcard
WHERE last_name LIKE '%smith'  -- ✗ can't use B-tree from the right

-- Type mismatch
WHERE id = '42'    -- ✗ if id is INT and '42' is VARCHAR (implicit cast)
WHERE id = 42      -- ✓

-- NOT, !=, <> (usually)
WHERE status != 'active'  -- May not use index (depends on cardinality)

-- OR across different columns (unless index covers both)
WHERE email = 'a@b.com' OR phone = '555-1234'  -- ✗ usually full scan

-- Arithmetic on indexed column
WHERE salary * 12 > 100000    -- ✗
WHERE salary > 100000 / 12    -- ✓ (move arithmetic to constant side)

-- Low selectivity (optimizer may prefer full scan)
WHERE is_active = TRUE  -- If 95% of rows are active, index not helpful
```

### Function-Based Indexes
```sql
-- PostgreSQL / Oracle
CREATE INDEX idx_email_lower ON users (LOWER(email));
SELECT * FROM users WHERE LOWER(email) = 'alice@company.com';

-- MySQL 8.0+ (expression index)
CREATE INDEX idx_email_upper ON users ((UPPER(email)));
SELECT * FROM users WHERE UPPER(email) = 'ALICE@COMPANY.COM';
```

---

## 9. Index Maintenance

### Check Index Usage (PostgreSQL)
```sql
SELECT
    indexrelname AS index_name,
    relname AS table_name,
    idx_scan AS times_used,
    idx_tup_read AS rows_returned,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan;

-- Find unused indexes
SELECT indexrelname, idx_scan, pg_size_pretty(pg_relation_size(indexrelid))
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;
```

### Rebuild / Reindex
```sql
-- PostgreSQL
REINDEX INDEX idx_employees_email;
REINDEX TABLE employees;
REINDEX DATABASE mydb;

-- REINDEX CONCURRENTLY (non-blocking, PostgreSQL 12+)
REINDEX INDEX CONCURRENTLY idx_employees_email;

-- MySQL
OPTIMIZE TABLE employees;   -- Also rebuilds indexes
ALTER TABLE employees ENGINE=InnoDB;  -- Full rebuild

-- SQL Server
ALTER INDEX idx_employees_email ON employees REBUILD;
ALTER INDEX ALL ON employees REBUILD;
ALTER INDEX idx_employees_email ON employees REORGANIZE;  -- Light rebuild
```

### Analyze / Update Statistics
```sql
-- PostgreSQL: update optimizer statistics
ANALYZE employees;
ANALYZE;  -- Analyze all tables

-- MySQL: update statistics
ANALYZE TABLE employees;

-- SQL Server
UPDATE STATISTICS employees;
UPDATE STATISTICS employees idx_employees_email;
```

---

## 10. Indexing Strategies

### Rules of Thumb

1. **Index foreign keys**: Every foreign key column should have an index
```sql
CREATE INDEX idx_orders_customer_id ON orders (customer_id);
CREATE INDEX idx_order_items_order_id ON order_items (order_id);
```

2. **Index columns used in WHERE, JOIN, ORDER BY, GROUP BY**
```sql
-- Frequent query: WHERE status = 'active' ORDER BY created_at DESC
CREATE INDEX idx_posts_status_created ON posts (status, created_at DESC);
```

3. **Don't index everything**: Each index slows down writes
```sql
-- Rule of thumb: ~5-10 indexes max per table
-- More for read-heavy tables, fewer for write-heavy tables
```

4. **High-cardinality columns benefit most from indexes**
```sql
-- Good candidates: email, user_id, order_id (many distinct values)
-- Poor candidates: boolean columns, gender, status with few values
```

5. **Composite index column order**: Put equality columns first, range last
```sql
-- Query: WHERE dept = 'Eng' AND salary > 50000
CREATE INDEX idx ON employees (dept, salary);
-- dept (equality) → salary (range)
```

6. **Consider query patterns**: Index to match your most frequent queries
```sql
-- For: SELECT name, email FROM users WHERE last_login > '2024-01-01'
CREATE INDEX idx_users_login ON users (last_login) INCLUDE (name, email);
```

### Index Size vs Performance

```sql
-- PostgreSQL: show index sizes
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_indexes
JOIN pg_class ON relname = tablename
WHERE tablename = 'employees'
ORDER BY pg_relation_size(indexrelid) DESC;

-- MySQL
SELECT
    INDEX_NAME,
    ROUND(STAT_VALUE * @@innodb_page_size / 1024 / 1024, 2) AS size_mb
FROM mysql.innodb_index_stats
WHERE database_name = 'mydb' AND table_name = 'employees' AND stat_name = 'size';
```

---

## Index Quick Reference

```sql
-- Create
CREATE INDEX idx_name ON t (col);
CREATE UNIQUE INDEX uq_name ON t (col);
CREATE INDEX idx_name ON t (col1, col2);         -- Composite
CREATE INDEX idx_name ON t (col) WHERE condition; -- Partial (PostgreSQL)
CREATE INDEX idx_name ON t (LOWER(col));          -- Functional
CREATE INDEX CONCURRENTLY idx ON t (col);         -- Non-blocking (PostgreSQL)

-- Drop
DROP INDEX idx_name;                    -- PostgreSQL
DROP INDEX idx_name ON t;               -- MySQL
DROP INDEX CONCURRENTLY idx_name;       -- Non-blocking (PostgreSQL)

-- View
SHOW INDEX FROM t;                      -- MySQL
SELECT * FROM pg_indexes WHERE tablename = 't';  -- PostgreSQL

-- Rebuild
REINDEX INDEX idx_name;                 -- PostgreSQL
ALTER INDEX idx_name ON t REBUILD;      -- SQL Server
OPTIMIZE TABLE t;                       -- MySQL

-- Analyze (update stats)
ANALYZE t;                              -- PostgreSQL
ANALYZE TABLE t;                        -- MySQL
UPDATE STATISTICS t;                    -- SQL Server
```
