# Indexes and Performance — Interview Questions

> **Difficulty Mix:** Easy (Q1–Q7) · Medium (Q8–Q14) · Hard (Q15–Q20)

---

### Q1. What is a database index and why do we use it?

**Answer:**
An index is a separate data structure that the database maintains alongside a table to enable fast data retrieval. It works like a book's index — instead of reading every page (full table scan), you jump directly to the right page.

**Without index (full table scan):**
```
SELECT * FROM employees WHERE email = 'alice@company.com';
→ Scans all N rows: O(N) — slow for large N
```

**With B-tree index on email:**
```
→ Traverses B-tree: O(log N) — fast regardless of N
```

**Trade-offs:**
| Benefit | Cost |
|---------|------|
| Faster SELECT, WHERE, JOIN, ORDER BY | Slower INSERT, UPDATE, DELETE (index must be updated) |
| Better sort performance | Additional disk space |
| Faster uniqueness enforcement | Memory overhead (index in buffer pool) |

---

### Q2. What is a B-tree index? How does it work?

**Answer:**
A B-tree (Balanced Tree) index is the default index type in all major RDBMS. It maintains sorted key values in a balanced tree structure.

```
                     [50]
                   /       \
            [20, 35]         [70, 90]
           /   |    \        /    |    \
       [5,15] [25,30] [40,45] [60,65] [80,85] [95,100]
       ↑leaf   ↑leaf   ↑leaf   ↑leaf   ↑leaf   ↑leaf
       (each leaf also has a pointer to the next leaf for range scans)
```

**Properties:**
- Height ≈ log_B(N) where B = branching factor (~100-1000 keys/node)
- For 1 billion rows, tree height ≈ 3-4 levels (3-4 page reads to find a row)
- Leaf nodes are linked for range scan efficiency
- Supports: =, <, >, <=, >=, BETWEEN, IN (small lists), LIKE 'prefix%', ORDER BY

---

### Q3. What types of indexes exist in PostgreSQL/MySQL?

**Answer:**

| Index Type | Best For | Does NOT Support |
|-----------|---------|-----------------|
| **B-tree** (default) | Equality, range, ORDER BY, LIKE prefix | LIKE '%suffix%' |
| **Hash** | Equality only (= operator) | Range queries |
| **GIN** (PG) | Full-text, arrays, JSONB, multi-value | Range |
| **GiST** (PG) | Geometric data, full-text, ranges | N/A |
| **BRIN** (PG) | Large naturally-ordered tables (timestamps) | Random access |
| **Full-text** (MySQL) | MATCH...AGAINST queries | Exact match |
| **Spatial** (MySQL) | Geographic queries (ST_CONTAINS, etc.) | Non-spatial |

```sql
-- PostgreSQL examples:
CREATE INDEX ON employees USING HASH  (email);      -- equality only
CREATE INDEX ON articles  USING GIN   (to_tsvector('english', body));  -- full-text
CREATE INDEX ON locations USING GIST  (coordinates);  -- geometric
CREATE INDEX ON events    USING BRIN  (created_at);   -- append-only time data
```

---

### Q4. What is the difference between a clustered and a non-clustered index?

**Answer:**

| Feature | Clustered Index | Non-Clustered Index |
|---------|----------------|---------------------|
| Table row order | Rows physically ordered by index key | Rows stored separately |
| Count per table | **Maximum 1** (one physical order) | Many |
| Row access | Leaf page IS the row data | Leaf → row pointer → table lookup |
| Default in MySQL InnoDB | PRIMARY KEY | All other indexes |
| Default in PostgreSQL | None (all are non-clustered by default) | All indexes |

**MySQL/InnoDB implication:**
- The PRIMARY KEY IS the clustered index — choose it carefully
- Secondary indexes store the primary key value as the row pointer
- This means a secondary index lookup requires two B-tree traversals (secondary → PK → row)
- UUID primary keys cause poor insert performance (random page writes vs sequential)

---

### Q5. What is a composite index? What is the leftmost prefix rule?

**Answer:**
A composite index spans multiple columns. The order of columns matters.

```sql
CREATE INDEX idx_emp ON employees (dept_id, status, salary);
```

**Leftmost prefix rule:** A composite index can only be used starting from the leftmost column. Skipping a column breaks the chain.

```
Index: (dept_id, status, salary)

Can use index:
  WHERE dept_id = 10                          ✓ uses (dept_id)
  WHERE dept_id = 10 AND status = 'active'    ✓ uses (dept_id, status)
  WHERE dept_id = 10 AND status = 'active' AND salary > 50000  ✓ all 3

Cannot use index:
  WHERE status = 'active'                     ✗ skips dept_id
  WHERE salary > 50000                        ✗ skips both
  WHERE dept_id = 10 AND salary > 50000       ✗ skips status (partially used)
  -- dept_id used, but salary can't skip status
```

**Column order rule:** Put equality columns first, range columns last.

---

### Q6. What is a covering index?

**Answer:**
A covering index includes all columns referenced by a query — the query can be answered entirely from the index without accessing the table (called an **index-only scan**).

```sql
-- Query needs: status (WHERE), name, email (SELECT)
SELECT name, email FROM users WHERE status = 'active';

-- Regular index: must fetch name, email from table
CREATE INDEX idx_status ON users (status);
-- Plan: Index Scan → table lookup for name, email

-- Covering index: all needed columns in index
CREATE INDEX idx_status_covering ON users (status, name, email);
-- Or with INCLUDE (non-key columns, PostgreSQL/SQL Server):
CREATE INDEX idx_status_inc ON users (status) INCLUDE (name, email);
-- Plan: Index Only Scan — never touches the table
```

**When to use:** High-frequency queries where the extra index size is acceptable. Covering indexes can reduce I/O by 10-100x on large tables.

---

### Q7. How do you read and interpret EXPLAIN output?

**Answer:**

**MySQL EXPLAIN — key columns:**
```sql
EXPLAIN SELECT * FROM employees WHERE dept_id = 10 AND salary > 70000;
+----+-------+-----+------+---------+--------+-------+
| id | type  | key | rows | filtered| Extra  | ...   |
+----+-------+-----+------+---------+--------+-------+
|  1 | ref   | idx | 25   |  80.00  |Using where |   |
```

| Column | What to look for |
|--------|-----------------|
| `type` | `ALL` = full scan (bad), `ref/range/eq_ref/const` = index used (good) |
| `key` | Which index is used (NULL = no index) |
| `rows` | Estimated rows examined (lower = better) |
| `Extra` | `Using filesort` or `Using temporary` = potential problem |

**PostgreSQL EXPLAIN ANALYZE — key nodes:**
```
Seq Scan          → Full table scan (look for missing index)
Index Scan        → B-tree lookup + table access
Index Only Scan   → Covering index (no table access, fastest)
Bitmap Heap Scan  → Multiple index entries combined
Nested Loop       → Good for small outer tables
Hash Join         → Good for large equi-joins
Merge Join        → Good for pre-sorted large inputs
Sort              → Check if avoidable via index
```

---

### Q8. When does an index NOT get used by the query optimizer?

**Answer:**
The optimizer skips an index when:

**1. Function applied to the indexed column:**
```sql
WHERE YEAR(hire_date) = 2020     -- ✗ index on hire_date not used
WHERE UPPER(email) = 'ALICE'     -- ✗ index on email not used

-- Fix: rewrite without function
WHERE hire_date BETWEEN '2020-01-01' AND '2020-12-31'  -- ✓
WHERE email = 'alice'  -- ✓ (normalize data to lowercase)
```

**2. Leading wildcard in LIKE:**
```sql
WHERE name LIKE '%smith'         -- ✗ can't use B-tree from the right
WHERE name LIKE '%smith%'        -- ✗ same issue
WHERE name LIKE 'smith%'         -- ✓ prefix search uses B-tree
```

**3. Type mismatch / implicit cast:**
```sql
WHERE id = '42'    -- id is INT, '42' is VARCHAR → cast prevents index use
WHERE id = 42      -- ✓
```

**4. Low selectivity** (optimizer prefers full scan):
```sql
WHERE is_active = TRUE  -- If 95% of rows are active, index costs more than full scan
```

**5. Skipping leading column of composite index:**
```sql
-- Index: (dept_id, salary)
WHERE salary > 80000   -- ✗ skips dept_id
```

---

### Q9. What is a partial index? When would you use one?

**Answer:**
A partial index (WHERE-filtered index) only indexes rows that satisfy a specified condition. This creates a smaller, faster index.

```sql
-- Index only pending orders (instead of all orders)
CREATE INDEX idx_pending_orders
ON orders (customer_id, created_at)
WHERE status = 'pending';

-- Only used when query includes the same condition:
SELECT * FROM orders WHERE status = 'pending' AND customer_id = 42;  -- ✓ uses index
SELECT * FROM orders WHERE customer_id = 42;                         -- ✗ doesn't use it
```

**When to use:**
1. **Soft-deleted records** — index only non-deleted rows
2. **Status fields** — only active/pending rows are queried
3. **NULL exclusion** — skip NULL values that are never searched
4. **Recent data** — index only last 30/90 days

```sql
-- Index non-deleted users only (common soft-delete pattern)
CREATE INDEX idx_users_active ON users (email, last_login)
WHERE deleted_at IS NULL;

-- Makes SELECT ... WHERE deleted_at IS NULL AND email = ? use this tiny index
```

---

### Q10. What is an expression index (functional index)?

**Answer:**
An expression index indexes the result of a function or expression rather than the raw column value.

```sql
-- PostgreSQL: index on LOWER(email) for case-insensitive searches
CREATE INDEX idx_email_lower ON users (LOWER(email));

-- Now this query uses the index:
SELECT * FROM users WHERE LOWER(email) = 'alice@company.com';

-- MySQL 8.0+: expression index
CREATE INDEX idx_email_upper ON users ((UPPER(email)));

-- PostgreSQL: computed expression
CREATE INDEX idx_full_name ON employees ((first_name || ' ' || last_name));
SELECT * FROM employees WHERE (first_name || ' ' || last_name) = 'Alice Johnson';

-- Date extraction:
CREATE INDEX idx_hire_year ON employees (EXTRACT(YEAR FROM hire_date));
SELECT * FROM employees WHERE EXTRACT(YEAR FROM hire_date) = 2020;
```

---

### Q11. What is the N+1 query problem? How do you fix it?

**Answer:**
N+1 occurs when an application executes **1 query to get N parent records, then N separate queries to get each child's data**.

```sql
-- BAD: N+1 pattern
SELECT id FROM departments;  -- Returns 10 departments
-- Then for each department (10 queries!):
SELECT * FROM employees WHERE dept_id = ?;
-- Total: 1 + 10 = 11 queries
```

**Fixes:**

**Fix 1: JOIN (single query)**
```sql
SELECT d.id, d.name, e.id AS emp_id, e.name AS emp_name
FROM departments d
LEFT JOIN employees e ON e.dept_id = d.id;
-- 1 query, returns all data
```

**Fix 2: Eager loading with IN (2 queries)**
```sql
SELECT id FROM departments;
-- Returns: [1, 2, 3, ..., 10]
SELECT * FROM employees WHERE dept_id IN (1, 2, 3, ..., 10);
-- 2 queries total
```

**Fix 3: Application-level batch:**
Collect all foreign keys first, then fetch all related records in one query.

---

### Q12. What is the difference between EXPLAIN and EXPLAIN ANALYZE?

**Answer:**

| Feature | EXPLAIN | EXPLAIN ANALYZE |
|---------|---------|----------------|
| Executes query | ✗ No | ✓ Yes (actually runs it) |
| Shows | Estimated plan + costs | Estimated AND actual rows/time |
| Safe on large tables | ✓ Yes | ⚠ Slower (runs the query) |
| Most useful for | Quick plan check | Diagnosing wrong row estimates |

```sql
-- EXPLAIN: shows the plan without executing
EXPLAIN SELECT * FROM orders WHERE customer_id = 42;
-- cost=0.00..18.50 rows=100 (ESTIMATED)

-- EXPLAIN ANALYZE: executes and shows actual vs estimated
EXPLAIN ANALYZE SELECT * FROM orders WHERE customer_id = 42;
-- cost=0.00..18.50 rows=100  (ESTIMATED)
-- actual time=0.05ms rows=3  (ACTUAL — very different from estimated!)
-- → stale statistics; run ANALYZE to fix
```

**For PostgreSQL, most useful form:**
```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT ...;
-- BUFFERS shows cache hits vs disk reads
-- High shared_read = data not in cache (cold start or too large)
```

---

### Q13. How would you diagnose and fix a slow query?

**Answer:**
**Step-by-step process:**

1. **Identify the slow query** (slow query log, pg_stat_statements, APM tool)
2. **Run EXPLAIN ANALYZE** to see the plan
3. **Look for:** Full table scans, wrong row estimates, high sort cost, nested loops on large tables
4. **Fix based on diagnosis:**

```sql
-- Problem: Seq Scan on large table
-- Fix: Add index
CREATE INDEX idx_orders_customer ON orders (customer_id);

-- Problem: estimated rows = 1000, actual = 1,000,000 (stale stats)
-- Fix: Update statistics
ANALYZE orders;                        -- PostgreSQL
ANALYZE TABLE orders;                  -- MySQL
UPDATE STATISTICS orders WITH FULLSCAN; -- SQL Server

-- Problem: filesort on large result
-- Fix: Add index matching ORDER BY
CREATE INDEX idx_orders_date ON orders (order_date DESC);

-- Problem: function on indexed column
WHERE YEAR(created_at) = 2024
-- Fix: rewrite
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01'

-- Problem: SELECT * fetching unnecessary columns
-- Fix: select only needed columns (possibly covering index)
SELECT id, total FROM orders WHERE customer_id = 42;
```

---

### Q14. What is index selectivity and why does it matter?

**Answer:**
Selectivity measures how many distinct values an index column has, as a fraction of total rows:

```
Selectivity = COUNT(DISTINCT col) / COUNT(*)
Range: 0 (all values the same) to 1 (all values unique)
```

**High selectivity** (close to 1.0):
- Unique columns: email, user_id → selectivity = 1.0
- Index is very useful — quickly narrows to a few rows

**Low selectivity** (close to 0.0):
- Boolean columns: is_active (2 values) → selectivity ≈ 0.02
- Status with few values: gender, status
- Index may be ignored — optimizer prefers full scan if >5-10% of rows match

```sql
-- Check selectivity
SELECT
    COUNT(DISTINCT dept_id) AS distinct_values,
    COUNT(*) AS total_rows,
    ROUND(COUNT(DISTINCT dept_id) * 1.0 / COUNT(*), 4) AS selectivity
FROM employees;
-- If 0.02, index on dept_id alone probably not used for most queries
```

**Rule:** For low-selectivity columns, use them as a **second or third column** in a composite index, not the first.

---

### Q15. What is index bloat and how do you fix it?

**Answer:**
Index bloat occurs when an index accumulates dead space (from deleted or updated rows) and becomes larger and slower than it should be.

**Causes:**
- Heavy UPDATE workload (old row versions remain in index pages)
- Heavy DELETE workload (pages become sparse)
- No regular VACUUM (PostgreSQL) or OPTIMIZE TABLE (MySQL)

**Detection:**
```sql
-- PostgreSQL: find bloated indexes
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    idx_scan AS times_used
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;

-- MySQL: check fragmentation
SELECT TABLE_NAME, DATA_FREE, DATA_LENGTH
FROM information_schema.TABLES WHERE TABLE_NAME = 'employees';
```

**Fix:**
```sql
-- PostgreSQL: rebuild index
REINDEX INDEX idx_employees_email;
REINDEX INDEX CONCURRENTLY idx_employees_email;  -- Non-blocking (PG 12+)

-- PostgreSQL: VACUUM to reclaim space
VACUUM ANALYZE employees;
VACUUM FULL employees;       -- Compact, but requires exclusive lock

-- MySQL: rebuild table + indexes
OPTIMIZE TABLE employees;
ALTER TABLE employees ENGINE=InnoDB;  -- Full rebuild
```

---

### Q16. Design an optimal indexing strategy for this query:

```sql
SELECT name, email, created_at
FROM users
WHERE status = 'active' AND country = 'US' AND age BETWEEN 18 AND 35
ORDER BY created_at DESC
LIMIT 20;
```

**Answer:**

**Analysis:**
- `status = 'active'` — equality (low selectivity)
- `country = 'US'` — equality (medium selectivity)
- `age BETWEEN 18 AND 35` — range
- `ORDER BY created_at DESC` — sort
- `SELECT: name, email, created_at` — these should be in the index (covering)

**Strategy — equality columns first, range last:**
```sql
-- Option 1: Composite index (equality first, range last, covering)
CREATE INDEX idx_users_optimal
ON users (status, country, age, created_at DESC)
INCLUDE (name, email);   -- covering columns (PostgreSQL/SQL Server)

-- Why this order:
-- status + country → equality filters narrow result to, say, 5% of rows
-- age → range filter within that 5%
-- created_at → ORDER BY can use index (no filesort needed)
-- INCLUDE(name, email) → covering index, no table lookup

-- Option 2: If status has low selectivity (most users are 'active'):
CREATE INDEX idx_users_country_age
ON users (country, age)
WHERE status = 'active'  -- Partial index (smaller, faster)
INCLUDE (name, email, created_at);
```

**Expected plan:** Index Only Scan (all columns covered) + Limit.

---

### Q17. What is the difference between optimistic and pessimistic locking? When does each affect index usage?

**Answer:**

**Pessimistic locking** acquires row locks immediately:
```sql
BEGIN;
SELECT * FROM products WHERE id = 42 FOR UPDATE;  -- Locks row immediately
UPDATE products SET stock = stock - 1 WHERE id = 42;
COMMIT;

-- FOR UPDATE causes the index to be used for a direct lookup
-- The locked row is pinned in buffer pool
```

**Optimistic locking** uses a version column, no row locks:
```sql
-- Read current version
SELECT id, stock, version FROM products WHERE id = 42;
-- Returns: stock=10, version=5

-- Update only if version hasn't changed
UPDATE products SET stock = 9, version = 6
WHERE id = 42 AND version = 5;

-- Check affected rows: 0 = conflict (retry), 1 = success
```

**Index impact:**
- `FOR UPDATE` / `FOR SHARE` uses the index for row lookup, then places a lock on that row
- In high-contention scenarios, index lookups can cause lock wait chains
- Optimistic locking does more reads but avoids lock waits — better for read-heavy, low-contention data

---

### Q18. What are the key differences in indexing strategy between OLTP and OLAP workloads?

**Answer:**

| Concern | OLTP | OLAP |
|---------|------|------|
| Query pattern | Many small reads/writes | Few complex reads |
| Index count | Few targeted indexes (~5-10) | Many indexes (or column store) |
| Write performance | Critical — minimize indexes | Less critical |
| Scan type | Index seeks (point lookups) | Full scans or column scans |
| Index type | B-tree | B-tree + Columnar/Bitmap |
| Partitioning | By access key | By date/time (analytical range scans) |

```sql
-- OLTP: narrow index for point lookups
CREATE INDEX idx_orders_customer ON orders (customer_id, status);  -- Narrow, point lookup

-- OLAP: covering index for wide analytical queries
CREATE INDEX idx_orders_analytical
ON orders (order_date, status, region)
INCLUDE (customer_id, total, product_id);

-- OLAP databases (Redshift, BigQuery, Snowflake): use column-store format
-- No traditional B-tree indexes; data sorted by sort key; columnar compression
```

---

### Q19. An INSERT operation is suddenly 10x slower than before. What would you check?

**Answer:**

**Checklist:**

1. **Too many indexes** — each INSERT updates all indexes on the table
```sql
SHOW INDEX FROM orders;  -- MySQL: how many indexes?
SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'orders';  -- PostgreSQL
-- If 15+ indexes on a heavily written table, consider removing unused ones
```

2. **Fragmented indexes** — bloated index pages cause slow writes
```sql
OPTIMIZE TABLE orders;   -- MySQL rebuild
REINDEX TABLE orders;    -- PostgreSQL
```

3. **Lock contention** — another long transaction holding table/row locks
```sql
SHOW PROCESSLIST;        -- MySQL
SELECT * FROM pg_stat_activity WHERE state = 'active';  -- PostgreSQL
```

4. **Primary key choice** — random UUID PKs cause random B-tree page writes
```sql
-- UUID as PK: random insertions → page splits → bloat
-- Fix: use ULID or sequential UUIDs (UUID v7), or auto-increment INT
```

5. **Triggers** — a trigger added to the table is executing heavy logic
6. **Disk I/O saturation** — check OS metrics (iostat, pgBadger)
7. **Auto-ANALYZE running** — VACUUM/ANALYZE competes for I/O

---

### Q20. How do you find unused indexes and when should you drop them?

**Answer:**

**Finding unused indexes:**
```sql
-- PostgreSQL: track index usage since last reset
SELECT
    indexrelname AS index_name,
    relname AS table_name,
    idx_scan AS times_used,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
WHERE idx_scan = 0                           -- Never used
  AND relname NOT LIKE 'pg_%'               -- Not system tables
ORDER BY pg_relation_size(indexrelid) DESC;  -- Biggest first

-- MySQL: via performance_schema
SELECT object_schema, object_name, index_name, count_read
FROM performance_schema.table_io_waits_summary_by_index_usage
WHERE count_read = 0 AND object_schema != 'mysql';
```

**Should you drop it?**

| Consider dropping | Keep even if "unused" |
|------------------|-----------------------|
| Never used since last pg_stat_reset | Unique constraints (enforce integrity) |
| High write-to-read ratio table | Foreign key support indexes |
| Large index (significant storage) | Backup/restore/ETL indexes |
| Duplicate of another index | Seasonal queries (may be used rarely) |

```sql
-- Always check: when were stats last reset?
SELECT stats_reset FROM pg_stat_database WHERE datname = current_database();

-- Test by disabling first (SQL Server), then drop if no performance change
-- PostgreSQL: no disable — use pg_stat_reset() and monitor for a week
```
