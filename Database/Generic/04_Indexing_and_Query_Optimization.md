# Indexing and Query Optimization — System Design Perspective

## Easy (Q1–Q7)

---

**Q1. What is a database index and what is the fundamental trade-off?**

An index is a separate data structure that maintains a sorted or hashed mapping from column values to physical row locations, allowing the database to find rows without scanning every row in the table.

```
Without index: Seq Scan
  "Find orders where customer_id = 1234"
  → Read every row in orders table (e.g., 100M rows)
  → Check each row: does customer_id = 1234?
  → O(N) → slow

With index on customer_id: Index Scan
  → Navigate B-tree (3 levels) to find customer_id = 1234
  → Retrieve only matching row pointers
  → O(log N) → fast
```

**The fundamental trade-off:**

| Benefit | Cost |
|---|---|
| Fast reads (SELECT) | Slower writes (every INSERT/UPDATE/DELETE updates the index) |
| Fast sorting (ORDER BY) | Extra storage (index occupies disk space) |
| Fast JOINs | Memory overhead (indexes loaded into buffer pool) |
| Can avoid table access (covering index) | Maintenance (VACUUM, bloat, fragmentation) |

**Rule of thumb:** Index columns that appear in WHERE, JOIN ON, and ORDER BY clauses for your most frequent queries. Avoid indexing columns with low cardinality (status with 3 values) or columns rarely queried.

---

**Q2. How does a B-tree index work and why is it the default?**

A B-tree (balanced tree) is a self-balancing sorted tree where all leaf nodes are at the same depth and linked together in sorted order.

```
Structure (simplified, branching factor ≈ 100):
                [50 | 150 | 250]                    ← Root
              /       |        \
    [10|20|30]  [60|80|100|120]  [160|200|230]      ← Internal nodes
        |              |                |
  leaf pages      leaf pages       leaf pages        ← Leaf nodes (linked list)
  [row pointers]  [row pointers]   [row pointers]

Lookup of customer_id = 80:
  1. Read root: 80 > 50, 80 < 150 → middle child
  2. Read internal: 80 = 80 → found in second leaf group
  3. Read leaf page: get row pointer → fetch row from heap (clustered) or via pointer
  Total I/Os: tree height = O(log_B N) ≈ 3–4 for million-row tables
```

**Why B-tree is default:**
- Supports: `=`, `<`, `>`, `BETWEEN`, `LIKE 'prefix%'` (but NOT `LIKE '%suffix'`)
- Supports range scans efficiently (leaf nodes are a linked list → just scan forward)
- Supports ORDER BY without re-sorting
- Self-balancing → no manual maintenance needed
- Height ≈ 3–4 for most real-world tables (100M rows → 4 levels)

---

**Q3. What is a covering index and why is it powerful for read performance?**

A covering index contains all columns needed by a query — the query can be answered entirely from the index without touching the main table (heap).

```sql
-- Without covering index:
CREATE INDEX idx_orders_cust ON orders (customer_id);

SELECT order_id, order_date, amount
FROM orders
WHERE customer_id = 1234;

-- Plan: Index Scan on idx_orders_cust
--       → finds row pointers via index
--       → fetches heap pages for order_id, order_date, amount (extra I/O!)

-- With covering index (INCLUDE clause):
CREATE INDEX idx_orders_cust_covering ON orders (customer_id)
    INCLUDE (order_id, order_date, amount);

-- Plan: Index Only Scan (no heap access at all!)
--       → All needed columns are in the index leaf nodes
--       → Zero heap I/O
```

**When covering indexes matter most:**
- High-frequency queries that read only a few columns from wide tables
- Tables where heap is on slow storage but index fits in buffer pool (RAM)
- Queries that would be Index Scan but table is large (fetching heap pages is the bottleneck)

**Storage trade-off:** INCLUDE columns inflate index leaf nodes (larger index). Internal nodes remain small (only the key column). Accept the larger index for the I/O savings on hot queries.

---

**Q4. What is index selectivity and why does it matter for query optimization?**

Selectivity is the fraction of rows an index predicate matches:

```
selectivity = distinct_values / total_rows

High selectivity (good for index):
  customer_id: 1M distinct values in 1M rows → selectivity = 1.0
  → Index lookup returns ~1 row → very useful

Low selectivity (index may not help):
  status (pending/shipped/delivered): 3 distinct values in 1M rows
  → selectivity = 0.000003 per value
  → "pending" status might match 400K rows (40% of table) → index scan + heap fetch = slower than seq scan
  → Query planner may choose Seq Scan over index (correct choice)
```

**PostgreSQL planner threshold:**
The planner estimates selectivity using column statistics (`pg_stats`). If the estimated fraction of rows returned is > ~5–10%, a sequential scan is preferred. Below that, an index scan wins.

```sql
-- Check column statistics:
SELECT attname, n_distinct, correlation
FROM pg_stats
WHERE tablename = 'orders' AND attname = 'customer_id';
-- n_distinct: -1.0 means all values are unique (cardinality = total rows)
-- correlation: 1.0 means column is physically sorted (range scans very fast)
```

**Implications for index design:**
- High-cardinality columns (user_id, order_id, email) → excellent index candidates
- Low-cardinality columns (status, boolean, gender) → poor standalone index; use as secondary column in composite index
- Partial index: index only the rare/selective values of a low-cardinality column

---

**Q5. What is a partial index and when should you use it?**

A partial index only indexes rows that satisfy a WHERE condition, making the index smaller and faster for the specific queries it serves.

```sql
-- Full index: indexes all 10M orders including 9M completed ones
CREATE INDEX idx_orders_status ON orders (status, created_at);

-- Partial index: only indexes the ~100K pending orders (1% of table)
CREATE INDEX idx_pending_orders ON orders (created_at, customer_id)
WHERE status = 'pending';

-- Query that benefits:
SELECT order_id, customer_id FROM orders
WHERE status = 'pending' AND created_at > NOW() - INTERVAL '1 day';
-- Uses partial index → scans only 100K row index (vs 10M full index)
-- Index is tiny → fits entirely in buffer pool (RAM) → extremely fast

-- Query that doesn't use partial index:
SELECT * FROM orders WHERE status = 'completed';
-- Partial index condition not satisfied → falls back to seq scan or full index
```

**Ideal use cases:**
- Soft-delete pattern: `WHERE deleted_at IS NULL` — index only live rows (typically 99%)
- Queue processing: `WHERE status = 'pending'` — index only work items not yet processed
- Recent data: `WHERE created_at > '2024-01-01'` — index only recent rows for dashboards
- Inactive users: `WHERE is_active = false` — if you have rare-value queries on a boolean

---

**Q6. How do composite indexes work and what is the leftmost prefix rule?**

A composite (multi-column) index stores rows sorted by the first column, then by the second within ties, then by the third, and so on.

```sql
CREATE INDEX idx_orders_composite ON orders (customer_id, order_date, status);
```

**The leftmost prefix rule:** The index can only be used for queries that include a leading subset of columns starting from the leftmost column.

```
Index: (customer_id, order_date, status)

Can use index:
  WHERE customer_id = 5                    -- leftmost column only ✓
  WHERE customer_id = 5 AND order_date > '2024-01-01'          ✓
  WHERE customer_id = 5 AND order_date = '2024-01-15' AND status = 'shipped'  ✓

Cannot use index (or uses it poorly):
  WHERE order_date > '2024-01-01'          -- skips leftmost column ✗
  WHERE status = 'shipped'                 -- only rightmost column ✗
  WHERE order_date = '2024-01-15' AND status = 'shipped'  -- skips customer_id ✗
  
Partial use (range on middle column stops further use):
  WHERE customer_id = 5 AND order_date > '2024-01-01'  -- uses both ✓
  WHERE customer_id = 5 AND order_date > '2024-01-01' AND status = 'shipped'
    -- uses customer_id + order_date for filtering; status not used as index condition
    -- but can be used as a filter after index range scan
```

**Column ordering rule:** Equality conditions first, range condition last.
```sql
-- For query: WHERE customer_id = 5 AND order_date > '2024-01-01' AND status = 'shipped'
-- Best index: (customer_id, status, order_date)  ← equality columns first, range last
--   → customer_id=5 AND status='shipped' pins the range tightly before scanning order_date range
```

---

**Q7. What are the most common reasons a query doesn't use an index even when one exists?**

1. **Function applied to the indexed column:**
```sql
-- Bad: wraps column in function → index cannot be used
WHERE YEAR(order_date) = 2024
WHERE LOWER(email) = 'alice@x.com'
WHERE DATE_TRUNC('month', created_at) = '2024-01-01'

-- Fix: rewrite to directly reference the column
WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01'
WHERE email = 'alice@x.com'  -- store emails already lowercased, or use expression index
WHERE created_at >= '2024-01-01' AND created_at < '2024-02-01'
```

2. **Implicit type cast:**
```sql
-- Column is VARCHAR, value is integer → implicit cast prevents index use
WHERE user_id = 1234        -- user_id is VARCHAR → cast to VARCHAR first → no index
WHERE user_id = '1234'      -- matches type → index used ✓
```

3. **Leading wildcard LIKE:**
```sql
WHERE name LIKE '%alice%'   -- cannot use B-tree (doesn't start from left boundary)
WHERE name LIKE 'alice%'    -- CAN use B-tree (starts from left) ✓
-- Fix for contains-search: use full-text index (GIN/tsvector) or pg_trgm trigram index
```

4. **Low selectivity:**
```sql
WHERE is_active = true      -- 95% of rows are active → seq scan wins
-- Fix: partial index (only index active=false if that's the rare case you query)
```

5. **Small table:**
```sql
-- 1000-row table → full scan reads 10 pages → faster than index lookup
-- Planner correctly chooses seq scan
```

6. **Outdated statistics:**
```sql
ANALYZE orders;  -- update planner statistics (run after large data changes)
```

---

## Medium (Q8–Q15)

---

**Q8. How do you identify and fix slow queries in a production PostgreSQL database?**

**Step 1: Identify slow queries**
```sql
-- pg_stat_statements: aggregate statistics per query (requires extension)
SELECT query,
       calls,
       round(mean_exec_time::numeric, 2) AS mean_ms,
       round(total_exec_time::numeric, 2) AS total_ms,
       rows
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;

-- Also check for queries running right now:
SELECT pid, now() - query_start AS duration, state, query
FROM pg_stat_activity
WHERE state = 'active' AND query_start < NOW() - INTERVAL '5 seconds'
ORDER BY duration DESC;
```

**Step 2: Analyze the slow query**
```sql
-- EXPLAIN ANALYZE: runs the query and shows actual vs estimated stats
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT o.order_id, c.name, SUM(oi.amount)
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.order_date > '2024-01-01'
GROUP BY o.order_id, c.name;

-- Key things to look for in the plan:
-- Seq Scan on large table → missing index
-- Rows=100 estimate vs Actual Rows=100000 → stale statistics → ANALYZE
-- Hash Join with high memory → work_mem too small → sort/hash spilling to disk
-- Nested Loop on large outer set → N+1 loop issue → missing index on inner table
-- Buffers: read=5000 hit=100 → very low cache hit → data not in shared_buffers
```

**Step 3: Fix**
```sql
-- Missing index:
CREATE INDEX CONCURRENTLY ON orders (order_date, customer_id);

-- Stale statistics:
ANALYZE orders;
ANALYZE customers;

-- Memory for sort/hash operations:
SET work_mem = '64MB';  -- per operation per query, can multiply with parallel workers

-- Rewrite the query (correlated subquery → JOIN):
-- Bad:
SELECT name FROM customers c
WHERE (SELECT SUM(amount) FROM orders WHERE customer_id = c.id) > 1000;
-- Good:
SELECT c.name FROM customers c
JOIN (SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id) o
  ON c.id = o.customer_id AND o.total > 1000;
```

---

**Q9. Explain the N+1 query problem and its solutions in a database context.**

**The N+1 problem:** One query returns N rows, then for each row, one more query is issued. Total: N+1 queries. At scale, this is the most common database performance killer.

```python
# N+1: 1 query for orders + N queries for customer name
orders = db.execute("SELECT order_id, customer_id, amount FROM orders LIMIT 100")
# Returns 100 rows

for order in orders:  # 100 iterations
    customer = db.execute(
        "SELECT name FROM customers WHERE id = %s", order.customer_id
    )
    # 100 individual queries
# Total: 101 queries → 100 round trips to database
```

**Solution 1: JOIN (single query)**
```sql
SELECT o.order_id, c.name, o.amount
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LIMIT 100;
-- 1 query, 1 round trip
```

**Solution 2: Batch IN query (when JOIN is not clean)**
```python
orders = db.execute("SELECT order_id, customer_id, amount FROM orders LIMIT 100")
customer_ids = [o.customer_id for o in orders]

customers = db.execute(
    "SELECT id, name FROM customers WHERE id = ANY(%s)", [customer_ids]
)
customer_map = {c.id: c.name for c in customers}
# Total: 2 queries (constant, not N+1)
```

**Solution 3: Avoid the second query with denormalization**
```sql
-- Store customer_name on the orders table (denormalized copy at order creation time)
SELECT order_id, customer_name, amount FROM orders LIMIT 100;
-- 1 query, no JOIN needed
-- Trade-off: if customer changes name, historical orders still show old name (often correct for orders!)
```

**Solution 4: DataLoader pattern (GraphQL / API context)**
```
Batch requests within one tick of the event loop:
  Request A: getCustomer(1)  ──┐
  Request B: getCustomer(2)  ──┤  All batched into:
  Request C: getCustomer(3)  ──┘  SELECT * FROM customers WHERE id IN (1,2,3)
```

---

**Q10. How does query plan caching work and when does it cause problems?**

**Query plan caching:** When the database first encounters a parameterized query, it compiles an execution plan. That plan is cached and reused for subsequent executions with different parameter values.

```sql
-- Prepared statement: plan cached after first execution
PREPARE get_orders(INT) AS
    SELECT * FROM orders WHERE customer_id = $1 ORDER BY order_date DESC LIMIT 50;

EXECUTE get_orders(1234);  -- Plan compiled for customer_id = 1234
EXECUTE get_orders(5678);  -- Reuses cached plan (no re-planning)
```

**When plan caching causes problems:**

**Parameter sniffing / plan instability:**
```
Customer 1: has 3 orders → index scan plan is optimal
Customer 2: has 1M orders → seq scan plan is optimal

If the plan was compiled for customer 1's data distribution (index scan):
  When executed for customer 2: uses wrong plan (index scan on 1M rows) → slow

In PostgreSQL: prepared statements use the first 5 executions as "generic" estimates
After 5 uses, a generic plan is created based on average statistics (not per-parameter)
```

**Fix options:**
```sql
-- Option 1: SET plan_cache_mode = force_custom_plan;
--            Force re-planning on every execution (no caching)
--            More CPU overhead but always optimal plan

-- Option 2: Use parameterized queries without PREPARE (PostgreSQL inlines statistics)
SELECT * FROM orders WHERE customer_id = 1234 LIMIT 50;
-- Direct query, not prepared → always fresh plan based on actual value statistics

-- Option 3: Use hints (PostgreSQL: pg_hint_plan extension)
/*+ IndexScan(orders idx_orders_cust) */ SELECT ...
```

**When plan caching is helpful:**
- Many short, repeated queries with similar data distributions
- OLTP workloads where the same queries repeat thousands of times per second
- Avoids parsing + planning overhead (~0.1–1ms per query) for every execution

---

**Q11. What is the difference between a hash join, nested loop join, and merge join? When does each apply?**

**Nested Loop Join:**
```
For each row in outer table:
  For each matching row in inner table (using index or scan)
  
Algorithm:
  FOR each order o:
    LOOKUP customer c WHERE c.id = o.customer_id  ← uses index on c.id

Time complexity: O(N * log M) if inner has index; O(N * M) without index
Best for: small outer table, indexed inner table
Bad for: large tables without indexes

Example plan output:
  -> Nested Loop
     -> Seq Scan on orders (outer, 100 rows estimated)
     -> Index Scan on customers using idx_customers_pk (inner)
```

**Hash Join:**
```
Phase 1 (build): scan the smaller table, build a hash table in memory (keyed by join column)
Phase 2 (probe): scan the larger table, probe hash table for each row

Time complexity: O(N + M) (two passes)
Memory: proportional to the smaller table
Best for: large tables without usable indexes; equi-joins only (not <, >)
Bad for: when smaller table doesn't fit in work_mem (spills to disk → slow)

Config: work_mem controls hash table size in memory
        Low work_mem → hash spill → slow → increase work_mem
```

**Merge Join:**
```
Phase 1: sort both tables by join key (or use existing sorted index)
Phase 2: merge sorted lists (like merge sort's combine step)

Time complexity: O(N log N + M log M) if sorting needed; O(N + M) if already sorted
Best for: large tables that are already sorted (sorted indexes) or sorted CTEs
Output: already sorted by join key → good if ORDER BY follows

EXPLAIN shows:
  -> Merge Join
     -> Index Scan on orders using idx_customer_id (provides sorted output)
     -> Index Scan on customers using idx_customer_pk (provides sorted output)
     -> Merge → no extra sort needed
```

**Planner chooses based on:**
- Table sizes (from statistics)
- Available indexes
- Available work_mem
- Whether output needs to be sorted

---

**Q12. How do you use EXPLAIN ANALYZE effectively to diagnose a slow query?**

```sql
EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT TEXT)
SELECT c.name, COUNT(o.order_id), SUM(oi.amount)
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.order_date > '2024-01-01'
GROUP BY c.customer_id, c.name
ORDER BY SUM(oi.amount) DESC
LIMIT 10;
```

**Reading the output:**

```
Hash Aggregate  (cost=15000..15010 rows=10 width=64)
                (actual time=2534.123..2534.145 rows=10 loops=1)
  Buffers: shared hit=450 read=5200              ← 5200 disk reads (bad: data not in cache)
  ->  Hash Join  (cost=1200..14500 rows=100000 width=48)
                 (actual time=150.2..2200.5 rows=850000 loops=1)  ← rows estimate: 100K, actual: 850K → bad estimates!
        Hash Cond: (o.customer_id = c.customer_id)
        Buffers: shared hit=400 read=4800
        ->  Seq Scan on orders o               ← Seq Scan on orders (needs index?)
              (cost=0..8000 rows=100000 width=28)
              (actual time=0.1..450.3 rows=850000 loops=1)
              Filter: (order_date > '2024-01-01'::date)
              Rows Removed by Filter: 150000   ← 150K rows filtered after scan → index would help
        ->  Hash  (cost=800..800 rows=32000 width=28)
                  (actual time=120.1..120.1 rows=32000 loops=1)
              Buckets: 32768  Batches: 1  Memory Usage: 2048kB   ← in memory, OK
              ->  Seq Scan on customers c  (cost=0..800 rows=32000 width=28)
```

**What to look for:**

| Signal | Meaning | Fix |
|---|---|---|
| Seq Scan on large table | Missing index | Add index on filter/join column |
| `actual rows` >> `rows estimate` | Stale statistics | `ANALYZE table_name` |
| `Buffers: read=N` high | Low cache hit rate | Increase shared_buffers; add RAM |
| Hash Join `Batches: >1` | Hash spill to disk | Increase `work_mem` |
| `loops=N` with N > 1 | Nested loop on large outer | Missing index on inner table |
| Filter: Rows Removed >> rows returned | Index would prune more | Add index that matches WHERE |

---

**Q13. What are expression/functional indexes and when do they save queries that would otherwise miss indexes?**

A functional index stores the result of a function applied to a column, rather than the raw column value.

```sql
-- Common problem: query uses LOWER() for case-insensitive search
SELECT * FROM users WHERE LOWER(email) = 'alice@x.com';
-- Regular index on email CANNOT be used (function wraps column)

-- Solution: functional index on LOWER(email)
CREATE INDEX idx_users_email_lower ON users (LOWER(email));
-- Now the index stores LOWER(email) values
-- Query planner can use this index for WHERE LOWER(email) = ...
```

**More use cases:**

```sql
-- Extract year from date (avoid YEAR() wrapping):
CREATE INDEX idx_orders_year ON orders ((EXTRACT(YEAR FROM order_date)::INT));
SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date)::INT = 2024;

-- JSON field extraction (JSONB):
CREATE INDEX idx_products_color ON products ((attributes->>'color'));
SELECT * FROM products WHERE attributes->>'color' = 'red';

-- Computed hash for long columns:
CREATE INDEX idx_users_phone_hash ON users (hashtext(phone_number));
SELECT * FROM users WHERE hashtext(phone_number) = hashtext('555-1234') AND phone_number = '555-1234';

-- Partial functional index (combine both features):
CREATE INDEX idx_active_users_email ON users (LOWER(email))
WHERE deleted_at IS NULL;
-- Only indexes active users with their lowercased email
```

**Important:** For the planner to use a functional index, the query's WHERE clause must match the expression **exactly** as defined in the index.

---

**Q14. How does connection overhead affect query performance at scale, and how is it measured?**

**Connection overhead components:**
```
1. TCP handshake: 1–3ms (if local network) or 50–200ms (cross-region)
2. TLS negotiation: 2–10ms (per new TLS connection)
3. PostgreSQL auth: 1–5ms (password verification, session setup)
4. First query plan: 0.1–2ms (parsing, planning, not cached yet)

Total per new connection: 5–50ms overhead before first query executes
```

**At scale:**
```
Application: 1000 concurrent requests, each creating a new connection
1000 × 20ms connection overhead = 20 seconds of overhead per second
→ Connection setup overhead > actual query execution time

Measurement:
SELECT count(*) FROM pg_stat_activity;  -- current connection count
SELECT count(*) FROM pg_stat_activity WHERE state = 'idle';  -- idle connections (wasted memory)
SELECT pid, state, query_start, now()-query_start AS age FROM pg_stat_activity ORDER BY age DESC;
```

**Impact on database:**
```
Each idle PostgreSQL connection:
  ~5–10MB RAM (process overhead)
  Counts against max_connections limit
  Holds locks on shared memory structures

1000 idle connections = 5–10GB RAM just for connection overhead
```

**Mitigation with PgBouncer:**
```
Application:        PgBouncer:         PostgreSQL:
1000 app threads → 1000 PgBouncer sockets → 20 real PG connections

Transaction mode: PG connection returned to pool after each COMMIT/ROLLBACK
→ 20 connections serve 1000 requests/second if queries are < 20ms each
→ Connection overhead eliminated from critical path

Measure effectiveness:
  Without PgBouncer: pg_stat_activity shows 500+ connections
  With PgBouncer: pg_stat_activity shows 20 connections, pgbouncer SHOW POOLS shows thousands of clients
```

---

**Q15. What is index bloat and how does it affect performance?**

Index bloat occurs when index pages contain many dead entries (from deleted or updated rows) that have not been reclaimed. The index grows larger than necessary, requiring more I/O to traverse.

**Cause in PostgreSQL (MVCC):**
```
UPDATE orders SET status = 'shipped' WHERE order_id = 1234
  → Old row version marked dead (xmax set)
  → New row version inserted
  → Index on status now has: one entry pointing to old dead row + one to new row
  → Dead index entry not immediately removed

Over millions of updates: index fills with dead entries
B-tree height increases → more I/Os per lookup
Index pages are mostly empty space → wasted memory/disk
```

**Measuring index bloat:**
```sql
-- pgstattuple extension:
CREATE EXTENSION pgstattuple;
SELECT * FROM pgstatindex('idx_orders_status');
-- leaf_fragmentation: 45% → 45% of leaf pages are wasted dead space
-- avg_leaf_density: 55% → pages are only 55% full

-- Simple approximation:
SELECT pg_size_pretty(pg_relation_size('orders')) AS table_size,
       pg_size_pretty(pg_relation_size('idx_orders_status')) AS index_size;
-- If index is larger than the table → suspect bloat
```

**Remediation:**
```sql
-- Option A: REINDEX (rebuilds index from scratch, blocks reads/writes briefly)
REINDEX INDEX idx_orders_status;

-- Option B: REINDEX CONCURRENTLY (PostgreSQL 12+, no blocking)
REINDEX INDEX CONCURRENTLY idx_orders_status;

-- Option C: Regular VACUUM (reclaims dead tuples, partially defragments)
VACUUM VERBOSE orders;
VACUUM ANALYZE orders;  -- also updates statistics

-- Prevention: tune autovacuum to run more aggressively
ALTER TABLE orders SET (
    autovacuum_vacuum_scale_factor = 0.05,  -- vacuum when 5% of rows are dead
    autovacuum_vacuum_cost_delay = 2        -- reduce throttling
);
```

---

## Hard (Q16–Q20)

---

**Q16. Design an indexing strategy for a 1-billion-row events table that supports both point lookups and time-range analytics.**

**Table definition:**
```sql
CREATE TABLE events (
    event_id    BIGINT       PRIMARY KEY,
    user_id     BIGINT       NOT NULL,
    event_type  TEXT         NOT NULL,   -- 'click', 'view', 'purchase', 'login'
    page        TEXT,
    amount      NUMERIC(10,2),           -- null for non-purchase events
    created_at  TIMESTAMPTZ  NOT NULL
) PARTITION BY RANGE (created_at);

-- Monthly partitions (auto-managed with pg_partman)
CREATE TABLE events_2024_01 PARTITION OF events
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
-- ... one partition per month
```

**Access patterns:**
```
1. Point lookup by event_id:                           PRIMARY KEY (auto) ✓
2. User's recent events: user_id + created_at DESC     → composite index
3. All purchases > $100 recently:                       → partial index on amount
4. Analytics: event_type distribution per day:          → no index (partition pruning + seq scan is fine for analytics)
5. User's events of a specific type:                   → composite index
```

**Index strategy:**
```sql
-- Per partition (inherited by all partitions via parent table index):

-- Pattern 2 + 5: user's events (with optional type filter)
-- Equality first (user_id), range last (created_at), filter (event_type)
CREATE INDEX ON events (user_id, event_type, created_at DESC)
    INCLUDE (page, amount);
-- Covers: WHERE user_id=X, WHERE user_id=X AND event_type='purchase', ORDER BY created_at

-- Pattern 3: recent purchase amounts (analytical, rare)
CREATE INDEX ON events (created_at DESC, amount)
WHERE event_type = 'purchase' AND amount > 100;
-- Partial index: only ~1% of events are large purchases → tiny index

-- No index on event_type alone (low cardinality, analytics uses partition pruning + seq scan)
-- No index on page (high cardinality but point-lookup pattern not required)
```

**Query patterns and their plans:**
```sql
-- User's last 50 events:
SELECT * FROM events WHERE user_id = 12345 ORDER BY created_at DESC LIMIT 50;
-- → Partition pruning on created_at if range given; otherwise hits all partitions
-- → Index (user_id, event_type, created_at DESC) → fast even across partitions

-- User's last 50 purchases:
SELECT * FROM events WHERE user_id = 12345 AND event_type = 'purchase'
ORDER BY created_at DESC LIMIT 50;
-- → Uses composite index (user_id=12345, event_type='purchase') → range scan on created_at

-- Analytics: event type distribution for January 2024:
SELECT event_type, COUNT(*) FROM events
WHERE created_at >= '2024-01-01' AND created_at < '2024-02-01'
GROUP BY event_type;
-- → Partition pruning: only events_2024_01 scanned
-- → Seq scan on single partition (much faster than index scan for full-partition aggregation)
-- → No index needed here (columnar OLAP would be better for this pattern if frequent)
```

---

**Q17. A query takes 30 seconds in production but runs in 200ms in your test environment. How do you diagnose and fix it?**

**Common causes of environment divergence:**

**1. Data volume difference (most common)**
```sql
-- Check table sizes:
SELECT relname, n_live_tup, n_dead_tup
FROM pg_stat_user_tables
WHERE relname IN ('orders', 'customers', 'order_items');
-- Production: orders = 500M rows
-- Test: orders = 10K rows
-- At 10K rows: seq scan is fast; at 500M rows: seq scan = minutes
-- Fix: ensure test data has same magnitude (or test with production EXPLAIN)
```

**2. Stale statistics in production**
```sql
-- Production table grew 10× since last ANALYZE → planner uses wrong cardinality estimates
SELECT last_analyze, last_autoanalyze FROM pg_stat_user_tables WHERE relname = 'orders';
-- If last_analyze is weeks ago on a rapidly changing table:
ANALYZE orders;  -- refresh statistics
-- Re-run query and compare plan
```

**3. Index exists in test but not production (or vice versa)**
```sql
-- Compare indexes:
-- Production:
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'orders';
-- Test:
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'orders';
-- If index missing in production: CREATE INDEX CONCURRENTLY ...
```

**4. Different PostgreSQL settings**
```sql
-- Check key settings:
SHOW work_mem;         -- production: 4MB (too small → hash spill); test: 256MB
SHOW shared_buffers;   -- production buffer cache (table hot?)
SHOW max_parallel_workers_per_gather;  -- parallel query enabled?
-- Fix: match settings or increase work_mem:
SET work_mem = '64MB'; EXPLAIN ANALYZE ...  -- test effect of higher work_mem
```

**5. Lock contention**
```sql
-- Is something blocking the query?
SELECT pid, granted, mode, relation::regclass, query
FROM pg_locks l JOIN pg_stat_activity a USING (pid)
WHERE NOT granted OR mode = 'ExclusiveLock';
-- If blocking: identify the holder, fix the long transaction
```

**6. Different query plans due to pg_stat_statements vs EXPLAIN mismatch**
```sql
-- Force production plan to be re-explained with actual values:
EXPLAIN (ANALYZE, BUFFERS) SELECT ... WHERE customer_id = 1234;
-- vs: check pg_stat_statements for cached plan
-- If plan differs: statistics mismatch → ANALYZE + pg_stat_reset()
```

---

**Q18. How would you optimize a query that aggregates data across 500 million rows?**

**The query:**
```sql
SELECT region,
       DATE_TRUNC('week', order_date) AS week,
       COUNT(*) AS order_count,
       SUM(amount) AS total_revenue,
       AVG(amount) AS avg_order_value
FROM orders
WHERE order_date >= '2023-01-01'
GROUP BY region, DATE_TRUNC('week', order_date)
ORDER BY week, region;
-- Full scan of 500M rows → minutes
```

**Optimization techniques:**

**1. Partition pruning (immediate win)**
```sql
-- Ensure orders is partitioned by order_date
-- Query with WHERE order_date >= '2023-01-01' → skips all pre-2023 partitions
-- Scans only 2023+ data (e.g., 200M rows instead of 500M)
```

**2. Partial aggregation via materialized view (best for repeated queries)**
```sql
-- Pre-aggregate to weekly summaries (replaces 500M row scan with 10K row scan)
CREATE MATERIALIZED VIEW weekly_order_summary AS
SELECT region,
       DATE_TRUNC('week', order_date) AS week,
       COUNT(*) AS order_count,
       SUM(amount) AS total_revenue
FROM orders
GROUP BY region, DATE_TRUNC('week', order_date);

CREATE UNIQUE INDEX ON weekly_order_summary (region, week);

-- Refresh nightly (or incrementally):
REFRESH MATERIALIZED VIEW CONCURRENTLY weekly_order_summary;

-- Query now: SELECT * FROM weekly_order_summary WHERE week >= '2023-01-01' ORDER BY week;
-- 10K rows instead of 500M → milliseconds
```

**3. Parallel query (PostgreSQL)**
```sql
-- Enable parallel aggregation:
SET max_parallel_workers_per_gather = 8;
-- PostgreSQL splits the table scan across 8 workers, each aggregates its chunk
-- Coordinator merges partial aggregates
-- Speedup ≈ 4–6× on 8 cores (I/O bound limits linear scaling)

-- Check plan for parallel execution:
EXPLAIN SELECT ... -- Look for "Gather" node with "Workers Planned: 8"
```

**4. Columnar storage (architectural)**
```sql
-- Move this analytics workload to ClickHouse (or TimescaleDB columnar chunks)
-- ClickHouse: vectorized execution + columnar storage
-- SELECT region, toStartOfWeek(order_date), count(), sum(amount) FROM orders WHERE ...
-- Same 500M rows: < 1 second in ClickHouse (vs minutes in PostgreSQL)
-- Use CDC to keep ClickHouse in sync with PostgreSQL
```

**5. Summary/rollup tables updated on write**
```sql
-- Increment summary table on each insert (no batch aggregation needed)
CREATE TABLE order_weekly_summary (
    region TEXT, week DATE, order_count INT, total_revenue NUMERIC,
    PRIMARY KEY (region, week)
);

-- On INSERT into orders, update summary:
INSERT INTO order_weekly_summary (region, week, order_count, total_revenue)
VALUES (NEW.region, DATE_TRUNC('week', NEW.order_date), 1, NEW.amount)
ON CONFLICT (region, week) DO UPDATE
    SET order_count = order_weekly_summary.order_count + 1,
        total_revenue = order_weekly_summary.total_revenue + EXCLUDED.total_revenue;
-- Summary reads: instant (reads 10K rows, not 500M)
-- Write overhead: +1 INSERT to summary per order insert (tiny)
```

---

**Q19. How does index usage change in a sharded database, and what new indexing challenges arise?**

**In a single database:**
```
Index on orders.customer_id:
  B-tree stores all customer_ids → find any customer's orders in O(log N)
  One index, one B-tree, one lookup
```

**In a sharded database (sharded by customer_id):**
```
Shard 0: customers with hash(customer_id) % 4 = 0
Shard 1: customers with hash(customer_id) % 4 = 1
...

Index on orders.customer_id within each shard:
  Each shard has its own B-tree covering only its subset of customer_ids
  Query for customer_id = 1234 → route to Shard 2 → local index lookup → fast
  Same efficiency as non-sharded (one shard, one B-tree traversal)
```

**New challenges:**

**1. Cross-shard indexes don't exist**
```
Query: "Find all orders with amount > $1000 (no shard key)"
  → Must scan all N shards (no global index to prune shards)
  → N parallel index scans + application-side merge
  
Solution:
  A) Avoid this query pattern (design queries to always include shard key)
  B) Maintain a separate global index in a dedicated database (e.g., Elasticsearch)
     orders with amount > 1000 → Elasticsearch → returns (customer_id, order_id) list
     Use customer_id to route to correct shard for full record fetch
```

**2. Global unique constraint enforcement**
```
Unique email across all users (users sharded by user_id):
  Cannot use DB-level UNIQUE constraint (only works per-shard)
  Two different shards could independently INSERT user with same email
  
Solution:
  A) Separate "email → user_id" lookup table in a dedicated shard (used for login routing)
     Unique constraint enforced by this dedicated shard: INSERT INTO email_registry ON CONFLICT FAIL
  B) Check-before-insert at application layer (race condition risk)
  C) Use a distributed ID service that guarantees uniqueness (e.g., UUID generated client-side)
```

**3. Index maintenance across N shards**
```
Adding an index: ALTER TABLE ... must run on all N shards
  Sequential: N × index build time
  Parallel: build on all N shards simultaneously (I/O spike)
  
Best practice: use CREATE INDEX CONCURRENTLY on each shard during off-peak hours
               Automate with schema migration tooling (Vitess schema changes, Flyway per shard)
```

---

**Q20. When should you move query workloads to a separate read-optimized database instead of adding more indexes?**

**Signs that indexes alone won't solve the problem:**

1. **Query must scan a large fraction of the table:**
```sql
SELECT region, SUM(amount) FROM orders WHERE order_date > '2020-01-01' GROUP BY region
-- 80% of rows match → index is useless (planner chooses seq scan correctly)
-- More indexes won't help — the problem is the scan itself
```

2. **Query requires reading many columns from wide rows:**
```
Row storage: to compute SUM(amount), PostgreSQL reads EVERY column of EVERY matching row
             (because pages store full rows: id + user_id + product_id + address + notes + amount)
With columnar storage: read ONLY the amount column (10x less data)
```

3. **Multiple analytical queries competing with OLTP:**
```
Long-running SELECT (30s) on primary → holds buffer pool pages → evicts OLTP data
Creates I/O contention with concurrent INSERTs/UPDATEs
Solution: move analytical queries to a read replica or separate OLAP DB
```

**Decision framework:**
```
Is the query read-only?               → Read replica or OLAP
Is the query aggregating millions of rows?  → OLAP (columnar)
Is the query time-series/metrics?     → TimescaleDB or ClickHouse
Is the query full-text search?        → Elasticsearch
Is the query a point-lookup?          → Index on the OLTP DB is correct solution
Is the query run by a human analyst?  → Data warehouse (Snowflake, BigQuery)
Is the query run in real-time?        → Stay on PostgreSQL with proper indexing + partitioning
```

**When to add a separate read DB:**
- Query takes > 1 second and cannot be optimized below 500ms with indexes
- Query runs more than 100 times/minute and accounts for >20% of DB CPU
- Query competes with OLTP writes and causes latency spikes on the primary
- The required index would be larger than the table itself (sign of wrong access pattern)

```
The rule: Indexes optimize access to existing storage layout.
          A separate read DB changes the storage layout for a different workload.
          Don't add 20 indexes trying to make a row-store fast at column aggregations.
          Instead: add a columnar DB.
```

---

## Quick Reference

```
Index types and use cases:
  B-tree:    equality, range, ORDER BY, LIKE 'prefix%'
  Hash:      equality only (PostgreSQL: rarely used)
  GIN:       full-text, array contains, JSONB @>
  GiST:      geometric, spatial, range types
  BRIN:      large sequential tables (time-series), very small index
  Partial:   index only rows matching WHERE (smaller, faster for selective queries)
  Covering:  add INCLUDE columns for Index Only Scans

Composite index column order:
  Equality conditions first, range condition last
  (customer_id, status, created_at) for WHERE customer_id=X AND status=Y AND created_at>Z

Common index misses:
  Function on column: WHERE LOWER(email)=... → create functional index on LOWER(email)
  Implicit type cast: string column = integer literal → match types
  Leading wildcard: LIKE '%text%' → use GIN/pg_trgm index
  Low selectivity: < 5% rows returned → index not used (correct!)

Query optimization order:
  1. EXPLAIN ANALYZE (find the bottleneck)
  2. Add missing indexes
  3. Rewrite query (eliminate N+1, push filters down, use window functions)
  4. ANALYZE (refresh statistics)
  5. Tune work_mem (for hash join / sort spills)
  6. Materialized view (pre-aggregate expensive queries)
  7. Separate OLAP database (if analytical queries compete with OLTP)
```
