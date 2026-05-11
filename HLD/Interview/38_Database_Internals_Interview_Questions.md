# 38 — Database Internals Interview Questions

## Easy (Q1–Q7)

---

### Q1. How does a B-tree index work — structure, insertion, and search?

A **B-tree** (Balanced Tree) is the dominant data structure used for database indexes. It keeps data sorted and maintains a balanced height, guaranteeing O(log N) search, insert, and delete operations.

```
B-TREE STRUCTURE (order 3: max 3 keys per node)
──────────────────────────────────────────────────────────────
                         [30 | 70]
                        /    |    \
               [10|20]  [40|50|60]  [80|90]
               / | \    / | | \    /  \
leaf pages: [5][15][25] [35][45][55][65] [75][85]

Each internal node: keys + pointers to child pages
Each leaf node:     keys + pointers to heap (actual row data)
Leaf nodes linked: ← [leaf1] → [leaf2] → [leaf3] →
                   (enables range scans efficiently)
```

**Search (O(log N)):**
- Start at root, compare search key with node keys
- Traverse left/right/middle pointer based on comparison
- Follow pointers down to leaf level
- In leaf: follow pointer to actual row in heap

**Insertion:**
```
1. Find the leaf node where the new key belongs
2. If leaf has space: insert key, update parent if needed
3. If leaf is full (overflow): SPLIT the node
   - Create new leaf, move upper half of keys
   - Push middle key UP to parent
   - If parent also overflows, split parent too (cascade up)
   - If root splits: tree grows one level (height +1)
```

**Why B-tree not binary search tree?**
- B-tree nodes hold many keys (hundreds) → shallow tree
- PostgreSQL: page size = 8KB, each internal node holds ~330 keys
- 4-level tree can index 330^4 ≈ 12 billion rows
- All leaf nodes at same depth → predictable search performance
- Leaf page links → O(n) range scan without backtracking

**B+ tree variant (used in most databases):**
Only leaf nodes hold data pointers. Internal nodes hold only keys (routing). This allows more keys per internal node → shallower tree → faster search.

---

### Q2. Why do database writes go to a WAL (Write-Ahead Log) before the data pages?

The **WAL** (Write-Ahead Log) guarantees **durability** and enables **crash recovery**. Before any data page is modified, the change is first appended to the WAL. "Write-ahead" means the log record must be flushed to disk before the data page modification is considered durable.

```
WRITE PATH WITHOUT WAL (dangerous)
──────────────────────────────────────────────────────────────
UPDATE users SET balance = 100 WHERE id = 1;
      │
      ▼ Modify page in buffer pool (in-memory only)
      │ Power fails here!
      ▼ Page write to disk never completed
Result: DATABASE IS IN INCONSISTENT STATE
──────────────────────────────────────────────────────────────

WRITE PATH WITH WAL (safe)
──────────────────────────────────────────────────────────────
UPDATE users SET balance = 100 WHERE id = 1;
      │
      ▼ Append to WAL buffer: (txn_id=42, op=UPDATE, table=users, ...)
      │
      ▼ On COMMIT: fsync WAL to disk (synchronous, durable)
      │ Power fails here → WAL is on disk, can replay
      │
      ▼ Modify page in buffer pool (async, may not be written yet)
      │
      ▼ Checkpoint: periodically flush dirty pages to disk
      │ After checkpoint: WAL records before checkpoint can be discarded
Result: On crash, replay WAL from last checkpoint → fully consistent
```

**WAL Benefits:**
| Benefit | How WAL enables it |
|---|---|
| Durability | fsync WAL before commit confirmation |
| Crash recovery | Replay WAL from last checkpoint |
| Replication | Stream WAL records to replicas |
| Point-in-time recovery | Apply WAL up to specific timestamp |
| Efficient writes | Sequential log append is faster than random page writes |

**Sequential writes are faster:** WAL writes are sequential (append to end of file). Data page writes are random (scattered across many pages). Sequential I/O is 10–100x faster on spinning disks and 5–10x faster on SSDs.

---

### Q3. How does MVCC (Multi-Version Concurrency Control) work in PostgreSQL?

**MVCC** allows readers and writers to never block each other. Instead of locking a row when it is updated, PostgreSQL keeps **multiple versions** of the same row and uses transaction IDs to determine which version is visible to each transaction.

```
MVCC ROW VERSIONING
──────────────────────────────────────────────────────────────
Each row has two hidden system columns:
  xmin: the transaction ID that CREATED this row version
  xmax: the transaction ID that DELETED/UPDATED this row version
        (0 if still current)

Timeline:
  txn_id=1: INSERT INTO users (id, name) VALUES (1, 'Alice')
  txn_id=5: UPDATE users SET name = 'Alice B' WHERE id = 1

Physical rows in heap:
  Row A: xmin=1, xmax=5, data=(id=1, name='Alice')     ← old version
  Row B: xmin=5, xmax=0, data=(id=1, name='Alice B')   ← current version
```

**Visibility Rules:**
A transaction with txn_id=T can see a row version if:
- `xmin < T` (row was created before this transaction)
- `xmin` is committed (not aborted)
- `xmax = 0` OR `xmax >= T` OR `xmax` is not committed

```python
# Simplified visibility check
def is_visible(row, current_txn_id, snapshot):
    if row.xmin not in snapshot.committed:
        return False                         # creator hasn't committed
    if row.xmin > current_txn_id:
        return False                         # created after us
    if row.xmax != 0:
        if row.xmax in snapshot.committed and row.xmax <= current_txn_id:
            return False                     # deleted before us
    return True
```

**Benefit:** A long-running `SELECT` doesn't block `UPDATE` and vice versa. They operate on different row versions.

**Cost:** Dead tuple accumulation. Rows A (old versions) are no longer visible to any active transaction but still occupy disk space. **VACUUM** reclaims this space by marking dead tuples as reusable.

---

### Q4. What is a database buffer pool and how does it affect performance?

The **buffer pool** (or buffer cache) is an in-memory cache of database pages. Since disk access is 1000x slower than memory access, the buffer pool is the single most important performance component in a database.

```
BUFFER POOL ARCHITECTURE
──────────────────────────────────────────────────────────────
Application Query
      │
      ▼ Execution Engine requests page P
      │
      ├─ Page P in buffer pool? → YES → Return immediately (memory access ~100ns)
      │
      └─ NO → Page fault
             ├─ Read page P from disk (SSD ~100μs, HDD ~10ms)
             ├─ Load into buffer pool frame
             │   (if pool is full: evict a page first — LRU policy)
             └─ Return page to query

BUFFER POOL MEMORY
┌────────────────────────────────────────────────┐
│  Frame 1: page_id=42,  dirty=false, pin=0      │
│  Frame 2: page_id=107, dirty=true,  pin=1      │  ← being written
│  Frame 3: page_id=33,  dirty=false, pin=0      │
│  ...                                            │
│  Frame N: (empty)                               │
└────────────────────────────────────────────────┘
```

**LRU Eviction:**
When a new page must be loaded but the pool is full, the Least Recently Used page is evicted. If it is dirty (modified but not yet written to disk), it must be flushed first.

**Performance Impact:**
```
Buffer pool hit rate = reads served from memory / total reads

hit_rate = 90%  → 1 in 10 reads goes to disk
hit_rate = 99%  → 1 in 100 reads goes to disk   (10x faster)
hit_rate = 99.9% → 1 in 1000 reads goes to disk  (100x faster)

PostgreSQL: shared_buffers controls buffer pool size
Recommended: 25-40% of total RAM
```

**Tuning:**
- Too small: constant evictions, high disk I/O, low performance
- Too large: OS has no memory for its own page cache (double-caching problem)
- PostgreSQL: `shared_buffers = 8GB` (on a 32GB server)
- MySQL InnoDB: `innodb_buffer_pool_size = 24GB` (on a 32GB server)

---

### Q5. What is the difference between a clustered and a non-clustered index?

**Clustered index:** The physical order of rows on disk matches the index order. The table IS the index — leaf nodes of the B-tree contain the actual row data. There can be only one clustered index per table.

**Non-clustered index:** A separate data structure that stores index keys + a pointer (row locator) back to the actual row. Multiple non-clustered indexes can exist per table.

```
CLUSTERED INDEX (InnoDB primary key / PostgreSQL with CLUSTER)
──────────────────────────────────────────────────────────────
B-tree leaf pages contain actual row data, ordered by key:
[PK=1 | name='Alice' | age=30] → [PK=2 | name='Bob' | age=25] → ...
                                     ↑ rows physically ordered by PK on disk
Range query on PK: reads sequential disk pages → fast

NON-CLUSTERED INDEX
──────────────────────────────────────────────────────────────
B-tree leaf pages contain: (index_key, row_pointer/heap_tid)
Index on email:
[alice@x.com | TID(page=42,slot=3)] → [bob@y.com | TID(page=107,slot=1)] → ...
                                                          ↑ pointer to heap
Lookup by email:
  1. Search non-clustered index B-tree (fast)
  2. Follow heap_tid to actual page (potentially random I/O)
  This step 2 is called a "heap fetch" or "table heap fetch"
```

**Performance Comparison:**
| Operation | Clustered | Non-Clustered |
|---|---|---|
| Range scan on key | Very fast (sequential pages) | Slower (random heap fetches) |
| Point lookup | Fast | Fast (index) + medium (heap fetch) |
| INSERT | Slower (must maintain order) | Faster for secondary indexes |
| Storage | Part of table storage | Additional storage required |

**PostgreSQL note:** PostgreSQL's heap is unordered. `CLUSTER` command physically reorders the table once, but the order degrades as rows are added. InnoDB (MySQL) always clusters by primary key.

---

### Q6. What is a covering index and why does it avoid a heap fetch?

A **covering index** (also called an index-only scan) is an index that contains all the columns needed to satisfy a query — so the database can answer the query entirely from the index without ever accessing the main table (heap).

```sql
-- Table: orders (id, user_id, amount, status, created_at)
-- Query: find total amount per user for completed orders

SELECT user_id, SUM(amount)
FROM orders
WHERE status = 'completed'
GROUP BY user_id;

-- Without covering index:
-- 1. Use index on status to find matching row IDs
-- 2. For EACH matching row: fetch the full row from heap to get amount
-- (potentially millions of random I/O operations)

-- With covering index:
CREATE INDEX idx_covering ON orders (status, user_id, amount);
-- Now the index contains status + user_id + amount
-- Query can be answered ENTIRELY from index pages
-- No heap access needed → index-only scan
```

**EXPLAIN output difference:**
```sql
-- Without covering index:
EXPLAIN SELECT user_id, SUM(amount) FROM orders WHERE status = 'completed' GROUP BY user_id;
-- Bitmap Index Scan on idx_status  (cost=...)
-- → Bitmap Heap Scan on orders     (cost=...) ← heap access!

-- With covering index:
-- Index Only Scan using idx_covering on orders  ← no heap access!
```

**PostgreSQL INCLUDE clause (covering without widening key):**
```sql
-- Include amount in index but not in the key (doesn't affect sort order)
CREATE INDEX idx_status_user ON orders (status, user_id) INCLUDE (amount);
-- amount is stored in leaf pages but not used for ordering
-- avoids heap fetch while keeping index smaller than a 3-column key index
```

**When it matters most:** Aggregation queries over large ranges, COUNT(*) queries, queries that select only a few columns from a wide table.

---

### Q7. How does PostgreSQL VACUUM work and why is it needed?

**VACUUM** is PostgreSQL's mechanism for reclaiming storage occupied by dead tuples (old row versions no longer visible to any transaction) created by MVCC.

```
WHY VACUUM IS NEEDED
──────────────────────────────────────────────────────────────
txn_id=5:  UPDATE users SET name='Alice B' WHERE id=1
  → Row A (xmin=1, xmax=5): DEAD TUPLE (old version, no longer visible)
  → Row B (xmin=5, xmax=0): LIVE TUPLE (current version)

  Row A still occupies disk space!
  Without VACUUM: table grows indefinitely even with no new data

VACUUM PROCESS
──────────────────────────────────────────────────────────────
1. Scan table pages linearly
2. Identify dead tuples (xmax is committed and below oldest active txn)
3. Mark dead tuple slots as "available for reuse" (doesn't shrink file)
4. Update Free Space Map (FSM) so new inserts can reuse space
5. Update Visibility Map (VM) for pages that are all-visible
   (enables index-only scans)
6. Remove index entries pointing to dead tuples
```

**VACUUM FULL vs regular VACUUM:**
```sql
-- Regular VACUUM: marks space as reusable, doesn't shrink file
VACUUM users;

-- VACUUM FULL: rewrites entire table, returns space to OS
-- WARNING: acquires exclusive lock! No reads/writes during execution!
VACUUM FULL users;

-- ANALYZE: updates table statistics for query planner
VACUUM ANALYZE users;

-- Check table bloat
SELECT relname, n_dead_tup, n_live_tup,
       round(100.0 * n_dead_tup / nullif(n_live_tup + n_dead_tup, 0), 2) as dead_pct
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;
```

**Autovacuum:** PostgreSQL's autovacuum daemon runs automatically when dead tuple count exceeds threshold:
```
scale_factor * n_live_tup + autovacuum_vacuum_threshold
default: 0.2 * live_rows + 50
```

**XID Wraparound (critical!):** Transaction IDs are 32-bit integers. After 2^32 ≈ 4 billion transactions, XID wraps around. VACUUM prevents wraparound by freezing old tuples (setting xmin to special "frozen" value). Ignoring wraparound alerts leads to database shutdown.

---

## Medium (Q8–Q15)

---

### Q8. What is the LSM tree vs B-tree trade-off — read amplification vs write amplification?

This is the most important data structure decision in database engine design. The choice between B-tree and LSM tree determines whether the system is read-optimized or write-optimized.

```
B-TREE: READ-OPTIMIZED
──────────────────────────────────────────────────────────────
Write path: find leaf node (random I/O) → modify in-place
Read path:  traverse tree (O(log N)) → one location for data

Write amplification: HIGH (random write to leaf + parent updates)
Read amplification:  LOW  (data in one place in tree)
Space amplification: MEDIUM (page fragmentation, free space in pages)

Best for: OLTP workloads with frequent point reads and updates
```

```
LSM TREE: WRITE-OPTIMIZED (Log-Structured Merge Tree)
──────────────────────────────────────────────────────────────
Write path: append to in-memory MemTable (very fast)
  MemTable full → flush to disk as immutable SSTable (L0)
  Background: merge/compact SSTables into larger levels

Read path: check MemTable → L0 SSTables → L1 → L2 → ...
  May need to check multiple SSTables (read amplification!)

Write amplification: LOW  (sequential writes only)
Read amplification:  HIGH (check multiple SSTable levels)
Space amplification: MEDIUM (stale versions in older SSTables until compaction)
```

```
AMPLIFICATION COMPARISON
──────────────────────────────────────────────────────────────
              B-Tree          LSM Tree
Write amp     HIGH            LOW (2-10x on disk, 1x on write path)
Read amp      LOW (1-2x)      HIGH (up to 10+ SSTables checked)
Space amp     MEDIUM          MEDIUM (until compaction)
Best for      High read ratio High write ratio
Used in       PostgreSQL,     Cassandra, RocksDB,
              MySQL InnoDB    LevelDB, HBase
```

**Mitigating LSM read amplification:**
- Bloom filters: check if a key exists in an SSTable before reading it
- Block cache: cache frequently read SSTable blocks in memory
- Compaction: merge SSTables to reduce levels (reduces read amplification)

---

### Q9. How does RocksDB/Cassandra LSM tree work — MemTable, SSTable, and compaction?

```
LSM TREE WRITE PATH
──────────────────────────────────────────────────────────────
Write(key=K, value=V)
      │
      ▼ 1. Write to WAL (durability)
      │
      ▼ 2. Insert into MemTable (in-memory sorted skip list)
      │    Fast: in-memory write, sorted order maintained
      │
      ▼ 3. MemTable full (e.g., 64MB)?
              YES → Flush to disk as immutable SSTable (L0)
                    New MemTable created for new writes
```

**SSTable (Sorted String Table):**
```
SSTable file layout:
┌─────────────────────────────────────────────────────┐
│ Data blocks: sorted key-value pairs                 │
│ Index block: key → offset in data blocks            │
│ Bloom filter: probabilistic set for key existence   │
│ Metadata: min/max key, level, creation time         │
└─────────────────────────────────────────────────────┘

File is immutable once written.
All updates/deletes are new entries (NEWER entry wins on read).
Delete = "tombstone" entry (key + deletion marker).
```

**Compaction (background process):**
```
Level 0: multiple small SSTables (may have overlapping key ranges)
  ↓ compaction (merge L0 files when count > threshold)
Level 1: larger SSTables, non-overlapping key ranges (10MB each)
  ↓ compaction (when level size exceeds limit)
Level 2: even larger (100MB each)
  ↓
Level N: largest SSTables

Compaction = merge-sort multiple SSTables:
  Read all files being compacted
  Merge-sort their entries (keep newest version per key)
  Discard tombstones for keys with no older versions
  Write output to next level
  Delete input files
```

**Read Path:**
```
Read(key=K)
  1. Check MemTable (in-memory) → found? return
  2. Check L0 SSTables (newest first, may overlap)
     Use bloom filter to skip if key definitely absent
  3. Check L1 (binary search on index, at most 1 file — non-overlapping)
  4. Check L2, L3... until found or exhausted
```

---

### Q10. What is index bloat and how do you fix it?

**Index bloat** occurs when an index grows larger than necessary due to dead index entries, page fragmentation, or logical deletion of rows. A bloated index wastes memory, slows queries, and occupies excessive disk space.

**Causes:**
```sql
-- Dead index entries from deleted rows
DELETE FROM orders WHERE created_at < '2020-01-01';
-- Index still contains entries pointing to these deleted heap rows
-- VACUUM cleans heap dead tuples but index cleanup is delayed

-- Update pattern: each UPDATE creates a new row version
UPDATE products SET price = price * 1.1;
-- 1M updates → 1M new row versions → index has 2M entries
-- (old versions are dead but bloat the index until VACUUM index cleanup)

-- Page fragmentation from random inserts
INSERT INTO events (id, ...) VALUES (random_uuid(), ...);
-- UUID inserts scatter across all index pages
-- Each page ends up 60-70% full → 30-40% wasted space
```

**Detecting Bloat:**
```sql
-- pgstattuple extension
CREATE EXTENSION pgstattuple;

SELECT
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    round(100.0 * dead_leaf_pages / total_leaf_pages, 2) AS dead_pct
FROM pg_stat_user_indexes
JOIN pg_index USING (indexrelid)
WHERE schemaname = 'public'
ORDER BY pg_relation_size(indexrelid) DESC;
```

**Fixing Bloat:**
```sql
-- Option 1: REINDEX CONCURRENTLY (online, no lock!)
REINDEX INDEX CONCURRENTLY idx_orders_user_id;
-- Builds new index alongside old one, swaps atomically
-- Reads and writes continue during rebuild

-- Option 2: Regular VACUUM (prevents accumulation)
VACUUM (ANALYZE) orders;

-- Option 3: For UUID primary keys — use UUIDv7 (time-sortable)
-- or ULID to avoid fragmentation from random ordering
-- UUIDv7 has timestamp prefix → sequential-ish inserts → less bloat
```

**Prevention:**
- Use sequential IDs (BIGSERIAL) instead of random UUIDs when possible
- Configure autovacuum more aggressively for high-churn tables
- Set `fillfactor = 70` on frequently updated tables: leave 30% free space to accommodate updates in-place

---

### Q11. How does database connection overhead work and why is connection pooling critical?

Opening a new database connection is **expensive**: it involves TCP handshake, authentication, TLS negotiation, process/thread creation, and session initialization. In PostgreSQL, each connection spawns a new backend process.

```
CONNECTION ESTABLISHMENT COST (PostgreSQL)
──────────────────────────────────────────────────────────────
Step                    Time
TCP handshake           1–5 ms
TLS negotiation         5–20 ms
Authentication          2–10 ms
Backend process fork    5–20 ms  (PostgreSQL forks a process per connection)
Session setup           1–5 ms
──────────────────────
Total                   15–60 ms per new connection

At 1000 req/sec: 1000 connections/sec × 50ms = impossible
```

**Connection Pooling (PgBouncer):**
```
WITHOUT POOLING
────────────────────────────────
App Server 1: open conn → query → close conn (50ms overhead each time)
App Server 2: open conn → query → close conn
...
100 app servers × 10 threads = 1000 connections to DB

WITH PGBOUNCER
────────────────────────────────
App Servers ──► PgBouncer (connection pool)
                 Maintains N persistent connections to PostgreSQL
                 (e.g., N = 20)
                 App gets a connection from pool (< 1ms)
                 Returns to pool after query

1000 app threads → PgBouncer → 20 persistent DB connections
                               (PostgreSQL only manages 20 processes)
```

**PgBouncer Modes:**
```
Session pooling:      connection held for entire client session
                      → minimal overhead benefit
Transaction pooling:  connection returned to pool after each transaction
                      → best performance, most common
Statement pooling:    connection returned after each statement
                      → can break applications that use session state
```

**Connection Pool Sizing:**
```
# Recommended formula (PostgreSQL):
pool_size = num_cores * 2 + num_spindles

# Example: 8 core server, SSD
pool_size = 8 * 2 + 1 = 17 connections

# Beyond ~100 connections, PostgreSQL performance DEGRADES
# (context switching between backend processes)
```

---

### Q12. What is write amplification in SSDs and how do LSM trees minimize it?

**Write amplification** is the ratio of actual bytes written to the storage device vs the bytes written by the application. It is the most important SSD health and performance metric.

```
WRITE AMPLIFICATION PROBLEM IN B-TREES
──────────────────────────────────────────────────────────────
Application writes: UPDATE users SET name='Bob' WHERE id=1
  → 8 bytes changed in a 8KB page
  → Database must write full 8KB page to disk (SSD minimum write unit)
  → Page may be in a 512KB SSD erase block
  → SSD must: read 512KB block → erase → write 512KB block
  → Write amplification = 512KB / 8B = 65,536x!

In practice, SSD write amplification: 5–100x for random writes
```

**How LSM Trees Minimize Write Amplification:**
```
LSM WRITE PATH
──────────────────────────────────────────────────────────────
Application write → MemTable (in-memory, no disk I/O)
                 → WAL (sequential append, minimal amplification)

When MemTable flushes to SSTable:
  Write is sequential (full SSTable file written once)
  SSD handles sequential writes efficiently (pre-erase at idle time)
  Write amplification ≈ 1–3x (much better than B-tree)

Compaction amplification:
  Each key may be rewritten multiple times during compaction
  Typical overall write amplification: 10–30x total
  Still better than random B-tree updates
```

**B-tree vs LSM Write Amplification:**
```
Random write workload:
  B-tree:  50–1000x write amplification (random page overwrites)
  LSM:     10–30x write amplification   (sequential writes)

Sequential write workload:
  B-tree:  5–20x
  LSM:     10–30x
  (B-tree can be competitive for sequential workloads)
```

**SSD Wear:** Write amplification directly reduces SSD lifespan. An SSD rated for 1 PBW (petabyte written) with 10x WA = only 100 TB of useful writes. LSM's lower WA on random workloads makes it much better for SSD longevity.

---

### Q13. How does EXPLAIN ANALYZE output reveal query performance issues?

`EXPLAIN ANALYZE` is the single most important diagnostic tool for slow database queries. It shows the query plan the planner chose AND the actual execution statistics.

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT u.name, COUNT(o.id)
FROM users u
JOIN orders o ON o.user_id = u.id
WHERE u.country = 'US'
GROUP BY u.id, u.name;
```

**Sample Output with Annotations:**
```
Finalize GroupAggregate  (cost=15000..16000 rows=10000 width=50)
                         (actual time=890..950 rows=8432 loops=1)
  -> Gather Merge  (cost=... actual time=880..885 rows=12000 loops=1)
       Workers Planned: 4
       Workers Launched: 4
  -> Partial GroupAggregate  (actual time=800..820 rows=3000 loops=5)
       -> Hash Join  (cost=5000..12000 rows=50000 width=32)
                     (actual time=50..780 rows=42000 loops=5)
             Hash Cond: (o.user_id = u.id)
             Buffers: shared hit=1200 read=8500  ← 8500 disk reads!
             -> Seq Scan on orders  (cost=0..8000 rows=400000 width=8)
                                    (actual time=0.1..400 rows=400000 loops=5)
             -> Hash  (cost=3000..3000 rows=40000 width=24)
                      (actual time=40..40 rows=42000 loops=5)
                  -> Index Scan on users  (cost=0.56..3000 rows=40000 width=24)
                     (actual time=0.1..35 rows=42000 loops=5)
                     Index Cond: ((country)::text = 'US'::text)
Planning Time: 5 ms
Execution Time: 960 ms
```

**Key Things to Look For:**

| What you see | What it means | Action |
|---|---|---|
| `Seq Scan` on large table | No usable index | Create index on filter column |
| `rows=50000` estimated vs `rows=42000` actual — close | Statistics are good | No action |
| `rows=1000` estimated vs `rows=400000` actual | Stale statistics → bad plan | `ANALYZE table_name` |
| `Buffers: read=8500` (vs hit=1200) | Most data fetched from disk | Increase shared_buffers, add covering index |
| `Hash Join` on large tables | Good for large datasets | Monitor memory: `work_mem` |
| `Nested Loop` with many inner iterations | Can be slow if outer set is large | Consider increasing `enable_hashjoin` or rewrite query |

---

### Q14. How do phantom reads work and how does SERIALIZABLE isolation prevent them?

**Phantom reads** occur when a transaction re-executes a query and finds NEW rows that weren't there before — because another transaction inserted rows that match the WHERE predicate.

```
PHANTOM READ SCENARIO
──────────────────────────────────────────────────────────────
Transaction A (READ COMMITTED / REPEATABLE READ):
  T1: SELECT COUNT(*) FROM orders WHERE amount > 1000;
      → Returns 50 rows

Transaction B (concurrent):
  T2: INSERT INTO orders (user_id, amount) VALUES (42, 1500);
  T3: COMMIT;

Transaction A (still running):
  T4: SELECT COUNT(*) FROM orders WHERE amount > 1000;
      → Returns 51 rows!  ← Phantom read!
      (new row appeared in same transaction)
```

**Isolation Levels and Phantom Read Protection:**
```
READ UNCOMMITTED  → dirty reads, phantom reads possible
READ COMMITTED    → no dirty reads, phantom reads POSSIBLE
REPEATABLE READ   → PostgreSQL: no phantom reads (snapshot isolation)
                    MySQL: phantom reads POSSIBLE (gap locks needed)
SERIALIZABLE      → no anomalies of any kind
```

**PostgreSQL SERIALIZABLE (Serializable Snapshot Isolation - SSI):**
PostgreSQL uses SSI (not traditional locking). Instead of locking ranges, it tracks read/write dependencies between concurrent transactions and **aborts** a transaction if a cycle would violate serializability.

```sql
-- Transaction A
BEGIN ISOLATION LEVEL SERIALIZABLE;
SELECT COUNT(*) FROM orders WHERE amount > 1000;  -- reads predicate

-- Transaction B (concurrent)
BEGIN ISOLATION LEVEL SERIALIZABLE;
INSERT INTO orders (amount) VALUES (1500);
COMMIT;

-- Transaction A tries to commit
COMMIT;
-- ERROR: could not serialize access due to concurrent update
-- Application must retry Transaction A
```

**Predicate Locking (MySQL SERIALIZABLE):**
MySQL uses actual predicate locks (gap locks, next-key locks) to prevent phantom inserts:
```
SELECT * FROM orders WHERE amount > 1000 FOR SHARE;
→ Locks the gap > 1000 in the index
→ Any concurrent INSERT with amount > 1000 is BLOCKED
```

---

### Q15. How do you diagnose a slow database query step by step?

A structured diagnostic approach prevents wasted time guessing. Follow this methodology for any slow query.

**Step-by-Step Diagnostic Process:**

```
STEP 1: Identify the slow query
────────────────────────────────
-- pg_stat_statements: top queries by total time
SELECT query, calls, total_exec_time/calls AS avg_ms,
       rows, shared_blks_hit, shared_blks_read
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;

Look for: high avg_ms (slow individual execution)
          high total_exec_time (called frequently)
          high shared_blks_read (lots of disk reads)

STEP 2: Get the query plan
────────────────────────────────
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT ...;  -- the slow query

Look for: Seq Scan on large tables
          Estimated rows >> Actual rows (stale stats)
          High Buffers: read count (disk I/O)
          Long actual times at specific nodes

STEP 3: Check table statistics freshness
────────────────────────────────
SELECT relname, last_analyze, last_autoanalyze, n_live_tup, n_dead_tup
FROM pg_stat_user_tables
WHERE relname = 'your_table';

If last_analyze is old: ANALYZE your_table;

STEP 4: Check for missing indexes
────────────────────────────────
-- Check if index exists on WHERE clause columns
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'your_table';

-- Check for sequential scans (indicator of missing index)
SELECT relname, seq_scan, seq_tup_read, idx_scan
FROM pg_stat_user_tables
WHERE seq_scan > idx_scan AND n_live_tup > 10000;

STEP 5: Check for lock contention
────────────────────────────────
SELECT pid, query, wait_event_type, wait_event, state
FROM pg_stat_activity
WHERE wait_event IS NOT NULL
ORDER BY wait_event_type;

STEP 6: Check for connection/resource pressure
────────────────────────────────
SELECT count(*), state FROM pg_stat_activity GROUP BY state;
-- Too many "idle in transaction" = connection leak or long transactions
-- Too many "active" = overloaded database
```

---

## Hard (Q16–Q20)

---

### Q16. How does the InnoDB doublewrite buffer protect against torn pages?

**Torn page** is a partial write failure that occurs when the database crashes during a page write. If a 16KB page is written in four 4KB OS writes and the system crashes after the first write, the page on disk contains half old data and half new data — corrupted.

```
TORN PAGE SCENARIO
──────────────────────────────────────────────────────────────
Database needs to write modified 16KB page (4 × 4KB OS writes):
  Write 1: first 4KB → disk (success)
  Write 2: second 4KB → disk (success)
  POWER FAILURE
  Write 3 and 4: never written

Result: page on disk = [new first 4KB | new second 4KB | OLD 4KB | OLD 4KB]
  → checksums don't match → corrupted page
  → WAL replay fails because WAL assumes the pre-update page is intact
```

**The WAL Limitation:**
WAL is typically **redo-only** (stores the new value, not the old). To replay a WAL record, the database reads the page from disk, applies the change, and writes back. If the page is torn (partially corrupted), WAL cannot safely apply the redo record.

**InnoDB Doublewrite Buffer Solution:**
```
DOUBLEWRITE WRITE PATH
──────────────────────────────────────────────────────────────
Step 1: Write dirty pages to DOUBLEWRITE BUFFER (sequential area on disk)
        Sequential write is fast; this is a sequential write to a contiguous region
        Flush to disk.

Step 2: Write dirty pages to their ACTUAL positions (random I/O)

On crash recovery:
  If actual page is intact → use it (normal case)
  If actual page is torn → copy from doublewrite buffer (recovery case)
  Recovery is always safe because either: actual page is good,
  OR doublewrite buffer has the complete pre- or post-write image
```

**PostgreSQL Alternative: Full Page Writes:**
PostgreSQL uses a different approach — on the first modification of a page after a checkpoint, it writes the ENTIRE page (not just the delta) to the WAL. This ensures crash recovery can always apply WAL from a good full-page image.

```
-- Control full_page_writes in PostgreSQL
SHOW full_page_writes;  -- should be 'on' (default)
-- Turning off full_page_writes is dangerous: only safe on storage with atomic writes
```

**Modern SSDs with 4K native sectors:** Have reduced (but not eliminated) torn page risk since a 4KB write is now atomic. Still, file system layers and RAID introduce multi-block writes.

---

### Q17. How does parallel query execution work in PostgreSQL?

PostgreSQL can split a single query across multiple CPU cores using parallel workers. This is particularly valuable for sequential scans, aggregations, and hash joins on large tables.

```
PARALLEL QUERY ARCHITECTURE
──────────────────────────────────────────────────────────────
Query: SELECT category, SUM(amount) FROM orders GROUP BY category;

Parallel execution plan:
  Finalize Aggregate (leader process)
       │
  Gather (coordination node — leader collects from workers)
       │
       ├─── Partial Aggregate (Worker 1) → reads pages 0–33%
       ├─── Partial Aggregate (Worker 2) → reads pages 34–66%
       └─── Partial Aggregate (Worker 3) → reads pages 67–100%

Each worker scans its portion of the table independently.
Workers compute partial aggregates (SUM per category within their portion).
Gather node collects all partial aggregates.
Finalize Aggregate merges partial SUMs into final SUM.
```

**Configuration:**
```sql
-- Maximum parallel workers per query
SET max_parallel_workers_per_gather = 4;

-- Total parallel workers across all queries
SET max_parallel_workers = 8;        -- should match CPU cores

-- Minimum table size to consider parallel scan (in pages = 8KB each)
SET min_parallel_table_scan_size = '8MB';   -- default

-- Parallel cost model: planner prefers parallel when estimated speedup
-- exceeds parallel_setup_cost (worker startup overhead)
SET parallel_setup_cost = 1000;     -- default
SET parallel_tuple_cost = 0.1;      -- cost per tuple passed between processes
```

**EXPLAIN with Parallel Plan:**
```sql
EXPLAIN SELECT category, SUM(amount) FROM orders GROUP BY category;
-- Finalize GroupAggregate  (cost=15000..16000 rows=100 width=20)
--   -> Gather Merge  (cost=14000..15000 rows=300 width=20)
--        Workers Planned: 3
--     -> Partial GroupAggregate  (cost=... rows=100 width=20)
--          -> Parallel Seq Scan on orders  (cost=0..10000 rows=133333 width=12)

-- Note: "Parallel Seq Scan" divides table pages among workers
```

**Limitations:**
- Not all operations are parallelizable: functions marked `PARALLEL UNSAFE` prevent parallelism
- Parallel overhead: worker startup takes ~1ms; only worthwhile for queries taking > 10ms
- Index scans: typically not parallelized (random I/O doesn't benefit from multiple workers as much)
- Memory: each worker gets its own `work_mem`; N workers × work_mem = total memory

---

### Q18. How does HBase / Bigtable store data — row key design, column families, and cell versioning?

HBase is a wide-column distributed database modeled after Google Bigtable. Understanding its storage model is essential for designing efficient schemas.

```
BIGTABLE/HBASE DATA MODEL
──────────────────────────────────────────────────────────────
Conceptual model:
  Table = sorted map of row keys
  Each row key → column families → columns → versioned cell values

  (row_key, column_family:column, timestamp) → value

Physical storage:
  Rows sorted lexicographically by row_key
  Column families stored separately (different HFiles)
  Each cell has multiple timestamped versions
```

**Row Key Design (critical for performance):**
```
BAD: user_id as row key
  user:00001 → all data for user 1
  user:00002 → all data for user 2
  ...
  Problem: "hotspot" — sequential keys → all writes go to last region server!

GOOD: salted or reversed key
  Option 1: Hash prefix (distribute writes evenly)
    row_key = MD5(user_id)[:2] + ":" + user_id
    "3a:00001", "9f:00002", "11:00003" → distributed across regions

  Option 2: Reversed timestamp (recent data first)
    row_key = (MAX_LONG - timestamp) + ":" + user_id
    Most recent rows stored first → "get latest" scan is efficient
```

**Column Families:**
```
CREATE TABLE user_activity (
  row_key      VARCHAR,  -- user:timestamp
  cf_events    COLUMN FAMILY (  -- stored in same HFile
    page_view  VARCHAR,
    click      VARCHAR,
    purchase   VARCHAR
  ),
  cf_profile   COLUMN FAMILY (  -- stored in DIFFERENT HFile
    name       VARCHAR,
    email      VARCHAR
  )
);

-- cf_events and cf_profile stored in different HFiles
-- Query that only needs profile doesn't touch events HFile → efficient
-- Rule: columns accessed together → same column family
```

**Cell Versioning:**
```
Get latest 3 versions of user profile:
  row_key="user:00001", cf="cf_profile", col="email"
    → T=1000: alice@old.com
    → T=2000: alice@new.com   ← most recent
    → T=3000: alice@final.com ← newest

HBase stores up to N versions per cell (configurable: VERSIONS=3)
```

**Region Servers and Compaction:**
Data stored in MemStore → flush to HFiles (like LSM SSTables). Compaction merges HFiles. Major compaction removes tombstones and old versions.

---

### Q19. What is a bloom filter and how does RocksDB use it to avoid unnecessary SSTable reads?

A **Bloom filter** is a probabilistic data structure that answers: "Is element X definitely NOT in this set?" It can return:
- **Definitely not present** (no false negatives) — skip this SSTable entirely
- **Possibly present** — read the SSTable to verify

This makes it ideal for LSM tree read optimization: avoid reading SSTables that definitely don't contain the key.

```
BLOOM FILTER MECHANICS
──────────────────────────────────────────────────────────────
Size: m bits, k hash functions

INSERT key K:
  Hash K with function 1 → set bit at position h1(K)
  Hash K with function 2 → set bit at position h2(K)
  Hash K with function 3 → set bit at position h3(K)

QUERY key K:
  Check bits at positions h1(K), h2(K), h3(K)
  ALL three set? → "Possibly in set" (could be false positive)
  ANY not set?  → "Definitely not in set" (no false negative possible)

Bit array: [0,0,1,0,1,0,0,1,1,0,0,1,...]  (m=12 bits)
```

**False Positive Rate:**
```python
# Optimal bloom filter parameters
import math

def bloom_filter_size(n_elements: int, false_positive_rate: float) -> tuple:
    m = -n_elements * math.log(false_positive_rate) / (math.log(2)**2)
    k = m / n_elements * math.log(2)
    return int(m), int(k)

# 1 million elements, 1% false positive rate
m, k = bloom_filter_size(1_000_000, 0.01)
# m ≈ 9.6 million bits (1.2 MB), k ≈ 7 hash functions
# 1% false positive rate means: 1 in 100 "possibly present" answers are wrong
```

**RocksDB Usage:**
```
Read(key=K):
  1. Check MemTable → not found
  2. For each SSTable (newest to oldest):
     ├─ Query bloom filter for key K
     │   "Definitely not present"? → SKIP THIS FILE (saves disk I/O)
     │   "Possibly present"?       → Read SSTable block index
     │                               → Read actual data block
     │                               → Found? return value
     └─ Continue to next level
```

**Impact:** Without bloom filters, every read of a missing key requires reading ALL SSTable files. With bloom filters, only ~1% of SSTables are read unnecessarily (at 1% false positive rate), reducing IOPS by 10–100x for read workloads.

**Memory cost:** RocksDB default: 10 bits per key. 1 billion keys = 10 billion bits = 1.25 GB. A worthwhile memory investment for dramatically reduced disk I/O.

---

### Q20. What are phantom reads, and how does Serializable Snapshot Isolation (SSI) prevent them without locks?

This is an advanced question that requires understanding the difference between lock-based serializability and optimistic concurrency control via SSI.

**Traditional Lock-Based Serializability (MySQL SERIALIZABLE):**
```sql
-- Transaction A acquires a shared predicate lock on the range amount > 1000
SELECT COUNT(*) FROM orders WHERE amount > 1000 FOR SHARE;
-- Now any INSERT with amount > 1000 is BLOCKED by a gap lock
-- Until Transaction A commits
-- Limitation: high contention, readers block writers
```

**PostgreSQL SSI (Optimistic, Lock-Free):**
PostgreSQL SERIALIZABLE uses **Serializable Snapshot Isolation** — it tracks dependencies between transactions and detects dangerous patterns at commit time.

```
SSI DEPENDENCY TRACKING
──────────────────────────────────────────────────────────────
Two types of dependencies:
  rw-conflict: T1 reads something that T2 later writes
  wr-conflict: T1 writes something that T2 reads

A "dangerous structure" (serialization anomaly) occurs when:
  T1 →rw→ T2 →rw→ T1   (cycle in dependency graph)

SSI PHANTOM READ PREVENTION
──────────────────────────────────────────────────────────────
Transaction A:
  T1: SELECT COUNT(*) FROM orders WHERE amount > 1000;
  → SSI records: T-A read predicate {amount > 1000}

Transaction B (concurrent):
  T2: INSERT INTO orders (amount) VALUES (1500);
  → SSI records: T-B write {amount=1500} which overlaps T-A's predicate
  → Creates rw-conflict: T-A rw→ T-B

Transaction A tries to commit:
  → SSI detects T-A read something T-B modified (rw-conflict)
  → If T-B has already committed: T-A's read is now stale
  → Abort T-A with serialization error
  → Application retries T-A (now reads the row T-B inserted)
```

**SSI Implementation Structures:**
```
SIREAD locks (predicate locks — track what was READ):
  Store predicate (e.g., "amount > 1000") with transaction ID
  When a new write overlaps any SIREAD predicate, record conflict

Transaction dependency graph:
  Edges are rw-conflicts between transactions
  Cycle detection on commit → abort one transaction in the cycle
```

**SSI vs 2PL (Two-Phase Locking) Comparison:**
```
                    2PL (traditional)    SSI (PostgreSQL)
Read blocks write?  YES (shared lock)    NO (optimistic)
Write blocks read?  YES (exclusive lock) NO (snapshot reads)
Throughput          Low under contention High read-write concurrency
Anomaly prevention  Serializable         Serializable
Cost on abort       Re-acquire locks     Retry transaction
Best for            Write-heavy, short   Read-heavy, some writes
                    transactions         long transactions
```

**Practical usage:**
```sql
-- Enable serializable for a transaction
BEGIN ISOLATION LEVEL SERIALIZABLE;
SELECT COUNT(*) FROM orders WHERE amount > 1000;
-- Do some computation
INSERT INTO audit_log (count) VALUES (50);
COMMIT;
-- If conflict detected: ERROR 40001 — retry in application
```

---

## Quick Reference

```
B-TREE PROPERTIES
──────────────────────────────────────────────────────────────
Height: O(log N) — 4 levels indexes billions of rows
Leaf nodes linked → efficient range scans
Search/Insert/Delete: O(log N)
Random write: pages modified in-place → write amplification
Used in: PostgreSQL, MySQL InnoDB, SQLite

LSM TREE PROPERTIES
──────────────────────────────────────────────────────────────
Write: MemTable (memory) → SSTable (disk), sequential only
Read:  check MemTable + all SSTable levels → read amplification
Bloom filter: skip SSTables that don't contain key
Compaction: merge SSTables → reduce read amplification
Used in: RocksDB, Cassandra, LevelDB, HBase

AMPLIFICATION COMPARISON
──────────────────────────────────────────────────────────────
            B-Tree       LSM Tree
Write amp   HIGH         LOW
Read amp    LOW          HIGH (mitigated by bloom filters)
Space amp   MEDIUM       MEDIUM

MVCC KEY FIELDS
──────────────────────────────────────────────────────────────
xmin: txn that created this row version
xmax: txn that deleted/updated this row version (0 = live)
VACUUM: reclaims dead tuples (xmax committed + no active readers)

ISOLATION LEVELS vs ANOMALIES
──────────────────────────────────────────────────────────────
READ COMMITTED     → dirty reads prevented; phantoms possible
REPEATABLE READ    → PG: phantoms prevented; MySQL: possible
SERIALIZABLE       → all anomalies prevented (SSI in PostgreSQL)

WAL PURPOSE
──────────────────────────────────────────────────────────────
Write WAL (sequential) before data pages (random)
On crash: replay WAL from last checkpoint → consistent state
Also enables: streaming replication, PITR

KEY BUFFER POOL SETTINGS
──────────────────────────────────────────────────────────────
PostgreSQL: shared_buffers = 25-40% of RAM
MySQL:      innodb_buffer_pool_size = 70-80% of RAM

CONNECTION POOLING
──────────────────────────────────────────────────────────────
New PostgreSQL connection: 15–60ms
PgBouncer transaction mode: < 1ms to get pooled connection
Max useful connections: ~100 (beyond this, context switching hurts)
Pool size formula: num_cores * 2 + spindles

SLOW QUERY DIAGNOSIS STEPS
──────────────────────────────────────────────────────────────
1. pg_stat_statements → identify slow query
2. EXPLAIN (ANALYZE, BUFFERS) → find Seq Scan, bad estimates
3. Check ANALYZE freshness (stale stats → bad plans)
4. Check for missing index on WHERE/JOIN columns
5. Check pg_stat_activity for lock contention
6. Check connection count and state distribution

INDEX BLOAT FIX
──────────────────────────────────────────────────────────────
REINDEX INDEX CONCURRENTLY idx_name;  -- online, no table lock
Prevention: autovacuum, sequential IDs, fillfactor = 70–80

BLOOM FILTER
──────────────────────────────────────────────────────────────
10 bits/key, 1% false positive rate
"Definitely absent" → skip entire SSTable (no disk I/O)
"Possibly present" → read SSTable to verify
```
