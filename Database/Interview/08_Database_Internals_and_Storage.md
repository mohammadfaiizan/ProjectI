# Database Internals and Storage — Interview Questions

## Easy (Q1–Q8)

---

**Q1. What is the difference between InnoDB and MyISAM in MySQL?**

| Feature | InnoDB | MyISAM |
|---|---|---|
| Transactions | Yes (ACID) | No |
| Foreign keys | Yes | No |
| Locking | Row-level | Table-level |
| Crash recovery | Yes (redo log) | No (manual repair) |
| Full-text search | Yes (MySQL 5.6+) | Yes |
| Clustered index | Yes (PK = clustered) | No (heap storage) |
| MVCC | Yes | No |
| Default since | MySQL 5.5 | Before 5.5 |

InnoDB is the only serious choice for production workloads today. MyISAM survives mainly for read-heavy reporting tables in legacy systems.

---

**Q2. What is a database page, and why does its size matter?**

A **page** (also called a block) is the smallest unit of I/O between memory and disk. PostgreSQL uses 8 KB pages by default; MySQL InnoDB uses 16 KB.

```
Disk ──read──▶ Buffer Pool (pages in memory) ──▶ Query Engine
              └── dirty pages written back to disk
```

Page size matters because:
- **Reads are whole-page** — reading one row fetches the entire 8/16 KB page into the buffer pool
- **Writes are whole-page** — a one-byte change dirtied the entire page, which must be flushed
- **Index branching factor** — larger pages fit more index keys → shallower B-tree → fewer I/Os
- **Small rows benefit** from larger pages (more rows per page → fewer reads for sequential scans)

---

**Q3. What is the buffer pool (buffer cache) and why is it important?**

The buffer pool is an area of RAM where the database caches data pages and index pages. Keeping frequently accessed pages in memory avoids slow disk reads.

```sql
-- MySQL: check buffer pool hit rate
SELECT FORMAT(
    (1 - innodb_buffer_pool_reads / innodb_buffer_pool_read_requests) * 100, 2
) AS hit_rate_pct
FROM information_schema.INNODB_METRICS
WHERE name IN ('buffer_pool_reads', 'buffer_pool_read_requests');
-- Target: > 99%

-- PostgreSQL: check shared_buffers hit rate
SELECT sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) AS ratio
FROM pg_statio_user_tables;
```

Tuning:
- **MySQL** `innodb_buffer_pool_size` — typically 70–80% of available RAM
- **PostgreSQL** `shared_buffers` — typically 25% of RAM; OS page cache handles the rest

---

**Q4. What is Write-Ahead Logging (WAL)?**

WAL (called the redo log in MySQL) is the mechanism by which databases guarantee durability without flushing data pages to disk on every write.

**How it works:**
1. Before any data page is modified, the change is first written to the WAL log (sequential write — fast)
2. The transaction is considered durable once the WAL record is flushed to disk (`fsync`)
3. The actual data page is written to disk later (asynchronously, by the background writer)
4. On crash recovery: replay WAL records to bring data pages up to date

```
Write path:
SQL statement → modify page in buffer pool → write WAL record → fsync WAL → COMMIT returns
                                                                ↓ (async)
                                                         dirty page flushed to disk
```

**Benefits:**
- Sequential WAL writes are much faster than random data page writes
- Groups many small changes into fewer large I/Os
- Enables point-in-time recovery and replication

---

**Q5. What is a checkpoint in a database?**

A checkpoint is a point at which the database guarantees that all dirty pages (modified data pages in the buffer pool) have been flushed to disk. After a checkpoint, crash recovery only needs to replay WAL records written after the checkpoint.

```
WAL:  ──[CP1]──── changes ────[CP2]──── changes ────[crash]
Disk: all pages current at CP1            all pages current at CP2
Recovery: only replay WAL after CP2
```

**PostgreSQL checkpoint settings:**
```
checkpoint_completion_target = 0.9   -- spread writes over 90% of checkpoint interval
checkpoint_timeout = 5min            -- force checkpoint every 5 minutes
max_wal_size = 1GB                   -- trigger checkpoint if WAL grows past 1GB
```

Frequent checkpoints = less recovery time but more I/O. Rare checkpoints = more recovery time but smoother I/O.

---

**Q6. What is the difference between a clustered and a non-clustered (secondary) index at the storage level?**

**Clustered index:**
- The table's actual row data is stored in the leaf nodes of the B-tree
- There is exactly one clustered index per table
- In MySQL InnoDB, this is always the primary key
- Row lookup by PK = navigate B-tree and row data is right there

```
PK B-tree leaf nodes:
[id=1 | name=Alice | salary=50000]
[id=2 | name=Bob   | salary=60000]
[id=3 | name=Carol | salary=70000]
```

**Non-clustered (secondary) index:**
- Leaf nodes contain the index key + a pointer to the actual row
- In InnoDB, the pointer is the primary key value (not a physical row address)
- A secondary index lookup requires: find PK in secondary B-tree → look up row in clustered B-tree (double B-tree traversal)

```
Secondary index on (name):
[name=Alice | PK=1]  →  lookup PK=1 in clustered index
[name=Bob   | PK=2]  →  lookup PK=2 in clustered index
```

This is why covering indexes matter: if the secondary index includes all needed columns, the second lookup can be skipped.

---

**Q7. What is table fragmentation and how is it addressed?**

Fragmentation occurs when pages are poorly utilized (many empty slots from deleted rows) or when rows are physically scattered across pages instead of being logically adjacent.

**Causes:**
- Heavy DELETE operations leave empty space in pages
- Random-order inserts (e.g., UUID primary keys) scatter rows across pages
- Updates that increase row size force row migration to a new page

**Effects:**
- Sequential scans read more pages than necessary
- Index range scans require more I/Os
- Buffer pool is used less efficiently

**Solutions:**
```sql
-- MySQL InnoDB
OPTIMIZE TABLE employees;         -- rebuilds table and indexes, reclaims space

-- PostgreSQL
VACUUM FULL employees;            -- rewrites table, full table lock
CLUSTER employees USING idx_id;   -- rewrites table ordered by index (blocks table)

-- SQL Server
ALTER INDEX ALL ON employees REBUILD;         -- rebuild all indexes
ALTER INDEX idx_name ON employees REORGANIZE; -- online defrag (less intrusive)
```

---

**Q8. What is the difference between OLTP and OLAP storage requirements?**

| Aspect | OLTP | OLAP |
|---|---|---|
| Query type | Point lookups, small inserts/updates | Large scans, aggregations |
| Row count per query | Few rows | Millions of rows |
| Schema | Normalized (3NF) | Denormalized (star/snowflake) |
| Storage layout | Row-oriented | Column-oriented preferred |
| Index strategy | Many narrow indexes for fast lookup | Few wide indexes or no index (full scan) |
| Concurrency | High (thousands of transactions/sec) | Low (batch queries) |
| Data freshness | Real-time | Periodic ETL / streaming |
| Examples | PostgreSQL, MySQL, SQL Server | Redshift, BigQuery, Snowflake, ClickHouse |

**Column-oriented storage for OLAP:** stores all values of a column together, enabling:
- Massive compression (same data type, often repeated values)
- Vectorized processing (CPU SIMD operations on column arrays)
- Skip reading columns not in the query

---

## Medium (Q9–Q15)

---

**Q9. How does a B-tree index work internally? Walk through a lookup.**

A B-tree (balanced tree) index is a self-balancing tree where all leaf nodes are at the same depth and linked in a doubly-linked list.

```
Structure (order = 3):
                    [30 | 70]                    ← root
                  /    |    \
           [10|20]  [40|60]  [80|90]             ← internal nodes
             |         |          |
       [10][20]   [40][60]   [80][90]            ← leaf nodes (linked →)
```

**Lookup of `id = 40`:**
1. Read root page: 40 > 30 and 40 < 70 → go to middle child
2. Read internal node [40|60]: 40 ≤ 40 → go to left child (leaf)
3. Read leaf page: find id=40, return row pointer (or full row for clustered)
4. Total I/Os: **tree height** = O(log_B N) where B = branching factor (~hundreds)

**Range scan of `id BETWEEN 40 AND 80`:**
1. Find leaf page containing 40 via top-down traversal
2. Scan rightward through linked leaf pages until 80
3. Much faster than full table scan

**Height example:**
- 1 million rows, page size 8KB, keys average 8 bytes + pointer 6 bytes → ~570 keys per internal page
- Height ≈ log₅₇₀(1,000,000) ≈ 2.6 → 3 levels
- 3 page reads to find any row in a million-row table

---

**Q10. What is the InnoDB row format and how does it affect storage?**

InnoDB stores rows in four possible formats:

| Format | Variable columns | Large columns | Notes |
|---|---|---|---|
| COMPACT | Page (inline up to 768 B) | Overflow pages | Default before 5.7 |
| DYNAMIC | Page (inline short; overflow if large) | Off-page BLOB pointer | Default MySQL 5.7+ |
| COMPRESSED | Compressed B-tree pages (KEY_BLOCK_SIZE) | Off-page BLOB | Saves space, more CPU |
| REDUNDANT | Legacy (stores field lengths) | Overflow | Oldest format |

```sql
-- Check row format
SHOW TABLE STATUS LIKE 'employees';
SELECT row_format FROM information_schema.TABLES WHERE table_name = 'employees';

-- Set row format
CREATE TABLE t (id INT, data TEXT) ROW_FORMAT=DYNAMIC;
ALTER TABLE t ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;
```

**DYNAMIC behavior:** VARCHAR/BLOB columns ≤ 40 bytes stay on the main page; longer ones use overflow pages. A row with many large TEXT columns may reference many overflow pages, making full-row reads expensive.

---

**Q11. How does PostgreSQL handle updates internally (HOT updates)?**

A normal UPDATE in PostgreSQL:
1. Marks the old row version as dead (sets `xmax`)
2. Inserts a new row version (new `xmin`)
3. Updates all indexes to point to the new row version

This is expensive when a frequently-updated column is not indexed.

**HOT (Heap-Only Tuple) update** is an optimization triggered when:
- The updated column is NOT covered by any index
- The new row fits on the same page as the old row

```
Normal UPDATE:
  Old row (dead) ← all index entries updated to new row
  New row (live)  ↑

HOT UPDATE:
  Old row (dead) → HOT chain pointer → New row (live)
  Indexes NOT updated — they still point to old row
  PostgreSQL follows the HOT chain automatically
```

HOT updates:
- Avoid index updates (major write amplification reduction)
- Reduce index bloat
- Are transparent to queries

Implication: setting `fillfactor < 100` leaves free space on data pages, allowing new row versions to land on the same page and enabling HOT updates more often.

```sql
CREATE TABLE employees (...) WITH (fillfactor = 80);
ALTER TABLE employees SET (fillfactor = 80);
```

---

**Q12. Explain the concept of write amplification and how databases minimize it.**

Write amplification is the ratio of physical bytes written to disk vs. logical bytes changed by the application.

**Sources of write amplification:**
1. **WAL + data page** — every change is written twice (WAL + eventual data page flush)
2. **B-tree page splits** — inserting a row into a full leaf page requires writing the split page + updating parent pointers
3. **Index updates** — one row update triggers N index page writes (one per index)
4. **Copy-on-write storage** — some systems (ZFS, Btrfs, some SSDs) write new pages rather than updating in place

**Minimization techniques:**

| Technique | How It Helps |
|---|---|
| WAL batching | Group many WAL records per `fsync` call |
| `full_page_writes = off` (PostgreSQL) | After a checkpoint, only WAL delta (not full page) — risky without backup filesystem |
| HOT updates | Skip index updates when non-indexed column changes |
| Fillfactor | Leave space in pages to allow in-place HOT chain |
| LSM tree (LevelDB, RocksDB, Cassandra) | Buffer all writes in memory, flush sequentially as sorted files — excellent write throughput at cost of read amplification |
| Larger WAL segments | Fewer file creation/rotation overheads |

---

**Q13. What is a covering index and how does it interact with the storage engine?**

A covering index contains all columns needed by a query — the query can be answered entirely from the index without touching the heap/data pages.

```sql
-- Without covering index: uses index to find rows, then fetches heap for salary
CREATE INDEX idx_dept ON employees (department_id);
SELECT employee_id, salary FROM employees WHERE department_id = 3;
-- Plan: Index Scan on idx_dept → fetch row from heap for salary

-- Covering index (PostgreSQL INCLUDE, SQL Server INCLUDE)
CREATE INDEX idx_dept_covering ON employees (department_id) INCLUDE (employee_id, salary);
SELECT employee_id, salary FROM employees WHERE department_id = 3;
-- Plan: Index Only Scan — no heap access at all
```

**Storage perspective:**
- The INCLUDE columns are stored only in the leaf nodes (not in internal nodes)
- Internal nodes stay small (only the key column) → shallower tree
- Leaf nodes store extra data → leaf pages are larger

**Visibility Map interaction (PostgreSQL):**
Index Only Scan still checks the Visibility Map to determine if heap reads are needed to verify row visibility. After `VACUUM`, most pages are in the VM and can be skipped. For a freshly populated table: nearly 100% VM coverage → nearly zero heap reads.

---

**Q14. How does an LSM tree differ from a B-tree for write-heavy workloads?**

**B-tree:**
- In-place updates (modify existing pages)
- Reads are fast (O(log N))
- Writes are random I/O (writing to existing page positions)
- Fragmentation accumulates, needs periodic defrag

**LSM tree (Log-Structured Merge Tree):**
- All writes go first to an in-memory buffer (MemTable)
- When full, MemTable is flushed to disk as an immutable sorted file (SSTable)
- Background compaction merges and sorts SSTables into larger levels
- Reads must check MemTable + multiple SSTable levels (bloom filters help avoid unnecessary reads)

```
Write path:
  Write → MemTable (RAM) → (when full) flush to SSTable on disk
  WAL (crash recovery for MemTable)

Read path:
  Check MemTable → Check L0 SSTables (most recent first) → L1 → L2 → ...
  Bloom filter: probabilistically skip SSTables that can't contain the key
```

| Metric | B-tree | LSM tree |
|---|---|---|
| Write throughput | Moderate | Very high (sequential writes) |
| Write amplification | Moderate | High during compaction |
| Read performance | Fast | Slower (may check many files) |
| Space amplification | Low | Higher (duplicate keys in levels) |
| Examples | PostgreSQL, MySQL InnoDB | RocksDB, Cassandra, LevelDB |

---

**Q15. What is the doublewrite buffer in MySQL InnoDB and why does it exist?**

The doublewrite buffer protects against **partial page writes** — a scenario where the OS or hardware writes only part of an 8/16 KB page before a crash, leaving a corrupted page that cannot be recovered from WAL alone.

**The problem:** WAL records describe changes at the byte/row level. To apply them during recovery, the original page must be intact. If the original page is half-written (corrupted), recovery cannot proceed.

**Solution:**
```
Write path with doublewrite buffer:
1. Before writing dirty pages to their final locations:
   a. Write all dirty pages to a sequential area called the doublewrite buffer (on disk)
   b. fsync doublewrite buffer
2. Now write dirty pages to their actual locations
3. On crash recovery: if a page looks torn, copy the clean version from the doublewrite buffer
```

**Cost:** Each data page write requires a write to the doublewrite buffer first — roughly doubles sequential write work. On filesystems with atomic writes (ZFS, ext4 with data=journal, file systems on newer SSDs), the doublewrite buffer can be disabled:

```sql
-- MySQL 8.0.20+
innodb_doublewrite = OFF  -- only safe on atomic-write filesystems
```

PostgreSQL uses full_page_writes (WAL contains full page images after each checkpoint) to solve the same problem.

---

## Hard (Q16–Q20)

---

**Q16. Explain PostgreSQL's TOAST mechanism and when it triggers.**

TOAST (The Oversized-Attribute Storage Technique) handles values that are too large to fit in a single 8KB page (PostgreSQL's hard limit: a single row must fit in one page).

**Trigger threshold:** A row wider than approximately 2KB triggers TOAST compression and/or out-of-line storage.

**Storage strategies per column:**

| Strategy | Behavior |
|---|---|
| `PLAIN` | No TOAST. Column must fit on page. |
| `EXTENDED` | Compress first; if still too large, move out of line (default for TEXT, BYTEA) |
| `EXTERNAL` | Move out of line without compression (good for already-compressed data like images) |
| `MAIN` | Compress first; prefer to keep on main page |

```sql
-- Check column storage strategy
SELECT attname, attstorage FROM pg_attribute
WHERE attrelid = 'large_table'::regclass AND attlen = -1;

-- Change strategy
ALTER TABLE docs ALTER COLUMN content SET STORAGE EXTERNAL;
```

**Out-of-line storage:**
- PostgreSQL creates a hidden `pg_toast_<oid>` table
- Large values are split into chunks (≤ 2KB each) stored in the TOAST table
- The main row stores a TOAST pointer (18 bytes)
- TOAST table has its own indexes

**Performance implications:**
- Accessing TOASTed columns requires reading the TOAST table (extra I/O)
- If a query only needs non-TOASTed columns, the TOAST table is never touched
- `SELECT count(*) FROM large_table` is fast even with huge TEXT columns if count uses an index

---

**Q17. How does InnoDB handle concurrent reads and writes using its MVCC implementation?**

InnoDB MVCC uses an **undo log** to reconstruct older row versions, unlike PostgreSQL which stores multiple row versions in the heap.

**Data structures:**
- Each row has hidden columns: `DB_TRX_ID` (last modifying transaction ID), `DB_ROLL_PTR` (pointer into undo log)
- **Undo log** stores the previous version of each modified row

**Read view (snapshot):**
When a transaction starts a consistent read, InnoDB creates a read view containing:
- `low_limit_id`: next transaction ID not yet assigned (all IDs ≥ this are invisible)
- `up_limit_id`: lowest active transaction ID (all IDs < this and committed are visible)
- `trx_ids`: list of active (in-progress) transaction IDs at snapshot time

**Row visibility rule:**
```
For a row with DB_TRX_ID = T:
  If T < up_limit_id and T is committed → row is visible
  If T >= low_limit_id → row is not visible (future transaction)
  If T is in trx_ids (active at snapshot time) → row is not visible (still in progress)
  Otherwise → visible
  If not visible: follow DB_ROLL_PTR to undo log to get previous version; repeat
```

**Purge thread:** InnoDB's background purge thread periodically removes undo log records that are no longer needed by any active read view. Long-running transactions delay purge → undo log grows → performance degrades (history list length).

```sql
-- Check undo log (history list) length
SHOW ENGINE INNODB STATUS;
-- Look for: "History list length NNN" — should be < 1000; high means long transactions or delayed purge
```

---

**Q18. What is index-organized table (IOT) / clustered table design and when is it beneficial?**

In an index-organized table, the row data IS the B-tree leaf node — there is no separate heap/data file. MySQL InnoDB uses this for all tables (the primary key is always clustered). Oracle has explicit IOT syntax.

```sql
-- MySQL InnoDB: every table is an IOT on the primary key
CREATE TABLE events (
    user_id    BIGINT  NOT NULL,
    event_time DATETIME NOT NULL,
    event_type VARCHAR(50),
    payload    JSON,
    PRIMARY KEY (user_id, event_time)  -- determines physical order
);
-- Rows are stored in (user_id, event_time) order on disk
-- Range scan for a user's events is a sequential read
```

**Benefits of IOT/clustered table:**
1. **Range queries on PK** are sequential reads (excellent for user_id + time_range queries)
2. **No secondary lookup** — the PK lookup returns the full row directly
3. **Locality** — related rows (same user) are physically adjacent → fewer page reads

**Drawbacks:**
1. **Random inserts** — if rows are inserted in non-PK order (e.g., UUID PK), page splits occur for nearly every insert
2. **Large PK overhead** — secondary indexes store the full PK as the row pointer → large PK = large secondary indexes
3. **One clustered order** — you pick one access pattern to optimize; other access patterns require secondary indexes

**Best practice:** Use a monotonically increasing PK (BIGINT AUTO_INCREMENT, ULID, UUID v7) for IOT tables to avoid page splits during bulk inserts.

---

**Q19. Explain how PostgreSQL's parallel query execution works and what limits it.**

PostgreSQL can parallelize certain query operations across multiple CPU cores using a leader + worker process model.

**Architecture:**
```
Client → Backend (leader process)
            ↓
    Parallel coordinator
   /         |          \
Worker-1   Worker-2   Worker-3
(reads partial table scan)
            ↓
    Results gathered + merged
```

**Operations that support parallelism:**
- Sequential scans (parallel seq scan)
- Index scans (parallel bitmap heap scan)
- Aggregations (parallel aggregate — workers compute partial aggregates)
- Hash joins (parallel hash join)
- Nested loop joins (limited)
- CREATE INDEX (parallel index build)

```sql
-- Control parallelism
SET max_parallel_workers_per_gather = 4;  -- max worker processes for one node
SET parallel_tuple_cost = 0.1;            -- cost estimate per row for parallel
SET parallel_setup_cost = 1000;           -- startup cost for launching workers

-- Force parallel plan (for testing)
SET min_parallel_table_scan_size = 0;
SET min_parallel_index_scan_size = 0;
SET parallel_leader_participation = on;
```

**Limits and exclusions:**
- Queries inside functions marked `PARALLEL UNSAFE` (any function modifying tables, using cursors, SETOF with side effects)
- Queries accessing temporary tables
- Scrollable cursors
- Plans that use `Append` with mixed parallel-safe and unsafe children
- Transactions using `FOR UPDATE/SHARE`

**EXPLAIN check:**
```sql
EXPLAIN SELECT count(*) FROM orders WHERE created_at > '2024-01-01';
-- Look for: "Gather" node → parallel query in use
--           "Workers Planned: 4" / "Workers Launched: 4"
```

---

**Q20. How would you diagnose and fix a table suffering from severe bloat in PostgreSQL?**

**Step 1: Detect bloat**
```sql
-- Quick check: dead tuple ratio
SELECT schemaname, tablename,
       n_dead_tup,
       n_live_tup,
       round(n_dead_tup::numeric / nullif(n_live_tup + n_dead_tup, 0) * 100, 2) AS dead_pct,
       last_autovacuum,
       last_autoanalyze
FROM pg_stat_user_tables
WHERE n_dead_tup > 10000
ORDER BY dead_pct DESC;

-- Detailed bloat estimate (pgstattuple extension)
CREATE EXTENSION IF NOT EXISTS pgstattuple;
SELECT * FROM pgstattuple('orders');
-- Look at: dead_tuple_percent, free_percent
```

**Step 2: Understand the cause**
```sql
-- Is autovacuum running?
SELECT * FROM pg_stat_activity WHERE query LIKE '%autovacuum%';

-- Is a long transaction blocking vacuum?
SELECT pid, now() - xact_start AS txn_age, query
FROM pg_stat_activity
WHERE xact_start IS NOT NULL
ORDER BY xact_start
LIMIT 10;
-- Kill a blocker if safe:
SELECT pg_terminate_backend(pid);
```

**Step 3: Immediate relief**
```sql
-- Online vacuum (no lock, marks space reusable but doesn't return to OS)
VACUUM VERBOSE ANALYZE orders;

-- If autovacuum is behind, temporarily boost it for this table
ALTER TABLE orders SET (autovacuum_vacuum_scale_factor = 0.01,
                        autovacuum_vacuum_threshold = 100);
```

**Step 4: Reclaim space to OS (requires exclusive lock)**
```sql
-- Option A: VACUUM FULL (full table lock, rewrites table)
VACUUM FULL orders;

-- Option B: pg_repack extension (online repack, no full table lock)
-- Install: apt install postgresql-<ver>-repack
pg_repack --table orders -d mydb
-- pg_repack builds new table and swaps atomically using triggers
```

**Step 5: Prevent recurrence**
```sql
-- Tune autovacuum per table
ALTER TABLE orders SET (
    autovacuum_vacuum_cost_delay = 2,       -- less throttling
    autovacuum_vacuum_scale_factor = 0.05,  -- vacuum when 5% of rows are dead
    autovacuum_vacuum_threshold = 500       -- always vacuum if > 500 dead rows
);

-- Set fillfactor to allow HOT updates
ALTER TABLE orders SET (fillfactor = 80);
-- CLUSTER or rebuild to apply fillfactor:
VACUUM FULL orders;  -- or pg_repack
```

**Decision tree:**
```
Dead % < 10%    → normal, autovacuum should handle it
Dead % 10–30%   → VACUUM ANALYZE, tune autovacuum
Dead % > 30%    → investigate blockers, pg_repack or VACUUM FULL in maintenance window
Table is huge   → always use pg_repack (avoids multi-hour table lock)
```

---

## Quick Reference

```sql
-- MySQL storage engine info
SHOW TABLE STATUS FROM mydb;
SHOW ENGINE INNODB STATUS;
SELECT * FROM information_schema.INNODB_BUFFER_PAGE_LRU LIMIT 10;

-- PostgreSQL storage info
SELECT * FROM pg_stat_user_tables;
SELECT * FROM pg_statio_user_tables;
SELECT * FROM pg_stat_bgwriter;
SELECT relpages, reltuples FROM pg_class WHERE relname = 'orders';

-- Fillfactor
ALTER TABLE t SET (fillfactor = 80);

-- TOAST strategy
ALTER TABLE docs ALTER COLUMN content SET STORAGE EXTERNAL;

-- Parallel query
SET max_parallel_workers_per_gather = 4;
SET parallel_leader_participation = on;

-- Bloat remediation
VACUUM VERBOSE ANALYZE orders;
VACUUM FULL orders;               -- with exclusive lock
-- pg_repack orders               -- online, no full lock

-- Check WAL
SELECT pg_current_wal_lsn();
SELECT pg_walfile_name(pg_current_wal_lsn());
SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), '0/0') / 1024 / 1024 AS wal_mb;
```
