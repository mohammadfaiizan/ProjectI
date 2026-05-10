# Database Performance Monitoring and Observability

---

## Easy (Q1–Q7)

---

**Q1. What is `pg_stat_statements` and why is it the first tool you enable on a production PostgreSQL instance?**

`pg_stat_statements` is a PostgreSQL extension that tracks execution statistics for every distinct query seen by the server — total calls, total/mean/min/max execution time, rows returned, buffer hits/reads, and planning time.

Enable it:
```sql
-- postgresql.conf
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.track = all          -- top (default) | all | none
pg_stat_statements.max = 10000

-- After restart
CREATE EXTENSION pg_stat_statements;
```

Key query to find slow queries:
```sql
SELECT
    left(query, 80)            AS query_snippet,
    calls,
    round(mean_exec_time::numeric, 2) AS mean_ms,
    round(total_exec_time::numeric / 1000, 2) AS total_sec,
    round(stddev_exec_time::numeric, 2) AS stddev_ms,
    rows / calls               AS avg_rows
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
```

Why it is essential: it answers "what query is consuming the most total database time?" — not just the slowest single execution but the cumulative cost of frequent queries.

---

**Q2. What information does `EXPLAIN ANALYZE` provide that `EXPLAIN` alone does not?**

`EXPLAIN` shows the query plan the planner *intends* to use with estimated row counts and costs.

`EXPLAIN ANALYZE` actually *executes* the query and adds:
- Actual rows returned vs estimated rows
- Actual time spent at each node (ms)
- Number of loops (how many times each node was executed)
- Buffer statistics (with `BUFFERS` option)

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM orders WHERE customer_id = 42 AND status = 'pending';
```

Critical things to look for:
| Signal | Meaning |
|--------|---------|
| Actual rows >> Estimated rows | Stale statistics — run `ANALYZE` |
| Seq Scan on large table | Missing index or low selectivity |
| Nested Loop with many loops | N+1 problem at DB layer |
| Hash Batches > 1 | `work_mem` too low for hash join |
| Buffers: read >> hit | Cold cache or I/O bottleneck |

**Warning:** `EXPLAIN ANALYZE` runs the query — use `BEGIN; EXPLAIN ANALYZE …; ROLLBACK;` for write queries to avoid side effects.

---

**Q3. What is the slow query log and how do you enable it in PostgreSQL and MySQL?**

**PostgreSQL:**
```ini
# postgresql.conf
log_min_duration_statement = 1000   # log queries taking > 1 second (ms)
log_duration = off                   # don't log all statement durations
log_statement = 'none'               # don't log all statements
```

**MySQL:**
```ini
# my.cnf
slow_query_log = ON
slow_query_log_file = /var/log/mysql/slow.log
long_query_time = 1          # seconds
log_queries_not_using_indexes = ON
```

**Analysis tools:**
- `pgBadger` — parses PostgreSQL logs, generates HTML reports with query frequency, duration percentiles, lock waits
- `pt-query-digest` (Percona Toolkit) — parses MySQL slow log, groups similar queries, shows percentile latencies
- `mysqldumpslow` — built-in MySQL tool for basic slow log analysis

**Production tip:** Start with `long_query_time = 5` to capture obvious problems, then lower to 1 or even 0.1 as you improve.

---

**Q4. What key metrics should you monitor for a production database?**

**Latency:**
- Query response time (p50, p95, p99) — from application APM or pg_stat_statements
- Transaction throughput (TPS/QPS)
- Replication lag (seconds behind primary)

**Throughput:**
- Rows read/written per second
- Bytes transferred per second (network I/O)

**Saturation:**
- CPU utilization
- Disk I/O utilization (iostat: `%util`, await time)
- Active connections vs max_connections
- Wait events (lock waits, I/O waits)

**Errors:**
- Deadlocks per minute (`pg_stat_database.deadlocks`)
- Rollback rate (high rollbacks → application errors or lock contention)
- Connection errors / refused connections

**Cache efficiency:**
```sql
-- Buffer cache hit ratio (target > 99%)
SELECT
    sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read) + 1) AS hit_ratio
FROM pg_statio_user_tables;
```

**Disk:**
- Table and index bloat percentage
- WAL generation rate (GB/hour)
- Free disk space (alert at < 20%)

---

**Q5. What is `pg_stat_activity` and what can you learn from it?**

`pg_stat_activity` shows one row per server process with:
- `pid` — process ID
- `usename`, `datname` — who and which DB
- `state` — `active`, `idle`, `idle in transaction`, `idle in transaction (aborted)`
- `wait_event_type`, `wait_event` — what the backend is waiting for
- `query` — current or last SQL
- `query_start`, `state_change` — timing
- `client_addr` — which application host

**Detecting problems:**
```sql
-- Long-running queries (> 30 seconds)
SELECT pid, now() - query_start AS duration, query, state, wait_event
FROM pg_stat_activity
WHERE state = 'active'
  AND query_start < now() - interval '30 seconds'
ORDER BY duration DESC;

-- Idle in transaction (holding locks)
SELECT pid, now() - state_change AS idle_duration, query
FROM pg_stat_activity
WHERE state = 'idle in transaction'
ORDER BY idle_duration DESC;

-- Lock waiters
SELECT pid, wait_event_type, wait_event, query
FROM pg_stat_activity
WHERE wait_event_type = 'Lock';
```

**Kill a rogue query:**
```sql
SELECT pg_cancel_backend(pid);   -- sends SIGINT (graceful)
SELECT pg_terminate_backend(pid); -- sends SIGTERM (forceful)
```

---

**Q6. What are wait events in PostgreSQL and why do they matter for performance diagnosis?**

Wait events tell you *why* a backend is not making progress. Categories:

| wait_event_type | wait_event examples | Meaning |
|-----------------|---------------------|---------|
| `Lock` | `relation`, `tuple`, `transactionid` | Waiting for a lock held by another transaction |
| `LWLock` | `buffer_mapping`, `wal_write` | Internal lightweight lock contention |
| `IO` | `relcachefileread`, `datafileread` | Waiting for disk I/O |
| `Client` | `ClientRead` | Waiting for client to send next command |
| `IPC` | `BgWorkerShutdown` | Inter-process communication |

**Diagnosis pattern:**
```sql
SELECT wait_event_type, wait_event, count(*)
FROM pg_stat_activity
WHERE state != 'idle'
GROUP BY 1, 2
ORDER BY 3 DESC;
```

If `Lock / relation` dominates → table-level lock contention (check for long DDL).
If `IO / datafileread` dominates → buffer cache miss / I/O bottleneck (increase `shared_buffers` or add I/O).
If `LWLock / buffer_mapping` dominates → very high concurrency thrashing the buffer pool (more CPUs or connection pooling).

---

**Q7. What is `pg_stat_user_tables` and what operational insights does it give?**

```sql
SELECT
    relname              AS table,
    n_live_tup           AS live_rows,
    n_dead_tup           AS dead_rows,
    round(n_dead_tup * 100.0 / (n_live_tup + n_dead_tup + 1), 1) AS dead_pct,
    last_autovacuum,
    last_autoanalyze,
    seq_scan,
    idx_scan,
    n_mod_since_analyze
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC
LIMIT 20;
```

**Insights:**
- `n_dead_tup` high → bloat accumulating, VACUUM not keeping up
- `seq_scan` >> `idx_scan` → sequential scans dominating (missing or unused indexes)
- `last_autovacuum` far in the past → autovacuum may be failing or not scheduled frequently enough
- `n_mod_since_analyze` large → statistics stale, planner making bad estimates

**Related views:**
- `pg_stat_user_indexes` — index scan counts (find unused indexes: `idx_scan = 0`)
- `pg_statio_user_tables` — heap/index block hits and reads (cache efficiency per table)

---

## Medium (Q8–Q15)

---

**Q8. How do you identify and fix table bloat in PostgreSQL?**

Bloat occurs when dead tuples (from UPDATE/DELETE) accumulate faster than VACUUM reclaims them.

**Detect bloat:**
```sql
-- Bloat estimate query (requires pgstattuple or formula approach)
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    n_dead_tup,
    n_live_tup,
    round(n_dead_tup * 100.0 / (n_live_tup + 1)) AS dead_pct
FROM pg_stat_user_tables
WHERE n_dead_tup > 10000
ORDER BY n_dead_tup DESC;

-- More accurate: use pgstattuple extension
SELECT * FROM pgstattuple('orders');
-- Returns: dead_tuple_count, dead_tuple_percent, free_space, free_percent
```

**Fix options:**

| Method | Downtime | Reclaims disk | Use when |
|--------|----------|---------------|----------|
| `VACUUM table` | None | No (marks free, not returned to OS) | Regular maintenance |
| `VACUUM FULL table` | Table lock | Yes (rewrites table) | Severe bloat, off-hours |
| `CLUSTER table USING idx` | Table lock | Yes + re-orders by index | Also want index order |
| `pg_repack` (extension) | None | Yes (online) | Zero-downtime bloat removal |

**Tune autovacuum:**
```sql
-- Per-table autovacuum tuning for high-churn tables
ALTER TABLE orders SET (
    autovacuum_vacuum_scale_factor = 0.01,  -- vacuum when 1% rows are dead (default 20%)
    autovacuum_analyze_scale_factor = 0.005
);
```

---

**Q9. How do you monitor and diagnose lock contention in PostgreSQL?**

**Find blocked and blocking queries:**
```sql
SELECT
    blocked.pid                  AS blocked_pid,
    blocked.query                AS blocked_query,
    blocking.pid                 AS blocking_pid,
    blocking.query               AS blocking_query,
    blocked.wait_event
FROM pg_stat_activity AS blocked
JOIN pg_stat_activity AS blocking
    ON blocking.pid = ANY(pg_blocking_pids(blocked.pid))
WHERE blocked.cardinality(pg_blocking_pids(blocked.pid)) > 0;
```

**Simpler version (works in all PG versions):**
```sql
SELECT
    bl.pid                     AS blocked_pid,
    bl.query                   AS blocked_query,
    kl.pid                     AS blocking_pid,
    kl.query                   AS blocking_query
FROM pg_locks bl
JOIN pg_stat_activity bl_act ON bl.pid = bl_act.pid
JOIN pg_locks kl ON bl.transactionid = kl.transactionid
    AND bl.pid != kl.pid
JOIN pg_stat_activity kl_act ON kl.pid = kl_act.pid
WHERE NOT bl.granted;
```

**Lock types by severity:**
- `RowExclusiveLock` — normal DML (INSERT/UPDATE/DELETE)
- `ShareLock` — SELECT FOR SHARE
- `ExclusiveLock` — SELECT FOR UPDATE
- `AccessExclusiveLock` — DDL (ALTER TABLE, DROP INDEX) — blocks all reads and writes

**Prevention:**
- Keep transactions short
- Acquire locks in consistent order
- Use `NOWAIT` / `SKIP LOCKED` to fail fast instead of waiting
- Use `CREATE INDEX CONCURRENTLY`, `ALTER TABLE ... ADD CONSTRAINT ... NOT VALID`
- Set `lock_timeout = '5s'` to prevent indefinite waits

---

**Q10. What is connection pool monitoring and what metrics matter?**

**PgBouncer monitoring:**
```
# Connect to PgBouncer admin console
psql -p 6432 -U pgbouncer pgbouncer

SHOW POOLS;
-- cl_active: clients using server connections
-- cl_waiting: clients queued waiting for a connection
-- sv_active: server connections in use
-- sv_idle: server connections available
-- sv_used: server connections last used but idle > server_check_delay

SHOW CLIENTS;   -- connected clients and their state
SHOW SERVERS;   -- backend connections and their state
SHOW STATS;     -- per-database throughput (total_xact_count, avg_xact_time)
```

**Alerts to set:**
- `cl_waiting > 0` for sustained periods → pool exhausted, increase `pool_size` or optimize queries
- `sv_idle = 0` continuously → maxed out, need more connections or faster queries
- `avg_xact_time` rising → slow queries, investigate with pg_stat_statements

**PostgreSQL side:**
```sql
-- Active connections vs max_connections
SELECT count(*), max_conn, count(*) * 100 / max_conn AS pct_used
FROM pg_stat_activity,
     (SELECT setting::int AS max_conn FROM pg_settings WHERE name = 'max_connections') s
GROUP BY max_conn;

-- Connections by state
SELECT state, count(*) FROM pg_stat_activity GROUP BY state;
```

---

**Q11. How do you implement database performance baselines and anomaly detection?**

**Step 1: Collect baseline metrics**

Use time-series metrics collection (Prometheus + postgres_exporter or Datadog):
```yaml
# Key metrics to export:
- pg_stat_statements_mean_exec_time_ms      # per query
- pg_stat_database_tup_fetched             # rows fetched/second  
- pg_stat_database_xact_commit             # TPS
- pg_stat_bgwriter_buffers_checkpoint      # checkpoint pressure
- pg_replication_slot_lag_bytes            # replica lag
- pg_stat_activity_count{state="active"}   # active connections
```

**Step 2: Define SLOs**

```
Query p99 latency < 100ms (critical path queries)
Query p99 latency < 500ms (background queries)
Replication lag < 30 seconds
Connection utilization < 80%
Buffer cache hit ratio > 99%
Dead tuple pct < 10% on any table
```

**Step 3: Alert thresholds (examples in Prometheus/AlertManager format)**

```yaml
- alert: SlowQueryP99
  expr: histogram_quantile(0.99, pg_stat_statements_exec_time_bucket) > 500
  for: 5m

- alert: ReplicationLagHigh
  expr: pg_replication_lag > 30
  for: 2m

- alert: ConnectionsSaturated
  expr: pg_stat_activity_count / pg_settings_max_connections > 0.85
  for: 1m
```

**Step 4: Anomaly detection approach**
- Compare current metric to rolling 7-day same-hour-of-week baseline
- Alert if deviation > 2 standard deviations
- Tools: Prometheus anomaly detection rules, Datadog anomaly monitors, or custom ML models

---

**Q12. What is auto_explain and how does it help in production performance tuning?**

`auto_explain` is a PostgreSQL contrib module that automatically logs query plans for slow queries without requiring manual `EXPLAIN ANALYZE`.

```ini
# postgresql.conf
shared_preload_libraries = 'auto_explain'
auto_explain.log_min_duration = 1000    # log plans for queries > 1s
auto_explain.log_analyze = on           # include actual rows/time (costs extra)
auto_explain.log_buffers = on           # include buffer stats
auto_explain.log_format = json          # JSON for structured log parsing
auto_explain.log_nested_statements = on # also log statements inside functions
auto_explain.sample_rate = 0.1          # only log 10% of qualifying queries (reduces log volume)
```

**What appears in the log:**
```
LOG:  duration: 1523.421 ms  plan:
{
  "Node Type": "Hash Join",
  "Actual Rows": 45234,
  "Plans": [
    { "Node Type": "Seq Scan", "Relation Name": "orders",
      "Actual Rows": 1000000, "Actual Loops": 1 },
    { "Node Type": "Hash", "Plans": [
        { "Node Type": "Index Scan", "Index Name": "idx_products_id" }
    ]}
  ]
}
```

**Advantage over pg_stat_statements:** captures the actual plan used at the time of slowness — useful when plans change due to parameter sniffing or statistics drift.

---

**Q13. How do you detect and fix index bloat?**

Indexes accumulate bloat from UPDATE and DELETE operations (old index entries remain until VACUUM cleans them).

**Detect index bloat:**
```sql
-- Check index sizes and usage
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;

-- pgstattuple for accurate bloat (extension required)
SELECT * FROM pgstatindex('idx_orders_customer_id');
-- avg_leaf_density should be > 70%; lower = bloated

-- Or use the public bloat estimation query from check_postgres
```

**Identifying unused indexes (candidates for removal):**
```sql
SELECT indexrelname, idx_scan, pg_size_pretty(pg_relation_size(indexrelid))
FROM pg_stat_user_indexes
WHERE idx_scan = 0
  AND indexrelid NOT IN (SELECT conindid FROM pg_constraint)  -- exclude PK/FK
ORDER BY pg_relation_size(indexrelid) DESC;
```

**Fix options:**

```sql
-- Non-blocking rebuild (PostgreSQL 12+)
REINDEX INDEX CONCURRENTLY idx_orders_customer_id;

-- Rebuild all indexes on a table concurrently
REINDEX TABLE CONCURRENTLY orders;

-- Or drop and recreate concurrently
DROP INDEX CONCURRENTLY idx_orders_customer_id;
CREATE INDEX CONCURRENTLY idx_orders_customer_id ON orders(customer_id);
```

**Note:** `REINDEX CONCURRENTLY` requires PostgreSQL 12+. On older versions, use the drop-and-recreate approach.

---

**Q14. What is the difference between database-level and query-level performance tuning?**

**Database-level tuning (configuration):**
```ini
# Memory
shared_buffers = 25% of RAM          # PostgreSQL buffer cache
effective_cache_size = 75% of RAM    # planner hint about OS cache
work_mem = RAM / (max_connections * 2)  # per sort/hash operation

# WAL/checkpoint
wal_buffers = 64MB
checkpoint_completion_target = 0.9   # spread checkpoint writes
max_wal_size = 4GB                   # allow longer checkpoints

# Parallelism
max_parallel_workers_per_gather = 4  # parallel query workers
max_parallel_workers = 8

# Autovacuum
autovacuum_max_workers = 5
autovacuum_vacuum_cost_delay = 2ms   # less aggressive for SSDs
```

**Query-level tuning:**
1. Add missing indexes (EXPLAIN ANALYZE shows Seq Scan → consider index)
2. Rewrite inefficient queries (avoid functions on indexed columns, use EXISTS instead of IN)
3. Add covering indexes to avoid table lookups
4. Partition tables to enable partition pruning
5. Materialized views for expensive aggregations
6. Denormalize hot read paths

**Application-level tuning:**
1. Fix N+1 queries (use JOINs or batch loading)
2. Add connection pooling (PgBouncer)
3. Cache results in Redis for read-heavy data
4. Use read replicas for reporting queries

**Priority order:** fix queries first (biggest wins), then schema/indexes, then DB configuration, then infrastructure.

---

**Q15. How do you monitor replication lag and what are acceptable thresholds?**

**PostgreSQL replication lag monitoring:**
```sql
-- On primary: bytes behind
SELECT
    client_addr,
    state,
    write_lag,
    flush_lag,
    replay_lag,
    pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS bytes_behind
FROM pg_stat_replication;

-- On replica: seconds behind
SELECT
    now() - pg_last_xact_replay_timestamp() AS replication_lag_seconds;

-- On replica: LSN position
SELECT pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn(),
       pg_is_in_recovery();
```

**Thresholds (context-dependent):**

| Use case | Acceptable lag | Concern | Critical |
|----------|---------------|---------|----------|
| Read replica for reporting | 60s | 5min | 30min |
| HA standby (auto-failover) | 1s | 10s | 30s |
| Synchronous standby | 0 (by definition) | N/A | N/A |
| Analytics replica | 10min | 1hr | 4hr |

**Causes of high lag:**
1. Heavy write workload (WAL generation exceeds replica apply rate)
2. Long-running queries on replica blocking WAL apply
3. Network bandwidth saturation between primary and replica
4. Replica CPU/disk I/O saturation

**Fix long-running replica queries:**
```sql
-- On replica: find query blocking WAL apply
SELECT pid, query, state, wait_event
FROM pg_stat_activity
WHERE wait_event_type = 'Lock' OR state = 'active';

-- Terminate blocking query on replica
SELECT pg_terminate_backend(pid);
```

---

## Hard (Q16–Q20)

---

**Q16. Design a complete database observability stack for a high-traffic e-commerce platform.**

**Context:** 500K orders/day, 10M active users, PostgreSQL primary + 2 replicas + Redis, PgBouncer pooler. Need to detect performance regressions before users notice.

**Architecture:**

```
[PostgreSQL] → [postgres_exporter] → [Prometheus] → [Grafana]
[PgBouncer]  → [pgbouncer_exporter] →     ↑
[Application]→ [APM (Datadog/NewRelic)]   |
[Slow Logs]  → [pgBadger/Vector] → [Elasticsearch] → [Kibana]
                                          ↓
                                    [AlertManager] → [PagerDuty/Slack]
```

**Key Grafana dashboards:**

**Dashboard 1: Query Performance**
```
Panel 1: Top 20 queries by total time (pg_stat_statements)
Panel 2: p50/p95/p99 latency per query fingerprint (time series)
Panel 3: Queries/second and TPS
Panel 4: Planning time vs execution time ratio (high planning = stats issue)
```

**Dashboard 2: Connection Health**
```
Panel 1: Active / idle / idle-in-transaction connections (stacked)
Panel 2: PgBouncer: cl_waiting over time (pool saturation signal)
Panel 3: Connection error rate
Panel 4: pg_stat_activity state distribution heatmap
```

**Dashboard 3: Storage Health**
```
Panel 1: Tables with highest dead tuple % (bar chart, auto-refreshing)
Panel 2: VACUUM and AUTOVACUUM frequency per table
Panel 3: Bloat trend (GB) per table
Panel 4: Index scan ratio (seq_scan / total scans) — anomaly when rising
Panel 5: WAL generation rate (GB/hr)
Panel 6: Checkpoint frequency and write bytes
```

**Dashboard 4: Replication**
```
Panel 1: Replication lag per replica (seconds) with SLO line
Panel 2: Replication slot oldest_lsn age (bytes)
Panel 3: replica pg_last_xact_replay_timestamp() delta
```

**Alerting rules:**
```yaml
# Tier 1: Page immediately
- query p99 > 2000ms for 2 min
- connections > 90% max_connections
- replication lag > 60s (HA replica)
- disk free < 15%

# Tier 2: Slack notification  
- query p95 > 500ms for 5 min
- dead tuple pct > 20% on any table
- buffer cache hit rate < 98%
- cl_waiting > 5 for 1 min (PgBouncer)
- WAL generation rate doubled vs 24hr avg

# Tier 3: Ticket / daily review
- Unused indexes (idx_scan = 0 for 7 days)
- Slow log queries not in known list
- Connection leaks (idle_in_transaction > 5 min)
```

**Automated remediation (runbook automation):**
```bash
# Auto-trigger ANALYZE when n_mod_since_analyze > 100K on critical tables
# Auto-kill queries running > 5 minutes on read replicas
# Alert + auto-pause non-critical batch jobs when cl_waiting > 20
```

**Key insight:** The most valuable signal is often the *rate of change* — not the absolute value. Replication lag going from 5s to 30s in 2 minutes is more alarming than a steady 30s lag.

---

**Q17. How do you perform a live performance regression investigation when a deploy caused a 3× increase in database response times?**

**Immediate triage (first 5 minutes):**

```sql
-- 1. Check if it's all queries or specific queries
SELECT
    left(query, 100) AS query,
    calls,
    round(mean_exec_time::numeric, 2) AS mean_ms,
    round((mean_exec_time - lag(mean_exec_time) OVER (ORDER BY queryid))::numeric, 2) AS delta_ms
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 20;

-- 2. Check for lock contention
SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock';

-- 3. Check for new sequential scans (missing index)
SELECT relname, seq_scan, idx_scan, 
       round(seq_scan * 100.0 / (seq_scan + idx_scan + 1), 1) AS seq_pct
FROM pg_stat_user_tables
ORDER BY seq_scan DESC
LIMIT 10;

-- 4. Check buffer cache hit ratio (cache eviction?)
SELECT round(sum(heap_blks_hit) * 100.0 / 
             (sum(heap_blks_hit) + sum(heap_blks_read) + 1), 2) AS hit_pct
FROM pg_statio_user_tables;
```

**Compare plans before/after deploy:**
```sql
-- Save the plan of a suspect query
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id
WHERE o.status = 'pending' AND c.tier = 'premium';

-- Compare: did it switch from Index Scan to Seq Scan?
-- Did a Hash Join become a Nested Loop with 1M loops?
```

**Common regression causes and fixes:**

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| New Seq Scan on large table | Deploy added new query without index | `CREATE INDEX CONCURRENTLY` |
| Plan switched from index to seq scan | New code passes NULL or different type | Check bind parameters, cast types |
| All queries slower by same factor | New code holding long transactions | Find and fix long transactions |
| Specific table queries slow | Migration ran `VACUUM FULL` / `CLUSTER` → statistics reset | `ANALYZE table_name` |
| Gradual slowdown post-deploy | New query has N+1 pattern | Fix in application + add index |
| p99 up, p50 unchanged | New code causes occasional lock waits | Find blocking transaction |

**Plan pinning (temporary stabilizer):**
```sql
-- If old plan was better, force it temporarily
ALTER SYSTEM SET enable_hashjoin = off;  -- force merge/nested loop
SELECT pg_reload_conf();
-- Revert after fixing root cause
ALTER SYSTEM RESET enable_hashjoin;
```

**Statistics refresh:**
```sql
-- New query touching columns the planner has stale stats for
ANALYZE orders (status, customer_id);
-- Or full table (may briefly slow down the table)
ANALYZE orders;
```

---

**Q18. How do you implement query performance SLOs with automated rollback triggers?**

**Design goal:** Automatically detect when a deploy degrades query performance beyond SLO thresholds and trigger rollback.

**Step 1: Capture pre-deploy baseline**
```sql
-- Save to a baseline table before each deploy
INSERT INTO query_baselines (deploy_id, query_hash, query_text, mean_ms, p95_ms, calls_per_min)
SELECT
    '2024-01-15-v2.4.1' AS deploy_id,
    queryid,
    left(query, 200),
    mean_exec_time,
    -- p95 approximation using mean + 2*stddev
    mean_exec_time + 2 * stddev_exec_time,
    calls / extract(epoch FROM (now() - stats_reset)) * 60
FROM pg_stat_statements
WHERE calls > 100
ORDER BY total_exec_time DESC
LIMIT 100;
```

**Step 2: Post-deploy comparison job (runs 5 min after deploy)**
```sql
-- Compare current performance to baseline
WITH current_stats AS (
    SELECT queryid, mean_exec_time, calls
    FROM pg_stat_statements
),
comparison AS (
    SELECT
        b.query_text,
        b.mean_ms AS baseline_ms,
        c.mean_exec_time AS current_ms,
        round((c.mean_exec_time / b.mean_ms - 1) * 100, 1) AS pct_regression
    FROM query_baselines b
    JOIN current_stats c USING (queryid)
    WHERE b.deploy_id = 'CURRENT_DEPLOY'
      AND c.calls > 50  -- enough samples
)
SELECT * FROM comparison
WHERE pct_regression > 50   -- flag > 50% regression
ORDER BY pct_regression DESC;
```

**Step 3: CI/CD integration**
```python
# post_deploy_check.py — runs in deployment pipeline
def check_performance_regression(deploy_id, threshold_pct=50):
    regressions = db.execute("""
        SELECT query_text, baseline_ms, current_ms, pct_regression
        FROM compare_to_baseline(%s)
        WHERE pct_regression > %s
          AND baseline_ms > 10  -- ignore trivially fast queries
    """, [deploy_id, threshold_pct]).fetchall()
    
    if len(regressions) > 0:
        # Post to Slack
        notify_team(f"Deploy {deploy_id} regressed {len(regressions)} queries")
        
        # Auto-rollback if critical query regressed
        critical = [r for r in regressions if r.baseline_ms > 100]
        if critical:
            trigger_rollback(deploy_id)
            return False
    
    return True
```

**Step 4: Canary deploy query monitoring**
```
- Route 5% of traffic to new version
- Compare p99 query latency between canary and stable
- If canary p99 > stable p99 * 1.5 → abort deploy
- Tools: Argo Rollouts + custom Prometheus metrics, or Datadog deployment tracking
```

---

**Q19. How do you diagnose a "mystery" slowdown that only happens during peak hours?**

Peak-hour slowdowns are usually not query problems in isolation but resource contention or queueing effects.

**Investigation framework:**

**Step 1: Correlate timing**
```
- Collect: TPS, p99 latency, connection count, CPU %, disk util%
- Plot as time series — find what changes 1-2 min BEFORE latency rises
- Common pattern: connections spike → latency spikes
```

**Step 2: Check resource saturation**
```bash
# During peak: run on DB server
iostat -x 1 10      # disk: look for %util > 80%, await > 10ms
vmstat 1 10         # CPU, memory swap
netstat -s          # TCP retransmits (network saturation)

# PostgreSQL: check checkpoint writes
SELECT * FROM pg_stat_bgwriter;
-- If buffers_checkpoint >> buffers_clean → checkpointer doing too much I/O
```

**Step 3: Check for lock queue buildup**
```sql
-- During peak: run repeatedly
SELECT
    count(*) FILTER (WHERE wait_event_type = 'Lock') AS lock_waiters,
    count(*) FILTER (WHERE state = 'active') AS active,
    count(*) FILTER (WHERE state = 'idle in transaction') AS idle_in_txn
FROM pg_stat_activity;
```

**Step 4: Check for autovacuum interference**
```sql
-- Autovacuum runs during peak and conflicts with writes?
SELECT schemaname, tablename, last_autovacuum, last_autoanalyze,
       autovacuum_count, autoanalyze_count
FROM pg_stat_user_tables
ORDER BY last_autovacuum DESC LIMIT 10;

-- Check autovacuum workers running now
SELECT pid, query, query_start
FROM pg_stat_activity
WHERE query LIKE 'autovacuum:%';
```

**Step 5: Work_mem spill to disk**
```sql
-- Check if sorts/hashes are spilling to disk
-- Look for "Buffers: temp read/written" in EXPLAIN ANALYZE during peak
-- Solution: increase work_mem for heavy queries
SET work_mem = '256MB';  -- session-level for reporting queries
```

**Typical root causes and fixes:**
| Pattern | Root cause | Fix |
|---------|-----------|-----|
| Latency tracks with connection count | Pool exhausted, queueing | Increase pool, optimize slow queries |
| Disk `%util` hits 100% | I/O bound, checkpoint storm | Spread checkpoints, add disk IOPS, move to SSD |
| Lock waiters spike | Batch job runs at peak | Reschedule batch to off-peak |
| Memory usage spikes | work_mem × connections = OOM | Reduce work_mem, use connection pooling |
| Autovacuum during peak | Not enough autovacuum workers | Increase workers, use autovacuum cost delay |

---

**Q20. A critical query that takes 200ms suddenly starts taking 8 seconds after data grows from 10M to 100M rows. Walk through the complete diagnosis and fix.**

**Query:**
```sql
SELECT p.name, p.price, c.name AS category, count(oi.id) AS times_ordered
FROM products p
JOIN categories c ON p.category_id = c.id
JOIN order_items oi ON oi.product_id = p.id
WHERE p.price > 50
  AND c.name = 'Electronics'
  AND oi.created_at > NOW() - INTERVAL '30 days'
GROUP BY p.id, p.name, p.price, c.name
ORDER BY times_ordered DESC
LIMIT 20;
```

**Step 1: Run EXPLAIN ANALYZE to see the plan**
```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
<query above>
```

Hypothetical output reveals:
```
Limit  (cost=... rows=20) (actual rows=20 loops=1)
  -> Sort  (actual time=7823ms)
    -> Hash Join (actual rows=84523 loops=1)
         Hash Cond: (oi.product_id = p.id)
         -> Seq Scan on order_items  (actual rows=9234567 loops=1)
              Filter: (created_at > '2024-01-01')
              Rows Removed by Filter: 78234567
         -> Hash  (actual rows=1234 loops=1)
              -> Hash Join
                   -> Seq Scan on products (actual rows=1000000 loops=1)
                        Filter: (price > 50)
                   -> Index Scan on categories ...
```

**Diagnosis:**
1. `Seq Scan on order_items` scanning 87M rows to filter to 9M — no index on `created_at`
2. `Seq Scan on products` scanning 1M rows — the `price > 50` filter without index
3. Planner estimated far fewer rows before data grew → plan didn't change, statistics stale

**Step 2: Apply fixes**

```sql
-- Fix 1: Index on order_items(created_at) — most critical
CREATE INDEX CONCURRENTLY idx_order_items_created_at 
    ON order_items(created_at);

-- Fix 2: Composite index for the JOIN + filter pattern
-- Query accesses product_id + created_at together
CREATE INDEX CONCURRENTLY idx_order_items_product_created 
    ON order_items(product_id, created_at DESC);

-- Fix 3: Index on products(price) with covering columns
CREATE INDEX CONCURRENTLY idx_products_price_category 
    ON products(category_id, price) 
    INCLUDE (name);

-- Fix 4: Refresh statistics (planner using stale estimates)
ANALYZE order_items;
ANALYZE products;
```

**Step 3: Rewrite query to use the indexes optimally**
```sql
-- Add explicit date range (more selective than interval expression)
-- Move category filter earlier using CTE
WITH electronics AS (
    SELECT id FROM categories WHERE name = 'Electronics'
),
recent_items AS (
    SELECT product_id, count(*) AS times_ordered
    FROM order_items
    WHERE created_at >= (CURRENT_DATE - 30)::timestamp   -- enables partition pruning
    GROUP BY product_id
)
SELECT p.name, p.price, 'Electronics' AS category, ri.times_ordered
FROM recent_items ri
JOIN products p ON p.id = ri.product_id
JOIN electronics e ON p.category_id = e.id
WHERE p.price > 50
ORDER BY ri.times_ordered DESC
LIMIT 20;
```

**Step 4: Consider partitioning for future scale**
```sql
-- Partition order_items by month (range partitioning)
-- enables partition pruning on created_at queries
-- Query on "last 30 days" touches at most 2 partitions instead of full table
```

**Result:** Query drops from 8s → ~50ms:
- `order_items` scan: 87M rows → 2 partition files (~1M rows each) with index
- Products scan: 1M rows → 1M/10 via category index → 100K with price filter
- Overall: ~160x improvement

**Quick Reference**

| pg_stat view | Key use |
|---|---|
| `pg_stat_statements` | Find queries consuming most total DB time |
| `pg_stat_activity` | See active/blocked/idle-in-transaction sessions |
| `pg_stat_user_tables` | Detect bloat, vacuum lag, sequential scan hotspots |
| `pg_stat_user_indexes` | Find unused indexes, index hit ratio |
| `pg_statio_user_tables` | Buffer cache efficiency per table |
| `pg_stat_replication` | Replication lag, WAL send/replay position |
| `pg_stat_bgwriter` | Checkpoint frequency, buffer write distribution |
| `pg_locks` | Active locks; join with `pg_stat_activity` for queries |

| Tool | Purpose |
|---|---|
| `pgBadger` | Parse slow log → HTML report |
| `auto_explain` | Auto-log plans for slow queries in production |
| `pg_repack` | Online bloat removal without locking |
| `postgres_exporter` | Prometheus metrics from PostgreSQL |
| `pt-query-digest` | MySQL slow log analysis |

| Alert threshold | Metric |
|---|---|
| > 99% | Buffer cache hit ratio (alarm if below) |
| > 85% | max_connections utilization |
| > 20% | Dead tuple percentage |
| > 30s | Replication lag (HA replica) |
| > 2× baseline | Query mean execution time post-deploy |
