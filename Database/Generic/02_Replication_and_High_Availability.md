# Database Replication and High Availability

## Easy (Q1–Q7)

---

**Q1. What is database replication and what problems does it solve?**

Replication is the process of maintaining identical copies of data on multiple database servers. One server is designated the **primary** (accepts writes); others are **replicas/standbys** (receive a stream of changes from the primary).

**Problems it solves:**

| Problem | How Replication Helps |
|---|---|
| Single point of failure | Replica can be promoted to primary if primary fails |
| Read throughput bottleneck | Route read queries to replicas, offloading the primary |
| Geographic latency | Place replicas closer to users in other regions |
| Backup without downtime | Take backups from a replica without impacting production |
| Upgrades without downtime | Upgrade replicas first, then promote, then upgrade old primary |

**What replication does NOT solve:**
- Write throughput beyond one server (all writes still go to one primary)
- Data corruption (corruption replicates to replicas too)
- Large dataset that cannot fit on one server (that's sharding)

---

**Q2. What is the difference between synchronous and asynchronous replication?**

**Synchronous replication:**
```
Primary → writes WAL to local disk
       → sends WAL to replica
       → waits for replica ACK
       → returns SUCCESS to client

RPO (Recovery Point Objective): 0 — no committed data can ever be lost
Write latency: local write time + network round trip to replica (adds 1–50ms)
```

**Asynchronous replication:**
```
Primary → writes WAL to local disk
       → returns SUCCESS to client (immediately)
       → ships WAL to replica in the background

RPO: > 0 — if primary crashes before shipping recent WAL, those commits are lost
Write latency: no overhead (replica not in the critical path)
```

**Semi-synchronous (MySQL):**
- Primary waits for at least one replica to confirm it received (but not applied) the WAL
- Prevents data loss when primary crashes; does not require replica to be fully up to date

**PostgreSQL config:**
```
# synchronous_standby_names controls which replicas must ACK before COMMIT returns
synchronous_standby_names = 'FIRST 1 (replica1, replica2)'
# Wait for the first available replica of the listed ones

synchronous_commit = on           -- wait for primary + replica WAL fsync (safest)
synchronous_commit = remote_write -- wait for replica to receive WAL (no fsync there)
synchronous_commit = local        -- only wait for primary local WAL fsync
synchronous_commit = off          -- don't even wait for local fsync (fastest, small loss window)
```

---

**Q3. What is RPO and RTO, and how do they drive replication strategy?**

**RPO (Recovery Point Objective):** Maximum acceptable data loss measured in time.
- RPO = 0: zero data loss allowed (use synchronous replication)
- RPO = 5 minutes: up to 5 minutes of writes can be lost (async replication with frequent backups)

**RTO (Recovery Time Objective):** Maximum acceptable downtime after a failure.
- RTO = 30 seconds: failover must complete within 30 seconds (automated failover with Patroni/MHA)
- RTO = 1 hour: manual failover acceptable (simpler setup, lower cost)

**How they drive strategy:**

| RPO | RTO | Strategy |
|---|---|---|
| 0 | < 30s | Synchronous replication + automated failover (Patroni) |
| < 1 min | < 5 min | Async replication + automated failover + WAL archiving |
| < 1 hour | < 1 hour | Async replication + manual failover + daily backups |
| Hours | Hours | Periodic backups only (dev/test environments) |

**Cost of lower RPO/RTO:** Synchronous replication adds write latency proportional to network RTT to the replica. For cross-region sync replication (US ↔ EU: ~80ms RTT), every write takes 80ms longer.

---

**Q4. What is a primary-replica (master-slave) replication setup and how does failover work?**

```
Primary  ──── WAL streaming ────▶  Replica 1 (sync standby)
                              ────▶  Replica 2 (async standby)

Reads:  route to Replica 1 or Replica 2
Writes: primary only

Failover (primary goes down):
  1. Detect failure (health check fails, heartbeat stops)
  2. Elect new primary (Replica 1 — most up-to-date, had sync replication)
  3. Promote Replica 1: pg_ctl promote / patronictl failover
  4. Point application connection string to Replica 1 (now primary)
  5. Rejoin old primary as replica when it recovers
```

**Tools for automated failover:**

| Tool | Database | What it does |
|---|---|---|
| Patroni | PostgreSQL | Watches cluster via etcd/ZooKeeper; auto-promotes on primary failure |
| PgBouncer | PostgreSQL | Connection pooler; reconnects to new primary transparently |
| AWS RDS Multi-AZ | PostgreSQL/MySQL | Managed sync standby; automatic failover in 60–120s |
| MHA (Master HA) | MySQL | Monitors and promotes on failure |
| ProxySQL | MySQL | Proxy that routes based on replication role |
| Orchestrator | MySQL | Topology management and auto-failover |

---

**Q5. What is replication lag and what problems does it cause?**

Replication lag is the delay between when a write is committed on the primary and when it is visible on the replica.

**Causes:**
- Network latency between primary and replica
- Replica is slower than primary (less CPU/disk)
- Replica is busy processing many previous transactions (streaming behind)
- Long-running transactions on primary block WAL streaming

**Problems caused by replication lag:**

1. **Read-after-write inconsistency:**
```
User updates their email → writes to primary
User's next request routed to replica → sees old email
User thinks their change was lost
```

2. **Replica used for "fresh" data that isn't fresh:**
```
Dashboard reads "current inventory" from replica
Replica is 10 seconds behind → shows inventory as 50 when primary has 0
→ Oversell risk
```

3. **Failover data loss:**
```
Primary crashes. Replica is 5 seconds behind (async).
Failover promotes replica → 5 seconds of writes are lost.
RPO = replication lag at time of failure
```

**Monitoring lag:**
```sql
-- PostgreSQL: check lag from primary
SELECT client_addr,
       sent_lsn,
       write_lsn,
       flush_lsn,
       replay_lsn,
       pg_wal_lsn_diff(sent_lsn, replay_lsn) AS lag_bytes,
       write_lag, flush_lag, replay_lag
FROM pg_stat_replication;
```

**Mitigations:**
- Route reads-after-write to primary (accept DB load for consistency)
- Check replica lag before reading (`replay_lag < 1s`); if lagged, read from primary
- Use synchronous replication for zero-lag critical replicas

---

**Q6. What is logical replication vs physical (streaming) replication in PostgreSQL?**

**Physical (streaming) replication:**
- Replicates at the byte/block level — exact copy of all data files
- Replica is an identical byte-for-byte copy of the primary
- Cannot apply writes to replica (read-only)
- Faster to set up; requires same PostgreSQL version

```
Primary WAL: [page 3, offset 240, bytes: 0x4E3A...] → replica applies binary patch
```

**Logical replication:**
- Replicates individual row changes (INSERT/UPDATE/DELETE) decoded from WAL
- Replica can write to other tables; can replicate to different PostgreSQL versions
- Can replicate specific tables (not entire cluster)
- Supports replication to different data types (heterogeneous targets)
- Replica can have its own indexes, views, additional tables

```
Primary WAL decoded: INSERT INTO orders (id, amount) VALUES (1234, 99.99)
Replica applies: same INSERT on its own local copy
```

| Aspect | Physical | Logical |
|---|---|---|
| Replica read-only | Yes | No (other tables writable) |
| Selective tables | No (all or nothing) | Yes |
| Cross-version | Must match major version | Can differ |
| Use case | Standby HA, read replicas | Selective replication, upgrades, CDC |
| CDC (Debezium) | Not suitable | Yes — logical decoding |

---

**Q7. How do read replicas affect application design?**

**Connection routing:**
```python
# Explicit routing by query type
class Database:
    def __init__(self):
        self.primary = create_pool(PRIMARY_URL)
        self.replica = create_pool(REPLICA_URL)  # round-robin pool of replicas
    
    def write(self, query, *args):
        return self.primary.execute(query, *args)
    
    def read(self, query, *args):
        return self.replica.execute(query, *args)

# Dangerous: reading from replica immediately after writing
def update_user_email(user_id, new_email):
    db.write("UPDATE users SET email = %s WHERE id = %s", new_email, user_id)
    user = db.read("SELECT * FROM users WHERE id = %s", user_id)  # may see old email!
    return user
```

**Read-after-write consistency strategies:**

```python
# Option A: always read from primary for a user's own data
def get_my_profile(user_id):
    return db.primary.execute("SELECT * FROM users WHERE id = %s", user_id)

# Option B: route to primary for N seconds after a write
def update_and_read(user_id, new_email):
    db.write("UPDATE users SET email = %s WHERE id = %s", new_email, user_id)
    redis.setex(f"primary_read:{user_id}", 5, "1")  # force primary reads for 5s

def get_profile(user_id):
    if redis.get(f"primary_read:{user_id}"):
        return db.primary.execute(...)
    return db.replica.execute(...)

# Option C: synchronous replica (lag = 0)
# Use synchronous_standby_names in PostgreSQL to make one replica always current
```

---

## Medium (Q8–Q15)

---

**Q8. What is multi-primary (multi-master) replication and when is it appropriate?**

Multi-primary replication allows multiple nodes to accept writes simultaneously, unlike primary-replica where only one node accepts writes.

```
Primary 1 ◄──────────────────▶ Primary 2
(accepts writes)                (accepts writes)
     ↑ bidirectional replication ↑
Both nodes sync changes to each other
```

**Write conflict problem:**
```
Primary 1: UPDATE users SET email = 'a@x.com' WHERE id = 1
Primary 2: UPDATE users SET email = 'b@x.com' WHERE id = 1  (same row, different value)
Both committed locally → conflict when replicating to each other

Resolution options:
  Last-write-wins (timestamp): higher timestamp wins → risk of data loss
  Application-level: application defines merge rules
  Avoid by design: route each row to only one primary (sticky routing)
```

**Appropriate use cases:**
1. **Geo-distributed writes with data locality:** US users always write to US node; EU users to EU node. Conflict-free if user data is owned by one region.
2. **High availability with zero-downtime failover:** both nodes always active; no failover delay.
3. **Horizontal write scale** (though sharding is usually cleaner)

**Implementations:**
- **MySQL Group Replication / Galera Cluster:** synchronous multi-primary, uses distributed locking (lower throughput)
- **CockroachDB / Spanner:** consensus-based, no conflicts, but higher latency
- **Cassandra:** leaderless multi-primary with LWW conflict resolution

**Rule:** Multi-primary is complex and conflict-prone. Use it only when you have a specific geo-distribution requirement. Primary-replica with automated failover handles most HA requirements simpler.

---

**Q9. How does PostgreSQL streaming replication work internally?**

```
Primary:
  1. WAL records written to local WAL buffers (in-memory)
  2. On COMMIT: WAL flushed to disk (WAL segment files: pg_wal/)
  3. WAL sender process streams new WAL records to connected replicas continuously

Replica:
  1. WAL receiver process connects to primary, requests WAL from last received LSN
  2. Receives WAL records, writes to local pg_wal/ directory
  3. WAL applier process reads from pg_wal/ and applies changes to data files
  4. Sends feedback (replay_lsn) to primary so primary knows it can advance pg_wal/ cleanup

Feedback loop:
  Primary tracks: sent_lsn, write_lsn, flush_lsn, replay_lsn per replica
  Primary won't vacuum/remove WAL until all replicas have confirmed replay past that LSN
  (configures via wal_keep_size or replication slots)
```

**Replication slot:**
```sql
-- Prevents primary from discarding WAL that a replica hasn't consumed yet
SELECT pg_create_physical_replication_slot('replica1_slot');
-- If replica disconnects for days: slot prevents WAL cleanup → WAL grows unboundedly
-- Monitor and set wal_keep_size as a safety cap
```

**Hot standby — reads on replica:**
```
Replica can serve SELECT queries (hot_standby = on)
Replicated transaction visibility:
  Replica tracks its own horizon: only shows rows committed before its consistent snapshot
  Queries on replica may conflict with ongoing WAL application → query cancelled if it conflicts
  max_standby_archive_delay and max_standby_streaming_delay control how long to wait before cancelling
```

---

**Q10. Explain the concept of failover split-brain and how to prevent it.**

**Split-brain:** A scenario where a network partition causes two nodes to both believe they are the primary, and both accept writes. When the partition heals, both have diverged data — it is impossible to merge without data loss.

```
                Network partition
Primary A ↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔ Replica B

A thinks B is dead → A continues accepting writes
B's health check on A times out → B promotes itself to primary
Now both A and B accept writes on the same dataset → split-brain

When partition heals:
  A has written: order #100, #101, #102
  B has written: order #100 (different data), #103
  → Impossible to reconcile without choosing a winner and discarding the other
```

**Prevention mechanisms:**

**1. Quorum / majority consensus:**
```
Only promote to primary if confirmed by majority of nodes (Patroni + etcd/ZooKeeper)
With 3 nodes: need 2 to agree → a partitioned single node cannot form quorum alone

Patroni + etcd:
  Primary holds a distributed lock in etcd
  Lock has a TTL (e.g., 30s)
  If primary cannot renew the lock → it demotes itself
  Replica can only promote by acquiring the lock
  → Only one node holds the lock at any time → no split-brain
```

**2. STONITH (Shoot The Other Node In The Head):**
```
When split-brain is detected: forcibly power off the other node via IPMI/AWS API
  before promoting the replica
→ Ensures old primary is definitely dead before new primary starts writing
```

**3. Fencing tokens:**
```
Each primary epoch is numbered (e.g., epoch 5)
Primary A is at epoch 5 — falls behind network
B promotes at epoch 6
A tries to write: storage layer rejects writes with epoch < 6
→ A's writes are rejected even if A doesn't know it's been replaced
```

---

**Q11. How does database HA differ across AWS RDS Multi-AZ, Aurora, and self-managed PostgreSQL with Patroni?**

**AWS RDS Multi-AZ:**
```
Architecture: Primary in AZ-1 + synchronous standby in AZ-2 (within same region)
Failover: automatic, ~60–120 seconds, DNS-based cutover
RPO: 0 (synchronous replication)
RTO: 1–2 minutes
Read scaling: no (standby not readable — it's a pure failover standby)
Storage: separate EBS volumes (not shared)
Operational overhead: near zero (fully managed)
```

**AWS Aurora (PostgreSQL/MySQL compatible):**
```
Architecture: 6-way storage replication across 3 AZs
  Compute and storage are separated
  Storage layer is a distributed log with quorum writes (4 of 6 copies confirmed)
  Up to 15 read replicas (all share the same storage — minimal lag)
Failover: < 30 seconds (replica becomes primary, storage already up to date)
RPO: 0
RTO: < 30 seconds
Read scaling: excellent — all replicas read from same storage, lag < 20ms
Global Database: async cross-region replica, RPL lag ~1 second
Best for: variable workloads (Aurora Serverless), read-heavy, managed simplicity
```

**Self-managed PostgreSQL + Patroni:**
```
Architecture: You manage nodes; Patroni orchestrates failover via etcd/ZooKeeper/Consul
Failover: typically 10–30 seconds with proper configuration
RPO: 0 with synchronous replication
RTO: 10–30 seconds
Read scaling: add as many read replicas as needed
Control: full — choose instance types, storage, network, PostgreSQL version
Operational overhead: significant — you manage OS, PostgreSQL, Patroni, etcd, monitoring
Best for: when you need full control, specific extensions, or cost optimization at scale
```

| | RDS Multi-AZ | Aurora | Patroni + PG |
|---|---|---|---|
| RPO | 0 | 0 | 0 (sync replica) |
| RTO | 60–120s | < 30s | 10–30s |
| Read scale | No | Yes (15 replicas) | Yes |
| Ops overhead | Very low | Very low | High |
| Cost | Medium | High | Low (infra only) |
| Flexibility | Low | Medium | Full |

---

**Q12. What is a replication cascade and when is it useful?**

A replication cascade is when a replica replicates from another replica rather than from the primary directly.

```
Primary ──▶ Replica 1 ──▶ Replica 2 ──▶ Replica 3
                      ──▶ Replica 4

Primary only streams WAL to Replica 1
Replica 1 streams to Replicas 2, 3, 4
```

**Use cases:**

1. **Cross-region replication (reduce primary WAN load):**
```
Primary (US-East) ──WAN──▶ Regional Hub (EU-West)
Regional Hub (EU-West) ──LAN──▶ EU Replica 1
                        ──LAN──▶ EU Replica 2
                        ──LAN──▶ EU Replica 3

Primary streams one WAN connection; regional hub fans out locally
Reduces WAN bandwidth cost and primary CPU for WAL streaming
```

2. **Tiered read scale:**
```
Primary → Analytics Replica 1 → Analytics Replica 2 (reads analytical reports)
Primary write load unchanged regardless of how many analytics replicas exist
```

3. **Minimize primary connection overhead:**
```
PostgreSQL default: each replica opens one connection to primary
With 100 replicas: 100 connections on primary just for replication
Cascade: 1 connection to primary; cascaded replicas connect to intermediaries
```

**PostgreSQL setup:**
```
# On Replica 1 (the intermediate cascaded source):
primary_conninfo = 'host=primary_db ...'   # receives from primary
wal_level = replica                         # must stream WAL to downstream replicas

# On Replica 2 (points to Replica 1, not Primary):
primary_conninfo = 'host=replica1 ...'     # receives from Replica 1 (not primary)
```

---

**Q13. How do you handle a failover scenario where the promoted replica has replication lag?**

**The problem:**
```
Primary has committed up to LSN 0/A001234
Replica's replay_lsn = 0/A000F00   (replica is behind by 0x334 bytes)
Primary crashes
Replica is promoted to new primary at 0/A000F00
→ Transactions from 0/A000F00 to 0/A001234 are lost
```

**What to do:**

**Before failover (prevention):**
```
1. Use synchronous replication to one standby (RPO = 0):
   synchronous_standby_names = 'replica1'
   → Primary blocks COMMIT until replica1 has flushed WAL
   → On failover, replica1 has ALL committed data

2. Monitor and alert on replication lag:
   SELECT replay_lag FROM pg_stat_replication;
   Alert if > 5 seconds lag (before it becomes a larger loss window)
```

**At failover time:**
```
3. Try to recover primary first:
   Is it a temporary network issue? Wait for primary to recover.
   Only promote replica if primary is definitively dead (confirmed via STONITH or health checks).

4. If you must promote a lagging replica:
   a. Stop all application writes (maintenance mode)
   b. Wait for replica to apply all available WAL
   c. Verify LSN matches as closely as possible
   d. Promote replica
   e. Resume application

5. Patroni handles this automatically:
   Patroni will choose the replica with the highest replay_lsn as the new primary
   Other replicas are rewound to the new primary's timeline using pg_rewind
```

**After failover (data loss audit):**
```sql
-- New primary: check timeline history
SELECT * FROM pg_control_checkpoint();  -- shows timeline ID and LSN

-- Compare with backup to identify what was lost
-- pg_waldump can decode missing WAL records if you saved WAL archives
```

---

**Q14. What is PITR (Point-in-Time Recovery) and how does it work with replication?**

PITR allows you to restore a database to any past moment in time, not just the last backup.

**How it works:**
```
Continuous WAL archiving:
  Every WAL segment (16MB by default) is shipped to durable storage (S3, GCS)
  as soon as it is completed and recycled on the primary

Base backup:
  pg_basebackup creates a consistent filesystem snapshot of all data files
  Taken periodically (daily, weekly)

Recovery:
  1. Restore base backup to a new server
  2. Apply archived WAL segments sequentially up to the target time/LSN
  
Example:
  Base backup: 2024-01-15 00:00
  WAL archives: 2024-01-15 00:00 → 2024-01-15 14:30 (continuous stream)
  Accidental DELETE at 2024-01-15 14:00
  Restore to 2024-01-15 13:59:59 → all data before the mistake
```

**PostgreSQL PITR setup:**
```
# archive_command: called when WAL segment is complete
archive_mode = on
archive_command = 'aws s3 cp %p s3://mybackups/wal/%f'

# Recovery (in recovery.conf / postgresql.conf):
restore_command = 'aws s3 cp s3://mybackups/wal/%f %p'
recovery_target_time = '2024-01-15 13:59:59'
recovery_target_action = 'promote'  -- promote after reaching target time
```

**PITR + replication:**
```
Replicas can also ship WAL to archive → multiple archive sources (redundancy)
If primary WAL archive is corrupted for a segment, another replica's archive can fill the gap
WAL-E, pgBackRest, Barman are tools that manage base backups + WAL archiving together
```

**RPO with PITR:**
```
Last WAL segment interval = 16MB of changes (could be seconds or minutes depending on write rate)
If write rate is 100MB/s: WAL segment completes every 0.16s → near-zero RPO
If write rate is 10KB/s: WAL segment takes 26 minutes → 26 minutes potential data loss
archive_timeout = 60  -- force WAL archive even for partial segments every 60s → max RPO = 60s
```

---

**Q15. How does Aurora's storage architecture provide both HA and read scalability simultaneously?**

**Traditional replication model (e.g., RDS Multi-AZ):**
```
Primary (compute + storage) → copies data → Standby (compute + storage)
Write: written to primary storage, then replicated to standby storage
Read replicas: receive binlog/WAL, apply changes, maintain their own full copy
→ Each replica needs full storage (cost) and full data transfer (bandwidth)
```

**Aurora's architecture:**
```
Compute tier:    Primary (writer) + up to 15 reader instances
Storage tier:    Distributed, shared log-structured storage (6 copies across 3 AZs)

Write path:
  1. Primary writes ONLY to storage tier (redo log records, not data pages)
  2. Storage tier: 4-of-6 quorum write confirmation → commit returned to client
  3. Storage nodes apply redo log to materialize pages in the background

Read path:
  All reader instances connect to the SAME storage tier
  Readers and writer share the same storage — no data copying between them
  Reader lag ≈ 10–20ms (just the time to apply the redo log at the storage layer)
```

**Why this provides both HA and read scale:**
```
HA: storage is always current (6-way replicated, quorum writes)
    Failover = a reader instance is promoted to writer → immediate (storage already up to date)
    No data copying needed during failover

Read scale: all readers share the same storage
    Adding a read replica = add a compute instance (no storage replication overhead)
    15 read replicas add negligible overhead to the writer
    Replica lag is storage-layer propagation only (~20ms), not full replication lag

Traditional read replica vs Aurora read replica:
  Traditional: writer copies full data to each replica → 5 replicas = 5x storage + 5x I/O
  Aurora:      5 replicas share one storage tier → near-zero additional overhead per replica
```

---

## Hard (Q16–Q20)

---

**Q16. Design a zero-downtime database upgrade strategy for a PostgreSQL cluster serving 100K requests/second.**

**Goal:** Upgrade PostgreSQL 14 → 16 with no downtime or data loss.

**Strategy: logical replication for version upgrade**

```
Current:  PostgreSQL 14 (primary) + 2 physical replicas

Phase 1: Set up PG16 in parallel (no traffic yet)
  Provision new PostgreSQL 16 cluster (primary + 1 replica)
  Install same extensions, run pg_upgrade in check mode to verify compatibility

Phase 2: Logical replication from PG14 → PG16
  -- On PG14 primary:
  CREATE PUBLICATION upgrade_pub FOR ALL TABLES;
  wal_level = logical   -- must be set (requires restart — do this in a maintenance window)

  -- On PG16 primary:
  CREATE SUBSCRIPTION upgrade_sub
    CONNECTION 'host=pg14 ...' PUBLICATION upgrade_pub;
  -- PG16 gets initial data copy, then streams changes continuously

Phase 3: Monitor lag
  SELECT * FROM pg_stat_subscription;
  -- Wait until received_lsn ≈ latest_end_lsn (near-zero lag)
  -- Can take hours for initial sync of large databases

Phase 4: Cutover (< 30 seconds of write pause)
  1. Set PG14 to read-only (block all writes):
     SET default_transaction_read_only = on;
     -- or: revoke INSERT/UPDATE/DELETE from application role temporarily

  2. Wait for PG16 subscription to catch up completely:
     SELECT latest_end_lsn, received_lsn FROM pg_stat_subscription;
     -- Wait until they match

  3. Drop subscription on PG16 (make it writable):
     DROP SUBSCRIPTION upgrade_sub;

  4. Promote PG16 to primary:
     Update application connection string to PG16 endpoint (via PgBouncer config reload)

  5. Re-enable writes (if you disabled them in step 1)

Total write pause: 5–30 seconds
```

**Verification:**
```sql
-- Compare row counts on PG14 and PG16 before cutover
SELECT relname, n_live_tup FROM pg_stat_user_tables ORDER BY relname;
-- Run checksum validation on a sample of critical tables
```

---

**Q17. How would you set up a globally distributed PostgreSQL database for a SaaS product with users in NA, EU, and APAC?**

**Architecture:**

```
NA Region (primary for NA users):
  PostgreSQL primary (Patroni HA: 1 primary + 2 sync standbys)
  Handles writes from NA users
  Read replicas: 2 (serve NA read traffic)

EU Region (primary for EU users):
  PostgreSQL primary (Patroni HA: 1 primary + 2 sync standbys)
  Handles writes from EU users (GDPR: EU data stays in EU)
  Read replicas: 2

APAC Region (primary for APAC users):
  PostgreSQL primary (Patroni HA: 1 primary + 2 sync standbys)

Cross-region logical replication (for read-only global data):
  Product catalog (NA primary) → async logical replication → EU replica, APAC replica
  Shared configuration data → replicated to all regions
  User-specific data: owned by home region, not cross-replicated (GDPR)

Routing:
  Application servers in each region connect to local cluster
  Tenant/user→region mapping stored in a global routing table (Redis or a small global PG cluster)
  User always routed to their home region (where their primary is)
```

**Data ownership model:**
```sql
-- Global routing metadata (tiny, fits anywhere):
CREATE TABLE user_region_map (
    user_id BIGINT PRIMARY KEY,
    home_region TEXT NOT NULL  -- 'NA', 'EU', 'APAC'
);

-- In EU cluster, enforce data residency:
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
CREATE POLICY eu_only ON users USING (region = 'EU');
-- EU cluster physically only stores EU users; application layer also enforces routing
```

**Cross-region query problem:**
```
EU user queries data that involves NA user (e.g., shared organization):
  Option A: Org data is owned by one region (the org creator's region); others use async replica
  Option B: Global shared data tier (small set of tables in a separate globally-replicated cluster)
  Option C: Accept cross-region API call (200ms latency) for rare cross-region operations
```

---

**Q18. What causes replication slot bloat and how do you safely clean it up?**

**What is replication slot bloat:**
```
PostgreSQL keeps WAL segments on disk until all replication slots have consumed them.
If a slot's consumer falls behind (replica disconnects, CDC connector pauses):
  Slot's confirmed_flush_lsn stops advancing
  pg_wal/ directory grows unboundedly
  Disk fills up → database crash

Monitor:
SELECT slot_name,
       active,
       pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn) / 1024 / 1024 AS lag_mb
FROM pg_replication_slots;
-- Alert if lag_mb > 10000 (10GB)
```

**Common causes:**
- Debezium/CDC connector paused or crashed
- Replica disconnected for maintenance and not reconnected
- Logical replication subscriber dropped without dropping the publisher slot
- Long-running transaction on subscriber preventing slot advancement

**Safe cleanup:**

**Step 1: Is the consumer coming back soon?**
```sql
-- Check when slot was last active:
SELECT slot_name, active, active_pid, xmin, catalog_xmin
FROM pg_replication_slots WHERE NOT active;

-- If connector is being fixed and will reconnect in < 1 hour: wait
-- If disconnected for days and disk is at risk: drop the slot
```

**Step 2: Drop inactive slot (if consumer will not reconnect)**
```sql
SELECT pg_drop_replication_slot('debezium_orders_slot');
-- WARNING: this discards unprocessed changes permanently
-- The consumer (Debezium) will need to do a full snapshot re-sync from the beginning
```

**Step 3: Prevention**
```sql
-- Limit how much WAL a slot can accumulate before PostgreSQL drops it automatically:
-- PostgreSQL 13+:
max_slot_wal_keep_size = '20GB'   -- slots are invalidated if they accumulate > 20GB WAL
-- Slot invalidation causes consumer to re-snapshot, but prevents disk death

-- Monitor with alerting:
-- Alert when any slot lag > 5GB
-- Alert when any slot is inactive for > 10 minutes
```

---

**Q19. Explain how you would design a database replication architecture that tolerates an entire region going down.**

**Requirements:**
- Primary region failure: zero data loss, automatic failover in < 60 seconds
- Any single region: 2 others still serve traffic
- 3 regions: US-East, EU-West, APAC-SE

**Architecture: CockroachDB or PostgreSQL with Patroni + global consensus**

**Option A: CockroachDB (recommended for new systems)**
```
3-region Raft cluster:
  US-East:  3 nodes (local Raft quorum for US-owned ranges)
  EU-West:  3 nodes
  APAC-SE:  3 nodes
  Total: 9 nodes

Data placement:
  US user data: primary Raft replicas in US-East; minority replicas in EU + APAC
  EU user data: primary Raft replicas in EU-West; minority in US + APAC
  
Raft quorum: 2 of 3 regions must acknowledge write
  → US-East fails: EU + APAC form quorum → writes continue
  → RPO: 0 (Raft ensures no committed data is lost)
  → RTO: seconds (Raft leader election within surviving regions)
```

**Option B: PostgreSQL with Patroni + etcd (existing PostgreSQL workloads)**
```
US-East:  PostgreSQL primary + 1 sync standby + etcd node
EU-West:  PostgreSQL sync standby + etcd node
APAC-SE:  PostgreSQL async standby + etcd node

etcd cluster (3 nodes, one per region):
  etcd quorum = 2/3 nodes → survives one region failure
  Patroni holds primary lease in etcd
  If US-East fails: etcd elects leader in EU + APAC → promotes EU sync standby

Data flow:
  US-East primary → sync replication → EU-West standby (RPO=0 for EU)
  US-East primary → async replication → APAC standby (RPO > 0 for APAC)

Trade-off: write latency includes US ↔ EU sync RTT (~80ms per write)
```

**DNS / Connection routing:**
```
Application → Global load balancer (Route 53 / Cloud DNS health-check routing)
              → US-East endpoint (healthy)
              → EU-West endpoint (failover if US-East unhealthy)
              → APAC endpoint

Patroni updates a shared endpoint via etcd:
  Current primary IP registered in etcd → PgBouncer reads from etcd, routes to current primary
```

---

**Q20. A replica is 2 hours behind the primary under heavy write load. How do you diagnose and fix it?**

**Step 1: Measure and characterize the lag**
```sql
-- On primary:
SELECT client_addr,
       write_lag,
       flush_lag,
       replay_lag,
       pg_wal_lsn_diff(sent_lsn, replay_lsn) AS lag_bytes
FROM pg_stat_replication;

-- Interpret:
-- write_lag large: replica disk is slow writing WAL
-- flush_lag large: replica fsync is slow
-- replay_lag large: replica CPU is slow applying WAL to data pages
-- Bytes large but time lag small: lots of WAL (write-heavy primary), replica keeping up in throughput but behind by volume
```

**Step 2: Identify the bottleneck on the replica**
```bash
# On replica host:
iostat -xz 1      # I/O utilization — is replica disk at 100%?
vmstat 1          # CPU utilization — is replica CPU saturated?
pg_top or top     # Is the WAL applier process using 100% CPU?

# PostgreSQL replica:
SHOW max_wal_size;               # Is WAL growing faster than replica can apply?
SELECT * FROM pg_stat_activity WHERE wait_event IS NOT NULL;  # Any wait events?
```

**Step 3: Common causes and fixes**

| Cause | Symptom | Fix |
|---|---|---|
| Replica disk I/O bottleneck | write_lag high, iostat 100% | Upgrade to faster disk (NVMe); move WAL to separate disk |
| Replica CPU bottleneck | replay_lag high, CPU 100% | Upgrade CPU; use parallel WAL apply (PG 14+) |
| Long-running query blocking WAL apply | replay_lag growing, `pg_stat_activity` shows conflict | Set `hot_standby_feedback = off` or `max_standby_streaming_delay = 0` to cancel conflicting queries |
| Primary generating too much WAL (table bloat) | lag_bytes enormous | `VACUUM` primary to reduce WAL churn; fix index bloat |
| Replication slot not advancing | `pg_replication_slots` slot inactive | Restart consumer; if stuck, drop and recreate slot |

**Parallel WAL apply (PostgreSQL 14+):**
```
recovery_parallelism = 4    -- apply WAL using 4 parallel workers
# Before PG14: WAL apply was single-threaded → bottleneck on write-heavy workloads
# PG14: parallel apply dramatically reduces replay_lag for large write burdens
```

**Step 4: Catchup monitoring**
```sql
-- After fix, monitor the lag shrinking:
SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) / 1024 / 1024 AS lag_mb,
       replay_lag
FROM pg_stat_replication;
-- Should see lag_mb decreasing over time
```

---

## Quick Reference

```
Replication types:
  Synchronous:  RPO=0, higher write latency (RTT to replica)
  Asynchronous: RPO>0, no write latency overhead
  Logical:      row-level, selective tables, cross-version, enables CDC

Key metrics:
  RPO: maximum acceptable data loss
  RTO: maximum acceptable downtime
  Replication lag: pg_stat_replication.replay_lag

HA tools:
  PostgreSQL: Patroni + etcd (auto-failover), PgBouncer (transparent routing)
  MySQL: MHA, Orchestrator, ProxySQL
  AWS: RDS Multi-AZ (60-120s RTO), Aurora (< 30s RTO)

Split-brain prevention:
  Quorum (Patroni + etcd): only one node holds distributed lock
  STONITH: fence old primary before promoting new one
  Fencing tokens: storage layer rejects writes from old epoch

Replication slot risk:
  max_slot_wal_keep_size = '20GB'  -- cap WAL retention per slot
  Monitor: lag_bytes from pg_replication_slots
  Alert at > 5GB; drop slot if consumer won't return

Zero-downtime upgrade:
  Logical replication from old version → new version
  Minimal write pause (< 30s) for final cutover
```
