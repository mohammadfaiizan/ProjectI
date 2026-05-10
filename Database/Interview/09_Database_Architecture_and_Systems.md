# Database Architecture and Systems — Interview Questions

## Easy (Q1–Q8)

---

**Q1. What is the difference between SQL and NoSQL databases?**

| Aspect | SQL (Relational) | NoSQL |
|---|---|---|
| Schema | Fixed, predefined schema | Flexible / schema-less |
| Query language | SQL (standardized) | Database-specific APIs |
| ACID | Yes (by default) | Often BASE; some support ACID |
| Relationships | JOINs across tables | Typically denormalized, embedded |
| Scaling | Primarily vertical; horizontal with effort | Designed for horizontal scaling |
| Consistency | Strong (default) | Often eventual |
| Examples | PostgreSQL, MySQL, SQL Server | MongoDB, Cassandra, Redis, DynamoDB |

**NoSQL categories:**
- **Document** (MongoDB, CouchDB) — JSON-like documents, good for nested/flexible data
- **Key-value** (Redis, DynamoDB) — simple map, extremely fast
- **Wide-column** (Cassandra, HBase) — column families, write-optimized, massive scale
- **Graph** (Neo4j, Amazon Neptune) — edges and nodes, best for relationship traversals

---

**Q2. What is the CAP theorem?**

CAP theorem states that a distributed system can guarantee at most **two** of the following three properties simultaneously:

| Property | Meaning |
|---|---|
| **Consistency (C)** | Every read returns the most recent write (all nodes see the same data) |
| **Availability (A)** | Every request receives a response (no timeout/error) |
| **Partition tolerance (P)** | The system continues operating even if network partitions separate some nodes |

**In practice:** Network partitions always happen in distributed systems, so P is non-negotiable. The real trade-off is **CP vs AP**:

- **CP systems** (sacrifice availability) — MongoDB, HBase, Zookeeper: during a partition, reject writes to preserve consistency
- **AP systems** (sacrifice consistency) — Cassandra, DynamoDB, CouchDB: during a partition, allow reads/writes but they may return stale data
- **CA systems** — only possible in single-node systems (no real partition to tolerate): PostgreSQL, MySQL

---

**Q3. What is eventual consistency and how does it differ from strong consistency?**

**Strong consistency:** After a write completes, any subsequent read from any node returns the updated value.

**Eventual consistency:** After a write, reads may return stale data for a period, but eventually (once all nodes sync) all reads will reflect the latest write.

```
Strong consistency:
  Write "x=10" → ACK → Read from any node → always returns 10

Eventual consistency:
  Write "x=10" → ACK
  Immediately after: Node A returns 10, Node B returns 9 (not yet synced)
  After replication lag: all nodes return 10
```

**Trade-off:** Strong consistency requires coordination between nodes (slower, blocks on network). Eventual consistency allows faster writes with local acknowledgment.

**Real-world examples:**
- DNS — updates propagate slowly (eventual consistency)
- Shopping cart — showing a slightly stale total is acceptable
- Bank balance — must be strongly consistent

---

**Q4. What is database replication and what are its primary uses?**

Replication is the process of copying data from one database (primary/master) to one or more other databases (replicas/standbys).

**Primary uses:**
1. **High availability** — if the primary fails, a replica can be promoted
2. **Read scaling** — route read queries to replicas, offloading the primary
3. **Geographic distribution** — replicas closer to users reduce read latency
4. **Backup** — replicas serve as live backups (though not a replacement for true backups)
5. **Analytics** — run heavy analytical queries on a replica without affecting OLTP on the primary

**Types:**
- **Synchronous** — primary waits for replica to confirm write before returning success. No data loss but higher write latency.
- **Asynchronous** — primary returns success after local write. Replica may lag. Risk of data loss on primary failure.
- **Semi-synchronous** (MySQL) — waits for at least one replica to acknowledge, then returns.

---

**Q5. What is the difference between horizontal and vertical scaling?**

**Vertical scaling (scale up):** Add more CPU, RAM, or faster disks to the existing server.
- Simple — no application changes
- Limited by hardware ceiling
- Single point of failure remains
- Good for: most databases up to a certain size

**Horizontal scaling (scale out):** Add more servers and distribute the load.
- Requires data distribution strategy (replication, sharding)
- Theoretically unlimited scale
- More complex (network, consistency challenges)
- Good for: very large datasets, globally distributed systems

```
Vertical:   [1 large server 64 CPU, 512GB RAM]
Horizontal: [Server 1] [Server 2] [Server 3] ... [Server N]
            each handling a partition of data
```

Most relational databases scale vertically first, then use read replicas (horizontal for reads), and sharding as a last resort. NoSQL databases are architected for horizontal scaling from the start.

---

**Q6. What is connection pooling and why is it important?**

Establishing a new database connection is expensive: TCP handshake + authentication + session setup can take 20–100ms. Connection pooling maintains a pool of pre-established connections that are reused by application threads.

```
Without pooling:
  Request 1 → open connection → query → close connection  (100ms overhead)
  Request 2 → open connection → query → close connection  (100ms overhead)

With pooling:
  Startup: open 20 connections, keep alive
  Request 1 → borrow connection → query → return to pool  (~1ms overhead)
  Request 2 → borrow connection → query → return to pool  (~1ms overhead)
```

**Common poolers:**
- **PgBouncer** (PostgreSQL) — very lightweight, supports transaction mode (connection returned after each transaction) and session mode
- **HikariCP** (Java/JDBC) — fast in-process pool
- **ProxySQL** (MySQL) — pooling + query routing + analytics
- **RDS Proxy** (AWS) — managed pooler for RDS

**Pool sizing:** A common formula: `pool_size = (num_cores * 2) + num_effective_spindles`. More connections than this causes more context switching than benefit. For PostgreSQL, 100 connections is a common max; use PgBouncer in transaction mode to serve thousands of application threads.

---

**Q7. What is database sharding?**

Sharding is a form of horizontal partitioning where data is distributed across multiple independent database servers (shards), each holding a subset of the total data.

```
Without sharding:  [All 1 billion users on one server]
With sharding:
  Shard 0: users where user_id % 4 = 0  (250M users)
  Shard 1: users where user_id % 4 = 1  (250M users)
  Shard 2: users where user_id % 4 = 2  (250M users)
  Shard 3: users where user_id % 4 = 3  (250M users)
```

**Sharding strategies:**
- **Range-based** — shard by ranges of a key (user_id 1–1M on shard 1, etc.) — risk of hot shards
- **Hash-based** — `shard = hash(shard_key) % num_shards` — even distribution, but hard to range-scan
- **Directory-based** — a lookup table maps each key to a shard — flexible, but the lookup table is a bottleneck

**Challenges:**
- **Cross-shard JOINs** — require data from multiple shards, usually done in application code
- **Cross-shard transactions** — require distributed transactions (2PC) or must be avoided
- **Rebalancing** — adding/removing shards requires moving data
- **Shard key choice** — bad choice causes hot spots

---

**Q8. What is a database proxy and what problems does it solve?**

A database proxy sits between the application and the database, intercepting all database traffic.

```
Application → [Database Proxy] → Primary
                              → Replica 1
                              → Replica 2
```

**Problems solved:**

| Problem | Proxy Solution |
|---|---|
| Too many connections | Connection pooling (PgBouncer, ProxySQL) |
| Read/write splitting | Route SELECTs to replicas, writes to primary |
| Failover transparency | Proxy redirects on primary failure (no app changes) |
| Query analytics | Log/analyze all queries passing through |
| Security | Firewall, query filtering, credential management |
| Schema migrations | Blue/green routing during migrations |

**Examples:**
- **PgBouncer** — lightweight PostgreSQL connection pooler
- **ProxySQL** — MySQL-compatible, adds query routing, caching, analytics
- **MaxScale** (MariaDB) — advanced query routing, read/write splitting
- **AWS RDS Proxy** — managed proxy for RDS/Aurora
- **Vitess** — sharding middleware for MySQL (used by YouTube)

---

## Medium (Q9–Q15)

---

**Q9. Compare primary-replica replication vs multi-primary (multi-master) replication.**

**Primary-Replica (Active-Passive):**
```
Primary  ──WAL/binlog──▶  Replica 1
                     ──▶  Replica 2
Writes: only to primary
Reads:  from any replica (may be slightly stale)
```
- Simple to implement
- No write conflicts
- Failover requires promoting a replica to primary (brief downtime or automated with tools like Patroni/MHA)

**Multi-Primary (Active-Active):**
```
Primary 1 ◄──────────────▶ Primary 2
Any writes to either primary
```
- Both nodes accept writes
- **Conflict resolution** required: what if both primaries update the same row simultaneously?
  - Last-write-wins (timestamp-based) — potential data loss
  - Application-level resolution
  - Avoid by routing the same row to the same primary (sticky routing)

| Aspect | Primary-Replica | Multi-Primary |
|---|---|---|
| Write throughput | Limited to one node | Can distribute writes |
| Complexity | Low | High |
| Conflicts | None | Possible |
| Use case | Most OLTP workloads | Geo-distributed writes, HA requiring zero downtime |
| Examples | PostgreSQL streaming replication | MySQL Group Replication, Galera Cluster, CockroachDB |

---

**Q10. How does Cassandra achieve high availability and what consistency model does it use?**

Cassandra is a wide-column, AP (Availability + Partition tolerance) database using a **peer-to-peer (ring) architecture** — there is no single primary.

**Key concepts:**

**Consistent hashing ring:** Data is distributed across nodes by hashing the partition key. Each node is responsible for a range of the ring. Virtual nodes (vnodes) allow fine-grained distribution.

**Replication factor (RF):** Each row is stored on RF nodes. RF=3 means 3 copies.

**Consistency levels (tunable):**
```
Write/Read consistency:
  ONE      — 1 node must acknowledge
  QUORUM   — majority (RF/2 + 1) must acknowledge
  ALL      — all RF nodes must acknowledge
  LOCAL_QUORUM — quorum within the local datacenter only
```

**Tunable consistency rule:**
For strong consistency: `write CL + read CL > RF`
- RF=3: QUORUM writes (2 nodes) + QUORUM reads (2 nodes) = 4 > 3 → guaranteed to read the latest write
- RF=3: ONE write + ONE read = 2 ≤ 3 → possible to read stale data

```
Write to RF=3 with QUORUM:
  Client → Coordinator → Node A (ACK), Node B (ACK), Node C (async)
  Client receives success after 2 ACKs
```

**Conflict resolution:** Last-write-wins using client-provided timestamps (microseconds). Cassandra does not use locks; concurrent writes to the same cell are resolved by highest timestamp.

---

**Q11. Explain database high availability architectures: active-passive vs active-active.**

**Active-Passive (Warm Standby):**
```
         [Load Balancer / VIP]
               |
          [Primary] ──────── [Standby]
          (serves all)       (replicating, idle)
```
- Standby becomes primary on failover (seconds to minutes)
- Tools: PostgreSQL + Patroni, MySQL + MHA, AWS RDS Multi-AZ
- **RTO** (Recovery Time Objective): seconds with automated failover
- **RPO** (Recovery Point Objective): seconds to zero with synchronous replication

**Active-Active:**
```
         [Load Balancer]
          /            \
    [Node A]          [Node B]
    (read+write)      (read+write)
         \              /
          [shared data / synced]
```
- Both nodes serve traffic simultaneously
- RTO ≈ 0 (no failover needed)
- Requires conflict resolution
- Harder to implement for transactional workloads
- Examples: Galera Cluster (MySQL), CockroachDB, Spanner

**Choosing:**
| Factor | Active-Passive | Active-Active |
|---|---|---|
| Complexity | Low | High |
| Write scale | Single node limit | Can distribute |
| Consistency | Easy to maintain | Requires coordination |
| Downtime on failure | Brief failover window | Near zero |

---

**Q12. What is the difference between synchronous and asynchronous replication, and how do they affect RPO?**

**RPO (Recovery Point Objective):** Maximum acceptable data loss measured in time. If RPO=0, no data loss is allowed.

**Synchronous replication:**
```
Primary: COMMIT
  → write to local WAL
  → send WAL to replica
  → wait for replica ACK
  → return success to client

RPO = 0 (no committed data can be lost — replica has everything)
Cost: Write latency += network round trip to replica
```

**Asynchronous replication:**
```
Primary: COMMIT
  → write to local WAL
  → return success to client (replica not involved)
  → (in background) ship WAL to replica

RPO > 0 (replica may lag; on primary failure, latest N seconds/transactions are lost)
Cost: No added latency for writes
```

**Semi-synchronous (MySQL):**
- Primary waits for at least one replica to acknowledge receipt (not necessarily applied)
- RPO ≈ 0 for at least one replica, but not all

**PostgreSQL synchronous_standby_names:**
```
-- Wait for at least one named standby to confirm
synchronous_standby_names = 'FIRST 1 (standby1, standby2)'
-- or require ALL standbys:
synchronous_standby_names = 'ANY 2 (standby1, standby2, standby3)'
```

---

**Q13. How does distributed SQL (NewSQL) differ from traditional SQL and NoSQL?**

NewSQL databases (CockroachDB, Google Spanner, TiDB, YugabyteDB) aim to combine the horizontal scalability of NoSQL with the ACID guarantees and SQL interface of relational databases.

| Aspect | Traditional SQL | NoSQL | NewSQL / Distributed SQL |
|---|---|---|---|
| SQL support | Full | None/limited | Full |
| ACID | Yes | Often not | Yes (distributed ACID) |
| Horizontal scale | Limited | Yes | Yes |
| Consistency | Strong | Often eventual | Strong (globally) |
| Latency | Low (local) | Low | Higher (coordination cost) |
| Sharding | Manual | Auto or manual | Automatic |
| Examples | PostgreSQL, MySQL | Cassandra, MongoDB | CockroachDB, Spanner |

**How they achieve distributed ACID:**
- **Consensus protocol** (Raft or Paxos): replication and linearizability without a single primary
- **MVCC + distributed timestamps**: Spanner uses TrueTime (GPS + atomic clocks); CockroachDB uses HLC (Hybrid Logical Clocks)
- **2PC over Raft groups**: each shard is a Raft group; cross-shard transactions use 2PC between Raft leaders

**Trade-off:** Distributed ACID is much more expensive than local ACID. A simple UPDATE that touches one row may require 2–3 Raft round trips. Best for globally distributed applications that truly need both strong consistency and unlimited write scale.

---

**Q14. What is the difference between a data warehouse, a data lake, and a lakehouse?**

**Data Warehouse:**
- Structured, processed data in defined schemas (star/snowflake)
- Optimized for analytical queries (column store, MPP — massively parallel processing)
- Data cleansed and transformed before loading (ETL)
- Examples: Redshift, BigQuery, Snowflake

**Data Lake:**
- Raw data in native format (structured, semi-structured, unstructured)
- Stored in cheap object storage (S3, GCS, ADLS)
- Schema-on-read: define schema when querying, not when storing
- No ACID guarantees; no versioning; risk of becoming a "data swamp"
- Examples: S3 + Athena, HDFS + Hive

**Data Lakehouse:**
- Combines data lake storage (cheap object storage) with data warehouse features (ACID, schema enforcement, time travel)
- Implemented via open table formats: **Delta Lake**, **Apache Iceberg**, **Apache Hudi**
- Supports both batch and streaming workloads
- Examples: Databricks Lakehouse, AWS Lake Formation with Iceberg, Snowflake Iceberg tables

| Feature | Warehouse | Data Lake | Lakehouse |
|---|---|---|---|
| Storage cost | High (proprietary) | Low (object storage) | Low (object storage) |
| ACID | Yes | No | Yes |
| Schema | Schema-on-write | Schema-on-read | Both |
| ML/unstructured | No | Yes | Yes |
| Query performance | Excellent | Variable | Good to excellent |

---

**Q15. What is a time-series database and when should you use one instead of a relational database?**

A time-series database (TSDB) is optimized for data that is indexed by time — metrics, events, IoT readings, logs.

**Why relational databases struggle at time-series scale:**
- Millions of inserts per second overwhelm B-tree indexes (random writes)
- Queries like "average CPU over last hour at 1-minute intervals" require full scans
- Old data is rarely updated but often deleted (retention policies)
- No built-in concepts like downsampling, gap filling, continuous aggregation

**TSDB features:**
- **Columnar storage + compression**: timestamps and values stored separately, delta/RLE encoding
- **Automatic partitioning by time** (chunks): recent data in hot storage, old data in cold
- **Native time functions**: time_bucket, date_trunc, interpolation, downsampling
- **Retention policies**: automatically drop data older than N days
- **Continuous aggregates**: precomputed summaries that auto-update

**Examples and comparison:**
| Database | Type | Notes |
|---|---|---|
| InfluxDB | Purpose-built TSDB | Flux query language, strong time functions |
| TimescaleDB | PostgreSQL extension | Full SQL + time features, ACID |
| Prometheus | Pull-based metrics | PromQL, ephemeral (not for long-term) |
| ClickHouse | OLAP column store | Excellent for event data, ad-hoc analytics |
| QuestDB | High-performance TSDB | Very fast ingestion |

**When to use a TSDB vs PostgreSQL:**
- Use PostgreSQL (with TimescaleDB) if you need JOINs with relational tables and time-series in one system
- Use a dedicated TSDB (InfluxDB, ClickHouse) for very high ingestion rates (>1M/sec) or pure monitoring use cases

---

## Hard (Q16–Q20)

---

**Q16. Design a multi-region database architecture for a global social media application with billions of users. Justify your choices.**

**Requirements analysis:**
- 5 billion users across NA, EU, APAC
- High read volume (browse feed), moderate writes (posts, likes)
- Latency requirement: < 100ms for reads globally
- Strong consistency for financial-like operations (not applicable here), eventual OK for feeds
- GDPR: EU user data must stay in EU

**Architecture:**

```
Tier 1: User Data (owns the user record)
  Region-sharded by user_id: NA shard (PostgreSQL cluster), EU shard, APAC shard
  Each shard: 1 primary + 2 synchronous replicas (HA within region)
  GDPR: EU users on EU shard — data never leaves EU region

Tier 2: Social Graph (follower/following relationships)
  Graph database (Neo4j Cluster or Cassandra with denormalized edges)
  Replicated to all regions (reads are local, writes async to other regions)
  Eventual consistency acceptable (follower count lag of seconds is fine)

Tier 3: Feed / Timeline
  Apache Cassandra (multi-region, tunable consistency)
  Pre-computed fan-out: when user posts, write to all followers' feed tables
  LOCAL_QUORUM reads/writes within region
  Async cross-region replication

Tier 4: Caching
  Redis cluster per region (read-through cache for profile data, session tokens)
  TTL-based invalidation

Tier 5: Media (photos, videos)
  Object storage (S3 / GCS) with CDN edge caching
  Not in the database tier at all

Routing layer:
  Global load balancer routes user to nearest region
  Auth token carries region hint (EU users always routed to EU)
```

**Key decisions justified:**
- **PostgreSQL for user data** — ACID for profile/account mutations, familiar SQL, strong consistency
- **Cassandra for feed** — write-optimized, scales horizontally, multi-region native, eventual consistency acceptable
- **Redis for cache** — sub-millisecond reads, reduces DB load by 90%+
- **Region-based sharding** — satisfies GDPR, reduces latency, avoids cross-region write amplification

---

**Q17. Explain the Raft consensus algorithm and how it provides replication with strong consistency.**

Raft is a consensus protocol that ensures a cluster of nodes agrees on a sequence of log entries (commands), even if some nodes fail or messages are delayed.

**Roles:**
- **Leader** — handles all client requests, replicates log entries to followers
- **Follower** — passive, replicates from leader
- **Candidate** — runs for leader election when it suspects leader failure

**Leader election:**
```
1. Followers track "election timeout" (150–300ms). If no heartbeat received → becomes Candidate
2. Candidate increments "term", votes for itself, sends RequestVote RPCs
3. Node grants vote if: it hasn't voted this term AND candidate's log is as up-to-date as theirs
4. Candidate receiving majority (n/2 + 1) → becomes Leader, sends heartbeats
```

**Log replication:**
```
1. Client sends command to Leader
2. Leader appends to its log (term + index)
3. Leader sends AppendEntries RPC to all Followers in parallel
4. Once majority ACK: Leader marks entry "committed", applies to state machine, returns to client
5. Leader notifies Followers of commit; they apply to their state machines
```

**Safety guarantees:**
- Only nodes with the most up-to-date log can become leader → committed entries are never lost
- At most one leader per term (split brain prevented by requiring majority votes)
- Log matching: if two logs have same index + term, all preceding entries are identical

**In database context (CockroachDB, TiDB, etcd):**
- Each shard/range is a Raft group
- WAL records ARE the Raft log entries
- A write is "committed" once a majority of the Raft group replicates it
- This is how you get strong consistency across nodes without a single point of failure

---

**Q18. How does Google Spanner achieve globally distributed ACID transactions?**

Spanner combines three innovations:

**1. TrueTime API:**
Spanner uses atomic clocks and GPS receivers in every datacenter to bound clock uncertainty. `TT.now()` returns an interval `[earliest, latest]` within which the true time lies. Google guarantees the interval is < 7ms.

```
If write commits at TrueTime T, any subsequent read at T' > T + ε will see it
Spanner uses TT.now().latest as the commit timestamp to ensure causal ordering
```

**2. Semi-relational data model with Interleaving:**
```sql
-- Parent table
CREATE TABLE Users (UserId INT64 NOT NULL, ...) PRIMARY KEY (UserId);

-- Interleaved child table: Albums rows are physically stored inside their parent User row
CREATE TABLE Albums (
    UserId INT64 NOT NULL,
    AlbumId INT64 NOT NULL,
    ...
) PRIMARY KEY (UserId, AlbumId), INTERLEAVE IN PARENT Users ON DELETE CASCADE;
```
Co-locating related rows means a transaction touching a user and their albums can be committed without coordination — it's all in one Paxos group.

**3. 2PC across Paxos groups:**
For cross-shard transactions:
- Each shard is a Paxos group with a leader
- Cross-shard transactions use 2PC: participant leaders prepare (Paxos-replicated), then commit
- Commit timestamp chosen ≥ `TT.now().latest` → external consistency: if transaction T1 commits before T2 starts, T2's timestamp > T1's timestamp → total order

**Result:** External consistency (linearizability at global scale) — the strongest consistency model.

**Cost:** Write latency is bounded by: `2 × (cross-datacenter RTT) + TrueTime uncertainty ≈ 10–20ms` even for cross-continent transactions.

---

**Q19. What is change data capture (CDC) and how is it implemented?**

CDC is the process of tracking and capturing every data change (INSERT, UPDATE, DELETE) in a source database and propagating those changes to downstream systems in real-time.

**Use cases:**
- Sync data to a search engine (Elasticsearch)
- Replicate to a data warehouse (Snowflake, BigQuery)
- Event-driven microservices (trigger actions on data changes)
- Cache invalidation
- Audit logging

**Implementation approaches:**

**1. Log-based CDC (preferred):**
```
Source DB WAL/binlog ──▶ CDC Connector ──▶ Kafka/message broker ──▶ Consumers
                    (reads replication slot)
```
- Reads the database's replication log (PostgreSQL WAL via logical decoding, MySQL binlog)
- Non-intrusive: no extra queries, no triggers, no impact on source
- Tools: **Debezium** (open source, supports PG/MySQL/SQL Server/MongoDB), **AWS DMS**, **Airbyte**

```sql
-- PostgreSQL: create replication slot for CDC
SELECT pg_create_logical_replication_slot('debezium_slot', 'pgoutput');
-- Debezium connects here and receives all changes as a stream
```

**2. Trigger-based CDC:**
```sql
-- Every change writes to an audit table
CREATE TRIGGER cdc_trigger AFTER INSERT OR UPDATE OR DELETE ON orders
FOR EACH ROW EXECUTE FUNCTION capture_changes();
```
- Works on any database but adds write overhead

**3. Timestamp-based CDC (polling):**
```sql
SELECT * FROM orders WHERE updated_at > :last_checked_time;
```
- Simple but misses DELETEs (deleted rows have no `updated_at`)
- Polling interval creates latency

**Debezium event example (Kafka message):**
```json
{
  "op": "u",          // u=update, c=create, d=delete
  "before": {"id": 1, "status": "pending"},
  "after":  {"id": 1, "status": "shipped"},
  "source": {"table": "orders", "lsn": "0/1A23456", "ts_ms": 1710000000000}
}
```

---

**Q20. Design the database layer for a real-time ride-sharing application at 100M active users. Cover schema, indexing, caching, and scaling strategy.**

**Core entities and schema decisions:**

```sql
-- Sharded by city_id (geographic locality)
CREATE TABLE drivers (
    driver_id   BIGINT       PRIMARY KEY,
    city_id     INT          NOT NULL,
    name        VARCHAR(100),
    status      TEXT         CHECK (status IN ('offline','available','on_trip')),
    location    GEOMETRY(Point, 4326),  -- PostGIS for spatial queries
    last_updated TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_drivers_city_status ON drivers (city_id, status);
CREATE INDEX idx_drivers_location ON drivers USING GIST (location);  -- spatial index

CREATE TABLE trips (
    trip_id     BIGINT       PRIMARY KEY DEFAULT nextval('trip_seq'),
    rider_id    BIGINT       NOT NULL,
    driver_id   BIGINT,
    city_id     INT          NOT NULL,
    status      TEXT         CHECK (status IN ('requested','accepted','in_progress','completed','cancelled')),
    pickup      GEOMETRY(Point, 4326),
    dropoff     GEOMETRY(Point, 4326),
    fare_cents  INT,
    requested_at TIMESTAMPTZ DEFAULT NOW(),
    started_at  TIMESTAMPTZ,
    ended_at    TIMESTAMPTZ
);
-- Partition trips by month (billions of rows expected)
-- PostgreSQL declarative partitioning:
CREATE TABLE trips (...) PARTITION BY RANGE (requested_at);
CREATE TABLE trips_2024_01 PARTITION OF trips FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

**Scaling strategy:**

**1. Geographic sharding** — shard by `city_id`. A trip is entirely within one city; drivers in a city are a small fraction of total. Cross-city queries are analytical, not operational.

**2. Hot data in Redis:**
```
driver:{driver_id}:location → geospatial (Redis GEOADD)
driver:{driver_id}:status   → string (TTL 60s — auto-expires if driver goes offline)
active_trips:{city_id}      → sorted set by time

-- Find nearby available drivers (GEORADIUS):
GEORADIUS drivers:city:1 -73.98 40.74 5 km ASC COUNT 10
```

**3. Matching service** — runs as a stateless microservice reading from Redis geo-index. When a ride is requested:
- Query Redis for drivers within 5km
- Sort by proximity and rating
- Offer to nearest driver → on accept, write to PostgreSQL

**4. Trip lifecycle writes** — go directly to PostgreSQL (ACID needed for fare calculation, payment). Use `SELECT FOR UPDATE` when transitioning trip status to prevent double-booking.

**5. Analytics** — CDC (Debezium → Kafka → BigQuery) streams all trip events to the data warehouse. Never run analytics queries on the operational database.

**6. Replication** — PostgreSQL: 1 synchronous standby per city cluster (RPO=0), 1 async read replica for admin/reporting.

**Capacity estimate:**
- 100M active users, 10M rides/day = ~115 rides/second peak
- Each ride: ~10 status updates → ~1,150 writes/sec to trips table
- PostgreSQL on good hardware: 10,000+ writes/sec → single city cluster is fine
- Redis driver locations: 100M GEOADD/sec is Redis's strong suit

---

## Quick Reference

```
CAP: Choose 2 of {Consistency, Availability, Partition Tolerance}
     CP: MongoDB, HBase | AP: Cassandra, DynamoDB | CA: single-node SQL

Replication modes:
  Synchronous: RPO=0, higher write latency
  Asynchronous: RPO>0, lower write latency

Sharding strategies:
  Range: easy range queries, hot spots possible
  Hash: even distribution, no range scans
  Directory: flexible, lookup table bottleneck

Connection pooling:
  PgBouncer (PostgreSQL), ProxySQL (MySQL)
  Pool size ≈ num_cores * 2 + num_spindles

CDC tools:
  Debezium (log-based), AWS DMS, Airbyte
  PostgreSQL: logical replication slot
  MySQL: binlog

NewSQL: CockroachDB, Spanner, TiDB
  = SQL + ACID + horizontal scale + consensus replication

Time-series DBs: InfluxDB, TimescaleDB, ClickHouse
  Use when: high ingestion rate, time-based retention, downsampling needed
```
