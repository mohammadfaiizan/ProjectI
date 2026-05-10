# Connection Pooling and Database Proxies

## Easy (Q1–Q7)

---

**Q1. Why is database connection pooling necessary and what problem does it solve?**

Opening a new database connection is expensive:

```
TCP handshake:       1–3ms
TLS negotiation:     2–10ms
Authentication:      1–5ms
Session setup:       0.5–2ms
Total overhead:      5–20ms per new connection
```

At scale, if every web request opens a new connection:
```
10,000 req/sec × 10ms connection overhead = 100 seconds of overhead per second
→ Connection setup costs more than the actual query execution
```

**Additional problems without pooling:**
- Each PostgreSQL connection is a separate OS process (~5–10MB RAM)
- `max_connections = 100` (default) → only 100 simultaneous connections allowed
- With 1,000 application threads all wanting connections: 900 requests wait or fail

**Connection pooling solution:**
```
Application threads (1000s) → Pool (holds 20–100 pre-established connections) → Database

Thread borrows a connection from the pool for its query duration (typically < 10ms)
Returns the connection to the pool when done (connection is NOT closed — stays open)
Next request reuses the same connection (no connection setup overhead)
```

**Result:** 100 application threads served by 20 actual database connections simultaneously.

---

**Q2. What are the three PgBouncer connection pooling modes and when do you use each?**

**Session mode:**
```
Connection assigned to client for the entire session duration
Released when client disconnects

Client A connects → gets DB connection #1 → uses it for entire session → disconnects → connection returned

Use when: application uses session-level features
          SET LOCAL, session variables, advisory locks, LISTEN/NOTIFY, prepared statements
Concurrency: low (1 client = 1 DB connection at all times)
```

**Transaction mode (most common):**
```
Connection assigned for the duration of one transaction
Released immediately after COMMIT or ROLLBACK

Client A: BEGIN → [DB connection #1 assigned] → query → COMMIT → [connection returned]
Client B: may use DB connection #1 on their next transaction

Use when: application uses standard BEGIN/COMMIT transactions
          Cannot use: session-level features, prepared statements across transactions
Concurrency: high (100 clients served by 10 DB connections if transactions are short)
```

**Statement mode:**
```
Connection assigned for a single statement
Released after each statement (even outside a transaction)

Use when: simple, single-statement queries only, no multi-statement transactions
Rarely used in practice (breaks most application frameworks)
```

**PgBouncer configuration:**
```ini
[pgbouncer]
pool_mode = transaction          ; most common
max_client_conn = 10000          ; how many clients can connect to PgBouncer
default_pool_size = 20           ; actual PostgreSQL connections per DB+user pair
reserve_pool_size = 5            ; extra connections when pool is exhausted
server_idle_timeout = 600        ; close idle DB connections after 10 min
```

---

**Q3. What is a database proxy and how is it different from a connection pooler?**

**Connection pooler** (e.g., PgBouncer): Multiplexes many client connections onto fewer database connections. Its primary function is connection management.

**Database proxy** (e.g., ProxySQL, AWS RDS Proxy, PgBouncer) does everything a pooler does, PLUS:

| Capability | Pooler Only | Database Proxy |
|---|---|---|
| Connection multiplexing | ✓ | ✓ |
| Read/write splitting | ✗ | ✓ |
| Query routing | ✗ | ✓ |
| Query rewriting | ✗ | ✓ |
| Connection caching on failover | ✗ | ✓ (RDS Proxy) |
| Query analytics/logging | Limited | ✓ |
| Authentication/credential management | Basic | ✓ |

**ProxySQL (MySQL proxy):**
```sql
-- Route writes to primary:
INSERT INTO mysql_query_rules (rule_id, match_pattern, destination_hostgroup)
VALUES (1, '^SELECT', 10),    -- reads → replica hostgroup
       (2, '.*',      20);    -- everything else → primary hostgroup

-- Application connects to ProxySQL; ProxySQL handles routing transparently
```

**AWS RDS Proxy:**
```
Application → RDS Proxy → RDS Primary (writes)
                        → RDS Read Replica (reads, if configured)
Benefits:
  Maintains warm pool of connections to RDS even when Lambda functions scale to zero and back
  Handles failover transparently (retries in-flight connections to new primary)
  IAM authentication (no DB credentials in application code)
```

---

**Q4. What is read/write splitting and how is it implemented at the proxy layer?**

Read/write splitting routes SELECT queries to read replicas and writes (INSERT/UPDATE/DELETE/DDL) to the primary.

```
Application writes:
  INSERT INTO orders ...       → Proxy → Primary DB
  UPDATE users SET ...         → Proxy → Primary DB
  BEGIN; ... COMMIT;           → Proxy → Primary DB

Application reads:
  SELECT * FROM products ...   → Proxy → Replica 1 (round-robin)
  SELECT COUNT(*) FROM orders  → Proxy → Replica 2
```

**ProxySQL example:**
```sql
-- Define hostgroups:
-- Hostgroup 10 = read replicas
-- Hostgroup 20 = primary (writer)

INSERT INTO mysql_servers (hostgroup_id, hostname, port) VALUES
    (20, 'primary.db.internal',  3306),
    (10, 'replica1.db.internal', 3306),
    (10, 'replica2.db.internal', 3306);

-- Routing rules:
-- Rule 1: read-only SELECTs go to replicas
INSERT INTO mysql_query_rules (rule_id, active, match_pattern, destination_hostgroup, apply)
VALUES (1, 1, '^SELECT\s', 10, 1);

-- Rule 2: SELECTs inside transactions go to primary (consistency)
INSERT INTO mysql_query_rules (rule_id, active, match_pattern, destination_hostgroup, apply, transaction_persistent)
VALUES (2, 1, '^SELECT\s', 20, 1, 1);
-- transaction_persistent=1: once a connection is in a transaction, keep it on primary

-- Default rule: everything else to primary
INSERT INTO mysql_query_rules (rule_id, active, destination_hostgroup, apply)
VALUES (3, 1, 20, 1);
```

**The read-after-write problem:**
```
User updates profile → routes to primary → committed
Immediately reads profile → routes to replica → replica hasn't replicated yet → stale data shown

Solutions at proxy level:
  - Route all queries in same session to primary for N seconds after a write
  - Use sticky sessions (same session always uses same replica)
  - Use synchronous replica (lag = 0) for the sticky session
```

---

**Q5. What is the `max_connections` parameter in PostgreSQL and how do connection pools interact with it?**

`max_connections` is the hard limit on simultaneous connections PostgreSQL accepts. Default: 100.

**Memory implications:**
```
Each connection = one OS process in PostgreSQL (not a thread)
Idle connection: ~5MB RAM (shared memory structures)
Active connection (executing query): ~10–50MB RAM (sort buffers, temp files)

max_connections = 1000 → 5GB RAM just for idle connections
With work_mem = 64MB and parallel queries: each connection could use 300MB+ during a sort
max_connections = 100, work_mem = 64MB → reserve ~6.4GB for active queries
```

**Optimal max_connections formula:**
```
max_connections ≈ (available_RAM_for_connections) / (work_mem * max_parallel_workers + idle_overhead)
For production: typically 100–500
For replicas (read-only): can be higher (read queries use less memory than write transactions)
```

**Connection pool interaction:**
```
PostgreSQL max_connections = 200

PgBouncer pool configuration:
  max_client_conn = 10000    -- clients connect to PgBouncer (no PostgreSQL limit applies)
  default_pool_size = 20     -- PgBouncer maintains ≤ 20 actual PostgreSQL connections
  
  With 5 application services:
    Service A: 20 PG connections
    Service B: 20 PG connections
    PgBouncer admin: 1 PG connection
    Total: 41 PG connections (well within max_connections = 200)
  
  Each service serves 2000 client connections via PgBouncer
  Total client-visible connections: 10000
  Total actual PostgreSQL connections: 41
```

---

**Q6. How does connection pooling interact with PostgreSQL's LISTEN/NOTIFY and prepared statements?**

**LISTEN/NOTIFY:**
```
LISTEN requires a persistent session (the notification is delivered to the specific session)
PgBouncer transaction mode: connections are returned to pool between transactions
→ LISTEN/NOTIFY DOES NOT WORK in transaction mode (connection may change between LISTEN and NOTIFY)

Solution:
  Use a dedicated connection (not from PgBouncer) for LISTEN/NOTIFY
  OR use PgBouncer in session mode for the LISTEN/NOTIFY channel
  OR use a separate pub/sub system (Redis Pub/Sub, Kafka) instead of LISTEN/NOTIFY at scale
```

**Prepared statements:**
```
Prepared statement: a named query template, stored per-connection
  PREPARE get_user AS SELECT * FROM users WHERE id = $1;
  EXECUTE get_user(1234);

PgBouncer transaction mode:
  Connection returned to pool after each transaction
  Next transaction may get a DIFFERENT connection
  → Named prepared statement from previous transaction doesn't exist on new connection
  → ERROR: prepared statement "get_user" does not exist

Solutions:
  1. Use unnamed prepared statements (protocol-level, PgBouncer intercepts and caches)
     JDBC, psycopg2, Go pgx: use protocol-level prepare by default → works with PgBouncer
  
  2. PgBouncer 1.21+: prepared statement support in transaction mode (tracks statements per client)
  
  3. Use PgBouncer session mode (loses connection multiplexing benefit)
  
  4. Use RDS Proxy (handles prepared statements transparently)
```

---

**Q7. What happens to in-flight queries when a database primary fails and how does a proxy handle it?**

**Without a proxy (direct connection):**
```
Application has 100 connections to primary
Primary crashes
All 100 connections receive TCP RST or timeout
Application gets connection errors
Must reconnect to new primary (after failover completes: 30–120 seconds)
All in-flight queries are lost
Application must implement reconnect logic with backoff
```

**With RDS Proxy:**
```
Application connects to RDS Proxy endpoint (stable DNS, never changes)
RDS Proxy maintains warm pool of connections to actual primary

Primary fails:
  Application: still connected to RDS Proxy (no TCP disconnect)
  RDS Proxy: detects primary failure, waits for new primary to be promoted (30–60s)
  RDS Proxy: reconnects its pool to the new primary
  Application: in-flight transactions are retried automatically (for read-only queries)
  Write transactions: may receive an error (cannot be safely retried without idempotency)

User experience: brief pause (< 30s) vs hard failure and reconnect storm
```

**With PgBouncer + Patroni:**
```
PgBouncer has a configuration file with primary endpoint
Patroni updates a "leader" key in etcd when primary changes

Patroni callback on promotion:
  1. New primary promoted (Replica 1)
  2. Patroni fires callback: update PgBouncer config (primary = new_primary_IP)
  3. PgBouncer reload: new connections go to new primary
  4. In-flight connections: client sees error, must retry

To minimize errors:
  PgBouncer "pause" mode: pause all new queries, drain in-flight, reconnect, unpause
  PauseSeconds: 5–10 second pause window during failover
```

---

## Medium (Q8–Q15)

---

**Q8. How do you size a connection pool for a production application? Walk through the calculation.**

**Inputs needed:**
1. Total RAM on database server
2. Average query execution time
3. Requests per second
4. Database server CPU count

**Formula:**
```
Optimal DB connections ≈ num_CPU_cores * 2 + effective_spindles
(From HikariCP's research / PostgreSQL guidelines)

For a 32-core server with NVMe SSD (1 spindle):
  Optimal connections ≈ 32 * 2 + 1 = 65
  
Why? Beyond this point, connections spend more time context-switching than doing useful work
More connections = more lock contention + more context switches + more shared memory overhead
```

**Verify with Little's Law:**
```
Little's Law: N = λ × W
  N = number of concurrent connections needed
  λ = request arrival rate (queries/second)
  W = average query execution time (seconds)

Example:
  λ = 5000 queries/second
  W = 10ms = 0.01 seconds per query
  N = 5000 × 0.01 = 50 concurrent connections

So 50 actual DB connections handle 5000 queries/second with 10ms average latency
PgBouncer pool_size = 50 → sufficient
```

**PgBouncer pool sizing example:**
```
Setup:
  PostgreSQL: 16 CPU cores, 64 GB RAM
  Applications: 5 services, each with 100 app threads
  max_connections setting: 200

Calculation:
  Optimal DB connections: 16 * 2 + 1 = 33 total
  Per service: 33 / 5 ≈ 7 connections each (set pool_size = 10 with some headroom)
  
  PgBouncer per service:
    max_client_conn = 200 (each service has 100 threads + overhead)
    default_pool_size = 10
    
  Total PostgreSQL connections: 5 services × 10 = 50 connections (well under max_connections=200)
  
  Headroom for admin tools, monitoring, migrations: 150 remaining connections available
```

---

**Q9. What is a connection storm and how do you prevent it?**

A connection storm (or connection stampede) occurs when many application instances simultaneously try to establish database connections — typically during a deployment, restart, or after a database failover.

**Scenario:**
```
Production: 50 application servers, each with pool of 20 connections = 1000 total connections
Database failover: all 1000 connections are dropped

Recovery:
  All 50 servers simultaneously try to reconnect
  1000 TCP connections attempted simultaneously
  Database connection setup: expensive (auth, session init)
  PostgreSQL: overwhelmed by 1000 simultaneous connection attempts
  → Slowdown → some connections time out → retry storms → cascade failure
```

**Prevention strategies:**

**1. Gradual reconnection with jitter:**
```python
import random, time

def connect_with_backoff():
    for attempt in range(10):
        try:
            return create_connection()
        except ConnectionError:
            delay = min(0.1 * (2 ** attempt) + random.uniform(0, 0.1), 30)
            time.sleep(delay)  # exponential backoff + jitter
    raise Exception("Could not connect after 10 attempts")
```

**2. PgBouncer as buffer (absorbs the storm):**
```
Application servers: reconnect to PgBouncer (PgBouncer stays up during DB failover)
PgBouncer: gradually re-establishes its pool to the new primary
→ 1000 app-to-PgBouncer reconnections are cheap (no DB auth overhead)
→ PgBouncer re-establishes 50 actual DB connections at its own pace
```

**3. Server-side connection limiting:**
```sql
-- Limit connections from application user to prevent overwhelming on reconnect:
ALTER ROLE app_user CONNECTION LIMIT 100;

-- With PgBouncer: this applies to PgBouncer-to-DB connections (not app-to-PgBouncer)
-- So set CONNECTION LIMIT = pool_size + a small buffer
ALTER ROLE app_user CONNECTION LIMIT 60;  -- pool_size=50 + 10 buffer
```

**4. Pre-warm connections before cutover:**
```
During deployment: new servers establish connections BEFORE taking traffic
Ensures pool is warm when first request arrives
Use health checks to verify connection pool is ready before routing traffic
```

---

**Q10. How does AWS RDS Proxy help with Lambda (serverless) database connections?**

**The serverless connection problem:**
```
Lambda functions: scale from 0 to 1000+ instances in seconds
Each Lambda invocation: needs a database connection
Without pooling:
  Each Lambda: opens new connection → runs query → closes connection (or holds it in container)
  1000 concurrent Lambdas = 1000 simultaneous connections → exceeds max_connections

Lambda container reuse:
  Lambda reuses warm containers → connection can be "kept" in global scope
  But: Lambda containers idle out → connections abandoned (not properly closed)
  PostgreSQL: sees zombie connections → exhausts max_connections
```

**RDS Proxy solves this:**
```
Lambda → RDS Proxy (persistent) → RDS instance

RDS Proxy:
  Maintains a warm pool of database connections (configured pool size)
  All Lambda invocations connect to RDS Proxy (lightweight TLS connection)
  RDS Proxy multiplexes many Lambda connections to few DB connections (like PgBouncer)
  
Lambda connection lifecycle:
  Lambda opens connection to RDS Proxy: fast (Proxy already connected to DB)
  Lambda runs query
  Lambda returns, connection to Proxy closes (or times out)
  DB connection stays in Proxy pool (not closed)
  
Benefits:
  max_connections on RDS: 100 (pool_size on Proxy)
  Lambda instances: 1000+ can connect to Proxy simultaneously
  No connection storms: Proxy maintains steady pool regardless of Lambda scaling

Failover handling:
  RDS fails over: Proxy retries connection to new primary
  Lambda sees a brief pause (< 30s) rather than a hard connection error
  Read queries: automatically retried by Proxy
  Write queries: error returned (must be retried with idempotency by application)
```

---

**Q11. How do you monitor and diagnose connection pool health in production?**

**Key metrics to monitor:**

**PgBouncer metrics:**
```
SHOW POOLS;
-- cl_active:    clients currently executing queries
-- cl_waiting:   clients waiting for a connection (pool exhausted — ALERT if > 0)
-- sv_active:    server connections currently in use
-- sv_idle:      available server connections in pool
-- sv_used:      server connections used but returned to pool (not idle)
-- maxwait:      longest wait time of a waiting client (ALERT if > 100ms)

SHOW STATS;
-- total_query_count: total queries processed
-- total_wait_time:   cumulative client wait time
-- avg_query_time:    average query duration

SHOW CLIENTS;
-- Shows all connected clients (application threads)
```

**PostgreSQL connection metrics:**
```sql
-- Active connections by state:
SELECT state, count(*) FROM pg_stat_activity GROUP BY state;
-- Expect: few 'active', many 'idle' (in pool)
-- Alert: many 'idle in transaction' (open transactions sitting idle = locking issue)

-- Connections by application:
SELECT application_name, count(*) FROM pg_stat_activity GROUP BY application_name ORDER BY count DESC;

-- Long-running idle in transaction (lock holders):
SELECT pid, now() - query_start AS idle_time, state, query
FROM pg_stat_activity
WHERE state = 'idle in transaction' AND query_start < NOW() - INTERVAL '30 seconds';
-- These should be investigated and killed if stuck:
SELECT pg_terminate_backend(pid) WHERE pid = ?;

-- Wait events (what are connections waiting for):
SELECT wait_event_type, wait_event, count(*)
FROM pg_stat_activity WHERE state = 'active' GROUP BY 1, 2 ORDER BY count DESC;
-- Lock: waiting for row lock (contention)
-- IO: waiting for disk read (slow queries / low cache hit rate)
```

**Alerting rules:**
```
CRITICAL: cl_waiting > 0 in PgBouncer (pool exhausted, requests queuing)
WARNING:  maxwait > 50ms (clients waiting for connections)
CRITICAL: pg_stat_activity idle in transaction > 60 seconds (stuck transaction)
WARNING:  active connections > 80% of max_connections
```

---

**Q12. What is connection leaking and how do you detect and fix it?**

A connection leak occurs when application code acquires a database connection but fails to return it to the pool — typically due to an unhandled exception, missing finally block, or forgotten cleanup.

**Effects:**
```
Pool has 20 connections
App has a bug: connections leak on exceptions
After 100 exceptions: all 20 pool connections are leaked
New requests: no available connection → timeout → 503 errors
Restoring: must restart application servers to reclaim leaked connections
```

**Detection:**
```sql
-- PostgreSQL: connections open but not doing anything for a long time
SELECT pid,
       application_name,
       now() - backend_start AS connection_age,
       now() - state_change AS time_in_state,
       state
FROM pg_stat_activity
WHERE state = 'idle'
  AND now() - state_change > INTERVAL '10 minutes'
ORDER BY time_in_state DESC;
-- Idle connections open for hours while the pool says it's empty → leak

-- PgBouncer: sv_used high, sv_idle low, cl_waiting growing
SHOW POOLS;
```

**Fixes:**

```python
# WRONG: connection leaked on exception
def get_user(user_id):
    conn = pool.getconn()
    result = conn.execute("SELECT * FROM users WHERE id = %s", user_id)  # exception here → conn never returned
    pool.putconn(conn)  # never reached!
    return result

# CORRECT: use context manager (always returns connection to pool)
def get_user(user_id):
    with pool.connection() as conn:
        return conn.execute("SELECT * FROM users WHERE id = %s", user_id).fetchone()
    # Connection automatically returned even on exception

# CORRECT: try/finally
def get_user(user_id):
    conn = pool.getconn()
    try:
        return conn.execute("SELECT * FROM users WHERE id = %s", user_id).fetchone()
    finally:
        pool.putconn(conn)  # always runs, even on exception
```

**Pool-side prevention:**
```python
# HikariCP (Java): connectionTimeout + leakDetectionThreshold
hikari.setConnectionTimeout(30000)         # fail fast if no connection available in 30s
hikari.setLeakDetectionThreshold(60000)    # log warning if connection held > 60s (likely leak)

# PgBouncer: server_idle_timeout forces idle connections back
server_idle_timeout = 600   # return idle server connections to pool after 10 min
```

---

**Q13. How do you implement zero-downtime database failover with transparent connection handling at the application layer?**

**Goal:** Primary database fails → application continues with < 30 seconds of interruption, no client-visible errors.

**Layer 1: Database HA (Patroni + etcd)**
```
Patroni monitors primary health every 5 seconds
On failure: acquires distributed lock in etcd → promotes replica → 10–15 seconds total
New primary: accepts connections
etcd: leader key updated to new primary's hostname
```

**Layer 2: Connection routing (PgBouncer)**
```
PgBouncer reads primary connection info from etcd (via patroni-pgbouncer script)

# patroni.yml callback:
on_role_change: "python /etc/patroni/update_pgbouncer.py"
# Script: reads current leader from etcd, updates pgbouncer.ini, reloads PgBouncer

# During failover (15-25 seconds):
PgBouncer: pauses new client connections (queues them)
           drains existing connections
           updates primary endpoint
           resumes connections to new primary
           unpauses client queue

# Client experience:
  Queries in-flight during failover: receive error (must retry)
  New queries during failover: wait in PgBouncer queue (up to pause_timeout)
  New queries after PgBouncer reloads: routed to new primary
```

**Layer 3: Application retry logic**
```python
import psycopg2, time, logging

def execute_with_retry(query, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            with get_connection() as conn:
                return conn.execute(query, params).fetchall()
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            # OperationalError: connection lost, server restarted, etc.
            if attempt == max_retries - 1:
                raise
            wait = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
            logging.warning(f"DB connection error, retry {attempt+1}/{max_retries} in {wait}s: {e}")
            time.sleep(wait)
            pool.reset()  # force pool to re-establish connections

# IMPORTANT: Only retry idempotent operations (SELECT, or INSERT with ON CONFLICT DO NOTHING)
# Never auto-retry writes without idempotency key (could double-charge, double-insert)
```

**Layer 4: Health checks prevent traffic to unhealthy app**
```yaml
# Kubernetes readiness probe:
readinessProbe:
  httpGet:
    path: /health/db
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 3

# /health/db endpoint:
def db_health_check():
    try:
        conn.execute("SELECT 1")
        return 200, "OK"
    except:
        return 503, "DB unavailable"
# If DB is unreachable: pod is marked NotReady → load balancer stops routing to it
```

---

**Q14. How does Vitess handle connection pooling for MySQL at scale, and what advantages does it provide over simple PgBouncer?**

**Vitess architecture:**
```
Application → VTGate (proxy layer) → VTablet (per-shard tablet server) → MySQL shard
```

**VTGate (query router):**
```
Accepts MySQL protocol connections from application
Maintains connection pool to each VTablet
Routes queries to correct shard based on routing rules (Vindex)
Handles scatter-gather for multi-shard queries
Load-balances reads across replicas

Connection management:
  Application: thousands of connections to VTGate (lightweight)
  VTGate: small pool to each VTablet (say, 50 per shard)
  With 8 shards: 8 × 50 = 400 actual MySQL connections
```

**VTablet (MySQL-side proxy):**
```
Runs alongside each MySQL instance (sidecar container)
Manages connections to its local MySQL
Enforces query timeouts (kills long-running queries)
Collects per-table query metrics
Handles schema changes (online DDL)
```

**Advantages over PgBouncer for multi-shard MySQL:**

| Feature | PgBouncer | Vitess |
|---|---|---|
| Sharding | None | Built-in (Vindex) |
| Multi-shard queries | None | Scatter-gather + merge |
| Online schema changes | None | Built-in (OSC/gh-ost) |
| Query timeouts | None | Per-query timeout |
| Read/write splitting | None | Built-in (replica routing) |
| Query analytics | None | Per-query latency tracking |
| Connection pool per shard | Manual | Automatic |
| Failover handling | Manual script | Automatic (via VTOrc) |

**Use PgBouncer:** Single PostgreSQL cluster (or a few replicas), simple connection multiplexing.
**Use Vitess:** MySQL at hyperscale (YouTube, Slack, GitHub), automated sharding, complex routing.

---

**Q15. What is a prepared statement and how does its interaction with connection pools affect performance?**

**Prepared statement lifecycle:**
```
1. PREPARE: client sends query template to server
   PostgreSQL parses and plans the query once → stores plan in memory (per-session)
   
2. EXECUTE: client sends parameter values
   PostgreSQL uses cached plan → skips parsing and planning
   
3. DEALLOCATE: client frees the prepared statement (or auto-freed on session close)
```

**Performance benefit:**
```
Without prepared statement:
  Every query: parse (1ms) + plan (0.5ms) + execute (10ms) = 11.5ms

With prepared statement (after first execution):
  PREPARE: parse + plan once = 1.5ms (paid once)
  EXECUTE: only execute = 10ms (every time)
  
For 10,000 executions of the same query:
  Without: 115,000ms total
  With:    1.5ms + 10,000 × 10ms = 100,001.5ms (saves ~15 seconds total)
  
Most impactful for: queries executed thousands of times per second (OLTP hot paths)
```

**Interaction with PgBouncer transaction mode:**
```
Problem: prepared statements are session-specific
  Session 1 creates PREPARE get_user
  PgBouncer transaction mode: session 1 borrows connection 5
  After transaction, connection 5 returned to pool
  Session 1's next transaction may get connection 8
  Connection 8 doesn't have PREPARE get_user → ERROR

Solutions:
  1. Protocol-level (unnamed) prepared statements:
     JDBC, psycopg2, Go pgx use protocol-level prepare (message type 'P')
     PgBouncer in transaction mode: intercepts and tracks these transparently
     → Works correctly with transaction mode pooling

  2. PgBouncer 1.21+ with prepared_statement_support = yes:
     PgBouncer tracks named prepared statements per client
     Maps client's statement names to actual server connections
     → Named prepared statements work in transaction mode

  3. Disable server-side prepared statements (disable_pqexec = 1 in PgBouncer):
     Force all queries to use simple query protocol (no prepare)
     → Loses plan caching benefit (higher CPU on PostgreSQL)
     → Use only when compatibility matters more than performance
```

---

## Hard (Q16–Q20)

---

**Q16. Design the connection pooling architecture for a microservices platform with 30 services, each deployed in Kubernetes with auto-scaling.**

**Problem:**
```
30 services × 100 pods each = 3000 pods
Each pod wants 5 DB connections = 15,000 connections requested
PostgreSQL max_connections = 500
Without pooling: 14,500 connections over limit → all connections fail
With horizontal pod autoscaling: connections spike on scale-out events
```

**Architecture: Sidecar PgBouncer + Central PgBouncer**

```
Tier 1: Sidecar PgBouncer (per service)
  Each service deployment has a PgBouncer sidecar container
  App pods: connect to localhost:5432 (sidecar PgBouncer)
  Sidecar: pool_size = 3 connections (per service pod) → to central PgBouncer
  
  100 pods × 3 connections = 300 connections (from this service to central)
  App pod: has 10 app threads → multiplexed through 3 connections → fast

Tier 2: Central PgBouncer
  Receives connections from all 30 services
  30 services × 300 = 9000 connections from sidecars to central PgBouncer
  Central pool_size = 10 per service (actual DB connections)
  30 services × 10 = 300 actual PostgreSQL connections
  
  PostgreSQL: sees 300 connections (within max_connections = 500)
```

**Kubernetes configuration:**
```yaml
# Deployment with sidecar PgBouncer:
spec:
  containers:
  - name: app
    image: my-service:latest
    env:
    - name: DB_HOST
      value: "localhost"  # connects to sidecar, not DB directly
    - name: DB_PORT
      value: "5432"
  
  - name: pgbouncer
    image: pgbouncer/pgbouncer:latest
    env:
    - name: DATABASE_URL
      value: "postgresql://user@central-pgbouncer:6432/mydb"
    - name: POOL_MODE
      value: "transaction"
    - name: DEFAULT_POOL_SIZE
      value: "3"  # connections per pod to central PgBouncer
    - name: MAX_CLIENT_CONN
      value: "50"  # allow up to 50 app threads per pod
```

**Auto-scaling considerations:**
```
HPA scales from 10 → 100 pods:
  Each new pod starts sidecar PgBouncer
  Sidecar establishes 3 connections to central PgBouncer
  90 new pods × 3 connections = 270 new connections to central PgBouncer
  Central PgBouncer: has capacity (it limits actual DB connections, not inbound)
  PostgreSQL: DB connections stay at 300 (central PgBouncer doesn't increase DB connections on scale-out)
  
→ Pod auto-scaling has no impact on PostgreSQL connection count
```

---

**Q17. How do you handle database credential rotation without downtime using a connection pool?**

**The problem:**
```
Security policy: rotate DB passwords every 90 days
Naive approach: update password in DB → update application config → restart application
Downtime: during restart, connections fail
With 30 microservices: 30 separate restart events → coordination nightmare
```

**Solution 1: Rolling credential rotation with PgBouncer**
```
Step 1: Create new password for app role (DB supports both old and new simultaneously):
  -- In PostgreSQL: add new password, keep old one valid for overlap period
  ALTER USER app_user PASSWORD 'new_password_xyz';
  -- PostgreSQL stores one password per user, so old password immediately invalid
  -- → Must use a wrapper approach

Step 2: Create a new DB user for the rotation:
  CREATE USER app_user_v2 PASSWORD 'new_password_xyz';
  GRANT SAME PRIVILEGES TO app_user_v2 AS app_user;

Step 3: Update PgBouncer config to use app_user_v2:
  RELOAD PgBouncer (no downtime — existing connections finish their transaction, new ones use new creds)
  
Step 4: After PgBouncer reload, gracefully retire app_user_v1:
  Wait for no active connections using app_user_v1
  DROP USER app_user_v1;

Step 5: At next rotation, create app_user_v3, retire app_user_v2
```

**Solution 2: AWS Secrets Manager + RDS Proxy (automated)**
```
AWS Secrets Manager: stores and auto-rotates DB credentials every N days
  Rotation Lambda: creates new password in RDS, updates secret in Secrets Manager
  Old password: kept valid for overlap window (1 hour)

RDS Proxy: reads current credentials from Secrets Manager on each new connection
  Application connects to RDS Proxy with IAM auth (no DB password in application code at all!)
  RDS Proxy resolves credentials → connects to RDS with current password
  
  Password rotation: RDS Proxy transparently picks up new credentials
  Application: sees nothing (connects to Proxy with IAM, not username/password)
  Zero downtime, zero coordination across services
```

**Solution 3: PgBouncer with auth_user and auth_query**
```ini
# PgBouncer: authenticate clients via a query to the DB
auth_type = md5
auth_user = pgbouncer_auth_user
auth_query = SELECT p_user, p_password FROM pgbouncer.get_auth($1);
# DB stores credentials in a function that can be rotated without restarting PgBouncer
# On rotation: update the auth function's return value
# PgBouncer: picks up new credentials for next connection → no restart
```

---

**Q18. Explain the impact of `idle_in_transaction_session_timeout` and `statement_timeout` on connection pool behavior.**

**`idle_in_transaction_session_timeout`:**
```sql
-- A session that is in a transaction but idle for more than this duration is killed
idle_in_transaction_session_timeout = '5min'

Scenario without this setting:
  Application starts a transaction: BEGIN;
  Application does some processing... crashes or hangs
  Transaction stays open: holds row-level locks on all modified rows
  Other transactions: blocked waiting for those locks
  Hours later: still locked; everything is stuck

With this setting:
  Transaction idle for 5 minutes → PostgreSQL terminates the session
  Locks released → other transactions can proceed
  
Impact on connection pooling:
  PgBouncer: if a pooled connection is "idle in transaction" when PgBouncer rotates it to another client:
    Other client starts using the connection mid-transaction (BAD!)
    PgBouncer transaction mode: this CANNOT happen (PgBouncer waits for COMMIT/ROLLBACK before returning)
    So idle_in_transaction_session_timeout is a safety net for stuck app connections that never committed
```

**`statement_timeout`:**
```sql
-- Any single query running longer than this is cancelled
statement_timeout = '30s'

Impact on connection pool:
  Long-running queries: hold the DB connection for the duration
  Pool has 20 connections, each running a 60-second analytical query:
    20 connections × 60 seconds = all connections tied up for 1 minute
    Other requests: wait in PgBouncer queue (cl_waiting grows)
    
  With statement_timeout = 30s:
    Long query cancelled after 30s → connection returned to pool sooner
    Other requests: get served
    
  Per-query override for legitimate long queries:
    SET statement_timeout = '5min' before the long query
    RESET statement_timeout after

Application implications:
  All queries in the pool should complete within statement_timeout
  If query legitimately needs more time: increase timeout for that specific query
  Never increase globally to accommodate one slow query (it will let all slow queries run)
```

---

**Q19. How does connection affinity (sticky connections) help with read-after-write consistency and when does it hurt performance?**

**Read-after-write problem:**
```
User updates their profile
Write goes to primary
Read is routed to replica (load balanced)
Replica hasn't replicated the write yet (lag: 50ms–2s)
User sees their old profile: "my update was lost!"

→ User experience issue even though data is not lost
```

**Connection affinity (sticky sessions) solution:**
```
After any write from a user:
  Mark that user's session: "must read from primary for next N seconds"
  Implement via Redis:
    redis.setex(f"primary_read:{user_id}", 5, "1")  # 5-second window
  
Read routing:
  if redis.get(f"primary_read:{user_id}"):
      conn = primary_pool.getconn()   # force primary read
  else:
      conn = replica_pool.getconn()   # normal replica read
```

**Proxy-level connection affinity:**
```
ProxySQL: can stick a session to primary after a write is detected
  transaction_persistent = 1: queries within a transaction always use same server
  After transaction: subsequent reads may route to replica

PgBouncer: does not have built-in read/write splitting
  → Use application-level tracking (Redis flag) or use separate connection strings
```

**When sticky connections hurt performance:**
```
Scenario: 90% of reads happen within 5 seconds of a write
  → 90% of reads are forced to primary
  → Read replicas sit idle
  → Primary is overwhelmed with read + write load
  → Benefit of read replicas: nearly zero

When this happens:
  Reduce affinity window: from 5s to 1s (most replication lag < 1s)
  Use synchronous replica for fast users: replica always current → no affinity needed
  Monitor replication lag: if consistently < 100ms, reduce window to 200ms
  Distinguish: user reading their OWN data (needs affinity) vs user reading OTHERS' data (no affinity needed)
```

---

**Q20. Design a global connection pooling strategy for a SaaS platform with databases in 5 geographic regions.**

**Challenge:**
- 30 microservices deployed in 5 regions (US-East, US-West, EU, APAC-SE, APAC-NE)
- Each region has: 1 PostgreSQL primary + 2 replicas
- Services in each region must connect to local DB (latency) + occasionally cross-region (consistency)

**Architecture per region:**
```
Region: US-East

[Service pods] → [Local PgBouncer (Kubernetes service)] → [US-East Primary (writes)]
                                                         → [US-East Replica 1 (reads)]
                                                         → [US-East Replica 2 (reads)]

PgBouncer deployment (DaemonSet — one per node, all pods on node share it):
  Write pool: 10 connections to primary
  Read pool:  20 connections to replicas (round-robin)
  Mode: transaction

Cross-region (rare, for globally-consistent data):
  Direct connection to primary of target region (no PgBouncer pooling — these are rare)
  Connection limit: 5 connections per service per cross-region destination
  Use sparingly: add latency budget (100–200ms) for cross-region calls
```

**Credential management:**
```
Each region: separate DB user (for audit isolation)
  app_use_us_east, app_user_eu_west, etc.
  Credentials in AWS Secrets Manager / GCP Secret Manager per region
  PgBouncer: uses auth_query to fetch credentials from secrets (no hardcoded passwords)
  Rotation: each region's secrets rotated independently
```

**Failover handling:**
```
US-East primary fails:
  Patroni: promotes US-East Replica 1 to primary (15 seconds)
  
  PgBouncer (local): watches etcd for leader change
    Patroni callback → PgBouncer config reload
    In-flight transactions: receive error (application retries with backoff)
    New connections: routed to new primary within 5 seconds of Patroni reload
    
  Services in EU/APAC using US-East cross-region connections:
    Reconnect to new primary endpoint (etcd provides updated endpoint)
    Cross-region writes: briefly unavailable (20–30s) during failover
    Application: retries with idempotency key
```

**Monitoring:**
```
Per-region PgBouncer metrics (Prometheus + Grafana):
  cl_waiting > 0: ALERT (pool exhausted)
  maxwait > 200ms: WARNING
  sv_idle < 3: WARNING (pool nearly exhausted)
  
Cross-region: 
  Alert if cross-region connection pool > 70% utilization
  Investigate: are services overusing cross-region queries?
  Fix: move data to correct region or accept higher latency budget
```

---

## Quick Reference

```
Connection pool sizing:
  Optimal DB connections ≈ num_CPU_cores × 2 + spindles
  Little's Law: N = λ × W (concurrent connections = RPS × avg query time)

PgBouncer modes:
  Session:     1 client = 1 DB connection (full session features)
  Transaction: 1 DB connection shared per-transaction (most common, best multiplexing)
  Statement:   1 DB connection per statement (rarely used)

PgBouncer limitations in transaction mode:
  - LISTEN/NOTIFY requires dedicated connection
  - Named prepared statements require PgBouncer 1.21+ or protocol-level statements
  - session-level SET variables don't persist across transactions

Connection storm prevention:
  - PgBouncer as buffer (absorbs reconnect storm; re-establishes pool gradually)
  - Jitter in application reconnect backoff
  - CONNECTION LIMIT per DB role

Key PostgreSQL settings:
  max_connections: hard limit (typically 100–500)
  idle_in_transaction_session_timeout: kill stuck transactions
  statement_timeout: prevent long queries from monopolizing pool connections

Read-after-write consistency:
  Short TTL Redis flag → route reads to primary for N seconds after write
  Synchronous replica → zero lag, no affinity needed (but higher write latency)
  Accept staleness for non-user-specific reads

Credential rotation (zero downtime):
  AWS RDS Proxy + Secrets Manager: fully automated, no application changes
  PgBouncer auth_query: fetch credentials from DB → rotate without restart
  Blue/green user rotation: create new user → switch PgBouncer → retire old user
```
