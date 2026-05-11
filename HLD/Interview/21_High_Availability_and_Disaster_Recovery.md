# High Availability and Disaster Recovery — Interview Q&A

---

## Easy (Q1–Q7)

---

### Q1. What is the difference between High Availability and Fault Tolerance?

**High Availability (HA)** means a system remains operational for a very high percentage of time, accepting that brief interruptions may occur during failover. The goal is to minimize downtime through redundancy, health checks, and automatic failover. A typical HA target is 99.99% uptime (~52 minutes downtime per year).

**Fault Tolerance (FT)** is a stronger guarantee: the system continues operating *without any interruption* even when a component fails. This is achieved through hardware-level redundancy (RAID, dual power supplies, redundant network paths) so that failures are completely masked from users.

```
High Availability:
  Normal  → Component fails → Brief blip → Failover completes → Healthy
                              (seconds to minutes downtime)

Fault Tolerance:
  Normal  → Component fails → Zero downtime → Redundant takes over seamlessly
```

| Property            | High Availability         | Fault Tolerance             |
|---------------------|---------------------------|-----------------------------|
| Downtime on failure | Seconds to minutes        | Zero (seamless)             |
| Cost                | Moderate                  | High (2x–3x redundancy)     |
| Complexity          | Medium                    | High                        |
| Example             | DB with standby + Patroni | RAID-1 disk mirroring       |
| Use case            | Web apps, APIs            | Avionics, financial trading |

In practice, most internet-scale systems target HA rather than full fault tolerance because FT requires expensive hardware duplication. FT is reserved for mission-critical scenarios where even milliseconds of downtime are unacceptable — such as hospital life-support systems or real-time trading engines.

---

### Q2. What are the "nines of availability" and what do they mean in practice?

The "nines" express uptime as a percentage and translate to concrete allowed downtime per year:

| Availability | Annual Downtime | Monthly Downtime | Notes                         |
|--------------|-----------------|------------------|-------------------------------|
| 99%          | 3.65 days       | 7.2 hours        | Single server, no redundancy  |
| 99.9%        | 8.76 hours      | 43.8 minutes     | Basic HA with standby         |
| 99.99%       | 52.6 minutes    | 4.4 minutes      | Production-grade, multi-AZ    |
| 99.999%      | 5.26 minutes    | 26.3 seconds     | Carrier-grade, very expensive |
| 99.9999%     | 31.5 seconds    | 2.6 seconds      | Rarely achievable end-to-end  |

**Calculation:** Downtime = (1 − availability) × seconds_per_year.

```
99.99% = 0.0001 × 31,536,000 seconds = 3,153.6 seconds ≈ 52.6 minutes/year
```

Moving from 99.9% to 99.99% reduces allowed downtime by ~10x but typically requires:
- Multi-AZ deployments
- Automated failover (not manual)
- Load balancers with health checks
- Zero-downtime deployment pipelines

Achieving 99.999% end-to-end is extremely difficult because composite availability across multiple services multiplies: a system with 5 components each at 99.99% has a composite availability of only 99.95% if they are in series.

---

### Q3. What are RPO and RTO? Give examples.

**RTO (Recovery Time Objective):** The maximum acceptable time the system can be down after a failure. It answers: *"How long can we be offline?"*

**RPO (Recovery Point Objective):** The maximum acceptable amount of data loss measured in time. It answers: *"How much data can we afford to lose?"*

```
Timeline:
  |---- Last Backup ----|---- Failure ----|---- Recovery ---|
  t=0                   t=4h              t=5h               t=7h
  
  RPO = 4 hours (data from last 4 hours lost)
  RTO = 2 hours (took 2 hours to restore service)
```

**Examples by system type:**

| System              | RPO Target  | RTO Target | Strategy                         |
|---------------------|-------------|------------|----------------------------------|
| Banking transaction | Near 0      | < 1 min    | Sync replication, hot standby    |
| E-commerce cart     | 1 hour      | 15 min     | Async replication, warm standby  |
| Analytics reports   | 24 hours    | 4 hours    | Daily backups, restore from S3   |
| Dev/test env        | 1 week      | 1 day      | Weekly snapshots                 |

Lowering RPO requires more frequent backups or synchronous replication (which adds write latency). Lowering RTO requires faster recovery procedures — pre-warmed standby instances are faster than cold restores from backup. Both have cost implications: stricter SLAs need more infrastructure and operational investment.

---

### Q4. What is the difference between active-active and active-passive failover?

**Active-Passive:** Only one instance serves traffic at a time. The passive instance is on standby and takes over when the active fails. Failover involves detecting the failure and routing traffic to the standby.

**Active-Active:** All instances serve traffic simultaneously. When one fails, the others absorb its load without any failover delay.

```
Active-Passive:
  Client → Load Balancer → [PRIMARY active] → DB
                        → [STANDBY passive] (idle, ready)
  On failure: LB detects, promotes standby → ~30s downtime

Active-Active:
  Client → Load Balancer → [Node A] → DB
                        → [Node B] → DB
                        → [Node C] → DB
  On failure: LB removes Node B → zero downtime, 33% more load on A, C
```

| Trade-off          | Active-Passive              | Active-Active               |
|--------------------|-----------------------------|-----------------------------|
| RTO                | Seconds to minutes          | Near zero                   |
| Resource usage     | 50% (standby idles)         | 100% (all serve traffic)    |
| Complexity         | Low                         | High (conflict resolution)  |
| Capacity on failure| Full (standby takes over)   | Reduced (N-1 nodes)         |
| Data consistency   | Easier (single writer)      | Hard (concurrent writes)    |
| Cost efficiency    | Wasteful (idle standby)     | Efficient                   |

Active-active is preferred for stateless services. For stateful systems (databases), active-passive is simpler to reason about correctness, while active-active requires distributed transaction coordination or conflict resolution.

---

### Q5. What is a Single Point of Failure (SPOF) and how do you eliminate them?

A Single Point of Failure is any component whose failure causes the entire system to go down. Identifying and eliminating SPOFs is the core exercise in HA design.

**Common SPOFs and their mitigations:**

```
System:  Client → DNS → LB → App Server → Cache → DB → Storage
SPOFs:         ^      ^    ^             ^       ^    ^
```

| Component     | SPOF Risk                        | Mitigation                                        |
|---------------|----------------------------------|---------------------------------------------------|
| DNS           | Single DNS server fails          | Use managed DNS (Route 53) with health checks     |
| Load Balancer | LB process crashes               | HA LB pairs (HAProxy + Keepalived), cloud LB      |
| App Servers   | Single app instance              | Run multiple instances across AZs                 |
| Cache         | Redis single instance            | Redis Sentinel or Redis Cluster                   |
| Database      | Single DB node                   | Primary + replica with Patroni auto-failover      |
| Storage       | Local disk failure               | Distributed storage (EBS, S3, RAID)               |
| Network       | Single NIC or switch             | Bonded NICs, redundant switches                   |
| Power         | Single PSU or datacenter         | Redundant PSUs, multi-AZ across different DCs     |

**HA checklist before going to production:**
- [ ] Is every component running with at least 2 instances?
- [ ] Is there an automated health check and failover for each component?
- [ ] Is the database primary/replica with auto-failover configured?
- [ ] Are all instances spread across multiple Availability Zones?
- [ ] Is there a redundant load balancer (not a single LB instance)?

---

### Q6. What is geographic redundancy and what is the difference between multi-AZ and multi-region?

**Geographic redundancy** means distributing infrastructure across physically separate locations so that a failure in one location does not take down the entire system.

**Multi-AZ (Availability Zone):** Multiple isolated data centers within the same geographic region, connected by low-latency, high-bandwidth private links (~1–2ms latency). AZs share the same region's infrastructure team but are on separate power grids, cooling, and networking.

**Multi-Region:** Completely separate geographic regions (e.g., us-east-1 and eu-west-1). Replication crosses public internet or dedicated WAN (~50–200ms latency). Protects against regional disasters (hurricane, power grid failure, cloud provider region outage).

```
Multi-AZ (same region, ~1ms):
  us-east-1:
    AZ-a: App + DB Primary
    AZ-b: App + DB Replica (auto-failover)
    AZ-c: App + DB Replica

Multi-Region (~100ms):
  us-east-1 (Primary):  App + DB Primary ──async──> eu-west-1 (Standby): App + DB Replica
```

| Dimension          | Multi-AZ                  | Multi-Region                  |
|--------------------|---------------------------|-------------------------------|
| Latency between AZ | 1–2ms                     | 50–200ms                      |
| Protects against   | Datacenter failure        | Regional disaster             |
| Replication        | Synchronous (low latency) | Usually async (higher RPO)    |
| Failover time      | Seconds (automatic)       | Minutes (often manual)        |
| Cost               | Moderate                  | High (2x infrastructure)      |
| Complexity         | Low                       | High (DNS, data conflicts)    |

Most production systems start with multi-AZ for HA and add multi-region when business requires geographic presence or compliance demands data residency.

---

### Q7. What are backup strategies and what is the 3-2-1 rule?

**Backup strategies:**

- **Full backup:** A complete copy of all data. Simple to restore but slow to create and requires the most storage.
- **Incremental backup:** Only backs up data changed since the last backup (full or incremental). Fast and small, but restoration requires applying the chain of incrementals on top of a full backup.
- **Differential backup:** Backs up all data changed since the last *full* backup. Larger than incremental but simpler to restore (base + one differential).

```
Full:        Mon[ALL] Tue[ALL] Wed[ALL] ... (large, slow)

Incremental: Mon[ALL] Tue[DELTA1] Wed[DELTA2] ... (fast, chain restore)

Differential:Mon[ALL] Tue[SINCE-MON] Wed[SINCE-MON] ... (medium, 2-step restore)
```

**The 3-2-1 Rule:**
- **3** copies of the data
- **2** different storage media types (e.g., SSD + S3 object storage)
- **1** copy stored offsite (different datacenter, cloud region, or physical location)

This rule protects against hardware failure, ransomware, and regional disasters. If your primary disk and local backup are destroyed in the same fire, the offsite copy saves you.

**Modern extension — 3-2-1-1-0:**
- 3 copies, 2 media, 1 offsite, 1 air-gapped (offline, immune to ransomware), 0 backup errors (regularly verify restores).

Always test backups by restoring to a staging environment periodically — a backup that has never been restored is a backup you cannot trust.

---

## Medium (Q8–Q15)

---

### Q8. How does data replication work for DR and what is the impact of synchronous vs asynchronous replication on RPO?

**Synchronous replication:** The primary waits for the replica to confirm the write before acknowledging success to the client. This guarantees zero data loss (RPO = 0) but adds latency equal to the round-trip to the replica.

**Asynchronous replication:** The primary acknowledges the write immediately and replicates to the replica in the background. This has near-zero write latency impact but means the replica may lag — any data in the replication lag window is lost if the primary fails (RPO = lag, typically seconds to minutes).

```
Synchronous:
  Client → Primary → writes WAL → waits for Replica ACK → replies to Client
  Latency added: 2 × network_RTT (e.g., +2ms in same AZ, +100ms cross-region)

Asynchronous:
  Client → Primary → replies to Client immediately
  Primary → Replica (background, ~seconds lag)
  On primary failure: lose last few seconds of writes
```

**RPO trade-off table:**

| Replication Type   | RPO             | Write Latency Impact | Use Case                          |
|--------------------|-----------------|----------------------|-----------------------------------|
| Synchronous        | 0 (zero loss)   | +RTT per write       | Financial transactions, banking   |
| Semi-synchronous   | Near 0          | Low (1 replica ack)  | MySQL semi-sync, good compromise  |
| Asynchronous       | Seconds–minutes | Near zero            | Read replicas, cross-region DR    |

Cross-region synchronous replication is generally impractical (adds 100ms+ to every write). Most multi-region DR architectures accept an RPO of seconds to minutes using async replication, with automated tooling to detect and minimize lag. PostgreSQL streaming replication supports both modes via `synchronous_commit` settings.

---

### Q9. How does automated failover work? Describe Patroni for PostgreSQL and Route 53 health checks.

**Automated failover** replaces manual operator intervention with software that detects failures, elects a new primary, and redirects traffic — reducing RTO from hours to seconds.

**Patroni (PostgreSQL HA):**
Patroni is a template for PostgreSQL HA using distributed consensus (etcd, ZooKeeper, or Consul) to coordinate leader election:

```
Architecture:
  [Patroni Agent] on each PostgreSQL node
       ↓ leader election via
  [etcd cluster] (3 nodes for quorum)
       ↓ winner becomes
  [Primary PostgreSQL] ← all writes
  [Replicas] ← streaming replication from primary

On primary failure:
  1. Patroni agents detect primary missing (heartbeat timeout ~10s)
  2. Agents compete for lock in etcd
  3. Winner (most up-to-date replica) promotes itself
  4. Other replicas repoint to new primary
  5. HAProxy/PgBouncer re-routes connections (via Patroni REST API)
Total failover: ~15-30 seconds
```

**Route 53 Health Checks (DNS-based failover):**
```
Route 53 health checker → polls /health endpoint every 10s
  If 3 consecutive failures → marks endpoint unhealthy
  DNS record → switches to secondary endpoint
  TTL (30–60s) controls how fast clients see the change

Active-Passive DNS:
  Primary:   app.example.com → 1.2.3.4 (PRIMARY, Failover=PRIMARY)
  Secondary: app.example.com → 5.6.7.8 (DR, Failover=SECONDARY)
  On failure: Route 53 serves secondary record automatically
```

Combining Patroni with Route 53 health checks gives database-level and DNS-level failover. The key insight is that health checks must be meaningfully testing the application — not just TCP connectivity but actual query execution.

---

### Q10. What is Point-in-Time Recovery (PITR) and how does it work?

PITR allows restoring a database to any specific point in time, not just the time of the last backup. This is critical for recovery from logical errors — a mistaken `DELETE`, a bad deployment that corrupted data, or data ingestion errors.

**How it works (PostgreSQL example):**

```
Continuous Archiving:
  1. Take base backup (full snapshot of data directory)
  2. Archive WAL (Write-Ahead Log) segments continuously to S3
  3. WAL contains every change: INSERT, UPDATE, DELETE with timestamps

Recovery process:
  1. Restore base backup to a new instance
  2. Apply WAL segments one by one up to target time
  3. Database state is exactly as it was at target_time

pg_restore command:
  recovery_target_time = '2025-10-15 14:32:00'
  restore_command = 'aws s3 cp s3://backups/wal/%f %p'
```

**PITR timeline example:**
```
08:00 → Base backup taken
09:00 → DBA accidentally runs: DELETE FROM orders WHERE 1=1
...
10:00 → Issue discovered

PITR target: restore to 08:59:59
  Result: all data from base backup + 59 minutes of WAL applied
  Lost: only the accidental deletes — not real data
```

**Key requirements for PITR:**
- Continuous WAL/binlog archiving to durable storage
- Known restore point (timestamp or LSN)
- Sufficient storage for WAL archives (typically 7–30 days)

Cloud databases (RDS, Cloud SQL) provide PITR natively with 5-minute granularity or better, making it accessible without manual WAL management.

---

### Q11. What is chaos engineering and how does the Netflix Simian Army validate HA?

**Chaos engineering** is the discipline of intentionally introducing failures into a production or staging system to discover weaknesses before real outages occur. The philosophy: if failures are inevitable, find them on your terms rather than the adversary's.

**Netflix Simian Army — key tools:**

```
Chaos Monkey        → randomly terminates EC2 instances in production
                      Tests: do app servers recover automatically?

Chaos Gorilla       → simulates entire AWS AZ going down
                      Tests: does multi-AZ traffic routing work?

Chaos Kong          → simulates entire AWS region failure
                      Tests: does multi-region DR activate?

Latency Monkey      → injects network latency between services
                      Tests: do timeouts and circuit breakers fire?

Doctor Monkey       → health-checks instances, removes sick ones
                      Tests: does health-check-based replacement work?

Security Monkey     → audits security group and IAM policy changes
                      Tests: are security regressions introduced?
```

**Chaos engineering process:**
1. Define the steady state (normal system behavior metrics: error rate, latency)
2. Hypothesize the failure won't affect steady state
3. Introduce failure in production (start small — one instance, one AZ)
4. Observe whether the hypothesis holds
5. Fix discovered weaknesses, repeat

**Key principle — blast radius minimization:** Start with low-traffic periods, small scopes (one instance), and gradually increase. Use feature flags to route a percentage of traffic through the chaos environment.

The goal is not to break things randomly but to build confidence that HA mechanisms actually work under real production conditions, not just on paper.

---

### Q12. What is blast radius reduction and how does cell architecture help?

**Blast radius** is the scope of impact when a failure occurs. Reducing blast radius means containing failures so they affect the smallest possible portion of users.

**Cell architecture** partitions the system into independent cells, each serving a subset of users. A failure in one cell affects only that cell's users, not the entire platform.

```
Without cells (monolithic pool):
  [All Users] → [Shared Service Pool] → [Shared DB]
  One bug in pool → ALL users affected

With cell architecture:
  Users A-M → Cell 1: [App1] → [DB1]
  Users N-Z → Cell 2: [App2] → [DB2]
  Datacenter failure in Cell1 → only Users A-M affected (50%)
  Bad deploy to Cell1 → only Users A-M affected (canary)
```

**Cell design principles:**
- Cells are independent: no shared state, no cross-cell calls
- Each cell can be deployed and scaled independently
- Cell assignment is based on user ID, tenant ID, or geography
- A bad deployment is rolled out one cell at a time (built-in canary)

**Other blast radius reduction techniques:**

| Technique            | How it reduces blast radius                        |
|----------------------|----------------------------------------------------|
| Feature flags        | Disable feature for all users without deploy       |
| Bulkhead pattern     | Separate thread pools per dependency               |
| Circuit breakers     | Stop sending to failing service, don't cascade     |
| Rate limiting        | One bad client can't exhaust shared resources      |
| Multi-tenancy quotas | One tenant's spike doesn't starve others           |

Amazon, Slack, and Stripe all use cell-based architectures. AWS calls them "shards" — each availability zone is itself a form of cell.

---

### Q13. How do you test disaster recovery? Describe runbooks, game days, and quarterly drills.

**DR testing** is non-negotiable: a DR plan that has never been tested is a plan that will fail under real disaster pressure. Testing validates both the technical procedures and the team's ability to execute them.

**Runbooks:**
A runbook is a documented, step-by-step procedure for responding to a specific incident or executing a DR scenario. Every alert should have an associated runbook.

```
Example Runbook: PostgreSQL Primary Failover
  Trigger: PagerDuty alert "postgres-primary-unreachable"
  
  Steps:
  1. Verify failure: ssh db-primary, check pg_isready
  2. Check Patroni status: patronictl -c /etc/patroni.yml list
  3. If Patroni did NOT auto-failover: patronictl failover cluster-name
  4. Verify new primary: patronictl list → leader should be a replica
  5. Check application connectivity: curl https://app/health
  6. Update monitoring dashboards, acknowledge alert
  7. Begin root cause analysis for original failure
  
  Estimated RTO: 5 minutes with runbook
  Without runbook: 30-60 minutes
```

**Game Days:**
Structured exercises where the team deliberately triggers failure scenarios in production or a production-like environment and executes the response. Teams are measured on RTO achieved vs. target.

**Quarterly DR Drills:**
- Full failover to DR region: validate RTO
- Backup restore validation: validate RPO
- Network partition simulation: validate split-brain handling
- Chaos engineering game day

**DR test scorecard:**
- Did the system fail over in under the RTO target?
- Was data loss within the RPO target?
- Did the runbook need corrections?
- Did the on-call engineer complete the steps without escalation?

Testing DR quarterly ensures runbooks stay current as infrastructure evolves.

---

### Q14. What is split-brain and how do you prevent it with quorum, STONITH, and fencing?

**Split-brain** occurs when a distributed system loses network connectivity between nodes, and multiple nodes simultaneously believe they are the primary/leader — each accepting writes independently, creating divergent data.

```
Split-brain scenario:
  [Primary DB] ←— network partition —→ [Standby DB]
  Primary: "I'm alive, accepting writes"
  Standby: "Primary is dead, I am now primary, accepting writes"
  Result: Two primaries writing different data → data corruption
```

**Quorum:** A node may only act as primary if it has the agreement of more than half (N/2 + 1) of cluster members. With 3 nodes, quorum requires 2. A partition that isolates 1 node cannot form quorum, so that node remains standby.

```
3-node cluster, network splits 1 vs 2:
  Side A (1 node): cannot reach quorum → stays standby
  Side B (2 nodes): has quorum → one may become primary
  Result: at most one primary at any time
```

**STONITH (Shoot The Other Node In The Head):** When a node is suspected of being in an inconsistent state, the surviving cluster forcibly powers it off via IPMI, iLO, or a cloud API kill call — ensuring the old primary cannot continue writing.

**Fencing:** The broader technique of isolating a failed node. Can be power fencing (STONITH), storage fencing (revoke disk access), or network fencing (block port access). The critical property: fencing must complete before the new primary starts writing.

```
Patroni fencing:
  1. Old primary health check fails
  2. etcd lock expires after TTL
  3. New leader candidate acquires lock
  4. DCS fencing script calls AWS API to stop old primary instance
  5. Only then does new primary start accepting writes
```

---

### Q15. How do you design zero-downtime deployments? Compare blue-green, canary, and rolling.

Zero-downtime deployments allow code changes to reach production without any period where the service is unavailable. Three primary strategies exist:

**Blue-Green Deployment:**
```
Blue (v1) environment: [App v1 x3] → [DB]  ← production traffic
Green (v2) environment: [App v2 x3] → [DB]  ← new version, tested

Switch: LB routing → Green instantly
Rollback: LB routing → Blue (seconds)

Pros: instant cutover, instant rollback
Cons: 2x infrastructure cost during deploy, DB schema changes are complex
```

**Canary Deployment:**
```
Production: 97% traffic → App v1
Canary:      3% traffic → App v2

Monitor error rates, latency for canary cohort.
Gradually: 3% → 10% → 25% → 50% → 100%
Rollback: route 0% to v2 if metrics degrade

Pros: real-traffic validation, gradual rollout, minimal blast radius
Cons: two versions live simultaneously, complex request routing
```

**Rolling Deployment:**
```
Start: [v1][v1][v1][v1]
Step1: [v2][v1][v1][v1]  ← 1 upgraded, 3 serving v1
Step2: [v2][v2][v1][v1]  ← 2 upgraded
Step3: [v2][v2][v2][v1]
Step4: [v2][v2][v2][v2]  ← complete

Pros: no extra infrastructure, gradual
Cons: slower rollback (must roll forward or re-deploy), two versions briefly live
```

| Strategy    | Rollback Speed | Infrastructure Cost | Risk Level | DB Schema Changes |
|-------------|----------------|---------------------|------------|-------------------|
| Blue-Green  | Instant        | 2x during deploy    | Low        | Complex           |
| Canary      | Fast           | ~1.03x              | Very Low   | Must be backward  |
| Rolling     | Slow           | 1x                  | Medium     | Must be backward  |

For all strategies, database schema changes must be backward-compatible (add columns only, never rename/drop while old version runs).

---

## Hard (Q16–Q20)

---

### Q16. What are the challenges of multi-region active-active architecture?

Multi-region active-active allows all regions to serve reads and writes simultaneously, offering the lowest latency globally and highest availability. However, it introduces fundamental distributed systems challenges.

**Challenge 1 — Data Conflicts:**
When two users in different regions modify the same record simultaneously, both writes succeed locally and then conflict during replication.

```
Region US writes: user_id=42, balance=100 → 90 (debit $10)
Region EU writes: user_id=42, balance=100 → 80 (debit $20)
After replication: conflict — which write wins?

Solutions:
  - Last-Write-Wins (LWW): use timestamps, risk losing the other write
  - CRDTs (Conflict-free Replicated Data Types): for commutative ops (counters)
  - Application-level conflict resolution (DynamoDB, Cassandra)
  - Avoid conflicts: shard users to regions (user 42 always in EU)
```

**Challenge 2 — Replication Lag:**
Writes in US take 100–150ms to reach EU. Reads in EU immediately after writes in US may see stale data.

```
Replication Lag:
  US write t=0 → EU replication t=150ms
  EU read at t=50ms → sees old value (reads own writes broken)

Mitigations:
  - Read-your-writes consistency: route reads to region where write occurred
  - Sticky sessions: user stays pinned to their home region
  - Fencing tokens or vector clocks to detect stale reads
```

**Challenge 3 — Global DNS TTL:**
```
DNS TTL too high: failover is slow (users cached old IP)
DNS TTL too low: high DNS query volume, propagation issues

Best practice:
  Normal TTL: 60–300 seconds
  Pre-failover: lower TTL to 30 seconds 24h before planned maintenance
  Failover: update record, wait for TTL to expire
```

**Challenge 4 — Global transactions:**
Distributed transactions across regions with 150ms latency make 2PC prohibitively slow. Solutions include saga pattern, eventual consistency, and designing data models to avoid cross-region transactions entirely.

For most systems, user-to-region affinity (shard users to their home region) eliminates 95% of conflicts while still providing geographic redundancy.

---

### Q17. What is graceful degradation and how do you implement it with stale data, read-only mode, and feature flags?

**Graceful degradation** means the system continues serving a reduced but functional experience when a dependency fails, rather than returning errors to users entirely.

**1. Serving Stale Data (Cache Fallback):**
```python
def get_product_price(product_id):
    try:
        # Try live price from DB (3s timeout)
        price = db.get_price(product_id, timeout=3)
        cache.set(f"price:{product_id}", price, ttl=300)
        return price
    except DatabaseTimeout:
        # Fall back to cached price (may be up to 5 min old)
        stale_price = cache.get(f"price:{product_id}")
        if stale_price:
            log.warn("Serving stale price for %s", product_id)
            return stale_price
        raise  # No cache entry, propagate error
```

**2. Read-Only Mode:**
When the primary DB is unavailable but replicas are healthy, switch to read-only mode: allow browsing, searching, viewing — block writes with user-friendly messages.

```
Implementation:
  - Feature flag: READ_ONLY_MODE = true
  - Middleware intercepts write requests: return HTTP 503 with Retry-After header
  - Reads continue from replica
  - User message: "We're experiencing issues. Orders are temporarily disabled."
```

**3. Feature Flags for Degradation:**
```python
# Kill switch for expensive recommendation engine
if feature_flag.is_enabled("recommendations"):
    recs = recommendation_service.get(user_id, timeout=200ms)
else:
    recs = popular_items_cache.get()  # cheap fallback

# Progressive degradation:
#   Level 1: Recommendations from ML model (normal)
#   Level 2: Recommendations from cached popular items (degraded)
#   Level 3: No recommendations section shown (fully degraded)
```

**Degradation hierarchy:**
```
Full service → Stale data → Cached only → Read-only → Static page → Maintenance page
```

Each step should be triggerable via a configuration change without a deployment. The key is that degraded is always better than down — users accept reduced functionality far better than error pages.

---

### Q18. How do you calculate composite availability for systems in series and in parallel?

Understanding how component availability combines is essential for system design — individual components' high availability can mask surprisingly low end-to-end availability.

**Systems in Series (all components must work):**
```
Composite Availability = A1 × A2 × A3 × ... × An

Example: 5 microservices in a request path
  Service A: 99.99% × Service B: 99.99% × Service C: 99.99%
            × Service D: 99.99% × Service E: 99.99%
  = 0.9999^5 = 0.9995 = 99.95%

Even with each service at four-nines, the chain is only three-nines!
```

```
Series system:
  Client → [A 99.99%] → [B 99.99%] → [C 99.99%] → Response
  Any component fails = request fails
  Composite: 0.9999 × 0.9999 × 0.9999 = 99.97%
```

**Systems in Parallel (any component can serve):**
```
Composite Availability = 1 − (1 − A1) × (1 − A2) × ... × (1 − An)

Example: 2 servers in parallel
  Server A: 99% availability → P(failure) = 1%
  Server B: 99% availability → P(failure) = 1%
  Both fail = 1% × 1% = 0.01%
  Composite: 1 − 0.0001 = 99.99%

Two 99% servers in parallel = 99.99% composite!
```

```
Parallel system:
  Client → LB → [A 99%]
              → [B 99%]
  Both must fail simultaneously → 0.01% chance
  Composite: 99.99%
```

**Mixed system example:**
```
[App (2 parallel nodes: 99.99%)] → [DB (99.99%)]

App composite:   1 - (0.0001)^2 ≈ 99.9999%
Series with DB:  0.999999 × 0.9999 ≈ 99.99%

Lesson: DB is the bottleneck — parallelizing app servers only helps if DB is also HA
```

This math reveals why every SPOF destroys end-to-end availability and why eliminating the weakest link is more valuable than over-engineering already-reliable components.

---

### Q19. How do you design HA for stateful services? Cover leader election, session replication, and external session stores.

Stateful services are significantly harder to make highly available than stateless services because state must be managed, replicated, and transferred during failover.

**Leader Election for Stateful Services:**
```
Problem: Distributed workers competing to process the same work.
Solution: Only the leader processes; others are hot standbys.

Using ZooKeeper/etcd ephemeral nodes:
  1. All nodes attempt to create /leader ephemeral node
  2. One succeeds → becomes leader, others watch for node deletion
  3. On leader crash: ephemeral node disappears
  4. Watchers trigger → competing election → new leader in ~seconds

Using Raft (etcd internals):
  - Requires quorum (n/2 + 1) votes to elect leader
  - Leader sends heartbeats; followers become candidates on timeout
  - Split vote → timeout → retry → guaranteed eventual single leader
```

**Session Replication (in-process):**
```
Java EE Session Replication:
  [App Node A]: User session {cart, auth_token}
  [App Node B]: Replica of same session
  
  On Node A failure: Node B already has session, seamless failover
  
  Drawbacks:
  - Multicast overhead for every session update
  - Memory usage doubles (every session on every node)
  - Does not scale beyond ~10 nodes without mesh overhead exploding
```

**External Session Store (recommended at scale):**
```python
# Flask with Redis session store
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = Redis(host='redis-cluster', port=6379)

# User request hits any app node:
# 1. Extract session_id from cookie
# 2. Lookup session in Redis: GET session:{session_id}
# 3. Process request
# 4. Write updated session back to Redis

# App nodes are now fully stateless
# Redis is HA via Redis Sentinel or Redis Cluster
```

```
Architecture with external session store:
  LB → [App Node 1]  ┐
     → [App Node 2]  ├──► Redis Cluster (HA, replicated)
     → [App Node 3]  ┘
  
  Any app node can handle any request
  Session data survives app node failure
  Scale app nodes freely without session concern
```

**JWT for sessionless HA:** For APIs, JWT tokens carry state in the token itself (signed, not encrypted by default). No server-side session store needed — but tokens cannot be invalidated before expiry without a blocklist (which re-introduces a store).

---

### Q20. How do you design a complete HA architecture for a global e-commerce platform? Walk through every layer.

This question tests the ability to synthesize all HA concepts into a coherent system design.

**Requirements:**
- 99.99% availability target (~52 min downtime/year)
- Global users (US, EU, APAC)
- Handles order processing, inventory, user sessions

**Complete Architecture:**

```
Global Layer:
  ┌─────────────────────────────────────────────┐
  │  Anycast DNS (Route 53 latency routing)     │
  │  CDN (CloudFront/Fastly) for static assets  │
  └─────────────┬───────────────────────────────┘
                │
  ┌─────────────▼───────────────────────────────┐
  │  Region: us-east-1    Region: eu-west-1      │
  │  (Active)             (Active)               │
  │  ┌──────────────┐    ┌──────────────┐        │
  │  │AZ-a  AZ-b    │    │AZ-a  AZ-b   │        │
  │  │ALB (HA pair) │    │ALB (HA pair)│        │
  │  │App x6 pods   │    │App x6 pods  │        │
  │  │Redis Cluster │    │Redis Cluster│        │
  │  │Aurora Global │◄──►│Aurora Global│        │
  │  │(RW primary)  │    │(RO replica) │        │
  │  └──────────────┘    └─────────────┘        │
  └─────────────────────────────────────────────┘
```

**Layer-by-layer HA design:**

| Layer          | HA Mechanism                                         | RTO     |
|----------------|------------------------------------------------------|---------|
| DNS            | Route 53 health checks, 30s TTL, latency routing     | 30–60s  |
| CDN            | Multiple PoPs, automatic failover between origins    | Instant |
| Load Balancer  | AWS ALB (managed, multi-AZ by default)               | Instant |
| App Tier       | Kubernetes: 3 replicas per AZ, HPA, PodDisruptionBudget | <30s |
| Cache          | Redis Cluster: 3 shards × 2 replicas, sentinel       | ~10s    |
| Database       | Aurora Global: primary in us-east, replica in eu-west, auto-failover | 30s |
| Storage        | S3 with cross-region replication (CRR)               | N/A     |
| Messaging      | SQS FIFO (managed, multi-AZ, durable)                | Instant |

**Failure scenarios and responses:**

```
Scenario 1: Single app pod crashes
  → Kubernetes restarts pod, readiness probe gates traffic
  → User impact: zero (other pods serve)

Scenario 2: Single AZ failure (AZ-a in us-east)
  → ALB routes to AZ-b pods only
  → Aurora replica in AZ-b promotes automatically
  → User impact: zero (seconds max)

Scenario 3: Full us-east-1 region failure
  → Route 53 health check fails for us-east endpoint
  → DNS switches to eu-west-1 (within TTL: 30–60s)
  → Aurora Global DB promotes eu-west replica (< 1 min)
  → RPO: < 1 second (Aurora Global uses synchronous local, async cross-region)
  → RTO: 1–2 minutes
```

**Operational practices:**
- Quarterly chaos game days: terminate AZ, test regional failover
- All alerts linked to runbooks in Confluence/Notion
- Feature flags for all new features (kill switch ready)
- Blue-green deploys for zero-downtime releases
- Automated backup restore tests weekly

---

## Quick Reference

### Availability Nines
| Nines | Annual Downtime   |
|-------|-------------------|
| 99%   | 3.65 days         |
| 99.9% | 8.76 hours        |
| 99.99%| 52.6 minutes      |
| 99.999%| 5.26 minutes     |

### RPO vs RTO
- **RPO** = max acceptable data loss (backup/replication frequency)
- **RTO** = max acceptable downtime (recovery speed)

### Availability Formulas
- Series:   `A_total = A1 × A2 × A3`
- Parallel: `A_total = 1 − (1−A1)(1−A2)`

### Failover Patterns
| Pattern       | RTO       | Cost  | Complexity |
|---------------|-----------|-------|------------|
| Active-Active | Near zero | High  | High       |
| Active-Passive| 30–120s   | Med   | Low        |
| Cold Standby  | Hours     | Low   | Low        |

### 3-2-1 Backup Rule
- **3** copies, **2** different media, **1** offsite

### Zero-Downtime Deploy Comparison
| Strategy   | Rollback  | Extra Cost |
|------------|-----------|------------|
| Blue-Green | Instant   | 2x infra   |
| Canary     | Fast      | ~1%        |
| Rolling    | Slow      | None       |

### Key Tools
- **Patroni**: PostgreSQL HA with etcd leader election
- **Route 53**: DNS-based health-check failover
- **Chaos Monkey**: Random instance termination (Netflix)
- **STONITH**: Fencing to prevent split-brain
- **Aurora Global**: Multi-region DB with <1s RPO
