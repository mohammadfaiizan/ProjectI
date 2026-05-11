# 26 — Multi-Region and Global Systems

---

## Easy (Q1–Q7)

---

### Q1. Why does multi-region deployment matter? What problems does it solve?

Multi-region deployment addresses three fundamental concerns: **latency**, **compliance**, and **resilience**.

**Latency** — A user in Tokyo hitting a server in Virginia adds ~150 ms of round-trip time before any application logic runs. A regional deployment reduces that to ~5–10 ms.

**Compliance and data sovereignty** — GDPR, PDPA (Thailand), LGPD (Brazil) and dozens of other regulations require that personal data about citizens be stored and processed within specific geographic boundaries. Multi-region architecture makes this structurally enforceable rather than a policy promise.

**Resilience** — A single-region system can be wiped out by a cloud provider's AZ failures, natural disasters, or large-scale network incidents. Netflix's Chaos Kong experiment deliberately took down entire AWS regions to prove their multi-region design was sound.

**Concrete impact:**

| Problem | Single-Region Risk | Multi-Region Solution |
|---|---|---|
| Latency | 150–300 ms for distant users | 5–30 ms via nearest region |
| Regulation | EU data stored in US = GDPR violation | Geo-routing keeps data in EU region |
| Availability | One region = one SPOF | Regional failover within minutes |
| Disaster recovery | RTO hours to days | RTO minutes, RPO near-zero |

Multi-region is not free — it multiplies infrastructure cost and introduces distributed-systems complexity (clock skew, replication lag, partition handling). The decision to go multi-region must be driven by a concrete business case: user geography, regulatory mandate, or an SLA that requires five-nines availability.

---

### Q2. What is the difference between active-passive and active-active multi-region?

These are the two fundamental topologies for multi-region deployment.

**Active-Passive:**
```
         Writes & Reads                 Standby (no traffic)
         ┌──────────────┐               ┌──────────────┐
Users ──▶│  Region A    │──replication─▶│  Region B    │
         │  (PRIMARY)   │               │  (REPLICA)   │
         └──────────────┘               └──────────────┘
         If A fails, DNS is updated to send traffic to B
```

- Region B is kept warm but serves no production traffic until failover.
- Failover requires a DNS TTL change or manual promotion.
- Simpler to reason about — no write conflicts.
- RPO and RTO depend on replication lag and DNS propagation time.

**Active-Active:**
```
         Writes & Reads                  Writes & Reads
         ┌──────────────┐               ┌──────────────┐
Users ──▶│  Region A    │◀─bi-directional▶│  Region B    │
         │  (ACTIVE)    │   replication  │  (ACTIVE)    │
         └──────────────┘               └──────────────┘
         Both regions serve live traffic simultaneously
```

- Both regions accept writes simultaneously — conflict resolution required.
- Higher complexity: concurrent writes to the same record from two regions create conflicts.
- Better utilisation: 100% of capacity in both regions is used.
- No cold-start delay on failover.

**Comparison table:**

| Dimension | Active-Passive | Active-Active |
|---|---|---|
| Complexity | Low | High |
| Cost efficiency | ~50% (passive sits idle) | ~100% utilisation |
| Failover time | Minutes (DNS + warm-up) | Near-instant |
| Write conflicts | None | Must resolve |
| Use case | DR-focused, compliance-critical | Global scale, latency-sensitive |

Most teams start with active-passive and graduate to active-active only when the operational complexity is justified by traffic volume or strict latency SLAs.

---

### Q3. What is data sovereignty and why must EU user data stay in the EU?

**Data sovereignty** means that data is subject to the laws of the country in which it resides — regardless of where the company owning it is headquartered.

**GDPR (General Data Protection Regulation)** — effective since May 2018 — establishes that:

1. Personal data about EU residents must not be transferred outside the EU/EEA unless the destination country provides "adequate protection" or a specific transfer mechanism (Standard Contractual Clauses, Binding Corporate Rules) is in place.
2. A company in California that stores EU user emails on a US-only server is in breach of GDPR — even if it never "transfers" data intentionally.
3. Fines can reach €20 million or 4% of annual global turnover, whichever is higher.

**Architectural implication:**

```
GeoDNS / Global Load Balancer
        │
        ├── EU users ──▶ eu-west-1 (Ireland) ─── EU-only DB cluster
        │                                          (no cross-region replication to US)
        │
        └── US users ──▶ us-east-1 (Virginia) ── US-only DB cluster
```

User identity must be established **before** routing — a common pattern is a lightweight global user-lookup service that only stores a user's home region, then redirects the client to that region for all further requests.

**Other regulations that impose similar constraints:**

| Regulation | Jurisdiction | Data type |
|---|---|---|
| GDPR | EU / EEA | All personal data |
| PDPA | Thailand | Personal data of Thai residents |
| LGPD | Brazil | Personal data of Brazilians |
| China PIPL | China | Personal information of Chinese citizens |
| India DPDP | India | Personal data of Indian residents |

Compliance is not optional — it is a hard constraint that must be designed into the data tier before launch.

---

### Q4. How does GeoDNS work, and how does it differ from Anycast?

Both are techniques to route users to the nearest or best-performing server, but they operate at different protocol layers.

**GeoDNS:**
- Operates at the **DNS application layer**.
- When a resolver queries for `api.example.com`, the authoritative DNS server looks up the resolver's IP address against a geo-IP database.
- Returns a different A/AAAA record depending on the detected geography.
- Example: a resolver in Frankfurt gets `1.2.3.4` (EU endpoint); a resolver in Singapore gets `5.6.7.8` (APAC endpoint).

```
Client in Germany
       │  DNS query: api.example.com
       ▼
DNS Resolver (Deutsche Telekom)
       │  Authoritative DNS sees resolver IP → Germany
       ▼
Returns: 1.2.3.4 (eu-west endpoint)
```

**Anycast:**
- Operates at the **IP routing layer (BGP)**.
- Multiple servers in different data centres advertise the **same IP address** into BGP.
- Routers naturally direct packets to the topologically closest server.
- No DNS involvement — routing is transparent to the client.
- Used by Cloudflare, Fastly, and root DNS servers (e.g., `1.1.1.1` is anycast globally).

**Comparison:**

| Feature | GeoDNS | Anycast |
|---|---|---|
| Layer | DNS (L7) | IP/BGP (L3) |
| Failover speed | Limited by DNS TTL (seconds–minutes) | Sub-second (BGP reconvergence) |
| Granularity | Region / country | AS-level (ISP-level proximity) |
| DDoS absorption | Limited | Excellent — attack traffic distributed globally |
| Complexity | DNS provider config | BGP peering with ISPs |

Cloudflare uses anycast for all its edge nodes — when a DDoS attack sends 1 Tbps of traffic at `104.16.x.x`, that traffic is absorbed by hundreds of PoPs simultaneously instead of overwhelming one location.

---

### Q5. What is the trade-off between short and long DNS TTL for multi-region failover?

DNS TTL (Time-To-Live) controls how long resolvers cache a DNS record before re-querying the authoritative server. Choosing the right TTL is a balancing act between failover speed and system load.

**Short TTL (e.g., 30–60 seconds):**
- Advantage: When a region goes down, you update the DNS record and most clients re-resolve within 60 seconds → fast failover.
- Disadvantage:
  - Every client must re-query DNS frequently → increased load on authoritative DNS servers.
  - Some resolvers ignore TTL and have minimum floors (often 30–60 s regardless).
  - More DNS queries = higher DNS cost at scale.
  - Increased DNS lookup latency per user request at high cache-miss rates.

**Long TTL (e.g., 3600 seconds = 1 hour):**
- Advantage: DNS resolvers cache the record for an hour → fewer queries, faster resolution for users.
- Disadvantage: If a region fails, clients continue hitting the dead IP for up to an hour before the cache expires.

**Recommended strategy:**
```
Normal operations:  TTL = 300–3600 s  (standard caching benefits)
Pre-failover warning: TTL = 60 s      (reduce proactively before maintenance)
During failover:    Update record
Post-failover:      TTL = 300 s       (restore after confirming new region stable)
```

AWS Route 53 and Cloudflare support **health-check-driven failover** — they automatically swap records when a health check fails, but TTL still governs how quickly stale records expire in downstream resolvers. The practical rule: **set TTL to 60 seconds for any record that could be involved in failover**. Accept the higher DNS query volume as the cost of fast recovery.

---

### Q6. What can and cannot be kept globally consistent in a distributed system?

The **CAP theorem** states that a distributed system can provide at most two of Consistency, Availability, and Partition tolerance simultaneously. In a multi-region setup, network partitions between regions are not hypothetical — they are routine (cross-region links fail several times a year). This means you must choose between **consistency** and **availability** when a partition occurs.

**What CAN be globally consistent (with trade-offs):**
- **Counters and monotonic values** — if you are willing to pay for synchronous cross-region writes (100–300 ms latency).
- **Configuration data** — changes infrequently; can use Paxos/Raft with quorum across regions.
- **Financial balances** — can be made consistent via two-phase commit or Spanner-style TrueTime, but at high latency cost.
- **Unique usernames** — consistent reservation via distributed lock or serializable transaction, accepting higher write latency.

**What CANNOT be globally consistent without prohibitive cost:**
- **Real-time inventory** at high write throughput — the conflict rate and latency would be unacceptable.
- **Social media feeds** — eventual consistency is acceptable and expected.
- **Session state** — consistency here provides no business value; sticky sessions or eventual sync is fine.
- **Metrics and analytics** — approximate counts are acceptable.

**The PACELC model** extends CAP by noting that even when there is no partition (E), you must still choose between Latency and Consistency (LC):

```
                     Partition?
                    /          \
                Yes              No
               /                  \
         Availability          Latency
         vs Consistency        vs Consistency
```

The practical answer: **define consistency requirements per data type**. Most data in a global system can be eventually consistent. Reserve strong consistency for financial and identity data, and pay the latency cost only where the business requires it.

---

### Q7. What is a CRDT? Explain the G-Counter and LWW-Register with examples.

A **CRDT (Conflict-free Replicated Data Type)** is a data structure designed so that replicas can be updated independently and merged without conflicts — the merge is always deterministic regardless of the order updates arrive.

CRDTs eliminate the need for coordination or conflict resolution logic. They are used in systems like Riak, Redis Cluster, and collaborative editors (Google Docs, Figma).

**G-Counter (Grow-only Counter):**
- Each replica maintains a vector of counts, one slot per replica.
- A replica only increments its own slot.
- Merge = take the max of each slot.
- Total value = sum of all slots.

```
Replica A increments: [A=3, B=0, C=0]
Replica B increments: [A=0, B=2, C=0]
Replica C increments: [A=0, B=0, C=5]

After merge (element-wise max):
[A=3, B=2, C=5] → total = 10
```

No conflicts possible — each replica only owns its own slot.

**LWW-Register (Last-Write-Wins Register):**
- Stores a single value with a timestamp.
- On conflict (two replicas have different values), the one with the higher timestamp wins.
- Requires reasonably synchronised clocks (or logical clocks).

```
Replica A: {value: "Alice", ts: 1000}
Replica B: {value: "Bob",   ts: 1001}

After merge: "Bob" wins (higher timestamp)
```

**Other common CRDTs:**

| CRDT | Description | Use case |
|---|---|---|
| G-Counter | Grow-only counter | Page views, like counts |
| PN-Counter | Positive-negative counter | Inventory deltas |
| OR-Set | Add/remove set without conflicts | Shopping cart |
| LWW-Register | Last-write-wins value | User profile field |
| MV-Register | Multi-value (keeps all concurrent) | Collaborative text |

The trade-off: CRDTs can only model data that fits their merge semantics. Not all business logic maps cleanly to a CRDT — but where it does, they provide conflict-free replication with zero coordination overhead.

---

## Medium (Q8–Q15)

---

### Q8. How do DynamoDB Global Tables and Cassandra multi-DC replication handle conflicts?

Both systems support multi-region writes but use different conflict resolution strategies.

**DynamoDB Global Tables:**
- Uses **Last-Writer-Wins (LWW)** based on wall-clock time.
- Each write is timestamped; if two regions write to the same item concurrently, the higher timestamp wins.
- Replication is asynchronous — replicas are eventually consistent.
- Global Tables use a **replication ring** topology where each region is both a source and a destination.

```
us-east-1 ◀──────────────▶ eu-west-1
     ▲                           ▲
     │                           │
     └──────── ap-southeast-1 ───┘
        (All three replicate to each other)
```

- **Conditional writes** are supported within a single region but NOT across regions atomically.
- For financial data, the recommendation is to use a **home region pattern**: route all writes for a user to their designated home region; other regions are read-only replicas for that user.

**Cassandra multi-DC replication:**
- Uses a configurable **replication factor per datacenter**.
- Conflict resolution is also LWW using a hybrid logical clock.
- `NetworkTopologyStrategy` allows specifying replication factor per DC:

```cql
CREATE KEYSPACE myapp WITH replication = {
  'class': 'NetworkTopologyStrategy',
  'us-east': 3,
  'eu-west': 3,
  'ap-south': 2
};
```

- **LOCAL_QUORUM** consistency ensures reads/writes reach quorum within the local DC — no cross-region latency for normal operations.
- **EACH_QUORUM** requires quorum in every DC — used for strong global consistency but incurs cross-region latency.

**Key difference:** DynamoDB Global Tables is fully managed with automatic conflict resolution; Cassandra gives operators more control over consistency levels per operation, making it more flexible but operationally heavier.

---

### Q9. Explain the read-local / write-global pattern and when to use it.

The **read-local / write-global pattern** allows reads to be served from the nearest replica (low latency) while writes are directed to a designated primary region (strong consistency), then asynchronously replicated to all replicas.

**Architecture:**

```
                    ┌─────────────────────────────────────┐
                    │         Global Primary (us-east-1)   │
                    │  All writes land here → replicated   │
                    └───────────┬──────────────────────────┘
                                │ async replication
                ┌───────────────┼────────────────┐
                ▼               ▼                ▼
         eu-west-1         ap-south-1      ap-southeast-1
         (read replica)   (read replica)  (read replica)
              ▲                ▲                ▲
         EU users         India users      SEA users
         (reads only)     (reads only)     (reads only)
```

**Read path:** User reads their profile → nearest replica → ~5–10 ms.
**Write path:** User updates their profile → routed to primary → ~150–250 ms → replicated to all replicas.

**When this pattern is appropriate:**
- Read-heavy workloads (social profiles, product catalogues, content).
- Data that can tolerate **replication lag** on reads (seconds to milliseconds).
- Workloads where write latency of 150–250 ms is acceptable.

**When it is NOT appropriate:**
- Financial transactions where reading stale data is dangerous.
- Systems where a write must be immediately visible to all concurrent readers.

**Implementation tips:**
- After a write, redirect the user's subsequent read to the primary region for a short window (e.g., 5 seconds) to avoid reading their own stale data — this is called **read-your-writes consistency**.
- Use conditional routing: `if (request is write) → primary; else → nearest replica`.

This pattern is used by many large-scale systems: GitHub (primary + read replicas), Twitter profiles, and most e-commerce product catalogues.

---

### Q10. What is the latency impact of cross-region synchronous writes, and how should you design around it?

Cross-region synchronous writes require a round-trip between the application server and the database in another region before the write can be acknowledged. Typical figures:

| Region Pair | Round-Trip Latency |
|---|---|
| us-east-1 ↔ us-west-2 | ~65 ms |
| us-east-1 ↔ eu-west-1 | ~80–100 ms |
| us-east-1 ↔ ap-southeast-1 | ~170–200 ms |
| eu-west-1 ↔ ap-southeast-1 | ~130–160 ms |

For a transaction that requires writes to two regions (e.g., two-phase commit), the latency is **at least one round-trip time** — making synchronous multi-region writes unsuitable for user-facing APIs with < 100 ms SLAs.

**Design strategies to avoid paying this cost:**

1. **Async replication with primary write** — write to one region, replicate async. Accept eventual consistency on replicas.

2. **Geo-partitioned data** — EU users' data lives entirely in eu-west-1; US users' data in us-east-1. No cross-region writes needed for normal operations.

3. **Event-driven replication** — write to local region, emit an event (Kafka / SQS), a consumer in each remote region applies the change asynchronously.

4. **CRDT-based data structures** — writes are always local, merges happen asynchronously. No need for coordination.

5. **Accept write latency for low-frequency operations** — if a user changes their billing address once a quarter, 200 ms is acceptable. Design for it explicitly rather than avoiding it entirely.

**Rule of thumb:** Every 10 ms of latency added to a checkout flow reduces conversion rate by ~1% (Amazon research). Cross-region synchronous writes on the critical path will measurably hurt business metrics. Always make them asynchronous unless consistency is a hard business requirement.

---

### Q11. How does Cloudflare use Anycast for DDoS mitigation and global latency reduction?

Cloudflare operates ~300 Points of Presence (PoPs) globally, all advertising the **same IP address prefixes** via BGP Anycast. This design provides two critical benefits: latency reduction and DDoS absorption.

**DDoS Mitigation via Anycast:**

```
Attacker (1 Tbps attack on 104.16.0.1)
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
  PoP: London  PoP: NYC    PoP: Tokyo
  (absorbs     (absorbs    (absorbs
  300 Gbps)    400 Gbps)   300 Gbps)
```

Because every PoP advertises the same IP, attack traffic is automatically distributed across all PoPs by BGP routing. Each PoP only sees a fraction of the total attack volume. A 1 Tbps attack is trivially handled if spread across 300 PoPs (average ~3.3 Gbps each).

Single-IP (unicast) architectures are vulnerable because all attack traffic hits one server. Anycast eliminates this single point of failure.

**Latency Reduction:**
- BGP naturally routes each user's packets to the **topologically nearest Cloudflare PoP**.
- TLS termination, caching, and WAF processing happen at that PoP — milliseconds from the user.
- Only cache misses or origin requests travel the full distance to the origin server.
- Typical improvement: 40–60% reduction in TTFB (Time to First Byte) for globally distributed users.

**How Cloudflare handles origin routing after anycast:**
- Edge PoP receives request → checks cache.
- Cache miss → Cloudflare's **Argo Smart Routing** uses private backbone tunnels between PoPs to reach the origin faster than the public Internet.
- Avoids congested public Internet paths, further reducing latency by 30% on average.

This architecture is why Cloudflare can simultaneously offer DDoS protection, CDN, and DNS at internet scale — all from a single anycast IP space.

---

### Q12. What is Google Spanner's TrueTime, and how does it achieve external consistency globally?

Most distributed databases use Lamport clocks or hybrid logical clocks for ordering events because physical clocks on different machines can differ by milliseconds. Google Spanner takes a different approach: it uses **TrueTime**, a globally synchronised clock with a bounded uncertainty interval, to achieve **external consistency** (the strongest form of linearizability) across globally distributed shards.

**TrueTime API:**
```
TT.now()   → returns interval [earliest, latest]
             the true current time lies somewhere in this interval
TT.after(t) → true if t has definitely passed
TT.before(t)→ true if t has definitely not passed
```

The key insight: Spanner does not know the exact current time — but it knows the **uncertainty bound** (typically 1–7 ms, using GPS receivers and atomic clocks in each data centre).

**Commit Wait mechanism:**
1. Spanner assigns a commit timestamp `s` = `TT.now().latest` (pessimistic — uses the upper bound).
2. Before committing, Spanner **waits** until `TT.after(s)` is true — i.e., the clock has definitely advanced past `s`.
3. This ensures no future transaction can receive a timestamp ≤ `s`.

```
Timeline:
  TT.now() = [1000, 1007]  ← 7ms uncertainty
  Assign s = 1007
  Wait until TT.now().earliest > 1007  ← commit wait (~7ms)
  Commit transaction with timestamp 1007
```

**Result:** Any transaction that starts after this commit will observe a timestamp > 1007 and will see this transaction's writes. External consistency is guaranteed — the observed order of transactions matches real-world time.

**Cost:** Commit wait adds 1–7 ms latency. For cross-region transactions, network latency (100–300 ms) dominates, making commit wait negligible. For single-region transactions, it is the dominant latency factor.

Spanner is used for Google's F1 (AdWords database) and Google Payments — systems where global consistency is a hard requirement.

---

### Q13. How does geographic data partitioning (sharding by region) work? What are its limitations?

**Geographic partitioning** assigns each user's data to a shard based on their region, ensuring that data access patterns are always local to that region. It is the most effective way to meet both latency and data sovereignty requirements simultaneously.

**Implementation:**

```
User signs up → Region determined by IP/preference → user_id prefixed with region code

user_id: eu-550e8400-e29b-41d4-a716 → stored in EU shard
user_id: us-3f5e4ab0-c2d3-4e8f-b912 → stored in US shard
user_id: ap-9a8b7c6d-5e4f-3a2b-1c0d → stored in APAC shard
```

**Routing layer:**

```
Client Request
      │
      ▼
Global Router (reads region prefix from user_id or JWT claim)
      │
      ├── eu-* ──▶ EU Database Cluster (eu-west-1)
      ├── us-* ──▶ US Database Cluster (us-east-1)
      └── ap-* ──▶ APAC Database Cluster (ap-southeast-1)
```

**Advantages:**
- Reads and writes are always local → single-digit millisecond database latency.
- Data sovereignty enforced structurally — EU data physically cannot end up in US shard.
- Failure in one region does not affect other regions' data.

**Limitations:**

| Limitation | Description |
|---|---|
| Cross-region queries | Aggregating data across all users globally requires scatter-gather across all shards |
| User migration | Moving a user between regions requires cross-region data copy + atomic rename |
| Hotspots | If APAC user base grows 10x, APAC shard becomes a bottleneck |
| Analytics | Global analytics must query all shards and merge results |
| Friend relationships | User A (EU) and User B (US) — their friendship data must live somewhere |

The cross-entity relationship problem is the hardest: if a EU user sends a message to a US user, which shard stores the message? Common solutions: store the message in the sender's shard with a pointer in the recipient's shard, or use a separate cross-region messaging service.

---

### Q14. What are NTP limitations for clock synchronisation in distributed systems?

**NTP (Network Time Protocol)** is the standard protocol for synchronising clocks across networked machines. It is widely used but has significant limitations that affect distributed systems design.

**How NTP works:**
1. Client sends request to NTP server with timestamp T1.
2. Server receives at T2, responds at T3.
3. Client receives response at T4.
4. Offset estimated as: `((T2-T1) + (T3-T4)) / 2`
5. Assumes symmetric network delay — often not true.

**Limitations:**

| Limitation | Impact |
|---|---|
| Accuracy | ~1–10 ms on LAN; 10–100 ms over WAN | |
| Asymmetric routing | Different latency in each direction → incorrect offset calculation |
| Leap seconds | NTP handles leap seconds inconsistently; some systems smear over 24 hours |
| VM clock drift | VMs can drift significantly when paused/migrated; NTP corrections cause jumps |
| Security | NTP amplification attacks; spoofing without NTPsec |
| Stepwise adjustments | ntpd can step the clock backward — dangerous for timestamps used as version numbers |

**Real-world incidents caused by clock skew:**
- A 10 ms clock difference between two database replicas can cause the "wrong" write to win in LWW conflict resolution.
- Kafka producers relying on wall-clock timestamps for ordering can produce incorrect orderings when clocks drift.
- Certificate validation failures when system clocks are significantly off.

**Better alternatives:**
- **Hybrid Logical Clocks (HLC)** — combine physical time with logical clock to provide causal ordering without requiring tight synchronisation.
- **Google TrueTime** — GPS + atomic clocks with bounded uncertainty (see Q12).
- **Amazon Time Sync Service** — uses a fleet of dedicated time servers with < 1 ms accuracy, available from within EC2 instances.

The practical rule: **never use wall-clock timestamps as the sole basis for ordering events in a distributed system**. Use vector clocks, HLC, or Lamport timestamps for causal ordering.

---

### Q15. How does MirrorMaker 2 replicate Kafka across regions?

**MirrorMaker 2 (MM2)** is Kafka's built-in cross-cluster replication tool, released with Kafka 2.4. It is built on the Kafka Connect framework and significantly improves on the original MirrorMaker 1.

**Architecture:**

```
Source Cluster (us-east-1)        Target Cluster (eu-west-1)
┌─────────────────────────┐       ┌─────────────────────────┐
│  Topic: orders          │       │  Topic: us-east-1.orders │
│  Topic: payments        │──MM2──▶  Topic: us-east-1.payments│
│  Topic: users           │       │  Topic: us-east-1.users  │
└─────────────────────────┘       └─────────────────────────┘
```

Key features:
1. **Topic renaming** — replicated topics are prefixed with the source cluster alias (`us-east-1.orders`) to avoid naming conflicts in active-active setups.
2. **Offset synchronisation** — MM2 maintains a mapping between source and target offsets via the `mm2-offsets` internal topic. Consumer groups can be migrated between clusters without message loss or reduplication.
3. **Bidirectional replication** — supports active-active replication between two clusters with cycle detection to prevent infinite replication loops.
4. **Heartbeat topics** — publishes heartbeat records to source and target to enable connectivity monitoring and offset translation.

**Configuration example:**
```properties
clusters = us-east, eu-west
us-east.bootstrap.servers = broker1.us-east:9092
eu-west.bootstrap.servers = broker1.eu-west:9092

# Replicate from us-east to eu-west
us-east->eu-west.enabled = true
us-east->eu-west.topics = orders, payments, users

# Replication factor on target
replication.factor = 3
```

**Limitations:**
- Replication is asynchronous — EU cluster will lag behind US cluster (seconds to minutes).
- No cross-cluster transactions — a producer cannot atomically write to both clusters.
- MM2 itself is a Kafka Connect cluster and must be sized and monitored separately.

For disaster recovery, MM2 enables RPO of seconds to minutes depending on replication lag. For zero-RPO requirements, synchronous cross-region writes (with their latency cost) are necessary.

---

## Hard (Q16–Q20)

---

### Q16. Design a global CDN architecture that handles both static and dynamic content.

A modern CDN must serve static assets (images, JS, CSS) at high cache hit rates while also accelerating dynamic, personalised content that cannot be cached in the traditional sense.

**Two-tier CDN Architecture:**

```
                        Users Globally
                              │
                    ┌─────────▼──────────┐
                    │  Edge PoPs (300+)   │  ← Anycast IP routing
                    │  Cache, TLS, WAF    │
                    └─────────┬──────────┘
                              │ Cache miss or dynamic
                    ┌─────────▼──────────┐
                    │  Regional Shield    │  ← 5–10 per continent
                    │  (Mid-tier cache)   │     collapses requests
                    └─────────┬──────────┘
                              │ Cache miss
                    ┌─────────▼──────────┐
                    │  Origin Servers     │  ← Protected from direct
                    │  (2–3 regions)      │     user traffic
                    └─────────────────────┘
```

**Static content handling:**
- Edge caches serve images, JS bundles, CSS with long TTLs (1 year + cache-busted by file hash in URL).
- Cache hit ratio target: > 95%.
- Origin fetch: only on first request per PoP or after invalidation.
- `Cache-Control: public, max-age=31536000, immutable` for hashed assets.

**Dynamic content acceleration:**
- Personalised pages cannot be cached by edge nodes directly.
- Strategies:
  1. **ESI (Edge Side Includes)** — decompose page into cacheable fragments (header, footer, navigation) and dynamic fragments (personalised section). Cache the static parts at the edge, fetch only the personalised fragment from origin.
  2. **Stale-While-Revalidate** — serve stale content immediately, trigger async refresh. Acceptable for non-critical personalisation.
  3. **Edge compute** (Cloudflare Workers, Lambda@Edge) — run lightweight personalisation logic at the edge using user JWT claims. No origin required for many personalisation decisions.
  4. **Protocol acceleration** — even for fully dynamic content, terminate TLS at edge and use HTTP/2 or HTTP/3 multiplexing over persistent connections to origin via Cloudflare Argo or Fastly's private backbone.

**Cache invalidation strategy:**
```
Deployment pipeline:
  1. New asset deployed → hash changes → new URL → automatic cache bust
  2. Content update (blog post) → API call to CDN purge API → targeted purge by URL
  3. Wildcard purge: purge /api/products/* on product catalogue update
  4. Surrogate keys (Fastly/Varnish) → tag all responses with entity ID → purge by tag
```

**Security layer at edge:**
- WAF rules evaluated at edge → malicious traffic dropped before reaching origin.
- Bot management, rate limiting, and DDoS mitigation at edge PoP.
- Cost saving: 95% of attack traffic never reaches origin.

---

### Q17. How do you test multi-region failover? What must you validate before declaring DR ready?

Declaring DR readiness requires **evidence from controlled failure testing**, not just documentation that failover is theoretically possible. Netflix's Chaos Kong is the gold standard — they intentionally route all traffic away from one AWS region to prove their system survives.

**What to test and validate:**

**1. RTO (Recovery Time Objective) measurement:**
```
Test: Simulate primary region failure
Measure: Time from failure detection to full traffic serving by secondary region
Target: Must be < stated RTO (e.g., < 5 minutes)
```

**2. RPO (Recovery Point Objective) measurement:**
```
Test: Inject 100 writes to primary region, kill primary
Measure: How many of those writes are visible in secondary after failover
Target: Must be < stated RPO (e.g., < 30 seconds of data loss)
```

**3. DNS failover timing:**
```
Steps:
  1. Health check marks primary region DOWN
  2. Route 53 / NS1 switches DNS record to secondary
  3. Clients with cached DNS re-resolve after TTL
Validate: All clients routing to secondary within TTL window
```

**4. Database state validation:**
```
Checks:
  - Replication lag at time of failover (actual RPO)
  - No data corruption in replica
  - Read-write promotion of replica completes correctly
  - Connection string updates propagated to application tier
```

**5. Dependent service validation:**

| Service | What to verify |
|---|---|
| Message queues | No messages lost; consumers reconnect to secondary |
| Cache | Cold cache in secondary doesn't cause DB overload (cache stampede) |
| Authentication | Auth service works independently in secondary region |
| Third-party integrations | Webhook URLs, OAuth callbacks still function |
| Certificates | TLS certificates valid in secondary region |

**6. Runbook execution test:**
- A human follows the DR runbook end-to-end during the test.
- Every step must be executable by an on-call engineer with no prior DR experience.
- Runbook that requires 15 manual steps with complex SQL commands is not DR-ready.

**7. Game Day methodology:**
- Schedule quarterly chaos experiments.
- Inject failures during business hours (not at 3am when nobody is watching).
- Measure MTTR, not just whether recovery eventually succeeded.
- Document gaps and remediate before the next exercise.

**Definition of DR ready:** Failover can be executed within the stated RTO, data loss is within stated RPO, and a non-expert can execute the runbook without escalation.

---

### Q18. Design an API that works offline and syncs when connectivity returns.

**Offline-first architecture** is required for mobile applications and progressive web apps that must function in low/no connectivity environments. Examples: Google Docs offline mode, Figma offline, airline check-in apps.

**Core components:**

```
┌──────────────────────────────────────────────────────┐
│                  Client (mobile/PWA)                  │
│                                                        │
│  ┌──────────────┐    ┌───────────────────────────┐   │
│  │  Local DB    │    │  Sync Engine               │   │
│  │  (SQLite /   │◀───│  - Tracks pending changes  │   │
│  │  IndexedDB)  │    │  - Detects connectivity     │   │
│  └──────────────┘    │  - Conflict resolution      │   │
│                       └──────────┬────────────────┘   │
└──────────────────────────────────┼─────────────────────┘
                                   │ HTTP / WebSocket
                         ┌─────────▼────────────┐
                         │  Sync API (cloud)     │
                         │  - Change log table   │
                         │  - Conflict detection │
                         └──────────────────────┘
```

**Change tracking — client side:**
- Every local mutation is appended to a **pending changes log** with:
  - `change_id` (UUID)
  - `entity_type` (e.g., "note")
  - `entity_id`
  - `operation` (CREATE / UPDATE / DELETE)
  - `payload` (the delta or full new state)
  - `client_timestamp`
  - `sync_status` (PENDING / SYNCED / CONFLICT)

**Sync protocol:**

```
1. Client comes online
2. Client sends all PENDING changes to /sync endpoint
3. Server applies changes with conflict detection
4. Server returns:
   - List of server changes since client's last_sync_token
   - Conflict resolutions (if any writes conflicted)
5. Client applies server changes to local DB
6. Client updates last_sync_token
7. Conflicted items presented to user for manual resolution (or auto-resolved by LWW)
```

**Conflict resolution strategies:**

| Strategy | Description | Use case |
|---|---|---|
| Last-Write-Wins | Higher timestamp wins | Profile fields |
| Server-Wins | Server state always authoritative | Configuration |
| Client-Wins | Client change always applied | Offline notes |
| Three-way merge | Diff against common ancestor | Collaborative text |
| Present to user | User resolves manually | Calendar events |

**Idempotency for sync:**
- Each change has a stable `change_id`.
- If the client retries the sync (connectivity dropped mid-sync), duplicate `change_id` values are ignored by the server (`INSERT ON CONFLICT DO NOTHING`).

**Progressive sync for large datasets:**
- Do not sync entire DB on reconnect — use **cursor-based delta sync**:
  ```
  GET /sync?since={last_sync_token}&limit=100
  ```
- Server returns changes in batches; client advances cursor after each batch.

**Real-world implementations:** Apple CloudKit, Firebase Firestore offline mode, CouchDB/PouchDB sync protocol, AWS AppSync with Amplify DataStore.

---

### Q19. How do you design for partial vs total network partition in a multi-region system?

A network partition in a multi-region system is not binary — it can range from **one region losing connectivity to one other region** to **all regions being fully isolated** from each other. Your design must handle both gracefully.

**Taxonomy of partition types:**

```
Type 1: Single-link failure
  us-east-1 ←───✗───→ eu-west-1
  us-east-1 ←────────→ ap-southeast-1   ← still connected
  eu-west-1 ←────────→ ap-southeast-1   ← still connected

Type 2: Region fully isolated
  us-east-1 ←───✗───→ eu-west-1
  us-east-1 ←───✗───→ ap-southeast-1
  us-east-1 is fully partitioned from the rest

Type 3: Total partition (split-brain)
  All regions isolated from each other simultaneously
```

**Design decisions per partition type:**

**Type 1 — Single link failure:**
- Quorum is still achievable across the 3 regions (2 of 3 still connected).
- Continue writes with quorum of 2 regions.
- Replication to the isolated link will resume when link restores.
- **No data divergence** if using quorum writes.

**Type 2 — Region fully isolated:**
- The isolated region should detect partition (health checks to peers failing) and enter a **degraded mode**:
  - Accept reads from its local replicas (potentially stale by the partition duration).
  - Reject writes (to avoid data divergence) OR accept writes locally and queue them for resync.
  - Which choice depends on your CAP trade-off: availability vs consistency.

**Type 3 — Total partition:**
```
Each region must independently decide: do I continue serving traffic?

Option A (CP): All regions stop accepting writes until quorum restored
  → Data consistency guaranteed, but system unavailable
  
Option B (AP): Each region continues independently
  → System remains available, but writes in different regions may conflict
  → Must resolve conflicts on reconnection
```

**Practical implementation pattern:**
```
Per-region state machine:
  NORMAL: serving reads and writes, all peers healthy
  DEGRADED: some peers unreachable, serving reads only
  ISOLATED: all peers unreachable, serving reads with stale warning
  RECOVERING: peers restored, replaying queued operations

Transition trigger: health checks to peer regions via dedicated monitoring link
```

**Split-brain prevention:**
- Use a **fencing token** from an external authority (e.g., a ZooKeeper or etcd cluster that itself has quorum) to determine which region is authoritative during partition.
- The region that cannot communicate with the quorum service must step down as primary.

The key insight: **design for graceful degradation, not binary availability**. A system that serves reads (possibly stale) while rejecting writes during partition is more useful than one that goes fully down.

---

### Q20. Case study: How does Netflix design for regional failure? What is the Chaos Kong experiment?

Netflix is the canonical example of a company that treats regional failure as a routine operational event rather than a rare catastrophe. Their resilience strategy is built around three pillars: **active-active multi-region**, **chaos engineering**, and **stateless compute**.

**Multi-Region Architecture:**

```
┌──────────────────────────────────────────────────────┐
│                   Route 53 (GeoDNS)                   │
│              + health check routing                   │
└─────────┬─────────────────────────────┬──────────────┘
          │                             │
┌─────────▼──────────┐     ┌────────────▼──────────────┐
│   us-east-1        │     │   eu-west-1                │
│   (Primary)        │     │   (Active)                 │
│                    │     │                            │
│  Zuul (API GW)     │     │  Zuul (API GW)            │
│  Microservices     │◀───▶│  Microservices             │
│  Cassandra (3 AZ)  │     │  Cassandra (3 AZ)          │
│  EVCache (memcache)│     │  EVCache                   │
└────────────────────┘     └────────────────────────────┘
        ▲                              ▲
        └──── Data replicated ─────────┘
              via Cassandra multi-DC
```

**Key architectural decisions:**
1. **Stateless services** — no application-level state stored in memory. All state in Cassandra or EVCache. Any instance can be killed without loss.
2. **Cassandra multi-DC replication** — each region has a full replica of user data. EU region can serve EU users independently even if US region is completely down.
3. **EVCache (memcached ring)** — caches user recommendation data regionally. Cold cache on failover causes temporary latency spike, not unavailability.
4. **Hystrix (circuit breakers)** — services degrade gracefully when dependencies are unavailable rather than cascading failure.

**Chaos Kong:**
- A **Chaos Kong** experiment deliberately evacuates all production traffic from one AWS region.
- Named after Chaos Monkey (kills single instances) — Chaos Kong kills an entire region.
- Process:
  1. Team pre-announces experiment date (not a surprise).
  2. Route 53 health checks for `us-east-1` are manually set to FAILING.
  3. All traffic re-routes to `us-west-2` and `eu-west-1`.
  4. Team observes: error rates, latency, cache hit ratios, and any services that fail to handle the load shift.
  5. Post-experiment: restore `us-east-1` to healthy and observe graceful traffic restoration.
- **What they validate:**
  - Remaining regions can absorb 100% of traffic without degradation.
  - No services have hardcoded `us-east-1` endpoints.
  - Monitoring correctly identifies the source of any anomalies.
  - Runbook steps are accurate and executable in the actual failure scenario.
  - Cache warm-up behaviour under load.

**Lessons Netflix has published:**
- Active-active is not a network topology decision — it is an application architecture decision. Every service must be designed to run in any region.
- Chaos experiments reveal integration failures that no amount of architecture review or staging tests will catch.
- The goal is not zero failures. The goal is **failing safely**: users experience a brief degradation (buffering, lower video quality) rather than a hard outage.

**Metrics Netflix targets:**
- RTO: < 5 minutes for full regional evacuation.
- RPO: Near-zero (Cassandra async replication lag = seconds).
- Availability: 99.99% (< 53 minutes downtime/year).

---

## Quick Reference

| Topic | Key Point |
|---|---|
| Multi-region motivation | Latency (< 30 ms local vs 150+ ms remote), compliance, resilience |
| Active-passive | Simple, idle standby, failover in minutes; cold start risk |
| Active-active | Both regions live, higher cost, conflict resolution required |
| Data sovereignty | GDPR requires EU data to stay in EU; geo-partition by user region |
| GeoDNS | Routes via DNS based on resolver IP; TTL limits failover speed |
| Anycast | Same IP advertised globally via BGP; instant failover, DDoS distribution |
| DNS TTL strategy | 60 s TTL for failover-critical records; pre-reduce before planned failover |
| CRDT G-Counter | Vector of per-replica counts; merge = element-wise max; no conflicts |
| CRDT LWW-Register | Last timestamp wins; requires reasonably synchronised clocks |
| DynamoDB Global Tables | Async multi-region replication; LWW conflict resolution |
| Cassandra multi-DC | LOCAL_QUORUM for low latency; EACH_QUORUM for global consistency |
| Read-local/write-global | Reads from nearest replica; writes to primary; eventual consistency |
| Cross-region write latency | 65–300 ms per round trip; avoid on critical user-facing paths |
| Cloudflare Anycast | 300+ PoPs absorb DDoS traffic; TLS terminates at edge |
| Google Spanner TrueTime | GPS+atomic clock; bounded uncertainty 1–7 ms; commit wait guarantees ordering |
| Geo-partitioning | Shard by user region; local reads/writes; cross-region queries require scatter-gather |
| NTP limitations | 1–100 ms accuracy; use HLC or TrueTime for event ordering |
| MirrorMaker 2 | Kafka cross-cluster async replication; topic prefixing; offset sync |
| Offline-first sync | Local DB + pending change log + idempotent sync API + conflict resolution |
| Partition handling | Single-link: quorum still possible. Full isolation: degrade gracefully |
| Netflix Chaos Kong | Deliberately evacuate entire AWS region; validate multi-region readiness |
| RTO / RPO | RTO = recovery time objective; RPO = data loss tolerance |
