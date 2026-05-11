# System Design Methodology and Requirements

> Interview-focused reference for structured system design thinking. Use this as your mental checklist before and during any system design interview.

---

## The 4-Step Interview Framework

A structured approach prevents rambling and demonstrates senior-level thinking. Every system design interview should follow this arc:

```
[Clarify Requirements] → [Estimate Scale] → [High-Level Design] → [Deep Dive]
     5 minutes               5 minutes          20 minutes          10 minutes
```

### Step 1 — Clarify (5 minutes)

Never start designing immediately. Ask questions to scope the problem. Interviewers deliberately leave questions open-ended to test whether you make assumptions or ask.

**What to clarify:**
- Who are the users? (consumers, businesses, internal engineers)
- What are the core features? (MVP vs full product)
- What is the scale? (10K users vs 10M users changes everything)
- Read-heavy or write-heavy?
- Any geographic constraints? (single region vs global)
- What consistency guarantees are required?
- Any existing infrastructure or tech stack constraints?

### Step 2 — Estimate (5 minutes)

Back-of-envelope calculations. Purpose: determine whether you need sharding, caching, CDN, etc. Don't over-engineer a system for 1,000 users.

**What to estimate:**
- Daily Active Users (DAU) → Queries Per Second (QPS)
- Storage requirements (3-year horizon)
- Bandwidth requirements (read + write)
- Memory requirements for cache

### Step 3 — High-Level Design (20 minutes)

Draw the system. Start with the client and work toward the data layer.

**Typical flow:**
```
Client → DNS → CDN → Load Balancer → App Servers → Cache → Database
                                                  → Message Queue → Workers
```

Cover: APIs, data models, key architectural decisions.

### Step 4 — Deep Dive (10 minutes)

Pick 2–3 critical components and explain them in detail. This is where you show depth:
- How does the database handle scale?
- How does the cache invalidation work?
- How do you handle failures?
- What are the trade-offs in your design?

---

## Functional vs Non-Functional Requirements

### Functional Requirements

What the system **does** — user-visible features and behaviors.

| Category | Examples |
|---|---|
| Core actions | User can post a tweet, upload a video, send a message |
| Data retrieval | User can view feed, search content, get notifications |
| User management | Register, login, follow/unfollow, block |
| Content operations | Create, read, update, delete, share |

**How to elicit:**
- "What are the top 3 features the system must support?"
- "Should we support editing after posting?"
- "Is search a required feature for this MVP?"

### Non-Functional Requirements

How the system **behaves** — the quality attributes.

| Requirement | Definition | Typical Target |
|---|---|---|
| Availability | % time system is operational | 99.9% – 99.999% |
| Latency (P99) | Max response time for 99% of requests | < 200ms API, < 50ms DB |
| Throughput | Requests/transactions per second | Depends on scale |
| Consistency | Are all users seeing the same data? | Eventual vs strong |
| Durability | Can we lose data? | 0 data loss for financial |
| Scalability | Can the system grow with load? | 10x growth in 2 years |
| Security | Auth, encryption, access control | HTTPS, RBAC, encryption at rest |
| Maintainability | Can the team operate and change it? | MTTR < 1 hour |

**Key insight:** Non-functional requirements often drive architecture decisions more than functional ones. A system that needs 99.999% availability requires a fundamentally different architecture than one needing 99%.

---

## Requirements Gathering Questions

### Universal Questions (Ask for Any System)

**Users and Usage:**
- Who are the primary users of this system?
- What is the expected number of users? (DAU/MAU)
- What is the geographic distribution of users?
- Are there user roles with different permissions?

**Scale and Traffic:**
- What is the expected QPS at peak? Average?
- What is the read-to-write ratio?
- Is traffic uniform or bursty? (e.g., sports events, flash sales)
- What is the expected growth rate? (2x/year? 10x?)

**Data:**
- What is the size of a typical data object?
- How long must data be retained?
- Does data ever get deleted? (GDPR implications)
- What are the access patterns? (recent data hot, old data cold)

**Reliability:**
- What is the acceptable downtime per year?
- Is partial availability acceptable? (degraded mode)
- What happens if a component fails — can users still read data?

**Consistency:**
- Is it acceptable if different users see slightly different data for a short window?
- Are there operations that require strict consistency? (payments, inventory)
- What is the acceptable staleness for reads?

**Constraints:**
- Any technology preferences or restrictions?
- Is this a greenfield build or migration?
- Any regulatory requirements? (HIPAA, GDPR, PCI-DSS)
- Budget considerations?

### System-Specific Questions

**For a Feed/Timeline system (Twitter, Instagram):**
- Is the feed ranked or chronological?
- How many people can a user follow?
- Do we need to support celebrities with millions of followers?
- Should the feed be pre-computed (push) or computed at read time (pull)?

**For a Storage/File system (Dropbox, Google Drive):**
- Maximum file size?
- Is deduplication required?
- Do we need versioning?
- Is offline access required?
- Collaboration features (simultaneous edits)?

**For a Messaging system (WhatsApp, Slack):**
- 1-1 messages only, or group chats?
- Maximum group size?
- Message persistence (chat history)?
- Read receipts, delivery receipts?
- Is end-to-end encryption required?

**For a Search system (Google, Elasticsearch):**
- Full-text search or structured search?
- Autocomplete required?
- How fresh must results be? (near real-time vs hourly index update)
- Ranking algorithm complexity?

**For a Recommendation system (Netflix, YouTube):**
- What signals drive recommendations? (views, ratings, time watched)
- Real-time personalization or batch computed?
- Cold start problem — how to handle new users?
- A/B testing support required?

---

## Availability, Latency, Throughput, Consistency — Defining SLAs

### Availability

**Definition:** The percentage of time a system is operational and accessible.

```
Availability = (Total Time - Downtime) / Total Time × 100%
```

**Availability Nines:**

| Availability | Annual Downtime | Monthly Downtime |
|---|---|---|
| 99% (two nines) | 3.65 days | 7.2 hours |
| 99.9% (three nines) | 8.76 hours | 43.8 minutes |
| 99.95% | 4.38 hours | 21.9 minutes |
| 99.99% (four nines) | 52.6 minutes | 4.4 minutes |
| 99.999% (five nines) | 5.26 minutes | 26.3 seconds |
| 99.9999% (six nines) | 31.5 seconds | 2.6 seconds |

**Interview insight:** Most services target 99.9% (three nines). Five nines is extremely expensive and rare. Financial systems, healthcare, and emergency services may require five nines.

**Composite availability** (systems in series multiply availabilities):
```
Two 99.9% services in series: 99.9% × 99.9% = 99.8%
Three 99.9% services in series: 99.9%³ = 99.7%
```

**Parallel redundancy improves availability:**
```
Two 99.9% services in parallel (either can serve):
1 - (1 - 0.999) × (1 - 0.999) = 1 - 0.000001 = 99.9999%
```

### Latency

**Definition:** Time from request sent to response received.

**Types:**
- **P50 (median):** 50% of requests complete within this time
- **P95:** 95% of requests complete within this time
- **P99:** 99% of requests complete within this time
- **P999:** 99.9% of requests complete within this time

**Why P99 matters more than average:**
- Average hides the long tail
- The user who waits 10 seconds is real — average of 50ms doesn't help them
- For microservices: a request touching 10 services at P99 each means 1 in 10^(10×-2) = ~10% see worst-case latency from at least one service

**Typical latency targets:**

| Operation | Target |
|---|---|
| User-facing API (web) | < 200ms P99 |
| User-facing API (mobile) | < 500ms P99 |
| Internal microservice call | < 50ms P99 |
| Database query (cached) | < 1ms |
| Database query (uncached) | < 10ms |
| Cache hit (Redis) | < 1ms |

### Throughput

**Definition:** Number of operations a system can process per unit of time.

- **QPS (Queries Per Second):** Read queries
- **TPS (Transactions Per Second):** Write operations
- **Bandwidth:** Data transferred per second (MB/s, Gbps)

**Throughput vs Latency trade-off:**
- Batching increases throughput but increases latency per item
- Processing individually minimizes latency but reduces throughput
- Choose based on user-facing or background system

### Consistency Models

**Strong Consistency:** After a write completes, all subsequent reads see that write. Required for: banking, inventory management, booking systems.

**Eventual Consistency:** All replicas will converge to the same value given enough time. Acceptable for: social media likes, follower counts, recommendations.

**Read-Your-Writes Consistency:** A user always sees their own writes. Important for: user profile updates, posting content.

**Monotonic Reads:** Once you read a value, you never read an older value. Important for: pagination, ordering.

**Causal Consistency:** Operations that are causally related are seen in order. Important for: comment threads (reply after post).

```
Stronger consistency ←——————————————→ Weaker consistency
[Linearizable] [Sequential] [Causal] [Read-Your-Writes] [Eventual]
    ↑                                                          ↑
Highest latency                                       Lowest latency
Lowest availability                               Highest availability
```

---

## Identifying Read-Heavy vs Write-Heavy Systems

This determines your architecture significantly.

### Read-Heavy Systems (10:1 to 100:1 read/write ratio)

**Examples:** Twitter feed, Wikipedia, Netflix, YouTube (viewing), product catalog

**Characteristics:**
- Optimize for read performance
- Caching is highly effective
- Can use read replicas
- CDN is beneficial

**Architecture implications:**
- Add Redis/Memcached caching layer
- Use read replicas in database
- Pre-compute expensive aggregations
- CDN for static and semi-static content

### Write-Heavy Systems (1:1 or more writes than reads)

**Examples:** IoT sensor data, logging systems, event tracking, analytics ingestion

**Characteristics:**
- Throughput for writes is the bottleneck
- Read can afford higher latency
- Data may be read much later than written

**Architecture implications:**
- Message queues to absorb write spikes (Kafka, SQS)
- Append-only storage (log-structured)
- Batch processing for reads
- LSM-tree databases (Cassandra, RocksDB) over B-tree for write performance

### Balanced Systems

**Examples:** Messaging apps, collaborative documents

- Separate read and write paths
- CQRS (Command Query Responsibility Segregation) pattern

### Practical Ratio Estimates:

| System | Read:Write Ratio |
|---|---|
| Social media feed | 100:1 |
| E-commerce product page | 1000:1 |
| Search engine | Very high |
| User activity logging | 1:1000 (write-heavy) |
| Database backup | Write-heavy |
| Chat application | ~1:1 |
| News website | 1000:1 |

---

## Stateless vs Stateful Design Decisions

### Stateless Services

**Definition:** The server does not store any client session state between requests. Each request contains all information needed to process it.

**Benefits:**
- Any server can handle any request → easy horizontal scaling
- Failure of one server does not affect other requests
- Load balancing is trivial (round robin works)
- Easy to add/remove server instances

**How to achieve statelessness:**
- Move session data to external store (Redis, DynamoDB)
- Use JWT tokens (self-contained, no server-side session)
- Pass state in the request (URL params, request body)
- Externalize file storage (S3 instead of local disk)

**Example — Session externalization:**
```
❌ Stateful (bad for scaling):
Client → Server1 (stores session in memory)
       → Server2 (no session! user gets logged out)

✓ Stateless (scalable):
Client → Server1 → Redis (read/write session) 
       → Server2 → Redis (same session data available)
```

### Stateful Services

**Definition:** The server maintains session or context between requests.

**When stateful is acceptable:**
- WebSocket connections (chat, gaming)
- Long-running operations with state machine
- When latency of external state lookup is unacceptable

**Stateful design challenges:**
- Sticky sessions required (route user to same server)
- Server failure causes session loss (need failover)
- Harder to scale horizontally

**Sticky session patterns:**
```
Option 1 — Client IP hash: same IP → same server
Option 2 — Session cookie: load balancer reads cookie, routes accordingly
Option 3 — Consistent hashing: by user ID → specific server
```

---

## Single Points of Failure (SPOF) Identification

A SPOF is any component whose failure causes the entire system to fail.

### Common SPOFs and Their Mitigations

| Component | SPOF Risk | Mitigation |
|---|---|---|
| Single database | High — entire system down | Primary-replica with automatic failover |
| Single load balancer | High | Active-active pair or cloud-managed LB |
| Single cache node | Medium | Redis cluster/sentinel, cache-aside fallback |
| Single message broker | High | Kafka replication, cluster mode |
| Single availability zone | High | Multi-AZ deployment |
| Single data center | High | Multi-region active-active or active-passive |
| DNS provider | Medium | Multiple DNS providers, low TTL |
| Single CDN | Low-medium | Multi-CDN strategy |
| Shared database connection pool | Medium | Connection pool per service |

### SPOF Checklist for System Design

```
For every component, ask:
1. What happens if this component fails?
2. Is there a backup or replica?
3. Does failover happen automatically or manually?
4. What is the recovery time (RTO)?
5. What data is lost on failure (RPO)?
```

**RTO (Recovery Time Objective):** How long can the system be down?
**RPO (Recovery Point Objective):** How much data loss is acceptable?

| Tier | RTO | RPO |
|---|---|---|
| Tier 1 (mission critical) | < 15 min | 0 |
| Tier 2 (business critical) | < 1 hour | < 15 min |
| Tier 3 (important) | < 4 hours | < 1 hour |
| Tier 4 (non-critical) | < 24 hours | < 24 hours |

---

## Trade-Off Analysis Framework

Every design decision is a trade-off. Senior engineers don't just know the right answer — they understand **why** and **what you give up**.

### Consistency vs Availability (CAP Theorem)

**CAP Theorem:** In the presence of a network partition, a distributed system can guarantee either Consistency or Availability, but not both.

```
     Consistency
          /\
         /  \
        /    \
       / CA   \
      /  (not  \
     /  partition \
    /   tolerant)  \
   /________________\
Availability   Partition
               Tolerance
```

**CP systems (Consistent + Partition Tolerant):** HBase, Zookeeper, MongoDB (w:majority)
- Returns error rather than stale data during partition
- Good for: banking, inventory, leader election

**AP systems (Available + Partition Tolerant):** Cassandra, DynamoDB, CouchDB
- Returns potentially stale data during partition
- Good for: social feeds, shopping carts, DNS

**Real-world note:** In practice, you choose between CP and AP only during network partitions (rare). Most of the time, all three are achievable.

### PACELC Trade-Off (Extension of CAP)

During normal operation (no partition), choose between:
- **Latency vs Consistency**

```
If Partition:  choose A (availability) or C (consistency)
ELse:          choose L (latency) or C (consistency)
```

| System | Partition behavior | Normal behavior |
|---|---|---|
| DynamoDB | Available | Low latency |
| Cassandra | Available | Low latency |
| MongoDB | Consistent | Low latency |
| HBase | Consistent | Consistent |

### Latency vs Throughput

| Approach | Latency | Throughput |
|---|---|---|
| Synchronous processing | Low for individual ops | Lower overall |
| Async batch processing | Higher per item | Much higher overall |
| Streaming (Kafka) | Medium | High |

### Cost vs Performance

| Strategy | Performance Gain | Cost |
|---|---|---|
| Add more servers | Linear | Linear |
| Add caching | 10x-100x | Low |
| Add CDN | 2x-10x for static | Medium |
| Database read replicas | 2x-10x reads | Medium |
| Sharding | Near linear | High (complexity) |

### Strong Consistency vs Performance

| Consistency Level | Latency | Availability | Use Case |
|---|---|---|---|
| Linearizable | Highest | Lowest | Financial transactions |
| Sequential | High | Low-medium | Leader election |
| Eventual | Lowest | Highest | Social media likes |

---

## Common Mistakes in System Design Interviews

### Mistake 1 — Jumping to Design Without Clarifying

**Problem:** You spend 30 minutes designing a Twitter-like system, then the interviewer says "I meant an internal Twitter for 500 employees."

**Fix:** Always spend 5 minutes asking clarifying questions. Verbalize your assumptions.

### Mistake 2 — Over-Engineering From the Start

**Problem:** Proposing Kubernetes, service mesh, multi-region active-active for a startup with 1,000 users.

**Fix:** Start simple. "I'll start with a monolith and MySQL. Let me show you how we'd scale when we hit 10x."

### Mistake 3 — Under-Engineering (No Scalability Plan)

**Problem:** "I'll just use one database server."

**Fix:** After designing the simple version, always discuss the scaling path. "When we exceed 10K QPS, we'd add read replicas. At 100K QPS, we'd consider sharding."

### Mistake 4 — Ignoring Non-Functional Requirements

**Problem:** Designing only for features without addressing availability, latency, or consistency.

**Fix:** After drawing the design, walk through: "This meets our 99.9% availability target because... Our P99 latency will be under 200ms because..."

### Mistake 5 — Not Discussing Trade-offs

**Problem:** "I chose MongoDB because it's better."

**Fix:** "I chose MongoDB over PostgreSQL here because the schema is document-oriented and flexibility outweighs the need for joins. The trade-off is we lose ACID transactions across documents, which is acceptable because..."

### Mistake 6 — Staying Too High-Level

**Problem:** Drawing boxes and arrows without explaining what's inside them.

**Fix:** For key components, go deep. "This is a Kafka cluster. Here's why I chose Kafka over RabbitMQ: ordered message delivery per partition, retention allows consumers to replay, high throughput write path."

### Mistake 7 — Ignoring the Data Model

**Problem:** Never discussing how data is stored, indexed, or queried.

**Fix:** Always include a data model section. "The User table has id, username, email, created_at. The Tweet table has id, user_id (FK), content, timestamp. I'll index on user_id and timestamp for feed queries."

### Mistake 8 — No Failure Handling

**Problem:** Designing only the happy path.

**Fix:** "What happens if the database goes down? What if the cache is cold? What if the message queue is full?"

### Mistake 9 — Not Prioritizing

**Problem:** Spending 20 minutes on user authentication and 2 minutes on the core feed algorithm.

**Fix:** Identify the hardest/most interesting component early. "The most challenging part is fan-out for the news feed. Let me focus there."

### Mistake 10 — Not Listening to Hints

**Problem:** Interviewer says "what if a celebrity has 100M followers?" and you ignore it.

**Fix:** Interviewers give hints on purpose. "Good point — celebrity accounts with 100M followers would overwhelm a push-based fanout. I'd use a hybrid approach..."

---

## Interview Time Allocation Strategy

### Minute-by-Minute Guide (45-minute interview)

```
Minutes 0–5:   CLARIFY
├── Ask 4-6 clarifying questions
├── State assumptions explicitly
└── Agree on MVP scope

Minutes 5–10:  ESTIMATE  
├── Calculate DAU → QPS
├── Estimate storage (3-year)
├── Determine read/write ratio
└── Conclude: need cache? sharding? CDN?

Minutes 10–30: HIGH-LEVEL DESIGN
├── 10–15: Draw components, explain APIs
├── 15–20: Data model and storage choices
└── 20–30: Key algorithms and flows

Minutes 30–40: DEEP DIVE
├── Pick 1-2 hard components
├── Justify choices with trade-offs
└── Handle failure scenarios

Minutes 40–45: WRAP-UP
├── Summarize key decisions
├── Mention what you'd do differently at 10x scale
└── Ask interviewer questions
```

### Pacing Signals

**You're going too slow if:**
- Still clarifying at minute 10
- Haven't drawn anything by minute 15
- Only covered 2 components at minute 35

**You're going too fast if:**
- Skipping trade-off explanations
- Not justifying technology choices
- Jumping to deep dive before finishing high-level

### How to Handle "I Don't Know"

"I'm not familiar with the internals of X, but here's how I'd think about it: [apply first principles]..."

This is always better than silence or guessing incorrectly.

---

## Quick Reference: Questions to Ask for Any System Design Problem

### The Opening Checklist

```
SCOPE
□ What are the top 3-5 features we need to support?
□ What features can we explicitly exclude from scope?

USERS
□ Who are the users — consumers, businesses, internal?
□ How many daily active users (DAU)?
□ Geographic distribution — single region or global?

SCALE
□ What is the expected QPS (reads and writes separately)?
□ What is the peak-to-average traffic ratio?
□ What is the data volume today? Expected in 3 years?

REQUIREMENTS
□ What availability SLA is required? (99.9%? 99.99%?)
□ What is the acceptable read latency? (P50, P99)
□ Is consistency critical or can we accept eventual consistency?
□ What is the data retention policy?

CONSTRAINTS
□ Any existing tech stack we must use?
□ Cloud provider preference?
□ Budget or operational complexity constraints?
□ Regulatory/compliance requirements?
```

### Red Flag Questions (to ask if they apply)

```
□ Are there celebrity/power users with very high follower counts?
  (Impacts fan-out design)
  
□ Is search required? Near-real-time or batch indexed?
  (Requires separate search infrastructure)
  
□ Are there operational constraints — can we afford 
  complex distributed systems?
  (Simplicity vs scalability)

□ Is this mobile-first? (Offline support? Bandwidth sensitivity)

□ Multi-tenancy requirements? (Data isolation per tenant)

□ Audit logging required? (Append-only event log needed)

□ GDPR/right-to-be-forgotten? (Delete propagation across replicas)
```

### Technology Decision Questions

```
DATABASE
□ Structured data with complex joins → Relational (PostgreSQL, MySQL)
□ Document data, flexible schema → MongoDB
□ Time-series data → InfluxDB, TimescaleDB
□ Wide-column, high write throughput → Cassandra
□ Graph relationships → Neo4j
□ Full-text search → Elasticsearch

CACHE
□ Simple key-value, high throughput → Redis
□ Distributed, multi-threaded → Memcached

MESSAGE QUEUE
□ High throughput, ordering, replay → Kafka
□ Simple async tasks → RabbitMQ, SQS
□ Delay queues, FIFO guarantees → SQS FIFO

STORAGE
□ Large files, blobs → S3, GCS
□ Relational + transactions → PostgreSQL
□ Globally distributed, multi-region → CockroachDB, Spanner
```

---

## Summary: The Mental Model

```
Every system design interview is really 3 questions:

1. WHAT does it do?
   → Functional requirements, APIs, data models

2. HOW WELL does it do it?
   → Non-functional requirements: availability, latency, throughput

3. HOW does it keep working when things break?
   → Fault tolerance, replication, failure modes

If you can answer these three questions with specific numbers,
concrete technology choices, and clear trade-off reasoning,
you are performing at the senior level.
```

---

*Reference: Alex Xu "System Design Interview" Vol 1 & 2, Martin Kleppmann "Designing Data-Intensive Applications"*
