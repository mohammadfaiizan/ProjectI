# 20. System Design Interview Framework

## Table of Contents
1. [The Complete Interview Playbook](#1-the-complete-interview-playbook)
2. [Phase 1: Clarification Questions](#2-phase-1-clarification-questions)
3. [Phase 2: Capacity Estimation](#3-phase-2-capacity-estimation)
4. [Phase 3: High-Level Design](#4-phase-3-high-level-design)
5. [Phase 4: Deep Dive](#5-phase-4-deep-dive)
6. [Key Trade-Offs to Know Cold](#6-key-trade-offs-to-know-cold)
7. [Common Building Blocks and When to Use](#7-common-building-blocks-and-when-to-use)
8. [Handling 10x Scale](#8-handling-10x-scale)
9. [Common Red Flags](#9-common-red-flags)
10. [Explaining Diagrams Verbally](#10-explaining-diagrams-verbally)
11. [Handling "What If Traffic 100x?" Follow-Ups](#11-handling-what-if-traffic-100x-follow-ups)
12. [Complete Checklist](#12-complete-checklist)
13. [Trade-Off Analysis Templates](#13-trade-off-analysis-templates)
14. [Study Plan: 30 System Design Problems](#14-study-plan-30-system-design-problems)
15. [Top 30 Most Common Questions](#15-top-30-most-common-questions)
16. [Component Selection Cheat Sheet](#16-component-selection-cheat-sheet)
17. [Numbers Every Engineer Must Know](#17-numbers-every-engineer-must-know)
18. [Interview Evaluation Rubric](#18-interview-evaluation-rubric)
19. [Common Mistakes and How to Avoid Them](#19-common-mistakes-and-how-to-avoid-them)
20. [Quick Reference](#20-quick-reference)

---

## 1. The Complete Interview Playbook

### 45-Minute Interview Timeline

```
[00:00 - 05:00] Phase 1: Clarification
  - Ask about scope, scale, constraints
  - Define functional and non-functional requirements
  - Goal: ensure you're solving the right problem

[05:00 - 10:00] Phase 2: Capacity Estimation
  - Estimate DAU, QPS, storage, bandwidth
  - Identify scale challenges upfront
  - Goal: size the system correctly

[10:00 - 20:00] Phase 3: High-Level Design
  - Draw the major components and data flows
  - Cover the happy path end-to-end
  - Goal: show you understand the full system

[20:00 - 35:00] Phase 4: Deep Dive
  - Pick 1-2 complex components and go deep
  - Address bottlenecks, edge cases, failure modes
  - Respond to interviewer hints about what to dig into
  - Goal: demonstrate technical depth

[35:00 - 45:00] Phase 5: Wrap-Up
  - Summarize key design decisions and trade-offs
  - Mention what you'd improve with more time
  - Answer remaining interviewer questions
```

### Mindset Principles

```
1. Think out loud: narrate your reasoning, don't go silent
2. Ask before you assume: clarify scope before designing
3. Top-down, not bottom-up: start at 30,000 feet, then zoom in
4. Justify decisions: "I chose X because Y, trade-off is Z"
5. Acknowledge trade-offs: there's no perfect system
6. Drive the conversation: propose what to deep-dive next
7. Accept hints gracefully: "Good point, let me reconsider that"
8. Draw as you talk: diagram and narration should match
```

---

## 2. Phase 1: Clarification Questions

### Universal Questions (Ask for EVERY system)

```
1. Scale:
   "How many users do we expect? DAU? MAU?"
   "What's the expected QPS for reads and writes?"
   "Is this global or single-region?"

2. Functional scope:
   "What are the core features for this design? Which can I deprioritize?"
   "Should I design the MVP or the full system?"
   
3. Non-functional priorities:
   "Is consistency or availability more important here?"
   "What's the acceptable latency? (P99?)"
   "What's the data retention requirement?"

4. Existing infrastructure:
   "Are we building on cloud (AWS/GCP/Azure) or on-prem?"
   "Can I use managed services or build everything from scratch?"

5. Edge cases to address:
   "Should I handle mobile clients differently from web?"
   "How important is data durability vs performance?"
```

### System-Specific Clarifying Questions

**URL Shortener:**
- "Should shortened URLs be custom or auto-generated?"
- "Do we need analytics (click tracking)?"
- "What's the URL expiration policy?"

**News Feed:**
- "Is this a social network or content-based feed?"
- "Can users create content or only consume?"
- "Should I support media (images/videos)?"

**Chat System:**
- "1-on-1, group chat, or both?"
- "Do we need message persistence and history?"
- "Push notifications for mobile?"

**Rate Limiter:**
- "Client-side or server-side? Per-user or per-IP?"
- "What algorithm? (Token bucket, sliding window?)"
- "What happens when rate limit exceeded? (Drop, queue, throttle?)"

---

## 3. Phase 2: Capacity Estimation

### Worked Example: Instagram-Scale System

```
Given:
  500M DAU
  Users post: 1% active posters → 5M posts/day
  Users read: avg 20 posts/session, 3 sessions/day
  Media: 50% of posts have images (avg 2MB each)

Step 1: QPS Estimates
  Write QPS = 5M posts/day ÷ 86,400s = ~58 writes/sec
  Read QPS  = 500M × 60 reads/day ÷ 86,400s = ~347,000 reads/sec
  Read:Write ratio = 6,000:1 → read-heavy, use caching heavily

Step 2: Storage Estimates
  DB storage:
    1 post metadata ≈ 1KB (text, user_id, timestamp, location)
    5M posts/day × 1KB = 5GB/day
    5 years: 5GB × 365 × 5 = ~9TB metadata
  
  Media storage:
    5M posts/day × 50% with image × 2MB = 5TB/day
    5 years = 5TB × 365 × 5 = ~9PB media
    CDN caches hot content → storage cost on object storage (S3)

Step 3: Bandwidth
  Write bandwidth: 58 writes/sec × 2MB = ~116 MB/sec
  Read bandwidth: 347K reads/sec × 2MB = ~700 GB/sec → need CDN!

Step 4: Cache estimation
  Hot posts: 20% of posts generate 80% of traffic (Pareto)
  Cache top 20%: 5M/day × 20% × 1KB = 1MB/day → trivial!
  Significance: aggressive caching covers most reads from memory
```

### Quick Estimation Numbers

```
Time conversions:
  1 day   = 86,400 seconds ≈ 100,000 seconds
  1 month = ~2.5M seconds
  1 year  = ~32M seconds

Request volume → QPS:
  1M requests/day  = ~12 req/sec
  10M requests/day = ~120 req/sec
  100M/day         = ~1,200 req/sec
  1B/day           = ~12,000 req/sec

Storage:
  1 char = 1 byte
  1 tweet (280 chars) ≈ 280 bytes ≈ 0.3 KB
  1 user metadata ≈ 1 KB
  1 thumbnail ≈ 100 KB
  1 photo ≈ 1-5 MB
  1 video (1 min, 720p) ≈ 100-300 MB

Network:
  Gigabit Ethernet = 125 MB/sec
  10 Gigabit = 1.25 GB/sec
  CDN edge → user: depends on ISP, often 1-100 Mbps
```

---

## 4. Phase 3: High-Level Design

### Always Mention These Components

```
Every system design should include explicit decisions about:

1. Load Balancer
   - Layer 4 (TCP) or Layer 7 (HTTP)?
   - Algorithm: round-robin, least connections, IP hash?
   - Health checks, SSL termination?

2. API Gateway / CDN
   - Rate limiting, authentication, request routing
   - CDN for static assets and media
   - Edge caching for API responses

3. Application Servers
   - Stateless (can be auto-scaled horizontally)
   - Session management (externalized to Redis)
   - Deployment: containers (Kubernetes) or VMs

4. Cache Layer
   - What to cache? (hot reads, session data, computed results)
   - Cache-aside vs write-through vs write-behind?
   - Eviction policy? Cache invalidation strategy?

5. Database
   - SQL or NoSQL? Why?
   - Primary-replica or multi-master?
   - Read replicas for scale?
   - Sharding if needed?

6. Message Queue
   - Async decoupling between services
   - Which operations can be async? (notifications, emails, analytics)
   - At-least-once delivery + idempotent consumers

7. Object Storage (if media involved)
   - S3 for unstructured data (images, videos, documents)
   - CDN in front of object storage

8. Search (if search involved)
   - Elasticsearch/OpenSearch
   - How does data get into search? (CDC, dual-write)
```

### Standard Architecture Template

```
[Client (Web/Mobile)]
        |
        v
[CDN] ←── static assets, cached API responses
        |
        v
[API Gateway]  ← auth, rate limiting, routing
        |
   _____|_____
  |     |     |
  v     v     v
[Service A] [Service B] [Service C]  ← stateless microservices
  |              |
  v              v
[Cache (Redis)]  [DB (Primary)]
                  |
              [DB Replica × N] ← read scaling
                  |
              [Object Storage (S3)] ← media
                  |
             [Message Queue (Kafka)]
                  |
           [Async Workers]
                  |
           [Search (ES)]
           [Notification Service]
           [Analytics Pipeline]
```

### Happy Path Narration Script

```
"Let me walk through the happy path for [operation]:

1. The user's request hits the CDN first for static assets.
   For API requests, it goes to the load balancer.

2. The load balancer (Layer 7, round-robin) forwards to one of N
   stateless API servers.

3. The API server checks the cache first [explain cache key/strategy].
   On cache hit: return immediately.
   On cache miss: query the database.

4. The database [explain schema/query].
   We write to the primary, reads to replicas.

5. For [async operation]: we enqueue to Kafka and return immediately.
   A consumer processes [X] asynchronously.

6. Response flows back: DB → API server → cache update → client.

Key design decisions:
  - Stateless API servers: any server can handle any request
  - Cache reduces DB load from 100K QPS to ~10K QPS (90% cache hit)
  - Async queue prevents [slow operation] from blocking user response"
```

---

## 5. Phase 4: Deep Dive

### How to Choose What to Deep-Dive

```
Let the problem guide you:

High availability problem → deep-dive replication, failover, leader election
High write throughput     → deep-dive sharding, write path, LSM vs B-tree
High read throughput      → deep-dive caching strategy, read replicas, CDN
Consistency concerns      → deep-dive transaction handling, saga pattern
Latency constraints       → deep-dive hot path optimization, data locality
Search functionality      → deep-dive inverted index, ranking, query processing
Real-time requirements    → deep-dive WebSocket, pub/sub, fan-out
Large media files         → deep-dive chunked upload, CDN, object storage

Interviewer hints to watch for:
  "What if one of those DB servers fails?"     → Failover / replication
  "How would you handle a hot user?"           → Celebrity problem / hybrid fan-out  
  "What if the cache goes down?"               → Cache stampede / circuit breaker
  "How do you prevent duplicate messages?"     → Idempotency
  "What if QPS is 10x your estimate?"          → Sharding / horizontal scale
```

### Deep-Dive Framework for Any Component

```
For any component being deep-dived, cover:

1. DATA MODEL: What data does it store? Schema? Indexes?
2. WRITE PATH: How does data get in? Batched? Synchronous?
3. READ PATH: How is data read? Cached? Aggregated?
4. FAILURE MODES: What happens if this component fails?
5. SCALE BOTTLENECK: Where does it break at 10x load?
6. OPTIMIZATION: What can be optimized for the primary concern?
```

---

## 6. Key Trade-Offs to Know Cold

### Consistency vs Availability (CAP Theorem)

```
CAP Theorem: In a distributed system, you can guarantee at most 2 of:
  C - Consistency   (all nodes see same data at same time)
  A - Availability  (every request gets a response)
  P - Partition Tolerance (system works despite network partitions)

Since network partitions are inevitable → choose CP or AP:

CP systems:  Refuse requests when can't guarantee consistency
  Examples: ZooKeeper, etcd, HBase, CockroachDB
  Use when: financial transactions, leader elections, config

AP systems:  Serve potentially stale data rather than refusing
  Examples: DynamoDB, Cassandra, CouchDB, DNS
  Use when: shopping carts, social feeds, user profiles

Nuance: Most systems offer tunable consistency
  DynamoDB: consistent_reads=True (CP) or False (AP) per query
  Cassandra: consistency level ONE (AP) vs QUORUM (CP) per query
```

### SQL vs NoSQL

```
Choose SQL (PostgreSQL, MySQL) when:
  ✓ Data is relational (foreign keys, JOINs)
  ✓ ACID transactions required
  ✓ Schema is well-defined and stable
  ✓ Complex queries (ad-hoc reporting)
  ✓ Data volume: up to a few TB (single node or read replicas)

Choose NoSQL when:
  ✓ Horizontal scaling is primary concern
  ✓ High write throughput (10K+ writes/sec)
  ✓ Schema is flexible or semi-structured
  ✓ Key-value or document access patterns
  ✓ Geographic distribution required

NoSQL sub-types:
  Key-Value (Redis, DynamoDB): O(1) get/put, no query capability
  Document (MongoDB):          JSON documents, flexible schema
  Wide-column (Cassandra):     Time-series, write-heavy workloads
  Graph (Neo4j):               Relationship traversal (friends-of-friends)
```

### Synchronous vs Asynchronous Processing

```
Choose Synchronous when:
  - Client needs immediate result
  - Operation must complete before response
  - Examples: login, payment initiation, search query

Choose Asynchronous when:
  - Operation is slow (>100ms) and result not needed immediately
  - Operation can fail and retry transparently
  - Fan-out to many downstream systems
  - Examples: email sending, push notifications, analytics, image resizing

Pattern for bridging:
  Client → Submit job (sync) → Get job ID
  Client → Poll status / WebSocket update (async)
  Worker → Process job, update status, notify client
```

### Push vs Pull

```
Push (fan-out on write):
  Server pushes data to consumers proactively
  + Low read latency
  - Write amplification, wasted resources for inactive consumers
  Use: messaging, real-time notifications, live scores

Pull (fan-out on read):
  Consumer requests data when needed
  + Simple, no wasted resources
  - Higher read latency, potential overload on read spikes
  Use: email (IMAP), batch jobs, low-frequency data

Hybrid:
  Push for active users, pull for inactive
  Use: Twitter-style feeds, news feeds
```

---

## 7. Common Building Blocks and When to Use

### Decision Matrix: Infrastructure Components

| Component | When to Use | When NOT to Use |
|-----------|------------|-----------------|
| Load Balancer | Any multi-server deployment | Single server |
| CDN | Static assets, media, global users | Internal APIs only |
| Redis Cache | Hot reads, sessions, rate limiting, pub/sub | Write-heavy, large objects |
| Kafka | Async decoupling, event streaming, fan-out | Simple job queue, small volume |
| Elasticsearch | Full-text search, log analysis, faceted search | Simple WHERE queries |
| S3/Object Storage | Media, backups, large files | Small structured records |
| Cassandra | Time-series, high-write, wide-column | Complex JOINs needed |
| PostgreSQL | ACID, relational, complex queries | Massive horizontal write scale |
| DynamoDB | Global, auto-scale, serverless, KV/document | Complex queries, transactions |

### When to Add Each Layer

```
Start with: Single server, single database
  ↓ When: App is slow
Add: Connection pooling (pgBouncer), query optimization

  ↓ When: Read QPS > 10,000
Add: Read replicas + cache (Redis)

  ↓ When: Write QPS > 50,000 or data > 5TB
Add: Database sharding (horizontal partitioning)

  ↓ When: Response time > 100ms for hot paths
Add: CDN for static assets, aggressive caching

  ↓ When: Slow operations (email, notifications) blocking user requests
Add: Message queue (Kafka/SQS) + async workers

  ↓ When: Single data center failure risk
Add: Multi-region deployment with data replication
```

---

## 8. Handling 10x Scale

### Progressive Scaling Strategy

```
Phase 1 (0 → 100K users): Single server
  App + DB on same machine
  No cache needed
  Simple deployment

Phase 2 (100K → 1M users): Separate concerns
  Separate app and DB servers
  Add read replica for DB
  Add Redis for sessions
  Add CDN for static assets

Phase 3 (1M → 10M users): Scale components
  Multiple app servers behind load balancer (stateless)
  DB replicas × 3 for read scale
  Redis cluster for cache + pub/sub
  Introduce message queue for async tasks

Phase 4 (10M → 100M users): Sharding + microservices
  DB sharding (hash by user_id or tenant_id)
  Extract high-traffic services (auth, search, media)
  Multi-region deployment
  Advanced caching (CDN for API responses)

Phase 5 (100M+ users): Global infrastructure
  Multi-region active-active or active-passive
  Geographically distributed databases (CockroachDB, Spanner)
  Edge computing for lowest latency
  Sophisticated rate limiting and traffic management
```

### Statelessness: The Foundation of Horizontal Scale

```
Rule: Application servers must be stateless
What to externalize:
  - Sessions           → Redis
  - Uploaded files     → S3 (don't store locally)
  - Locks              → Redis / ZooKeeper
  - Long-running tasks → Database or Redis state
  - Config             → Environment variables / config service

Test: "Can I kill any app server and the system still works?"
If yes: truly stateless ✓
If no: find what's stored in-process and externalize it
```

---

## 9. Common Red Flags

### Single Points of Failure

```
Red flag: "The primary database handles all traffic"
Fix: Add read replicas + automatic failover (PostgreSQL Patroni, MySQL InnoDB Cluster)

Red flag: "One message broker for everything"
Fix: Kafka cluster (replicated partitions), ZooKeeper ensemble, or managed service

Red flag: "Load balancer with no redundancy"
Fix: Active-passive LB pair, or use cloud LB (ALB is managed and inherently HA)

Red flag: "Single datacenter"
Fix: Multi-AZ at minimum, multi-region for global products

Red flag: "All traffic routed through one service"
Fix: Service mesh, circuit breakers (Netflix Hystrix, Resilience4j)
```

### Scale Anti-Patterns

```
Anti-pattern: "We'll add indexes later"
Fix: Design schema with query patterns in mind; add indexes during design

Anti-pattern: No caching mentioned
Fix: Identify hot read paths early; propose cache-aside with Redis

Anti-pattern: "The monolith handles everything"
Fix: Identify I/O-bound vs CPU-bound work; separate concerns into services or workers

Anti-pattern: Synchronous calls for everything
Fix: Identify operations that can be async; use queues for decoupling

Anti-pattern: Growing database indefinitely
Fix: Data retention policy, archiving, time-series index rotation
```

### Security Red Flags

```
Never mention in interview without also mentioning security:
  - User authentication: "JWT tokens stored in httpOnly cookies, validated on every request"
  - API rate limiting: "per-user and per-IP rate limiting at API gateway"
  - Input validation: "sanitize all user inputs to prevent injection"
  - Data in transit: "TLS/HTTPS everywhere, internal services use mTLS"
  - Data at rest: "encrypt PII columns, S3 bucket encryption"
  - Access control: "principle of least privilege for service accounts"
```

---

## 10. Explaining Diagrams Verbally

### What to Say as You Draw Each Component

```
When drawing Load Balancer:
  "I'm adding an L7 load balancer here, which will do SSL termination
   and use round-robin to distribute requests across app servers.
   It also handles health checks and removes unhealthy instances."

When drawing Database:
  "Here's the primary database — I'll use PostgreSQL here because we need
   ACID transactions for [reason]. I'll add a read replica to handle
   read-heavy traffic. The primary handles all writes."

When drawing Cache:
  "I'm adding Redis as a caching layer in front of the database.
   We'll use cache-aside pattern: check cache first, on miss go to DB
   and populate cache. Cache hit rate should be high because [reason].
   TTL of [X] handles cache invalidation."

When drawing Message Queue:
  "This async queue decouples the [slow operation] from the user request.
   The API server enqueues the job and returns immediately. Workers
   consume and process asynchronously. This prevents [slow thing] from
   blocking user response time."

When drawing CDN:
  "CDN sits in front of our media storage and API layer for static content.
   It caches at edge nodes close to users, reducing origin server load
   and improving latency for global users."
```

---

## 11. Handling "What If Traffic 100x?" Follow-Up Questions

### Structured Response Template

```
"Good question. At 100x traffic, the bottlenecks would be:

1. [Identify the bottleneck]
   Currently we're at [X QPS]. At 100x, we'd have [100X QPS].
   The first thing to break would be [database/cache/service] because [reason].

2. [Propose solution]
   To handle this, I'd:
   - [Solution 1]: e.g., shard the database by user_id, giving us N × capacity
   - [Solution 2]: e.g., add more aggressive caching to reduce DB QPS
   - [Solution 3]: e.g., move to async processing for non-critical paths

3. [Identify next bottleneck]
   After solving that, the next bottleneck would be [X] because [Y].
   We'd address that by [Z].

4. [Acknowledge trade-offs]
   This does mean [trade-off: consistency/complexity/cost increase]."
```

### Common 100x Solutions by Component

```
Database reads hitting limit:
  → Add read replicas
  → Introduce caching layer (cache hit rate 80-90%)
  → Denormalize hot query results
  → Introduce CQRS (separate read and write models)

Database writes hitting limit:
  → Shard by user_id or tenant_id
  → Move to append-only log + async aggregation
  → Use NoSQL with higher write throughput (Cassandra)
  → Batch writes (collect 1000 writes, batch insert)

Network bandwidth limit:
  → Add CDN for static and media content
  → Compress responses (gzip, Brotli)
  → Reduce payload size (GraphQL vs REST for mobile)
  → Edge caching to serve from nearest location

CPU-bound computation:
  → Scale out (more app servers)
  → Optimize hot paths
  → Move to pre-computation / materialized views
  → Consider ASIC/GPU if truly CPU-bound (ML inference)
```

---

## 12. Complete Checklist

### Before You Start

```
[ ] Asked about user scale (DAU/MAU)
[ ] Asked about read vs write ratio
[ ] Asked about latency requirements
[ ] Asked about consistency requirements
[ ] Defined functional requirements (top 3 features)
[ ] Defined non-functional requirements (latency, availability, durability)
[ ] Asked about geographic distribution
[ ] Asked about mobile vs web vs API usage
```

### During Design

```
[ ] Capacity estimation: QPS, storage, bandwidth
[ ] API design: endpoints, request/response schemas
[ ] Data model: schema, indexes, relationships
[ ] High-level architecture diagram
[ ] Cache strategy (what, where, when to invalidate)
[ ] Database choice and justification
[ ] Handling async operations (queue?)
[ ] CDN strategy (what's cached at edge?)
[ ] Authentication and authorization
[ ] Rate limiting
[ ] Error handling and retries
[ ] Failure scenarios addressed
```

### Before Finishing

```
[ ] Addressed single points of failure
[ ] Mentioned monitoring/alerting
[ ] Discussed data retention/cleanup
[ ] Summarized key design decisions
[ ] Mentioned trade-offs of your choices
[ ] Offered to deep-dive any specific area
[ ] Asked if interviewer wants to explore any part further
```

---

## 13. Trade-Off Analysis Templates

### Template for Every Major Decision

```
"I'm choosing [Option A] over [Option B] for [this component] because:

Pros of Option A in this context:
  - [Primary reason matching the main system constraint]
  - [Secondary reason]

The trade-off is:
  - [What we give up by choosing A]
  - [When this trade-off matters]

If the requirement were [X instead of Y], I'd choose Option B."
```

### 10 Common System Trade-Off Analyses

**1. URL Shortener: UUID vs Base62 encoding vs Hash**
```
Base62 counter: predictable, easy to decode, risk of enumeration
MD5 hash: distributed, not enumerable, collision risk
Custom: user-chosen vanity URLs, requires uniqueness check
→ Use: Base62 on auto-increment ID from DB, resolve enumeration with auth
```

**2. News Feed: Fan-out on write vs read**
```
Write: O(1) reads, O(N followers) writes, hotspot for celebrities
Read: O(N followees) reads, O(1) writes, poor latency for power users
Hybrid: write for regular users, read merge for celebrities (Instagram/Twitter)
→ Choose based on read:write ratio, celebrity concentration in user base
```

**3. Chat: Single server vs distributed**
```
Single: simple, no routing, limited scale
Distributed (Redis pub/sub): complex routing, unlimited scale
→ Use Redis pub/sub backbone when > 1 server needed
```

**4. Database: SQL vs NoSQL for user profiles**
```
SQL: joins with other data, ACID, complex queries
NoSQL: simple by user_id, high-scale reads, flexible schema
→ User profiles: DynamoDB if simple CRUD + high scale; PostgreSQL if joins needed
```

**5. Cache: Write-through vs cache-aside**
```
Cache-aside: stale reads possible, cache only what's needed
Write-through: consistent but every write hits cache (memory waste for cold data)
→ Cache-aside for read-heavy with tolerable staleness; write-through for always-needed data
```

**6. Search: Elasticsearch vs PostgreSQL full-text**
```
PG FTS: simpler ops, same DB, good for <1M documents
Elasticsearch: more features, better performance, separate infra to manage
→ PG FTS for simple cases; ES for complex search, facets, large scale
```

**7. Rate Limiting: Token bucket vs sliding window**
```
Token bucket: burst handling, simple, good for API quotas
Sliding window: precise fairness, slightly more storage
→ Token bucket for most cases; sliding window for strict fairness requirements
```

**8. Notifications: Push vs pull**
```
Push (FCM/WebSocket): real-time, server initiates, requires persistent connection
Pull (polling): simpler, works behind NAT, higher latency
→ Push for real-time requirements; pull as fallback
```

**9. ID Generation: UUID vs Snowflake vs DB auto-increment**
```
UUID: decentralized, large (128-bit), random (poor index locality)
Snowflake: timestamp-sorted, 64-bit, requires central service
DB auto-increment: simple, sequential, doesn't scale to distributed writes
→ Snowflake for distributed systems needing sortable IDs
```

**10. Consistency: Eventual vs strong for shopping cart**
```
Strong: prevents lost updates, requires locking, slower
Eventual: fast, scales, risk of conflicting concurrent adds
→ Eventual with merge (union of add operations) for shopping cart; strong for checkout
```

---

## 14. Study Plan: 30 System Design Problems

### Week 1-2: Foundation Systems (Must-Know)

```
Day 1-2:  URL Shortener
  Key concepts: hash functions, redirect (301 vs 302), analytics, custom URLs
  
Day 3-4:  Key-Value Store
  Key concepts: consistent hashing, replication, conflict resolution, gossip protocol
  
Day 5-6:  Rate Limiter
  Key concepts: token bucket, sliding window, Redis atomic operations, distributed rate limiting
  
Day 7-8:  Web Crawler
  Key concepts: URL frontier, politeness, deduplication, distributed BFS
  
Day 9-10: Notification System
  Key concepts: FCM/APNs, fan-out, user preferences, delivery guarantees
  
Day 11-12: News Feed
  Key concepts: fan-out models, timeline storage, real-time updates
  
Day 13-14: Review + Mock interview
```

### Week 3-4: Intermediate Systems

```
Day 15-16: Chat System (WhatsApp)
  Key concepts: WebSocket, message routing, delivery receipts, presence
  
Day 17-18: Search System (Typeahead + Search)
  Key concepts: trie, Elasticsearch, BM25, autocomplete
  
Day 19-20: YouTube / Video Streaming
  Key concepts: chunked upload, transcoding, CDN, HLS/DASH
  
Day 21-22: Ride-Sharing (Uber)
  Key concepts: geospatial indexing, matching, real-time location, ETA
  
Day 23-24: Hotel/Ticket Booking
  Key concepts: distributed locking, inventory management, double-booking prevention
  
Day 25-26: Payment System
  Key concepts: idempotency, saga pattern, 2PC, currency handling
  
Day 27-28: Review + Mock interview
```

### Week 5-6: Advanced Systems

```
Day 29-30: Distributed Cache (Redis)
Day 31-32: Distributed Message Queue (Kafka)
Day 33-34: Object Storage (S3)
Day 35-36: Google Maps / Location Services
Day 37-38: Metrics and Monitoring (Datadog)
Day 39-40: Ad Click Aggregation
Day 41-42: Mock interviews × 2
```

### Study Method for Each Problem

```
For each system, spend 2-3 hours:

30 min: Understand requirements
  - Read the problem carefully
  - Identify functional requirements
  - Identify non-functional requirements

45 min: Design independently
  - Draw the architecture without looking at answers
  - Estimate capacity
  - Think through deep-dive areas

45 min: Review and compare
  - Compare to reference solution
  - Note what you missed
  - Understand WHY different choices were made

30 min: Create your own notes
  - Write down 5 key insights
  - Note the trade-offs
  - Add to your component cheat sheet
```

---

## 15. Top 30 Most Common Questions

### Tier 1: Extremely Common (Master These First)

```
1.  Design a URL shortener (bit.ly)              ← Asked at almost every FAANG interview
2.  Design a news feed (Facebook/Twitter)         ← Fan-out, real-time, scale
3.  Design a chat system (WhatsApp/Messenger)     ← WebSocket, delivery guarantees
4.  Design YouTube / Netflix                      ← Video pipeline, CDN, streaming
5.  Design Twitter / Instagram                    ← Social graph, feed, search
6.  Design a notification system                  ← Multi-channel, fan-out, push
7.  Design an autocomplete / typeahead            ← Trie, ranking, Elasticsearch
8.  Design a rate limiter                         ← Algorithms, distributed, Redis
9.  Design a key-value store (Redis)              ← Consistent hashing, replication
10. Design a web crawler                          ← Distributed, politeness, dedup
```

### Tier 2: Very Common

```
11. Design Uber / Lyft (ride-sharing)
12. Design Airbnb / hotel booking
13. Design a distributed message queue (Kafka)
14. Design a payment system / Stripe
15. Design a search engine (Google-scale)
16. Design Google Drive / Dropbox (file storage)
17. Design a distributed cache
18. Design an API gateway / rate limiter
19. Design a stock trading system
20. Design a metrics and logging system (Datadog)
```

### Tier 3: Frequently Asked at Senior/Staff Level

```
21. Design a distributed job scheduler
22. Design ad click aggregation
23. Design a live streaming platform (Twitch)
24. Design a proximity-based recommendation (Yelp)
25. Design a distributed transaction system
26. Design a recommendation system (Netflix/Spotify)
27. Design a gaming leaderboard
28. Design Google Maps
29. Design a code deployment system (CI/CD pipeline)
30. Design a multi-player online game
```

---

## 16. Component Selection Cheat Sheet

### Kafka vs SQS vs RabbitMQ

| Feature | Kafka | SQS (AWS) | RabbitMQ |
|---------|-------|-----------|----------|
| Throughput | Very High (millions/sec) | High | Medium |
| Message Retention | Days to forever (log) | 14 days max | Until consumed |
| Replay | Yes (offset-based) | No | No |
| Fan-out | Topics + consumer groups | Multiple queues | Exchanges/bindings |
| Ordering | Per-partition ordering | FIFO queues (limited) | Per-queue ordering |
| Latency | Low (~10ms) | Medium (~100ms) | Low (~5ms) |
| Ops complexity | High | None (managed) | Medium |
| Use case | Event streaming, audit log | Decoupling microservices | Task queues, RPC |

**Choose Kafka when:** Need replay, event sourcing, high throughput, multiple consumers
**Choose SQS when:** AWS native, simple decoupling, don't want to manage infra
**Choose RabbitMQ when:** Complex routing, RPC patterns, moderate volume

### Redis vs Memcached

| Feature | Redis | Memcached |
|---------|-------|-----------|
| Data structures | Rich (strings, hashes, sets, sorted sets, streams) | Strings only |
| Persistence | RDB snapshots + AOF log | None |
| Replication | Master-replica + Sentinel + Cluster | Client-side only |
| Pub/Sub | Yes | No |
| Lua scripting | Yes | No |
| Horizontal scale | Redis Cluster | Built-in sharding |
| Memory efficiency | Slightly lower (overhead for types) | Slightly higher |

**Choose Redis when:** Sessions, leaderboards, pub/sub, rate limiting, queues, any complex structure
**Choose Memcached when:** Pure simple caching, need maximum memory efficiency for strings only

### PostgreSQL vs DynamoDB vs Cassandra

| Feature | PostgreSQL | DynamoDB | Cassandra |
|---------|-----------|---------|---------|
| Query model | Full SQL | Key/GSI only | CQL, partition+cluster keys |
| ACID | Full ACID | Single-table ACID, multi-table limited | Row-level; no cross-partition |
| Scale | Vertical + read replicas | Horizontal (auto) | Horizontal (consistent hashing) |
| Write throughput | Moderate (~10K/sec) | Very High (auto-scale) | Very High (~100K/sec) |
| Schema | Rigid (migrations) | Flexible (schemaless) | Semi-flexible |
| Joins | Yes | No | No |
| Global distribution | Manual or Citus | DynamoDB Global Tables | Multi-datacenter native |
| Operations | Self-managed or RDS | Fully managed | Self-managed or Astra |

**Choose PostgreSQL:** Relational data, ACID needed, complex queries, < few TB
**Choose DynamoDB:** AWS ecosystem, simple access patterns, need auto-scale, serverless
**Choose Cassandra:** Very high write throughput, time-series, wide-column data, multi-region

### Elasticsearch vs PostgreSQL Full-Text Search

| Feature | Elasticsearch | PG Full-Text |
|---------|--------------|-------------|
| Setup | Separate cluster | Same DB |
| BM25 scoring | Yes (built-in) | Custom (ts_rank) |
| Faceted search | Aggregations | GROUP BY (limited) |
| Autocomplete | Completion suggester | Trigram index |
| Scale | Horizontal sharding | Limited |
| Consistency | Near-real-time | Immediate |
| Operations | Complex | Simple |

**Choose ES:** Complex search, facets, large scale, analytics
**Choose PG FTS:** Simple search on small dataset, don't want another infra component

---

## 17. Numbers Every Engineer Must Know

### Latency Numbers (2024)

```
Operation                          | Latency
------------------------------------|----------
L1 cache reference                  | 0.5 ns
Branch mispredict                   | 5 ns
L2 cache reference                  | 7 ns
Mutex lock/unlock                   | 25 ns
Main memory reference               | 100 ns
Compress 1KB with Snappy            | 3,000 ns  (3 µs)
Send 2KB over 1Gbps network         | 20,000 ns (20 µs)
Read 1MB sequentially from memory   | 250 µs
Round trip within same datacenter   | 500 µs  (0.5 ms)
Disk seek                           | 10 ms
Read 1MB sequentially from SSD      | 1 ms
Read 1MB sequentially from disk     | 20 ms
Send packet CA → Netherlands → CA   | 150 ms
```

### Throughput and Capacity

```
Single-core CPU:                ~1 billion operations/sec
Goroutine / green thread:       handle ~10,000 concurrent
Thread (OS):                    handle ~10,000 total per server
Redis (single-threaded):        ~100,000 ops/sec
PostgreSQL (single primary):    ~10,000 writes/sec, ~100,000 reads/sec
Cassandra (per node):           ~50,000 writes/sec
Kafka (per partition):          ~1M messages/sec
SSD sequential read/write:      ~500 MB/sec read, ~200 MB/sec write
HDD sequential read/write:      ~100 MB/sec read, ~50 MB/sec write
1 Gbps network:                 ~125 MB/sec
10 Gbps network:                ~1.25 GB/sec
```

### Storage Sizes

```
1 KB   = 1,000 bytes         (1 tweet, 1 user metadata row)
1 MB   = 1,000,000 bytes     (1 photo thumbnail)
1 GB   = 10^9 bytes          (short video clip)
1 TB   = 10^12 bytes         (large DB, user photo library)
1 PB   = 10^15 bytes         (YouTube: ~1PB new video per day)
1 EB   = 10^18 bytes         (Google's total storage: ~15 EB)

Practical sizes:
  1 user row (SQL):     ~1 KB
  1 tweet:              ~300 bytes
  1 photo (Instagram):  ~3 MB (original), 100 KB (thumbnail)
  1 video (1 min 720p): ~100 MB
  1 page HTML:          ~50 KB
```

---

## 18. Interview Evaluation Rubric

### What Interviewers Actually Grade

```
1. Problem Clarification (10%)
   ✓ Asked about scale, constraints, priorities
   ✓ Defined scope before designing
   ✗ Jumped to design without clarifying
   ✗ Made assumptions without stating them

2. Estimation / Quantitative Thinking (10%)
   ✓ Calculated QPS, storage, bandwidth
   ✓ Used estimates to drive architecture decisions
   ✗ No estimation, or estimation not connected to design

3. High-Level Design (30%)
   ✓ Drew coherent architecture covering all main components
   ✓ Explained data flow clearly
   ✓ Identified right components for each function
   ✗ Missing key components (no cache, no queue when needed)
   ✗ Can't explain why each component is there

4. Deep Dive / Technical Depth (30%)
   ✓ Went deep on 1-2 components with implementation detail
   ✓ Discussed data models, algorithms, protocols
   ✓ Addressed failure modes and edge cases
   ✗ Stayed at high level throughout ("we'd use a database")
   ✗ Could not answer follow-up technical questions

5. Trade-Off Analysis (15%)
   ✓ Explicitly stated why you chose A over B
   ✓ Acknowledged limitations of your design
   ✓ Showed awareness of alternatives
   ✗ "It's the best approach" without reasoning
   ✗ No mention of what was given up

6. Communication and Collaboration (5%)
   ✓ Structured explanation, easy to follow
   ✓ Responsive to hints without being defensive
   ✓ Drew diagrams while explaining
   ✗ Silent for long periods
   ✗ Dismissed interviewer suggestions
```

### Levels of Performance

```
Not Hire:
  - Cannot design a basic working system
  - No estimation, no data modeling
  - Cannot handle simple follow-ups
  - Designs have obvious critical flaws

Hire (SDE-2 equivalent):
  - Designs working system for core requirements
  - Some estimation, basic data modeling
  - Handles straightforward follow-ups
  - Aware of most important trade-offs

Strong Hire (Senior SDE):
  - Comprehensive design covering edge cases
  - Accurate estimation driving architecture
  - Proactively addresses failure modes
  - Deep technical knowledge in at least 2 areas
  - Excellent trade-off reasoning

Principal/Staff Hire:
  - Above + novel insights or non-obvious optimizations
  - Drives conversation, identifies hidden constraints
  - Cross-system thinking (how design affects other systems)
  - Quantified trade-offs (latency/cost/consistency numbers)
```

---

## 19. Common Mistakes and How to Avoid Them

### Mistake 1: Jumping to a solution immediately

```
Bad: "I'll use microservices with Kubernetes and Kafka and..."
Good: "Before I start designing, let me clarify a few things..."
```

### Mistake 2: Designing a perfect system (over-engineering)

```
Bad: "I'll add a neural network to predict cache evictions..."
Good: "For now, LRU eviction is standard. We can optimize with ML if profiling shows a bottleneck."
```

### Mistake 3: Forgetting failure modes

```
Bad: "The system receives the message and processes it."
Good: "The system receives the message. If processing fails, we retry with exponential backoff. After 3 failures, we move to DLQ for manual inspection."
```

### Mistake 4: Not quantifying

```
Bad: "We'll use caching to improve performance."
Good: "With 80% cache hit rate on 350K QPS reads, we reduce DB load from 350K to 70K QPS, well within PostgreSQL's capacity."
```

### Mistake 5: Ignoring data consistency

```
Bad: "We write to DB and publish to Kafka."
Good: "We use the Outbox pattern: write to DB and outbox table in one transaction. A CDC (Debezium) process reads the outbox and publishes to Kafka — this prevents the dual-write inconsistency problem."
```

### Mistake 6: Inconsistent depth across components

```
Bad: (Spends 20 minutes on DB schema, 30 seconds on caching)
Good: Allocate time based on the most challenging and interesting components; check in with interviewer: "Should I go deeper on the DB schema or the caching strategy?"
```

### Mistake 7: Not considering the read/write ratio

```
Bad: "I'll add more database servers to handle load."
Good: "Our system is 95% reads. Adding read replicas and caching will 10x our capacity more cost-effectively than adding primaries."
```

---

## 20. Quick Reference

### Complete System Design Checklist

```
Phase 1 - Clarify (5 min):
  [ ] Scale: DAU, QPS, data volume
  [ ] Features: top 3 functional requirements
  [ ] Constraints: latency, consistency, availability SLA

Phase 2 - Estimate (5 min):
  [ ] Read QPS, Write QPS
  [ ] Storage: objects/day × size × retention
  [ ] Bandwidth: QPS × object size
  [ ] Cache size: hot data × hit target

Phase 3 - Design (10 min):
  [ ] API: key endpoints and schemas
  [ ] Data model: tables, keys, indexes
  [ ] Architecture diagram: LB, App, Cache, DB, Queue, CDN
  [ ] Happy path: trace a request end-to-end

Phase 4 - Deep Dive (15 min):
  [ ] Most complex component: data model detail
  [ ] Failure handling: what happens when X fails
  [ ] Scale bottleneck: where does it break, how to fix
  [ ] Edge cases: large users, spikes, geographic distribution

Phase 5 - Wrap-Up (5 min):
  [ ] Summarize: 3 key design decisions
  [ ] Trade-offs: what you gave up and why
  [ ] Future: what you'd improve with more time
```

### Component Decision Matrix (One-Line Guide)

```
Need to handle high read QPS:    → Read replicas + Redis cache
Need to handle high write QPS:   → Shard DB or switch to Cassandra
Need full-text search:           → Elasticsearch
Need real-time updates to client:→ WebSocket (bidirectional) or SSE (one-way)
Need to decouple services:       → Kafka (streaming) or SQS (simple queue)
Need to store files/media:       → S3 + CloudFront CDN
Need distributed locking:        → Redis (Redlock) or ZooKeeper
Need globally unique sorted IDs: → Snowflake ID
Need rate limiting:              → Redis token bucket
Need session storage:            → Redis with TTL
Need geospatial queries:         → PostGIS or geohashing
Need to prevent double-spend:    → DB transaction + SELECT FOR UPDATE
Need async job processing:       → Kafka + worker pool + DLQ
Need config/service discovery:   → etcd or ZooKeeper or Consul
Need analytics at scale:         → ClickHouse or BigQuery or Redshift
```

### Final Pre-Interview Cheat Codes

```
If asked about consistency, mention: "strong, eventual, or tunable (Cassandra)"
If asked about DB choice, mention: "read/write ratio, consistency needs, query patterns"
If asked about scaling, mention: "stateless + horizontal scale + cache + shard"
If asked about reliability, mention: "redundancy + retries + circuit breakers + monitoring"
If asked about performance, mention: "cache hot reads, async slow writes, CDN for media"
If asked about security, mention: "TLS, auth tokens, rate limiting, input validation, principle of least privilege"
```
