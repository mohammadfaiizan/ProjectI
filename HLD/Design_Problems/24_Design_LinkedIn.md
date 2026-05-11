# System Design: LinkedIn

## 1. Problem Statement & Clarifying Questions

### Problem Statement
Design a professional social network like LinkedIn where users build professional profiles, connect with others, follow thought leaders, search for jobs, and consume a professional content feed.

### Clarifying Questions
1. **Scale**: 900M members — how many DAU? (~150M DAU)
2. **Graph depth**: For "People You May Know" and 2nd/3rd degree connections — how deep? (2nd degree primary, 3rd degree secondary signal)
3. **Feed type**: Timeline from connections, or algorithm-curated mix?
4. **Messaging**: InMail (paid, any member) vs. free messages (connections only)?
5. **Job search**: Active job applications tracked in system?
6. **Endorsements**: Do endorsements require mutual connection?
7. **Recommendations**: Written text recommendations vs. skill endorsements?
8. **Notifications**: Email + in-app? (both)
9. **Company pages**: Do companies have profiles?
10. **Search**: People + jobs + companies + content — all scopes?

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
- Users have profiles: name, headline, current role, education, skills, experience
- Connection system: send request → accept/decline (bidirectional friendship)
- Follow system: unidirectional follow (like Twitter) for content consumption
- 1st/2nd/3rd degree connection discovery and display
- "People You May Know" (PYMK) recommendations based on mutual connections + shared employer/school
- Professional content feed: posts from connections + followed topics + promoted content
- Job search with filters: location, industry, experience level, skills, remote
- Job application tracking: save, apply, track status
- Skill endorsements: connections can endorse your skills
- InMail: paid direct messaging to any member
- Notifications: connection requests, likes, mentions, job alerts
- Company pages with follower counts and job listings

### Non-Functional Requirements
- **Scale**: 900M members, 150M DAU, ~2B feed requests/day
- **Availability**: 99.99% uptime
- **Feed latency**: < 300ms p99
- **Graph queries**: 2nd degree connection lookup < 200ms
- **Job search**: < 200ms p99
- **Storage**: ~900M profiles × 5KB = ~4.5TB; graph edges ~50B × 16 bytes = ~800GB
- **Read/Write**: Feed 90:10, Graph 80:20

---

## 3. Capacity Estimation

### Traffic
- **Feed**: 150M DAU × 10 feed views/day = 1.5B → ~17K RPS average, 60K peak
- **Profile views**: 150M DAU × 5 views/day = 750M → ~8,700 RPS
- **Job searches**: 50M searches/day → ~580 RPS
- **Connection requests**: 10M/day → ~115 RPS
- **Posts**: 1M posts/day → ~12 RPS (write)

### Storage
- **Profiles**: 900M × 5KB = ~4.5TB
- **Connections**: 900M users × 500 avg connections × 16 bytes / 2 = ~3.6TB
- **Posts**: 1M/day × 1KB × 365 = ~365GB/year
- **Job listings**: 20M active × 2KB = ~40GB
- **Messages**: 500M messages/day × 200 bytes × 365 = ~36.5TB/year

### Graph Size
- 900M nodes
- ~900M × 500 / 2 = ~225B undirected edges
- At 16 bytes/edge: ~3.6TB raw graph data
- Graph partitioned across Neo4j or adjacency list in PostgreSQL

---

## 4. High-Level Architecture

```
                     ┌──────────────────────────────────────────────────────┐
                     │                     Clients                           │
                     │          Web / iOS / Android                          │
                     └─────────────────────┬────────────────────────────────┘
                                           │
                     ┌─────────────────────▼────────────────────────────────┐
                     │              API Gateway + Load Balancer              │
                     └──┬──────────┬───────────┬────────────┬───────────────┘
                        │          │           │            │
            ┌───────────▼──┐ ┌─────▼──────┐ ┌──▼───────┐ ┌▼──────────────┐
            │  Feed Svc    │ │ Graph Svc  │ │ Job Svc  │ │ Messaging Svc │
            └──────┬───────┘ └─────┬──────┘ └──┬───────┘ └───────┬───────┘
                   │               │            │                 │
         ┌─────────▼──────┐ ┌──────▼─────┐ ┌───▼──────┐  ┌──────▼──────┐
         │  Redis Sorted   │ │ Graph DB   │ │  ES      │  │  Kafka +    │
         │  Sets (feeds)   │ │ (Neo4j /   │ │  (Jobs + │  │  Cassandra  │
         │                 │ │ Adj. List) │ │  People) │  │  (Messages) │
         └─────────────────┘ └────────────┘ └──────────┘  └─────────────┘

  PYMK Service: graph BFS service → score by mutual connections
  Notification Service: event bus (Kafka) → push/email workers
  Search Service: Elasticsearch for people, companies, jobs, content
```

---

## 5. Component Deep-Dive

### 5.1 Professional Graph

LinkedIn's core is a professional graph:
- **Connections**: Undirected edges (both parties agree)
- **Follows**: Directed edges (you follow them, they don't need to follow back)
- **Company/School affiliations**: Hyperedges linking members to organizations

**Storage Options:**
1. **Adjacency List in PostgreSQL**: `connections(user_a BIGINT, user_b BIGINT)` with `user_a < user_b` invariant. B-tree index on both columns. Fast for 1st degree lookups.
2. **Graph Database (Neo4j)**: Native graph traversal, Cypher queries for BFS, but harder to scale horizontally.
3. **Custom adjacency list sharded by user_id**: LinkedIn's actual approach — shard users by user_id, store adjacency list per shard.

**2nd Degree BFS:**
```
Start: user U
Level 1: direct_connections(U) = set C1
Level 2: for each c in C1: direct_connections(c) - C1 - {U}
Deduplicate, sort by mutual_connection_count DESC
```
LinkedIn caches 2nd degree results (TTL 1 hour) since the graph doesn't change frequently.

### 5.2 People You May Know (PYMK)

PYMK scoring algorithm:
```
pymk_score(candidate) = 
  mutual_connections × 10
  + same_company × 25
  + same_school × 15
  + same_industry × 5
  + profile_completeness × 2
  + recently_connected_to_your_connection × 8
```

**Offline vs Online computation:**
- **Offline (batch)**: Spark job runs nightly, computes PYMK list for all users, stores in Cassandra. Fresh once per day.
- **Online (real-time)**: On profile page load, BFS 2nd degree and score. Fresh but slower.
- **Chosen**: Hybrid — offline batch for most users, real-time BFS for recent connection changes

### 5.3 Feed System

LinkedIn uses a **push-pull hybrid** feed:
- **Push (fanout on write)**: When user with < 1000 followers posts, push to all followers' feed caches
- **Pull (fanout on read)**: When celebrity/influencer with 1M followers posts, don't push to all feeds; pull on read
- Threshold: ~1000 followers → switch from push to pull
- Feed items stored in Redis sorted set per user: `feed:{user_id}` → ZADD score=timestamp

**Feed Ranking:**
- LinkedIn feed is not purely chronological; uses engagement signals:
  - Recency (newer = higher base score)
  - Relationship strength (close connection = multiplier)
  - Content type (video > article > text)
  - Past engagement (user liked similar content before)

### 5.4 Job Search

Elasticsearch powers job search with:
- Full-text on job title, description, company name
- Faceted filters: location (geo), skills (keyword), experience level (range), remote (boolean)
- Personalized ranking: match user's skills/experience to job requirements
- Job alerts: Kafka-based notification when new jobs match saved search criteria

**Skill Matching Score:**
```
match_score = |user_skills ∩ job_required_skills| / |job_required_skills|
```
Jobs with > 70% skill match ranked higher.

### 5.5 Endorsements

- Connection endorses one of your skills → creates `endorsement(endorser_id, endorsee_id, skill)`
- Aggregate endorsement count stored in profile (de-normalized)
- Top 3 skills by endorsement count displayed prominently
- Endorsements require mutual 1st degree connection (prevent spam)
- Users can reorder skills on their profile; endorsement count remains

### 5.6 Messaging (InMail)

**Regular message**: Both users must be 1st degree connections
**InMail**: Paid feature; any member can message any member
- InMail credits: premium members get N credits/month
- InMail delivery: stored in Cassandra (wide-column) for high write throughput
- Message threads: Cassandra partition key = min(user_a, user_b), clustering key = timestamp
- Real-time delivery: WebSocket connection for online users; push notification for mobile

---

## 6. Database Design

### Users Table
```sql
CREATE TABLE users (
    id              BIGSERIAL PRIMARY KEY,
    email           VARCHAR(255) UNIQUE NOT NULL,
    name            VARCHAR(200),
    headline        VARCHAR(220),
    location        VARCHAR(100),
    profile_photo   VARCHAR(500),
    is_premium      BOOLEAN DEFAULT false,
    connection_count INT DEFAULT 0,
    follower_count  INT DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

### Connections Table (Graph)
```sql
CREATE TABLE connections (
    user_a      BIGINT NOT NULL,    -- always user_a < user_b
    user_b      BIGINT NOT NULL,
    status      VARCHAR(10) DEFAULT 'pending',  -- pending, accepted, declined
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_a, user_b)
);
CREATE INDEX idx_connections_a ON connections(user_a) WHERE status='accepted';
CREATE INDEX idx_connections_b ON connections(user_b) WHERE status='accepted';
```

### Experience & Education (for PYMK signals)
```sql
CREATE TABLE experiences (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT REFERENCES users(id),
    company_id  BIGINT,
    title       VARCHAR(200),
    start_year  SMALLINT,
    end_year    SMALLINT,
    is_current  BOOLEAN DEFAULT false
);

CREATE TABLE education (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT REFERENCES users(id),
    school_id   BIGINT,
    degree      VARCHAR(100),
    field       VARCHAR(100),
    grad_year   SMALLINT
);
```

### Skills & Endorsements
```sql
CREATE TABLE skills (
    id      BIGSERIAL PRIMARY KEY,
    name    VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE user_skills (
    user_id     BIGINT REFERENCES users(id),
    skill_id    BIGINT REFERENCES skills(id),
    endorsement_count INT DEFAULT 0,
    display_order INT,
    PRIMARY KEY (user_id, skill_id)
);

CREATE TABLE endorsements (
    endorser_id BIGINT REFERENCES users(id),
    user_id     BIGINT REFERENCES users(id),
    skill_id    BIGINT REFERENCES skills(id),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (endorser_id, user_id, skill_id)
);
```

### Jobs Table
```sql
CREATE TABLE jobs (
    id              BIGSERIAL PRIMARY KEY,
    company_id      BIGINT,
    title           VARCHAR(200),
    description     TEXT,
    location        VARCHAR(200),
    lat             DECIMAL(9,6),
    lon             DECIMAL(9,6),
    remote_type     VARCHAR(10),    -- onsite, remote, hybrid
    experience_min  SMALLINT,
    experience_max  SMALLINT,
    skills_required TEXT[],
    is_active       BOOLEAN DEFAULT true,
    posted_at       TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 7. API Design

### Connection API
```
POST /api/v1/connections/request
Body: { target_user_id }
Response: { connection_request_id, status: "pending" }

PUT /api/v1/connections/request/{request_id}
Body: { action: "accept" | "decline" }

GET /api/v1/users/{user_id}/connections?limit=20&cursor=...
GET /api/v1/users/{user_id}/connections/2nd-degree?limit=20
GET /api/v1/users/{user_id}/pymk?limit=20
```

### Feed API
```
GET /api/v1/feed?limit=20&cursor=...
Response: { posts: [...], cursor: "...", sponsored: [...] }

POST /api/v1/posts
Body: { content, media_ids[], visibility: "connections"|"public" }

POST /api/v1/posts/{post_id}/reactions
Body: { reaction_type: "like"|"celebrate"|"support"|"love"|"insightful"|"funny" }
```

### Job API
```
GET /api/v1/jobs/search?q=...&location=...&lat&lon&radius_km&skills=...&remote=...&exp_min&exp_max
GET /api/v1/jobs/{job_id}
POST /api/v1/jobs/{job_id}/apply
Body: { resume_id, cover_letter }
GET /api/v1/jobs/saved
POST /api/v1/jobs/{job_id}/save
GET /api/v1/jobs/recommendations?limit=10
```

### Messaging API
```
POST /api/v1/messages
Body: { recipient_id, body, type: "message"|"inmail" }
GET /api/v1/messages/threads?limit=20
GET /api/v1/messages/threads/{thread_id}?limit=50&before=message_id
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: Graph BFS at Scale
- 2nd degree BFS on 900M users with avg 500 connections = up to 500 × 500 = 250K nodes scanned
- Solution: Cache 2nd degree list in Redis (TTL 1 hour); use graph DB (Neo4j) for native BFS
- PYMK computed offline nightly in Spark for bulk, real-time for recent users

### Bottleneck 2: Feed Generation
- 150M DAU × 10 feed loads = 1.5B feed requests/day
- Push model for normal users, pull for celebrities
- Redis sorted set per user stores feed item IDs; actual post content fetched separately
- Redis cluster sharded by user_id: 150M × 200 feed items × 16 bytes = ~480GB in Redis

### Bottleneck 3: Connection Graph Storage
- 225B edges × 16 bytes = 3.6TB — doesn't fit in single PostgreSQL
- Solution: Shard connections table by user_a's user_id range
- Graph traversal that crosses shards uses scatter-gather pattern

### Bottleneck 4: Job Search Personalization
- 20M active jobs; personalize for 150M DAU = cannot precompute all combinations
- Solution: Compute match score at query time using Elasticsearch function_score
- User's top 10 skills stored in search request; script scores each job by skill overlap

### Bottleneck 5: Notification Thundering Herd
- A viral post from a 5M-follower influencer triggers 5M notifications
- Solution: Kafka topic per notification type; consumer groups process in parallel
- Rate limit: max 20 notifications/day per user (beyond that, digest)

---

## 9. Trade-offs & Design Decisions

### Decision 1: Graph Storage — SQL vs Graph DB
- **PostgreSQL adjacency list**: Familiar ops, ACID, good for 1st degree queries
- **Neo4j**: Native BFS, Cypher language, but operationally complex and harder to shard
- **Custom sharded adjacency list**: LinkedIn's actual approach — user partition, local BFS within shard, merge across shards
- **Choice**: PostgreSQL adjacency list (interview simplicity) with Redis cache for BFS results

### Decision 2: Feed — Push vs Pull
- **Push (fanout on write)**: Fast reads, but expensive for celebrities (1M followers → 1M writes per post)
- **Pull (fanout on read)**: Simple writes, but slow reads (merge N followed users' posts per feed request)
- **Choice**: Hybrid — push for users with < 1000 followers, pull for celebrities; merge at read time
- **Trade-off**: Feed consistency: push feeds are near-real-time, pull feeds may miss very recent posts

### Decision 3: PYMK — Batch vs Real-time
- **Batch**: Accurate, offline computed, but stale (up to 24h old)
- **Real-time**: Always fresh, but expensive BFS on every profile load
- **Choice**: Batch for 90% of users; real-time triggered by new connection events
- **Trade-off**: New connection won't appear in PYMK immediately (up to 24h for batch refresh)

### Decision 4: Endorsement Constraints
- **Any user can endorse**: Open to spam; gaming the system
- **Only connections can endorse**: Reduces spam but limits reach
- **Choice**: Only 1st degree connections; max 1 endorsement per skill per endorser
- **Trade-off**: Reduces endorsement diversity; power users may game through connect → endorse → disconnect

### Decision 5: InMail vs Free Messages
- Unlimited free messages → spam problem (sales people would abuse)
- Strict connection requirement → reduces discovery and opportunity
- **Choice**: Free messages for 1st degree connections; InMail (paid credits) for non-connections
- **Trade-off**: Creates friction for legitimate cold outreach

---

## 10. Key Interview Talking Points

### 1. BFS for Degree Discovery
LinkedIn's most interesting algorithmic problem. BFS with depth limit 2 for 2nd degree:
```
Level 0: {user}
Level 1: adjacency_list[user]
Level 2: union of adjacency_list[c] for c in Level 1, minus Level 0 + Level 1
```
At 500 avg connections: Level 2 = up to 250K candidates. In practice, significant overlap means ~50K unique 2nd degree connections.

### 2. PYMK Algorithm
Don't just say "mutual connections." The complete signal set:
- Mutual connections count (strongest signal)
- Current/past same employer (LinkedIn mines this from profiles)
- Same school + overlapping years
- Same industry
- Both connected to same set of people (density)
- Profile similarity score

### 3. Feed Push-Pull Hybrid
The celebrity problem: with push model, a post from user with 1M followers creates 1M writes. LinkedIn's solution:
- Users < N followers: push their posts to followers' feed caches
- Users > N followers: don't push; followers pull on feed load
- Feed assembly merges both types at read time

### 4. Graph Scaling
900M nodes, ~225B edges → can't fit in single machine. Options:
- LinkedIn uses a custom in-house graph store (Leo) and offline Hadoop/Spark for PYMK
- Interview answer: shard by user_id, 2nd degree queries may cross shards (scatter-gather)
- Neo4j for smaller scale; LinkedIn-scale needs custom sharded solution

### 5. Skill Endorsement Gaming
A naive endorsement system can be gamed: create accounts, endorse each other. Defenses:
- Only 1st degree connections can endorse
- Account age requirement (> 30 days)
- Max 1 endorsement per skill per endorser
- Endorsements from high-karma connections weighted more

### 6. Job Recommendation Personalization
The skill match score formula is the key: intersection of user skills vs. required job skills. Beyond skills:
- Location proximity (job in user's city ranked higher)
- Company connection count (you know someone there = warm intro signal)
- Career trajectory match (next logical role in career progression)
- Application success rate (jobs where similar profiles got responses)

### 7. Scale Numbers
- 900M members, 150M DAU
- 2B feed requests/day = 23K RPS average
- Graph: 225B edges, 3.6TB raw storage
- Feed Redis storage: ~480GB for 150M users × 200 items × 16 bytes
