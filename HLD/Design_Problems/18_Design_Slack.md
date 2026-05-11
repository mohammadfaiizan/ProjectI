# Design Slack — Real-Time Team Messaging Platform

---

## 1. Problem Statement & Clarifying Questions

Design a team messaging platform like Slack that supports workspaces, channels, direct messages, real-time delivery, and rich features like threads, reactions, and search.

### Clarifying Questions

| Question | Assumption |
|---|---|
| How many DAU? | 10M DAU |
| How many messages per day? | 5 billion messages/day |
| Do we support file sharing? | Yes — up to 1GB files via S3 + CDN |
| Do we need message search? | Yes — full-text search across workspace |
| Do we support threads? | Yes — threaded replies on any message |
| What delivery guarantees? | At-least-once with client-side deduplication |
| Do we need message editing/deletion? | Yes — edit history retained, soft delete |
| Do we support emoji reactions? | Yes |
| How many concurrent connections? | 5M concurrent WebSocket connections |
| Retention policy? | Free tier: 90 days; paid: unlimited |

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
1. **Workspaces** — Isolated tenants; users belong to one or more workspaces
2. **Channels** — Public/private channels within workspaces; DMs between users
3. **Messages** — Send, edit, delete text messages with formatting
4. **Threads** — Reply to any message in a thread
5. **Real-Time Delivery** — Messages delivered via WebSocket < 200ms
6. **Emoji Reactions** — Add/remove emoji reactions on messages
7. **File Sharing** — Upload and share files in channels
8. **@Mentions** — Notify users when mentioned
9. **Presence** — Online/away/offline status per workspace
10. **Unread Counts** — Per-channel unread message tracking
11. **Search** — Full-text search across channels and DMs
12. **Slash Commands** — Extensible bot/integration framework

### Non-Functional Requirements
1. **Availability** — 99.99% (< 52 min downtime/year)
2. **Latency** — Message delivery < 200ms P99
3. **Scalability** — 10M DAU, 5B messages/day, 5M concurrent WebSocket connections
4. **Durability** — No message loss; Cassandra replication factor 3
5. **Consistency** — Eventual consistency for delivery; message ordering per channel guaranteed

---

## 3. Capacity Estimation

### Message Volume
- Messages per day: 5 billion
- Messages per second (peak, 3x average): ~175,000 msg/sec
- Average message size: 1 KB (text + metadata)
- Daily storage: 5B × 1 KB = **5 TB/day**
- Yearly: ~1.8 PB/year (compressed ~900 GB/year)

### Connection Estimation
- 5M concurrent WebSocket connections
- Each connection server handles 10,000 connections
- Connection servers needed: **500 servers**
- Each server: 4 vCPU, 16 GB RAM

### Channel Estimation
- 10M workspaces × 50 channels average = 500M channels
- 1000 messages/channel/day average
- Read:Write ratio ~10:1 (mostly reads/renders)

### Search Index
- 5B messages × 1 KB = 5 TB raw text/day
- Elasticsearch index overhead: ~3x → 15 TB/day of index data
- Hot index (30 days): 450 TB — requires large Elasticsearch cluster

---

## 4. High-Level Architecture

```
           ┌──────────────────────────────────────┐
           │          Slack Client (Browser/App)   │
           └──────────────┬───────────────────────┘
                          │ WebSocket (persistent)
           ┌──────────────▼───────────────────────┐
           │       Connection Server Pool          │
           │   (500 servers × 10K connections)     │
           └──────┬─────────────────┬─────────────┘
                  │ Redis Pub/Sub   │ REST
       ┌──────────▼──────┐  ┌──────▼──────────────┐
       │  Redis Cluster  │  │    API Gateway       │
       │  (pub/sub per   │  │    (REST + Auth)     │
       │   channel_id)   │  └──────┬───────────────┘
       └─────────────────┘         │
                          ┌────────▼────────────────────────┐
                          │         Microservices            │
                          │  ┌──────────┐  ┌─────────────┐ │
                          │  │ Message  │  │ Channel     │ │
                          │  │ Service  │  │ Service     │ │
                          │  └──────────┘  └─────────────┘ │
                          │  ┌──────────┐  ┌─────────────┐ │
                          │  │ Presence │  │ Notification│ │
                          │  │ Service  │  │ Service     │ │
                          │  └──────────┘  └─────────────┘ │
                          │  ┌──────────┐  ┌─────────────┐ │
                          │  │ Search   │  │ File        │ │
                          │  │ Service  │  │ Service     │ │
                          │  └──────────┘  └─────────────┘ │
                          └────────┬────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
     ┌────────▼───────┐  ┌─────────▼──────┐  ┌────────▼───────┐
     │   Cassandra    │  │  Elasticsearch │  │    S3 + CDN    │
     │  (messages,   │  │  (full-text    │  │   (files,      │
     │   reactions,  │  │   search       │  │   thumbnails)  │
     │   threads)    │  │   index)       │  └────────────────┘
     └────────────────┘  └────────────────┘
              │
     ┌────────▼───────┐
     │   PostgreSQL   │
     │  (workspaces,  │
     │   channels,    │
     │   users,       │
     │   members)     │
     └────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Message Storage — Cassandra Design

Cassandra is ideal because:
- Write-optimized (LSM tree) — perfect for append-only messages
- Partition by `channel_id` for fast channel history queries
- TIMEUUID for message ordering — no clock skew issues

**Read Patterns:**
1. Load channel history (most common): partition by `channel_id`, range query by time
2. Load thread replies: partition by `parent_message_id`

### 5.2 WebSocket Connection Architecture

**Problem:** User A and User B may be connected to different connection servers.

**Solution: Redis Pub/Sub as Message Bus:**
```
When User A sends a message to #general:
1. Message Service validates and persists to Cassandra
2. Message Service publishes to Redis topic: "channel:{channel_id}"
3. All connection servers subscribed to that topic receive the message
4. Each server delivers to locally-connected users who are in #general
```

**Why Redis Pub/Sub over Kafka here:**
- Ephemeral delivery (real-time only — historical messages are in Cassandra)
- Lower latency than Kafka
- No persistence needed — if user is offline, they'll catch up via REST on next login

### 5.3 Presence Service

Presence tracks online/away/offline status. Challenges:
- Must be workspace-scoped (you can be online in one workspace, DND in another)
- Heartbeat-based: client sends heartbeat every 30 seconds
- 5M × heartbeat/30s = 167K heartbeats/sec — significant load!

**Implementation:**
- Redis with TTL: `SET presence:{user_id}:{workspace_id} "online" EX 60`
- If key expires (no heartbeat) → user goes offline
- Redis pub/sub notifies workspace members of status changes
- Coalesce presence updates: batch 100ms before broadcasting

### 5.4 Unread Count Tracking

Per-user, per-channel unread tracking is critical for UX:

```
Redis: HSET unread:{user_id} {channel_id} {count}
  - Increment on new message posted to channel (if user is not currently viewing)
  - Reset to 0 when user opens the channel (marks as read)
```

**Challenge:** Fan-out on message to all channel members
- For channel with 1000 members: 1 message → 1000 Redis HSET commands
- Batched with Lua script to reduce round-trips
- For very large channels (> 10K members): lazy unread calculation on open

### 5.5 Message Delivery Guarantees

**At-Least-Once + Client Dedup:**
1. Client sends message with client-generated `idempotency_key`
2. Server persists and acks with `message_id`
3. If no ack (network failure), client retries same `idempotency_key`
4. Server deduplicates: `message_id` already exists → return existing message_id, don't store again

### 5.6 Search Architecture

Full-text search powered by Elasticsearch:
- Index populated asynchronously from Kafka (message created event → ES indexer)
- Supports: keyword search, @user filter, in-channel filter, date range
- Elasticsearch indices partitioned by workspace for multi-tenancy isolation
- Paid tier: full history indexed; Free tier: last 90 days only

### 5.7 Channel Router (Routing Messages to Connection Servers)

```
Message flow:
  1. Client sends via WebSocket to its connection server
  2. Connection server → HTTP POST to Message Service
  3. Message Service persists → publishes to Redis channel:{channel_id}
  4. All 500 connection servers that have subscribers for channel_{id} 
     receive the Redis message and push to their clients
```

---

## 6. Database Design

### 6.1 Workspaces (PostgreSQL)
```sql
CREATE TABLE workspaces (
    workspace_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name          VARCHAR(255) NOT NULL,
    slug          VARCHAR(100) UNIQUE NOT NULL,    -- Used in URLs
    owner_id      UUID NOT NULL,
    plan          ENUM('free','pro','enterprise') DEFAULT 'free',
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

### 6.2 Channels (PostgreSQL)
```sql
CREATE TABLE channels (
    channel_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id  UUID REFERENCES workspaces(workspace_id),
    name          VARCHAR(80) NOT NULL,
    is_private    BOOLEAN DEFAULT FALSE,
    is_dm         BOOLEAN DEFAULT FALSE,
    created_by    UUID,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (workspace_id, name)
);
```

### 6.3 Messages (Cassandra)
```cql
CREATE TABLE messages (
    channel_id    UUID,
    message_id    TIMEUUID,
    user_id       UUID,
    workspace_id  UUID,
    content       TEXT,
    thread_ts     TIMEUUID,       -- Parent message if this is a thread reply
    is_edited     BOOLEAN,
    is_deleted    BOOLEAN,
    file_ids      LIST<UUID>,
    created_at    TIMESTAMP,
    PRIMARY KEY (channel_id, message_id)
) WITH CLUSTERING ORDER BY (message_id DESC)
  AND default_time_to_live = 7776000;  -- 90 days for free tier
```

### 6.4 Reactions (Cassandra)
```cql
CREATE TABLE reactions (
    message_id    UUID,
    emoji         TEXT,
    user_id       UUID,
    created_at    TIMESTAMP,
    PRIMARY KEY (message_id, emoji, user_id)
);
```

### 6.5 Channel Members (PostgreSQL)
```sql
CREATE TABLE channel_members (
    channel_id    UUID REFERENCES channels(channel_id),
    user_id       UUID,
    role          ENUM('admin','member'),
    last_read_ts  TIMESTAMPTZ,
    joined_at     TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (channel_id, user_id)
);
```

---

## 7. API Design

### Send Message
```
POST /api/v1/channels/{channel_id}/messages
Body: { content, thread_ts (optional), file_ids (optional), idempotency_key }
Response: { message_id, created_at, channel_id }
```

### Get Channel History
```
GET /api/v1/channels/{channel_id}/messages?limit=50&before={message_id}
Response: { messages: [{message_id, user_id, content, created_at, reactions, thread_count}] }
```

### Add Reaction
```
POST /api/v1/messages/{message_id}/reactions
Body: { emoji }
Response: { message_id, emoji, count }
```

### Search
```
GET /api/v1/search?q={query}&workspace_id={id}&channel_id={id}&from={date}&to={date}
Response: { results: [{message_id, channel_id, content, highlighted_content, timestamp}] }
```

### Set Presence
```
PUT /api/v1/users/presence
Body: { workspace_id, status: "online"|"away"|"dnd" }
Response: { status, expires_at }
```

### Get Unread Counts
```
GET /api/v1/users/unread?workspace_id={id}
Response: { unread: [{channel_id, count, last_message_id}] }
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: Connection Server Fan-Out
- **Problem:** Popular channel #general with 50K members — 1 message fan-out to 50K WebSocket clients
- **Solution:** Two-tier fan-out. Tier 1: broadcast to connection servers via Redis pub/sub (O(500) publishes to servers). Tier 2: each server delivers to local connected users. Batch within 100ms windows.

### Bottleneck 2: Cassandra Hot Partitions
- **Problem:** Extremely active channel → single Cassandra partition gets hammered
- **Solution:** Partition by `(channel_id, bucket_id)` where bucket_id = floor(timestamp / 1_hour). Spread writes across multiple partitions per channel. Read merges results from time-ranged buckets.

### Bottleneck 3: Presence at Scale
- **Problem:** 5M online users × heartbeat/30s = 167K ops/sec to Redis
- **Solution:** Shard Redis presence cluster by user_id. 10 Redis shards → 16.7K ops/sec per shard — manageable.

### Bottleneck 4: Search Indexing Lag
- **Problem:** Kafka consumer for Elasticsearch indexing can fall behind during spikes
- **Solution:** Separate Kafka consumer groups for real-time and catch-up indexing. Real-time index has 30-second lag max. Dead letter queue for failed indexing ops.

### Bottleneck 5: DM Delivery (Two Users, Different Servers)
- **Problem:** Alice (server 1) sends DM to Bob (server 2)
- **Solution:** DMs are just channels with is_dm=true. Same pub/sub mechanism. Redis pub/sub channel: `channel:{dm_channel_id}` — both servers subscribed.

---

## 9. Trade-offs & Design Decisions

### Decision 1: Cassandra vs. PostgreSQL for Messages
- **Cassandra:** Write-optimized, horizontal scale, time-series friendly, eventual consistency
- **PostgreSQL:** ACID, complex queries, thread counts/aggregations easier
- **Choice:** Cassandra for messages. Scale requirement (5B msgs/day) mandates write-optimized storage. Thread counts maintained in Redis cache.

### Decision 2: Redis Pub/Sub vs. Kafka for Real-Time Delivery
- **Kafka:** Durable, ordered, replay. Overhead: consumer groups, offset management.
- **Redis Pub/Sub:** Ephemeral, fire-and-forget, lower latency. No delivery guarantee.
- **Choice:** Redis pub/sub for WebSocket fan-out (ephemeral, low-latency). Kafka for async processing (search indexing, notifications, analytics).

### Decision 3: Per-User vs. Per-Channel Unread Counting
- **Per-Channel Pointer:** Store `last_read_message_ts` per user per channel. Unread = count messages after that timestamp. Cheap on write, expensive on read.
- **Cached Counter:** Maintain explicit count in Redis, increment on new message. Cheap on read.
- **Choice:** Redis cached counter + PostgreSQL `last_read_ts` as source of truth on reconnect.

### Decision 4: Message Edit History
- **Overwrite:** Simple, loses history
- **Append Edits:** Store each edit as a new record
- **Choice:** Store original content + edit history as a JSON array field in Cassandra. View history on demand.

### Decision 5: Free-Tier Message Retention
- **Delete old messages:** Free users lose history after 90 days — drives upgrades
- **Full history always:** Expensive storage
- **Choice:** Cassandra TTL = 90 days for free tier. Paid tier: no TTL.

---

## 10. Key Interview Talking Points

1. **Cassandra Partition Key Design:** Partitioning messages by `channel_id` gives you all messages in a channel on one (set of) node(s). The TIMEUUID clustering key gives you time-ordered, globally unique message IDs without clock skew problems.

2. **Redis Pub/Sub for Cross-Server Delivery:** The insight is that you can't know which connection server a user is on. Redis pub/sub makes every server a subscriber to every channel it has users in — messages fan out automatically.

3. **Unread Count Trade-off:** Storing explicit counts in Redis is fast for the client but expensive to maintain on the write path (every message → fan out increment to all N channel members). For huge channels, switch to lazy computation (count on channel open).

4. **Presence is Heartbeat-Based:** TTL in Redis is your "health check." No heartbeat in 60 seconds → key expires → user goes offline. This is more reliable than explicit disconnect handling (network can silently drop connections).

5. **At-Least-Once + Idempotency Key:** The client generates a UUID before sending. On server, deduplicate by idempotency key. Client retries on timeout. Duplicate messages never reach the channel.

6. **Thread Model:** Thread replies are just messages with `thread_ts` set to the parent message's TIMEUUID. Thread summary (reply count, last reply preview) is a denormalized field on the parent message, updated asynchronously.

7. **Search is Eventually Consistent:** Elasticsearch index is updated asynchronously from Kafka. Messages appear in search ~5 seconds after posting. This is acceptable for search (users expect slight delay) but not for real-time delivery.

8. **Scale Math:** 5B msgs/day / 86,400 = 57,870 msg/sec average. Peak (3x) = 173,600 msg/sec. At 1 KB per message = 174 MB/sec write throughput. Cassandra cluster: 20 nodes × 10 MB/sec write = 200 MB/sec capacity — fits.

9. **DM vs. Channel Unification:** DMs are just channels with `is_dm=true` and exactly 2 members. This unification simplifies the entire messaging stack — one code path for all message types.

10. **Slack vs. Discord Architecture Difference:** Discord serves gaming communities with massive channels (100K+ members). Discord uses event-driven fan-out with gateway servers that shard users. Slack's workspace model has more bounded channel sizes, making per-channel pub/sub more tractable.
