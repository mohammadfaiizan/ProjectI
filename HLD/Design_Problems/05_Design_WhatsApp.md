# System Design: WhatsApp (Messaging Platform)

---

## 1. Problem Statement

Design a messaging platform like WhatsApp. Users can send one-on-one messages and participate in group chats. Messages must be delivered reliably, including when recipients are offline (with offline queuing). The system must support message status tracking (sent, delivered, read), presence (online/offline/last seen), media sharing, and end-to-end encryption. Scale to 2 billion users sending 100 billion messages per day.

---

## 2. Clarifying Questions to Ask

- What is the scale? (DAU, messages/day)
- Do we need group messaging? If yes, what is the max group size?
- What media types must be supported? (photos, videos, voice notes, documents)
- Must we support end-to-end encryption?
- How long should messages be retained? (WhatsApp: indefinitely for delivered, 30 days for offline)
- Do we need message editing or deletion (for me vs for everyone)?
- What is the acceptable delivery latency? (< 1 second for online users)
- Do we need read receipts? (double tick = delivered, blue tick = read)
- Do we need voice/video calling?
- What platforms? (iOS, Android, Web)

---

## 3. Functional Requirements

1. One-on-one messaging between users.
2. Group messaging (up to 1,024 members per group).
3. Message status: sent (server received) → delivered (recipient got it) → read (recipient opened).
4. Offline message storage: messages queued when recipient is offline, delivered when they reconnect.
5. Presence tracking: online, offline, last seen timestamp.
6. Media sharing: photos, videos, voice notes, documents (up to 16MB per file).
7. Push notifications for offline users.
8. Message ordering guaranteed within a conversation.
9. End-to-end encryption (E2EE) — server cannot read message content.
10. Web and mobile client support with message sync across devices.

---

## 4. Non-Functional Requirements

- **Availability**: 99.99% — message delivery must be highly available
- **Latency**: Online-to-online message P99 < 500ms; P50 < 100ms
- **Durability**: Delivered messages must never be lost; offline queue persisted
- **Consistency**: Messages within a conversation must maintain order (FIFO)
- **Scalability**: 2B users, 100B messages/day, 50M concurrent online users
- **Security**: E2EE using Signal Protocol; server sees only metadata
- **Storage**: Messages retained until device download; server purges after delivery confirmation

---

## 5. Capacity Estimation

### Scale
- Total users: 2B
- DAU: 500M
- Concurrent online users: 50M (10% of DAU)
- Messages per day: 100B
- Messages per second: 100B / 86,400 = ~1.16M messages/sec
- Peak message QPS: ~2.5M messages/sec

### Per-User Stats
- Average messages sent per DAU: 200 messages/day
- Average message size (text): 100 bytes (compressed, encrypted)
- Average media message size: 200 KB (compressed thumbnail + S3 URL)

### Storage
- Text messages: 100B * 100B = 10 TB/day
- With metadata (conversation_id, sender, timestamp, status): ~300 bytes/msg → 30 TB/day
- Annual: 30 TB * 365 = ~11 PB/year
- Cassandra with 3x replication: ~33 PB/year
- Media in S3: 100B * 5% media * 200KB = 1 PB/day → separate cold storage tier

### Connections
- 50M concurrent WebSocket connections
- Each connection server handles 50K connections → need 1,000 connection servers
- Connection server memory: 50K * 50KB per connection = ~2.5 GB per server (feasible)

### Bandwidth
- Inbound: 2.5M msgs/sec * 300B = 750 MB/s
- Outbound: same (1:1 messaging) plus group fan-out = ~3 GB/s
- WebSocket keep-alive: negligible (30s ping/pong frames)

---

## 6. High-Level Architecture

```
[Client App]
    |
    | (TLS WebSocket)
    v
[Connection Server Cluster]         [Push Notification Service]
(1,000 servers, 50K conn each)           (APNs / FCM)
    |                                           ^
    | (internal gRPC)                           |
    v                                           |
[Message Router]                      [Notification Service]
    |         |                                 |
    v         v                                 |
[Message   [Presence                            |
 Service]   Service]                            |
    |           |                               |
    v           v                               |
[Cassandra]  [Redis]                            |
(messages)  (online users,                      |
             last seen,                         |
             offline queue)                     |
    |                                           |
    v                                           |
[Media Service] --> [S3 / Object Store]         |
[CDN (CloudFront)]                              |
                                                |
[Kafka] <-- events (msg delivered, read)    ----+
```

### Message Send Flow (Online Recipient)
```
1. Sender's client sends msg via WebSocket to Connection Server A
2. Connection Server A → Message Service (store + route)
3. Message Service stores encrypted msg in Cassandra
4. Message Service queries Presence Service: is recipient online?
5. YES: Route to recipient's Connection Server B via internal RPC
6. Connection Server B pushes msg to recipient's WebSocket
7. Recipient client sends "delivered" ACK
8. Message Service updates msg status in Cassandra
9. Sender's client notified of delivered status (double tick turns grey→grey)
```

### Message Send Flow (Offline Recipient)
```
1-4. Same as above
5. NO: Message Service writes to offline message queue (Redis or Cassandra)
6. Notification Service sends push notification via APNs/FCM
7. When recipient comes online → Connection Server fetches offline queue
8. Delivers all queued messages in order
9. Sends "delivered" ACKs back to senders
```

---

## 7. Component Deep-Dive

### 7.1 WebSocket Connection Management

**Why WebSocket over HTTP?**
- HTTP is request-response: client must poll for new messages (wasteful, high latency)
- WebSocket is full-duplex: server can push messages to client instantly
- Long-lived connection maintained for the session

**Connection Server Architecture**:
- Each connection server maintains a map: `{user_id → WebSocket connection object}`
- When message arrives for user X, look up X's connection server (from Presence Service)
- Forward message to that specific server via internal gRPC
- Connection server pushes to client

**Scaling WebSocket servers**:
- Stateful — connections are "sticky" (client always connects to same server)
- Use consistent hashing: user_id → connection server assignment
- If a server goes down: clients reconnect via WebSocket reconnection logic (with backoff)

**Connection server selection**:
- Client gets assigned server from a Connection Router based on user_id hash
- Allows load balancing while keeping session affinity

### 7.2 Message Delivery & Storage (Cassandra)

**Why Cassandra?**
- 100B messages/day = 1.16M writes/sec → Cassandra excels at high write throughput
- Messages are naturally partitioned by conversation_id
- Time-series data: messages ordered by timestamp within a conversation
- Linear horizontal scalability
- Multi-datacenter replication for global availability

**Message ordering**:
- Use TIMEUUID (UUID Version 1 with timestamp) as message_id
- TIMEUUID is monotonically increasing per node
- Cassandra clustering key on message_id DESC → newest messages first
- Client assigns a local sequence number; server assigns canonical order

**Message TTL**:
- Undelivered messages: TTL = 30 days (Cassandra TTL feature)
- Delivered messages: Clients are the primary storage; server may keep 30 days for sync
- Media: S3 lifecycle policies move to Glacier after 90 days, delete after 1 year

### 7.3 Message Status System (Sent / Delivered / Read)

```
PENDING (client) → SENT (server received) → DELIVERED (recipient got it) → READ (opened)
```

**Visual representation (WhatsApp-style)**:
- Single grey tick: SENT (server stored it)
- Double grey tick: DELIVERED (recipient's device received it)
- Double blue tick: READ (recipient opened the conversation)

**Implementation**:
- Status field in messages table: 0=pending, 1=sent, 2=delivered, 3=read
- When recipient's device receives msg: client sends `{type: "delivered", msg_id: X}`
- When user opens conversation: client sends `{type: "read", conversation_id: X, up_to_msg_id: Y}`
- Server updates Cassandra and notifies original sender

**Group read receipts**:
- Track per-member delivery: `{msg_id, user_id, status, timestamp}`
- Message shows delivered only when ALL members have received it
- Message shows read when ALL members have read it (or show individual status)

### 7.4 Presence Service

**Tracking online/offline status**:
- User comes online → client establishes WebSocket → Presence Service marks user ONLINE in Redis
- `HSET presence:{user_id} status online last_seen {timestamp} server_id {conn_server_id}`
- TTL: 30 seconds (refreshed by heartbeat ping every 15 seconds)
- If heartbeat stops → TTL expires → user marked offline
- `last_seen` timestamp stored in Redis for "last seen today at 3:00pm"

**Privacy controls**:
- User can hide last seen (only status change: no last_seen timestamp shown to others)
- "Last seen recently", "last seen today", "last seen this week" — bucketed for privacy

**Scaling presence**:
- 50M concurrent users → 50M Redis keys → ~500 MB (fits in Redis Cluster easily)
- Presence queries: O(1) Redis HGET

### 7.5 End-to-End Encryption (Signal Protocol Overview)

**Key concepts**:
- E2EE means only sender and recipient can read messages; WhatsApp server cannot
- Based on Signal Protocol (open source, audited)
- Uses Double Ratchet Algorithm for perfect forward secrecy

**Key exchange**:
1. Each device generates: identity key, signed prekey, one-time prekeys
2. Public keys uploaded to WhatsApp server (server stores public keys only)
3. When Alice sends first message to Bob:
   - Alice fetches Bob's public keys from server
   - Alice derives shared secret using X3DH (Extended Triple Diffie-Hellman)
   - Alice encrypts message with derived key
4. Bob receives encrypted message and derives same shared secret independently
5. Server only sees: {sender_id, recipient_id, ciphertext, timestamp} — cannot decrypt

**Double Ratchet**:
- Every message uses a new encryption key derived from previous key
- Compromise of one key doesn't compromise past or future messages
- "Perfect forward secrecy"

### 7.6 Group Messaging Fan-out

**When user sends message to group with N members**:
- Server stores one copy of the message
- Fans out to N-1 other members' message queues
- For N=1024: 1024 writes per group message
- Fan-out workers: dedicated Kafka consumers for group message delivery
- At 1M group messages/day with avg 100 members: 100M fan-out writes/day

**Group message storage**:
- Conversation-level storage: one partition per group_id in Cassandra
- Each member fetches messages from the group's conversation partition
- Avoids N copies of the same message (store once, fan-out delivery status)

### 7.7 Media Sharing Pipeline

1. Client compresses media (photo: up to 720px, video: up to 720p)
2. Client generates AES-256 key, encrypts media locally
3. Client uploads encrypted media directly to S3 (pre-signed URL)
4. Client sends message with {media_url, encryption_key, hash} — key is E2EE-wrapped
5. Recipient downloads encrypted media from CDN, decrypts locally with key from message
6. CDN caches encrypted media (server cannot decrypt — it's encrypted ciphertext)

**Media deduplication**:
- Hash of plaintext media stored client-side
- Same video forwarded to 1000 groups: uploaded once, referenced 1000 times

---

## 8. Database Design

### Cassandra: messages
```cql
CREATE TABLE messages (
    conversation_id UUID,              -- partition key: could be user_pair or group_id
    message_id      TIMEUUID,          -- clustering key: time-sortable UUID
    sender_id       BIGINT,
    content         BLOB,              -- encrypted ciphertext (E2EE)
    message_type    VARCHAR,           -- text, image, video, audio, document
    media_url       TEXT,
    status          TINYINT,           -- 0=sent, 1=delivered, 2=read
    is_deleted      BOOLEAN,
    created_at      TIMESTAMP,
    PRIMARY KEY (conversation_id, message_id)
) WITH CLUSTERING ORDER BY (message_id DESC)
  AND default_time_to_live = 2592000;  -- 30 days TTL for undelivered
```

### Cassandra: message_receipts (for group messages)
```cql
CREATE TABLE message_receipts (
    message_id      TIMEUUID,
    user_id         BIGINT,
    status          TINYINT,
    updated_at      TIMESTAMP,
    PRIMARY KEY (message_id, user_id)
);
```

### Cassandra: conversations
```cql
CREATE TABLE user_conversations (
    user_id         BIGINT,
    conversation_id UUID,
    last_message_id TIMEUUID,
    unread_count    INT,
    updated_at      TIMESTAMP,
    PRIMARY KEY (user_id, updated_at)
) WITH CLUSTERING ORDER BY (updated_at DESC);
```

### PostgreSQL: users
```sql
CREATE TABLE users (
    id              BIGSERIAL PRIMARY KEY,
    phone_number    VARCHAR(20) UNIQUE NOT NULL,
    name            VARCHAR(100),
    about           TEXT,
    profile_pic_url TEXT,
    last_seen       TIMESTAMP,
    privacy_last_seen VARCHAR(20) DEFAULT 'everyone',
    created_at      TIMESTAMP DEFAULT NOW()
);
```

### PostgreSQL: groups
```sql
CREATE TABLE groups (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(100) NOT NULL,
    description     TEXT,
    icon_url        TEXT,
    created_by      BIGINT REFERENCES users(id),
    created_at      TIMESTAMP DEFAULT NOW(),
    max_members     INT DEFAULT 1024
);

CREATE TABLE group_members (
    group_id        UUID REFERENCES groups(id),
    user_id         BIGINT REFERENCES users(id),
    role            VARCHAR(20) DEFAULT 'member',  -- member, admin
    joined_at       TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (group_id, user_id)
);
```

### Redis Data Structures
```
presence:{user_id}          → Hash { status, last_seen, server_id }  TTL=30s
offline_queue:{user_id}     → List of message_ids (FIFO)             TTL=30days
unread:{user_id}:{conv_id}  → Integer (unread count)
conn_server:{user_id}       → String (connection_server_id)
```

---

## 9. API Design

### Send Message (WebSocket frame)
```json
// Client → Server
{
  "type": "message",
  "client_msg_id": "uuid-locally-generated",
  "conversation_id": "conv_abc123",
  "content": "<encrypted_base64>",
  "message_type": "text",
  "timestamp": 1705312200000
}

// Server → Sender (ACK)
{
  "type": "ack",
  "client_msg_id": "uuid-locally-generated",
  "server_msg_id": "timeuuid-from-cassandra",
  "status": "sent"
}

// Server → Recipient
{
  "type": "message",
  "server_msg_id": "timeuuid",
  "conversation_id": "conv_abc123",
  "sender_id": 12345,
  "content": "<encrypted_base64>",
  "timestamp": 1705312200001
}
```

### Message Status Update (WebSocket frame)
```json
// Recipient → Server (delivered)
{ "type": "delivered", "message_id": "timeuuid" }

// Recipient → Server (read)
{ "type": "read", "conversation_id": "conv_abc123", "up_to": "timeuuid" }

// Server → Original Sender
{ "type": "status_update", "message_id": "timeuuid", "status": "read" }
```

### Get Message History (REST)
```
GET /api/v1/conversations/{conv_id}/messages?before={timeuuid}&limit=50
Authorization: Bearer {token}

Response 200:
{
  "messages": [
    {
      "message_id": "timeuuid",
      "sender_id": 12345,
      "content": "<encrypted>",
      "message_type": "text",
      "status": 2,
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "has_more": true
}
```

### Presence Status (REST)
```
GET /api/v1/users/{user_id}/presence

Response 200:
{
  "user_id": 12345,
  "status": "online",
  "last_seen": "2024-01-15T10:30:00Z"
}
```

---

## 10. Scalability & Bottlenecks

### Bottleneck 1: WebSocket Connection Scale (50M concurrent)
- 1 connection server handles ~50K connections
- Need 1,000 connection servers
- Load balancer uses sticky sessions (consistent hashing by user_id)
- Horizontal scaling: add more connection servers; rehash reassignments

### Bottleneck 2: Message Write Throughput (1.16M writes/sec)
- Cassandra with 20+ nodes handles this easily (100K writes/sec per node)
- Write ahead log + eventual consistency
- Batch writes with async acknowledgment

### Bottleneck 3: Fan-out for Large Groups (1024 members)
- 1 group message → 1024 write operations (to each member's queue or conversation)
- Async fan-out via Kafka consumers
- For very large groups: store message once, clients pull from group partition

### Bottleneck 4: Presence at Scale (50M users)
- 50M Redis HSET operations per session start (manageable)
- Heartbeat: 50M * 1 ping/15s = 3.3M Redis ops/sec → need Redis Cluster
- Gossip protocol alternative: distributed presence using consistent hashing

### Bottleneck 5: Offline Queue Delivery Order
- User comes online → must deliver messages in order
- Redis list per user maintains FIFO order
- Cassandra query: `SELECT * WHERE conversation_id = X AND message_id > last_seen_id`

---

## 11. Trade-offs & Design Decisions

### Cassandra vs MySQL for Messages
- MySQL: Strong consistency, but doesn't scale to 1.16M writes/sec
- Cassandra: Designed for exactly this write pattern (time-series, partition by conversation)
- Decision: Cassandra for messages; PostgreSQL for user/group metadata

### TIMEUUID vs Auto-increment for Message IDs
- Auto-increment: Sequential, predictable, but single point of failure
- TIMEUUID: Distributed generation, time-sortable, globally unique
- Decision: TIMEUUID — aligns with Cassandra clustering, no coordination needed

### Store Messages on Server vs Client-only
- Client-only (pure E2EE): No server storage, maximum privacy
- Server-stored (WhatsApp approach): Server stores encrypted messages until delivery confirmed
- Decision: Store encrypted on server with TTL; purge after delivery + 30 days

### Push vs Long-polling for Offline Notifications
- Long-polling: Simple but wasteful (keeps connection open)
- Push notifications (APNs/FCM): Battery-efficient, platform-optimized
- WebSocket: Best for online, but drain battery if persistent
- Decision: WebSocket when app is in foreground; push notifications for background/offline

### Group Message Storage: Fan-out vs Shared
- Fan-out: Each member has their own copy → simple reads, expensive writes
- Shared (one copy per group message): Storage efficient, requires group membership check on read
- Decision: Shared Cassandra partition per group_id with per-member status tracking

---

## 12. Key Interview Talking Points

1. **WebSocket is non-negotiable**: 1.16M messages/sec with < 500ms latency is impossible with HTTP polling.

2. **Cassandra for messages**: Time-series, high write volume, partition by conversation_id — classic Cassandra use case. TIMEUUID as clustering key gives ordering for free.

3. **Two-phase delivery**: Message stored on server (sent) → delivered to device → status update sent back. This guarantees no message loss.

4. **TTL for offline messages**: Cassandra's native TTL feature — messages auto-expire after 30 days. Mention this to show database feature awareness.

5. **Presence heartbeat**: 15-second heartbeat with 30-second TTL in Redis. If network drops, presence expires naturally.

6. **E2EE with Signal Protocol**: Server stores only ciphertext + public keys. Cannot read content. This is a differentiator for WhatsApp.

7. **Group fan-out challenge**: 1,024 members * 100M group messages = 100B fan-out writes/day. Kafka consumers handle async fan-out.

8. **Connection server state**: WebSocket servers are stateful (hold connections). Use consistent hashing for routing, not pure load balancing.

9. **Media upload flow**: Pre-signed S3 URL + client-side encryption. Server never sees plaintext media. CDN caches encrypted ciphertext.

10. **Read receipts privacy**: WhatsApp allows users to turn off read receipts. If turned off, sender doesn't see blue ticks — but sender still sees single grey tick for server receipt.

11. **Scale math**: 2B users, 100B messages/day = 1.16M msg/sec. This drives the entire architecture — Cassandra, Kafka, WebSocket cluster size.

12. **Offline queue vs persistent subscription**: Kafka or Redis list as offline queue; delivered in order when user reconnects.
