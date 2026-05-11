# HLD Interview Q&A — File 17: Real-Time Systems

> 20 questions across Easy (Q1–7), Medium (Q8–15), and Hard (Q16–20).
> Each answer is 150–300+ words with diagrams, tables, or code where helpful.

---

## EASY (Q1–Q7)

---

### Q1. What are the differences between WebSocket, Server-Sent Events (SSE), and Long Polling? When should you use each?

**Answer:**

These three technologies all enable servers to push data to clients, but they work fundamentally differently.

**Long Polling:**
```
Client → GET /events HTTP/1.1
Server: holds connection open until event or timeout
Server → 200 OK {event data}
Client immediately re-issues request
```
- HTTP-based, works everywhere
- High overhead (new HTTP request per event cycle)
- Half-duplex (client polls, server responds)

**Server-Sent Events (SSE):**
```
Client → GET /events HTTP/1.1
         Accept: text/event-stream

Server → HTTP/1.1 200 OK
         Content-Type: text/event-stream

         data: {"user": "alice", "msg": "hello"}\n\n
         data: {"user": "bob", "msg": "hi"}\n\n
```
- Persistent HTTP connection, server pushes
- Unidirectional (server → client only)
- Auto-reconnect built into browser EventSource API
- Works over HTTP/2 (multiplexed)

**WebSocket:**
```
Client → GET /chat HTTP/1.1
         Upgrade: websocket
         Connection: Upgrade

Server → HTTP/1.1 101 Switching Protocols
         Upgrade: websocket

[Full-duplex binary/text frames in both directions]
```
- Persistent TCP connection, full-duplex
- Low overhead after handshake (~2 bytes per frame header)
- Custom protocol — not cacheable, not load-balanced easily

**Decision Matrix:**

| Factor | Long Polling | SSE | WebSocket |
|---|---|---|---|
| Direction | Half-duplex | Server → Client | Full-duplex |
| Protocol | HTTP | HTTP | WS |
| Reconnect | Manual | Auto | Manual |
| Load balancer support | Easy | Easy | Sticky sessions needed |
| Use case | Notifications, simple chat | Live feeds, dashboards | Chat, gaming, collaborative |
| Mobile battery | High | Medium | Low |

**Rule of thumb:** Use SSE when data flows only server→client (live scores, dashboards). Use WebSocket when both sides send frequently (chat, multiplayer games). Use long polling as a fallback when WebSocket/SSE are blocked.

---

### Q2. How do you scale WebSocket connections horizontally?

**Answer:**

**The Problem:**

WebSocket connections are stateful and persistent. If you have 3 servers and User A is connected to Server 1, a message from User B (connected to Server 2) cannot simply be forwarded to User A unless there is a shared backbone.

```
Server 1: [Alice, Carol]
Server 2: [Bob, Dave]
Bob sends to Alice → Server 2 doesn't have Alice's connection!
```

**Solution: Redis Pub/Sub as message backbone**

```
Architecture:
  Client Alice ←→ WebSocket Server 1 ←→ Redis Pub/Sub
  Client Bob   ←→ WebSocket Server 2 ←→ Redis Pub/Sub

Flow (Bob sends to Alice):
1. Bob's message arrives at Server 2
2. Server 2 publishes to Redis channel: pub("user:alice", message)
3. Server 1 is subscribed to "user:alice" channel
4. Redis delivers to Server 1
5. Server 1 pushes to Alice's WebSocket connection
```

**Code example (Node.js + ioredis):**
```javascript
// Each server subscribes to channels for its connected users
const subscriber = new Redis();
const publisher = new Redis();

ws.on('connection', (socket, req) => {
  const userId = getUserId(req);
  
  // Subscribe to this user's channel
  subscriber.subscribe(`user:${userId}`);
  subscriber.on('message', (channel, message) => {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(message);
    }
  });
  
  socket.on('message', (data) => {
    const { to, content } = JSON.parse(data);
    publisher.publish(`user:${to}`, JSON.stringify({ from: userId, content }));
  });
  
  socket.on('close', () => {
    subscriber.unsubscribe(`user:${userId}`);
  });
});
```

**Scaling considerations:**
- **Redis Cluster** for high fan-out (millions of channels)
- **Sticky sessions** (IP hash or cookie) as alternative — simpler but less flexible
- **Horizontal pod autoscaling** in Kubernetes — but requires draining connections gracefully
- **Connection limits:** Single server handles ~50K-100K concurrent WebSocket connections; with Redis pub/sub you can scale to millions

---

### Q3. How do you design a presence service (online/offline/last-seen)?

**Answer:**

A **presence service** tracks which users are currently online and their last activity. Core to chat apps (WhatsApp, Slack, Discord).

**Architecture:**

```
Client                  Presence Service         Redis
  |                          |                    |
  |--heartbeat(user_id)----->|                    |
  |                          |--SET user:123:ts NX EX 60-->|
  |                          |                    |
  |<--OK--------------------|                    |
  
  [60s passes with no heartbeat]
  
  Redis TTL expires → key deleted → user is offline
```

**Data model in Redis:**
```
# Online indicator (auto-expires)
SET presence:user:{id} 1 EX 60  # expires if no heartbeat in 60s

# Last seen timestamp (persists)
SET presence:last_seen:{id} {unix_timestamp}
```

**Client heartbeat:**
```javascript
// Send heartbeat every 30 seconds
setInterval(() => {
  websocket.send(JSON.stringify({ type: 'heartbeat' }));
}, 30000);
```

**Polling vs Subscription for presence updates:**
```python
# Option 1: Subscribers get push notification when friend goes offline
# Use Redis keyspace notifications
redis_config.set("notify-keyspace-events", "Ex")  # Expired events
pubsub.subscribe("__keyevent@0__:expired")

# Option 2: Client polls presence for visible contacts
# GET /presence?user_ids=1,2,3,4,5
def get_presence_batch(user_ids):
    pipe = redis.pipeline()
    for uid in user_ids:
        pipe.exists(f"presence:user:{uid}")
        pipe.get(f"presence:last_seen:{uid}")
    return pipe.execute()
```

**Scalability considerations:**
- Heartbeats from 1M users every 30s = 33K requests/second to presence service
- Shard by user_id modulo N Redis nodes
- For mobile: push notifications replace WebSocket heartbeats (battery efficiency)
- **Batching:** Don't send presence updates for every user to every subscriber — only send updates for friends/contacts

---

### Q4. What is fan-out on write vs fan-out on read for real-time feeds?

**Answer:**

**Fan-out on Write (Push Model):**

When a user creates a post, immediately push it to all followers' feed caches.

```
Alice posts → Fan-out Service
               ├── Write to Bob's feed cache
               ├── Write to Carol's feed cache
               └── Write to Dave's feed cache (... n followers)
```

```python
def publish_post(author_id, post):
    followers = db.get_followers(author_id)  # Could be 100M for celebrities
    for follower_id in followers:
        feed_cache.lpush(f"feed:{follower_id}", serialize(post))
        feed_cache.ltrim(f"feed:{follower_id}", 0, 999)  # Keep 1000 items
```

**Fan-out on Read (Pull Model):**

When a user opens their feed, query all followees' recent posts and merge.

```
Bob opens feed → Feed Service
                  ├── Query Alice's posts (last 100)
                  ├── Query Carol's posts (last 100)
                  └── Merge + sort + deduplicate
```

**Comparison:**

| Aspect | Fan-out on Write | Fan-out on Read |
|---|---|---|
| Read latency | Very fast (pre-computed) | Slower (compute on read) |
| Write latency | Slow for high-follower users | Fast |
| Storage | High (N copies per post) | Low (one copy) |
| Celebrity problem | Catastrophic (100M fan-outs) | Easy |
| Real-time | Immediate | Depends on polling |

**Hybrid Approach (Twitter's Solution):**

```
Regular users (< threshold, e.g., 10K followers):
  → Fan-out on write to follower timelines

Celebrity users (> threshold):
  → No fan-out; stored in celebrity's own timeline

On read:
  → Load pre-computed timeline from cache
  → Merge in posts from celebrities the user follows
  → Return merged, sorted result
```

This avoids the "Katy Perry problem" (1M+ followers) causing massive write amplification while keeping reads fast for regular users.

---

### Q5. How do push notifications work (FCM/APNs delivery pipeline)?

**Answer:**

**Push notification flow:**

```
Your Server → Push Gateway (FCM/APNs) → Device OS → App
```

**Step-by-step:**

```
1. App Registration:
   App → FCM/APNs: "Register me"
   FCM/APNs → App: device_token = "abc123xyz..."
   App → Your Server: "My token is abc123xyz"
   Your Server: stores {user_id → device_token}

2. Sending notification:
   Your Server → FCM API:
   POST https://fcm.googleapis.com/fcm/send
   {
     "to": "abc123xyz",
     "notification": {
       "title": "New message from Alice",
       "body": "Hey, are you free tonight?"
     },
     "data": { "chat_id": "123", "sender": "alice" }
   }

3. Delivery:
   FCM → Google Push Network → Device
   Device OS → Wake app / show notification
```

**FCM vs APNs:**

| Aspect | FCM (Android/Cross-platform) | APNs (iOS) |
|---|---|---|
| Auth | Server API key / OAuth2 | JWT or certificate |
| Protocol | HTTP/2 | HTTP/2 |
| Max payload | 4KB | 4KB |
| Priority | Normal (delayed) / High (immediate) | Normal (5) / High (10) |

**Reliability considerations:**
- Notifications are **best-effort** — no delivery guarantee
- FCM stores offline notifications for up to 4 weeks (with TTL)
- Collapsed messages: multiple notifications → one delivery (use collapse_key)
- Device token rotation: tokens change; your server must update on `registration_ids` error response

**Deduplication:** Use a notification ID; app checks if already shown before displaying.

---

### Q6. How do you design a real-time leaderboard using Redis?

**Answer:**

**Redis Sorted Set** is purpose-built for leaderboards — it maintains elements sorted by score with O(log N) updates and O(log N + M) range queries.

**Core operations:**
```redis
# Add/update player score
ZADD leaderboard NX 1500 "player:alice"     # Add only if not exists
ZINCRBY leaderboard 50 "player:alice"       # Increment score

# Get top 10 (rank 0 = highest score)
ZREVRANGE leaderboard 0 9 WITHSCORES

# Get player's rank (0-indexed)
ZREVRANK leaderboard "player:alice"

# Get player's score
ZSCORE leaderboard "player:alice"

# Get rank range (e.g., positions 100-200)
ZREVRANGE leaderboard 99 199 WITHSCORES
```

**Application code:**
```python
def update_score(player_id, delta):
    new_score = redis.zincrby("leaderboard:global", delta, f"player:{player_id}")
    rank = redis.zrevrank("leaderboard:global", f"player:{player_id}")
    return {"score": new_score, "rank": rank + 1}  # 1-indexed

def get_leaderboard(page=1, page_size=50):
    start = (page - 1) * page_size
    end = start + page_size - 1
    entries = redis.zrevrange("leaderboard:global", start, end, withscores=True)
    return [{"rank": start + i + 1, "player": p, "score": s} 
            for i, (p, s) in enumerate(entries)]

def get_surrounding_ranks(player_id, window=5):
    rank = redis.zrevrank("leaderboard:global", f"player:{player_id}")
    start = max(0, rank - window)
    end = rank + window
    return redis.zrevrange("leaderboard:global", start, end, withscores=True)
```

**Time-windowed leaderboards:**
```python
# Daily leaderboard — key per day
today = datetime.now().strftime("%Y-%m-%d")
redis.zincrby(f"leaderboard:daily:{today}", delta, player_id)
redis.expire(f"leaderboard:daily:{today}", 7 * 86400)  # Keep 7 days
```

**Scale:** Redis Sorted Set handles millions of players efficiently. For global scale, shard by region or game. Merge top-N from each shard into a global leaderboard.

---

### Q7. What is the heartbeat mechanism for connection liveness in distributed systems?

**Answer:**

A **heartbeat** is a periodic signal sent between components to confirm they are alive and the connection is healthy. Without heartbeats, neither side knows if the other has crashed silently or if the network has been severed.

**Client-initiated heartbeat (WebSocket):**
```javascript
// WebSocket ping/pong — built into the protocol
// Server sends PING frame
// Client responds with PONG frame
// If no PONG received within timeout → close connection

// Application-level heartbeat
const heartbeatInterval = setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
  }
}, 25000);  // Every 25 seconds (before typical 30s TCP timeout)

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'pong') {
    clearTimeout(reconnectTimer);  // Connection confirmed alive
  }
};
```

**Server-side timeout handling:**
```python
class ConnectionManager:
    def __init__(self):
        self.connections = {}  # user_id → {ws, last_heartbeat}
    
    async def on_heartbeat(self, user_id):
        self.connections[user_id]['last_heartbeat'] = time.time()
    
    async def check_stale_connections(self):
        while True:
            now = time.time()
            for user_id, conn in list(self.connections.items()):
                if now - conn['last_heartbeat'] > 60:  # 60s timeout
                    await self.disconnect(user_id, reason='heartbeat_timeout')
            await asyncio.sleep(10)  # Check every 10 seconds
```

**Distributed heartbeat (service mesh / health checks):**
```
Consul health check:
  Every 10s: HTTP GET /health
  If fails 3 times → service marked unhealthy → removed from load balancer
```

**Gossip protocol heartbeats (Cassandra/Redis Cluster):**
Each node periodically sends its state vector to random peers. Nodes learn about others' health through gossip propagation — no single point for heartbeat collection.

---

## MEDIUM (Q8–Q15)

---

### Q8. What are Operational Transformation (OT) and how does it enable collaborative editing?

**Answer:**

**Operational Transformation (OT)** is the algorithm behind Google Docs' real-time collaboration. It allows multiple users to edit a document simultaneously by transforming operations against each other to ensure convergence.

**The Problem:**
```
Initial state: "Hello"

User A (position 5): Insert " World"  → Op A: Insert(" World", 5)
User B (position 0): Insert "Say "    → Op B: Insert("Say ", 0)

If both apply their ops to "Hello":
A sees: "Hello World"    → then applies B → "Say Hello World" ✓
B sees: "Say Hello"      → then applies A → "Say Hello World" ✓
But positions are now wrong! A's op inserted at 5, but B shifted everything by 4.
```

**OT Solution — Transform function:**

```python
def transform_insert_against_insert(op_a, op_b):
    """Transform op_a assuming op_b was already applied."""
    if op_b.position <= op_a.position:
        # op_b inserted before op_a's position → shift op_a right
        return InsertOp(op_a.text, op_a.position + len(op_b.text))
    else:
        # op_b inserted after op_a's position → no change
        return op_a

# Transform sequence:
Op_A_prime = transform(Op_A, Op_B)  # Op_A after Op_B applied
Op_B_prime = transform(Op_B, Op_A)  # Op_B after Op_A applied

# Final state:
# Start → apply(Op_A) → apply(Op_B_prime) = "Say Hello World"
# Start → apply(Op_B) → apply(Op_A_prime) = "Say Hello World"  ✓ Convergence!
```

**Server-side OT architecture (Google Docs model):**
```
Client A sends Op_A at version 5
Server is at version 7 (Ops 6, 7 arrived from others)

Server transforms Op_A against Ops [6, 7]:
  Op_A' = transform(Op_A, Op_6)
  Op_A'' = transform(Op_A', Op_7)

Server applies Op_A'' to document, broadcasts to all clients
```

**OT limitations:**
- Server must be the **arbiter** — true P2P OT is exponentially complex
- Transform functions must be defined for every pair of operation types
- Multi-user scenarios require careful diamond property proofs

---

### Q9. How does CRDT differ from OT for real-time collaboration, and when would you choose each?

**Answer:**

Both CRDTs and OT solve the same problem — concurrent edits in distributed systems — but with fundamentally different approaches.

**OT (Operational Transformation):**
- Transforms operations to account for concurrent edits
- Requires a central server to order and transform operations
- Works on the **operation** level — what changed

**CRDT (Conflict-free Replicated Data Type):**
- Designs the data structure so merges are always correct
- Works fully peer-to-peer without central coordination
- Works on the **state** level — what the data is

**CRDT for text editing — sequence CRDTs:**

```
Each character gets a unique, immutable identifier:
  "Hello" = [(H, uid:a1), (e, uid:a2), (l, uid:a3), (l, uid:a4), (o, uid:a5)]

Insert "X" between 'H' and 'e':
  Create new element: (X, uid:b1) with parent=uid:a1
  CRDT rule: new element always sorts after its parent

Result regardless of order applied:
  H → X → e → l → l → o   ✓ No transformation needed
```

**Popular CRDT text libraries:**
- **Yjs:** Used by Notion, used with CodeMirror/Monaco
- **Automerge:** Used by various collaborative tools
- **diamond-types:** Rust-based, extremely fast

**Comparison:**

| Aspect | OT | CRDT |
|---|---|---|
| Architecture | Requires central server | Fully P2P capable |
| Complexity | High (transform functions) | High (data structure design) |
| Offline support | Poor (need server to transform) | Excellent (merge on reconnect) |
| Storage | Only current state needed | Document grows (tombstones) |
| Performance | Good | Good (but GC needed for tombstones) |
| Conflict resolution | Explicit transforms | Implicit in structure |

**When to choose:**
- **OT:** Already have server, simple document types (text), need maximum control, existing Google Docs API compatibility
- **CRDT:** Need offline-first, P2P, or local-first software; complex merge semantics (JSON, rich text, code); mobile apps with intermittent connectivity

---

### Q10. How does WebRTC work for P2P video calls? What are STUN, TURN, ICE, and SDP?

**Answer:**

WebRTC enables direct peer-to-peer audio/video communication in browsers without plugins. The challenge: most devices are behind NAT routers and don't have public IPs.

**The problem — NAT traversal:**
```
Alice: 192.168.1.5 (private) → router → 203.0.113.1:54321 (public)
Bob:   10.0.0.7   (private) → router → 198.51.100.2:43210 (public)

Direct connection impossible without knowing each other's public IP:port
```

**STUN (Session Traversal Utilities for NAT):**
```
Alice → STUN Server: "What is my public IP:port?"
STUN Server → Alice: "You appear as 203.0.113.1:54321"
Alice can now share this with Bob via signaling
```

**ICE (Interactive Connectivity Establishment):**
ICE is the framework that gathers all possible connection paths ("candidates") and picks the best one:
```
ICE Candidates gathered by Alice:
  1. Host candidate:    192.168.1.5:49152 (local)
  2. SRFLX candidate:   203.0.113.1:54321 (via STUN — server-reflexive)
  3. Relay candidate:   198.51.100.100:49152 (via TURN — if P2P fails)
```

**SDP (Session Description Protocol):**
A text format that describes the media session — codecs, encryption keys, ICE candidates:
```
v=0
o=alice 123456 654321 IN IP4 203.0.113.1
s=VideoCall
m=video 9 UDP/TLS/RTP/SAVPF 96 97
a=rtpmap:96 VP8/90000
a=candidate:1 1 UDP 2130706431 192.168.1.5 49152 typ host
a=candidate:2 1 UDP 1694498815 203.0.113.1 54321 typ srflx
```

**TURN (Traversal Using Relays around NAT):**
When P2P fails (symmetric NAT), TURN relays all traffic through a server. Expensive but always works.

**Complete WebRTC signaling flow:**
```
Alice                  Signaling Server            Bob
  |                          |                      |
  |--createOffer()---------->|                      |
  |<--setLocalDescription----|                      |
  |                          |--offer SDP---------->|
  |                          |<--answer SDP---------|
  |<--setRemoteDescription---|                      |
  |                          |                      |
  [ICE candidate exchange via signaling]
  |                          |                      |
  [P2P connection established — data flows directly]
```

---

### Q11. What is the difference between SFU, MCU, and P2P for group video calls?

**Answer:**

As the number of participants in a video call grows, P2P becomes impractical. SFU and MCU are server-based architectures that solve this.

**P2P (Peer-to-Peer):**
```
3 participants = 3 connections (mesh)
N participants = N*(N-1)/2 connections

Alice ←→ Bob
Alice ←→ Carol
Bob ←→ Carol

Upload: Alice sends N-1 streams (one per peer)
```
- Works well for 2-4 people
- Each participant's upload bandwidth grows with N
- 10 participants = 9 outgoing streams per person (infeasible)

**SFU (Selective Forwarding Unit):**
```
Each participant sends ONE stream to SFU
SFU forwards selected streams to each participant

Alice → SFU → Bob (gets Alice's stream)
Bob   → SFU → Alice (gets Bob's stream)
Carol → SFU → Alice, Bob
```
- Server doesn't decode/re-encode (forwards RTP packets as-is)
- Server CPU: low (just routing)
- Client download: still N-1 streams (but upload is only 1)
- Can use **simulcast**: send 3 quality levels (360p/720p/1080p), SFU chooses per-receiver
- Examples: Zoom, Google Meet (SFU-based)

**MCU (Multipoint Control Unit):**
```
Each participant sends to MCU
MCU decodes ALL streams, composites into one mixed video
Sends ONE composed stream to each participant

Alice → MCU ← Bob ← Carol
        ↓
   [Composed video grid]
        ↓
Alice ← MCU → Bob → Carol
```
- Server CPU: very high (decode + encode + mix)
- Client: only downloads 1 stream (great for low-bandwidth clients)
- High latency (encode/decode pipeline)
- Historically used by enterprise video conferencing

**Comparison:**

| Aspect | P2P | SFU | MCU |
|---|---|---|---|
| Server load | None | Low (forwarding) | Very High (transcoding) |
| Client upload | High | Low (1 stream) | Low (1 stream) |
| Client download | High | Medium (N-1 streams) | Low (1 composed) |
| Latency | Lowest | Low | Medium-High |
| Quality flexibility | N/A | Per-subscriber | Fixed composition |
| Best for | 2-4 people | 5-50 people | Low-bandwidth clients, recording |

**Modern approach:** Zoom, Discord, and Google Meet all use SFU. Cascaded SFU (multiple SFU servers per call) handles large webinars.

---

### Q12. How does Uber show real-time driver location updates?

**Answer:**

Uber's real-time location system must handle millions of driver GPS updates per minute and deliver them to nearby rider apps within seconds.

**End-to-end architecture:**
```
Driver App               Uber Backend                 Rider App
    |                        |                            |
    |--GPS update (every 4s)->|                           |
    |                        |                            |
    |                   Location Service                  |
    |                   (geo-partitioned)                 |
    |                        |                            |
    |                   Geohash index                     |
    |                   Redis + S2 geometry               |
    |                        |                            |
    |                        |<--driver query(lat/lng)----|
    |                        |--nearby drivers----------->|
```

**Driver location ingestion:**
```python
# Driver app sends GPS every 4 seconds
# Uber uses a custom binary protocol over HTTP/2 for efficiency

class LocationService:
    def update_driver_location(self, driver_id, lat, lng, timestamp):
        geohash = encode_geohash(lat, lng, precision=7)  # ~150m precision
        
        # Update driver's current position
        redis.geoadd("drivers:active", lng, lat, driver_id)
        redis.expire(f"driver:{driver_id}:location", 30)  # Remove if offline
        
        # Pub/Sub for riders who are tracking this driver
        if rider_id := self.get_assigned_rider(driver_id):
            pubsub.publish(f"rider:{rider_id}:driver_location",
                          json.dumps({"lat": lat, "lng": lng, "ts": timestamp}))
```

**Rider querying nearby drivers:**
```python
# Find drivers within 2km
def get_nearby_drivers(rider_lat, rider_lng, radius_km=2):
    # Redis GEORADIUS command
    drivers = redis.georadius(
        "drivers:active",
        rider_lng, rider_lat,
        radius_km, "km",
        withdist=True, withcoord=True,
        count=20, sort="ASC"
    )
    return drivers
```

**Dispatch matching (simplified):**
```
1. Rider requests ride at (lat, lng)
2. Find top 10 nearest available drivers
3. Send request to driver #1 (15s timeout)
4. If no response → send to driver #2
5. Driver accepts → trip begins
```

**Scale numbers:** ~5M driver updates/minute → ~83K updates/second. Uber geo-partitions its systems — each city shard handles its own drivers. S2 geometry library for hierarchical geo-indexing.

---

### Q13. How do you implement read receipts (double tick / blue tick) in a chat system?

**Answer:**

Read receipts require tracking three states per message per recipient: **Sent → Delivered → Read**.

**Message states:**
```
Sent (✓):      Message reached the server
Delivered (✓✓): Message reached the recipient's device
Read (✓✓ blue): Recipient opened the conversation
```

**Data model:**
```sql
CREATE TABLE message_receipts (
    message_id BIGINT,
    recipient_id BIGINT,
    delivered_at TIMESTAMP,   -- when device received it
    read_at TIMESTAMP,        -- when user opened chat
    PRIMARY KEY (message_id, recipient_id)
);
```

**Flow — Delivered receipt:**
```
Alice sends message → Server stores (status: sent)
Server pushes to Bob's device (FCM/WebSocket)
Bob's device receives → Device ACKs to server
Server updates message status to 'delivered'
Server notifies Alice: "message_delivered" event
```

**Flow — Read receipt:**
```
Bob opens conversation → Client sends read receipt
POST /messages/read
{ "conversation_id": "123", "last_read_message_id": "456" }

Server:
  UPDATE messages SET read_at = NOW()
  WHERE conversation_id = 123
    AND recipient_id = Bob
    AND id <= 456;

Server → Alice WebSocket: "messages_read" { "through": 456, "by": Bob }
```

**Optimizations:**
```python
# Batch read receipts — don't send one per message
# Use "read through" model: mark all messages up to ID X as read
def mark_read(conversation_id, user_id, last_message_id):
    redis.zadd(f"read_receipt:{conversation_id}:{user_id}",
               {str(last_message_id): time.time()})
    # Async flush to DB via worker

# Coalesce receipts: if user reads 50 messages, send 1 receipt event
# not 50 individual events
```

**Privacy considerations:**
- WhatsApp allows users to disable read receipts
- If disabled: no blue tick shown to sender, but you also can't see others' blue ticks
- Group chats: show per-member delivery/read status (expensive for large groups → typically capped at 100 members)

---

### Q14. How would you design a live sports score system?

**Answer:**

A live sports score system must push score updates to millions of concurrent viewers within seconds of real events.

**Requirements:**
- Low latency score updates (< 2 seconds from event to display)
- Handle traffic spikes (100x normal during big games)
- Global reach
- Handle stale connections gracefully

**Architecture:**
```
Data Source (Official API/Stadium feed)
         ↓
    Score Ingestion Service
         ↓
    Kafka topic: score-updates
         ↓
    Score Processor → Redis (current scores)
         ↓
    Push Gateway (SSE/WebSocket)
         ↓
    CDN Edge Nodes
         ↓
    Client browsers/apps
```

**Score ingestion:**
```python
class ScoreIngestionService:
    def process_update(self, event):
        score_update = {
            "match_id": event["match_id"],
            "home_score": event["home"],
            "away_score": event["away"],
            "minute": event["minute"],
            "event_type": event["type"],  # goal/card/substitution
            "timestamp": time.time()
        }
        
        # Atomic update
        redis.hset(f"match:{event['match_id']}", mapping=score_update)
        
        # Notify subscribers
        redis.publish(f"match:{event['match_id']}:updates",
                     json.dumps(score_update))
```

**Client delivery (SSE - Server-Sent Events):**
```python
@app.route('/scores/live/<match_id>')
def live_scores(match_id):
    def generate():
        # Send current state immediately
        current = redis.hgetall(f"match:{match_id}")
        yield f"data: {json.dumps(current)}\n\n"
        
        # Subscribe to updates
        pubsub = redis.pubsub()
        pubsub.subscribe(f"match:{match_id}:updates")
        for message in pubsub.listen():
            if message['type'] == 'message':
                yield f"data: {message['data']}\n\n"
    
    return Response(generate(),
                   mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache'})
```

**CDN strategy:** Score API responses (current state) can be cached for 1-2 seconds at edge. Live push channels bypass CDN. CDN handles static assets (team logos, page templates).

**Traffic spike handling:** Pre-scale before known high-demand matches. Use auto-scaling based on active WebSocket connections. Rate-limit score polling to prevent stampede on reconnection.

---

### Q15. How does backpressure work in real-time streaming systems?

**Answer:**

**Backpressure** is the mechanism by which a slower consumer signals to a faster producer to slow down, preventing buffer overflow and system crashes.

**Without backpressure:**
```
Producer (1M events/sec)
      ↓
  [Buffer fills up → OOM error or messages dropped]
      ↓
Consumer (100K events/sec)
```

**Backpressure mechanisms:**

**1. TCP Flow Control (network level):**
TCP receiver window advertises how much buffer space is available. Producer stops sending when window = 0.

**2. Reactive Streams (application level):**
```java
// Java Reactive Streams (Project Reactor / RxJava)
Flux.range(1, 1_000_000)
    .onBackpressureBuffer(1000)          // Buffer up to 1000
    .onBackpressureDrop(item ->          // Drop if buffer full
        log.warn("Dropping: {}", item))
    .subscribe(
        item -> processSlowly(item),
        error -> log.error("Error", error)
    );
```

**3. Kafka consumer groups:**
```
Kafka topic: 10M messages
Kafka consumer: 100K/sec processing rate
Kafka: consumers control their own offset
       → consumer only pulls what it can handle
       → partition lag increases (monitoring alert) but no crash
```

**4. Rate limiting with token bucket:**
```python
class TokenBucketBackpressure:
    def __init__(self, rate, capacity):
        self.tokens = capacity
        self.rate = rate  # tokens per second
        self.last_refill = time.time()
    
    def try_consume(self):
        now = time.time()
        tokens_to_add = (now - self.last_refill) * self.rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True  # Allow
        return False  # Backpressure — reject or queue
```

**WebSocket backpressure:**
```javascript
// Check bufferedAmount before sending more
function sendWithBackpressure(ws, data) {
    if (ws.bufferedAmount < 16384) {  // 16KB threshold
        ws.send(data);
    } else {
        // Socket buffer is full — pause sending
        setTimeout(() => sendWithBackpressure(ws, data), 50);
    }
}
```

---

## HARD (Q16–Q20)

---

### Q16. How do you design a notification system that deduplicates across multiple channels?

**Answer:**

A notification system for a large platform (1B users) must route notifications through multiple channels (push, email, SMS, in-app) while ensuring the same logical notification isn't delivered twice through different channels or retries.

**Architecture:**
```
Event Bus (Kafka)
      ↓
Notification Router
      ↓ Routing rules engine
      ↓
 ┌────┴────┬──────────┬──────┐
Push     Email      SMS    In-App
Worker   Worker   Worker  Worker
  ↓        ↓        ↓       ↓
FCM/APNs  SES    Twilio  Redis
```

**Deduplication at multiple levels:**

**Level 1: Logical deduplication (same event, multiple channels):**
```python
class NotificationRouter:
    def route(self, event):
        user_prefs = self.get_user_preferences(event.user_id)
        channels = self.determine_channels(event, user_prefs)
        
        # Choose BEST channel, not all channels
        # e.g., if user is online → in-app only (no push)
        #       if offline → push
        #       if push fails after 30min → email
        
        dedup_key = f"notif:{event.type}:{event.user_id}:{event.entity_id}"
        if redis.set(dedup_key, 1, nx=True, ex=3600):
            # Not yet sent for this (event_type, user, entity) combo
            self.send_via_channels(event, channels)
```

**Level 2: Delivery deduplication (retry safety):**
```python
def send_push(notification_id, device_token, payload):
    # Idempotency key: if FCM times out, retry with same key
    dedup_key = f"fcm_sent:{notification_id}"
    if redis.set(dedup_key, 1, nx=True, ex=3600):
        fcm.send(device_token, payload, message_id=notification_id)
    else:
        logger.info(f"Notification {notification_id} already sent, skipping")
```

**Level 3: Notification batching (reduce notification fatigue):**
```python
# Instead of 10 separate "X liked your photo" notifications,
# batch them: "Alice, Bob, and 8 others liked your photo"

class NotificationBatcher:
    def queue(self, user_id, event):
        batch_key = f"batch:{user_id}:{event.type}:{event.entity_id}"
        redis.zadd(batch_key, {json.dumps(event): time.time()})
        redis.expire(batch_key, 300)  # Batch window: 5 minutes
        
        # Schedule flush if not already scheduled
        if not redis.exists(f"batch_timer:{user_id}:{event.type}"):
            self.schedule_flush(user_id, event.type, delay=300)
    
    def flush_batch(self, user_id, event_type):
        events = redis.zrange(f"batch:{user_id}:{event_type}:*", 0, -1)
        if len(events) == 1:
            self.send_single(events[0])
        else:
            self.send_aggregated(user_id, events)
```

**Notification preference engine:**
```json
{
  "user_id": 12345,
  "channels": {
    "push": { "enabled": true, "quiet_hours": "22:00-08:00" },
    "email": { "enabled": true, "frequency": "daily_digest" },
    "sms": { "enabled": false }
  },
  "categories": {
    "marketing": { "channels": ["email"] },
    "security": { "channels": ["push", "email", "sms"] },
    "social": { "channels": ["push"], "max_per_hour": 5 }
  }
}
```

---

### Q17. How do you design a real-time analytics dashboard that updates every second?

**Answer:**

A real-time analytics dashboard (think Cloudflare's analytics, Stripe's payment dashboard) needs fresh metrics without overloading backends.

**Architecture:**
```
Raw Events → Kafka → Stream Processor (Flink/Spark Streaming)
                                ↓
                         Pre-aggregated metrics
                                ↓
                    TimeSeries DB (ClickHouse/InfluxDB)
                                ↓
                    Query Cache (Redis)
                                ↓
                    Dashboard API (SSE)
                                ↓
                    Browser Dashboard
```

**Pre-aggregation in stream processor:**
```python
# Apache Flink — tumbling window aggregation
stream = env.from_source(kafka_source)

# 1-second tumbling windows
windowed = (stream
    .key_by(lambda e: e["merchant_id"])
    .window(TumblingProcessingTimeWindows.of(Time.seconds(1)))
    .aggregate(
        lambda: {"count": 0, "amount": 0.0, "errors": 0},
        lambda acc, e: {
            "count": acc["count"] + 1,
            "amount": acc["amount"] + e["amount"],
            "errors": acc["errors"] + (1 if e["status"] == "error" else 0)
        }
    )
)
```

**Dashboard API with SSE:**
```python
@app.route('/dashboard/metrics/<merchant_id>')
def dashboard_stream(merchant_id):
    def generate():
        last_metrics = None
        while True:
            metrics = get_current_metrics(merchant_id)
            
            # Only send if changed (reduce noise)
            if metrics != last_metrics:
                yield f"data: {json.dumps(metrics)}\n\n"
                last_metrics = metrics
            
            time.sleep(1)  # 1-second resolution
    
    return Response(generate(), mimetype='text/event-stream')
```

**Avoiding thundering herd on dashboard load:**
```python
def get_current_metrics(merchant_id):
    cache_key = f"metrics:{merchant_id}:current"
    
    # Try cache first (pre-warmed by stream processor)
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Fallback to DB query (should rarely happen)
    return clickhouse.query(f"""
        SELECT count() as requests, sum(amount) as revenue,
               countIf(status='error') as errors
        FROM events
        WHERE merchant_id = {merchant_id}
          AND timestamp >= now() - INTERVAL 1 SECOND
    """)
```

**Scaling to 100K concurrent dashboard viewers:**
- SSE connections are cheap (HTTP/2 multiplexing)
- Pre-compute metrics → don't query DB per connection
- Push from stream processor → Redis → SSE workers
- Shard SSE workers by merchant_id

---

### Q18. How do you handle offline users in a chat system?

**Answer:**

When a message is sent to an offline user, the system must durably store it and deliver it when the user reconnects or via push notification.

**Delivery state machine:**
```
SENT → QUEUED (recipient offline) → DELIVERED (reconnects/push) → READ
```

**Architecture:**
```
Alice sends message
      ↓
Chat Server checks Bob's presence
      ├── Online: push via WebSocket → mark DELIVERED
      └── Offline: persist to message store + send push notification
                        ↓
                 Bob reconnects
                        ↓
                 Client sends "sync from offset X"
                        ↓
                 Server returns all undelivered messages
                        ↓
                 Client ACKs each message
```

**Message persistence (Cassandra schema):**
```sql
CREATE TABLE messages (
    conversation_id UUID,
    message_id TIMEUUID,     -- time-ordered UUID for natural sorting
    sender_id UUID,
    content TEXT,
    status TEXT,             -- 'sent', 'delivered', 'read'
    created_at TIMESTAMP,
    PRIMARY KEY (conversation_id, message_id)
) WITH CLUSTERING ORDER BY (message_id ASC)
  AND default_time_to_live = 7776000;  -- 90 days TTL
```

**Sync on reconnect:**
```python
class MessageSyncHandler:
    def on_client_reconnect(self, user_id, last_seen_message_id):
        # Get all messages delivered while user was offline
        undelivered = cassandra.execute("""
            SELECT * FROM user_inbox
            WHERE user_id = %s AND message_id > %s
            LIMIT 1000
        """, [user_id, last_seen_message_id])
        
        # Send in batches of 50
        for batch in chunks(undelivered, 50):
            websocket.send_batch(user_id, batch)
            # Wait for ACK before next batch (backpressure)
            await websocket.wait_ack(user_id, timeout=5)
```

**Push notification for offline delivery:**
```python
def deliver_to_offline_user(user_id, message):
    device_tokens = db.get_device_tokens(user_id)
    
    push_payload = {
        "notification": {
            "title": message.sender_name,
            "body": truncate(message.content, 100)
        },
        "data": {
            "conversation_id": str(message.conversation_id),
            "message_id": str(message.message_id),
            "deep_link": f"myapp://chat/{message.conversation_id}"
        }
    }
    
    for token in device_tokens:
        fcm.send(token, push_payload, 
                 ttl=86400,  # Keep notification for 24h if device offline
                 collapse_key=f"chat_{message.conversation_id}")
```

---

### Q19. How would you design the fan-out system for a Twitter-like feed with celebrity accounts having 100M followers?

**Answer:**

This is one of the canonical systems design challenges. The core tension: fan-out on write is O(followers) per tweet — catastrophic for celebrities.

**Detailed hybrid architecture:**

```
Tweet published by user X
         ↓
    Fan-out Service
         ↓
  Is X a "celebrity"? (followers > threshold T)
  ├── No (regular user): → Async fan-out worker
  │         ↓
  │   For each follower: LPUSH feed:follower_id tweet_id
  │
  └── Yes (celebrity): No fan-out, tweet stored in celebrity:X:tweets
```

**Timeline construction on read:**
```python
def get_timeline(user_id, count=20, cursor=None):
    # 1. Get pre-computed feed (from fan-out writes of regular users)
    feed = redis.lrange(f"feed:{user_id}", 0, count + 50)
    
    # 2. Get celebrities this user follows
    celebrities = db.get_followed_celebrities(user_id)
    
    # 3. Fetch recent tweets from each celebrity
    celebrity_tweets = []
    for celeb_id in celebrities:
        tweets = redis.lrange(f"celebrity:{celeb_id}:tweets", 0, 20)
        celebrity_tweets.extend(tweets)
    
    # 4. Merge and sort by timestamp
    all_tweet_ids = list(set(feed + celebrity_tweets))
    
    # 5. Fetch tweet details (batch get from Redis/DB)
    tweets = batch_get_tweets(all_tweet_ids)
    
    # 6. Apply filters (blocked users, muted words)
    tweets = apply_filters(tweets, user_id)
    
    # 7. Sort by time, apply ranking (engagement signals)
    return sorted(tweets, key=lambda t: t.score, reverse=True)[:count]
```

**Fan-out worker with priority queues:**
```python
class FanoutWorker:
    def process_tweet(self, tweet):
        followers = db.get_followers_paginated(tweet.author_id)
        
        # Prioritize active users (online/recently active)
        active, inactive = partition_by_activity(followers)
        
        # Fan out to active users first (real-time experience)
        for follower_id in active:
            redis.lpush(f"feed:{follower_id}", tweet.id)
            redis.ltrim(f"feed:{follower_id}", 0, 799)  # Keep 800 tweets
        
        # Fan out to inactive users lazily (batch, lower priority)
        self.lazy_fanout_queue.enqueue_batch(inactive, tweet.id)
```

**Scale math:**
- Celebrity with 100M followers sends 1 tweet
- Fan-out on write: 100M Redis writes × 10 bytes = 1GB write amplification per tweet
- At 10 celebrity tweets/minute peak: 1GB × 10 = 10GB/minute writes to Redis
- **Solution:** Hybrid approach — only fan-out to ACTIVE followers (< 5% of followers are active at any moment)
- 100M × 5% = 5M writes per celebrity tweet — manageable

---

### Q20. How do you design a WebRTC-based group video call system supporting 500 concurrent participants?

**Answer:**

500 participants is far beyond what SFU can naively handle per room. This requires a **cascaded SFU** architecture plus aggressive optimization.

**Naïve SFU limitation:**
```
500 participants × each sends 1 stream
= 500 incoming streams to one SFU
= 500 × 499 = 249,500 forwarding decisions per second
= Completely infeasible on one server
```

**Cascaded SFU Architecture:**
```
Global Load Balancer
        ↓
   [SFU Cluster]
  SFU-1  SFU-2  SFU-3  SFU-4
  (125 participants each)
     ↕       ↕       ↕       ↕
   [SFU Interconnect — selected streams only]
```

```
Layout:
  Each SFU handles its local participants
  "Active speaker" stream is forwarded to all other SFUs
  Only 1-3 active speakers' streams cross SFU boundaries
  → 4 SFUs × 3 cross-streams = 12 inter-SFU streams (manageable)
```

**Simulcast + adaptive bitrate:**
```javascript
// Each participant sends 3 quality layers
const senderConstraints = [
    { scaleResolutionDownBy: 4, maxBitrate: 150000 },  // 240p
    { scaleResolutionDownBy: 2, maxBitrate: 500000 },  // 480p
    { scaleResolutionDownBy: 1, maxBitrate: 1500000 }, // 1080p
];

// SFU selects appropriate layer per receiver
// Participant seeing speaker in full screen → gets 1080p
// Participant in thumbnail grid → gets 240p
```

**Active speaker detection:**
```python
class ActiveSpeakerDetector:
    def __init__(self):
        self.audio_levels = {}  # participant_id → rolling average dB
    
    def update(self, participant_id, audio_level_db):
        self.audio_levels[participant_id] = (
            0.9 * self.audio_levels.get(participant_id, -80) +
            0.1 * audio_level_db
        )
    
    def get_active_speakers(self, top_n=3):
        sorted_by_level = sorted(self.audio_levels.items(),
                                  key=lambda x: x[1], reverse=True)
        # Apply hysteresis — don't switch active speaker too quickly
        return [p for p, level in sorted_by_level[:top_n] if level > -50]
```

**Bandwidth optimization:**
```
Regular attendee (500 total):
  Sends: 1 video stream (720p) + audio
  Receives: 3 active speaker streams + 49 thumbnail streams (240p)
  
  Download per participant:
    3 × 1.5Mbps (speakers) + 49 × 150Kbps (thumbnails)
    = 4.5Mbps + 7.35Mbps = ~12Mbps
  
  With viewport-based visibility:
    Only request streams for thumbnails actually visible on screen
    Reduces download to ~5Mbps per participant
```

**Recording architecture:**
```
One SFU node designated as recorder
Receives all streams → passes to media server
Media server: Gstreamer pipeline → mux → MP4/WebM
Store to S3 → process async for replay
```

---

## Quick Reference

### WebSocket vs SSE vs Long Polling
| | Long Poll | SSE | WebSocket |
|---|---|---|---|
| Direction | Half | Server→Client | Full |
| Use when | Fallback | Dashboards/feeds | Chat/games |
| Load balancer | Easy | Easy | Sticky |

### Scaling WebSockets
```
Client → WS Server 1 → Redis Pub/Sub ← WS Server 2 ← Client
```

### Presence Service
```
Client heartbeat every 30s → Redis SET user:X EX 60
No heartbeat for 60s → key expires → offline
```

### Fan-Out Decision
- Regular users (<10K followers) → fan-out on write
- Celebrities (>10K followers) → fan-out on read (merge at query time)

### Redis Sorted Set for Leaderboard
```
ZADD lb 1500 "player:alice"
ZREVRANGE lb 0 9 WITHSCORES   ← top 10
ZREVRANK lb "player:alice"    ← alice's rank
```

### WebRTC Connection Flow
```
1. STUN → get public IP:port
2. Exchange SDP (offer/answer) via signaling server
3. ICE candidate exchange
4. P2P connection (or TURN relay if NAT blocks P2P)
```

### SFU vs MCU vs P2P
- P2P: best for 2-4 people, no server needed
- SFU: best for 5-50 people, low server CPU
- MCU: best for low-bandwidth clients, high server CPU

### Backpressure Strategies
1. TCP window (transport level)
2. Kafka consumer pull (application level)
3. Reactive streams (programmatic)
4. Token bucket rate limiter

### Heartbeat Timeouts
```
Client sends heartbeat every 25s
Server closes connection if no heartbeat for 60s
```
