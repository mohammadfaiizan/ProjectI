# 19. Real-Time and Notification Systems

## Table of Contents
1. [Real-Time Communication Patterns](#1-real-time-communication-patterns)
2. [WebSocket Architecture](#2-websocket-architecture)
3. [Scaling WebSockets](#3-scaling-websockets)
4. [Server-Sent Events (SSE)](#4-server-sent-events-sse)
5. [Push Notifications](#5-push-notifications)
6. [Notification System Design](#6-notification-system-design)
7. [Fan-Out Patterns](#7-fan-out-patterns)
8. [Real-Time Feed Design](#8-real-time-feed-design)
9. [Chat System Architecture](#9-chat-system-architecture)
10. [Presence Service](#10-presence-service)
11. [Notification Delivery Guarantees](#11-notification-delivery-guarantees)
12. [Email Delivery Pipeline](#12-email-delivery-pipeline)
13. [SMS Delivery](#13-sms-delivery)
14. [In-App Notification Center](#14-in-app-notification-center)
15. [Live Collaboration](#15-live-collaboration)
16. [Real-Time Gaming Considerations](#16-real-time-gaming-considerations)
17. [Video and Audio Streaming Protocols](#17-video-and-audio-streaming-protocols)
18. [Quick Reference](#18-quick-reference)

---

## 1. Real-Time Communication Patterns

### Overview and Comparison

```
Pattern           Direction     Latency    Overhead   Reconnection   Use Case
Polling           C → S         High       High       N/A            Simple dashboards
Long Polling      C ← S         Medium     Medium     Manual         Fallback, simple push
SSE               S → C         Low        Low        Automatic      Feeds, notifications
WebSocket         Bidirectional Very Low   Very Low   Manual         Chat, games, collab
WebRTC            P2P           Lowest     Medium     Manual         Video/audio calls
```

### Short Polling

```
Client: "Any updates?"  (every N seconds)
Server: "Here's data" or "No updates"

HTTP request → response cycle repeats

Problems:
  - Wastes bandwidth (most responses are empty)
  - Latency = polling interval (default 30s → 30s average lag)
  - Server load proportional to client count × polling frequency
  
Use when:
  - Simplest possible implementation
  - Update frequency matches polling interval
  - Client count is small

Client implementation:
  setInterval(() => fetch('/api/notifications'), 30000);
```

### Long Polling

```
Client sends request → Server holds request open until data available → 
Response sent → Client immediately re-opens request

Client:  GET /notifications (hanging open)
         (10 seconds pass)
Server:  Here's a notification!  (response after event occurs)
Client:  GET /notifications (immediately reconnects)

Problems:
  - Server must track open connections (1 thread per connection in blocking servers)
  - Reconnect gap: notifications during reconnect window may be missed
  - Not efficient under heavy message volume
  
Use when:
  - WebSocket not supported
  - Messages are infrequent (< 1/second)
  - Fallback for older clients
```

### Comparison: When to Use What

```
Use Short Polling when:
  - Data changes infrequently (hourly analytics dashboard)
  - Simplicity > efficiency
  
Use Long Polling when:
  - Can't use WebSocket (proxy limitations)
  - Need reliable fallback
  
Use SSE when:
  - Server-to-client only (notifications, live feeds, stock tickers)
  - HTTP/2 available (multiplexing handles many connections)
  - Built-in reconnect behavior needed
  
Use WebSocket when:
  - Bidirectional real-time communication (chat, games, collaborative editing)
  - High message frequency
  - Low latency critical
  
Use WebRTC when:
  - Peer-to-peer audio/video
  - Avoiding server as intermediary for media
```

---

## 2. WebSocket Architecture

### HTTP Upgrade Handshake

```
WebSocket connection starts as HTTP, then upgrades:

Client → Server (HTTP Upgrade Request):
GET /chat HTTP/1.1
Host: example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13

Server → Client (101 Switching Protocols):
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=

After this: same TCP connection, WebSocket framing protocol
```

### WebSocket Frame Format

```
WebSocket Frame:
  Bit 0:    FIN (1 if last fragment)
  Bits 1-3: RSV1-3 (reserved, 0 unless extension used)
  Bits 4-7: Opcode
    0x0: continuation
    0x1: text frame (UTF-8)
    0x2: binary frame
    0x8: connection close
    0x9: ping
    0xA: pong
  Bit 8:    MASK (1 if payload masked, required from client)
  Bits 9-15: Payload length (7 bits; 126→16-bit ext; 127→64-bit ext)
  [4 bytes masking key if MASK=1]
  [Payload data, XOR'd with masking key]
```

### WebSocket Server Implementation (Node.js)

```javascript
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

// Connection registry: userID → WebSocket
const connections = new Map();

wss.on('connection', (ws, req) => {
  const userId = authenticateFromRequest(req);
  connections.set(userId, ws);
  
  ws.on('message', (data) => {
    const msg = JSON.parse(data);
    handleMessage(userId, msg, ws);
  });
  
  ws.on('close', (code, reason) => {
    connections.delete(userId);
    updatePresence(userId, 'offline');
  });
  
  ws.on('error', (err) => {
    console.error(`WebSocket error for user ${userId}:`, err);
  });
  
  // Send a ping every 30 seconds
  const heartbeat = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.ping();
    } else {
      clearInterval(heartbeat);
    }
  }, 30000);
  
  ws.on('pong', () => {
    ws.isAlive = true;  // Mark as alive on pong response
  });
});

// Send message to specific user
function sendToUser(userId, message) {
  const ws = connections.get(userId);
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(message));
  }
}
```

### Connection Persistence and Heartbeat

```
Heartbeat pattern:
  Server: sends PING frame every 30 seconds
  Client: responds with PONG automatically (handled by browser)
  Server: if no PONG within 60 seconds → close connection as dead

Why needed:
  - Idle TCP connections silently dropped by firewalls/load balancers
  - NAT tables expire idle entries (~30-120 seconds depending on config)
  - Heartbeat keeps connection alive AND detects dead clients

Client reconnect strategy (exponential backoff):
  let delay = 1000; // Start 1 second
  function connect() {
    ws = new WebSocket('wss://api.example.com/ws');
    ws.onclose = () => {
      setTimeout(() => {
        delay = Math.min(delay * 2, 30000);  // Cap at 30 seconds
        connect();
      }, delay + Math.random() * 1000);  // Add jitter
    };
    ws.onopen = () => { delay = 1000; };  // Reset on success
  }
```

---

## 3. Scaling WebSockets

### The Problem: Sticky State

```
WebSocket connections are stateful and long-lived.
When a user receives a message, it must be routed to the server holding their connection.

Problem with naive round-robin LB:
  User A connects to Server 1
  User B sends message to User A → goes to Server 2 (round robin)
  Server 2 has no connection to User A → message lost!
```

### Solution 1: Sticky Sessions (Session Affinity)

```
Load balancer routes all requests from same client to same server:

IP Hash:
  server = hash(client_ip) % num_servers
  Pro:  Simple, no shared state needed
  Con:  Uneven distribution (many clients behind same NAT/proxy)

Cookie-based affinity:
  LB sets cookie on first request: SERVERID=server-3
  Subsequent requests routed to server-3
  Pro:  Works even behind NAT
  Con:  LB must remember mapping

AWS ALB target group stickiness:
  enable_load_balancer_stickiness = true
  stickiness_duration = 86400  # 1 day
```

### Solution 2: Redis Pub/Sub Backbone

```
Architecture:
  Server 1 ──── Redis Pub/Sub ──── Server 2
  |                                    |
  User A                            User B

Message flow (User A sends to User B, both on different servers):
  1. User A → Server 1
  2. Server 1 looks up: "User B is on Server 2"  (or broadcasts to all servers)
  3. Server 1 publishes to Redis channel "user:B"
  4. Server 2 subscribed to "user:B" → receives message
  5. Server 2 → WebSocket → User B

User-to-server mapping:
  Redis Hash: user:{userId}:server → serverId
  Update on connect/disconnect
  TTL: heartbeat_interval × 2 (auto-cleanup of dead entries)
```

### Redis Pub/Sub Implementation

```python
import redis
import asyncio

redis_client = redis.Redis(host='redis-cluster')

async def handle_connection(user_id: str, ws):
    # Register this server as handler for user
    redis_client.hset('ws_servers', user_id, SERVER_ID)
    
    # Subscribe to this user's personal channel
    pubsub = redis_client.pubsub()
    pubsub.subscribe(f"ws:user:{user_id}")
    
    async def redis_listener():
        for message in pubsub.listen():
            if message['type'] == 'message':
                await ws.send(message['data'])
    
    # Run redis listener and WebSocket receiver concurrently
    await asyncio.gather(
        redis_listener(),
        websocket_receiver(user_id, ws)
    )

def send_to_user(target_user_id: str, message: dict):
    redis_client.publish(
        f"ws:user:{target_user_id}",
        json.dumps(message)
    )
```

### Horizontal Scaling with WebSocket Clusters

```
Production architecture:

                     [L7 Load Balancer (nginx/ALB)]
                     (IP hash for session affinity)
                    /         |           \
           [WS Server 1] [WS Server 2] [WS Server 3]
                    \         |           /
                     [Redis Pub/Sub Cluster]
                              |
                     [Message Queue (Kafka)]
                              |
                     [Business Logic Services]

Capacity:
  Each WebSocket server: ~50,000 concurrent connections (2GB RAM server)
  With 3 servers: 150,000 concurrent users
  Add more servers as needed (stateless via Redis)
```

---

## 4. Server-Sent Events (SSE)

### Protocol

```
SSE is a simple HTTP-based protocol for server-to-client streaming:

Server response:
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"type": "message", "text": "Hello!"}\n\n

event: notification\n
data: {"id": 456, "text": "New like"}\n
id: 789\n
\n

retry: 3000\n   ← Client should retry after 3 seconds on disconnect
\n
```

### SSE Server Implementation (Node.js)

```javascript
app.get('/events', (req, res) => {
  // SSE headers
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no'  // Important for nginx not to buffer
  });
  
  const userId = req.session.userId;
  const clientId = `${userId}-${Date.now()}`;
  
  // Add to connected clients registry
  sseClients.set(clientId, { res, userId });
  
  // Send initial connection event
  res.write(`data: ${JSON.stringify({type: 'connected'})}\n\n`);
  
  // Heartbeat to keep connection alive
  const heartbeat = setInterval(() => {
    res.write(':ping\n\n');  // SSE comment line (no event fired on client)
  }, 15000);
  
  req.on('close', () => {
    clearInterval(heartbeat);
    sseClients.delete(clientId);
  });
});

// Send event to specific user
function sendSSEToUser(userId, event) {
  for (const [id, client] of sseClients) {
    if (client.userId === userId) {
      client.res.write(`event: ${event.type}\n`);
      client.res.write(`data: ${JSON.stringify(event.data)}\n\n`);
    }
  }
}
```

### SSE Client (Browser)

```javascript
const evtSource = new EventSource('/events', { withCredentials: true });

evtSource.onmessage = (e) => {
  const data = JSON.parse(e.data);
  updateUI(data);
};

evtSource.addEventListener('notification', (e) => {
  showNotification(JSON.parse(e.data));
});

evtSource.onerror = (err) => {
  console.error('SSE error:', err);
  // Browser auto-reconnects using 'retry' field value
};

// Close when done
evtSource.close();
```

### SSE vs WebSocket

| Feature | SSE | WebSocket |
|---------|-----|-----------|
| Direction | Server → Client only | Bidirectional |
| Protocol | HTTP | Custom over TCP |
| Reconnect | Automatic (browser handles) | Manual |
| Browser support | All modern browsers | All modern browsers |
| HTTP/2 | Multiple SSE streams multiplexed | N/A (separate TCP) |
| Proxy/firewall | Works through most proxies | May be blocked |
| Overhead | Low (simple text protocol) | Very low (binary frames) |
| Use case | Notifications, feeds, live data | Chat, games, collaboration |

---

## 5. Push Notifications

### Architecture Overview

```
[Your Backend]
      |
      | HTTPS
      v
[Firebase Cloud Messaging (FCM) / APNs]
      |
      | Mobile network / persistent connection
      v
[User's Device]
      |
      v
[App / Notification Tray]
```

### Firebase Cloud Messaging (FCM)

```python
import firebase_admin
from firebase_admin import messaging

# Send to single device (FCM token)
def send_push_notification(fcm_token: str, title: str, body: str, data: dict):
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        data=data,  # Custom key-value pairs
        token=fcm_token,
        android=messaging.AndroidConfig(
            priority='high',
            notification=messaging.AndroidNotification(
                icon='notification_icon',
                color='#f45342',
                channel_id='default'
            )
        ),
        apns=messaging.APNSConfig(
            payload=messaging.APNSPayload(
                aps=messaging.Aps(
                    badge=1,
                    sound='default',
                    content_available=True
                )
            )
        )
    )
    response = messaging.send(message)
    return response

# Send to topic (fan-out to all subscribers)
def send_to_topic(topic: str, notification_data: dict):
    message = messaging.Message(
        topic=topic,  # e.g., 'breaking-news-sports'
        data=notification_data
    )
    messaging.send(message)

# Send to multiple devices (up to 500 per call)
def send_multicast(tokens: List[str], title: str, body: str):
    message = messaging.MulticastMessage(
        notification=messaging.Notification(title=title, body=body),
        tokens=tokens
    )
    response = messaging.send_multicast(message)
    
    # Handle failures
    if response.failure_count > 0:
        for idx, resp in enumerate(response.responses):
            if not resp.success:
                handle_invalid_token(tokens[idx], resp.exception)
```

### APNs (Apple Push Notification Service)

```
APNs uses HTTP/2 with JWT or certificate authentication:

POST https://api.push.apple.com/3/device/{device-token}

Headers:
  apns-topic: com.example.MyApp
  apns-push-type: alert  (or background, voip, fileprovider)
  apns-priority: 10  (10=immediate, 5=conserve power)
  apns-expiration: 0  (0=expire immediately if device offline)
  authorization: bearer {JWT_TOKEN}

Body:
{
  "aps": {
    "alert": {"title": "Hello", "body": "World"},
    "badge": 5,
    "sound": "default",
    "content-available": 1,
    "mutable-content": 1
  },
  "customKey": "customValue"
}
```

### Web Push (VAPID)

```javascript
// Server-side (Node.js with web-push library)
const webpush = require('web-push');

webpush.setVapidDetails(
  'mailto:admin@example.com',
  process.env.VAPID_PUBLIC_KEY,
  process.env.VAPID_PRIVATE_KEY
);

async function sendWebPush(subscription, payload) {
  try {
    await webpush.sendNotification(subscription, JSON.stringify(payload));
  } catch (error) {
    if (error.statusCode === 410) {
      // Subscription expired — remove from DB
      await removeSubscription(subscription.endpoint);
    }
  }
}

// Client-side service worker
self.addEventListener('push', event => {
  const data = event.data.json();
  event.waitUntil(
    self.registration.showNotification(data.title, {
      body: data.body,
      icon: '/icon.png',
      badge: '/badge.png',
      data: { url: data.url }
    })
  );
});
```

---

## 6. Notification System Design

### Multi-Channel Notification Architecture

```
[Event Producers]
  - Order Service
  - Payment Service
  - Social Service
        |
        v
[Kafka Topic: notifications_raw]
        |
        v
[Notification Service]
  1. Fetch user preferences (which channels enabled)
  2. Fetch message template (localized)
  3. Render message
  4. Route to channel handlers
        |
   _____|_____
  |     |     |
  v     v     v
[Email] [SMS] [Push]
[Queue] [Queue] [Queue]
  |       |      |
  v       v      v
[SendGrid] [Twilio] [FCM/APNs]
```

### User Preferences Management

```python
# User notification preferences
class NotificationPreferences:
    user_id: str
    channels: dict  # {channel: enabled}
    quiet_hours: dict  # {start_hour, end_hour, timezone}
    frequency_cap: dict  # {notification_type: max_per_day}
    
# Example preferences:
{
  "user_id": "u123",
  "channels": {
    "email": True,
    "push": True,
    "sms": False,   # User disabled SMS
    "in_app": True
  },
  "quiet_hours": {
    "enabled": True,
    "start": 22,  # 10 PM
    "end": 8,     # 8 AM
    "timezone": "America/New_York"
  },
  "frequency_cap": {
    "marketing": 1,      # Max 1 marketing notification per day
    "transactional": -1  # Unlimited (always send)
  }
}
```

### Notification Templates

```python
# Template-based notification rendering
class NotificationTemplate:
    template_id: str
    channel: str
    locale: str
    subject_template: str  # Jinja2 template
    body_template: str
    
# Template example (email):
subject_template = "Your order #{{ order_id }} has {{ status }}"
body_template = """
Hello {{ user_name }},

Your order #{{ order_id }} placed on {{ order_date }} has been {{ status }}.

{% if status == 'shipped' %}
Tracking number: {{ tracking_number }}
Estimated delivery: {{ estimated_delivery }}
{% endif %}

Thank you for shopping with us!
"""

def render_notification(template, context):
    return {
        'subject': jinja2.from_string(template.subject_template).render(**context),
        'body': jinja2.from_string(template.body_template).render(**context)
    }
```

### Rate Limiting and Priority

```
Notification priority levels:
  CRITICAL:    Payment failure, security alert (always deliver, bypass rate limit)
  HIGH:        Order shipped, password reset (bypass quiet hours)
  MEDIUM:      New message, social interaction (respect quiet hours)
  LOW:         Marketing, weekly digest (aggregate, batch sending)

Frequency capping:
  - Redis key: f"notif_cap:{user_id}:{type}:{date}"
  - Value: count
  - TTL: until end of day
  
  def check_frequency_cap(user_id, notif_type, limit) -> bool:
      key = f"cap:{user_id}:{notif_type}:{today()}"
      count = redis.incr(key)
      redis.expire(key, 86400)
      return count <= limit
```

---

## 7. Fan-Out Patterns

### Fan-Out on Write (Push Model)

```
When User A posts:
  1. Find all followers of User A (potentially millions)
  2. Write post to each follower's timeline cache
  
Pros:
  - Read is O(1): timeline is pre-computed
  - Low read latency
  
Cons:
  - Write amplification: 1 post → millions of writes
  - Hot user problem: celebrity with 10M followers slows the system
  - Wasted storage for inactive followers

Implementation:
  for follower_id in get_followers(user_a_id):
      redis.lpush(f"timeline:{follower_id}", post_id)
      redis.ltrim(f"timeline:{follower_id}", 0, 999)  # Keep last 1000
```

### Fan-Out on Read (Pull Model)

```
When User B opens feed:
  1. Find all users User B follows
  2. Fetch recent posts from each user
  3. Merge and sort chronologically
  
Pros:
  - Write is cheap (just write to own feed)
  - No wasted storage for inactive users
  
Cons:
  - Read is expensive: N users followed × N posts each = N² work
  - High read latency
  - Cache miss problem for users with many followees

Implementation:
  def get_timeline(user_id):
      followees = get_following(user_id)
      posts = []
      for f_id in followees:
          posts.extend(get_recent_posts(f_id, limit=20))
      return sorted(posts, key=lambda p: p.timestamp, reverse=True)[:100]
```

### Hybrid Fan-Out (Twitter/Instagram Approach)

```
Strategy:
  - Regular users (< N followers): fan-out on write
  - Celebrities (> N followers, e.g., 1M): fan-out on read
  
Implementation:
  On post:
    1. Write post to author's post store
    2. If author is not celebrity:
         fan_out_to_all_followers(post_id)
    3. If author IS celebrity:
         just record post_id in celebrity post store
    
  On read:
    1. Fetch pre-computed timeline (fan-out-on-write users)
    2. Fetch recent posts from followed celebrities
    3. Merge both streams → final sorted timeline

Why it works:
  - Celebrities post infrequently relative to their follower count
  - Read merging with few celebrities is cheap (N celebrity posts, not N×M)
  - Regular users: fast write fan-out, O(1) reads
```

---

## 8. Real-Time Feed Design

### Redis Sorted Set for Timeline

```python
# Timeline stored as sorted set: score = timestamp, member = post_id
# O(log N) insert, O(log N + k) range query

class TimelineService:
    def add_post(self, user_id: str, post_id: str, timestamp: float):
        key = f"timeline:{user_id}"
        redis.zadd(key, {post_id: timestamp})
        redis.zremrangebyrank(key, 0, -1001)  # Keep only 1000 most recent
    
    def get_timeline(self, user_id: str, cursor: float = None, limit: int = 20):
        key = f"timeline:{user_id}"
        if cursor:
            # Cursor-based pagination: get posts older than cursor
            return redis.zrevrangebyscore(key, cursor - 1, '-inf', 
                                          start=0, num=limit, withscores=True)
        else:
            return redis.zrevrange(key, 0, limit - 1, withscores=True)
    
    def delete_post(self, user_id: str, post_id: str):
        key = f"timeline:{user_id}"
        redis.zrem(key, post_id)
```

### Pub/Sub for Live Updates

```python
# When user has feed open, stream new posts in real-time
# User subscribes to a "new posts" channel

async def subscribe_to_feed_updates(user_id: str, ws):
    channel = f"feed_updates:{user_id}"
    
    async with redis.subscribe(channel) as sub:
        async for message in sub.listen():
            if message['type'] == 'message':
                post_id = message['data']
                post = await get_post_details(post_id)
                await ws.send(json.dumps({
                    'type': 'new_post',
                    'post': post
                }))

# Publisher side (when a new post is fan-out)
def publish_to_active_followers(post_id: str, follower_ids: List[str]):
    for follower_id in follower_ids:
        # Only publish if user is currently online
        if is_user_online(follower_id):
            redis.publish(f"feed_updates:{follower_id}", post_id)
```

---

## 9. Chat System Architecture

### Message Flow

```
Client A sends message to Client B:

1. Client A → WebSocket → Chat Server (via LB with sticky sessions)
2. Chat Server validates message
3. Stores message in DB (Cassandra for high-write throughput)
4. Publishes to message queue (Kafka topic: "messages")
5. Message Consumer looks up: "Which server is Client B connected to?"
6. Consumer publishes to Redis channel for that server
7. Target WS Server delivers message to Client B via WebSocket
8. Client B sends ACK
9. Message marked as "delivered" in DB

For group chats:
  Step 5-7 repeated for each group member (fan-out)
```

### Message Storage Schema (Cassandra)

```cql
-- Optimized for "get messages in a conversation, newest first"
CREATE TABLE messages (
  conversation_id UUID,
  message_id      TIMEUUID,   -- Sorted chronologically (time + random UUID)
  sender_id       UUID,
  content         TEXT,
  message_type    TEXT,       -- text, image, video, file
  status          TEXT,       -- sent, delivered, read
  created_at      TIMESTAMP,
  PRIMARY KEY (conversation_id, message_id)
) WITH CLUSTERING ORDER BY (message_id DESC)
  AND compaction = {'class': 'TimeWindowCompactionStrategy',
                    'compaction_window_size': '1',
                    'compaction_window_unit': 'DAYS'};

-- Fetch latest 50 messages
SELECT * FROM messages
WHERE conversation_id = ?
ORDER BY message_id DESC
LIMIT 50;

-- Pagination using message_id as cursor
SELECT * FROM messages
WHERE conversation_id = ?
AND message_id < ?   -- cursor (exclusive)
ORDER BY message_id DESC
LIMIT 50;
```

### Read Receipts

```
States: SENT → DELIVERED → READ

SENT:      Message stored in server DB
DELIVERED: Device received message (even if app in background)
READ:      User opened and viewed the message

Implementation:
  1. Single read receipt: simple status column update
  2. Group read receipts: separate table
  
CREATE TABLE message_receipts (
  message_id    UUID,
  user_id       UUID,
  status        TEXT,  -- delivered, read
  timestamp     TIMESTAMP,
  PRIMARY KEY (message_id, user_id)
);

-- Broadcast read receipt to group:
When user opens a group conversation:
  1. Mark all unread messages as "read" for this user
  2. Publish read receipt events to Kafka
  3. Fan-out read receipt notifications to all group members
  4. Other clients update read receipt indicators
```

---

## 10. Presence Service

### Architecture

```
Presence service tracks who is online, when they were last seen, and their status.

Components:
  1. Heartbeat receiver: users send heartbeat every 5 seconds
  2. Presence store: Redis (fast reads, TTL-based auto-expiry)
  3. Presence publisher: broadcasts status changes
  4. Presence query: look up status of specific users

User online status:
  Redis key: f"presence:{user_id}"
  Value: {status, last_seen, device}
  TTL: heartbeat_interval × 3 = 15 seconds (miss 3 heartbeats = offline)
```

### Heartbeat Implementation

```python
# Client side: send heartbeat every 5 seconds
async def send_heartbeats(ws):
    while True:
        await ws.send(json.dumps({
            "type": "heartbeat",
            "timestamp": time.time()
        }))
        await asyncio.sleep(5)

# Server side: update presence
class PresenceService:
    def heartbeat(self, user_id: str, device: str = "web"):
        key = f"presence:{user_id}"
        data = {
            "user_id": user_id,
            "status": "online",
            "last_seen": datetime.utcnow().isoformat(),
            "device": device
        }
        redis.setex(key, 15, json.dumps(data))  # TTL=15s
    
    def get_presence(self, user_id: str) -> dict:
        data = redis.get(f"presence:{user_id}")
        if data:
            return json.loads(data)  # Online
        
        # Check last_seen from DB for "2 hours ago" display
        return {
            "status": "offline",
            "last_seen": db.get_last_seen(user_id)
        }
    
    def get_bulk_presence(self, user_ids: List[str]) -> dict:
        pipeline = redis.pipeline()
        for uid in user_ids:
            pipeline.get(f"presence:{uid}")
        results = pipeline.execute()
        
        return {
            uid: json.loads(r) if r else {"status": "offline"}
            for uid, r in zip(user_ids, results)
        }
```

### Status Broadcasting

```
When user comes online/goes offline:
  1. Detect: TTL expiry (offline) or heartbeat (online)
  2. Publish to Kafka topic "presence_changes"
  3. Presence consumer fan-outs to interested parties:
     - Friends/followers who are online
     - Active chat conversations

Scaling:
  - Don't subscribe to presence for 10,000 friends (too much traffic)
  - Limit: track presence only for recent contacts / open conversations
  - On-demand: fetch presence when user opens a conversation
```

---

## 11. Notification Delivery Guarantees

### Delivery Guarantee Levels

```
At-most-once:   Send and forget. Possible message loss.
At-least-once:  Retry until acknowledged. Possible duplicates.
Exactly-once:   Practically: at-least-once + idempotent consumer.

For notifications: at-least-once + deduplication is standard.

Why not exactly-once?
  - FCM/APNs don't support exactly-once semantics
  - Network failures require retries
  - Better to show duplicate notification than miss critical alert
```

### Deduplication Strategy

```python
def send_notification(notification: Notification):
    dedup_key = f"notif:dedup:{notification.id}"
    
    # Atomic check-and-set
    if not redis.set(dedup_key, "1", nx=True, ex=3600):
        # Already processed
        logger.info(f"Duplicate notification {notification.id}, skipping")
        return
    
    try:
        deliver_to_channel(notification)
    except Exception as e:
        # Remove dedup key so retry can proceed
        redis.delete(dedup_key)
        raise e
```

### Notification Expiry

```python
class Notification:
    id: str
    type: str
    priority: str
    expires_at: datetime  # Don't deliver after this time

def should_send(notification: Notification) -> bool:
    if notification.expires_at and datetime.utcnow() > notification.expires_at:
        log_expired(notification)
        return False
    return True

# Expiry policies by type:
expiry_policies = {
    "order_shipped":    timedelta(days=7),
    "flash_sale":       timedelta(hours=4),
    "breaking_news":    timedelta(hours=1),
    "password_reset":   timedelta(minutes=30),
    "marketing":        timedelta(days=1),
    "chat_message":     None  # Never expire (deliver when online)
}
```

---

## 12. Email Delivery Pipeline

### Architecture

```
[Notification Service]
        |
        v
[Email Queue (SQS/Kafka)]   ← Rate-limited consumption
        |
        v
[Email Service]
  1. Render template
  2. Check unsubscribe list
  3. Check bounce/complaint list
  4. Send via SendGrid/SES/Mailgun
        |
        v
[SendGrid / AWS SES]
        |
        v
[SMTP → Recipient's Mail Server]
        |
   [Webhook callbacks]
        |
        v
[Event Processor]
  - delivered → update status
  - opened    → update engagement
  - clicked   → update engagement  
  - bounced   → mark email as invalid
  - spam      → unsubscribe immediately
```

### SendGrid Integration

```python
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content

class EmailService:
    def __init__(self):
        self.sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    
    def send_email(self, to: str, subject: str, html_content: str, 
                   from_email: str = 'noreply@example.com',
                   custom_args: dict = None):
        message = Mail(
            from_email=from_email,
            to_emails=to,
            subject=subject,
            html_content=html_content
        )
        
        if custom_args:
            message.custom_args = custom_args  # For webhook correlation
        
        try:
            response = self.sg.send(message)
            return response.status_code
        except Exception as e:
            logger.error(f"SendGrid error: {e}")
            raise

# Bounce handling
def handle_sendgrid_webhook(event_data: List[dict]):
    for event in event_data:
        if event['event'] == 'bounce':
            db.mark_email_bounced(event['email'], event['type'])
            if event['type'] == 'permanent':
                db.blacklist_email(event['email'])
        
        elif event['event'] == 'spamreport':
            db.unsubscribe_email(event['email'], reason='spam')
        
        elif event['event'] == 'unsubscribe':
            db.unsubscribe_email(event['email'], reason='manual')
```

### Email Rate Limiting

```
Sending limits:
  SendGrid:   100/second (paid), 100/day (free)
  AWS SES:    14 emails/second, 50,000/day (sandbox: 200/day)
  
Rate limiting strategy:
  - Token bucket per provider
  - Per-user rate limit (max N emails/day per recipient)
  - Warm-up new IPs gradually (ISPs trust established senders more)
  
IP reputation:
  - Dedicated IPs for transactional (critical) email
  - Shared IPs for marketing (cheaper, but affected by others)
  - Monitor: Sender Score, Postmaster Tools
```

---

## 13. SMS Delivery

### Twilio Integration

```python
from twilio.rest import Client

class SMSService:
    def __init__(self):
        self.client = Client(
            os.environ['TWILIO_ACCOUNT_SID'],
            os.environ['TWILIO_AUTH_TOKEN']
        )
        self.from_number = os.environ['TWILIO_PHONE_NUMBER']
    
    def send_sms(self, to: str, body: str) -> str:
        message = self.client.messages.create(
            to=to,
            from_=self.from_number,
            body=body
        )
        return message.sid
    
    def handle_webhook(self, request_data: dict):
        message_sid = request_data.get('MessageSid')
        status = request_data.get('MessageStatus')
        
        # Statuses: queued, sending, sent, delivered, undelivered, failed
        if status == 'delivered':
            db.update_notification(message_sid, status='delivered')
        elif status in ('undelivered', 'failed'):
            db.update_notification(message_sid, status='failed')
            # Retry logic or fallback to email
```

### International SMS Considerations

```
E.164 format: +[country code][number] → +14155552671

Carrier lookups:
  - Before sending, verify number validity
  - Determine carrier (needed for some regional routing)
  - Detect if landline (SMS not possible) vs mobile

Country-specific issues:
  - India: require DLT (Distributed Ledger Technology) registration
  - China: requires local China SMS providers
  - USA/Canada: 10DLC (10-digit long code) registration required for A2P
  - Short codes vs long codes vs toll-free numbers

Cost optimization:
  - Short codes: premium, fast delivery
  - Long codes: cheaper, higher spam risk
  - WhatsApp API: ~80% cheaper for international messages
```

---

## 14. In-App Notification Center

### Unread Count with Redis

```python
class NotificationCenter:
    def add_notification(self, user_id: str, notification: dict):
        notif_id = generate_id()
        
        # Store notification details
        key = f"notification:{notif_id}"
        redis.setex(key, 86400 * 30, json.dumps(notification))  # 30-day TTL
        
        # Add to user's notification list (sorted set, score=timestamp)
        redis.zadd(f"notifications:{user_id}", 
                   {notif_id: time.time()})
        
        # Increment unread count (atomic)
        redis.incr(f"unread_count:{user_id}")
        
        # Cap to last 100 notifications
        redis.zremrangebyrank(f"notifications:{user_id}", 0, -101)
    
    def get_notifications(self, user_id: str, offset: int = 0, limit: int = 20):
        # Get notification IDs, newest first
        ids = redis.zrevrange(f"notifications:{user_id}", offset, offset + limit - 1)
        
        # Fetch details in batch
        pipeline = redis.pipeline()
        for notif_id in ids:
            pipeline.get(f"notification:{notif_id}")
        results = pipeline.execute()
        
        return [json.loads(r) for r in results if r]
    
    def mark_as_read(self, user_id: str, notification_ids: List[str]):
        pipeline = redis.pipeline()
        for notif_id in notification_ids:
            # Update notification status
            data = json.loads(redis.get(f"notification:{notif_id}") or '{}')
            data['read'] = True
            pipeline.set(f"notification:{notif_id}", json.dumps(data))
        
        # Decrease unread count
        unread = redis.decrby(f"unread_count:{user_id}", len(notification_ids))
        redis.set(f"unread_count:{user_id}", max(0, unread))  # Never negative
        pipeline.execute()
    
    def get_unread_count(self, user_id: str) -> int:
        count = redis.get(f"unread_count:{user_id}")
        return int(count) if count else 0
```

### Mark All As Read

```python
def mark_all_read(self, user_id: str):
    redis.set(f"unread_count:{user_id}", 0)
    
    # Update all notification statuses
    ids = redis.zrange(f"notifications:{user_id}", 0, -1)
    pipeline = redis.pipeline()
    for notif_id in ids:
        raw = redis.get(f"notification:{notif_id}")
        if raw:
            data = json.loads(raw)
            if not data.get('read'):
                data['read'] = True
                pipeline.set(f"notification:{notif_id}", json.dumps(data))
    pipeline.execute()
```

---

## 15. Live Collaboration

### Operational Transformation (OT)

OT ensures that concurrent edits to a shared document produce a consistent result:

```
Initial state: "HELLO"

User A types "W" at position 5: "HELLOW"     → operation A: insert("W", 5)
User B types "!" at position 5: "HELLO!"     → operation B: insert("!", 5)

Without OT:
  A applies first, then B: insert("!", 5) → "HELLOW!" (correct)
  B applies first, then A: insert("W", 5) → "HELLO!W" (W before !)

With OT:
  transform(A, B) → A' = insert("W", 6)  (adjust position since B inserted at 5)
  transform(B, A) → B' = insert("!", 5)  (position unchanged)
  
  Apply A then B': "HELLOW" then insert("!", 5) → "HELLO!W" ← still wrong in this example
  
  Actually: B' transforms with A already applied:
  Result: "HELLO!W" vs "HELLOW!" → OT guarantees both converge to same state
```

**OT properties:**
- **Causality preservation:** Each operation sees all operations that causally preceded it
- **Convergence:** All sites converge to the same document state

### CRDTs for Collaborative Editing

CRDT-based approach (used by Figma, Google Docs newer versions):

```python
# RGA (Replicated Growable Array) - CRDT for sequences
class RGANode:
    def __init__(self, char: str, uid: str, prev_uid: str):
        self.char = char
        self.uid = uid         # Unique ID (timestamp + node_id)
        self.prev_uid = prev_uid  # Points to predecessor
        self.deleted = False

class RGADocument:
    def __init__(self):
        self.nodes = {}  # uid → RGANode
        self.head = None
    
    def insert(self, char: str, prev_uid: str, uid: str):
        node = RGANode(char, uid, prev_uid)
        self.nodes[uid] = node
        # No transformation needed! Just insert after prev_uid
        # Concurrent inserts at same position: sorted by uid (deterministic)
    
    def delete(self, uid: str):
        self.nodes[uid].deleted = True  # Tombstone, never remove
    
    def get_text(self) -> str:
        # Traverse linked list, skip tombstones
        ...
```

### OT vs CRDT Comparison

| Dimension | OT | CRDT |
|-----------|----|----|
| Complexity | High (transformation functions) | Moderate (data structure design) |
| Memory | Low (no tombstones needed) | High (tombstones accumulate) |
| Performance | O(n²) transform in worst case | O(1) insert/delete |
| Correctness | Hard to prove correct | Mathematically proven |
| History | Can reconstruct history | May need separate history log |
| Used by | Google Docs (original), Wave | Figma, Atom Teletype, many modern systems |

---

## 16. Real-Time Gaming Considerations

### UDP for Low Latency

```
TCP vs UDP for gaming:

TCP problems for games:
  - Head-of-line blocking: lost packet stalls all subsequent data
  - Retransmission adds latency (200ms RTT → 400ms delay for retry)
  - Nagle algorithm buffers small packets (disable with TCP_NODELAY)

UDP benefits:
  - No connection setup delay
  - No retransmission (send and forget)
  - No head-of-line blocking
  - Application controls resend policy

Custom reliability over UDP:
  - Sequence numbers to detect out-of-order / lost packets
  - Application-level ACK for critical state (health, scores)
  - Don't retry: send fresh state instead of retransmitting old state

QUIC (used by HTTP/3):
  - UDP-based with stream multiplexing
  - 0-RTT connection establishment
  - No head-of-line blocking between streams
  - Built-in encryption (TLS 1.3)
```

### Client-Side Prediction

```
Without prediction:
  Player presses W → sends to server → server validates → sends new position → player moves
  Latency: 50-200ms → feels sluggish

With client-side prediction:
  Player presses W → instantly moves locally (prediction)
               → also sends to server
               → server validates and sends authoritative position
  
  If server position differs from predicted:
    → "Reconcile": interpolate to correct position (rubber banding if large delta)

Dead reckoning:
  Client predicts OTHER players' positions:
  "Player B was at (100, 200) moving at 5 units/sec east"
  "Predict: B is now at (125, 200)"
  Update when server correction arrives
```

### Server Reconciliation

```
Sequence of events:
  1. Client sends input with sequence number: {seq: 42, action: "move_right"}
  2. Client saves input history: {42: "move_right", 43: "jump", ...}
  3. Server processes, sends: {authoritative_pos: (150, 0), last_processed_seq: 42}
  4. Client:
     a. Correct position to (150, 0) (authoritative)
     b. Re-apply unacknowledged inputs (seq 43 onward)
     c. Resulting position = correct position after all pending inputs
```

---

## 17. Video and Audio Streaming Protocols

### WebRTC for P2P

```
WebRTC connection setup (ICE/STUN/TURN):

1. Browser A (behind NAT) → STUN server: "What's my public IP:port?"
   STUN server → A: "You are 203.0.113.1:50000"

2. Browser B (behind NAT) → STUN server: "What's my public IP:port?"
   STUN server → B: "You are 198.51.100.5:51000"

3. A and B exchange ICE candidates via signaling server (WebSocket)
4. A tries to connect directly to B's public IP:port
5. If NAT traversal fails: use TURN server (relay, more expensive)

Signaling server (not part of WebRTC spec, you build this):
  - Exchanges SDP (Session Description Protocol) offers/answers
  - Exchanges ICE candidates
  - WebSocket or HTTP-based

Media flows peer-to-peer (bypasses server → low latency, low server cost)
```

### SFU vs MCU for Group Calls

```
SFU (Selective Forwarding Unit):
  Each participant sends 1 stream to SFU
  SFU forwards each stream to all other participants
  Each participant receives N-1 streams
  
  Pros:
    - Server is simple relay (CPU-efficient)
    - Each participant controls their own receive quality (simulcast)
  Cons:
    - Client upload: 1 stream
    - Client download: N-1 streams (bandwidth heavy for large groups)
  Used by: Zoom, Discord, most modern video platforms

MCU (Multipoint Control Unit):
  Each participant sends 1 stream to MCU
  MCU decodes, mixes, re-encodes into 1 composite stream
  Each participant receives 1 mixed stream
  
  Pros:
    - Client bandwidth: constant (1 up, 1 down)
    - Client CPU: low
  Cons:
    - Server CPU: very high (decode/encode all streams)
    - Quality loss from re-encoding
    - Can't customize layout per viewer
  Used by: older enterprise video conferencing

Scale limits:
  SFU: 50-100 participants per room (typical)
  MCU: 500+ participants (server handles all mixing)
  Broadcast (HLS/DASH): unlimited viewers
```

### HLS/DASH for Broadcast Streaming

```
HLS (HTTP Live Streaming, Apple):
  1. Encoder → Segments of 2-10 seconds (.ts files or fMP4)
  2. Upload to CDN (S3 → CloudFront)
  3. Master playlist (.m3u8) lists quality renditions
  4. Client downloads playlist → downloads segments sequentially
  
DASH (Dynamic Adaptive Streaming over HTTP, MPEG standard):
  Similar to HLS, but open standard
  Segments can be MPEG-4 or WebM

Latency:
  Traditional HLS/DASH: 15-30 seconds latency (3-5 segments buffered)
  Low-latency HLS (LLHLS): 2-5 seconds latency (smaller segments, partial segments)
  WebRTC P2P: 50-150ms latency

ABR (Adaptive Bitrate):
  Client monitors download speed
  Switches quality based on bandwidth:
    1080p (8 Mbps) → 720p (4 Mbps) → 480p (2 Mbps) → 360p (1 Mbps)
```

---

## 18. Quick Reference

### Real-Time Protocol Comparison Matrix

| Protocol | Direction | Latency | Overhead | Reconnect | Scale | Use Case |
|---------|-----------|---------|---------|-----------|-------|---------|
| Short Poll | Pull | High (~30s) | Very High | N/A | Good | Dashboards |
| Long Poll | Pull | Medium (~500ms) | High | Manual | Moderate | Simple push |
| SSE | Push | Low (~100ms) | Low | Automatic | Good | Feeds, alerts |
| WebSocket | Both | Very Low (~50ms) | Very Low | Manual | Complex | Chat, games |
| WebRTC | P2P | Lowest (~50ms) | Medium | Manual | P2P | Video calls |

### Notification Channel Trade-Offs

| Channel | Delivery Rate | Latency | Cost | Open Rate | Best For |
|---------|--------------|---------|------|----------|---------|
| Push notification | 70-90% | Seconds | Very Low | 5-10% | Re-engagement, alerts |
| In-app | 100% (if online) | <1s | Free | 100% | Active users only |
| SMS | 95%+ | Seconds | High | 98% | Critical alerts, OTP |
| Email | 85-95% | Minutes | Low | 20-30% | Digests, transactional |
| WhatsApp | 95%+ | Seconds | Medium | 80%+ | Global, conversational |

### Fan-Out Pattern Decision Guide

```
Choose Fan-Out on Write when:
  - Read latency is critical (social media feed)
  - Most users have moderate follower counts
  - Users access feed frequently

Choose Fan-Out on Read when:
  - Write performance is critical
  - Many inactive users (wasted fan-out)
  - Follower counts are high for all users

Choose Hybrid when:
  - Power users (celebrities) have huge follower counts
  - Regular users have normal follower counts
  - (Twitter, Instagram approach)
```

### Common Interview Questions

**Q: Design a notification system for 100 million users.**
Answer: Event producers (services) → Kafka topic → Notification service (reads user preferences, renders templates, routes to channels) → per-channel queues → delivery workers → FCM/APNs/SendGrid/Twilio. Use rate limiting per user, idempotency keys, and exponential backoff for retries.

**Q: How would you scale WebSocket connections to 10M concurrent users?**
Answer: Deploy WebSocket servers behind IP-hash load balancer. Each server handles 50K connections. Use Redis Pub/Sub as message backbone. Store user→server mapping in Redis. With 10M users: 200 servers × 50K = 10M. Use Kubernetes HPA to auto-scale.

**Q: What is the difference between fan-out on read vs fan-out on write?**
Answer: Fan-out on write precomputes timelines when posts are created (fast reads, expensive writes, wasted storage for inactive users). Fan-out on read fetches posts from all followees at read time (cheap writes, expensive reads). Hybrid approach: fan-out on write for regular users, fan-out on read for celebrities.

**Q: How does WebRTC establish a peer-to-peer connection?**
Answer: ICE (Interactive Connectivity Establishment) process: both peers contact STUN servers to discover public IP/port. Exchange ICE candidates and SDP via signaling server (your WebSocket server). Try direct connection. If NAT traversal fails, relay through TURN server.
