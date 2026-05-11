# Design a Notification System — High-Level Design

---

## 1. Problem Statement & Clarifying Questions

**Problem Statement:**
Design a large-scale notification system that sends notifications to users through multiple channels (push notifications, email, SMS, in-app) with different priority levels, delivery guarantees, and user preference management.

**Clarifying Questions:**
- What notification channels must be supported? (Push/FCM/APNs, Email, SMS, In-App)
- What is the expected volume? (1B notifications/day)
- Should we support transactional (OTP, order confirmation) and marketing notifications?
- What delivery guarantee is needed — at-least-once, exactly-once?
- Do users need to manage notification preferences per channel?
- Should we support notification templates?
- Do we need delivery status tracking (sent/delivered/opened)?
- Should we support A/B testing on notifications?
- What are the latency requirements for critical notifications?

**Assumptions:**
- 1 Billion notifications per day
- Support Push (FCM/APNs), Email (SendGrid), SMS (Twilio), In-App
- At-least-once delivery with idempotency for deduplication
- User preference management per channel and notification type
- Templates with variable substitution and localization
- Priority queues: Critical (OTP) > Transactional > Marketing

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
1. **Send Notification:** Trigger notification to one or multiple users
2. **Multi-Channel:** Deliver via push, email, SMS, in-app based on preferences
3. **Templates:** Create and render notification templates with variables
4. **User Preferences:** Users control which channels they receive per notification type
5. **Notification Center:** In-app notification history with read/unread status
6. **Fan-out:** Send one event to millions of users (broadcast)
7. **Deduplication:** Prevent duplicate notifications
8. **Retry Logic:** Retry failed deliveries with exponential backoff
9. **Delivery Tracking:** Track sent/delivered/opened status

### Non-Functional Requirements
1. **Latency:** Critical notifications (OTP) delivered in <5 seconds
2. **Throughput:** 1B notifications/day = ~12K/second average, 50K peak
3. **Availability:** 99.99% for notification intake, 99.9% for delivery
4. **Durability:** No notification loss after intake acknowledgment
5. **Consistency:** At-least-once delivery (idempotency handles duplicates)
6. **Scalability:** Horizontal scaling of all components

---

## 3. Capacity Estimation

### Volume
- Daily notifications: 1 Billion
- Average QPS: 1B / 86400 ≈ 12K notifications/second
- Peak QPS: ~50K notifications/second (morning delivery, sales events)

### Channel Distribution
- Push notifications: 60% = 600M/day (mobile apps)
- Email: 25% = 250M/day
- In-App: 10% = 100M/day
- SMS: 5% = 50M/day (expensive, limited to critical)

### Storage
- Notification record: 500 bytes
- Daily new records: 1B * 500B = 500GB/day
- In-app notification history (30 days): 30 * 100M * 500B = 1.5TB
- Templates: 10,000 templates * 5KB = 50MB
- User preferences: 1B users * 200B = 200GB

### Throughput
- Fan-out events: 1 event → 10K users (average broadcast)
- 100 broadcast events/day → 100 * 10K = 1M fan-out messages
- Single user notification events: 999M/day
- Kafka throughput needed: 50K events/second (peak)

---

## 4. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRIGGER SOURCES                                  │
│   User Action    Scheduled Job    System Alert    Marketing Campaign     │
│   (like, follow) (daily digest)   (fraud alert)   (promo email)         │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    NOTIFICATION API SERVICE                              │
│                                                                          │
│  POST /notify                                                            │
│  ├── Validate request (auth, schema)                                    │
│  ├── Lookup user preferences                                            │
│  ├── Deduplicate (idempotency key check)                                │
│  ├── Persist notification to DB (status=PENDING)                        │
│  └── Publish to Kafka                                                   │
└──────────────────────────────────────┬──────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    KAFKA TOPICS (Priority-Based)                         │
│                                                                          │
│  notifications.critical    (OTP, fraud alerts, security)                │
│  notifications.transactional (order confirm, password reset)            │
│  notifications.marketing   (promotions, newsletters, recommendations)   │
│  notifications.fan_out     (social: likes, follows, comments)           │
└───────┬────────────┬────────────────┬──────────────────┬───────────────┘
        │            │                │                  │
        ▼            ▼                ▼                  ▼
┌───────────┐ ┌───────────┐ ┌──────────────┐ ┌──────────────────┐
│  Push     │ │  Email    │ │  SMS         │ │  In-App          │
│  Worker   │ │  Worker   │ │  Worker      │ │  Worker          │
│           │ │           │ │              │ │                  │
│ FCM/APNs  │ │ SendGrid  │ │  Twilio      │ │  WebSocket/DB    │
└───────────┘ └───────────┘ └──────────────┘ └──────────────────┘
        │            │                │                  │
        ▼            ▼                ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DELIVERY TRACKING SERVICE                             │
│              Update notification status: sent/delivered/failed          │
│              Retry failed deliveries (exponential backoff)              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                       │
│                                                                          │
│  ┌──────────────────┐  ┌────────────────────┐  ┌───────────────────┐  │
│  │  PostgreSQL       │  │  Redis Cluster     │  │  Cassandra        │  │
│  │  (notifications, │  │  (dedup cache,     │  │  (in-app notifs,  │  │
│  │   templates,     │  │   rate limits,     │  │   user history)   │  │
│  │   preferences)   │  │   FCM tokens)      │  │                   │  │
│  └──────────────────┘  └────────────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Notification Types

| Type | Example | Latency Req | Channel |
|------|---------|------------|---------|
| Critical | OTP, fraud alert, password reset | <5s | Push + SMS |
| Transactional | Order confirm, shipping update | <30s | Push + Email |
| Social | Like, follow, comment | <5min | Push + In-App |
| Marketing | Promotions, newsletters | Hours (batched) | Email + Push |
| Digest | Daily summary | Scheduled | Email |

### 5.2 Notification Pipeline

**Step 1: Intake**
```
Client → POST /api/v1/notify
{
    "idempotency_key": "order-123-confirm",
    "user_id": "user-456",
    "type": "ORDER_CONFIRMED",
    "priority": "TRANSACTIONAL",
    "data": {
        "order_id": "ORD-123",
        "total": "$49.99",
        "items": ["Laptop", "Mouse"]
    },
    "channels": ["PUSH", "EMAIL"]  // optional override
}
```

**Step 2: Preference Resolution**
- Fetch user preferences: `{EMAIL: on, PUSH: on, SMS: off, MARKETING: off}`
- Check notification type against preferences
- Result: send via EMAIL and PUSH (both allowed and user has enabled them)

**Step 3: Template Rendering**
```
Template: "Your order {{order_id}} for {{total}} has been confirmed!"
Variables: {order_id: "ORD-123", total: "$49.99"}
Rendered:  "Your order ORD-123 for $49.99 has been confirmed!"
```

**Step 4: Fan-out (for broadcast)**
- For events targeting millions of users (e.g., "system maintenance"):
  - Write one event record to DB
  - Fan-out worker reads target user IDs from segments
  - Produces individual notification messages to Kafka
  - Rate-limited fan-out: 10K messages/second per worker

### 5.3 Push Notification Architecture

**FCM (Firebase Cloud Messaging) — Android/Web:**
```
Notification Worker → POST https://fcm.googleapis.com/fcm/send
{
    "to": "{device_token}",
    "notification": { "title": "...", "body": "..." },
    "data": { "click_action": "ORDER_DETAIL", "order_id": "123" }
}
```

**APNs (Apple Push Notification Service) — iOS:**
```
Notification Worker → HTTP/2 POST to api.push.apple.com
Headers: apns-topic, apns-priority, apns-expiration, Authorization (JWT)
Body: { "aps": { "alert": { "title": "...", "body": "..." }, "badge": 1 } }
```

**Device Token Management:**
- Tokens stored in Redis: `device:{user_id}:tokens` → Set of {platform: token}
- Invalid tokens (FCM returns 404) → Remove from Redis
- Multi-device: user has 3 devices → send push to all 3 concurrently

### 5.4 Retry Logic with Exponential Backoff

```
Attempt 1: immediate
Attempt 2: wait 5 seconds
Attempt 3: wait 25 seconds
Attempt 4: wait 125 seconds (2 min)
Attempt 5: wait 625 seconds (10 min)
Max attempts: 5
After max: move to dead-letter queue (DLQ)

Jitter: add random(0, wait_time * 0.1) to prevent synchronized retries

Retry triggers:
- Network timeout
- 5xx from FCM/SendGrid/Twilio
- Rate limit (429) from provider → honor Retry-After header

No retry:
- Invalid device token (404) → clean up token
- Invalid email address (400)
- User unsubscribed (bounce)
```

### 5.5 Deduplication

**Idempotency Key:**
- Client provides idempotency key: `"order-123-confirmed"`
- Server checks Redis before processing: `dedup:{idempotency_key}` → exists?
- If exists: return cached response (200 with original notification_id)
- If not exists: process and store `SETEX dedup:{key} 86400 {notification_id}`

**Why Redis with TTL?**
- Fast O(1) lookup
- Auto-expiry after 24 hours (stale dedup keys cleaned up automatically)
- Atomic SETNX ensures only one request wins the race

**Content-Based Dedup:**
- For notifications without idempotency key:
  - Hash = MD5(user_id + notification_type + content + round_to_minute(timestamp))
  - Prevent same notification sending to same user within 1-minute window

### 5.6 User Preference Management

```
User Preferences Schema:
{
    "user_id": "user-456",
    "channels": {
        "PUSH": true,
        "EMAIL": true,
        "SMS": false,
        "IN_APP": true
    },
    "notification_types": {
        "ORDER_UPDATES": ["PUSH", "EMAIL"],
        "MARKETING": [],            // opted out of marketing
        "SECURITY": ["PUSH", "SMS", "EMAIL"],  // cannot disable
        "SOCIAL": ["IN_APP"]
    },
    "quiet_hours": {
        "enabled": true,
        "start": "22:00",
        "end": "08:00",
        "timezone": "America/New_York"
    },
    "frequency_limits": {
        "MARKETING": "max_1_per_day"
    }
}
```

**Quiet Hours:**
- If notification during quiet hours and priority < CRITICAL:
  - Schedule delivery for `quiet_hours_end` time
  - Store in "scheduled" Kafka topic with timestamp
  - Scheduler service reads and publishes at scheduled time

### 5.7 Notification Center (In-App)

**Storage: Cassandra**
```
CREATE TABLE user_notifications (
    user_id     UUID,
    created_at  TIMESTAMP,
    notif_id    UUID,
    type        TEXT,
    title       TEXT,
    body        TEXT,
    data        TEXT,    -- JSON
    is_read     BOOLEAN,
    channel     TEXT,
    PRIMARY KEY (user_id, created_at, notif_id)
) WITH CLUSTERING ORDER BY (created_at DESC)
  AND default_time_to_live = 2592000;  -- 30 days TTL
```

**Real-time via WebSocket:**
- User connects via WebSocket to Notification Center Service
- Service subscribes to Redis pub/sub channel: `notif:{user_id}`
- When notification arrives: push via WebSocket → immediate in-app display
- On reconnect: fetch unread count + last N notifications from Cassandra

**Unread Count:**
- Redis counter: `notif:{user_id}:unread` → INCR on new, DECR on mark-read
- Badge count on app icon from this counter

### 5.8 A/B Testing Notifications

```
Experiment:
{
    "experiment_id": "promo-email-subject-test",
    "type": "marketing",
    "variants": [
        {"id": "A", "weight": 50, "template": "subject_v1"},
        {"id": "B", "weight": 50, "template": "subject_v2"}
    ],
    "metrics": ["open_rate", "click_rate", "conversion_rate"]
}

Assignment:
- user_variant = hash(user_id + experiment_id) % 100
- variant = "A" if user_variant < 50 else "B"
- Consistent assignment (same user always gets same variant)

Tracking:
- Log {experiment_id, variant, user_id, event_type} to analytics pipeline
- Measure: open_rate = opened / sent, click_rate = clicked / opened
```

---

## 6. Database Design

### PostgreSQL Schema

```sql
CREATE TABLE notifications (
    notification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key VARCHAR(200) UNIQUE,
    user_id         UUID NOT NULL,
    type            VARCHAR(100) NOT NULL,
    priority        ENUM('CRITICAL','TRANSACTIONAL','SOCIAL','MARKETING'),
    title           VARCHAR(200),
    body            TEXT,
    data            JSONB,
    status          ENUM('PENDING','SENT','DELIVERED','FAILED','SKIPPED'),
    channel         ENUM('PUSH','EMAIL','SMS','IN_APP'),
    template_id     UUID,
    sent_at         TIMESTAMP,
    delivered_at    TIMESTAMP,
    opened_at       TIMESTAMP,
    retry_count     SMALLINT DEFAULT 0,
    created_at      TIMESTAMP DEFAULT NOW(),
    INDEX(user_id, created_at DESC),
    INDEX(status, created_at),
    INDEX(idempotency_key)
);

CREATE TABLE notification_templates (
    template_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(100) UNIQUE NOT NULL,
    type            VARCHAR(100),
    channel         ENUM('PUSH','EMAIL','SMS','IN_APP'),
    subject         VARCHAR(200),
    body_template   TEXT NOT NULL,
    variables       JSONB,            -- [{name, required, default}]
    locale          CHAR(5) DEFAULT 'en_US',
    version         INTEGER DEFAULT 1,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE user_notification_preferences (
    user_id         UUID PRIMARY KEY,
    channel_settings JSONB NOT NULL DEFAULT '{}',
    type_settings   JSONB NOT NULL DEFAULT '{}',
    quiet_hours     JSONB,
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE device_tokens (
    token_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL,
    platform        ENUM('IOS','ANDROID','WEB'),
    token           VARCHAR(500) UNIQUE NOT NULL,
    app_version     VARCHAR(20),
    is_active       BOOLEAN DEFAULT TRUE,
    last_used_at    TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW(),
    INDEX(user_id, platform)
);
```

---

## 7. API Design

### Send Notification
```
POST /api/v1/notifications/send
Authorization: Bearer {api_key}
{
    "idempotency_key": "order-456-shipped",
    "user_id": "user-789",
    "type": "ORDER_SHIPPED",
    "priority": "TRANSACTIONAL",
    "data": { "tracking_number": "1Z999AA1", "carrier": "UPS" },
    "channels": ["PUSH", "EMAIL"]  // optional, default from preferences
}
Response 202: { "notification_id": "notif-123", "status": "QUEUED" }
```

### Bulk Send (Fan-out)
```
POST /api/v1/notifications/broadcast
{
    "segment_id": "premium_users",
    "type": "PROMO_SUMMER_SALE",
    "priority": "MARKETING",
    "data": { "discount": "30%", "code": "SUMMER30" }
}
Response 202: { "broadcast_id": "bc-456", "estimated_recipients": 5000000 }
```

### Notification Center
```
GET /api/v1/users/{user_id}/notifications?limit=20&before_id=notif-100
Response: { notifications: [...], unread_count: 5, has_more: true }

PATCH /api/v1/users/{user_id}/notifications/{notification_id}/read
Response: { status: "read", unread_count: 4 }

PATCH /api/v1/users/{user_id}/notifications/read-all
Response: { marked_read: 5, unread_count: 0 }
```

### User Preferences
```
GET  /api/v1/users/{user_id}/notification-preferences
PUT  /api/v1/users/{user_id}/notification-preferences
{
    "channels": { "EMAIL": true, "SMS": false },
    "types": { "MARKETING": { "channels": [] } },
    "quiet_hours": { "enabled": true, "start": "22:00", "end": "08:00" }
}

POST /api/v1/unsubscribe?token={unsubscribe_token}&type=all|marketing
```

---

## 8. Scalability & Bottlenecks

| Bottleneck | Problem | Solution |
|-----------|---------|----------|
| Fan-out for viral events | 100M user broadcast → 100M DB writes | Batch writes, async fan-out workers, Kafka |
| FCM rate limits | Google limits push rates | Connection pooling, batch FCM API |
| Email delivery | ISP rate limits, spam filters | Multiple IP pools, domain warmup, bounce handling |
| Preference lookup | Check preferences per notification | Cache in Redis (TTL 5min), invalidate on update |
| Dedup at scale | 1B idempotency checks/day | Redis with TTL, in-memory bloom filter as pre-filter |
| Notification history | Cassandra scan for user notifications | Partition by user_id, pagination with page tokens |

---

## 9. Trade-offs & Design Decisions

### Fan-out on Write vs On Read for Notification Feed
- **Fan-out on write:** Each event immediately written to each subscriber's notification queue — fast reads, expensive writes for broadcast
- **Fan-out on read:** Pull all events at read time — expensive reads, cheap writes
- **Choice:** Fan-out on write for social notifications (<10K subscribers); fan-out on read for system-wide broadcasts
- **Trade-off:** Read latency vs write amplification

### At-Least-Once vs Exactly-Once Delivery
- **At-least-once:** Simple, Kafka + idempotency keys handle duplicates
- **Exactly-once:** Complex, requires distributed transactions
- **Choice:** At-least-once with client-side idempotency
- **Trade-off:** Some duplicate suppression logic needed vs complex distributed transactions

### Synchronous vs Asynchronous Notification Sending
- **Synchronous:** API waits for FCM/email ACK — simple but slow (200-500ms)
- **Asynchronous (Kafka):** API returns immediately, workers deliver — faster API response, better scalability
- **Choice:** Asynchronous Kafka-based pipeline
- **Trade-off:** Eventual delivery vs synchronous confirmation

### Push vs In-App for Critical Alerts
- **Push only:** Immediate but requires app installed and notifications enabled
- **In-App + Push:** Push for immediate, in-app for history and missed push
- **Choice:** Both — push for immediate delivery, in-app for notification center
- **Trade-off:** More channels to manage vs better reliability

---

## 10. Key Interview Talking Points

1. **Multi-Channel Design:** Explain the abstraction layer — NotificationChannel interface with FCMChannel, EmailChannel, SMSChannel implementations. Factory pattern selects channel. New channels added without changing core logic.

2. **Priority Queues:** Critical notifications (OTP) MUST arrive in seconds — dedicated Kafka consumer group, dedicated worker threads, no batching. Marketing can be delayed, batched, and scheduled for optimal times.

3. **Fan-out Problem:** 1 social event → 1M followers → 1M notifications. Naive approach writes 1M records synchronously. Kafka-based async fan-out with rate limiting prevents system overload. Show the write amplification calculation.

4. **Idempotency:** Clients retry on network failure — need idempotency to prevent double notifications. Redis SETNX with idempotency key is atomic, O(1), with auto-expiry. Essential pattern for at-least-once systems.

5. **Retry Exponential Backoff:** Explain jitter (prevent synchronized retry storms). Dead-letter queue for exhausted retries. Alerting on DLQ depth. Handle transient (retry) vs permanent failures (invalid token — don't retry).

6. **User Preferences:** Store in DB (source of truth), cache in Redis (fast lookup). When preference changes, invalidate cache. Quiet hours require scheduling delayed delivery — use a scheduled Kafka topic.

7. **Deduplication Cache:** Two-layer dedup — idempotency key (explicit) + content hash (implicit). Redis with 24h TTL handles idempotency. Bloom filter as fast pre-filter before Redis lookup.

8. **WebSocket for Real-time In-App:** Persistent connection per user → push notification immediately. Horizontal scaling with sticky sessions or Redis pub/sub for cross-instance delivery. Reconnect/backfill from Cassandra.

9. **Template System:** Templates enable consistency, reuse, and A/B testing. Variable substitution with validation. Localization support (different template per locale). Version management for template updates without breaking in-flight notifications.

10. **Back-of-Envelope:** 1B notifications/day = 12K/sec average, 50K peak. 60% push = 600M FCM calls/day (7K/sec). FCM allows ~1M/sec. Distributed Kafka with 10 partitions handles 500K events/sec. Worker pool scales horizontally — add workers during peak.
