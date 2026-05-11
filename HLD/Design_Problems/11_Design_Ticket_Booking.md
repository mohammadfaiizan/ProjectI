# System Design: Ticket Booking System

---

## 1. Problem Statement & Clarifying Questions

### Problem Statement
Design a ticket booking system (like BookMyShow or Ticketmaster) that allows users to browse events, select seats, hold them temporarily, and complete purchases. The system must handle high concurrency during popular event launches without double-booking seats.

### Clarifying Questions

**Scale & Traffic:**
- How many concurrent users during peak launch? *500K concurrent users*
- How many events are live at once? *~10K active events*
- How many seats per event on average? *~50K seats (stadiums)*
- What is the read:write ratio? *~100:1 (browsing >> booking)*

**Functionality:**
- Is seat selection mandatory or can users get auto-assigned? *Both modes*
- How long is the hold window? *10 minutes*
- Do we support partial bookings (some seats fail)? *No — all-or-nothing per booking*
- Is waitlisting required? *Yes, virtual waiting room for popular events*
- What payment methods are supported? *Credit card, UPI, wallets*

**Business Rules:**
- Can seats be transferred/resold? *No (out of scope)*
- Is overbooking allowed (airline-style)? *Optional, configurable per event*
- What happens on payment failure — is the seat re-held? *No, seat returns to available*

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
1. Browse events by category, city, date
2. View seat map with real-time availability
3. Hold selected seats for 10 minutes
4. Complete payment to confirm booking
5. Cancel booking and trigger refund
6. Receive confirmation email/SMS asynchronously
7. Virtual waiting room for high-demand events
8. Admin tools: create event, configure overbooking, view analytics

### Non-Functional Requirements
| Property | Target |
|---|---|
| Availability | 99.99% (4 nines) |
| Consistency | Strong (no double-booking) |
| Latency (seat hold) | < 500ms p99 |
| Latency (seat availability read) | < 100ms p99 |
| Durability | Zero booking loss |
| Throughput | 500K concurrent users, 50K bookings/minute at peak |
| Scalability | Horizontal scale-out |

### Out of Scope
- Event organizer portal (beyond admin APIs)
- Secondary ticket marketplace
- Mobile app specifics

---

## 3. Capacity Estimation

### Traffic
- Peak concurrent users: 500K
- Peak booking attempts: ~8,000/second (assuming 10% convert in first minute)
- Seat availability reads: ~800,000/second (100:1 read:write)
- Events: 10K active; ~500 "hot" events at any time

### Storage
- Events table: 10K events × 1 KB = 10 MB
- Seats table: 10K events × 50K seats × 200 bytes = **100 GB**
- Reservations: 1M bookings/day × 500 bytes × 365 days = **~180 GB/year**
- Payments: similar to reservations

### Cache (Redis)
- Seat availability bitmap per event: 50K bits = 6.25 KB/event → 10K events = **62.5 MB**
- Hot seat maps cached in Redis; TTL = 30 seconds

### Network
- Booking payload: ~2 KB/request
- Peak inbound: 8K req/s × 2 KB = **16 MB/s**

---

## 4. High-Level Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              Clients                     │
                        │    (Web / iOS / Android / 3rd Party)    │
                        └─────────────┬───────────────────────────┘
                                      │ HTTPS
                        ┌─────────────▼───────────────────────────┐
                        │           API Gateway / CDN              │
                        │   (Rate limiting, Auth, SSL termination) │
                        └──┬──────────┬──────────┬────────────────┘
                           │          │          │
              ┌────────────▼──┐  ┌────▼─────┐  ┌▼──────────────┐
              │  Event Service│  │ Booking  │  │ Waiting Room  │
              │  (read-heavy) │  │ Service  │  │ Service       │
              └────────┬──────┘  └────┬─────┘  └───────┬───────┘
                       │              │                 │
              ┌────────▼──────────────▼─────────────────▼───────┐
              │                  Message Bus (Kafka)             │
              └────────┬──────────────────────────────┬─────────┘
                       │                              │
          ┌────────────▼──────┐          ┌────────────▼──────────┐
          │  Notification     │          │  Analytics / Reporting │
          │  Service          │          │  Service               │
          │  (Email/SMS/Push) │          └───────────────────────┘
          └───────────────────┘

              ┌─────────────────────────────────────────────────┐
              │                 Data Layer                       │
              │  ┌──────────────┐    ┌───────────────────────┐  │
              │  │  PostgreSQL  │    │  Redis Cluster        │  │
              │  │  (Primary +  │    │  - Seat availability  │  │
              │  │   3 Replicas)│    │  - Hold expiry keys   │  │
              │  └──────────────┘    │  - Rate limit counters│  │
              │                      └───────────────────────┘  │
              │  ┌──────────────┐    ┌───────────────────────┐  │
              │  │  Payment     │    │  Elasticsearch        │  │
              │  │  Gateway     │    │  (Event search)       │  │
              │  │  (Stripe)    │    └───────────────────────┘  │
              └─────────────────────────────────────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Booking Service (Critical Path)

The Booking Service handles the transactional core:

**Hold Flow:**
1. Client sends `POST /bookings/hold` with `{event_id, seat_ids[], user_id}`
2. Service acquires distributed lock per `event_id:seat_id` via Redis `SET NX PX 10000`
3. Reads seat state from DB with `SELECT ... FOR UPDATE` (pessimistic lock on the DB row)
4. Transitions seat from `AVAILABLE → HELD`, records expiry timestamp
5. Writes hold record with TTL; schedules expiry in priority queue
6. Releases DB row lock, keeps Redis hold key alive

**Confirm Flow (Payment 2-Phase):**
1. Client calls `POST /bookings/confirm` with payment token
2. Service charges via PSP (Stripe); on success → transitions `HELD → CONFIRMED`
3. Updates DB atomically; fires Kafka event for notification

**Expiry Flow:**
1. Background worker reads min-heap of expiry timestamps
2. For each expired hold: transitions `HELD → AVAILABLE`, clears Redis key
3. Also handled by Redis key TTL expiry with keyspace notifications as backup

### 5.2 Seat State Machine

```
                    hold()
    AVAILABLE ─────────────────► HELD
         ▲                         │
         │   hold_expired()        │ confirm() + payment_ok
         ◄─────────────────────    │
         │                         ▼
         │   cancel()          CONFIRMED
         ◄──────────────────────────│
         │                         │ cancel_after_confirm()
         │                         ▼
         └────────────────── CANCELLED
```

### 5.3 Virtual Waiting Room

For events expecting >50K concurrent users at launch:
1. Users are placed in a queue with a signed JWT ticket (position, timestamp)
2. Waiting room service drains N users/second into the booking flow
3. Frontend polls `/queue/status` → returns `{position, estimated_wait}`
4. Prevents thundering herd on DB; acts as a token bucket at application layer

### 5.4 Seat Map Service

- Seat map stored as a 2D grid configuration in DB (section, row, seat_number, category)
- Availability overlay is a bitmap in Redis (1 bit per seat; 0=available, 1=taken)
- Reads hit Redis cache; misses fall through to read replica
- Bitmap updated on every state transition via Lua script (atomic)

### 5.5 Concurrency Control: Optimistic vs Pessimistic Locking

| Strategy | Mechanism | Use Case | Downside |
|---|---|---|---|
| **Pessimistic** | `SELECT FOR UPDATE` | Seat hold (contention expected) | Held locks during payment; lower throughput |
| **Optimistic** | Version number check (`WHERE version=N`) | Low-contention updates (profile, event details) | CAS retry storms under high contention |
| **Distributed Lock** | Redis `SET NX PX` | Cross-service coordination | Lock expiry must be > operation time |

**Decision:** Use pessimistic locking for seat hold operations since contention is expected and holding a lock for < 50ms is acceptable. Use optimistic locking for event metadata updates.

---

## 6. Database Design

```sql
-- Events
CREATE TABLE events (
    event_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL,
    venue_id        UUID NOT NULL,
    event_time      TIMESTAMPTZ NOT NULL,
    total_seats     INT NOT NULL,
    available_seats INT NOT NULL,
    overbooking_pct DECIMAL(5,2) DEFAULT 0.0,  -- e.g. 5.00 = 5% overbooking
    status          VARCHAR(20) DEFAULT 'ACTIVE',  -- ACTIVE, CANCELLED, COMPLETED
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    INDEX idx_event_time (event_time),
    INDEX idx_status (status)
);

-- Seats (one row per physical seat)
CREATE TABLE seats (
    seat_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id    UUID NOT NULL REFERENCES events(event_id),
    section     VARCHAR(50),
    row_num     VARCHAR(10),
    seat_num    INT,
    category    VARCHAR(20),  -- PREMIUM, STANDARD, ECONOMY
    price       BIGINT NOT NULL,  -- in minor units (cents)
    status      VARCHAR(20) DEFAULT 'AVAILABLE',  -- AVAILABLE, HELD, CONFIRMED, CANCELLED
    version     INT DEFAULT 0,  -- for optimistic locking
    held_by     UUID,  -- user_id holding this seat
    held_until  TIMESTAMPTZ,
    INDEX idx_event_status (event_id, status),
    INDEX idx_held_until (held_until) WHERE status = 'HELD'
);

-- Reservations (booking header)
CREATE TABLE reservations (
    reservation_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL,
    event_id        UUID NOT NULL,
    status          VARCHAR(20) DEFAULT 'PENDING',  -- PENDING, CONFIRMED, CANCELLED
    total_amount    BIGINT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    confirmed_at    TIMESTAMPTZ,
    cancelled_at    TIMESTAMPTZ,
    INDEX idx_user_event (user_id, event_id),
    INDEX idx_status (status)
);

-- Reservation Items (seats within a booking)
CREATE TABLE reservation_items (
    item_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reservation_id  UUID NOT NULL REFERENCES reservations(reservation_id),
    seat_id         UUID NOT NULL REFERENCES seats(seat_id),
    price_at_booking BIGINT NOT NULL
);

-- Payments
CREATE TABLE payments (
    payment_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reservation_id  UUID NOT NULL REFERENCES reservations(reservation_id),
    user_id         UUID NOT NULL,
    amount          BIGINT NOT NULL,
    currency        CHAR(3) DEFAULT 'USD',
    status          VARCHAR(20),  -- PENDING, COMPLETED, FAILED, REFUNDED
    psp_reference   VARCHAR(255),  -- Stripe charge ID
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    INDEX idx_reservation (reservation_id),
    INDEX idx_psp_ref (psp_reference)
);
```

**Sharding Strategy:**
- Shard `seats` and `reservations` by `event_id` — keeps all data for one event co-located
- Events table on a single shard (small, read-heavy, cached)
- Consider hash-partitioning within PostgreSQL before moving to full sharding

---

## 7. API Design

### REST Endpoints

```
GET    /events?city=NYC&date=2025-01-15&category=concert
GET    /events/{event_id}
GET    /events/{event_id}/seats                    # Seat map with availability
GET    /events/{event_id}/seats?section=A&status=AVAILABLE

POST   /bookings/hold
Body:  { event_id, seat_ids: [uuid, ...], user_id }
Response: { hold_id, expires_at, total_amount }

POST   /bookings/confirm
Body:  { hold_id, payment_token }
Response: { reservation_id, confirmation_number, seats: [...] }

POST   /bookings/{reservation_id}/cancel
Response: { status, refund_amount, refund_eta }

GET    /bookings/{reservation_id}
GET    /users/{user_id}/bookings

POST   /queue/join              # Enter virtual waiting room
Body:  { event_id }
Response: { queue_token, position, estimated_wait_seconds }

GET    /queue/status?token=xxx
Response: { position, estimated_wait_seconds, can_proceed: bool }
```

### Response Codes
- `200 OK` — successful read
- `201 Created` — booking created
- `409 Conflict` — seat already held/booked
- `410 Gone` — hold expired
- `429 Too Many Requests` — rate limit
- `503 Service Unavailable` — waiting room active

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: Seat Availability Reads (800K req/s)
**Solution:**
- Redis bitmap cache per event (hot path, sub-millisecond)
- Read replicas for PostgreSQL (3 replicas with async replication)
- HTTP caching headers (Cache-Control: max-age=5) for CDN layer
- Accept slight staleness (a user sees a seat as available but gets conflict on hold — show error and refresh)

### Bottleneck 2: Thundering Herd at Event Launch
**Solution:**
- Virtual waiting room queues users before granting booking access
- Rate limit booking attempts per user (max 3 attempts per 5 min)
- Pre-warm Redis cache with event data before launch
- Read replicas absorb browse traffic

### Bottleneck 3: Lock Contention on Hot Seats
**Solution:**
- Pessimistic lock scope is minimal (single seat row, held < 50ms)
- Shard DB by event_id so each event has its own shard — locks don't cross shards
- Redis distributed lock provides first-line guard before DB lock

### Bottleneck 4: Payment Latency
**Solution:**
- Async payment processing: hold seats synchronously, charge asynchronously
- If payment fails: seat returns to AVAILABLE immediately
- Timeout: if PSP doesn't respond in 30s, release hold and mark payment FAILED

### Bottleneck 5: Notification Volume (50K emails at once)
**Solution:**
- Kafka topic `booking.confirmed` → Notification Service consumers
- Partitioned by event_id for ordering; scaled with consumer group
- Async — does not block booking confirmation response

---

## 9. Trade-offs & Design Decisions

### Decision 1: Pessimistic vs Optimistic Locking for Seat Hold
- **Chosen:** Pessimistic (`SELECT FOR UPDATE`)
- **Why:** For seats, contention is expected at launch. Optimistic locking leads to retry storms where 1000 users try to book the same seat and 999 retry, amplifying DB load.
- **Trade-off:** Lower throughput on the locked rows, but correctness is guaranteed.

### Decision 2: Seat Hold Expiry — DB Polling vs Redis TTL
- **Chosen:** Both (belt-and-suspenders)
- **Redis TTL:** `EXPIRE hold:event_id:seat_id 600` — fast, but Redis can lose data
- **DB polling:** Background worker queries `WHERE status='HELD' AND held_until < NOW()` every 30s
- **Why:** Redis handles 99% of expirations; DB polling handles Redis failures

### Decision 3: Synchronous vs Asynchronous Hold Confirmation
- **Chosen:** Hold is synchronous; payment → confirmation is async-friendly but acknowledged synchronously
- **Why:** Users need immediate feedback that their seats are secured

### Decision 4: Overbooking Strategy
- Configurable `overbooking_pct` per event (default 0)
- When enabled: allow `total_seats × (1 + pct/100)` confirmations
- Excess bookings enter a standby list; compensation offered if standby doesn't clear
- **Use case:** Airline-style concerts with high expected no-show rates

### Decision 5: Strong Consistency vs Eventual Consistency
- **Booking path:** Strong consistency required — reads and writes go to primary DB
- **Browse path:** Eventual consistency acceptable — read replicas with 1-2s lag
- **Cache:** Short TTL (30s) on seat availability; accept showing stale data to browsing users

---

## 10. Key Interview Talking Points

1. **Double-Booking Prevention:** The combination of `SELECT FOR UPDATE` at the DB level and a Redis distributed lock (`SET NX PX`) at the application level creates two layers of protection. The DB lock is the final arbiter.

2. **Seat Hold Timeout:** Implemented as a priority queue (min-heap by expiry time) in the Booking Service, backed by Redis TTL as a safety net. Expired holds transition atomically to AVAILABLE.

3. **Two-Phase Booking:** Phase 1 holds seats (fast, transactional). Phase 2 processes payment (slower, can fail). This decouples seat reservation from payment latency.

4. **Virtual Waiting Room:** Prevents 500K users from hitting the DB simultaneously. Acts as a controlled token-bucket drain. Users get a signed JWT with queue position; a separate service manages the queue.

5. **Kafka for Notifications:** Booking confirmation triggers a Kafka event. Notification consumers handle email/SMS without blocking the critical booking path. Enables fan-out to multiple downstream systems.

6. **Read Scalability:** 3 PostgreSQL read replicas handle browse traffic. Redis cache handles hot seat-map reads. CDN caches event listings. Booking writes always go to primary.

7. **Sharding by Event ID:** Keeps all seats and reservations for one event on the same DB shard, enabling co-located transactions without cross-shard distributed transactions.

8. **Idempotent Booking:** Each booking attempt generates a client-side idempotency key. The server stores it to prevent duplicate charges on network retry.

9. **Failure Handling:** If payment PSP times out, the hold is released and the user is prompted to retry. If Kafka is down, notification failures are logged and retried via dead-letter queue; they do not block booking.

10. **Metrics to Monitor:** Hold-to-confirm conversion rate, hold expiry rate (high = bad UX), double-booking attempt rate (should be 0), payment success rate, queue drain rate during peak events.
