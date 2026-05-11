# System Design: Uber (Ride-Sharing Platform)

---

## 1. Problem Statement

Design a ride-sharing platform like Uber. Riders can request rides, the system finds the nearest available driver, matches them, tracks the ride in real-time, calculates surge pricing, handles payments, and stores trip history. The system must support millions of concurrent drivers and riders globally.

---

## 2. Clarifying Questions to Ask

- What is the scale? (DAU riders, concurrent drivers)
- What geographic regions must be supported?
- Should we support different ride types (UberX, UberXL, Uber Black)?
- How accurate must ETAs be? (±1 min? ±5 min?)
- What is the frequency of driver location updates? (every 4 seconds?)
- Should we design the payment processing system?
- How does surge pricing work? (demand/supply ratio? geographic zones?)
- Should we support scheduling rides in advance?
- What is the driver-to-rider matching radius? (configurable per city?)
- Do we need to handle cancellations and refunds?

---

## 3. Functional Requirements

1. Riders can request a ride from their current location to a destination.
2. System finds the nearest available driver and sends them the request.
3. Driver can accept or reject the ride request.
4. Rider sees real-time driver location on map during pickup/ride.
5. System calculates ETA for pickup and destination.
6. Surge pricing is applied based on local demand/supply ratio.
7. Trip fare is calculated at completion based on distance and time.
8. Riders can rate drivers (1-5 stars) and vice versa.
9. Full trip history available for both riders and drivers.
10. Driver receives navigation to pickup and then to destination.

---

## 4. Non-Functional Requirements

- **Availability**: 99.99% — must always be able to request a ride
- **Latency**: Driver match P99 < 5 seconds; location update P99 < 1 second
- **Accuracy**: Driver location must be accurate within 50 meters
- **Consistency**: Strong consistency for trip state (can't have two drivers matched to same ride)
- **Scalability**: 20M rides/day; 1M concurrent drivers; 5M concurrent riders
- **Real-time**: Location updates every 4 seconds from drivers
- **Durability**: All trip records and payment data must be durable

---

## 5. Capacity Estimation

### Scale
- Daily rides: 20M
- Concurrent active drivers: 1M
- Concurrent active riders: 5M
- Location updates per driver: every 4 seconds
- Total location updates/sec: 1M drivers / 4 sec = 250,000 location writes/sec
- Peak location updates/sec: ~500,000

### Ride Requests
- 20M rides/day → 231 rides/sec average
- Peak (rush hour, 3x): ~700 rides/sec

### Storage
- Per trip record: ~2 KB (start/end location, timestamps, fare, driver, rider)
- Daily trip storage: 20M * 2KB = 40 GB/day
- Annual storage: 40GB * 365 = ~15 TB/year
- Driver location (Redis, volatile): 1M drivers * ~100 bytes = ~100 MB (fits in RAM)

### Bandwidth
- Location updates: 250,000 * 100B = 25 MB/s inbound
- Location pushes to riders (1M active rides * 1 update/4s): ~25 MB/s outbound
- Map/routing data: ~50 MB/s (cached heavily)

---

## 6. High-Level Architecture

```
[Rider App]                   [Driver App]
    |                               |
    | (HTTPS REST)          (HTTPS REST + WebSocket)
    v                               v
[Load Balancer (Layer 7)]
          |
    [API Gateway]
   /       |       \
[Ride     [Location  [User
 Service]  Service]  Service]
    |          |          |
    v          v          v
[Matching  [Redis GEO   [PostgreSQL]
 Service]  (driver locs)]
    |
[Surge
 Pricing]
    |
[Routing/ETA
 Service]
    |
[Google Maps
 / MapBox API]

[Trip Service] <--> [PostgreSQL - trips table]
[Payment Service] <--> [Stripe / Payment Gateway]
[Notification Service] <--> [APNs / FCM]
[Analytics] <--> [Kafka → Spark → ClickHouse]
```

### Request a Ride Flow
```
1. Rider sends POST /rides with pickup + destination
2. Ride Service creates pending ride record in DB
3. Matching Service queries Redis GEO for nearby available drivers
4. Matching Service sends ride request to top 3 nearest drivers
5. First driver to accept → assigned via atomic Redis lock
6. Ride status updated to ACCEPTED in PostgreSQL
7. Rider notified via WebSocket: driver ETA, driver info, license plate
8. Driver location pushed to rider every 4 seconds via WebSocket
```

---

## 7. Component Deep-Dive

### 7.1 Location Tracking Service

**Driver sends location every 4 seconds**:
- POST /location with {driver_id, lat, lng, heading, speed, timestamp}
- Location Service receives update
- Writes to Redis GEO: `GEOADD drivers:available {lng} {lat} {driver_id}`
- If driver is on a trip: also updates trip's live location for rider

**Redis GEO commands**:
- `GEOADD drivers:available -73.935242 40.730610 "driver_123"` — add/update
- `GEORADIUS drivers:available -73.9 40.7 5 km ASC COUNT 10` — find nearby drivers
- `GEOPOS drivers:available driver_123` — get driver's current position
- `GEODIST drivers:available driver_1 driver_2 km` — distance between two drivers

**Driver state management**:
- When driver goes offline: `ZREM drivers:available driver_id`
- When driver accepts ride: move from `drivers:available` to `drivers:on_trip` set

### 7.2 Geohash-based Spatial Indexing

**What is Geohash?**
- Divides Earth into a grid of rectangular cells
- Each cell has a string code (e.g., "dr5ru" = lower Manhattan at precision 5)
- Nearby cells share a common prefix
- Precision 6 (~1.2km x 0.6km cells) is ideal for ride matching

**Geohash approach**:
- Drivers are stored by their geohash cell
- Finding nearby drivers: look in driver's cell + 8 adjacent cells
- Index: `drivers:{geohash}` → set of driver_ids
- Update on location change: remove from old cell, add to new cell

**Geohash vs Redis GEO**:
| Feature | Geohash | Redis GEO |
|---------|---------|-----------|
| Implementation | Manual | Built-in |
| Nearby search | 9 cells (manual) | GEORADIUS (automatic) |
| Precision control | geohash length | radius parameter |
| Speed | O(1) set lookup | O(N+log N) sorted set |
| Production use | Manual sharding | Simpler at scale |

**Decision**: Use Redis GEO for simplicity; use Geohash for manual sharding when traffic exceeds a single Redis node's capacity.

### 7.3 Driver-Rider Matching Algorithm

1. Query `GEORADIUS drivers:available {rider_lat} {rider_lng} 5 km ASC COUNT 10`
2. Get top 10 nearest drivers sorted by distance
3. Filter by: driver rating ≥ 4.0, correct vehicle type, not previously rejected this ride
4. Send ride request to top 3 drivers simultaneously (avoid waiting on slowest driver)
5. First to accept wins (use Redis atomic `SET if not exists` to prevent double-accept)
6. Other 2 drivers receive cancellation notice
7. If all 3 reject within 30 seconds: try next batch of 3 drivers

**Matching Service**:
- Stateless service; scales horizontally
- Uses Redis for distributed locking (ride can only be accepted once)
- Exponential backoff radius: 5km → 10km → 15km if no drivers available

### 7.4 Ride Lifecycle State Machine

```
REQUESTED → SEARCHING → ACCEPTED → DRIVER_ARRIVING → RIDE_IN_PROGRESS → COMPLETED
                |              |
                |              └── DRIVER_CANCELLED → SEARCHING (retry)
                └── NO_DRIVER_FOUND → FAILED
                
COMPLETED → RATED (optional)
REQUESTED → RIDER_CANCELLED
ACCEPTED → RIDER_CANCELLED (cancellation fee if too late)
```

**State transitions are persisted to PostgreSQL (trips table)**:
- Each state change is atomic (optimistic locking with version number)
- Prevents double-accept race condition
- Audit trail for disputes

### 7.5 Surge Pricing

**Algorithm: Demand/Supply Ratio**
```
surge_multiplier = 1.0 + k * max(0, (demand_rate - supply_rate) / supply_rate)
```

Where:
- `demand_rate` = ride requests in last 5 minutes in this geographic zone
- `supply_rate` = available drivers in this zone
- `k` = calibration constant (typically 0.5)

**Surge Zones**:
- City divided into hexagonal cells (Uber uses H3 — Hexagonal Hierarchical Spatial Index)
- Each zone has its own demand/supply counter
- Surge updated every 1 minute
- Stored in Redis: `surge:{zone_id}` → multiplier (float)

**Dynamic Pricing Display**:
- Price shown to rider BEFORE they request (with 5-minute lock)
- Rider must confirm if surge > 2x
- Surge multiplier capped at 4.9x (regulatory limits in some regions)

### 7.6 ETA Calculation

**Pickup ETA**:
1. Driver's current GPS position → destination (rider's pickup)
2. Routing API call (Google Maps / Mapbox)
3. Adjust for: current traffic, time of day, driver's heading/speed
4. Cache similar routes for 5 minutes

**Trip ETA (to destination)**:
1. Pickup location → destination
2. Include predicted traffic for departure time
3. Recalculate every 2 minutes during ride

**Offline ETA estimation**:
- Pre-computed road graph stored locally
- Haversine distance as lower bound: `2R * arcsin(sqrt(sin²(Δlat/2) + cos(lat1)*cos(lat2)*sin²(Δlon/2)))`

### 7.7 Real-time Driver Location Push to Rider

- WebSocket connection maintained between rider and server
- When driver sends location update (every 4s):
  1. Location Service writes to Redis
  2. If driver is on active trip → publish to Pub/Sub channel `trip:{trip_id}:location`
  3. WebSocket server subscribed to that channel pushes to rider's connection
- WebSocket servers are stateful — use consistent hashing to route rider to same server

---

## 8. Database Design

### PostgreSQL: trips
```sql
CREATE TABLE trips (
    id              BIGSERIAL PRIMARY KEY,
    rider_id        BIGINT NOT NULL,
    driver_id       BIGINT,
    status          VARCHAR(30) NOT NULL DEFAULT 'requested',
    pickup_lat      DECIMAL(9, 6) NOT NULL,
    pickup_lng      DECIMAL(9, 6) NOT NULL,
    pickup_address  TEXT,
    dest_lat        DECIMAL(9, 6) NOT NULL,
    dest_lng        DECIMAL(9, 6) NOT NULL,
    dest_address    TEXT,
    pickup_time     TIMESTAMP,
    dropoff_time    TIMESTAMP,
    base_fare       DECIMAL(8, 2),
    surge_mult      DECIMAL(3, 1) DEFAULT 1.0,
    total_fare      DECIMAL(8, 2),
    distance_km     DECIMAL(6, 2),
    duration_min    INT,
    rider_rating    SMALLINT,
    driver_rating   SMALLINT,
    created_at      TIMESTAMP DEFAULT NOW(),
    version         INT DEFAULT 0,         -- optimistic locking
    INDEX idx_rider (rider_id, created_at),
    INDEX idx_driver (driver_id, created_at),
    INDEX idx_status (status)
);
```

### PostgreSQL: drivers
```sql
CREATE TABLE drivers (
    id              BIGSERIAL PRIMARY KEY,
    user_id         BIGINT UNIQUE NOT NULL,
    vehicle_type    VARCHAR(20) NOT NULL,
    license_plate   VARCHAR(20) NOT NULL,
    rating          DECIMAL(3, 2) DEFAULT 5.00,
    total_trips     INT DEFAULT 0,
    is_verified     BOOLEAN DEFAULT FALSE,
    is_available    BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT NOW()
);
```

### PostgreSQL: riders
```sql
CREATE TABLE riders (
    id              BIGSERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    phone           VARCHAR(20),
    rating          DECIMAL(3, 2) DEFAULT 5.00,
    payment_method  VARCHAR(50),
    created_at      TIMESTAMP DEFAULT NOW()
);
```

### Redis Data Structures
```
drivers:available            → GEO Set (driver_id: lat/lng)
drivers:on_trip              → Set of driver_ids currently on trips
driver:status:{driver_id}    → Hash { state, current_trip_id, lat, lng, last_update }
surge:{zone_id}              → Float (multiplier, TTL=60s)
ride:lock:{ride_id}          → String ("driver_id", TTL=30s) — prevents double-accept
trip:location:{trip_id}      → Hash { driver_lat, driver_lng, eta_seconds }
```

---

## 9. API Design

### Request a Ride
```
POST /api/v1/rides
Authorization: Bearer {rider_token}

Request:
{
  "pickup": { "lat": 40.730610, "lng": -73.935242 },
  "destination": { "lat": 40.758896, "lng": -73.985130 },
  "ride_type": "uberx"
}

Response 200:
{
  "ride_id": "trip_abc123",
  "status": "searching",
  "estimated_fare": { "min": 12.50, "max": 15.00, "surge_multiplier": 1.0 },
  "estimated_pickup_eta_min": 3
}
```

### Driver Accepts Ride
```
POST /api/v1/rides/{ride_id}/accept
Authorization: Bearer {driver_token}

Response 200:
{
  "accepted": true,
  "ride_id": "trip_abc123",
  "rider_info": { "name": "Alice", "rating": 4.8 },
  "pickup": { "lat": 40.730610, "lng": -73.935242, "address": "..." },
  "navigation_url": "https://maps.uber.com/nav?trip=abc123"
}

Response 409 (already taken):
{ "error": "ride already accepted by another driver" }
```

### Update Driver Location
```
POST /api/v1/driver/location
Authorization: Bearer {driver_token}

Request:
{
  "lat": 40.730610,
  "lng": -73.935242,
  "heading": 180,
  "speed_kmh": 35
}

Response 200: { "received": true }
```

### Complete a Ride
```
POST /api/v1/rides/{ride_id}/complete
Authorization: Bearer {driver_token}

Response 200:
{
  "ride_id": "trip_abc123",
  "status": "completed",
  "fare": 13.50,
  "distance_km": 5.2,
  "duration_min": 18
}
```

### Get Surge Info
```
GET /api/v1/surge?lat=40.730610&lng=-73.935242

Response 200:
{
  "surge_multiplier": 1.8,
  "zone_id": "zone_manhattan_midtown",
  "message": "Prices are higher due to high demand",
  "valid_until": "2024-01-15T18:45:00Z"
}
```

---

## 10. Scalability & Bottlenecks

### Bottleneck 1: Location Update Writes (250K writes/sec)
- Redis can handle ~1M ops/sec single node
- Solution: Redis Cluster with 5 nodes, each handling ~50K driver locations
- Partition by geographic region (city-level shards)

### Bottleneck 2: Driver Matching Radius Query (700 queries/sec)
- GEORADIUS on large sets (1M drivers) is O(N+log N) — could be slow
- Solution: Geohash-based partitioning — only query drivers in relevant cells
- Redis sharding: `drivers:available:NYC`, `drivers:available:SF`, etc.

### Bottleneck 3: WebSocket Connection Scale (5M concurrent riders)
- Each WebSocket connection = 1 TCP connection on server
- 1 server handles ~50K WebSockets → need 100 WebSocket servers
- Solution: Message broker (Redis Pub/Sub) to fan-out updates across WebSocket servers

### Bottleneck 4: Trip State Race Conditions
- Two drivers accept same ride simultaneously
- Solution: Redis distributed lock `SET ride:lock:{ride_id} driver_id NX EX 30`
- Only first SET succeeds; second returns nil (conflict detected)

### Bottleneck 5: Surge Calculation at Scale
- Real-time aggregation across many zones
- Solution: Kafka stream of ride requests → stream processor (Flink) → Redis zone counters
- Surge recalculated every 60 seconds per zone

---

## 11. Trade-offs & Design Decisions

### Redis GEO vs Geohash
- Redis GEO: Built-in, exact radius queries, but single-node bottleneck
- Geohash: Manual but allows custom sharding strategies
- Decision: Redis GEO for initial scale; migrate to Geohash-based sharding for 10M+ drivers

### WebSocket vs HTTP Polling for Location
- HTTP Polling: Simple, stateless, but high latency (depends on poll interval)
- WebSocket: Real-time, efficient, but stateful (harder to scale)
- Decision: WebSocket for riders (location updates every 4s); HTTP for historical data

### Strong Consistency for Trip State
- Trips use PostgreSQL with optimistic locking (version column)
- Cannot use eventual consistency — must prevent double-accept
- Redis lock provides distributed locking during the accept window

### Surge Zones: Fixed vs Dynamic
- Fixed hexagonal grids (H3): Consistent, predictable
- Dynamic ML-based zones: More accurate but complex
- Decision: Fixed H3 hexagonal zones; ML layer on top for fare prediction

### ETA Accuracy: Pre-computed vs Real-time Routing
- Pre-computed routes: Fast but stale
- Real-time (Google Maps): Accurate but adds latency and cost
- Decision: Cache routes per origin-destination pair for 5 minutes; fall back to Haversine for instant estimates

---

## 12. Key Interview Talking Points

1. **Redis GEO for driver locations**: GEORADIUS command is the key primitive. 1M driver locations fit easily in Redis (~100MB).

2. **Driver location update frequency**: Every 4 seconds is Uber's actual number. 1M drivers / 4s = 250K writes/sec — needs Redis Cluster.

3. **Distributed lock for match acceptance**: Redis `SET NX EX` (Set if Not Exists with Expiry) prevents two drivers from accepting the same ride.

4. **Ride state machine**: Walk through all state transitions. Interviewers love seeing you handle cancellations, timeouts, and failures.

5. **Surge pricing algorithm**: Demand/supply ratio calculation. Explain zone-based calculation (geohash/H3 cells).

6. **Geohash prefix property**: Cells sharing prefix are geographically adjacent — this is what makes prefix-based spatial queries efficient.

7. **WebSocket for real-time tracking**: 5M concurrent WebSocket connections = horizontal scaling with Redis Pub/Sub fanout.

8. **ETA calculation**: Google Maps API + Haversine fallback. Cache results for nearby origin-destination pairs.

9. **Consecutive retries for matching**: 3 drivers at a time, expand radius if all reject, configurable per city.

10. **H3 hexagonal indexing**: Mention that Uber actually uses H3 (open-sourced) for surge zones because hexagons tesselate without gaps and all neighbors are equidistant.
