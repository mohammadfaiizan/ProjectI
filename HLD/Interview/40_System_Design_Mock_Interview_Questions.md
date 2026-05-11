# 40 — System Design Mock Interview Questions

> All 20 questions are Hard-level, open-ended, and modeled after senior/staff engineer interviews.
> Model answers cover key architectural decisions, trade-offs, and scale considerations.
> Format: Easy (Q1–7) / Medium (Q8–15) bands are replaced here with a single Hard band
> since all 20 questions are senior/staff level. Difficulty increases from Q1 to Q20.

---

### Q1. Design a global content moderation system for a platform with 500M daily posts (text, image, video). What are the key architectural decisions?

**Scale framing:**
500M posts/day = ~5,800 posts/second average, with peak 3x higher (~17K/sec). Content types have wildly different processing costs: text (< 10ms), image (100ms–2s), video (minutes). You cannot block submission waiting for moderation.

**Architecture: Async pipeline with tiered classifiers**

```
User Submission
      │
      ▼ Accept immediately (return post_id)
      │
   Kafka Topic: raw_content
      │
      ├──► Fast Lane (Synchronous pre-check)
      │    Text only, < 50ms classifier
      │    Block obvious violations before storage
      │    (hash-based deduplication of known CSAM hashes)
      │
      └──► Async Moderation Pipeline
            │
            ├─ Text classifier (BERT fine-tuned): 50ms, batch 1000/s per GPU
            ├─ Image classifier (ResNet): 200ms, GPU worker pool
            └─ Video: extract keyframes → image classifier → audio transcript
                      (most expensive: dedicate worker fleet, 5-30 min SLA)
```

**Key Architectural Decisions:**

**1. Tiered classification (cost vs accuracy vs latency):**
- Tier 1: Hash-based lookup (known bad content, PhotoDNA for CSAM) — microseconds, no ML cost
- Tier 2: Fast classifier — 50–200ms, handles 90% of decisions automatically
- Tier 3: Expensive model — for borderline cases flagged by Tier 2
- Tier 4: Human review queue — for cases where ML confidence < threshold

**2. Visibility states (not binary approve/reject):**
```
post_state:
  PENDING   → submitted, awaiting moderation (visible only to poster)
  VISIBLE   → passed moderation (visible to all)
  SHADOW    → flagged but not told to poster (research policy decision)
  REMOVED   → violated policy (poster notified or not, depending on policy)
  APPEALING → removed, under human appeal review
```

**3. Scaling across content types:**
```
Text:    Stream processing (Flink), 1 GPU = 500 req/s
         500M posts/day ÷ 86400s = 5800/s → 12 GPUs
Image:   200ms/image → 1 GPU handles 5/s
         500M × 50% have images → 2,900/s → 580 GPU workers
Video:   Background workers, SLA = 5 minutes not 5 seconds
         Keyframe extraction → reuse image classifier
```

**4. Human review queue:**
ML sends ~0.5% of content to human review = 2.5M items/day. Requires: queue system (prioritized by severity), reviewer tool (context, user history, similar cases), geographic routing (language-specific reviewers), SLA tracking, reviewer mental health protections (limited exposure sessions).

**5. Appeals system:**
Store decision + model confidence + features. Appeals trigger re-evaluation by different model + senior reviewer. Track false positive rate by content category.

**6. Data storage:**
- Decision log: immutable append-only store (Kafka → S3 → Iceberg table) for audit, model training
- Policy rules: versioned in DB + cached (policy changes are frequent)
- Known bad content: bloom filter + hash store (deduplication)

**Trade-offs:** Pre-moderation (block until reviewed) vs post-moderation (allow then remove). Post-moderation is standard at scale for non-CSAM content — too slow to pre-moderate 17K posts/sec. The harm window (between post and removal) is a business/legal decision.

---

### Q2. You need to build a real-time collaborative spreadsheet like Google Sheets. How does it differ from Google Docs in design?

**Key difference from Google Docs:** A spreadsheet has **formula dependencies** between cells. Editing cell A1 may recalculate 10,000 downstream cells. A document has no such cascading recalculation.

**Conflict Resolution Differences:**

```
Google Docs (text):
  Operational Transformation (OT) or CRDT
  Two users type at position 5 simultaneously → merge by position shift
  Conflict resolution: position-based, straightforward

Google Sheets (cells + formulas):
  Cell edits at different coordinates: EASIER (no position shift needed)
  A1 edited by two users → Last Write Wins is often acceptable per cell
  But: formula recalculation cascade is the hard part
```

**Formula Dependency Graph:**
```
Cell D1 = A1 + B1 + C1
Cell E1 = D1 * 2
Cell F1 = SUM(A1:E1)

Edit A1 → must recalculate: D1, E1, F1 (topological order!)

Dependency Graph (DAG):
  A1 → D1 → E1 → F1
           →  F1
```

**Architecture:**

```
Client Browser (Google Sheets)
  Local cell model + formula evaluator (JavaScript)
  Optimistic local update
      │
      ▼
WebSocket connection (persistent, bidirectional)
      │
      ▼
Spreadsheet Server
  Conflict resolution (last-write-wins per cell + version vectors)
  Formula dependency graph
  Recalculation engine (topological sort)
  Broadcast delta to all connected clients
      │
      ▼
Storage: Bigtable (row = spreadsheet_id + row_id, columns = cell values)
         Versioned: keep last 1000 versions per cell for undo/history
```

**Concurrent Edit Protocol:**
```
User A edits A1 = "100"
User B edits A1 = "200" (concurrent)

Server receives both:
  Vector clock: A1:{A:1} and A1:{B:1}
  Conflict detected. Policy: last writer (by server timestamp) wins.
  Server broadcasts: A1 = "200" (B's value accepted)
  Server sends correction to User A: your value was overridden

For formula cells: recalculate D1, E1, F1 after conflict resolution
  Broadcast recalculated values to all clients
```

**Performance challenges unique to sheets:**
- Large sheets (1M rows): don't load full sheet — virtualize rows, only load visible viewport
- Volatile functions: `=NOW()`, `=RAND()` — recalculate every second for all clients? Use server-push model, not client-poll
- Circular references: detect cycles in dependency DAG on formula save
- Cross-sheet references: `=Sheet2!A1` — dependency graph spans sheets

**Scaling:** Shard by spreadsheet ID. Each spreadsheet has a "session server" responsible for all concurrent edits to that document. If the session server fails, reconnect to new server (load state from Bigtable). Handle this with sticky sessions at load balancer, with failover.

---

### Q3. Design a system that can execute 1 million database queries per second without hitting the database more than 100K times per second.

**Math:** Cache hit rate required = (1M - 100K) / 1M = 90%. This is achievable but requires careful cache key design, proper TTL, and cache warming.

**Architecture: Multi-layer caching**

```
1,000,000 queries/sec incoming
      │
      ├─ Layer 1: In-process cache (per-server memory)
      │   Capacity: 100MB per server, 100 servers = 10GB total
      │   TTL: 5 seconds (hot data stays hot)
      │   Hit rate: ~60% (most hot queries answered here)
      │   Zero network latency
      │
      │  After L1 miss: 400K/sec remaining
      │
      ├─ Layer 2: Distributed cache (Redis Cluster)
      │   Capacity: 500GB across 10 Redis nodes
      │   TTL: 60 seconds
      │   Hit rate: ~75% of misses (L2 hits 300K/sec of L1 misses)
      │   Latency: 0.5–2ms (single hop to Redis)
      │
      │  After L2 miss: 100K/sec remaining
      │
      └─ Layer 3: Database (PostgreSQL / primary)
          100K queries/sec — within database capacity
```

**Cache Key Design:**
```python
def cache_key(query_type: str, params: dict) -> str:
    # Canonical form: sort params, hash for short key
    canonical = json.dumps(params, sort_keys=True)
    param_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return f"{query_type}:{param_hash}"

# Example: user profile
cache_key("user_profile", {"user_id": 123}) → "user_profile:3a7b2f..."
```

**Cache Invalidation Strategy:**
```python
# Write-through: update DB and cache simultaneously
def update_user_profile(user_id: int, data: dict):
    db.update("users", data, where={"id": user_id})
    # Invalidate all cache levels
    key = cache_key("user_profile", {"user_id": user_id})
    local_cache.delete(key)           # L1: instant
    redis.delete(key)                 # L2: ~1ms
    # Database change propagates to L1 on next cold miss

# Event-driven invalidation (for complex dependencies)
kafka.publish("cache_invalidation", {
    "keys": [f"user_profile:{user_id}",
             f"user_settings:{user_id}",
             f"user_permissions:{user_id}"]
})
# All servers subscribe to this topic and invalidate their L1
```

**Hot Key Problem (celebrity problem):**
If one user (Beyoncé's profile) gets 100K queries/sec, a single Redis node becomes a bottleneck.
```python
# Solution: replicate hot keys across multiple Redis nodes
HOT_KEY_REPLICAS = 10

def get_hot_key(key: str) -> any:
    # Distribute reads across N replicas
    replica_idx = random.randint(0, HOT_KEY_REPLICAS - 1)
    replica_key = f"{key}:r{replica_idx}"
    value = redis.get(replica_key)
    if value:
        return value
    # Fetch from DB and populate all replicas
    value = db.get(key)
    for i in range(HOT_KEY_REPLICAS):
        redis.setex(f"{key}:r{i}", 60, value)
    return value
```

**Cache Warming:** On startup, pre-populate L1 from Redis before serving traffic. Prevents thundering herd on deployment.

---

### Q4. How would you design the infrastructure for a live-streaming platform (like Twitch) that needs to handle 100K concurrent streamers and 10M concurrent viewers?

**Scale math:**
- 100K streamers × 6 Mbps average bitrate = 600 Gbps ingest bandwidth
- 10M viewers × 4 Mbps delivery = 40 Tbps delivery bandwidth
- 40 Tbps cannot come from origin servers — CDN is mandatory

**Architecture: Ingest → Transcode → CDN → Edge Delivery**

```
STREAMER (OBS / broadcast software)
  RTMP or WebRTC push (low latency)
      │
      ▼
INGEST EDGE NODES (geographically distributed)
  Accept RTMP stream, buffer 2-4 GOP (Group of Pictures)
  Assign stream to transcoding cluster
      │
      ▼
TRANSCODING CLUSTER (GPU fleet)
  Input: 1080p60 (streamer quality)
  Output: HLS/DASH adaptive bitrate ladder:
    1080p60, 720p60, 720p30, 480p, 360p, 160p
  Latency target: < 3 seconds from capture to viewer
      │
      ▼
ORIGIN SERVERS (stream manifest + segments)
  HLS: .m3u8 manifest + 2-second .ts segments
  Push segments to CDN edge on creation (push model, not wait for pull)
      │
      ▼
CDN EDGE NODES (Akamai, Fastly, CloudFront — 10K+ PoPs globally)
  Cache each 2-second segment for its lifetime (2 seconds + buffer)
  10M viewers × 4 Mbps = 40 Tbps → distributed across CDN PoPs
```

**Low-Latency Considerations:**
```
Traditional HLS: 30-45 second latency (too slow for interactive streaming)
LL-HLS (Apple):  2-5 second latency (partial segments + push)
WebRTC relay:    < 500ms latency (gaming, interactive)

Twitch uses LL-HLS for most streams
  Partial segments: CDN serves partial .ts files as they're written
  HTTP/2 push: server pushes new segments to player before requested
```

**Chat System (1M messages/minute):**
```
Separate from video pipeline entirely.
WebSocket connections to chat servers (one per channel).
Redis Pub/Sub for message fan-out to all viewers of a channel.
Channel with 1M viewers: 1 message → pub/sub → 1M subscriber sockets.
Shard: partition large channels across multiple chat servers.
Rate limiting: 20 messages/30 seconds per user (prevents spam).
```

**Viewer Count (approximate):**
```
Exact count is expensive (requires counting unique sockets).
Use HyperLogLog (Redis PFADD/PFCOUNT) for approximate unique viewer count.
Update every 5 seconds. 99% accuracy within 1% error margin.
```

**Failover:** If a transcoding node fails mid-stream: streamer reconnects to ingest (RTMP reconnect), stream reassigned to new transcoder. Viewer sees up to 10-second gap (HLS manifest has retry logic). Chat continues uninterrupted.

---

### Q5. Design a distributed job scheduler that can handle 10 million scheduled tasks per day with exactly-once execution guarantees.

**Scale:** 10M tasks/day = ~116 tasks/second average. Peak may be 10x (1160/s) around common scheduling times (top of hour, midnight).

**Exactly-once execution challenges:**
- If scheduler crashes after dispatching a task but before recording it: task runs twice
- If worker crashes after starting but before completing: task needs retry but should not duplicate
- Clock skew across distributed nodes causes missed or double triggers

**Architecture:**

```
SCHEDULING DATABASE (PostgreSQL)
──────────────────────────────────────────────────────────────
CREATE TABLE scheduled_tasks (
    task_id         UUID PRIMARY KEY,
    scheduled_at    TIMESTAMPTZ NOT NULL,
    payload         JSONB NOT NULL,
    status          VARCHAR(20) DEFAULT 'PENDING',
        -- PENDING, DISPATCHED, RUNNING, COMPLETED, FAILED
    lease_owner     VARCHAR(100),   -- which scheduler node owns this
    lease_expires_at TIMESTAMPTZ,  -- heartbeat-based lease
    attempts        INT DEFAULT 0,
    max_attempts    INT DEFAULT 3,
    idempotency_key VARCHAR(200) UNIQUE  -- prevents double-execution
);
CREATE INDEX ON scheduled_tasks (scheduled_at, status)
WHERE status = 'PENDING';
```

**Dispatcher (runs on multiple nodes, uses DB-based leader election):**
```python
def dispatch_due_tasks(node_id: str):
    while True:
        now = datetime.utcnow()

        # Claim tasks with optimistic locking (FOR UPDATE SKIP LOCKED)
        # SKIP LOCKED: other nodes skip rows locked by this transaction
        tasks = db.execute("""
            UPDATE scheduled_tasks
            SET status = 'DISPATCHED',
                lease_owner = %s,
                lease_expires_at = %s,
                attempts = attempts + 1
            WHERE task_id IN (
                SELECT task_id FROM scheduled_tasks
                WHERE status IN ('PENDING', 'DISPATCHED')
                  AND scheduled_at <= %s
                  AND (lease_expires_at IS NULL OR lease_expires_at < %s)
                ORDER BY scheduled_at
                LIMIT 100
                FOR UPDATE SKIP LOCKED  ← key: no row-level contention
            )
            RETURNING *
        """, node_id, now + timedelta(minutes=5), now, now)

        for task in tasks:
            job_queue.enqueue(task)   # Kafka / SQS
```

**Worker (exactly-once via idempotency key):**
```python
def execute_task(task: Task):
    # Check if already completed (idempotency)
    if db.scalar("SELECT 1 FROM task_results WHERE idempotency_key = %s",
                 task.idempotency_key):
        logger.info(f"Task {task.task_id} already completed, skipping")
        return

    try:
        result = run_task(task.payload)
        # Atomic: write result + update status in one transaction
        with db.transaction():
            db.insert("task_results", {
                "idempotency_key": task.idempotency_key,
                "result": result
            })
            db.execute("UPDATE scheduled_tasks SET status='COMPLETED' WHERE task_id=%s",
                       task.task_id)
    except Exception as e:
        db.execute("""
            UPDATE scheduled_tasks
            SET status = CASE WHEN attempts >= max_attempts THEN 'FAILED' ELSE 'PENDING' END,
                lease_owner = NULL, lease_expires_at = NULL
            WHERE task_id = %s
        """, task.task_id)
```

**Handling the 12:00 AM spike** (millions of "run at midnight" tasks):
Pre-shard: assign each midnight task to a random 60-second window around midnight. This spreads 1M midnight tasks over 1 minute = 16K/sec instead of a single-second spike.

---

### Q6. How would you design the autocomplete for a search engine that serves 10 billion queries per day across 50 languages?

**Scale:** 10B queries/day = ~115K queries/second. Each query generates autocomplete suggestions at ~100ms keystroke intervals. Multiple keystrokes per query = likely 3–5 autocomplete calls per query = 500K–600K autocomplete requests/second.

**Core Data Structure: Trie + Frequency Ranking**

```
Trie for prefix matching:
  "app" → ["apple", "application", "app store", "apple music"]
  "appl" → ["apple", "application", "apple music", "apple tv"]

Each node stores: character + set of top-K completions (pre-computed)
  (storing completions at each node avoids full trie traversal per query)

Memory: English vocabulary ~500K words × prefix nodes × top-10 completions
        ~5GB in-memory per language tier
```

**At 10B queries/day, 50 languages:**
```
Architecture per language cluster:
  Top languages (English, Spanish, Mandarin, etc.):
    Dedicated in-memory trie service: 5GB RAM per instance
    Multiple read replicas (Redis cluster or custom trie service)
    Updates: batch-rebuilt nightly from query log analysis

  Long-tail languages (rare languages):
    Shared multi-language service
    Lower freshness requirement (weekly rebuild)
```

**Serving Architecture:**
```
Keystroke event → CDN edge PoP
                      │ cache hit? (popular prefixes cached at edge)
                      │ YES → return from edge (~5ms)
                      │ NO → forward to regional autocomplete service
                              │
                              │ Trie lookup: O(prefix_length) = O(10)
                              │ Return top-10 completions
                              │
                              → Cache result at edge (TTL = 1 hour for
                                high-frequency prefixes)
```

**Real-Time Trend Injection:**
```python
# Breaking news: "earthquake" starts trending
# Solution: real-time event injection into autocomplete
# Maintain a separate "trending terms" list updated every 5 minutes
# Merge trending terms with static trie results:

def get_completions(prefix: str, lang: str) -> list:
    static_completions  = trie.get(prefix, lang)       # historical popularity
    trending_completions = trending.get(prefix, lang)  # last 1-hour query spikes

    # Boost trending terms: insert at top if query volume spike > 2x
    return merge_and_rank(static_completions, trending_completions, top_k=10)
```

**Personalization (search history):**
For signed-in users: blend global completions with personal query history.
```
global_completions × 0.7 + personal_history × 0.3
```
Personal history is stored in user profile service, fetched in parallel with trie lookup.

**Query suggestion vs autocomplete:** Autocomplete = complete the current term. Query suggestion = suggest related queries ("people also searched for"). Different models; autocomplete is prefix-match, suggestion is semantic similarity.

---

### Q7. Design a system to detect fraudulent transactions in real-time for a payment network processing 100K transactions per second.

**SLA:** Decision in < 100ms per transaction. False positive rate < 0.5% (blocking legitimate transactions is expensive). False negative rate < 0.1% of fraud (by value, not count — a $10K fraud matters more than a $5 fraud).

**Architecture: Multi-layer scoring**

```
Transaction Event (100K TPS)
      │
      ├─ Layer 1: Rules Engine (< 5ms)
      │  Simple deterministic rules:
      │  - Amount > $10,000 → flag for review
      │  - Country mismatch (card issued in US, transaction in Nigeria, < 2h gap) → deny
      │  - Known fraudulent merchant ID → deny
      │  - Card velocity: > 5 transactions in 1 minute → deny
      │  Handles ~30% of obvious fraud decisions
      │
      ├─ Layer 2: ML Model (< 50ms)
      │  Features from Redis (real-time):
      │    user_txn_count_1h, user_spend_1h, merchant_fraud_rate,
      │    device_fingerprint_age, geo_velocity, card_age
      │  Model: Gradient Boosted Trees (XGBoost)
      │  Output: fraud_score 0.0–1.0
      │  score > 0.8 → deny; 0.5–0.8 → 3DS challenge; < 0.5 → approve
      │
      └─ Layer 3: Human Review Queue
         score 0.5–0.8 AND amount > $500 → queue for analyst
         SLA: 30-minute human review for high-value borderline cases
```

**Feature Store (Redis, sub-millisecond reads):**
```python
features = {
    # User-level velocity (sliding window counters)
    "txn_count_1h":        redis.zcount(f"txn:{user}", now-3600, now),
    "spend_1h":            redis.get(f"spend:{user}:1h"),
    "unique_merchants_24h": redis.pfcount(f"merchants:{user}:24h"),

    # Card-level features
    "card_declined_7d":    redis.get(f"card:{card_id}:declines:7d"),
    "card_age_days":       (now - card_issue_date).days,

    # Merchant features
    "merchant_fraud_rate_30d": feature_store.get(merchant_id, "fraud_rate_30d"),
    "merchant_avg_txn":    feature_store.get(merchant_id, "avg_txn_amount"),

    # Device/session features
    "device_seen_count":   redis.get(f"device:{device_id}:count"),
    "is_new_device":       device_first_seen > datetime.now() - timedelta(days=7),

    # Geo velocity (impossible travel)
    "geo_velocity_km_h":   compute_geo_velocity(user_id, lat, lon, now),
}
```

**Feedback Loop (retraining):**
```
Ground truth labels arrive delayed (chargebacks occur 30-90 days later)
  Denied transactions: labeled as fraud (confirmed) or false positive (dispute)
  Approved transactions: labeled as fraud when chargeback arrives

Weekly retraining pipeline:
  Pull labeled transactions from last 90 days
  Retrain XGBoost model
  Shadow evaluate new model vs champion model for 1 week
  Promote if: lower false negatives AND lower false positives
```

**Operational controls:**
Real-time dashboard showing: fraud rate by geography, merchant category, card type, time of day. SRE can flip rules (kill switch) to adjust thresholds during active fraud campaigns without code deployment.

---

### Q8. How would you design the ride-matching system for a global ride-sharing platform operating in 80 countries with different regulations?

**Core matching challenge:** Given a rider at (lat, lng), find available drivers within N km, sorted by ETA, assign optimally considering surge pricing, driver preferences, and local regulations.

**Geo-spatial index for driver location:**
```python
# Redis GEO: O(log N) radius search
redis.geoadd("drivers:available:london", longitude, latitude, driver_id)
drivers = redis.georadius(
    "drivers:available:london",
    rider_lon, rider_lat,
    radius=3, unit="km",
    withcoord=True, count=20, sort="ASC"   # nearest 20 drivers
)
# Driver location updates: 60M drivers × update every 4 seconds
# = 15M location updates/second globally
# Solution: partition by city/region (separate Redis keys per city)
```

**ETA Calculation:**
```
Approach: Pre-computed ETA vs real-time routing

Driver candidates: 20 nearest (from GEO search)
ETA needed: actual driving time (not straight-line distance)

Option A: Call routing API for each driver → 20 API calls × 50ms = slow
Option B: Pre-computed road segment travel times + Dijkstra locally
  → Build road graph for each city
  → Bidirectional Dijkstra from driver to rider
  → Update segment times every 5 minutes from historical data
  → Faster: 5–10ms per route calculation
```

**Matching Algorithm:**
```python
def match_rider_to_driver(rider: Rider) -> Driver:
    # Step 1: Get candidate drivers within radius
    candidates = geo_index.nearby_drivers(rider.location, radius_km=3, limit=20)

    # Step 2: Score each candidate
    scored = []
    for driver in candidates:
        eta = routing.estimate_eta(driver.location, rider.location)
        score = compute_match_score(
            eta=eta,
            driver_rating=driver.rating,
            vehicle_type_match=driver.vehicle == rider.vehicle_preference,
            acceptance_rate=driver.acceptance_rate,
            is_heading_toward_rider=heading_similarity(driver.heading,
                                                        rider.location)
        )
        scored.append((score, driver, eta))

    # Step 3: Assign highest-scored available driver
    # Atomic: use Redis SETNX to "lock" driver to prevent double-assignment
    for score, driver, eta in sorted(scored, reverse=True):
        if redis.setnx(f"driver:assigned:{driver.id}", rider.id, ex=30):
            return MatchResult(driver=driver, eta=eta)

    return None  # no drivers available
```

**Regulatory Differences (80 countries):**
```
Config-driven rules per country:
{
  "FR": {
    "max_surge_multiplier": 3.0,      # France caps surge pricing
    "require_professional_license": true,
    "require_commercial_insurance": true,
    "allow_motorcycle_rides": false
  },
  "US/CA": {
    "max_surge_multiplier": null,     # no cap
    "require_background_check": true,
    "allow_shared_rides": true
  },
  "CN": {
    "max_surge_multiplier": 2.0,
    "require_local_driver_license": true,
    "data_residency": "required"      # China: data must stay in China
  }
}
```

**Surge pricing:** Demand/supply ratio per grid cell (H3 hexagonal grid). When `demand/supply > 1.3`, surge = 1.3x; ratio > 2.0, surge = 2.0x. Real-time marketplace clearing.

---

### Q9. Design a configuration management system that allows feature rollouts to be targeted to specific users, rolled back instantly, and audited completely.

This is a combination of a **feature flag service** (File 36) and a **configuration management system** with complete auditability.

**Core Requirements:**
- Targeting: specific users, segments, orgs, regions, percentage
- Instant rollback: < 5 second propagation to all services
- Complete audit: who changed what, when, from what value to what value, why

**Data Model:**
```sql
CREATE TABLE config_flags (
    flag_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flag_key     VARCHAR(200) UNIQUE NOT NULL,
    flag_type    VARCHAR(20) NOT NULL,  -- 'boolean', 'integer', 'string', 'json'
    current_value JSONB NOT NULL,
    description  TEXT,
    owner_team   VARCHAR(100) NOT NULL,
    expires_at   TIMESTAMPTZ,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE config_audit_log (
    log_id       BIGSERIAL PRIMARY KEY,
    flag_key     VARCHAR(200) NOT NULL,
    changed_by   VARCHAR(200) NOT NULL,  -- email of person
    change_type  VARCHAR(20),            -- 'created', 'updated', 'deleted', 'rollback'
    old_value    JSONB,
    new_value    JSONB,
    justification TEXT,                  -- required field: "why are you changing this?"
    jira_ticket  VARCHAR(50),            -- link to change request
    changed_at   TIMESTAMPTZ DEFAULT NOW(),
    ip_address   INET,
    session_id   UUID
);

CREATE TABLE targeting_rules (
    rule_id       UUID PRIMARY KEY,
    flag_key      VARCHAR(200) REFERENCES config_flags(flag_key),
    priority      INT NOT NULL,       -- lower = higher priority
    rule_type     VARCHAR(30),        -- 'user_ids', 'org_ids', 'percentage', 'segment'
    rule_config   JSONB NOT NULL,     -- {"user_ids": [123, 456]} or {"percent": 10}
    variant_value JSONB NOT NULL,     -- what value to return when rule matches
    enabled       BOOLEAN DEFAULT TRUE
);
```

**Instant Propagation (< 5 seconds):**
```
Change made via API
      │
      ▼ Write to PostgreSQL (source of truth)
      │
      ▼ Publish to Redis Pub/Sub: channel "config_changes"
      │  Payload: {flag_key, version, new_value}
      │
      ▼ All SDK instances subscribed to Pub/Sub
        Each SDK receives message → invalidates local cache entry
        Next evaluation: fetches fresh value from Redis
        Total propagation: < 500ms
```

**Rollback UI:**
```python
@app.post("/flags/{flag_key}/rollback")
def rollback_flag(flag_key: str, target_version: int, user=Depends(require_auth)):
    # Find the audit log entry for target_version
    historical = db.scalar("""
        SELECT new_value FROM config_audit_log
        WHERE flag_key = %s ORDER BY log_id LIMIT 1 OFFSET %s
    """, flag_key, target_version)

    if not historical:
        raise HTTPException(404, "Version not found")

    current = db.get_flag(flag_key)

    # Apply rollback as a new audit entry (preserve full history)
    db.insert("config_audit_log", {
        "flag_key": flag_key,
        "changed_by": user.email,
        "change_type": "rollback",
        "old_value": current.value,
        "new_value": historical,
        "justification": f"Rollback to version {target_version}",
    })
    db.update_flag(flag_key, value=historical)
    redis.publish("config_changes", json.dumps({"flag_key": flag_key}))
    return {"status": "rolledback", "value": historical}
```

**Approval Workflow (for production flags):**
High-risk flags (ops flags, billing flags) require 2-person approval before applying. Implement as: change goes to `PENDING_APPROVAL` state, sends Slack/email to approvers, second person clicks approve, then flag is applied.

---

### Q10. How would you migrate a 10TB monolithic PostgreSQL database to a microservices architecture with zero downtime?

This is one of the hardest real-world problems. 10TB of data with complex relationships across tables that will be split across independent services, while the application is live.

**Strategy: Strangler Fig + Expand-Contract + Dual Write**

```
MIGRATION PHASES
──────────────────────────────────────────────────────────────
Phase 1: Identify seam (2-4 weeks)
  Map all tables to owning domain (user tables, order tables, inventory)
  Identify foreign key relationships that cross domain boundaries
  These cross-domain FKs are the hardest to break

Phase 2: Add service layer (4-8 weeks)
  Extract first domain (e.g., User service)
  New service has its OWN database (PostgreSQL, subset of data)
  Monolith still owns the data; new service reads from monolith DB
  (no data movement yet, just new service code)

Phase 3: Dual write (4-8 weeks per domain)
  New service writes to BOTH monolith DB and its own DB
  Background job: sync historical data from monolith to service DB
  Validation: compare monolith vs service DB results continuously

Phase 4: Read migration (2-4 weeks)
  Gradually shift READ traffic from monolith to service DB
  Monitor: same results? If yes, 100% reads to service DB

Phase 5: Cut over (1 week)
  Stop dual write to monolith
  Service DB is now authoritative
  Remove foreign keys from monolith pointing to migrated tables

Phase 6: Cleanup
  Drop migrated tables from monolith (after 2-week observation period)
```

**Dual Write Implementation:**
```python
class UserRepository:
    def create_user(self, user_data: dict) -> User:
        # Phase 3: write to BOTH databases
        with db_monolith.transaction():
            user = db_monolith.insert("users", user_data)

        # Async write to new service DB (don't let failure here block)
        try:
            db_user_service.insert("users", user_data)
        except Exception as e:
            # Log for reconciliation; don't fail the request
            reconciliation_queue.publish({"action": "create_user",
                                          "data": user_data})

        return user
```

**Cross-Domain Foreign Key Breaking:**
```sql
-- Before: orders.user_id has FK constraint to users.id (same DB)
-- After: orders.user_id is just an integer (no DB-enforced FK)
-- Consistency enforced by application layer, not DB constraint

-- Step 1: Drop FK constraint (non-disruptive in PostgreSQL)
ALTER TABLE orders DROP CONSTRAINT orders_user_id_fkey;
-- (orders still contain user_id values, just no DB enforcement)

-- Step 2: Application now validates user existence via API call, not JOIN
-- Step 3: Eventually, user service and order service are separate DBs
```

**Data Validation:**
Run continuous reconciliation jobs during Phase 3/4:
```python
def validate_user_migration(user_id: int):
    monolith_user = db_monolith.get_user(user_id)
    service_user  = db_user_service.get_user(user_id)
    if monolith_user != service_user:
        alert(f"Divergence detected for user {user_id}")
        reconciliation_queue.publish({"user_id": user_id})
```

---

### Q11. Design a system for a stock exchange that needs to process orders with sub-millisecond latency while maintaining a complete audit trail.

**Latency requirements:** Order matching < 1ms end-to-end (network + processing + ACK). This rules out traditional databases for the hot path.

**Architecture: In-Memory Matching Engine + Async Persistence**

```
EXCHANGE ARCHITECTURE
──────────────────────────────────────────────────────────────
Order Submission (FIX protocol / REST for retail)
      │
      ▼ Hardware timestamping (FPGA NIC → nanosecond accuracy)
      │
      ▼ Order Gateway (C++ / Rust, kernel bypass networking)
        Validates order: symbol, quantity, price, account balance
        Publishes to Order Bus (LMAX Disruptor: lock-free ring buffer)
      │
      ▼ Matching Engine (single-threaded, in-memory)
        Per-symbol Order Book:
          Buy side: max-heap sorted by price (desc), then FIFO
          Sell side: min-heap sorted by price (asc), then FIFO
        Match algorithm: Price-Time Priority
        Throughput: 1M orders/sec per symbol on modern hardware
      │
      ├──► Trade Events → Disruptor (non-blocking)
      │                        │
      │                        ├─► Market Data Feed (real-time quote distribution)
      │                        ├─► Audit Log Writer (write-ahead log → disk)
      │                        └─► Risk System (post-trade checks)
      │
      ▼ ACK to submitter (< 1ms from order received to ACK)
```

**Order Book Data Structure:**
```python
from sortedcontainers import SortedList

class OrderBook:
    def __init__(self, symbol: str):
        self.symbol = symbol
        # Buy side: sorted descending by price (best bid first)
        self.bids = SortedList(key=lambda x: (-x.price, x.timestamp))
        # Sell side: sorted ascending by price (best ask first)
        self.asks = SortedList(key=lambda x: (x.price, x.timestamp))

    def add_order(self, order: Order):
        if order.side == "BUY":
            self.bids.add(order)
        else:
            self.asks.add(order)
        self._match()

    def _match(self):
        while self.bids and self.asks:
            best_bid = self.bids[0]
            best_ask = self.asks[0]
            if best_bid.price >= best_ask.price:
                trade_qty = min(best_bid.remaining_qty, best_ask.remaining_qty)
                trade_price = best_ask.price  # price-time priority: ask's price
                self._execute_trade(best_bid, best_ask, trade_qty, trade_price)
            else:
                break  # no match possible
```

**Audit Trail (complete, immutable):**
Every event is written to an immutable, append-only event log:
```
Order received → Order matched → Trade executed → Settlement initiated
Each event: { event_id, timestamp_ns, event_type, payload, sequence_number }

Technology: Kafka (ordered, replicated, immutable) + S3/HDFS (cold storage, 7-year retention for regulatory compliance)
Sequence numbers: monotonic per symbol → detect any gaps (missing events)
```

**Compliance Requirements:**
- MiFID II / SEC: must be able to reconstruct entire market state at any historical nanosecond
- Every order, cancel, modify, trade stored with nanosecond timestamp
- Audit query: "What was the order book for AAPL at 09:30:00.000123456?"

---

### Q12. How would you design a multi-player online game's backend to handle 1 million concurrent players with real-time state synchronization?

**Challenge:** Game state changes 30–60 times per second. 1M players × 30 updates/s = 30M updates/second. Full state broadcast is impossible at scale — you need zone-based visibility.

**Architecture: Zone-based game servers + Interest Management**

```
GAME WORLD PARTITIONING
──────────────────────────────────────────────────────────────
Game world: 2D/3D space partitioned into zones (e.g., 200x200 tile chunks)
  Each zone handled by one game server instance
  Players in same zone see each other's state
  Players in different zones: invisible to each other

Scalable because:
  1M players across 10,000 zones = 100 players/zone average
  Each zone server handles 100 players = trivial
  Add zones = add servers = linear scaling
```

**State Synchronization Protocol:**
```python
# Server: authoritative game state
# Client: runs prediction (show immediate response) + reconciliation

# Server broadcast per zone (30 Hz / 33ms tick rate):
{
  "tick": 12345,
  "entities": [
    {"id": "p1", "x": 102.3, "y": 45.1, "health": 80, "action": "running"},
    {"id": "p2", "x": 103.7, "y": 44.9, "health": 100, "action": "idle"},
    # Only entities in SAME zone + nearby zones (interest management)
  ]
}

# Client-side prediction (reduces perceived latency):
# 1. User presses "move right" → immediately move local player (prediction)
# 2. Send input to server: {"input": "move_right", "tick": 12345}
# 3. Server authoritative update arrives
# 4. If server says different position → interpolate/snap to correct position
```

**Interest Management (visibility culling):**
```
Without: 1M players × 30 updates/s = 30M msg/s (impossible)
With:    Each player only receives updates for entities within sight radius

sight_radius = 50 tiles
Zone size = 200 tiles → player sees ~4 zones
Players per zone: 100 → player receives ~400 entity updates/s
400 entities × 30Hz = 12,000 updates/sec per player (manageable)
```

**Zone Transition (player moves from zone A to zone B):**
```
Player crosses zone boundary:
  1. Player connects to zone B server
  2. Zone A server sends "goodbye" event (remove entity from zone A state)
  3. Zone B server adds entity, broadcasts to zone B players
  4. Handoff is seamless: < 100ms gap (player buffer masks the transition)
```

**Persistent State (character inventory, achievements):**
Separate from real-time game state. Written to PostgreSQL asynchronously after combat, item pickup, etc. Real-time = in-memory on game server. Persistent = async write to DB (fire-and-forget with acknowledgment).

---

### Q13. Design a system that detects when a user has been inactive for exactly 30 days and sends them a re-engagement notification.

**The challenge:** This sounds simple but is surprisingly hard at scale. With 100M users, you cannot run `SELECT * FROM users WHERE last_active < now() - 30 days` every second — it would never finish.

**Why naive approaches fail:**
```
Approach: Cron job at midnight, scan all users
  100M users, 1ms per user = 28 hours per scan (too slow)

Approach: Query with index on last_active
  SELECT * FROM users WHERE last_active = now() - 30 days (date)
  Issue: "exactly 30 days" = 86,400 second window to check per day
  Still scans millions of rows per day
```

**Correct Approach: Event-driven with time-bucketed scheduling**

```
When user last_active changes → schedule a "check at T+30 days" event

Architecture:
  User activity event → Kafka → Activity Processor
                                    │
                                    ▼
                              Upsert user_last_active:
                                users SET last_active = NOW() WHERE id = user_id
                              AND schedule delayed job:
                                job_queue.schedule_at(
                                    "check_re_engagement",
                                    payload={"user_id": user_id,
                                             "last_active": now},
                                    run_at=now + timedelta(days=30)
                                )

At T+30 days, job fires:
  Look up user.last_active
  If last_active >= (30 days ago - 1 hour):
    User was active within last 30 days after all (new activity since we scheduled)
    → cancel / reschedule
  Else:
    Send re-engagement notification
    Mark notification_sent_at = NOW()
```

**Delayed Job Implementation Options:**
```
Option A: Kafka with future timestamp (Kafka native scheduling, limited support)
Option B: Redis sorted set (score = execute_at timestamp)
  redis.zadd("delayed_jobs", {job_id: execute_at_unix_timestamp})
  Worker polls: ZRANGEBYSCORE delayed_jobs 0 now LIMIT 100 every second

Option C: Dedicated scheduler (Quartz, Celery Beat, custom service from Q5)
  Most robust for large-scale delayed jobs
```

**De-duplication:** If user is inactive for 30 days, we send 1 notification (not repeated every day after 30 days). After notification sent, don't check again until user re-engages and goes inactive for another 30 days.

---

### Q14. How would you design the comment ranking system for a social platform where some posts have 10 million comments?

**The problem:** 10M comments cannot be sorted in real-time on every page load. The top comments must be pre-computed, and threaded replies add hierarchical complexity.

**Ranking Signals:**
```
score = f(upvotes, downvotes, time_decay, reply_count, reporter_rate)

Reddit-style Wilson Score (accounts for sample size):
  score = (upvotes + 1.9208) / (total_votes + 3.8415) -
          1.96 * sqrt(upvotes * downvotes / total_votes^2 + 0.9604 / total_votes^2) /
          (1 + 3.8415 / total_votes)

Time decay (Hacker News style):
  score = votes / (age_hours + 2) ^ gravity   # gravity ≈ 1.8
```

**Pre-computed Rankings (essential at 10M scale):**
```python
# Background job: re-score and re-rank top comments periodically
def rerank_comments(post_id: str):
    # Only top 10,000 comments are worth ranking (99.9% of views)
    top_comments = db.query("""
        SELECT comment_id, upvotes, downvotes, created_at, reply_count
        FROM comments WHERE post_id = %s
        ORDER BY upvotes DESC LIMIT 10000
    """, post_id)

    scored = [(compute_score(c), c.comment_id) for c in top_comments]
    scored.sort(reverse=True)

    # Store ranked list in Redis sorted set
    with redis.pipeline() as pipe:
        pipe.delete(f"comments:ranked:{post_id}")
        pipe.zadd(f"comments:ranked:{post_id}",
                  {comment_id: score for score, comment_id in scored})
        pipe.execute()

# Real-time update: when comment gets upvoted, update score atomically
def on_upvote(comment_id: str, post_id: str):
    new_score = compute_score(db.get_comment(comment_id))
    redis.zadd(f"comments:ranked:{post_id}", {comment_id: new_score})
```

**Threaded Comments:**
```
Flat list approach (Reddit/Twitter): show top-level comments, collapse threads
  Adjacency list: parent_comment_id column
  Query top-level: WHERE parent_id IS NULL ORDER BY score DESC LIMIT 20
  Lazy-load replies: click to expand → fetch WHERE parent_id = X

Materialized path (for deep nesting):
  path = "0003.0012.0001"  (root → reply → reply-to-reply)
  ORDER BY path, score → natural nesting order in query
```

**Pagination at 10M scale:**
Never `OFFSET 5000000` — that scans 5M rows. Use cursor-based:
```sql
SELECT comment_id, score FROM comments
WHERE post_id = $post_id AND score < $cursor_score
ORDER BY score DESC LIMIT 20;
-- cursor_score from last comment of previous page
```

---

### Q15. Design a global DNS system (simplified). What data structure stores records, how do you handle cache invalidation, and how do you prevent DDoS?

**DNS Record Storage:**
```
DNS record types:
  A:    domain → IPv4 address
  AAAA: domain → IPv6 address
  CNAME: alias → canonical domain
  MX:   mail exchange server
  TXT:  arbitrary text (SPF, DKIM, domain verification)
  NS:   name server for a zone

Storage: Distributed hierarchical database
  Root zone (13 root server clusters, anycast):
    Stores: NS records for TLDs (.com, .net, .org)
  TLD nameservers:
    .com zone: NS records for all registered .com domains
  Authoritative nameservers:
    api.example.com A 1.2.3.4 TTL=300
```

**Cache (Recursive Resolver):**
```
ISP Recursive Resolver caches results for TTL seconds:

Query: "What is the IP of api.example.com?"
  1. Check local cache: miss
  2. Query root server: "Where is .com?"  → ns1.verisign.com
  3. Cache: .com NS = ns1.verisign.com TTL=172800 (2 days)
  4. Query .com nameserver: "Where is example.com?" → ns1.example.com
  5. Cache: example.com NS = ns1.example.com TTL=86400 (1 day)
  6. Query ns1.example.com: "What is api.example.com?" → 1.2.3.4
  7. Cache: api.example.com A = 1.2.3.4 TTL=300 (5 minutes)
  8. Return 1.2.3.4 to client

Future queries: served from cache until TTL expires
```

**Cache Invalidation:**
```
DNS has no push invalidation — TTL is the only mechanism.
To "invalidate" a record: lower TTL BEFORE making change.

Best practice for IP change:
  Day -2: Lower TTL from 86400 to 300 (wait 2 days for old TTL to expire)
  Day 0:  Change A record to new IP
  Day 0+:  Old IP serves traffic for 300 seconds max (5 minutes)
  Day +7: Restore TTL to 86400

Emergency propagation: minimum TTL is 0 but resolvers may cache for 30-60s anyway
```

**DDoS Prevention:**
```
1. Anycast (most important)
   13 root server clusters each announce the SAME IP via BGP anycast
   Packets go to nearest cluster (physically distributed)
   DDoS traffic from one region → absorbed by that region's cluster
   Other regions unaffected

2. Rate limiting at resolver
   Per-IP query rate limiting at recursive resolver
   Block obviously abusive clients

3. Response Rate Limiting (RRL)
   Authoritative server: limit responses per source IP per domain
   Prevents amplification attack:
     Small query (40 bytes) → large response (3000 bytes) = 75x amplification
     With RRL: rate limit responses to same source

4. DNSSEC (prevents spoofing, not DDoS)
   Zone signing with private key
   Resolvers verify signature against public key
   Prevents DNS cache poisoning attacks
```

---

### Q16. How would you design a system that aggregates metrics from 10,000 microservice instances in real-time with a less than 30 second delay?

**Scale:** 10,000 instances × 1,000 metrics each × 10-second intervals = 1M metric data points per second.

**Architecture: Push-based collection with hierarchical aggregation**

```
METRICS PIPELINE
──────────────────────────────────────────────────────────────
Each service instance (10,000)
  StatsD client: batches metrics every 10s
  Push to regional metrics aggregator (UDP, low overhead)
      │
      ▼
Regional Aggregator Tier (20 nodes, one per 500 service instances)
  Receives: 50,000 metrics/second per aggregator
  Aggregates: sum, count, min, max, percentiles per 10-second window
  Writes to: TimeSeries DB (Victoria Metrics / Prometheus remote write)
      │
      ▼
TimeSeries Database (Victoria Metrics cluster)
  Stores: aggregated metrics (post-aggregation, not raw events)
  Retention: 15-second resolution for 1 week, 1-minute for 1 month
  Query: PromQL for dashboards and alerts
      │
      ▼
Grafana (dashboards) + Alertmanager (alerts)
  End-to-end latency: metric generated → visible in dashboard
  = collection interval (10s) + aggregation window (10s) + write latency (5s)
  = ~25 seconds → within 30-second SLA
```

**Metric Schema:**
```
metric_name{label1="value1", label2="value2"} value timestamp

http_request_duration_ms{service="user-service",
                          instance="pod-42",
                          endpoint="/users",
                          status="200"} 45.2 1714000000
```

**Aggregation to avoid cardinality explosion:**
```
BAD: store per-instance per-endpoint per-status = 10K × 500 endpoints × 10 statuses
  = 50M time series → too high cardinality

GOOD: aggregate at collection tier:
  p50/p95/p99 per service per endpoint (drop instance dimension)
  = 100 services × 500 endpoints × 3 percentiles = 150K time series
  Acceptable cardinality

Keep instance dimension only for: CPU, memory, restart count, error rate
  (instance-level debugging metrics, much lower volume)
```

**Pull model alternative (Prometheus native):**
Each service exposes `/metrics` endpoint. Prometheus scrapes every 15 seconds. At 10K instances: scrape 10K endpoints every 15s = 667 scrapes/second. Manageable with Prometheus sharding (different Prometheus per 1K instances + Thanos for global aggregation).

---

### Q17. Design a system to handle 1 billion webhook deliveries per day with guaranteed delivery and configurable retry logic.

**Scale:** 1B/day = 11,600 deliveries/second. Each delivery involves: dequeue message, make HTTP POST to customer endpoint, handle success/failure, retry on failure.

**Architecture:**

```
WEBHOOK DELIVERY PIPELINE
──────────────────────────────────────────────────────────────
Event occurs (payment completed, user signed up, etc.)
      │
      ▼ Publish to Kafka topic: webhooks_pending
      │  Partitioned by: subscriber_id (guarantees ordering per subscriber)
      │
      ▼ Webhook Dispatcher Service (horizontally scaled, 100 workers)
      │  Reads from Kafka, manages delivery state
      │
      ▼ Delivery Attempt:
        HTTP POST to customer URL
        Timeout: 30 seconds
        Success: 2xx → mark delivered in DB, commit Kafka offset
        Failure: 4xx/5xx/timeout → schedule retry with backoff
```

**Retry Logic (configurable per subscriber):**
```python
RETRY_SCHEDULE = {
    "attempt_1": timedelta(minutes=5),
    "attempt_2": timedelta(minutes=30),
    "attempt_3": timedelta(hours=2),
    "attempt_4": timedelta(hours=8),
    "attempt_5": timedelta(hours=24)
}

def handle_delivery_failure(delivery: Delivery, attempt: int, error: str):
    if attempt >= delivery.max_attempts:  # subscriber-configured, default 5
        db.update_delivery_status(delivery.id, "FAILED",
                                   failure_reason=error)
        # Alert customer: their endpoint is down
        notify_subscriber(delivery.subscriber_id, "endpoint_down")
        return

    next_attempt_at = datetime.utcnow() + RETRY_SCHEDULE[f"attempt_{attempt+1}"]

    # Schedule for retry (use delayed job queue from Q5)
    delayed_queue.schedule_at("retry_webhook", {
        "delivery_id": delivery.id,
        "attempt": attempt + 1
    }, run_at=next_attempt_at)
```

**Guaranteed Delivery (at-least-once):**
```
Idempotency: every webhook carries a unique event_id
  {"event": "payment.completed", "event_id": "evt_abc123", "data": {...}}

Customer endpoint should be idempotent:
  if already_processed(event_id): return 200
  else: process_event(); mark_processed(event_id)

Delivery record:
  CREATE TABLE webhook_deliveries (
    delivery_id   UUID PRIMARY KEY,
    event_id      UUID NOT NULL,
    subscriber_id UUID NOT NULL,
    url           VARCHAR(2000) NOT NULL,
    payload       JSONB NOT NULL,
    status        VARCHAR(20),  -- PENDING, DELIVERED, FAILED
    attempt_count INT DEFAULT 0,
    delivered_at  TIMESTAMPTZ,
    created_at    TIMESTAMPTZ DEFAULT NOW()
  );
```

**Protecting slow/down endpoints:**
```python
# Circuit breaker per subscriber endpoint
# If endpoint fails 5 consecutive times, pause delivery for 1 hour
circuit_breaker = {
    "subscriber_123": CircuitBreaker(
        threshold=5,
        timeout=timedelta(hours=1)
    )
}

def deliver(delivery: Delivery):
    cb = circuit_breaker[delivery.subscriber_id]
    if cb.is_open():
        # Endpoint is down; queue for later without attempting
        schedule_retry(delivery, delay=cb.retry_after)
        return
    try:
        response = http_post(delivery.url, delivery.payload, timeout=30)
        cb.on_success()
    except Exception:
        cb.on_failure()
```

---

### Q18. How would you design the data pipeline for a recommendation system that needs to incorporate user events in near-real-time with less than 5 minute lag?

**The core challenge:** Training data needs historical features (batch, cheap, rich) but must also incorporate very recent events (streaming, expensive, limited).

**Hybrid Lambda Architecture:**

```
USER EVENTS (clicks, purchases, searches, ratings)
      │
      ├──► STREAMING PATH (< 5 minute lag)
      │    Kafka → Flink
      │    Computes: sliding window aggregates (clicks_last_30min, etc.)
      │    Writes to: Redis Online Store (real-time features)
      │    Feeds: Model inference at serving time
      │
      └──► BATCH PATH (daily jobs)
           Kafka → S3 (data lake, Parquet format)
           Spark: compute rich historical features (90-day history, embeddings)
           Writes to: Offline Feature Store (BigQuery / S3)
           Feeds: Model retraining
```

**Flink Streaming Job (real-time features):**
```python
# Flink: process each event and update rolling aggregates in Redis
def process_event(event: UserEvent):
    user_id = event.user_id
    now = event.timestamp

    # Real-time feature updates (atomic Redis operations)
    redis.zadd(f"events:{user_id}", {event.event_id: now})
    redis.zremrangebyscore(f"events:{user_id}", 0, now - 1800)  # keep 30min

    redis.incr(f"click_count:{user_id}:30m")
    redis.expire(f"click_count:{user_id}:30m", 1800)

    if event.type == "purchase":
        redis.incrbyfloat(f"spend:{user_id}:24h", event.amount)
        redis.expire(f"spend:{user_id}:24h", 86400)

    # Update user's recent item history (capped at 50 items)
    redis.lpush(f"history:{user_id}", event.item_id)
    redis.ltrim(f"history:{user_id}", 0, 49)
```

**Model Serving (uses both batch + real-time features):**
```python
def get_recommendations(user_id: str) -> list:
    # Real-time features from Redis (< 2ms)
    realtime_features = {
        "click_count_30m": redis.get(f"click_count:{user_id}:30m"),
        "spend_24h":        redis.get(f"spend:{user_id}:24h"),
        "recent_items":     redis.lrange(f"history:{user_id}", 0, 9)
    }

    # Batch features from offline store (refreshed daily, cheap lookup)
    batch_features = feature_store.get_online(user_id, [
        "avg_spend_30d", "preferred_category", "price_sensitivity_score"
    ])

    # Assemble and score
    features = {**realtime_features, **batch_features}
    candidates = retrieval_model.get_candidates(user_id, k=1000)
    return ranking_model.score_and_rank(candidates, features)[:20]
```

**Watermark handling for late events:**
Mobile apps batch events when offline. Define 10-minute watermark: accept events up to 10 minutes late. After watermark, late events are processed in next window.

---

### Q19. You're designing an API for a financial trading platform. Users submit orders that must be processed in strict FIFO order. Design the complete system.

**FIFO guarantee is the hardest constraint.** Most distributed systems sacrifice strict ordering for availability. For a trading platform, ordering IS correctness.

**Why distributed systems make FIFO hard:**
```
User A submits Order1 at T=0 (hits Server 1)
User A submits Order2 at T=1 (hits Server 2, due to load balancing)

If Server2 processes faster: Order2 processed before Order1!
Violation: Order1 (buy 100 shares at $150) + Order2 (sell 100 at market)
  In correct order: buy then sell (net: 0 shares)
  Wrong order: sell first (short sell, may be prohibited) then buy

This MUST not happen.
```

**Architecture: Partitioned, ordered queue with sequence numbers**

```
Order Submission API
      │ Step 1: Assign monotonic sequence number to order
      │         (per-account sequence, not global)
      │
      ▼
Sequence Server (single writer per account/symbol)
  Redis INCR: account_seq = redis.incr(f"seq:{account_id}")
  Order stamped with sequence_number
  Written to Kafka: partition_key = account_id
  (Kafka partitions preserve order within a partition key)
      │
      ▼
Kafka Topic: orders (partitioned by account_id)
  All orders from account_id=123 go to same partition
  Kafka guarantees in-partition ordering → FIFO per account
      │
      ▼
Order Processor (one consumer per partition)
  Reads orders IN ORDER (Kafka consumer API preserves order)
  Validates: sequence_number = last_processed_seq + 1?
    YES: process order
    NO: gap detected → hold and wait (out-of-order delivery recovery)
  Processing: update account balance + submit to exchange
      │
      ▼
Matching Engine (single-threaded per symbol, File 38-style)
  Receives pre-validated, in-order orders
  Executes: price-time priority matching
      │
      ▼
Trade Confirmation + Audit Log
```

**Sequence gap handling:**
```python
def process_order(order: Order, expected_seq: int):
    if order.sequence_number < expected_seq:
        # Duplicate (already processed) — idempotent skip
        logger.info(f"Duplicate order seq {order.sequence_number}, skipping")
        return

    if order.sequence_number > expected_seq:
        # Gap: previous order not yet received
        # Hold this order in pending buffer
        pending_buffer[order.sequence_number] = order
        # Wait for missing sequence(s)
        return

    # Process in-order
    execute_order(order)
    # Check if buffered orders are now in-sequence
    while expected_seq + 1 in pending_buffer:
        expected_seq += 1
        execute_order(pending_buffer.pop(expected_seq))
```

**Idempotency for retries:**
Each order has a client-generated `client_order_id` (UUID). If client retries (network timeout), server checks: `SELECT 1 FROM orders WHERE client_order_id = $id`. If exists → return existing result (don't re-submit to exchange).

---

### Q20. How would you design a system that allows 50 engineers to deploy to production 100 times per day safely, with automatic rollback on error spikes?

**This is a complete CI/CD + SRE system design question.** 100 deployments/day / 50 engineers = 2 deployments/engineer/day, which requires near-zero-friction deployment with strong safety nets.

**Architecture: Progressive Delivery Pipeline**

```
DEPLOY PIPELINE (fully automated)
──────────────────────────────────────────────────────────────
Engineer: git push to main branch
      │
      ▼ CI Pipeline (< 10 minutes)
      │  Unit tests (parallel, < 3 min)
      │  Integration tests (< 5 min)
      │  Security scan (Snyk, Trivy)
      │  Docker build + push to registry
      │  If any check fails: block deploy, notify Slack
      │
      ▼ CD Pipeline (automated, progressive)
      │
      ├─ Step 1: Deploy to STAGING (100% traffic)
      │   Run smoke tests (synthetic traffic)
      │   Pass? → advance automatically
      │
      ├─ Step 2: CANARY (5% production traffic, 10 minutes)
      │   Monitor: error rate, p99 latency, business metrics
      │   Auto-rollback if: error_rate > baseline + 2%
      │                     OR p99 latency > baseline × 1.3
      │   Pass? → advance automatically
      │
      ├─ Step 3: 25% production traffic (20 minutes)
      │   Same monitoring + auto-rollback
      │
      ├─ Step 4: 100% production traffic
      │   Monitor for 30 minutes post-deploy
      │   If regression detected: auto-rollback to previous version
      │
      └─ SUCCESS: Record deployment metadata in deployment database
```

**Auto-Rollback System:**
```python
class DeploymentWatchdog:
    BASELINE_WINDOW = timedelta(hours=1)     # compare to this window before deploy
    OBSERVATION_WINDOW = timedelta(minutes=10)
    ERROR_RATE_THRESHOLD = 0.02              # 2% absolute increase
    LATENCY_THRESHOLD = 1.3                  # 30% increase

    def watch(self, deployment_id: str, service: str):
        pre_deploy_metrics = self.get_metrics(service, self.BASELINE_WINDOW)

        # Watch for OBSERVATION_WINDOW
        start_time = datetime.utcnow()
        while datetime.utcnow() - start_time < self.OBSERVATION_WINDOW:
            post_deploy_metrics = self.get_metrics(service, timedelta(minutes=2))

            # Check error rate regression
            error_delta = (post_deploy_metrics.error_rate -
                           pre_deploy_metrics.error_rate)
            if error_delta > self.ERROR_RATE_THRESHOLD:
                self.auto_rollback(deployment_id, service,
                    reason=f"Error rate increased by {error_delta:.2%}")
                return

            # Check latency regression
            latency_ratio = (post_deploy_metrics.p99_latency /
                             pre_deploy_metrics.p99_latency)
            if latency_ratio > self.LATENCY_THRESHOLD:
                self.auto_rollback(deployment_id, service,
                    reason=f"p99 latency increased by {(latency_ratio-1):.0%}")
                return

            time.sleep(30)  # check every 30 seconds

    def auto_rollback(self, deployment_id: str, service: str, reason: str):
        # Get previous successful deployment
        previous = db.scalar("""
            SELECT image_tag FROM deployments
            WHERE service = %s AND status = 'SUCCESS'
            ORDER BY deployed_at DESC LIMIT 1 OFFSET 1
        """, service)

        # Roll back Kubernetes deployment
        k8s.set_image(service, previous)

        # Alert on-call
        pagerduty.alert(f"Auto-rollback: {service}. Reason: {reason}")
        slack.post("#deploys", f"🔴 Auto-rollback: `{service}` → `{previous}`\nReason: {reason}")

        # Record in deployment audit log
        db.insert("deployment_events", {
            "deployment_id": deployment_id,
            "event_type": "auto_rollback",
            "reason": reason,
            "rolled_back_to": previous
        })
```

**Feature Flags Integration:**
Every risky change goes behind a feature flag. The deploy only changes code — the flag release is a separate step. Rollback = flip flag (< 5 seconds), not re-deploy (5–10 minutes).

**Deployment Frequency Enablers:**
```
Required organizational practices:
  1. Trunk-based development (no long-lived branches)
  2. Feature flags for incomplete work (deploy != release)
  3. Small commits (easier to isolate regressions)
  4. Fast CI (< 10 min → engineers get fast feedback)
  5. Comprehensive automated tests (high confidence per commit)
  6. Observability (can detect regressions in minutes)
  7. Rollback is easy (< 5 min) and practiced regularly
  8. On-call rotation (someone owns each service)
  9. Deployment windows: allow deploys any time (not just "deploy windows")
     Because risky deploys are the ones done INFREQUENTLY in batches
```

**Capacity Planning:** 100 deploys/day = ~4/hour. CD pipeline parallelism: 4–8 deployments running simultaneously. Queue if more than 8 active deployments (unusual burst).

---

## Quick Reference

```
CONTENT MODERATION (Q1)
──────────────────────────────────────────────────────────────
Hash-based (CSAM, known bad) → ML classifier (90%) → Human review (0.5%)
Async: accept submission immediately, moderate in background
Visibility states: PENDING → VISIBLE or SHADOW or REMOVED

COLLABORATIVE SPREADSHEET (Q2)
──────────────────────────────────────────────────────────────
Cells vs text: Last-write-wins per cell (vs OT/CRDT for text position)
Formula DAG: dependency tracking, topological recalculation on edit
WebSocket: 30Hz tick for live cursors; REST for saves

CACHE MATH (Q3)
──────────────────────────────────────────────────────────────
L1 (in-process, 5s TTL): ~60% hit rate
L2 (Redis, 60s TTL): 75% of L1 misses
Database: 100K/sec after L1+L2 absorption
Hot key solution: N replica keys in Redis, round-robin reads

LIVESTREAMING (Q4)
──────────────────────────────────────────────────────────────
Ingest → Transcode (adaptive bitrate) → CDN edge (40 Tbps delivery)
LL-HLS: 2–5 second latency via partial segments + HTTP push
Chat: WebSocket + Redis Pub/Sub, sharded per channel

JOB SCHEDULER (Q5)
──────────────────────────────────────────────────────────────
SELECT FOR UPDATE SKIP LOCKED: concurrent dispatchers, no contention
Idempotency key: prevent double-execution on retry
Lease + heartbeat: detect crashed workers, reassign stuck jobs
Midnight spike: randomize ±30 seconds around scheduled time

AUTOCOMPLETE (Q6)
──────────────────────────────────────────────────────────────
Trie with pre-computed top-K per node
CDN-cached popular prefixes (~50% edge hit rate)
Trending terms: 5-minute hot list injected into static results

FRAUD DETECTION (Q7)
──────────────────────────────────────────────────────────────
Rules (5ms) → ML model (50ms) → Human review queue
Features: velocity (Redis), historical (offline store)
Model: XGBoost for tabular features; retrain weekly on chargebacks

RIDE MATCHING (Q8)
──────────────────────────────────────────────────────────────
Redis GEO for driver location index (O(log N) radius search)
Pre-computed road graph for ETA estimation (5–10ms)
Atomic assignment: Redis SETNX prevents double-assignment

WEBHOOK DELIVERY (Q17)
──────────────────────────────────────────────────────────────
Kafka for ordered, durable queue
Idempotency: event_id on every delivery
Retry: exponential backoff (5m, 30m, 2h, 8h, 24h)
Circuit breaker: pause delivery to failing endpoints

TRADING FIFO (Q19)
──────────────────────────────────────────────────────────────
Sequence number per account (Redis INCR)
Kafka partition by account_id → in-partition ordering = FIFO
Gap detection: buffer out-of-order, wait for missing sequence

CI/CD SAFE DEPLOYMENT (Q20)
──────────────────────────────────────────────────────────────
Staging → Canary 5% (10min) → 25% (20min) → 100%
Auto-rollback: error_rate delta > 2% OR p99 > 1.3x baseline
Feature flags: deploy != release; rollback = flag flip (< 5s)
Enablers: trunk-based dev, fast CI (< 10min), feature flags
```
