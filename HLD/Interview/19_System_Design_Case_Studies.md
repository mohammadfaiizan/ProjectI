# HLD Interview Q&A — File 19: System Design Case Studies

> 20 questions across Easy (Q1–7), Medium (Q8–15), and Hard (Q16–20).
> Each answer is 150–300+ words with diagrams, tables, or code where helpful.

---

## EASY (Q1–Q7)

---

### Q1. What are the 3 most important design decisions in a URL shortener?

**Answer:**

A URL shortener (like bit.ly, TinyURL) is a classic interview warm-up. The three most important decisions are:

**Decision 1: Short code generation strategy**

Option A — Random base62:
```python
import secrets, string

ALPHABET = string.ascii_letters + string.digits  # 62 chars
CODE_LENGTH = 7  # 62^7 = 3.5 trillion possibilities

def generate_code():
    return ''.join(secrets.choice(ALPHABET) for _ in range(CODE_LENGTH))
```

Option B — Counter + base62 encoding (bijective, no collisions):
```python
def encode_base62(num):
    chars = []
    while num:
        chars.append(ALPHABET[num % 62])
        num //= 62
    return ''.join(reversed(chars)).zfill(7)

# Global counter: 1 → "0000001", 2 → "0000002" (sequential but predictable)
```

**Tradeoff:** Random codes are unpredictable (harder to enumerate) but need collision checking. Counter-based is sequential (enumerable = privacy risk) but collision-free.

**Decision 2: Storage and data model**

```sql
CREATE TABLE short_urls (
    code        VARCHAR(10) PRIMARY KEY,   -- "abc1234"
    long_url    TEXT NOT NULL,             -- original URL
    user_id     BIGINT,                    -- NULL for anonymous
    created_at  TIMESTAMP DEFAULT NOW(),
    expires_at  TIMESTAMP,                 -- NULL = never expires
    click_count BIGINT DEFAULT 0
);

-- Index for reverse lookup (optional, for dedup)
CREATE INDEX idx_long_url ON short_urls(MD5(long_url));
```

**Decision 3: Read path optimization (caching)**

Redirects are read-heavy (1000:1 read/write ratio). Every access needs to look up the short code.

```
Read flow:
  Browser → GET /abc1234
  Server → Redis CACHE lookup
    Hit  → 302 redirect (< 1ms)
    Miss → DB query → populate cache → 302 redirect (5-10ms)
```

```python
def redirect(code):
    # L1: Redis cache (TTL = 24h for hot codes)
    cached_url = redis.get(f"url:{code}")
    if cached_url:
        increment_click_async(code)
        return redirect_to(cached_url)
    
    # L2: Database
    url_record = db.query("SELECT long_url FROM short_urls WHERE code = %s", code)
    if not url_record:
        return 404
    
    redis.setex(f"url:{code}", 86400, url_record.long_url)
    increment_click_async(code)
    return redirect_to(url_record.long_url)
```

---

### Q2. How does Twitter handle the celebrity/whale problem in feed generation?

**Answer:**

The "celebrity problem" (also called the "hotspot" or "whale" problem) occurs when a single user has tens of millions of followers and posts a tweet. Naive fan-out-on-write would trigger millions of writes immediately.

**The magnitude:**
```
Katy Perry: 100M followers
She tweets once → fan-out writes = 100M Redis LPUSH operations
At 100K writes/sec → takes 1000 seconds (16 minutes!) to fan-out
Users at the end of the queue see the tweet 16 minutes late
```

**Twitter's hybrid solution:**

**Step 1: Classify users by follower count**
```python
CELEBRITY_THRESHOLD = 10_000  # Adjust based on system capacity

def classify_user(user_id):
    follower_count = db.get_follower_count(user_id)
    return "celebrity" if follower_count > CELEBRITY_THRESHOLD else "regular"
```

**Step 2: Write path**
```
Regular user tweets:
  → Fan-out to all follower home timelines in Redis
  → Async, completes in seconds

Celebrity tweets:
  → Store tweet ONLY in celebrity's own tweet list
  → NO fan-out to follower timelines
```

**Step 3: Read path (timeline construction)**
```python
def get_home_timeline(user_id, count=20):
    # 1. Get pre-computed timeline (filled by fan-out of regular users)
    timeline_ids = redis.lrange(f"timeline:{user_id}", 0, count + 50)
    
    # 2. Get celebrities this user follows
    celebrities = db.get_followed_celebrities(user_id)
    
    # 3. Merge celebrity tweets at read time
    celebrity_tweets = []
    for celeb_id in celebrities:
        tweets = redis.lrange(f"user_tweets:{celeb_id}", 0, 20)
        celebrity_tweets.extend(tweets)
    
    # 4. Merge + sort by timestamp + deduplicate
    all_ids = list(set(timeline_ids + celebrity_tweets))
    tweets = batch_fetch_tweets(all_ids)
    return sorted(tweets, key=lambda t: t.timestamp, reverse=True)[:count]
```

**Result:** Celebrity tweets are merged at read time (adds ~5-10ms), but avoids the 16-minute fan-out. Normal user tweets are still pre-computed for fast reads.

---

### Q3. Why does Netflix use its own CDN (Open Connect) instead of a third-party CDN?

**Answer:**

Netflix built **Open Connect**, a custom CDN with hardware appliances placed inside ISP data centers. At Netflix's scale, this is dramatically cheaper and better than any commercial CDN.

**The economics:**
```
Netflix bandwidth (2023): ~700 Gbps peak
Commercial CDN cost: ~$0.05/GB average
Monthly bandwidth: 700Gbps × 30 days × 86400s × 0.000000001 GB/bit
                 = ~226 PB/month
CDN cost: 226,000,000 GB × $0.05 = $11.3M/month = $135M/year

Open Connect cost (amortized):
  Hardware: ~$50M/year (appliances + install)
  Operations: ~$30M/year
  Total: ~$80M/year → saves ~$55M/year
```

**Beyond cost — quality:**

```
Without Open Connect:
  Netflix → Akamai PoP → Akamai backbone → ISP peering → ISP network → User
  Multiple hops, shared infrastructure, less control

With Open Connect:
  Netflix (AWS) → Open Connect Appliance (inside ISP) → User
  One hop, dedicated hardware, Netflix controls quality
```

**Proactive caching:**
```python
# Netflix knows what will be popular (recommendations algorithm)
# They pre-push popular content to OCAs during off-peak hours

class OpenConnectManager:
    def nightly_prefetch(self):
        popular_titles = get_top_1000_titles_for_tomorrow()
        for title in popular_titles:
            for oca_region in ALL_OCA_REGIONS:
                if not oca_region.has_title(title):
                    schedule_transfer(title, oca_region)
```

**ISP partnership:** ISPs install OCAs for free because it saves THEM money — Netflix traffic doesn't have to travel over expensive upstream transit links. It's a mutual benefit arrangement.

**Key insight:** Third-party CDNs make money by marking up bandwidth. At Netflix's 1M+ concurrent streams, eliminating the middleman saves hundreds of millions per year.

---

### Q4. How does Uber match riders with drivers at scale?

**Answer:**

Uber's matching system is a real-time geospatial optimization problem. It must match millions of riders to drivers within seconds, globally.

**Core architecture:**
```
Rider requests ride (lat, lng)
         ↓
Supply Service: finds available drivers within radius
         ↓
ETA Service: calculates pickup time for each candidate driver
         ↓
Matching Service: scores and ranks drivers
         ↓
Dispatch Service: sends offer to top driver (15s timeout)
         ↓
Driver accepts → trip assigned
Driver declines / timeout → offer to next driver
```

**Geospatial indexing:**
```python
# Uber uses geohash and H3 (Hexagonal Hierarchical Spatial Index)
import h3

def find_nearby_drivers(rider_lat, rider_lng, radius_km=2):
    # Get H3 cell for rider location (resolution 9 ≈ 174m hexagons)
    rider_cell = h3.geo_to_h3(rider_lat, rider_lng, resolution=9)
    
    # Get all cells within radius (ring)
    search_cells = h3.k_ring(rider_cell, k=3)  # ~2km radius
    
    # Query Redis for drivers in these cells
    drivers = []
    for cell in search_cells:
        cell_drivers = redis.smembers(f"drivers:h3:{cell}")
        drivers.extend(cell_drivers)
    
    return drivers
```

**Driver location updates:**
```python
# Driver app sends GPS every 4 seconds
def update_driver_location(driver_id, lat, lng):
    new_cell = h3.geo_to_h3(lat, lng, resolution=9)
    old_cell = redis.get(f"driver:{driver_id}:cell")
    
    if new_cell != old_cell:
        # Move between H3 cells
        if old_cell:
            redis.srem(f"drivers:h3:{old_cell}", driver_id)
        redis.sadd(f"drivers:h3:{new_cell}", driver_id)
        redis.set(f"driver:{driver_id}:cell", new_cell)
    
    # Update precise GPS
    redis.geoadd("driver_positions", lng, lat, driver_id)
```

**Surge pricing integration:**
Uber calculates demand/supply ratio per H3 cell to determine surge multiplier. Cells with more rider requests than available drivers get surge pricing to incentivize driver movement.

**Scale:** Uber handles ~3M trips/day with a P99 match latency of ~30 seconds (including driver acceptance).

---

### Q5. What database does WhatsApp use for messages and why?

**Answer:**

WhatsApp uses **Mnesia** (an Erlang distributed database, built into the BEAM VM) for real-time message routing and presence, and **FreeBSD + custom storage** for message persistence.

**More importantly — why Erlang/Mnesia:**

WhatsApp's founding engineering philosophy: use Erlang because:
- The BEAM VM was designed for telecom systems (high concurrency, fault tolerance)
- Lightweight processes (millions of concurrent connections per node)
- Built-in distribution (Mnesia replicates across nodes natively)

```
Traditional thread model:
  1 connection = 1 OS thread = 1-2MB RAM
  1M connections = 1-2 TB RAM (infeasible)

Erlang process model:
  1 connection = 1 Erlang process = 2-4 KB RAM
  1M connections = 2-4 GB RAM (feasible)
```

**Message flow:**
```
Sender → WhatsApp Server
  ├── Recipient online: route via Mnesia process registry → WebSocket delivery
  └── Recipient offline: persist to message store
                              ↓
                        Recipient connects
                              ↓
                        Pull messages from store → deliver → ACK
                              ↓
                        Delete from server (WhatsApp deletes after delivery)
```

**Post-Facebook acquisition:** WhatsApp's message store shifted toward more scalable solutions. Facebook (Meta) uses their own distributed storage stack (Haystack for media, MyRocks for metadata).

**Key insight:** WhatsApp consistently serves ~100B messages/day. At the time of Facebook acquisition, they had ~50 engineers serving 450M users — efficiency from Erlang's concurrency model.

**2024 architecture:** WhatsApp now uses a combination of Erlang (real-time messaging/presence), RocksDB (on-device storage), and custom distributed storage at Meta's data centers.

---

### Q6. How does Dropbox achieve sync with minimal bandwidth using delta sync and chunking?

**Answer:**

Dropbox's core engineering challenge: sync files across devices efficiently, uploading only what changed.

**Block-based chunking:**
```
File: large_document.docx (50 MB)
Split into fixed-size blocks (4MB each):

Block 1: [bytes 0 - 4MB]     hash: sha256("a1b2c3...")
Block 2: [bytes 4MB - 8MB]   hash: sha256("d4e5f6...")
Block 3: [bytes 8MB - 12MB]  hash: sha256("g7h8i9...")
...
Block 13: [bytes 48MB - 50MB] hash: sha256("z1y2x3...")
```

**Delta sync (only upload changed blocks):**
```python
def sync_file(local_path, dropbox_client):
    local_blocks = compute_blocks(local_path)
    remote_metadata = dropbox_client.get_file_metadata(local_path)
    
    if not remote_metadata:
        # New file: upload all blocks
        upload_all_blocks(local_blocks)
        return
    
    remote_block_hashes = set(remote_metadata.block_hashes)
    
    # Only upload blocks that changed
    blocks_to_upload = [
        block for block in local_blocks
        if block.hash not in remote_block_hashes
    ]
    
    if blocks_to_upload:
        upload_blocks(blocks_to_upload)
        # Tell server new block layout
        dropbox_client.update_file_manifest(local_path, 
                                             [b.hash for b in local_blocks])
    
    print(f"Uploaded {len(blocks_to_upload)}/{len(local_blocks)} blocks")
    # A 1-paragraph edit in a 50MB file: upload 1/13 blocks = 4MB instead of 50MB
```

**Deduplication across users:**
```python
# If two users have the same file (by block hash), Dropbox stores it once
def upload_block(block_hash, block_data):
    if block_store.exists(block_hash):
        return  # Block already stored (another user has same content)
    block_store.put(block_hash, block_data)

# This is why Dropbox can sync instantly when you copy a file
# that's already in another user's Dropbox — no upload needed
```

**Bandwidth savings:** A typical 1-sentence edit in a 100-page Word document uploads only the changed 4MB block, not the full 10MB document — 60-80% bandwidth savings for incremental edits.

---

### Q7. How does Google Docs handle concurrent edits from multiple users?

**Answer:**

Google Docs uses **Operational Transformation (OT)** to allow multiple users to edit simultaneously and converge to the same document state.

**The core problem:**
```
Initial state: "Hello world"
Position:       0123456789

User A inserts " beautiful" at position 5: "Hello beautiful world"
User B deletes "world" (positions 6-10): "Hello "

If both apply without coordination:
  A's result: "Hello beautiful world"
  B's result: "Hello "
  They see different documents — divergence!
```

**OT solution:**
```python
# Operations have type, position, and content
Op_A = Insert(pos=5, text=" beautiful")
Op_B = Delete(pos=6, length=5)  # delete "world"

# Transform B against A: A inserted 10 chars at pos 5
# B's original delete starts at pos 6 (was "world")
# After A's insert, "world" is now at pos 6 + 10 = 16
Op_B_prime = Delete(pos=16, length=5)

# Transform A against B: B deleted 5 chars at pos 6
# A's insert is at pos 5 (before B's delete) → no change
Op_A_prime = Insert(pos=5, text=" beautiful")

# Both users apply:
# A's state: "Hello beautiful world" + Op_B_prime = "Hello beautiful "
# B's state: "Hello " + Op_A_prime = "Hello beautiful "
# Both converge!
```

**Google Docs server model:**
```
All operations flow through a central server
Server assigns global sequence numbers (revision IDs)
Server transforms client operations against concurrent operations
Broadcasts transformed operations to all clients

Client state:
  current_revision: 42
  pending_operations: [Op_A]  (sent to server, awaiting ACK)
  buffered_operations: []     (typed after pending, held until ACK)

On server ACK (revision 43):
  Apply buffered_operations to local state
  Transform buffered against server response if needed
```

**Why a central server for OT:** True peer-to-peer OT is exponentially complex for multi-user scenarios. The server acts as a "tie-breaker" — operations arrive in a defined order, so only O(n) transformations are needed. CRDTs remove this need but have different trade-offs.

---

## MEDIUM (Q8–Q15)

---

### Q8. What is the key design insight that makes a distributed key-value store like DynamoDB scale?

**Answer:**

The fundamental insight behind DynamoDB's design (documented in the 2007 Amazon Dynamo paper) is: **avoid coordination for the common case**.

Traditional databases coordinate all writes through a single node (leader). This creates a bottleneck. Dynamo's insight: for many use cases, you can trade perfect consistency for linear scalability.

**Key design decisions (Dynamo paper):**

**1. Consistent Hashing — avoid full rebalancing:**
```
Standard sharding: add a node → rehash all keys → massive data movement
Consistent hashing: add a node → only move keys in its range

Ring with virtual nodes:
  Node A owns: keys hashing to [0, 25%]
  Node B owns: keys hashing to [25%, 50%]
  Node C owns: keys hashing to [50%, 75%]
  Node D owns: keys hashing to [75%, 100%]

Add Node E:
  E takes [25%, 37.5%] from B → only 12.5% of keyspace moves
```

**2. Quorum reads/writes — tunable consistency:**
```
N = 3 replicas (in 3 different AZs)
W = 2 (wait for 2 writes to succeed before ACK)
R = 2 (read from 2 replicas, take latest)

If W + R > N → strong consistency (2+2=4 > 3 ✓)
If W=1, R=1 → highest availability, eventual consistency
```

**3. Sloppy quorum + hinted handoff — partition tolerance:**
```
If the designated replica is down:
  Write to a different available node with a "hint" that it belongs to the down node
  When down node recovers → transfer the hinted data back
  
This is why DynamoDB is always available for writes — even if N-1 replicas are down
```

**4. Vector clocks — conflict detection (classic Dynamo, DynamoDB simplified this):**
```
Multiple concurrent writes → create siblings (conflicting versions)
Return all versions to the client
Application logic resolves the conflict (merge or choose winner)
DynamoDB today uses last-writer-wins with server-side timestamps instead
```

**The key insight in one sentence:** By giving up consistency as a global invariant and making it a per-request tunable parameter, DynamoDB achieves essentially unlimited write throughput through horizontal sharding with minimal coordination.

---

### Q9. How does Kafka achieve high throughput (millions of messages per second)?

**Answer:**

Kafka achieves extraordinary throughput through a combination of OS-level, hardware-level, and protocol-level optimizations.

**1. Sequential disk writes (no random I/O):**
```
Traditional queue: Random I/O → 100-200 IOPS (spinning disk)
Kafka: Sequential append-only log → 500 MB/s+ sustained write throughput

Messages are appended to end of log file:
[msg1][msg2][msg3][msg4] → [msg1][msg2][msg3][msg4][msg5]

OS page cache handles buffering; Kafka flushes periodically
OS is very efficient at sequential reads/writes
```

**2. Zero-copy transfer (sendfile syscall):**
```
Traditional copy:
  Disk → Kernel buffer → User space → Kernel socket buffer → Network
  4 copies, 2 syscalls

Kafka zero-copy:
  Disk → Kernel buffer → Network (via sendfile)
  2 copies, 1 syscall
  
kafka.fileRecords.writeTo(channel, position, length)
// This ultimately calls Java's FileChannel.transferTo() → OS sendfile()
```

**3. Batching:**
```python
# Producer batches messages before sending
producer = KafkaProducer(
    batch_size=65536,      # 64KB batch
    linger_ms=5,           # Wait up to 5ms to fill batch
    compression_type='lz4' # Compress batch (typically 4-8x ratio)
)

# Broker batches messages per partition before writing
# Consumer reads entire batch in one fetch (amortizes network round-trips)
```

**4. Partitioning for parallelism:**
```
Topic: orders (100 partitions)

Producer → sends to partition based on key hash
          orders for customer_id=123 always go to partition 37

Consumer group (100 consumers, one per partition):
  Consumer 0 → reads partition 0
  Consumer 1 → reads partition 1
  ...
  Consumer 99 → reads partition 99

100 consumers reading in parallel = 100x throughput vs single consumer
```

**5. Log compaction (for compacted topics):**
```
Instead of storing every version of a key, keep only the latest:
[user:123=alice][user:123=alice-smith][user:456=bob]
→ compacts to →
[user:123=alice-smith][user:456=bob]

Reduces log size, speeds up consumer catchup after restart
```

---

### Q10. What makes a stock exchange different from other systems in terms of design requirements?

**Answer:**

A stock exchange is perhaps the most demanding system to design due to extreme requirements across every dimension.

**Unique requirements:**

**1. Strict ordering — total order of events:**
```
Traditional distributed systems: eventual consistency acceptable
Stock exchange: EVERY transaction must be processed in strict order

If Buy(AAPL, 100) and Sell(AAPL, 100) arrive simultaneously:
  Order matters: who gets priority? (price-time priority)
  Must be deterministic: same order on all nodes
  
Solution: Single-threaded matching engine (no parallelism!)
  → 1-5 microsecond order processing
  → LMAX Disruptor ring buffer pattern
```

**2. Ultra-low latency requirements:**
```
Acceptable latency:
  Web application: 50-200ms
  Gaming server: 5-50ms
  Stock exchange matching engine: 10-100 microseconds

Techniques:
  - Co-location: traders pay to place servers in same data center as exchange
  - Kernel bypass: DPDK, SR-IOV (bypass OS network stack entirely)
  - FPGA: offload order matching to hardware
  - No garbage collection: C++ or Java with off-heap memory (chronicle.map)
```

**3. Deterministic replay — audit trail:**
```
Every exchange must be able to replay from day 0 to audit any transaction
Complete event log stored immutably:
  [09:30:00.000001] OrderReceived: OrderID=123, BUY AAPL 100 @ $150
  [09:30:00.000002] OrderMatched: BuyID=123, SellID=456, 100 shares @ $150
  [09:30:00.000003] TradeConfirmed: BuyID=123 buyer=Fund_A ...

This is why Kafka's append-only log is perfect for exchange architectures
```

**4. Regulatory requirements:**
```
- MiFID II (EU): Timestamp accuracy to 1 microsecond
- Reg NMS (US): Best execution across all exchanges
- Complete audit trail: 7 years retention
- Circuit breakers: halt trading if price moves > X% in Y minutes
```

**5. Fairness:**
```
Co-location fees and network topology must give equal access
Some exchanges deliberately randomize order receipt within 1ms windows
to prevent latency arbitrage
```

**LMAX Disruptor — the key pattern:**
```java
// Single-producer, multi-consumer lock-free ring buffer
// Avoids garbage collection, lock contention
// 6M+ transactions/second on commodity hardware
RingBuffer<OrderEvent> ringBuffer = RingBuffer.createSingleProducer(
    OrderEvent.FACTORY, 1024,  // Power of 2 size
    new YieldingWaitStrategy()  // Busy-spin for lowest latency
);
```

---

### Q11. How does a search engine like Google rank pages (simplified)?

**Answer:**

Google's ranking involves hundreds of signals, but the foundational algorithm is **PageRank** combined with content relevance and user signals.

**PageRank — link graph analysis:**
```
Idea: A page is important if many important pages link to it
(Recursive definition, solved iteratively)

PageRank formula:
  PR(A) = (1-d) + d × Σ(PR(B) / OutLinks(B))
  
  Where:
    d = damping factor (0.85 — probability user follows a link vs goes to random page)
    B = all pages that link to A
    OutLinks(B) = number of outbound links from B

Example:
  Page A has links from Wikipedia (PR=95), a blog (PR=30)
  PR(A) ≈ 0.15 + 0.85 × (95/5 + 30/10) ≈ 0.15 + 0.85 × 22 ≈ 18.8
```

**Full ranking pipeline:**
```
Query: "best coffee shops in Seattle"

Step 1: Query Processing
  Tokenize + normalize: ["best", "coffee", "shop", "seattle"]
  Identify intent: local search → use location-aware index
  Expand: ["café", "espresso", "coffee house", ...]

Step 2: Document Retrieval (inverted index lookup)
  For each token → get list of (doc_id, term_frequency, position) tuples
  Intersect lists → candidate documents containing all/most terms

Step 3: Scoring (many signals)
  TF-IDF: term frequency × inverse document frequency
  PageRank: authority of the page
  Freshness: when was page last updated?
  Click-through rate: do users click this result and stay?
  Core Web Vitals: page load speed
  Mobile-friendly: responsive design?
  HTTPS: secure connection?
  User location: near Seattle?

Step 4: BERT/MUM (neural ranking)
  Deep learning models understand query intent
  "best coffee" → intent = quality, not cheapest
  Rerank top-1000 candidates using transformer model

Step 5: Diversity + Personalization
  Avoid returning 10 results from same domain
  Personalize based on user's search history (if signed in)
  Local results boost (your distance to coffee shop)
```

**Simplified index structure:**
```
Inverted index:
  "coffee" → [(doc:42, tf:5, pos:[12,45,89]), (doc:157, tf:3, pos:[7,23,91]), ...]
  "seattle" → [(doc:42, tf:2, pos:[3,67]), (doc:891, tf:8, pos:[1,4,9,15,...]), ...]

Forward index (for document info):
  doc:42 → {url: "visitseattle.com/coffee", pr: 42.3, last_modified: 2024-01-15}
```

---

### Q12. What database choices would you make for an e-commerce platform like Amazon and why?

**Answer:**

A large e-commerce platform needs multiple specialized databases — no single database handles all workloads optimally.

**Workload analysis:**
| Data Type | Access Pattern | Volume | Consistency |
|---|---|---|---|
| Product catalog | Read-heavy, search | 100M products | Eventual OK |
| User accounts | Read/write balanced | 300M users | Strong |
| Orders | Write-heavy, ACID | 10M orders/day | Strict |
| Inventory | Write-heavy, concurrent | 100M SKUs | Strong |
| Shopping cart | Session-scoped, ephemeral | 50M active sessions | Eventual |
| Reviews/ratings | Write once, read-heavy | 1B reviews | Eventual |
| Search | Complex queries, full-text | All products | Near-real-time |

**Database selection:**
```
Product Catalog: DynamoDB
  - Key: product_id → product details
  - Read-heavy, horizontal scaling, flexible schema
  - Cache with ElastiCache (DAX): product pages are read millions of times

User Accounts: Aurora PostgreSQL (MySQL-compatible)
  - ACID for account creation, login, security events
  - Multi-AZ for high availability
  - Read replicas for profile reads

Orders: Aurora PostgreSQL or CockroachDB
  - ACID transactions (prevent double-charging)
  - ORDER by customer_id, ORDER by status
  - Sharding by customer_id at scale

Inventory: DynamoDB with conditional writes
  - Atomic decrement: UPDATE inventory SET quantity = quantity - 1
                      WHERE sku = 'X' AND quantity > 0
  - Conditional write prevents oversell

Shopping Cart: Redis
  - Ephemeral (TTL = 30 days)
  - HSET cart:{session_id} {sku: quantity, sku2: quantity2}
  - Fast read/write, acceptable loss on Redis failure (reconstruct from DB)

Product Search: Elasticsearch / OpenSearch
  - Full-text search, faceted filtering (by brand, price, rating)
  - Near-real-time index updates (sync from DynamoDB via Lambda/streams)

Reviews/Analytics: Cassandra or ClickHouse
  - Write-heavy append-only workload (reviews never deleted)
  - Time-series analytics (sales trends, review frequency)
```

**Data flow:**
```
New order → Aurora (ACID commit)
         → DynamoDB (inventory decrement, conditional)
         → Kafka (order event)
         → Inventory service (update availability)
         → Notification service (send confirmation)
         → Analytics pipeline (ClickHouse)
```

---

### Q13. How does Airbnb prevent double-booking of the same property?

**Answer:**

Double-booking is catastrophic for a marketplace platform — a guest shows up to find another guest already there. Airbnb must ensure each night of a property is reserved by at most one booking.

**The race condition:**
```
Guest A views listing: July 1-7 available
Guest B views listing: July 1-7 available (same time)

Guest A begins booking July 1-7
Guest B begins booking July 1-7

Without protection:
  A checks: available? YES → reserve July 1-7 → CONFIRMED
  B checks: available? YES → reserve July 1-7 → CONFIRMED
  Result: Double booking!
```

**Solution 1: Database-level constraints**

```sql
-- Reservations table
CREATE TABLE reservations (
    id BIGINT PRIMARY KEY,
    listing_id BIGINT NOT NULL,
    check_in DATE NOT NULL,
    check_out DATE NOT NULL,
    status VARCHAR(20)  -- 'PENDING', 'CONFIRMED', 'CANCELLED'
);

-- Exclusion constraint (PostgreSQL): no overlapping dates for same listing
CREATE EXTENSION btree_gist;
ALTER TABLE reservations ADD CONSTRAINT no_double_booking
    EXCLUDE USING gist (
        listing_id WITH =,
        daterange(check_in, check_out, '[)') WITH &&
    )
    WHERE (status != 'CANCELLED');
```

This is the most reliable approach — the database enforces the invariant at the storage layer. The `&&` operator means "overlaps". The constraint fires even under high concurrency.

**Solution 2: Optimistic locking with version vectors**

```python
def attempt_booking(listing_id, check_in, check_out, guest_id):
    # Read availability
    availability = db.get_availability(listing_id)
    version = availability.version
    
    if not is_available(availability, check_in, check_out):
        return {"error": "Already booked"}
    
    # Try to reserve (with version check)
    rows_updated = db.execute("""
        UPDATE availability
        SET booked_nights = array_append(booked_nights, %s),
            version = version + 1
        WHERE listing_id = %s
          AND version = %s  -- Optimistic lock
          AND NOT dates_overlap(booked_nights, %s, %s)
    """, (check_in, listing_id, version, check_in, check_out))
    
    if rows_updated == 0:
        # Conflict! Someone else booked simultaneously
        return attempt_booking(listing_id, check_in, check_out, guest_id)  # Retry
```

**Solution 3: Distributed lock during booking window**

```python
def book_with_lock(listing_id, check_in, check_out, guest_id):
    lock_key = f"booking_lock:{listing_id}:{check_in}:{check_out}"
    
    with redis.lock(lock_key, timeout=30):  # 30 second timeout
        # Double-check after acquiring lock
        if is_already_booked(listing_id, check_in, check_out):
            return {"error": "Already booked"}
        
        # Safe to book now
        create_reservation(listing_id, check_in, check_out, guest_id)
        return {"status": "CONFIRMED"}
```

**Best approach:** Database exclusion constraints (Solution 1) as the authoritative guard, with optimistic locking for better user experience (show conflict before hitting DB constraint).

---

### Q14. What makes designing a rate limiter at global scale (Stripe/Cloudflare) challenging?

**Answer:**

A rate limiter at global scale must handle millions of API clients with consistent enforcement across a globally distributed infrastructure.

**Core algorithm choices:**

**Fixed window:**
```
Window: 1 minute, limit: 100 requests
Counter resets at :00 of each minute

Problem: 100 requests at :59, 100 requests at :01 = 200 in 2 seconds!
```

**Sliding window (most accurate):**
```
Use Redis sorted set with timestamps as scores:
ZADD rate:user123 {timestamp: requestId}  # Log each request
ZREMRANGEBYSCORE rate:user123 0 (now - 60s)  # Remove old requests
count = ZCARD rate:user123  # Count remaining
```

**Token bucket (allows bursting):**
```python
class TokenBucketRateLimiter:
    def __init__(self, rate, capacity):
        self.rate = rate        # tokens per second
        self.capacity = capacity  # max burst size
    
    def check(self, user_id):
        lua_script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or now
        
        -- Refill tokens based on elapsed time
        local elapsed = now - last_refill
        tokens = math.min(capacity, tokens + elapsed * rate)
        
        if tokens >= 1 then
            tokens = tokens - 1
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', key, 3600)
            return 1  -- Allow
        else
            return 0  -- Deny
        end
        """
        result = redis.eval(lua_script, 1, f"rate:{user_id}", 
                           self.rate, self.capacity, time.time())
        return result == 1
```

**Global scale challenges:**

**1. Distributed counter synchronization:**
```
Problem: 5 Redis shards, user makes 100 requests
  Shard 1 sees: 25 requests (at limit 20 → blocks!)
  Shard 2 sees: 18 requests (under limit → allows)
  Correct total: 43 — should be under limit of 100!

Solutions:
  A) Route all requests for same user to same Redis shard (hash routing)
  B) Gossip protocol: shards sync counts periodically (approximate)
  C) Local + global counters: local allows burst, global enforces ceiling
```

**2. Clock skew:**
```
Different servers have different clocks
Sliding window boundaries are inconsistent across nodes
Solution: Redis time (TIME command) as authoritative clock
```

**3. Latency — every request adds Redis round-trip:**
```
Stripe's solution: middleware in every region with local Redis
  + Cross-region sync for global rate limits
  
Local limit: 80% of global limit (absorbs local bursts)
Global enforcement: asynchronous sync checks global counter
```

---

### Q15. How would you design the notification system for a platform with 1 billion users?

**Answer:**

A notification system at 1B user scale must handle diverse notification types, multiple delivery channels, and massive fan-out while being reliable and respecting user preferences.

**Scale estimate:**
```
1B users
Average 10 notifications/user/day = 10B notifications/day
Peak: 10x average = 1.16M notifications/second during peak
Channels: push (mobile), email, SMS, in-app, web push
```

**Architecture:**

```
Event Bus (Kafka)
     ↓ (events: new_message, order_shipped, friend_request, etc.)
Notification Service
     ↓
Preference Engine ──→ User Preferences DB
     ↓               (channels, frequency, quiet hours, DND)
Routing Engine
     ↓
┌────┬─────┬────┬─────────┐
Push  Email  SMS  In-App   Web Push
Workers (Kafka consumers per channel)
     ↓
Channel-specific SDKs
  FCM/APNs → Mobile
  SendGrid/SES → Email
  Twilio → SMS
  WebSocket Gateway → In-App
```

**Notification prioritization:**
```python
class NotificationPriority:
    CRITICAL = 1    # Security alerts, payment failures → all channels immediately
    HIGH = 2        # Direct messages, order updates → push + in-app
    MEDIUM = 3      # Social interactions → in-app, batched push
    LOW = 4         # Marketing → email daily digest

def route_notification(notification, user_prefs):
    if notification.priority == CRITICAL:
        return ALL_CHANNELS
    
    if user_prefs.dnd_active():
        if notification.priority >= HIGH:
            return [PUSH]  # Only urgent push during DND
        else:
            return queue_for_after_dnd(notification)
    
    return user_prefs.preferred_channels_for(notification.category)
```

**Fan-out at scale:**
```python
# For viral events (celebrity posts, breaking news):
# Don't fan-out to all subscribers immediately
# Use layered fan-out with priority queue

class FanOutService:
    def fan_out(self, event, subscriber_ids):
        # Chunk subscribers into batches
        for batch in chunks(subscriber_ids, 1000):
            # Push to Kafka for async processing
            kafka.produce('notification-fanout', {
                'event': event,
                'subscribers': batch,
                'priority': 'normal'
            })
    
    # 1000 subscribers/batch × 1000 batches/topic × 100 consumers = massive throughput
```

**Deduplication and rate limiting:**
```python
def should_send(user_id, notification_type, entity_id):
    dedup_key = f"notif:{user_id}:{notification_type}:{entity_id}"
    
    # Don't send same notification twice in 1 hour
    if redis.set(dedup_key, 1, nx=True, ex=3600):
        # Rate limit: max 5 push notifications/hour
        rate_key = f"notif_rate:{user_id}:push"
        count = redis.incr(rate_key)
        redis.expire(rate_key, 3600)
        return count <= 5
    return False
```

---

## HARD (Q16–Q20)

---

### Q16. What are the key challenges in designing a real-time collaborative document editor?

**Answer:**

A real-time collaborative editor (Google Docs, Notion, Figma) is one of the most complex distributed systems challenges because it must combine strong consistency semantics with real-time performance.

**Challenge 1: Concurrent edit convergence**

Without coordination, two users editing the same character position produce inconsistent results. The system must ensure all clients converge to the same state.

```
OT approach (Google Docs):
  Central server serializes operations
  Each operation transformed against all concurrent operations
  O(n²) transformations for n concurrent users
  
CRDT approach (Figma, some Notion features):
  Each character gets globally unique position identifier
  Merge is always correct regardless of order
  No server serialization needed
  Trade-off: document grows (tombstones never removed)
```

**Challenge 2: Latency — showing local changes immediately**

```
Optimistic local application:
  User types character → immediately appears in their editor (local state)
  Character sent to server → server processes → echoes back with confirmation
  If server rejects (conflict) → revert local change

This is "operation echo" — you see your own change instantly,
even though confirmation hasn't arrived yet
```

**Challenge 3: Cursor and presence synchronization**

```python
# Cursor positions must be transformed too
# If User A inserts before User B's cursor → shift B's cursor right

class CollaborativeCursorManager:
    def transform_cursor(self, cursor_pos, applied_op):
        if applied_op.type == 'insert' and applied_op.position <= cursor_pos:
            return cursor_pos + len(applied_op.text)
        elif applied_op.type == 'delete' and applied_op.position < cursor_pos:
            return max(cursor_pos - applied_op.length, applied_op.position)
        return cursor_pos
```

**Challenge 4: Offline editing and reconnection**

```
User goes offline, makes 50 edits
User reconnects → must merge 50 local operations against N server operations

OT: Server transforms all 50 against server history → complex
CRDT: Merge state vectors → simpler but requires full state transfer or operation log
```

**Challenge 5: Undo/redo in collaborative context**

```
Alice types "Hello"
Bob types " World" (after Alice's text)
Alice hits CTRL+Z (undo) — should undo "Hello" but not " World"

Selective undo: Must undo only Alice's operations, even if Bob's operations
are interleaved. This requires tracking operation authorship and
dependency graphs. Extremely complex to implement correctly.
```

**Challenge 6: Scale — shared documents with 1000 concurrent editors**

```
Solution:
  1. Shard document into sections (pages, sections)
  2. Use section-level locks for coarse operations
  3. Segment presence updates (don't broadcast every cursor move to 1000 users)
  4. Throttle cursor updates: batch at 50ms intervals
```

---

### Q17. How does YouTube handle 500 hours of video uploaded per minute?

**Answer:**

YouTube processes an extraordinary volume of video content — 500 hours per minute = 8.3 hours of video every second. The system must transcode, store, index, and make content available globally.

**Upload pipeline:**
```
User uploads video
      ↓
Upload Service (chunked upload, resumable)
      ↓
Raw Storage (GCS - Google Cloud Storage)
      ↓
Processing Queue (Pub/Sub)
      ↓
Transcoding Service
  ├── Multiple resolutions: 360p, 480p, 720p, 1080p, 4K
  ├── Multiple codecs: VP9, H.264, AV1 (for newer devices)
  └── Multiple formats: WebM, MP4
      ↓
Processed Storage (GCS, globally replicated)
      ↓
CDN Distribution (Google's global network)
```

**Chunked upload (resumable):**
```python
# Client-side: upload in 256KB-5MB chunks
def resumable_upload(video_file, upload_url):
    chunk_size = 5 * 1024 * 1024  # 5MB chunks
    
    for offset in range(0, file_size, chunk_size):
        chunk = video_file.read(chunk_size)
        
        response = requests.put(
            upload_url,
            data=chunk,
            headers={
                'Content-Range': f'bytes {offset}-{offset+len(chunk)-1}/{file_size}',
                'Content-Length': str(len(chunk))
            }
        )
        
        if response.status_code == 308:  # Resume Incomplete
            offset = int(response.headers['Range'].split('-')[1]) + 1
        elif response.status_code == 200:
            return "Upload complete"
```

**Parallel transcoding:**
```
Input: raw_video.mov (4K, 1 hour = ~50GB)

Transcoding pipeline:
  Split into 30-second segments
  Transcode each segment in parallel:
    Segment 1 → 1080p, 720p, 480p (simultaneously on 3 workers)
    Segment 2 → 1080p, 720p, 480p (simultaneously on 3 workers)
    ...
  Merge segments per resolution

1 hour video that takes 4 hours to transcode sequentially
→ Split into 120 segments × 3 resolutions = 360 parallel jobs
→ Completes in ~5 minutes (wall clock)
```

**Infrastructure scale:**
- YouTube transcodes on a fleet of thousands of machines
- Uses Google's Borg (predecessor to Kubernetes) for job scheduling
- AV1 encoding is CPU-intensive — dedicated encoding farms with custom ASICs

**Metadata and search pipeline:**
```
Video uploaded → metadata stored (Spanner for global consistency)
                → thumbnail generated (frame extraction)
                → audio processed (speech-to-text for captions)
                → content moderation (ML models for policy violations)
                → indexed for search (Indexing pipeline → search serving)
```

---

### Q18. How does LinkedIn find 2nd-degree connections efficiently at 900M users?

**Answer:**

"People You May Know" and 2nd-degree connections require graph traversal over a social graph with hundreds of billions of edges.

**Scale:**
```
900M users
Average connections per user: ~300
Total edges: 900M × 300 / 2 = 135 billion edges
2nd-degree connections per user: ~300² = 90,000 (before deduplication)
```

**Challenge:** Graph databases (Neo4j) struggle at this scale. LinkedIn uses custom graph infrastructure.

**LinkedIn's Leo + Voldemort stack:**

```
Graph stored as adjacency list per user:
  user:123 → [456, 789, 1011, ...]  (1st-degree connections)

Stored in custom distributed key-value store (Voldemort/Espresso)
Partitioned by user_id: all connections of user X on same shard
```

**2nd-degree query algorithm:**
```python
def get_second_degree_connections(user_id):
    # Step 1: Get 1st degree connections (fast — single KV lookup)
    first_degree = graph_store.get(user_id)  # Returns [456, 789, ...]
    
    # Step 2: For each 1st-degree, get their connections
    # Fan-out: 300 parallel requests → returns 300 sets of ~300 IDs each
    second_degree_sets = parallel_fetch([
        graph_store.get(fid) for fid in first_degree
    ])
    
    # Step 3: Union + deduplicate + exclude self + exclude 1st degree
    all_second_degree = set().union(*second_degree_sets)
    all_second_degree -= set(first_degree)  # Remove 1st degree
    all_second_degree.discard(user_id)       # Remove self
    
    return all_second_degree  # ~90K candidates before ranking
```

**Ranking the candidates:**
```python
def rank_2nd_degree(candidates, user_id, first_degree):
    scores = {}
    for candidate in candidates:
        mutual_connections = len(
            set(graph_store.get(candidate)) & set(first_degree)
        )
        score = (
            mutual_connections * 10 +          # Primary signal
            profile_completeness(candidate) +  # Quality signal
            company_overlap(user_id, candidate) * 5 +  # Relevance
            school_overlap(user_id, candidate) * 3
        )
        scores[candidate] = score
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
```

**Optimization techniques:**
- **Pre-computation:** Offline jobs compute 2nd-degree connections nightly for each user, store in candidate store
- **Graph sampling:** For users with 10K+ connections, sample a subset for query
- **Bloom filters:** Quickly check if candidate is already a 1st-degree connection (O(1) membership check)

---

### Q19. What is the hardest part of designing a payment system, and how do you solve it?

**Answer:**

The hardest part of a payment system is ensuring **exactly-once execution** of financial transactions — money should move exactly once even when networks fail, servers crash, and users retry.

**The fundamental problem:**
```
1. Application charges credit card via Stripe API
2. Stripe charges the card → SUCCESS
3. Network failure — application never receives response
4. Application assumes failure → retries the charge
5. Card charged TWICE → customer angry, chargeback risk

OR:

1. Application debits account A
2. Server crashes
3. Account B never credited
4. Money lost in system
```

**Solution 1: Idempotency keys (for external APIs)**
```python
def charge_customer(customer_id, amount, order_id):
    # Idempotency key = stable identifier for this specific charge attempt
    idempotency_key = f"charge:{order_id}"  # stable, not random per retry
    
    # Check if already processed
    existing = db.get_payment_by_order(order_id)
    if existing:
        return existing  # Return original result, don't charge again
    
    response = stripe.charges.create(
        amount=amount,
        currency='usd',
        customer=customer_id,
        idempotency_key=idempotency_key  # Stripe deduplicates on this key
    )
    
    # Store result durably before returning
    db.save_payment({
        'order_id': order_id,
        'stripe_charge_id': response.id,
        'status': response.status,
        'amount': amount
    })
    
    return response
```

**Solution 2: Two-phase ledger (for internal transfers)**
```sql
-- Atomic debit-credit using database transaction
BEGIN;

-- Debit source account
UPDATE accounts
SET balance = balance - 100, version = version + 1
WHERE account_id = 'A'
  AND balance >= 100
  AND version = 5;  -- Optimistic lock

-- Credit destination account
UPDATE accounts
SET balance = balance + 100, version = version + 1
WHERE account_id = 'B';

-- Record ledger entry (immutable)
INSERT INTO ledger_entries (id, from_account, to_account, amount, status, created_at)
VALUES ('txn-001', 'A', 'B', 100, 'COMPLETED', NOW());

COMMIT;
```

**Solution 3: Saga with compensating transactions for multi-service payments**
```
Multi-step payment (e-commerce checkout):
  1. Reserve inventory (idempotent: reserve_id)
  2. Authorize credit card (idempotent: auth_code)
  3. Capture payment (idempotent: capture_id)
  4. Confirm order (idempotent: order_id)

Failure at step 3:
  → Compensate step 2: void authorization
  → Compensate step 1: release inventory

Each step stores: {step_id, status, result, compensating_action}
Saga coordinator retries failed steps with same idempotency key
```

**Solution 4: Reconciliation jobs**
```python
# Nightly reconciliation: compare internal records with payment processor
def nightly_reconciliation():
    stripe_transactions = stripe.get_all_transactions(date=yesterday)
    internal_records = db.get_payments(date=yesterday)
    
    discrepancies = find_discrepancies(stripe_transactions, internal_records)
    
    for discrepancy in discrepancies:
        if discrepancy.type == 'MISSING_CAPTURE':
            # We charged but didn't record → retroactively update ledger
            fix_missing_ledger_entry(discrepancy)
        elif discrepancy.type == 'DOUBLE_CHARGE':
            # Refund the duplicate
            issue_refund(discrepancy)
        else:
            alert_finance_team(discrepancy)
```

---

### Q20. How would you scale a chat system from 1M to 1B users — what changes at each order of magnitude?

**Answer:**

Scaling a chat system is a journey through fundamentally different architectural decisions at each stage.

**1M Users — Monolith + Vertical Scaling**
```
Architecture: Single monolithic chat server
Database: Single PostgreSQL instance
Storage: Local file system for media

Config:
  1 beefy server: 64 vCPU, 256GB RAM
  1 PostgreSQL: 32 vCPU, 128GB RAM
  20K concurrent WebSocket connections / server

Pain points at this stage:
  - Code deployment takes down all connections
  - Any DB migration is risky (table locks)
  - Geographic latency (one data center)
```

**10M Users — Service decomposition + Caching**
```
Changes:
  ├── Split: WebSocket gateway / Message service / User service
  ├── Add: Redis for session storage and message cache
  ├── Add: S3/GCS for media storage
  ├── Add: Read replicas for PostgreSQL
  └── Add: Load balancer (ALB) in front of WebSocket servers

Message delivery: Redis pub/sub for cross-server routing
Database: PostgreSQL with connection pooling (PgBouncer)
Connections: 10K × 10 WS servers = 100K concurrent

Pain points:
  - Message ordering across servers
  - Delivery guarantees (at-least-once vs exactly-once)
```

**100M Users — Distributed storage + Message queue**
```
Changes:
  ├── Database: Cassandra for message history (wide rows per conversation)
  │            PostgreSQL only for user metadata and contacts
  ├── Add: Kafka for reliable message delivery and fan-out
  ├── Add: Multi-region deployment (US + EU + APAC)
  ├── Add: CDN for media delivery
  ├── Add: Dedicated push notification service
  └── Add: Separate presence service

Message sharding: by conversation_id → same partition per conversation
Multi-region: route user to nearest region; inter-region sync for cross-region chats

Pain points:
  - Cross-region message delivery latency
  - Global presence aggregation
  - Data residency laws (EU GDPR — messages must stay in EU)
```

**1B Users — Global mesh + Specialized systems**
```
Changes:
  ├── Per-service dedicated infrastructure
  │   (each service has its own DB cluster, cache cluster, queue)
  ├── Message storage: Custom distributed log (Meta's ZippyDB, Cassandra at scale)
  ├── Add: Edge PoPs in every major city for WebSocket termination
  ├── Add: Separate analytics pipeline (ClickHouse) — don't query production DBs
  ├── Add: Machine learning pipeline (spam detection, content moderation)
  ├── Add: Custom CDN infrastructure (like WhatsApp's)
  └── Add: Comprehensive observability (distributed tracing, anomaly detection)

Scale numbers:
  100B messages/day
  1.16M messages/second peak
  5M concurrent connections
  1PB+ of messages stored (with encryption)

Key architectural shifts:
  1. No single database can hold all user data → partition by phone number prefix
  2. Encryption at rest and in transit for every message (E2E encryption)
  3. Operations team dedicated to capacity planning 6 months ahead
  4. Feature flags for gradual rollout (can't do big-bang deploys at this scale)
  5. Chaos engineering (regularly kill components to test resilience)
```

**Summary table:**
| Scale | DB | Transport | Storage | Key Addition |
|---|---|---|---|---|
| 1M | Single PostgreSQL | WebSocket | Local | N/A |
| 10M | PostgreSQL + replicas | WS + Redis pub/sub | S3 | Caching layer |
| 100M | Cassandra + PostgreSQL | WS + Kafka | CDN media | Message queue |
| 1B | Custom distributed | Edge WS + Kafka | Custom CDN | Full observability |

---

## Quick Reference

### URL Shortener Key Decisions
1. Code generation: random (unpredictable) vs counter (collision-free)
2. Storage: SQL (ACID) with Redis cache
3. Read path: cache-first (99%+ cache hit rate)

### Fan-Out Strategy
- Regular users → fan-out on write to follower caches
- Celebrities → fan-out on read (merge at query time)
- Threshold: ~10K followers separates regular from celebrity

### Database Selection Matrix
| Use case | Database choice |
|---|---|
| Product catalog | DynamoDB (NoSQL, read-heavy) |
| User accounts | PostgreSQL (ACID) |
| Orders | PostgreSQL/CockroachDB (ACID + sharding) |
| Search | Elasticsearch / OpenSearch |
| Sessions/cache | Redis |
| Analytics | ClickHouse / BigQuery |
| Time-series | InfluxDB / TimescaleDB |

### Preventing Double-Booking
1. Database exclusion constraints (most reliable)
2. Optimistic locking + retry
3. Distributed lock (Redis/ZooKeeper)

### Payment Exactly-Once
- External APIs: idempotency keys
- Internal transfers: database transactions + ledger
- Multi-service: Saga pattern
- Recovery: nightly reconciliation

### Kafka Throughput Enablers
1. Sequential disk I/O (append-only log)
2. Zero-copy (sendfile syscall)
3. Batching at producer, broker, consumer
4. Partitions for parallel consumers

### Chat Scaling Summary
```
1M  → Monolith, single DB
10M → Services + Redis pub/sub
100M → Cassandra + Kafka + multi-region
1B  → Custom everything + edge PoPs
```
