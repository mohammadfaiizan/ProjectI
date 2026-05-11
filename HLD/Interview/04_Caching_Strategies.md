# Caching Strategies — Interview Q&A

> 20 questions | Easy: Q1–Q7 | Medium: Q8–Q15 | Hard: Q16–Q20

---

## EASY (Q1–Q7)

---

### Q1. What is cache-aside (lazy loading) and when do you use it?

**Answer:**

Cache-aside (also called lazy loading) is the most common caching pattern. The application code is responsible for loading data into the cache. The cache does not interact with the storage directly.

**Read flow:**
```
Application requests data:
  1. Check cache: cache.get(key)
  2. CACHE HIT: return cached value → done
  3. CACHE MISS:
     a. Fetch from DB: db.query(...)
     b. Store in cache: cache.set(key, value, ttl)
     c. Return value to caller

Diagram:
  App ──get(key)──> Cache ──miss──> App ──query──> DB
   ↑                                              │
   └──────────────── return data ◄────────────────┘
                     + cache.set(key, data, ttl)
```

**Write flow:**
```
Application writes data:
  1. Write to DB: db.update(...)
  2. Invalidate cache: cache.delete(key)  [NOT update — see below]

Why invalidate rather than update?
  - Avoids race conditions between write and cache update
  - If write fails, cache is not inconsistent
  - Next read will repopulate cache with fresh data
```

**Advantages:**
- Only requested data is cached (lazy = populated on demand).
- Cache failures are non-fatal — app falls back to DB.
- Works well for read-heavy workloads.
- Cache can hold different data structures than the DB.

**Disadvantages:**
- First request after cache miss is slow (cache cold start).
- Cache stampede risk (many concurrent misses hit DB simultaneously — see Q9).
- Stale data possible between write and cache expiry.

**Best for:** Read-heavy workloads, varied access patterns, large datasets where only a fraction is hot.

```python
def get_user(user_id):
    key = f"user:{user_id}"
    
    # Try cache first
    user = cache.get(key)
    if user:
        return user  # Cache hit
    
    # Cache miss — fetch from DB
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # Populate cache with TTL
    cache.set(key, user, ttl=3600)
    return user
```

---

### Q2. What is write-through caching and when do you use it?

**Answer:**

In write-through caching, every write goes to both the cache AND the database synchronously. The cache is always kept in sync with the database.

**Write flow:**
```
Application writes:
  1. Write to CACHE (cache.set(key, value))
  2. Write to DB (db.update(...))
  3. Return success to caller (after BOTH complete)

         App
          │
    ┌─────┴─────┐
    ▼           ▼
  Cache        DB
  (write)    (write)
          Both synchronous
```

**Read flow:**
```
Application reads:
  1. Check cache: almost always a HIT (cache was pre-populated on write)
  2. CACHE MISS (cold start only): read from DB, populate cache
  
  Very high cache hit ratio because cache is kept warm on every write
```

**Advantages:**
- No stale data: cache is always up to date.
- After a cache restart, reads are slow only until writes repopulate.
- Consistent cache hits for recently written data.

**Disadvantages:**
- Write latency is doubled (must write to both cache and DB).
- Writes of data that is never read waste cache space.
- If DB write fails and cache write succeeded, inconsistency exists (use transactions or write cache only after successful DB write).

**Best for:** Write-then-immediately-read patterns. User profile updates, settings, configuration data where reads follow writes closely.

**Comparison: cache-aside vs write-through:**
| Property | Cache-Aside | Write-Through |
|---|---|---|
| Read path | Check cache, fetch DB on miss | Read from cache (always warm) |
| Write path | Write DB, invalidate cache | Write cache + DB |
| Cache warmth | Cold after restart or miss | Always warm after write |
| Stale data risk | Yes (TTL window) | No |
| Write overhead | Low | Double write |
| Wasted cache | No (lazy) | Yes (write rarely-read data) |

---

### Q3. What is write-back (write-behind) caching?

**Answer:**

In write-back (write-behind) caching, writes go to the cache first and are asynchronously flushed to the database later. The cache acknowledges the write immediately.

**Write flow:**
```
Application writes:
  1. Write to CACHE → immediately return success to caller
  2. Mark cache entry as "dirty"
  3. Background process periodically flushes dirty entries to DB

App ──write──> Cache ──ACK immediately──> App (done)
                 │
                 │ (asynchronously, later)
                 ▼
                DB

Timeline:
  t=0:   App writes to cache → returns success in 1ms
  t=5s:  Background flusher writes to DB
  t=10s: Another write coalesced with first → single DB write
```

**Advantages:**
- Very low write latency (just a cache write — memory speed).
- Batching: multiple writes to same key coalesce into one DB write.
- DB write I/O significantly reduced.

**Disadvantages:**
- Data loss risk: If cache crashes before flush, writes are lost.
- Complex implementation: Dirty tracking, flush scheduling, failure handling.
- Stale DB reads: If another service reads DB directly, it gets stale data.
- Ordering guarantees: Writes must be flushed in order.

**Best for:** Write-heavy workloads where some data loss is acceptable, or where writes can be reconstructed (gaming leaderboards, counters, analytics).

**Not suitable for:** Financial transactions, order records — use write-through or no-cache for these.

**Redis use case:**
```
Redis AOF (Append Only File) write-back:
  - All writes go to Redis in-memory
  - Redis buffers writes to AOF file (disk) every 1 second
  - If Redis crashes, up to 1 second of data may be lost
  - Trade-off: high write throughput vs durability
```

---

### Q4. What are cache eviction policies and which do you use when?

**Answer:**

When the cache reaches capacity, an eviction policy determines which entries to remove to make space for new ones.

**LRU — Least Recently Used (most common):**
```
Evict the entry that has not been accessed for the longest time.
Assumption: recently used data is more likely to be used again.

Cache: [D(t=10)] [A(t=5)] [B(t=3)] [C(t=1)]
                                     ↑ evict C (last used at t=1)

Use: General-purpose caching, web sessions, database query cache
```

**LFU — Least Frequently Used:**
```
Evict the entry that has been accessed the fewest times.
Assumption: infrequently accessed data is unlikely to be needed.

Cache: [A:freq=50] [B:freq=30] [C:freq=2] [D:freq=1]
                                            ↑ evict D (accessed only once)

Use: When access frequency matters more than recency (media file cache)
Problem: Old but frequently accessed items stuck in cache forever ("cache pollution")
```

**FIFO — First In, First Out:**
```
Evict the entry that was inserted first, regardless of usage.
Simple queue-based eviction.

Use: Simple implementations; when insertion order correlates with relevance
Problem: Ignores actual usage patterns — rarely the best choice
```

**TTL-based expiry (not strictly eviction, but related):**
```
Each entry has a time-to-live; expired entries are removed.
Can be combined with any eviction policy.

cache.set("key", value, ttl=3600)  # expires after 1 hour

Use: Data with known staleness deadline (session tokens, API responses)
```

**Random eviction:**
```
Evict a random entry.
Surprisingly effective at large scale (avoids pathological cases of LRU/LFU).
Used in Redis when maxmemory-policy is set to allkeys-random.
```

**Summary table:**

| Policy | Evicts | Memory Overhead | Best For |
|---|---|---|---|
| LRU | Least recently used | Doubly linked list | General-purpose (most common) |
| LFU | Least frequently used | Frequency counter | Content with stable popularity |
| FIFO | Oldest inserted | Queue pointer | Simple, ordered caches |
| Random | Random entry | None | Simple fallback; large caches |
| TTL | Expired entries | Expiry timestamp | Time-sensitive data |
| LRU-2 | Not used in last 2 accesses | Two timestamps | Scan resistance |

**Redis maxmemory-policy options:**
```
noeviction          — Return error when full (default)
allkeys-lru         — LRU across all keys
allkeys-lfu         — LFU across all keys (Redis 4.0+)
volatile-lru        — LRU only on keys with TTL set
volatile-ttl        — Evict keys with soonest TTL expiry
allkeys-random      — Random across all keys
```

---

### Q5. How do you implement LRU cache in O(1) time complexity?

**Answer:**

A naive LRU using only a hashmap allows O(1) lookup but O(n) eviction. Achieving O(1) for both get and put requires combining a hashmap with a doubly linked list.

**Data structure:**
```
HashMap: key → node pointer (O(1) lookup)
Doubly Linked List: maintains access order (head = most recent, tail = least recent)

            head (MRU)                          tail (LRU)
              │                                    │
              ▼                                    ▼
None ◄── [key:A, val:1] ◄──► [key:B, val:2] ◄──► [key:C, val:3] ──► None
              ↑                    ↑                    ↑
         HashMap["A"]         HashMap["B"]         HashMap["C"]
```

**Operations:**

**GET(key) — O(1):**
```
1. Lookup key in hashmap → get node pointer
2. Move node to HEAD of list (mark as recently used)
3. Return node.value
```

**PUT(key, value) — O(1):**
```
1. If key exists: update value, move node to HEAD
2. If key does not exist:
   a. Create new node, insert at HEAD
   b. Add to hashmap
   c. If capacity exceeded: remove node at TAIL, delete from hashmap
```

**Python implementation:**
```python
class ListNode:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {}  # key → node
        # Sentinel head and tail (simplify edge cases)
        self.head = ListNode(0, 0)  # most recently used side
        self.tail = ListNode(0, 0)  # least recently used side
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove(self, node):
        """Remove node from linked list"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _insert_at_head(self, node):
        """Insert node right after head (most recent)"""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
    
    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._insert_at_head(node)
            return node.val
        return -1
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        node = ListNode(key, value)
        self._insert_at_head(node)
        self.cache[key] = node
        if len(self.cache) > self.cap:
            # Evict LRU (node before tail)
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
```

**Time/Space complexity:**
| Operation | Complexity |
|---|---|
| get | O(1) |
| put | O(1) |
| Space | O(capacity) |

---

### Q6. What is cache hit ratio and why does it matter?

**Answer:**

Cache hit ratio is the percentage of requests that are served from the cache (rather than requiring a database or backend fetch).

```
Hit Ratio = Cache Hits / (Cache Hits + Cache Misses) × 100%

Example:
  Total requests: 1,000,000
  Cache hits:       950,000
  Cache misses:      50,000
  
  Hit Ratio = 950,000 / 1,000,000 = 95%
```

**Why it matters — cost and performance:**

```
Assume:
  DB read latency:    50ms
  Cache read latency:  1ms
  DB cost per read:   $0.0001
  Cache cost per read: $0.00001

At 10,000 QPS, 95% hit ratio:
  Cache reads: 9,500/sec × $0.00001 = $0.095/sec
  DB reads:      500/sec × $0.0001  = $0.050/sec
  Total:                               $0.145/sec

At 10,000 QPS, 50% hit ratio:
  Cache reads: 5,000/sec × $0.00001 = $0.050/sec
  DB reads:    5,000/sec × $0.0001  = $0.500/sec
  Total:                               $0.550/sec  ← 3.8× more expensive

Average latency:
  95% hit: 0.95×1ms + 0.05×50ms = 0.95 + 2.5 = 3.45ms
  50% hit: 0.50×1ms + 0.50×50ms = 0.5 + 25   = 25.5ms  ← 7× slower
```

**Factors affecting hit ratio:**

| Factor | Effect |
|---|---|
| Cache size | Larger cache → higher hit ratio (more data fits) |
| TTL | Shorter TTL → more misses (data expires faster) |
| Access pattern | Uniform (low hit ratio) vs. skewed/popular (high hit ratio) |
| Eviction policy | Poor policy → hot data evicted |
| Number of distinct keys | More distinct keys relative to cache size → lower ratio |

**Target hit ratios:**
- > 95%: Excellent — cache is very effective.
- 80–95%: Good for general use cases.
- < 80%: Investigate: cache too small, wrong keys cached, high TTL churn.

---

### Q7. What is the difference between Redis and Memcached?

**Answer:**

Both are in-memory key-value stores used for caching. Redis has significantly more features; Memcached is simpler and faster for basic use cases.

**Feature comparison:**

| Feature | Redis | Memcached |
|---|---|---|
| Data structures | String, Hash, List, Set, ZSet, Bitmap, HyperLogLog, Stream | String only (bytes) |
| Persistence | RDB snapshots + AOF log | None (data lost on restart) |
| Replication | Built-in primary-replica | No native replication |
| Clustering | Redis Cluster (built-in sharding) | Client-side sharding only |
| Pub/Sub | Yes | No |
| Lua scripting | Yes | No |
| Transactions | Yes (MULTI/EXEC) | No |
| TTL on keys | Yes | Yes |
| Max value size | 512 MB | 1 MB |
| Multi-threading | Single-threaded (I/O, 6.0+ multi-threaded I/O) | Multi-threaded |
| Memory efficiency | Lower (metadata per key) | Higher (simpler) |

**When to use Memcached:**
- Pure caching of simple strings (serialized objects).
- Need multi-threaded performance for extremely high QPS on simple gets/sets.
- Simple horizontal scaling by adding nodes (no coordination).
- Minimal operational complexity.

**When to use Redis:**
- Need data structures beyond strings (e.g., sorted sets for leaderboards).
- Need persistence (cache that survives restarts).
- Need pub/sub messaging.
- Need transactions or Lua scripts.
- Need built-in replication and clustering.
- Rate limiting, session storage, distributed locking.

**Practical advice:** In most new systems, choose Redis. Its richer feature set and built-in clustering make it more versatile. The only case to choose Memcached is if you have very high throughput, simple cache-only workloads, and want to avoid Redis's single-threaded bottleneck on older versions.

---

## MEDIUM (Q8–Q15)

---

### Q8. What is a cache stampede (thundering herd) and how do you prevent it?

**Answer:**

A cache stampede occurs when many concurrent requests find the same key has expired (cache miss), all simultaneously query the database, overwhelming it.

**The problem:**
```
Key "popular_product:123" expires at t=100

At t=100:
  Request 1: cache.get("popular_product:123") → MISS
  Request 2: cache.get("popular_product:123") → MISS  (same instant)
  Request 3: cache.get("popular_product:123") → MISS
  ...
  Request 500: cache.get("popular_product:123") → MISS

All 500 requests simultaneously hit the database!
DB gets 500 queries for same key → overload → cascade failure
```

**Prevention strategies:**

**1. Mutex / Distributed Lock (most reliable):**
```python
def get_with_lock(key):
    value = cache.get(key)
    if value:
        return value  # Cache hit
    
    # Try to acquire lock
    lock_key = f"lock:{key}"
    if cache.setnx(lock_key, "1", ex=30):  # Atomic set if not exists
        try:
            # This process fetches from DB
            value = db.query(key)
            cache.set(key, value, ttl=3600)
            return value
        finally:
            cache.delete(lock_key)
    else:
        # Another process is fetching — wait and retry
        time.sleep(0.05)
        return get_with_lock(key)  # Retry (will hit cache on next attempt)
```

**2. Probabilistic Early Expiry (PER — elegant, no locks):**
```python
def get_probabilistic(key):
    entry = cache.get_with_expiry(key)  # Returns (value, expiry_time)
    
    if entry:
        ttl_remaining = entry.expiry_time - time.time()
        beta = 1.0  # tuning parameter
        
        # Randomly recompute early based on remaining TTL
        if -beta * math.log(random.random()) >= ttl_remaining:
            # Probabilistically decide to refresh early
            value = db.query(key)
            cache.set(key, value, ttl=3600)
            return value
        
        return entry.value  # Return existing value
    
    # Hard miss — fetch from DB
    value = db.query(key)
    cache.set(key, value, ttl=3600)
    return value

# Effect: As TTL approaches 0, probability of early refresh increases
#         Single worker refreshes before mass expiry → no stampede
```

**3. Stale-While-Revalidate:**
```
Cache stores: value + expiry + extended_expiry

At normal expiry: Return stale value immediately
                  Trigger async background refresh
At extended expiry: Must synchronously refresh (true expiry)

Client gets fast response (stale) while cache refreshes in background
```

**4. Jitter on TTL:**
```python
# Instead of fixed TTL=3600:
ttl = 3600 + random.randint(-300, 300)  # ±5 minutes jitter

# Keys set at similar times expire at different times
# Prevents simultaneous mass expiry of a cohort of keys
```

---

### Q9. Explain TTL-based vs event-driven cache invalidation.

**Answer:**

Cache invalidation is notoriously one of the hardest problems in computer science. There are two main approaches for keeping caches fresh.

**TTL-based invalidation:**
```
Every cache entry has a Time-To-Live
Entry expires automatically after TTL seconds

cache.set("product:123", data, ttl=3600)  # expires in 1 hour

At t+3600: entry deleted, next request gets fresh data from DB

Pros:
  + Simple to implement (no invalidation logic in application)
  + Bounded staleness (max stale = TTL)
  + Self-healing (stale entries clean up automatically)

Cons:
  - Data may be stale for up to TTL seconds
  - Short TTL → more DB pressure; Long TTL → more staleness
  - Cannot react to updates immediately

Use: Non-critical data with acceptable staleness:
  - Product listings (1-hour TTL is fine)
  - Search results (10-minute TTL)
  - Configuration data (5-minute TTL)
```

**Event-driven invalidation:**
```
Application explicitly invalidates cache when data changes

On DB write:
  cache.delete("product:123")

Next read:
  cache.get("product:123") → miss → fetch fresh from DB → repopulate

Pros:
  + Near-zero staleness (cache updated immediately on change)
  + Cache is accurate

Cons:
  - More complex: must invalidate in every write path
  - Cache misses on write → potential stampede
  - Risk of missed invalidation (application bug → stale cache forever)
  - Distributed systems: must invalidate ALL cache nodes

Patterns:
  1. Delete-on-write: cache.delete(key) after DB write
  2. Update-on-write: cache.set(key, new_value) after DB write
     (write-through pattern — see Q2)
```

**Cache invalidation via CDC (Change Data Capture):**
```
Decouple DB writes from cache invalidation using DB transaction log:

DB Primary
    │
    ├── Row updated in table "products"
    │
    └── CDC connector (Debezium) reads binlog/WAL
              │
              ▼
         [Kafka Topic: db.products.changes]
              │
              ▼
         [Cache Invalidation Service]
              │
              ├── cache.delete("product:123")
              └── cache.delete("product_list:category:electronics")

Pros:
  + Application code does NOT need to explicitly invalidate
  + Works for any write source (multiple services, direct DB writes)
  + Reliable ordering (follows DB commit order)
Cons:
  + Complex infrastructure
  + Small delay: CDC lag + Kafka lag + consumer lag (typically < 1 second)
```

---

### Q10. How does distributed cache consistent hashing work?

**Answer:**

When you have multiple cache nodes, you need a consistent way to determine which node stores a given key. Consistent hashing minimizes key redistribution when nodes are added or removed.

**Naive modulo approach (problematic):**
```
3 cache nodes: hash(key) % 3

"user:123" → hash=450 → 450%3=0 → Node 0
"user:456" → hash=891 → 891%3=0 → Node 0

Add 4th node: hash(key) % 4
"user:123" → 450%4=2 → Node 2  ← MOVED! (cache miss)
"user:456" → 891%4=3 → Node 3  ← MOVED! (cache miss)

~75% of keys remapped on each scale event → mass cache miss → DB overload
```

**Consistent hashing for distributed cache:**
```
Hash ring: 0 → 2^32

Place nodes on ring:
  hash("cache-node-1") → position 100
  hash("cache-node-2") → position 250
  hash("cache-node-3") → position 400

For each key: hash(key) → walk clockwise to nearest node

"user:123" → hash=150 → clockwise → Node 2 (at 250)
"user:456" → hash=350 → clockwise → Node 3 (at 400)
"user:789" → hash=450 → clockwise → Node 1 (at 100 → wraps)
```

**Adding/removing nodes:**
```
Add Node 4 at position 300:
  Keys between 250 and 300 now route to Node 4 (were on Node 3)
  Only keys in that range are remapped → ~25% remapped vs 75% with modulo

Remove Node 2 (at position 250):
  Keys between 100 and 250 now route to Node 3 (were on Node 2)
  Only keys on Node 2 remapped → ~33% of keys (1/3 of the ring)
```

**Virtual nodes (vnodes) for uniform distribution:**
```
With 3 nodes and 150 vnodes each:
  Node 1: positions [50, 150, 300, 600, ...]  (150 virtual positions)
  Node 2: positions [100, 200, 450, 700, ...]
  Node 3: positions [75, 250, 350, 800, ...]

Keys are uniformly distributed across all nodes even with small node count
Weighted capacity: stronger nodes get more vnodes
```

**Redis Cluster implementation:**
```
Redis Cluster uses 16,384 hash slots (not full consistent hashing):
  hash_slot = CRC16(key) % 16384

Slots assigned to nodes:
  Node 1: slots 0–5460
  Node 2: slots 5461–10922
  Node 3: slots 10923–16383

Add node: Move some slots from existing nodes to new node
Remove node: Move all its slots to remaining nodes
Only keys in moved slots are affected
```

---

### Q11. Explain Redis data structures and their use cases.

**Answer:**

Redis supports multiple data structures beyond simple key-value strings. Choosing the right structure dramatically impacts both performance and code simplicity.

**String:**
```
Commands: SET, GET, INCR, DECR, APPEND, SETEX
Use cases:
  - Caching serialized objects (JSON/Protobuf)
  - Counters (page views, API call count): INCR counter:api:user123
  - Rate limiting: INCR + EXPIRE
  - Session storage: SET session:abc123 "{userId:1}" EX 3600
  - Feature flags: SET feature:new_ui "true"

SET user:123 '{"name":"Alice","email":"alice@example.com"}' EX 3600
GET user:123
INCR page_views:home
```

**Hash:**
```
Commands: HSET, HGET, HGETALL, HMSET, HINCRBY
A map of field → value within a single key

Use cases:
  - User objects (avoid serializing/deserializing full JSON for partial updates)
  - Session data with individual field updates
  - Shopping cart items

HSET user:123 name "Alice" email "alice@example.com" age 30
HGET user:123 name          → "Alice"
HINCRBY user:123 login_count 1  → atomic increment of one field
HGETALL user:123            → all fields

Memory efficient: < 64 fields with small values → stored as ziplist
```

**List:**
```
Commands: LPUSH, RPUSH, LPOP, RPOP, LRANGE, LLEN
Ordered list of strings; efficient push/pop at both ends

Use cases:
  - Message queues (LPUSH to add, RPOP to consume)
  - Activity feeds (latest N items): LPUSH feed:user:123 "event"
  - Recent browsing history: LPUSH history:user:123 "product:456" → LTRIM to last 20

LPUSH queue:tasks "task1" "task2"
RPOP queue:tasks        → consume from right (FIFO)
LRANGE recent:user:123 0 19  → last 20 items
```

**Set:**
```
Commands: SADD, SMEMBERS, SISMEMBER, SUNION, SINTER, SDIFF
Unordered collection of unique strings

Use cases:
  - Unique visitor tracking: SADD daily_visitors:2024-01-01 "user123"
  - Tag sets: SADD tags:product:123 "electronics" "laptop"
  - Mutual friends: SINTER friends:alice friends:bob
  - Access control: SISMEMBER admin_users "user456"

SADD online_users "user123" "user456"
SCARD online_users          → count unique online users
SISMEMBER online_users "user789"  → 0 (not online)
```

**Sorted Set (ZSet):**
```
Commands: ZADD, ZRANK, ZRANGE, ZREVRANGE, ZSCORE, ZINCRBY
Ordered set with float score per member

Use cases:
  - Leaderboards: ZADD leaderboard 1500.0 "player:123"
  - Rate limiting (sliding window): ZADD user:123:requests timestamp timestamp
  - Priority queues: score = priority or timestamp
  - Geospatial index (Redis GEO uses ZSet internally)

ZADD leaderboard 5000 "player:alice" 4200 "player:bob" 6100 "player:carol"
ZREVRANGE leaderboard 0 9 WITHSCORES  → top 10 players
ZRANK leaderboard "player:alice"       → rank (0-indexed)
ZINCRBY leaderboard 500 "player:alice" → add 500 to alice's score
```

**Stream:**
```
Append-only log (like Kafka, but in Redis)
Commands: XADD, XREAD, XGROUP, XACK

Use: Message streaming, audit logs, event sourcing

XADD events:orders * type "order_placed" user "123" amount "49.99"
XREAD COUNT 10 STREAMS events:orders 0  → read 10 events
```

---

### Q12. What is the difference between Redis RDB and AOF persistence?

**Answer:**

Redis offers two persistence mechanisms that trade off durability against performance.

**RDB (Redis Database Backup) — Snapshots:**
```
Redis forks a child process and writes the entire dataset to a binary .rdb file
at configured intervals.

Configuration (redis.conf):
  save 900 1      # Save if ≥1 change in 900 seconds
  save 300 10     # Save if ≥10 changes in 300 seconds
  save 60  10000  # Save if ≥10,000 changes in 60 seconds

Snapshot process:
  t=0:   Redis forks child process
  t=0:   Child serializes all keys to dump.rdb (copy-on-write)
  t=10s: Snapshot complete; dump.rdb replaces old file

Recovery:
  On restart: Load dump.rdb → last snapshot restored
  Data loss: All writes SINCE last snapshot are lost
  Typical RPO: 1-15 minutes
```

**AOF (Append Only File) — Write Log:**
```
Redis appends every write command to an append-only file.

fsync options:
  appendfsync always:    fsync on every write → zero data loss, very slow
  appendfsync everysec:  fsync every second → at most 1 second of data loss
  appendfsync no:        OS decides → fast, variable data loss

AOF file:
  *3
  $3
  SET
  $6
  user:1
  $5
  Alice
  *3
  $4
  INCR
  ...

Recovery:
  On restart: Replay all commands in AOF → fully reconstructed state
  Data loss: Depends on fsync policy (0 to 1 second with everysec)

AOF rewrite:
  Over time, AOF grows large (many updates to same key logged separately)
  BGREWRITEAOF: Redis rewrites AOF compactly (only final state per key)
  Automatic: When AOF size exceeds auto-aof-rewrite-percentage threshold
```

**Comparison:**

| Property | RDB | AOF |
|---|---|---|
| Data loss (RPO) | Minutes (last snapshot) | 0–1 second (with everysec) |
| File size | Compact (binary) | Large (grows over time) |
| Restart recovery speed | Fast (load binary) | Slow (replay all commands) |
| Write performance impact | Low (fork periodically) | Higher (append per write) |
| Use case | Backup, fast restart | Durability requirement |

**Recommended configuration:**
```
Both enabled (hybrid approach):
  RDB:  Compact snapshot for fast restarts
  AOF:  Fine-grained durability between snapshots

On restart:
  If AOF exists: use AOF (more complete)
  Else: use RDB

This is the AWS ElastiCache Redis default for persistence-enabled clusters.
```

---

### Q13. What is the difference between Redis Cluster and Redis Sentinel?

**Answer:**

Both solve high availability for Redis, but they address different problems.

**Redis Sentinel:**
```
Purpose: High availability for a single Redis deployment (failover only)
Architecture:
  ┌─────────────────────────────────────┐
  │  Sentinel Process 1                 │
  │  Sentinel Process 2  (quorum: 2/3) │
  │  Sentinel Process 3                 │
  └────────────────┬────────────────────┘
                   │ Monitor + Failover
         ┌─────────┴─────────┐
    [Redis Primary]    [Redis Replica 1]
                       [Redis Replica 2]

Sentinel responsibilities:
  - Monitor: Continuously checks primary and replicas
  - Notification: Alert when state changes
  - Automatic failover: Promote replica to primary on primary failure
  - Config provider: Clients ask Sentinel for current primary IP

Failover process:
  1. Primary goes down
  2. Sentinels vote (quorum required to avoid split-brain)
  3. Best replica promoted to primary (least replication lag)
  4. Other replicas reconfigured to replicate from new primary
  5. Client reconnects to new primary (via Sentinel config service)

Data capacity: Limited to single server's RAM (no sharding)
```

**Redis Cluster:**
```
Purpose: Horizontal scaling + high availability
Architecture:
  6 nodes (3 primary + 3 replica):
  
  [Primary A (slots 0-5460)]    + [Replica A]
  [Primary B (slots 5461-10922)] + [Replica B]
  [Primary C (slots 10923-16383)]+ [Replica C]

Data sharding:
  hash_slot = CRC16(key) % 16384
  Each primary owns a portion of slots (keys)

High availability:
  If Primary A fails → Replica A promoted automatically
  Cluster continues serving (2/3 primaries still up)
  Cluster becomes unavailable if primary fails with no replica

Capacity: Horizontal scaling by adding shards (primary+replica pairs)

Client requirements:
  Redis Cluster-aware clients needed
  Client must handle MOVED and ASK redirects (slot migrations)
  Most Redis clients (redis-py, Jedis, ioredis) support clustering
```

**Comparison:**

| Property | Sentinel | Redis Cluster |
|---|---|---|
| Primary purpose | HA failover | Horizontal sharding + HA |
| Data sharding | No | Yes (16,384 slots) |
| Scale limit | Single server RAM | Effectively unlimited (add shards) |
| Operations | Simpler | More complex |
| Multi-key commands | All supported | Only within same slot |
| Minimum nodes | 3 sentinel + 2 Redis | 6 nodes (3 primary + 3 replica) |
| Best for | < 100 GB dataset | > 100 GB or > 100K writes/sec |

---

### Q14. What is the Redlock algorithm for distributed locking?

**Answer:**

Redlock is a distributed mutual exclusion algorithm for Redis that provides a safe distributed lock even when using multiple independent Redis nodes.

**Problem with single-node Redis locking:**
```
Standard lock on one Redis:
  SET lock:resource "client-id" NX EX 30  (atomic: set if not exists)
  
  Problem: If Redis primary fails, lock is lost even if client holds it.
  Replica may not have the lock yet (async replication).
  Another client could acquire the same "lock" from a promoted replica.
```

**Redlock algorithm (5 independent Redis nodes):**
```
ACQUIRE lock on N=5 independent Redis nodes:

1. Get current time T1 (milliseconds)

2. Try to SET lock on EACH of the 5 nodes:
   SET lock:resource "unique-client-id" NX PX 30000
   (atomic: set only if key doesn't exist, 30s TTL)
   
   Non-blocking: if one node is down, skip it quickly

3. Get current time T2
   Elapsed = T2 - T1
   
4. Consider lock ACQUIRED if:
   a. Set successfully on at least N/2 + 1 = 3 nodes (majority)
   b. Elapsed < lock_ttl - safety_margin
      (time to acquire < lock validity time)

5. Lock validity time = lock_ttl - elapsed - clock_drift_factor

RELEASE lock:
  Delete the lock key on ALL nodes:
  Lua script: if GET key == "my-unique-id" then DEL key end
  (atomic compare-and-delete prevents releasing another client's lock)
```

**Failure handling:**
```
Scenario: 2 Redis nodes fail (3/5 still running)
  Client sets lock on 3/5 nodes → majority → lock acquired
  
Scenario: Client acquires lock on 3 nodes, then crashes
  Lock expires on all nodes after TTL → auto-release
  Other clients can acquire after TTL

Scenario: Clock skew between Redis nodes
  Redlock accounts for clock_drift_factor in validity calculation
```

**Controversies (Martin Kleppmann critique):**
```
Flaw: Network pauses can violate mutual exclusion
  1. Client 1 acquires Redlock, gets 30s validity
  2. Client 1 pauses (GC, network, swap) for 31 seconds
  3. Lock expires on all nodes
  4. Client 2 acquires lock
  5. Client 1 resumes — thinks it still has lock!
  Both clients hold the "lock" simultaneously

Mitigation: Fencing tokens
  Lock manager provides monotonically increasing token
  Storage layer rejects writes with lower token than previously seen
  Client 1 token=5 (expired), Client 2 token=6 → Client 1's writes rejected
```

**Practical alternatives:**
- For most use cases, single-node Redis locking with NX+EX is sufficient.
- For strict mutual exclusion, use ZooKeeper or etcd with sequential ephemeral nodes.
- Redlock is a middle ground when you already have Redis and need reasonable distributed locking.

---

### Q15. What is the write-around cache pattern?

**Answer:**

Write-around is a pattern where writes bypass the cache entirely and go directly to permanent storage. The cache is populated only on a read miss.

**Flow:**
```
WRITE:
  Application ──write──> Database (directly, NO cache update)
  Cache is NOT updated on write

READ:
  Application ──get(key)──> Cache → MISS (since write bypassed cache)
  Application ──query──> Database → get data
  Application ──set(key, data, ttl)──> Cache (populate on first read)
  Application ←── data
```

**Comparison with cache-aside:**

Write-around IS cache-aside — the distinction is emphasizing that writes go directly to the database without touching the cache. Cache-aside describes the full pattern (lazy population); write-around specifically emphasizes the write behavior.

**When write-around is appropriate:**

| Scenario | Reason |
|---|---|
| Large data writes that will not be re-read | Avoid polluting cache with cold data |
| Batch imports (loading 1M records into DB) | None of the imported data should displace hot cache entries |
| Log entries / audit trails | Written once, rarely read — cache pollution waste |
| One-time operations | Password reset tokens, one-time codes |

**Problem it solves:**
```
Without write-around (write-through for write-heavy batch):
  Batch import of 5M product records → all 5M go to cache
  Evicts 5M hot user/session cache entries (cold data displaces hot data)
  Cache hit ratio drops from 95% to 20%
  DB gets hammered as all hot data must be reloaded

With write-around:
  Batch import → goes straight to DB, cache untouched
  Hot user/session data stays in cache
  Cache hit ratio maintains 95%
```

---

## HARD (Q16–Q20)

---

### Q16. How do you handle hot keys in a distributed cache?

**Answer:**

A hot key is a cache key accessed so frequently that a single cache node cannot handle the load. This is a common problem for viral content, popular products, or shared configuration keys.

**The problem:**
```
Redis Cluster: 6 shards
Key "product:iphone15" → always routes to Shard 3

All 500K reads/second for "product:iphone15" → Shard 3 only
Shard 3: CPU 100%, saturated → errors
Other shards: CPU 5%, idle

The load cannot be distributed — key always maps to same shard
```

**Solution 1: Key replication with random suffix:**
```python
# Instead of one key, create N copies across N shards
REPLICATION_FACTOR = 10

def set_hot_key(key, value, ttl):
    for i in range(REPLICATION_FACTOR):
        replicated_key = f"{key}:replica:{i}"
        cache.set(replicated_key, value, ttl=ttl)

def get_hot_key(key):
    # Read from random replica
    i = random.randint(0, REPLICATION_FACTOR - 1)
    replicated_key = f"{key}:replica:{i}"
    return cache.get(replicated_key)

# CRC16("product:iphone15:replica:0") → Shard 2
# CRC16("product:iphone15:replica:1") → Shard 5
# CRC16("product:iphone15:replica:2") → Shard 1
# ... spread across shards!
```

**Solution 2: Local application-tier cache:**
```python
# Each app server maintains a small in-memory LRU cache
from functools import lru_cache

local_cache = TTLCache(maxsize=1000, ttl=5)  # cachetools

def get_product(product_id):
    # L1: Local in-process cache (zero network, sub-microsecond)
    if product_id in local_cache:
        return local_cache[product_id]
    
    # L2: Distributed Redis cache (1ms network hop)
    value = redis_cache.get(f"product:{product_id}")
    if value:
        local_cache[product_id] = value  # Populate L1
        return value
    
    # L3: Database
    value = db.get_product(product_id)
    redis_cache.set(f"product:{product_id}", value, ttl=60)
    local_cache[product_id] = value
    return value

# 100 app servers × 1000 items cached locally
# Hot key hit rate: near 100% from L1, zero Redis pressure
```

**Solution 3: Dedicated hot key shards:**
```
Detect hot keys: Monitor Redis with MONITOR or CLIENT GETNAME + key tracking
When key exceeds threshold (e.g., > 10K reads/sec):
  Migrate key to dedicated Redis instance for hot keys only
  Hot key Redis: higher memory, isolated from regular traffic
```

**Solution 4: Read-through with cache-aside stampede protection:**
```
For hot keys, use probabilistic early expiry (see Q8) to prevent stampede
+ In combination with key replication to distribute load
```

**Monitoring for hot keys:**
```
Redis built-in (Redis 4.0+):
  redis-cli --hotkeys    # Finds top-N most frequently accessed keys
  
AWS ElastiCache:
  CloudWatch: CacheMisses, EngineCPUUtilization
  Redis SlowLog: SLOWLOG GET 10
  
Custom: Implement hit tracking in application code
  On each cache.get(key): metrics.increment(f"cache_hit:{key}")
  Alert: if key > 10K accesses/sec in 1 minute → hot key detected
```

---

### Q17. Design a multi-level cache architecture for a high-traffic e-commerce platform.

**Answer:**

Multi-level (L1/L2/L3) caching reduces latency and reduces load on each subsequent tier. Think of it like CPU cache hierarchy applied to distributed systems.

**Architecture:**
```
Request flow:
  Browser → Edge CDN (L0) → API Server L1 (in-process) → Redis L2 → DB L3

┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT BROWSER                              │
│                  Browser Cache (Cache-Control headers)              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Cache miss
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    L0: CDN EDGE (Cloudflare/CloudFront)             │
│    Static assets: HTML, CSS, JS, Images (TTL: 1hr-1yr)             │
│    Product pages (TTL: 5min) | Search pages (TTL: 1min)            │
│    Hit ratio target: 80%+ for static, 40%+ for dynamic             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Cache miss
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│               L1: APPLICATION TIER (per app server)                 │
│    In-process cache (Python: cachetools, Java: Caffeine)            │
│    Capacity: 1,000–10,000 hot items per server                      │
│    TTL: 5–60 seconds (very short; must be fresh)                   │
│    Hit ratio target: 60-80% for hot items                           │
│    ─────────────────────────────────────────────────────────        │
│    100 servers × 5,000 items = 500K items total (distributed L1)   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Cache miss
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   L2: DISTRIBUTED CACHE (Redis Cluster)             │
│    6 shards (3 primary + 3 replica)                                 │
│    Capacity: 50 GB total                                            │
│    TTL: 1min–1hr (by data type)                                     │
│    Hit ratio target: 90%+ of L2 queries                            │
│    Products, user profiles, session data, rate limit counters       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Cache miss
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     L3: DATABASE (PostgreSQL)                        │
│    Read replicas for read scaling                                   │
│    Only ~5% of requests reach here (with L0+L1+L2 combined)        │
└─────────────────────────────────────────────────────────────────────┘
```

**Cache assignment by data type:**

| Data Type | L0 CDN | L1 In-Process | L2 Redis | DB |
|---|---|---|---|---|
| Static assets | 1 year | No | No | Origin |
| Product details | 5 min | 30 sec | 10 min | Source of truth |
| User profile | No | 5 sec | 1 hour | Source of truth |
| Session data | No | No | 24 hours | Session service |
| Search results | 1 min | 10 sec | 5 min | Elasticsearch |
| Cart contents | No | No | 1 hour | DB (volatile) |
| Order history | No | No | 30 min | DB (append-only) |
| Rate limit counters | No | No | 60 sec sliding | Never |

**Cache invalidation strategy:**

```
Product price update:
  1. DB: Update products table
  2. CDC (Debezium): Detects change in binlog
  3. Kafka: Publishes "product:123:updated" event
  4. Cache invalidation service:
     - DELETE cache L2: "product:123"
     - Publish CDN purge: /products/123
  5. L1 expires naturally (30-sec TTL too short to worry about)
  
  Result: Max staleness = 30 seconds (L1 TTL) for most users
          5 seconds for CDN-cached product pages (aggressive CDN purge)
```

---

### Q18. How do you ensure cache consistency in distributed systems?

**Answer:**

Cache consistency means the cache accurately reflects the underlying data source. In distributed systems, this is inherently difficult due to replication lag, network partitions, and concurrent operations.

**Consistency models for caches:**

**Strong consistency (expensive):**
```
Read-your-writes: After a write, immediately see the new value.
Implementation: Write to DB + synchronously update/invalidate cache
                Read always checks if cache is fresh before serving

Cost: High — synchronous operations; cache write and DB write in same transaction
Use: Financial balances, inventory counts, medical records
```

**Eventual consistency (practical):**
```
After a write, cache will eventually reflect the new value.
TTL ensures data is not stale beyond a maximum window.

Staleness window = max(replication lag, TTL)

Use: Product descriptions, user preferences, non-critical counts
```

**Read-after-write consistency for specific users:**
```
Problem: User updates profile → gets their request routed to replica
         Replica not yet caught up → user sees stale profile

Solution: Session-based consistency
  After user writes: set flag "user:123:wrote_recently" with TTL=5s
  On read: if flag exists → route to primary DB (not replica)
           else → route to replica (default)
  
  After 5s: replication has caught up; read from replica is safe
```

**Cache consistency with distributed locks:**
```
Write + cache update atomically:

# Problem: race between two writers updating cache
Thread 1: db.update(user:123, v1) → cache.set(user:123, v1)
Thread 2: db.update(user:123, v2) → cache.set(user:123, v2)

If Thread 2's cache.set executes before Thread 1's:
  Cache: v1 (stale!) DB: v2 (correct)

Solution: Version numbers + conditional writes
  db.update(user:123, v2, version=2)  # DB has version column
  cache.set(user:123, v2) only if version in cache < 2
  
  Or: Use event sourcing — DB CDC drives cache invalidation in order
```

**"Delete on write" vs "update on write":**
```
DELETE on write (safer):
  db.update(key, new_value)
  cache.delete(key)  ← delete, don't update
  
  Why? If update fails after DB write: cache has stale data indefinitely
       If delete fails: just a cache miss; next read gets fresh data from DB
       Delete failure is always recoverable via TTL

UPDATE on write (write-through, for read-after-write):
  db.update(key, new_value)
  cache.set(key, new_value, ttl=X)
  
  Use only when you need read-after-write consistency
  Risk: Cache and DB can diverge if either write fails (need transactions)
```

---

### Q19. What data should NOT be cached?

**Answer:**

Understanding what NOT to cache is as important as understanding what to cache. Caching the wrong data causes correctness bugs, security vulnerabilities, or wasted resources.

**1. Highly dynamic data (changes every request):**
```
Stock prices updating 100× per second:
  Cache TTL of 1s → still stale for some users
  Cache with no TTL → serves massively stale data
  
  Better: Real-time pub/sub (WebSocket) or very short TTL (1s) accepted

Live scores during games:
  Accept: eventual consistency with 1-5s TTL
  Avoid: Caching with longer TTL
```

**2. User-specific sensitive data (security risk):**
```
Bank account balance — NEVER cache shared:
  Two users using same browser (library, shared device)
  User A logs in → balance cached in browser
  User A logs out, User B logs in → browser may return User A's cached balance
  
  Fix: Cache-Control: no-store for sensitive financial data
       Or: Cache with user-specific key, short TTL, HTTPS-only, Vary: Cookie

Authentication tokens and passwords:
  NEVER cache in shared caches (CDN, reverse proxy)
  Vary: Authorization or Vary: Cookie header
```

**3. Rarely accessed data (low benefit, wastes space):**
```
An archive of 10-year-old order records:
  Accessed by < 0.01% of users
  Caching displaces hot data (user sessions, current products)
  
  Better: Serve directly from DB; data is cold, latency acceptable
```

**4. Large objects that are infrequently accessed:**
```
100 MB video file in Redis:
  Redis is RAM-based; 100 MB per video is wasteful
  Displaces thousands of small frequently-accessed items
  
  Better: Object storage (S3) with CloudFront CDN for large files
          Redis for metadata only (title, duration, thumbnail URL)
```

**5. Unpredictably invalidated data:**
```
Data that changes based on complex business rules:
  Product price changes based on real-time demand, competitor prices
  + Cache invalidation requires tracking all affected keys
  + Multiple writes → multiple invalidations → high complexity
  
  Better: Very short TTL (1 minute) or no cache; complexity not worth it
```

**6. Write-heavy data where reads are rare:**
```
Log entries: written millions of times per second, read occasionally
  Caching adds no benefit (rarely read → no cache hits)
  Wastes cache space
  
  Better: Write directly to storage (S3, Elasticsearch, ClickHouse)
```

**Decision checklist:**
```
Cache if ALL of these are true:
  ✓ Read more than twice (cache warmup cost is amortized)
  ✓ Data is not extremely sensitive (or properly secured)
  ✓ Data changes infrequently relative to TTL
  ✓ Data is not too large for cache tier
  ✓ Access pattern is not perfectly uniform (some keys are hotter)
```

---

### Q20. How do you size a cache? Walk through the calculation.

**Answer:**

Cache sizing determines how much memory to allocate to achieve a target hit ratio. The goal is to find the knee of the hit-ratio-vs-size curve.

**Step 1: Understand your working set:**
```
Working set = set of data actively accessed within a time window

For an e-commerce site with 5M products:
  Products accessed in last 24 hours: ~50,000 (1% of catalog → 80% of traffic)
  Products accessed in last 7 days:   ~200,000 (4% of catalog → 95% of traffic)
  
  This is the "80/20 rule" of caching: 20% of data receives 80% of reads
```

**Step 2: Estimate hit ratio at different cache sizes:**
```
Model with Zipf distribution (real-world access patterns are Zipf-distributed):

At cache_size = 10K items (20% of hot working set):   hit ratio ≈ 70%
At cache_size = 50K items (= hot working set):        hit ratio ≈ 90%
At cache_size = 200K items (= 7-day working set):     hit ratio ≈ 95%
At cache_size = 1M items:                              hit ratio ≈ 97%
At cache_size = 5M items (full catalog):               hit ratio ≈ 99%

Hit ratio has diminishing returns — going from 50K to 5M (100× more memory)
gains only 9% additional hit ratio
```

**Step 3: Estimate per-item memory:**
```
Product data cached:
  Product struct:  500 bytes (JSON serialized: id, name, price, description, tags)
  Redis overhead:  100 bytes (key, TTL, metadata, hash entry)
  Total per item:  ~600 bytes

Cache size calculation:
  50K items × 600 bytes = 30 MB        (70% hit ratio — too small)
  200K items × 600 bytes = 120 MB      (90% hit ratio — good)
  1M items × 600 bytes = 600 MB        (95% hit ratio)
  5M items × 600 bytes = 3 GB          (97%+ hit ratio — full catalog)
```

**Step 4: Calculate cost benefit:**
```
DB read cost (RDS):       $0.0002 per query
Cache read cost (Redis):  $0.000002 per operation (100× cheaper)
DB reads per day at 10M QPS, 10% miss:  1M reads/sec × 86,400 = 86.4B reads/day
  → Very expensive without cache

Break-even analysis:
  Cache 200K items (120 MB Redis memory): ~$0.01/GB/hr = ~$0.0012/hr = ~$10/mo
  Saves: 90% of DB reads → massive DB cost reduction

At 10K QPS and 90% hit ratio:
  DB reads without cache: 10,000/sec
  DB reads with cache:     1,000/sec (10% miss)
  DB cost saved: 9,000 queries/sec × $0.0002 = $1.80/sec = $4,665/day
  Redis cost: ~$100/month
  ROI: 46× return on cache investment
```

**Step 5: Operational headroom:**
```
Never fill cache to 100%:
  80% target utilization (leave 20% headroom)
  
  Target: 90% hit ratio requires 200K items
  With 20% headroom: provision for 250K items
  Memory: 250K × 600 bytes / 0.8 = 187 MB
  
  Redis instance: t3.medium (3.68 GB RAM) → supports 10× growth
  Redis Cluster: Start with single node, shard when > 50% RAM used
```

---

## Quick Reference

### Caching Patterns
| Pattern | Write Path | Read Path | Best For |
|---|---|---|---|
| Cache-Aside | DB only, invalidate cache | Check cache, fallback to DB | Read-heavy, general use |
| Write-Through | Cache + DB (sync) | Cache (always warm) | Read-after-write consistency |
| Write-Back | Cache only, async DB flush | Cache (always warm) | Write-heavy, loss-tolerant |
| Write-Around | DB only (bypass cache) | Check cache, fallback to DB | Large one-time writes |

### Eviction Policies
| Policy | Evicts | Use When |
|---|---|---|
| LRU | Least recently used | General-purpose |
| LFU | Least frequently used | Stable popularity patterns |
| TTL | Expired entries | Time-sensitive data |
| Random | Random entry | Very large caches |

### Redis Data Structures
| Structure | Command Examples | Best For |
|---|---|---|
| String | SET, GET, INCR | Sessions, counters, serialized objects |
| Hash | HSET, HGET | User objects, partial updates |
| List | LPUSH, RPOP | Queues, activity feeds |
| Set | SADD, SMEMBERS, SINTER | Unique sets, tags, access control |
| Sorted Set | ZADD, ZRANGE | Leaderboards, rate limiting, priority queues |
| Stream | XADD, XREAD | Event logs, messaging |

### Cache Sizing Rules
| Target Hit Ratio | Cache Size Guideline |
|---|---|
| 70% | ~10–20% of working set |
| 90% | ~50–100% of hot working set (24h) |
| 95% | ~100% of warm working set (7-day) |
| 99% | Full dataset (expensive, use sparingly) |

### What NOT to Cache
- Highly dynamic data (changing > 1/sec)
- Sensitive data without per-user keys
- Rarely accessed data (< twice total)
- Data too large for RAM tier
- Write-heavy, rarely-read data
