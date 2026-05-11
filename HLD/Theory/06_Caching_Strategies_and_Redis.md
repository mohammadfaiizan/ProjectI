# Caching Strategies and Redis

## Table of Contents
1. [Why Caching?](#why-caching)
2. [Cache Hit Ratio](#cache-hit-ratio)
3. [Cache Placement](#cache-placement)
4. [Cache-Aside (Lazy Loading) Pattern](#cache-aside-lazy-loading-pattern)
5. [Write-Through Cache](#write-through-cache)
6. [Write-Back (Write-Behind) Cache](#write-back-write-behind-cache)
7. [Write-Around Cache](#write-around-cache)
8. [Cache Eviction Policies](#cache-eviction-policies)
9. [Cache Invalidation Strategies](#cache-invalidation-strategies)
10. [Cache Stampede / Thundering Herd](#cache-stampede--thundering-herd)
11. [Distributed Cache Design](#distributed-cache-design)
12. [Cache Consistency](#cache-consistency)
13. [Multi-Level Caching Architecture](#multi-level-caching-architecture)
14. [Redis Architecture Deep Dive](#redis-architecture-deep-dive)
15. [Redis Persistence](#redis-persistence)
16. [Redis Replication and Cluster](#redis-replication-and-cluster)
17. [Redis Sentinel vs Redis Cluster](#redis-sentinel-vs-redis-cluster)
18. [Common Redis Patterns](#common-redis-patterns)
19. [Quick Reference](#quick-reference)

---

## Why Caching?

Caching stores the result of expensive computations or slow I/O operations so that subsequent requests can be served faster. It is one of the most impactful optimizations in distributed systems.

### Latency Reduction

| Storage Layer | Typical Latency |
|---|---|
| CPU L1 Cache | ~0.5 ns |
| CPU L2 Cache | ~7 ns |
| RAM | ~100 ns |
| Redis (local network) | ~0.5 ms |
| SSD (local) | ~0.1 ms |
| HDD (local) | ~10 ms |
| Database query (simple) | 1–50 ms |
| Database query (complex join) | 50–500 ms |
| Cross-datacenter round trip | 50–150 ms |

A cache hit serving data from Redis at 0.5ms vs a database query at 50ms is a 100x latency improvement for that request.

### Throughput Increase

A typical PostgreSQL instance handles ~5,000–20,000 simple queries/second. Redis handles ~100,000–1,000,000 operations/second. Caching frequently-read data can multiply effective read throughput by 10–100x.

### Cost Reduction

- Fewer database queries -> smaller RDS/Cloud SQL instance needed.
- Fewer origin requests -> lower CDN origin bandwidth costs.
- Example: If 80% of reads are cache hits, your database receives only 20% of the original traffic, potentially allowing you to downsize by 80%.

### When NOT to Cache

- Data that changes on every read (real-time stock prices, live sensor data).
- Highly personalized data that differs per user (unless per-user caching is acceptable).
- Data with strict consistency requirements where stale reads cause business problems.
- Small datasets that fit in DB memory (the DB already caches them in its buffer pool).

---

## Cache Hit Ratio

The cache hit ratio is the percentage of requests served from cache vs total requests.

```
Hit Ratio = Cache Hits / (Cache Hits + Cache Misses)
```

### Impact on Performance

```
Effective latency = (hit_ratio * cache_latency) + (miss_ratio * (cache_latency + db_latency))

Example:
  cache_latency = 1ms, db_latency = 50ms
  
  Hit ratio 50%: 0.5*1 + 0.5*(1+50) = 0.5 + 25.5 = 26ms
  Hit ratio 80%: 0.8*1 + 0.2*(1+50) = 0.8 + 10.2 = 11ms
  Hit ratio 95%: 0.95*1 + 0.05*(1+50) = 0.95 + 2.55 = 3.5ms
  Hit ratio 99%: 0.99*1 + 0.01*(1+50) = 0.99 + 0.51 = 1.5ms
```

The 95th percentile hit ratio inflection point: going from 80% to 95% reduces effective latency by ~3x. Going from 95% to 99% reduces it by another 2x. This is why optimizing cache hit ratio is high-leverage work.

### Factors Affecting Hit Ratio

1. **Cache size** — Larger cache holds more unique keys; more hits.
2. **TTL policy** — Too-short TTL forces frequent re-population.
3. **Data access patterns** — Zipfian distribution (few hot keys) is cache-friendly; uniform access is not.
4. **Key design** — Granular keys waste cache space; coarse keys cause too many invalidations.
5. **Eviction policy** — LRU is usually optimal for web workloads.

---

## Cache Placement

### Client-Side Cache

Data cached in the browser (HTTP Cache) or mobile app.

```
Browser -> checks local cache
  HIT:  serve from local cache (0 network latency)
  MISS: request to server, cache response
```

- HTTP Cache headers control browser caching: `Cache-Control: max-age=3600`
- Service Workers can provide sophisticated client-side caching logic.
- Limitations: cache is per-client, can't be invalidated server-side (only via TTL or versioned URLs).

### CDN Cache

Geographically distributed caches at the edge, between the client and origin server.

```
Client -> [CDN PoP: cache hit] -> return cached response (no origin contact)
Client -> [CDN PoP: cache miss] -> [Origin Server] -> cache response at CDN
```

- Best for: static assets (JS, CSS, images, videos), public API responses.
- Reduces latency for global users and offloads origin traffic.

### Reverse Proxy Cache

Nginx, Varnish, or Squid caches entire HTTP responses before they reach the application.

```nginx
proxy_cache_path /data/nginx/cache levels=1:2 keys_zone=my_cache:10m max_size=10g inactive=60m;

location /api/ {
    proxy_cache my_cache;
    proxy_cache_valid 200 10m;
    proxy_cache_key "$host$request_uri";
    proxy_pass http://backend;
}
```

### Application Cache (In-Process)

Cache within the application's memory (JVM heap, Python dict, Go map).

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_user_profile(user_id: int) -> dict:
    return database.query("SELECT * FROM users WHERE id = ?", user_id)
```

- Zero network latency (in-process).
- Not shared between instances — each instance has its own cache.
- Lost on restart, not invalidatable from outside.
- Best for: immutable reference data (country codes, product catalog), computation results.

### Shared Cache (Out-of-Process)

Redis or Memcached accessed over the network. Shared by all application instances.

```
App Instance 1  \
App Instance 2  -> [Redis] -> [Database]
App Instance 3  /
```

- Small network overhead (~0.5ms), but shared and persistent.
- Cache can be invalidated by any instance.
- Survives application restarts.

### Database Query Cache

Some databases have built-in query caches (MySQL Query Cache — now removed). More commonly, use the DB's buffer pool (InnoDB buffer pool caches hot pages in RAM).

---

## Cache-Aside (Lazy Loading) Pattern

The most common caching pattern. The application is responsible for loading data into cache.

### Flow

```
READ:
  1. Application checks cache for key K
  2. HIT: return cached value
  3. MISS:
     a. Application queries database for K
     b. Application stores result in cache with TTL
     c. Application returns result

WRITE:
  1. Application writes to database
  2. Application DELETES (not updates) cache key K
     (or lets TTL expire it naturally)
```

```python
def get_user(user_id: str) -> dict:
    # Step 1: Check cache
    cached = redis.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)  # cache HIT

    # Step 2: Cache MISS — query database
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    
    # Step 3: Populate cache
    redis.setex(f"user:{user_id}", 3600, json.dumps(user))  # TTL: 1 hour
    
    return user

def update_user(user_id: str, data: dict):
    # Update database
    db.execute("UPDATE users SET ... WHERE id = %s", user_id)
    
    # Invalidate cache (don't update — avoid race conditions)
    redis.delete(f"user:{user_id}")
```

### Pros
- Only caches data that is actually read (no wasted memory).
- Resilient to cache failures — data is still in the DB.
- Cache can be a different data model than the DB (denormalized for reads).

### Cons
- Cache miss always causes a database read (cold start problem).
- Race condition: two threads may simultaneously miss cache and both query DB.
- Potential for stale data if invalidation is missed.

### Cold Start Problem

When the cache is empty (after restart, new deployment, cache eviction), all requests hit the database simultaneously.

**Mitigation:**
- **Cache warming:** Pre-populate cache with hot data at startup.
- **Gradual rollout:** Bring new instances up gradually so cache fills before full traffic.
- **Mutex/lock:** Only one thread populates the cache; others wait (see Cache Stampede section).

---

## Write-Through Cache

The cache is always written at the same time as the database. On write, update both the cache and the database before acknowledging success.

### Flow

```
WRITE:
  1. Application writes to cache
  2. Cache synchronously writes to database
  3. Acknowledge success to application

READ:
  1. Application checks cache
  2. HIT: return cached value (always fresh)
  3. MISS: read from DB, populate cache
```

```python
def update_user(user_id: str, data: dict):
    # Write to cache AND database atomically
    redis.setex(f"user:{user_id}", 3600, json.dumps(data))
    db.execute("UPDATE users SET ... WHERE id = %s", user_id)
    # If DB write fails, compensate by deleting cache
```

### Pros
- Cache is always consistent with the database.
- No stale reads — cache always has the latest data.
- Read performance is excellent (always cache-first).

### Cons
- Write latency increases (must write to both cache and DB).
- Writes data that may never be read (cache pollution for write-heavy, read-rare data).
- Cache and DB must succeed together — requires compensation on partial failure.

### Consistency Guarantee

Write-through provides read-your-writes consistency: after a write, any subsequent read will see the new value (from cache).

---

## Write-Back (Write-Behind) Cache

The application writes to the cache only. The cache asynchronously writes to the database in the background.

### Flow

```
WRITE:
  1. Application writes to cache -> immediately acknowledge success
  2. Cache asynchronously flushes to database (batched, after delay)

READ:
  1. Application reads from cache
  2. Cache data may be newer than DB
```

```
Timeline:
t=0: Write to cache (acknowledged immediately)
t=5s: Batch flush: write 100 pending changes to DB
      (if cache fails between t=0 and t=5, those writes are LOST)
```

### Pros
- Very low write latency (only writes to cache).
- Write batching reduces DB write operations (good for write-heavy workloads).
- Can coalesce multiple updates to the same key into one DB write.

### Cons
- **Data loss risk:** Writes in cache not yet flushed to DB are lost if cache crashes.
- Complex implementation (need a write queue, flush logic, error handling).
- Database may be temporarily inconsistent with cache.

### When to Use Write-Back

- Analytics counters (exact count can lag by seconds).
- Session data (losing a session on cache failure is acceptable).
- IoT telemetry where occasional data loss is tolerable.
- Shopping cart before checkout (within a session).

**NOT suitable for:** Financial transactions, inventory updates, any data where loss is unacceptable.

---

## Write-Around Cache

Writes go directly to the database, bypassing the cache. The cache is only populated on reads (like cache-aside).

### When to Use

- Data is written once and read rarely or never.
- Write-intensive operations that would pollute the cache.
- Uploading files/media: write to S3, read from CDN (don't cache at app level).
- Log entries: write directly to DB/Elasticsearch, never needs caching.

---

## Cache Eviction Policies

When the cache is full, which item should be removed to make room for new data?

### LRU (Least Recently Used)

Evicts the item that was accessed least recently. Maintains a linked list + hash map for O(1) access and O(1) eviction.

```
Cache state (access order, newest to oldest):
[D, C, A, B]

Access E (miss, cache full):
-> Evict B (least recently used)
-> Cache: [E, D, C, A]
```

**Best for:** Web caches, API response caches — recently accessed data is likely to be accessed again (temporal locality).

### LFU (Least Frequently Used)

Evicts the item accessed the fewest times. Requires tracking access counts per key.

```
Key: A, count: 50
Key: B, count: 3   <- evict this
Key: C, count: 12
```

**Best for:** Long-running caches where access frequency is a better predictor than recency. Good for CDN caches.
**Cons:** New items start with count=1 and are immediately eviction candidates. Requires decay/aging to handle changing popularity.

### FIFO (First In, First Out)

Evicts the oldest inserted item, regardless of access frequency.

**Pros:** Simple, predictable.
**Cons:** Evicts hot items that happen to be old. Rarely optimal.

### Random Replacement

Evicts a randomly selected item.

**Pros:** Simple, no bookkeeping overhead. Surprisingly competitive with LRU in some workloads.
**Cons:** May evict hot items.

### MRU (Most Recently Used)

Evicts the most recently used item. Used in scan-resistant caches where a sequential scan would pollute an LRU cache.

**Example:** Reading a large file sequentially. LRU would evict older hot data because every file block is "recently used."

### Redis Eviction Policies

Redis supports 8 eviction policies (set via `maxmemory-policy`):

| Policy | Description | Best For |
|---|---|---|
| `noeviction` | Return error when memory full | Critical data (default) |
| `allkeys-lru` | LRU across all keys | General caching |
| `volatile-lru` | LRU among keys with TTL | Mix of persistent + cached data |
| `allkeys-lfu` | LFU across all keys | Varying access frequency |
| `volatile-lfu` | LFU among keys with TTL | Same as above but selective |
| `allkeys-random` | Random eviction | Uniform access distribution |
| `volatile-random` | Random among TTL keys | Similar to above |
| `volatile-ttl` | Evict key with nearest expiry | Prefer removing soon-expiring items |

---

## Cache Invalidation Strategies

"There are only two hard things in Computer Science: cache invalidation and naming things." — Phil Karlton

### TTL-Based Expiry

Set a time-to-live on each cached item. After TTL expires, the item is automatically removed.

```python
redis.setex("user:123", ttl=3600, value=json.dumps(user))  # expires in 1 hour
```

**Pros:** Simple, automatic, requires no write-path changes.
**Cons:** Data is stale up to TTL duration. Setting TTL too short increases cache miss rate; too long increases staleness.

**TTL guidelines by data type:**

| Data Type | Suggested TTL |
|---|---|
| User profile | 5–60 minutes |
| Product catalog | 1–24 hours |
| Exchange rates | 60 seconds |
| Session data | 30 minutes (sliding) |
| Static reference data | 24 hours |
| Rendered HTML pages | 1–5 minutes |

### Event-Driven Invalidation

Invalidate the cache entry immediately when the underlying data changes.

```python
def update_product(product_id: str, data: dict):
    db.update(product_id, data)
    redis.delete(f"product:{product_id}")        # exact key
    redis.delete(f"product_list:category:*")     # pattern (use with care)
```

**Pros:** Cache is always fresh after a write.
**Cons:** Tightly couples write path to cache. Pattern-based deletes are O(N) in Redis (use SCAN, avoid in production at scale).

### Version Tags / Cache Versioning

Embed a version number in the cache key. Invalidation means incrementing the version.

```python
# Store version in Redis
user_version = redis.incr("user:123:version")  # now version=5

# Cache key includes version
cache_key = f"user:123:v{user_version}"
redis.setex(cache_key, 3600, json.dumps(user))

# Invalidation: just increment version
redis.incr("user:123:version")  # now version=6, old key becomes orphaned
```

**Pros:** Instant logical invalidation without deleting data.
**Cons:** Old versions remain in cache until evicted; wastes memory. Requires reading the version key on every request.

### Cache Invalidation via Message Bus

Publish invalidation events to a message bus (Kafka, Redis Pub/Sub). All application instances subscribe and delete their local cache entries.

```python
# On write:
db.update(user_id, data)
kafka.publish("cache-invalidation", {"key": f"user:{user_id}"})

# All instances consume:
@kafka.subscribe("cache-invalidation")
def on_invalidation(event):
    local_cache.delete(event["key"])
    redis.delete(event["key"])
```

**Best for:** Multi-level caches (in-process + Redis) where the in-process cache needs to be invalidated on other nodes.

---

## Cache Stampede / Thundering Herd

### The Problem

When a popular cached key expires, many concurrent requests simultaneously experience a cache miss and all try to populate the cache by querying the database. This creates a sudden spike in database load.

```
t=0:  key "popular_item" expires
t=0+: 500 concurrent requests all get cache miss
      500 simultaneous DB queries for the same data
      DB becomes overloaded
      All 500 requests finally populate the cache with the same data
```

### Solution 1: Mutex Lock

Only one request rebuilds the cache. Others wait for it to complete.

```python
import redis
import threading

def get_with_lock(key: str, compute_fn, ttl: int):
    value = redis.get(key)
    if value:
        return json.loads(value)
    
    lock_key = f"lock:{key}"
    acquired = redis.set(lock_key, "1", nx=True, ex=30)  # NX = only if not exists
    
    if acquired:
        try:
            value = compute_fn()  # query database
            redis.setex(key, ttl, json.dumps(value))
            return value
        finally:
            redis.delete(lock_key)
    else:
        # Wait and retry
        time.sleep(0.1)
        return get_with_lock(key, compute_fn, ttl)  # retry
```

**Cons:** All other requests are blocked waiting; if the lock holder crashes, the lock must expire (30s delay).

### Solution 2: Probabilistic Early Expiration (PER)

Before the TTL actually expires, randomly re-compute the cache value based on a probability that increases as expiry approaches.

```python
import random
import math

def get_with_per(key: str, compute_fn, ttl: int, beta: float = 1.0):
    cached = redis.get(key)
    remaining_ttl = redis.ttl(key)
    
    if cached:
        # Probability of early re-computation increases near expiry
        # XFetch algorithm: recompute if -beta * delta * log(random()) > remaining_ttl
        delta = time.time() - start_time  # time to compute fn (use stored value)
        if -beta * delta * math.log(random.random()) <= remaining_ttl:
            return json.loads(cached)
    
    # Recompute
    value = compute_fn()
    redis.setex(key, ttl, json.dumps(value))
    return value
```

**Pros:** No locks; stampede is avoided by spreading re-computations over time.

### Solution 3: Background Refresh

Serve the stale value immediately and asynchronously refresh the cache.

```python
def get_with_background_refresh(key: str, compute_fn, soft_ttl: int, hard_ttl: int):
    cached = redis.get(key)
    
    if not cached:
        # Complete cache miss — must block
        value = compute_fn()
        redis.setex(key, hard_ttl, json.dumps({"data": value, "computed_at": time.time()}))
        return value
    
    entry = json.loads(cached)
    age = time.time() - entry["computed_at"]
    
    if age > soft_ttl:
        # Soft TTL exceeded — refresh in background, return stale data now
        threading.Thread(target=lambda: refresh_cache(key, compute_fn, hard_ttl)).start()
    
    return entry["data"]  # return immediately (may be slightly stale)
```

**Pros:** Zero latency penalty; users never see cache miss latency.
**Cons:** Users may see stale data briefly; background tasks need error handling.

---

## Distributed Cache Design

### Consistent Hashing for Cache Nodes

When you have multiple Redis nodes, use consistent hashing to distribute keys.

```
Ring: Node1 at 0°, Node2 at 120°, Node3 at 240°

Key K -> hash(K) -> position on ring -> nearest node clockwise

Adding Node4: only keys between Node3 and Node4 need to move
```

Redis Cluster uses hash slots (see Redis Cluster section).

### Cache Replication

For read-heavy workloads, replicate cache data to multiple nodes.

```
Write: [Primary Cache] -> [Replica 1, Replica 2]
Read:  [Any Replica] (load-balanced)
```

Redis supports primary-replica replication natively.

### Cache Failover

If a cache node fails:
1. **Without replication:** Cache misses increase (thundering herd risk). Database handles extra load.
2. **With replication:** Promote a replica to primary. Automated by Redis Sentinel or Redis Cluster.

**Graceful degradation:** Application must handle cache failures:
```python
def get_user(user_id):
    try:
        cached = redis.get(f"user:{user_id}")
        if cached:
            return json.loads(cached)
    except redis.RedisError as e:
        log.warning(f"Cache unavailable: {e}")
        # Fall through to database
    
    return db.query_user(user_id)
```

---

## Cache Consistency

### Stale Reads

A stale read occurs when the cache contains outdated data that has been updated in the database.

**Causes:**
- Cache-aside with long TTL
- Failed cache invalidation
- Replication lag in cache cluster

**Mitigation:**
- Shorter TTL for frequently-updated data
- Event-driven invalidation for critical data
- Read from primary cache node for strong consistency reads

### Cache Poisoning

An attacker (or bug) writes malicious or corrupt data into the cache.

**Prevention:**
- Input validation before caching
- Cache data signing/checksums for sensitive data
- Restrict cache write access to trusted services only

### The Cache Invalidation Race Condition

```
Thread A: read user (miss), query DB, gets old value
Thread B: update user in DB, delete cache key
Thread A: write OLD value to cache  <- stale data!
```

**Fix: Cache-aside should DELETE (not write) on invalidation, and use short TTLs.**

More robust fix: use conditional writes with versioning.

```python
# Write only if version hasn't changed
lua_script = """
if redis.call('get', KEYS[1]..':version') == ARGV[1] then
    redis.call('set', KEYS[1], ARGV[2])
    return 1
else
    return 0
end
"""
```

---

## Multi-Level Caching Architecture

Inspired by CPU cache hierarchy (L1/L2/L3), multi-level caching places faster but smaller caches closer to the compute.

```
Request
  |
  v
[L1: In-process Cache]  <- Python dict / Guava Cache / Caffeine
  | miss (rare)
  v
[L2: Shared Redis Cache] <- Shared by all instances
  | miss (uncommon)
  v
[L3: CDN / Reverse Proxy Cache] <- For HTTP responses
  | miss
  v
[Database / Origin]
```

### In-Process L1 Cache Example (Python)

```python
from cachetools import TTLCache

class UserService:
    def __init__(self):
        self.local_cache = TTLCache(maxsize=1000, ttl=60)  # 1000 users, 60s TTL
    
    def get_user(self, user_id: str) -> dict:
        # L1: in-process
        if user_id in self.local_cache:
            return self.local_cache[user_id]
        
        # L2: Redis
        cached = redis.get(f"user:{user_id}")
        if cached:
            user = json.loads(cached)
            self.local_cache[user_id] = user  # populate L1
            return user
        
        # Database
        user = db.get_user(user_id)
        redis.setex(f"user:{user_id}", 3600, json.dumps(user))  # populate L2
        self.local_cache[user_id] = user  # populate L1
        return user
```

**Trade-off:** L1 cache is not shared — invalidation must be broadcast to all instances (via Redis Pub/Sub or message bus) or rely on short TTL.

---

## Redis Architecture Deep Dive

### Single-Threaded I/O Model

Redis processes commands in a single thread. This eliminates the complexity of thread synchronization and makes all operations inherently atomic.

```
Network I/O thread(s) -> Command Queue -> Single execution thread -> I/O thread(s)
```

- Why single-threaded? Context switching overhead exceeds the benefit for in-memory operations.
- Since Redis 6.0: I/O threading for network reads/writes, but command execution remains single-threaded.
- Result: 100,000+ simple operations/second on commodity hardware.

### Redis Data Structures

**1. String**

The simplest type. Stores strings, integers, or binary data (up to 512 MB).

```redis
SET user:123:name "Alice"
GET user:123:name                  # "Alice"
INCR page:views                    # atomic increment
INCRBY score:player1 10
SET session:abc123 "token" EX 1800  # with TTL
SETNX lock:resource "owner"        # SET if Not eXists (distributed lock)
```

**2. Hash**

A map of field-value pairs. Ideal for objects.

```redis
HSET user:123 name "Alice" age 30 email "alice@example.com"
HGET user:123 name                 # "Alice"
HMGET user:123 name email          # multiple fields
HGETALL user:123                   # all fields
HINCRBY user:123 age 1             # increment field
```

Memory-efficient for small hashes (Redis uses ziplist internally for < 128 fields).

**3. List**

An ordered collection of strings. Implemented as a doubly-linked list or quicklist.

```redis
RPUSH jobs "job1" "job2" "job3"    # append to right
LPUSH jobs "job0"                   # prepend to left
LPOP jobs                           # pop from left (queue behavior)
RPOP jobs                           # pop from right (stack behavior)
LRANGE jobs 0 -1                    # all elements
BLPOP jobs 5                        # blocking pop with 5s timeout (message queue)
```

Use case: Job queues, activity feeds, chat message history.

**4. Set**

Unordered collection of unique strings.

```redis
SADD online:users "user1" "user2" "user3"
SISMEMBER online:users "user1"     # check membership
SMEMBERS online:users              # all members
SCARD online:users                 # cardinality (count)
SINTER online:users premium:users  # intersection
SUNION set1 set2                   # union
SDIFF set1 set2                    # difference
```

Use case: Unique visitors, tags, friends lists, real-time presence.

**5. Sorted Set (ZSet)**

Set where each member has a floating-point score. Members are ordered by score.

```redis
ZADD leaderboard 1500 "alice" 2300 "bob" 900 "carol"
ZRANK leaderboard "alice"          # rank (0-indexed)
ZREVRANK leaderboard "bob"         # rank from highest (0 = highest)
ZSCORE leaderboard "alice"         # score
ZRANGE leaderboard 0 2             # top 3 (lowest score first)
ZREVRANGE leaderboard 0 2          # top 3 (highest score first)
ZRANGEBYSCORE leaderboard 1000 2000  # members in score range
ZINCRBY leaderboard 100 "alice"    # increment score
```

Use case: Leaderboards, priority queues, rate limiting (sliding window), timeline feeds.

**6. Stream**

Append-only log of messages with consumer groups. Similar to Kafka topics but within Redis.

```redis
XADD events * user_id 123 action "login" ip "1.2.3.4"
XLEN events
XREAD COUNT 10 STREAMS events 0    # read from beginning
XREADGROUP GROUP mygroup consumer1 COUNT 10 STREAMS events >  # consumer group
XACK events mygroup <message_id>   # acknowledge processing
```

Use case: Event sourcing, audit logs, inter-service messaging.

**7. Bitmap**

Bit-level operations on strings. Space-efficient boolean flags.

```redis
SETBIT user:logins:2024-01-15 user_id 1   # mark user logged in on date
GETBIT user:logins:2024-01-15 user_id     # check if logged in
BITCOUNT user:logins:2024-01-15            # count active users today
BITOP AND active_both week1_active week2_active  # users active both weeks
```

Use case: Feature flags, daily active user tracking, A/B testing segments.

**8. HyperLogLog**

Probabilistic data structure for counting unique elements. Uses ~12KB regardless of cardinality.

```redis
PFADD unique_visitors "user1" "user2" "user3"
PFCOUNT unique_visitors             # ~3 (±0.81% error)
PFMERGE combined visitors_day1 visitors_day2
```

Use case: Counting unique visitors, unique search queries, distinct IPs. Error rate ~0.81%.

---

## Redis Persistence

### No Persistence

```
save ""           # disable RDB
appendonly no     # disable AOF
```

Redis operates as a pure cache. All data is lost on restart. Acceptable for disposable caches.

### RDB (Redis Database) Snapshots

Periodically saves a point-in-time snapshot of the dataset to disk as a binary `.rdb` file.

```
# redis.conf
save 900 1        # save if at least 1 key changed in 900 seconds
save 300 10       # save if at least 10 keys changed in 300 seconds
save 60 10000     # save if at least 10000 keys changed in 60 seconds
dbfilename dump.rdb
dir /var/lib/redis/
```

**How it works:** Redis forks a child process. The child writes the snapshot while the parent continues serving requests (copy-on-write). No I/O blocking on the main thread.

**Pros:** Compact file, fast restarts, ideal for backups.
**Cons:** Data since last snapshot is lost on crash. Forking can be slow with large datasets (fork latency).

### AOF (Append-Only File)

Logs every write command to a file. On restart, Redis replays the log.

```
# redis.conf
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec  # flush to disk every second (compromise)
# appendfsync always  # flush every command (safest, slowest)
# appendfsync no      # OS decides (fastest, riskiest)
```

**AOF rewrite:** Over time, AOF grows large. Redis can rewrite it (BGREWRITEAOF) to produce a compact file that achieves the same state with fewer commands.

**Pros:** Up to 1-second data durability (with `everysec`). Human-readable log.
**Cons:** Larger files than RDB. Slower recovery (must replay all commands). More disk I/O.

### RDB + AOF Hybrid (Recommended for Production)

```
# redis.conf
appendonly yes
aof-use-rdb-preamble yes  # AOF file starts with RDB snapshot, then appends commands
```

**On restart:** Load the embedded RDB snapshot (fast), then replay only the commands since the snapshot (few). Best of both worlds.

---

## Redis Replication and Cluster

### Redis Replication (Primary-Replica)

```
Primary (reads + writes)
    |
    |-- Replica 1 (reads only)
    |-- Replica 2 (reads only)
    |-- Replica 3 (reads only)
```

```
# On replica:
replicaof 10.0.1.1 6379

# Replication is asynchronous by default
# For sync replication (at the cost of latency):
min-replicas-to-write 1
min-replicas-max-lag 10
```

**Replication lag:** Replicas are slightly behind the primary. For read replicas, this is usually acceptable (< 1ms on LAN).

### Redis Cluster

Redis Cluster shards data across multiple primary nodes using hash slots.

```
Total hash slots: 16384 (0 to 16383)
key -> CRC16(key) % 16384 -> hash slot -> node

Example with 3 nodes:
Node A: slots 0 - 5460
Node B: slots 5461 - 10922
Node C: slots 10923 - 16383
```

Each primary node can have replica nodes for failover.

```
Node A Primary  <--> Node A Replica
Node B Primary  <--> Node B Replica
Node C Primary  <--> Node C Replica
```

**Multi-key operations:** Keys on different slots cannot be used in the same command. Use hash tags to force keys to the same slot:

```redis
# {} forces keys to same slot
MSET {user:123}:name "Alice" {user:123}:email "alice@example.com"
```

**Cluster commands:**
```redis
CLUSTER INFO
CLUSTER NODES
CLUSTER SLOTS
```

---

## Redis Sentinel vs Redis Cluster

| Feature | Redis Sentinel | Redis Cluster |
|---|---|---|
| Purpose | HA for a single Redis instance | Horizontal sharding + HA |
| Sharding | No (single dataset) | Yes (16384 hash slots) |
| Min nodes | 3 Sentinel + 1 Primary + 1 Replica | 3 Primaries + 3 Replicas (minimum) |
| Automatic failover | Yes | Yes |
| Max dataset size | Limited by single node RAM | Sum of all primary node RAM |
| Multi-key ops | All keys on one node | Keys must be on same slot |
| Client complexity | Simple (connect to Sentinel) | Cluster-aware client needed |
| Use case | Smaller datasets, simple HA | Large datasets, horizontal scale |

**Redis Sentinel Architecture:**
```
Sentinel 1
Sentinel 2  ->  quorum decision -> promote replica to primary
Sentinel 3

Clients query Sentinel for current primary address.
```

---

## Common Redis Patterns

### Rate Limiting

Using sorted sets for sliding window (see Load Balancing file for full implementation):

```redis
-- Lua script (atomic)
ZREMRANGEBYSCORE ratelimit:user123 0 (NOW - WINDOW)
ZADD ratelimit:user123 NOW NOW
ZCARD ratelimit:user123
EXPIRE ratelimit:user123 WINDOW
```

### Session Store

```python
def create_session(user_id: str) -> str:
    session_id = secrets.token_urlsafe(32)
    session_data = {"user_id": user_id, "created_at": time.time()}
    redis.setex(f"session:{session_id}", 1800, json.dumps(session_data))
    return session_id

def get_session(session_id: str) -> dict:
    data = redis.get(f"session:{session_id}")
    if data:
        redis.expire(f"session:{session_id}", 1800)  # sliding expiry
        return json.loads(data)
    return None
```

### Leaderboard (Sorted Set)

```python
# Update score
def update_score(player_id: str, score_delta: float):
    redis.zincrby("game:leaderboard", score_delta, player_id)

# Get top 10
def get_top_players(n: int = 10) -> list:
    return redis.zrevrange("game:leaderboard", 0, n-1, withscores=True)

# Get player rank
def get_rank(player_id: str) -> int:
    return redis.zrevrank("game:leaderboard", player_id)
```

### Pub/Sub (Real-Time Notifications)

```python
# Publisher
def publish_notification(user_id: str, message: dict):
    redis.publish(f"notifications:{user_id}", json.dumps(message))

# Subscriber (in a separate thread/process)
pubsub = redis.pubsub()
pubsub.subscribe("notifications:user123")
for message in pubsub.listen():
    if message["type"] == "message":
        handle_notification(json.loads(message["data"]))
```

**Note:** Redis Pub/Sub does not persist messages. Use Redis Streams for durability.

### Distributed Lock (Redlock Algorithm)

```python
import uuid

def acquire_lock(resource: str, ttl_ms: int = 30000) -> str | None:
    lock_id = str(uuid.uuid4())
    acquired = redis.set(
        f"lock:{resource}",
        lock_id,
        nx=True,          # only set if key does NOT exist
        px=ttl_ms         # expire after ttl_ms milliseconds
    )
    return lock_id if acquired else None

def release_lock(resource: str, lock_id: str) -> bool:
    # Lua script ensures atomicity: only delete if we own the lock
    lua_script = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
        return redis.call('del', KEYS[1])
    else
        return 0
    end
    """
    result = redis.eval(lua_script, 1, f"lock:{resource}", lock_id)
    return result == 1

# Usage
lock_id = acquire_lock("payment:order123", ttl_ms=10000)
if lock_id:
    try:
        process_payment()
    finally:
        release_lock("payment:order123", lock_id)
```

**Redlock (Multi-node distributed lock):**
Acquire the lock on N/2+1 Redis nodes (majority). If acquired on majority within TTL/2 time, the lock is considered acquired. This handles single Redis node failures.

---

## Quick Reference

### Cache Pattern Decision Tree

```
Q: Is the data read-heavy or write-heavy?
  READ-HEAVY:
    Q: Can you tolerate slightly stale data?
      YES -> Cache-Aside (lazy loading) with appropriate TTL
      NO  -> Write-through or read from primary DB
  WRITE-HEAVY:
    Q: Can you tolerate eventual DB persistence?
      YES -> Write-back (write-behind) cache
      NO  -> Write-through or write-around

Q: How is data updated?
  UPDATED RARELY    -> Long TTL, TTL-based invalidation
  UPDATED OFTEN     -> Short TTL or event-driven invalidation
  WRITTEN ONCE      -> Write-around (don't cache writes at all)

Q: Is data personalized per user?
  YES -> Per-user cache key (user:{id}:...), shorter TTL
  NO  -> Shared cache key, can use longer TTL
```

### Redis Data Structure Use Cases

| Data Structure | Use Case | Key Operations |
|---|---|---|
| String | Sessions, counters, locks, simple KV | GET, SET, INCR, SETNX |
| Hash | User objects, config, product details | HGET, HSET, HGETALL |
| List | Job queues, activity feeds, chat history | LPUSH, RPOP, BLPOP |
| Set | Tags, friends, unique visitors, permissions | SADD, SISMEMBER, SINTER |
| Sorted Set | Leaderboards, priority queues, rate limiting | ZADD, ZRANGE, ZREVRANK |
| Stream | Event logs, audit trails, message bus | XADD, XREAD, XREADGROUP |
| Bitmap | Daily active users, feature flags | SETBIT, BITCOUNT, BITOP |
| HyperLogLog | Unique visitor count, distinct items | PFADD, PFCOUNT |

### Cache Strategy Comparison

| Strategy | Write Latency | Read Latency | Consistency | Data Loss Risk | Use Case |
|---|---|---|---|---|---|
| Cache-Aside | DB only | Cache or DB | Eventual (TTL) | None | Most read-heavy apps |
| Write-Through | Cache + DB | Cache | Strong | None | Read-heavy, consistency required |
| Write-Back | Cache only (fast) | Cache | Eventual | Yes (unflushed writes) | Write-heavy, loss tolerable |
| Write-Around | DB only | DB (first miss) | Strong | None | Write-once, rarely-read data |

### Eviction Policy Quick Selection

| Scenario | Recommended Policy |
|---|---|
| Pure cache (all data re-fetchable) | `allkeys-lru` |
| Mix of persistent + cached data | `volatile-lru` |
| All keys accessed at similar frequency | `allkeys-random` |
| Prefer removing soon-expiring data | `volatile-ttl` |
| Must never evict any data | `noeviction` (+ monitor memory!) |
| Highly skewed popularity | `allkeys-lfu` |
