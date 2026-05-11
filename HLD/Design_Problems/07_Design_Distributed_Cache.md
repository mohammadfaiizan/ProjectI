# Design a Distributed Cache (like Memcached/Redis Cluster) — High-Level Design

---

## 1. Problem Statement & Clarifying Questions

**Problem Statement:**
Design a distributed in-memory caching system that provides low-latency key-value storage across multiple nodes. The system must support horizontal scaling, fault tolerance, and consistent hashing to minimize cache misses during topology changes.

**Clarifying Questions:**
- What is the expected read/write QPS? (10K QPS per node, horizontal scaling)
- What eviction policies are needed? (LRU as primary)
- Should we support TTL (time-to-live) on keys?
- Do we need persistence (survive restarts)?
- What consistency model is acceptable? (eventual is fine for cache)
- Should we support replication for high availability?
- What data types do we need? (simple key-value, or Redis-like data structures)
- What is the maximum value size?

**Assumptions:**
- Key-value store with string keys, arbitrary binary values
- LRU eviction when memory is full
- TTL support on all keys
- Replication factor of 3 for high availability
- Read/write quorum configurable (default W=2, R=2, N=3)
- Consistent hashing for partitioning

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
1. **GET key:** Retrieve value by key
2. **SET key value [TTL]:** Store key-value pair with optional expiry
3. **DELETE key:** Remove a key
4. **EXISTS key:** Check if key exists
5. **EXPIRE key seconds:** Set TTL on existing key
6. **Cluster management:** Add/remove nodes with minimal data movement

### Non-Functional Requirements
1. **Latency:** P99 read < 1ms, write < 2ms
2. **Availability:** 99.99% uptime
3. **Consistency:** Eventual consistency within cluster
4. **Throughput:** 1M+ ops/second across cluster
5. **Scalability:** Linear horizontal scaling
6. **Memory efficiency:** Minimize overhead per key
7. **Fault tolerance:** Survive node failures without data loss (with replication)

---

## 3. Capacity Estimation

### Per-Node
- Memory per cache node: 64GB RAM
- Overhead per key: ~100 bytes (metadata, pointers)
- Average value size: 1KB
- Keys per node: 64GB / (100B + 1KB) ≈ 58M keys
- QPS per node: 10,000 read/write operations/second

### Cluster Sizing (Example: 100-node cluster)
- Total cache memory: 100 nodes * 64GB = 6.4TB
- Total keys: 100 * 58M = 5.8B keys
- Total cluster QPS: 100 * 10K = 1M QPS
- Network: Each request ~1KB → 1M QPS * 1KB = 1 Gbps network load

### Consistent Hashing
- Virtual nodes per physical node: 150 vnodes
- Total ring positions: 2^32 ≈ 4B positions
- With 100 nodes: 100 * 150 = 15,000 virtual positions
- Load distribution variance: ~5% with 150 vnodes

---

## 4. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENT APPLICATIONS                              │
│           App Server 1     App Server 2     App Server 3                │
└──────────────────┬─────────────────┬──────────────────┬─────────────────┘
                   │                 │                  │
                   ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    CACHE CLIENT LIBRARY (Embedded in App)                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Consistent Hash Ring → Route to correct node(s)                │   │
│  │  Connection Pool → TCP connections to each cache node            │   │
│  │  Read/Write Quorum → W=2 of 3 replicas must ACK                 │   │
│  │  Retry Logic → Failover to next node on timeout                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
                   │ Routes to responsible nodes
                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│                      CACHE CLUSTER                                     │
│                                                                        │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│   │  Node A      │    │  Node B      │    │  Node C      │           │
│   │  (Primary)   │    │  (Primary)   │    │  (Primary)   │           │
│   │  Vnodes:     │    │  Vnodes:     │    │  Vnodes:     │           │
│   │  [0-120]     │    │  [121-240]   │    │  [241-360]   │           │
│   │              │    │              │    │              │           │
│   │  LRU Cache   │    │  LRU Cache   │    │  LRU Cache   │           │
│   │  TTL Heap    │    │  TTL Heap    │    │  TTL Heap    │           │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘           │
│          │ replicates        │ replicates         │                    │
│          ▼                   ▼                    ▼                    │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│   │  Node D      │    │  Node E      │    │  Node F      │           │
│   │  (Replica)   │    │  (Replica)   │    │  (Replica)   │           │
│   └──────────────┘    └──────────────┘    └──────────────┘           │
│                                                                        │
└───────────────────────────────────────────────────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│               CLUSTER COORDINATOR / GOSSIP PROTOCOL                    │
│     Node membership, failure detection, ring updates, rebalancing      │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Consistent Hashing Ring

**Problem with Naive Hashing:**
- `hash(key) % N` — adding/removing node rehashes ~all keys
- Cache invalidation storm when scaling

**Consistent Hashing Solution:**
- Place both nodes and keys on a circular ring (0 to 2^32)
- Key is assigned to the next node clockwise on the ring
- Adding a node: only keys between previous node and new node move
- Removing a node: only that node's keys move to next node
- On average, only `K/N` keys move when adding/removing a node

**Virtual Nodes (Vnodes):**
- Each physical node maps to 150 positions on the ring
- Prevents hot spots from uneven distribution
- When node fails, its load spreads across all other nodes (not just neighbors)
- Trade-off: more memory for ring metadata

**Hashing:**
```python
ring_position = int(MD5(f"{node_id}:{vnode_index}").hexdigest(), 16) % (2**32)
key_position  = int(MD5(key).hexdigest(), 16) % (2**32)
# Find first ring_position >= key_position (with wrap-around)
```

### 5.2 LRU Eviction — O(1) Implementation

**Data Structure:** Doubly Linked List + HashMap

```
HashMap: key → node_pointer (O(1) lookup)
Doubly Linked List: MRU ←→ ... ←→ LRU
  - On GET: move accessed node to head (O(1))
  - On SET new: add to head, if over capacity evict tail (O(1))
  - On SET existing: update value, move to head (O(1))
```

**Why not other data structures?**
- Array: O(n) for move-to-front
- Single linked list: O(n) for node removal without tail pointer
- Doubly linked list + hashmap: O(1) all operations

**TTL Management:**
- Min-heap ordered by expiry timestamp
- Background thread checks heap every 100ms
- Lazy expiry: also check TTL on every GET

### 5.3 Replication & Quorum

**Replication Factor N=3:**
- Every key is written to 3 nodes (determined by consistent hash ring)
- Replication is synchronous for W=2, async for the 3rd replica

**Quorum Reads/Writes (Dynamo-style):**
- Write quorum W=2: write confirmed when 2/3 replicas ACK
- Read quorum R=2: read from 2/3 replicas, return latest version
- W + R > N ensures we always read at least one node with latest write
- With W=2, R=2, N=3: W+R=4 > 3 ✓

**Version Vectors (Conflict Resolution):**
- Each write tagged with vector clock
- On read, compare versions — return highest
- On conflict: last-write-wins (by timestamp) or application-defined merge

### 5.4 Cache Patterns

**Cache-Aside (Lazy Loading):**
```
1. App checks cache for key
2. Cache MISS → App queries DB → App stores result in cache → return
3. Cache HIT → return directly
```
- Pros: Only requested data cached, resilient to cache failure
- Cons: Cache miss = 3 operations (read miss, DB query, write cache)

**Write-Through:**
```
1. App writes to cache
2. Cache synchronously writes to DB
3. Always consistent — no stale data
```
- Pros: Cache always consistent with DB
- Cons: Write latency = cache + DB latency, cold cache on startup

**Write-Back (Write-Behind):**
```
1. App writes to cache only
2. Cache asynchronously batches writes to DB
```
- Pros: Lowest write latency
- Cons: Risk of data loss if cache crashes before flush

### 5.5 Cache Stampede Prevention

**Problem:** Cache key expires → 1000 requests simultaneously hit DB (thundering herd)

**Solutions:**

**1. Mutex Lock:**
- First thread to detect miss acquires lock, fetches from DB, populates cache
- Other threads wait for lock, then read from cache
- Risk: Lock contention, single point of delay

**2. Probabilistic Early Expiration (PER):**
```python
def should_refresh(expiry_time, beta=1.0):
    current_time = time.time()
    time_to_expiry = expiry_time - current_time
    # Refresh probabilistically before expiry
    return current_time - beta * math.log(random.random()) > expiry_time
```
- Stochastically refreshes before expiry
- No locks, no thundering herd
- Slight over-fetching but distributed load

**3. Stale-While-Revalidate:**
- Serve stale data while background thread refreshes
- Key has two TTLs: soft (when to refresh) and hard (when to evict)

### 5.6 Hot Key Problem

**Problem:** Single key receives disproportionate traffic (celebrity, viral content)
- Single node overwhelmed while others are idle
- 100K requests/second to one key, but node handles 10K

**Solutions:**

**1. Key Replication:**
- Store same key on multiple nodes: `key:0`, `key:1`, ... `key:9`
- Read: `hash(key + random(0,9))` → distributes read load
- Write: write to all replicas (expensive but rare)

**2. Local In-Process Cache:**
- App server caches hot keys locally in JVM heap / Python dict
- TTL = 1-5 seconds (short to avoid staleness)
- Absorbs 99%+ of hot key traffic before hitting distributed cache

**3. Cache Sharding by User:**
- Instead of one key, shard as `key:{user_id % 100}`
- Distribute individual user requests across nodes

### 5.7 Node Failure and Recovery

**Failure Detection:**
- Gossip protocol: each node sends heartbeat every 500ms
- If no heartbeat for 3 seconds → node suspected
- After 30 seconds without recovery → node removed from ring
- Ring updated: keys remapped to next node clockwise

**Recovery (Hinted Handoff):**
- During failure, writes intended for failed node saved as "hints" on recipient
- When failed node recovers, hints are replayed to it
- Ensures eventual consistency

**Anti-Entropy (Read Repair):**
- On a read that returns different values from replicas → fix stale replica
- Background Merkle tree comparison to find divergence

---

## 6. Database Design (Cache Node Internal)

### In-Memory Data Structures

```
CacheNode:
├── hash_map: Dict[str, CacheEntry]     # key → entry (O(1) lookup)
├── lru_list: DoublyLinkedList          # ordered by recency
├── ttl_heap: MinHeap[(expiry, key)]    # min-heap for TTL cleanup
├── memory_used: int                    # track memory consumption
└── max_memory: int                     # eviction threshold

CacheEntry:
├── key: str
├── value: bytes
├── size: int                           # value size in bytes
├── expiry: float | None                # Unix timestamp
├── version: int                        # vector clock
└── node: DLLNode                      # pointer into LRU list
```

### Cluster Metadata (Stored in ZooKeeper/etcd)
```
/cache/nodes/node-1 → {"host": "10.0.0.1", "port": 6379, "status": "up", "vnodes": [...]}
/cache/nodes/node-2 → {"host": "10.0.0.2", "port": 6379, "status": "up", "vnodes": [...]}
/cache/ring/version  → 42
/cache/ring/nodes    → [sorted vnode positions with node assignments]
```

---

## 7. API Design

### Client-Facing API
```
GET   /cache/{key}
      Response 200: { "value": "...", "ttl_remaining": 3600 }
      Response 404: { "error": "key not found" }
      Response 410: { "error": "key expired" }

SET   /cache/{key}
      Body: { "value": "...", "ttl": 3600 }
      Response 200: { "status": "OK", "written_to": ["node-1", "node-2"] }

DELETE /cache/{key}
       Response 200: { "status": "OK", "deleted": true }
```

### Internal Cluster API
```
REPLICATE /internal/replicate
Body: { "key": "...", "value": "...", "version": 42, "expiry": 1234567890 }

GOSSIP /internal/gossip
Body: { "node_id": "...", "known_nodes": [...], "ring_version": 42 }

STATS /internal/stats
Response: { "memory_used": "48GB", "hit_rate": 0.95, "evictions": 1234 }
```

---

## 8. Scalability & Bottlenecks

| Bottleneck | Root Cause | Solution |
|-----------|-----------|----------|
| Hot keys | Uneven key distribution | Key replication, local in-process cache |
| Memory pressure | Key accumulation | LRU eviction, memory-capped per node |
| Network bandwidth | High QPS with large values | Value compression, connection multiplexing |
| Ring rebalancing | Node add/remove | Virtual nodes minimize data movement |
| Read latency | Cross-rack network | Client-side routing, affinity to local replica |
| Clock skew | Version conflicts | NTP sync, vector clocks for conflict resolution |

---

## 9. Trade-offs & Design Decisions

### Consistent Hashing vs Hash Slots
- **Consistent Hashing:** O(log N) lookup, minimal key movement on change
- **Hash Slots (Redis Cluster):** 16,384 slots, simpler management, manual slot assignment
- **Choice:** Consistent hashing with vnodes for automatic rebalancing
- **Trade-off:** More complex ring management vs simpler slot approach

### Replication: Sync vs Async
- **Synchronous (all replicas):** Strong consistency, higher write latency
- **Asynchronous:** Low latency, possible data loss on failure
- **Quorum (W=2 of 3):** Balance — durable without waiting for all replicas
- **Trade-off:** Latency vs durability

### LRU vs LFU Eviction
- **LRU:** Evicts least recently used — good for temporal locality
- **LFU:** Evicts least frequently used — good for frequency-biased workloads
- **Choice:** LRU as default (simpler, predictable, covers most workloads)
- **Trade-off:** LRU scans can evict frequently-used cold-start keys

### Single-threaded (Redis model) vs Multi-threaded
- **Single-threaded:** No lock contention, simple, predictable performance
- **Multi-threaded:** Better CPU utilization on multi-core, complex locking
- **Choice:** Single-threaded event loop for data operations (like Redis), multi-threaded I/O
- **Trade-off:** CPU bound at high QPS vs complexity of concurrent data access

---

## 10. Key Interview Talking Points

1. **Consistent Hashing:** Always explain why naive `hash % N` fails (all keys rehash). Consistent hashing moves only K/N keys. Virtual nodes solve uneven distribution. Draw the ring.

2. **LRU O(1) Implementation:** HashMap for O(1) key lookup, doubly linked list for O(1) move-to-front and eviction. This is a classic LeetCode problem — know it cold.

3. **Quorum Reads/Writes:** W+R>N ensures overlap. With N=3, W=2, R=2: you always read from at least one node that has the latest write. Explain the trade-off: higher W = more durable but slower writes.

4. **Cache Stampede:** Classic interview problem. Explain mutex (single bottleneck), probabilistic early expiry (elegant, distributed), and stale-while-revalidate (best user experience).

5. **Hot Key Problem:** Mention it proactively — shows production experience. Key replication distributes reads. Local in-process cache with short TTL is the pragmatic solution.

6. **Memcached vs Redis:** Memcached: simpler, pure LRU, multithreaded. Redis: more data structures, persistence, pub/sub, Lua scripting, single-threaded. Redis wins for most use cases.

7. **Cache Patterns:** Cache-aside is most common. Write-through for consistency. Write-back for performance. Know when to use each.

8. **Failure Handling:** Gossip protocol for node failure detection. Hinted handoff to not lose writes during temporary failures. Read repair for eventual consistency.

9. **Monitoring Metrics:** Hit rate (>95% is healthy), eviction rate (high eviction = undersized cache), memory usage, latency percentiles (P50, P99), connection count.

10. **Back-of-Envelope:** 64GB RAM node, 1KB avg value = 60M keys/node. 150 vnodes balances load within 5%. Adding 1 node to 100-node cluster moves 1% of keys.
