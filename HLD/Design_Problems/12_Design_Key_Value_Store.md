# System Design: Distributed Key-Value Store

---

## 1. Problem Statement & Clarifying Questions

### Problem Statement
Design a distributed key-value store (like Amazon DynamoDB or Apache Cassandra) that can store billions of key-value pairs across thousands of nodes, providing high availability and partition tolerance with tunable consistency.

### Clarifying Questions

**Scale:**
- How many nodes? *Up to 10,000 nodes*
- How many keys? *Billions of keys, up to 10 KB per value*
- Read/write ratio? *Balanced — 50:50 to 80:20 read-heavy*
- What throughput per node? *~10K ops/sec per node*

**Consistency:**
- Strong vs eventual consistency? *Tunable: default eventual (quorum configurable)*
- What replication factor? *N=3 (3 replicas per key)*
- Quorum settings? *Default W=2, R=2 for eventual; W=3, R=3 for strong*

**Operations:**
- Required operations? *GET, PUT, DELETE with optional TTL*
- Transactions/multi-key operations? *No — single key only*
- Range queries? *No — point lookups only*

**Availability:**
- Acceptable downtime? *99.99% availability (4 nines)*
- Behavior during network partition? *Prefer availability (AP system per CAP theorem)*

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
1. PUT(key, value, ttl=None) — store or update a value
2. GET(key) — retrieve value by key
3. DELETE(key) — remove a key
4. Configurable replication factor N
5. Configurable quorum reads R and writes W
6. Automatic failure detection and data re-replication
7. Gossip-based cluster membership

### Non-Functional Requirements
| Property | Target |
|---|---|
| Availability | 99.99% |
| Consistency | Tunable (eventual default) |
| Latency (GET) | < 10ms p99 |
| Latency (PUT) | < 20ms p99 |
| Throughput | 10K nodes × 10K ops = 100M ops/sec cluster-wide |
| Scalability | Linear scale-out by adding nodes |
| Durability | Data survives N-1 node failures |

---

## 3. Capacity Estimation

### Storage
- 1 billion keys × 10 KB average value = **10 TB raw**
- Replication factor N=3: **30 TB total across cluster**
- 10,000 nodes: **3 GB per node average**
- With LSM overhead (compaction, WAL): ~**5 GB per node**

### Network
- Write quorum W=2: each PUT fans out to N=3 nodes → 3× write amplification
- At 1M writes/second: 1M × 10 KB × 3 = **30 GB/s** cluster-wide (manageable with rack-aware routing)

### Memory (MemTable per node)
- MemTable before flush: 64 MB
- Write buffer + bloom filters: ~512 MB per node
- Gossip state (10K nodes × 100 bytes): 1 MB per node

---

## 4. High-Level Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              Client                      │
                    │   (SDK with consistent hash routing)    │
                    └────────────────┬────────────────────────┘
                                     │ SDK routes to coordinator
                    ┌────────────────▼────────────────────────┐
                    │         Coordinator Node                 │
                    │  (Any node can be coordinator for req)  │
                    └──────┬─────────────┬────────────────────┘
                           │ Replication │ (N=3)
           ┌───────────────▼──────┐  ┌──▼──────────────────┐
           │     Replica 1        │  │    Replica 2         │
           │  ┌────────────────┐  │  │  ┌────────────────┐  │
           │  │  Storage Engine│  │  │  │  Storage Engine│  │
           │  │  (LSM Tree)    │  │  │  │  (LSM Tree)    │  │
           │  │  WAL           │  │  │  │  WAL           │  │
           │  │  MemTable      │  │  │  │  MemTable      │  │
           │  │  SSTables      │  │  │  │  SSTables      │  │
           │  └────────────────┘  │  │  └────────────────┘  │
           └──────────────────────┘  └──────────────────────┘
                                  (+ Replica 3, similar)

    ┌─────────────────────────────────────────────────────────────┐
    │                   Cluster Management                        │
    │  ┌────────────────┐   ┌────────────────┐  ┌─────────────┐  │
    │  │ Gossip Protocol│   │ Consistent Hash│  │ Merkle Tree │  │
    │  │ (membership,   │   │ Ring (virtual  │  │ (anti-      │  │
    │  │  failure detect│   │  nodes, token  │  │  entropy)   │  │
    │  │  node state)   │   │  ranges)       │  └─────────────┘  │
    │  └────────────────┘   └────────────────┘                   │
    └─────────────────────────────────────────────────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Consistent Hashing Ring

**Problem:** How to distribute N billion keys across 10K nodes without full reshuffling when nodes join/leave?

**Solution:** Hash ring [0, 2^128) with virtual nodes.

1. Each physical node owns V=150 virtual nodes (tokens on the ring)
2. Key is hashed to a position; assigned to the first V-node clockwise
3. When a node is added: only its V-node ranges are migrated (1/N of data)
4. When a node is removed: its ranges are taken over by successor V-nodes

**Virtual Nodes benefit:** Ensures even distribution even with heterogeneous hardware; new nodes take load from many existing nodes uniformly.

### 5.2 Replication Strategy

**Coordinator node** receives the write and replicates to N-1 additional nodes:
1. Coordinator hashes the key → finds primary V-node position
2. Walk the ring clockwise to find N distinct physical nodes (skip duplicates of same physical node)
3. Rack-aware placement: prefer replicas on different racks/AZs

**Quorum Reads/Writes:**
```
N = 3 replicas
W = 2 write quorum (wait for ACK from 2 replicas before confirming)
R = 2 read quorum (read from 2 replicas, return latest version)

Strong consistency:  W + R > N  →  2 + 2 = 4 > 3 ✓
Eventual consistency: W=1, R=1  →  1 + 1 = 2 ≤ 3 (faster, less consistent)
```

### 5.3 Vector Clocks for Conflict Detection

When two clients concurrently update the same key and only W=1 replicas are written, divergent versions can exist:

```
Node A: key="x", value="foo", clock={A:1}
Node B: key="x", value="bar", clock={B:1}  ← concurrent write while A is down
```

Vector clock comparison:
- If `clock_a[i] <= clock_b[i]` for all i → B is newer, discard A
- If neither dominates → **conflict**: return both versions to client for resolution

**Conflict Resolution Strategies:**
- **LWW (Last Write Wins):** Use wall clock timestamp; simpler, may lose writes
- **Application-level merge:** Return both values (DynamoDB returns a "siblings" list)
- **CRDT:** Use conflict-free data types (counters, sets) that merge deterministically

### 5.4 Storage Engine: LSM Tree

**Why LSM over B-Tree?**
- Write-optimized: all writes are sequential (MemTable → WAL → SSTables)
- B-Tree has random I/O for inserts; LSM converts random writes to sequential

**LSM Structure:**
```
Write path: WAL (append) → MemTable (in-memory sorted map)
             → When MemTable full: flush to SSTable on disk
             → Background: merge/compact SSTables

Read path:  MemTable → L0 SSTables → L1 SSTables → ... → Bloom filter check per level
```

**Compaction Strategies:**
- **Size-tiered:** Merge SSTables of similar size (write-optimized, higher space amplification)
- **Leveled (LevelDB/RocksDB style):** Each level 10× bigger; merges reduce read amplification
- **TWCS (Time Window):** Group SSTables by time window; good for time-series data

### 5.5 Bloom Filter

A Bloom filter sits in front of SSTable reads:
- 1-2% false positive rate with ~10 bits per key
- Before reading an SSTable, check Bloom filter: if it returns false, skip file entirely
- Dramatically reduces disk I/O for keys that don't exist (negative lookups)
- 1 billion keys × 10 bits = **10 Gb = 1.25 GB** for bloom filters in memory

### 5.6 Gossip Protocol for Failure Detection

- Each node maintains a **membership list** with heartbeat counters per node
- Every 1 second: gossip with 3 random peers, exchange membership lists
- Merge rule: take the max heartbeat counter per node
- If a node's heartbeat hasn't increased in T=10 heartbeat rounds → marked **SUSPECT**
- After T_fail=30s without update → marked **DOWN**; data re-replication triggered

**Hinted Handoff:**
- If the target replica node is DOWN during a write:
  - Coordinator writes to any available node with a "hint" (original target, key, value)
  - When target node recovers, hint is replayed to it
  - Ensures durability during transient failures (sloppy quorum)

### 5.7 Anti-Entropy with Merkle Trees

**Problem:** After failures/repairs, replicas may diverge. How to detect and sync differences efficiently?

**Solution:**
1. Each replica builds a **Merkle tree** over its key ranges
2. Root hash summarizes all data; subtree hashes narrow down differences
3. Two replicas exchange root hashes — if they match, data is in sync
4. If different: drill down the tree to identify the specific key ranges that differ
5. Only transfer the differing data (not the entire dataset)

**Time complexity:** O(K) to detect differences vs O(N×K) for naive full scan.

---

## 6. Database Design

```
Physical Storage Layout per Node:
├── WAL (Write-Ahead Log)
│   └── wal_000001.log  ← append-only, fsync on write
├── MemTable (in-memory, sorted red-black tree)
├── SSTables/
│   ├── L0/
│   │   ├── sst_001.data  ← key-value pairs sorted by key
│   │   └── sst_001.idx   ← bloom filter + key index
│   ├── L1/
│   │   └── sst_010.data
│   └── L2/
│       └── sst_100.data
└── metadata.json  ← version, node_id, token ranges

SSTable entry format:
┌──────────┬───────────┬──────────────┬────────────┬──────────────────┐
│ key_len  │   key     │  value_len   │  value     │  timestamp + ttl │
│ (4 bytes)│ (variable)│  (4 bytes)   │ (variable) │  (16 bytes)      │
└──────────┴───────────┴──────────────┴────────────┴──────────────────┘

Cluster Metadata (stored in all nodes via gossip):
- node_id → (ip, port, status, tokens[], datacenter, rack)
- Tokens define the virtual node positions on the hash ring
```

---

## 7. API Design

### Client-Facing API

```
# Simple KV operations
PUT  /kv/{key}
     Body: { value, ttl_seconds? }
     Headers: X-Consistency: eventual|strong
     Response: 200 OK, { version, timestamp }

GET  /kv/{key}
     Headers: X-Consistency: eventual|strong
     Response: 200 { key, value, version, timestamp }
             : 404 Key not found
             : 300 { siblings: [{value, clock}, ...] }  ← conflict

DELETE /kv/{key}
     Response: 200 OK

# Admin API
GET  /admin/ring           ← view consistent hash ring
GET  /admin/nodes          ← cluster membership status
POST /admin/nodes/{id}/decommission
GET  /admin/stats          ← reads, writes, cache hits, compaction status
```

### Internal Node-to-Node API

```
# Replication (coordinator → replicas)
POST /internal/replicate
     Body: { key, value, clock, timestamp, ttl }

# Hinted handoff replay
POST /internal/hints/replay
     Body: { target_node_id }

# Anti-entropy (merkle sync)
GET  /internal/merkle/root?range=start,end
POST /internal/merkle/sync   ← exchange and repair
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: Write Hotspots
**Problem:** Sequential keys (timestamps, auto-increments) hash to adjacent ring positions.
**Solution:**
- Recommend clients use random UUIDs or prefix keys with random bytes
- Virtual nodes spread load even with sequential keys
- Monitor per-node throughput; rebalance tokens if variance > 20%

### Bottleneck 2: Compaction Blocking Reads
**Problem:** Background compaction causes read latency spikes (I/O competition).
**Solution:**
- Rate-limit compaction I/O (e.g., max 50 MB/s)
- Use dedicated compaction threads with lower I/O priority
- Leveled compaction reduces read amplification vs size-tiered

### Bottleneck 3: Gossip Overhead at 10K Nodes
**Problem:** O(N) gossip state per node × 10K nodes = 1 MB state per node.
**Solution:**
- Use compressed gossip messages (delta encoding)
- Bound gossip fanout to 3-5 peers per round
- Use indirect ping (ask C to ping B if A→B fails) to reduce false positives

### Bottleneck 4: Read Amplification
**Problem:** Reads may scan many SSTables (especially for tombstoned/overwritten keys).
**Solution:**
- Bloom filter eliminates 99% of unnecessary SSTable reads
- Leveled compaction reduces read amplification to O(L) levels
- Cache hot SSTables in OS page cache (don't use Java heap; use mmap)

### Bottleneck 5: Node Recovery Time
**Problem:** After a node failure, re-replication of 3 GB can take hours.
**Solution:**
- Streaming repair: copy data in sorted order directly between nodes (SSTables, not row-by-row)
- Prioritize most-read key ranges first
- Hinted handoff reduces divergence window for short outages (< repair threshold)

---

## 9. Trade-offs & Design Decisions

### Decision 1: AP vs CP (CAP Theorem)
- **Chosen:** AP (Availability + Partition Tolerance) — like Cassandra/DynamoDB
- **Why:** KV store should remain writable during network partitions (sloppy quorum, hinted handoff)
- **Trade-off:** Possible read of stale data; conflicts need resolution
- **Alternative:** CP (like ZooKeeper/etcd) — stricter consistency, lower availability

### Decision 2: LSM Tree vs B-Tree Storage Engine
- **Chosen:** LSM Tree
- **Why:** Workloads are write-heavy; LSM converts random writes to sequential I/O, enabling SSD throughput
- **Trade-off:** Higher read amplification (multiple files), requires compaction
- **B-Tree alternative:** Better for read-heavy with random access; used in PostgreSQL, MySQL

### Decision 3: Consistent Hashing vs Range Partitioning
- **Chosen:** Consistent hashing with virtual nodes
- **Why:** No central metadata server needed; nodes can join/leave with minimal rebalancing
- **Alternative:** Range partitioning (HBase/BigTable style) — better for range scans, requires master node

### Decision 4: Quorum N=3, W=2, R=2
- **Why W=2, R=2:** Overlap of 1 guarantees at least one node has the latest version
- **Trade-off:** 33% slower than W=1, R=1 but far more consistent
- **For analytics (stale reads OK):** R=1 acceptable; for banking: W=3, R=3

### Decision 5: LWW vs Vector Clock Conflict Resolution
- **LWW (Last Write Wins):** Simple, loses data on concurrent writes
- **Vector Clocks:** Detects true conflicts, returns siblings to client
- **Chosen:** LWW as default (DynamoDB style); vector clocks for conflict tracking in audit log
- **Reason:** Most clients don't handle siblings; LWW sufficient for cache-like workloads

---

## 10. Key Interview Talking Points

1. **Consistent Hashing:** 150 virtual nodes per physical node ensures even distribution. When a node is added, it takes 1/N of the data from its immediate predecessors on the ring. Virtual nodes absorb uneven hashing distribution.

2. **Quorum Formula:** W + R > N guarantees overlap. With N=3, W=2, R=2: at least one node in both quorum sets has the latest write. Strong consistency (W=3, R=1) sacrifices write availability for read speed.

3. **Sloppy Quorum + Hinted Handoff:** During temporary node failure, writes go to available nodes with hints. When the failed node recovers, hints are replayed. This maintains W availability without waiting for recovery.

4. **LSM Tree Write Path:** All writes are sequential: WAL → MemTable → SSTable flush. This is why Cassandra/RocksDB can handle 100K+ writes/second on SSDs (sequential I/O is 10× faster than random).

5. **Bloom Filter ROI:** 10 bits per key, 1% false positive rate. For a 100 GB SSTable, a 100 MB bloom filter saves 99% of unnecessary disk reads for missing keys. Essential for read performance.

6. **Gossip Protocol Convergence:** With fanout=3, information propagates to all N nodes in O(log N) rounds. At 10K nodes, every node knows about a failure within ~14 rounds = ~14 seconds.

7. **Anti-Entropy Purpose:** Handles scenarios where hinted handoff was lost (node down > hint expiry). Merkle tree comparison is O(differences), not O(total data), making full cluster repair practical.

8. **Vector Clocks vs Timestamps:** Wall clocks can skew (NTP drift ±100ms). Vector clocks are logical — they track causality, not time. Two writes are concurrent if neither vector clock dominates the other.

9. **Compaction Trade-off:** Compaction maintains read performance but competes with foreground reads/writes. Key metrics: write amplification factor (WAF), read amplification factor (RAF), space amplification factor (SAF). Leveled: low RAF, high WAF. Size-tiered: low WAF, high RAF.

10. **Why DynamoDB chose Consistent Hashing:** Before DynamoDB, Amazon used RDBMS for their shopping cart. During a Black Friday partition, the entire cart service went down. DynamoDB's AP design means "partial writes are better than no writes" — you can always add to cart even during network issues.
