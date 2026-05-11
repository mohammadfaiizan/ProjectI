# 10 — Distributed Systems Fundamentals

---

## Table of Contents
1. [The 8 Fallacies of Distributed Computing](#1-the-8-fallacies-of-distributed-computing)
2. [CAP Theorem Deep Dive](#2-cap-theorem-deep-dive)
3. [PACELC Theorem](#3-pacelc-theorem)
4. [Consistency Models Spectrum](#4-consistency-models-spectrum)
5. [Distributed Consensus: Paxos and Raft](#5-distributed-consensus-paxos-and-raft)
6. [Leader Election Patterns](#6-leader-election-patterns)
7. [Vector Clocks and Lamport Timestamps](#7-vector-clocks-and-lamport-timestamps)
8. [Gossip Protocol](#8-gossip-protocol)
9. [Service Discovery](#9-service-discovery)
10. [Heartbeat and Failure Detection](#10-heartbeat-and-failure-detection)
11. [Quorum Reads and Writes](#11-quorum-reads-and-writes)
12. [Anti-Entropy with Merkle Trees](#12-anti-entropy-with-merkle-trees)
13. [Bloom Filters](#13-bloom-filters)
14. [Consistent Hashing](#14-consistent-hashing)
15. [Distributed Locking](#15-distributed-locking)
16. [Idempotency in Distributed Systems](#16-idempotency-in-distributed-systems)
17. [Retry Patterns](#17-retry-patterns)
18. [Split-Brain Problem](#18-split-brain-problem)
19. [Chaos Engineering](#19-chaos-engineering)
20. [Quick Reference](#20-quick-reference)

---

## 1. The 8 Fallacies of Distributed Computing

Originally articulated by Peter Deutsch (Sun Microsystems). Engineers who ignore these build fragile distributed systems.

### Fallacy 1: The Network is Reliable

Reality: Packets are dropped, duplicated, reordered. TCP handles some issues but not all.

```
Consequence: Must design for partial failure, retries, idempotency
Design response: Circuit breakers, retry with backoff, idempotent operations
```

### Fallacy 2: Latency is Zero

Reality: Even local network has ~0.1ms latency. Cross-datacenter: 10–100ms. External APIs: 100ms+.

```
Consequence: Synchronous chained calls multiply latency
Example: 5 synchronous service calls × 20ms = 100ms minimum
Design response: Async messaging, caching, data locality, avoid N+1 queries
```

### Fallacy 3: Bandwidth is Infinite

Reality: Network bandwidth is finite, shared, and can become a bottleneck.

```
Consequence: Large payload serialization, chatty microservices hurt performance
Design response: Compression, pagination, batch APIs, efficient serialization (Protobuf)
```

### Fallacy 4: The Network is Secure

Reality: Networks can be compromised, intercepted, or spoofed.

```
Design response: mTLS for service-to-service, TLS everywhere, zero-trust networking
```

### Fallacy 5: Topology Doesn't Change

Reality: Servers crash, scale up/down, IPs change, data centers shift.

```
Design response: Service discovery (Consul/Eureka), DNS-based routing, health checks
```

### Fallacy 6: There is One Administrator

Reality: Multiple teams, different organizations, competing policies.

```
Design response: Clear ownership, API contracts, rate limiting, separate failure domains
```

### Fallacy 7: Transport Cost is Zero

Reality: Serialization/deserialization, network transit, protocol overhead cost CPU and time.

```
Design response: Choose efficient protocols (gRPC/Protobuf over REST/JSON for internal)
```

### Fallacy 8: The Network is Homogeneous

Reality: Different OS versions, NIC drivers, proxy configs, MTU sizes, network hardware.

```
Design response: Don't rely on network-layer assumptions; use application-level protocols
```

### Summary Table

| # | Fallacy | Real Danger | Mitigation |
|---|---|---|---|
| 1 | Network reliable | Silent failures | Retries, circuit breakers |
| 2 | Zero latency | Response time SLAs missed | Async, caching |
| 3 | Infinite bandwidth | Throughput bottlenecks | Compression, batching |
| 4 | Network secure | Data interception | mTLS, encryption |
| 5 | Static topology | Hardcoded IPs fail | Service discovery |
| 6 | One admin | Conflicting changes | API contracts, ownership |
| 7 | Free transport | Performance degradation | Efficient serialization |
| 8 | Homogeneous | Interoperability issues | Application-level protocols |

---

## 2. CAP Theorem Deep Dive

### Statement

In the presence of a network **Partition**, a distributed system can guarantee at most two of three properties:
- **C**onsistency — every read receives the most recent write (or an error)
- **A**vailability — every request receives a response (not an error)
- **P**artition Tolerance — system continues operating despite network partition

### The Key Insight

Partition tolerance is **not optional** in real distributed systems. Networks fail. Therefore, the real choice is:

> **CP or AP — what do you sacrifice when a partition occurs?**

```
Venn Diagram (Text):

         [C]onsistency
          /          \
         /   CA zone   \
        /  (single node)\
       /                 \
     [CP]               [AP]
      |                   |
   HBase                Cassandra
   Zookeeper            DynamoDB
   RDBMS(dist)          Couchbase
      \                   /
       \                 /
        \_______________/
              [P]artition
             Tolerance
             (always needed)
```

### CP Systems — Choose Consistency over Availability

When a partition occurs: return error rather than potentially stale data.

**HBase:**
- Uses HDFS for storage; ZooKeeper for coordination
- During partition: master becomes unreachable → HBase becomes unavailable
- When available: strongly consistent reads

**ZooKeeper:**
- Requires majority quorum for writes
- If quorum lost → ZooKeeper stops serving writes (CP)
- Configuration management, distributed locks

**MongoDB (default write concern: majority):**
- With w:majority → waits for majority of replicas
- If primary isolated → new primary elected, old rejects writes

### AP Systems — Choose Availability over Consistency

When a partition occurs: serve potentially stale data rather than returning an error.

**Cassandra:**
- Eventual consistency by default
- Each node can accept writes even during partition
- Conflicts resolved by last-write-wins or custom logic
- Tunable: `consistency=QUORUM` trades availability for consistency

**DynamoDB:**
- Eventually consistent reads by default
- Strongly consistent reads available (higher latency, less available)

**Couchbase, Riak:**
- Available, eventually consistent

### CA Systems (Theory Only)

- Consistent and available but not partition tolerant
- Only possible in single-node systems or systems in same rack with no network failure risk
- PostgreSQL (single node), MySQL (single instance) — not truly distributed

### Real-World Nuance

"CAP is misleading" — Martin Kleppmann: most systems are actually tunable between C and A per-operation.

```
Cassandra with quorum settings:
  ONE:      Available, weakly consistent
  QUORUM:   Balanced
  ALL:      Consistent, less available
```

---

## 3. PACELC Theorem

CAP only addresses what happens **during** a partition. PACELC asks: what about **without** a partition?

### PACELC Statement

- If there is a **P**artition → choose between **A**vailability and **C**onsistency (same as CAP)
- **E**lse (no partition) → choose between **L**atency and **C**onsistency

```
PA/EL:  During partition: available. Else: low latency.     → DynamoDB, Cassandra
PC/EC:  During partition: consistent. Else: consistent.     → HBase, Zookeeper
PA/EC:  During partition: available. Else: consistent.      → MongoDB (tunable)
PC/EL:  During partition: consistent. Else: low latency.    → PNUTS (Yahoo)
```

### Why PACELC Matters

Even without failures, consistency costs latency:
- Synchronous replication to all replicas → higher latency
- Single-leader sequential writes → bottleneck

```
DynamoDB (PA/EL):
  Partition → serves from local replica (available)
  No partition → returns quickly from local replica (low latency)
  Trade-off: may serve stale data

HBase (PC/EC):
  Partition → refuses service (consistent / unavailable)
  No partition → waits for all acks (consistent / higher latency)
```

### PACELC Comparison Table

| System | Partition | Else | Notes |
|---|---|---|---|
| DynamoDB | Availability | Low latency | Eventual consistency default |
| Cassandra | Availability | Low latency | Tunable consistency |
| HBase | Consistency | Consistency | Strong consistency |
| ZooKeeper | Consistency | Consistency | Quorum-based |
| MongoDB | Availability | Consistency | Depends on write concern |
| Spanner | Consistency | Consistency | TrueTime for global ordering |

---

## 4. Consistency Models Spectrum

From strongest to weakest:

```
Strongest  ──────────────────────────────  Weakest
  │                                           │
  ▼                                           ▼
Linearizable → Sequential → Causal → Eventual
```

### Linearizable (Strict) Consistency

- Strongest guarantee: reads always see the most recent write
- Operations appear instantaneous and ordered by real time
- Every operation takes effect between its start and end time

```
Timeline:
T1: Write(x=1) [start=0ms, end=10ms]
T2: Read(x)    [start=11ms] → must return 1

If Read starts before Write ends but both overlap:
  Linearizable: Read may return either 0 or 1 but must be consistent for all readers
```

Used by: etcd, ZooKeeper (within quorum), CockroachDB

### Sequential Consistency

- All operations appear to execute in some sequential order
- Order consistent with per-process order
- Does NOT guarantee real-time ordering

```
Process A: Write(x=1), Write(y=2)
Process B: Read(y=2), Read(x=0)  ← allowed in sequential (reads can be stale)
```

### Causal Consistency

- Causally related operations seen in same order by all
- Concurrent operations can be seen in different orders

```
A writes x=1 (cause)
B reads x=1 and writes y=2 (effect of reading x=1)
C must see x=1 before y=2 (causal order preserved)
C may see unrelated write z=5 in any order
```

### Eventual Consistency

- Given no new updates, all replicas eventually converge to same value
- No guarantees about when
- Read may return stale data

```
Write x=1 to replica A
Read x from replica B → may return x=0 (stale)
Eventually: all replicas have x=1
```

### Consistency Model Comparison

| Model | Real-time order | Causal order | Concurrent order | Performance |
|---|---|---|---|---|
| Linearizable | Yes | Yes | Yes | Slowest |
| Sequential | No | Yes | Yes | Slow |
| Causal | No | Yes | No | Medium |
| Eventual | No | No | No | Fastest |

---

## 5. Distributed Consensus: Paxos and Raft

### Why Consensus?

Multiple nodes need to agree on a single value (e.g., who is the leader, what is the next log entry) despite node failures and network partitions.

### Paxos Overview

Three roles: **Proposers**, **Acceptors**, **Learners**

**Phase 1 (Prepare):**
```
Proposer → Acceptors: "Prepare(n)" (proposal number n)
Acceptors → Proposer: "Promise(n, v)" (promise not to accept proposals < n)
```

**Phase 2 (Accept):**
```
Proposer → Acceptors: "Accept(n, v)"
Acceptors: if n >= promised → accept and notify learners
Learner: if majority accepted → value v is chosen
```

Paxos is notoriously hard to understand and implement correctly. Multi-Paxos (for log replication) requires additional complexity.

### Raft Algorithm

Designed to be **more understandable** than Paxos. Used by etcd, CockroachDB, TiKV, Consul.

**Three States:** Leader, Follower, Candidate

#### Leader Election

```
1. All nodes start as Followers
2. Follower doesn't hear from leader for election_timeout (150–300ms):
   → becomes Candidate
   → increments term
   → votes for itself
   → sends RequestVote to all nodes

3. Node grants vote if:
   - term >= candidate's term
   - not voted for anyone else in this term
   - candidate's log is at least as up-to-date

4. Candidate receives majority votes → becomes Leader
5. Leader sends heartbeats every 150ms to prevent new elections
```

#### Log Replication

```
1. Client sends command to Leader
2. Leader appends to its log (uncommitted)
3. Leader sends AppendEntries to all Followers
4. Followers append to their log, respond OK
5. When majority confirmed → Leader commits entry
6. Leader sends committed index in next heartbeat
7. Followers apply committed entries to state machine
```

#### Safety Guarantees

- **Election Safety:** At most one leader per term
- **Leader Append-Only:** Leader never overwrites its log
- **Log Matching:** If two logs have same index and term, entries before that point are identical
- **State Machine Safety:** If server applies entry at index i, no other server applies different entry at i

#### Term Numbers

```
Term 1: Node A is leader
Network partition occurs
Term 2: Node B elected in majority partition (Node A isolated)
Partition heals: Node A has smaller term → steps down as follower
```

### Raft vs Paxos

| Aspect | Paxos | Raft |
|---|---|---|
| Understandability | Very hard | Designed for clarity |
| Leader | No explicit leader (Multi-Paxos does) | Explicit leader election |
| Log replication | Complex | Built into design |
| Used in | Chubby (Google) | etcd, CockroachDB, Consul |
| Membership changes | Not specified | Included (joint consensus) |

---

## 6. Leader Election Patterns

### Why Leader Election?

Designate one node as coordinator to prevent conflicting operations (write conflicts, duplicate work).

### Bully Algorithm

Node with highest ID wins ("bullies" others into submission).

```
Nodes: A(ID=3), B(ID=5), C(ID=7)

C crashes. A starts election:
  A sends Election to B, C
  B responds OK, starts own election
  B sends Election to C → no response
  B declares itself Leader, sends Coordinator to A
```

**Problem:** Many messages (O(n²)); highest ID node always wins (may not be most capable).

### Raft Leader Election

(See Section 5) — timeout-based, term-based, majority vote.

### ZooKeeper-Based Election

Use ephemeral sequential znodes:

```
All nodes create /election/node-<seq>
  Node A → /election/node-0001
  Node B → /election/node-0002
  Node C → /election/node-0003

Leader = node with lowest sequence number
Each node watches the node with next-lower sequence number
If watched node deleted (crashed) → re-check if now leader
```

**Advantages:**
- No thundering herd (each node watches only one predecessor)
- ZooKeeper handles failure detection
- Ephemeral nodes auto-deleted on session close

### Comparison

| Algorithm | Complexity | SPOF | Use Case |
|---|---|---|---|
| Bully | O(n²) messages | No | Simple clusters |
| Raft | O(n) messages | No | Consensus groups |
| ZooKeeper | O(1) per failure | ZK ensemble | Production systems |

---

## 7. Vector Clocks and Lamport Timestamps

### The Problem

In distributed systems, there is no global clock. How do we order events?

### Lamport Timestamps

Each node maintains a logical counter:
1. Increment before each local event
2. Increment on send, attach to message
3. On receive: `clock = max(local_clock, received_clock) + 1`

```
Node A:  1 → 2 → 3 (send msg with ts=3) ──────────────►
Node B:  1 → 2 (receive ts=3) → max(2,3)+1=4 → 5 → 6

A:ts=3 happened before B:ts=4
BUT: cannot infer causality from timestamps alone
```

**Limitation:** Lamport timestamps establish partial order but cannot detect concurrent events.

### Vector Clocks

Each node maintains a vector of clocks: one per node in the system.

```
3 nodes: A, B, C
Each node's vector: [A_count, B_count, C_count]

Initial: A=[0,0,0]  B=[0,0,0]  C=[0,0,0]

A processes event: A=[1,0,0]
A sends message to B with [1,0,0]
B receives, merges: B=[1,1,0] (max each component, increment own)
B sends to C with [1,1,0]
C receives, merges: C=[1,1,1]
```

**Comparing vectors:**
- V1 < V2: all components of V1 ≤ V2, at least one strictly less → V1 causally before V2
- V1 and V2 concurrent: neither V1 ≤ V2 nor V2 ≤ V1

```
V1=[1,2,0]  V2=[0,3,1]
V1[A]=1 > V2[A]=0, but V1[B]=2 < V2[B]=3 → CONCURRENT (conflict!)
```

### Vector Clocks Use Cases

- **DynamoDB:** Vector clocks for conflict detection
- **Riak:** Version vectors for multi-value conflict resolution
- **CRDTs:** Conflict-free replicated data types use vector clocks

### Dotted Version Vectors (Improvement)

Amazon's improvement to vector clocks: reduces false conflicts and storage overhead.

---

## 8. Gossip Protocol

### What is Gossip?

Epidemic-style communication protocol. Each node periodically selects random peers and exchanges information.

```
Round 0: Node A has new info
Round 1: A tells B (random)
Round 2: A+B each tell random node: B→D, A→C
Round 3: A,B,C,D each tell random node
...
Convergence: O(log N) rounds
```

### Properties

- **Probabilistic:** Not guaranteed per-round but statistically reliable
- **Decentralized:** No coordinator
- **Fault-tolerant:** Works despite node failures
- **Scalable:** O(log N) rounds, O(N log N) messages total

### Failure Detection via Gossip

**SWIM Protocol (Scalable Weakly-Consistent Infection-style Membership):**

```
Node A suspects B (no heartbeat):
  1. A sends Ping to B
  2. No response → A sends PingReq to K random nodes: "ping B for me"
  3. If K nodes also get no response → B marked suspect
  4. After timeout: B marked failed, disseminated via gossip
```

### Gossip Uses

| System | Use of Gossip |
|---|---|
| Cassandra | Cluster membership, node state, schema |
| DynamoDB | Node membership, failure detection |
| Redis Cluster | Cluster topology propagation |
| Consul | Health check dissemination |
| Bitcoin | Transaction propagation |

### Convergence Time

```
N = cluster size
k = fanout (nodes contacted per round)
Rounds to converge ≈ log_k(N)

For N=1000, k=3:
  log_3(1000) ≈ 7 rounds
```

---

## 9. Service Discovery

### Why Service Discovery?

Services run on dynamic IPs (containers, auto-scaling). Hard-coding IPs is fragile.

### Client-Side Discovery

Client queries service registry and selects which instance to call.

```
Client → [Service Registry (Eureka/Consul)] → gets list of OrderService instances
Client → load balances (round-robin, etc.) → calls OrderService instance
```

**Netflix Eureka:**
- Services register on startup, deregister on shutdown
- Heartbeats maintain registration
- Client caches registry locally (resilient if Eureka down)

**HashiCorp Consul:**
- Service registration + health checking
- DNS or HTTP interface for discovery
- Supports multi-datacenter

**Pros:** Client controls load balancing strategy
**Cons:** Client must know about registry; coupling to discovery library

### Server-Side Discovery

Client calls a load balancer/router that queries the registry.

```
Client → [Load Balancer (AWS ALB, Nginx)] → queries registry → routes to service
```

**AWS ALB + ECS:**
- ECS registers containers with ALB target groups
- ALB routes based on path/host rules

**Kubernetes Service:**
- ClusterIP: virtual IP proxied by kube-proxy to pod endpoints
- DNS: `my-service.my-namespace.svc.cluster.local`

**Pros:** Client agnostic; centralized routing logic
**Cons:** Load balancer can be bottleneck; extra hop

### DNS-Based Discovery

```
_http._tcp.orders.service.consul → SRV records
  orders.node1.dc1.consul 8080
  orders.node2.dc1.consul 8080
```

### Comparison

| Method | Coupling | LB control | Failure mode | Example |
|---|---|---|---|---|
| Client-side | Requires library | Client-controlled | Registry down = use cache | Eureka, Consul |
| Server-side | None | Centralized | LB failure = outage | AWS ALB, K8s Service |
| DNS-based | Minimal | DNS TTL-dependent | DNS cache staleness | Consul DNS, CoreDNS |

---

## 10. Heartbeat and Failure Detection

### Simple Heartbeat

- Each node sends periodic "I'm alive" messages to coordinator
- Coordinator marks node dead after timeout

```
Node A → heartbeat every 5s → Coordinator
Coordinator: if no heartbeat for 15s → mark A dead
```

**Problem:** False positives — node may be alive but network slow (latency > timeout).

### Phi Accrual Failure Detector

Used by Cassandra and Akka. Instead of binary (alive/dead), outputs a **suspicion level φ**.

```
φ = -log10(probability that node is still alive)

φ < 1: definitely alive
φ = 5: 99.999% probability dead
φ = 8: 99.9999999% probability dead
```

**How it works:**
1. Track arrival times of heartbeats as a statistical distribution
2. Calculate probability that heartbeat is overdue based on historical inter-arrival times
3. φ grows the longer the heartbeat is overdue relative to historical pattern

**Advantages:**
- Adapts to network conditions (slower network = higher normal interval = higher threshold before suspicion)
- Tunable sensitivity

### Heartbeat vs Gossip Failure Detection

| Method | Scalability | False positives | Detection speed |
|---|---|---|---|
| Central heartbeat | Poor (O(N) to coordinator) | Higher | Fast |
| Gossip-based SWIM | Good (O(N log N) total) | Low | Configurable |
| Phi Accrual | Good | Very low | Configurable |

---

## 11. Quorum Reads and Writes

### Quorum Definition

In a cluster of N replicas:
- **W** = write quorum (replicas that must acknowledge write)
- **R** = read quorum (replicas that must respond to read)

**Consistency condition:** `R + W > N`

### Why R + W > N Guarantees Consistency

```
N=3, W=2, R=2:  R+W=4 > 3

Write: must write to 2 of 3 nodes
Read:  must read from 2 of 3 nodes

Overlap: at least 1 node in both read set and write set
→ at least one reader has the latest write
```

### Common Quorum Configurations

```
N=3:
  W=1, R=3: fast writes, slow reads (relaxed consistency for reads)
  W=3, R=1: slow writes, fast reads (strong durability)
  W=2, R=2: balanced (most common)
  W=1, R=1: fastest, no consistency (use for cache-like behavior)
```

### Sloppy Quorum and Hinted Handoff (DynamoDB)

When preferred nodes are unavailable:
- Write to "hint" nodes (any available node)
- Hint stored with original target info
- When target recovers: hint node forwards data
- Allows writes to succeed even with many failures (more available, less consistent)

### Cassandra Consistency Levels

| Level | W or R quorum | Description |
|---|---|---|
| ONE | 1 | Fastest, weakest consistency |
| QUORUM | (N/2)+1 | Balanced |
| LOCAL_QUORUM | (N/2)+1 in local DC | Low latency for multi-DC |
| ALL | N | Strongest, least available |
| EACH_QUORUM | QUORUM per DC | Consistent in all DCs |

---

## 12. Anti-Entropy with Merkle Trees

### The Problem

Replicas can diverge due to node failures, network partitions, or bugs. How do we detect and fix divergence efficiently?

### Naive Approach

Compare all data between replicas — too slow for large datasets.

### Merkle Tree Solution

A hash tree where:
- Leaf nodes = hash of individual data blocks
- Internal nodes = hash of children's hashes
- Root = single hash representing entire dataset

```
        [Root: H(H(AB)+H(CD))]
           /              \
    [H(AB)]              [H(CD)]
    /      \             /      \
[H(A)]  [H(B)]      [H(C)]   [H(D)]
  A        B           C        D
```

### Anti-Entropy Process

```
Replica 1 and Replica 2 exchange root hashes:
  Same root → data is identical → done (O(1) comparison!)
  
Different root → compare left subtrees:
  Same → divergence is in right subtree
  Different → go deeper → narrow down to specific key ranges
  
Repair only the divergent leaf ranges
```

### DynamoDB Anti-Entropy

- Each node maintains Merkle trees for key ranges
- Background process compares trees with neighbors
- Detects divergence without scanning entire dataset

### Benefits

```
N = number of data items
Without Merkle: O(N) data transferred to detect divergence
With Merkle: O(log N) tree comparison, then only transfer divergent data
```

---

## 13. Bloom Filters

### What is a Bloom Filter?

Probabilistic data structure that answers: "Is element X in the set?"
- **Definitely not** (100% accurate for "no")
- **Probably yes** (false positives possible)
- Space-efficient: no false negatives

### How it Works

```
m = bit array of m bits (all 0 initially)
k = hash functions

Insert "apple":
  h1("apple") = 3 → set bit 3
  h2("apple") = 7 → set bit 7
  h3("apple") = 12 → set bit 12

Query "mango":
  h1("mango") = 5 → bit 5 = 0 → DEFINITELY NOT IN SET

Query "orange":
  h1("orange") = 3 → bit 3 = 1
  h2("orange") = 7 → bit 7 = 1
  h3("orange") = 2 → bit 2 = 0 → DEFINITELY NOT IN SET

Query "apple":
  h1("apple") = 3 → bit 3 = 1
  h2("apple") = 7 → bit 7 = 1
  h3("apple") = 12 → bit 12 = 1 → PROBABLY IN SET ✓
```

### False Positive Rate

```
p ≈ (1 - e^(-kn/m))^k

Where:
  n = number of inserted elements
  m = size of bit array
  k = number of hash functions

Optimal k = (m/n) * ln(2)

Example: 1 million items, 1% false positive rate
  m = -n*ln(p) / (ln(2))^2 ≈ 9.585 million bits ≈ 1.2 MB
  k = 7 hash functions
```

### Use Cases

| System | Usage |
|---|---|
| Cassandra | Avoid disk reads for keys that don't exist |
| HBase | Same — avoid I/O for missing keys |
| Chrome | Malicious URL detection (local filter before DB check) |
| CDN | Prevent one-hit wonders from entering cache |
| Bitcoin | SPV (Simplified Payment Verification) |
| Database query optimizer | Join optimization |

### Limitations

- Cannot delete elements (use Counting Bloom Filter for deletion)
- False positives grow as more elements inserted
- Cannot retrieve stored elements

---

## 14. Consistent Hashing

### The Problem with Simple Modulo Hashing

```
hash(key) % N (N = number of servers)

Problem: Add/remove server → almost ALL keys remap
  N=4: key → server 2
  N=5: key → server (hash % 5) → likely different server
  99.99% cache miss on scaling!
```

### Consistent Hashing Ring

1. Map servers to positions on a virtual ring [0, 2^32)
2. Map keys to positions on same ring
3. Each key assigned to **first server clockwise** on the ring

```
Ring (0 to 360°):
  Server A at 60°
  Server B at 180°
  Server C at 300°

Key X hashes to 90°  → assigned to Server B (next clockwise)
Key Y hashes to 200° → assigned to Server C
Key Z hashes to 320° → assigned to Server A (wraps around)
```

**Adding server D at 120°:**
- Keys between 60° and 120° (previously B's) now go to D
- All other keys unaffected
- Average keys remapped: 1/N of total

### Virtual Nodes (Vnodes)

Problem: uneven distribution with few physical nodes.

Solution: each physical node gets K virtual nodes (positions on ring).

```
Physical: Server A, B, C (3 servers)
Virtual: A1, A2, A3, A4, B1, B2, B3, B4, C1, C2, C3, C4 (12 positions)

Benefits:
  - More even distribution
  - Natural load balancing for heterogeneous hardware
  - Smaller key ranges per vnode during rebalancing
```

### DynamoDB and Cassandra Use

- Cassandra: uses consistent hashing for token ring
- DynamoDB: uses consistent hashing for partition routing

### Properties

| Property | Value |
|---|---|
| Keys remapped when node added | ~1/N |
| Keys remapped when node removed | ~1/N |
| With virtual nodes | Even distribution |
| Lookup complexity | O(log N) with binary search |

---

## 15. Distributed Locking

### When Needed

- Prevent duplicate order processing
- Distributed cron: only one node runs a job
- Leader election (before Raft is available)
- Resource reservation

### Redis-Based Locking (Basic)

```python
# Acquire lock (SET NX PX)
lock_key = "lock:order:12345"
result = redis.set(lock_key, unique_id, nx=True, px=30000)  # 30s TTL
if result:
    try:
        process_order(12345)
    finally:
        # Only release if we own the lock
        if redis.get(lock_key) == unique_id:
            redis.delete(lock_key)
```

**Problem with single Redis node:** Node failure = lock lost or permanently held.

### Redlock Algorithm (Distributed)

Martin Kleppmann describes this for high-availability locking across 5 Redis nodes:

```
1. Get current time T1
2. Try to acquire lock on all 5 nodes (with short timeout per node)
3. Lock acquired if:
   a. Majority of nodes (3+) responded positively
   b. Elapsed time < lock TTL
4. Validity = TTL - (T2 - T1) - clock drift
5. If failed: release lock on all nodes

Release:
  Send Lua script to all nodes: if value matches, delete
```

**Controversy:** Martin Kleppmann argues Redlock still has issues with clock drift and process pauses. For strong safety: use ZooKeeper or etcd.

### ZooKeeper Distributed Lock

```
1. Create ephemeral sequential node: /locks/resource-0001
2. Get all children of /locks/
3. If your node has lowest sequence → you have the lock
4. Else: watch the next-lower node
5. When watched node deleted → re-check step 3

Ephemeral node: auto-deleted if session closes (handles crashes)
```

### Database Advisory Locks (PostgreSQL)

```sql
-- Acquire advisory lock (session-level)
SELECT pg_try_advisory_lock(12345);  -- returns true if acquired

-- Do work...

-- Release
SELECT pg_advisory_unlock(12345);
```

### Comparison

| Method | Durability | Complexity | Failure handling | Use case |
|---|---|---|---|---|
| Redis (single) | Low | Simple | Manual TTL | Cache locks |
| Redlock | Medium | Moderate | Multi-node | Medium criticality |
| ZooKeeper | High | Higher | Session-based | Leader election |
| DB advisory | High | Simple | Transaction-based | Simple coordination |
| etcd/lease | High | Simple | TTL + watch | Production locking |

---

## 16. Idempotency in Distributed Systems

### Why Idempotency?

At-least-once delivery means operations may be retried. If the operation is not idempotent: duplicate charges, duplicate emails, data corruption.

**Idempotent operation:** `f(f(x)) = f(x)` — applying multiple times = applying once.

### Idempotency Keys

Client generates unique key per operation. Server deduplicates on key.

```
POST /payments
Headers: Idempotency-Key: uuid-client-generated-abc123

Server logic:
  Check if key "abc123" exists in idempotency table:
    EXISTS → return cached response (don't re-process)
    NOT EXISTS → process payment, store result with key

CREATE TABLE idempotency_keys (
  key         VARCHAR PRIMARY KEY,
  response    JSONB,
  created_at  TIMESTAMP,
  expires_at  TIMESTAMP
);
```

### Stripe's Idempotency Implementation

- 24-hour TTL on idempotency keys
- Locks the key during processing (prevents concurrent duplicate requests)
- Stores full response for replay

### Database-Level Idempotency

```sql
-- Idempotent upsert
INSERT INTO orders (order_id, user_id, amount, status)
VALUES ($1, $2, $3, 'pending')
ON CONFLICT (order_id) DO NOTHING;

-- Idempotent increment (only if not already applied)
UPDATE accounts
SET balance = balance + $amount,
    last_transaction_id = $txn_id
WHERE account_id = $id
  AND last_transaction_id != $txn_id;
```

### Idempotency + At-Least-Once = Exactly-Once Semantics

```
At-least-once delivery + Idempotent processing = Effectively exactly-once
```

---

## 17. Retry Patterns

### Fixed Retry

```python
for attempt in range(max_retries):
    try:
        response = call_service()
        return response
    except TransientError:
        time.sleep(retry_delay)
raise MaxRetriesExceeded()
```

**Problem:** All clients retry simultaneously → thundering herd → overwhelms recovering service.

### Exponential Backoff

```python
def retry_with_backoff(fn, max_retries=5, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except TransientError as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)  # 1, 2, 4, 8, 16 seconds
            time.sleep(delay)
```

### Jitter (Prevent Thundering Herd)

```python
import random

delay = base_delay * (2 ** attempt)
jitter = random.uniform(0, delay)
time.sleep(jitter)  # Full jitter: spread retries across the window
```

**AWS SDK uses "Full Jitter":**
```
sleep = random(0, min(cap, base * 2^attempt))
```

### Circuit Breaker Pattern

Prevents cascading failures. Three states:

```
CLOSED (normal): requests pass through
  → failure_count > threshold → OPEN

OPEN (failing): requests immediately fail (fail-fast)
  → after timeout → HALF-OPEN

HALF-OPEN (testing): allow limited requests
  → success → CLOSED
  → failure → OPEN
```

```python
class CircuitBreaker:
    def __init__(self, threshold=5, timeout=60):
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = None
        self.threshold = threshold
        self.timeout = timeout
    
    def call(self, fn):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitOpenError()
        
        try:
            result = fn()
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.threshold:
                self.state = "OPEN"
            raise
```

---

## 18. Split-Brain Problem

### What is Split-Brain?

Network partition causes cluster to split into two isolated groups, each believing they are the primary/leader.

```
5-node cluster:
[A, B, C] ←partition→ [D, E]

Both partitions elect their own leader:
  ABC partition: A is leader
  DE partition:  D is leader
  
Both leaders accept writes → divergent state → data corruption
```

### Quorum-Based Prevention

Require majority quorum (N/2 + 1) for leadership:

```
5-node cluster: quorum = 3
  ABC partition (3 nodes): CAN form quorum → elects leader
  DE partition (2 nodes): CANNOT form quorum → refuses leadership

Only one partition can have majority → only one leader
```

### STONITH (Shoot The Other Node In The Head)

Used in high-availability clusters (Pacemaker):
- When split detected: each partition tries to "fence" the other
- Fencing = power off, reset, or block storage access to other partition
- Only one wins the race → other partition shut down

### etcd/Raft Split-Brain Handling

```
3-node etcd cluster [A, B, C]:
  A isolated from B, C

B and C form quorum:
  Elect B as leader
  A cannot form quorum → A stops accepting writes (becomes read-only)
  
Partition heals:
  A reconnects, sees higher term from B
  A steps down, syncs log from B
```

---

## 19. Chaos Engineering

### Definition

Deliberately injecting failures into production systems to discover weaknesses before they cause incidents.

### Principles of Chaos Engineering

1. Define steady state (normal behavior: latency p99, error rate, throughput)
2. Hypothesize that steady state will continue during failure
3. Introduce variables (failures) in production (or production-like staging)
4. Look for difference from steady state → weakness found

### Netflix Simian Army

| Tool | What it does |
|---|---|
| Chaos Monkey | Randomly terminates EC2 instances in production |
| Chaos Gorilla | Terminates entire AWS Availability Zone |
| Latency Monkey | Injects artificial network delays |
| Conformity Monkey | Finds and shuts down non-conforming instances |
| Security Monkey | Finds security violations |
| Janitor Monkey | Cleans up unused cloud resources |
| Chaos Kong | Takes down entire AWS region |

### Fault Injection Techniques

```
Network faults:
  - Packet loss (10%, 50%, 100%)
  - Latency injection (50ms, 500ms, 5s)
  - Network partition between services
  - Bandwidth throttling

Resource exhaustion:
  - CPU spike (100% load)
  - Memory exhaustion (OOM kill)
  - Disk fill
  - Connection pool exhaustion

Application faults:
  - Return HTTP 500 from dependency
  - Slow response (timeout simulation)
  - Corrupt response payloads
```

### Game Days

Scheduled chaos experiments with full team awareness:
1. Define scope and hypothesis
2. Prepare rollback plan
3. Execute experiment
4. Observe and measure
5. Document findings
6. Fix discovered weaknesses

### Chaos Engineering Tools

| Tool | Type | Features |
|---|---|---|
| Chaos Monkey (Netflix OSS) | AWS EC2 | Random instance termination |
| Chaos Toolkit | Open-source | Multi-platform, extensible |
| Gremlin | Commercial | Network, CPU, memory, I/O |
| LitmusChaos | Kubernetes | Native K8s chaos |
| AWS Fault Injection Simulator | Managed | AWS-native chaos |
| Toxiproxy (Shopify) | Proxy | Network fault injection |

---

## 20. Quick Reference

### Consistency Model Comparison

| Model | Every read sees latest? | Concurrent order agreed? | Real-time? | Example |
|---|---|---|---|---|
| Linearizable | Yes | Yes | Yes | etcd, Zookeeper |
| Sequential | Yes (logical) | Yes | No | Theoretical |
| Causal | Causal only | No | No | COPS, DynamoDB (conditional) |
| Eventual | No | No | No | Cassandra, DynamoDB default |

### CAP vs PACELC Comparison Table

| System | CAP | P→? | E→? | Notes |
|---|---|---|---|---|
| DynamoDB | AP | A | L | Available + low latency; eventual consistency |
| Cassandra | AP | A | L | Tunable; default eventual |
| HBase | CP | C | C | Consistent; unavailable during partition |
| ZooKeeper | CP | C | C | Quorum required |
| MongoDB | CP/AP | Tunable | Tunable | Depends on write concern |
| Spanner | CP | C | C | TrueTime global consistency |

### Distributed Systems Interview Cheat Sheet

1. **CAP theorem** → During partition: CP (HBase, ZK) vs AP (Cassandra, Dynamo)
2. **PACELC** → Else (no partition): Latency (Dynamo) vs Consistency (HBase)
3. **Raft** → Leader election + log replication; term numbers prevent split-brain
4. **Vector clocks** → Detect causality and concurrent events; DynamoDB conflict detection
5. **Gossip** → O(log N) rounds to propagate info; failure detection in Cassandra
6. **Quorum** → R + W > N → at least 1 reader overlaps with last writer
7. **Consistent hashing** → Minimize remapping on scale; vnodes for even distribution
8. **Bloom filter** → No false negatives; avoid disk reads for missing keys
9. **Merkle tree** → Efficient anti-entropy; only sync divergent ranges
10. **Circuit breaker** → CLOSED→OPEN→HALF-OPEN; prevent cascading failures

### The 8 Fallacies Quick Reference

```
1. Network reliable    → Design for failure (retries, circuit breakers)
2. Zero latency        → Async, cache, minimize hops
3. Infinite bandwidth  → Compress, paginate, efficient serialization
4. Secure network      → mTLS, encryption, zero trust
5. Static topology     → Service discovery, health checks
6. One administrator   → API contracts, versioning
7. Zero transport cost → Binary protocols (gRPC/Protobuf)
8. Homogeneous network → Application-level protocols
```
