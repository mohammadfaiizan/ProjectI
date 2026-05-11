# Distributed Systems Fundamentals — HLD Interview Q&A

---

## Easy (Q1–Q7)

---

### Q1. What are the 8 fallacies of distributed computing?

**Answer:**

The 8 fallacies of distributed computing (originally by Peter Deutsch and James Gosling at Sun Microsystems) describe incorrect assumptions developers commonly make when building distributed systems. Violating these leads to fragile, incorrect systems.

| # | Fallacy | Reality |
|---|---------|---------|
| 1 | The network is reliable | Networks drop packets, connections reset, links fail |
| 2 | Latency is zero | Cross-datacenter calls add 50-200ms; local calls add 1-5ms |
| 3 | Bandwidth is infinite | High-volume data transfer saturates links |
| 4 | The network is secure | Traffic can be intercepted; services can be impersonated |
| 5 | Topology doesn't change | Nodes are added/removed; IPs change; DNS entries expire |
| 6 | There is one administrator | Multiple teams, cloud providers, ISPs manage the stack |
| 7 | Transport cost is zero | Serialization, TLS, compression have CPU/memory costs |
| 8 | The network is homogeneous | Different hardware, OS, drivers, versions coexist |

**Engineering implications:**

- **Fallacy 1 (reliability):** Implement retries with exponential backoff, circuit breakers, and timeouts.
- **Fallacy 2 (latency):** Use async communication, caching, avoid chatty interfaces (batch requests).
- **Fallacy 3 (bandwidth):** Compress payloads, paginate large responses, use streaming.
- **Fallacy 4 (security):** Encrypt in transit (TLS), authenticate all calls (mTLS, JWT), authorize at every hop.
- **Fallacy 5 (topology):** Use service discovery (DNS, Consul) rather than hardcoded IPs.
- **Fallacy 6 (one admin):** Automate configuration, use infrastructure-as-code.
- **Fallacy 7 (transport cost):** Profile serialization overhead; choose efficient formats (Protobuf > JSON).
- **Fallacy 8 (homogeneous):** Use standard protocols (HTTP/2, gRPC) that work across heterogeneous environments.

Understanding these fallacies is foundational because every design decision in distributed systems is a response to one or more of them.

---

### Q2. What does CAP theorem actually mean? Is it really a choice of 2 of 3?

**Answer:**

**CAP theorem** (Brewer, 2000) states that a distributed data store can provide at most two of the following three guarantees simultaneously:

- **C — Consistency:** Every read receives the most recent write or an error.
- **A — Availability:** Every request receives a (non-error) response, without guarantee it's the most recent write.
- **P — Partition tolerance:** The system continues to operate despite an arbitrary number of network partitions.

**The common misconception — "choose 2 of 3":**
In practice, **network partitions are unavoidable** in any distributed system (the network will eventually fail). You cannot choose to sacrifice P. The real choice is: **during a partition, do you sacrifice C or A?**

```
Normal operation (no partition):
  [Node A] <----replication----> [Node B]
  Both nodes agree on data state.

During partition:
  [Node A] ~~~X~~~ [Node B]  (network cut)
  
  Client writes "x=5" to Node A.
  Client reads from Node B.
  
  CP choice: Node B returns ERROR (refuses to serve stale data)
  AP choice: Node B returns x=3 (old value, may be stale)
```

**CP systems (prefer consistency during partition):**
HBase, ZooKeeper, Spanner, Consul. Refuse to serve requests when they can't guarantee consistency.

**AP systems (prefer availability during partition):**
Cassandra, CouchDB, DynamoDB (default), Riak. Serve potentially stale data.

**The nuance** (from Gilbert and Lynch's proof): The theorem applies to a model where you must choose a behavior *at the exact moment of a partition*. Outside of partitions, both C and A can be satisfied simultaneously.

```
Partition occurring?
  NO  -> System can be both Consistent AND Available (ACID databases during normal ops)
  YES -> Choose one:
           CP: Return error / wait for quorum
           AP: Return potentially stale data
```

**Practical takeaway:** Don't design around "which 2 do I pick?" — instead design for: "When a partition occurs, what behavior should my system exhibit, and what can my application tolerate?"

---

### Q3. What is the PACELC theorem, and why was it proposed?

**Answer:**

**PACELC** extends CAP theorem to address its incomplete picture. CAP only describes behavior **during** a partition. PACELC also covers the trade-off **during normal operation**.

**PACELC stands for:**
- **P:** If there is a **Partition** →
- **A:** sacrifice **Availability** or
- **C:** sacrifice **Consistency**
- **E:** **Else** (no partition) →
- **L:** trade-off between **Latency** and
- **C:** **Consistency**

```
Normal operation: no partition
  [Node A] ---fast replication--> [Node B]
  
  LOW LATENCY choice: Write to A, return immediately (replicate async)
    -> Low latency, but reads from B may be stale
    -> EL (Else Latency)
  
  HIGH CONSISTENCY choice: Write to A, wait for B to confirm
    -> Higher latency, but both nodes always agree
    -> EC (Else Consistency)
```

**Why CAP alone is insufficient:**
CAP says Spanner is CP. But so is HBase. PACELC distinguishes them further: Spanner chooses EC (consistency over latency during normal ops — uses TrueTime for global ordering). HBase also chooses EC but with different latency characteristics.

**System classifications:**

| System | Partition | Normal Op | Classification |
|--------|-----------|-----------|---------------|
| DynamoDB (default) | AP | EL | PA/EL |
| Cassandra | AP | EL | PA/EL |
| HBase | CP | EC | PC/EC |
| ZooKeeper | CP | EC | PC/EC |
| Google Spanner | CP | EC | PC/EC |
| MySQL (single node) | N/A | EC | N/A |
| Cosmos DB | Configurable | Configurable | Varies by level |

**Tuning:** Many modern databases (Cassandra, DynamoDB, Cosmos DB) are tunable along this spectrum. You can configure consistency level per operation, choosing the trade-off at request time:
- `QUORUM` writes: slower, more consistent
- `ONE` writes: faster, less consistent

---

### Q4. What are the main consistency models in distributed systems?

**Answer:**

Consistency models define what guarantees a system makes about the visibility of reads and writes across nodes. From strongest to weakest:

**1. Linearizability (Strong Consistency)**
Every read sees the most recently committed write. Operations appear instantaneous and globally ordered. The strongest model.
```
T1: write(x=5)
T2: read(x) -> must return 5 (if T1 completed before T2 started)
```
Cost: High latency (requires coordination/quorum). Used in: ZooKeeper, Google Spanner, etcd.

**2. Sequential Consistency**
All operations appear to execute in some sequential order consistent with each node's program order, but not necessarily real-time order.
```
Node A: write(x=1), write(x=2)
Node B: read(x) -> 2, read(x) -> 1  // NOT allowed (violates program order)
Node B: read(x) -> 1, read(x) -> 2  // OK (consistent with A's order)
```
Weaker than linearizable: doesn't require real-time ordering. Used in some CPU memory models.

**3. Causal Consistency**
Operations that are causally related are seen in the same order by all nodes. Concurrent (causally unrelated) operations may be seen in different orders.
```
A writes post P
B reads P, then writes comment C (causally related to P)
All nodes must see P before C

D writes post Q (concurrent with P)
Nodes may see P before Q or Q before P — both valid
```
Used in: MongoDB causal sessions, some Cassandra configurations.

**4. Eventual Consistency**
If no new writes are made, all replicas will eventually converge to the same value. No guarantees about when, or what intermediate values are seen.
```
Node A: write(x=5)
Node B: read(x) -> 3 (stale, not yet propagated)
... some time later...
Node B: read(x) -> 5 (converged)
```
Used in: DynamoDB (default), Cassandra (ONE consistency), DNS.

**Comparison:**

| Model | Ordering Guarantee | Latency | Use Case |
|-------|-------------------|---------|---------|
| Linearizable | Global real-time | High | Financial transactions |
| Sequential | Program order | Medium | Shared memory systems |
| Causal | Causally related | Low | Social feeds, comments |
| Eventual | No guarantee | Lowest | DNS, shopping carts |

**The read-your-writes guarantee** (session consistency): A client always reads its own writes. Weaker than linearizable but practically important (e.g., after updating profile, user should see updated profile).

---

### Q5. What is the Raft consensus algorithm, and how does leader election work?

**Answer:**

**Raft** is a consensus algorithm designed to be more understandable than Paxos. It ensures that a cluster of nodes agrees on a sequence of values (log entries) even when some nodes fail.

**Raft roles:**
- **Leader:** Handles all client requests, replicates log to followers. One leader at a time.
- **Follower:** Passive; only responds to requests from leader or candidates.
- **Candidate:** Trying to become leader.

**Leader Election process:**

```
Normal state:
  Leader ---heartbeat (AppendEntries with no entries)---> Followers
  
  If follower doesn't receive heartbeat within election_timeout (150-300ms random):
    -> Follower becomes Candidate
    -> Increments its term (term=2)
    -> Votes for itself
    -> Sends RequestVote to all other nodes

RequestVote logic (each node votes YES if):
  1. candidate.term >= my.term (not outdated)
  2. candidate.log is at least as up-to-date as mine
  3. I haven't already voted in this term

Election outcome:
  Candidate receives votes from majority (>N/2 nodes) -> becomes Leader
  Another node wins first -> Candidate becomes Follower
  Split vote (no majority) -> Election timeout, new election with term+1
```

**Log replication:**
```
Client -> Leader: "set x=5"
Leader: Appends to own log at index 7, term 3
Leader: Sends AppendEntries(index=7, term=3, entry="set x=5") to all Followers
Followers: Append to log, respond SUCCESS
Leader: Once majority ACKs, COMMITS entry, applies to state machine
Leader: Notifies followers to commit in next heartbeat
```

**Safety guarantee:** Raft ensures that if two logs have the same index and term, they contain the same entry and all preceding entries are identical. This prevents split-brain where two nodes apply different commands at the same position.

**Used in:** etcd (Kubernetes), CockroachDB, TiKV, Consul.

---

### Q6. What is split-brain in distributed systems, and how do you prevent it?

**Answer:**

**Split-brain** occurs when a network partition causes a distributed cluster to divide into two or more isolated sub-clusters, each believing it is the authoritative partition and accepting writes independently. When the partition heals, the clusters have divergent data with no clear winner.

```
Before partition:
  [Node1] --- [Node2] --- [Node3]
  All three form a single cluster, Node1 is leader.

Partition event:
  [Node1] | [Node2] --- [Node3]
  
  Node1: "I'm still the leader!" (accepts writes W1)
  Nodes 2,3: "Node1 is dead, elect new leader!" -> Node2 is new leader (accepts writes W2)
  
  Both W1 and W2 committed independently.
  
  Partition heals: W1 and W2 conflict. Which is correct?
```

**Prevention strategies:**

**1. Majority Quorum (most common)**
Only a partition with a **majority of nodes** can elect a leader or accept writes. With N=3 nodes, need 2 votes.
```
Node1 (isolated): cannot reach majority -> steps down, refuses writes
Nodes 2,3: have majority -> elect leader, accept writes
```
This is what Raft, Paxos, and ZooKeeper implement.

**2. Fencing tokens**
When a new leader is elected, it gets a fencing token (monotonically increasing integer). Old leader's writes are rejected if its token is lower.
```
Old leader: token=5 -> sends write with token=5
Storage system: "Current valid token is 7. Reject."
```

**3. STONITH (Shoot The Other Node In The Head)**
Force-kill the isolated node via out-of-band mechanism (IPMI, power switch) to guarantee it can't accept writes.
Used in high-availability database clusters (Pacemaker/Corosync).

**4. Lease-based leadership**
Leader holds a time-bounded lease. When lease expires, it must stop serving. If it can't renew (partition), it steps down voluntarily before the new leader is elected.

**5. Odd number of nodes**
Always use an odd number of nodes (3, 5, 7) to avoid a 50/50 split where no partition has a majority.

---

### Q7. What is a Bloom filter, and what are its use cases in distributed systems?

**Answer:**

A **Bloom filter** is a space-efficient probabilistic data structure that tests whether an element is a member of a set. It can return:
- **Definitely NOT in the set** (100% certain)
- **Possibly in the set** (may be a false positive)

It **never produces false negatives**. It uses multiple hash functions and a bit array.

```
Insert "apple":
  hash1("apple") = 3  -> set bit[3] = 1
  hash2("apple") = 7  -> set bit[7] = 1
  hash3("apple") = 12 -> set bit[12] = 1

Query "banana":
  hash1("banana") = 3  -> bit[3] = 1 (set by apple)
  hash2("banana") = 5  -> bit[5] = 0 -> DEFINITELY NOT IN SET

Query "cherry":
  hash1("cherry") = 3  -> bit[3] = 1
  hash2("cherry") = 7  -> bit[7] = 1
  hash3("cherry") = 12 -> bit[12] = 1 -> POSSIBLY IN SET (false positive!)
```

**False positive rate:**
```
p ≈ (1 - e^(-k*n/m))^k

where:
  k = number of hash functions
  n = number of inserted elements
  m = size of bit array
```

For m=10 bits per element and k=7 hash functions: ~1% false positive rate.

**Use cases in distributed systems:**

| Use Case | How Bloom Filter Helps |
|----------|----------------------|
| Cassandra/HBase SSTable | Skip reading SSTables that don't contain a key (avoid disk IO) |
| Google BigTable | Avoid disk lookups for non-existent rows |
| Chrome (Safe Browsing) | Check URLs against malicious list locally |
| CDN | Check if an object exists before fetching from origin |
| Database (avoid null lookups) | Don't query DB for keys that definitely don't exist |
| Distributed cache (Redis) | Prevent cache penetration (querying DB for non-existent keys) |
| Network routers | Detect duplicate packets |

**Limitations:**
- Cannot remove elements (use Counting Bloom Filter for deletion).
- False positive rate increases as more elements are added.
- Not suitable when false positives are costly (financial transactions).

---

## Medium (Q8–Q15)

---

### Q8. How do vector clocks differ from Lamport timestamps, and when do you use each?

**Answer:**

Both are logical clocks that assign timestamps to events in distributed systems to establish ordering without relying on synchronized physical clocks.

**Lamport Timestamps (1978)**
A single integer counter per process. Rules:
1. Before sending a message: increment own counter.
2. On receiving a message: counter = max(local, received) + 1.

```
Process A: [1]  [2]------>  [5]
                             ^
Process B: [1]  [2]  [3]  [4]---> [6]
                         sends msg with ts=4
                         B receives: max(2,4)+1=5 -> A's counter=5
```

Lamport timestamps establish a **partial order**: if A→B then ts(A) < ts(B). But the converse is NOT true: ts(A) < ts(B) does not mean A→B. **Cannot detect concurrent events.**

**Vector Clocks**
A vector of counters, one per process. Rules:
1. Before an event: increment own entry.
2. Before sending: increment own entry, attach full vector.
3. On receive: component-wise max of local and received, then increment own.

```
3 nodes: A, B, C
Initial: A=[0,0,0], B=[0,0,0], C=[0,0,0]

A does event: A=[1,0,0]
A sends to B:  A=[2,0,0], B receives -> B=[2,1,0]
B sends to C:  B=[2,2,0], C receives -> C=[2,2,1]
A does event: A=[3,0,0]  (concurrent with B's activity after receiving from A)
```

**Comparing vector clocks:**
- V1 < V2 if V1[i] ≤ V2[i] for all i (and strictly less for at least one): V1 happened before V2.
- V1 and V2 are **concurrent** if neither V1 ≤ V2 nor V2 ≤ V1.
- This is the key advantage: vector clocks **can detect concurrent events**.

**When to use each:**

| Feature | Lamport | Vector Clock |
|---------|---------|-------------|
| Detect causality | Yes (partial) | Yes (complete) |
| Detect concurrency | No | Yes |
| Space | O(1) | O(N) — N = nodes |
| Use case | Simple ordering, logging | Conflict detection (DynamoDB), CRDTs |

**DynamoDB uses vector clocks** (with LWW fallback) to detect conflicting writes from different clients during network partitions, allowing application-level conflict resolution.

---

### Q9. How does a gossip protocol work, and what are its use cases?

**Answer:**

A **gossip protocol** (also called epidemic protocol) is a peer-to-peer communication method where each node periodically exchanges state information with a randomly selected subset of neighbors. Information spreads through the cluster similarly to how rumors or diseases spread through a population.

**Basic gossip algorithm:**
```
Every T milliseconds, each node:
  1. Select K random peers from known node list
  2. Send current state (or state delta) to each peer
  3. Receive state from peers
  4. Merge received state with local state

Convergence: with fanout K, information spreads in O(log N) rounds
```

**Example — membership/health state propagation:**
```
Time 0: Node A marks Node F as "DOWN"
         A's state: {A:UP, B:UP, C:UP, D:UP, E:UP, F:DOWN}

Time 1: A gossips to random peers B, D
         B and D learn F:DOWN

Time 2: B gossips to C, E; D gossips to C, G
         C, E, G learn F:DOWN

Time 3: Nearly all nodes know F:DOWN (exponential spread)
```

**Properties:**
- **Eventual consistency:** All nodes converge to the same state within O(log N) rounds.
- **Fault tolerant:** Works even if some nodes crash mid-gossip.
- **Scalable:** No central coordinator; communication is O(N log N) total messages per round.
- **Decentralized:** No single point of failure.

**Use cases in production systems:**

| System | Gossip Use |
|--------|-----------|
| Cassandra | Node membership, failure detection, schema propagation |
| DynamoDB | Ring membership, token ring updates |
| Redis Cluster | Cluster state, slot assignments |
| Consul | Service health, member lists |
| Bitcoin/blockchain | Transaction and block propagation |
| Amazon S3 | Metadata replication hints |

**Limitations:**
- **Eventually consistent** — brief windows where nodes have different views.
- **Bandwidth:** O(N) messages per round; manageable but not zero.
- **Amplification:** Without deduplication, popular updates can be re-gossiped unnecessarily. Use version numbers / generation counters.

---

### Q10. What is quorum reads/writes (R+W>N), and when does it guarantee strong consistency?

**Answer:**

In a distributed system with **N replicas**, a **quorum** is the minimum number of nodes that must agree for an operation to succeed.

- **W** = write quorum (number of replicas that must acknowledge a write)
- **R** = read quorum (number of replicas that must respond to a read)

**The rule for strong consistency: R + W > N**
This ensures that the read set and write set always overlap by at least one node, guaranteeing the read will see the most recent write.

```
N=3 replicas

W=2, R=2: R+W=4 > 3 ✓ (strong consistency)
  Write must be on at least 2 nodes.
  Read must contact at least 2 nodes.
  Overlap guaranteed: at least 1 node has the latest write.

W=1, R=3: R+W=4 > 3 ✓ (strong consistency, but W=1 means fast writes, slow reads)

W=1, R=1: R+W=2 < 3 ✗ (eventual consistency — no overlap guarantee)
```

**Visual overlap:**
```
3 replicas: [R1] [R2] [R3]

Write with W=2: [R1✓] [R2✓] [R3 ]  <- R1,R2 have latest
Read  with R=2: [R1 ] [R2✓] [R3✓]  <- R2 is the overlap (has latest write)
```

**Common configurations:**

| Config (N=3) | W | R | Consistency | Availability |
|-------------|---|---|-------------|-------------|
| Strong | 2 | 2 | Strong | Tolerate 1 failure |
| Fast reads | 3 | 1 | Strong | All must be up |
| Fast writes | 1 | 3 | Strong | All must be up |
| Eventual | 1 | 1 | Eventual | High |
| Cassandra LOCAL_QUORUM | 2 | 2 | Strong (local DC) | Tolerates 1 |

**Cassandra quorum settings:**
```
Consistency Levels:
  ONE    -> W=1, R=1 (fastest, eventual)
  QUORUM -> W=ceil(N/2)+1, R=ceil(N/2)+1 (strong)
  ALL    -> W=N, R=N (strongest, least available)
  LOCAL_QUORUM -> Quorum within local datacenter
```

**Important nuance:** Even with R+W>N, linearizability is not guaranteed without additional coordination (like version vectors or compare-and-swap). Quorum prevents reading stale data but doesn't prevent concurrent writes from creating conflicts.

---

### Q11. How does consistent hashing work, and what problem do virtual nodes solve?

**Answer:**

**Consistent hashing** is a technique for distributing data across nodes such that when a node is added or removed, only a minimal fraction of keys need to be remapped (K/N keys on average, where K=keys and N=nodes).

**Basic consistent hashing:**
```
Hash ring (0 to 2^32-1):

              0
           /     \
    NodeC(300)  NodeA(100)
          |       |
    NodeB(200)---/

Key assignment: hash(key) -> find nearest node clockwise

hash("user:1")  = 150 -> NodeB (next clockwise after 150)
hash("order:5") = 250 -> NodeC (next clockwise after 250)
hash("item:9")  = 50  -> NodeA (next clockwise after 50)

Adding NodeD at position 180:
  Only keys between 150 and 180 move from NodeB to NodeD
  All other keys unchanged
```

**Problem with basic consistent hashing: Hotspots and imbalance**
With few real nodes, hash positions are unlikely to be perfectly distributed. One node may get 60% of the keyspace while another gets 10%.

**Virtual nodes (vnodes) solution:**
Each physical node owns multiple positions on the ring.

```
Physical nodes: NodeA, NodeB, NodeC (each gets 100 virtual nodes)

Ring (showing only some vnodes):
  NodeA-v1(45), NodeB-v1(73), NodeC-v1(112), NodeA-v2(150),
  NodeB-v2(210), NodeC-v2(270), NodeA-v3(320), ...

Benefits:
  1. Even key distribution (statistical averaging over many points)
  2. When NodeA is added: its vnodes spread load across ALL existing nodes
  3. When NodeA fails: its vnodes' keys spread to ALL remaining nodes
```

**Virtual nodes enable:**
- **Heterogeneous nodes:** A node with 2x hardware gets 2x vnodes → 2x data automatically.
- **Smooth rebalancing:** Adding a node takes a little from every existing node, not just neighbors.
- **Fault tolerance:** A node failure is shared by all nodes, not just the successor.

**Used in:** Cassandra (default 256 vnodes), Amazon Dynamo, Riak.

---

### Q12. How does anti-entropy with Merkle trees work?

**Answer:**

**Anti-entropy** is a background process that ensures data consistency between replicas by comparing and synchronizing their data. The challenge is doing this efficiently without transferring all data.

**Merkle trees (hash trees)** solve the "which data differs?" problem efficiently.

A Merkle tree is a binary tree where:
- Leaf nodes = hash of individual data chunks/rows.
- Parent nodes = hash of concatenation of children hashes.
- Root node = single hash representing the entire dataset.

```
Data: [D1, D2, D3, D4]

Leaf level:   H(D1)    H(D2)    H(D3)    H(D4)
Level 2:      H(H1+H2)          H(H3+H4)
Root:         H(H12 + H34)

If D3 changes on Replica B:
  Replica A root:  abc123
  Replica B root:  xyz789  <- different!

Compare children of root:
  Left subtree H(H1+H2): SAME on both -> skip, no diff in D1,D2
  Right subtree H(H3+H4): DIFFERENT -> recurse
    H(D3): DIFFERENT -> D3 needs sync
    H(D4): SAME -> skip
```

**Efficiency:** With N data items, comparing trees takes O(log N) hash comparisons to find the differing subtree, rather than O(N) full data comparison.

**Anti-entropy process:**
```
1. Node A builds Merkle tree over its key range
2. Node A sends root hash to Node B
3. If roots match: no diff, done
4. If mismatch: exchange tree level by level (BFS)
5. Identify leaf nodes that differ
6. Sync only the differing data chunks
```

**Real-world usage:**
- **Cassandra:** Uses Merkle trees for read repair and node repair operations.
- **DynamoDB:** Anti-entropy process for replica synchronization.
- **Git:** Content-addressed objects; commit trees are effectively Merkle trees.
- **Blockchain:** Transaction Merkle trees in blocks (Bitcoin, Ethereum).

**Limitation:** Building a Merkle tree is CPU and I/O intensive. Cassandra runs `nodetool repair` during off-peak hours to avoid impacting production traffic.

---

### Q13. What are the main approaches to distributed locking, and what are their trade-offs?

**Answer:**

Distributed locking ensures mutual exclusion across multiple nodes/processes for shared resources. Unlike a single-machine mutex, distributed locks must handle network failures and node crashes.

**1. Redis-based lock (Redlock)**
```python
# Acquire lock:
SET resource_name unique_id NX PX 30000
# NX = only set if Not eXists
# PX 30000 = expire in 30000ms (safety timeout)

# Release lock (Lua script for atomicity):
if redis.call("get", key) == unique_id then
    return redis.call("del", key)
end
```
- Single Redis: Simple but SPOF.
- Redlock (N=5 Redis nodes): Acquire on majority (3+); immune to single node failure.
- **Concern (Martin Kleppmann):** Clock drift can cause safety violations in edge cases with process pauses (GC stops).

**2. ZooKeeper-based lock (most reliable)**
```
1. Create ephemeral sequential node: /locks/resource-00000001
2. List all children: /locks/resource-*
3. If you own the lowest-numbered node: you hold the lock
4. Else: watch the node just before yours (avoid herd effect)
5. When that node is deleted: recheck if you're now lowest
6. Release: delete your node (or it auto-deletes on client disconnect)
```
ZooKeeper uses ZAB (atomic broadcast) — strongly consistent. Lock is safe because ZooKeeper nodes are ephemeral (auto-released on session timeout).

**3. Database-based lock**
```sql
-- Pessimistic: SELECT FOR UPDATE
SELECT * FROM locks WHERE resource_id = ? FOR UPDATE;
-- Holds lock for transaction duration

-- Optimistic: version-based
UPDATE resource SET version = version+1, ...
WHERE id = ? AND version = ?;
-- Retry if rows_affected = 0
```

**4. etcd-based lock**
Similar to ZooKeeper but uses Raft consensus. Used in Kubernetes for leader election.
```
etcd lease: PUT /locks/resource <holder> --lease=<lease_id>
Lease renews via heartbeat; auto-expires if process dies
```

**Comparison:**

| Approach | Consistency | Fault Tolerance | Complexity | Latency |
|----------|------------|----------------|------------|---------|
| Redis single | Weak | Low (SPOF) | Low | Very Low |
| Redlock | Weak* | Medium | Medium | Low |
| ZooKeeper | Strong | High | Medium | Medium |
| etcd | Strong | High | Medium | Medium |
| DB SELECT FOR UPDATE | Strong (DB) | DB-dependent | Low | Medium |

**Key advice:** Distributed locks are hard to get right. Prefer designing systems to avoid needing them (idempotent operations, CAS operations). When you must use them, prefer ZooKeeper or etcd for safety.

---

### Q14. What is the two generals problem, and why does it matter for distributed systems?

**Answer:**

The **Two Generals Problem** is a thought experiment that proves **it is impossible to achieve guaranteed agreement** between two parties communicating over an unreliable channel.

**The scenario:**
Two generals (A and B) must coordinate an attack on a city. They can only communicate via messengers who may be captured (message lost). They need to agree on a time to attack simultaneously.

```
General A: "Attack at dawn?" -> [Messenger] -> General B (messenger captured? unknown)

If General A sends: "Attack at dawn"
  General B receives it and agrees: "OK, attack at dawn" -> [Messenger] -> General A
  General A receives confirmation... but needs to confirm he received it
  -> Sends confirmation: "Confirmed" -> [Messenger] -> General B
     General B needs to confirm he received the confirmation...
     -> Infinite regress. No message can be the last.
```

**The proof:** For any final message either side sends, they cannot know if the other side received it. Both generals remain uncertain. The problem is unsolvable — there is no protocol that guarantees agreement with unreliable communication.

**Why it matters in distributed systems:**

**1. TCP handshake analogy:** TCP's 3-way handshake is a practical approximation. After the third message (ACK of SYN-ACK), the client sends data optimistically — we accept a tiny window of uncertainty.

**2. Distributed transactions:** You can never guarantee that all participants in a distributed transaction atomically committed or rolled back. 2PC has the same fundamental uncertainty in coordinator failure scenarios.

**3. At-least-once delivery:** Messaging systems choose "retry until acknowledged" precisely because you can never be certain a single message was received. This causes duplicates, which must be handled via idempotency.

**4. Exactly-once impossibility:** True exactly-once across network boundaries is impossible at the pure theory level. Kafka's "exactly-once" is actually "effectively exactly-once" — it handles the practical cases through sequence numbers and transactions, but relies on acknowledged commits.

**Practical implication:** Always design for the possibility that the last message in any protocol was lost. Design with retries, idempotency, and timeouts as first-class concerns.

---

### Q15. What is the Phi Accrual failure detector, and how does it improve on simple timeout-based detection?

**Answer:**

**Simple timeout failure detection** uses a fixed heartbeat interval and declares a node dead if no heartbeat is received within a fixed timeout. Problem: the threshold must be set conservatively high (to avoid false positives during GC pauses or network hiccups), leading to slow failure detection. Or if set low, legitimate slow responses cause false failures.

**Phi Accrual Failure Detector** (Hayashibara, 2004) replaces the binary "alive/dead" decision with a continuous **suspicion level** phi (φ) that grows over time when heartbeats are not received.

**How it works:**
```
1. Track heartbeat arrival intervals: [t1, t2, t3, ...]
2. Fit a distribution (normal/exponential) to the intervals
3. At query time T, compute phi:

phi = -log10(P_later(T - t_last))

where P_later = probability of NOT receiving a heartbeat by time T
               given the observed distribution

phi = 0   -> node is healthy, heartbeat expected soon
phi = 1   -> 90% probability the node is dead
phi = 2   -> 99% probability
phi = 3   -> 99.9% probability
phi = 10  -> very high confidence, node is dead
```

**Adaptive behavior:**
```
Normal operation (50ms intervals):
  phi rises slowly -> threshold not reached quickly

GC pause (one missed heartbeat, then resumes):
  phi rises above threshold momentarily, but distribution adapts
  Next heartbeats recalibrate the expected interval
  Less likely to false-positive flag a GC-paused node

Actual failure (no heartbeats forever):
  phi grows unboundedly -> eventually crosses threshold confidently
```

**Configurable threshold (φ_threshold):**
- φ = 8: suitable for fast LAN environments (low latency, low variance)
- φ = 12: suitable for WANs or environments with high jitter

**Used in:** Cassandra (configurable phi_convict_threshold), Akka clustering.

**Comparison:**

| Method | False Positives | Detection Speed | Adaptability |
|--------|----------------|----------------|-------------|
| Fixed timeout | High (conservative) | Slow (conservative) | None |
| Adaptive timeout | Medium | Medium | Basic |
| Phi Accrual | Low | Fast (adaptive) | High |

---

## Hard (Q16–Q20)

---

### Q16. How does DynamoDB handle write conflicts using LWW and vector clocks?

**Answer:**

Amazon Dynamo (the internal paper; DynamoDB shares its lineage) is an AP system designed for high availability with eventual consistency. During network partitions, multiple replicas can accept conflicting writes for the same key.

**Last Write Wins (LWW):**
The simplest conflict resolution: when two conflicting versions of a key exist, keep the one with the latest wall-clock timestamp.

```
Replica A: key="cart:user1", value=[item1, item2], timestamp=T1
Replica B: key="cart:user1", value=[item1, item3], timestamp=T2 (T2 > T1)

After merge: keep T2 version -> [item1, item3]
item2 is silently lost!
```

LWW is simple but problematic: clock skew between nodes can cause newer writes to lose. A write with a slightly earlier timestamp is discarded even if it logically happened later.

**Vector Clocks in Dynamo:**
Dynamo uses vector clocks to track causality and detect true conflicts (concurrent writes that cannot be ordered).

```
Initial state:
  key="cart" -> value=[item1], vclock={}

Client C1 reads: gets ([item1], {})
Client C1 writes [item1, item2]:
  Coordinator node Sx increments Sx's counter
  -> value=[item1, item2], vclock={Sx:1}

Concurrent: Client C2 also read the initial version and writes [item1, item3]:
  Coordinator node Sy: -> value=[item1, item3], vclock={Sy:1}

Now two versions exist:
  V1: ([item1, item2], {Sx:1})
  V2: ([item1, item3], {Sy:1})

Neither V1 ≤ V2 nor V2 ≤ V1 -> CONFLICT (concurrent writes)
```

**Conflict resolution options:**

1. **Application-level resolution:** Dynamo returns all conflicting versions to the client on next read. Client must merge them.
```
Read returns: [([item1, item2], {Sx:1}), ([item1, item3], {Sy:1})]
Client merges: [item1, item2, item3]
Client writes merged version with merged vclock
```

2. **LWW fallback:** If application doesn't handle conflicts, use timestamps as tiebreaker. This is what DynamoDB's default mode does with its "last writer wins" attribute.

3. **CRDTs (Conflict-free Replicated Data Types):** Data structures that support automatic mathematically correct merging (e.g., a shopping cart as a grow-only set).

**Vector clock pruning:** In practice, vector clocks grow as more coordinators handle writes. Dynamo truncates old (coordinator, counter) pairs when the clock exceeds a size threshold — this can cause false conflicts but keeps memory bounded.

---

### Q17. How does ZooKeeper work internally using the ZAB protocol?

**Answer:**

**ZooKeeper** is a centralized coordination service (leader election, configuration, distributed locks, naming). It uses **ZAB (ZooKeeper Atomic Broadcast)** — a total-order broadcast protocol similar to Raft but optimized for high read throughput.

**ZooKeeper data model:**
A hierarchical namespace (like a filesystem) of **znodes**:
```
/
├── /services
│   ├── /services/db (ephemeral — auto-deleted when client disconnects)
│   └── /services/cache
├── /locks
│   └── /locks/resource-00000001 (sequential ephemeral)
└── /config
    └── /config/timeout  -> "30000"
```

**ZAB protocol — two phases:**

**Phase 1: Leader Election (crash-recovery)**
```
1. Any server can be leader, follower, or observer.
2. On startup or leader failure:
   - Servers vote for the server with the highest (zxid, server_id)
   - zxid = epoch(32 bits) + counter(32 bits) — uniquely identifies transactions
   - Majority agreement required to elect a leader
3. New leader:
   - Discovers highest committed zxid across quorum members
   - Synchronizes followers: sends all uncommitted transactions
   - Starts new epoch
```

**Phase 2: Active Messaging (broadcast)**
```
Client write request to any server:
  1. Follower forwards write to Leader
  2. Leader assigns zxid (monotonically increasing)
  3. Leader sends PROPOSAL to all followers (like 2PC prepare)
  4. Followers write to transaction log, send ACK to leader
  5. When majority ACKs: Leader sends COMMIT to all followers
  6. Followers apply to in-memory data tree
  7. Leader responds to client
```

**Reads are served locally** from any ZooKeeper server (not just leader). This gives ZooKeeper excellent read throughput but means reads may be slightly stale. For linearizable reads, use `sync()` before read (forces sync with leader).

**Watchers:**
Clients can set watches on znodes. When the znode changes, ZooKeeper sends a one-time notification to the watcher. Used for:
- Service discovery (watch /services/*)
- Configuration change detection (watch /config/*)
- Lock acquisition (watch the node ahead in the queue)

**Comparison with Raft:**

| Feature | ZAB | Raft |
|---------|-----|------|
| Read from followers | Yes (possibly stale) | Yes (same) |
| Leader failover | Fast (pre-computed) | Election-based |
| Used in | ZooKeeper | etcd, Consul, CockroachDB |
| Transaction model | 2PC-like broadcast | Log replication |

---

### Q18. What is a distributed transaction, and why is it fundamentally hard?

**Answer:**

A **distributed transaction** is a transaction that spans multiple nodes, services, or databases, requiring all participants to either commit or rollback atomically. It must satisfy ACID properties across the distributed participants.

**Why it's hard — the fundamental tension:**
Distributed transactions require both **atomicity** (all-or-nothing) and **network reliability** (impossible to guarantee, as per the Two Generals Problem). Additionally, holding locks across multiple systems for the duration of the transaction severely limits throughput.

**Two-Phase Commit (2PC):**
```
Phase 1 — Prepare:
  Coordinator -> Participant A: "Prepare to commit transaction T"
  Coordinator -> Participant B: "Prepare to commit transaction T"
  Participant A: Acquires locks, writes to WAL, replies "READY"
  Participant B: Acquires locks, writes to WAL, replies "READY"

Phase 2 — Commit:
  If all READY: Coordinator -> A, B: "COMMIT"
  If any ABORT: Coordinator -> A, B: "ROLLBACK"
```

**Problems with 2PC:**

1. **Blocking:** If coordinator crashes after Phase 1 and before Phase 2, participants are blocked holding locks indefinitely. They can't unilaterally decide to commit or rollback (they don't know what other participants said).

2. **Coordinator SPOF:** Single coordinator failure halts all in-flight transactions.

3. **Latency:** Two network round trips minimum, plus all participants must be synchronously available.

4. **Scalability:** Doesn't work well across microservices with different databases and tech stacks.

**Three-Phase Commit (3PC):** Adds a "pre-commit" phase to allow participants to timeout and rollback if coordinator fails. Solves the blocking problem but introduces new issues with network partitions (split decisions).

**Modern alternatives:**

| Approach | Consistency | Availability | Complexity |
|----------|-------------|-------------|------------|
| 2PC | Strong | Low (blocking) | Medium |
| Saga | Eventual | High | High |
| Google Spanner TrueTime | Strong | Medium | Very High |
| CRDT-based | Eventual | High | Medium |

**Sagas** (discussed in Q11 of file 06) are the practical answer for microservices: break the distributed transaction into a sequence of local transactions with compensating actions. Accept eventual consistency, design for business-level recovery.

**XA Transactions:** Standard protocol for 2PC across heterogeneous databases (MySQL, Oracle, etc.). Works but suffers the same 2PC problems.

**Google Spanner** achieves distributed transactions with external consistency using TrueTime (GPS + atomic clocks), which bounds clock uncertainty to within ~7ms globally. This allows commit timestamps to be assigned such that T1 commits before T2 if ts(T1) < ts(T2) — a mathematically provable external consistency property unavailable in most systems.

---

### Q19. What are CRDTs (Conflict-free Replicated Data Types) and how do they work?

**Answer:**

A **CRDT** is a data structure designed so that concurrent updates on different replicas can always be merged automatically and deterministically, without coordination or conflict resolution logic.

**The core mathematical property:**
A CRDT must form a **join-semilattice**: there exists a merge function ⊔ that is:
- **Commutative:** A ⊔ B = B ⊔ A (order of merge doesn't matter)
- **Associative:** (A ⊔ B) ⊔ C = A ⊔ (B ⊔ C)
- **Idempotent:** A ⊔ A = A (merging same state twice is safe)

These properties mean: regardless of message ordering, reordering, or duplication, all replicas will converge to the same state.

**Types of CRDTs:**

**1. G-Counter (Grow-only Counter)**
```
Each node has its own counter slot.
Increment: local_vector[my_node]++
Value:      sum(all slots)
Merge:      component-wise max

Node A: [3, 0, 0]
Node B: [0, 5, 0]
Merge:  [3, 5, 0]  <- correct: total = 8
```

**2. PN-Counter (Positive-Negative Counter)**
Two G-Counters: one for increments (P), one for decrements (N).
```
Value = sum(P) - sum(N)
```

**3. G-Set (Grow-only Set)**
```
Add element: add to local set
Merge: union of both sets
Cannot remove elements (no conflict)
```

**4. 2P-Set (Two-Phase Set)**
Two G-Sets: added set A and removed set R.
```
Add element: add to A
Remove element: add to R (element must be in A first)
Membership: x in (A - R)
Merge: union A, union R
Limitation: once removed, can never re-add
```

**5. LWW-Element-Set**
Each element has a timestamp. Latest timestamp wins.
```
Add(x, t=5) from A; Remove(x, t=3) from B
Merge: t=5 > t=3 -> x is present
```

**6. OR-Set (Observed-Remove Set)**
Add/remove with unique tags to correctly handle concurrent add/remove:
```
Add(x) -> generates unique tag: (x, tag1)
Remove(x) -> removes all observed tags for x
Concurrent add(x) with new tag and remove(x) -> add wins (new tag not seen by remove)
```

**Real-world CRDT usage:**
- **Redis:** CRDT support in Redis Enterprise for multi-region replication.
- **Riak:** CRDT data types (counters, sets, maps).
- **Figma:** Uses CRDT-like approaches for collaborative document editing.
- **Google Docs / Operational Transform:** OT is an alternative to CRDT for collaborative text.

**When to use CRDTs:**
- Multi-region active-active databases where writes happen simultaneously at different regions.
- Offline-first applications (mobile apps that sync later).
- Collaborative real-time editing.

---

### Q20. What are the principles of chaos engineering, and how does it apply to distributed systems design?

**Answer:**

**Chaos Engineering** is the discipline of experimenting on a distributed system to build confidence in its ability to withstand turbulent and unexpected conditions. Coined by Netflix and formalized in the "Principles of Chaos Engineering" manifesto.

**Core principle:** "Build confidence in the system's capability to withstand turbulent conditions by proactively injecting failure in production (or production-like) environments."

**The scientific method applied:**

```
1. Define steady state (baseline normal behavior):
   "99.9% of API requests respond in <200ms"
   "Order completion rate is 98.5%"

2. Hypothesize: "This behavior will continue when X fails"

3. Inject failure X (a controlled experiment)

4. Observe: Does the steady state hold?

5. Analyze: If not, why not? Fix the weakness.
```

**Types of experiments:**

| Chaos Type | Example |
|------------|---------|
| Instance failure | Kill random EC2 instances (Chaos Monkey) |
| Network partition | Block traffic between AZ A and AZ B |
| Latency injection | Add 200ms to all calls to Service X |
| Resource exhaustion | Fill disk or exhaust CPU on a node |
| Clock skew | Artificially advance clocks on subset of nodes |
| Dependency failure | Return 500 errors from a downstream service |
| Region failure | Blackhole all traffic to us-east-1 |
| DNS failure | Break DNS resolution for a service |

**Netflix chaos tools:**
- **Chaos Monkey:** Randomly kills instances in production.
- **Chaos Kong:** Evacuates entire AWS regions.
- **Latency Monkey:** Induces artificial delays.
- **Conformity Monkey:** Finds instances that don't adhere to best practices.

**Chaos engineering in design:**
The goal is not just to run experiments post-deployment — it's to design systems that survive chaos. Key architectural patterns chaos engineering validates:

1. **Circuit breakers** — do they open when a dependency degrades?
2. **Fallbacks and graceful degradation** — does the system serve degraded results instead of errors?
3. **Retry with exponential backoff** — does retry logic create thundering herds?
4. **Bulkheads** — does a failure in Service A isolate from Service B?
5. **Health checks and auto-recovery** — does the orchestrator detect and replace failed instances?

**Starting small:**
```
Chaos maturity levels:
  Level 1: Run experiments in dev/staging (no production risk)
  Level 2: Run in production during business hours (engineers watching)
  Level 3: Automated chaos in production continuously
  Level 4: Automated chaos + automated detection + automated remediation
```

**Key insight:** Chaos engineering reveals that distributed systems fail in ways that are not predicted by reading the code. Network partitions, cascading failures, thundering herds, and resource exhaustion manifest differently in production than in testing. The only way to build true confidence is to prove the system survives failure empirically.

---

## Quick Reference

### CAP Theorem
| Choice | Behavior During Partition | Systems |
|--------|--------------------------|---------|
| CP | Refuse requests (error) | ZooKeeper, HBase, Spanner |
| AP | Serve stale data | Cassandra, DynamoDB, Riak |
| Note | P is not optional — partition will happen | Design for C vs A trade-off |

### PACELC Summary
| System | During Partition | During Normal Ops | Type |
|--------|-----------------|-------------------|------|
| DynamoDB default | Availability | Low Latency | PA/EL |
| Cassandra | Availability | Low Latency | PA/EL |
| ZooKeeper | Consistency | Consistency | PC/EC |
| Spanner | Consistency | Consistency | PC/EC |

### Consistency Models (Strongest → Weakest)
1. Linearizability — global real-time ordering
2. Sequential — program order preserved
3. Causal — causally related ops ordered
4. Eventual — converges eventually

### Raft Roles
| Role | Behavior |
|------|---------|
| Leader | Handles all writes, sends heartbeats |
| Follower | Passive, responds to leader/candidate |
| Candidate | Requesting votes to become leader |

### Quorum Formula
```
R + W > N  =>  strong consistency
N = replicas, W = write quorum, R = read quorum
Common: N=3, W=2, R=2
```

### Consistent Hashing
```
Virtual nodes: each physical node owns V positions on ring
Adding node: takes keys from ALL nodes (not just neighbors)
Removing node: distributes keys to ALL remaining nodes
Heterogeneous nodes: larger nodes get more vnodes
```

### Clock Types
| Clock | Detects Concurrent? | Space | Used In |
|-------|---------------------|-------|---------|
| Lamport timestamp | No | O(1) | Logging, ordering |
| Vector clock | Yes | O(N) | DynamoDB, conflict detection |

### Split-Brain Prevention
1. Majority quorum (most robust)
2. Fencing tokens (monotonic)
3. STONITH (force-kill isolated node)
4. Odd number of nodes always

### Bloom Filter
```
False positive possible: YES
False negative possible: NO
Space: O(m) where m = bit array size
Tune: k hash functions, m/n bits per element
```
