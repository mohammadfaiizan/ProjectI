# 29 — Concurrency and Parallelism

---

## Easy (Q1–Q7)

---

### Q1. What is the difference between concurrency and parallelism?

This is one of the most commonly misunderstood distinctions in systems design. Rob Pike (Go creator) captures it precisely: **Concurrency is about dealing with lots of things at once. Parallelism is about doing lots of things at once.**

**Concurrency:**
- A system is concurrent if it is designed to handle multiple tasks that are in-progress simultaneously — even if only one is executing at any instant.
- The tasks interleave on a single processor via context switching.
- Concurrency is a **design property** — how you structure your program.

```
Single CPU, two tasks (A and B):
Time: 1  2  3  4  5  6  7  8
CPU:  A  A  B  B  A  B  A  B
      (interleaved — concurrent but not parallel)
```

**Parallelism:**
- A system is parallel if multiple tasks are literally executing at the same time on multiple processors or cores.
- Parallelism requires multiple hardware execution units.
- Parallelism is a **runtime property** — how your program actually runs.

```
Two CPUs, two tasks:
Time: 1  2  3  4
CPU1: A  A  A  A
CPU2: B  B  B  B
      (simultaneously — parallel)
```

**Key insight:** You can have concurrency without parallelism (cooperative multitasking on a single core) and parallelism without concurrency (SIMD vectorisation on independent data). A concurrent program **can** exploit parallelism on multi-core hardware.

**Practical implications for system design:**

| Scenario | Technique | Why |
|---|---|---|
| Web server handling 1000 connections | Concurrency (async I/O) | Most time is waiting for I/O, not computing |
| Image encoding 10,000 files | Parallelism (thread pool) | CPU-bound; needs multiple cores |
| Database query planner | Both | Concurrent query scheduling, parallel query execution |

Node.js and Python asyncio provide **concurrency without parallelism** via event loops. Go's goroutines provide **concurrency with parallelism** — the runtime schedules goroutines across available CPUs.

---

### Q2. What are the resource trade-offs between threads, processes, and coroutines?

These three abstractions offer different trade-offs in memory, isolation, and context-switch cost.

**Threads:**
- Lightweight unit of execution within a process.
- Shared memory space with other threads in the same process.
- OS manages scheduling; context switch involves kernel mode transition.

```
Process (100 MB)
├── Thread 1 (1 MB stack)   ← shared heap, file descriptors
├── Thread 2 (1 MB stack)
└── Thread 3 (1 MB stack)
```

**Processes:**
- Fully isolated execution context — separate memory space, file descriptors, signal handlers.
- IPC (pipes, sockets, shared memory) required for communication.
- Context switch is expensive: full memory map switch.

**Coroutines (green threads / fibers):**
- User-space concurrency — no OS involvement in scheduling.
- Cooperative (yield explicitly) or scheduled by a user-space runtime.
- Extremely lightweight: 2–8 KB stack vs 1–8 MB for OS threads.

**Resource comparison:**

| Resource | OS Thread | OS Process | Coroutine |
|---|---|---|---|
| Stack size | 1–8 MB | 1–8 MB | 2–64 KB |
| Creation time | ~10 µs | ~100 µs | ~1 µs |
| Context switch | ~1–10 µs (kernel) | ~10–100 µs | ~100 ns (user space) |
| Max count (typical) | ~10,000 | ~1,000 | ~1,000,000 |
| Memory isolation | No | Yes | No |
| Crash isolation | No | Yes | No |
| Parallelism | Yes (multi-core) | Yes (multi-core) | Runtime-dependent |

**Concurrency at scale:**
- Go: 1M goroutines is feasible (2 KB initial stack, grown dynamically).
- Python: 10K asyncio coroutines is common; threads are heavier.
- Java: Virtual threads (Project Loom, Java 21) bring coroutine-like lightweight threads to the JVM.

**When to use each:**
- **Threads**: CPU-bound workloads, shared in-memory data structures, compute parallelism.
- **Processes**: Strong isolation required (security, fault containment), different languages/runtimes.
- **Coroutines**: I/O-bound workloads with high concurrency (web servers, network proxies).

---

### Q3. How do you size a thread pool for CPU-bound vs I/O-bound workloads?

Thread pool sizing is one of the most impactful performance tuning decisions. The optimal size depends on whether work is CPU-bound (computation) or I/O-bound (waiting for network/disk).

**CPU-bound workloads:**
- Threads spend time computing, not waiting.
- Having more threads than CPU cores causes context-switching overhead without throughput benefit.
- Formula: `thread_count = N_cores` or `N_cores + 1` (the +1 handles the occasional I/O wait).

```
8-core machine, image encoding job:
  Optimal thread pool: 8–9 threads
  With 16 threads: CPU thrashing, context switches, slower than 8
  With 4 threads: 4 cores idle — half throughput wasted
```

**I/O-bound workloads:**
- Threads spend most time waiting (network, disk, database).
- While waiting, the CPU is available for other threads.
- More threads than cores is beneficial — fill CPU idle time with other threads' compute.
- Formula: `thread_count = N_cores × (1 + wait_time / compute_time)`

```
Example: API call to database takes 50ms, processes result in 5ms
Wait ratio: 50ms / 5ms = 10

thread_count = 8 cores × (1 + 10) = 88 threads

This keeps all 8 cores busy servicing ~88 concurrent database calls
```

**Little's Law approach:**
```
L = λ × W
L = number of threads in the pool
λ = requests/second
W = average request latency in seconds

At 100 req/s with 200ms average latency:
L = 100 × 0.2 = 20 threads needed
```

**Practical sizing:**

| Workload Type | Formula | Example |
|---|---|---|
| CPU-bound | N_cores | 8-core → 8 threads |
| Lightly I/O-bound | N_cores × 2 | 8-core → 16 threads |
| Heavily I/O-bound | N_cores × (1 + wait/compute) | 8 × 11 = 88 threads |
| Database pool | Derived from DB connection limit | Min(100, db_max_connections × 0.8) |

**Warning signs of incorrect sizing:**
- CPU < 50% but threads blocked → pool too small, I/O-bound.
- CPU > 90% with many threads → too many threads competing, CPU-bound pool too large.
- High context-switch rate (`vmstat cs` column) → too many threads.

---

### Q4. Compare mutex, semaphore, and read-write lock. When should you use each?

These three primitives solve different concurrency control problems. Choosing the wrong one leads to either deadlocks, excessive serialization, or incorrect synchronization.

**Mutex (Mutual Exclusion Lock):**
- Binary lock — either locked or unlocked.
- Only the thread that acquired the lock can release it.
- Used to protect a critical section from concurrent access.

```python
import threading
lock = threading.Lock()

def update_counter():
    with lock:          # Acquire
        counter += 1   # Critical section
    # Release (on exit from 'with' block)
```

**Semaphore:**
- Generalisation of mutex with a count (allows N threads simultaneously).
- Any thread can release (not just the acquirer) — used for signalling.
- `Semaphore(1)` == binary semaphore == similar to mutex (but without ownership semantics).

```python
# Limit concurrent database connections to 10
db_semaphore = threading.Semaphore(10)

def query_database(sql):
    with db_semaphore:   # Blocks if 10 threads already inside
        return db.execute(sql)
```

**Read-Write Lock (RWLock):**
- Multiple readers allowed simultaneously.
- Exclusive access for writers (blocks all readers and other writers).
- Ideal for data structures read frequently but written rarely.

```python
import threading
rwlock = threading.RWLock()

def read_user_profile(user_id):
    with rwlock.read():       # Multiple readers allowed
        return cache[user_id]

def update_user_profile(user_id, data):
    with rwlock.write():      # Exclusive access
        cache[user_id] = data
        db.update(user_id, data)
```

**Comparison table:**

| Primitive | Concurrency | Ownership | Use case |
|---|---|---|---|
| Mutex | 1 thread at a time | Acquirer must release | Protecting counter, linked list node |
| Semaphore | N threads at a time | Any thread can release | Rate limiting, producer-consumer signalling |
| RW Lock | N readers OR 1 writer | Separate read/write owners | Config cache, in-memory index |

**When RW lock is better than mutex:**
- Read:write ratio > 10:1 — a mutex forces all readers to queue behind each other unnecessarily.
- With RW lock, 100 concurrent readers proceed without blocking each other.

---

### Q5. What is a deadlock? What are the four necessary conditions (CLAD) and how do you prevent them?

A **deadlock** is a state where two or more threads are each waiting for a resource held by another thread in the group, and none can proceed. Deadlocks do not crash programs — they cause them to hang silently, often detectable only by monitoring thread states.

**The four necessary conditions (all must be present for deadlock):**

```
C - Circular Wait:    A waits for B's lock, B waits for A's lock
L - Lock holding:     A thread holds at least one lock while requesting another
A - Acquire & hold:   No preemption — locks not forcibly taken
D - Denial of sharing: Each resource can only be held by one thread at a time
```

**Classic deadlock example:**
```python
lock_A = threading.Lock()
lock_B = threading.Lock()

def thread_1():
    with lock_A:          # Acquires A
        time.sleep(0.1)
        with lock_B:      # Waits for B (Thread 2 holds B)
            do_work()

def thread_2():
    with lock_B:          # Acquires B
        time.sleep(0.1)
        with lock_A:      # Waits for A (Thread 1 holds A)
            do_work()
```

**Prevention strategies (break one of the four conditions):**

**1. Lock ordering (breaks Circular Wait) — most common:**
```python
def transfer(account_a, account_b, amount):
    # Always acquire locks in consistent order (by account ID)
    first = min(account_a.id, account_b.id)
    second = max(account_a.id, account_b.id)
    with get_lock(first):
        with get_lock(second):
            account_a.balance -= amount
            account_b.balance += amount
```

**2. Lock timeout (breaks Acquire & Hold):**
```python
if lock_A.acquire(timeout=5):
    if lock_B.acquire(timeout=5):
        # Do work
        lock_B.release()
    lock_A.release()
else:
    # Retry after backoff
```

**3. Try-lock without blocking (breaks Lock Holding):**
```python
if lock_A.acquire(blocking=False) and lock_B.acquire(blocking=False):
    # Do work
else:
    # Release any partial locks and retry
```

**4. Single global lock (breaks need for multiple locks — drastic, serialises everything).**

**Detection (when prevention isn't fully applied):**
- Database systems detect deadlocks via wait-for graph cycle detection.
- PostgreSQL: automatically detects and kills one of the deadlocked transactions.
- Java: thread dump shows threads in `BLOCKED` state waiting for each other.

---

### Q6. What are race conditions in distributed systems? Give examples and solutions.

A **race condition** occurs when the correct behaviour of a system depends on the relative ordering or timing of concurrent operations, and that ordering is not guaranteed. In distributed systems, race conditions are more insidious because they involve multiple services, databases, and network latency.

**Example 1: Double-spend (banking):**
```
Account balance: $100

Thread A: reads balance = $100
Thread B: reads balance = $100
Thread A: balance - $80 = $20, writes $20
Thread B: balance - $80 = $20, writes $20
Result: $160 withdrawn from a $100 account

Solution: SELECT FOR UPDATE (pessimistic lock) or optimistic lock (version check)
```

**Example 2: Lost update (distributed cache):**
```
Redis key "inventory:item_x" = 100

Service A: reads 100, calculates 100 - 10 = 90
Service B: reads 100, calculates 100 - 5 = 95
Service A: writes 90
Service B: writes 95 (overwrites A's update — 10 units lost)

Solution: Redis WATCH + MULTI/EXEC (optimistic transaction) or DECRBY (atomic operation)
```

**Example 3: Time-of-check vs time-of-use (TOCTOU):**
```
Booking system: "Is seat 14A available?"
User A: checks seat → available
User B: checks seat → available
User A: books seat 14A ✓
User B: books seat 14A ✓ (double booking!)

Solution: Database row lock (SELECT FOR UPDATE on the seat row) or optimistic lock
```

**Example 4: Event ordering in distributed systems:**
```
Order service: publishes OrderCreated at T=100
Inventory service: publishes InventoryReserved at T=101
Notification service: receives InventoryReserved before OrderCreated (network reordering)
→ Cannot process InventoryReserved without the OrderCreated context

Solution: Event ordering by entity key in Kafka (same partition for same order_id)
         or causal ordering with vector clocks
```

**Solution patterns summary:**

| Race Condition Type | Solution |
|---|---|
| Shared counter | Atomic operations (CAS, INCR, DECR) |
| Database read-modify-write | SELECT FOR UPDATE or optimistic locking |
| Distributed counter | Redis INCR (atomic), or CRDT |
| Event ordering | Partition by entity key; vector clocks |
| Check-then-act | Make check and act atomic (DB transaction, CAS) |

---

### Q7. What is compare-and-swap (CAS) and how do lock-free data structures use it?

**Compare-and-swap (CAS)** is a hardware-level atomic instruction that performs the following sequence as an indivisible operation:

```
CAS(memory_location, expected_value, new_value):
  if *memory_location == expected_value:
      *memory_location = new_value
      return SUCCESS
  else:
      return FAILURE (current value returned)
```

Because CAS is a single CPU instruction (`CMPXCHG` on x86), it cannot be interrupted — there is no window for another thread to observe an intermediate state.

**Lock-free counter using CAS:**
```python
import ctypes

def atomic_increment(counter):
    while True:
        current = counter.value
        if counter.compare_and_swap(current, current + 1):
            return current + 1
        # CAS failed: another thread changed counter between read and CAS
        # Retry (spin)
```

**Lock-free stack (Treiber stack):**
```
push(value):
  new_node = Node(value)
  while True:
      top = stack.head              # Read current head
      new_node.next = top           # New node points to current head
      if CAS(stack.head, top, new_node):   # Atomically update head
          return                    # Success
      # CAS failed: another push happened — retry

pop():
  while True:
      top = stack.head
      if top is None: return None
      if CAS(stack.head, top, top.next):   # Atomically remove head
          return top.value
```

**Why CAS-based lock-free structures are useful:**
- No mutex: eliminates lock acquisition cost and deadlock risk.
- Progress guarantee: at least one thread always makes progress (obstruction-free to lock-free).
- Better cache performance: no lock state to invalidate across CPU caches.

**The ABA problem:**
```
Thread 1 reads head = A
Thread 2: pops A, pops B, pushes A back → head is A again
Thread 1: CAS(head, A, new_node) → SUCCEEDS (head is A)
But: A now points to nothing useful (B was popped between reads)
→ Corruption

Solution: Tagged pointers — attach a version counter to each pointer
  CAS((head, version), (A, 5), (new_node, 6))
  Thread 2 would change version to 7, so CAS would fail
```

Java's `AtomicReference`, `AtomicInteger`, and `AtomicLong` all use CAS under the hood via `sun.misc.Unsafe`.

---

## Medium (Q8–Q15)

---

### Q8. How does the Actor model avoid shared state? Compare to traditional threading.

The **Actor model** is a concurrency paradigm where the fundamental unit is an **actor** — an independent entity with its own private state that communicates exclusively through asynchronous message passing. No shared memory, no locks.

**Actor model principles:**
1. Each actor has its own private state — no other actor can access it directly.
2. Actors communicate only by sending immutable messages.
3. Upon receiving a message, an actor can: process it, create new actors, send messages, or change its own state.
4. Messages are processed one at a time per actor — no intra-actor concurrency.

```
Traditional OOP + Threads:            Actor Model:
┌──────────────────┐                  ┌──────────────────┐
│ Shared counter   │                  │  Actor A         │
│ int value = 0    │                  │  (private: n=0)  │
└────────┬─────────┘                  └────────┬─────────┘
  Thread1─┘─writes (need lock)          sends message
  Thread2─┘─writes (need lock)          "increment"
  Lock contention, deadlock risk        ▼
                                  ┌──────────────────┐
                                  │  Actor B          │
                                  │  processes msg    │
                                  │  n += 1 (safe)    │
                                  └──────────────────┘
                                  (no locks needed)
```

**Akka actor example (Scala):**
```scala
import akka.actor.{Actor, Props}

class CounterActor extends Actor {
  var count = 0   // Private state — not accessible from outside

  def receive = {
    case "increment" => count += 1
    case "get"       => sender() ! count   // Reply with current count
    case "reset"     => count = 0
  }
}

// Usage:
val counter = system.actorOf(Props[CounterActor], "counter")
counter ! "increment"    // Fire and forget
counter ! "increment"
val future = (counter ? "get").mapTo[Int]  // Ask pattern — awaits reply
```

**Comparison:**

| Aspect | Traditional Threads | Actor Model |
|---|---|---|
| State sharing | Shared mutable state | No sharing — message passing only |
| Synchronisation | Locks, monitors | None (actors process one message at a time) |
| Deadlock risk | High (lock ordering required) | Eliminated by design |
| Debugging | Hard (non-deterministic) | Easier (message log reveals execution order) |
| Fault tolerance | Thread crash can affect others | Each actor can be supervised independently |
| Overhead | Thread stack (~1 MB) | Actor (~300 bytes in Akka) |

Erlang's entire concurrency model is actor-based — this is why Erlang processes (actors) can number in the millions and why systems like WhatsApp (built on Erlang) handle 2M+ connections per server.

---

### Q9. How does async/await work in Node.js? How does an event loop enable concurrency without threads?

Node.js achieves high I/O concurrency with a **single-threaded event loop** — no thread creation, no context switching, no locks. This model is highly effective for I/O-bound workloads.

**The event loop:**
```
┌─────────────────────────────────────────────────────────┐
│                    Event Loop                            │
│                                                          │
│  1. timers:        setTimeout, setInterval callbacks    │
│  2. I/O callbacks: completed I/O operations             │
│  3. idle, prepare: internal use                         │
│  4. poll:          retrieve new I/O events (blocks here)│
│  5. check:         setImmediate callbacks               │
│  6. close callbacks: socket/file close events           │
└─────────────────────────────────────────────────────────┘
```

**How non-blocking I/O works:**
```javascript
// Synchronous (blocking — bad for concurrency):
const data = fs.readFileSync('/large-file');  // Thread blocked here
processData(data);

// Asynchronous (non-blocking — good):
fs.readFile('/large-file', (err, data) => {
    processData(data);         // Called when I/O completes
});
// Event loop continues here immediately — not waiting
```

**async/await as syntactic sugar over Promises:**
```javascript
// Promise-based:
function fetchUserOrders(userId) {
    return fetchUser(userId)
        .then(user => fetchOrders(user.id))
        .then(orders => computeTotal(orders))
        .catch(err => handleError(err));
}

// async/await (same behaviour, more readable):
async function fetchUserOrders(userId) {
    const user = await fetchUser(userId);     // Suspends, yields control
    const orders = await fetchOrders(user.id); // Suspends again
    return computeTotal(orders);
}
```

When `await` encounters a Promise, it:
1. Suspends the current async function (does NOT block the thread).
2. Returns control to the event loop.
3. Resumes the function when the Promise resolves.

**Concurrency via Promise.all:**
```javascript
// Sequential (slow — waits for each):
const user = await fetchUser(userId);      // 50ms
const prefs = await fetchPrefs(userId);    // 50ms
// Total: 100ms

// Concurrent (fast — both in flight simultaneously):
const [user, prefs] = await Promise.all([
    fetchUser(userId),    // 50ms
    fetchPrefs(userId)    // 50ms — started at same time
]);
// Total: ~50ms (limited by slower of the two)
```

**Node.js limitation:** CPU-intensive work (image encoding, JSON parsing of large files) **blocks the event loop** — while the CPU is computing, no other callbacks can execute. Solution: `worker_threads` for CPU-bound work, keeping the main event loop free for I/O.

---

### Q10. How do databases handle concurrent writes? Cover MVCC, row-level locking, and SELECT FOR UPDATE.

Databases face the same concurrency challenges as application code but must maintain ACID guarantees across all concurrent transactions. Three key mechanisms address this.

**MVCC (Multi-Version Concurrency Control):**
- Instead of locking rows during reads, the database keeps **multiple versions** of each row.
- Each transaction sees a snapshot of the database as of its start time.
- Readers never block writers; writers never block readers.

```
Timeline:
  T=100: Row "Alice" = {balance: 1000, version: 1}
  T=101: Transaction A starts (snapshot = T=100 state)
  T=102: Transaction B updates Alice.balance = 800, commits
         Row now has: {balance: 800, version: 2} AND {balance: 1000, version: 1} (old)
  T=103: Transaction A reads Alice.balance → sees 1000 (version: 1, before its start)
  T=104: Transaction A commits → no conflict (it only read, did not write)
```

PostgreSQL, MySQL InnoDB, Oracle, SQL Server all use MVCC.

**Row-level locking:**
- A transaction acquires a lock on specific rows it intends to modify.
- Other transactions can read the row (MVCC serves their snapshot) but cannot acquire conflicting locks.

```sql
-- Transaction A:
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 'alice';
-- Row lock on alice's row acquired until COMMIT or ROLLBACK

-- Transaction B (concurrent):
UPDATE accounts SET balance = balance + 100 WHERE id = 'bob';
-- No conflict — different row, different lock
```

**SELECT FOR UPDATE (pessimistic locking):**
- Explicitly acquires a write lock on selected rows before any modification.
- Prevents phantom reads and lost updates in read-modify-write patterns.

```sql
-- Safe transfer: lock both rows before reading balances
BEGIN;
SELECT balance FROM accounts WHERE id IN ('alice', 'bob') FOR UPDATE;
-- Now locked: no other transaction can modify alice or bob until this commits

UPDATE accounts SET balance = balance - 100 WHERE id = 'alice';
UPDATE accounts SET balance = balance + 100 WHERE id = 'bob';
COMMIT;
```

**`SELECT FOR UPDATE SKIP LOCKED` (job queues):**
```sql
-- Worker claims next available job — other workers skip locked rows
SELECT id, payload FROM job_queue
WHERE status = 'pending'
ORDER BY created_at
LIMIT 1
FOR UPDATE SKIP LOCKED;
```
This allows multiple workers to claim different jobs from the same table concurrently without blocking each other.

---

### Q11. What is connection pool exhaustion? How do you prevent it?

**Connection pool exhaustion** occurs when all database connections in the pool are in use and new requests must wait or fail. It is a common production failure mode that causes cascading timeouts across the entire application tier.

**How it happens:**
```
Connection pool: 20 connections

Normal operation: 15 connections in use, 5 free → healthy

Traffic spike or slow query:
  Query A: takes 10s (table scan due to missing index)
  20 concurrent requests × 10s each = 200 connection-seconds needed
  Pool only has 20 connections = pool exhausted in < 1s
  
Requests 21-100: wait in queue for a connection
  → Timeout after 30s (connection checkout timeout)
  → HTTP 500 errors returned to users
  → More requests pile up → queue grows → memory issues → service crash
```

**Prevention strategies:**

**1. Right-size the connection pool:**
```python
# The formula (PgBouncer / HikariCP):
# pool_size = (core_count × 2) + effective_spindle_count
# For PostgreSQL on 8-core server with SSD: 8 × 2 + 1 = 17 connections

# HikariCP (Java):
HikariConfig config = new HikariConfig();
config.setMaximumPoolSize(17);           # Max connections
config.setMinimumIdle(5);               # Keep 5 warm
config.setConnectionTimeout(3000);      # 3s checkout timeout
config.setIdleTimeout(600000);          # Remove idle after 10min
config.setMaxLifetime(1800000);         # Replace connection after 30min
```

**2. Use a connection pooler (PgBouncer for PostgreSQL):**
```
Without PgBouncer:
  1000 app servers × 10 connections each = 10,000 connections to PostgreSQL
  PostgreSQL max_connections = 500 → overloaded

With PgBouncer (transaction pooling):
  1000 app servers → PgBouncer (1000 client connections accepted)
  PgBouncer → PostgreSQL (50 server connections maintained)
  PgBouncer reuses a server connection per transaction, not per client session
```

**3. Set query timeouts to release connections:**
```sql
-- PostgreSQL: kill queries taking > 5s
SET statement_timeout = '5s';

-- Application-level:
cursor.execute("SET statement_timeout TO '5000'")
```

**4. Monitor pool metrics:**
```
Key metrics to alert on:
  - pool_connections_waiting > 0  → pool pressure building
  - pool_connections_waiting > 5 for > 30s → near exhaustion
  - checkout_timeout rate > 0     → pool already exhausted
  
Set alerting thresholds before exhaustion, not after.
```

**5. Circuit breaker on DB calls:**
- If DB call failure rate > 50% for 10 seconds → open circuit → return error immediately rather than holding connections waiting.

---

### Q12. Explain the producer-consumer pattern with a bounded queue. Why is bounding essential?

The **producer-consumer pattern** decouples work production from work processing. A bounded queue between producers and consumers limits memory usage and applies back-pressure when consumers fall behind.

**Architecture:**
```
Producers                 Bounded Queue              Consumers
┌──────────┐              ┌───────────┐              ┌──────────┐
│ Producer │──enqueue()──▶│ [||||   ] │──dequeue()──▶│ Worker 1 │
│ Thread 1 │              │ capacity=N│              └──────────┘
└──────────┘              └───────────┘              ┌──────────┐
┌──────────┐                 (blocks when full)      │ Worker 2 │
│ Producer │──enqueue()──▶                           └──────────┘
│ Thread 2 │
└──────────┘
```

**Python implementation with bounded queue:**
```python
import queue
import threading

QUEUE_SIZE = 100  # Bounded: max 100 items

work_queue = queue.Queue(maxsize=QUEUE_SIZE)

def producer(num_items):
    for i in range(num_items):
        item = produce_item(i)
        work_queue.put(item, block=True, timeout=5)
        # Blocks here if queue is full → BACK-PRESSURE applied to producer

def consumer():
    while True:
        try:
            item = work_queue.get(block=True, timeout=1)
            process_item(item)
            work_queue.task_done()
        except queue.Empty:
            break  # No more items, worker exits

# Start 5 consumer threads
consumers = [threading.Thread(target=consumer) for _ in range(5)]
for t in consumers:
    t.start()

# Start producer
producer(1_000_000)

# Wait for all items to be processed
work_queue.join()
```

**Why bounding is essential:**

```
Unbounded queue scenario:
  Producer generates 100,000 items/second
  Consumer processes 50,000 items/second
  Queue grows: +50,000 items/second
  
  After 60 seconds: 3,000,000 items in memory
  At 1 KB per item: 3 GB memory consumed
  JVM/Python process OOM killed → all queued work lost
```

**Back-pressure:** When the queue is full, `queue.put(block=True)` **blocks the producer**. This is intentional — it signals to the producer that consumers are overwhelmed, slowing the entire pipeline to the sustainable rate of the slowest stage.

**Monitoring a bounded queue:**
- Queue depth approaching `maxsize` → add consumers or reduce producer rate.
- Producer blocks often → consumers are the bottleneck (increase workers or optimise processing).
- Queue always empty → consumers are faster than producers (reduce worker count to save resources).

---

### Q13. What is the thundering herd problem? How do you solve mutex contention at scale?

The **thundering herd** problem occurs when a large number of processes or threads are woken up simultaneously to compete for a resource, but only one can proceed — the rest must go back to sleep. This causes a burst of CPU and context-switch overhead that can temporarily degrade performance.

**Cache stampede (most common form in web systems):**
```
Scenario: 10,000 concurrent users requesting the same product page
Cache TTL expires at 12:00:00

At 12:00:00:
  All 10,000 in-flight requests miss cache simultaneously
  All 10,000 threads try to query the database
  Database: receives 10,000 identical queries in < 1 second
  Result: database overwhelmed, query latency spikes, cascading timeout
```

**Solutions:**

**1. Mutex/Lock-based cache stampede prevention:**
```python
import threading
import time

cache = {}
locks = {}
locks_mutex = threading.Lock()

def get_product(product_id):
    if product_id in cache:
        return cache[product_id]
    
    # Only one thread should fetch; others wait
    with locks_mutex:
        if product_id not in locks:
            locks[product_id] = threading.Lock()
    
    lock = locks[product_id]
    
    with lock:
        # Double-check: another thread may have populated cache while we waited
        if product_id in cache:
            return cache[product_id]
        
        # This thread fetches the data
        result = db.fetch_product(product_id)
        cache[product_id] = result
        return result
```

**2. Probabilistic early expiration:**
```python
def get_with_early_expiry(key, beta=1.0):
    value, expiry = cache.get(key)
    current_time = time.time()
    
    # Stochastically recompute before expiry
    # Higher beta = more aggressive early refresh
    if current_time - beta * math.log(random.random()) >= expiry:
        # Proactively refresh before others
        value = recompute(key)
        cache.set(key, value, ttl=300)
    
    return value
```

**3. Striped locks (reduce contention for high-throughput scenarios):**
```java
// Instead of one lock for the entire cache:
Lock[] stripes = new Lock[256];  // 256 independent locks
for (int i = 0; i < 256; i++) stripes[i] = new ReentrantLock();

Lock getLock(String key) {
    return stripes[Math.abs(key.hashCode()) % 256];
}
// Keys hash to different stripes → 256x reduction in contention
// Used by Java's ConcurrentHashMap internally
```

**4. Background refresh with stale-while-revalidate:**
```python
def get_product(product_id):
    value, ttl = cache.get(product_id)
    
    if ttl < 30:  # Less than 30s remaining
        # Trigger async refresh (non-blocking)
        threading.Thread(target=refresh_product, args=[product_id]).start()
    
    return value  # Return current value immediately (possibly slightly stale)
```

This ensures the cache is always pre-populated before expiry — the thundering herd never forms.

---

### Q14. Explain transaction isolation levels and their concurrency trade-offs.

Transaction isolation levels define which concurrency anomalies are permitted for a transaction. Higher isolation prevents more anomalies but requires more locking, reducing concurrency.

**The anomalies:**
- **Dirty read**: Reading uncommitted data from another transaction.
- **Non-repeatable read**: Reading a row twice in one transaction yields different values (another transaction committed a change between reads).
- **Phantom read**: A re-executed query returns different rows (another transaction inserted/deleted rows matching the WHERE clause).
- **Serialization anomaly**: Results inconsistent with any serial ordering of transactions.

**Isolation levels (SQL standard, PostgreSQL/MySQL):**

| Level | Dirty Read | Non-Repeatable Read | Phantom Read | Concurrency |
|---|---|---|---|---|
| READ UNCOMMITTED | Possible | Possible | Possible | Highest |
| READ COMMITTED | Prevented | Possible | Possible | High |
| REPEATABLE READ | Prevented | Prevented | Possible* | Medium |
| SERIALIZABLE | Prevented | Prevented | Prevented | Lowest |

(*PostgreSQL's REPEATABLE READ also prevents phantom reads via MVCC snapshots)

**Practical guidance:**

```sql
-- READ COMMITTED (default in PostgreSQL, Oracle):
-- Good for: OLTP, most web application queries
-- Risk: non-repeatable reads in multi-statement transactions
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- REPEATABLE READ (default in MySQL InnoDB):
-- Good for: reports that re-read rows, consistent aggregate calculations
-- Risk: phantom reads (new rows can appear)
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- SERIALIZABLE:
-- Good for: financial calculations, inventory management, anything requiring true isolation
-- Cost: highest lock contention, possible serialization errors requiring retry
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

**SERIALIZABLE implementation — SSI (Serializable Snapshot Isolation):**
PostgreSQL uses SSI rather than strict two-phase locking. SSI tracks read/write dependencies and aborts transactions only when a dependency cycle is detected — much less blocking than traditional SERIALIZABLE.

**Choosing isolation level:**
```
Low conflict + high concurrency → READ COMMITTED
Report/analytics (consistent snapshot) → REPEATABLE READ
Financial operations → SERIALIZABLE
Interactive user queries (fastest) → READ COMMITTED + idempotent retry logic
```

---

### Q15. How do you design a concurrent rate limiter using atomic operations in Redis?

A **distributed rate limiter** must handle concurrent requests from multiple application servers accurately without race conditions. Redis provides atomic operations that eliminate the need for distributed locks.

**Algorithm: Sliding window log (accurate, higher memory):**
```python
import time
import redis

r = redis.Redis()

def is_rate_limited(user_id, max_requests, window_seconds):
    now = time.time()
    window_start = now - window_seconds
    key = f"rate_limit:{user_id}"
    
    # MULTI/EXEC ensures all operations are atomic
    pipe = r.pipeline()
    
    # Remove old entries outside the window
    pipe.zremrangebyscore(key, 0, window_start)
    
    # Count requests in current window
    pipe.zcard(key)
    
    # Add current request (score = timestamp)
    pipe.zadd(key, {str(now): now})
    
    # Set TTL to avoid stale keys
    pipe.expire(key, window_seconds)
    
    results = pipe.execute()
    
    request_count = results[1]   # Count before adding current request
    
    return request_count >= max_requests  # True = rate limited
```

**Algorithm: Fixed window counter (simpler, slight boundary issue):**
```python
def is_rate_limited_fixed(user_id, max_requests, window_seconds):
    window_key = int(time.time() // window_seconds)
    key = f"rate_limit:{user_id}:{window_key}"
    
    # INCR is atomic: increment and return new value in one operation
    current_count = r.incr(key)
    
    if current_count == 1:
        # First request in this window: set TTL
        r.expire(key, window_seconds)
    
    return current_count > max_requests
```

**Algorithm: Token bucket (smooth bursting, most production-friendly):**
```lua
-- Lua script executes atomically in Redis
local key = KEYS[1]
local max_tokens = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])   -- tokens per second
local now = tonumber(ARGV[3])

local data = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(data[1]) or max_tokens
local last_refill = tonumber(data[2]) or now

-- Refill tokens based on elapsed time
local elapsed = now - last_refill
local new_tokens = math.min(max_tokens, tokens + elapsed * refill_rate)

if new_tokens >= 1 then
    -- Token available: allow request
    redis.call('HMSET', key, 'tokens', new_tokens - 1, 'last_refill', now)
    redis.call('EXPIRE', key, 3600)
    return 1    -- allowed
else
    return 0    -- rate limited
end
```

**Comparison:**

| Algorithm | Memory | Accuracy | Burst handling | Complexity |
|---|---|---|---|---|
| Fixed window | O(1) | Boundary effect at window edge | Allows 2x burst at window edge | Simplest |
| Sliding window log | O(n requests) | Exact | Smooth | Medium |
| Token bucket | O(1) | Near-exact | Configurable burst | Medium |

For most APIs: **token bucket** is the best choice — configurable burst capacity, O(1) memory, and the Lua script ensures atomicity across the entire check-and-update.

---

## Hard (Q16–Q20)

---

### Q16. How does MVCC enable high-concurrency databases? Walk through a concurrent transaction scenario.

**MVCC (Multi-Version Concurrency Control)** is the key innovation that allows databases to serve high concurrency without readers blocking writers or vice versa. PostgreSQL, MySQL InnoDB, and Oracle all implement MVCC, each with slight variations.

**Core mechanics in PostgreSQL:**

Every row version has system columns:
- `xmin` — transaction ID that created this row version.
- `xmax` — transaction ID that deleted/updated this row version (0 if not deleted).
- `ctid` — physical location of the row.

```sql
-- Examine MVCC internal columns:
SELECT xmin, xmax, ctid, balance FROM accounts WHERE id = 'alice';
-- xmin=500, xmax=0 → created by txn 500, not yet deleted
```

**Concurrent transaction scenario:**

```
Initial state: alice.balance = 1000, xmin=100, xmax=0

T1 (txn_id=501): BEGIN; SELECT balance FROM accounts WHERE id='alice';
  → Reads row where xmin <= 501 AND xmax = 0 → sees 1000 ✓

T2 (txn_id=502): BEGIN; UPDATE accounts SET balance = 800 WHERE id='alice';
  → Creates new row version: {balance=800, xmin=502, xmax=0}
  → Marks old row: {balance=1000, xmin=100, xmax=502}
  T2: COMMIT;

T1 (still running): SELECT balance FROM accounts WHERE id='alice' again;
  → Reads rows visible at T1's snapshot time (before txn 502 started)
  → Old row: xmin=100 < 501, xmax=502 > 501 → VISIBLE (xmax is from a future txn)
  → New row: xmin=502 > 501 → NOT VISIBLE (created after T1's snapshot)
  → Sees 1000 (still, even though T2 committed 800)
```

This is REPEATABLE READ behaviour — T1 always sees the snapshot from its BEGIN time.

**Vacuum and row version cleanup:**
- Old row versions accumulate — `VACUUM` removes them once no active transaction can see them.
- `AUTOVACUUM` runs automatically in PostgreSQL.
- Heavy write workloads need tuned autovacuum — table bloat from dead rows is a common performance issue.

**MVCC write conflict (Serializable isolation):**
```
T1 and T2 both read and update the same row:
T1: reads alice.balance = 1000
T2: reads alice.balance = 1000
T2: sets balance = 800, commits
T1: sets balance = 900 (based on stale read)

Under REPEATABLE READ: T1 commits successfully (lost update — T2's change overwritten)
Under SERIALIZABLE (SSI): T1 detects that alice.balance was modified after T1's read
  → T1 aborted with: ERROR: could not serialize access due to concurrent update
  → Application retries T1 with fresh data
```

---

### Q17. Explain coroutine-based vs thread-based I/O. Compare throughput for a web server scenario.

This comparison reveals why Python asyncio, Node.js, and Nginx outperform thread-per-connection servers for I/O-bound workloads at high concurrency.

**Thread-based I/O (Apache httpd prefork, Java Tomcat blocking):**
```
Each request: 1 OS thread
Thread lifecycle:
  - Accept connection
  - Read request (blocks, waiting for network bytes)
  - Process request
  - Await database response (blocks, waiting for DB)
  - Write response
  - Thread released

1000 concurrent connections = 1000 OS threads
Thread stack: 1 MB each = 1 GB RAM just for stacks
Context switches: OS switches between 1000 threads, each adding ~5 µs overhead
Total context switch overhead at 1000 req/s: 1000 × 5µs = 5ms per second of CPU wasted
```

**Coroutine-based I/O (asyncio, Node.js, Nginx, Go):**
```
1 OS thread (or N worker threads, not 1 per connection)
Each request: 1 coroutine/goroutine (~2-8 KB stack)

await database_query()  → coroutine suspends, returns control to event loop
                           event loop services OTHER requests while waiting
                           resumes coroutine when DB response arrives

1000 concurrent connections = 1000 coroutines = ~4 MB RAM (vs 1 GB for threads)
Context switches: user-space only, ~100ns each (50x cheaper than OS thread switch)
```

**Throughput comparison for a web API (each request makes 1 DB call, 50ms DB latency):**

```
Thread-per-connection model:
  1000 threads × (50ms waiting + 5ms processing) = saturated at ~16,000 req/s
  Memory: 1000 × 1MB = 1 GB
  At 10,000 concurrent: 10 GB RAM, thread scheduling becomes bottleneck

Asyncio (Python uvicorn):
  1000 coroutines: 4 MB total memory
  All 1000 "in-flight" simultaneously — all waiting on their DB calls
  When DB responses arrive, event loop processes completions rapidly
  Theoretical throughput: 1000 coroutines / 50ms latency = 20,000 req/s per worker
  With 8 workers: 160,000 req/s  (vs 16,000 for thread model)
  
Go (goroutines):
  Goroutines scheduled across all CPU cores
  GOMAXPROCS=8 → 8 OS threads, millions of goroutines
  I/O-bound: goroutines multiplex onto OS threads when not blocked
  CPU-bound: goroutines use true parallelism via 8 OS threads
```

**Python asyncio benchmark (FastAPI vs Flask):**
```
Flask (thread-based, 4 workers, 4 threads each):
  1000 concurrent requests with 50ms sleep: ~320 req/s (4×4=16 threads)

FastAPI/uvicorn (asyncio, 4 workers):
  1000 concurrent requests with asyncio.sleep(0.05): ~12,000 req/s (75x faster)
```

**When threads still win:** CPU-bound workloads (image processing, ML inference). Python's GIL prevents true parallelism in asyncio; use multiprocessing or Go/Java for CPU-parallelism.

---

### Q18. Explain fork-join parallelism and how map-reduce applies it for CPU-bound work.

**Fork-join parallelism** is a pattern where a parent task **forks** into parallel subtasks, each subtask executes independently (possibly forking further), and then all results are **joined** back together.

**Abstract pattern:**
```
                     fork()
              ┌──────┼──────┐
              ▼      ▼      ▼
           Task1   Task2  Task3  ← execute in parallel
              │      │      │
              └──────┼──────┘
                    join()
                      │
                  Combine results
```

**Java ForkJoinPool example — parallel sum of large array:**
```java
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

class SumTask extends RecursiveTask<Long> {
    private final int[] array;
    private final int start, end;
    private static final int THRESHOLD = 10_000;

    public SumTask(int[] array, int start, int end) {
        this.array = array; this.start = start; this.end = end;
    }

    @Override
    protected Long compute() {
        if (end - start <= THRESHOLD) {
            // Base case: compute directly (sequential)
            long sum = 0;
            for (int i = start; i < end; i++) sum += array[i];
            return sum;
        }
        
        // Recursive case: fork two subtasks
        int mid = (start + end) / 2;
        SumTask left  = new SumTask(array, start, mid);
        SumTask right = new SumTask(array, mid, end);
        
        left.fork();                    // Execute left in background
        long rightResult = right.compute();  // Execute right in this thread
        long leftResult  = left.join();      // Wait for left to complete
        
        return leftResult + rightResult;
    }
}

// Usage:
ForkJoinPool pool = new ForkJoinPool(Runtime.getRuntime().availableProcessors());
int[] bigArray = new int[100_000_000];
long total = pool.invoke(new SumTask(bigArray, 0, bigArray.length));
```

**Map-Reduce as fork-join:**
```python
from concurrent.futures import ProcessPoolExecutor
import math

def count_words_in_file(filepath):
    """Map function: applied to each chunk in parallel"""
    with open(filepath) as f:
        return len(f.read().split())

def parallel_word_count(filepaths):
    """Fork-join word count across multiple files"""
    with ProcessPoolExecutor(max_workers=8) as executor:
        # Fork: submit all tasks in parallel
        futures = {executor.submit(count_words_in_file, f): f for f in filepaths}
        
        # Join: collect results as they complete
        total = 0
        for future in futures:
            total += future.result()
    
    return total
```

**Work stealing (ForkJoinPool's key optimisation):**
- Each thread maintains a deque of pending tasks.
- When a thread finishes its tasks, it **steals** tasks from the end of another thread's deque.
- This dynamically load-balances work across CPU cores without manual partitioning.

**When to use fork-join:**
- CPU-bound work that can be subdivided: sorting, searching, encoding, simulation.
- Work units are independent (no shared mutable state between subtasks).
- Subtask granularity: not too fine (overhead > work) and not too coarse (poor load balancing).

---

### Q19. How does lock contention affect throughput at scale? Solutions: striped locks and lock-free structures.

Lock contention is the primary throughput ceiling in concurrent systems. As the number of competing threads grows, time spent waiting for locks grows — often faster than linearly.

**Amdahl's Law applied to locking:**
```
If 10% of your workload is serialised under a lock:
  With 2 threads:  speedup = 1 / (0.9 + 0.1/2)   = 1.05x  (5% gain only)
  With 10 threads: speedup = 1 / (0.9 + 0.1/10)  = 1.09x  (9% gain)
  With 100 threads:speedup = 1 / (0.9 + 0.1/100) = 1.10x  (10% gain — plateau)
  
With 100% serialisation: no benefit from additional threads at all.
```

**Measuring lock contention:**
```bash
# Linux perf: show lock contention events
perf stat -e lock:contention_begin,lock:contention_end java -jar app.jar

# Java: log monitor contention time
java -XX:+PrintConcurrentLocks -jar app.jar

# Prometheus: track time spent waiting for locks
lock_wait_duration_seconds_histogram{lock="user_cache"}
```

**Striped locking:**
```java
// One lock for entire map → all operations serialise
Map<String, User> users = new HashMap<>();
Lock lock = new ReentrantLock();

// Striped locking → 256 independent locks, each protects 1/256 of keys
private static final int STRIPE_COUNT = 256;
private final Lock[] stripes = new Lock[STRIPE_COUNT];
private final Map<String, User>[] maps = new Map[STRIPE_COUNT];

Lock getLock(String key) {
    return stripes[Math.abs(key.hashCode() % STRIPE_COUNT)];
}
Map getMap(String key) {
    return maps[Math.abs(key.hashCode() % STRIPE_COUNT)];
}

void put(String key, User user) {
    Lock lock = getLock(key);
    lock.lock();
    try { getMap(key).put(key, user); }
    finally { lock.unlock(); }
}
// Thread contention reduced by ~256x for keys uniformly distributed
```

Java's `ConcurrentHashMap` uses segment-level striping (16 segments by default in Java 7, fully striped in Java 8+ via CAS operations).

**Lock-free counter for statistics:**
```java
// Bad: single AtomicLong causes cache line bouncing across CPUs
AtomicLong counter = new AtomicLong(0);

// Better: LongAdder uses striped approach internally
LongAdder counter = new LongAdder();
counter.increment();    // Updates a per-CPU cell, no contention
long total = counter.sum();  // Sum all cells at read time

// LongAdder throughput: ~10x higher than AtomicLong under high contention
```

**Read-copy-update (RCU) for read-heavy data structures:**
```python
import copy
import threading

class RCUDict:
    """Lock-free reads, copy-on-write updates"""
    def __init__(self):
        self._dict = {}     # Immutable reference
        self._write_lock = threading.Lock()
    
    def read(self, key):
        return self._dict.get(key)  # Lock-free!
    
    def update(self, key, value):
        with self._write_lock:
            new_dict = copy.copy(self._dict)  # Copy
            new_dict[key] = value              # Modify copy
            self._dict = new_dict             # Atomic pointer swap
            # Readers see either old or new dict, never partial state
```

---

### Q20. Design a job scheduler with concurrency control: max concurrent jobs per queue.

A production job scheduler must prevent individual queues from monopolising resources, ensure fair scheduling, and handle failure gracefully — all while operating correctly under concurrent access from multiple workers.

**Requirements:**
- Multiple queues (one per tenant or job type).
- Each queue has a configured max-concurrency limit.
- Workers poll for jobs across all queues.
- If a queue is at its concurrency limit, skip to the next queue.
- Jobs that fail are retried with exponential backoff.

**Database schema:**
```sql
CREATE TABLE queues (
    id           VARCHAR(50) PRIMARY KEY,
    max_concurrency INT NOT NULL DEFAULT 5
);

CREATE TABLE jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    queue_id        VARCHAR(50) REFERENCES queues(id),
    payload         JSONB NOT NULL,
    status          VARCHAR(20) DEFAULT 'pending',
    retry_count     INT DEFAULT 0,
    max_retries     INT DEFAULT 3,
    scheduled_at    TIMESTAMP DEFAULT NOW(),
    started_at      TIMESTAMP,
    completed_at    TIMESTAMP,
    error_message   TEXT,
    worker_id       VARCHAR(100)
);

CREATE INDEX ON jobs (queue_id, status, scheduled_at)
    WHERE status IN ('pending', 'failed');
```

**Atomic job claim (prevents multiple workers from claiming same job):**
```sql
-- Worker claims a job from any queue that has available concurrency
WITH queue_concurrency AS (
    -- Count running jobs per queue
    SELECT queue_id, COUNT(*) as running_count
    FROM jobs
    WHERE status = 'running'
    GROUP BY queue_id
),
available_queues AS (
    -- Find queues not at their limit
    SELECT q.id as queue_id
    FROM queues q
    LEFT JOIN queue_concurrency qc ON q.id = qc.queue_id
    WHERE COALESCE(qc.running_count, 0) < q.max_concurrency
),
next_job AS (
    -- Find next eligible job
    SELECT j.id
    FROM jobs j
    JOIN available_queues aq ON j.queue_id = aq.queue_id
    WHERE j.status = 'pending'
      AND j.scheduled_at <= NOW()
    ORDER BY j.scheduled_at ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED  -- Skip jobs claimed by other workers
)
UPDATE jobs
SET status = 'running',
    started_at = NOW(),
    worker_id = $1
FROM next_job
WHERE jobs.id = next_job.id
RETURNING jobs.*;
```

**Worker implementation:**
```python
import time
import uuid
import psycopg2

class JobWorker:
    def __init__(self, db_url, worker_id=None):
        self.db = psycopg2.connect(db_url)
        self.worker_id = worker_id or str(uuid.uuid4())
    
    def run(self):
        while True:
            job = self.claim_job()
            if job:
                self.execute_job(job)
            else:
                time.sleep(1)  # No jobs available, back-off
    
    def claim_job(self):
        with self.db.cursor() as cur:
            cur.execute(CLAIM_JOB_SQL, [self.worker_id])
            return cur.fetchone()
        self.db.commit()
    
    def execute_job(self, job):
        try:
            handler = get_handler(job['queue_id'])
            handler(job['payload'])
            self.mark_complete(job['id'])
        except Exception as e:
            self.mark_failed(job['id'], str(e), job['retry_count'])
    
    def mark_failed(self, job_id, error, retry_count):
        max_retries = 3
        if retry_count < max_retries:
            # Exponential backoff: retry after 2^retry_count minutes
            backoff = 2 ** retry_count
            next_attempt = f"NOW() + INTERVAL '{backoff} minutes'"
            self.db.execute("""
                UPDATE jobs SET status='pending', retry_count=retry_count+1,
                    scheduled_at={next_attempt}, error_message=%s
                WHERE id=%s
            """, [error, job_id])
        else:
            self.db.execute("""
                UPDATE jobs SET status='dead_letter', error_message=%s
                WHERE id=%s
            """, [error, job_id])
        self.db.commit()
```

**Stale job recovery (handles worker crashes):**
```sql
-- Background job: reclaim jobs stuck 'running' for > 10 minutes
UPDATE jobs
SET status = 'pending', worker_id = NULL, started_at = NULL
WHERE status = 'running'
  AND started_at < NOW() - INTERVAL '10 minutes';
```

**Concurrency safety analysis:**
- `FOR UPDATE SKIP LOCKED` ensures two workers never claim the same job.
- The CTE-based claim is a single atomic SQL statement — no time-of-check to time-of-use gap.
- Queue concurrency counting is snapshotted within the same transaction — consistent.
- Stale job recovery handles the failure case where a worker dies holding a job.

---

## Quick Reference

| Topic | Key Point |
|---|---|
| Concurrency vs parallelism | Concurrency = dealing with many things (structure); parallelism = doing many things (execution) |
| Thread vs coroutine | Thread: 1–8 MB stack, OS-scheduled; coroutine: 2–64 KB, user-space, 1M+ feasible |
| CPU-bound thread pool | N_cores or N_cores + 1 threads |
| I/O-bound thread pool | N_cores × (1 + wait_time/compute_time) |
| Mutex | 1 thread at a time; owner must release |
| Semaphore | N threads at a time; any thread can release |
| RW lock | N readers OR 1 writer; best when reads >> writes |
| CLAD deadlock conditions | Circular wait, Lock holding, Acquire-and-hold, Denial of sharing |
| Deadlock prevention | Lock ordering (most common); lock timeout; try-lock |
| CAS | Atomic read-modify-write; hardware instruction; basis of lock-free structures |
| ABA problem | Use tagged pointers with version counter to prevent CAS false success |
| Actor model | Private state + message passing; no shared memory; no locks needed |
| async/await | Suspends coroutine on I/O; event loop serves other work; no thread blocking |
| MVCC | Multiple row versions; readers never block writers |
| SELECT FOR UPDATE | Pessimistic row lock for read-modify-write; SKIP LOCKED for job queues |
| Connection pool exhaustion | Pool full → new requests timeout → cascading failure; monitor pool_waiting metric |
| Bounded queue | Back-pressure: producer blocks when full; prevents OOM |
| Thundering herd | All threads wake for one resource; solutions: mutex, probabilistic expiry, stale-while-revalidate |
| Serializable isolation | Prevents all anomalies; SSI in PostgreSQL; aborts conflicts, needs retry |
| Striped locks | 256 independent locks; reduce contention 256x; used in ConcurrentHashMap |
| LongAdder | Per-CPU cells; 10x throughput vs AtomicLong under high contention |
| Fork-join | Divide work recursively; execute in parallel; join results; work-stealing for load balance |
| Job scheduler | SELECT FOR UPDATE SKIP LOCKED; max-concurrency per queue; stale job recovery |
