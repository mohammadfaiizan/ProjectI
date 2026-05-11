# HLD Interview Q&A — File 20: Advanced Topics and Trade-offs

> 20 questions across Easy (Q1–7), Medium (Q8–15), and Hard (Q16–20).
> Each answer is 150–300+ words with diagrams, tables, or code where helpful.

---

## EASY (Q1–Q7)

---

### Q1. How do you decide between strong consistency and eventual consistency for a given feature?

**Answer:**

The choice between strong consistency and eventual consistency is a fundamental architectural decision that should be driven by business requirements, not technical preference.

**Framework for the decision:**

Ask three questions:

**1. What happens if a user reads stale data?**
```
Scenario A: User reads their own profile picture as the old version for 2 seconds
→ Acceptable → Eventual consistency fine

Scenario B: User sees their bank balance as $500 after spending $600
→ Catastrophic → Strong consistency required

Scenario C: User refreshes a product page and sees old inventory count briefly
→ Depends on conversion rate impact
```

**2. What is the cost of strong consistency?**
```
Strong consistency requires coordination:
  - Higher latency (must wait for quorum acknowledgment)
  - Lower availability during network partitions
  - Typically 2-10x more expensive (more coordination overhead)
```

**3. Can you fix stale reads with application logic?**
```
Read-your-writes consistency (weaker than full strong consistency):
  "You always see your own updates"
  
Implementation: Route reads to primary for 5 seconds after write,
then allow replica reads

This handles 95% of user-visible staleness at much lower cost
```

**Decision table:**

| Feature | Recommendation | Reason |
|---|---|---|
| Account balance | Strong | Money must be exact |
| User preferences | Eventual | Stale settings are harmless |
| Inventory count | Strong (or lease-based) | Overselling has real cost |
| Tweet like count | Eventual | Approximate count acceptable |
| Login / authentication | Strong | Security token must be current |
| Feed / recommendations | Eventual | Slightly stale content is fine |
| Payment status | Strong | User needs accurate status |
| Notification count | Eventual | Badge off by 1 is acceptable |

**Key principle:** Default to eventual consistency for read-heavy features where stale data is a minor UX issue. Default to strong consistency for anything involving money, security, or invariants you cannot violate.

---

### Q2. What is the N+1 problem and how does it manifest at different levels?

**Answer:**

The **N+1 problem** occurs when code executes 1 query to fetch N records, then executes N additional queries to fetch related data for each — resulting in N+1 total queries instead of 1 or 2.

**Database Level (ORM):**
```python
# BAD: N+1 query problem
orders = Order.objects.filter(user_id=123)  # 1 query
for order in orders:
    print(order.customer.name)  # 1 query per order = N queries
    # 100 orders = 101 queries!

# GOOD: Use JOIN or prefetch
orders = Order.objects.filter(user_id=123).select_related('customer')  # 1 query with JOIN
orders = Order.objects.filter(user_id=123).prefetch_related('items')   # 2 queries total
```

**API / Microservices Level:**
```python
# BAD: API calls N+1 downstream services
def get_enriched_orders(order_ids):
    orders = order_service.get_orders(order_ids)  # 1 API call
    for order in orders:
        order['user_details'] = user_service.get_user(order['user_id'])  # N API calls
    return orders  # If 100 orders: 1 + 100 = 101 API calls

# GOOD: Batch API call
def get_enriched_orders(order_ids):
    orders = order_service.get_orders(order_ids)  # 1 API call
    user_ids = [o['user_id'] for o in orders]
    users = user_service.get_users_batch(user_ids)  # 1 batch API call
    user_map = {u['id']: u for u in users}
    for order in orders:
        order['user_details'] = user_map[order['user_id']]
    return orders  # 2 API calls total
```

**GraphQL Level:**
```javascript
// BAD: GraphQL without DataLoader
const resolvers = {
    Order: {
        customer: async (order) => {
            return await UserService.find(order.customerId);  // Called for every order
        }
    }
};

// GOOD: DataLoader batches and caches
const userLoader = new DataLoader(async (userIds) => {
    const users = await UserService.findMany(userIds);
    return userIds.map(id => users.find(u => u.id === id));
});

const resolvers = {
    Order: {
        customer: async (order) => userLoader.load(order.customerId)
    }
};
```

**Impact at scale:** 100 orders × 100ms per user lookup = 10 seconds. With batching: 2 queries × 100ms = 200ms. A 50x improvement.

---

### Q3. How do you handle the thundering herd problem at multiple layers?

**Answer:**

The **thundering herd** occurs when many clients simultaneously request a resource that is unavailable (cache miss, server restart, scheduled event), overwhelming the backend.

**Layer 1: Cache stampede (most common)**
```
Scenario: 10,000 concurrent requests hit CDN
Cache entry for /homepage expires at 12:00:00
All 10,000 requests get cache miss simultaneously
All 10,000 forward to origin → origin dies

Solutions:
```

**Solution A: Cache locking (mutex on cache fill)**
```python
def get_with_lock(key, ttl, compute_fn):
    value = cache.get(key)
    if value:
        return value
    
    lock_key = f"lock:{key}"
    if redis.set(lock_key, 1, nx=True, ex=10):  # Only one winner
        # This process populates cache
        value = compute_fn()
        cache.setex(key, ttl, value)
        redis.delete(lock_key)
    else:
        # Other processes wait and retry
        time.sleep(0.1)
        return get_with_lock(key, ttl, compute_fn)  # Retry
    
    return value
```

**Solution B: Probabilistic early expiry (XFetch)**
```python
def get_with_early_expiry(key, ttl, compute_fn, beta=1):
    value, expiry = cache.get_with_ttl(key)
    if value:
        remaining_ttl = expiry - time.time()
        # Probabilistically refresh before expiry
        should_refresh = (-beta * math.log(random.random()) > remaining_ttl)
        if not should_refresh:
            return value
    
    value = compute_fn()
    cache.setex(key, ttl, value)
    return value
```

**Layer 2: Service restart stampede**
```
Scenario: Service restarts → all clients reconnect simultaneously
→ Auth service receives 100K connection requests in 1 second

Solution: Exponential backoff with jitter in clients
import random, time

def connect_with_backoff(base_delay=1, max_delay=60, max_retries=10):
    for attempt in range(max_retries):
        try:
            return establish_connection()
        except ConnectionError:
            delay = min(base_delay * (2 ** attempt), max_delay)
            # Jitter: randomize by ±30% to spread reconnections
            jitter = delay * 0.3 * (2 * random.random() - 1)
            time.sleep(delay + jitter)
```

**Layer 3: Cron job / scheduled spike**
```
Scenario: Daily report runs at 00:00 for 10M users
→ 10M simultaneous database queries

Solution: Stagger the load
import random

def schedule_daily_report(user_id):
    # Spread jobs over a 2-hour window instead of all at midnight
    delay_seconds = random.randint(0, 7200)  # 0-2 hours
    schedule_job(run_daily_report, user_id, delay=delay_seconds)
```

---

### Q4. What are the trade-offs between a monorepo and polyrepo for microservices?

**Answer:**

The choice between storing all services in one repository (monorepo) or separate repositories (polyrepo) has significant engineering culture and tooling implications.

**Monorepo (Google, Meta, Twitter):**
All services in a single repository:
```
/repo
  /services
    /user-service
    /order-service
    /payment-service
  /shared
    /proto-definitions
    /shared-libraries
    /common-utils
```

**Polyrepo:**
Each service in its own repository:
```
github.com/company/user-service
github.com/company/order-service
github.com/company/payment-service
github.com/company/shared-proto
```

**Comparison:**

| Aspect | Monorepo | Polyrepo |
|---|---|---|
| Atomic cross-service changes | Easy (one PR) | Hard (multiple PRs, coordination) |
| Shared library updates | Instant (all services see it) | Version pinning required |
| Build time | Slow (build whole repo) | Fast per service |
| CI/CD complexity | High (need selective builds) | Simple per service |
| Code discovery | Easy (search everywhere) | Hard (across repos) |
| Service independence | Low (easy to create coupling) | High (forced decoupling) |
| Onboarding | Easy (one clone) | Complex (N repos to understand) |
| Access control | Coarse-grained | Fine-grained per repo |

**When monorepo wins:**
- Shared schema/proto definitions change frequently
- Teams need to make atomic cross-service refactors
- Strong tooling investment (Bazel for selective builds, Nx, Turborepo)

**When polyrepo wins:**
- Services have completely different release cadences and owners
- Security requirements for code access isolation
- Different tech stacks (Python ML service + Go API + Node.js frontend)

**Hybrid approach (most large companies):** Monorepo for closely related services (e.g., all Python backend services), separate repos for unrelated platforms (mobile app, data science, infrastructure).

---

### Q5. How do you design for multi-tenancy? Compare shared table, shared schema, and separate database approaches.

**Answer:**

**Multi-tenancy** is designing a single software system to serve multiple customers (tenants) while isolating their data from each other.

**Option 1: Shared Table (tenant_id column)**
```sql
-- All tenants share the same tables
CREATE TABLE orders (
    id BIGINT PRIMARY KEY,
    tenant_id UUID NOT NULL,  -- Discriminator column
    customer_name VARCHAR(200),
    amount DECIMAL(10,2),
    created_at TIMESTAMP
);

-- Every query MUST include tenant_id filter
SELECT * FROM orders WHERE tenant_id = 'tenant-abc' AND id = 456;

-- Row-Level Security (PostgreSQL) enforces isolation
CREATE POLICY tenant_isolation ON orders
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
```

**Option 2: Shared Schema (separate tables per tenant)**
```sql
-- Each tenant has own table (schema segregation)
CREATE TABLE tenant_abc.orders ( ... );
CREATE TABLE tenant_xyz.orders ( ... );

-- Application switches schema based on tenant
SET search_path TO tenant_abc;
SELECT * FROM orders WHERE id = 456;
```

**Option 3: Separate Database**
```
tenant_abc → database_abc (separate cluster)
tenant_xyz → database_xyz (separate cluster)

Router: SELECT connection_string FROM tenant_registry WHERE tenant_id = ?
```

**Comparison:**

| Aspect | Shared Table | Shared Schema | Separate DB |
|---|---|---|---|
| Infrastructure cost | Lowest | Medium | Highest |
| Isolation level | Logical only | Schema-level | Physical |
| Data leak risk | Higher (bugs) | Medium | Lowest |
| Cross-tenant analytics | Easy | Medium | Hard |
| Tenant customization | Hard | Medium | Easy |
| Compliance (GDPR) | Harder | Medium | Easiest |
| Scale per tenant | Limited | Better | Best |
| Onboarding new tenant | Instant | Fast (create schema) | Slow (provision DB) |

**Best practice — tiered approach:**
```
Freemium / Small tenants → Shared table (low isolation needs, high density)
Business tier → Shared schema (some isolation)
Enterprise tier → Separate database (full isolation, compliance, custom config)
```

---

### Q6. What is the difference between orchestration and choreography, and when does each break down?

**Answer:**

**Orchestration** uses a central coordinator that directs all participants, explicitly calling each service in sequence.

**Choreography** uses events — services react to events and emit new events, with no central controller.

**Orchestration example:**
```python
# Order Orchestrator (explicit control flow)
class OrderOrchestrator:
    def process_order(self, order):
        # Step 1
        inventory = inventory_service.reserve(order.items)
        if not inventory.success:
            return {"error": "Out of stock"}
        
        # Step 2
        payment = payment_service.charge(order.payment, order.total)
        if not payment.success:
            inventory_service.release(inventory.reservation_id)
            return {"error": "Payment failed"}
        
        # Step 3
        shipping = shipping_service.schedule(order.address, inventory)
        return {"order_id": order.id, "tracking": shipping.tracking_number}
```

**Choreography example:**
```
OrderCreated event → Inventory Service (listens): reserves stock → fires InventoryReserved
InventoryReserved event → Payment Service (listens): charges → fires PaymentCompleted
PaymentCompleted event → Shipping Service (listens): schedules → fires ShipmentScheduled
```

**When orchestration breaks down:**
1. **Orchestrator becomes God Object:** All business logic centralizes in one place — any change requires touching the orchestrator
2. **Single point of failure:** Orchestrator crash = nothing works
3. **Scalability bottleneck:** All workflow state flows through orchestrator
4. **Synchronous coupling:** Steps must complete before proceeding (lower parallelism)

**When choreography breaks down:**
1. **Spaghetti flow:** 10 services reacting to 20 events → impossible to trace a single business process
2. **Debugging difficulty:** "Why was order 456 never shipped?" requires reconstructing events from logs
3. **Circular event loops:** Service A fires event → Service B reacts → fires event → Service A reacts → infinite loop
4. **Implicit contract:** Adding a new service that needs to react to an event requires finding all relevant event publishers

**Decision rule:**
- **Orchestration:** Complex, branching workflows with many failure modes and compensation requirements. Better observability.
- **Choreography:** Simple, linear flows; when loose coupling and independent deployability is paramount.

---

### Q7. How do you handle schema migrations in a zero-downtime system?

**Answer:**

Zero-downtime migrations require careful sequencing — the database schema must be compatible with both old and new versions of application code simultaneously during the deployment window.

**The fundamental rule:** Never make a change that is incompatible with the currently running version of the application.

**Strategy — Expand/Contract (Blue/Green database migrations):**

**Phase 1 — Expand (additive change):**
```sql
-- Add new column with nullable/default — safe while old code runs
ALTER TABLE users ADD COLUMN display_name VARCHAR(200);
-- Old code: ignores display_name (doesn't know about it)
-- New code: reads display_name if present
```

**Phase 2 — Migrate data:**
```python
# Backfill existing rows in batches (avoid table lock)
def backfill_display_name():
    last_id = 0
    while True:
        rows = db.execute("""
            UPDATE users
            SET display_name = CONCAT(first_name, ' ', last_name)
            WHERE id > %s AND display_name IS NULL
            LIMIT 10000
            RETURNING id
        """, [last_id])
        if not rows:
            break
        last_id = max(r.id for r in rows)
        time.sleep(0.1)  # Throttle to avoid overloading DB
```

**Phase 3 — Deploy new code:**
```
Deploy new application version that:
  - Reads from display_name
  - Writes to BOTH name (old) AND display_name (new)
```

**Phase 4 — Contract (remove old column):**
```sql
-- Only safe after ALL old code is retired
-- And after verifying display_name is fully populated
ALTER TABLE users DROP COLUMN first_name;
ALTER TABLE users DROP COLUMN last_name;
```

**Dangerous operations and their safe alternatives:**

| Dangerous | Safe Alternative |
|---|---|
| `ALTER TABLE ... ADD COLUMN NOT NULL` | Add nullable first, backfill, then add constraint |
| `DROP COLUMN` | First stop reading it in code, then drop |
| `RENAME COLUMN` | Add new column, dual-write, migrate, drop old |
| `ADD INDEX` (locks table) | `CREATE INDEX CONCURRENTLY` (PostgreSQL) |
| Change data type | Add new column, migrate data, swap |

---

## MEDIUM (Q8–Q15)

---

### Q8. What is the cell architecture pattern, and when is it worth the complexity?

**Answer:**

**Cell architecture** (also called "cell-based architecture" or "swim lane" architecture) partitions a system into independent, isolated cells — each a complete, self-contained replica of the full stack, serving a subset of users.

**Architecture:**
```
Users 1-10M    → Cell A (Region US-East)
                 [API servers + DB + Cache + Queue]

Users 10M-20M  → Cell B (Region US-West)  
                 [API servers + DB + Cache + Queue]

Users 20M-30M  → Cell C (Region EU)
                 [API servers + DB + Cache + Queue]

Cell Router:
  user_id % num_cells → routes to appropriate cell
```

**Key properties of a cell:**
- **Full stack:** Each cell has its own API servers, database, cache, message queue — completely self-contained
- **No cross-cell data dependencies:** A request always stays within one cell
- **Independent deployability:** Deploy to Cell A without affecting Cell B
- **Failure isolation:** Cell A outage affects only 10M users (1/3 of total), not all 30M

**Why this matters:**
```
Traditional architecture: single global deployment
  One bad config change → affects 100% of users
  One DB overload → affects 100% of users
  One noisy neighbor tenant → slows everyone

Cell architecture:
  Bad config change deployed to Cell A only → affects 33% of users
  Blast radius contained
  Runaway tenant isolated to their cell
```

**Implementation:**
```python
class CellRouter:
    def __init__(self, cells):
        self.cells = cells  # List of cell configs
    
    def get_cell_for_user(self, user_id):
        cell_index = hash(user_id) % len(self.cells)
        return self.cells[cell_index]
    
    def get_cell_for_tenant(self, tenant_id):
        # Enterprise tenants may be assigned to dedicated cells
        if dedicated_cell := self.dedicated_cells.get(tenant_id):
            return dedicated_cell
        cell_index = hash(tenant_id) % len(self.shared_cells)
        return self.shared_cells[cell_index]
```

**When is cell architecture worth the complexity?**

Worth it when:
- Running stateful services that are hard to migrate (databases)
- Need blast radius reduction for compliance (GDPR: EU data stays in EU cells)
- Have noisy neighbor problems between enterprise and freemium tenants
- Need to test new versions in production gradually (canary cells)
- Scale requires geographic distribution

Not worth it when:
- System is simple enough that standard blue/green deployments provide sufficient isolation
- Cross-user queries are frequent (cells make cross-cell queries very expensive)
- Team is small (operational overhead is significant)

---

### Q9. How do you design a system that needs to be both GDPR compliant and performant?

**Answer:**

GDPR (General Data Protection Regulation) creates specific technical constraints that can conflict with performance-optimizing techniques like caching, denormalization, and global data replication.

**GDPR requirements that affect system design:**

**1. Right to be forgotten (Art. 17):**
```
User requests deletion → all their data must be deleted within 30 days
from ALL systems: primary DB, replicas, backups, analytics, caches, logs

Challenge: Data is intentionally replicated for performance
```

**Solution — Crypto shredding:**
```python
# Instead of storing PII directly, encrypt it with a per-user key
# Deletion = delete the key → all encrypted data becomes unreadable

class CryptoShredder:
    def store_user_data(self, user_id, pii):
        # Generate per-user encryption key
        user_key = generate_key()
        key_store.put(f"key:{user_id}", user_key)
        
        # Encrypt PII before storing
        encrypted = encrypt(pii, user_key)
        db.put(f"user:{user_id}", encrypted)
    
    def delete_user(self, user_id):
        # Delete encryption key → all data becomes cryptographic garbage
        key_store.delete(f"key:{user_id}")
        # Data still exists physically but is unreadable
        # Can be purged from backups on normal rotation schedule
```

**2. Data residency (Art. 44-49):**
```
EU user data must remain in EU unless adequacy decision exists

Architecture:
  EU users → EU database cluster (Frankfurt/Dublin)
  US users → US database cluster (Virginia)
  Cross-region data transfer: anonymized aggregates only

Implementation:
  Cell architecture (Q8) with geo-cells per regulatory region
  EU Cell: processes and stores all EU user data
  US Cell: processes and stores all US user data
```

**3. Data minimization — affects caching:**
```python
# WRONG: Cache full user profile (includes PII)
redis.set(f"user:{user_id}", json.dumps(full_user_profile))

# RIGHT: Cache only non-PII data
redis.set(f"user:{user_id}:public", json.dumps({
    "display_name": user.display_name,  # User chose to make public
    "member_since": user.created_at,
    "role": user.role
}))
# Never cache: email, phone, address, payment info
```

**4. Audit logging:**
```sql
-- Every PII access must be logged
CREATE TABLE gdpr_access_log (
    id BIGINT PRIMARY KEY,
    accessor_user_id BIGINT,
    accessed_user_id BIGINT,
    data_category VARCHAR(100),  -- 'email', 'profile', 'payment_info'
    access_reason VARCHAR(200),  -- 'user_request', 'support_ticket_123'
    accessed_at TIMESTAMP,
    ip_address INET
);
```

**Performance trade-offs:**
- Crypto shredding: ~5-10% overhead for encryption/decryption
- Data residency: increases latency for cross-region operations by 50-100ms
- Audit logging: adds write overhead; mitigate with async Kafka pipeline

---

### Q10. What are the trade-offs between synchronous and asynchronous inter-service communication?

**Answer:**

**Synchronous** (HTTP/gRPC): Caller waits for response before proceeding.
**Asynchronous** (Kafka/RabbitMQ/SQS): Caller publishes and continues; consumer processes later.

**Synchronous (HTTP/gRPC):**
```python
# Direct HTTP call — synchronous
def create_order(cart_id, user_id):
    # Call inventory service synchronously
    reservation = inventory_service.reserve(cart_id)  # Blocks until response
    if not reservation.success:
        return {"error": "Out of stock"}
    
    # Call payment service synchronously
    payment = payment_service.charge(user_id, reservation.total)
    if not payment.success:
        inventory_service.release(reservation.id)  # Compensate
        return {"error": "Payment failed"}
    
    return {"order_id": new_order_id}
```

**Asynchronous (Event-driven):**
```python
# Publish event — fire and forget
def create_order(cart_id, user_id):
    order_id = generate_order_id()
    
    # Persist to DB
    db.save_order(order_id, cart_id, user_id, status='PENDING')
    
    # Publish event — consumer processes asynchronously
    kafka.produce('order-created', {
        'order_id': order_id,
        'cart_id': cart_id,
        'user_id': user_id
    })
    
    return {"order_id": order_id, "status": "PENDING"}  # Returns immediately
    # Client polls status or receives webhook when complete
```

**Comparison:**

| Aspect | Synchronous | Asynchronous |
|---|---|---|
| Latency | Request latency = sum of all calls | Request latency = local only |
| Coupling | Tight (caller knows callee) | Loose (publish to topic) |
| Error handling | Immediate (return error to caller) | Deferred (retry in background) |
| Transaction semantics | Can use distributed 2PC | Saga required |
| Debugging | Easy (request trace) | Hard (event correlation) |
| Availability | Caller unavailable if callee down | Caller unaffected by consumer down |
| Throughput | Limited by slowest service | Producer throughput limited by queue only |

**Decision guide:**
- **Synchronous:** User needs immediate feedback (payment status, search results, login)
- **Asynchronous:** Operations can be deferred (send email, process report, trigger workflow)
- **Hybrid:** Use sync for critical path; decouple non-critical side effects

**The hidden cost of synchronous chains:**
```
If service A calls B calls C calls D:
  A availability = A% × B% × C% × D%
  Each service at 99.9% → chain availability: 99.9%⁴ = 99.6%
  
Make it async: A only depends on queue
  A availability = A% × Queue% (queues are very reliable: 99.99%)
```

---

### Q11. How do you design a feature flag system that works at scale?

**Answer:**

A feature flag system allows enabling/disabling features dynamically without code deployment, enabling gradual rollouts, A/B tests, and instant kill switches.

**Data model:**
```json
{
  "flag_key": "new_checkout_flow",
  "type": "percentage",            // boolean, percentage, user-segment, user-list
  "enabled": true,
  "percentage": 5,                 // 5% of users
  "targeting_rules": [
    {
      "type": "user_segment",
      "segment": "beta_testers",
      "override": "enabled"
    },
    {
      "type": "user_list",
      "user_ids": [123, 456, 789],
      "override": "enabled"
    },
    {
      "type": "country",
      "countries": ["US", "CA"],
      "override": "disabled"
    }
  ]
}
```

**Evaluation engine:**
```python
class FeatureFlagEvaluator:
    def is_enabled(self, flag_key, user_context):
        flag = self.get_flag(flag_key)  # From cache
        
        if not flag or not flag['enabled']:
            return False
        
        # Check targeting rules in order (first match wins)
        for rule in flag.get('targeting_rules', []):
            if self.matches_rule(rule, user_context):
                return rule['override'] == 'enabled'
        
        # Default: percentage rollout (deterministic per user)
        if flag['type'] == 'percentage':
            # Deterministic hash so same user always gets same result
            bucket = (hash(f"{flag_key}:{user_context['user_id']}") % 100)
            return bucket < flag['percentage']
        
        return False
```

**Distribution architecture:**
```
Flag Store (database)
      ↓
Flag Config Service (HTTP API)
      ↓
SDK Cache (in-memory per service)  ← SDK polls every 30s or receives push update
      ↓
Application code evaluates locally (no network call per feature check)
```

**SDK pattern (critical for performance):**
```python
# WRONG: Network call per flag evaluation
def render_button(user_id):
    if flag_service.is_enabled('new_button', user_id):  # HTTP call every time!
        return new_button()
    return old_button()

# RIGHT: Local evaluation with cached flags
class FeatureFlagClient:
    def __init__(self):
        self.flags = {}  # Local in-memory cache
        self._start_polling()  # Background refresh
    
    def is_enabled(self, key, context):
        flag = self.flags.get(key)  # Microsecond local lookup
        return self._evaluate(flag, context) if flag else False

# One network call every 30s for all flags (not per-evaluation)
```

**Kill switch pattern:**
```python
# Instant emergency disable — no deployment required
def emergency_disable(flag_key, reason):
    flag_store.update(flag_key, {
        'enabled': False,
        'disabled_reason': reason,
        'disabled_at': now()
    })
    # Pub/Sub push notification to all SDK instances
    redis.publish('flag-updates', f'disabled:{flag_key}')
    # All instances disable the feature within milliseconds
```

---

### Q12. What is the back-pressure problem, and how do different systems handle it?

**Answer:**

**Back-pressure** occurs when data is produced faster than it can be consumed. Without back-pressure mechanisms, buffers overflow, memory exhausts, or data is silently dropped.

**The problem:**
```
Producer: 1M events/second
Consumer: 100K events/second
Buffer: 10M events

After 10 seconds: buffer full
Options without back-pressure:
  A) Drop new events (data loss)
  B) Block producer (cascades upstream)
  C) Crash (OOM error)
```

**System-specific handling:**

**Kafka:**
```python
# Kafka consumer owns its offset — pulls only what it can handle
consumer = KafkaConsumer('orders', max_poll_records=100)

for messages in consumer:
    process_messages(messages)  # Process 100 at a time
    consumer.commit()            # ACK only when done
    # If processing is slow, consumer just lags — producer unaffected
    # Consumer lag is monitored (alert when > 10,000 messages)
```

**TCP (transport level):**
```
Receive window in TCP header (0-65535 bytes)
Receiver advertises window = how much buffer space it has
Sender may NOT send more than window allows
Window = 0 → sender blocks completely
```

**Node.js streams:**
```javascript
const readable = fs.createReadStream('huge-file.csv');
const writable = fs.createWriteStream('output.json');

// Without back-pressure:
readable.on('data', (chunk) => {
    writable.write(chunk);  // Buffer fills up if writing is slow
});

// With back-pressure (pipe handles it automatically):
readable.pipe(writable);  // pipe() respects highWaterMark and pauses readable when needed
```

**RxJS / Reactive Streams:**
```typescript
from(dataSource)
    .pipe(
        bufferTime(1000, null, 100),  // Buffer 1s or 100 items (whichever first)
        concatMap(batch => processSlowly(batch)),  // Sequential, maintains order
        // OR:
        mergeMap(batch => processSlowly(batch), 5)  // Parallel but max 5 concurrent
    )
    .subscribe();
```

**Back-pressure signals:**

| System | Mechanism | Signal to Producer |
|---|---|---|
| TCP | Receive window | Reduce packet rate |
| Kafka | Consumer pull | Consumer controls rate |
| HTTP/2 | Flow control frames | Explicit WINDOW_UPDATE |
| Reactive Streams | request(n) | Pull-based demand signaling |
| Redis Streams | MAXLEN | Trim oldest messages |

---

### Q13. How do you approach capacity planning for a system expecting 10x growth?

**Answer:**

Capacity planning is not just buying bigger servers — it's an analytical process combining current metrics, growth projections, and architectural headroom.

**Step 1: Establish current baselines**
```python
# Measure actual production metrics (last 30 days)
baseline = {
    "requests_per_second": {
        "p50": 1200, "p95": 3400, "p99": 8200, "peak": 12000
    },
    "db_connections_per_second": 450,
    "kafka_messages_per_second": 8500,
    "storage_growth_per_day_gb": 120,
    "cpu_utilization_avg": 0.45,  # 45%
    "memory_utilization_avg": 0.68,
    "p99_latency_ms": 250
}
```

**Step 2: Project 10x load**
```python
# Simple linear projection (conservative for most resources)
projected_10x = {
    "requests_per_second": {
        "p50": 12000, "p95": 34000, "p99": 82000, "peak": 120000
    },
    "storage_growth_per_day_gb": 1200,  # 10x growth
    "storage_per_year_tb": 438,          # 1200 × 365 / 1024
}

# But many bottlenecks don't scale linearly:
# - DB connections: N threads × M connection per thread
# - N+1 queries: 10x data = 10x query amplification
# - Lock contention: may get worse than linear with concurrent writers
```

**Step 3: Identify bottlenecks**
```
Current architecture ceiling (before optimization):
  Web servers:  Current peak utilization 60% → can handle ~17K RPS per server
                10x = 120K RPS → need 7 servers (have 3 → add 4)
  
  Database:     Current p99 query time 50ms at 450 QPS
                10x = 4500 QPS → likely approaches connection limit (500 max)
                BOTTLENECK: Add read replicas + query optimization first
  
  Kafka:        Linear scaling with partitions → add partitions + consumers
  
  Redis:        Single node at 30% CPU → can handle 10x without change
  
  Storage:      438TB/year → evaluate S3 tiering (lifecycle to Glacier for old data)
```

**Step 4: Make targeted improvements**
```
Priority 1 (blocks scaling): 
  DB: Add 3 read replicas (immediate 4x read capacity increase)
  DB: Connection pooling (PgBouncer — 5x connection efficiency)
  
Priority 2 (within 3 months):
  DB: Partition largest tables (orders, events)
  App: Fix N+1 queries found in profiling
  
Priority 3 (within 6 months):
  Evaluate: Move analytics to ClickHouse (offload DB)
  Evaluate: Introduce caching layer for hot queries
```

**Load testing before launch:**
```bash
# k6 load test simulating 10x traffic
k6 run --vus 10000 --duration 30m load_test.js
# Monitor: error rate, latency percentiles, DB connection wait time
# Target: p99 < 500ms at 10x load, error rate < 0.1%
```

---

### Q14. What is the difference between read-through and cache-aside in failure scenarios?

**Answer:**

**Cache-aside (Lazy loading):**
Application code manages cache explicitly — on miss, application fetches from DB and populates cache.

```python
def get_user_cache_aside(user_id):
    # 1. Try cache
    user = redis.get(f"user:{user_id}")
    if user:
        return json.loads(user)
    
    # 2. Cache miss → fetch from DB (application handles this)
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    
    # 3. Populate cache
    redis.setex(f"user:{user_id}", 3600, json.dumps(user))
    return user
```

**Read-through:**
Cache sits in front of DB and handles misses automatically — application only ever talks to cache.

```python
# Application code is simpler:
def get_user_read_through(user_id):
    return cache.get(f"user:{user_id}")  # Cache handles DB fetch on miss

# Cache is configured with a "loader function":
cache = Cache(loader=lambda key: db.query("SELECT * FROM users WHERE id = ?", key))
```

**Failure scenarios — key differences:**

**Redis cache failure:**

Cache-aside:
```python
def get_user_cache_aside(user_id):
    try:
        user = redis.get(f"user:{user_id}")
    except RedisConnectionError:
        # Fallback: go directly to DB
        return db.query("SELECT * FROM users WHERE id = %s", user_id)
    # Application controls fallback
```

Read-through:
```python
def get_user_read_through(user_id):
    return cache.get(f"user:{user_id}")
    # If cache is down: behavior depends entirely on cache library's fallback
    # May throw exception, may fallback to DB, may return None
    # Application has less control
```

**DB failure:**

Cache-aside:
```python
# Can serve stale data explicitly
def get_user_resilient(user_id):
    user = redis.get(f"user:{user_id}")
    if user:
        return user
    try:
        user = db.query(...)
        redis.setex(f"user:{user_id}", 3600, json.dumps(user))
    except DBConnectionError:
        # Return extended stale data from cache
        stale_user = redis.get(f"user:{user_id}:stale")
        return stale_user  # Explicitly return stale data
```

**Summary:** Cache-aside gives more control and explicit fallback behavior, making it easier to design graceful degradation. Read-through is simpler code but you depend on the cache library for failure behavior.

---

### Q15. How do you handle data consistency across microservices without distributed transactions?

**Answer:**

Distributed transactions (2PC) across microservices are problematic — they create tight coupling, reduce availability, and can cause blocking. The alternatives use eventual consistency patterns.

**Pattern 1: Saga with compensating transactions (covered in File 16 Q11)**
Handle each step locally; compensate on failure.

**Pattern 2: Outbox Pattern (transactional messaging)**
Guarantee that events are published exactly once by co-locating them with the business transaction.
```python
# Application transaction
def create_order(items, user_id):
    with db.transaction():
        order = db.insert_order(items, user_id)
        # Outbox in same transaction
        db.insert_outbox({
            'event_type': 'ORDER_CREATED',
            'payload': order.to_dict(),
            'aggregate_id': order.id
        })
    # Outbox relay publishes to Kafka asynchronously
```

**Pattern 3: Event sourcing**
Store state as a sequence of events — the event log IS the source of truth, and other services project their own view.
```python
# Event store (Kafka / EventStoreDB)
event_log = [
    {'type': 'OrderPlaced', 'order_id': 1, 'items': [...], 'ts': T1},
    {'type': 'PaymentAuthorized', 'order_id': 1, 'amount': 100, 'ts': T2},
    {'type': 'InventoryReserved', 'order_id': 1, 'ts': T3},
    {'type': 'OrderConfirmed', 'order_id': 1, 'ts': T4},
]

# Each service projects its own materialized view
# Order Service projects: orders table
# Inventory Service projects: reservations table
# Payment Service projects: authorizations table
```

**Pattern 4: Change Data Capture (CDC)**
Use database transaction log as the integration event bus.
```
PostgreSQL WAL → Debezium → Kafka → Consumer services
Any change to orders table automatically publishes a CDC event
Services react to their relevant changes
```

**Anti-patterns to avoid:**
```python
# WRONG: Synchronous distributed transaction
def create_order(items):
    order_service.create(items)        # Service A
    inventory_service.reserve(items)   # Service B — will this fail?
    payment_service.charge(amount)     # Service C — all or nothing? NO!

# If payment fails: must manually undo order + inventory
# Creates synchronous coupling → tight dependency
```

**Eventual consistency acceptance:**
The hardest part is accepting that cross-service consistency is eventual, not immediate. Design UX around this:
- Show "Order Processing" instead of "Order Confirmed" until all services complete
- Use optimistic UI updates on client
- Provide reconciliation/correction mechanisms for edge cases

---

## HARD (Q16–Q20)

---

### Q16. What are the trade-offs of using a service mesh vs. an API gateway?

**Answer:**

Both service meshes and API gateways handle cross-cutting concerns for service communication, but at different layers and with different scope.

**API Gateway:**
Sits at the **north-south** boundary (external traffic entering the cluster):
```
Internet → API Gateway → Internal Services
                ↑
  Handles: Auth, rate limiting, routing, SSL termination,
           request transformation, API versioning, logging
```

**Service Mesh:**
Sits at the **east-west** boundary (service-to-service communication):
```
Service A ←→ [Proxy Sidecar] ←→ [Proxy Sidecar] ←→ Service B
                    ↑
  Handles: mTLS, circuit breaking, load balancing, observability,
           retries, timeout, traffic splitting (canary)
```

**Detailed comparison:**

| Aspect | API Gateway | Service Mesh |
|---|---|---|
| Traffic direction | North-South (ingress) | East-West (internal) |
| Primary concern | External API management | Service-to-service reliability |
| Authentication | External auth (JWT, OAuth) | Internal mTLS (service identity) |
| Implementation | Single component | Sidecar proxy per pod (Envoy) |
| Overhead | Low | High (sidecar per pod, ~3-5ms latency added) |
| Observability | Request-level metrics | Service-level distributed tracing |
| Configuration | Centralized | Distributed (control plane) |
| Examples | Kong, AWS API GW, Nginx | Istio, Linkerd, Consul Connect |

**When API Gateway alone is sufficient:**
- Small number of services (< 10)
- Simple internal communication patterns
- Teams don't need per-service fine-grained policies
- Cost is a concern (service mesh has real operational overhead)

**When service mesh is worth it:**
- Many services need mutual authentication (zero-trust networking)
- Complex traffic management (canary, A/B at service level)
- Distributed tracing across 50+ services
- Teams enforce SLAs for internal service communication
- Compliance requires encrypted internal traffic (PCI DSS, HIPAA)

**The hybrid approach (common):**
```
API Gateway: handles external traffic, external auth, rate limiting
Service Mesh: handles internal mTLS, circuit breaking, observability

Internet → [API Gateway: Kong/Nginx] → Kubernetes Cluster
                                         [Istio Service Mesh]
                                          ↓
                               Service A ←→ Service B ←→ Service C
```

**Operational reality:** Service meshes are complex. Istio requires significant expertise to configure correctly. Many teams adopt Linkerd (simpler) or a lightweight sidecar approach instead of full Istio.

---

### Q17. How do you design an idempotent API that handles retries safely?

**Answer:**

An idempotent API can be called multiple times with the same input and produce the same result without additional side effects. This is critical for retry-safe clients.

**HTTP method idempotency (standard):**
```
GET    → Idempotent (read-only, no side effects)
PUT    → Idempotent (replace resource; calling 5× = same result as calling 1×)
DELETE → Idempotent (delete same resource multiple times = same end state)
POST   → NOT idempotent by default (creates new resource each time)
PATCH  → Depends on implementation
```

**Making POST idempotent with idempotency keys:**
```python
@app.route('/api/payments', methods=['POST'])
def create_payment():
    idempotency_key = request.headers.get('Idempotency-Key')
    
    if not idempotency_key:
        return {"error": "Idempotency-Key header required"}, 400
    
    # Validate key format
    if not is_valid_uuid(idempotency_key):
        return {"error": "Invalid Idempotency-Key format"}, 400
    
    # Check for existing request with this key
    lock_key = f"lock:idempotency:{idempotency_key}"
    with redis.lock(lock_key, timeout=30):
        existing = db.get_idempotency_record(idempotency_key)
        
        if existing:
            # Validate payload matches original
            if existing.request_hash != hash_request(request.json):
                return {"error": "Idempotency-Key reused with different payload"}, 422
            # Return original response
            return existing.response_body, existing.status_code
        
        # Process the new payment
        result = process_payment(
            amount=request.json['amount'],
            customer_id=request.json['customer_id']
        )
        
        # Store result with idempotency key
        db.store_idempotency_record(
            key=idempotency_key,
            request_hash=hash_request(request.json),
            response_body=result,
            status_code=200,
            expires_at=datetime.now() + timedelta(hours=24)
        )
        
    return result, 200
```

**Database schema for idempotency:**
```sql
CREATE TABLE idempotency_records (
    idempotency_key VARCHAR(64) PRIMARY KEY,
    request_hash    VARCHAR(64) NOT NULL,
    status_code     INTEGER     NOT NULL,
    response_body   JSONB       NOT NULL,
    created_at      TIMESTAMP   NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMP   NOT NULL
);

-- Auto-clean expired records
CREATE INDEX idx_expires_at ON idempotency_records(expires_at);
```

**Retry-safe design for distributed workflows:**
```python
# Each step in a workflow has a stable idempotency key
def process_order(order_id):
    # Key derived from business entity ID, not random
    reserve_key = f"reserve:{order_id}"
    pay_key = f"pay:{order_id}"
    ship_key = f"ship:{order_id}"
    
    # Each call is safe to retry
    inventory.reserve(order_id, idempotency_key=reserve_key)
    payment.charge(order_id, idempotency_key=pay_key)
    shipping.schedule(order_id, idempotency_key=ship_key)
```

**Client-side retry policy:**
```python
def retry_with_backoff(fn, max_retries=3):
    idempotency_key = str(uuid4())  # Generate ONCE per logical operation
    
    for attempt in range(max_retries):
        try:
            return fn(idempotency_key=idempotency_key)  # Reuse same key
        except (NetworkError, TimeoutError):
            if attempt < max_retries - 1:
                wait = 2 ** attempt + random.random()
                time.sleep(wait)
    raise MaxRetriesExceeded()
```

---

### Q18. What is the strangler fig pattern, and what can go wrong during migration?

**Answer:**

The **Strangler Fig Pattern** (named after the strangler fig plant that grows around and eventually replaces a host tree) is a migration strategy for incrementally replacing a legacy monolith with microservices.

**The pattern:**
```
Phase 1: Route all traffic through monolith (unchanged)
  Client → [Strangler Proxy] → Monolith

Phase 2: Extract Feature A to new service, proxy routes selectively
  Client → [Strangler Proxy] → Feature A Service (new)
                              → Monolith (everything else)

Phase 3: Extract more features over time
  Client → [Strangler Proxy] → Feature A Service
                              → Feature B Service
                              → Feature C Service
                              → Monolith (remaining)

Phase 4: Monolith is empty → decommission
  Client → [Load Balancer] → Services A, B, C, D, E...
```

**Implementation:**
```python
# Strangler proxy (Nginx or application-level router)
class StranglerProxy:
    def route(self, request):
        # Routing table driven by feature flags
        if request.path.startswith('/api/users') and feature_flags.is_enabled('new_user_service'):
            return self.forward(request, 'http://user-service:8080')
        
        if request.path.startswith('/api/payments') and feature_flags.is_enabled('new_payment_service'):
            return self.forward(request, 'http://payment-service:8080')
        
        # Default: monolith
        return self.forward(request, 'http://legacy-monolith:3000')
```

**What can go wrong:**

**1. Data synchronization nightmare:**
```
Problem: Both monolith and new service write to data
         Monolith: writes to MySQL users table
         New service: writes to PostgreSQL users table
         → Two sources of truth → divergence

Solution: Cut over data access at the same time as routing
  OR: New service writes to same DB initially, migrate DB later
```

**2. Distributed transactions across strangler boundary:**
```
Order service (new) needs to check user balance (monolith)
→ Synchronous HTTP call creates coupling you're trying to eliminate
→ Solution: Publish events from monolith; new service subscribes
```

**3. Proxy becomes a bottleneck:**
```
All traffic through one proxy → proxy is SPOF
Solution: Make proxy stateless + horizontally scalable (Nginx + replicas)
```

**4. Feature flag sprawl:**
```
After 2 years: 50 feature flags in the proxy, some forgotten
"Can we remove this flag?" → "I don't know if anything depends on old behavior"
Solution: Time-bound all migration flags; review quarterly
```

**5. Never finishing the migration:**
```
Common trap: 80% migrated → last 20% is hard (oldest, scariest code)
→ Teams stop and "live with it"
Solution: Executive commitment to complete migration; track remaining monolith lines of code
```

---

### Q19. How do you evaluate whether to build vs buy a component (e.g., message queue, search)?

**Answer:**

Build vs buy is one of the most consequential architectural decisions, affecting cost, velocity, and operational complexity for years.

**Decision framework:**

**Step 1: Assess differentiation**
```
Ask: "Does this component differentiate our product?"

Message queue: NO → no customer pays you because your queue is better than Kafka
Search for your e-commerce: Maybe → if search quality is key differentiator
Content recommendation: YES → core product feature, unique to your data
```

**Step 2: Evaluate build cost accurately (teams consistently underestimate)**
```
Build cost for a message queue:
  Engineering: 3 senior engineers × 12 months = 36 person-months
  Infrastructure: servers, monitoring, ops overhead
  Ongoing: maintenance, security patches, capacity planning
  Opportunity cost: 3 engineers NOT building product features

Total 3-year cost (fully loaded): $1.5M - $3M

Buy cost (Kafka on Confluent Cloud):
  $10K-$200K/year depending on volume
  3-year total: $30K-$600K
  + integration work: 1-2 weeks

→ Build is only justified if you need > $600K/year worth of customization
```

**Step 3: Assess operational burden**

```
Build:
  Your team is on-call for outages at 3AM
  Security vulnerabilities → your team patches
  Capacity planning → your team figures it out
  Onboarding new engineers → they learn your custom system

Buy (managed service):
  Vendor handles operations
  SLAs typically: 99.9-99.99% availability
  Security patches handled by vendor
  Scaling is someone else's problem
```

**Decision matrix:**

| Scenario | Recommendation | Reason |
|---|---|---|
| Message queue (Kafka) | Buy (Confluent/MSK/Pub/Sub) | Commodity; build cost >> buy cost |
| Full-text search | Buy (Elasticsearch/Algolia) | Complex to build correctly |
| Database | Buy (Aurora, Spanner, DynamoDB) | Never build unless FAANG-scale |
| Recommendation engine | Build | Core differentiator, unique data |
| Authentication | Buy (Auth0/Cognito) for startups, Build for scale | Security is hard; buy until 10M users |
| CDN | Buy until Netflix-scale | Economics favor building only at massive scale |
| Monitoring | Buy (Datadog/Grafana) until custom needs arise | Not a differentiator |

**Hybrid approach:**
```
Use open-source self-hosted (not managed service):
  Kafka self-hosted > build from scratch
  Elasticsearch self-hosted > Algolia (cost at scale)
  
This reduces cost vs managed but increases operational burden
Good for: teams with strong infrastructure expertise + large volume
```

---

### Q20. You're the principal engineer at a startup that just got 100x traffic overnight. What is your immediate action plan?

**Answer:**

This is a crisis response question. The goal is: restore stability first, then optimize. Every decision must be time-boxed and prioritized by impact.

**Immediate (0-30 minutes): Stop the bleeding**

```
Step 1: Assess the situation (5 minutes)
  - Check monitoring dashboards: what is actually failing?
  - CPU? Memory? DB connections? Disk I/O? Network?
  - Is it ALL traffic or specific endpoints?
  - Is revenue impacted? What is the SLA?

Step 2: Quick wins — scale out what you can (10 minutes)
  - Auto-scaling groups: increase max instances NOW (manually trigger if needed)
  - If on Kubernetes: kubectl scale deployment api --replicas=20
  - Add spot/preemptible instances immediately (cost is secondary right now)

Step 3: Reduce load on the bottleneck (10 minutes)
  - Enable rate limiting at API gateway / CDN level
  - Drop or queue non-critical traffic (analytics, batch jobs)
  - Enable maintenance page for non-essential features
  - Increase cache TTLs aggressively (serve stale rather than crash)
```

**Short-term (30 min - 4 hours): Systematic mitigation**

```python
# Priority order: address bottlenecks in this sequence

bottleneck_priority = [
    "Database (most common single-node bottleneck)",
    "Application servers (easiest to scale)",
    "Cache layer",
    "Message queue",
    "Storage/CDN"
]

# Database emergency actions:
# 1. Add read replicas immediately (AWS RDS: takes 20-30 minutes)
# 2. Route read-only queries to replicas
# 3. Increase max_connections if hitting connection limit
# 4. Identify and kill long-running queries that are blocking others
# 5. Enable connection pooling (PgBouncer) if not already in place

# Application emergency actions:
# 1. Scale out API servers (horizontal scaling)
# 2. Enable circuit breakers for non-critical external calls
# 3. Increase worker thread pool sizes carefully
# 4. Enable queue-based processing for write-heavy endpoints

# Cache emergency actions:
# 1. Increase Redis node size (vertical scale is faster than resharding)
# 2. Increase cache TTLs to reduce DB pressure
# 3. Add cache warming for top N hottest keys
```

**Communication (parallel to technical actions):**
```
T+0:  Page on-call team, assemble war room (Slack channel or video call)
T+15: Status page update: "We are experiencing high traffic and investigating"
T+30: Engineering leadership update (short, factual: impact + actions taken)
T+60: Customer support briefed on talking points
T+2h: Updated ETA for full resolution
```

**Medium-term (4 hours - 24 hours): Structural fixes**
```
1. Database sharding / partitioning for hotspot tables
2. Add dedicated read replicas for analytics queries
3. Implement request queuing for write endpoints
4. Add CDN in front of any static/semi-static content
5. Implement proper cache warming strategy
6. Profile and fix top 3 slowest endpoints (usually 20% of endpoints cause 80% of load)
```

**Post-incident (24-72 hours): Root cause and prevention**
```
Load test to find the actual breaking point:
  k6 run --vus 1000 --duration 10m load_test.js
  Find: at what RPS does p99 latency exceed SLA?
  
Define capacity thresholds:
  Alert at 60% of breaking capacity
  Auto-scale at 70%
  Emergency playbook at 80%

Architecture review:
  What would need to change for 1000x traffic?
  Identify the next three bottlenecks BEFORE you hit them
  
Cost review:
  What did this cost? ($X in emergency EC2 + engineer time)
  How to prevent next time? (auto-scaling, load testing in CI/CD)
```

**The key mental model:**
```
Traffic spike response = fire fighting
  Instrument (see the fire)
  Contain (stop it spreading)
  Extinguish (fix the cause)
  Prevent recurrence (sprinkler system)

Never optimize prematurely,
but always have a playbook for when the load arrives.
```

---

## Quick Reference

### Consistency Decision Matrix
| Feature | Consistency | Reason |
|---|---|---|
| Balance, payments | Strong | Money must be exact |
| Feed, like count | Eventual | Approximation is fine |
| Auth tokens | Strong | Security |
| UI preferences | Eventual | Harmless if stale |

### N+1 Solutions
- Database: `select_related()` / JOIN / `prefetch_related()`
- API: Batch endpoints (`/users/batch?ids=1,2,3`)
- GraphQL: DataLoader (batch + cache per request)

### Thundering Herd Solutions
1. Cache mutex (only one filler)
2. Probabilistic early refresh (XFetch)
3. Staggered expiry (add random jitter to TTL)
4. Client jitter on reconnect

### Schema Migration (Zero Downtime)
```
Expand → (Dual write) → Contract
Add column (nullable) → Backfill → Enforce constraint → Drop old column
NEVER: DROP, RENAME, or ADD NOT NULL in one step
```

### Monorepo vs Polyrepo
- Monorepo: atomic cross-service changes, shared libraries, good tooling required
- Polyrepo: independent teams, different cadences, access control

### Build vs Buy Heuristics
- Differentiates product → Build
- Commodity infrastructure → Buy managed service
- Open source + self-host = middle ground

### Feature Flag Architecture
```
Flag Store → Config Service → SDK (local cache, 30s refresh)
Application evaluates locally → microsecond latency
Kill switch: pub/sub notification to all SDK instances
```

### Back-pressure Mechanisms
| System | Mechanism |
|---|---|
| TCP | Receive window |
| Kafka | Consumer pull rate |
| Reactive Streams | request(n) backpressure |
| Node.js streams | pipe() with highWaterMark |

### 100x Traffic Response Priority
1. Scale app servers (horizontal, immediate)
2. Add DB read replicas (20-30 min)
3. Increase cache TTLs (immediate)
4. Rate limit at CDN/gateway (immediate)
5. Profile and fix top 3 slowest endpoints (hours)
6. Load test to find actual breaking point (post-incident)

### Service Mesh vs API Gateway
- API Gateway = north-south traffic, external auth, API management
- Service Mesh = east-west traffic, mTLS, circuit breaking, tracing
- Both needed for large microservices deployments
