# 32 — Data Modeling and Schema Design

---

## Easy (Q1–Q7)

---

### Q1. What is normalization vs denormalization? When do you choose each?

**Answer:**

**Normalization** organizes data to eliminate redundancy by splitting data into related tables and enforcing referential integrity. **Denormalization** deliberately introduces redundancy to improve read performance by combining tables.

**Normalization forms (1NF → 3NF is standard target):**
```
1NF: Atomic values, no repeating groups
  Bad:  orders(id, items="pen,notebook,ruler")
  Good: orders(id), order_items(order_id, item_name)

2NF: No partial dependencies (all non-key columns depend on full PK)
  Bad:  order_items(order_id, product_id, product_name) ← product_name depends only on product_id
  Good: products(product_id, product_name), order_items(order_id, product_id)

3NF: No transitive dependencies
  Bad:  employees(id, dept_id, dept_name) ← dept_name depends on dept_id, not id
  Good: departments(dept_id, dept_name), employees(id, dept_id)
```

**Decision matrix:**

| Factor | Normalize | Denormalize |
|--------|-----------|-------------|
| Workload | OLTP (many writes) | OLAP / read-heavy |
| Data integrity | Critical | Acceptable eventual consistency |
| Storage | Constrained | Abundant |
| Query patterns | Unknown / varied | Known, repeated aggregate queries |
| Update frequency | High | Low (batch updates ok) |
| Join cost | Acceptable | Too high (100M+ row joins) |

**Practical example:**

```sql
-- Normalized (OLTP)
SELECT o.id, u.name, p.name, oi.quantity
FROM orders o
JOIN users u ON o.user_id = u.id
JOIN order_items oi ON oi.order_id = o.id
JOIN products p ON oi.product_id = p.id
WHERE o.id = 42;
-- Fast single-row lookups, correct after updates

-- Denormalized (OLAP/reporting)
SELECT order_id, user_name, product_name, quantity
FROM orders_flat
WHERE order_id = 42;
-- One-table scan, no joins, fast aggregations
-- Stale until ETL refresh runs
```

**Rule of thumb:** Start normalized, denormalize only when you have measured performance data showing that joins are the bottleneck.

---

### Q2. What are the trade-offs between surrogate keys (auto-increment/UUID) and natural keys?

**Answer:**

A **natural key** uses a real-world attribute as the primary key (email address, SSN, product SKU). A **surrogate key** is a system-generated identifier with no real-world meaning (auto-increment integer, UUID).

**Comparison table:**

| Dimension | Natural Key | Surrogate Key |
|-----------|-------------|---------------|
| Meaningfulness | Business-meaningful | Opaque, no meaning |
| Uniqueness | Must enforce (may have violations) | System-guaranteed |
| Stability | Can change (email changes, SKU reformat) | Never changes |
| Join efficiency | Often varchar/large → slow joins | Integer/UUID → fast joins |
| URL exposure | Exposes business data (email) | Hides internal structure |
| Debugging | Easier ("find user bob@example.com") | Harder (need to know UUID) |
| Index size | Often large (varchar(255)) | Small (4 or 16 bytes) |

**Natural key failure example:**
```sql
-- Natural key: email
CREATE TABLE users (email VARCHAR(255) PRIMARY KEY, name TEXT);
CREATE TABLE orders (user_email VARCHAR(255) REFERENCES users(email));

-- User changes email: CASCADE UPDATE required across all tables
UPDATE users SET email = 'new@x.com' WHERE email = 'old@x.com';
-- Also updates all orders referencing old email — dangerous at scale
```

**Surrogate key (correct approach):**
```sql
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,      -- surrogate
    email VARCHAR(255) UNIQUE NOT NULL,  -- natural key still has UNIQUE constraint
    name TEXT NOT NULL
);
CREATE TABLE orders (user_id BIGINT REFERENCES users(id));
-- User changes email: only one row updated, no cascade needed
```

**When natural keys are acceptable:**
- Junction/link tables (user_id, role_id composite PK)
- Lookup/reference tables (country_code CHAR(2) PRIMARY KEY)
- When the natural key is truly immutable and globally unique (ISO standard codes)

**Recommendation:** Always use a surrogate key as the PK; add a UNIQUE constraint on natural keys for integrity.

---

### Q3. Compare UUID v4, ULID, and Snowflake ID for distributed systems.

**Answer:**

In distributed systems, IDs must be globally unique without coordination. The choice affects sort order, index efficiency, URL safety, and operational debuggability.

**UUID v4:**
```
Format:  550e8400-e29b-41d4-a716-446655440000
Size:    128 bits (16 bytes), string representation: 36 chars
Source:  Cryptographically random (122 random bits)
```

**ULID (Universally Unique Lexicographically Sortable Identifier):**
```
Format:  01ARZ3NDEKTSV4RRFFQ69G5FAV
Size:    128 bits, string: 26 chars (Crockford base32)
Source:  48-bit millisecond timestamp + 80 random bits
```

**Snowflake ID (Twitter/Discord):**
```
Format:  64-bit integer (e.g., 1234567890123456789)
Size:    64 bits (8 bytes)
Source:  41-bit timestamp + 10-bit machine ID + 12-bit sequence
```

**Detailed comparison:**

| Property | UUID v4 | ULID | Snowflake |
|----------|---------|------|-----------|
| Size | 16 bytes | 16 bytes | 8 bytes |
| Sortable by creation time | No | Yes (ms precision) | Yes (ms precision) |
| B-tree index efficiency | Poor (random inserts cause splits) | Good (mostly sequential) | Excellent |
| Coordination needed | None | None | Machine ID assignment |
| Theoretical uniqueness | 5.3 × 10^36 | 2.8 × 10^24/ms | 4096/ms/machine |
| URL safe | No (contains hyphens) | Yes | Yes |
| Embeds creation time | No | Yes | Yes |
| Monotonic within same ms | No | Yes (optional) | Yes |

**Index performance illustration:**
```
UUID v4 inserts into B-tree index:
  Random IDs → random page writes → B-tree page splits → fragmentation
  1M rows: ~40% index bloat typical

ULID/Snowflake inserts:
  Time-sorted → mostly sequential writes → minimal page splits
  1M rows: < 5% index bloat
```

**When to use each:**
- **UUID v4**: No coordination possible, privacy important, small-scale
- **ULID**: Distributed, need sortability, want simple implementation
- **Snowflake**: High-throughput, need smallest possible ID, can assign machine IDs

```python
# ULID generation
import ulid
id = ulid.new()  # 01ARZ3NDEKTSV4RRFFQ69G5FAV
created_at = id.timestamp().datetime  # Extract creation time

# Snowflake (simplified)
def snowflake_id(machine_id: int) -> int:
    ts = int(time.time() * 1000) - EPOCH
    seq = next_sequence()
    return (ts << 22) | (machine_id << 12) | seq
```

---

### Q4. What is the difference between star schema and snowflake schema for analytical workloads?

**Answer:**

Both are dimensional modeling patterns for data warehouses (OLAP). They organize data into a central fact table surrounded by dimension tables.

**Star Schema:**
```
              ┌──────────────┐
              │  dim_product │
              │  product_id  │
              │  name        │
              │  category    │  ← Denormalized: category stored here
              │  brand       │     (not in a separate dim_category table)
              └──────┬───────┘
                     │
┌──────────┐  ┌──────▼───────────┐  ┌──────────────┐
│ dim_date │  │   fact_sales      │  │  dim_customer │
│ date_id  ├──┤ date_id           ├──┤ customer_id  │
│ year     │  │ product_id        │  │ name         │
│ quarter  │  │ customer_id       │  │ city         │
│ month    │  │ store_id          │  │ country      │  ← Denormalized
└──────────┘  │ quantity_sold     │  └──────────────┘
              │ revenue           │
              └──────┬────────────┘
                     │
              ┌──────▼───────┐
              │  dim_store   │
              │  store_id    │
              └──────────────┘
```

**Snowflake Schema:**
```
dim_product → dim_category → dim_department  (normalized dimension)
dim_customer → dim_city → dim_country        (normalized dimension)

Each dimension is further normalized into sub-dimensions.
```

**Comparison:**

| Property | Star Schema | Snowflake Schema |
|----------|-------------|------------------|
| Joins per query | Fewer (1 join per dimension) | More (multiple joins per dimension) |
| Storage | More (denormalized redundancy) | Less (normalized) |
| Query performance | Faster | Slower (more joins) |
| ETL complexity | Simpler | More complex |
| Data consistency | Managed by ETL | Enforced by FK |
| Tool compatibility | Better (BI tools prefer star) | Okay |
| Typical use | Most OLAP/BI workloads | When storage is critical |

**Query example — Star is faster:**
```sql
-- Star schema: 4 joins
SELECT d.year, p.category, SUM(f.revenue)
FROM fact_sales f
JOIN dim_date d ON f.date_id = d.date_id
JOIN dim_product p ON f.product_id = p.product_id
GROUP BY d.year, p.category;

-- Snowflake schema: 6 joins (extra category and department joins)
SELECT d.year, cat.name, SUM(f.revenue)
FROM fact_sales f
JOIN dim_date d ON f.date_id = d.date_id
JOIN dim_product p ON f.product_id = p.product_id
JOIN dim_category cat ON p.category_id = cat.category_id
GROUP BY d.year, cat.name;
```

**Industry practice:** Star schema is preferred in most modern data warehouses (Snowflake DW, BigQuery, Redshift). Use snowflake schema only when dimensions have very high cardinality or strict normalization requirements.

---

### Q5. What are Slowly Changing Dimensions (SCD) Types 1, 2, and 3? How do they track history?

**Answer:**

SCDs handle how dimension data changes over time in a data warehouse. The choice determines how much history is preserved.

**SCD Type 1 — Overwrite (no history):**
```sql
-- Customer moves city: just update
UPDATE dim_customer SET city = 'New York' WHERE customer_id = 42;
-- Previous city 'Boston' is lost forever
-- Use when: history doesn't matter (fix data errors, irrelevant attributes)
```

**SCD Type 2 — Add new row (full history):**
```sql
CREATE TABLE dim_customer (
    sk          BIGSERIAL PRIMARY KEY,  -- surrogate key
    customer_id INT NOT NULL,           -- natural/business key
    name        VARCHAR(100),
    city        VARCHAR(100),
    is_current  BOOLEAN DEFAULT TRUE,
    valid_from  DATE NOT NULL,
    valid_to    DATE DEFAULT '9999-12-31'
);

-- Customer moves from Boston to New York on 2025-06-01:
-- Step 1: Expire old row
UPDATE dim_customer
SET is_current = FALSE, valid_to = '2025-05-31'
WHERE customer_id = 42 AND is_current = TRUE;

-- Step 2: Insert new row
INSERT INTO dim_customer (customer_id, name, city, valid_from)
VALUES (42, 'Alice', 'New York', '2025-06-01');
```

Result:
```
sk | customer_id | city      | is_current | valid_from | valid_to
1  | 42          | Boston    | FALSE      | 2020-01-01 | 2025-05-31
2  | 42          | New York  | TRUE       | 2025-06-01 | 9999-12-31
```

Historical fact rows join to the correct dimension row based on date ranges.

**SCD Type 3 — Previous column (limited history):**
```sql
CREATE TABLE dim_customer (
    customer_id    INT PRIMARY KEY,
    name           VARCHAR(100),
    current_city   VARCHAR(100),
    previous_city  VARCHAR(100),   -- Only keeps ONE previous value
    city_changed   DATE
);
-- Customer moves: current becomes previous, new is current
```

**Comparison:**

| Type | History Kept | Storage | Complexity | Use Case |
|------|-------------|---------|------------|----------|
| SCD 1 | None | Minimal | Simple | Corrections, irrelevant attrs |
| SCD 2 | Full | High | Medium | Most business tracking |
| SCD 3 | One previous | Low | Simple | Two-state tracking only |

Type 2 is the most commonly used SCD in enterprise data warehousing.

---

### Q6. What is the EAV (Entity-Attribute-Value) pattern? Why is it used and why is it problematic?

**Answer:**

EAV stores arbitrary attributes as rows instead of columns:

**Standard schema:**
```sql
CREATE TABLE products (
    id INT PRIMARY KEY, name TEXT, price DECIMAL, weight DECIMAL
);
-- Fixed set of known columns
```

**EAV schema:**
```sql
CREATE TABLE entities (id INT PRIMARY KEY, type VARCHAR(50));
CREATE TABLE attributes (id INT PRIMARY KEY, name VARCHAR(100));
CREATE TABLE values (
    entity_id INT REFERENCES entities(id),
    attribute_id INT REFERENCES attributes(id),
    value TEXT,  -- Everything is a string!
    PRIMARY KEY (entity_id, attribute_id)
);
```

**Why EAV is used:**
- Schema-less flexibility: add new attributes without ALTER TABLE
- SaaS products where each customer defines custom fields
- Medical records (patients have different tests with different attributes)
- CMS/e-commerce with arbitrary product attributes (size, color, material vary by category)

**Why EAV is problematic:**

```sql
-- Retrieve a product with 5 attributes (EAV):
SELECT
    MAX(CASE WHEN a.name = 'color' THEN v.value END) AS color,
    MAX(CASE WHEN a.name = 'size'  THEN v.value END) AS size,
    MAX(CASE WHEN a.name = 'weight' THEN v.value END) AS weight
FROM entities e
JOIN values v ON e.id = v.entity_id
JOIN attributes a ON v.attribute_id = a.id
WHERE e.id = 42
GROUP BY e.id;
-- This gets worse with each additional attribute
```

**Problems:**

| Issue | Description |
|-------|-------------|
| No data types | Everything stored as TEXT; money stored as "19.99" string |
| No constraints | Can't enforce NOT NULL, FK, or CHECK on values |
| Query complexity | Pivoting rows to columns is expensive and ugly |
| Performance | Cannot index value column efficiently across attributes |
| No schema documentation | No way to see what attributes exist without querying |
| Reporting nightmare | BI tools can't understand EAV structure |

**Better alternatives:**
- PostgreSQL JSONB: flexible schema with indexing support
- Separate tables per product type (table-per-hierarchy)
- RDBMS + document store hybrid

---

### Q7. When should you use JSONB columns in PostgreSQL instead of a separate NoSQL database?

**Answer:**

PostgreSQL's JSONB type stores parsed binary JSON with full indexing support, bridging the gap between relational and document models.

**JSONB advantages over separate NoSQL:**
```sql
-- Store flexible product metadata alongside structured data
CREATE TABLE products (
    id BIGSERIAL PRIMARY KEY,
    sku VARCHAR(50) UNIQUE NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    category_id INT REFERENCES categories(id),
    attributes JSONB  -- flexible per-product attributes
);

-- GIN index for fast JSON queries
CREATE INDEX idx_products_attrs ON products USING GIN(attributes);

-- Query: find all red products under $50
SELECT sku, price, attributes->>'color'
FROM products
WHERE attributes @> '{"color": "red"}'
  AND price < 50;

-- Update nested JSON
UPDATE products
SET attributes = jsonb_set(attributes, '{dimensions, weight}', '1.5')
WHERE id = 42;
```

**When to use JSONB (keep in PostgreSQL):**

| Scenario | Reasoning |
|----------|-----------|
| Occasional flexible fields (< 20%) | Not worth operating another DB |
| Need ACID transactions across JSON + relational data | Single DB transaction |
| Schema varies by product/entity type | JSONB handles per-row variability |
| JSON is queried infrequently | GIN index adequate |
| Small-medium scale (< 100M JSON docs) | PostgreSQL handles well |

**When to use a separate document DB (MongoDB, DynamoDB):**

| Scenario | Reasoning |
|----------|-----------|
| JSON is the primary data model (> 80% of queries hit JSON) | Document DB is optimized for this |
| Deeply nested documents (> 5 levels) | JSONB query syntax becomes complex |
| Need document-level horizontal sharding | MongoDB has native shard keys |
| Schema-free at massive scale (> 1B documents) | Dedicated document DB wins |
| Team has no PostgreSQL expertise | Operational risk |

**Cost of NoSQL vs JSONB:**
```
Separate MongoDB cluster:
  - Extra infrastructure cost (3+ nodes)
  - Cross-service join impossible (must join in application code)
  - Two systems to monitor, backup, upgrade
  - No ACID across PostgreSQL + MongoDB

JSONB in PostgreSQL:
  - Zero extra infrastructure
  - Join JSON fields with relational data in one query
  - Single ACID transaction
  - One system to operate
```

**Rule:** Use JSONB when the JSON is an extension of relational data. Use a document DB when JSON IS the relational data.

---

## Medium (Q8–Q15)

---

### Q8. How do you design a multi-tenant schema? Compare shared table + RLS vs separate schemas vs separate databases.

**Answer:**

Multi-tenant architecture determines data isolation, operational complexity, and cost. Three main patterns exist:

**Pattern 1: Shared table + Row-Level Security (RLS)**
```sql
-- All tenants in one table
CREATE TABLE orders (
    id BIGSERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL,  -- Every table has this column
    user_id BIGINT,
    total DECIMAL(10,2),
    created_at TIMESTAMP
);

-- RLS policy: tenants only see their own rows
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON orders
    USING (tenant_id = current_setting('app.tenant_id')::UUID);

-- Application sets context at connection time
SET LOCAL app.tenant_id = '550e8400-e29b-41d4-a716-446655440000';
-- All subsequent queries automatically filtered by tenant
```

**Pattern 2: Separate schemas per tenant**
```sql
-- Schema per tenant
CREATE SCHEMA tenant_acme;
CREATE SCHEMA tenant_globex;

-- Same tables in each schema
CREATE TABLE tenant_acme.orders (...);
CREATE TABLE tenant_globex.orders (...);

-- Set search_path per connection
SET search_path = tenant_acme;
-- Now: SELECT * FROM orders; hits tenant_acme.orders
```

**Pattern 3: Separate database per tenant**
```
tenant_acme  → postgres://db-acme.internal/acme_db
tenant_globex → postgres://db-globex.internal/globex_db
```

**Comparison:**

| Dimension | Shared Table + RLS | Separate Schemas | Separate Databases |
|-----------|-------------------|-----------------|-------------------|
| Isolation level | Row-level (logical) | Schema (logical) | Full (physical) |
| Data breach risk | Medium (policy bug exposes all) | Low | Minimal |
| Compliance (HIPAA/PCI) | May not satisfy | Often acceptable | Best |
| Operational cost | Low | Medium | High |
| Cross-tenant queries | Easy (admin bypasses RLS) | Per-schema queries | Need data lake |
| Tenant-specific schema | Impossible | Possible | Full flexibility |
| Max tenants | Millions | Thousands | Hundreds |
| DB migrations | One migration | Run per schema | Run per DB |
| Performance isolation | None | None | Full |

**Choosing the right pattern:**
```
SMB SaaS (Slack, Notion model):
  → Shared table + RLS
  → Thousands of small tenants, similar needs

Mid-market SaaS:
  → Separate schemas
  → Hundreds of tenants, some customization needed

Enterprise/regulated (Salesforce, enterprise banking):
  → Separate databases
  → Dozens of large tenants with strict isolation requirements
```

**Hybrid approach (most common in practice):**
- Standard tenants: shared tables + RLS
- Enterprise tenants (paying premium): dedicated schema or database
- This lets you scale efficiently while offering isolation to high-value customers

---

### Q9. How do you model time-series data in a relational database? Discuss BRIN indexes, partitioning, and downsampling.

**Answer:**

Time-series data has unique characteristics: high write volume, time-ordered access, range queries, and data that becomes less relevant over time.

**Basic schema:**
```sql
CREATE TABLE metrics (
    time       TIMESTAMPTZ NOT NULL,
    device_id  UUID NOT NULL,
    metric     VARCHAR(50) NOT NULL,
    value      DOUBLE PRECISION NOT NULL
) PARTITION BY RANGE (time);

-- Create monthly partitions
CREATE TABLE metrics_2025_05
    PARTITION OF metrics
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');

CREATE TABLE metrics_2025_06
    PARTITION OF metrics
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
```

**Why BRIN indexes (Block Range INdexes):**
```
B-tree index on 1B time-series rows:
  Size: ~20GB (stores value for every row)
  Insert cost: High (random writes into tree)

BRIN index on same data:
  Size: ~256KB (stores min/max per 128-page block)
  Insert cost: Near-zero (just updates min/max of last block)
  Query: Scans only blocks where min <= query_time <= max
  
BRIN works well because time-series data is naturally ordered:
  Block 1: rows 1-128 → time range 2025-05-01 to 2025-05-01 00:01
  Block 2: rows 129-256 → time range 2025-05-01 00:01 to 2025-05-01 00:02
  
Query WHERE time > '2025-05-01 12:00':
  BRIN skips blocks 1 through ~720 in one operation
```

```sql
-- BRIN index creation
CREATE INDEX CONCURRENTLY idx_metrics_time_brin
ON metrics USING BRIN (time)
WITH (pages_per_range = 128);
```

**Downsampling (retention policy):**
```sql
-- Raw data: 10-second granularity, keep 7 days
-- 1-minute aggregates: keep 30 days
-- 1-hour aggregates: keep 1 year
-- 1-day aggregates: keep forever

-- Continuous aggregate (TimescaleDB syntax)
CREATE MATERIALIZED VIEW metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    device_id,
    metric,
    AVG(value) AS avg_value,
    MAX(value) AS max_value,
    MIN(value) AS min_value,
    COUNT(*) AS sample_count
FROM metrics
GROUP BY bucket, device_id, metric;

-- Retention policy: drop raw data after 7 days
SELECT add_retention_policy('metrics', INTERVAL '7 days');
SELECT add_retention_policy('metrics_hourly', INTERVAL '30 days');
```

**Partition pruning efficiency:**
```sql
-- Without partitioning: scans all 365 daily partitions
-- With partitioning:
EXPLAIN SELECT * FROM metrics WHERE time BETWEEN '2025-05-01' AND '2025-05-02';
-- Shows: Seq Scan on metrics_2025_05 (only 1 of 12 partitions!)
```

**Column ordering for compression:**
```sql
-- Bad: device_id, time (random order per device)
-- Good: time, device_id (time-ordered; better compression & BRIN efficiency)
```

For heavy time-series workloads (> 100k writes/sec), consider TimescaleDB (PostgreSQL extension) or InfluxDB.

---

### Q10. Explain hierarchical data modeling: adjacency list, materialized path, nested sets, and closure table.

**Answer:**

Hierarchical data (org charts, file systems, comment threads, category trees) can be modeled four ways, each with different read/write trade-offs.

**The hierarchy example:**
```
CEO
├── CTO
│   ├── Engineering Manager
│   │   ├── Alice (Engineer)
│   │   └── Bob (Engineer)
│   └── DevOps Lead
└── CFO
    └── Finance Manager
```

**1. Adjacency List (simplest)**
```sql
CREATE TABLE employees (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    parent_id INT REFERENCES employees(id)
);

-- Find all direct reports of CTO:
SELECT * FROM employees WHERE parent_id = (SELECT id FROM employees WHERE name = 'CTO');

-- Find full subtree (requires recursive CTE):
WITH RECURSIVE subtree AS (
    SELECT id, name, parent_id FROM employees WHERE name = 'CTO'
    UNION ALL
    SELECT e.id, e.name, e.parent_id
    FROM employees e JOIN subtree s ON e.parent_id = s.id
)
SELECT * FROM subtree;
```
- Simple to understand and maintain
- Full tree traversal requires recursive SQL (slow on deep trees)

**2. Materialized Path**
```sql
CREATE TABLE categories (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    path VARCHAR(500)  -- e.g., '/1/3/7/12/'
);

-- Find all descendants of node 3:
SELECT * FROM categories WHERE path LIKE '/1/3/%';

-- Find all ancestors of node 12:
-- path = '/1/3/7/12/' → ancestors are 1, 3, 7
```
- Fast subtree queries with LIKE (can use varchar prefix index)
- Path can become long; changing parent requires updating all descendants

**3. Nested Sets (Preorder tree traversal)**
```sql
CREATE TABLE categories (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    lft INT NOT NULL,
    rgt INT NOT NULL
);
-- CEO: lft=1, rgt=14
-- CTO: lft=2, rgt=9
-- Alice: lft=4, rgt=5

-- Find all descendants of CTO (lft=2, rgt=9):
SELECT * FROM categories WHERE lft > 2 AND rgt < 9;
-- No recursion needed! O(1) query

-- But: inserting/moving nodes requires updating many lft/rgt values
```

**4. Closure Table (best general-purpose)**
```sql
CREATE TABLE employees (id INT PRIMARY KEY, name VARCHAR(100));

CREATE TABLE employee_hierarchy (
    ancestor_id   INT REFERENCES employees(id),
    descendant_id INT REFERENCES employees(id),
    depth         INT NOT NULL,
    PRIMARY KEY (ancestor_id, descendant_id)
);
-- Every node stores a row for EACH of its ancestors (including itself at depth=0)

-- Find all reports under CTO (id=2):
SELECT e.* FROM employees e
JOIN employee_hierarchy h ON e.id = h.descendant_id
WHERE h.ancestor_id = 2;
-- Fast! Single index lookup

-- Find all managers above Alice (id=5):
SELECT e.* FROM employees e
JOIN employee_hierarchy h ON e.id = h.ancestor_id
WHERE h.descendant_id = 5
ORDER BY h.depth;
```

**Summary:**

| Method | Read (subtree) | Read (ancestors) | Insert | Move | Storage |
|--------|---------------|-----------------|--------|------|---------|
| Adjacency List | Recursive CTE | Recursive CTE | O(1) | O(1) | Low |
| Materialized Path | O(log n) LIKE | Parse string | O(1) | O(n) | Low |
| Nested Sets | O(1) | O(1) | O(n) | O(n) | Low |
| Closure Table | O(1) | O(1) | O(depth) | O(n) | High |

Use closure table for most web applications. Use nested sets only for read-heavy trees with rare modifications.

---

### Q11. How do you design a schema for a notification system with notifications, preferences, and delivery log?

**Answer:**

A notification system requires: generating notifications, respecting user preferences, and tracking delivery state across channels.

**Core schema:**
```sql
-- Notification templates (reusable, localizable)
CREATE TABLE notification_templates (
    id          SERIAL PRIMARY KEY,
    type        VARCHAR(50) UNIQUE NOT NULL,  -- 'order_shipped', 'password_reset'
    channel     VARCHAR(20) NOT NULL,          -- 'email', 'sms', 'push', 'in_app'
    subject     TEXT,
    body        TEXT NOT NULL,               -- Handlebars/Jinja template
    variables   JSONB                        -- Schema of expected variables
);

-- Actual notifications (one per user per event)
CREATE TABLE notifications (
    id            BIGSERIAL PRIMARY KEY,
    user_id       BIGINT NOT NULL REFERENCES users(id),
    template_id   INT REFERENCES notification_templates(id),
    channel       VARCHAR(20) NOT NULL,
    subject       TEXT,
    body          TEXT NOT NULL,    -- Rendered body (stored for audit)
    variables     JSONB,            -- Variables used for rendering
    status        VARCHAR(20) NOT NULL DEFAULT 'pending',
                  -- pending | queued | sent | delivered | failed | read
    priority      SMALLINT NOT NULL DEFAULT 5,  -- 1=critical, 10=marketing
    scheduled_at  TIMESTAMPTZ DEFAULT NOW(),
    sent_at       TIMESTAMPTZ,
    read_at       TIMESTAMPTZ,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_notif_user_status ON notifications(user_id, status, created_at DESC);
CREATE INDEX idx_notif_scheduled ON notifications(scheduled_at) WHERE status = 'pending';

-- User preferences per channel and notification type
CREATE TABLE notification_preferences (
    user_id         BIGINT NOT NULL REFERENCES users(id),
    notification_type VARCHAR(50) NOT NULL,
    channel         VARCHAR(20) NOT NULL,
    enabled         BOOLEAN NOT NULL DEFAULT TRUE,
    quiet_hours_start TIME,           -- Don't disturb 22:00
    quiet_hours_end   TIME,           -- Don't disturb until 08:00
    frequency_limit   INT,            -- Max N per day
    PRIMARY KEY (user_id, notification_type, channel)
);

-- Delivery log (append-only, every state transition)
CREATE TABLE notification_delivery_log (
    id              BIGSERIAL PRIMARY KEY,
    notification_id BIGINT NOT NULL REFERENCES notifications(id),
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event           VARCHAR(30) NOT NULL,  -- 'queued','sent','delivered','bounced','failed'
    channel         VARCHAR(20) NOT NULL,
    provider        VARCHAR(50),           -- 'sendgrid', 'twilio', 'apns', 'fcm'
    provider_msg_id VARCHAR(200),          -- External ID for tracking
    metadata        JSONB,                 -- Error details, open tracking
    CONSTRAINT delivery_log_no_delete CHECK (TRUE)  -- Reminder: never DELETE
);

CREATE INDEX idx_delivery_log_notif ON notification_delivery_log(notification_id, timestamp);
```

**Notification processing flow:**
```
Event occurs (order shipped)
    ↓
NotificationService.create_notification(user_id, type='order_shipped', vars={order_id: 42})
    ↓
Check notification_preferences: Is email enabled for this user + type?
    ↓ Yes
Render template with vars → stored as notification.body
    ↓
Check quiet hours: Is it 23:00 for this user?
    ↓ Yes → schedule_at = tomorrow 08:00
    ↓ No  → schedule_at = NOW()
    ↓
Insert into notifications (status='pending')
    ↓
Worker picks up pending notifications, sends via provider
    ↓
Insert delivery_log event: 'sent' with provider_msg_id
    ↓
Provider webhook → 'delivered' or 'bounced'
    ↓
Insert delivery_log event, update notifications.status
```

**Querying unread in-app notifications:**
```sql
SELECT id, subject, body, created_at
FROM notifications
WHERE user_id = $1
  AND channel = 'in_app'
  AND status != 'read'
ORDER BY priority ASC, created_at DESC
LIMIT 20;
```

---

### Q12. Explain the expand-contract pattern for zero-downtime schema migrations.

**Answer:**

The expand-contract (also called parallel-change) pattern allows schema changes without taking the database or application offline.

**Problem with naive migrations:**
```
v1 code runs against old schema
            ↓
ALT TABLE: rename column user_name → full_name
            ↓
v2 code runs against new schema

During migration: v1 code uses user_name, schema says full_name → ERROR
```

**Expand-contract in three phases:**

```
Phase 1: EXPAND (backward compatible addition)
  - Add new column (full_name) WITHOUT removing old (user_name)
  - Deploy v1.5 code: writes to BOTH columns, reads from user_name
  - Old v1 code: still reads/writes user_name (no breakage)

Phase 2: MIGRATE (backfill)
  - Backfill full_name for all existing rows:
    UPDATE users SET full_name = user_name WHERE full_name IS NULL;
  - Deploy v2 code: writes to BOTH, reads from full_name
  - Verify: all rows have full_name populated

Phase 3: CONTRACT (remove old)
  - Only after v1 code is 100% retired
  - Deploy v2.5 code: writes only to full_name
  - Drop old column: ALTER TABLE users DROP COLUMN user_name;
```

**Timeline:**
```
Week 1:  Deploy schema change (ADD COLUMN full_name)
Week 1:  Deploy v1.5 app (dual write)
Week 2:  Run backfill migration
Week 2:  Deploy v2 app (reads from full_name)
Week 3:  Verify no app reads user_name in logs
Week 4:  Deploy v2.5 (remove dual write)
Week 4:  Drop old column
```

**SQL migration steps:**
```sql
-- Phase 1: EXPAND
ALTER TABLE users ADD COLUMN full_name VARCHAR(200);
-- Immediately safe — adds nullable column, no lock on most databases

-- Phase 2: BACKFILL (do in batches to avoid lock)
DO $$
DECLARE
    batch_size INT := 10000;
    last_id BIGINT := 0;
BEGIN
    LOOP
        UPDATE users
        SET full_name = user_name
        WHERE id > last_id AND full_name IS NULL
        LIMIT batch_size
        RETURNING MAX(id) INTO last_id;
        EXIT WHEN NOT FOUND;
        PERFORM pg_sleep(0.1);  -- Reduce I/O pressure
    END LOOP;
END $$;

-- Add NOT NULL constraint after backfill (two-step)
ALTER TABLE users ADD CONSTRAINT full_name_not_null CHECK (full_name IS NOT NULL) NOT VALID;
ALTER TABLE users VALIDATE CONSTRAINT full_name_not_null;  -- Validates without full table lock

-- Phase 3: CONTRACT (weeks later)
ALTER TABLE users DROP COLUMN user_name;
```

**Key rule:** Never do a breaking change (rename, drop, type change) in a single deployment. Always use expand-contract.

---

### Q13. How do you design an append-only ledger schema for financial systems using double-entry bookkeeping?

**Answer:**

Financial ledgers must be immutable (append-only), auditable, and balanced. Double-entry bookkeeping ensures every transaction affects at least two accounts and the sum of all entries is always zero.

**Core principle:**
```
Every transaction: debits = credits
  Debit account A: +$100
  Credit account B: -$100
  Net: 0 (always balanced)

Account types:
  Assets:     Debit increases, Credit decreases
  Liabilities: Credit increases, Debit decreases
  Revenue:    Credit increases
  Expenses:   Debit increases
```

**Schema:**
```sql
-- Accounts (wallets, bank accounts, virtual ledger accounts)
CREATE TABLE accounts (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT REFERENCES users(id),
    type        VARCHAR(30) NOT NULL,    -- 'asset', 'liability', 'revenue', 'expense'
    name        VARCHAR(100) NOT NULL,   -- 'User Wallet - Alice', 'Platform Revenue'
    currency    CHAR(3) NOT NULL DEFAULT 'USD',
    created_at  TIMESTAMPTZ DEFAULT NOW()
    -- NO BALANCE COLUMN: balance computed from entries
);

-- Transactions (immutable metadata)
CREATE TABLE transactions (
    id            BIGSERIAL PRIMARY KEY,
    reference     VARCHAR(100) UNIQUE NOT NULL,  -- Idempotency key
    description   TEXT,
    initiated_by  BIGINT REFERENCES users(id),
    created_at    TIMESTAMPTZ DEFAULT NOW()
    -- NEVER UPDATE OR DELETE
);

-- Journal entries (immutable double-entry lines)
CREATE TABLE journal_entries (
    id             BIGSERIAL PRIMARY KEY,
    transaction_id BIGINT NOT NULL REFERENCES transactions(id),
    account_id     BIGINT NOT NULL REFERENCES accounts(id),
    amount         NUMERIC(20, 8) NOT NULL,  -- Positive = debit, Negative = credit
    currency       CHAR(3) NOT NULL,
    description    TEXT,
    created_at     TIMESTAMPTZ DEFAULT NOW()
    -- NEVER UPDATE OR DELETE
);

-- Enforce immutability via trigger
CREATE OR REPLACE FUNCTION prevent_journal_modification()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Journal entries are immutable. Create a reversal entry instead.';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER no_journal_updates
    BEFORE UPDATE OR DELETE ON journal_entries
    FOR EACH ROW EXECUTE FUNCTION prevent_journal_modification();

-- Constraint: sum of entries per transaction = 0
-- (Enforced at application level or via check function)
```

**Recording a payment ($50 from user wallet to merchant):**
```sql
BEGIN;
-- Insert transaction
INSERT INTO transactions (reference, description)
VALUES ('PAY-20250511-001', 'Payment: Alice to Merchant A')
RETURNING id INTO tx_id;

-- Debit user wallet (reduces asset = credit)
INSERT INTO journal_entries (transaction_id, account_id, amount)
VALUES (tx_id, user_wallet_account_id, -50.00);

-- Credit merchant wallet (increases asset = debit)
INSERT INTO journal_entries (transaction_id, account_id, amount)
VALUES (tx_id, merchant_wallet_account_id, +50.00);

-- Verify balance (safety check)
SELECT SUM(amount) FROM journal_entries WHERE transaction_id = tx_id;
-- Must equal 0, else ROLLBACK

COMMIT;
```

**Computing balance (fast with indexed aggregate):**
```sql
-- Current balance of account 42
SELECT SUM(amount) AS balance
FROM journal_entries
WHERE account_id = 42;

-- Balance at a specific point in time
SELECT SUM(amount) AS balance_on_date
FROM journal_entries
WHERE account_id = 42 AND created_at < '2025-05-01';
```

**Correction = reversal, not update:**
```sql
-- Wrong entry: debit was $50, should be $30
-- Step 1: Reverse original
INSERT INTO journal_entries (transaction_id, account_id, amount)
VALUES (reversal_tx_id, user_wallet_account_id, +50.00);  -- Reverse debit
-- Step 2: Apply correct amount
INSERT INTO journal_entries (transaction_id, account_id, amount)
VALUES (correction_tx_id, user_wallet_account_id, -30.00);
```

---

### Q14. How do you design a schema for tracking a follower graph at scale (millions of users)?

**Answer:**

The follower graph (Twitter/Instagram model) is a many-to-many relationship that must support: follow/unfollow, get followers, get following, check if A follows B, and compute mutual followers.

**Relational adjacency list:**
```sql
-- Simple relational approach
CREATE TABLE follows (
    follower_id  BIGINT NOT NULL REFERENCES users(id),
    followee_id  BIGINT NOT NULL REFERENCES users(id),
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (follower_id, followee_id),
    CHECK (follower_id != followee_id)
);

-- Index for both lookup directions
CREATE INDEX idx_follows_follower ON follows(follower_id, created_at DESC);
CREATE INDEX idx_follows_followee ON follows(followee_id, created_at DESC);

-- Get all followers of user 42:
SELECT follower_id FROM follows WHERE followee_id = 42;

-- Get all users that user 42 follows:
SELECT followee_id FROM follows WHERE follower_id = 42;

-- Check if Alice (1) follows Bob (2):
SELECT EXISTS (SELECT 1 FROM follows WHERE follower_id = 1 AND followee_id = 2);
-- Uses PK index: O(1) lookup

-- Mutual follows (both A→B and B→A):
SELECT f1.followee_id AS mutual
FROM follows f1
JOIN follows f2 ON f1.follower_id = f2.followee_id
                AND f1.followee_id = f2.follower_id
WHERE f1.follower_id = 42;
```

**Scaling problem at 100M+ users:**
```
Celebrities (Justin Bieber problem):
  selena_gomez has 250M followers
  SELECT COUNT(*) FROM follows WHERE followee_id = selena_gomez_id;
  → Touches 250M rows → slow

Fanout-on-write for news feed:
  When Justin posts, insert into feed of 250M followers
  → 250M writes per post → impossible synchronously
```

**Redis SET for hot graph data:**
```
# Store follower/following as Redis SETs
SADD followers:42 1 5 7 99 1042  # User 42's followers
SADD following:42 2 8 100         # Users that 42 follows

# Check if 1 follows 42: O(1)
SISMEMBER followers:42 1

# Mutual follows between 42 and 5:
SINTER followers:42 following:42  # Intersection

# Count followers: O(1)
SCARD followers:42

# Paginated followers: Use ZSORTEDSET with timestamp as score
ZADD followers_sorted:42 1683000000 user_id_1
ZRANGE followers_sorted:42 0 99  # First 100 followers, time-ordered
```

**Hybrid architecture:**
```
PostgreSQL (source of truth):
  follows table: follower_id, followee_id, created_at
  Used for: precise queries, reporting, backfilling Redis

Redis (hot cache for graph traversal):
  followers:{user_id}  → SORTED SET (score = timestamp)
  following:{user_id}  → SORTED SET
  Used for: real-time follow checks, feed generation

Write path:
  Alice follows Bob
  → INSERT INTO follows (Alice, Bob) [PostgreSQL]
  → ZADD followers:Bob score=now Alice [Redis]
  → ZADD following:Alice score=now Bob [Redis]

Read path (feed generation):
  → ZRANGE following:Alice 0 -1 [get all followed users from Redis]
  → For each user: fetch recent posts from posts cache
```

---

### Q15. How do you handle GDPR data deletion in a schema that uses event sourcing or append-only tables?

**Answer:**

GDPR Article 17 (Right to Erasure) conflicts directly with append-only/event-sourced systems. You cannot delete from an immutable log — so you need a strategy.

**Problem:**
```
Event log (append-only):
  {event: "user_registered", user_id: 42, email: "alice@x.com", ip: "1.2.3.4"}
  {event: "address_updated", user_id: 42, address: "123 Main St"}
  {event: "order_placed", user_id: 42, items: [...]}

GDPR deletion request from user 42:
  - Cannot delete rows from event log (breaks audit trail)
  - Cannot update rows (immutable)
  - Must remove PII within 30 days
```

**Strategy 1: Crypto-shredding (best for event sourcing)**
```python
# At write time: encrypt PII with user-specific key
class PIIEncryptionService:
    def __init__(self, key_store: KeyStore):
        self.key_store = key_store
    
    def write_event(self, user_id: int, event: dict):
        # Get or create encryption key for this user
        key = self.key_store.get_or_create_key(user_id)
        
        # Encrypt PII fields
        encrypted_event = {
            **event,
            'email': encrypt(event.get('email'), key),
            'ip': encrypt(event.get('ip'), key),
            'address': encrypt(event.get('address'), key)
        }
        return append_to_log(encrypted_event)
    
    def delete_user_data(self, user_id: int):
        # Delete only the encryption key
        self.key_store.delete_key(user_id)
        # All events containing PII are now unreadable garbage
        # Event IDs and timestamps remain for audit
```

**Strategy 2: Anonymization table**
```sql
-- PII stored separately, events reference it
CREATE TABLE user_pii (
    user_id  BIGINT PRIMARY KEY,
    email    TEXT,
    name     TEXT,
    phone    TEXT,
    deleted  BOOLEAN DEFAULT FALSE,
    deleted_at TIMESTAMPTZ
);

CREATE TABLE events (
    id        BIGSERIAL PRIMARY KEY,
    user_id   BIGINT NOT NULL,  -- Foreign key but NOT enforced after deletion
    event_type VARCHAR(50),
    payload   JSONB             -- No PII stored in events
);

-- GDPR deletion: anonymize PII table only
UPDATE user_pii
SET email = 'deleted@gdpr.invalid',
    name  = '[DELETED]',
    phone = NULL,
    deleted = TRUE,
    deleted_at = NOW()
WHERE user_id = 42;

-- Events remain intact (no PII in payload)
-- user_id still present for analytics (pseudonymous)
```

**Strategy 3: PII vault (separate high-security store)**
```
Events:       {user_id: 42, event: "purchase", amount: 99.99}
PII Vault:    {user_id: 42, email: "alice@x.com", name: "Alice"}

On GDPR deletion:
  PII Vault: Hard delete user 42's row
  Events: Remain as-is (user_id is pseudonymous — not PII on its own)
  
GDPR position: user_id without accompanying PII is not personal data
(Confirm with legal counsel — varies by jurisdiction)
```

**Deletion verification:**
```sql
-- Verify no PII remains after deletion
SELECT * FROM user_pii WHERE user_id = 42;
-- Should return: deleted=true, email='deleted@gdpr.invalid'

-- Audit log of deletion (itself GDPR-compliant)
CREATE TABLE gdpr_deletion_log (
    user_id    BIGINT,
    requested_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    deletion_scope TEXT[],  -- ['pii_table', 'file_store', 's3_uploads']
    requester  VARCHAR(100) -- 'user_self', 'admin', 'legal_team'
);
```

---

## Hard (Q16–Q20)

---

### Q16. Design a complete schema for a multi-tenant SaaS platform supporting custom fields per tenant, with full migration safety.

**Answer:**

A SaaS platform must balance flexibility (tenants want different fields) with performance and schema safety.

**Architecture: Core tables + JSONB custom fields + metadata registry**

```sql
-- Tenant registry
CREATE TABLE tenants (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       VARCHAR(200) NOT NULL,
    plan       VARCHAR(20) NOT NULL DEFAULT 'standard',
    schema_name VARCHAR(63) UNIQUE,    -- For dedicated-schema tenants
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Custom field definitions per tenant
CREATE TABLE tenant_field_definitions (
    id             BIGSERIAL PRIMARY KEY,
    tenant_id      UUID NOT NULL REFERENCES tenants(id),
    entity_type    VARCHAR(50) NOT NULL,     -- 'contact', 'deal', 'ticket'
    field_key      VARCHAR(100) NOT NULL,    -- JSON key used in JSONB column
    field_label    VARCHAR(200) NOT NULL,    -- Display name
    field_type     VARCHAR(30) NOT NULL,     -- 'text','number','date','enum','boolean'
    required       BOOLEAN DEFAULT FALSE,
    enum_values    JSONB,                    -- For type='enum': ["Open","Closed","Pending"]
    validation     JSONB,                    -- {"min": 0, "max": 100, "pattern": "..."}
    display_order  INT DEFAULT 0,
    is_active      BOOLEAN DEFAULT TRUE,
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (tenant_id, entity_type, field_key)
);

-- Contacts with built-in + custom fields
CREATE TABLE contacts (
    id          BIGSERIAL PRIMARY KEY,
    tenant_id   UUID NOT NULL REFERENCES tenants(id),
    -- Built-in fields (fast, typed, indexed)
    email       VARCHAR(255) NOT NULL,
    first_name  VARCHAR(100),
    last_name   VARCHAR(100),
    phone       VARCHAR(50),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    -- Custom fields (flexible per tenant)
    custom_data JSONB DEFAULT '{}',
    -- Constraints
    UNIQUE (tenant_id, email)
);

-- RLS for tenant isolation
ALTER TABLE contacts ENABLE ROW LEVEL SECURITY;
CREATE POLICY contacts_tenant_isolation ON contacts
    USING (tenant_id = current_setting('app.tenant_id')::UUID);

-- GIN index for custom field queries
CREATE INDEX idx_contacts_custom ON contacts USING GIN(custom_data);
-- Composite index for common queries
CREATE INDEX idx_contacts_tenant_email ON contacts(tenant_id, email);
CREATE INDEX idx_contacts_tenant_created ON contacts(tenant_id, created_at DESC);
```

**Custom field validation at write time:**
```python
class ContactRepository:
    async def create(self, tenant_id: str, data: dict) -> Contact:
        # Load field definitions for this tenant
        field_defs = await self.get_field_definitions(tenant_id, 'contact')
        
        # Validate custom fields
        custom_data = {}
        for field_def in field_defs:
            key = field_def.field_key
            value = data.get(key)
            
            if field_def.required and value is None:
                raise ValidationError(f"{field_def.field_label} is required")
            
            if value is not None:
                value = self._coerce_type(value, field_def.field_type)
                self._validate(value, field_def.validation)
                custom_data[key] = value
        
        return await self.db.execute(
            "INSERT INTO contacts (tenant_id, email, first_name, custom_data) "
            "VALUES ($1, $2, $3, $4) RETURNING *",
            tenant_id, data['email'], data.get('first_name'), json.dumps(custom_data)
        )
```

**Zero-downtime migration for adding a built-in field:**
```sql
-- Phase 1: Add nullable column (safe, instant)
ALTER TABLE contacts ADD COLUMN company_name VARCHAR(200);

-- Phase 2: Backfill from custom_data where present
UPDATE contacts
SET company_name = custom_data->>'company_name'
WHERE custom_data ? 'company_name'
  AND company_name IS NULL;

-- Phase 3: Validate constraint (non-blocking)
ALTER TABLE contacts ADD CONSTRAINT company_name_length 
    CHECK (length(company_name) <= 200) NOT VALID;
ALTER TABLE contacts VALIDATE CONSTRAINT company_name_length;

-- Phase 4 (weeks later): Remove from custom_data, set NOT NULL if needed
UPDATE contacts SET custom_data = custom_data - 'company_name'
WHERE custom_data ? 'company_name';
```

---

### Q17. Design an event sourcing schema for an e-commerce order system with complete audit trail and CQRS read models.

**Answer:**

Event sourcing stores state as a sequence of immutable events rather than current state. CQRS separates the write model (events) from read models (projections).

**Write side — Event store:**
```sql
-- Aggregate versions (optimistic concurrency control)
CREATE TABLE aggregates (
    id         UUID PRIMARY KEY,
    type       VARCHAR(50) NOT NULL,   -- 'Order', 'User', 'Inventory'
    version    BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (id, type)
);

-- Event store (immutable, append-only)
CREATE TABLE events (
    id             BIGSERIAL PRIMARY KEY,
    aggregate_id   UUID NOT NULL REFERENCES aggregates(id),
    aggregate_type VARCHAR(50) NOT NULL,
    event_type     VARCHAR(100) NOT NULL,  -- 'OrderPlaced', 'OrderShipped', 'OrderCancelled'
    version        BIGINT NOT NULL,         -- Monotonic per aggregate
    payload        JSONB NOT NULL,
    metadata       JSONB,                   -- correlation_id, user_agent, ip, actor_id
    recorded_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (aggregate_id, version)          -- Optimistic locking enforcement
);

CREATE INDEX idx_events_aggregate ON events(aggregate_id, version);
CREATE INDEX idx_events_type_time ON events(event_type, recorded_at);
```

**Recording events with optimistic concurrency:**
```python
async def append_events(
    aggregate_id: UUID,
    expected_version: int,
    new_events: list[Event]
) -> None:
    async with db.transaction():
        # Check current version (optimistic locking)
        current = await db.fetchval(
            "SELECT version FROM aggregates WHERE id = $1 FOR UPDATE",
            aggregate_id
        )
        if current != expected_version:
            raise ConcurrencyConflict(
                f"Expected version {expected_version}, got {current}"
            )
        
        # Append events
        for i, event in enumerate(new_events):
            new_version = expected_version + i + 1
            await db.execute(
                "INSERT INTO events (aggregate_id, aggregate_type, event_type, version, payload, metadata) "
                "VALUES ($1, $2, $3, $4, $5, $6)",
                aggregate_id, 'Order', event.type, new_version,
                json.dumps(event.payload), json.dumps(event.metadata)
            )
        
        # Update aggregate version
        await db.execute(
            "UPDATE aggregates SET version = $1 WHERE id = $2",
            expected_version + len(new_events), aggregate_id
        )
```

**Read side — Denormalized projections (CQRS):**
```sql
-- Order list view (fast list queries)
CREATE TABLE order_list_projection (
    order_id        UUID PRIMARY KEY,
    user_id         BIGINT NOT NULL,
    status          VARCHAR(30) NOT NULL,
    total_amount    DECIMAL(10,2) NOT NULL,
    item_count      INT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL,
    last_updated_at TIMESTAMPTZ NOT NULL
);

-- Order detail view (fast single-order reads)
CREATE TABLE order_detail_projection (
    order_id     UUID PRIMARY KEY,
    user_id      BIGINT NOT NULL,
    user_name    VARCHAR(200),
    user_email   VARCHAR(255),
    status       VARCHAR(30) NOT NULL,
    items        JSONB NOT NULL,    -- Denormalized items array
    address      JSONB,
    payments     JSONB,
    history      JSONB,             -- Denormalized event history
    created_at   TIMESTAMPTZ NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL
);

-- Projection offset tracking (for catchup/replay)
CREATE TABLE projection_checkpoints (
    projection_name VARCHAR(100) PRIMARY KEY,
    last_event_id   BIGINT NOT NULL DEFAULT 0,
    last_updated    TIMESTAMPTZ DEFAULT NOW()
);
```

**Projection rebuilding (for bug fixes or new projections):**
```python
async def rebuild_projection(projection_name: str):
    """Replay all events to rebuild a projection from scratch."""
    await db.execute(f"TRUNCATE TABLE {projection_name}_projection")
    await db.execute(
        "UPDATE projection_checkpoints SET last_event_id = 0 WHERE projection_name = $1",
        projection_name
    )
    
    # Process events in batches
    last_id = 0
    while True:
        events = await db.fetch(
            "SELECT * FROM events WHERE id > $1 ORDER BY id LIMIT 1000",
            last_id
        )
        if not events:
            break
        for event in events:
            await apply_event_to_projection(projection_name, event)
        last_id = events[-1]['id']
```

---

### Q18. Design a schema that handles time-series sensor data at 1 million writes per second using PostgreSQL partitioning and compression.

**Answer:**

1M writes/sec requires extreme partitioning, minimal index overhead, and aggressive compression.

**Partition strategy:**
```sql
-- Parent table (range partition by time)
CREATE TABLE sensor_readings (
    time       TIMESTAMPTZ NOT NULL,
    sensor_id  INT NOT NULL,
    metric     SMALLINT NOT NULL,   -- Use INT codes instead of strings
    value      FLOAT4 NOT NULL      -- 4-byte float (not 8-byte double)
)
PARTITION BY RANGE (time);

-- Create 1-hour partitions (at 1M/s that's 3.6B rows/hour — use hourly not daily)
-- Script to create partitions for next 7 days
DO $$
DECLARE
    start_time TIMESTAMPTZ := date_trunc('hour', NOW());
    end_time   TIMESTAMPTZ;
BEGIN
    FOR i IN 0..167 LOOP  -- 24*7 = 168 hours
        end_time := start_time + INTERVAL '1 hour';
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS sensor_readings_%s PARTITION OF sensor_readings '
            'FOR VALUES FROM (%L) TO (%L)',
            to_char(start_time, 'YYYYMMDDHH24'),
            start_time,
            end_time
        );
        start_time := end_time;
    END LOOP;
END $$;
```

**Write optimization:**
```sql
-- UNLOGGED tables for staging (10x faster writes, no WAL)
-- Then copy to main table in batches
CREATE UNLOGGED TABLE sensor_staging (LIKE sensor_readings);

-- Minimal indexing on write path (add indexes on read partitions)
-- No indexes on current partition → writes at memory speed
-- Add BRIN after partition is "closed" (no more writes):
CREATE INDEX CONCURRENTLY idx_readings_2025050100_brin
ON sensor_readings_2025050100 USING BRIN (time);
```

**TimescaleDB for production scale:**
```sql
-- If using TimescaleDB extension:
SELECT create_hypertable('sensor_readings', 'time',
    chunk_time_interval => INTERVAL '1 hour',
    partitioning_column => 'sensor_id',
    number_partitions => 16   -- Additional space partitioning
);

-- Native compression (up to 95% compression ratio on time-series)
ALTER TABLE sensor_readings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'sensor_id',
    timescaledb.compress_orderby = 'time DESC'
);

-- Compress chunks older than 1 hour
SELECT add_compression_policy('sensor_readings', INTERVAL '1 hour');
```

**Ingestion pipeline:**
```
Sensors → Kafka (10M msgs/sec, 100 partitions)
              ↓
        Consumer group (100 workers)
              ↓
        COPY batch (10,000 rows/batch, 100 batches/sec)
              ↓
        PostgreSQL partition (async, no foreign keys, no triggers)
```

```python
async def batch_writer(events: list[SensorReading]):
    """Use COPY for maximum PostgreSQL write throughput."""
    async with pool.acquire() as conn:
        # COPY is 10x faster than INSERT for bulk loads
        await conn.copy_records_to_table(
            'sensor_readings',
            records=[(e.time, e.sensor_id, e.metric, e.value) for e in events],
            columns=['time', 'sensor_id', 'metric', 'value']
        )
```

**Throughput benchmarks:**
```
Single INSERT: ~5,000 rows/sec per connection
Batched INSERT (1000 rows): ~50,000 rows/sec per connection
COPY: ~200,000 rows/sec per connection
COPY + unlogged + no indexes: ~500,000 rows/sec per connection
20 parallel COPY workers: ~1,000,000+ rows/sec
```

---

### Q19. How do you design a graph schema in PostgreSQL using recursive CTEs, and when do you migrate to a dedicated graph database?

**Answer:**

PostgreSQL can handle graph workloads via recursive CTEs, but has fundamental limitations that drive teams to Neo4j or Amazon Neptune at scale.

**Graph schema in PostgreSQL:**
```sql
-- Nodes
CREATE TABLE nodes (
    id         BIGSERIAL PRIMARY KEY,
    type       VARCHAR(50) NOT NULL,     -- 'user', 'product', 'category'
    properties JSONB
);

-- Edges (directed)
CREATE TABLE edges (
    id          BIGSERIAL PRIMARY KEY,
    from_id     BIGINT NOT NULL REFERENCES nodes(id),
    to_id       BIGINT NOT NULL REFERENCES nodes(id),
    type        VARCHAR(50) NOT NULL,    -- 'follows', 'purchased', 'belongs_to'
    weight      FLOAT,
    properties  JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_edges_from ON edges(from_id, type);
CREATE INDEX idx_edges_to ON edges(to_id, type);
```

**Recursive CTE for graph traversal:**
```sql
-- Shortest path using BFS (PostgreSQL recursive CTE)
WITH RECURSIVE bfs AS (
    -- Base case: start from node 1
    SELECT
        id AS node_id,
        ARRAY[id] AS path,
        0 AS depth
    FROM nodes WHERE id = 1
    
    UNION ALL
    
    -- Recursive case: follow edges
    SELECT
        e.to_id,
        bfs.path || e.to_id,
        bfs.depth + 1
    FROM bfs
    JOIN edges e ON e.from_id = bfs.node_id
    WHERE e.to_id != ALL(bfs.path)  -- Prevent cycles
      AND bfs.depth < 6              -- Max depth limit (CRITICAL for performance)
)
SELECT path, depth
FROM bfs
WHERE node_id = 999  -- Target node
ORDER BY depth
LIMIT 1;
```

**Performance limitations of PostgreSQL for graphs:**
```
Query: "Find all users within 4 hops who purchased Product X"
  Nodes: 10M users, Edges: 500M follows

PostgreSQL WITH RECURSIVE:
  Depth 1: ~1,000 results (fast)
  Depth 2: ~1,000,000 results (slow — table scan territory)
  Depth 3: Billions of paths — PostgreSQL grinds to halt
  
Neo4j Cypher (same query):
  MATCH (u:User)-[:FOLLOWS*1..4]->(other:User)
  WHERE (other)-[:PURCHASED]->(:Product {id: 42})
  RETURN u
  
  Time: 200ms (uses native graph index, no table scans)
  Why: Neo4j stores edges as direct pointers (constant-time traversal)
       PostgreSQL must do index lookups at each hop
```

**Migration trigger criteria:**

| Scenario | PostgreSQL OK | Move to Graph DB |
|----------|--------------|-----------------|
| Max traversal depth | < 3 hops | > 3 hops |
| Graph query frequency | < 10% of queries | > 30% of queries |
| Graph size | < 10M edges | > 100M edges |
| Query patterns | Simple parent-child | Complex pattern matching |
| Path finding | Rare | Core feature |
| Team expertise | SQL expert | Can hire graph expertise |

**Hybrid architecture (common in practice):**
```
PostgreSQL:
  Source of truth for all entity data
  Handles OLTP (user writes, order processing)

Neo4j/Neptune (read-only replica):
  Synced via CDC (change data capture)
  Handles: recommendation engine, fraud detection,
           social graph queries, network analysis
  
Sync mechanism:
  Debezium → Kafka → Graph DB consumer
  Lag: typically 1-5 seconds
```

---

### Q20. Design a complete schema evolution strategy for a NoSQL document store handling millions of versioned documents.

**Answer:**

NoSQL databases have implicit schemas — documents can have different structures. Without a strategy, the schema diverges over time into an unmaintainable mess.

**The schema drift problem:**
```javascript
// Version 1 (2020): simple address string
{ "_id": "user:1", "name": "Alice", "address": "123 Main St" }

// Version 2 (2021): structured address
{ "_id": "user:2", "name": "Bob", "address": { "street": "456 Oak Ave", "city": "NYC" } }

// Version 3 (2022): added preferences
{ "_id": "user:3", "name": "Carol", "address": {...}, "preferences": { "email": true } }

// Now: 3 schemas in production, all must be handled simultaneously
```

**Strategy 1: Versioned documents**
```javascript
// Add schema_version to every document
{
  "_id": "user:42",
  "schema_version": 3,
  "name": "Alice",
  "address": { "street": "123 Main", "city": "Boston" },
  "preferences": { "email": true, "sms": false }
}

// Application: always specify and migrate on read
class UserRepository:
    CURRENT_VERSION = 3
    
    def get(self, user_id: str) -> User:
        raw = self.db.find_one({"_id": f"user:{user_id}"})
        return self.migrate(raw)
    
    def migrate(self, doc: dict) -> User:
        version = doc.get('schema_version', 1)
        
        if version == 1:
            # Migrate v1 → v2: string address → structured
            doc['address'] = {'street': doc['address'], 'city': None}
            doc['schema_version'] = 2
        
        if version == 2:
            # Migrate v2 → v3: add preferences
            doc['preferences'] = {'email': True, 'sms': False}
            doc['schema_version'] = 3
        
        return User(**doc)
    
    def save(self, user: User):
        # Always write at current version
        doc = user.to_dict()
        doc['schema_version'] = self.CURRENT_VERSION
        self.db.replace_one({"_id": doc["_id"]}, doc, upsert=True)
```

**Strategy 2: Schema registry with Avro (for Kafka/event streams)**
```json
// Schema registry: version 1
{
  "type": "record",
  "name": "UserEvent",
  "namespace": "com.example",
  "fields": [
    {"name": "user_id", "type": "long"},
    {"name": "email", "type": "string"}
  ]
}

// Version 2: adding optional field (backward compatible)
{
  "fields": [
    {"name": "user_id", "type": "long"},
    {"name": "email", "type": "string"},
    {"name": "phone", "type": ["null", "string"], "default": null}
  ]
}
```

**Compatibility rules:**

| Compatibility | Old reader + New data | New reader + Old data | Use When |
|--------------|----------------------|----------------------|----------|
| BACKWARD | Yes | No | Old consumers can read new events |
| FORWARD | No | Yes | New consumers can read old events |
| FULL | Yes | Yes | Both directions (safest) |

**Background migration script:**
```python
async def migrate_documents_batch():
    """Migrate old-version documents in the background."""
    
    batch_size = 1000
    migrated = 0
    
    # Find documents not at current version
    cursor = db.users.find(
        {"schema_version": {"$lt": UserRepository.CURRENT_VERSION}},
        batch_size=batch_size
    )
    
    async for doc in cursor:
        user = repo.migrate(doc)
        
        # Write migrated version back
        await db.users.replace_one(
            {"_id": doc["_id"], "schema_version": doc["schema_version"]},  # Optimistic lock
            user.to_dict()
        )
        migrated += 1
        
        if migrated % 10000 == 0:
            logger.info(f"Migrated {migrated} documents")
            await asyncio.sleep(0.01)  # Throttle to avoid overloading DB
    
    logger.info(f"Migration complete: {migrated} documents migrated")
```

**MongoDB-specific index management:**
```javascript
// Safe index creation (background, non-blocking)
db.users.createIndex(
    { "address.city": 1 },
    { background: true, sparse: true }  // sparse: ignore docs without field
)

// Check index usage
db.users.aggregate([{ $indexStats: {} }])
```

**Monitoring schema health:**
```python
# Weekly schema health report
def schema_health_report():
    pipeline = [
        {"$group": {"_id": "$schema_version", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    
    dist = list(db.users.aggregate(pipeline))
    total = sum(d['count'] for d in dist)
    
    for d in dist:
        pct = d['count'] / total * 100
        print(f"v{d['_id']}: {d['count']:,} docs ({pct:.1f}%)")
    
    # Alert if > 5% of docs are on old schema
    old_schema = sum(d['count'] for d in dist if d['_id'] < CURRENT_VERSION)
    if old_schema / total > 0.05:
        alert("Schema migration backlog > 5% — accelerate background migration")
```

---

## Quick Reference

### Normal Forms Cheat Sheet

| Form | Rule | Violation Example |
|------|------|-------------------|
| 1NF | Atomic values, no arrays | tags = "python,sql,go" |
| 2NF | No partial PK dependencies | order_items stores product_name |
| 3NF | No transitive dependencies | employees stores dept_name via dept_id |
| BCNF | Every determinant is a candidate key | Rare edge cases |

### ID Type Comparison

| Type | Size | Sortable | Coordination | Best For |
|------|------|----------|-------------|----------|
| Auto-increment | 4-8 bytes | Yes | Single DB | Single-node |
| UUID v4 | 16 bytes | No | None | Privacy, distributed |
| ULID | 16 bytes | Yes (ms) | None | Distributed + sortable |
| Snowflake | 8 bytes | Yes (ms) | Machine ID | High-throughput |

### Schema Migration Safety Rules

1. Never rename a column in one step — use expand-contract
2. Never add a NOT NULL column without a default in one step
3. Always backfill in batches with pg_sleep throttling
4. Add constraints as NOT VALID first, then VALIDATE separately
5. Use CONCURRENTLY for index creation on live tables
6. Test migrations on production-size data clone first

### SCD Type Selection

| Scenario | SCD Type |
|----------|----------|
| Fix data error | Type 1 (overwrite) |
| Track full history (audit, bi-temporal analysis) | Type 2 (new row) |
| Just need "what changed from previous" | Type 3 (extra column) |
| Need both current and historical queries | Type 2 |

### Hierarchy Pattern Performance

| Pattern | Find subtree | Find ancestors | Insert | Move |
|---------|-------------|----------------|--------|------|
| Adjacency list | O(n) recursive | O(n) recursive | O(1) | O(1) |
| Materialized path | O(log n) | O(1) parse | O(1) | O(n) |
| Nested sets | O(1) | O(1) | O(n) | O(n) |
| Closure table | O(1) | O(1) | O(d) | O(n) |

### GDPR Deletion Strategies

| Strategy | Mutability Required | Complexity | Best For |
|----------|--------------------|-----------|---------|----|
| Hard delete | Tables | Low | Simple RDBMS |
| Anonymization | Tables | Low | Audit tables, analytics |
| Crypto-shredding | Immutable logs | High | Event sourcing |
| PII vault | Mixed | Medium | Large-scale, mixed systems |
