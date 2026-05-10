# Database Schema Design for Scale

## Easy (Q1–Q7)

---

**Q1. What schema design principles help databases scale?**

1. **Choose the right primary key** — monotonically increasing PKs (BIGSERIAL, ULID) avoid B-tree page splits. UUIDs v4 cause random writes → page fragmentation at scale.

2. **Partition by the most frequent query dimension** — if 90% of queries filter by `created_at`, range-partition by date. Query only touches one partition, not the whole table.

3. **Index the right columns** — every unnecessary index slows down writes (each INSERT/UPDATE must maintain all indexes). Index only what queries actually filter/sort on.

4. **Denormalize deliberately** — for read-heavy data, store redundant copies to avoid JOINs. For write-heavy data, normalize to avoid update anomalies.

5. **Avoid wide tables with many NULLs** — a 200-column table where each row uses 10 columns is better modeled as a narrow core table + attribute tables or JSONB.

6. **Use appropriate data types** — `INT` vs `BIGINT`, `TEXT` vs `VARCHAR(N)`, `TIMESTAMPTZ` vs `TIMESTAMP`. Proper types reduce storage, improve index density (more keys per page), and prevent silent truncation.

7. **Design for your access patterns** — schema that is beautiful in ER but requires 10-table JOINs for every API request is a performance problem. The schema should match the queries.

---

**Q2. How does primary key choice affect database performance at scale?**

**Auto-increment integer (BIGSERIAL/AUTO_INCREMENT):**
```sql
CREATE TABLE orders (order_id BIGSERIAL PRIMARY KEY, ...);
-- New rows always insert at the end of the B-tree
-- B-tree leaf pages fill sequentially → minimal page splits
-- Cache-friendly: recently inserted rows are in the same pages
-- Problem: single sequence generator can become a bottleneck in distributed setup
```

**UUID v4 (random):**
```sql
CREATE TABLE orders (order_id UUID PRIMARY KEY DEFAULT gen_random_uuid(), ...);
-- Random UUID → random position in B-tree → random page writes
-- At high insert rates: almost every insert causes a different page to be loaded into cache
-- "Write everywhere": buffer pool thrashes, cache hit rate drops, disk I/O spikes
-- 50M rows: UUID insert rate typically 3–5× slower than sequential integer
```

**ULID / UUID v7 (time-ordered):**
```sql
-- ULID: 26-character time-ordered unique identifier
-- First 48 bits = timestamp (milliseconds) → mostly sequential
-- Last 80 bits = random → globally unique
CREATE TABLE orders (order_id TEXT DEFAULT generate_ulid() PRIMARY KEY, ...);
-- Inserts mostly sequential (same timestamp = sequential) → good B-tree behavior
-- Still globally unique and distributed-safe → best of both worlds
```

**Rule:** Use `BIGSERIAL` for single-server databases. Use ULID or UUID v7 for distributed/sharded databases where sequence generators cannot be centralized.

---

**Q3. What is the difference between a star schema and a snowflake schema in a data warehouse?**

Both are dimensional modeling techniques for OLAP databases.

**Star schema:**
```
Central fact table → directly connected to dimension tables

    ┌─────────────┐     ┌──────────────┐
    │  dim_product│     │  dim_customer │
    └──────┬──────┘     └──────┬───────┘
           │                   │
           └────┐   ┌──────────┘
                ▼   ▼
          ┌──────────────────┐
          │  fact_sales      │
          │  (order_id PK)   │
          │  product_id FK   │
          │  customer_id FK  │
          │  date_id FK      │
          │  amount          │
          └──────────────────┘
                 │
          ┌──────▼──────┐
          │  dim_date   │
          └─────────────┘

Dimension tables are denormalized (e.g., dim_product contains category name directly)
Queries: 1 fact table JOIN N dimension tables = simple, fast
```

**Snowflake schema:**
```
Dimension tables are normalized (further split into sub-dimensions)

dim_product → dim_category → dim_category_group
dim_customer → dim_city → dim_state → dim_country

Queries: require JOINs through multiple dimension tables → more complex
Storage: less redundancy → slightly smaller
Performance: typically slower than star (more JOINs)
```

**When to use each:**
- **Star schema:** Most OLAP workloads. Fewer JOINs = faster queries. Storage is cheap.
- **Snowflake:** When dimension tables are very large and normalization saves significant storage, or when dimension data changes frequently (fewer places to update).

---

**Q4. What is the SCD (Slowly Changing Dimension) Type 2 pattern and when do you use it?**

SCD Type 2 keeps historical records when a dimension attribute changes, allowing you to analyze "what was true at the time" rather than just the current state.

**Without SCD2:**
```sql
-- If we update the customer's email:
UPDATE customers SET email = 'new@email.com' WHERE customer_id = 1;
-- Historical analysis: "who were our top customers in 2022?" now uses their 2024 email → wrong
```

**With SCD Type 2:**
```sql
CREATE TABLE customers_history (
    surrogate_key  BIGSERIAL    PRIMARY KEY,
    customer_id    BIGINT       NOT NULL,            -- natural key (same person)
    name           TEXT         NOT NULL,
    email          TEXT         NOT NULL,
    address        TEXT,
    effective_from TIMESTAMPTZ  NOT NULL,
    effective_to   TIMESTAMPTZ,                      -- NULL = current record
    is_current     BOOLEAN      NOT NULL DEFAULT TRUE,
    CONSTRAINT one_current_record UNIQUE (customer_id, is_current) DEFERRABLE INITIALLY DEFERRED
);

-- When email changes:
UPDATE customers_history SET effective_to = NOW(), is_current = FALSE
WHERE customer_id = 1 AND is_current = TRUE;

INSERT INTO customers_history (customer_id, name, email, address, effective_from, is_current)
VALUES (1, 'Alice', 'new@email.com', '123 Main St', NOW(), TRUE);
```

**Query at a point in time:**
```sql
-- What was Alice's email when she placed order #5678 in 2022?
SELECT ch.email
FROM orders o
JOIN customers_history ch
  ON o.customer_id = ch.customer_id
  AND o.order_date >= ch.effective_from
  AND (ch.effective_to IS NULL OR o.order_date < ch.effective_to)
WHERE o.order_id = 5678;
```

**Use cases:** Data warehouses, audit requirements, regulatory compliance (what did the user's profile look like when they agreed to terms?), chargeback analysis.

---

**Q5. How does the EAV (Entity-Attribute-Value) anti-pattern work and why is it problematic?**

EAV is a schema design where instead of fixed columns, you store attribute names and values as rows.

```sql
-- EAV approach:
CREATE TABLE product_attributes (
    product_id   INT,
    attr_name    TEXT,
    attr_value   TEXT,
    PRIMARY KEY (product_id, attr_name)
);

INSERT INTO product_attributes VALUES (1, 'color', 'red');
INSERT INTO product_attributes VALUES (1, 'size', 'M');
INSERT INTO product_attributes VALUES (1, 'material', 'cotton');

-- To get all attributes for a product:
SELECT attr_name, attr_value FROM product_attributes WHERE product_id = 1;
-- Or to get specific attributes as columns (pivot):
SELECT
    MAX(CASE WHEN attr_name = 'color'    THEN attr_value END) AS color,
    MAX(CASE WHEN attr_name = 'size'     THEN attr_value END) AS size,
    MAX(CASE WHEN attr_name = 'material' THEN attr_value END) AS material
FROM product_attributes WHERE product_id = 1;
```

**Problems with EAV:**
1. **No type safety** — everything is TEXT; price stored as "99.99" not as NUMERIC
2. **No constraints** — can't enforce NOT NULL on specific attributes
3. **Terrible query performance** — getting one product's attributes requires N rows scan + pivot
4. **No referential integrity** — cannot FK to attr_value
5. **Query complexity** — simple "find all red shirts under $50" becomes a multi-level pivot JOIN nightmare

**Better alternatives:**
```sql
-- Option A: JSONB column (PostgreSQL) — flexible, indexed, typed
CREATE TABLE products (
    product_id BIGSERIAL PRIMARY KEY,
    name TEXT, category TEXT, price NUMERIC,
    attributes JSONB    -- {"color":"red","size":"M","material":"cotton"}
);
CREATE INDEX ON products USING GIN (attributes);
SELECT * FROM products WHERE attributes @> '{"color":"red"}' AND price < 50;

-- Option B: Category-specific tables (for known categories)
CREATE TABLE shirt_attributes (product_id INT PRIMARY KEY, color TEXT, size TEXT, material TEXT);
CREATE TABLE laptop_attributes (product_id INT PRIMARY KEY, ram_gb INT, cpu TEXT, storage_gb INT);
```

---

**Q6. What is the outbox table pattern from a schema design perspective?**

The outbox pattern solves the dual-write problem (write to DB AND publish event) by making the event publication part of the same local ACID transaction as the data change.

```sql
-- Outbox table: stores events that need to be published
CREATE TABLE outbox (
    message_id      UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_type  TEXT         NOT NULL,   -- e.g., 'order', 'user'
    aggregate_id    TEXT         NOT NULL,   -- e.g., order_id value
    event_type      TEXT         NOT NULL,   -- e.g., 'order.placed', 'user.updated'
    payload         JSONB        NOT NULL,   -- event payload
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    processed_at    TIMESTAMPTZ,             -- NULL = not yet published
    retry_count     INT          NOT NULL DEFAULT 0
);

CREATE INDEX ON outbox (processed_at, created_at) WHERE processed_at IS NULL;

-- On order placement (same transaction):
BEGIN;
INSERT INTO orders (order_id, customer_id, amount) VALUES (...);
INSERT INTO outbox (aggregate_type, aggregate_id, event_type, payload)
VALUES ('order', order_id::TEXT, 'order.placed',
        jsonb_build_object('order_id', order_id, 'customer_id', customer_id, 'amount', amount));
COMMIT;
-- Either both records exist, or neither (atomic)

-- Relay worker: reads unprocessed outbox entries, publishes to Kafka, marks processed
UPDATE outbox SET processed_at = NOW() WHERE message_id = ? AND processed_at IS NULL;
```

---

**Q7. How do you model a hierarchical / tree structure (e.g., org chart, category tree) in a relational database?**

**Approach 1: Adjacency list (simplest)**
```sql
CREATE TABLE employees (
    emp_id     INT PRIMARY KEY,
    name       TEXT,
    manager_id INT REFERENCES employees(emp_id)  -- self-referential FK
);
-- Root node: manager_id IS NULL

-- Pros: simple writes (INSERT/UPDATE one row)
-- Cons: fetching the full subtree requires recursive CTE (O(depth) queries or one recursive query)

-- Fetch all reports of manager 5 (recursive):
WITH RECURSIVE reports AS (
    SELECT emp_id, name, 0 AS depth FROM employees WHERE emp_id = 5
    UNION ALL
    SELECT e.emp_id, e.name, r.depth + 1
    FROM employees e JOIN reports r ON e.manager_id = r.emp_id
)
SELECT * FROM reports ORDER BY depth, name;
```

**Approach 2: Materialized path**
```sql
CREATE TABLE categories (
    cat_id   INT PRIMARY KEY,
    name     TEXT,
    path     TEXT NOT NULL  -- e.g., '/electronics/laptops/gaming'
);
CREATE INDEX ON categories (path text_pattern_ops);  -- prefix search

-- All descendants of /electronics:
SELECT * FROM categories WHERE path LIKE '/electronics/%';
-- Pros: O(1) ancestor queries, easy breadcrumb generation
-- Cons: updates require updating all descendant paths (expensive for deep moves)
```

**Approach 3: Nested sets**
```sql
CREATE TABLE categories (cat_id INT, name TEXT, lft INT, rgt INT);
-- All descendants: WHERE lft > parent.lft AND rgt < parent.rgt
-- Pros: O(1) descendant queries
-- Cons: inserts/moves require updating many rows (renum left/right values)
```

**Approach 4: Closure table (best for frequent reads + updates)**
```sql
CREATE TABLE category_closure (
    ancestor_id   INT NOT NULL REFERENCES categories(cat_id),
    descendant_id INT NOT NULL REFERENCES categories(cat_id),
    depth         INT NOT NULL,
    PRIMARY KEY (ancestor_id, descendant_id)
);
-- One row per ancestor-descendant pair (including self: depth=0)
-- All descendants: WHERE ancestor_id = X
-- All ancestors: WHERE descendant_id = X
-- Pros: fast reads for any traversal pattern
-- Cons: more storage; writes update O(depth) rows in closure table
```

---

## Medium (Q8–Q15)

---

**Q8. How do you design a multi-tenant database schema? Compare row-level, schema-level, and database-level isolation.**

**Option 1: Shared tables (row-level isolation)**
```sql
-- All tenants in the same table, separated by tenant_id column
CREATE TABLE orders (
    tenant_id  INT  NOT NULL,   -- always first column
    order_id   BIGINT NOT NULL,
    ...
    PRIMARY KEY (tenant_id, order_id)
);

-- Enforce isolation via Row-Level Security:
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON orders
    USING (tenant_id = current_setting('app.tenant_id')::INT);

-- Pros: simple schema, efficient storage, easy cross-tenant queries (for admin)
-- Cons: noisy neighbor (one tenant's heavy query hurts others), RLS bug = data leak
-- Best for: 10,000+ small tenants, low data isolation requirements
```

**Option 2: Separate schemas per tenant**
```sql
-- Each tenant gets their own schema (namespace) within one PostgreSQL instance
CREATE SCHEMA tenant_42;
CREATE TABLE tenant_42.orders (order_id BIGSERIAL PRIMARY KEY, ...);

SET search_path = tenant_42;
SELECT * FROM orders;  -- tenant 42's data only

-- Pros: strong namespace isolation, separate indexes per tenant, no RLS bugs
-- Cons: schema migrations must run N times (one per tenant), complex for 1000+ tenants
-- Best for: 10–1000 tenants that need strong isolation or different schemas
```

**Option 3: Separate databases per tenant**
```sql
-- Each tenant has their own database (or database instance)
-- Connection string: postgresql://db_tenant_42/orders_db

-- Pros: maximum isolation (different connection string, different VACUUM, different locks)
--       GDPR compliance (physically isolated data for EU tenants)
-- Cons: very high operational overhead, separate backup/monitoring per tenant
-- Best for: enterprise tenants with compliance requirements, < 100 tenants
```

**Recommendation by scale:**
```
< 100 tenants, enterprise:           Option 3 (separate DB per tenant)
100–10,000 tenants, mid-market:     Option 2 (separate schemas)
10,000+ tenants, SMB/freemium:      Option 1 (shared tables + RLS)
```

---

**Q9. How do you design an audit trail table that tracks all changes to critical data?**

```sql
-- Approach 1: Generic audit log (works for any table)
CREATE TABLE audit_log (
    audit_id    BIGSERIAL    PRIMARY KEY,
    table_name  TEXT         NOT NULL,
    record_id   TEXT         NOT NULL,   -- stringified PK of changed row
    operation   TEXT         NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    old_data    JSONB,                   -- NULL for INSERT
    new_data    JSONB,                   -- NULL for DELETE
    changed_by  TEXT         NOT NULL,  -- user/service that made the change
    changed_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    ip_address  INET,
    request_id  TEXT                    -- correlation ID for distributed tracing
);

CREATE INDEX ON audit_log (table_name, record_id, changed_at DESC);
CREATE INDEX ON audit_log (changed_by, changed_at DESC);
CREATE INDEX ON audit_log (changed_at DESC) WHERE table_name = 'payments';

-- Trigger to auto-populate (PostgreSQL):
CREATE OR REPLACE FUNCTION audit_trigger_func() RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log (table_name, record_id, operation, old_data, new_data, changed_by, changed_at)
    VALUES (
        TG_TABLE_NAME,
        CASE TG_OP WHEN 'DELETE' THEN row_to_json(OLD)::jsonb->>'id'
                   ELSE row_to_json(NEW)::jsonb->>'id' END,
        TG_OP,
        CASE TG_OP WHEN 'INSERT' THEN NULL ELSE row_to_json(OLD)::jsonb END,
        CASE TG_OP WHEN 'DELETE' THEN NULL ELSE row_to_json(NEW)::jsonb END,
        current_setting('app.current_user', true),
        NOW()
    );
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER orders_audit
AFTER INSERT OR UPDATE OR DELETE ON orders
FOR EACH ROW EXECUTE FUNCTION audit_trigger_func();
```

**Retention and partitioning:**
```sql
-- Audit logs grow fast — partition by month
CREATE TABLE audit_log (...) PARTITION BY RANGE (changed_at);
CREATE TABLE audit_log_2024_01 PARTITION OF audit_log
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Retention: detach and drop partitions older than 2 years
ALTER TABLE audit_log DETACH PARTITION audit_log_2022_01;
DROP TABLE audit_log_2022_01;
```

---

**Q10. How do you design a schema for a feed system (social media, news) that needs to be both fast to write and fast to read?**

**Write-heavy path:** Users post content → store in posts table.
**Read-heavy path:** Users read their personalized feed → retrieve 50 most recent posts from followed users.

**Approach A: Fan-out on write (pre-computed feeds)**
```sql
-- User's posts:
CREATE TABLE posts (
    post_id     BIGINT       PRIMARY KEY DEFAULT nextval('post_seq'),
    author_id   BIGINT       NOT NULL,
    content     TEXT         NOT NULL,
    media_url   TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX ON posts (author_id, created_at DESC);

-- Pre-computed feed per user:
CREATE TABLE user_feed (
    user_id     BIGINT       NOT NULL,
    post_id     BIGINT       NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL,  -- copy from post (for sorting)
    PRIMARY KEY (user_id, created_at DESC, post_id)
) PARTITION BY HASH (user_id);  -- distribute feed data across partitions

-- On post creation: fan-out to all followers' feeds
-- INSERT INTO user_feed SELECT follower_id, post_id, created_at FROM follows WHERE following_id = author_id;
-- Read feed: SELECT p.* FROM user_feed f JOIN posts p USING (post_id) WHERE f.user_id = ? ORDER BY f.created_at DESC LIMIT 50;
```

**Approach B: Fan-out on read (pull-based)**
```sql
-- No feed table; query posts from followed users at read time:
SELECT p.* FROM posts p
JOIN follows f ON p.author_id = f.following_id
WHERE f.follower_id = ? AND p.created_at > NOW() - INTERVAL '7 days'
ORDER BY p.created_at DESC LIMIT 50;

-- Index to make this efficient:
CREATE INDEX ON posts (author_id, created_at DESC);
CREATE INDEX ON follows (follower_id, following_id);
-- Works if user follows < 1000 people; at 10K follows this query becomes slow
```

**Hybrid (recommended at scale):**
```
Regular users (< 10K followers): fan-out on write (pre-computed feed)
Celebrity users (> 10K followers): fan-out on read for their posts
Feed assembly: pre-computed feed + recent celebrity posts (merged at read time in application)
```

---

**Q11. What is the difference between a soft delete and a hard delete, and how does each affect schema design?**

**Hard delete:**
```sql
DELETE FROM users WHERE user_id = 1234;
-- Row is gone — cannot be recovered
-- Foreign key constraints: must cascade delete or set null all child records first
-- GDPR "right to be forgotten": hard delete is the correct approach
-- Performance: simpler — no filter needed; fewer rows → smaller indexes
```

**Soft delete:**
```sql
-- Add a deletion marker:
ALTER TABLE users ADD COLUMN deleted_at TIMESTAMPTZ;  -- NULL = active, non-null = deleted

-- "Delete":
UPDATE users SET deleted_at = NOW() WHERE user_id = 1234;

-- All queries must filter deleted rows:
SELECT * FROM users WHERE deleted_at IS NULL AND user_id = 1234;
```

**Schema implications of soft delete:**

```sql
-- Unique constraint problem: deleted users should free up their email for re-registration
CREATE UNIQUE INDEX ON users (email) WHERE deleted_at IS NULL;  -- partial unique index
-- Allows same email to be used again after deletion; enforces uniqueness only for active users

-- Foreign key problem: child records referencing "deleted" parent
-- Option A: keep child records (order history should remain even if user is "deleted")
-- Option B: cascade soft-delete (set child.deleted_at when parent.deleted_at is set)

-- Performance: index all queries on active records only
CREATE INDEX ON users (last_active_at DESC) WHERE deleted_at IS NULL;  -- partial index
-- This index is tiny compared to full index (99% of users are active)

-- GDPR complication: soft delete does NOT satisfy "right to erasure" (data still exists)
-- For GDPR: anonymize the row (overwrite PII with nulls/hashes) AND set deleted_at
UPDATE users SET
    email = 'deleted_' || user_id || '@deleted.invalid',
    name = 'Deleted User',
    phone = NULL,
    address = NULL,
    deleted_at = NOW()
WHERE user_id = ?;
```

---

**Q12. How do you design a database schema for a flexible product catalog that supports hundreds of product types, each with different attributes?**

**The challenge:** A shirt has color, size, material. A laptop has RAM, CPU, storage, GPU. A book has ISBN, author, publisher, page_count. No single fixed schema works for all.

**Solution: Core + Extensions pattern**

```sql
-- Core product table (common to all products):
CREATE TABLE products (
    product_id    BIGSERIAL    PRIMARY KEY,
    name          TEXT         NOT NULL,
    category_id   INT          NOT NULL REFERENCES categories(category_id),
    base_price    NUMERIC(10,2) NOT NULL CHECK (base_price >= 0),
    status        TEXT         NOT NULL DEFAULT 'active',
    created_at    TIMESTAMPTZ  DEFAULT NOW(),
    
    -- Variable attributes: JSONB for flexibility
    attributes    JSONB        NOT NULL DEFAULT '{}'
);

-- GIN index for attribute queries:
CREATE INDEX ON products USING GIN (attributes);
-- Functional index for specific high-frequency attribute:
CREATE INDEX ON products ((attributes->>'color')) WHERE category_id = 1;  -- shirts only

-- For each major category: strongly-typed extension table (optional, for validation)
CREATE TABLE laptop_attributes (
    product_id   INT  PRIMARY KEY REFERENCES products(product_id),
    ram_gb       INT  NOT NULL CHECK (ram_gb IN (8, 16, 32, 64, 128)),
    cpu_model    TEXT NOT NULL,
    storage_gb   INT  NOT NULL,
    gpu_model    TEXT,
    screen_size  NUMERIC(4,1)
);

-- Query all red shirts under $50 (using JSONB):
SELECT product_id, name, base_price
FROM products
WHERE category_id = 1  -- shirts
  AND attributes @> '{"color":"red"}'
  AND base_price < 50;

-- Query laptops with ≥16GB RAM (using extension table):
SELECT p.product_id, p.name, p.base_price, l.ram_gb, l.cpu_model
FROM products p
JOIN laptop_attributes l USING (product_id)
WHERE l.ram_gb >= 16
  AND p.base_price < 1000;
```

**Trade-off between JSONB and extension tables:**
```
JSONB: maximum flexibility, no migration for new attributes, slightly looser type safety
Extension tables: strict types and constraints, faster for strongly-typed queries, migration required for new attributes
```

---

**Q13. How does data archival strategy affect schema design? What patterns exist for managing large historical tables?**

**The problem:** An orders table grows 10M rows per month. After 1 year: 120M rows. Queries slow down. Backups take longer. Autovacuum takes longer.

**Strategy 1: Time-based partitioning + partition archival**
```sql
-- Orders partitioned by month:
CREATE TABLE orders (...) PARTITION BY RANGE (order_date);
CREATE TABLE orders_2024_01 PARTITION OF orders FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Archive old partitions (12+ months):
ALTER TABLE orders DETACH PARTITION orders_2022_01;
-- Dump to cold storage:
pg_dump -t orders_2022_01 -F c mydb > s3://backups/orders_2022_01.dump
-- Drop from hot database:
DROP TABLE orders_2022_01;
-- Active table: only 12 months of data → manageable size

-- Restore if needed (rare):
pg_restore -t orders_2022_01 -d mydb orders_2022_01.dump
ALTER TABLE orders ATTACH PARTITION orders_2022_01
    FOR VALUES FROM ('2022-01-01') TO ('2022-02-01');
```

**Strategy 2: Archive table (hot/cold split)**
```sql
-- Active orders: last 90 days
CREATE TABLE orders_active (
    LIKE orders INCLUDING ALL
) PARTITION BY RANGE (order_date);

-- Historical orders: > 90 days
CREATE TABLE orders_archive (
    LIKE orders INCLUDING ALL
) PARTITION BY RANGE (order_date);
-- orders_archive on cheaper storage tier (NFS, HDD)

-- Nightly migration job:
INSERT INTO orders_archive SELECT * FROM orders_active WHERE order_date < NOW() - INTERVAL '90 days';
DELETE FROM orders_active WHERE order_date < NOW() - INTERVAL '90 days';

-- Application: reads from orders_active for normal use, queries orders_archive for historical reports
-- Unified view (optional):
CREATE VIEW orders_all AS SELECT * FROM orders_active UNION ALL SELECT * FROM orders_archive;
```

**Strategy 3: Vertical partitioning (store wide rows partially)**
```sql
-- Wide orders table: many columns rarely queried
CREATE TABLE orders (order_id BIGINT PRIMARY KEY, customer_id BIGINT, amount NUMERIC, created_at TIMESTAMPTZ);
-- Rarely accessed columns in separate table (joined on demand):
CREATE TABLE orders_details (
    order_id    BIGINT PRIMARY KEY REFERENCES orders(order_id),
    shipping_address TEXT, billing_address TEXT, coupon_code TEXT,
    notes TEXT, internal_flags JSONB
);
-- Core orders table is narrow → more rows per page → faster scans
-- Details table not touched unless needed
```

---

**Q14. What are the schema design considerations for a time-series workload in PostgreSQL?**

```sql
-- Time-series table design principles:
CREATE TABLE sensor_readings (
    -- Natural time-series PK: (device, time) — time is the sort key
    device_id  UUID         NOT NULL,
    read_time  TIMESTAMPTZ  NOT NULL,
    metric     TEXT         NOT NULL,
    value      DOUBLE PRECISION,
    PRIMARY KEY (device_id, metric, read_time)  -- note: NOT (device_id, read_time) alone
    -- reason: reading multiple metrics for one device at one time is common
) PARTITION BY RANGE (read_time);

-- Monthly partitions (auto-managed):
CREATE TABLE sensor_readings_2024_01 PARTITION OF sensor_readings
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- TTL via partition drop (much faster than DELETE):
-- pg_partman handles this automatically with retention settings

-- DO NOT use UUID as primary key for time-series:
-- UUIDs are random → scatter writes across B-tree → page thrash
-- Compound (device_id, read_time) PK: sequential within a device's readings

-- Index for time-range queries per device (most common query):
-- The PK (device_id, metric, read_time) already serves: WHERE device_id=X AND metric=Y AND read_time BETWEEN A AND B
-- The clustering order IS the query order: no separate index needed

-- For "latest reading per device" query pattern:
-- BRIN index is excellent (time-series data is physically sorted by read_time):
CREATE INDEX ON sensor_readings USING BRIN (read_time);
-- BRIN is tiny (stores min/max per 128 pages) → fast range pruning with tiny index
```

**Compression (TimescaleDB):**
```sql
-- Enable time-based chunking and compression:
SELECT create_hypertable('sensor_readings', 'read_time', chunk_time_interval => INTERVAL '1 day');
-- Compress chunks older than 7 days:
ALTER TABLE sensor_readings SET (timescaledb.compress, timescaledb.compress_orderby = 'read_time DESC');
SELECT add_compression_policy('sensor_readings', INTERVAL '7 days');
-- Compression ratio: 10–40× for typical metrics data
```

---

**Q15. How do you design a schema for a notifications system that scales to millions of users with billions of notifications?**

**Requirements:**
- Store notifications for 100M users
- Mark as read/unread
- Count unread notifications (fast — shown in UI)
- Paginate through notifications (newest first)
- TTL: delete notifications older than 90 days

```sql
-- Notifications partitioned by user_id (hash) and created_at (range):
-- Use range partition by time for easy TTL via partition drop
CREATE TABLE notifications (
    notification_id  BIGINT       NOT NULL DEFAULT nextval('notif_seq'),
    user_id          BIGINT       NOT NULL,
    type             TEXT         NOT NULL,  -- 'like', 'comment', 'follow', 'mention'
    actor_id         BIGINT,                 -- who triggered it (nullable: system notifications)
    resource_type    TEXT,                   -- 'post', 'comment', 'order'
    resource_id      BIGINT,
    payload          JSONB,                  -- additional context
    is_read          BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, created_at DESC, notification_id)  -- clustered by user + time
) PARTITION BY RANGE (created_at);

-- Monthly partitions:
CREATE TABLE notifs_2024_01 PARTITION OF notifications
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Unread count: maintain as a separate Redis counter (much faster than COUNT(*))
-- On notification INSERT: INCR redis user:{id}:unread_notif_count
-- On notification READ: DECR redis user:{id}:unread_notif_count
-- (If Redis and DB diverge: periodically reconcile:
--   UPDATE redis FROM: SELECT COUNT(*) FROM notifications WHERE user_id=X AND is_read=FALSE)

-- Queries:
-- Get user's 20 most recent notifications:
SELECT * FROM notifications WHERE user_id = 12345 ORDER BY created_at DESC LIMIT 20;
-- Uses PK scan (user_id + created_at DESC) → single partition scan → fast

-- Mark all read:
UPDATE notifications SET is_read = TRUE WHERE user_id = 12345 AND is_read = FALSE;
-- Partial index to make this fast:
CREATE INDEX ON notifications (user_id) WHERE is_read = FALSE;  -- per partition

-- TTL: detach + drop 90-day-old partitions monthly (instant operation)
ALTER TABLE notifications DETACH PARTITION notifs_2021_12;
DROP TABLE notifs_2021_12;
```

---

## Hard (Q16–Q20)

---

**Q16. Design the complete schema for a financial ledger system that must maintain an accurate, immutable audit trail of every balance change.**

**Design principle:** A ledger is append-only. Balances are computed from transactions, not stored directly (prevents silent tampering).

```sql
-- Accounts: the entities that hold balances
CREATE TABLE accounts (
    account_id    UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id      BIGINT       NOT NULL REFERENCES users(user_id),
    account_type  TEXT         NOT NULL CHECK (account_type IN ('checking', 'savings', 'escrow')),
    currency      CHAR(3)      NOT NULL DEFAULT 'USD',
    status        TEXT         NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'frozen', 'closed')),
    opened_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX ON accounts (owner_id, status);

-- Ledger entries: immutable, append-only
-- NEVER UPDATE or DELETE rows in this table
CREATE TABLE ledger_entries (
    entry_id       BIGSERIAL    PRIMARY KEY,
    account_id     UUID         NOT NULL REFERENCES accounts(account_id),
    transaction_id UUID         NOT NULL,   -- groups debit + credit entries for one transfer
    entry_type     TEXT         NOT NULL CHECK (entry_type IN ('debit', 'credit')),
    amount_cents   BIGINT       NOT NULL CHECK (amount_cents > 0),  -- always positive; type determines sign
    balance_after  BIGINT       NOT NULL,   -- running balance after this entry (denormalized for speed)
    description    TEXT,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    created_by     TEXT         NOT NULL,   -- user or service that initiated
    idempotency_key UUID        UNIQUE      -- prevents duplicate entries
);
-- Partition by created_at for manageability:
CREATE TABLE ledger_entries_2024 PARTITION OF ledger_entries
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- Index for account history queries:
CREATE INDEX ON ledger_entries (account_id, created_at DESC);

-- Transactions: groups debit + credit for double-entry bookkeeping
CREATE TABLE ledger_transactions (
    transaction_id  UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    from_account    UUID         REFERENCES accounts(account_id),
    to_account      UUID         REFERENCES accounts(account_id),
    amount_cents    BIGINT       NOT NULL CHECK (amount_cents > 0),
    status          TEXT         NOT NULL DEFAULT 'pending',
    reference_type  TEXT,                   -- 'order_payment', 'refund', 'withdrawal'
    reference_id    TEXT,
    initiated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

-- Current balance view (computed from ledger, not stored):
-- (In practice, use balance_after from the most recent ledger entry for performance)
CREATE VIEW account_balances AS
SELECT account_id, balance_after AS current_balance_cents, created_at AS as_of
FROM ledger_entries
WHERE entry_id = (
    SELECT MAX(entry_id) FROM ledger_entries le2 WHERE le2.account_id = ledger_entries.account_id
);
-- Or materialize this as a separate table maintained by triggers
```

**Double-entry bookkeeping enforcement:**
```sql
-- Every transfer must have exactly two entries: one debit + one credit of equal amount
-- Enforced by procedure:
CREATE OR REPLACE FUNCTION post_transfer(
    p_txn_id UUID, p_from UUID, p_to UUID, p_amount BIGINT, p_idempotency_key UUID
) RETURNS VOID AS $$
DECLARE v_from_balance BIGINT; v_to_balance BIGINT;
BEGIN
    SELECT balance_after INTO v_from_balance FROM ledger_entries
    WHERE account_id = p_from ORDER BY entry_id DESC LIMIT 1 FOR UPDATE;
    
    IF v_from_balance < p_amount THEN RAISE EXCEPTION 'Insufficient funds'; END IF;
    
    SELECT balance_after INTO v_to_balance FROM ledger_entries
    WHERE account_id = p_to ORDER BY entry_id DESC LIMIT 1;
    
    INSERT INTO ledger_entries (account_id, transaction_id, entry_type, amount_cents, balance_after, idempotency_key)
    VALUES (p_from, p_txn_id, 'debit', p_amount, v_from_balance - p_amount, p_idempotency_key);
    
    INSERT INTO ledger_entries (account_id, transaction_id, entry_type, amount_cents, balance_after, idempotency_key)
    VALUES (p_to, p_txn_id, 'credit', p_amount, v_to_balance + p_amount, gen_random_uuid());
    
    UPDATE ledger_transactions SET status = 'completed', completed_at = NOW() WHERE transaction_id = p_txn_id;
END;
$$ LANGUAGE plpgsql;
```

---

**Q17. How would you design a schema that supports both OLTP operations and near-real-time analytics without impacting production performance?**

**Architecture: HTAP-lite (Hybrid Transactional/Analytical Processing)**

```
OLTP Primary (PostgreSQL):
  → Handles writes and point-lookup reads
  → Schema: normalized (3NF), indexed for OLTP patterns
  
  ↓ WAL / logical replication ↓ (100ms lag)
  
Analytical Read Replica (PostgreSQL + columnar extension):
  → Handles analytical queries (aggregations, large scans)
  → No impact on primary write performance
  → Can create additional OLAP-style indexes without touching primary
  → Schema: same as primary, but with added materialized views

  ↓ CDC (Debezium → Kafka → ClickHouse) ↓ (5-60s lag)

ClickHouse Analytics:
  → Deep analytics, arbitrary aggregations on full history
  → Columnar storage: scans 100× faster than row storage for aggregations
  → Schema: denormalized (pre-joined orders + customers + products)
```

**Schema design on the replica:**
```sql
-- On analytical replica: create materialized views for common reports
CREATE MATERIALIZED VIEW daily_revenue AS
SELECT DATE_TRUNC('day', order_date) AS day,
       region,
       SUM(amount) AS total_revenue,
       COUNT(*) AS order_count
FROM orders
GROUP BY 1, 2;

CREATE UNIQUE INDEX ON daily_revenue (day, region);

-- Refresh with no blocking (PostgreSQL):
REFRESH MATERIALIZED VIEW CONCURRENTLY daily_revenue;
-- Schedule: every 15 minutes via pg_cron

-- Application: read daily_revenue from replica, write orders to primary
-- Never run analytical queries on the primary
```

**Table inheritance / columnar extension (pg_mooncake, Citus columnar):**
```sql
-- Citus columnar: append-only columnar storage as a PostgreSQL table
CREATE TABLE order_analytics (...) USING columnar;
-- 3-5× storage compression, fast column scans, no UPDATE/DELETE support
-- Perfect for analytics read replica
```

---

**Q18. How do you handle schema migrations in a system with 24/7 availability requirements (zero-downtime migrations)?**

**The problem:** Most `ALTER TABLE` operations in PostgreSQL require a full table rewrite or an `ACCESS EXCLUSIVE LOCK` (blocks all reads and writes).

**Safe migration patterns:**

**Pattern 1: Adding a nullable column (safe, instant)**
```sql
ALTER TABLE orders ADD COLUMN shipped_at TIMESTAMPTZ;
-- PostgreSQL: instant (updates catalog, no table rewrite for nullable column)
-- No lock contention for reads/writes
```

**Pattern 2: Adding a NOT NULL column (risky without this pattern)**
```sql
-- WRONG: ALTER TABLE orders ADD COLUMN shipped_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
-- Rewrites entire table → locks for minutes on large tables

-- RIGHT: Multi-phase migration
-- Phase 1: Add nullable column (instant):
ALTER TABLE orders ADD COLUMN shipped_at TIMESTAMPTZ;

-- Phase 2: Backfill in batches (no exclusive lock):
UPDATE orders SET shipped_at = created_at
WHERE id BETWEEN 1 AND 100000 AND shipped_at IS NULL;
-- Repeat in batches, rate-limited, during low-traffic hours

-- Phase 3: Add NOT NULL constraint (PostgreSQL 12+ supports NOT VALID first):
ALTER TABLE orders ADD CONSTRAINT orders_shipped_at_not_null
    CHECK (shipped_at IS NOT NULL) NOT VALID;
-- NOT VALID: constraint created without scanning existing rows → instant

VALIDATE CONSTRAINT orders_shipped_at_not_null;
-- Validates existing rows while allowing concurrent writes (no exclusive lock in PG 12+)

-- Phase 4: Set actual NOT NULL (now safe):
ALTER TABLE orders ALTER COLUMN shipped_at SET NOT NULL;
-- Fast because constraint already validated
```

**Pattern 3: Adding an index (use CONCURRENTLY)**
```sql
-- WRONG: CREATE INDEX ON orders (customer_id);
-- Takes ShareLock → blocks writes for the duration

-- RIGHT:
CREATE INDEX CONCURRENTLY idx_orders_customer ON orders (customer_id);
-- Builds index in background
-- Allows concurrent reads AND writes during build
-- Slightly slower (multiple table scans) but zero blocking
```

**Pattern 4: Renaming a column**
```sql
-- Cannot rename without downtime easily
-- EXPAND/CONTRACT migration:

-- Phase 1: Add new column:
ALTER TABLE orders ADD COLUMN customer_id BIGINT;

-- Phase 2: Dual-write: application writes to both old_customer_id and customer_id
-- Phase 3: Backfill: UPDATE orders SET customer_id = old_customer_id WHERE customer_id IS NULL;
-- Phase 4: Switch reads to customer_id
-- Phase 5: Remove old_customer_id: ALTER TABLE orders DROP COLUMN old_customer_id;
```

---

**Q19. How do you design a schema for a multi-currency financial system?**

```sql
-- Store all monetary values as integers (cents/smallest unit) — NEVER use FLOAT
-- FLOAT cannot represent 0.1 + 0.2 = 0.30000000000000004 (binary floating point)
-- NUMERIC can, but is slower and larger

CREATE TABLE currencies (
    currency_code  CHAR(3)  PRIMARY KEY,  -- ISO 4217: 'USD', 'EUR', 'JPY'
    name           TEXT     NOT NULL,
    decimal_places INT      NOT NULL,     -- USD: 2, JPY: 0, KWD: 3
    symbol         TEXT     NOT NULL      -- '$', '€', '¥'
);

CREATE TABLE accounts (
    account_id   UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id     BIGINT  NOT NULL,
    currency     CHAR(3) NOT NULL REFERENCES currencies(currency_code),
    balance_minor BIGINT NOT NULL DEFAULT 0  -- in minor units (cents for USD)
    -- USD: 1099 = $10.99 | JPY: 1099 = ¥1099 | KWD: 10990 = 10.990 KWD
);

-- Exchange rates (point-in-time):
CREATE TABLE exchange_rates (
    rate_id       BIGSERIAL    PRIMARY KEY,
    from_currency CHAR(3)      NOT NULL REFERENCES currencies(currency_code),
    to_currency   CHAR(3)      NOT NULL REFERENCES currencies(currency_code),
    rate          NUMERIC(20,10) NOT NULL,  -- high precision for rate
    effective_at  TIMESTAMPTZ  NOT NULL,
    source        TEXT         NOT NULL    -- 'ECB', 'internal', 'Stripe'
);
CREATE UNIQUE INDEX ON exchange_rates (from_currency, to_currency, effective_at);

-- Conversion function:
CREATE OR REPLACE FUNCTION convert_currency(
    amount_minor BIGINT,
    from_curr CHAR(3),
    to_curr CHAR(3),
    at_time TIMESTAMPTZ DEFAULT NOW()
) RETURNS BIGINT AS $$
DECLARE
    v_rate NUMERIC;
    v_from_decimals INT;
    v_to_decimals INT;
BEGIN
    SELECT rate INTO v_rate FROM exchange_rates
    WHERE from_currency = from_curr AND to_currency = to_curr
      AND effective_at <= at_time
    ORDER BY effective_at DESC LIMIT 1;
    
    SELECT decimal_places INTO v_from_decimals FROM currencies WHERE currency_code = from_curr;
    SELECT decimal_places INTO v_to_decimals FROM currencies WHERE currency_code = to_curr;
    
    -- Convert: amount_minor → major → apply rate → minor
    RETURN ROUND((amount_minor::NUMERIC / power(10, v_from_decimals)) * v_rate * power(10, v_to_decimals));
END;
$$ LANGUAGE plpgsql IMMUTABLE;
```

**Key rules for multi-currency:**
1. Store in minor units (integers, not decimals): `1099` not `10.99`
2. Store the currency alongside every monetary value: `(amount_minor BIGINT, currency CHAR(3))`
3. Never store rates with floating point: use `NUMERIC(20,10)`
4. Audit every conversion: record the rate used at the time of conversion

---

**Q20. How does denormalization of specific data in a relational schema improve performance, and when does it introduce correctness risks?**

**Denormalization: storing redundant copies of data to avoid JOINs.**

**Case 1: Denormalize at write time (snapshot pattern — correct)**
```sql
-- Orders table stores customer info AS IT WAS at order time (not current)
CREATE TABLE orders (
    order_id         BIGINT  PRIMARY KEY,
    customer_id      BIGINT  NOT NULL REFERENCES customers(customer_id),
    -- Denormalized snapshot:
    customer_name    TEXT    NOT NULL,
    customer_email   TEXT    NOT NULL,
    shipping_address TEXT    NOT NULL,
    -- Current product price might change; store it at order time:
    unit_price       NUMERIC(10,2) NOT NULL  -- price at time of order, not current price
);
-- If customer changes their name: historical orders correctly show the name they had
-- This is CORRECT denormalization — data is intentionally historical
```

**Case 2: Denormalize for query performance (cache pattern — risky if not maintained)**
```sql
-- Denormalize: store post_count on user profile (avoids COUNT query)
CREATE TABLE users (
    user_id    BIGINT PRIMARY KEY,
    username   TEXT UNIQUE NOT NULL,
    post_count INT NOT NULL DEFAULT 0   -- denormalized: must be kept in sync
);

-- Must maintain via trigger or application:
CREATE TRIGGER update_post_count AFTER INSERT OR DELETE ON posts
FOR EACH ROW EXECUTE FUNCTION adjust_post_count();
-- Risk: trigger bug → post_count drifts from actual count → incorrect data shown

-- Safer: don't denormalize, use materialized view instead:
CREATE MATERIALIZED VIEW user_stats AS
SELECT user_id, COUNT(*) AS post_count FROM posts GROUP BY user_id;
REFRESH MATERIALIZED VIEW CONCURRENTLY user_stats;  -- periodic refresh
-- Slightly stale (acceptable for stats) but always derivable from source of truth
```

**When denormalization introduces correctness risk:**
```
1. The source data changes independently:
   orders.product_name = "Laptop X" but products.name changed to "Laptop X Pro"
   → Historical orders: correct (they show the name at purchase time)
   → Live product listing that denormalized into orders for display: wrong

2. The denormalized value has a many-to-one relationship:
   category_name stored on products, but many products share one category
   → Category renamed: must UPDATE all products with that category → expensive + race conditions

3. Aggregates that should always reflect latest data:
   Storing total_revenue on the customer row
   → Each order INSERT must UPDATE customer.total_revenue
   → If an order is refunded: must also UPDATE customer.total_revenue
   → Any missed update = permanent drift
```

**Decision framework:**
```
Denormalize when:
  ✓ The value is immutable at write time (price at order time)
  ✓ The source changes very infrequently (country_name, currency_symbol)
  ✓ You can enforce consistency via DB constraint or trigger (low risk of drift)

Don't denormalize when:
  ✗ The source changes frequently and the copy must always be current
  ✗ The value is an aggregate that depends on many rows
  ✗ You can't guarantee the copy stays in sync (no trigger, no constraint)
  → Use materialized view instead (stale but always correctable from source)
```

---

## Quick Reference

```
Primary key choice:
  BIGSERIAL:   single-server, sequential inserts, best B-tree performance
  UUID v4:     distributed-safe, random → page splits at high insert rate
  ULID/UUIDv7: distributed-safe + mostly sequential → best of both

Partitioning patterns:
  Range by date: time-series, archive old partitions by detach+drop
  Hash by ID:    even distribution within one DB instance
  List by region/type: geographic or categorical isolation

Multi-tenant options:
  Shared tables + RLS: 10K+ small tenants
  Separate schemas:    100–10K medium tenants
  Separate databases:  < 100 enterprise tenants with compliance needs

Schema migration (zero-downtime):
  Add nullable column:      instant (safe)
  Add index:                CREATE INDEX CONCURRENTLY (no lock)
  Add NOT NULL column:      add nullable → backfill → add CHECK NOT VALID → VALIDATE → SET NOT NULL
  Rename column:            expand/contract (add new → dual-write → switch reads → drop old)

Denormalization safety:
  Immutable at write time (order snapshot): safe
  Aggregate that changes: risky → use materialized view instead
  One-to-many source: risky (bulk updates needed on source change)
```
