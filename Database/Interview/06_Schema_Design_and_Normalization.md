# Schema Design and Normalization — Interview Questions

> **Difficulty Mix:** Easy (Q1–Q7) · Medium (Q8–Q14) · Hard (Q15–Q20)

---

### Q1. What is database normalization? Why do we do it?

**Answer:**
Normalization is the process of organizing a relational database to **reduce data redundancy** and **improve data integrity** by structuring it according to normal forms.

**Goals:**
- Eliminate redundant (duplicated) data
- Ensure data dependencies make sense (only storing related data together)
- Prevent update/insert/delete anomalies

**Without normalization (anomalies):**
```
Table: employee_projects
emp_id | emp_name | dept | project_id | project_name
1      | Alice    | Eng  | 101        | Rocket
1      | Alice    | Eng  | 102        | Shuttle
2      | Bob      | Mkt  | 103        | Campaign

Update anomaly: Changing Alice's dept requires updating 2 rows
Insert anomaly: Can't add a project without assigning an employee
Delete anomaly: Deleting project 103 deletes Bob from the database
```

---

### Q2. Explain First Normal Form (1NF).

**Answer:**
A table is in **1NF** when:
1. Every column contains **atomic** (indivisible) values — no arrays, lists, or sets in a single cell
2. Every row is **uniquely identifiable** (has a primary key)
3. No repeating groups (no multi-valued columns)

**Violates 1NF:**
```
employee_id | name  | skills
1           | Alice | Python, SQL, Java   ← not atomic (comma-separated list)
2           | Bob   | Go, Rust
```

**Fixed (1NF):**
```
employee_id | name  | skill
1           | Alice | Python
1           | Alice | SQL
1           | Alice | Java
2           | Bob   | Go
2           | Bob   | Rust
PRIMARY KEY (employee_id, skill)
```

---

### Q3. Explain Second Normal Form (2NF).

**Answer:**
A table is in **2NF** when:
1. It is in 1NF
2. Every non-key attribute depends on the **whole** primary key (no partial dependencies)

Partial dependency: A non-key column depends on only **part** of a composite primary key.

**Violates 2NF (composite PK = {order_id, product_id}):**
```
order_id | product_id | quantity | product_name | product_price
1        | 101        | 2        | Laptop       | 999.99
1        | 102        | 1        | Mouse        | 29.99

product_name depends only on product_id (part of PK) — partial dependency!
```

**Fixed (2NF):**
```sql
CREATE TABLE order_items (
    order_id INT, product_id INT, quantity INT,
    PRIMARY KEY (order_id, product_id)
);
CREATE TABLE products (
    product_id INT PRIMARY KEY, name VARCHAR(100), price DECIMAL(10,2)
);
-- product_name and price moved to products table
```

---

### Q4. Explain Third Normal Form (3NF).

**Answer:**
A table is in **3NF** when:
1. It is in 2NF
2. No transitive dependencies — non-key attributes depend **only** on the primary key, not on other non-key attributes

Transitive dependency: A → B → C, where A is the PK, B and C are non-key columns.

**Violates 3NF:**
```
employee_id | name  | dept_id | dept_name | dept_location
1           | Alice | 10      | Engineering | New York
2           | Bob   | 20      | Marketing   | Chicago

dept_name and dept_location depend on dept_id, not employee_id → transitive dependency!
```

**Fixed (3NF):**
```sql
CREATE TABLE employees (emp_id INT PK, name VARCHAR, dept_id INT FK);
CREATE TABLE departments (dept_id INT PK, dept_name VARCHAR, dept_location VARCHAR);
-- dept_name and dept_location moved to their own table
```

---

### Q5. What is BCNF (Boyce-Codd Normal Form)?

**Answer:**
BCNF is a stronger version of 3NF. A table is in BCNF when **every determinant is a candidate key**.

A determinant is any column (or set) that functionally determines another column.

**Violates BCNF (but satisfies 3NF):**
```
student | course | teacher
Alice   | Math   | Dr. Smith    ← teacher determines course (teaches only one)
Bob     | Math   | Dr. Smith
Carol   | Physics| Dr. Jones

Functional dependencies:
(student, course) → teacher      ← composite PK
teacher → course                 ← teacher is a determinant but NOT a candidate key!
This violates BCNF.
```

**Fixed (BCNF):**
```sql
CREATE TABLE teacher_courses (teacher VARCHAR, course VARCHAR, PRIMARY KEY(teacher, course));
CREATE TABLE student_teachers (student VARCHAR, teacher VARCHAR, PRIMARY KEY(student, teacher));
```

---

### Q6. What are insertion, update, and deletion anomalies?

**Answer:**
Data anomalies are problems caused by poor schema design (lack of normalization):

**Update anomaly:** Same fact stored multiple times — updating one copy creates inconsistency:
```
employees_denormalized:
emp_id | emp_name | dept_id | dept_name
1      | Alice    | 10      | Engineering   ← dept_name stored here
2      | Bob      | 10      | Engineering   ← and here
-- Renaming Engineering to "Software Eng" requires updating 2 rows
-- If only one is updated → inconsistency
```

**Insert anomaly:** Cannot insert a valid entity without also providing unrelated data:
```
-- Can't add a new department without also adding an employee to it
INSERT INTO employees_denormalized (dept_id, dept_name) VALUES (99, 'HR');  -- FAILS (no emp data)
```

**Delete anomaly:** Deleting one entity inadvertently deletes information about another:
```
-- If the only HR employee is deleted, the HR department record is also lost
DELETE FROM employees_denormalized WHERE emp_id = 5;  -- HR dept disappears from DB
```

**Fix:** Normalize — separate employees and departments into different tables.

---

### Q7. When should you denormalize? What are the trade-offs?

**Answer:**
Denormalization deliberately introduces redundancy to improve **read performance**, at the cost of write complexity.

**When to denormalize:**
- Read-heavy systems (OLAP, reporting, dashboards)
- Frequently needed JOIN results (acceptable to store them pre-computed)
- High query latency is unacceptable
- Complex JOINs involve many tables

**Trade-offs:**

| | Normalized | Denormalized |
|--|-----------|-------------|
| Data redundancy | Low | High |
| Write performance | Better | Worse (update multiple places) |
| Read performance | Slower (JOINs) | Faster |
| Data integrity | Strong | Must maintain manually / with triggers |
| Storage | Less | More |

**Common denormalization techniques:**
```sql
-- 1. Precompute columns
ALTER TABLE orders ADD COLUMN item_count INT;  -- Cached count from order_items
-- Maintain with trigger:
UPDATE orders SET item_count = (SELECT COUNT(*) FROM order_items WHERE order_id = ?)

-- 2. Duplicate columns across tables
-- orders.customer_name (copied from customers.name)
-- Avoid JOINs on the hot path

-- 3. Materialized views (best of both worlds — maintained automatically in some DBs)
CREATE MATERIALIZED VIEW order_summary AS SELECT ...;
REFRESH MATERIALIZED VIEW order_summary;
```

---

### Q8. What is the difference between OLTP and OLAP schemas?

**Answer:**

**OLTP (Online Transactional Processing):**
- Normalized (3NF) to minimize redundancy
- Many small read/write operations
- Many tables, simple queries
- Examples: banking, e-commerce, ERP

**OLAP (Online Analytical Processing):**
- Denormalized (star/snowflake schema) for fast reads
- Few complex analytical queries
- Fewer, wider tables
- Examples: data warehouses, BI dashboards

**Star Schema:**
```
              ┌──────────────┐
              │   Fact Table  │
              │ (fact_sales)  │
              └──────┬───────┘
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
 dim_customer   dim_product    dim_time
 (flat, wide)   (flat, wide)   (flat, wide)
```

**Snowflake Schema:** Like star schema, but dimension tables are further normalized:
```
dim_product → dim_category → dim_subcategory
```
Snowflake saves space but requires more JOINs for analytical queries.

---

### Q9. What is an Entity-Relationship (ER) model? What are its components?

**Answer:**
An ER model is a visual representation of the data and its relationships, used for database design.

**Components:**

| Component | Description | Symbol |
|-----------|-------------|--------|
| Entity | A real-world object/thing | Rectangle |
| Attribute | A property of an entity | Oval |
| Relationship | How entities relate to each other | Diamond |
| Cardinality | How many entities participate | 1, M (crow's foot notation) |

**Relationship types:**
- **One-to-One (1:1):** One person has one passport
- **One-to-Many (1:M):** One department has many employees
- **Many-to-Many (M:M):** Students enroll in many courses; courses have many students

```sql
-- Many-to-Many requires a junction/bridge table:
CREATE TABLE student_courses (
    student_id INT REFERENCES students(id),
    course_id  INT REFERENCES courses(id),
    enrolled_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (student_id, course_id)
);
```

---

### Q10. How would you design a schema for a many-to-many relationship with attributes?

**Answer:**
When a many-to-many relationship has its own attributes, the junction table becomes an entity of its own.

**Example: Students and Courses with a grade attribute:**
```sql
-- students M:M courses, with grade and enrollment date

CREATE TABLE students (
    id   INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE courses (
    id       INT PRIMARY KEY AUTO_INCREMENT,
    name     VARCHAR(100) NOT NULL,
    credits  INT NOT NULL CHECK (credits > 0)
);

-- Junction table with attributes
CREATE TABLE enrollments (
    student_id   INT NOT NULL,
    course_id    INT NOT NULL,
    enrolled_at  DATE NOT NULL DEFAULT CURRENT_DATE,
    grade        CHAR(2),        -- Can be NULL if not yet graded
    attendance   DECIMAL(5,2),   -- Percentage

    PRIMARY KEY (student_id, course_id),
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    FOREIGN KEY (course_id)  REFERENCES courses(id)  ON DELETE RESTRICT
);

-- Query: students with GPA
SELECT s.name, AVG(CASE e.grade
    WHEN 'A' THEN 4.0 WHEN 'B' THEN 3.0 WHEN 'C' THEN 2.0 ELSE 0 END) AS gpa
FROM students s
JOIN enrollments e ON e.student_id = s.id
GROUP BY s.id, s.name;
```

---

### Q11. What is the EAV (Entity-Attribute-Value) pattern? Why is it often an anti-pattern?

**Answer:**
EAV stores dynamic attributes in rows instead of columns:

```sql
-- EAV table
CREATE TABLE product_attributes (
    product_id  INT,
    attr_name   VARCHAR(50),   -- e.g., 'color', 'weight', 'size'
    attr_value  VARCHAR(200),  -- e.g., 'red', '5kg', 'XL'
    PRIMARY KEY (product_id, attr_name)
);
```

**Why it seems useful:** Flexible — add new attributes without schema changes.

**Why it's an anti-pattern:**
1. **No data types** — all values stored as strings (no numeric indexing, no date sorting)
2. **No NOT NULL constraints** — can't enforce required attributes
3. **Terrible query performance** — pivoting EAV back to columns requires many self-joins
4. **Hard to query** — `WHERE color = 'red'` becomes a JOIN nightmare

```sql
-- To find products where color=red AND size=XL:
SELECT p.id FROM products p
JOIN product_attributes ca ON ca.product_id = p.id AND ca.attr_name = 'color' AND ca.attr_value = 'red'
JOIN product_attributes sa ON sa.product_id = p.id AND sa.attr_name = 'size'  AND sa.attr_value = 'XL';
-- Gets much worse with more attributes!
```

**Better alternatives:**
- **JSON columns** (PostgreSQL JSONB, MySQL JSON) — flexible but typed
- **Wide table** — add columns for common attributes
- **Inheritance/polymorphism** — separate tables per category

---

### Q12. How would you design a self-referential (hierarchical) table?

**Answer:**
A self-referential table has a foreign key back to itself, enabling tree/hierarchy storage.

```sql
-- Adjacency list model (simplest)
CREATE TABLE categories (
    id        INT PRIMARY KEY AUTO_INCREMENT,
    name      VARCHAR(100) NOT NULL,
    parent_id INT REFERENCES categories(id) ON DELETE SET NULL
    -- NULL parent_id = root category
);

-- Example data:
-- id=1, name='Electronics', parent_id=NULL  (root)
-- id=2, name='Computers',   parent_id=1
-- id=3, name='Laptops',     parent_id=2
-- id=4, name='Phones',      parent_id=1

-- Query: all children of Electronics (one level)
SELECT * FROM categories WHERE parent_id = 1;

-- Query: all descendants (use recursive CTE)
WITH RECURSIVE descendants AS (
    SELECT id, name, parent_id, 0 AS depth
    FROM categories WHERE id = 1  -- Start at Electronics

    UNION ALL

    SELECT c.id, c.name, c.parent_id, d.depth + 1
    FROM categories c JOIN descendants d ON c.parent_id = d.id
)
SELECT * FROM descendants ORDER BY depth, name;
```

**Alternative models for hierarchies:**
- **Materialized path** — store full path as string: `/Electronics/Computers/Laptops`
- **Nested sets** — store left/right values for range-based subtree queries
- **Closure table** — store all ancestor-descendant pairs (fastest reads, most storage)

---

### Q13. What is the difference between a wide table and a narrow table? When to use each?

**Answer:**

**Narrow table:** Few columns, many rows (normalized)
```sql
CREATE TABLE user_events (
    user_id    INT,
    event_type VARCHAR(50),
    event_date DATE,
    value      DECIMAL(10,2)
);
-- Many rows per user, one row per event
```

**Wide table:** Many columns, fewer rows (denormalized)
```sql
CREATE TABLE user_monthly_stats (
    user_id    INT,
    year_month CHAR(7),
    pageviews  INT, sessions INT, purchases INT, revenue DECIMAL(10,2),
    clicks INT, impressions INT, ...  -- many metrics as columns
);
```

| | Narrow | Wide |
|--|--------|------|
| Flexibility | Easy to add new event types | Hard to add new metrics (schema change) |
| Query for specific metric | Requires GROUP BY + CASE pivot | Direct column access |
| Storage | More rows, less wasted space | Sparse (NULLs for unused metrics) |
| Use case | OLTP event logging | OLAP pre-aggregated reporting |
| Columnar storage (OLAP) | Inefficient | Efficient |

---

### Q14. How would you model an audit trail / change history in a database?

**Answer:**

**Option 1: Audit table (simple)**
```sql
CREATE TABLE employees_audit (
    audit_id    INT PRIMARY KEY AUTO_INCREMENT,
    employee_id INT NOT NULL,
    operation   CHAR(1) NOT NULL,   -- 'I'=Insert, 'U'=Update, 'D'=Delete
    old_data    JSON,               -- Previous values (NULL for insert)
    new_data    JSON,               -- New values (NULL for delete)
    changed_by  VARCHAR(100),
    changed_at  TIMESTAMP DEFAULT NOW()
);

-- Maintained by trigger or application code
```

**Option 2: Temporal table / Slowly Changing Dimension (SCD Type 2)**
```sql
CREATE TABLE employee_history (
    id          INT PRIMARY KEY AUTO_INCREMENT,
    employee_id INT NOT NULL,       -- Natural key
    name        VARCHAR(100),
    salary      DECIMAL(10,2),
    dept_id     INT,
    valid_from  TIMESTAMP NOT NULL,
    valid_to    TIMESTAMP,          -- NULL = current record
    is_current  BOOLEAN DEFAULT TRUE,

    INDEX idx_emp_current (employee_id, is_current),
    INDEX idx_emp_time (employee_id, valid_from, valid_to)
);

-- Query current record:
SELECT * FROM employee_history WHERE employee_id = 1 AND is_current = TRUE;

-- Query state at a point in time:
SELECT * FROM employee_history
WHERE employee_id = 1
  AND valid_from <= '2023-06-01' AND (valid_to IS NULL OR valid_to > '2023-06-01');
```

**Option 3: PostgreSQL temporal tables (SQL:2011 standard)**
```sql
-- PostgreSQL 16+
CREATE TABLE employees (
    id INT PRIMARY KEY,
    salary DECIMAL(10,2),
    valid_at TSTZRANGE    -- built-in time range
);
```

---

### Q15. How would you design a "tags" or "labels" system for a blog platform?

**Answer:**
A blog platform has posts that can have multiple tags, and tags belong to many posts.

```sql
-- Core tables
CREATE TABLE posts (
    id         INT PRIMARY KEY AUTO_INCREMENT,
    title      VARCHAR(200) NOT NULL,
    body       TEXT,
    author_id  INT NOT NULL,
    status     ENUM('draft', 'published', 'archived') DEFAULT 'draft',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE tags (
    id   INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) NOT NULL UNIQUE,
    slug VARCHAR(50) NOT NULL UNIQUE      -- URL-friendly version: 'machine-learning'
);

-- Many-to-many junction
CREATE TABLE post_tags (
    post_id INT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    tag_id  INT NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (post_id, tag_id)
);

-- Indexes for fast lookups
CREATE INDEX idx_post_tags_tag ON post_tags (tag_id);   -- "all posts with this tag"
CREATE INDEX idx_post_tags_post ON post_tags (post_id); -- "all tags on this post"

-- Query: all posts tagged 'SQL' with their other tags
SELECT p.title, GROUP_CONCAT(t2.name) AS all_tags
FROM posts p
JOIN post_tags pt1 ON p.id = pt1.post_id
JOIN tags t1 ON pt1.tag_id = t1.id AND t1.slug = 'sql'
JOIN post_tags pt2 ON p.id = pt2.post_id
JOIN tags t2 ON pt2.tag_id = t2.id
GROUP BY p.id, p.title;

-- Optimization: store tag array in JSON for fast "list tags per post"
ALTER TABLE posts ADD COLUMN tag_cache JSON;  -- Denormalized cache
```

---

### Q16. Compare UUID vs Auto-Increment as a primary key. What are the performance implications?

**Answer:**

**Auto-increment INT/BIGINT:**
```sql
id INT AUTO_INCREMENT PRIMARY KEY  -- Sequential: 1, 2, 3, ...
```
- **Pros:** Sequential inserts → B-tree pages filled left-to-right (no page splits), compact (4 bytes), readable
- **Cons:** Predictable/guessable (security), requires centralized sequence, not unique across shards

**UUID (v4 — random):**
```sql
id CHAR(36) DEFAULT UUID() PRIMARY KEY  -- Random: '550e8400-e29b-41d4-...'
```
- **Pros:** Globally unique (safe for distributed systems), not guessable
- **Cons:** 16x larger than INT (36 chars or 16 bytes), **random inserts cause B-tree page splits** → index fragmentation → slower writes and larger index

**UUID v7 (time-ordered — best of both):**
```sql
id BINARY(16) DEFAULT gen_random_uuid()  -- Time-ordered: monotonic within a millisecond
```
- UUID v7 is time-ordered → sequential inserts like auto-increment, but globally unique
- Available in PostgreSQL 17 with `gen_random_uuid()` producing UUIDv4; use pgcrypto for v7

**Performance rule of thumb:**
- For single-server apps: auto-increment INT
- For distributed apps: UUID v7 or ULID (lexicographically sortable)
- For security-sensitive IDs: UUID v4 (unpredictable) or ULID

---

### Q17. How do you design for soft delete? What are the trade-offs?

**Answer:**
Soft delete marks rows as deleted instead of physically removing them.

```sql
-- Implementation 1: Boolean flag
ALTER TABLE users ADD COLUMN is_deleted BOOLEAN DEFAULT FALSE;
ALTER TABLE users ADD COLUMN deleted_at TIMESTAMP;
ALTER TABLE users ADD COLUMN deleted_by INT;

-- Implementation 2: deleted_at (NULL = not deleted)
-- More informative — records when deletion occurred
CREATE INDEX idx_users_active ON users (email) WHERE deleted_at IS NULL;  -- Partial index

-- Always filter in queries:
SELECT * FROM users WHERE deleted_at IS NULL;  -- Active users only
SELECT * FROM users WHERE deleted_at IS NOT NULL;  -- Deleted users

-- Use a view to hide complexity:
CREATE VIEW active_users AS SELECT * FROM users WHERE deleted_at IS NULL;
```

**Trade-offs:**

| | Hard Delete | Soft Delete |
|--|------------|------------|
| Storage | ✓ Row is gone | ✗ Rows accumulate |
| Compliance | Data truly deleted | Easy to restore |
| Query complexity | Simple | Every query needs WHERE deleted_at IS NULL |
| Index efficiency | ✓ Clean | ✗ Indexes include deleted rows (use partial index!) |
| Referential integrity | FK handles it | Must handle manually |
| GDPR "right to be forgotten" | ✓ Easy | ✗ Must truly delete personal data |

**Best practice:** Use soft delete with a partial index, and periodically archive/hard-delete old soft-deleted rows.

---

### Q18. What is schema versioning and migration? How do you manage it in production?

**Answer:**
Schema migrations are controlled, versioned changes to a database schema. They must be:
- **Versioned** — tracked with sequential version numbers or timestamps
- **Reproducible** — can be applied to any environment (dev/staging/prod)
- **Rollback-able** — each migration has an up and down script

**Common tools:** Flyway, Liquibase (Java), Alembic (Python), django-migrations, Active Record (Rails)

**Migration best practices for zero-downtime deployments:**

```sql
-- 1. Add nullable column first (backward compatible)
ALTER TABLE users ADD COLUMN display_name VARCHAR(100);  -- Nullable, no default needed

-- 2. Deploy new code that writes both old and new column
-- 3. Backfill the new column
UPDATE users SET display_name = CONCAT(first_name, ' ', last_name)
WHERE display_name IS NULL;

-- 4. Add NOT NULL constraint (after backfill is complete)
ALTER TABLE users ALTER COLUMN display_name SET NOT NULL;

-- 5. Deploy code that reads new column
-- 6. Drop old columns (after confirming no code uses them)
ALTER TABLE users DROP COLUMN first_name, DROP COLUMN last_name;
```

**Dangerous patterns to avoid:**
- Adding a NOT NULL column without a default (locks table on large data)
- Renaming a column in one deployment (break existing code)
- Dropping a column before removing all references in code

---

### Q19. How would you design a schema to support multi-tenancy?

**Answer:**
Multi-tenancy means multiple customers share the same application, with data isolation between them.

**Option 1: Shared database, shared schema (add tenant_id to every table)**
```sql
-- Every table has a tenant_id column
CREATE TABLE orders (
    id         INT PRIMARY KEY,
    tenant_id  INT NOT NULL,    -- Which customer this belongs to
    customer_id INT NOT NULL,
    total      DECIMAL(10,2),
    FOREIGN KEY (tenant_id) REFERENCES tenants(id)
);

-- All queries MUST include tenant_id
SELECT * FROM orders WHERE tenant_id = :current_tenant AND customer_id = 42;

-- Use Row-Level Security (PostgreSQL) to enforce tenant isolation at DB level
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON orders USING (tenant_id = current_setting('app.tenant_id')::INT);
```

**Option 2: Shared database, separate schema (one schema per tenant)**
```sql
CREATE SCHEMA tenant_123;
CREATE TABLE tenant_123.orders (...);
-- Set search_path = tenant_123 per session
```

**Option 3: Separate database per tenant (highest isolation, most overhead)**

| Option | Isolation | Complexity | Cost | Best for |
|--------|-----------|-----------|------|---------|
| Shared schema | Low | Low | Low | Many small tenants |
| Separate schema | Medium | Medium | Medium | Dozens of tenants |
| Separate DB | High | High | High | Regulated/enterprise |

---

### Q20. Design a database schema for a ride-sharing app (like Uber/Lyft).

**Answer:**

```sql
-- Core entities and their relationships

CREATE TABLE users (
    id            INT PRIMARY KEY AUTO_INCREMENT,
    email         VARCHAR(100) UNIQUE NOT NULL,
    phone         VARCHAR(20) UNIQUE NOT NULL,
    full_name     VARCHAR(100) NOT NULL,
    user_type     ENUM('rider', 'driver', 'both') NOT NULL,
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE driver_profiles (
    user_id       INT PRIMARY KEY REFERENCES users(id),
    license_no    VARCHAR(50) UNIQUE NOT NULL,
    vehicle_make  VARCHAR(50),
    vehicle_model VARCHAR(50),
    vehicle_year  INT,
    vehicle_plate VARCHAR(20) UNIQUE,
    rating        DECIMAL(3,2) DEFAULT 5.00,
    is_available  BOOLEAN DEFAULT FALSE
);

CREATE TABLE rides (
    id              INT PRIMARY KEY AUTO_INCREMENT,
    rider_id        INT NOT NULL REFERENCES users(id),
    driver_id       INT REFERENCES users(id),         -- NULL until accepted
    status          ENUM('requested','accepted','in_progress','completed','cancelled') DEFAULT 'requested',
    pickup_lat      DECIMAL(10,8) NOT NULL,
    pickup_lng      DECIMAL(11,8) NOT NULL,
    dropoff_lat     DECIMAL(10,8) NOT NULL,
    dropoff_lng     DECIMAL(11,8) NOT NULL,
    pickup_address  VARCHAR(300),
    dropoff_address VARCHAR(300),
    fare_estimate   DECIMAL(8,2),
    actual_fare     DECIMAL(8,2),
    distance_km     DECIMAL(8,3),
    duration_min    INT,
    requested_at    TIMESTAMP DEFAULT NOW(),
    accepted_at     TIMESTAMP,
    started_at      TIMESTAMP,
    completed_at    TIMESTAMP,

    INDEX idx_rides_rider  (rider_id, requested_at DESC),
    INDEX idx_rides_driver (driver_id, requested_at DESC),
    INDEX idx_rides_status (status, requested_at)
);

CREATE TABLE payments (
    id            INT PRIMARY KEY AUTO_INCREMENT,
    ride_id       INT UNIQUE NOT NULL REFERENCES rides(id),
    rider_id      INT NOT NULL REFERENCES users(id),
    amount        DECIMAL(8,2) NOT NULL,
    method        ENUM('card','cash','wallet'),
    status        ENUM('pending','completed','failed','refunded') DEFAULT 'pending',
    processed_at  TIMESTAMP
);

CREATE TABLE ratings (
    id         INT PRIMARY KEY AUTO_INCREMENT,
    ride_id    INT NOT NULL REFERENCES rides(id),
    rater_id   INT NOT NULL REFERENCES users(id),  -- Who gave the rating
    ratee_id   INT NOT NULL REFERENCES users(id),  -- Who was rated
    score      TINYINT NOT NULL CHECK (score BETWEEN 1 AND 5),
    comment    VARCHAR(500),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (ride_id, rater_id)    -- One rating per person per ride
);
```

**Key design decisions:**
- `driver_profiles` separated — not all users are drivers
- Location as lat/lng DECIMAL (consider PostGIS for spatial queries)
- Status as ENUM with progression
- Timestamps for each state transition (auditability)
- Separate payments table for payment complexity
- Bidirectional ratings (rider rates driver, driver rates rider)
