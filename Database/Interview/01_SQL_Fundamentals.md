# SQL Fundamentals — Interview Questions

> **Difficulty Mix:** Easy (Q1–Q8) · Medium (Q9–Q15) · Hard (Q16–Q20)

---

### Q1. What is SQL and what are its main categories of commands?

**Answer:**
SQL (Structured Query Language) is the standard language for managing relational databases. Commands fall into five categories:

| Category | Purpose | Commands |
|----------|---------|----------|
| **DDL** — Data Definition Language | Define/modify structure | CREATE, ALTER, DROP, TRUNCATE, RENAME |
| **DML** — Data Manipulation Language | Manipulate data | INSERT, UPDATE, DELETE, MERGE |
| **DQL** — Data Query Language | Retrieve data | SELECT |
| **DCL** — Data Control Language | Manage access | GRANT, REVOKE |
| **TCL** — Transaction Control Language | Manage transactions | COMMIT, ROLLBACK, SAVEPOINT |

---

### Q2. What is the difference between DELETE, TRUNCATE, and DROP?

**Answer:**

| Feature | DELETE | TRUNCATE | DROP |
|---------|--------|----------|------|
| Scope | Specific rows | All rows | Entire table |
| WHERE clause | ✓ Yes | ✗ No | ✗ No |
| Can ROLLBACK | ✓ Yes | ✗ No (usually) | ✗ No |
| Fires triggers | ✓ Yes | ✗ No | ✗ No |
| Resets identity | ✗ No | ✓ Yes | N/A |
| Speed | Slow (row-logged) | Fast | Fast |
| Keeps structure | ✓ Yes | ✓ Yes | ✗ No |

**Key rule:** Use DELETE when you need selective or rollback-able removal. Use TRUNCATE to wipe all data quickly. Use DROP to eliminate the table entirely.

---

### Q3. What is NULL in SQL? How do you check for it?

**Answer:**
NULL represents the **absence of a value** — it is not zero, not an empty string, not false. It is unknown.

**Rules:**
- Any arithmetic with NULL produces NULL: `5 + NULL = NULL`
- Any comparison with NULL produces UNKNOWN (not TRUE/FALSE): `NULL = NULL` → UNKNOWN
- Aggregate functions (SUM, AVG, COUNT) **ignore** NULLs; `COUNT(*)` counts all rows including NULLs but `COUNT(col)` skips them

**How to check:**
```sql
WHERE col IS NULL        -- ✓ Correct
WHERE col IS NOT NULL    -- ✓ Correct
WHERE col = NULL         -- ✗ Always returns 0 rows (comparison with NULL = UNKNOWN)
```

**Utility functions:**
```sql
COALESCE(a, b, c)   -- first non-NULL value
NULLIF(a, b)        -- returns NULL if a = b
IFNULL(a, default)  -- MySQL
ISNULL(a, default)  -- SQL Server
```

---

### Q4. What is the SQL execution order?

**Answer:**
SQL is **written** in one order but **executed** in a different order:

```
Written:   SELECT ... FROM ... JOIN ... WHERE ... GROUP BY ... HAVING ... ORDER BY ... LIMIT
Executed:
  1. FROM           → identify source tables
  2. JOIN           → combine tables
  3. WHERE          → filter rows (before grouping)
  4. GROUP BY       → group rows
  5. HAVING         → filter groups (after grouping)
  6. SELECT         → compute column expressions
  7. DISTINCT       → remove duplicates
  8. ORDER BY       → sort
  9. LIMIT / OFFSET → restrict output rows
```

**Why it matters:**
- You **cannot** use a SELECT alias in WHERE (WHERE runs before SELECT)
- You **can** use a SELECT alias in ORDER BY (ORDER BY runs after SELECT)
- Aggregate functions cannot appear in WHERE — use HAVING instead

---

### Q5. What is the difference between WHERE and HAVING?

**Answer:**

| Feature | WHERE | HAVING |
|---------|-------|--------|
| Execution stage | Before GROUP BY | After GROUP BY |
| Filters | Individual rows | Groups |
| Aggregate functions | ✗ Not allowed | ✓ Allowed |

```sql
-- WHERE: filter rows before grouping
SELECT dept_id, AVG(salary)
FROM employees
WHERE is_active = TRUE          -- ✓ row filter
GROUP BY dept_id
HAVING AVG(salary) > 70000;    -- ✓ group filter

-- This FAILS:
WHERE AVG(salary) > 70000      -- ✗ aggregate in WHERE — error
```

---

### Q6. What is the difference between CHAR and VARCHAR?

**Answer:**

| Feature | CHAR(n) | VARCHAR(n) |
|---------|---------|------------|
| Storage | Fixed-length, always n bytes | Variable-length, only actual data |
| Trailing spaces | Padded with spaces | Not padded |
| Speed | Slightly faster for reads | Slightly slower (length metadata) |
| Use case | Fixed-size data (country codes, codes) | Variable-length strings |

```sql
CHAR(2)       -- 'US' stored as 'US', 'A' stored as 'A '
VARCHAR(100)  -- 'Hello' stored as 5 bytes, not 100
```

**Rule of thumb:** Use CHAR for data that is always a fixed length (ISO codes, hash digests). Use VARCHAR for everything else.

---

### Q7. What is a primary key? What are its rules?

**Answer:**
A primary key uniquely identifies each row in a table.

**Rules:**
1. Must be **UNIQUE** — no two rows can have the same value
2. Must be **NOT NULL** — cannot be absent
3. Only **one** primary key per table (can be composite)
4. Automatically creates a **clustered index** (in MySQL InnoDB)

```sql
-- Single column
id INT PRIMARY KEY AUTO_INCREMENT

-- Composite (spans multiple columns)
PRIMARY KEY (order_id, product_id)
```

**Natural key** (business meaningful: email, SSN) vs **Surrogate key** (system-generated: auto-increment id, UUID) — surrogate keys are generally preferred for stability.

---

### Q8. What is a foreign key and what does it enforce?

**Answer:**
A foreign key is a column in one table that references the **primary key** (or unique key) of another table. It enforces **referential integrity** — you cannot insert a row that references a non-existent parent, and (by default) cannot delete a parent row that has children.

```sql
CREATE TABLE orders (
    id          INT PRIMARY KEY,
    customer_id INT,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
        ON DELETE CASCADE    -- Delete orders when customer deleted
        ON UPDATE CASCADE    -- Update orders when customer id changes
);
```

**Referential actions:**
- `CASCADE` — propagate the change
- `RESTRICT` / `NO ACTION` — error if children exist (default)
- `SET NULL` — set FK column to NULL
- `SET DEFAULT` — set FK to its default value

---

### Q9. What are constraints in SQL? Name them all.

**Answer:**
Constraints enforce data integrity rules at the column or table level.

| Constraint | Purpose |
|-----------|---------|
| `PRIMARY KEY` | Uniquely identifies each row; NOT NULL + UNIQUE |
| `FOREIGN KEY` | Enforces referential integrity between tables |
| `UNIQUE` | All values in column(s) must be distinct (NULLs may repeat) |
| `NOT NULL` | Column cannot contain NULL |
| `CHECK` | Column value must satisfy a condition |
| `DEFAULT` | Provides a default value when none is specified |

```sql
CREATE TABLE employees (
    id         INT           PRIMARY KEY AUTO_INCREMENT,
    email      VARCHAR(100)  NOT NULL UNIQUE,
    salary     DECIMAL(10,2) DEFAULT 0.00 CHECK (salary >= 0),
    dept_id    INT           REFERENCES departments(id),
    status     VARCHAR(20)   CHECK (status IN ('active','inactive','suspended'))
);
```

---

### Q10. What is the difference between UNION and UNION ALL?

**Answer:**

| Feature | UNION | UNION ALL |
|---------|-------|-----------|
| Duplicates | Removed (implicit DISTINCT) | Kept |
| Performance | Slower (requires sort/hash to dedup) | Faster (no dedup step) |
| Use case | When duplicates must be eliminated | When all rows needed or guaranteed no duplicates |

```sql
-- Both require same number of columns and compatible types
SELECT id, name FROM employees_us
UNION ALL              -- Keep all; don't deduplicate
SELECT id, name FROM employees_eu
ORDER BY name;         -- ORDER BY applies to final result, not each SELECT
```

**Rule:** Always prefer `UNION ALL` unless you explicitly need deduplication — it can be 2-10x faster on large datasets.

---

### Q11. Explain the difference between a clustered and non-clustered index.

**Answer:**

| Feature | Clustered Index | Non-Clustered Index |
|---------|----------------|---------------------|
| Data order | Table data physically stored in index order | Separate structure; points to table rows |
| Count per table | **Only 1** (one physical order) | Many (up to ~999 in SQL Server) |
| Row access | Direct (data IS the leaf node) | Indirect (leaf → row pointer → table) |
| Default in MySQL (InnoDB) | PRIMARY KEY | All other indexes |

**Implication:** In MySQL InnoDB, the primary key IS the clustered index. Choosing a good primary key (sequential INT vs random UUID) significantly affects insert performance because sequential values avoid page splits.

---

### Q12. What is DISTINCT and when should you avoid it?

**Answer:**
`DISTINCT` removes duplicate rows from the result set. Internally it requires sorting or hashing, which is expensive.

```sql
SELECT DISTINCT dept_id FROM employees;
```

**When to avoid DISTINCT:**
1. When duplicates come from a poorly written JOIN — fix the JOIN instead
2. When you only need to check existence — use EXISTS
3. When GROUP BY achieves the same result with better semantics

```sql
-- Bad: using DISTINCT to hide JOIN multiplicity
SELECT DISTINCT e.name FROM employees e JOIN orders o ON o.employee_id = e.id;

-- Better: understand why duplicates exist; or use EXISTS
SELECT e.name FROM employees e WHERE EXISTS (SELECT 1 FROM orders WHERE employee_id = e.id);
```

---

### Q13. What is a CASE expression? What are its two forms?

**Answer:**
CASE is SQL's conditional expression (like if-else). It returns a value and can appear anywhere an expression is valid — SELECT, WHERE, ORDER BY, GROUP BY.

**Simple CASE** (compares one value):
```sql
CASE dept_id
    WHEN 10 THEN 'Engineering'
    WHEN 20 THEN 'Marketing'
    ELSE 'Other'
END
```

**Searched CASE** (each WHEN has its own condition):
```sql
CASE
    WHEN salary < 50000  THEN 'Junior'
    WHEN salary < 90000  THEN 'Mid-level'
    WHEN salary < 140000 THEN 'Senior'
    ELSE                      'Executive'
END
```

**Use in ORDER BY (custom sort):**
```sql
ORDER BY CASE status WHEN 'urgent' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END
```

---

### Q14. How does the LIKE operator work? What are its wildcards?

**Answer:**

| Wildcard | Meaning |
|----------|---------|
| `%` | Zero or more characters |
| `_` | Exactly one character |

```sql
WHERE name LIKE 'A%'       -- Starts with A
WHERE name LIKE '%son'     -- Ends with son
WHERE name LIKE '%ann%'    -- Contains ann
WHERE code LIKE 'A_-__'    -- A, then 1 char, then -, then 2 chars
```

**Performance notes:**
- `LIKE 'prefix%'` can use a B-tree index
- `LIKE '%suffix'` and `LIKE '%middle%'` **cannot** use B-tree index (full scan)
- For full-text search use FULLTEXT index (MySQL) or tsvector/GIN (PostgreSQL)
- For wildcard prefix/suffix search, use trigram indexes (`pg_trgm` in PostgreSQL)

---

### Q15. What is a view? How does it differ from a table?

**Answer:**
A view is a stored SELECT query that behaves like a virtual table. It does **not store data** — the underlying query executes every time the view is accessed.

```sql
CREATE VIEW active_employees AS
SELECT id, name, salary, dept_id FROM employees WHERE is_active = TRUE;

SELECT * FROM active_employees WHERE salary > 70000;  -- Uses the view
```

| Feature | Table | View |
|---------|-------|------|
| Stores data | ✓ Yes | ✗ No (re-executes query) |
| Indexes | ✓ Yes | ✗ No (use materialized view) |
| DML | ✓ Always | Only if updatable (simple single-table views) |
| Purpose | Store data | Abstraction, security, reusability |

**Materialized View** (PostgreSQL, Oracle) physically stores the result and must be refreshed — combines view's simplicity with table's performance.

---

### Q16. What is the three-valued logic in SQL and why does it matter?

**Answer:**
SQL uses **three-valued logic**: TRUE, FALSE, and **UNKNOWN** (which NULL comparisons produce). This departs from standard boolean logic and causes subtle bugs.

```
NULL = 5    → UNKNOWN
NULL != 5   → UNKNOWN
NULL > 5    → UNKNOWN
NULL = NULL → UNKNOWN (not TRUE!)
```

**Truth table with UNKNOWN:**
| A | B | A AND B | A OR B |
|---|---|---------|--------|
| TRUE | UNKNOWN | UNKNOWN | TRUE |
| FALSE | UNKNOWN | FALSE | UNKNOWN |
| UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |

**Practical implication — NOT IN with NULLs:**
```sql
-- If orders.customer_id has any NULL rows, this returns 0 rows:
SELECT * FROM customers WHERE id NOT IN (SELECT customer_id FROM orders);
-- Because: id NOT IN (..., NULL) → id != NULL → UNKNOWN → row excluded

-- Safe replacement:
SELECT * FROM customers c WHERE NOT EXISTS (SELECT 1 FROM orders WHERE customer_id = c.id);
```

---

### Q17. What is the difference between RANK(), DENSE_RANK(), and ROW_NUMBER()?

**Answer:**
All three are window functions that assign a number to each row based on ORDER BY. They differ in how they handle **ties**:

| Function | Ties | Gaps |
|----------|------|------|
| ROW_NUMBER() | No ties (always unique) | No gaps |
| RANK() | Same rank for ties | Gaps after ties |
| DENSE_RANK() | Same rank for ties | No gaps |

```sql
-- salary: 90000, 80000, 80000, 70000
ROW_NUMBER → 1, 2, 3, 4   (unique always)
RANK       → 1, 2, 2, 4   (gap: no rank 3)
DENSE_RANK → 1, 2, 2, 3   (no gap)
```

**Use cases:**
- ROW_NUMBER: pagination, deduplication (keep 1 row per group)
- RANK: competition-style ranking where ties share a rank
- DENSE_RANK: "which salary tier" — no gaps in tier numbering

---

### Q18. What is the difference between implicit and explicit JOINs?

**Answer:**

**Implicit JOIN** (old SQL-89 syntax — avoid this):
```sql
SELECT e.name, d.name
FROM employees e, departments d
WHERE e.dept_id = d.id;
```

**Explicit JOIN** (SQL-92 syntax — use this):
```sql
SELECT e.name, d.name
FROM employees e
JOIN departments d ON e.dept_id = d.id;
```

**Why implicit joins are dangerous:**
- Forgetting the WHERE condition produces a CROSS JOIN (Cartesian product) silently
- Harder to read with multiple tables
- Makes the join type unclear (is this INNER or LEFT?)
- Modern explicit JOIN syntax is clearer, safer, and supported by all query optimizers the same way

---

### Q19. What is a correlated subquery? When is it inefficient?

**Answer:**
A **correlated subquery** references a column from the outer query. It is re-evaluated **for each row** of the outer query.

```sql
-- For every employee row, this subquery runs once:
SELECT e1.name, e1.salary
FROM employees e1
WHERE e1.salary > (
    SELECT AVG(e2.salary)
    FROM employees e2
    WHERE e2.dept_id = e1.dept_id  -- ← correlated to outer row
);
```

**Why it's inefficient:** If the outer query returns N rows, the subquery executes N times — O(N) subquery executions. For a 100,000-row table, this means 100,000 subquery executions.

**Rewrite as JOIN (runs once):**
```sql
SELECT e.name, e.salary
FROM employees e
JOIN (SELECT dept_id, AVG(salary) AS avg_sal FROM employees GROUP BY dept_id) d
    ON e.dept_id = d.dept_id
WHERE e.salary > d.avg_sal;
```

**Or with window function:**
```sql
SELECT name, salary FROM (
    SELECT name, salary, AVG(salary) OVER (PARTITION BY dept_id) AS dept_avg FROM employees
) t WHERE salary > dept_avg;
```

---

### Q20. Explain the concept of a surrogate key vs a natural key. Which do you prefer and why?

**Answer:**

**Natural key:** A key derived from the actual data that has business meaning.
- Examples: email address, SSN, ISBN, phone number, product code

**Surrogate key:** A system-generated key with no business meaning.
- Examples: auto-increment integer, UUID/GUID

| Dimension | Natural Key | Surrogate Key |
|-----------|------------|---------------|
| Meaningfulness | Has business meaning | No inherent meaning |
| Stability | Can change (email changes) | Never changes |
| Size | Varies (string = larger) | Small (INT = 4 bytes) |
| Index performance | Larger, slower B-tree | Compact, fast B-tree |
| Joins | Complex multi-column JOINs | Simple `WHERE id = ?` |
| Data entry errors | Must validate format | Auto-generated |

**Preferred approach:** Surrogate keys as primary keys, with a UNIQUE constraint on the natural key:
```sql
CREATE TABLE customers (
    id    INT PRIMARY KEY AUTO_INCREMENT,  -- surrogate
    email VARCHAR(100) UNIQUE NOT NULL     -- natural key, still enforced
);
```

This gives you stable join keys, compact indexes, and business rule enforcement via the UNIQUE constraint.
