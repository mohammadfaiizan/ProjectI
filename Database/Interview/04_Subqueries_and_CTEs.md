# Subqueries and CTEs — Interview Questions

> **Difficulty Mix:** Easy (Q1–Q7) · Medium (Q8–Q14) · Hard (Q15–Q20)

---

### Q1. What is a subquery? What are the different types?

**Answer:**
A subquery (also called inner query or nested query) is a SELECT statement embedded inside another SQL statement, enclosed in parentheses.

**Types by position:**

| Type | Location | Returns |
|------|----------|---------|
| Scalar subquery | SELECT list or WHERE with comparison | Exactly one value |
| Multi-row subquery | WHERE with IN / NOT IN / ANY / ALL | A column of values |
| Derived table (inline view) | FROM clause | A result set |
| Correlated subquery | WHERE / HAVING (references outer query) | Any of the above |

```sql
-- Scalar subquery in WHERE
WHERE salary > (SELECT AVG(salary) FROM employees)

-- Multi-row in WHERE
WHERE dept_id IN (SELECT id FROM departments WHERE location = 'NY')

-- Derived table in FROM
FROM (SELECT dept_id, AVG(salary) avg FROM employees GROUP BY dept_id) d

-- Scalar subquery in SELECT
SELECT name, (SELECT name FROM departments WHERE id = e.dept_id) AS dept FROM employees e
```

---

### Q2. What is the difference between a subquery and a JOIN?

**Answer:**

| | Subquery | JOIN |
|-|----------|------|
| Output | Can return a scalar, column, or table | Returns rows combining both tables |
| Readability | Can be cleaner for existence checks | Cleaner when you need both tables' columns |
| Performance | Optimizer often rewrites to join | Usually same after optimization |
| Duplicates | Rarely causes fan-out | Can cause row multiplication |
| Use case | Existence checks, scalar lookups | Need columns from both tables |

```sql
-- Subquery — only need employees table in output:
SELECT name FROM employees
WHERE dept_id IN (SELECT id FROM departments WHERE location = 'NY');

-- JOIN — need columns from both tables:
SELECT e.name, d.location
FROM employees e JOIN departments d ON e.dept_id = d.id
WHERE d.location = 'NY';
```

Modern optimizers often rewrite IN subqueries as JOINs internally. The execution plan may be identical.

---

### Q3. What is a CTE (Common Table Expression)?

**Answer:**
A CTE is a named temporary result set defined with the `WITH` clause, scoped to a single SQL statement. It improves readability by breaking complex queries into named logical steps.

```sql
WITH dept_averages AS (
    SELECT dept_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_id
)
SELECT e.name, e.salary, d.avg_sal
FROM employees e
JOIN dept_averages d ON e.dept_id = d.dept_id
WHERE e.salary > d.avg_sal;
```

**Advantages over subqueries:**
1. Can be referenced **multiple times** in the same query
2. More readable — complex logic broken into named steps
3. Supports **recursive** queries (hierarchies, graphs)
4. Easier to debug — test each CTE independently

---

### Q4. What is a correlated subquery? Why can it be slow?

**Answer:**
A correlated subquery references a column from the **outer query**. It re-executes **once per row** of the outer query.

```sql
-- Correlated: runs once per employee row
SELECT e1.name, e1.salary
FROM employees e1
WHERE e1.salary > (
    SELECT AVG(e2.salary)
    FROM employees e2
    WHERE e2.dept_id = e1.dept_id   -- ← references outer e1.dept_id
);
```

**Why slow:** If the outer query returns 10,000 rows, the subquery executes 10,000 times — O(N) subquery runs. Each execution may itself be a full scan.

**Rewrite as JOIN for O(1) subquery cost:**
```sql
SELECT e.name, e.salary
FROM employees e
JOIN (SELECT dept_id, AVG(salary) avg_sal FROM employees GROUP BY dept_id) d
    ON e.dept_id = d.dept_id
WHERE e.salary > d.avg_sal;
```

---

### Q5. What is the difference between IN and EXISTS?

**Answer:**

| Feature | IN | EXISTS |
|---------|-----|--------|
| Evaluates | Materializes full subquery result | Stops at first match |
| NULLs in subquery | Dangerous with NOT IN | Safe with NOT EXISTS |
| Performance | Better for small subquery results | Better for large subquery / existence-only check |
| Correlated | Usually not | Usually is |

```sql
-- IN: materializes all customer_ids, then checks membership
SELECT * FROM orders
WHERE customer_id IN (SELECT id FROM customers WHERE country = 'US');

-- EXISTS: stops as soon as one US customer is found for this order
SELECT * FROM orders o
WHERE EXISTS (SELECT 1 FROM customers WHERE id = o.customer_id AND country = 'US');
```

**Critical NULL behavior:**
```sql
-- If customers has any NULL id, NOT IN returns 0 rows:
SELECT * FROM orders WHERE customer_id NOT IN (SELECT id FROM customers);
-- id != NULL is UNKNOWN → row excluded!

-- NOT EXISTS is always safe:
SELECT * FROM orders o WHERE NOT EXISTS (SELECT 1 FROM customers WHERE id = o.customer_id);
```

---

### Q6. What is ANY / ALL in SQL?

**Answer:**
`ANY` and `ALL` compare a value against a **set of values** returned by a subquery.

```sql
-- = ANY is equivalent to IN
WHERE salary = ANY (SELECT salary FROM managers)

-- > ANY means "greater than at least one" = "greater than the minimum"
WHERE salary > ANY (SELECT avg_salary FROM dept_stats)
-- Returns employees earning more than the LOWEST dept average

-- > ALL means "greater than every value" = "greater than the maximum"
WHERE salary > ALL (SELECT avg_salary FROM dept_stats)
-- Returns employees earning more than the HIGHEST dept average

-- < ALL = less than the minimum
-- <> ALL is equivalent to NOT IN (with same NULL gotcha)
```

**SOME** is a synonym for **ANY** (rarely used).

---

### Q7. What is a derived table?

**Answer:**
A derived table is a subquery in the FROM clause. It acts as a temporary table for the outer query.

```sql
-- Derived table: avg salary per dept, filtered to > 70000
SELECT dept_id, avg_sal
FROM (
    SELECT dept_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_id
) AS dept_stats                    -- ← must have an alias (MySQL requires it)
WHERE avg_sal > 70000;

-- Derived table with JOIN
SELECT e.name, stats.avg_sal
FROM employees e
JOIN (
    SELECT dept_id, AVG(salary) AS avg_sal, COUNT(*) AS cnt
    FROM employees GROUP BY dept_id
) stats ON e.dept_id = stats.dept_id
WHERE stats.cnt >= 5;
```

**Difference from CTE:** A derived table can only be used **once** in the query. A CTE can be **referenced multiple times**.

---

### Q8. Multiple CTEs — how do you chain them?

**Answer:**
Multiple CTEs are separated by commas after the `WITH` keyword. Later CTEs can reference earlier ones.

```sql
WITH
-- Step 1: Calculate dept averages
dept_avgs AS (
    SELECT dept_id, AVG(salary) AS avg_sal, COUNT(*) AS headcount
    FROM employees
    GROUP BY dept_id
),
-- Step 2: Find high-performing depts (avg > 80000)
top_depts AS (
    SELECT dept_id
    FROM dept_avgs
    WHERE avg_sal > 80000 AND headcount >= 5
),
-- Step 3: Get employees in those depts
top_employees AS (
    SELECT e.* FROM employees e
    JOIN top_depts td ON e.dept_id = td.dept_id
    WHERE e.salary > (SELECT avg_sal FROM dept_avgs WHERE dept_id = e.dept_id)
)
SELECT t.name, t.salary, d.avg_sal
FROM top_employees t
JOIN dept_avgs d ON t.dept_id = d.dept_id
ORDER BY t.salary DESC;
```

Each CTE is computed in the order defined. The final SELECT uses the last CTE's output.

---

### Q9. What is a recursive CTE? What are its parts?

**Answer:**
A recursive CTE references itself to process hierarchical or sequential data. It has two mandatory parts connected by `UNION ALL`:

```sql
WITH RECURSIVE cte_name AS (
    -- 1. ANCHOR MEMBER (non-recursive, base case)
    SELECT base_row_data FROM table WHERE starting_condition

    UNION ALL

    -- 2. RECURSIVE MEMBER (references cte_name)
    SELECT more_data
    FROM table
    JOIN cte_name ON ...     -- references itself
    WHERE stopping_condition -- must eventually become FALSE
)
SELECT * FROM cte_name;
```

**Example — employee hierarchy:**
```sql
WITH RECURSIVE org_tree AS (
    -- Anchor: CEO (no manager)
    SELECT id, name, manager_id, 0 AS depth
    FROM employees WHERE manager_id IS NULL

    UNION ALL

    -- Recursive: each level's direct reports
    SELECT e.id, e.name, e.manager_id, ot.depth + 1
    FROM employees e
    JOIN org_tree ot ON e.manager_id = ot.id
)
SELECT REPEAT('  ', depth) || name AS hierarchy, depth
FROM org_tree ORDER BY depth, name;
```

**Termination:** The recursion stops when the recursive SELECT returns 0 rows. Always include a condition that shrinks the dataset each iteration.

---

### Q10. How do you use a CTE in an UPDATE or DELETE statement?

**Answer:**

```sql
-- UPDATE using CTE (PostgreSQL)
WITH dept_averages AS (
    SELECT dept_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_id
)
UPDATE employees e
SET salary = da.avg_sal
FROM dept_averages da
WHERE e.dept_id = da.dept_id
  AND e.salary < da.avg_sal * 0.80;  -- Employees 20% below dept avg

-- DELETE using CTE
WITH old_orders AS (
    SELECT id FROM orders
    WHERE order_date < NOW() - INTERVAL '2 YEAR'
      AND status = 'archived'
)
DELETE FROM order_items
WHERE order_id IN (SELECT id FROM old_orders);

-- INSERT using CTE
WITH validated_customers AS (
    SELECT name, email FROM staging_customers
    WHERE email LIKE '%@%.%' AND name IS NOT NULL
)
INSERT INTO customers (name, email)
SELECT name, email FROM validated_customers
ON CONFLICT (email) DO NOTHING;
```

---

### Q11. What is a materialized CTE? When does it matter?

**Answer:**
In PostgreSQL, a CTE is **materialized** by default (evaluated once, result stored). In MySQL, CTEs are always inlined (treated as derived tables).

**Materialized (default in PostgreSQL):**
```sql
-- Computed once, result cached
WITH MATERIALIZED expensive_calc AS (
    SELECT id, complex_function(data) AS result FROM big_table
)
SELECT * FROM expensive_calc WHERE result > 100
UNION ALL
SELECT * FROM expensive_calc WHERE result < 0;
-- expensive_calc is computed only ONCE
```

**Inlined (allow optimizer to push filters):**
```sql
-- NOT MATERIALIZED: optimizer may push WHERE inside the CTE
WITH NOT MATERIALIZED dept_emps AS (
    SELECT * FROM employees WHERE dept_id = 10
)
SELECT * FROM dept_emps WHERE salary > 80000;
-- Optimizer can push salary > 80000 into the CTE scan
```

**When it matters:**
- Expensive CTEs referenced multiple times → materialized saves cost
- CTEs with selective outer filters → `NOT MATERIALIZED` lets optimizer push filters inside

---

### Q12. Write a recursive CTE to generate a date range.

**Answer:**
```sql
-- MySQL / PostgreSQL — generate all dates in 2024
WITH RECURSIVE date_series AS (
    SELECT CAST('2024-01-01' AS DATE) AS dt    -- anchor: first date

    UNION ALL

    SELECT dt + INTERVAL '1 DAY'               -- add one day each iteration
    FROM date_series
    WHERE dt < '2024-12-31'                    -- stop at last date
)
SELECT dt FROM date_series;
-- Returns 366 rows (2024 is a leap year)

-- Alternative: PostgreSQL generate_series (no recursion needed)
SELECT generate_series('2024-01-01'::DATE, '2024-12-31'::DATE, '1 day'::INTERVAL)::DATE AS dt;

-- Use case: left-join with sales to find days with zero sales
WITH dates AS (
    SELECT generate_series('2024-01-01'::DATE, '2024-12-31'::DATE, '1 day') AS dt
)
SELECT d.dt, COALESCE(SUM(s.amount), 0) AS revenue
FROM dates d
LEFT JOIN sales s ON s.sale_date = d.dt
GROUP BY d.dt ORDER BY d.dt;
```

---

### Q13. How do you find the Nth highest value without using LIMIT/TOP?

**Answer:**
```sql
-- Using correlated subquery — find 3rd highest salary
SELECT DISTINCT salary
FROM employees e1
WHERE 2 = (
    SELECT COUNT(DISTINCT salary)
    FROM employees e2
    WHERE e2.salary > e1.salary
    -- When exactly 2 salaries are higher, e1.salary is 3rd
);

-- General formula: for Nth highest, use N-1 as the count

-- More readable with DENSE_RANK:
SELECT salary
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employees
) t
WHERE rnk = 3     -- Replace 3 with N
LIMIT 1;

-- Using CTE:
WITH ranked AS (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS dr
    FROM employees
)
SELECT MIN(salary) FROM ranked WHERE dr = 3;
```

---

### Q14. What is the difference between a CTE and a temporary table?

**Answer:**

| Feature | CTE | Temporary Table |
|---------|-----|----------------|
| Scope | Single query | Entire session |
| Storage | Not stored (re-executed unless materialized) | Stored in tempdb/temp schema |
| Indexable | ✗ No | ✓ Yes |
| Reusable across queries | ✗ No | ✓ Yes |
| Recursive | ✓ Yes | ✗ No |
| Performance on large data | May be slower (repeated execution) | Faster (indexed, stored once) |

```sql
-- CTE: scope of one query
WITH temp_data AS (SELECT id, salary FROM employees WHERE dept_id = 10)
SELECT * FROM temp_data;  -- temp_data gone after this query

-- Temp table: persists for the session
CREATE TEMP TABLE temp_data AS SELECT id, salary FROM employees WHERE dept_id = 10;
CREATE INDEX ON temp_data (id);      -- Can be indexed!
SELECT * FROM temp_data;             -- Available in next query too
DROP TABLE temp_data;                -- Must clean up
```

**Rule of thumb:** Use CTE for readability within one query. Use temp table for complex multi-step processing or when you need indexes on intermediate results.

---

### Q15. Write a recursive CTE to find all descendants of a given employee in an org chart.

**Answer:**
```sql
-- employees: (id, name, manager_id)
-- Find all direct and indirect reports of employee with id = 1 (the CEO)

WITH RECURSIVE subordinates AS (
    -- Anchor: start with the target employee
    SELECT id, name, manager_id, 0 AS depth
    FROM employees
    WHERE id = 1

    UNION ALL

    -- Recursive: find direct reports of each found employee
    SELECT e.id, e.name, e.manager_id, s.depth + 1
    FROM employees e
    INNER JOIN subordinates s ON e.manager_id = s.id
)
SELECT id, name, depth,
    REPEAT('  ', depth) || name AS indented_name
FROM subordinates
ORDER BY depth, name;

-- To find only direct reports (depth = 1):
-- Add WHERE depth = 1 in outer SELECT

-- With path tracking:
WITH RECURSIVE subordinates AS (
    SELECT id, name, manager_id, 0 AS depth, CAST(name AS VARCHAR(1000)) AS path
    FROM employees WHERE id = 1

    UNION ALL

    SELECT e.id, e.name, e.manager_id, s.depth + 1, s.path || ' → ' || e.name
    FROM employees e JOIN subordinates s ON e.manager_id = s.id
)
SELECT id, name, depth, path FROM subordinates;
```

---

### Q16. Explain relational division using SQL. What is the "who ordered all products" problem?

**Answer:**
Relational division: Find all entities in Table A that are associated with **every** entity in Table B. Classic example: find customers who ordered **all** products in a given list.

```sql
-- Tables:
-- customers (id, name)
-- orders (customer_id, product_id)
-- required_products: products we need every customer to have ordered

-- Method 1: Double NOT EXISTS (classic relational division)
SELECT DISTINCT c.id, c.name
FROM customers c
WHERE NOT EXISTS (
    -- Is there a required product that this customer did NOT order?
    SELECT 1 FROM products p
    WHERE p.is_required = TRUE
    AND NOT EXISTS (
        SELECT 1 FROM orders o
        WHERE o.customer_id = c.id AND o.product_id = p.id
    )
);

-- Method 2: COUNT approach (cleaner, same result)
SELECT o.customer_id
FROM orders o
JOIN products p ON o.product_id = p.id
WHERE p.is_required = TRUE
GROUP BY o.customer_id
HAVING COUNT(DISTINCT o.product_id) = (
    SELECT COUNT(*) FROM products WHERE is_required = TRUE
);
```

---

### Q17. How do you detect and remove duplicate rows using a CTE?

**Answer:**
```sql
-- Step 1: Identify duplicates (same email = duplicate)
WITH duplicates AS (
    SELECT
        id,
        email,
        ROW_NUMBER() OVER (PARTITION BY email ORDER BY id ASC) AS rn
    FROM customers
    -- rn = 1: the "original" (lowest id)
    -- rn > 1: duplicates to remove
)

-- Step 2: View the duplicates
SELECT * FROM duplicates WHERE rn > 1;

-- Step 3: Delete them
DELETE FROM customers
WHERE id IN (SELECT id FROM duplicates WHERE rn > 1);

-- Or in PostgreSQL with CTE in DELETE:
WITH dupes AS (
    SELECT id, ROW_NUMBER() OVER (PARTITION BY email ORDER BY id) AS rn
    FROM customers
)
DELETE FROM customers WHERE id IN (SELECT id FROM dupes WHERE rn > 1);
```

---

### Q18. What happens if a recursive CTE has no termination condition?

**Answer:**
Without a termination condition, a recursive CTE runs **indefinitely** until:
- The database's recursion depth limit is hit (error)
- The database detects a cycle (if it has cycle detection)
- Server runs out of memory or time

```sql
-- INFINITE RECURSION — no stopping condition:
WITH RECURSIVE bad AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM bad    -- No WHERE clause — runs forever!
)
SELECT * FROM bad;
-- ERROR: maximum recursion depth exceeded (or OOM)

-- CORRECT — with stopping condition:
WITH RECURSIVE good AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM good WHERE n < 100   -- stops at 100
)
SELECT * FROM good;

-- PostgreSQL: set recursion limit
SET max_recursion_depth = 100;

-- MySQL: default max recursion = 1000
-- Override: SET @@cte_max_recursion_depth = 10000;
```

**Protection strategies:**
1. Always include a WHERE stopping condition
2. Include a depth counter and limit it
3. For cyclic graphs, track visited nodes in an array

---

### Q19. Write a query using a CTE to find managers whose entire team earns less than the company average.

**Answer:**
```sql
WITH company_avg AS (
    SELECT AVG(salary) AS avg_sal FROM employees
),
team_max_salary AS (
    SELECT
        e.manager_id,
        MAX(e.salary)  AS team_max,
        COUNT(e.id)    AS team_size
    FROM employees e
    WHERE e.manager_id IS NOT NULL
    GROUP BY e.manager_id
)
SELECT
    m.id AS manager_id,
    m.name AS manager_name,
    tm.team_size,
    tm.team_max AS highest_team_salary,
    ca.avg_sal AS company_avg
FROM employees m
JOIN team_max_salary tm ON m.id = tm.manager_id
CROSS JOIN company_avg ca
WHERE tm.team_max < ca.avg_sal   -- entire team earns less than company average
ORDER BY tm.team_max;
```

---

### Q20. How do you paginate efficiently using subqueries/CTEs vs OFFSET?

**Answer:**
**OFFSET pagination** (naive — avoid for large pages):
```sql
-- Page 1001 (10 rows per page) — scans first 10,010 rows!
SELECT * FROM orders ORDER BY id LIMIT 10 OFFSET 10000;
-- Performance degrades linearly with page number: O(page_number × page_size)
```

**Keyset (cursor) pagination** (efficient — use this):
```sql
-- Page 1: no cursor
SELECT * FROM orders ORDER BY id LIMIT 10;
-- Returns rows with ids ending at, say, 10050

-- Page 2: use last seen id as cursor
SELECT * FROM orders WHERE id > 10050 ORDER BY id LIMIT 10;
-- Always O(log n) — just an index seek

-- CTE version (cleaner):
WITH page AS (
    SELECT id, order_date, total
    FROM orders
    WHERE id > :last_seen_id    -- cursor from previous page
    ORDER BY id
    LIMIT :page_size
)
SELECT *, (SELECT MAX(id) FROM page) AS next_cursor FROM page;
-- Return next_cursor to the client for the next request
```

**When OFFSET is acceptable:** Small datasets, pages < 1000, report generation (not interactive). Always use keyset for user-facing paginated APIs on large tables.
