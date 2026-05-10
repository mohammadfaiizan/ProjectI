# SQL Joins — Interview Questions

> **Difficulty Mix:** Easy (Q1–Q7) · Medium (Q8–Q14) · Hard (Q15–Q20)

---

### Q1. What are the different types of JOINs in SQL?

**Answer:**

| JOIN Type | Returns |
|-----------|---------|
| `INNER JOIN` | Only rows that match in **both** tables |
| `LEFT JOIN` | All rows from left + matching from right (NULL for non-matches) |
| `RIGHT JOIN` | Matching from left + all rows from right (NULL for non-matches) |
| `FULL OUTER JOIN` | All rows from both tables (NULL on non-matching sides) |
| `CROSS JOIN` | Cartesian product — every row of left × every row of right |
| `SELF JOIN` | A table joined to itself (using aliases) |

```sql
-- Visual mnemonic using sets A (left) and B (right):
INNER JOIN       → A ∩ B
LEFT JOIN        → A (with B where matched)
RIGHT JOIN       → B (with A where matched)
FULL OUTER JOIN  → A ∪ B
CROSS JOIN       → A × B (all combinations)
```

---

### Q2. What is INNER JOIN? Write an example.

**Answer:**
INNER JOIN returns only the rows where the join condition is satisfied in **both** tables. Non-matching rows are excluded from both sides.

```sql
SELECT e.name, e.salary, d.name AS department
FROM employees e
INNER JOIN departments d ON e.dept_id = d.id;

-- "INNER" is optional — JOIN alone defaults to INNER JOIN
SELECT e.name, d.name
FROM employees e
JOIN departments d ON e.dept_id = d.id;
```

If an employee has a NULL `dept_id`, they are excluded. If a department has no employees, it is excluded. Only **matching pairs** appear.

---

### Q3. What is the difference between LEFT JOIN and RIGHT JOIN?

**Answer:**
Both are OUTER JOINs — they preserve all rows from one side and NULL-fill the other side for non-matches.

- **LEFT JOIN:** All rows from the **left** (first) table. Right-side columns are NULL when no match.
- **RIGHT JOIN:** All rows from the **right** (second) table. Left-side columns are NULL when no match.

```sql
-- LEFT JOIN: all employees, even those without a department
SELECT e.name, d.name AS dept
FROM employees e          -- left table (all rows kept)
LEFT JOIN departments d ON e.dept_id = d.id;

-- RIGHT JOIN: all departments, even those with no employees
SELECT e.name, d.name AS dept
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.id;  -- right table (all rows kept)

-- RIGHT JOIN can always be rewritten as LEFT JOIN by swapping table order:
FROM departments d LEFT JOIN employees e ON e.dept_id = d.id;
```

**Preference:** Most developers use LEFT JOIN exclusively and swap table order as needed. RIGHT JOIN is rarely seen in practice.

---

### Q4. What is a SELF JOIN? Give a practical example.

**Answer:**
A SELF JOIN joins a table to **itself**. You must use table aliases to distinguish the two "instances" of the same table. Useful for hierarchical or comparative data within a single table.

**Example 1 — Employee–Manager hierarchy:**
```sql
-- Find each employee and their manager's name
SELECT
    e.name  AS employee,
    m.name  AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```

**Example 2 — Find pairs of employees in the same department:**
```sql
SELECT a.name AS emp1, b.name AS emp2, a.dept_id
FROM employees a
JOIN employees b
    ON a.dept_id = b.dept_id
   AND a.id < b.id;   -- a.id < b.id avoids (Alice, Bob) and (Bob, Alice) duplicates
```

---

### Q5. What is a CROSS JOIN and when is it used?

**Answer:**
A CROSS JOIN returns the **Cartesian product** of two tables — every row from the left table combined with every row from the right table. With M rows in left and N rows in right, it produces M×N rows.

```sql
SELECT s.size, c.color
FROM sizes s
CROSS JOIN colors c;
-- 4 sizes × 6 colors = 24 combinations
```

**Practical uses:**
- Generate all combinations (size × color, date × product)
- Build a date spine (all dates in a range)
- Load testing with repeated data

```sql
-- Generate a date range using CROSS JOIN + numbering
SELECT DATE_ADD('2024-01-01', INTERVAL n DAY) AS dt
FROM (SELECT 0 UNION SELECT 1 UNION SELECT 2 ... UNION SELECT 365) nums(n);
```

**Warning:** A CROSS JOIN without a WHERE condition on large tables can produce billions of rows and crash the server.

---

### Q6. What is the USING clause in a JOIN?

**Answer:**
`USING(col)` is a shorthand for `ON t1.col = t2.col` when **both tables have a column with the same name**.

```sql
-- Using ON:
SELECT e.name, d.name
FROM employees e JOIN departments d ON e.dept_id = d.dept_id;

-- Using USING (cleaner):
SELECT e.name, d.name
FROM employees e JOIN departments d USING (dept_id);
```

**Benefit:** With `USING`, the column appears only **once** in `SELECT *` output (not duplicated from both tables). With `ON`, both `e.dept_id` and `d.dept_id` appear in `SELECT *`.

---

### Q7. How do you find rows in table A that do NOT exist in table B (anti-join)?

**Answer:**
Three equivalent approaches — all find "orphan" rows in A:

**Method 1: LEFT JOIN + IS NULL (most common)**
```sql
SELECT a.*
FROM table_a a
LEFT JOIN table_b b ON a.id = b.a_id
WHERE b.id IS NULL;           -- No match found in B
```

**Method 2: NOT EXISTS (safest with NULLs)**
```sql
SELECT * FROM table_a a
WHERE NOT EXISTS (
    SELECT 1 FROM table_b b WHERE b.a_id = a.id
);
```

**Method 3: NOT IN (avoid if subquery can have NULLs)**
```sql
SELECT * FROM table_a
WHERE id NOT IN (SELECT a_id FROM table_b WHERE a_id IS NOT NULL);
-- Must add IS NOT NULL — if table_b.a_id has any NULL, NOT IN returns empty set!
```

**Best practice:** Use NOT EXISTS. It handles NULLs correctly and is usually optimized as efficiently as LEFT JOIN.

---

### Q8. What is a FULL OUTER JOIN? How do you simulate it in MySQL?

**Answer:**
FULL OUTER JOIN returns **all rows from both tables**. Where no match exists on either side, the other side's columns are NULL.

```sql
-- PostgreSQL / SQL Server
SELECT e.name, d.name
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.id;
-- Rows with no dept: e.name = 'Eve', d.name = NULL
-- Depts with no employees: e.name = NULL, d.name = 'HR'
```

**MySQL has no FULL OUTER JOIN** — simulate with UNION:
```sql
SELECT e.name, d.name
FROM employees e LEFT JOIN departments d ON e.dept_id = d.id

UNION

SELECT e.name, d.name
FROM employees e RIGHT JOIN departments d ON e.dept_id = d.id;
```

**Use case:** Find unmatched rows on both sides simultaneously (data reconciliation, finding orphans in either direction).

---

### Q9. Explain the difference between JOIN ON and JOIN ON + WHERE for outer joins.

**Answer:**
The placement of a filter condition significantly changes the result for OUTER JOINs:

**Filter in ON clause** — applied *before* the outer join (unmatched rows still returned):
```sql
SELECT e.name, d.name
FROM employees e
LEFT JOIN departments d
    ON e.dept_id = d.id
   AND d.location = 'NY';      -- Only match NY departments
-- Result: all employees; non-NY employees show d.name = NULL
-- Employees not in any NY dept are still returned
```

**Filter in WHERE clause** — applied *after* the outer join (converts LEFT JOIN to INNER JOIN for filtered rows):
```sql
SELECT e.name, d.name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id
WHERE d.location = 'NY';       -- Removes rows where d.location IS NULL or not 'NY'
-- Result: only employees in NY departments (same as INNER JOIN!)
```

**Rule:** To truly use a LEFT JOIN and filter by an attribute of the right table, put the filter in the `ON` clause, not `WHERE`.

---

### Q10. What is a non-equi join?

**Answer:**
A non-equi join uses an inequality operator (>, <, >=, <=, BETWEEN, !=) instead of `=` in the join condition.

**Example 1 — Salary bands:**
```sql
SELECT e.name, e.salary, b.band_name
FROM employees e
JOIN salary_bands b
    ON e.salary BETWEEN b.min_salary AND b.max_salary;
```

**Example 2 — Find employees who earn more than their manager:**
```sql
SELECT e.name, e.salary, m.name AS manager, m.salary AS manager_salary
FROM employees e
JOIN employees m ON e.manager_id = m.id
WHERE e.salary > m.salary;
```

**Example 3 — Version history (find the active price for each order):**
```sql
SELECT o.id, o.order_date, p.price
FROM orders o
JOIN price_history p
    ON p.product_id = o.product_id
   AND o.order_date BETWEEN p.valid_from AND p.valid_to;
```

---

### Q11. What is a LATERAL JOIN (or CROSS APPLY / OUTER APPLY)?

**Answer:**
A LATERAL join allows the **right-side subquery to reference columns from the left-side table** — like a correlated subquery that returns multiple rows/columns.

```sql
-- PostgreSQL: For each department, get the top 2 earners
SELECT d.name, top_emp.name, top_emp.salary
FROM departments d
JOIN LATERAL (
    SELECT name, salary
    FROM employees
    WHERE dept_id = d.id           -- References outer d.id
    ORDER BY salary DESC
    LIMIT 2
) top_emp ON TRUE;
```

**SQL Server equivalent:**
```sql
SELECT d.name, e.name, e.salary
FROM departments d
CROSS APPLY (
    SELECT TOP 2 name, salary FROM employees WHERE dept_id = d.id ORDER BY salary DESC
) e;

-- OUTER APPLY includes departments with no employees (NULL for emp columns)
```

**Use cases:** Top-N per group, running totals per row, calling table-valued functions with row arguments.

---

### Q12. How many rows does a JOIN produce? Give examples.

**Answer:**
The number of rows depends on the join type and cardinality of the relationship:

```
Tables:
employees: 5 rows (emp1-emp5)
departments: 3 rows (dept1-dept3)
emp3 has no dept_id; dept3 has no employees

INNER JOIN:     4 rows (only matched pairs)
LEFT JOIN:      5 rows (all employees; emp3 shows NULL dept)
RIGHT JOIN:     5 rows (4 matched + dept3 with NULL emp)
FULL OUTER JOIN: 6 rows (4 matched + emp3 NULL + dept3 NULL)
CROSS JOIN:     5 × 3 = 15 rows (all combinations)
```

**Many-to-many multiplication:**
```sql
-- If employee has 3 skills and we join skills table:
-- 1 employee × 3 skills = 3 rows per employee
-- 100 employees × avg 3 skills = ~300 rows
-- DISTINCT or proper aggregation needed to avoid "fan-out" bug
```

---

### Q13. What is the "fan-out" problem in JOINs?

**Answer:**
Fan-out occurs when a JOIN causes row multiplication — you join a fact table to a detail table and unknowingly double-count aggregates.

```sql
-- orders: 1 row per order (total = 100)
-- order_items: 3 rows per order

-- WRONG: double-counts order total
SELECT o.customer_id, SUM(o.total) AS revenue   -- total counted 3x per order!
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
GROUP BY o.customer_id;

-- CORRECT approach 1: aggregate items first
SELECT o.customer_id, SUM(o.total) AS revenue
FROM orders o
GROUP BY o.customer_id;

-- CORRECT approach 2: use a window function
SELECT DISTINCT o.customer_id,
    SUM(o.total) OVER (PARTITION BY o.customer_id) AS revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.id;

-- CORRECT approach 3: aggregate items then join
SELECT o.customer_id, SUM(o.total) AS order_revenue, SUM(items.item_total) AS item_revenue
FROM orders o
JOIN (SELECT order_id, SUM(unit_price * quantity) AS item_total FROM order_items GROUP BY order_id) items
    ON o.id = items.order_id
GROUP BY o.customer_id;
```

---

### Q14. Write a query to get the department with the most employees using a JOIN.

**Answer:**
```sql
-- Method 1: Subquery + GROUP BY
SELECT d.name, COUNT(e.id) AS headcount
FROM departments d
LEFT JOIN employees e ON e.dept_id = d.id
GROUP BY d.id, d.name
ORDER BY headcount DESC
LIMIT 1;

-- Method 2: WITH TIES (SQL Server — include tied departments)
SELECT TOP 1 WITH TIES d.name, COUNT(e.id) AS headcount
FROM departments d
LEFT JOIN employees e ON e.dept_id = d.id
GROUP BY d.id, d.name
ORDER BY COUNT(e.id) DESC;

-- Method 3: Using RANK (handles ties, no LIMIT needed)
SELECT name, headcount
FROM (
    SELECT d.name, COUNT(e.id) AS headcount,
        RANK() OVER (ORDER BY COUNT(e.id) DESC) AS rnk
    FROM departments d
    LEFT JOIN employees e ON e.dept_id = d.id
    GROUP BY d.id, d.name
) t
WHERE rnk = 1;
```

---

### Q15. What is the difference between JOIN and subquery in terms of performance?

**Answer:**
Modern query optimizers often rewrite subqueries as joins internally, so performance is frequently equivalent. However, there are cases where they differ:

**JOINs are generally better when:**
- You need columns from both tables in SELECT
- The subquery is non-correlated (same result regardless of outer row)
- The optimizer can choose the join algorithm (nested loop, hash join, merge join)

**Subqueries can be better when:**
- Using EXISTS — stops at first match (no need to read all rows)
- Scalar subquery that returns one value (may be cached)
- Correlated subquery with a highly selective condition

```sql
-- Equivalent — optimizer usually rewrites to the same plan:
-- 1. IN subquery
SELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE country = 'US');

-- 2. JOIN
SELECT DISTINCT o.* FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.country = 'US';

-- 3. EXISTS (stops at first match — most efficient for existence check)
SELECT * FROM orders o WHERE EXISTS (SELECT 1 FROM customers WHERE id = o.customer_id AND country = 'US');
```

**Always verify with EXPLAIN** — the optimizer may transform them identically or not.

---

### Q16. How would you find all employees who share the same manager and also work in the same department?

**Answer:**
```sql
-- Self-join to find pairs with same manager AND same department
SELECT
    a.name  AS employee1,
    b.name  AS employee2,
    a.manager_id,
    a.dept_id
FROM employees a
JOIN employees b
    ON  a.manager_id = b.manager_id   -- Same manager
    AND a.dept_id    = b.dept_id       -- Same department
    AND a.id < b.id                    -- Avoid (Alice,Bob) AND (Bob,Alice); and (Alice,Alice)
ORDER BY a.manager_id, a.dept_id;
```

---

### Q17. Explain NATURAL JOIN. Why is it generally avoided in production?

**Answer:**
NATURAL JOIN automatically joins on **all columns with the same name** in both tables — no ON clause needed.

```sql
SELECT * FROM employees NATURAL JOIN departments;
-- Joins on all shared column names
```

**Why it's avoided:**
1. **Fragile** — if a new column with the same name is added to either table, the join condition silently changes
2. **Implicit** — not obvious from the query which columns are being joined on
3. **Accidental matches** — columns like `created_at` or `updated_at` appear in many tables; NATURAL JOIN would join on these too

```sql
-- Safer explicit alternative using USING:
SELECT * FROM employees JOIN departments USING (dept_id);
-- Or explicit ON:
SELECT * FROM employees JOIN departments ON employees.dept_id = departments.id;
```

---

### Q18. You have a products table and a sales table. Write a query to find products that have NEVER been sold.

**Answer:**
```sql
-- Method 1: LEFT JOIN anti-join pattern (most common)
SELECT p.id, p.name
FROM products p
LEFT JOIN sales s ON s.product_id = p.id
WHERE s.id IS NULL;            -- No matching sale found

-- Method 2: NOT EXISTS
SELECT p.id, p.name
FROM products p
WHERE NOT EXISTS (
    SELECT 1 FROM sales WHERE product_id = p.id
);

-- Method 3: NOT IN (fragile if sales.product_id has NULLs)
SELECT id, name FROM products
WHERE id NOT IN (SELECT product_id FROM sales WHERE product_id IS NOT NULL);

-- Method 4: EXCEPT (PostgreSQL)
SELECT id FROM products
EXCEPT
SELECT DISTINCT product_id FROM sales;
```

**Best choice:** LEFT JOIN IS NULL or NOT EXISTS. NOT EXISTS is preferred when NULLs might exist in the FK column.

---

### Q19. What is the difference between ON and WHERE when filtering in a JOIN for INNER vs OUTER joins?

**Answer:**

For **INNER JOIN**, ON and WHERE are functionally equivalent — the optimizer treats them the same:
```sql
-- These produce identical results for INNER JOIN:
FROM a JOIN b ON a.id = b.a_id AND b.status = 'active'
FROM a JOIN b ON a.id = b.a_id WHERE b.status = 'active'
```

For **OUTER JOINs**, they behave differently:
```sql
-- ON: filter applied before outer join — all left rows still returned
FROM customers c LEFT JOIN orders o ON c.id = o.customer_id AND o.status = 'shipped'
-- Returns ALL customers; o columns are NULL if customer has no shipped orders

-- WHERE: filter applied after outer join — NULL rows removed (converts to INNER JOIN)
FROM customers c LEFT JOIN orders o ON c.id = o.customer_id WHERE o.status = 'shipped'
-- Returns ONLY customers who have shipped orders (LEFT JOIN semantics lost!)
```

**Rule:** To preserve outer join semantics, put right-table filters in the `ON` clause, not `WHERE`.

---

### Q20. Describe a JOIN that is causing performance issues. How would you diagnose and fix it?

**Answer:**
**Symptom:** A query with JOINs runs slowly (minutes instead of seconds).

**Diagnosis steps:**
```sql
-- 1. Run EXPLAIN to see the execution plan
EXPLAIN ANALYZE
SELECT e.name, d.name, COUNT(o.id)
FROM employees e
JOIN departments d ON e.dept_id = d.id
JOIN orders o ON o.employee_id = e.id
GROUP BY e.id, d.id;

-- Look for:
-- - Seq Scan on large tables (missing index)
-- - Hash/Nested Loop with high "rows" estimates
-- - Actual rows >> Estimated rows (stale statistics)
-- - Sorts on large datasets
```

**Common fixes:**

| Problem | Fix |
|---------|-----|
| Missing index on join column | `CREATE INDEX idx_orders_employee ON orders(employee_id)` |
| Missing index on filter column | `CREATE INDEX idx ON employees(dept_id)` |
| Stale statistics | `ANALYZE employees; ANALYZE orders;` |
| Fan-out from 1:many | Aggregate the many-side first, then join |
| Joining large tables on non-indexed strings | Normalize to integer keys |
| Cartesian product (forgot join condition) | Check ON clause — add the missing condition |
| Implicit type mismatch | Ensure joined columns have same data type |

```sql
-- After adding index:
CREATE INDEX idx_orders_emp ON orders (employee_id);
CREATE INDEX idx_emp_dept   ON employees (dept_id);
ANALYZE TABLE employees, departments, orders;
-- Re-run EXPLAIN — should show Index Scans instead of Seq Scans
```
