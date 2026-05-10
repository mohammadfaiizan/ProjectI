# Subqueries and CTEs

## Table of Contents
1. [Subqueries](#1-subqueries)
2. [Subquery Types by Position](#2-subquery-types-by-position)
3. [Correlated Subqueries](#3-correlated-subqueries)
4. [EXISTS and NOT EXISTS](#4-exists-and-not-exists)
5. [ANY / ALL / SOME](#5-any--all--some)
6. [CTEs (Common Table Expressions)](#6-ctes-common-table-expressions)
7. [Recursive CTEs](#7-recursive-ctess)
8. [Subquery vs JOIN vs CTE](#8-subquery-vs-join-vs-cte)

---

## 1. Subqueries

A subquery (inner query / nested query) is a SELECT statement embedded inside another SQL statement.

### Key Rules
- Must be enclosed in parentheses
- Can return a scalar (single value), a single column, or a full result set
- Executed before the outer query (usually)
- Cannot use ORDER BY inside a subquery (unless also using LIMIT/TOP)

### Basic Example
```sql
-- Find employees earning more than the average salary
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```

---

## 2. Subquery Types by Position

### In WHERE (Scalar Subquery)
Returns one value. Used with comparison operators.
```sql
-- Employees in the Engineering department
SELECT name
FROM employees
WHERE dept_id = (
    SELECT id FROM departments WHERE name = 'Engineering'
);

-- Employees earning the highest salary
SELECT name, salary
FROM employees
WHERE salary = (SELECT MAX(salary) FROM employees);

-- Employees earning above company average
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```

### In WHERE with IN (Multi-row Subquery)
Returns multiple values.
```sql
-- Employees in NY departments
SELECT name
FROM employees
WHERE dept_id IN (
    SELECT id FROM departments WHERE location = 'New York'
);

-- Orders with electronics items
SELECT DISTINCT order_id
FROM order_items
WHERE product_id IN (
    SELECT id FROM products WHERE category = 'Electronics'
);
```

### In FROM (Derived Table / Inline View)
The subquery acts as a temporary table.
```sql
-- Average salary per dept, then filter those > 70000
SELECT dept_id, avg_sal
FROM (
    SELECT dept_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_id
) AS dept_averages
WHERE avg_sal > 70000;

-- Rank employees within their department
SELECT outer_query.*
FROM (
    SELECT
        name,
        dept_id,
        salary,
        ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rn
    FROM employees
) AS outer_query
WHERE rn <= 3;

-- Note: MySQL requires subquery alias; PostgreSQL does not
```

### In SELECT (Scalar Subquery in projection)
Returns exactly one value per row.
```sql
-- Each employee's salary vs. department average
SELECT
    e.name,
    e.salary,
    (SELECT AVG(salary) FROM employees WHERE dept_id = e.dept_id) AS dept_avg,
    e.salary - (SELECT AVG(salary) FROM employees WHERE dept_id = e.dept_id) AS diff
FROM employees e;

-- Count of orders per customer
SELECT
    c.name,
    (SELECT COUNT(*) FROM orders WHERE customer_id = c.id) AS order_count
FROM customers c;
```

### In HAVING
```sql
-- Departments whose avg salary > company avg
SELECT dept_id, AVG(salary) AS dept_avg
FROM employees
GROUP BY dept_id
HAVING AVG(salary) > (SELECT AVG(salary) FROM employees);
```

### In INSERT / UPDATE / DELETE
```sql
-- INSERT
INSERT INTO salary_history (emp_id, salary, snapshot_date)
SELECT id, salary, CURRENT_DATE FROM employees WHERE dept_id = 10;

-- UPDATE
UPDATE employees
SET salary = salary * 1.10
WHERE dept_id = (SELECT id FROM departments WHERE name = 'Engineering');

-- DELETE
DELETE FROM employees
WHERE dept_id IN (SELECT id FROM departments WHERE is_active = FALSE);
```

---

## 3. Correlated Subqueries

A correlated subquery references a column from the outer query. It is re-executed for each row of the outer query.

### Basic Correlated Subquery
```sql
-- Find employees earning more than their department's average
SELECT e1.name, e1.salary, e1.dept_id
FROM employees e1
WHERE e1.salary > (
    SELECT AVG(e2.salary)
    FROM employees e2
    WHERE e2.dept_id = e1.dept_id  -- References outer query's row
);
```

### Find Latest Record per Group
```sql
-- Most recent order per customer
SELECT *
FROM orders o1
WHERE order_date = (
    SELECT MAX(o2.order_date)
    FROM orders o2
    WHERE o2.customer_id = o1.customer_id
);
```

### Find Nth Highest Value
```sql
-- 3rd highest salary (using correlated subquery)
SELECT DISTINCT salary
FROM employees e1
WHERE 2 = (
    SELECT COUNT(DISTINCT salary)
    FROM employees e2
    WHERE e2.salary > e1.salary
);
-- When COUNT = 2, e1.salary is the 3rd highest
```

### Performance Note
Correlated subqueries can be slow (O(n) subquery executions). Often replaceable with JOINs or window functions.

```sql
-- Correlated subquery (slow for large tables)
SELECT name, salary
FROM employees e1
WHERE salary > (
    SELECT AVG(salary) FROM employees e2 WHERE e2.dept_id = e1.dept_id
);

-- Equivalent JOIN (faster)
SELECT e.name, e.salary
FROM employees e
JOIN (
    SELECT dept_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_id
) d ON e.dept_id = d.dept_id
WHERE e.salary > d.avg_sal;
```

---

## 4. EXISTS and NOT EXISTS

EXISTS checks if a subquery returns any rows. It stops as soon as one match is found (efficient).

### EXISTS
```sql
-- Customers who have placed at least one order
SELECT c.name
FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders WHERE customer_id = c.id
);

-- Departments that have at least one employee
SELECT d.name
FROM departments d
WHERE EXISTS (
    SELECT 1 FROM employees WHERE dept_id = d.id
);
```

### NOT EXISTS (Anti-join)
```sql
-- Customers with no orders
SELECT c.name
FROM customers c
WHERE NOT EXISTS (
    SELECT 1 FROM orders WHERE customer_id = c.id
);

-- Products never ordered
SELECT p.name
FROM products p
WHERE NOT EXISTS (
    SELECT 1 FROM order_items WHERE product_id = p.id
);

-- Better than NOT IN when data contains NULLs
-- (NOT IN with NULLs returns 0 rows; NOT EXISTS handles NULLs correctly)
```

### EXISTS vs IN vs JOIN

```sql
-- All three find customers with at least one order

-- EXISTS (stops at first match — efficient)
SELECT name FROM customers c
WHERE EXISTS (SELECT 1 FROM orders WHERE customer_id = c.id);

-- IN (materializes the subquery)
SELECT name FROM customers
WHERE id IN (SELECT DISTINCT customer_id FROM orders);

-- JOIN (may produce duplicates without DISTINCT)
SELECT DISTINCT c.name
FROM customers c
JOIN orders o ON o.customer_id = c.id;
```

---

## 5. ANY / ALL / SOME

### ANY / SOME
Returns TRUE if the comparison is true for at least one row.
```sql
-- = ANY is equivalent to IN
SELECT name FROM employees
WHERE salary = ANY (SELECT salary FROM managers);

-- > ANY means "greater than at least one" = "greater than minimum"
SELECT name, salary
FROM employees
WHERE salary > ANY (
    SELECT AVG(salary) FROM employees GROUP BY dept_id
);
-- Returns employees earning more than the lowest departmental average

-- SOME is a synonym for ANY
SELECT name FROM employees
WHERE dept_id = SOME (SELECT id FROM departments WHERE location = 'NY');
```

### ALL
Returns TRUE if the comparison is true for every row.
```sql
-- Employees earning more than ALL managers
SELECT name, salary
FROM employees
WHERE salary > ALL (SELECT salary FROM managers);

-- > ALL means "greater than maximum"
SELECT name, salary
FROM employees
WHERE salary > ALL (
    SELECT AVG(salary) FROM employees GROUP BY dept_id
);
-- Returns employees earning more than the highest departmental average

-- = ALL makes sense only when all values are the same
-- < ALL means "less than minimum"
```

### NULL Gotcha with ANY/ALL
```sql
-- If subquery returns NULL, ALL comparisons with NULL return UNKNOWN
-- Be careful with NOT IN and ALL — use EXISTS/NOT EXISTS instead
```

---

## 6. CTEs (Common Table Expressions)

A CTE defines a named temporary result set using WITH clause. More readable than subqueries.

### Basic CTE Syntax
```sql
WITH cte_name AS (
    -- CTE query
    SELECT ...
)
SELECT * FROM cte_name;
```

### Simple CTE
```sql
-- Equivalent to derived table but more readable
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

### Multiple CTEs
```sql
WITH
high_earners AS (
    SELECT id, name, salary, dept_id
    FROM employees
    WHERE salary > 80000
),
dept_counts AS (
    SELECT dept_id, COUNT(*) AS total_emp
    FROM employees
    GROUP BY dept_id
)
SELECT
    h.name,
    h.salary,
    d.total_emp AS dept_size
FROM high_earners h
JOIN dept_counts d ON h.dept_id = d.dept_id
ORDER BY h.salary DESC;
```

### CTEs for Readability
```sql
-- Complex query with CTEs (readable)
WITH
raw_sales AS (
    SELECT
        DATE_TRUNC('month', sale_date) AS month,
        product_id,
        SUM(amount) AS total
    FROM sales
    WHERE sale_date >= '2024-01-01'
    GROUP BY 1, 2
),
ranked_sales AS (
    SELECT
        month,
        product_id,
        total,
        RANK() OVER (PARTITION BY month ORDER BY total DESC) AS rnk
    FROM raw_sales
)
SELECT month, product_id, total
FROM ranked_sales
WHERE rnk <= 5
ORDER BY month, rnk;
```

### CTEs in DML
```sql
-- CTE in UPDATE
WITH avg_salaries AS (
    SELECT dept_id, AVG(salary) AS avg_sal FROM employees GROUP BY dept_id
)
UPDATE employees e
SET salary = (SELECT avg_sal FROM avg_salaries a WHERE a.dept_id = e.dept_id)
WHERE salary < (SELECT avg_sal FROM avg_salaries a WHERE a.dept_id = e.dept_id) * 0.8;

-- CTE in DELETE
WITH old_orders AS (
    SELECT id FROM orders WHERE order_date < NOW() - INTERVAL '2 YEAR'
)
DELETE FROM order_items
WHERE order_id IN (SELECT id FROM old_orders);

-- PostgreSQL: CTE in INSERT
WITH new_customers AS (
    SELECT name, email FROM staging WHERE is_valid = TRUE
)
INSERT INTO customers (name, email)
SELECT name, email FROM new_customers;
```

### Materialized CTEs (PostgreSQL)
```sql
-- By default, PostgreSQL may inline or materialize CTEs
-- Force materialization (CTE evaluated once, result stored)
WITH MATERIALIZED expensive_calc AS (
    SELECT id, complex_function(data) AS result FROM big_table
)
SELECT * FROM expensive_calc WHERE result > 100;

-- Force inlining (allow optimizer to inline)
WITH NOT MATERIALIZED dept_avgs AS (
    SELECT dept_id, AVG(salary) FROM employees GROUP BY dept_id
)
SELECT * FROM dept_avgs;
```

---

## 7. Recursive CTEs

Recursive CTEs allow a CTE to reference itself. Used for hierarchical data (org charts, tree structures, path traversal).

### Syntax
```sql
WITH RECURSIVE cte_name AS (
    -- Anchor member (base case — non-recursive part)
    SELECT ...

    UNION ALL

    -- Recursive member (references cte_name)
    SELECT ...
    FROM cte_name
    WHERE termination_condition
)
SELECT * FROM cte_name;
```

### Hierarchy Traversal (Org Chart)
```sql
-- employees table:
-- id | name    | mgr_id
-- 1  | Alice   | NULL    <- CEO
-- 2  | Bob     | 1
-- 3  | Carol   | 1
-- 4  | Dave    | 2
-- 5  | Eve     | 2

WITH RECURSIVE org_chart AS (
    -- Anchor: start with CEO (top-level employees)
    SELECT id, name, mgr_id, 0 AS level, CAST(name AS VARCHAR(500)) AS path
    FROM employees
    WHERE mgr_id IS NULL

    UNION ALL

    -- Recursive: find direct reports of current level
    SELECT
        e.id,
        e.name,
        e.mgr_id,
        oc.level + 1,
        oc.path || ' > ' || e.name
    FROM employees e
    JOIN org_chart oc ON e.mgr_id = oc.id
)
SELECT
    REPEAT('  ', level) || name AS org_tree,
    level,
    path
FROM org_chart
ORDER BY path;

-- Result:
-- Alice               level 0
--   Bob               level 1
--     Dave            level 2
--     Eve             level 2
--   Carol             level 1
```

### Generate Number Series
```sql
-- PostgreSQL has generate_series(), but here's the recursive approach
WITH RECURSIVE nums AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM nums WHERE n < 100
)
SELECT n FROM nums;
```

### Generate Date Range
```sql
WITH RECURSIVE date_range AS (
    SELECT CAST('2024-01-01' AS DATE) AS dt
    UNION ALL
    SELECT dt + INTERVAL '1 DAY'
    FROM date_range
    WHERE dt < '2024-12-31'
)
SELECT dt FROM date_range;
```

### Find All Paths in Graph
```sql
-- category hierarchy
WITH RECURSIVE category_path AS (
    SELECT id, name, parent_id, CAST(name AS TEXT) AS full_path
    FROM categories
    WHERE parent_id IS NULL  -- Root categories

    UNION ALL

    SELECT c.id, c.name, c.parent_id, cp.full_path || ' > ' || c.name
    FROM categories c
    JOIN category_path cp ON c.parent_id = cp.id
)
SELECT id, full_path FROM category_path ORDER BY full_path;
```

### Cycle Detection
```sql
-- Prevent infinite loops in cyclic graphs
WITH RECURSIVE graph_traverse AS (
    SELECT
        node_id,
        ARRAY[node_id] AS visited,  -- Track visited nodes
        FALSE AS is_cycle
    FROM graph_nodes
    WHERE node_id = 1

    UNION ALL

    SELECT
        e.to_node,
        gt.visited || e.to_node,
        e.to_node = ANY(gt.visited)  -- Check if we've seen this node
    FROM graph_edges e
    JOIN graph_traverse gt ON e.from_node = gt.node_id
    WHERE NOT gt.is_cycle
)
SELECT * FROM graph_traverse;
```

### BFS (Breadth-First Search)
```sql
WITH RECURSIVE bfs AS (
    SELECT id, parent_id, 1 AS depth
    FROM tree_nodes
    WHERE parent_id IS NULL  -- Root

    UNION ALL

    SELECT t.id, t.parent_id, b.depth + 1
    FROM tree_nodes t
    JOIN bfs b ON t.parent_id = b.id
    WHERE b.depth < 5  -- Limit depth to prevent infinite recursion
)
SELECT * FROM bfs ORDER BY depth, id;
```

### MySQL Recursive CTE
```sql
-- Supported in MySQL 8.0+
WITH RECURSIVE subordinates AS (
    SELECT id, name, mgr_id FROM employees WHERE id = 1
    UNION ALL
    SELECT e.id, e.name, e.mgr_id
    FROM employees e
    INNER JOIN subordinates s ON e.mgr_id = s.id
)
SELECT * FROM subordinates;
```

---

## 8. Subquery vs JOIN vs CTE

### Performance Comparison

| Approach | Pros | Cons |
|----------|------|------|
| Subquery in WHERE | Simple to read | Can be slow if correlated |
| Derived table (subquery in FROM) | Flexible | Can't be referenced multiple times |
| JOIN | Usually fastest | Can be complex; may need DISTINCT |
| CTE | Readable, reusable in same query | May be materialized (PostgreSQL) |
| Window Function | No grouping needed | Syntax can be complex |

### When to Use Each

```sql
-- Use IN subquery when:
-- - List is small, query is simple
WHERE id IN (SELECT id FROM small_table WHERE condition)

-- Use EXISTS when:
-- - Checking existence only (don't need values)
-- - Subquery could return NULLs
WHERE EXISTS (SELECT 1 FROM related WHERE fk = t.id)

-- Use JOIN when:
-- - You need columns from both tables
-- - Performance is critical
JOIN related r ON r.fk = t.id

-- Use CTE when:
-- - Query needs to be reused multiple times
-- - Improves readability
-- - Recursive traversal needed
WITH cte AS (...) SELECT ...

-- Use Window Functions when:
-- - Ranking or running totals needed without collapsing rows
RANK() OVER (PARTITION BY dept_id ORDER BY salary)
```

---

## Quick Reference

```sql
-- Scalar subquery
WHERE col = (SELECT single_val FROM t WHERE ...)

-- Multi-row subquery
WHERE col IN (SELECT col FROM t WHERE ...)

-- Derived table (subquery in FROM)
FROM (SELECT ... FROM t) AS alias

-- Correlated subquery
WHERE col > (SELECT expr FROM t2 WHERE t2.fk = t1.id)

-- EXISTS
WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.fk = t1.id)

-- NOT EXISTS
WHERE NOT EXISTS (SELECT 1 FROM t2 WHERE t2.fk = t1.id)

-- ANY / ALL
WHERE val > ANY (SELECT val FROM t)
WHERE val > ALL (SELECT val FROM t)

-- CTE
WITH name AS (SELECT ...) SELECT ... FROM name;

-- Multiple CTEs
WITH a AS (...), b AS (...) SELECT ... FROM a JOIN b;

-- Recursive CTE
WITH RECURSIVE name AS (
    SELECT ...         -- Anchor
    UNION ALL
    SELECT ... FROM name WHERE stop_condition
)
SELECT * FROM name;
```
