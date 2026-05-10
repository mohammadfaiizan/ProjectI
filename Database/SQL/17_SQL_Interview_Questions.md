# SQL Interview Questions

## Table of Contents
1. [Fundamentals](#1-fundamentals)
2. [Joins](#2-joins)
3. [Aggregations and Grouping](#3-aggregations-and-grouping)
4. [Subqueries and CTEs](#4-subqueries-and-ctes)
5. [Window Functions](#5-window-functions)
6. [Schema Design and Normalization](#6-schema-design-and-normalization)
7. [Indexes and Performance](#7-indexes-and-performance)
8. [Transactions and Concurrency](#8-transactions-and-concurrency)
9. [Advanced Problems](#9-advanced-problems)
10. [Practical Coding Problems](#10-practical-coding-problems)

---

## 1. Fundamentals

**Q: What is the difference between WHERE and HAVING?**
> WHERE filters rows before grouping. HAVING filters groups after GROUP BY. You cannot use aggregate functions in WHERE; you can in HAVING.

**Q: What is the SQL execution order?**
> FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT

**Q: What is the difference between CHAR and VARCHAR?**
> CHAR(n) is fixed-length, padded with spaces, and slightly faster for fixed-size data. VARCHAR(n) is variable-length, stores only actual data, better for strings of varying length.

**Q: What is NULL and how is it different from 0 or empty string?**
> NULL represents the absence of a value. NULL != 0 and NULL != ''. Comparisons with NULL return UNKNOWN, not TRUE/FALSE. Use IS NULL / IS NOT NULL to check for NULL.

**Q: What does COALESCE do?**
> COALESCE returns the first non-NULL argument. `COALESCE(a, b, c)` returns a if not null, else b if not null, else c.

**Q: What is the difference between UNION and UNION ALL?**
> UNION removes duplicate rows (expensive — does a dedup step). UNION ALL keeps all rows including duplicates and is faster. Use UNION ALL when duplicates are acceptable.

**Q: What is a primary key? Can it be NULL?**
> A primary key uniquely identifies each row. It combines UNIQUE and NOT NULL constraints. A primary key CANNOT be NULL.

**Q: Can a table have multiple primary keys?**
> No. A table can have only one PRIMARY KEY, but the primary key can span multiple columns (composite primary key).

**Q: What is a foreign key?**
> A foreign key is a column (or set of columns) that references the primary key of another table, enforcing referential integrity.

**Q: What are DDL, DML, DQL, DCL, and TCL?**
> - DDL: CREATE, ALTER, DROP, TRUNCATE (structure)
> - DML: INSERT, UPDATE, DELETE, MERGE (data)
> - DQL: SELECT (query)
> - DCL: GRANT, REVOKE (access control)
> - TCL: COMMIT, ROLLBACK, SAVEPOINT (transactions)

**Q: What is the difference between DELETE, TRUNCATE, and DROP?**
| | DELETE | TRUNCATE | DROP |
|--|--------|----------|------|
| Scope | Selected rows | All rows | Entire table |
| WHERE | Yes | No | No |
| Rollback | Yes | No (usually) | No |
| Triggers | Fired | Not fired | Not fired |
| Speed | Slow | Fast | Fast |
| Structure | Kept | Kept | Removed |

---

## 2. Joins

**Q: What is the difference between INNER JOIN and LEFT JOIN?**
> INNER JOIN returns only matching rows from both tables. LEFT JOIN returns all rows from the left table and matching rows from the right table (NULL for non-matches).

**Q: What is a SELF JOIN? Give an example.**
```sql
-- Find each employee and their manager's name
SELECT e.name AS employee, m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```

**Q: What is a CROSS JOIN?**
> Returns the Cartesian product of two tables — every combination of rows from both tables. With N rows in A and M rows in B, CROSS JOIN produces N×M rows.

**Q: What is the difference between USING and ON in JOINs?**
> ON lets you specify any join condition. USING is shorthand when both tables share a column with the same name: `JOIN t USING (col)` is equivalent to `JOIN t ON t.col = a.col`.

**Q: What is a non-equi join?**
```sql
-- Join on inequality (salary within a salary band)
SELECT e.name, sb.band
FROM employees e
JOIN salary_bands sb ON e.salary BETWEEN sb.min AND sb.max;
```

**Q: Write a query to find all customers who have NOT placed any orders.**
```sql
-- Method 1: LEFT JOIN + IS NULL
SELECT c.name FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.id IS NULL;

-- Method 2: NOT EXISTS
SELECT c.name FROM customers c
WHERE NOT EXISTS (SELECT 1 FROM orders WHERE customer_id = c.id);

-- Method 3: NOT IN (careful with NULLs!)
SELECT c.name FROM customers c
WHERE c.id NOT IN (SELECT customer_id FROM orders WHERE customer_id IS NOT NULL);
```

**Q: What is the difference between FULL OUTER JOIN and CROSS JOIN?**
> FULL OUTER JOIN returns all rows from both tables, with NULLs for non-matching sides. CROSS JOIN returns every combination of rows (Cartesian product), regardless of any condition.

---

## 3. Aggregations and Grouping

**Q: What is the difference between COUNT(*), COUNT(col), and COUNT(DISTINCT col)?**
> - COUNT(*): counts all rows including NULLs
> - COUNT(col): counts non-NULL values in col
> - COUNT(DISTINCT col): counts unique non-NULL values in col

**Q: Can you use aggregate functions in WHERE? Why not?**
> No. WHERE executes before grouping (GROUP BY), so aggregates haven't been computed yet. Use HAVING to filter on aggregate results.

**Q: Find departments with more than 5 employees and average salary > 70000.**
```sql
SELECT dept_id, COUNT(*) AS cnt, AVG(salary) AS avg_sal
FROM employees
GROUP BY dept_id
HAVING COUNT(*) > 5 AND AVG(salary) > 70000;
```

**Q: What does ROLLUP do?**
> GROUP BY ROLLUP generates subtotals and a grand total. For GROUP BY ROLLUP(a, b), it generates:
> - (a, b) level subtotals
> - (a) level subtotals
> - () grand total

**Q: Write a query to get the total salary and percentage of total for each department.**
```sql
SELECT
    dept_id,
    SUM(salary) AS dept_total,
    ROUND(SUM(salary) * 100.0 / SUM(SUM(salary)) OVER (), 2) AS pct_of_total
FROM employees
GROUP BY dept_id;
```

---

## 4. Subqueries and CTEs

**Q: What is a correlated subquery? What are its performance implications?**
> A correlated subquery references a column from the outer query. It is re-executed once for each row of the outer query (O(n) executions), making it slow. Usually replaceable with a JOIN.

**Q: What is the difference between IN and EXISTS?**
> IN materializes the subquery result and checks membership. EXISTS stops as soon as one match is found and is more efficient when checking existence. EXISTS handles NULLs better — NOT EXISTS is safe; NOT IN fails silently when the subquery contains NULLs.

**Q: What is a CTE? How does it differ from a subquery?**
> A CTE (WITH clause) is a named temporary result set scoped to a single query. Unlike a subquery in FROM, a CTE can be referenced multiple times in the same query. CTEs improve readability and can be recursive.

**Q: What is a recursive CTE? Give an example.**
```sql
-- Traverse an employee hierarchy
WITH RECURSIVE org AS (
    SELECT id, name, manager_id, 0 AS level
    FROM employees WHERE manager_id IS NULL  -- CEO
    UNION ALL
    SELECT e.id, e.name, e.manager_id, o.level + 1
    FROM employees e
    JOIN org o ON e.manager_id = o.id
)
SELECT REPEAT('  ', level) || name AS hierarchy FROM org ORDER BY level, name;
```

**Q: Find the second highest salary.**
```sql
-- Method 1: Subquery
SELECT MAX(salary) FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- Method 2: LIMIT/OFFSET
SELECT DISTINCT salary FROM employees ORDER BY salary DESC LIMIT 1 OFFSET 1;

-- Method 3: Dense rank
SELECT salary FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employees
) t WHERE rnk = 2;
```

---

## 5. Window Functions

**Q: What is a window function? How does it differ from GROUP BY?**
> Window functions compute values across a set of rows related to the current row without collapsing them. GROUP BY collapses rows into one per group. Window functions preserve all rows and add a new computed column.

**Q: What is PARTITION BY in window functions?**
> PARTITION BY divides rows into groups (like GROUP BY for window functions) but doesn't collapse them. The window function is applied independently within each partition.

**Q: What is the difference between RANK, DENSE_RANK, and ROW_NUMBER?**
| Function | Ties | Gaps |
|----------|------|------|
| ROW_NUMBER | No ties (always unique) | No |
| RANK | Tied values get same rank | Yes (skips numbers) |
| DENSE_RANK | Tied values get same rank | No |

**Q: Find the top 3 earners per department.**
```sql
SELECT name, dept_id, salary
FROM (
    SELECT name, dept_id, salary,
        ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rn
    FROM employees
) t WHERE rn <= 3;
```

**Q: Calculate a 7-day moving average of daily sales.**
```sql
SELECT
    sale_date,
    daily_revenue,
    AVG(daily_revenue) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d
FROM daily_sales;
```

**Q: What is LAG and LEAD?**
```sql
-- LAG: access value from previous row
-- LEAD: access value from next row
SELECT
    month, revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_month,
    revenue - LAG(revenue) OVER (ORDER BY month) AS change
FROM monthly_revenue;
```

**Q: What is the default frame for window functions?**
> When ORDER BY is specified: `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` (all rows up to and including the current row's peers).
> When ORDER BY is not specified: `RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` (entire partition).

---

## 6. Schema Design and Normalization

**Q: What is normalization? Explain 1NF, 2NF, 3NF.**

**1NF (First Normal Form):**
> - Atomic values (no repeating groups, no arrays in a cell)
> - Each column has a single data type
> - Each row is uniquely identifiable

**2NF (Second Normal Form):**
> - Must be in 1NF
> - No partial dependencies (every non-key column depends on the WHOLE primary key, not part of it)
> - Applies when there's a composite primary key

**3NF (Third Normal Form):**
> - Must be in 2NF
> - No transitive dependencies (non-key columns depend only on the primary key, not on other non-key columns)

**Q: When would you denormalize a schema?**
> For read-heavy workloads where query performance is critical, you might denormalize to avoid expensive JOINs. Trade-off: redundant data, more complex updates. Common for analytics/reporting databases, data warehouses.

**Q: What is the difference between OLTP and OLAP?**
| OLTP | OLAP |
|------|------|
| Transactional (inserts/updates) | Analytical (complex reads) |
| Many small operations | Few complex queries |
| Normalized schema | Denormalized / star schema |
| Current data | Historical data |
| Example: e-commerce app | Example: BI dashboard |

**Q: What is a star schema?**
> A denormalized schema used in data warehouses with one central fact table and dimension tables. Example: orders fact table with product, customer, time dimension tables. Optimized for analytical queries.

---

## 7. Indexes and Performance

**Q: What is an index? How does it work?**
> An index is a separate data structure (typically a B-tree) that allows the database to find rows quickly without scanning the entire table. It stores sorted values with pointers to the actual rows, enabling O(log n) lookups instead of O(n) full scans.

**Q: What is the downside of adding more indexes?**
> Each index slows down INSERT/UPDATE/DELETE operations because the index must be updated for every data change. Indexes also consume disk space and memory.

**Q: What is a covering index?**
> An index that includes all columns referenced in a query (SELECT, WHERE, JOIN, ORDER BY), so the query can be satisfied entirely from the index without accessing the table. Results in "Index Only Scan".

**Q: What is the leftmost prefix rule for composite indexes?**
> For an index on (a, b, c), queries can use the index for:
> - WHERE a = ...
> - WHERE a = ... AND b = ...
> - WHERE a = ... AND b = ... AND c = ...
> But NOT for WHERE b = ... (skips the leftmost column).

**Q: Why would an index not be used?**
> - Function applied to indexed column: `WHERE UPPER(email) = ...`
> - LIKE with leading wildcard: `WHERE name LIKE '%smith'`
> - Type mismatch causing implicit cast
> - Low selectivity (optimizer prefers full scan)
> - Table is very small (full scan is faster)

**Q: What is EXPLAIN and how do you use it?**
> EXPLAIN shows the query execution plan without running the query. EXPLAIN ANALYZE (PostgreSQL) also executes the query and shows actual vs estimated rows and timing. Key things to look for: full table scans (Seq Scan, ALL type), missing index seeks.

---

## 8. Transactions and Concurrency

**Q: What are ACID properties?**
> - **A**tomicity: All operations succeed or all are rolled back
> - **C**onsistency: Database moves from one valid state to another
> - **I**solation: Concurrent transactions don't interfere with each other
> - **D**urability: Committed changes persist even after failure

**Q: What is a deadlock? How is it resolved?**
> A deadlock occurs when two transactions each hold a lock the other needs, so both wait forever. The database detects deadlocks and rolls back one transaction. Prevention: access resources in consistent order, keep transactions short.

**Q: What are the four transaction isolation levels?**
| Level | Dirty Read | Non-Repeatable | Phantom |
|-------|-----------|----------------|---------|
| READ UNCOMMITTED | Yes | Yes | Yes |
| READ COMMITTED | No | Yes | Yes |
| REPEATABLE READ | No | No | Yes* |
| SERIALIZABLE | No | No | No |

**Q: What is the difference between COMMIT and ROLLBACK?**
> COMMIT permanently saves all changes made in the transaction. ROLLBACK undoes all changes since the last BEGIN (or to a SAVEPOINT).

**Q: What is a SAVEPOINT?**
> A SAVEPOINT marks a point within a transaction to which you can partially roll back, without rolling back the entire transaction. Useful for loop processing where some rows might fail.

**Q: What is MVCC?**
> Multi-Version Concurrency Control keeps multiple versions of rows to allow readers and writers to not block each other. Each transaction sees a snapshot of the data as of when the transaction started.

---

## 9. Advanced Problems

**Q: Write a query to find duplicates in a table.**
```sql
-- Find duplicate emails (more than one row with same email)
SELECT email, COUNT(*) AS cnt
FROM employees
GROUP BY email
HAVING COUNT(*) > 1;

-- Get the duplicate rows themselves (with all details)
SELECT * FROM employees
WHERE email IN (
    SELECT email FROM employees GROUP BY email HAVING COUNT(*) > 1
)
ORDER BY email;
```

**Q: Delete duplicate rows, keeping the one with the lowest ID.**
```sql
-- Delete all duplicates, keep the MIN id per group
DELETE FROM employees
WHERE id NOT IN (
    SELECT MIN(id) FROM employees GROUP BY email
);

-- PostgreSQL (using CTE)
WITH duplicates AS (
    SELECT id, ROW_NUMBER() OVER (PARTITION BY email ORDER BY id) AS rn
    FROM employees
)
DELETE FROM employees WHERE id IN (
    SELECT id FROM duplicates WHERE rn > 1
);
```

**Q: Find employees who earn more than their department's average salary.**
```sql
SELECT e.name, e.salary, d.avg_sal
FROM employees e
JOIN (
    SELECT dept_id, AVG(salary) AS avg_sal FROM employees GROUP BY dept_id
) d ON e.dept_id = d.dept_id
WHERE e.salary > d.avg_sal;
```

**Q: Write a query to find the Nth highest salary.**
```sql
SELECT DISTINCT salary
FROM employees e1
WHERE N - 1 = (
    SELECT COUNT(DISTINCT salary)
    FROM employees e2
    WHERE e2.salary > e1.salary
);

-- More modern (replace N with actual number, e.g., 3):
SELECT salary FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employees
) t WHERE rnk = 3 LIMIT 1;
```

**Q: Write a query to get cumulative salary by hire date.**
```sql
SELECT
    name,
    hire_date,
    salary,
    SUM(salary) OVER (ORDER BY hire_date ROWS UNBOUNDED PRECEDING) AS cumulative_salary
FROM employees
ORDER BY hire_date;
```

**Q: Find managers with more than 3 direct reports.**
```sql
SELECT m.name AS manager, COUNT(e.id) AS direct_reports
FROM employees e
JOIN employees m ON e.manager_id = m.id
GROUP BY m.id, m.name
HAVING COUNT(e.id) > 3;
```

**Q: Write a query to compute year-over-year revenue growth.**
```sql
SELECT
    year,
    revenue,
    LAG(revenue) OVER (ORDER BY year) AS prev_year_revenue,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY year))
        / NULLIF(LAG(revenue) OVER (ORDER BY year), 0) * 100, 2
    ) AS yoy_growth_pct
FROM annual_revenue
ORDER BY year;
```

---

## 10. Practical Coding Problems

### Problem 1: Employees and Departments
```sql
-- Table: employees (id, name, salary, dept_id, manager_id)
-- Table: departments (id, name, location)

-- Q: List departments with their employee count and avg salary
SELECT
    d.name,
    COUNT(e.id) AS headcount,
    ROUND(AVG(e.salary), 2) AS avg_salary
FROM departments d
LEFT JOIN employees e ON e.dept_id = d.id
GROUP BY d.id, d.name;

-- Q: Find employees earning above their department average
SELECT e.name, e.salary, dept_avg.avg_sal
FROM employees e
JOIN (SELECT dept_id, AVG(salary) AS avg_sal FROM employees GROUP BY dept_id) dept_avg
    ON e.dept_id = dept_avg.dept_id
WHERE e.salary > dept_avg.avg_sal;

-- Q: Find the highest paid employee in each department
SELECT dept_id, name, salary
FROM (
    SELECT dept_id, name, salary,
        RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk
    FROM employees
) t WHERE rnk = 1;
```

### Problem 2: Orders and Products
```sql
-- Table: customers (id, name, email, created_at)
-- Table: orders (id, customer_id, order_date, total, status)
-- Table: order_items (id, order_id, product_id, quantity, unit_price)
-- Table: products (id, name, category, price)

-- Q: Top 5 customers by total spend
SELECT c.name, SUM(o.total) AS total_spent
FROM customers c
JOIN orders o ON o.customer_id = c.id
GROUP BY c.id, c.name
ORDER BY total_spent DESC
LIMIT 5;

-- Q: Month-over-month revenue
SELECT
    DATE_TRUNC('month', order_date) AS month,
    SUM(total) AS revenue,
    LAG(SUM(total)) OVER (ORDER BY DATE_TRUNC('month', order_date)) AS prev_month,
    ROUND(
        (SUM(total) - LAG(SUM(total)) OVER (ORDER BY DATE_TRUNC('month', order_date)))
        / NULLIF(LAG(SUM(total)) OVER (ORDER BY DATE_TRUNC('month', order_date)), 0) * 100, 2
    ) AS mom_growth
FROM orders
GROUP BY DATE_TRUNC('month', order_date);

-- Q: Customers who ordered in January but not February
SELECT DISTINCT customer_id
FROM orders
WHERE MONTH(order_date) = 1 AND YEAR(order_date) = 2024
  AND customer_id NOT IN (
    SELECT customer_id FROM orders
    WHERE MONTH(order_date) = 2 AND YEAR(order_date) = 2024
  );

-- Q: Best-selling product per category
SELECT category, product_id, total_qty
FROM (
    SELECT
        p.category,
        oi.product_id,
        SUM(oi.quantity) AS total_qty,
        RANK() OVER (PARTITION BY p.category ORDER BY SUM(oi.quantity) DESC) AS rnk
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    GROUP BY p.category, oi.product_id
) t WHERE rnk = 1;
```

### Problem 3: Sessions / Activity
```sql
-- Table: user_activity (user_id, activity_date)

-- Q: Find consecutive active days (streaks) per user
WITH days AS (
    SELECT DISTINCT user_id, activity_date FROM user_activity
),
grouped AS (
    SELECT
        user_id,
        activity_date,
        activity_date - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY activity_date) * INTERVAL '1 DAY' AS grp
    FROM days
)
SELECT user_id, MIN(activity_date) AS streak_start, MAX(activity_date) AS streak_end,
       COUNT(*) AS streak_days
FROM grouped
GROUP BY user_id, grp
ORDER BY user_id, streak_start;

-- Q: 30-day retention (users active 30 days after signup)
SELECT
    COUNT(DISTINCT u.id) AS total_users,
    COUNT(DISTINCT a.user_id) AS retained_users,
    COUNT(DISTINCT a.user_id) * 100.0 / COUNT(DISTINCT u.id) AS retention_pct
FROM users u
LEFT JOIN user_activity a
    ON a.user_id = u.id
    AND a.activity_date BETWEEN u.signup_date + INTERVAL '25 DAY'
                             AND u.signup_date + INTERVAL '35 DAY';
```

### Problem 4: Miscellaneous
```sql
-- Q: Transpose rows to columns (pivot)
SELECT
    employee_id,
    MAX(CASE WHEN skill = 'Python' THEN 1 ELSE 0 END) AS has_python,
    MAX(CASE WHEN skill = 'SQL'    THEN 1 ELSE 0 END) AS has_sql,
    MAX(CASE WHEN skill = 'Java'   THEN 1 ELSE 0 END) AS has_java
FROM employee_skills
GROUP BY employee_id;

-- Q: Running total resetting each month
SELECT
    sale_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY DATE_FORMAT(sale_date, '%Y-%m')
        ORDER BY sale_date
    ) AS monthly_running_total
FROM sales;

-- Q: Find gaps in a numeric sequence
SELECT a.id + 1 AS gap_start, MIN(b.id) - 1 AS gap_end
FROM sequence a
JOIN sequence b ON a.id < b.id
WHERE NOT EXISTS (SELECT 1 FROM sequence WHERE id = a.id + 1)
  AND NOT EXISTS (SELECT 1 FROM sequence WHERE id = b.id - 1 AND b.id > a.id + 1)
GROUP BY a.id;

-- Simpler gap detection:
SELECT s.id + 1 AS missing_id
FROM sequence s
WHERE NOT EXISTS (SELECT 1 FROM sequence WHERE id = s.id + 1)
  AND s.id < (SELECT MAX(id) FROM sequence);
```

---

## Common SQL Interview Tips

1. **Clarify the problem** — Ask about edge cases: NULLs, duplicates, ties, empty tables
2. **Start simple** — Write a basic version first, then optimize
3. **Use aliases** — Makes queries readable; especially for self-joins
4. **Think about NULLs** — They often trip up NOT IN, comparisons, and aggregations
5. **Window functions** — Know ROW_NUMBER, RANK, LAG/LEAD for ranking and time-series problems
6. **CTEs for readability** — Break complex queries into named steps
7. **EXISTS vs IN** — EXISTS handles NULLs safely and can be more efficient
8. **Test with examples** — Trace through your query with sample data mentally
9. **Know the execution order** — FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
10. **Aggregate + Window combo** — `SUM(col) OVER (PARTITION BY ...)` doesn't need GROUP BY and keeps all rows
