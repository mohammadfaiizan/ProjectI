# Practical Coding Challenges — Interview Questions

All problems use standard SQL (PostgreSQL syntax unless noted). Schema is defined at the start of each section. Solve before reading the solution.

---

## Schema Reference

```sql
-- Employees / Departments
CREATE TABLE departments (
    dept_id   INT PRIMARY KEY,
    dept_name VARCHAR(100) NOT NULL
);

CREATE TABLE employees (
    emp_id     INT PRIMARY KEY,
    name       VARCHAR(100) NOT NULL,
    dept_id    INT REFERENCES departments(dept_id),
    manager_id INT REFERENCES employees(emp_id),
    salary     NUMERIC(10,2),
    hire_date  DATE,
    title      VARCHAR(100)
);

-- Orders / Products / Customers
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    name        VARCHAR(100),
    country     VARCHAR(50),
    join_date   DATE
);

CREATE TABLE products (
    product_id  INT PRIMARY KEY,
    name        VARCHAR(100),
    category    VARCHAR(50),
    price       NUMERIC(10,2)
);

CREATE TABLE orders (
    order_id    INT PRIMARY KEY,
    customer_id INT REFERENCES customers,
    order_date  DATE,
    status      TEXT
);

CREATE TABLE order_items (
    order_id   INT REFERENCES orders,
    product_id INT REFERENCES products,
    quantity   INT,
    unit_price NUMERIC(10,2),
    PRIMARY KEY (order_id, product_id)
);

-- Sessions / Events
CREATE TABLE user_sessions (
    session_id  BIGINT PRIMARY KEY,
    user_id     INT,
    started_at  TIMESTAMPTZ,
    ended_at    TIMESTAMPTZ
);

CREATE TABLE page_events (
    event_id   BIGINT PRIMARY KEY,
    user_id    INT,
    event_type VARCHAR(50),   -- 'view', 'click', 'purchase'
    page       VARCHAR(100),
    created_at TIMESTAMPTZ
);
```

---

## Easy Challenges (Q1–Q6)

---

**Q1. List employees who earn more than the average salary in their own department.**

```sql
SELECT e.emp_id, e.name, e.salary, e.dept_id
FROM employees e
WHERE e.salary > (
    SELECT AVG(salary)
    FROM employees
    WHERE dept_id = e.dept_id
)
ORDER BY e.dept_id, e.salary DESC;
```

**Alternative with window function (single scan):**
```sql
SELECT emp_id, name, salary, dept_id
FROM (
    SELECT emp_id, name, salary, dept_id,
           AVG(salary) OVER (PARTITION BY dept_id) AS dept_avg
    FROM employees
) sub
WHERE salary > dept_avg
ORDER BY dept_id, salary DESC;
```

---

**Q2. Find all customers who have never placed an order.**

```sql
-- Method 1: LEFT JOIN + IS NULL (most common)
SELECT c.customer_id, c.name
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;

-- Method 2: NOT EXISTS
SELECT customer_id, name
FROM customers c
WHERE NOT EXISTS (
    SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id
);

-- Method 3: NOT IN (safe because customer_id is NOT NULL)
SELECT customer_id, name
FROM customers
WHERE customer_id NOT IN (SELECT customer_id FROM orders);
```

All three return the same result. NOT EXISTS is generally safest and most readable for large subqueries.

---

**Q3. For each department, return the name of the highest-paid employee.**

```sql
-- Method 1: Window function (handles ties correctly)
SELECT dept_id, emp_id, name, salary
FROM (
    SELECT dept_id, emp_id, name, salary,
           RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk
    FROM employees
) ranked
WHERE rnk = 1;

-- Method 2: Correlated subquery
SELECT e.dept_id, e.emp_id, e.name, e.salary
FROM employees e
WHERE e.salary = (
    SELECT MAX(salary) FROM employees WHERE dept_id = e.dept_id
);

-- Method 3: JOIN on max salary
SELECT e.dept_id, e.emp_id, e.name, e.salary
FROM employees e
INNER JOIN (
    SELECT dept_id, MAX(salary) AS max_sal
    FROM employees
    GROUP BY dept_id
) m ON e.dept_id = m.dept_id AND e.salary = m.max_sal;
```

---

**Q4. Calculate total revenue per product category, ordered by revenue descending.**

```sql
SELECT p.category,
       COUNT(DISTINCT oi.order_id)          AS order_count,
       SUM(oi.quantity)                      AS units_sold,
       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
GROUP BY p.category
ORDER BY total_revenue DESC;
```

---

**Q5. Show the month-over-month revenue change as an absolute value and percentage.**

```sql
WITH monthly AS (
    SELECT DATE_TRUNC('month', o.order_date) AS month,
           SUM(oi.quantity * oi.unit_price)   AS revenue
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY 1
),
with_lag AS (
    SELECT month,
           revenue,
           LAG(revenue) OVER (ORDER BY month) AS prev_revenue
    FROM monthly
)
SELECT month,
       ROUND(revenue, 2)                                             AS revenue,
       ROUND(revenue - prev_revenue, 2)                             AS change,
       ROUND((revenue - prev_revenue) / prev_revenue * 100, 1)     AS pct_change
FROM with_lag
ORDER BY month;
```

---

**Q6. Return the second-highest salary in the employees table. Handle ties (second-distinct value).**

```sql
-- Method 1: DENSE_RANK (cleanest)
SELECT salary
FROM (
    SELECT DISTINCT salary,
           DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employees
) r
WHERE rnk = 2;

-- Method 2: Subquery with MAX exclusion (works in all databases)
SELECT MAX(salary)
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- Method 3: OFFSET/LIMIT (simple but only works for Nth distinct)
SELECT DISTINCT salary
FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET 1;
-- Note: returns NULL if there is no second-distinct salary
```

---

## Medium Challenges (Q7–Q14)

---

**Q7. Find customers who placed orders in January 2024 but NOT in February 2024.**

```sql
-- Method 1: EXCEPT (cleanest)
SELECT DISTINCT customer_id
FROM orders
WHERE order_date >= '2024-01-01' AND order_date < '2024-02-01'

EXCEPT

SELECT DISTINCT customer_id
FROM orders
WHERE order_date >= '2024-02-01' AND order_date < '2024-03-01';

-- Method 2: NOT EXISTS
SELECT DISTINCT o1.customer_id
FROM orders o1
WHERE o1.order_date >= '2024-01-01' AND o1.order_date < '2024-02-01'
  AND NOT EXISTS (
      SELECT 1 FROM orders o2
      WHERE o2.customer_id = o1.customer_id
        AND o2.order_date >= '2024-02-01'
        AND o2.order_date < '2024-03-01'
  );
```

---

**Q8. For each customer, show the running total of their spending and their cumulative spend rank at each order.**

```sql
SELECT o.customer_id,
       c.name,
       o.order_date,
       SUM(oi.quantity * oi.unit_price) AS order_total,
       SUM(SUM(oi.quantity * oi.unit_price))
           OVER (PARTITION BY o.customer_id ORDER BY o.order_date
                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total,
       RANK() OVER (PARTITION BY o.customer_id
                    ORDER BY SUM(oi.quantity * oi.unit_price) DESC) AS order_rank
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN customers c    ON o.customer_id = c.customer_id
GROUP BY o.customer_id, c.name, o.order_id, o.order_date
ORDER BY o.customer_id, o.order_date;
```

Note: `SUM(SUM(...)) OVER (...)` — the inner SUM aggregates per order_id (GROUP BY), the outer SUM is a window function over those aggregated values.

---

**Q9. Write a query to detect and list duplicate rows in the customers table (same name and country, different customer_id). Show all duplicates.**

```sql
-- Show all rows that share (name, country) with at least one other row
WITH dupes AS (
    SELECT name, country,
           COUNT(*) OVER (PARTITION BY name, country) AS cnt,
           ROW_NUMBER() OVER (PARTITION BY name, country ORDER BY customer_id) AS rn
    FROM customers
)
SELECT customer_id, name, country
FROM customers
WHERE (name, country) IN (
    SELECT name, country FROM dupes WHERE cnt > 1
)
ORDER BY name, country, customer_id;

-- To keep only the first and delete the rest:
DELETE FROM customers
WHERE customer_id IN (
    SELECT customer_id FROM (
        SELECT customer_id,
               ROW_NUMBER() OVER (PARTITION BY name, country ORDER BY customer_id) AS rn
        FROM customers
    ) r
    WHERE rn > 1
);
```

---

**Q10. Pivot the order count per customer per status into columns: pending, completed, cancelled.**

```sql
-- Manual pivot (works on all databases)
SELECT customer_id,
       COUNT(*) FILTER (WHERE status = 'pending')   AS pending,
       COUNT(*) FILTER (WHERE status = 'completed') AS completed,
       COUNT(*) FILTER (WHERE status = 'cancelled') AS cancelled
FROM orders
GROUP BY customer_id
ORDER BY customer_id;

-- Equivalent without FILTER (MySQL compatible):
SELECT customer_id,
       SUM(CASE WHEN status = 'pending'   THEN 1 ELSE 0 END) AS pending,
       SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
       SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled
FROM orders
GROUP BY customer_id
ORDER BY customer_id;
```

---

**Q11. Find the top 3 best-selling products in each category by total units sold.**

```sql
WITH ranked AS (
    SELECT p.category,
           p.name AS product_name,
           SUM(oi.quantity) AS total_units,
           RANK() OVER (PARTITION BY p.category ORDER BY SUM(oi.quantity) DESC) AS rnk
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    GROUP BY p.category, p.product_id, p.name
)
SELECT category, product_name, total_units, rnk
FROM ranked
WHERE rnk <= 3
ORDER BY category, rnk;
```

Use `RANK()` to allow ties at position 3; use `ROW_NUMBER()` if you want exactly 3 rows per category regardless of ties; use `DENSE_RANK()` to allow more than 3 when there are ties.

---

**Q12. Write a recursive CTE to return all employees in a given manager's reporting chain (direct and indirect reports), including their depth level.**

```sql
WITH RECURSIVE hierarchy AS (
    -- Anchor: the manager themselves at depth 0
    SELECT emp_id, name, manager_id, title, 0 AS depth,
           ARRAY[emp_id] AS path
    FROM employees
    WHERE emp_id = :manager_id   -- parameter: starting manager

    UNION ALL

    -- Recursive: add each direct report of the current level
    SELECT e.emp_id, e.name, e.manager_id, e.title, h.depth + 1,
           h.path || e.emp_id
    FROM employees e
    INNER JOIN hierarchy h ON e.manager_id = h.emp_id
    WHERE NOT e.emp_id = ANY(h.path)  -- cycle guard
)
SELECT emp_id, name, title, depth,
       array_to_string(path, ' → ') AS reporting_path
FROM hierarchy
ORDER BY path;
```

---

**Q13. Find the longest consecutive streak of days on which a user placed at least one order.**

```sql
WITH daily_orders AS (
    -- Distinct days the user ordered
    SELECT DISTINCT customer_id,
                    order_date::date AS day
    FROM orders
),
with_groups AS (
    -- Gap-and-island: subtract row_number to create a group ID per streak
    SELECT customer_id,
           day,
           day - CAST(ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY day) AS INT) AS grp
    FROM daily_orders
),
streaks AS (
    SELECT customer_id,
           MIN(day) AS streak_start,
           MAX(day) AS streak_end,
           COUNT(*) AS streak_length
    FROM with_groups
    GROUP BY customer_id, grp
)
SELECT customer_id, streak_start, streak_end, streak_length
FROM (
    SELECT customer_id, streak_start, streak_end, streak_length,
           RANK() OVER (PARTITION BY customer_id ORDER BY streak_length DESC) AS rnk
    FROM streaks
) r
WHERE rnk = 1
ORDER BY streak_length DESC;
```

**Key insight:** If dates are consecutive, `date - row_number` stays constant within a streak. Any gap creates a new group ID.

---

**Q14. For each product, calculate the 3-month moving average of monthly revenue.**

```sql
WITH monthly_revenue AS (
    SELECT p.product_id,
           p.name,
           DATE_TRUNC('month', o.order_date) AS month,
           SUM(oi.quantity * oi.unit_price)   AS revenue
    FROM order_items oi
    JOIN orders   o ON oi.order_id   = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    GROUP BY p.product_id, p.name, DATE_TRUNC('month', o.order_date)
)
SELECT product_id, name, month,
       ROUND(revenue, 2) AS monthly_revenue,
       ROUND(AVG(revenue) OVER (
           PARTITION BY product_id
           ORDER BY month
           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
       ), 2) AS moving_avg_3m
FROM monthly_revenue
ORDER BY product_id, month;
```

`ROWS BETWEEN 2 PRECEDING AND CURRENT ROW` = current month + 2 previous months = 3-month window.

---

## Hard Challenges (Q15–Q20)

---

**Q15. Find all pairs of employees who share the same manager AND the same department, but are not the same person. Return each pair once.**

```sql
SELECT e1.emp_id AS emp1_id, e1.name AS emp1_name,
       e2.emp_id AS emp2_id, e2.name AS emp2_name,
       e1.manager_id, e1.dept_id
FROM employees e1
JOIN employees e2
  ON e1.manager_id = e2.manager_id
 AND e1.dept_id    = e2.dept_id
 AND e1.emp_id     < e2.emp_id   -- avoid (A,B) and (B,A) and (A,A)
WHERE e1.manager_id IS NOT NULL
ORDER BY e1.dept_id, e1.manager_id, e1.emp_id;
```

Using `e1.emp_id < e2.emp_id` ensures each pair appears exactly once.

---

**Q16. Implement relational division: find customers who have ordered ALL products in a given category.**

```sql
-- Find all customers who have ordered every product in the 'Electronics' category
-- Approach 1: Double NOT EXISTS (cleanest relational division)
SELECT c.customer_id, c.name
FROM customers c
WHERE NOT EXISTS (
    -- There is no Electronics product that this customer has NOT ordered
    SELECT 1
    FROM products p
    WHERE p.category = 'Electronics'
      AND NOT EXISTS (
          SELECT 1
          FROM orders o
          JOIN order_items oi ON o.order_id = oi.order_id
          WHERE o.customer_id = c.customer_id
            AND oi.product_id = p.product_id
      )
);

-- Approach 2: COUNT comparison (simpler to read)
SELECT o.customer_id
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p     ON oi.product_id = p.product_id
WHERE p.category = 'Electronics'
GROUP BY o.customer_id
HAVING COUNT(DISTINCT oi.product_id) = (
    SELECT COUNT(*) FROM products WHERE category = 'Electronics'
);
```

---

**Q17. Detect gaps in a sequential order_id series. Return the start and end of each gap.**

```sql
WITH ordered AS (
    SELECT order_id,
           LEAD(order_id) OVER (ORDER BY order_id) AS next_id
    FROM orders
),
gaps AS (
    SELECT order_id + 1 AS gap_start,
           next_id - 1  AS gap_end
    FROM ordered
    WHERE next_id - order_id > 1  -- there is at least one missing ID
)
SELECT gap_start, gap_end,
       gap_end - gap_start + 1 AS missing_count
FROM gaps
ORDER BY gap_start;
```

**Example output:**
```
gap_start | gap_end | missing_count
----------+---------+---------------
       5  |      7  |      3        (IDs 5, 6, 7 are missing)
      15  |     15  |      1        (ID 15 is missing)
```

---

**Q18. Write a query that returns each user's session data, including session duration and whether it overlaps with any other session for the same user.**

```sql
WITH sessions AS (
    SELECT session_id, user_id,
           started_at,
           ended_at,
           EXTRACT(EPOCH FROM (ended_at - started_at)) / 60 AS duration_minutes
    FROM user_sessions
),
overlap_check AS (
    SELECT s1.session_id,
           s1.user_id,
           s1.started_at,
           s1.ended_at,
           s1.duration_minutes,
           EXISTS (
               SELECT 1 FROM user_sessions s2
               WHERE s2.user_id = s1.user_id
                 AND s2.session_id <> s1.session_id
                 AND s2.started_at < s1.ended_at   -- s2 starts before s1 ends
                 AND s2.ended_at   > s1.started_at  -- s2 ends after s1 starts
           ) AS has_overlap
    FROM sessions s1
)
SELECT session_id, user_id,
       started_at, ended_at,
       ROUND(duration_minutes::numeric, 2) AS duration_min,
       has_overlap
FROM overlap_check
ORDER BY user_id, started_at;
```

---

**Q19. Given a table of stock prices (one row per ticker per day), compute the maximum drawdown for each ticker. Drawdown = peak-to-trough decline as a percentage.**

```sql
CREATE TABLE stock_prices (
    ticker  VARCHAR(10),
    price_date DATE,
    close_price NUMERIC(10,2),
    PRIMARY KEY (ticker, price_date)
);

WITH running_peak AS (
    SELECT ticker,
           price_date,
           close_price,
           MAX(close_price) OVER (
               PARTITION BY ticker
               ORDER BY price_date
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) AS peak_so_far
    FROM stock_prices
),
drawdowns AS (
    SELECT ticker,
           price_date,
           close_price,
           peak_so_far,
           ROUND((close_price - peak_so_far) / peak_so_far * 100, 2) AS drawdown_pct
    FROM running_peak
)
SELECT ticker,
       MIN(drawdown_pct) AS max_drawdown_pct,
       (SELECT price_date FROM drawdowns d2
        WHERE d2.ticker = d.ticker
          AND d2.drawdown_pct = MIN(d.drawdown_pct)
        LIMIT 1) AS trough_date
FROM drawdowns d
GROUP BY ticker
ORDER BY max_drawdown_pct;
```

**Explanation:** Running peak = highest close seen up to each date. Drawdown = (current - peak) / peak × 100 (negative number). Max drawdown = minimum drawdown (most negative).

---

**Q20. Design and write a query for a 30-day retention analysis: for each acquisition cohort (week users joined), what percentage of users returned to place an order in their first 30 days?**

```sql
WITH cohorts AS (
    -- Define cohort: the week each customer joined
    SELECT customer_id,
           DATE_TRUNC('week', join_date) AS cohort_week
    FROM customers
),
user_orders AS (
    -- All orders within 30 days of the customer's join date
    SELECT c.customer_id,
           co.cohort_week,
           MIN(o.order_date) AS first_order_date,
           c.join_date
    FROM customers c
    JOIN cohorts co ON c.customer_id = co.customer_id
    LEFT JOIN orders o
           ON o.customer_id = c.customer_id
          AND o.order_date BETWEEN c.join_date AND c.join_date + INTERVAL '30 days'
    GROUP BY c.customer_id, co.cohort_week, c.join_date
),
cohort_stats AS (
    SELECT cohort_week,
           COUNT(DISTINCT customer_id)                             AS cohort_size,
           COUNT(DISTINCT customer_id) FILTER (WHERE first_order_date IS NOT NULL) AS converted
    FROM user_orders
    GROUP BY cohort_week
)
SELECT cohort_week,
       cohort_size,
       converted,
       ROUND(converted::numeric / cohort_size * 100, 1) AS retention_pct_30d
FROM cohort_stats
ORDER BY cohort_week;
```

**Output example:**
```
cohort_week | cohort_size | converted | retention_pct_30d
------------+-------------+-----------+-------------------
2024-01-01  |     1240    |    847    |       68.3
2024-01-08  |     1105    |    712    |       64.4
2024-01-15  |      980    |    621    |       63.4
```

**Extension — day-by-day retention curve (cohort × day_number):**
```sql
WITH cohort_orders AS (
    SELECT c.customer_id,
           DATE_TRUNC('week', c.join_date) AS cohort_week,
           (o.order_date - c.join_date)    AS days_since_join
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.order_date >= c.join_date
      AND o.order_date <= c.join_date + INTERVAL '30 days'
)
SELECT cohort_week,
       days_since_join,
       COUNT(DISTINCT customer_id) AS active_users
FROM cohort_orders
GROUP BY cohort_week, days_since_join
ORDER BY cohort_week, days_since_join;
```

---

## Common Patterns Cheat Sheet

```sql
-- Nth highest value
SELECT salary FROM (
    SELECT DISTINCT salary,
           DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employees
) r WHERE rnk = :N;

-- Top-N per group
SELECT * FROM (
    SELECT *, RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk
    FROM employees
) r WHERE rnk <= :N;

-- Gap-and-island (consecutive streak)
SELECT grp, MIN(day) AS start, MAX(day) AS end, COUNT(*) AS length
FROM (
    SELECT day,
           day - ROW_NUMBER() OVER (ORDER BY day)::int AS grp
    FROM daily_table
) t GROUP BY grp;

-- Gaps in sequence
SELECT id + 1 AS gap_start, next_id - 1 AS gap_end
FROM (SELECT id, LEAD(id) OVER (ORDER BY id) AS next_id FROM t) x
WHERE next_id - id > 1;

-- Relational division (all X for all Y)
SELECT x FROM X
WHERE NOT EXISTS (
    SELECT y FROM Y
    WHERE NOT EXISTS (
        SELECT 1 FROM XY WHERE XY.x = X.x AND XY.y = Y.y
    )
);

-- Running total
SUM(value) OVER (PARTITION BY key ORDER BY date ROWS UNBOUNDED PRECEDING)

-- Moving average (3 periods)
AVG(value) OVER (PARTITION BY key ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)

-- YoY change
SELECT year, revenue,
       LAG(revenue) OVER (ORDER BY year) AS prev_year,
       revenue - LAG(revenue) OVER (ORDER BY year) AS yoy_change
FROM yearly_revenue;

-- Deduplication (keep first by some ordering)
DELETE FROM t WHERE id IN (
    SELECT id FROM (
        SELECT id, ROW_NUMBER() OVER (PARTITION BY key ORDER BY id) AS rn
        FROM t
    ) r WHERE rn > 1
);

-- Pivot (manual)
SELECT key,
       SUM(CASE WHEN category = 'A' THEN value END) AS A,
       SUM(CASE WHEN category = 'B' THEN value END) AS B
FROM t GROUP BY key;

-- Recursive hierarchy
WITH RECURSIVE hier AS (
    SELECT id, parent_id, name, 0 AS depth FROM t WHERE parent_id IS NULL
    UNION ALL
    SELECT t.id, t.parent_id, t.name, h.depth + 1
    FROM t JOIN hier h ON t.parent_id = h.id
)
SELECT * FROM hier ORDER BY depth, id;
```
