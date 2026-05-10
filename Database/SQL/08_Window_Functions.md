# Window Functions

## Table of Contents
1. [What are Window Functions?](#1-what-are-window-functions)
2. [OVER Clause Syntax](#2-over-clause-syntax)
3. [Ranking Functions](#3-ranking-functions)
4. [Offset Functions](#4-offset-functions)
5. [Aggregate Window Functions](#5-aggregate-window-functions)
6. [Frame Specification](#6-frame-specification)
7. [Named Windows](#7-named-windows)
8. [Practical Patterns](#8-practical-patterns)

---

## 1. What are Window Functions?

Window functions perform calculations across a set of related rows (a "window") without collapsing them into a single output row — unlike GROUP BY.

### Key Difference from GROUP BY

```sql
-- GROUP BY: collapses rows
SELECT dept_id, AVG(salary) AS avg_sal
FROM employees
GROUP BY dept_id;
-- Returns one row per dept_id

-- Window Function: keeps all rows
SELECT
    name,
    dept_id,
    salary,
    AVG(salary) OVER (PARTITION BY dept_id) AS dept_avg
FROM employees;
-- Returns all rows, with dept_avg added to each
```

### Syntax Structure
```sql
function_name([args]) OVER (
    [PARTITION BY partition_expression, ...]
    [ORDER BY sort_expression [ASC|DESC], ...]
    [frame_clause]
)
```

### Components
- **PARTITION BY**: Divides rows into groups (like GROUP BY but doesn't collapse)
- **ORDER BY**: Defines the order within each partition (required for ranking/offset functions)
- **Frame**: Defines which rows relative to the current row are included

---

## 2. OVER Clause Syntax

```sql
-- Empty OVER(): entire result set is one window
SELECT name, salary, AVG(salary) OVER () AS company_avg FROM employees;

-- PARTITION BY: one window per group
SELECT name, dept_id, salary,
    AVG(salary) OVER (PARTITION BY dept_id) AS dept_avg
FROM employees;

-- ORDER BY: defines row ordering within each partition
SELECT name, dept_id, salary,
    RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS salary_rank
FROM employees;

-- PARTITION BY + ORDER BY + Frame
SELECT name, dept_id, salary,
    SUM(salary) OVER (PARTITION BY dept_id ORDER BY hire_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM employees;
```

---

## 3. Ranking Functions

### ROW_NUMBER()
Assigns unique sequential integers to rows. No ties.
```sql
SELECT
    name,
    dept_id,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS row_num,
    ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS dept_row_num
FROM employees;

-- Paginate using ROW_NUMBER (SQL Server / older approach)
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (ORDER BY name) AS rn
    FROM employees
) t
WHERE rn BETWEEN 11 AND 20;  -- Page 2 (10 per page)

-- Deduplicate: keep first occurrence per email
DELETE FROM employees
WHERE id NOT IN (
    SELECT MIN(id) FROM (
        SELECT id, ROW_NUMBER() OVER (PARTITION BY email ORDER BY id) AS rn
        FROM employees
    ) t
    WHERE rn = 1
);
```

### RANK()
Assigns ranks with gaps for ties.
```sql
SELECT
    name,
    salary,
    RANK() OVER (ORDER BY salary DESC) AS rnk
FROM employees;

-- With ties:
-- Alice   90000  → rank 1
-- Bob     80000  → rank 2
-- Carol   80000  → rank 2  (tie)
-- Dave    70000  → rank 4  (gap: no rank 3)
```

### DENSE_RANK()
Assigns ranks without gaps for ties.
```sql
SELECT
    name,
    salary,
    DENSE_RANK() OVER (ORDER BY salary DESC) AS dense_rnk
FROM employees;

-- With ties:
-- Alice   90000  → rank 1
-- Bob     80000  → rank 2
-- Carol   80000  → rank 2  (tie)
-- Dave    70000  → rank 3  (no gap)
```

### NTILE(n)
Divides rows into n roughly equal buckets (percentile buckets).
```sql
SELECT
    name,
    salary,
    NTILE(4) OVER (ORDER BY salary) AS quartile    -- Q1, Q2, Q3, Q4
FROM employees;

-- Top 10%:
SELECT * FROM (
    SELECT name, salary, NTILE(10) OVER (ORDER BY salary DESC) AS decile
    FROM employees
) t WHERE decile = 1;
```

### Ranking Functions Comparison

| Function | Ties | Gaps |
|----------|------|------|
| ROW_NUMBER | No ties (unique) | No gaps |
| RANK | Ties allowed | Gaps after ties |
| DENSE_RANK | Ties allowed | No gaps |
| NTILE | Buckets (approximation) | N/A |

---

## 4. Offset Functions

### LAG()
Access a value from a previous row.
```sql
LAG(column, offset, default) OVER (ORDER BY ...)

-- Default offset = 1 (immediately preceding row)
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_month_revenue,
    revenue - LAG(revenue) OVER (ORDER BY month) AS monthly_change,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY month))
        / NULLIF(LAG(revenue) OVER (ORDER BY month), 0) * 100, 2
    ) AS pct_change
FROM monthly_revenue;

-- LAG with offset and default
SELECT
    name,
    salary,
    LAG(salary, 2, 0) OVER (ORDER BY hire_date) AS salary_2_hires_ago
FROM employees;
```

### LEAD()
Access a value from a following row.
```sql
LEAD(column, offset, default) OVER (ORDER BY ...)

SELECT
    name,
    hire_date,
    LEAD(hire_date) OVER (ORDER BY hire_date) AS next_hire_date,
    LEAD(hire_date) OVER (ORDER BY hire_date) - hire_date AS days_until_next_hire
FROM employees;

-- Detect churn: customers with no future orders
SELECT
    customer_id,
    order_date,
    LEAD(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS next_order_date
FROM orders;
```

### FIRST_VALUE() and LAST_VALUE()
```sql
-- FIRST_VALUE: value from the first row in the window
SELECT
    name,
    dept_id,
    salary,
    FIRST_VALUE(name)   OVER (PARTITION BY dept_id ORDER BY salary DESC) AS top_earner,
    FIRST_VALUE(salary) OVER (PARTITION BY dept_id ORDER BY salary DESC) AS max_salary
FROM employees;

-- LAST_VALUE: value from the last row in the window
-- Note: default frame is RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
-- This means LAST_VALUE only sees rows up to current row, not end of partition!
-- Must specify full frame:
SELECT
    name,
    dept_id,
    salary,
    LAST_VALUE(name) OVER (
        PARTITION BY dept_id
        ORDER BY salary
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS min_earner_name
FROM employees;
```

### NTH_VALUE()
```sql
-- Value at a specific position in the window
SELECT
    name,
    dept_id,
    salary,
    NTH_VALUE(name, 2) OVER (
        PARTITION BY dept_id
        ORDER BY salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS second_highest_earner
FROM employees;
```

---

## 5. Aggregate Window Functions

All aggregate functions can be used as window functions with OVER().

### Running Totals (Cumulative Sum)
```sql
SELECT
    order_date,
    amount,
    SUM(amount) OVER (ORDER BY order_date) AS running_total
FROM orders;

-- Running total per customer
SELECT
    customer_id,
    order_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY customer_id
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_spend
FROM orders;
```

### Running Average
```sql
SELECT
    order_date,
    amount,
    AVG(amount) OVER (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7day_avg
FROM daily_sales;
```

### Running MIN / MAX
```sql
SELECT
    name,
    hire_date,
    salary,
    MAX(salary) OVER (ORDER BY hire_date) AS highest_salary_at_hire_time,
    MIN(salary) OVER (ORDER BY hire_date) AS lowest_salary_at_hire_time
FROM employees;
```

### COUNT Window
```sql
SELECT
    name,
    dept_id,
    COUNT(*) OVER (PARTITION BY dept_id) AS dept_size,
    COUNT(*) OVER () AS company_size
FROM employees;
```

### Percentage of Total
```sql
SELECT
    name,
    dept_id,
    salary,
    ROUND(salary * 100.0 / SUM(salary) OVER (PARTITION BY dept_id), 2) AS pct_of_dept,
    ROUND(salary * 100.0 / SUM(salary) OVER (), 2) AS pct_of_company
FROM employees;
```

---

## 6. Frame Specification

The frame defines which rows relative to the current row are included in the window.

### Frame Syntax
```sql
OVER (
    ORDER BY col
    {ROWS | RANGE | GROUPS}
    BETWEEN frame_start AND frame_end
)
```

### Frame Boundaries
```
UNBOUNDED PRECEDING  -- First row of the partition
n PRECEDING          -- n rows before current row
CURRENT ROW          -- Current row
n FOLLOWING          -- n rows after current row
UNBOUNDED FOLLOWING  -- Last row of the partition
```

### ROWS vs RANGE vs GROUPS

| Mode | Unit |
|------|------|
| `ROWS` | Physical rows (counts by position) |
| `RANGE` | Logical rows (groups rows with equal ORDER BY values) |
| `GROUPS` | Groups of peer rows |

```sql
-- ROWS: exact row count
SUM(amount) OVER (ORDER BY date ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
-- Includes current row + 3 rows before (exactly 4 rows)

-- RANGE: includes all rows with same ORDER BY value as current row
SUM(amount) OVER (ORDER BY date RANGE BETWEEN 3 PRECEDING AND CURRENT ROW)
-- Includes all rows within 3 days of current (logical range)
```

### Default Frames

```sql
-- When ORDER BY is specified, default frame is:
-- RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
-- (includes all rows up to and including current row's peers)

-- When ORDER BY is NOT specified, default frame is:
-- RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
-- (entire partition)
```

### Frame Examples

```sql
-- Running sum (cumulative): all rows from start to current
SUM(sal) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)

-- 3-row moving average (current + 2 preceding)
AVG(sal) OVER (ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)

-- 7-day moving average (3 before, current, 3 after)
AVG(sal) OVER (ORDER BY date ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING)

-- Full partition (same as no ORDER BY)
SUM(sal) OVER (PARTITION BY dept ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)

-- Trailing 30-day window (RANGE — date-based)
SUM(revenue) OVER (
    ORDER BY sale_date
    RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
)  -- PostgreSQL
```

---

## 7. Named Windows

Define a window once and reuse it.

```sql
SELECT
    name,
    dept_id,
    salary,
    RANK()    OVER w AS rnk,
    DENSE_RANK() OVER w AS dense_rnk,
    ROW_NUMBER() OVER w AS rn,
    AVG(salary) OVER w AS avg_sal
FROM employees
WINDOW w AS (PARTITION BY dept_id ORDER BY salary DESC);

-- Named window with frame
SELECT
    order_date,
    amount,
    SUM(amount) OVER w AS running_total,
    AVG(amount) OVER w AS running_avg
FROM orders
WINDOW w AS (ORDER BY order_date ROWS UNBOUNDED PRECEDING);
```

---

## 8. Practical Patterns

### Top N Per Group
```sql
-- Top 3 earners per department
SELECT name, dept_id, salary
FROM (
    SELECT
        name, dept_id, salary,
        ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rn
    FROM employees
) ranked
WHERE rn <= 3;

-- Same but keep ties for position 3 (use RANK instead of ROW_NUMBER)
SELECT name, dept_id, salary
FROM (
    SELECT
        name, dept_id, salary,
        RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk
    FROM employees
) ranked
WHERE rnk <= 3;
```

### Year-over-Year Comparison
```sql
SELECT
    year,
    revenue,
    LAG(revenue) OVER (ORDER BY year) AS prev_year,
    revenue - LAG(revenue) OVER (ORDER BY year) AS yoy_change,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY year))
        / LAG(revenue) OVER (ORDER BY year) * 100, 2
    ) AS yoy_pct
FROM annual_revenue;
```

### Moving Average
```sql
-- 7-day moving average
SELECT
    sale_date,
    daily_revenue,
    ROUND(AVG(daily_revenue) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2) AS moving_avg_7d
FROM daily_sales;
```

### Running Total with Reset
```sql
-- Cumulative sales resetting each month
SELECT
    sale_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY DATE_TRUNC('month', sale_date)
        ORDER BY sale_date
    ) AS monthly_running_total
FROM sales;
```

### Median Salary per Department
```sql
SELECT DISTINCT
    dept_id,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary)
        OVER (PARTITION BY dept_id) AS median_salary
FROM employees;
-- Note: PERCENTILE_CONT is an ordered-set aggregate (not window), use differently:
SELECT dept_id,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees GROUP BY dept_id;
```

### Session / Gap-and-Island Analysis
```sql
-- Identify consecutive active days (islands)
WITH flagged AS (
    SELECT
        user_id,
        activity_date,
        activity_date - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY activity_date) * INTERVAL '1 DAY' AS grp
    FROM user_activity
)
SELECT user_id, MIN(activity_date) AS session_start, MAX(activity_date) AS session_end
FROM flagged
GROUP BY user_id, grp
ORDER BY user_id, session_start;
```

### Deduplication — Keep Latest
```sql
-- Keep only the most recent record per customer
DELETE FROM customers
WHERE id NOT IN (
    SELECT id FROM (
        SELECT id,
            ROW_NUMBER() OVER (PARTITION BY email ORDER BY created_at DESC) AS rn
        FROM customers
    ) t
    WHERE rn = 1
);
```

### Cumulative Distribution
```sql
SELECT
    name,
    salary,
    CUME_DIST()  OVER (ORDER BY salary) AS cume_dist,   -- 0 to 1
    PERCENT_RANK() OVER (ORDER BY salary) AS pct_rank   -- 0 to 1
FROM employees;

-- CUME_DIST: fraction of rows with value <= current
-- PERCENT_RANK: (rank - 1) / (total_rows - 1)
```

---

## Window Functions Quick Reference

```sql
-- Ranking
ROW_NUMBER() OVER (PARTITION BY p ORDER BY o)  -- unique rank
RANK()       OVER (PARTITION BY p ORDER BY o)  -- rank with gaps
DENSE_RANK() OVER (PARTITION BY p ORDER BY o)  -- rank without gaps
NTILE(n)     OVER (PARTITION BY p ORDER BY o)  -- bucket 1..n
CUME_DIST()  OVER (PARTITION BY p ORDER BY o)  -- 0-1 cumulative
PERCENT_RANK() OVER (...)                       -- 0-1 relative rank

-- Offset
LAG(col, n, default)   OVER (ORDER BY o)  -- prev n-th row value
LEAD(col, n, default)  OVER (ORDER BY o)  -- next n-th row value
FIRST_VALUE(col)       OVER (ORDER BY o)  -- first row in window
LAST_VALUE(col)        OVER (ORDER BY o ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
NTH_VALUE(col, n)      OVER (ORDER BY o)  -- n-th row value

-- Aggregates as windows
SUM(col)   OVER (PARTITION BY p ORDER BY o ROWS UNBOUNDED PRECEDING)
AVG(col)   OVER (PARTITION BY p ORDER BY o ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
COUNT(col) OVER (PARTITION BY p)
MIN(col)   OVER (PARTITION BY p ORDER BY o)
MAX(col)   OVER (PARTITION BY p ORDER BY o)

-- Named window
WINDOW w AS (PARTITION BY p ORDER BY o)
func() OVER w
```
