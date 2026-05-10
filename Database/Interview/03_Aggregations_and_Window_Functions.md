# Aggregations and Window Functions — Interview Questions

> **Difficulty Mix:** Easy (Q1–Q7) · Medium (Q8–Q14) · Hard (Q15–Q20)

---

### Q1. What are aggregate functions in SQL? List the main ones.

**Answer:**
Aggregate functions operate on a **set of rows** and return a single summary value per group.

| Function | Description | NULL behavior |
|----------|-------------|---------------|
| `COUNT(*)` | Count all rows | Includes NULLs |
| `COUNT(col)` | Count non-NULL values | Ignores NULLs |
| `COUNT(DISTINCT col)` | Count unique non-NULL values | Ignores NULLs |
| `SUM(col)` | Sum of values | Ignores NULLs |
| `AVG(col)` | Average of values | Ignores NULLs |
| `MIN(col)` | Minimum value | Ignores NULLs |
| `MAX(col)` | Maximum value | Ignores NULLs |
| `STDDEV(col)` | Standard deviation | Ignores NULLs |
| `VARIANCE(col)` | Variance | Ignores NULLs |

```sql
SELECT
    COUNT(*)              AS total_rows,
    COUNT(bonus)          AS rows_with_bonus,    -- NULLs excluded
    COUNT(DISTINCT dept_id) AS unique_depts,
    AVG(salary)           AS avg_salary,         -- NULLs excluded from avg
    SUM(salary)           AS total_payroll
FROM employees;
```

---

### Q2. What is the difference between COUNT(*), COUNT(col), and COUNT(DISTINCT col)?

**Answer:**

```sql
-- Setup: 10 rows; 2 have NULL salary; 4 distinct dept_ids

SELECT
    COUNT(*)              AS c1,   -- 10  (all rows, including NULLs)
    COUNT(salary)         AS c2,   -- 8   (excludes 2 NULL salary rows)
    COUNT(DISTINCT dept_id) AS c3  -- 4   (unique non-NULL dept values)
FROM employees;
```

| Expression | Counts |
|-----------|--------|
| `COUNT(*)` | Every row in the group (NULLs included) |
| `COUNT(col)` | Rows where col is NOT NULL |
| `COUNT(DISTINCT col)` | Unique non-NULL values of col |

**Common mistake:** Using `COUNT(col)` when you want all rows — if col has NULLs, the result is wrong. Use `COUNT(*)` for total row count.

---

### Q3. How does NULL affect aggregate functions?

**Answer:**
All aggregate functions **except COUNT(*)** ignore NULL values.

```sql
-- Table: bonuses = {100, NULL, 200, NULL, 300}
SELECT
    COUNT(*)          AS rows,     -- 5 (counts NULL rows)
    COUNT(bonus)      AS non_null, -- 3 (ignores NULLs)
    SUM(bonus)        AS total,    -- 600 (ignores NULLs)
    AVG(bonus)        AS average   -- 200 (600/3, not 600/5!)
FROM bonuses;
```

**Trap:** `AVG(bonus)` divides by the **count of non-NULL values** (3), not total rows (5). This can misrepresent the true average if NULLs should be treated as 0.

```sql
-- To include NULLs as 0 in average:
SELECT AVG(COALESCE(bonus, 0)) AS avg_including_zeros FROM bonuses;
-- = 120 (600 / 5)
```

---

### Q4. What is GROUP BY? What rules must you follow?

**Answer:**
GROUP BY divides rows into groups and applies aggregate functions to each group.

**Rules:**
1. Every column in SELECT that is **not** an aggregate function **must** appear in GROUP BY
2. GROUP BY executes **before** SELECT — you cannot GROUP BY a SELECT alias (in most databases)
3. GROUP BY executes **after** WHERE — WHERE filters rows before grouping
4. ORDER BY can use aggregate results; HAVING filters groups

```sql
-- CORRECT: dept_id in GROUP BY matches SELECT
SELECT dept_id, COUNT(*) AS cnt, AVG(salary) AS avg_sal
FROM employees
WHERE is_active = TRUE
GROUP BY dept_id
HAVING COUNT(*) > 3
ORDER BY avg_sal DESC;

-- WRONG: name not in GROUP BY
SELECT dept_id, name, COUNT(*)      -- ERROR: 'name' must be in GROUP BY
FROM employees GROUP BY dept_id;
```

---

### Q5. What is HAVING and how does it differ from WHERE?

**Answer:**

| | WHERE | HAVING |
|--|-------|--------|
| Filters | Individual rows | Groups |
| Runs | Before GROUP BY | After GROUP BY |
| Aggregates | ✗ Not allowed | ✓ Allowed |

```sql
SELECT dept_id, AVG(salary) AS avg_sal, COUNT(*) AS cnt
FROM employees
WHERE is_active = TRUE          -- ① Filter rows first (no aggregates)
GROUP BY dept_id
HAVING AVG(salary) > 75000      -- ② Filter groups (aggregates allowed)
   AND COUNT(*) >= 5;
```

**Memory trick:** WHERE = "which rows to include before grouping", HAVING = "which groups to include after grouping".

---

### Q6. What is a window function and how does it differ from GROUP BY?

**Answer:**

| Feature | GROUP BY + Aggregate | Window Function |
|---------|---------------------|----------------|
| Collapses rows | ✓ Yes — one row per group | ✗ No — all rows preserved |
| Access to non-grouped cols | ✗ No | ✓ Yes |
| Syntax | `SELECT dept_id, AVG(salary) ... GROUP BY dept_id` | `AVG(salary) OVER (PARTITION BY dept_id)` |

```sql
-- GROUP BY: collapses to 1 row per dept
SELECT dept_id, AVG(salary) AS dept_avg
FROM employees GROUP BY dept_id;
-- 4 rows (one per dept)

-- Window function: all rows preserved with dept avg added
SELECT name, dept_id, salary,
    AVG(salary) OVER (PARTITION BY dept_id) AS dept_avg
FROM employees;
-- 50 rows — each employee's row now has their dept average
```

---

### Q7. What is PARTITION BY in window functions?

**Answer:**
`PARTITION BY` divides the rows into independent groups (windows). The window function is calculated separately for each partition.

```sql
SELECT
    name,
    dept_id,
    salary,
    -- Global ranking across all employees
    RANK() OVER (ORDER BY salary DESC) AS global_rank,
    -- Ranking within each department
    RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS dept_rank,
    -- Department average (recomputed per partition)
    AVG(salary) OVER (PARTITION BY dept_id) AS dept_avg,
    -- Company-wide average (no PARTITION BY = one big window)
    AVG(salary) OVER () AS company_avg
FROM employees;
```

`PARTITION BY` without `ORDER BY` treats the entire partition as a single group (like GROUP BY but without collapsing). `PARTITION BY` with `ORDER BY` creates an ordered partition where frame-based functions have meaning.

---

### Q8. Explain ROW_NUMBER, RANK, and DENSE_RANK with a concrete example.

**Answer:**

```sql
-- Salaries: 90000, 80000, 80000, 70000
SELECT name, salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS row_num,
    RANK()       OVER (ORDER BY salary DESC) AS rank_val,
    DENSE_RANK() OVER (ORDER BY salary DESC) AS dense_rank
FROM employees ORDER BY salary DESC;
```

| Name | Salary | ROW_NUMBER | RANK | DENSE_RANK |
|------|--------|-----------|------|------------|
| Alice | 90000 | 1 | 1 | 1 |
| Bob | 80000 | 2 | 2 | 2 |
| Carol | 80000 | 3 | 2 | 2 |
| Dave | 70000 | 4 | 4 | 3 |

- **ROW_NUMBER**: Always unique — no ties
- **RANK**: Ties share the same rank; next rank skips (gap: 3 is skipped)
- **DENSE_RANK**: Ties share rank; no gap in sequence (3 still used)

**Interview question:** "Top 3 earners per department" — use `ROW_NUMBER` if you want exactly 3, `RANK` or `DENSE_RANK` if ties should all be included.

---

### Q9. What are LAG and LEAD functions? Give a business use case.

**Answer:**
`LAG(col, n, default)` accesses the value **n rows before** the current row.
`LEAD(col, n, default)` accesses the value **n rows after** the current row.

```sql
SELECT
    sale_month,
    revenue,
    LAG(revenue)  OVER (ORDER BY sale_month) AS prev_month_rev,
    LEAD(revenue) OVER (ORDER BY sale_month) AS next_month_rev,

    -- Month-over-month growth
    revenue - LAG(revenue) OVER (ORDER BY sale_month) AS mom_change,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY sale_month))
        / NULLIF(LAG(revenue) OVER (ORDER BY sale_month), 0) * 100, 2
    ) AS mom_pct
FROM monthly_revenue
ORDER BY sale_month;
```

**Business use cases:**
- Month-over-month / year-over-year growth
- Detect churn: customers with no next order (`LEAD` returns NULL)
- Compute time between events (order → delivery)
- Identify gaps in sequences

---

### Q10. What is the frame specification in window functions? Explain ROWS vs RANGE.

**Answer:**
The frame defines **which rows relative to the current row** are included in the window computation.

```sql
OVER (
    ORDER BY col
    {ROWS | RANGE | GROUPS}
    BETWEEN frame_start AND frame_end
)
```

**Common frame boundaries:**
- `UNBOUNDED PRECEDING` — first row of partition
- `n PRECEDING` — n rows/range units before current
- `CURRENT ROW` — current row
- `n FOLLOWING` — n rows/range units after current
- `UNBOUNDED FOLLOWING` — last row of partition

**ROWS vs RANGE:**
- `ROWS` counts physical rows by position (exact)
- `RANGE` groups rows with equal ORDER BY values together (logical)

```sql
-- 7-day rolling average using ROWS (exact 7 physical rows)
AVG(revenue) OVER (ORDER BY sale_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)

-- Running total (all rows from start to current, inclusive of ties)
SUM(revenue) OVER (ORDER BY sale_date RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)

-- Full partition (same as no ORDER BY)
SUM(revenue) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
```

**Default frame when ORDER BY is present:** `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`

---

### Q11. How do you calculate a running total (cumulative sum) in SQL?

**Answer:**

```sql
-- Running total using window function
SELECT
    order_date,
    amount,
    SUM(amount) OVER (
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM orders;

-- Running total per customer (reset for each customer)
SELECT
    customer_id,
    order_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY customer_id
        ORDER BY order_date
        ROWS UNBOUNDED PRECEDING    -- shorthand for BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS customer_running_total
FROM orders;

-- Running total resetting each month
SELECT
    sale_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY DATE_TRUNC('month', sale_date)
        ORDER BY sale_date
    ) AS monthly_running_total
FROM sales;
```

---

### Q12. How would you compute the percentage of total for each group?

**Answer:**

```sql
-- Percentage of total company payroll per department
SELECT
    dept_id,
    SUM(salary) AS dept_total,
    SUM(SUM(salary)) OVER () AS company_total,    -- nested aggregate + window
    ROUND(
        SUM(salary) * 100.0 / SUM(SUM(salary)) OVER ()
    , 2) AS pct_of_total
FROM employees
GROUP BY dept_id;

-- Per-row percentage (salary as % of dept total)
SELECT
    name,
    dept_id,
    salary,
    ROUND(
        salary * 100.0 / SUM(salary) OVER (PARTITION BY dept_id)
    , 2) AS pct_of_dept
FROM employees;
```

**Key insight:** `SUM(SUM(salary)) OVER ()` is a window function applied to an aggregate — the inner SUM groups by dept_id, the outer SUM OVER () is the grand total of those grouped values.

---

### Q13. What is conditional aggregation? How is it different from a WHERE clause?

**Answer:**
Conditional aggregation computes multiple aggregates in **one pass** over the data using CASE inside an aggregate function.

```sql
-- Multiple groups in one query (no separate GROUP BYs needed)
SELECT
    YEAR(hire_date) AS year,
    COUNT(*) AS total_hires,
    COUNT(CASE WHEN dept_id = 10 THEN 1 END) AS eng_hires,
    COUNT(CASE WHEN dept_id = 20 THEN 1 END) AS mkt_hires,
    SUM(CASE WHEN salary > 80000 THEN salary ELSE 0 END) AS high_salary_payroll,
    AVG(CASE WHEN is_active THEN salary END) AS active_avg_salary
FROM employees
GROUP BY YEAR(hire_date);
```

**PostgreSQL FILTER clause** (cleaner syntax):
```sql
SELECT
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE dept_id = 10) AS eng_count,
    AVG(salary) FILTER (WHERE is_active = TRUE) AS active_avg
FROM employees;
```

**vs WHERE:** WHERE would require running separate queries for each condition. Conditional aggregation does it in one pass — far more efficient.

---

### Q14. Explain ROLLUP, CUBE, and GROUPING SETS.

**Answer:**
These are GROUP BY extensions that produce multiple levels of aggregation in one query.

**ROLLUP** — subtotals + grand total along one hierarchy:
```sql
SELECT dept_id, job_title, SUM(salary) AS total
FROM employees
GROUP BY ROLLUP (dept_id, job_title);
-- Produces: (dept, job) subtotals + (dept) subtotals + () grand total
```

**CUBE** — all possible combinations:
```sql
GROUP BY CUBE (dept_id, location, job_title)
-- 2^3 = 8 combinations: all groupings possible
```

**GROUPING SETS** — explicit control over which combinations:
```sql
GROUP BY GROUPING SETS (
    (dept_id, job_title),   -- both
    (dept_id),              -- dept only
    ()                      -- grand total only
);
```

**GROUPING() function** — tells you if a NULL is from rollup aggregation or actual NULL data:
```sql
SELECT
    CASE GROUPING(dept_id) WHEN 1 THEN 'ALL DEPTS' ELSE CAST(dept_id AS VARCHAR) END AS dept,
    SUM(salary)
FROM employees
GROUP BY ROLLUP (dept_id);
```

---

### Q15. Write a query to find the top 2 earners in each department using window functions.

**Answer:**
```sql
-- Method 1: ROW_NUMBER (exactly 2 per dept, no ties)
SELECT name, dept_id, salary
FROM (
    SELECT
        name, dept_id, salary,
        ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rn
    FROM employees
) ranked
WHERE rn <= 2;

-- Method 2: DENSE_RANK (include all tied employees at position 2)
SELECT name, dept_id, salary
FROM (
    SELECT
        name, dept_id, salary,
        DENSE_RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS dr
    FROM employees
) ranked
WHERE dr <= 2;

-- Method 3: RANK (same as dense_rank but with gaps)
SELECT name, dept_id, salary
FROM (
    SELECT name, dept_id, salary,
        RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk
    FROM employees
) t WHERE rnk <= 2;
```

**Which to choose?**
- Use `ROW_NUMBER` when you need exactly N rows per partition
- Use `DENSE_RANK` when tied employees should all qualify
- Use `RANK` when standard competition ranking is needed

---

### Q16. How would you compute a 3-month moving average of revenue?

**Answer:**
```sql
SELECT
    sale_month,
    revenue,
    AVG(revenue) OVER (
        ORDER BY sale_month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW    -- current + 2 prior = 3 months
    ) AS moving_avg_3m,

    -- Alternative: RANGE-based (date arithmetic, works for irregular dates)
    AVG(revenue) OVER (
        ORDER BY sale_month
        RANGE BETWEEN INTERVAL '2 months' PRECEDING AND CURRENT ROW
    ) AS range_based_avg
FROM monthly_revenue
ORDER BY sale_month;
```

**Difference:**
- `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW` — always exactly 3 physical rows
- `RANGE BETWEEN INTERVAL '2 months' PRECEDING AND CURRENT ROW` — all months within the 2-month date range

**Note:** For the first 2 months, the window is smaller (1 or 2 rows). If you need exactly 3 months, filter the result:
```sql
WHERE sale_month >= (SELECT MIN(sale_month) FROM monthly_revenue) + INTERVAL '2 months'
```

---

### Q17. What is NTILE and how is it used?

**Answer:**
`NTILE(n)` divides ordered rows into `n` approximately equal buckets (numbered 1 to n). Useful for quartile/decile analysis.

```sql
SELECT
    name,
    salary,
    NTILE(4) OVER (ORDER BY salary)  AS quartile,    -- Q1, Q2, Q3, Q4
    NTILE(10) OVER (ORDER BY salary) AS decile,      -- 1-10
    NTILE(100) OVER (ORDER BY salary) AS percentile  -- 1-100
FROM employees;
```

**Use case — find employees in the top 25%:**
```sql
SELECT name, salary
FROM (
    SELECT name, salary, NTILE(4) OVER (ORDER BY salary DESC) AS quartile
    FROM employees
) t
WHERE quartile = 1;
```

**Caveat:** If rows don't divide evenly, earlier buckets get one extra row. `NTILE(4)` on 10 rows → buckets of size 3, 3, 2, 2.

---

### Q18. What is CUME_DIST and PERCENT_RANK?

**Answer:**
Both return a value between 0 and 1 representing the relative position of each row.

```sql
SELECT
    name,
    salary,
    CUME_DIST()    OVER (ORDER BY salary) AS cum_dist,   -- Fraction of rows ≤ this value
    PERCENT_RANK() OVER (ORDER BY salary) AS pct_rank    -- (rank-1) / (n-1)
FROM employees;
```

| Function | Formula | Returns |
|---------|---------|---------|
| `CUME_DIST()` | rows_with_value_≤_current / total_rows | 0 < value ≤ 1 |
| `PERCENT_RANK()` | (rank - 1) / (total_rows - 1) | 0 ≤ value ≤ 1 |

```sql
-- Find employees in top 20% by salary
SELECT name, salary
FROM (
    SELECT name, salary, CUME_DIST() OVER (ORDER BY salary DESC) AS cd
    FROM employees
) t
WHERE cd <= 0.20;
```

---

### Q19. What is the difference between FIRST_VALUE and LAST_VALUE? What frame gotcha should you know?

**Answer:**
- `FIRST_VALUE(col)` — returns the value of `col` from the **first row** of the window
- `LAST_VALUE(col)` — returns the value of `col` from the **last row** of the window

**The LAST_VALUE gotcha:**
The default frame is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, meaning LAST_VALUE only sees rows up to and including the current row — NOT the end of the partition.

```sql
-- WRONG: returns current row's value, not last in partition
SELECT name, salary,
    LAST_VALUE(name) OVER (PARTITION BY dept_id ORDER BY salary DESC) AS last_name  -- wrong!
FROM employees;

-- CORRECT: extend the frame to UNBOUNDED FOLLOWING
SELECT name, salary,
    LAST_VALUE(name) OVER (
        PARTITION BY dept_id
        ORDER BY salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING  -- full partition
    ) AS lowest_earner_name
FROM employees;

-- FIRST_VALUE doesn't have this problem (first row is always in the default frame)
```

---

### Q20. Write a query to compute year-over-year growth and rank months by performance within each year.

**Answer:**
```sql
WITH monthly AS (
    SELECT
        YEAR(sale_date)  AS yr,
        MONTH(sale_date) AS mo,
        SUM(amount)      AS revenue
    FROM sales
    GROUP BY YEAR(sale_date), MONTH(sale_date)
),
with_growth AS (
    SELECT
        yr,
        mo,
        revenue,
        LAG(revenue) OVER (ORDER BY yr, mo) AS prev_year_revenue,
        ROUND(
            (revenue - LAG(revenue, 12) OVER (ORDER BY yr, mo))
            / NULLIF(LAG(revenue, 12) OVER (ORDER BY yr, mo), 0) * 100, 2
        ) AS yoy_growth_pct
    FROM monthly
)
SELECT
    yr,
    mo,
    revenue,
    yoy_growth_pct,
    RANK() OVER (PARTITION BY yr ORDER BY revenue DESC) AS rank_in_year,
    ROUND(revenue * 100.0 / SUM(revenue) OVER (PARTITION BY yr), 2) AS pct_of_year
FROM with_growth
ORDER BY yr, mo;
```

**Key techniques used:**
- `LAG(revenue, 12)` — look back exactly 12 months for year-over-year
- `RANK() OVER (PARTITION BY yr ORDER BY revenue DESC)` — rank within year
- `SUM(revenue) OVER (PARTITION BY yr)` — year total for percentage
- CTE to stage the computation in readable steps
