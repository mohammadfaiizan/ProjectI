# Aggregations and Grouping

## Table of Contents
1. [Aggregate Functions](#1-aggregate-functions)
2. [GROUP BY](#2-group-by)
3. [HAVING](#3-having)
4. [GROUP BY Extensions](#4-group-by-extensions)
5. [Combining Aggregations](#5-combining-aggregations)
6. [Conditional Aggregation](#6-conditional-aggregation)
7. [String Aggregation](#7-string-aggregation)
8. [Statistical Functions](#8-statistical-functions)

---

## 1. Aggregate Functions

Aggregate functions operate on a set of rows and return a single value.

### COUNT

```sql
-- Count all rows (including NULLs)
SELECT COUNT(*) FROM employees;

-- Count non-NULL values in a column
SELECT COUNT(salary) FROM employees;

-- Count distinct values
SELECT COUNT(DISTINCT dept_id) FROM employees;

-- Count with condition (CASE trick)
SELECT COUNT(CASE WHEN salary > 80000 THEN 1 END) AS high_earners
FROM employees;

-- Count per group
SELECT dept_id, COUNT(*) AS headcount
FROM employees
GROUP BY dept_id;
```

### SUM

```sql
SELECT SUM(salary) AS total_payroll FROM employees;

-- SUM with condition
SELECT SUM(CASE WHEN is_active THEN salary ELSE 0 END) AS active_payroll
FROM employees;

-- SUM per group
SELECT dept_id, SUM(salary) AS dept_payroll
FROM employees
GROUP BY dept_id;

-- Cumulative / running sum → use Window Functions (see file 08)
```

### AVG

```sql
SELECT AVG(salary) AS avg_salary FROM employees;

-- AVG ignores NULLs
SELECT AVG(bonus) FROM employees;   -- Only counts rows with non-NULL bonus

-- Round average
SELECT ROUND(AVG(salary), 2) AS avg_salary FROM employees;

-- Average per group
SELECT dept_id, AVG(salary) AS avg_dept_salary
FROM employees
GROUP BY dept_id;

-- Weighted average
SELECT
    SUM(price * quantity) / SUM(quantity) AS weighted_avg_price
FROM order_items;
```

### MIN and MAX

```sql
SELECT MIN(salary) AS lowest, MAX(salary) AS highest FROM employees;

SELECT
    dept_id,
    MIN(salary) AS min_salary,
    MAX(salary) AS max_salary,
    MAX(salary) - MIN(salary) AS salary_range
FROM employees
GROUP BY dept_id;

-- MIN/MAX on dates
SELECT MIN(hire_date) AS first_hire, MAX(hire_date) AS latest_hire
FROM employees;

-- MIN/MAX on strings (alphabetical)
SELECT MIN(last_name) AS first_alpha, MAX(last_name) AS last_alpha
FROM employees;
```

### NULL Behavior in Aggregates

```sql
-- All aggregates (except COUNT(*)) ignore NULLs
SELECT
    COUNT(*)       AS total_rows,    -- Includes NULL rows
    COUNT(bonus)   AS has_bonus,     -- Excludes NULLs
    SUM(bonus)     AS total_bonus,   -- NULLs excluded from sum
    AVG(bonus)     AS avg_bonus,     -- Average of non-NULL values only
    MIN(bonus)     AS min_bonus,
    MAX(bonus)     AS max_bonus
FROM employees;

-- Replace NULL before aggregating
SELECT AVG(COALESCE(bonus, 0)) AS avg_bonus_including_zeros
FROM employees;
```

---

## 2. GROUP BY

GROUP BY splits rows into groups and applies aggregate functions to each group.

### Basic GROUP BY

```sql
SELECT dept_id, COUNT(*) AS count, AVG(salary) AS avg_salary
FROM employees
GROUP BY dept_id;

-- Group by multiple columns
SELECT dept_id, job_title, COUNT(*) AS count, AVG(salary) AS avg_salary
FROM employees
GROUP BY dept_id, job_title;

-- Group by expression
SELECT
    YEAR(hire_date) AS hire_year,
    COUNT(*) AS hires
FROM employees
GROUP BY YEAR(hire_date)
ORDER BY hire_year;

-- Group by alias (PostgreSQL allows this)
SELECT YEAR(hire_date) AS hire_year, COUNT(*) AS hires
FROM employees
GROUP BY hire_year;  -- PostgreSQL
```

### Rules for GROUP BY

```sql
-- Every column in SELECT that is NOT an aggregate function
-- MUST appear in GROUP BY

-- WRONG: name not in GROUP BY
SELECT dept_id, name, COUNT(*)
FROM employees
GROUP BY dept_id;  -- ERROR

-- CORRECT:
SELECT dept_id, COUNT(*)
FROM employees
GROUP BY dept_id;

-- CORRECT (all non-aggregate cols in GROUP BY):
SELECT dept_id, job_title, COUNT(*), SUM(salary)
FROM employees
GROUP BY dept_id, job_title;

-- Exception: MySQL's non-standard GROUP BY allows non-grouped columns
-- (returns arbitrary value from the group — avoid this)
```

### GROUP BY with ORDER BY

```sql
SELECT dept_id, COUNT(*) AS cnt, AVG(salary) AS avg_sal
FROM employees
GROUP BY dept_id
ORDER BY avg_sal DESC;   -- Order by aggregate result
```

### GROUP BY with JOIN

```sql
SELECT
    d.name AS department,
    COUNT(e.id) AS headcount,
    AVG(e.salary) AS avg_salary,
    SUM(e.salary) AS total_payroll
FROM departments d
LEFT JOIN employees e ON e.dept_id = d.id
GROUP BY d.id, d.name
ORDER BY total_payroll DESC;
```

---

## 3. HAVING

HAVING filters groups (after GROUP BY). WHERE filters rows (before GROUP BY).

### Basic HAVING

```sql
-- Departments with more than 5 employees
SELECT dept_id, COUNT(*) AS headcount
FROM employees
GROUP BY dept_id
HAVING COUNT(*) > 5;

-- Departments where average salary > 70000
SELECT dept_id, AVG(salary) AS avg_sal
FROM employees
GROUP BY dept_id
HAVING AVG(salary) > 70000;
```

### HAVING vs WHERE

```sql
-- WHERE filters BEFORE grouping
-- HAVING filters AFTER grouping

-- Find departments with avg salary > 70000, excluding interns
SELECT dept_id, AVG(salary) AS avg_sal
FROM employees
WHERE job_title != 'Intern'          -- Filter rows first
GROUP BY dept_id
HAVING AVG(salary) > 70000;          -- Then filter groups

-- Wrong: using WHERE on aggregate
SELECT dept_id, AVG(salary) AS avg_sal
FROM employees
WHERE AVG(salary) > 70000            -- ERROR: can't use aggregate in WHERE
GROUP BY dept_id;
```

### HAVING with Multiple Conditions

```sql
SELECT
    dept_id,
    COUNT(*) AS cnt,
    AVG(salary) AS avg_sal
FROM employees
GROUP BY dept_id
HAVING COUNT(*) >= 3
   AND AVG(salary) BETWEEN 60000 AND 100000;
```

### HAVING without GROUP BY

```sql
-- Applies to the entire table as one group
SELECT AVG(salary) AS avg_sal
FROM employees
HAVING AVG(salary) > 70000;
-- Returns one row if avg > 70000, else empty
```

---

## 4. GROUP BY Extensions

### ROLLUP

Generates subtotals and a grand total.

```sql
-- MySQL / PostgreSQL / SQL Server
SELECT
    dept_id,
    job_title,
    SUM(salary) AS total
FROM employees
GROUP BY ROLLUP (dept_id, job_title);

-- Result:
-- (10, Engineer,  250000)  <- group subtotal
-- (10, Manager,   90000)   <- group subtotal
-- (10, NULL,     340000)   <- dept_id=10 subtotal
-- (20, Analyst,  130000)
-- (20, NULL,     130000)   <- dept_id=20 subtotal
-- (NULL, NULL,   470000)   <- Grand total

-- MySQL syntax
GROUP BY dept_id, job_title WITH ROLLUP;
```

### CUBE

Generates all possible subtotals for all combinations.

```sql
-- PostgreSQL / SQL Server
SELECT
    dept_id,
    job_title,
    location,
    SUM(salary) AS total
FROM employees
GROUP BY CUBE (dept_id, job_title, location);
-- Produces 2^3 = 8 grouping combinations

-- MySQL: no CUBE natively — use UNION manually
```

### GROUPING SETS

Explicitly specify which grouping combinations to compute.

```sql
-- PostgreSQL / SQL Server
SELECT
    dept_id,
    job_title,
    SUM(salary) AS total
FROM employees
GROUP BY GROUPING SETS (
    (dept_id, job_title),   -- Both columns
    (dept_id),              -- Dept subtotal
    (job_title),            -- Job subtotal
    ()                      -- Grand total
);
-- Equivalent to UNION of 4 GROUP BY queries
```

### GROUPING() Function

Identifies whether a column is aggregated (NULL due to ROLLUP/CUBE) or actually NULL.

```sql
SELECT
    CASE GROUPING(dept_id)   WHEN 1 THEN 'ALL DEPTS' ELSE CAST(dept_id AS VARCHAR) END AS dept,
    CASE GROUPING(job_title) WHEN 1 THEN 'ALL JOBS'  ELSE job_title                 END AS job,
    SUM(salary) AS total
FROM employees
GROUP BY ROLLUP (dept_id, job_title);
-- GROUPING() = 1 means the NULL is from rollup aggregation
-- GROUPING() = 0 means the NULL is actual data NULL
```

---

## 5. Combining Aggregations

### Multiple Aggregates in One Query

```sql
SELECT
    dept_id,
    COUNT(*)                  AS headcount,
    COUNT(DISTINCT job_title) AS unique_titles,
    MIN(salary)               AS min_sal,
    MAX(salary)               AS max_sal,
    AVG(salary)               AS avg_sal,
    SUM(salary)               AS total_sal,
    ROUND(STDDEV(salary), 2)  AS sal_std_dev
FROM employees
GROUP BY dept_id
ORDER BY total_sal DESC;
```

### Nested Aggregates (Aggregate of Aggregates)

SQL doesn't allow nesting aggregates directly. Use a subquery or CTE:

```sql
-- Find the department with the highest average salary
SELECT dept_id, avg_sal
FROM (
    SELECT dept_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_id
) dept_avgs
ORDER BY avg_sal DESC
LIMIT 1;

-- Or with a CTE:
WITH dept_avgs AS (
    SELECT dept_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_id
)
SELECT dept_id, avg_sal
FROM dept_avgs
WHERE avg_sal = (SELECT MAX(avg_sal) FROM dept_avgs);
```

### FILTER clause (PostgreSQL)

```sql
-- More readable than CASE inside aggregate
SELECT
    dept_id,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE salary > 80000) AS high_earners,
    AVG(salary) FILTER (WHERE is_active = TRUE) AS active_avg_salary
FROM employees
GROUP BY dept_id;
```

---

## 6. Conditional Aggregation

### SUM / COUNT with CASE

```sql
-- Pivot-style: count by status per department
SELECT
    dept_id,
    COUNT(*) AS total,
    SUM(CASE WHEN is_active = TRUE  THEN 1 ELSE 0 END) AS active,
    SUM(CASE WHEN is_active = FALSE THEN 1 ELSE 0 END) AS inactive,
    -- COUNT equivalent:
    COUNT(CASE WHEN is_active = TRUE THEN 1 END) AS active_count
FROM employees
GROUP BY dept_id;

-- Conditional SUM
SELECT
    dept_id,
    SUM(salary) AS total_payroll,
    SUM(CASE WHEN job_title = 'Manager' THEN salary ELSE 0 END) AS manager_payroll,
    SUM(CASE WHEN job_title != 'Manager' THEN salary ELSE 0 END) AS staff_payroll
FROM employees
GROUP BY dept_id;
```

### Boolean Aggregations (PostgreSQL)

```sql
SELECT
    dept_id,
    BOOL_AND(is_active) AS all_active,   -- TRUE if all rows are TRUE
    BOOL_OR(is_active)  AS any_active    -- TRUE if any row is TRUE
FROM employees
GROUP BY dept_id;
```

### Percentage Calculations

```sql
SELECT
    dept_id,
    COUNT(*) AS dept_count,
    ROUND(
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM employees),
        2
    ) AS pct_of_total
FROM employees
GROUP BY dept_id;

-- Using window function (no subquery needed)
SELECT
    dept_id,
    COUNT(*) AS dept_count,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (),
        2
    ) AS pct_of_total
FROM employees
GROUP BY dept_id;
```

---

## 7. String Aggregation

### GROUP_CONCAT (MySQL)

```sql
-- Comma-separated list of names per department
SELECT
    dept_id,
    GROUP_CONCAT(first_name ORDER BY first_name SEPARATOR ', ') AS names
FROM employees
GROUP BY dept_id;

-- With DISTINCT
SELECT
    dept_id,
    GROUP_CONCAT(DISTINCT job_title ORDER BY job_title SEPARATOR ' | ') AS roles
FROM employees
GROUP BY dept_id;

-- Limit length
SELECT
    dept_id,
    GROUP_CONCAT(first_name SEPARATOR ', ') AS names
FROM employees
GROUP BY dept_id;
-- Note: default max length is 1024; increase with:
SET SESSION group_concat_max_len = 1000000;
```

### STRING_AGG (PostgreSQL / SQL Server)

```sql
-- PostgreSQL
SELECT
    dept_id,
    STRING_AGG(first_name, ', ' ORDER BY first_name) AS names
FROM employees
GROUP BY dept_id;

-- With DISTINCT (PostgreSQL)
SELECT
    dept_id,
    STRING_AGG(DISTINCT job_title, ' | ' ORDER BY job_title) AS roles
FROM employees
GROUP BY dept_id;

-- SQL Server 2017+
SELECT
    dept_id,
    STRING_AGG(first_name, ', ') WITHIN GROUP (ORDER BY first_name) AS names
FROM employees
GROUP BY dept_id;
```

### ARRAY_AGG (PostgreSQL)

```sql
-- Aggregate into an array
SELECT
    dept_id,
    ARRAY_AGG(first_name ORDER BY first_name) AS name_array,
    ARRAY_AGG(salary) AS salary_array
FROM employees
GROUP BY dept_id;

-- JSON aggregation
SELECT
    dept_id,
    JSON_AGG(JSON_BUILD_OBJECT('name', first_name, 'salary', salary)) AS emp_json
FROM employees
GROUP BY dept_id;
```

---

## 8. Statistical Functions

```sql
-- Standard Deviation and Variance
SELECT
    dept_id,
    ROUND(AVG(salary), 2)              AS mean,
    ROUND(STDDEV(salary), 2)           AS stddev,       -- Population std dev
    ROUND(STDDEV_SAMP(salary), 2)      AS stddev_sample,-- Sample std dev
    ROUND(VARIANCE(salary), 2)         AS variance,     -- Population variance
    ROUND(VAR_SAMP(salary), 2)         AS var_sample    -- Sample variance
FROM employees
GROUP BY dept_id;

-- PostgreSQL
SELECT
    dept_id,
    STDDEV_POP(salary)  AS pop_stddev,
    STDDEV_SAMP(salary) AS sample_stddev,
    VAR_POP(salary)     AS pop_variance,
    VAR_SAMP(salary)    AS sample_variance,
    CORR(salary, bonus) AS salary_bonus_correlation
FROM employees
GROUP BY dept_id;

-- Median (no built-in in MySQL)
-- PostgreSQL:
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees;

-- Quartiles
SELECT
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY salary) AS q1,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY salary) AS median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY salary) AS q3
FROM employees;

-- MySQL median workaround
SELECT AVG(salary) AS median
FROM (
    SELECT salary
    FROM employees
    ORDER BY salary
    LIMIT 2 - (SELECT COUNT(*) FROM employees) % 2
    OFFSET (SELECT (COUNT(*) - 1) / 2 FROM employees)
) AS subquery;
```

---

## Aggregations Quick Reference

```sql
-- Basic aggregates
SELECT COUNT(*), COUNT(col), COUNT(DISTINCT col) FROM t;
SELECT SUM(col), AVG(col), MIN(col), MAX(col) FROM t;

-- Grouping
SELECT col, COUNT(*) FROM t GROUP BY col;
SELECT c1, c2, SUM(v) FROM t GROUP BY c1, c2;

-- Filtering groups
SELECT col, AVG(v) FROM t GROUP BY col HAVING AVG(v) > 100;

-- Conditional aggregate
SELECT SUM(CASE WHEN cond THEN val ELSE 0 END) FROM t GROUP BY col;
SELECT COUNT(*) FILTER (WHERE cond) FROM t GROUP BY col;  -- PostgreSQL

-- Rollup / Cube
GROUP BY ROLLUP(c1, c2)           -- Subtotals + grand total
GROUP BY CUBE(c1, c2)             -- All combinations
GROUP BY GROUPING SETS((c1,c2),(c1),())  -- Custom sets

-- String aggregation
GROUP_CONCAT(col ORDER BY col SEPARATOR ',')  -- MySQL
STRING_AGG(col, ',' ORDER BY col)             -- PostgreSQL / SQL Server

-- Statistical
STDDEV(col), VARIANCE(col)
PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col)  -- Median (PostgreSQL)
```
