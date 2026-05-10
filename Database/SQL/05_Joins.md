# SQL Joins

## Table of Contents
1. [What is a Join?](#1-what-is-a-join)
2. [INNER JOIN](#2-inner-join)
3. [LEFT JOIN (LEFT OUTER JOIN)](#3-left-join-left-outer-join)
4. [RIGHT JOIN (RIGHT OUTER JOIN)](#4-right-join-right-outer-join)
5. [FULL OUTER JOIN](#5-full-outer-join)
6. [CROSS JOIN](#6-cross-join)
7. [SELF JOIN](#7-self-join)
8. [NATURAL JOIN](#8-natural-join)
9. [Multiple Joins](#9-multiple-joins)
10. [JOIN with Conditions](#10-join-with-conditions)
11. [Non-Equi Joins](#11-non-equi-joins)
12. [LATERAL JOIN](#12-lateral-join)
13. [Join Performance Tips](#13-join-performance-tips)

---

## Setup: Sample Tables

```sql
-- departments
+----+-------------+----------+
| id | name        | location |
+----+-------------+----------+
| 10 | Engineering | NY       |
| 20 | Marketing   | Chicago  |
| 30 | Finance     | NY       |
| 40 | HR          | Boston   |
+----+-------------+----------+

-- employees
+----+----------+---------+--------+---------+
| id | name     | dept_id | salary | mgr_id  |
+----+----------+---------+--------+---------+
|  1 | Alice    |    10   | 90000  |    NULL |
|  2 | Bob      |    10   | 75000  |       1 |
|  3 | Carol    |    20   | 65000  |    NULL |
|  4 | Dave     |    30   | 80000  |    NULL |
|  5 | Eve      |  NULL   | 70000  |       1 |
+----+----------+---------+--------+---------+
```

---

## 1. What is a Join?

A JOIN combines rows from two or more tables based on a related column. The relationship is typically defined by primary key → foreign key pairs.

### Join Types Visual

```
employees             departments
    |                      |
    |--- INNER JOIN --------|  (only matching rows)
    |                      |
    |--- LEFT JOIN ---------|  (all from left + matching from right)
    |                      |
    |--- RIGHT JOIN --------|  (matching from left + all from right)
    |                      |
    |--- FULL OUTER JOIN ---|  (all from both sides)
    |                      |
    |--- CROSS JOIN --------|  (every combination: cartesian product)
```

---

## 2. INNER JOIN

Returns only rows where the join condition is satisfied in **both** tables.

### Syntax
```sql
SELECT columns
FROM table1
INNER JOIN table2 ON table1.column = table2.column;

-- INNER is optional (JOIN defaults to INNER JOIN)
SELECT columns
FROM table1
JOIN table2 ON table1.column = table2.column;
```

### Example
```sql
SELECT e.id, e.name, e.salary, d.name AS department
FROM employees e
INNER JOIN departments d ON e.dept_id = d.id;

-- Result: (Eve excluded — NULL dept_id; HR excluded — no employees)
+----+-------+--------+-------------+
| id | name  | salary | department  |
+----+-------+--------+-------------+
|  1 | Alice | 90000  | Engineering |
|  2 | Bob   | 75000  | Engineering |
|  3 | Carol | 65000  | Marketing   |
|  4 | Dave  | 80000  | Finance     |
+----+-------+--------+-------------+
```

### INNER JOIN with Multiple Conditions
```sql
SELECT e.name, d.name AS dept, p.title AS project
FROM employees e
JOIN departments d ON e.dept_id = d.id
JOIN assignments a ON a.employee_id = e.id AND a.is_lead = TRUE
JOIN projects p ON a.project_id = p.id;
```

---

## 3. LEFT JOIN (LEFT OUTER JOIN)

Returns **all rows from the left table** and matching rows from the right table. Non-matching right-side columns are NULL.

### Syntax
```sql
SELECT columns
FROM table1
LEFT JOIN table2 ON table1.column = table2.column;

-- LEFT OUTER JOIN is the same as LEFT JOIN
LEFT OUTER JOIN table2 ON ...
```

### Example
```sql
SELECT e.id, e.name, e.salary, d.name AS department
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id;

-- Result: All employees, including Eve (NULL dept_id → NULL department)
+----+-------+--------+-------------+
| id | name  | salary | department  |
+----+-------+--------+-------------+
|  1 | Alice | 90000  | Engineering |
|  2 | Bob   | 75000  | Engineering |
|  3 | Carol | 65000  | Marketing   |
|  4 | Dave  | 80000  | Finance     |
|  5 | Eve   | 70000  | NULL        |
+----+-------+--------+-------------+
```

### Find Unmatched Rows (Anti-Join Pattern)
```sql
-- Employees with NO department
SELECT e.id, e.name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id
WHERE d.id IS NULL;

-- Result:
+----+------+
| id | name |
+----+------+
|  5 | Eve  |
+----+------+
```

### Common Use Case
```sql
-- All customers and their total order amount (0 if no orders)
SELECT
    c.id,
    c.name,
    COALESCE(SUM(o.total), 0) AS total_spent
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.id
GROUP BY c.id, c.name;
```

---

## 4. RIGHT JOIN (RIGHT OUTER JOIN)

Returns matching rows from left + **all rows from the right table**. Non-matching left-side columns are NULL.

```sql
SELECT e.id, e.name, d.name AS department
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.id;

-- Result: All departments, including HR (no employees)
+------+-------+-------------+
| id   | name  | department  |
+------+-------+-------------+
|    1 | Alice | Engineering |
|    2 | Bob   | Engineering |
|    3 | Carol | Marketing   |
|    4 | Dave  | Finance     |
| NULL | NULL  | HR          |
+------+-------+-------------+
```

### Note
RIGHT JOIN is rarely used. Prefer LEFT JOIN by swapping table order:
```sql
-- These are equivalent:
SELECT * FROM a RIGHT JOIN b ON a.id = b.a_id;
SELECT * FROM b LEFT  JOIN a ON a.id = b.a_id;
```

---

## 5. FULL OUTER JOIN

Returns **all rows from both tables**. Non-matching columns from either side are NULL.

```sql
SELECT e.id, e.name, d.id AS dept_id, d.name AS department
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.id;

-- Result: All employees AND all departments
+------+-------+---------+-------------+
| id   | name  | dept_id | department  |
+------+-------+---------+-------------+
|    1 | Alice |      10 | Engineering |
|    2 | Bob   |      10 | Engineering |
|    3 | Carol |      20 | Marketing   |
|    4 | Dave  |      30 | Finance     |
|    5 | Eve   |    NULL | NULL        |
| NULL | NULL  |      40 | HR          |
+------+-------+---------+-------------+
```

### MySQL Workaround (no FULL OUTER JOIN support)
```sql
-- Emulate FULL OUTER JOIN with UNION of LEFT and RIGHT
SELECT e.name, d.name AS department
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id

UNION

SELECT e.name, d.name AS department
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.id;
```

### Find All Unmatched Rows
```sql
SELECT e.name, d.name AS department
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.id
WHERE e.id IS NULL OR d.id IS NULL;
```

---

## 6. CROSS JOIN

Returns the **Cartesian product** — every combination of rows from both tables.

```sql
SELECT e.name, d.name AS department
FROM employees e
CROSS JOIN departments d;

-- With 5 employees and 4 departments: 5 × 4 = 20 rows
```

### Implicit CROSS JOIN (old syntax)
```sql
-- Comma-separated tables without ON clause = CROSS JOIN
SELECT e.name, d.name
FROM employees e, departments d;  -- Avoid this style
```

### Practical Uses
```sql
-- Generate all size/color combinations for a product
SELECT s.size, c.color
FROM sizes s
CROSS JOIN colors c;

-- Generate a date range table
SELECT DATE_ADD('2024-01-01', INTERVAL n DAY) AS date
FROM (
    SELECT 0 UNION SELECT 1 UNION SELECT 2 -- ... up to 365
) AS nums(n);

-- PostgreSQL: generate_series
SELECT CURRENT_DATE + s AS date
FROM generate_series(0, 364) AS s;
```

---

## 7. SELF JOIN

A table joined to itself. Used for hierarchical data, comparing rows within the same table.

### Manager Hierarchy
```sql
-- List employees with their manager's name
SELECT
    e.name AS employee,
    m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.mgr_id = m.id;

-- Result:
+---------+---------+
| employee| manager |
+---------+---------+
| Alice   | NULL    |
| Bob     | Alice   |
| Carol   | NULL    |
| Dave    | NULL    |
| Eve     | Alice   |
+---------+---------+
```

### Compare Rows
```sql
-- Find pairs of employees in same department with different salaries
SELECT
    e1.name AS emp1,
    e2.name AS emp2,
    e1.salary,
    e2.salary,
    e1.dept_id
FROM employees e1
JOIN employees e2
    ON e1.dept_id = e2.dept_id
   AND e1.id < e2.id;  -- Avoid duplicates and self-pairing
```

### Find Duplicate Records
```sql
SELECT a.id, a.email
FROM customers a
JOIN customers b ON a.email = b.email AND a.id > b.id;
```

---

## 8. NATURAL JOIN

Automatically joins on all columns with the same name. Generally avoided due to fragility.

```sql
-- Joins on all common column names (here: dept_id? only if named identically)
SELECT * FROM employees NATURAL JOIN departments;

-- Danger: if a new column with the same name is added to either table,
-- the join condition changes silently
```

### USING clause (safer alternative to NATURAL JOIN)
```sql
-- Join on a specific shared column name
SELECT e.name, d.name AS dept
FROM employees e
JOIN departments d USING (dept_id);

-- Equivalent to:
FROM employees e JOIN departments d ON e.dept_id = d.dept_id
-- But USING avoids duplicating the column in SELECT *
```

---

## 9. Multiple Joins

```sql
-- Three-table join
SELECT
    e.name      AS employee,
    d.name      AS department,
    p.title     AS project,
    r.role_name AS role
FROM employees e
JOIN departments  d ON e.dept_id      = d.id
JOIN assignments  a ON a.employee_id  = e.id
JOIN projects     p ON a.project_id   = p.id
JOIN roles        r ON a.role_id      = r.id
WHERE p.status = 'active';

-- Mix of join types
SELECT
    c.name                          AS customer,
    o.id                            AS order_id,
    COALESCE(p.name, 'No product')  AS product
FROM customers c
LEFT JOIN orders  o ON o.customer_id = c.id
LEFT JOIN order_items oi ON oi.order_id = o.id
LEFT JOIN products p ON oi.product_id  = p.id
WHERE c.is_active = TRUE;
```

---

## 10. JOIN with Conditions

### ON vs WHERE for Outer Joins
```sql
-- Filtering in ON clause: applied BEFORE the outer join
-- (keeps all left-side rows regardless)
SELECT e.name, d.name AS dept
FROM employees e
LEFT JOIN departments d
    ON e.dept_id = d.id
   AND d.location = 'NY';         -- Only join NY departments; others show NULL

-- Filtering in WHERE clause: applied AFTER the outer join
-- (converts LEFT JOIN to INNER JOIN for the filtered rows)
SELECT e.name, d.name AS dept
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id
WHERE d.location = 'NY';          -- Removes rows where department is NULL or not NY
```

### Joining on Multiple Columns
```sql
SELECT *
FROM order_items oi
JOIN product_prices pp
    ON oi.product_id = pp.product_id
   AND oi.order_date BETWEEN pp.valid_from AND pp.valid_to;
```

### JOIN on Inequality
```sql
-- Salary bands
SELECT e.name, e.salary, b.band_name
FROM employees e
JOIN salary_bands b
    ON e.salary BETWEEN b.min_salary AND b.max_salary;
```

---

## 11. Non-Equi Joins

Joins using operators other than `=`.

```sql
-- Range join: match to salary band
SELECT e.name, sb.band
FROM employees e
JOIN salary_bands sb
    ON e.salary >= sb.min_salary
   AND e.salary <  sb.max_salary;

-- Self-join: find employees earning more than their manager
SELECT e.name AS employee, e.salary, m.name AS manager, m.salary AS mgr_salary
FROM employees e
JOIN employees m ON e.mgr_id = m.id
WHERE e.salary > m.salary;

-- Join with LIKE
SELECT e.name, r.role_description
FROM employees e
JOIN roles r ON e.job_title LIKE CONCAT('%', r.keyword, '%');
```

---

## 12. LATERAL JOIN

A LATERAL join allows the right-side subquery to reference columns from the left-side table. Like a correlated subquery that returns multiple rows/columns.

### PostgreSQL LATERAL
```sql
-- For each department, get the top 3 highest-paid employees
SELECT d.name AS dept, top_emp.name, top_emp.salary
FROM departments d
JOIN LATERAL (
    SELECT name, salary
    FROM employees
    WHERE dept_id = d.id        -- References d from outer query
    ORDER BY salary DESC
    LIMIT 3
) AS top_emp ON TRUE;
```

### MySQL (8.0+) LATERAL
```sql
SELECT d.name, top_e.name, top_e.salary
FROM departments d
JOIN LATERAL (
    SELECT name, salary
    FROM employees
    WHERE dept_id = d.id
    ORDER BY salary DESC
    LIMIT 3
) top_e ON TRUE;
```

### Cross Apply / Outer Apply (SQL Server — equivalent of LATERAL)
```sql
-- CROSS APPLY = INNER JOIN LATERAL
SELECT d.name, e.name, e.salary
FROM departments d
CROSS APPLY (
    SELECT TOP 3 name, salary
    FROM employees
    WHERE dept_id = d.id
    ORDER BY salary DESC
) e;

-- OUTER APPLY = LEFT JOIN LATERAL
SELECT d.name, e.name, e.salary
FROM departments d
OUTER APPLY (
    SELECT TOP 3 name, salary
    FROM employees
    WHERE dept_id = d.id
    ORDER BY salary DESC
) e;
```

---

## 13. Join Performance Tips

### Use Indexes on Join Columns
```sql
-- Always index foreign keys and join columns
CREATE INDEX idx_employees_dept_id ON employees (dept_id);
CREATE INDEX idx_orders_customer_id ON orders (customer_id);
```

### Filter Early (Push Predicates)
```sql
-- Good: filter before joining (reduces rows to join)
SELECT e.name, d.name
FROM (SELECT * FROM employees WHERE salary > 50000) e
JOIN departments d ON e.dept_id = d.id;

-- Also works with WHERE — optimizer usually handles this
SELECT e.name, d.name
FROM employees e
JOIN departments d ON e.dept_id = d.id
WHERE e.salary > 50000;
```

### Avoid Functions on Joined Columns
```sql
-- Bad: index on dept_id cannot be used
ON UPPER(e.dept_code) = UPPER(d.dept_code)

-- Good: normalize data so no function needed
ON e.dept_code = d.dept_code
```

### Choose Join Type Carefully
```sql
-- INNER JOIN: fastest (fewest rows)
-- LEFT JOIN: slightly more work (must preserve unmatched rows)
-- CROSS JOIN: most expensive (avoid unless intended)
-- FULL OUTER JOIN: expensive (use only when needed)
```

---

## Join Quick Reference

```sql
-- INNER JOIN: rows matching in both tables
FROM a JOIN b ON a.id = b.a_id

-- LEFT JOIN: all from a, matching from b
FROM a LEFT JOIN b ON a.id = b.a_id

-- RIGHT JOIN: matching from a, all from b
FROM a RIGHT JOIN b ON a.id = b.a_id

-- FULL OUTER JOIN: all from both
FROM a FULL OUTER JOIN b ON a.id = b.a_id

-- CROSS JOIN: Cartesian product
FROM a CROSS JOIN b

-- SELF JOIN: table joined to itself
FROM employees e1 JOIN employees e2 ON e1.mgr_id = e2.id

-- Anti-join (unmatched LEFT):
FROM a LEFT JOIN b ON a.id = b.a_id WHERE b.id IS NULL

-- USING (shared column name):
FROM a JOIN b USING (shared_col)

-- LATERAL (right side sees left side's columns):
FROM a JOIN LATERAL (SELECT ... WHERE x = a.col) sub ON TRUE
```
