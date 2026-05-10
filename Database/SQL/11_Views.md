# Views

## Table of Contents
1. [What is a View?](#1-what-is-a-view)
2. [CREATE VIEW](#2-create-view)
3. [Querying Views](#3-querying-views)
4. [Updatable Views](#4-updatable-views)
5. [WITH CHECK OPTION](#5-with-check-option)
6. [Altering and Dropping Views](#6-altering-and-dropping-views)
7. [Materialized Views](#7-materialized-views)
8. [View Use Cases and Patterns](#8-view-use-cases-and-patterns)

---

## 1. What is a View?

A view is a stored SELECT query that acts like a virtual table. It doesn't store data (unless materialized); it re-executes the underlying query each time it's accessed.

### Benefits
- **Abstraction**: Hide complex query logic
- **Security**: Expose only certain rows/columns to users
- **Reusability**: Define once, use in many queries
- **Simplification**: Give complex joins a simple name
- **Backward compatibility**: Shield applications from schema changes

### Limitations
- **Performance**: Re-executed every time (unless materialized)
- **No indexes**: Cannot index a regular view (only materialized)
- **Write restrictions**: Not all views are updatable

---

## 2. CREATE VIEW

### Basic Syntax
```sql
CREATE VIEW view_name AS
SELECT ...;
```

### Simple View
```sql
CREATE VIEW active_employees AS
SELECT id, first_name, last_name, email, dept_id, salary
FROM employees
WHERE is_active = TRUE;
```

### View with JOIN
```sql
CREATE VIEW employee_details AS
SELECT
    e.id,
    e.first_name,
    e.last_name,
    e.salary,
    d.name   AS department,
    d.location,
    m.first_name || ' ' || m.last_name AS manager_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id
LEFT JOIN employees   m ON e.mgr_id  = m.id
WHERE e.is_active = TRUE;
```

### View with Aggregation
```sql
CREATE VIEW dept_salary_summary AS
SELECT
    d.id         AS dept_id,
    d.name       AS department,
    COUNT(e.id)  AS employee_count,
    AVG(e.salary) AS avg_salary,
    MIN(e.salary) AS min_salary,
    MAX(e.salary) AS max_salary,
    SUM(e.salary) AS total_payroll
FROM departments d
LEFT JOIN employees e ON e.dept_id = d.id
GROUP BY d.id, d.name;
```

### View with Window Functions
```sql
CREATE VIEW employee_rankings AS
SELECT
    id,
    first_name,
    last_name,
    dept_id,
    salary,
    RANK()       OVER (PARTITION BY dept_id ORDER BY salary DESC) AS dept_rank,
    DENSE_RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS dept_dense_rank,
    salary * 100.0 / SUM(salary) OVER (PARTITION BY dept_id)     AS pct_of_dept_payroll
FROM employees
WHERE is_active = TRUE;
```

### View with Subquery
```sql
CREATE VIEW above_avg_earners AS
SELECT id, first_name, last_name, dept_id, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```

### CREATE OR REPLACE VIEW
```sql
-- Update the view definition without dropping it
CREATE OR REPLACE VIEW active_employees AS
SELECT id, first_name, last_name, email, dept_id, salary, hire_date
FROM employees
WHERE is_active = TRUE;
```

### View with Column Aliases
```sql
-- Name the view's columns explicitly
CREATE VIEW emp_summary (emp_id, full_name, annual_salary, dept) AS
SELECT
    id,
    first_name || ' ' || last_name,
    salary * 12,
    dept_id
FROM employees;
```

---

## 3. Querying Views

Views behave like tables in SELECT statements.

```sql
-- Query a view like a table
SELECT * FROM active_employees;

SELECT first_name, last_name, salary
FROM active_employees
WHERE salary > 70000
ORDER BY salary DESC;

-- Join views together
SELECT ed.first_name, ed.department, ds.avg_salary
FROM employee_details ed
JOIN dept_salary_summary ds ON ed.dept_id = ds.dept_id;

-- Join view with table
SELECT ae.first_name, ae.salary, d.location
FROM active_employees ae
JOIN departments d ON ae.dept_id = d.id;

-- Use view in subquery
SELECT * FROM (
    SELECT * FROM employee_rankings WHERE dept_rank <= 3
) top_earners
ORDER BY dept_id, dept_rank;

-- Use view in CTE
WITH top_dept AS (
    SELECT dept_id FROM dept_salary_summary
    ORDER BY total_payroll DESC LIMIT 1
)
SELECT ae.* FROM active_employees ae
JOIN top_dept t ON ae.dept_id = t.dept_id;
```

---

## 4. Updatable Views

Views are updatable when they meet certain criteria. DML through the view affects the underlying base table.

### Conditions for Updatable Views
1. Based on a single table (no JOINs)
2. No GROUP BY or HAVING
3. No aggregate functions (SUM, COUNT, etc.)
4. No DISTINCT
5. No subqueries in the WHERE clause (usually)
6. All NOT NULL columns without DEFAULT must be included

```sql
-- This view is updatable
CREATE VIEW active_employees AS
SELECT id, first_name, last_name, email, salary
FROM employees
WHERE is_active = TRUE;

-- INSERT through view
INSERT INTO active_employees (first_name, last_name, email, salary)
VALUES ('Alice', 'Smith', 'alice@co.com', 75000);
-- Note: is_active defaults to TRUE (or whatever the table default is)

-- UPDATE through view
UPDATE active_employees
SET salary = salary * 1.10
WHERE id = 1;
-- Updates the underlying employees table

-- DELETE through view
DELETE FROM active_employees WHERE id = 5;
-- Deletes from the underlying employees table
```

### Non-Updatable Views
```sql
-- This view is NOT updatable (has JOIN)
CREATE VIEW employee_details AS
SELECT e.id, e.name, d.name AS dept
FROM employees e
JOIN departments d ON e.dept_id = d.id;

-- Attempting to update will fail:
UPDATE employee_details SET dept = 'HR' WHERE id = 1;  -- ERROR
```

---

## 5. WITH CHECK OPTION

Ensures that DML operations through the view satisfy the view's WHERE condition.

```sql
CREATE VIEW high_salary_employees AS
SELECT id, first_name, salary
FROM employees
WHERE salary > 80000
WITH CHECK OPTION;

-- This INSERT is REJECTED (violates WHERE salary > 80000)
INSERT INTO high_salary_employees (first_name, salary) VALUES ('Bob', 50000);
-- ERROR: new row violates check option for view "high_salary_employees"

-- This INSERT is ALLOWED (salary > 80000)
INSERT INTO high_salary_employees (first_name, salary) VALUES ('Alice', 90000);

-- This UPDATE is REJECTED (would make row invisible to the view)
UPDATE high_salary_employees SET salary = 70000 WHERE id = 1;
-- ERROR: new row violates check option

-- This UPDATE is ALLOWED
UPDATE high_salary_employees SET salary = 95000 WHERE id = 1;
```

### LOCAL vs CASCADED CHECK OPTION
```sql
-- Create a base view
CREATE VIEW dept_10_employees AS
SELECT id, name, dept_id, salary
FROM employees WHERE dept_id = 10
WITH CHECK OPTION;

-- Create a view on top of another view
CREATE VIEW dept_10_high_earners AS
SELECT * FROM dept_10_employees WHERE salary > 80000
WITH LOCAL CHECK OPTION;   -- Only checks this view's condition (salary > 80000)

-- With CASCADED (default): checks all views in the chain
CREATE VIEW dept_10_high_earners AS
SELECT * FROM dept_10_employees WHERE salary > 80000
WITH CASCADED CHECK OPTION;  -- Checks both: dept_id=10 AND salary > 80000
```

---

## 6. Altering and Dropping Views

### Alter View
```sql
-- Replace the view definition
CREATE OR REPLACE VIEW active_employees AS
SELECT id, first_name, last_name, email, salary, hire_date
FROM employees
WHERE is_active = TRUE;

-- SQL Server
ALTER VIEW active_employees AS
SELECT id, first_name, last_name, email, salary
FROM employees
WHERE is_active = TRUE;
```

### Rename View
```sql
-- PostgreSQL
ALTER VIEW active_employees RENAME TO current_employees;

-- MySQL
RENAME TABLE active_employees TO current_employees;
```

### Drop View
```sql
DROP VIEW view_name;
DROP VIEW IF EXISTS view_name;

-- Drop multiple views
DROP VIEW IF EXISTS view1, view2, view3;

-- PostgreSQL: drop with dependencies
DROP VIEW employee_details CASCADE;  -- Also drops views that depend on this one
DROP VIEW employee_details RESTRICT; -- Fail if other objects depend on it (default)
```

### View Information
```sql
-- MySQL: list views
SELECT TABLE_NAME, VIEW_DEFINITION
FROM information_schema.VIEWS
WHERE TABLE_SCHEMA = 'mydb';

SHOW FULL TABLES WHERE TABLE_TYPE = 'VIEW';

-- PostgreSQL
SELECT viewname, definition FROM pg_views WHERE schemaname = 'public';
\dv   -- psql command

-- SQL Server
SELECT name, definition
FROM sys.objects o
JOIN sys.sql_modules m ON o.object_id = m.object_id
WHERE o.type = 'V';
```

---

## 7. Materialized Views

Materialized views store the result of the query physically. They must be refreshed to reflect base table changes.

### PostgreSQL Materialized Views
```sql
-- Create
CREATE MATERIALIZED VIEW dept_monthly_sales AS
SELECT
    d.name AS department,
    DATE_TRUNC('month', o.order_date) AS month,
    COUNT(o.id) AS order_count,
    SUM(o.total) AS revenue
FROM departments d
JOIN employees e ON e.dept_id = d.id
JOIN orders o ON o.employee_id = e.id
GROUP BY d.name, DATE_TRUNC('month', o.order_date)
WITH DATA;   -- Populate immediately
-- Without WITH DATA: CREATE MATERIALIZED VIEW ... WITH NO DATA;

-- Query (reads from stored data — fast)
SELECT * FROM dept_monthly_sales WHERE month = '2024-01-01';

-- Refresh (re-run the query and store new results)
REFRESH MATERIALIZED VIEW dept_monthly_sales;

-- Refresh without locking reads
REFRESH MATERIALIZED VIEW CONCURRENTLY dept_monthly_sales;
-- Note: requires a UNIQUE index on the materialized view

-- Create index on materialized view
CREATE UNIQUE INDEX idx_dept_monthly ON dept_monthly_sales (department, month);

-- Drop
DROP MATERIALIZED VIEW dept_monthly_sales;
DROP MATERIALIZED VIEW IF EXISTS dept_monthly_sales;
```

### MySQL: No Native Materialized Views
```sql
-- Workaround 1: Manual refresh table
CREATE TABLE mat_dept_summary AS
SELECT dept_id, COUNT(*) AS cnt, AVG(salary) AS avg_sal
FROM employees GROUP BY dept_id;

-- Refresh (scheduled or triggered)
TRUNCATE TABLE mat_dept_summary;
INSERT INTO mat_dept_summary
SELECT dept_id, COUNT(*), AVG(salary) FROM employees GROUP BY dept_id;

-- Workaround 2: Use a scheduled event (MySQL Event Scheduler)
CREATE EVENT refresh_dept_summary
ON SCHEDULE EVERY 1 HOUR
DO
BEGIN
    TRUNCATE TABLE mat_dept_summary;
    INSERT INTO mat_dept_summary SELECT dept_id, COUNT(*), AVG(salary) FROM employees GROUP BY dept_id;
END;
```

### SQL Server Indexed Views (equivalent to materialized views)
```sql
-- Create view with SCHEMABINDING (required for indexed views)
CREATE VIEW dept_summary
WITH SCHEMABINDING AS
SELECT
    dept_id,
    COUNT_BIG(*) AS cnt,    -- Must use COUNT_BIG (not COUNT)
    SUM(salary) AS total_salary
FROM dbo.employees  -- Must use schema.table
GROUP BY dept_id;

-- Create a clustered index to materialize it
CREATE UNIQUE CLUSTERED INDEX idx_dept_summary ON dept_summary (dept_id);

-- Now SQL Server automatically maintains and queries this view
SELECT * FROM dept_summary WHERE dept_id = 10;  -- Uses materialized data
```

### Regular vs Materialized View Comparison

| Feature | Regular View | Materialized View |
|---------|-------------|-------------------|
| Storage | Query only | Data stored |
| Speed | Slow (re-executes query) | Fast (pre-computed) |
| Freshness | Always current | May be stale |
| Indexes | No | Yes |
| Refresh | Automatic | Manual or scheduled |
| Use case | Simple abstraction | Complex aggregations, reporting |

---

## 8. View Use Cases and Patterns

### Security: Expose Subset of Data
```sql
-- Public view: hide sensitive columns
CREATE VIEW public_employees AS
SELECT id, first_name, last_name, dept_id, job_title
FROM employees;
-- salary, SSN, performance_notes hidden

-- Row-level security: show only own department
CREATE VIEW my_department_employees AS
SELECT e.* FROM employees e
JOIN employees me ON me.id = CURRENT_USER_ID()  -- app-specific function
WHERE e.dept_id = me.dept_id;
```

### Backward Compatibility
```sql
-- After renaming/restructuring a table, keep old view name working
-- Original table: customers (id, customer_name, customer_email)
-- New table: users (id, name, email, created_at)

CREATE VIEW customers AS
SELECT id, name AS customer_name, email AS customer_email FROM users;
-- Old queries using "customers" table still work
```

### Reporting Views
```sql
CREATE VIEW monthly_revenue_report AS
SELECT
    DATE_FORMAT(o.order_date, '%Y-%m') AS month,
    p.category,
    COUNT(DISTINCT o.id) AS order_count,
    COUNT(oi.id)         AS item_count,
    SUM(oi.quantity * oi.unit_price) AS gross_revenue,
    SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS net_revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
JOIN products p ON oi.product_id = p.id
GROUP BY DATE_FORMAT(o.order_date, '%Y-%m'), p.category;
```

### Audit / History View
```sql
CREATE VIEW recent_changes AS
SELECT
    table_name,
    record_id,
    action,
    old_values,
    new_values,
    changed_by,
    changed_at
FROM audit_log
WHERE changed_at >= NOW() - INTERVAL '30 DAY'
ORDER BY changed_at DESC;
```

---

## Views Quick Reference

```sql
-- Create
CREATE VIEW name AS SELECT ...;
CREATE OR REPLACE VIEW name AS SELECT ...;
CREATE VIEW name (col1, col2) AS SELECT ...;  -- With column names

-- With check
CREATE VIEW name AS SELECT ... WHERE cond WITH CHECK OPTION;
CREATE VIEW name AS SELECT ... WHERE cond WITH LOCAL CHECK OPTION;

-- Alter / rename
ALTER VIEW name AS SELECT ...;           -- SQL Server
CREATE OR REPLACE VIEW name AS ...;     -- MySQL / PostgreSQL rename via re-create
ALTER VIEW name RENAME TO new_name;     -- PostgreSQL

-- Drop
DROP VIEW name;
DROP VIEW IF EXISTS name;
DROP VIEW name CASCADE;                 -- PostgreSQL

-- Materialized (PostgreSQL)
CREATE MATERIALIZED VIEW name AS SELECT ... WITH DATA;
REFRESH MATERIALIZED VIEW name;
REFRESH MATERIALIZED VIEW CONCURRENTLY name;
DROP MATERIALIZED VIEW name;

-- Query
SELECT * FROM view_name;
SELECT * FROM view_name WHERE col = val;
INSERT INTO updatable_view (...) VALUES (...);
UPDATE updatable_view SET col = val WHERE ...;
DELETE FROM updatable_view WHERE ...;
```
