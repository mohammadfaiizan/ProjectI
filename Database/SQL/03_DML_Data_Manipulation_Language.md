# DML — Data Manipulation Language

## Table of Contents
1. [INSERT](#1-insert)
2. [UPDATE](#2-update)
3. [DELETE](#3-delete)
4. [MERGE (UPSERT)](#4-merge-upsert)
5. [RETURNING Clause](#5-returning-clause)
6. [DML with Subqueries](#6-dml-with-subqueries)

---

## 1. INSERT

### Insert Single Row
```sql
-- Explicit column list (recommended)
INSERT INTO employees (first_name, last_name, email, salary, dept_id)
VALUES ('Alice', 'Johnson', 'alice@company.com', 75000, 10);

-- Without column list (must match all columns in order)
INSERT INTO employees
VALUES (DEFAULT, 'Bob', 'Smith', 'bob@company.com', 80000, 20, CURRENT_DATE, TRUE);
```

### Insert Multiple Rows
```sql
-- Single INSERT with multiple value sets (efficient — one round trip)
INSERT INTO departments (name, location)
VALUES
    ('Engineering', 'New York'),
    ('Marketing',   'Chicago'),
    ('Finance',     'Los Angeles'),
    ('HR',          'Boston');
```

### Insert with Expressions
```sql
INSERT INTO orders (customer_id, total, created_at, order_number)
VALUES (
    42,
    199.99,
    NOW(),
    CONCAT('ORD-', YEAR(NOW()), '-', LPAD(FLOOR(RAND() * 10000), 4, '0'))
);
```

### INSERT ... SELECT (Insert from query)
```sql
-- Copy active employees to archive
INSERT INTO employees_archive (id, first_name, last_name, email, salary, dept_id)
SELECT id, first_name, last_name, email, salary, dept_id
FROM employees
WHERE is_active = FALSE;

-- Insert with transformation
INSERT INTO salary_history (employee_id, old_salary, change_date)
SELECT id, salary, CURRENT_DATE
FROM employees
WHERE dept_id = 10;
```

### INSERT OR IGNORE / INSERT IGNORE (Skip on conflict)
```sql
-- MySQL: skip rows that would cause a constraint violation
INSERT IGNORE INTO employees (id, email, first_name)
VALUES (1, 'alice@company.com', 'Alice');

-- PostgreSQL: skip on conflict
INSERT INTO employees (id, email, first_name)
VALUES (1, 'alice@company.com', 'Alice')
ON CONFLICT DO NOTHING;
```

### INSERT OR REPLACE (Replace on conflict)
```sql
-- MySQL: delete old row + insert new on primary key conflict
REPLACE INTO employees (id, first_name, last_name, email)
VALUES (1, 'Alice', 'Johnson', 'newalice@company.com');

-- SQLite
INSERT OR REPLACE INTO employees (id, first_name, email)
VALUES (1, 'Alice', 'newalice@company.com');
```

### INSERT ... ON DUPLICATE KEY UPDATE (MySQL UPSERT)
```sql
INSERT INTO employees (id, first_name, email, salary)
VALUES (1, 'Alice', 'alice@company.com', 90000)
ON DUPLICATE KEY UPDATE
    salary = VALUES(salary),
    email  = VALUES(email);

-- Using new VALUES() alias (MySQL 8.0.19+)
INSERT INTO employees (id, first_name, salary)
VALUES (1, 'Alice', 90000) AS new_val
ON DUPLICATE KEY UPDATE
    salary = new_val.salary;
```

### INSERT ... ON CONFLICT (PostgreSQL UPSERT)
```sql
-- On conflict do nothing
INSERT INTO employees (id, email) VALUES (1, 'alice@co.com')
ON CONFLICT (id) DO NOTHING;

-- On conflict update (upsert)
INSERT INTO employees (id, first_name, salary)
VALUES (1, 'Alice', 90000)
ON CONFLICT (id)
DO UPDATE SET
    first_name = EXCLUDED.first_name,
    salary     = EXCLUDED.salary,
    updated_at = NOW();

-- On conflict with condition
INSERT INTO employees (id, salary)
VALUES (1, 95000)
ON CONFLICT (id)
DO UPDATE SET salary = EXCLUDED.salary
WHERE EXCLUDED.salary > employees.salary;
```

### Bulk Insert Performance Tips
```sql
-- Wrap in transaction for speed
BEGIN;
INSERT INTO large_table VALUES (...);
INSERT INTO large_table VALUES (...);
-- ... many rows ...
COMMIT;

-- Disable indexes temporarily (MySQL, large bulk load)
ALTER TABLE large_table DISABLE KEYS;
-- bulk inserts...
ALTER TABLE large_table ENABLE KEYS;

-- MySQL LOAD DATA INFILE (fastest for CSV)
LOAD DATA INFILE '/path/to/data.csv'
INTO TABLE employees
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(first_name, last_name, email, salary);

-- PostgreSQL COPY (fastest)
COPY employees (first_name, last_name, email, salary)
FROM '/path/to/data.csv'
WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');
```

---

## 2. UPDATE

### Update Single Column
```sql
UPDATE employees
SET salary = 85000
WHERE id = 1;
```

### Update Multiple Columns
```sql
UPDATE employees
SET
    salary    = 90000,
    email     = 'alice.new@company.com',
    updated_at = NOW()
WHERE id = 1;
```

### Update with Expression
```sql
-- Give 10% raise to all Engineering employees
UPDATE employees
SET salary = salary * 1.10
WHERE dept_id = (SELECT id FROM departments WHERE name = 'Engineering');

-- Increment a counter
UPDATE page_views
SET view_count = view_count + 1
WHERE page_id = 42;
```

### Update All Rows (no WHERE)
```sql
-- Be careful — updates every row!
UPDATE employees
SET is_active = TRUE;

-- Always double-check with SELECT first:
SELECT * FROM employees;  -- Then run UPDATE
```

### UPDATE with JOIN (MySQL)
```sql
-- Give 15% raise to employees in Chicago department
UPDATE employees e
JOIN departments d ON e.dept_id = d.id
SET e.salary = e.salary * 1.15
WHERE d.location = 'Chicago';

-- Update from another table
UPDATE employees e
INNER JOIN salary_adjustments sa ON e.id = sa.employee_id
SET e.salary = e.salary + sa.adjustment
WHERE sa.effective_date <= CURRENT_DATE;
```

### UPDATE with Subquery
```sql
-- PostgreSQL / Standard SQL: UPDATE with FROM
UPDATE employees
SET salary = salary * 1.10
FROM departments d
WHERE employees.dept_id = d.id
  AND d.name = 'Engineering';

-- Using subquery in WHERE
UPDATE employees
SET salary = salary * 1.20
WHERE dept_id IN (
    SELECT id FROM departments WHERE budget > 1000000
);

-- Update to a value from another table
UPDATE employees e
SET salary = (
    SELECT AVG(salary)
    FROM employees e2
    WHERE e2.dept_id = e.dept_id
)
WHERE salary < (
    SELECT AVG(salary) * 0.8
    FROM employees e3
    WHERE e3.dept_id = e.dept_id
);
```

### Conditional UPDATE with CASE
```sql
UPDATE employees
SET salary = CASE
    WHEN dept_id = 10 THEN salary * 1.15
    WHEN dept_id = 20 THEN salary * 1.10
    WHEN dept_id = 30 THEN salary * 1.05
    ELSE salary * 1.03
END;

-- Update status based on date
UPDATE orders
SET status = CASE
    WHEN shipped_date IS NULL                             THEN 'pending'
    WHEN shipped_date <= CURRENT_DATE - INTERVAL '7 DAY' THEN 'delivered'
    ELSE 'in_transit'
END;
```

### Limiting Updates (MySQL)
```sql
-- Update only first 100 rows matching condition
UPDATE employees
SET bonus = 1000
WHERE dept_id = 10
LIMIT 100;
```

---

## 3. DELETE

### Delete Specific Rows
```sql
DELETE FROM employees
WHERE id = 5;

DELETE FROM employees
WHERE is_active = FALSE AND hire_date < '2020-01-01';
```

### Delete All Rows
```sql
-- Logged, transactional (slower but safe)
DELETE FROM employees;

-- Better: use TRUNCATE for full table clear (not transactional)
TRUNCATE TABLE employees;
```

### DELETE with Subquery
```sql
-- Delete employees in departments with no budget
DELETE FROM employees
WHERE dept_id IN (
    SELECT id FROM departments WHERE budget = 0 OR budget IS NULL
);

-- Delete duplicate rows, keep one with lowest id
DELETE FROM employees
WHERE id NOT IN (
    SELECT MIN(id)
    FROM employees
    GROUP BY email
);
```

### DELETE with JOIN (MySQL)
```sql
-- Delete employees whose department was closed
DELETE e
FROM employees e
JOIN departments d ON e.dept_id = d.id
WHERE d.is_active = FALSE;

-- Multi-table delete
DELETE e, d
FROM employees e
JOIN departments d ON e.dept_id = d.id
WHERE d.name = 'Temp Department';
```

### DELETE with USING (PostgreSQL)
```sql
-- PostgreSQL: DELETE ... USING (like JOIN)
DELETE FROM employees
USING departments d
WHERE employees.dept_id = d.id
  AND d.is_active = FALSE;
```

### Soft Delete (Common Pattern)
```sql
-- Instead of deleting, mark as inactive
UPDATE employees
SET
    is_active  = FALSE,
    deleted_at = NOW(),
    deleted_by = CURRENT_USER
WHERE id = 5;

-- Query active records
SELECT * FROM employees WHERE is_active = TRUE;
-- Or: WHERE deleted_at IS NULL
```

### Limiting Deletes (MySQL)
```sql
-- Delete only first 10 matching rows
DELETE FROM log_entries
WHERE log_date < '2023-01-01'
LIMIT 10;

-- Delete in batches (useful for large tables to avoid locking)
DELETE FROM log_entries
WHERE log_date < '2023-01-01'
LIMIT 1000;
-- Run repeatedly until 0 rows affected
```

---

## 4. MERGE (UPSERT)

MERGE combines INSERT, UPDATE, and DELETE in one statement.

### Standard SQL MERGE (SQL Server / Oracle / PostgreSQL 15+)
```sql
MERGE INTO employees AS target
USING (
    SELECT id, first_name, last_name, salary
    FROM new_employee_data
) AS source
ON target.id = source.id

WHEN MATCHED AND source.salary <> target.salary THEN
    UPDATE SET
        salary     = source.salary,
        updated_at = NOW()

WHEN MATCHED AND source.salary IS NULL THEN
    DELETE

WHEN NOT MATCHED BY TARGET THEN
    INSERT (id, first_name, last_name, salary)
    VALUES (source.id, source.first_name, source.last_name, source.salary)

WHEN NOT MATCHED BY SOURCE THEN
    UPDATE SET is_active = FALSE;
```

### PostgreSQL INSERT ... ON CONFLICT (UPSERT)
```sql
INSERT INTO employees (id, first_name, last_name, salary, updated_at)
SELECT id, first_name, last_name, salary, NOW()
FROM staging_employees
ON CONFLICT (id)
DO UPDATE SET
    first_name = EXCLUDED.first_name,
    last_name  = EXCLUDED.last_name,
    salary     = EXCLUDED.salary,
    updated_at = EXCLUDED.updated_at;
```

### MySQL INSERT ... ON DUPLICATE KEY UPDATE
```sql
INSERT INTO products (id, name, price, stock)
SELECT id, name, price, stock FROM new_products
ON DUPLICATE KEY UPDATE
    price = VALUES(price),
    stock = stock + VALUES(stock);
```

---

## 5. RETURNING Clause

Get the affected rows back after INSERT/UPDATE/DELETE.

### PostgreSQL RETURNING
```sql
-- INSERT and get generated ID
INSERT INTO employees (first_name, last_name, email)
VALUES ('Carol', 'White', 'carol@co.com')
RETURNING id, created_at;

-- INSERT multiple and get all IDs
INSERT INTO orders (customer_id, total)
VALUES (1, 100), (2, 200), (3, 300)
RETURNING id, customer_id, total;

-- UPDATE and get updated values
UPDATE employees
SET salary = salary * 1.10
WHERE dept_id = 10
RETURNING id, first_name, salary AS new_salary;

-- DELETE and get deleted rows
DELETE FROM employees
WHERE hire_date < '2015-01-01'
RETURNING id, first_name, last_name;
```

### MySQL: Last Insert ID
```sql
-- Get last auto-incremented ID
INSERT INTO employees (first_name) VALUES ('Dave');
SELECT LAST_INSERT_ID();

-- In application code (PHP example)
-- $id = $conn->insert_id;
```

### SQL Server OUTPUT clause
```sql
-- INSERT ... OUTPUT
INSERT INTO employees (first_name, salary)
OUTPUT INSERTED.id, INSERTED.first_name, INSERTED.created_at
VALUES ('Eve', 70000);

-- UPDATE ... OUTPUT
UPDATE employees
SET salary = salary * 1.10
OUTPUT DELETED.salary AS old_salary, INSERTED.salary AS new_salary, INSERTED.id
WHERE dept_id = 10;

-- DELETE ... OUTPUT
DELETE FROM employees
OUTPUT DELETED.*
WHERE is_active = FALSE;

-- Capture OUTPUT into a table variable
DECLARE @updated TABLE (id INT, old_salary DECIMAL, new_salary DECIMAL);
UPDATE employees
SET salary = salary * 1.10
OUTPUT DELETED.id, DELETED.salary, INSERTED.salary
INTO @updated;
SELECT * FROM @updated;
```

---

## 6. DML with Subqueries

### INSERT with Subquery
```sql
-- Insert aggregated data
INSERT INTO dept_salary_summary (dept_id, avg_salary, employee_count, as_of_date)
SELECT
    dept_id,
    AVG(salary),
    COUNT(*),
    CURRENT_DATE
FROM employees
GROUP BY dept_id;
```

### UPDATE with Correlated Subquery
```sql
-- Set each employee's salary to their department average
UPDATE employees e
SET salary = (
    SELECT AVG(e2.salary)
    FROM employees e2
    WHERE e2.dept_id = e.dept_id
);
```

### DELETE with EXISTS
```sql
-- Delete customers who have no orders
DELETE FROM customers c
WHERE NOT EXISTS (
    SELECT 1 FROM orders o WHERE o.customer_id = c.id
);
```

### TRUNCATE vs DELETE vs DROP Summary
```sql
-- TRUNCATE: fast, resets sequence, no WHERE, no rollback (usually)
TRUNCATE TABLE logs;

-- DELETE: slow, with WHERE, transactional
DELETE FROM logs WHERE created_at < NOW() - INTERVAL '90 DAY';

-- DROP: removes the entire table
DROP TABLE logs;
```

---

## DML Quick Reference

```sql
-- INSERT
INSERT INTO t (c1, c2) VALUES (v1, v2);
INSERT INTO t (c1, c2) VALUES (v1, v2), (v3, v4);
INSERT INTO t (c1, c2) SELECT c1, c2 FROM other_t WHERE condition;
INSERT INTO t (c1) VALUES (v1) ON CONFLICT (c1) DO UPDATE SET c2 = EXCLUDED.c2;  -- PG
INSERT INTO t (c1) VALUES (v1) ON DUPLICATE KEY UPDATE c2 = VALUES(c2);           -- MySQL

-- UPDATE
UPDATE t SET c1 = v1 WHERE condition;
UPDATE t SET c1 = v1, c2 = v2 WHERE condition;
UPDATE t SET c1 = CASE WHEN x THEN v1 ELSE v2 END;
UPDATE t SET c1 = (SELECT ... FROM other_t WHERE ...);

-- DELETE
DELETE FROM t WHERE condition;
DELETE FROM t WHERE id IN (SELECT id FROM other_t WHERE ...);
DELETE FROM t WHERE EXISTS (SELECT 1 FROM other_t WHERE ...);

-- MERGE (SQL Server / PostgreSQL 15+)
MERGE INTO target USING source ON key
WHEN MATCHED THEN UPDATE SET ...
WHEN NOT MATCHED THEN INSERT ...;
```
