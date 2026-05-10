# Advanced SQL

## Table of Contents
1. [PIVOT and UNPIVOT](#1-pivot-and-unpivot)
2. [JSON in SQL](#2-json-in-sql)
3. [UNION, INTERSECT, EXCEPT](#3-union-intersect-except)
4. [Advanced CASE Expressions](#4-advanced-case-expressions)
5. [COALESCE, NULLIF, NVL](#5-coalesce-nullif-nvl)
6. [Set Operations and Relational Division](#6-set-operations-and-relational-division)
7. [Dynamic SQL](#7-dynamic-sql)
8. [DCL — Permissions and Security](#8-dcl--permissions-and-security)
9. [System Tables and Information Schema](#9-system-tables-and-information-schema)
10. [Advanced String and Regex](#10-advanced-string-and-regex)

---

## 1. PIVOT and UNPIVOT

### PIVOT (Rows to Columns)

#### SQL Server Native PIVOT
```sql
-- Sales per product per quarter
SELECT *
FROM (
    SELECT product_id, quarter, amount FROM sales
) src
PIVOT (
    SUM(amount) FOR quarter IN ([Q1], [Q2], [Q3], [Q4])
) AS pivot_table;

-- Result:
-- product_id  Q1     Q2     Q3     Q4
-- 1           5000   6000   4500   7000
-- 2           3000   3500   4000   5500
```

#### Manual PIVOT with CASE (Works everywhere)
```sql
SELECT
    product_id,
    SUM(CASE WHEN quarter = 'Q1' THEN amount ELSE 0 END) AS Q1,
    SUM(CASE WHEN quarter = 'Q2' THEN amount ELSE 0 END) AS Q2,
    SUM(CASE WHEN quarter = 'Q3' THEN amount ELSE 0 END) AS Q3,
    SUM(CASE WHEN quarter = 'Q4' THEN amount ELSE 0 END) AS Q4
FROM sales
GROUP BY product_id;
```

#### PostgreSQL CROSSTAB (requires tablefunc extension)
```sql
-- Enable extension
CREATE EXTENSION IF NOT EXISTS tablefunc;

SELECT * FROM crosstab(
    'SELECT product_id, quarter, SUM(amount)
     FROM sales
     GROUP BY product_id, quarter
     ORDER BY 1, 2',
    'VALUES (''Q1''),(''Q2''),(''Q3''),(''Q4'')'
) AS pivot (product_id INT, Q1 DECIMAL, Q2 DECIMAL, Q3 DECIMAL, Q4 DECIMAL);
```

### UNPIVOT (Columns to Rows)

#### SQL Server Native UNPIVOT
```sql
-- Turn quarterly columns back into rows
SELECT product_id, quarter, amount
FROM quarterly_sales
UNPIVOT (
    amount FOR quarter IN (Q1, Q2, Q3, Q4)
) AS unpivot_table;
```

#### Manual UNPIVOT with UNION ALL
```sql
SELECT product_id, 'Q1' AS quarter, Q1 AS amount FROM quarterly_sales
UNION ALL
SELECT product_id, 'Q2', Q2 FROM quarterly_sales
UNION ALL
SELECT product_id, 'Q3', Q3 FROM quarterly_sales
UNION ALL
SELECT product_id, 'Q4', Q4 FROM quarterly_sales;
```

#### PostgreSQL with VALUES
```sql
SELECT product_id, quarter, amount
FROM quarterly_sales
CROSS JOIN LATERAL (
    VALUES ('Q1', Q1), ('Q2', Q2), ('Q3', Q3), ('Q4', Q4)
) AS t(quarter, amount);
```

---

## 2. JSON in SQL

### MySQL JSON

```sql
-- JSON column
CREATE TABLE events (
    id       INT PRIMARY KEY,
    payload  JSON
);

-- Insert JSON
INSERT INTO events (id, payload)
VALUES (1, '{"user_id": 42, "action": "login", "tags": ["web", "mobile"]}');

-- Extract value
SELECT payload->'$.user_id'         FROM events;  -- Returns "42" (with quotes)
SELECT payload->>'$.user_id'        FROM events;  -- Returns 42 (unquoted)
SELECT JSON_EXTRACT(payload, '$.user_id') FROM events;
SELECT JSON_UNQUOTE(JSON_EXTRACT(payload, '$.action')) FROM events;

-- Access nested
SELECT payload->'$.address.city' FROM events;

-- Access array element
SELECT payload->'$.tags[0]' FROM events;  -- "web"

-- Modify JSON
UPDATE events
SET payload = JSON_SET(payload, '$.action', 'logout')
WHERE id = 1;

UPDATE events
SET payload = JSON_INSERT(payload, '$.timestamp', NOW())
WHERE id = 1;

UPDATE events
SET payload = JSON_REMOVE(payload, '$.tags[1]')
WHERE id = 1;

-- Aggregation
SELECT JSON_ARRAYAGG(name) FROM employees WHERE dept_id = 10;
SELECT JSON_OBJECTAGG(id, name) FROM employees WHERE dept_id = 10;

-- Filter by JSON value
SELECT * FROM events WHERE payload->>'$.action' = 'login';
SELECT * FROM events WHERE JSON_CONTAINS(payload->'$.tags', '"web"');

-- JSON index (virtual generated column)
ALTER TABLE events ADD COLUMN user_id INT
    GENERATED ALWAYS AS (payload->>'$.user_id') VIRTUAL;
CREATE INDEX idx_events_user ON events (user_id);
```

### PostgreSQL JSON / JSONB

```sql
-- Two JSON types:
-- json: stored as text, preserves whitespace & key order
-- jsonb: stored as binary, faster querying (preferred)

CREATE TABLE events (
    id       SERIAL PRIMARY KEY,
    payload  JSONB
);

-- Insert
INSERT INTO events (payload)
VALUES ('{"user_id": 42, "action": "login", "tags": ["web", "mobile"]}');

-- Extract operators
SELECT payload->'user_id'         FROM events;  -- Returns JSON "42"
SELECT payload->>'user_id'        FROM events;  -- Returns text "42"
SELECT payload->'address'->>'city' FROM events;  -- Nested
SELECT payload#>'{address,city}'  FROM events;  -- Path operator
SELECT payload#>>'{address,city}' FROM events;  -- Text version

-- Array element
SELECT payload->'tags'->0 FROM events;   -- "web"
SELECT payload->'tags'->>0 FROM events;  -- web (text)

-- Modify JSONB
SELECT payload || '{"verified": true}'::JSONB FROM events;  -- Merge
SELECT payload - 'action' FROM events;                       -- Remove key

UPDATE events
SET payload = payload || '{"status": "processed"}'::JSONB
WHERE id = 1;

UPDATE events
SET payload = payload - 'old_key'
WHERE id = 1;

-- JSONB operators
payload ? 'user_id'              -- Key exists
payload ?| ARRAY['a', 'b']       -- Any key exists
payload ?& ARRAY['a', 'b']       -- All keys exist
payload @> '{"action":"login"}'  -- Contains
payload <@ '{"action":"login"}'  -- Is contained by

-- Build JSON
SELECT JSON_BUILD_OBJECT('id', id, 'name', first_name) FROM employees;
SELECT JSON_AGG(row_to_json(e)) FROM employees e WHERE dept_id = 10;
SELECT JSONB_BUILD_ARRAY(1, 'two', TRUE, NULL);
SELECT JSONB_OBJECT(ARRAY['a','b'], ARRAY['1','2']);

-- Expand JSON to rows
SELECT * FROM JSONB_EACH('{"a":1,"b":2}');   -- (key, value) rows
SELECT * FROM JSONB_EACH_TEXT('{"a":"1"}');
SELECT * FROM JSONB_ARRAY_ELEMENTS('[1,2,3]');
SELECT JSONB_ARRAY_LENGTH('[1,2,3]');

-- Keys and values
SELECT JSONB_OBJECT_KEYS('{"a":1,"b":2}');   -- Returns rows

-- Path queries
SELECT JSONB_PATH_QUERY(payload, '$.tags[*]') FROM events;
SELECT JSONB_PATH_QUERY_ARRAY(payload, '$.tags[*]') FROM events;

-- GIN index for JSONB
CREATE INDEX idx_events_payload ON events USING GIN (payload);
CREATE INDEX idx_events_ops ON events USING GIN (payload jsonb_path_ops);
```

---

## 3. UNION, INTERSECT, EXCEPT

### UNION (removes duplicates)
```sql
-- Combine customers from two regions
SELECT id, name FROM customers_us
UNION
SELECT id, name FROM customers_eu;
-- Removes duplicate rows (same as UNION DISTINCT)
```

### UNION ALL (keeps duplicates, faster)
```sql
SELECT id, name, 'US' AS region FROM customers_us
UNION ALL
SELECT id, name, 'EU' AS region FROM customers_eu;
-- Keeps all rows, much faster (no dedup step)
```

### INTERSECT (common rows)
```sql
-- Customers who exist in both regions
SELECT id FROM customers_us
INTERSECT
SELECT id FROM customers_eu;
```

### EXCEPT / MINUS (rows in first but not second)
```sql
-- US customers NOT in EU
SELECT id FROM customers_us
EXCEPT
SELECT id FROM customers_eu;

-- Oracle uses MINUS instead of EXCEPT
SELECT id FROM customers_us
MINUS
SELECT id FROM customers_eu;
```

### Combining Multiple Sets
```sql
-- Column count and types must match across all queries
SELECT employee_id AS person_id, 'employee' AS type FROM employees
UNION ALL
SELECT contractor_id, 'contractor' FROM contractors
UNION ALL
SELECT vendor_id, 'vendor' FROM vendors
ORDER BY type, person_id;
```

### Set Operations Comparison

| Operation | Keeps Duplicates | Returns |
|-----------|-----------------|---------|
| UNION | No | Rows in A OR B |
| UNION ALL | Yes | All rows from A and B |
| INTERSECT | No | Rows in A AND B |
| EXCEPT / MINUS | No | Rows in A but NOT in B |

---

## 4. Advanced CASE Expressions

### CASE in ORDER BY (Custom Sort)
```sql
SELECT * FROM tickets
ORDER BY
    CASE priority
        WHEN 'critical' THEN 1
        WHEN 'high'     THEN 2
        WHEN 'medium'   THEN 3
        WHEN 'low'      THEN 4
        ELSE 5
    END,
    created_at ASC;
```

### CASE for Pivot
```sql
SELECT
    month,
    SUM(CASE WHEN status = 'completed' THEN revenue ELSE 0 END) AS completed_rev,
    SUM(CASE WHEN status = 'refunded'  THEN revenue ELSE 0 END) AS refunded_rev,
    SUM(CASE WHEN status = 'pending'   THEN revenue ELSE 0 END) AS pending_rev,
    SUM(revenue) AS total_rev
FROM orders
GROUP BY month
ORDER BY month;
```

### Nested CASE
```sql
SELECT
    id,
    salary,
    CASE
        WHEN dept_id = 10 THEN
            CASE WHEN salary > 100000 THEN 'Sr Engineer' ELSE 'Jr Engineer' END
        WHEN dept_id = 20 THEN
            CASE WHEN salary > 80000 THEN 'Sr Analyst' ELSE 'Jr Analyst' END
        ELSE 'Other'
    END AS job_level
FROM employees;
```

---

## 5. COALESCE, NULLIF, NVL

```sql
-- COALESCE: first non-NULL value
SELECT COALESCE(preferred_name, first_name, 'Unknown') AS display_name
FROM users;

-- Chain for fallback
SELECT COALESCE(mobile_phone, home_phone, work_phone, 'No phone') AS contact
FROM contacts;

-- COALESCE in calculations (treat NULL as 0)
SELECT id, salary + COALESCE(bonus, 0) AS total_comp FROM employees;

-- NULLIF: return NULL if a = b (avoid division by zero)
SELECT total_revenue / NULLIF(total_units, 0) AS avg_price FROM sales;

-- NULLIF to normalize empty strings to NULL
SELECT NULLIF(TRIM(phone), '') AS phone FROM contacts;

-- IIF (SQL Server): inline IF
SELECT IIF(salary > 80000, 'High', 'Normal') AS salary_band FROM employees;

-- DECODE (Oracle): similar to CASE
SELECT DECODE(status, 'A', 'Active', 'I', 'Inactive', 'Unknown') FROM users;

-- NVL (Oracle / some others): NVL(val, default) = COALESCE with 2 args
SELECT NVL(commission, 0) FROM employees;

-- NVL2 (Oracle): NVL2(val, if_not_null, if_null)
SELECT NVL2(commission, 'Has Commission', 'No Commission') FROM employees;
```

---

## 6. Set Operations and Relational Division

### Relational Division (Find rows that match ALL items)

**Problem**: Find customers who ordered ALL products in a given list.

```sql
-- Products that customer must have ordered
-- Method 1: Double NOT EXISTS (classic)
SELECT DISTINCT c.id, c.name
FROM customers c
WHERE NOT EXISTS (
    SELECT 1 FROM products p
    WHERE p.category = 'Essential'
    AND NOT EXISTS (
        SELECT 1 FROM orders o
        WHERE o.customer_id = c.id
        AND o.product_id = p.id
    )
);

-- Method 2: COUNT approach
SELECT customer_id
FROM orders o
JOIN products p ON o.product_id = p.id
WHERE p.category = 'Essential'
GROUP BY customer_id
HAVING COUNT(DISTINCT o.product_id) = (
    SELECT COUNT(*) FROM products WHERE category = 'Essential'
);
```

### Find Common Values Between Tables
```sql
-- Students enrolled in ALL courses of a specific program
SELECT student_id
FROM enrollments
WHERE program_id = 5
GROUP BY student_id
HAVING COUNT(DISTINCT course_id) = (
    SELECT COUNT(*) FROM courses WHERE program_id = 5
);
```

---

## 7. Dynamic SQL

### MySQL Dynamic SQL (Prepared Statements)
```sql
-- Build and execute dynamic query
DELIMITER $$
CREATE PROCEDURE dynamic_search(IN p_column VARCHAR(50), IN p_value VARCHAR(100))
BEGIN
    SET @sql = CONCAT('SELECT * FROM employees WHERE ', p_column, ' = ?');
    SET @val = p_value;
    PREPARE stmt FROM @sql;
    EXECUTE stmt USING @val;
    DEALLOCATE PREPARE stmt;
END$$
DELIMITER ;

CALL dynamic_search('dept_id', '10');
CALL dynamic_search('last_name', 'Smith');
```

### PostgreSQL Dynamic SQL with EXECUTE
```sql
CREATE OR REPLACE PROCEDURE search_table(p_table TEXT, p_col TEXT, p_val TEXT)
AS $$
DECLARE
    v_sql TEXT;
    v_result RECORD;
BEGIN
    -- Always quote identifiers to prevent SQL injection
    v_sql := FORMAT('SELECT * FROM %I WHERE %I = %L', p_table, p_col, p_val);
    RAISE NOTICE 'Executing: %', v_sql;

    FOR v_result IN EXECUTE v_sql LOOP
        RAISE NOTICE '%', v_result;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- FORMAT with:
-- %I = identifier (quoted: "column_name")
-- %L = literal (quoted: 'value')
-- %s = as-is (UNSAFE for user input!)
```

### SQL Server Dynamic SQL
```sql
DECLARE @sql NVARCHAR(MAX);
DECLARE @col NVARCHAR(100) = N'last_name';
DECLARE @val NVARCHAR(100) = N'Smith';

SET @sql = N'SELECT * FROM employees WHERE ' + QUOTENAME(@col) + N' = @p_val';

EXEC sp_executesql @sql,
    N'@p_val NVARCHAR(100)',  -- parameter definition
    @p_val = @val;             -- parameter value
-- sp_executesql is safe — parameterized, cached, injection-resistant
```

---

## 8. DCL — Permissions and Security

### GRANT
```sql
-- Grant SELECT on a table
GRANT SELECT ON employees TO 'analytics_user'@'%';           -- MySQL
GRANT SELECT ON employees TO analytics_user;                   -- PostgreSQL

-- Grant multiple privileges
GRANT SELECT, INSERT, UPDATE ON employees TO hr_user;

-- Grant on all tables in schema
GRANT SELECT ON ALL TABLES IN SCHEMA public TO reporting_user; -- PostgreSQL
GRANT SELECT ON *.* TO 'readonly'@'%';                         -- MySQL (all DBs)

-- Grant with GRANT OPTION (user can grant to others)
GRANT SELECT ON employees TO manager WITH GRANT OPTION;

-- Create role and grant to user (PostgreSQL)
CREATE ROLE readonly_role;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_role;
GRANT readonly_role TO alice, bob, carol;

-- MySQL 8+: roles
CREATE ROLE 'app_read', 'app_write';
GRANT SELECT ON mydb.* TO 'app_read';
GRANT INSERT, UPDATE, DELETE ON mydb.* TO 'app_write';
GRANT 'app_read' TO 'alice'@'%';
```

### REVOKE
```sql
-- Revoke a privilege
REVOKE SELECT ON employees FROM analytics_user;              -- PostgreSQL
REVOKE SELECT ON employees FROM 'analytics_user'@'%';       -- MySQL

-- Revoke all privileges
REVOKE ALL PRIVILEGES ON employees FROM analytics_user;      -- PostgreSQL
REVOKE ALL ON mydb.* FROM 'analytics_user'@'%';             -- MySQL

-- Revoke the GRANT OPTION
REVOKE GRANT OPTION FOR SELECT ON employees FROM manager;
```

### Row-Level Security (PostgreSQL)
```sql
-- Enable RLS on table
ALTER TABLE employees ENABLE ROW LEVEL SECURITY;

-- Create policy: users can only see their own department
CREATE POLICY emp_isolation ON employees
    USING (dept_id = current_setting('app.current_dept_id')::INT);

-- Policy for specific commands
CREATE POLICY emp_select ON employees
    FOR SELECT
    USING (dept_id = current_setting('app.dept_id')::INT);

CREATE POLICY emp_insert ON employees
    FOR INSERT
    WITH CHECK (dept_id = current_setting('app.dept_id')::INT);

-- Admin bypasses RLS
CREATE POLICY emp_admin ON employees
    USING (current_user = 'admin');

-- Set context variable in session
SET app.current_dept_id = '10';
```

### Create User / Role
```sql
-- MySQL
CREATE USER 'alice'@'%' IDENTIFIED BY 'secure_password';
CREATE USER 'app_user'@'localhost' IDENTIFIED BY 'pass';
ALTER USER 'alice'@'%' IDENTIFIED BY 'new_password';
DROP USER 'alice'@'%';

-- PostgreSQL
CREATE USER alice WITH PASSWORD 'secure_password';
CREATE ROLE bob WITH LOGIN PASSWORD 'pass';
ALTER USER alice WITH PASSWORD 'new_password';
DROP USER alice;
```

---

## 9. System Tables and Information Schema

### Information Schema (Standard SQL — works across databases)
```sql
-- List all tables
SELECT TABLE_NAME, TABLE_TYPE
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = 'mydb';

-- List columns of a table
SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT, CHARACTER_MAXIMUM_LENGTH
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = 'mydb' AND TABLE_NAME = 'employees'
ORDER BY ORDINAL_POSITION;

-- List indexes
SELECT TABLE_NAME, INDEX_NAME, COLUMN_NAME, NON_UNIQUE
FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA = 'mydb'
ORDER BY TABLE_NAME, INDEX_NAME;

-- List foreign keys
SELECT
    TABLE_NAME, CONSTRAINT_NAME, COLUMN_NAME,
    REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
FROM information_schema.KEY_COLUMN_USAGE
WHERE TABLE_SCHEMA = 'mydb'
  AND REFERENCED_TABLE_NAME IS NOT NULL;

-- List views
SELECT TABLE_NAME, VIEW_DEFINITION
FROM information_schema.VIEWS
WHERE TABLE_SCHEMA = 'mydb';
```

### PostgreSQL System Catalogs
```sql
-- Table sizes
SELECT
    relname AS table_name,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
    pg_size_pretty(pg_relation_size(relid)) AS table_size,
    pg_size_pretty(pg_total_relation_size(relid) - pg_relation_size(relid)) AS index_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;

-- Row counts estimate
SELECT relname, reltuples::BIGINT AS row_estimate
FROM pg_class
WHERE relkind = 'r'
ORDER BY reltuples DESC;

-- Current connections
SELECT pid, usename, application_name, state, query
FROM pg_stat_activity
WHERE state != 'idle';

-- Long-running queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity
WHERE state = 'active' AND (now() - query_start) > INTERVAL '5 minutes';

-- Kill a query
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid = 12345;

-- Index usage stats
SELECT indexrelname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes
ORDER BY idx_scan;

-- Cache hit ratio (should be > 99%)
SELECT
    100 * heap_blks_hit / NULLIF(heap_blks_hit + heap_blks_read, 0) AS cache_hit_ratio
FROM pg_statio_user_tables
WHERE relname = 'employees';
```

---

## 10. Advanced String and Regex

### Regular Expressions (PostgreSQL)
```sql
-- ~ : case-sensitive match
SELECT * FROM users WHERE email ~ '^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$';

-- ~* : case-insensitive match
SELECT * FROM products WHERE name ~* 'laptop|notebook';

-- !~ : does not match
SELECT * FROM employees WHERE phone !~ '^\d{3}-\d{3}-\d{4}$';

-- REGEXP_REPLACE: replace with regex
SELECT REGEXP_REPLACE(phone, '[^0-9]', '', 'g') AS clean_phone FROM contacts;

-- REGEXP_MATCHES: extract groups
SELECT REGEXP_MATCHES(email, '^(.+)@(.+)\.(.+)$') AS parts FROM users;
-- Returns array: {user, domain, tld}

-- REGEXP_SPLIT_TO_TABLE: split by regex
SELECT REGEXP_SPLIT_TO_TABLE('one,two,,three', ',') AS word;

-- REGEXP_SPLIT_TO_ARRAY
SELECT REGEXP_SPLIT_TO_ARRAY('one,two,three', ',') AS words;
```

### MySQL Regular Expressions
```sql
-- REGEXP / RLIKE
SELECT * FROM products WHERE name REGEXP '^[A-Z]{2}-[0-9]{4}$';

-- REGEXP_REPLACE (MySQL 8.0+)
SELECT REGEXP_REPLACE(phone, '[^0-9]', '') FROM contacts;

-- REGEXP_SUBSTR (MySQL 8.0+)
SELECT REGEXP_SUBSTR(description, '[0-9]+(\.[0-9]+)?') AS extracted_number
FROM products;

-- REGEXP_INSTR (MySQL 8.0+)
SELECT REGEXP_INSTR(name, '[0-9]+') AS first_digit_pos FROM products;
```

### Advanced String Processing
```sql
-- Split a comma-separated string into rows (PostgreSQL)
SELECT UNNEST(STRING_TO_ARRAY('a,b,c,d', ',')) AS element;

-- Word count
SELECT
    id,
    content,
    ARRAY_LENGTH(REGEXP_SPLIT_TO_ARRAY(TRIM(content), '\s+'), 1) AS word_count
FROM articles;

-- Levenshtein distance (fuzzy matching) — PostgreSQL fuzzystrmatch
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
SELECT LEVENSHTEIN('Smith', 'Smithe');  -- 1 (one edit)
SELECT SOUNDEX('Smith') = SOUNDEX('Smithe');  -- Same sound
SELECT METAPHONE('Smith', 10);  -- Phonetic encoding

-- Generate slug from title
SELECT LOWER(REGEXP_REPLACE(TRIM(title), '[^a-zA-Z0-9]+', '-', 'g')) AS slug
FROM articles;
```

---

## Quick Reference

```sql
-- PIVOT (manual)
SELECT col, SUM(CASE WHEN cat='A' THEN val END) AS A,
            SUM(CASE WHEN cat='B' THEN val END) AS B
FROM t GROUP BY col;

-- UNPIVOT (manual)
SELECT id, 'Q1' AS q, Q1 FROM t UNION ALL SELECT id, 'Q2', Q2 FROM t;

-- JSON (PostgreSQL)
payload->>'key'              -- Extract text
payload->'key'               -- Extract JSON
payload @> '{"k":"v"}'      -- Contains
payload ? 'key'             -- Key exists
payload || '{"k":"v"}'::jsonb -- Merge
payload - 'key'             -- Remove key
JSONB_EACH(payload)         -- Expand to rows

-- JSON (MySQL)
JSON_EXTRACT(col, '$.key')
JSON_SET(col, '$.key', val)
JSON_CONTAINS(col, val, path)

-- Set operations
UNION / UNION ALL / INTERSECT / EXCEPT (MINUS)

-- Dynamic SQL (PostgreSQL)
EXECUTE FORMAT('SELECT * FROM %I WHERE %I = %L', tbl, col, val);

-- Permissions
GRANT SELECT ON t TO user;
REVOKE SELECT ON t FROM user;
CREATE ROLE r; GRANT r TO user;

-- Regex (PostgreSQL)
col ~ 'pattern'       -- Match
col ~* 'pattern'      -- Case-insensitive
REGEXP_REPLACE(col, 'pattern', 'replacement', 'flags')
REGEXP_MATCHES(col, 'pattern')
```
