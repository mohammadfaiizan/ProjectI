# DQL — SELECT and Filtering

## Table of Contents
1. [SELECT Basics](#1-select-basics)
2. [WHERE Clause](#2-where-clause)
3. [Comparison Operators](#3-comparison-operators)
4. [Logical Operators](#4-logical-operators)
5. [LIKE and Pattern Matching](#5-like-and-pattern-matching)
6. [IN and NOT IN](#6-in-and-not-in)
7. [BETWEEN](#7-between)
8. [NULL Handling](#8-null-handling)
9. [ORDER BY](#9-order-by)
10. [LIMIT and OFFSET](#10-limit-and-offset)
11. [DISTINCT](#11-distinct)
12. [Column Aliases and Expressions](#12-column-aliases-and-expressions)
13. [CASE Expression](#13-case-expression)
14. [String Functions](#14-string-functions)
15. [Date and Time Functions](#15-date-and-time-functions)
16. [Math Functions](#16-math-functions)
17. [Type Casting](#17-type-casting)

---

## 1. SELECT Basics

### Select All Columns
```sql
SELECT * FROM employees;
```

### Select Specific Columns
```sql
SELECT first_name, last_name, salary FROM employees;
```

### Select with Expression
```sql
SELECT
    first_name,
    last_name,
    salary,
    salary * 12          AS annual_salary,
    salary * 0.10        AS bonus,
    salary + salary * 12 AS total_comp
FROM employees;
```

### Select Constants / Literals
```sql
SELECT
    'Hello'              AS greeting,
    42                   AS the_answer,
    CURRENT_DATE         AS today,
    NOW()                AS right_now,
    3.14                 AS pi;
```

### Select from Dual (Oracle / MySQL)
```sql
SELECT 1 + 1 FROM DUAL;          -- Oracle
SELECT 1 + 1;                     -- MySQL / PostgreSQL (no FROM needed)
```

### SELECT with Schema Qualification
```sql
SELECT e.first_name, d.name AS department
FROM hr.employees e
JOIN hr.departments d ON e.dept_id = d.id;
```

---

## 2. WHERE Clause

The WHERE clause filters rows before any grouping.

```sql
-- Single condition
SELECT * FROM employees WHERE dept_id = 10;

-- Multiple conditions with AND
SELECT * FROM employees
WHERE dept_id = 10 AND salary > 60000;

-- Multiple conditions with OR
SELECT * FROM employees
WHERE dept_id = 10 OR dept_id = 20;

-- Combined AND / OR (use parentheses to control precedence)
SELECT * FROM employees
WHERE (dept_id = 10 OR dept_id = 20)
  AND salary > 50000
  AND is_active = TRUE;
```

---

## 3. Comparison Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `=` | Equal | `salary = 75000` |
| `<>` or `!=` | Not equal | `status <> 'inactive'` |
| `<` | Less than | `age < 30` |
| `>` | Greater than | `salary > 100000` |
| `<=` | Less than or equal | `hire_date <= '2020-01-01'` |
| `>=` | Greater than or equal | `score >= 90` |
| `<=>` | NULL-safe equal (MySQL) | `col <=> NULL` |

```sql
SELECT * FROM employees WHERE salary >= 75000;
SELECT * FROM employees WHERE hire_date < '2020-01-01';
SELECT * FROM employees WHERE last_name <> 'Smith';

-- NULL-safe comparison (MySQL)
SELECT * FROM employees WHERE manager_id <=> NULL;
-- Equivalent to:
SELECT * FROM employees WHERE manager_id IS NULL;
```

---

## 4. Logical Operators

### AND
Both conditions must be true.
```sql
SELECT * FROM employees
WHERE salary > 50000 AND dept_id = 10;
```

### OR
At least one condition must be true.
```sql
SELECT * FROM employees
WHERE dept_id = 10 OR dept_id = 20 OR dept_id = 30;
```

### NOT
Negates a condition.
```sql
SELECT * FROM employees WHERE NOT (dept_id = 10);
SELECT * FROM employees WHERE NOT is_active;
SELECT * FROM products WHERE NOT category IN ('A', 'B');
```

### Operator Precedence (highest to lowest)
```
1. NOT
2. AND
3. OR
```

```sql
-- These are different!
WHERE dept_id = 10 OR dept_id = 20 AND salary > 50000
-- Parsed as:
WHERE dept_id = 10 OR (dept_id = 20 AND salary > 50000)

-- Use parentheses for clarity:
WHERE (dept_id = 10 OR dept_id = 20) AND salary > 50000
```

---

## 5. LIKE and Pattern Matching

### LIKE
Case-insensitive pattern matching with wildcards.

| Wildcard | Meaning |
|----------|---------|
| `%` | Zero or more characters |
| `_` | Exactly one character |

```sql
-- Names starting with 'A'
SELECT * FROM employees WHERE first_name LIKE 'A%';

-- Names ending with 'son'
SELECT * FROM employees WHERE last_name LIKE '%son';

-- Names containing 'an'
SELECT * FROM employees WHERE first_name LIKE '%an%';

-- Names where second character is 'l'
SELECT * FROM employees WHERE first_name LIKE '_l%';

-- Exactly 5 characters
SELECT * FROM products WHERE code LIKE '_____';

-- NOT LIKE
SELECT * FROM employees WHERE email NOT LIKE '%@gmail.com';
```

### ILIKE (PostgreSQL — case-insensitive)
```sql
SELECT * FROM employees WHERE first_name ILIKE 'alice';
SELECT * FROM employees WHERE email ILIKE '%@gmail.com';
```

### SIMILAR TO (PostgreSQL — regex-like)
```sql
-- Matches regex-style patterns
SELECT * FROM employees WHERE first_name SIMILAR TO 'A(lic|my)%';
```

### REGEXP / RLIKE (MySQL)
```sql
-- Regular expression matching
SELECT * FROM employees WHERE first_name REGEXP '^[A-C]';
SELECT * FROM employees WHERE phone REGEXP '^[0-9]{3}-[0-9]{3}-[0-9]{4}$';
```

### PostgreSQL Regular Expressions
```sql
-- ~ operator (case-sensitive match)
SELECT * FROM employees WHERE first_name ~ '^Alice';

-- ~* operator (case-insensitive match)
SELECT * FROM employees WHERE first_name ~* '^alice';

-- !~ (does not match)
SELECT * FROM employees WHERE email !~ '@gmail\.com$';
```

### Escape Special Characters in LIKE
```sql
-- Find literal % in data
SELECT * FROM discounts WHERE description LIKE '%50\%%' ESCAPE '\';

-- PostgreSQL
SELECT * FROM discounts WHERE description LIKE '%50!%%' ESCAPE '!';
```

---

## 6. IN and NOT IN

```sql
-- IN with literal list
SELECT * FROM employees WHERE dept_id IN (10, 20, 30);

-- NOT IN
SELECT * FROM employees WHERE dept_id NOT IN (10, 20);

-- IN with subquery
SELECT * FROM employees
WHERE dept_id IN (
    SELECT id FROM departments WHERE location = 'New York'
);

-- IN vs OR (equivalent)
WHERE dept_id IN (10, 20)
-- same as:
WHERE dept_id = 10 OR dept_id = 20

-- NULL gotcha with NOT IN
-- If the list contains NULL, NOT IN returns no rows (UNKNOWN)
SELECT * FROM employees WHERE dept_id NOT IN (10, NULL);
-- Returns 0 rows! Because (dept_id != NULL) is always UNKNOWN

-- Safe approach with NOT EXISTS
SELECT * FROM employees e
WHERE NOT EXISTS (
    SELECT 1 FROM excluded_depts ed WHERE ed.id = e.dept_id
);
```

---

## 7. BETWEEN

```sql
-- Inclusive range (includes both endpoints)
SELECT * FROM employees WHERE salary BETWEEN 50000 AND 100000;
-- Same as: WHERE salary >= 50000 AND salary <= 100000

-- NOT BETWEEN
SELECT * FROM employees WHERE salary NOT BETWEEN 50000 AND 100000;

-- Date range
SELECT * FROM orders
WHERE order_date BETWEEN '2024-01-01' AND '2024-12-31';

-- BETWEEN with timestamps (careful about time component)
SELECT * FROM orders
WHERE order_date BETWEEN '2024-01-01 00:00:00' AND '2024-12-31 23:59:59';

-- Better for dates with timestamps:
SELECT * FROM orders
WHERE order_date >= '2024-01-01'
  AND order_date <  '2025-01-01';  -- exclusive end
```

---

## 8. NULL Handling

```sql
-- IS NULL
SELECT * FROM employees WHERE manager_id IS NULL;

-- IS NOT NULL
SELECT * FROM employees WHERE manager_id IS NOT NULL;

-- COALESCE: returns first non-NULL value
SELECT
    first_name,
    COALESCE(phone, mobile, email, 'No contact') AS contact
FROM employees;

-- IFNULL (MySQL) / NVL (Oracle): replace NULL with default
SELECT first_name, IFNULL(bonus, 0) AS bonus FROM employees;   -- MySQL
SELECT first_name, NVL(bonus, 0)    AS bonus FROM employees;   -- Oracle

-- NULLIF: return NULL if two values are equal (avoid divide by zero)
SELECT
    total_sales / NULLIF(total_units, 0) AS avg_price
FROM sales_summary;

-- ISNULL (SQL Server)
SELECT first_name, ISNULL(bonus, 0) AS bonus FROM employees;   -- SQL Server

-- NULL in comparisons
SELECT * FROM employees WHERE manager_id = NULL;    -- WRONG (returns 0 rows)
SELECT * FROM employees WHERE manager_id IS NULL;   -- CORRECT

-- NULL in ORDER BY
SELECT * FROM employees ORDER BY salary ASC;   -- NULLs sort last (default varies by DB)
SELECT * FROM employees ORDER BY salary ASC NULLS LAST;   -- PostgreSQL
SELECT * FROM employees ORDER BY salary ASC NULLS FIRST;  -- PostgreSQL
```

---

## 9. ORDER BY

```sql
-- Ascending (default)
SELECT * FROM employees ORDER BY last_name;
SELECT * FROM employees ORDER BY last_name ASC;

-- Descending
SELECT * FROM employees ORDER BY salary DESC;

-- Multiple columns
SELECT * FROM employees ORDER BY dept_id ASC, salary DESC;

-- By column position (not recommended — fragile)
SELECT first_name, last_name, salary FROM employees ORDER BY 3 DESC;

-- By alias
SELECT salary * 12 AS annual_salary FROM employees ORDER BY annual_salary DESC;

-- By expression
SELECT * FROM employees ORDER BY LENGTH(first_name);

-- NULL handling in ORDER BY
SELECT * FROM employees ORDER BY salary DESC NULLS LAST;  -- PostgreSQL
SELECT * FROM employees ORDER BY ISNULL(salary), salary DESC;  -- MySQL trick

-- Random order
SELECT * FROM employees ORDER BY RAND();      -- MySQL
SELECT * FROM employees ORDER BY RANDOM();    -- PostgreSQL
SELECT TOP 5 * FROM employees ORDER BY NEWID();  -- SQL Server
```

---

## 10. LIMIT and OFFSET

### LIMIT (MySQL / PostgreSQL)
```sql
-- First 10 rows
SELECT * FROM employees LIMIT 10;

-- Skip first 20, return next 10 (pagination)
SELECT * FROM employees LIMIT 10 OFFSET 20;

-- MySQL shorthand: LIMIT offset, count
SELECT * FROM employees LIMIT 20, 10;  -- skip 20, return 10
```

### FETCH FIRST (SQL Standard)
```sql
SELECT * FROM employees FETCH FIRST 10 ROWS ONLY;
SELECT * FROM employees OFFSET 20 ROWS FETCH NEXT 10 ROWS ONLY;
```

### TOP (SQL Server)
```sql
SELECT TOP 10 * FROM employees;
SELECT TOP 10 PERCENT * FROM employees;
SELECT TOP 10 WITH TIES * FROM employees ORDER BY salary DESC;
```

### Pagination Pattern
```sql
-- Page 1: rows 1-10
SELECT * FROM employees ORDER BY id LIMIT 10 OFFSET 0;

-- Page 2: rows 11-20
SELECT * FROM employees ORDER BY id LIMIT 10 OFFSET 10;

-- Page N: rows (N-1)*10+1 to N*10
-- OFFSET = (page_number - 1) * page_size
-- LIMIT  = page_size

-- Keyset pagination (faster for large offsets)
SELECT * FROM employees
WHERE id > :last_seen_id
ORDER BY id
LIMIT 10;
```

---

## 11. DISTINCT

```sql
-- Remove duplicate rows
SELECT DISTINCT dept_id FROM employees;

-- Distinct on multiple columns (combination must be unique)
SELECT DISTINCT dept_id, job_title FROM employees;

-- DISTINCT with COUNT
SELECT COUNT(DISTINCT dept_id) FROM employees;

-- DISTINCT vs GROUP BY (often equivalent)
SELECT DISTINCT dept_id FROM employees;
-- equivalent to:
SELECT dept_id FROM employees GROUP BY dept_id;

-- PostgreSQL: DISTINCT ON (keep first row per group)
SELECT DISTINCT ON (dept_id)
    dept_id, first_name, salary
FROM employees
ORDER BY dept_id, salary DESC;
-- Returns one row per dept_id: the one with highest salary
```

---

## 12. Column Aliases and Expressions

```sql
-- AS keyword (can be omitted)
SELECT first_name AS fname, last_name AS lname FROM employees;
SELECT first_name fname, last_name lname FROM employees;   -- same thing

-- Quoted aliases (for spaces or reserved words)
SELECT salary * 12 AS "Annual Salary" FROM employees;
SELECT salary * 12 AS "annual salary" FROM employees;

-- Computed columns
SELECT
    first_name || ' ' || last_name AS full_name,     -- PostgreSQL
    CONCAT(first_name, ' ', last_name) AS full_name,  -- MySQL
    UPPER(first_name) AS upper_name,
    salary * 1.10 AS proposed_salary,
    DATEDIFF(NOW(), hire_date) AS days_employed        -- MySQL
FROM employees;

-- Alias in ORDER BY (works because ORDER BY executes after SELECT)
SELECT salary * 12 AS annual_salary
FROM employees
ORDER BY annual_salary DESC;
```

---

## 13. CASE Expression

### Simple CASE
```sql
-- Simple CASE: compare one value against options
SELECT
    first_name,
    dept_id,
    CASE dept_id
        WHEN 10 THEN 'Engineering'
        WHEN 20 THEN 'Marketing'
        WHEN 30 THEN 'Finance'
        ELSE 'Other'
    END AS department_name
FROM employees;
```

### Searched CASE
```sql
-- Searched CASE: each WHEN has its own condition
SELECT
    first_name,
    salary,
    CASE
        WHEN salary < 50000  THEN 'Junior'
        WHEN salary < 80000  THEN 'Mid-level'
        WHEN salary < 120000 THEN 'Senior'
        ELSE                      'Executive'
    END AS salary_band
FROM employees;
```

### CASE in WHERE
```sql
SELECT * FROM employees
WHERE
    CASE
        WHEN dept_id = 10 THEN salary > 60000
        WHEN dept_id = 20 THEN salary > 70000
        ELSE salary > 50000
    END;
```

### CASE in ORDER BY
```sql
-- Custom sort order
SELECT * FROM orders
ORDER BY
    CASE status
        WHEN 'urgent'    THEN 1
        WHEN 'normal'    THEN 2
        WHEN 'low'       THEN 3
        ELSE                  4
    END;
```

### CASE in Aggregation
```sql
-- Conditional counting
SELECT
    dept_id,
    COUNT(*) AS total,
    COUNT(CASE WHEN salary > 80000 THEN 1 END) AS high_earners,
    SUM(CASE WHEN is_active = TRUE THEN salary ELSE 0 END) AS active_payroll
FROM employees
GROUP BY dept_id;
```

---

## 14. String Functions

```sql
-- Length
SELECT LENGTH('Hello');        -- 5 (MySQL / PostgreSQL)
SELECT CHAR_LENGTH('Hello');   -- 5 (MySQL, counts characters not bytes)
SELECT LEN('Hello');           -- 5 (SQL Server)

-- Case conversion
SELECT UPPER('hello');         -- HELLO
SELECT LOWER('HELLO');         -- hello
SELECT INITCAP('hello world'); -- Hello World (PostgreSQL)

-- Trimming
SELECT TRIM('  hello  ');       -- 'hello'
SELECT LTRIM('  hello');        -- 'hello'
SELECT RTRIM('hello  ');        -- 'hello'
SELECT TRIM(BOTH 'x' FROM 'xhellox');  -- 'hello'

-- Padding
SELECT LPAD('42', 5, '0');     -- '00042'
SELECT RPAD('hi', 5, '-');     -- 'hi---'

-- Substring extraction
SELECT SUBSTRING('Hello World', 1, 5);  -- 'Hello' (1-based index)
SELECT SUBSTR('Hello World', 7);        -- 'World'
SELECT LEFT('Hello World', 5);          -- 'Hello'
SELECT RIGHT('Hello World', 5);         -- 'World'
SELECT MID('Hello World', 7, 5);        -- 'World' (MySQL)

-- Position / Find
SELECT POSITION('World' IN 'Hello World');  -- 7
SELECT CHARINDEX('World', 'Hello World');   -- 7 (SQL Server)
SELECT STRPOS('Hello World', 'World');      -- 7 (PostgreSQL)
SELECT LOCATE('World', 'Hello World');      -- 7 (MySQL)
SELECT INSTR('Hello World', 'World');       -- 7 (MySQL / Oracle)

-- Concatenation
SELECT CONCAT('Hello', ' ', 'World');           -- 'Hello World'
SELECT 'Hello' || ' ' || 'World';               -- 'Hello World' (PostgreSQL)
SELECT CONCAT_WS(', ', 'Alice', 'Bob', 'Carol'); -- 'Alice, Bob, Carol'

-- Replace
SELECT REPLACE('Hello World', 'World', 'SQL');  -- 'Hello SQL'

-- Repeat
SELECT REPEAT('ab', 3);  -- 'ababab'

-- Reverse
SELECT REVERSE('Hello');  -- 'olleH'

-- Space
SELECT SPACE(5);  -- '     '

-- ASCII / CHAR
SELECT ASCII('A');   -- 65
SELECT CHAR(65);     -- 'A'

-- Soundex / Difference (for fuzzy matching)
SELECT SOUNDEX('Smith');   -- S530
SELECT DIFFERENCE('Smith', 'Smythe');  -- 4 (0-4, 4 = identical sound)

-- Format number as string
SELECT FORMAT(1234567.89, 2);  -- '1,234,567.89' (MySQL)
SELECT TO_CHAR(1234567.89, '9,999,999.99');  -- PostgreSQL

-- String aggregation
SELECT dept_id, GROUP_CONCAT(first_name ORDER BY first_name SEPARATOR ', ') AS names
FROM employees GROUP BY dept_id;   -- MySQL

SELECT dept_id, STRING_AGG(first_name, ', ' ORDER BY first_name) AS names
FROM employees GROUP BY dept_id;   -- PostgreSQL
```

---

## 15. Date and Time Functions

```sql
-- Current date/time
SELECT CURRENT_DATE;             -- date only
SELECT CURRENT_TIME;             -- time only
SELECT CURRENT_TIMESTAMP;        -- date + time (standard)
SELECT NOW();                    -- date + time (MySQL / PostgreSQL)
SELECT GETDATE();                -- SQL Server
SELECT SYSDATE;                  -- Oracle

-- Extract parts
SELECT YEAR(hire_date) FROM employees;        -- MySQL
SELECT MONTH(hire_date) FROM employees;
SELECT DAY(hire_date) FROM employees;
SELECT HOUR(created_at) FROM orders;
SELECT MINUTE(created_at) FROM orders;

-- PostgreSQL EXTRACT
SELECT EXTRACT(YEAR  FROM hire_date) FROM employees;
SELECT EXTRACT(MONTH FROM hire_date) FROM employees;
SELECT EXTRACT(DOW   FROM hire_date) FROM employees;  -- 0=Sunday
SELECT EXTRACT(EPOCH FROM hire_date) FROM employees;  -- Unix timestamp

-- PostgreSQL DATE_PART (same as EXTRACT)
SELECT DATE_PART('year', hire_date) FROM employees;

-- Date arithmetic
SELECT hire_date + INTERVAL 90 DAY AS probation_end FROM employees;   -- MySQL
SELECT hire_date + INTERVAL '90 days' AS probation_end FROM employees;-- PostgreSQL
SELECT DATEADD(DAY, 90, hire_date) AS probation_end FROM employees;   -- SQL Server

-- Date difference
SELECT DATEDIFF(NOW(), hire_date) AS days_employed FROM employees;    -- MySQL (days)
SELECT AGE(NOW(), hire_date) FROM employees;                          -- PostgreSQL
SELECT DATEDIFF(DAY, hire_date, GETDATE()) FROM employees;            -- SQL Server

-- Format date
SELECT DATE_FORMAT(hire_date, '%Y-%m-%d') FROM employees;    -- MySQL
SELECT TO_CHAR(hire_date, 'YYYY-MM-DD') FROM employees;      -- PostgreSQL
SELECT FORMAT(hire_date, 'yyyy-MM-dd') FROM employees;        -- SQL Server

-- Parse string to date
SELECT STR_TO_DATE('01/15/2024', '%m/%d/%Y');   -- MySQL
SELECT TO_DATE('01/15/2024', 'MM/DD/YYYY');      -- PostgreSQL / Oracle
SELECT CAST('2024-01-15' AS DATE);               -- Standard SQL

-- Truncate to period
SELECT DATE_TRUNC('month', hire_date) FROM employees;  -- PostgreSQL: first day of month
SELECT DATE_TRUNC('year',  hire_date) FROM employees;  -- First day of year
SELECT LAST_DAY(hire_date) FROM employees;              -- MySQL: last day of month

-- Day of week
SELECT DAYOFWEEK(hire_date) FROM employees;    -- MySQL: 1=Sunday
SELECT WEEKDAY(hire_date) FROM employees;      -- MySQL: 0=Monday
SELECT TO_CHAR(hire_date, 'Day') FROM employees; -- PostgreSQL: 'Monday'
```

---

## 16. Math Functions

```sql
-- Basic
SELECT ABS(-42);         -- 42
SELECT CEIL(4.3);        -- 5
SELECT CEILING(4.3);     -- 5
SELECT FLOOR(4.7);       -- 4
SELECT ROUND(4.567, 2);  -- 4.57
SELECT ROUND(4.565, 2);  -- 4.57 (banker's rounding may apply)
SELECT TRUNCATE(4.999, 2); -- 4.99 (MySQL)
SELECT TRUNC(4.999, 2);    -- 4.99 (PostgreSQL)

-- Power and roots
SELECT POWER(2, 10);     -- 1024
SELECT POW(2, 10);       -- 1024
SELECT SQRT(16);         -- 4
SELECT CBRT(27);         -- 3 (PostgreSQL: cube root)

-- Modulo
SELECT MOD(17, 5);       -- 2
SELECT 17 % 5;           -- 2

-- Logarithm
SELECT LOG(100);         -- 2 (base-10 log)
SELECT LOG10(100);       -- 2
SELECT LN(2.718);        -- ~1 (natural log)
SELECT LOG(2, 8);        -- 3 (log base 2 of 8)

-- Trigonometry (radians)
SELECT SIN(PI()/2);      -- 1
SELECT COS(0);           -- 1
SELECT TAN(PI()/4);      -- 1
SELECT DEGREES(PI());    -- 180
SELECT RADIANS(180);     -- PI

-- Random
SELECT RAND();           -- Random float 0 to 1 (MySQL)
SELECT RANDOM();         -- Random float 0 to 1 (PostgreSQL)
SELECT RAND() * 100;     -- Random 0-100
SELECT FLOOR(RAND() * 6) + 1;  -- Random integer 1-6

-- Sign
SELECT SIGN(-42);        -- -1
SELECT SIGN(0);          --  0
SELECT SIGN(42);         --  1

-- Greatest / Least
SELECT GREATEST(10, 20, 5);   -- 20
SELECT LEAST(10, 20, 5);      -- 5
```

---

## 17. Type Casting

```sql
-- CAST (SQL Standard)
SELECT CAST('42' AS INT);
SELECT CAST('3.14' AS DECIMAL(5,2));
SELECT CAST('2024-01-15' AS DATE);
SELECT CAST(salary AS VARCHAR(20));
SELECT CAST(is_active AS INT);       -- TRUE -> 1, FALSE -> 0

-- CONVERT (MySQL / SQL Server)
SELECT CONVERT(INT, '42');                          -- SQL Server
SELECT CONVERT('42', UNSIGNED INT);                 -- MySQL
SELECT CONVERT(salary, CHAR);                       -- MySQL
SELECT CONVERT(DATETIME, '2024-01-15 10:30:00');   -- SQL Server

-- PostgreSQL :: shorthand cast
SELECT '42'::INT;
SELECT '3.14'::DECIMAL;
SELECT '2024-01-15'::DATE;
SELECT salary::VARCHAR;

-- Implicit vs Explicit casting
SELECT '5' + 3;            -- 8 in MySQL (implicit), error in PostgreSQL
SELECT CAST('5' AS INT) + 3;  -- 8 (explicit, portable)

-- Safe casting (PostgreSQL)
SELECT TRY_CAST('abc' AS INT);  -- NULL instead of error (SQL Server)
-- PostgreSQL equivalent:
SELECT CASE WHEN '123' ~ '^\d+$' THEN '123'::INT ELSE NULL END;

-- Numeric formatting
SELECT TO_CHAR(salary, '$999,999.00') FROM employees;  -- PostgreSQL
SELECT FORMAT(salary, 'C', 'en-US') FROM employees;    -- SQL Server
```

---

## SELECT Quick Reference

```sql
-- Basic SELECT
SELECT col1, col2, expr AS alias FROM t WHERE condition;

-- Filtering
WHERE col = val
WHERE col != val / col <> val
WHERE col > val / col >= val / col < val / col <= val
WHERE col BETWEEN a AND b
WHERE col IN (v1, v2, v3)
WHERE col NOT IN (v1, v2)
WHERE col LIKE 'pattern%'
WHERE col IS NULL / IS NOT NULL
WHERE NOT condition
WHERE cond1 AND cond2
WHERE cond1 OR cond2

-- Sorting
ORDER BY col ASC / DESC
ORDER BY col1 ASC, col2 DESC
ORDER BY col NULLS LAST   -- PostgreSQL

-- Limiting
LIMIT n
LIMIT n OFFSET m
FETCH FIRST n ROWS ONLY   -- Standard SQL
TOP n                     -- SQL Server

-- Uniqueness
SELECT DISTINCT col FROM t

-- Conditional
CASE WHEN ... THEN ... ELSE ... END

-- NULL
COALESCE(a, b, c)   -- first non-NULL
NULLIF(a, b)        -- NULL if a = b
IFNULL(a, default)  -- MySQL
ISNULL(a, default)  -- SQL Server
```
