# SQL Introduction

## Table of Contents
1. [What is SQL?](#1-what-is-sql)
2. [History of SQL](#2-history-of-sql)
3. [RDBMS Concepts](#3-rdbms-concepts)
4. [SQL Categories](#4-sql-categories)
5. [Data Types](#5-data-types)
6. [NULL in SQL](#6-null-in-sql)
7. [SQL Execution Order](#7-sql-execution-order)
8. [Naming Conventions](#8-naming-conventions)

---

## 1. What is SQL?

**SQL (Structured Query Language)** is a domain-specific language used to manage and manipulate relational databases. It is the standard language for Relational Database Management Systems (RDBMS).

SQL allows you to:
- Create, modify, and delete database structures (tables, views, indexes)
- Insert, update, and delete data
- Query and retrieve data
- Control access to data

---

## 2. History of SQL

| Year | Milestone |
|------|-----------|
| 1970 | E.F. Codd publishes "A Relational Model of Data for Large Shared Data Banks" |
| 1974 | IBM develops SEQUEL (Structured English Query Language) |
| 1979 | Oracle releases first commercial SQL RDBMS |
| 1986 | ANSI publishes first SQL standard (SQL-86) |
| 1992 | SQL-92 (SQL2) — major revision |
| 1999 | SQL:1999 (SQL3) — adds procedural elements, triggers |
| 2003 | SQL:2003 — adds XML support, window functions |
| 2008 | SQL:2008 — TRUNCATE, FETCH FIRST |
| 2011 | SQL:2011 — temporal data |
| 2016 | SQL:2016 — JSON support |

---

## 3. RDBMS Concepts

### Database
A structured collection of data organized into tables.

### Table
A collection of related data stored in rows and columns.
```
employees
+----+----------+---------+------------+
| id | name     | dept_id | salary     |
+----+----------+---------+------------+
|  1 | Alice    |    10   |   75000    |
|  2 | Bob      |    20   |   80000    |
|  3 | Charlie  |    10   |   90000    |
+----+----------+---------+------------+
```

### Row (Record/Tuple)
A single entry in a table. Each row represents one instance of the entity.

### Column (Field/Attribute)
A specific data attribute of the table. All values in a column share the same data type.

### Schema
A logical container that groups related database objects (tables, views, procedures).

### Relationships
Tables can be related via keys:
- **One-to-One**: One row in Table A maps to one row in Table B
- **One-to-Many**: One row in Table A maps to many rows in Table B
- **Many-to-Many**: Many rows in Table A map to many rows in Table B (requires junction table)

### Popular RDBMS Systems

| System | Notes |
|--------|-------|
| MySQL | Open-source, widely used for web apps |
| PostgreSQL | Open-source, advanced features, ACID compliant |
| Microsoft SQL Server | Enterprise-grade, Windows-focused |
| Oracle Database | Enterprise, high performance |
| SQLite | Lightweight, file-based, embedded |
| MariaDB | MySQL fork, open-source |

---

## 4. SQL Categories

SQL commands are grouped into five categories:

### DDL — Data Definition Language
Commands that define or modify database structure.
```sql
CREATE, ALTER, DROP, TRUNCATE, RENAME
```

### DML — Data Manipulation Language
Commands that manipulate data within tables.
```sql
INSERT, UPDATE, DELETE, MERGE
```

### DQL — Data Query Language
Commands that retrieve data.
```sql
SELECT
```

### DCL — Data Control Language
Commands that control access/permissions.
```sql
GRANT, REVOKE
```

### TCL — Transaction Control Language
Commands that manage transactions.
```sql
COMMIT, ROLLBACK, SAVEPOINT, SET TRANSACTION
```

---

## 5. Data Types

### Numeric Types

| Type | Description | Range |
|------|-------------|-------|
| `TINYINT` | Very small integer | -128 to 127 |
| `SMALLINT` | Small integer | -32,768 to 32,767 |
| `MEDIUMINT` | Medium integer | -8,388,608 to 8,388,607 |
| `INT` / `INTEGER` | Standard integer | -2,147,483,648 to 2,147,483,647 |
| `BIGINT` | Large integer | -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807 |
| `DECIMAL(p,s)` / `NUMERIC(p,s)` | Exact fixed-point | p = precision, s = scale |
| `FLOAT` | Approximate single-precision | 4 bytes |
| `DOUBLE` / `REAL` | Approximate double-precision | 8 bytes |

```sql
-- Examples
salary DECIMAL(10, 2)   -- Up to 99999999.99
percentage FLOAT        -- 3.14159
count INT               -- Whole numbers
```

### String / Character Types

| Type | Description |
|------|-------------|
| `CHAR(n)` | Fixed-length string, padded with spaces |
| `VARCHAR(n)` | Variable-length string, up to n characters |
| `TEXT` | Variable-length, large text (no length limit) |
| `TINYTEXT` | Up to 255 characters |
| `MEDIUMTEXT` | Up to 16,777,215 characters |
| `LONGTEXT` | Up to 4,294,967,295 characters |
| `BINARY(n)` | Fixed-length binary |
| `VARBINARY(n)` | Variable-length binary |
| `BLOB` | Binary Large Object |

```sql
-- Examples
first_name VARCHAR(50)
country_code CHAR(2)    -- Always 2 chars: 'US', 'IN'
description TEXT
```

### Date and Time Types

| Type | Format | Example |
|------|--------|---------|
| `DATE` | YYYY-MM-DD | 2024-01-15 |
| `TIME` | HH:MM:SS | 14:30:00 |
| `DATETIME` | YYYY-MM-DD HH:MM:SS | 2024-01-15 14:30:00 |
| `TIMESTAMP` | YYYY-MM-DD HH:MM:SS | 2024-01-15 14:30:00 UTC |
| `YEAR` | YYYY | 2024 |
| `INTERVAL` | varies | '1 year 3 months' (PostgreSQL) |

```sql
-- Examples
birth_date DATE
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
updated_at DATETIME ON UPDATE CURRENT_TIMESTAMP
```

### Boolean Type

| Type | Description |
|------|-------------|
| `BOOLEAN` / `BOOL` | True/False (stored as TINYINT 0/1 in MySQL) |
| `BIT(n)` | Bit-field, stores n bits |

```sql
is_active BOOLEAN DEFAULT TRUE
```

### JSON Type (Modern SQL)
```sql
-- PostgreSQL / MySQL 5.7+
metadata JSON
settings JSONB  -- PostgreSQL binary JSON (faster querying)
```

### UUID Type
```sql
-- PostgreSQL
id UUID DEFAULT gen_random_uuid()

-- MySQL
id CHAR(36)  -- Stored as string
```

### PostgreSQL-Specific Types

| Type | Description |
|------|-------------|
| `SERIAL` | Auto-incrementing integer |
| `BIGSERIAL` | Auto-incrementing bigint |
| `ARRAY` | Array of any type |
| `JSONB` | Binary JSON |
| `HSTORE` | Key-value store |
| `INET` | IP address |
| `CIDR` | IP network |
| `UUID` | Universally unique identifier |

---

## 6. NULL in SQL

`NULL` represents the absence of a value — it is NOT the same as zero, empty string, or false.

### NULL Behavior
```sql
-- NULL comparisons always return NULL (not TRUE or FALSE)
SELECT NULL = NULL;     -- NULL (not TRUE)
SELECT NULL != NULL;    -- NULL
SELECT NULL > 5;        -- NULL

-- Use IS NULL / IS NOT NULL
SELECT * FROM employees WHERE manager_id IS NULL;
SELECT * FROM employees WHERE manager_id IS NOT NULL;

-- NULL in arithmetic
SELECT 5 + NULL;    -- NULL
SELECT 5 * NULL;    -- NULL

-- NULL in aggregations (NULLs are ignored by aggregate functions)
SELECT AVG(salary) FROM employees;  -- NULLs excluded from average

-- COALESCE — returns first non-NULL value
SELECT COALESCE(phone, email, 'No contact') FROM customers;

-- NULLIF — returns NULL if two values are equal
SELECT NULLIF(salary, 0) FROM employees;  -- Returns NULL if salary = 0

-- IFNULL (MySQL) / NVL (Oracle) — replace NULL
SELECT IFNULL(commission, 0) FROM employees;
```

### Three-Valued Logic
SQL uses three-valued logic: TRUE, FALSE, NULL (UNKNOWN)

| A | B | A AND B | A OR B |
|---|---|---------|--------|
| TRUE | TRUE | TRUE | TRUE |
| TRUE | FALSE | FALSE | TRUE |
| TRUE | NULL | NULL | TRUE |
| FALSE | NULL | FALSE | NULL |
| NULL | NULL | NULL | NULL |

---

## 7. SQL Execution Order

SQL queries are **written** in one order but **executed** in a different order:

### Written Order
```sql
SELECT    col1, col2
FROM      table
JOIN      other_table ON condition
WHERE     filter_condition
GROUP BY  col1
HAVING    group_condition
ORDER BY  col1
LIMIT     n;
```

### Execution Order
```
1. FROM          -- Identify source tables
2. JOIN          -- Combine tables
3. WHERE         -- Filter rows (before grouping)
4. GROUP BY      -- Group rows
5. HAVING        -- Filter groups (after grouping)
6. SELECT        -- Select columns / compute expressions
7. DISTINCT      -- Remove duplicates
8. ORDER BY      -- Sort results
9. LIMIT/OFFSET  -- Limit output rows
```

This matters because:
- You cannot use a `SELECT` alias in a `WHERE` clause (WHERE runs before SELECT)
- You CAN use a `SELECT` alias in `ORDER BY` (ORDER BY runs after SELECT)
- `HAVING` filters groups; `WHERE` filters rows

```sql
-- This FAILS — alias used in WHERE (WHERE runs before SELECT)
SELECT salary * 12 AS annual_salary
FROM employees
WHERE annual_salary > 100000;  -- ERROR

-- This WORKS
SELECT salary * 12 AS annual_salary
FROM employees
WHERE salary * 12 > 100000;

-- This WORKS — alias in ORDER BY
SELECT salary * 12 AS annual_salary
FROM employees
ORDER BY annual_salary DESC;
```

---

## 8. Naming Conventions

### Best Practices

| Element | Convention | Example |
|---------|-----------|---------|
| Table names | Lowercase, snake_case, plural | `employees`, `order_items` |
| Column names | Lowercase, snake_case | `first_name`, `created_at` |
| Primary key | `id` or `table_id` | `id`, `employee_id` |
| Foreign key | `referenced_table_id` | `department_id` |
| Index names | `idx_table_column` | `idx_employees_email` |
| Constraint names | `pk_`, `fk_`, `uq_`, `chk_` | `pk_employees`, `fk_emp_dept` |
| Stored procedures | Verb + noun | `get_employee`, `update_salary` |

### Reserved Words
Avoid using SQL reserved words as identifiers. If necessary, quote them:
```sql
-- Bad
SELECT order, select FROM table;

-- Good (quoted identifiers)
SELECT "order", "select" FROM "table";

-- Or use backticks (MySQL)
SELECT `order`, `select` FROM `table`;
```

---

## Quick Reference: Common SQL Data Type Mappings

| Concept | MySQL | PostgreSQL | SQL Server |
|---------|-------|------------|------------|
| Auto-increment | `AUTO_INCREMENT` | `SERIAL` / `GENERATED` | `IDENTITY` |
| String | `VARCHAR(n)` | `VARCHAR(n)` | `VARCHAR(n)` / `NVARCHAR(n)` |
| Large text | `TEXT` | `TEXT` | `VARCHAR(MAX)` |
| Current time | `NOW()` | `NOW()` / `CURRENT_TIMESTAMP` | `GETDATE()` |
| Boolean | `TINYINT(1)` | `BOOLEAN` | `BIT` |
| JSON | `JSON` | `JSON` / `JSONB` | `NVARCHAR(MAX)` |
| Unique ID | `CHAR(36)` | `UUID` | `UNIQUEIDENTIFIER` |
