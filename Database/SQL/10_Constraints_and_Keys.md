# Constraints and Keys

## Table of Contents
1. [PRIMARY KEY](#1-primary-key)
2. [FOREIGN KEY](#2-foreign-key)
3. [UNIQUE Constraint](#3-unique-constraint)
4. [NOT NULL Constraint](#4-not-null-constraint)
5. [CHECK Constraint](#5-check-constraint)
6. [DEFAULT Constraint](#6-default-constraint)
7. [Managing Constraints](#7-managing-constraints)
8. [Referential Actions](#8-referential-actions)
9. [Constraint Best Practices](#9-constraint-best-practices)

---

## 1. PRIMARY KEY

The primary key uniquely identifies each row in a table. It cannot be NULL and must be unique.

### Column-Level
```sql
CREATE TABLE employees (
    id INT PRIMARY KEY,
    name VARCHAR(100)
);

-- With auto-increment (MySQL)
CREATE TABLE employees (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100)
);

-- PostgreSQL SERIAL
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

-- PostgreSQL IDENTITY (SQL standard)
CREATE TABLE employees (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(100)
);
```

### Table-Level (Composite Primary Key)
```sql
CREATE TABLE order_items (
    order_id   INT NOT NULL,
    product_id INT NOT NULL,
    quantity   INT NOT NULL,
    PRIMARY KEY (order_id, product_id)
);

-- Named constraint
CREATE TABLE order_items (
    order_id   INT NOT NULL,
    product_id INT NOT NULL,
    CONSTRAINT pk_order_items PRIMARY KEY (order_id, product_id)
);
```

### Natural vs Surrogate Keys

| Type | Description | Example |
|------|-------------|---------|
| Natural key | Business meaningful | email, SSN, ISBN |
| Surrogate key | System-generated, no business meaning | AUTO_INCREMENT id, UUID |

```sql
-- Surrogate key (recommended for most cases)
CREATE TABLE customers (
    id    INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(100) UNIQUE NOT NULL
);

-- Natural key (use when it's guaranteed unique and stable)
CREATE TABLE countries (
    code CHAR(2) PRIMARY KEY,
    name VARCHAR(100)
);
```

### UUID as Primary Key
```sql
-- PostgreSQL
CREATE TABLE users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name VARCHAR(100)
);

-- MySQL
CREATE TABLE users (
    id CHAR(36) DEFAULT (UUID()) PRIMARY KEY,  -- MySQL 8.0+
    name VARCHAR(100)
);
```

### Add Primary Key to Existing Table
```sql
ALTER TABLE employees ADD PRIMARY KEY (id);
ALTER TABLE employees ADD CONSTRAINT pk_employees PRIMARY KEY (id);
```

### Drop Primary Key
```sql
ALTER TABLE employees DROP PRIMARY KEY;              -- MySQL
ALTER TABLE employees DROP CONSTRAINT pk_employees;  -- PostgreSQL / SQL Server
```

---

## 2. FOREIGN KEY

A foreign key enforces referential integrity between tables.

### Basic Foreign Key
```sql
CREATE TABLE employees (
    id        INT PRIMARY KEY,
    dept_id   INT,
    FOREIGN KEY (dept_id) REFERENCES departments(id)
);

-- Named constraint
CREATE TABLE employees (
    id      INT PRIMARY KEY,
    dept_id INT,
    CONSTRAINT fk_emp_dept FOREIGN KEY (dept_id) REFERENCES departments(id)
);
```

### Column-Level Foreign Key
```sql
CREATE TABLE employees (
    id      INT PRIMARY KEY,
    dept_id INT REFERENCES departments(id)  -- Inline syntax
);
```

### Composite Foreign Key
```sql
CREATE TABLE shipment_items (
    shipment_id  INT,
    product_id   INT,
    warehouse_id INT,
    FOREIGN KEY (product_id, warehouse_id)
        REFERENCES inventory (product_id, warehouse_id)
);
```

### Add Foreign Key to Existing Table
```sql
ALTER TABLE employees
ADD CONSTRAINT fk_emp_dept
FOREIGN KEY (dept_id) REFERENCES departments(id);
```

### Drop Foreign Key
```sql
ALTER TABLE employees DROP FOREIGN KEY fk_emp_dept;           -- MySQL
ALTER TABLE employees DROP CONSTRAINT fk_emp_dept;            -- PostgreSQL / SQL Server
```

---

## 3. UNIQUE Constraint

Ensures all values in a column (or combination of columns) are distinct.

### Column-Level UNIQUE
```sql
CREATE TABLE employees (
    id    INT PRIMARY KEY,
    email VARCHAR(100) UNIQUE
);
```

### Table-Level UNIQUE
```sql
CREATE TABLE employees (
    id         INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name  VARCHAR(50),
    email      VARCHAR(100),
    UNIQUE (email),
    UNIQUE (first_name, last_name)  -- Combination must be unique
);

-- Named constraint
CREATE TABLE employees (
    id    INT PRIMARY KEY,
    email VARCHAR(100),
    CONSTRAINT uq_employee_email UNIQUE (email)
);
```

### NULL in UNIQUE
```sql
-- Most databases (PostgreSQL, SQL Server, Oracle): multiple NULLs are allowed
-- NULLs are not considered equal to each other in UNIQUE constraints
-- Exception: MySQL treats multiple NULLs as a violation in UNIQUE

INSERT INTO employees (email) VALUES (NULL);  -- Allowed in PostgreSQL
INSERT INTO employees (email) VALUES (NULL);  -- Also allowed (second NULL)
```

### Add UNIQUE Constraint
```sql
ALTER TABLE employees ADD UNIQUE (email);
ALTER TABLE employees ADD CONSTRAINT uq_email UNIQUE (email);
```

### Drop UNIQUE Constraint
```sql
ALTER TABLE employees DROP INDEX uq_email;           -- MySQL
ALTER TABLE employees DROP CONSTRAINT uq_email;      -- PostgreSQL / SQL Server
```

---

## 4. NOT NULL Constraint

Prevents NULL values from being inserted in a column.

### Inline NOT NULL
```sql
CREATE TABLE employees (
    id         INT NOT NULL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name  VARCHAR(50) NOT NULL,
    email      VARCHAR(100) NOT NULL,
    salary     DECIMAL(10,2)     -- NULLable (no NOT NULL)
);
```

### Add NOT NULL to Existing Column
```sql
-- First ensure no existing NULLs, then add constraint
UPDATE employees SET email = 'unknown@company.com' WHERE email IS NULL;

-- MySQL
ALTER TABLE employees MODIFY COLUMN email VARCHAR(100) NOT NULL;

-- PostgreSQL
ALTER TABLE employees ALTER COLUMN email SET NOT NULL;

-- SQL Server
ALTER TABLE employees ALTER COLUMN email VARCHAR(100) NOT NULL;
```

### Remove NOT NULL Constraint
```sql
-- MySQL
ALTER TABLE employees MODIFY COLUMN email VARCHAR(100) NULL;

-- PostgreSQL
ALTER TABLE employees ALTER COLUMN email DROP NOT NULL;
```

---

## 5. CHECK Constraint

Validates that column values satisfy a specified condition.

### Column-Level CHECK
```sql
CREATE TABLE employees (
    id      INT PRIMARY KEY,
    salary  DECIMAL(10,2) CHECK (salary >= 0),
    age     INT CHECK (age BETWEEN 18 AND 120)
);
```

### Table-Level CHECK (can reference multiple columns)
```sql
CREATE TABLE events (
    id         INT PRIMARY KEY,
    start_date DATE NOT NULL,
    end_date   DATE NOT NULL,
    CHECK (end_date >= start_date)
);

-- Named CHECK constraint
CREATE TABLE products (
    id         INT PRIMARY KEY,
    price      DECIMAL(10,2),
    sale_price DECIMAL(10,2),
    CONSTRAINT chk_price        CHECK (price > 0),
    CONSTRAINT chk_sale_price   CHECK (sale_price >= 0),
    CONSTRAINT chk_price_logic  CHECK (sale_price <= price)
);
```

### CHECK with IN for Enumeration
```sql
CREATE TABLE orders (
    id     INT PRIMARY KEY,
    status VARCHAR(20),
    CONSTRAINT chk_status CHECK (status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled'))
);

-- PostgreSQL alternative: use ENUM type
CREATE TYPE order_status AS ENUM ('pending', 'processing', 'shipped', 'delivered', 'cancelled');
CREATE TABLE orders (
    id     INT PRIMARY KEY,
    status order_status NOT NULL DEFAULT 'pending'
);
```

### CHECK with Regular Expression (PostgreSQL)
```sql
CREATE TABLE employees (
    email VARCHAR(100),
    phone VARCHAR(20),
    CONSTRAINT chk_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT chk_phone CHECK (phone ~ '^\d{3}-\d{3}-\d{4}$')
);
```

### Add CHECK Constraint
```sql
ALTER TABLE employees
ADD CONSTRAINT chk_salary CHECK (salary >= 0);

-- PostgreSQL: check existing data too
ALTER TABLE employees
ADD CONSTRAINT chk_salary CHECK (salary >= 0) NOT VALID;  -- Skip historical data check
ALTER TABLE employees VALIDATE CONSTRAINT chk_salary;      -- Validate later
```

### Drop CHECK Constraint
```sql
ALTER TABLE employees DROP CHECK chk_salary;           -- MySQL
ALTER TABLE employees DROP CONSTRAINT chk_salary;      -- PostgreSQL / SQL Server
```

---

## 6. DEFAULT Constraint

Provides a default value when no value is specified on INSERT.

### Inline DEFAULT
```sql
CREATE TABLE employees (
    id         INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    is_active  BOOLEAN     DEFAULT TRUE,
    salary     DECIMAL(10,2) DEFAULT 0.00,
    hire_date  DATE          DEFAULT CURRENT_DATE,
    created_at TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP  -- MySQL
);
```

### DEFAULT with Expression (PostgreSQL)
```sql
CREATE TABLE orders (
    id           SERIAL PRIMARY KEY,
    order_number VARCHAR(20) DEFAULT 'ORD-' || nextval('order_seq'),
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    expires_at   TIMESTAMPTZ DEFAULT NOW() + INTERVAL '30 days'
);
```

### Add DEFAULT to Existing Column
```sql
-- MySQL
ALTER TABLE employees MODIFY COLUMN is_active BOOLEAN DEFAULT TRUE;

-- PostgreSQL
ALTER TABLE employees ALTER COLUMN is_active SET DEFAULT TRUE;

-- SQL Server
ALTER TABLE employees ADD CONSTRAINT df_is_active DEFAULT (1) FOR is_active;
```

### Remove DEFAULT
```sql
-- MySQL
ALTER TABLE employees MODIFY COLUMN is_active BOOLEAN;

-- PostgreSQL
ALTER TABLE employees ALTER COLUMN is_active DROP DEFAULT;

-- SQL Server
ALTER TABLE employees DROP CONSTRAINT df_is_active;
```

### DEFAULT with INSERT
```sql
-- Using default values
INSERT INTO employees (first_name) VALUES ('Alice');  -- Uses defaults for other columns
INSERT INTO employees (first_name, is_active) VALUES ('Bob', DEFAULT);  -- Explicit DEFAULT

-- Using DEFAULT keyword
INSERT INTO employees (first_name, salary, is_active)
VALUES ('Carol', DEFAULT, TRUE);
```

---

## 7. Managing Constraints

### List All Constraints

```sql
-- MySQL: information_schema
SELECT
    CONSTRAINT_NAME,
    CONSTRAINT_TYPE,
    TABLE_NAME
FROM information_schema.TABLE_CONSTRAINTS
WHERE TABLE_SCHEMA = 'mydb' AND TABLE_NAME = 'employees';

-- MySQL: key constraints
SELECT
    TABLE_NAME, CONSTRAINT_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
FROM information_schema.KEY_COLUMN_USAGE
WHERE TABLE_SCHEMA = 'mydb';

-- PostgreSQL
SELECT
    conname AS constraint_name,
    contype AS type,  -- p=primary, f=foreign, u=unique, c=check
    pg_get_constraintdef(oid) AS definition
FROM pg_constraint
WHERE conrelid = 'employees'::regclass;

-- SQL Server
SELECT
    name AS constraint_name,
    type_desc AS constraint_type,
    OBJECT_NAME(parent_object_id) AS table_name
FROM sys.objects
WHERE type_desc LIKE '%CONSTRAINT%' AND OBJECT_NAME(parent_object_id) = 'employees';
```

### Enable / Disable Constraints

```sql
-- SQL Server: disable constraint
ALTER TABLE employees NOCHECK CONSTRAINT fk_emp_dept;
ALTER TABLE employees NOCHECK CONSTRAINT ALL;

-- SQL Server: re-enable
ALTER TABLE employees CHECK CONSTRAINT fk_emp_dept;
ALTER TABLE employees CHECK CONSTRAINT ALL;

-- MySQL: disable foreign key checks
SET FOREIGN_KEY_CHECKS = 0;
-- ... do operations ...
SET FOREIGN_KEY_CHECKS = 1;

-- PostgreSQL: defer constraint to end of transaction
SET CONSTRAINTS fk_emp_dept DEFERRED;
SET CONSTRAINTS ALL DEFERRED;
SET CONSTRAINTS fk_emp_dept IMMEDIATE;
```

### Deferrable Constraints (PostgreSQL)

```sql
-- Constraint checked at end of transaction, not each statement
CREATE TABLE employees (
    id      INT PRIMARY KEY,
    mgr_id  INT,
    CONSTRAINT fk_manager FOREIGN KEY (mgr_id) REFERENCES employees(id)
        DEFERRABLE INITIALLY DEFERRED
);

-- Now you can insert manager and employee in same transaction (any order)
BEGIN;
INSERT INTO employees (id, mgr_id) VALUES (2, 1);  -- mgr_id=1 doesn't exist yet
INSERT INTO employees (id, mgr_id) VALUES (1, NULL); -- Now it exists
COMMIT;  -- Constraint checked here — passes
```

---

## 8. Referential Actions

Defines what happens to child rows when a parent row is updated or deleted.

### ON DELETE Options

| Action | Description |
|--------|-------------|
| `RESTRICT` / `NO ACTION` | Error if parent row has children (default) |
| `CASCADE` | Delete/update child rows automatically |
| `SET NULL` | Set foreign key column to NULL |
| `SET DEFAULT` | Set foreign key column to its default value |

```sql
CREATE TABLE orders (
    id          INT PRIMARY KEY,
    customer_id INT,

    -- When customer is deleted, delete their orders too
    CONSTRAINT fk_orders_customer
        FOREIGN KEY (customer_id) REFERENCES customers(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE order_items (
    id       INT PRIMARY KEY,
    order_id INT,

    -- When order is deleted, set order_id to NULL
    CONSTRAINT fk_items_order
        FOREIGN KEY (order_id) REFERENCES orders(id)
        ON DELETE SET NULL
);

CREATE TABLE employees (
    id         INT PRIMARY KEY,
    manager_id INT,

    -- When manager is deleted, manager_id becomes NULL
    CONSTRAINT fk_emp_mgr
        FOREIGN KEY (manager_id) REFERENCES employees(id)
        ON DELETE SET NULL
        ON UPDATE CASCADE
);
```

### Cascade Example
```sql
-- departments → employees (CASCADE)
-- If we delete Engineering department, all its employees are deleted too
DELETE FROM departments WHERE name = 'Engineering';
-- Automatically deletes all employees with dept_id = Engineering's id
```

---

## 9. Constraint Best Practices

### Naming Conventions
```sql
-- pk_tablename
CONSTRAINT pk_employees PRIMARY KEY (id)

-- fk_child_parent
CONSTRAINT fk_orders_customers FOREIGN KEY (customer_id) REFERENCES customers(id)

-- uq_tablename_column
CONSTRAINT uq_employees_email UNIQUE (email)

-- chk_tablename_rule
CONSTRAINT chk_employees_salary CHECK (salary >= 0)

-- df_tablename_column
CONSTRAINT df_employees_active DEFAULT TRUE FOR is_active
```

### Always Name Constraints
```sql
-- Bad: anonymous constraint (hard to drop/reference)
salary DECIMAL(10,2) CHECK (salary >= 0)

-- Good: named constraint
salary DECIMAL(10,2),
CONSTRAINT chk_salary_positive CHECK (salary >= 0)
```

### Foreign Key Indexes
```sql
-- Always create an index on foreign key columns
-- Without it, lookups from parent → child are full scans
CREATE TABLE orders (
    id          INT PRIMARY KEY,
    customer_id INT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
CREATE INDEX idx_orders_customer ON orders (customer_id);
```

### Defer Validation for Bulk Loads
```sql
-- PostgreSQL: temporarily skip constraint checks for bulk load
BEGIN;
SET CONSTRAINTS ALL DEFERRED;
-- ... bulk insert ...
COMMIT;  -- All constraints checked at commit
```

---

## Constraints Quick Reference

```sql
-- PRIMARY KEY
id INT PRIMARY KEY                                      -- Column level
PRIMARY KEY (col1, col2)                                -- Table level (composite)
CONSTRAINT pk_name PRIMARY KEY (id)                     -- Named

-- FOREIGN KEY
dept_id INT REFERENCES departments(id)                  -- Inline
FOREIGN KEY (dept_id) REFERENCES departments(id)        -- Table level
CONSTRAINT fk_name FOREIGN KEY (col) REFERENCES t(col) ON DELETE CASCADE

-- UNIQUE
email VARCHAR(100) UNIQUE                               -- Column level
UNIQUE (email)                                          -- Table level
CONSTRAINT uq_name UNIQUE (col1, col2)                  -- Named composite

-- NOT NULL
col VARCHAR(100) NOT NULL                               -- Column level

-- CHECK
salary DECIMAL CHECK (salary >= 0)                      -- Column level
CHECK (end_date >= start_date)                          -- Table level (multi-col)
CONSTRAINT chk_name CHECK (condition)                   -- Named

-- DEFAULT
col BOOLEAN DEFAULT TRUE                                -- Inline
ALTER TABLE t ALTER COLUMN col SET DEFAULT value        -- PostgreSQL
ALTER TABLE t MODIFY COLUMN col TYPE DEFAULT value      -- MySQL

-- ALTER TABLE: add / drop constraints
ALTER TABLE t ADD CONSTRAINT name PRIMARY KEY (col);
ALTER TABLE t ADD CONSTRAINT name FOREIGN KEY (col) REFERENCES t2(col);
ALTER TABLE t ADD CONSTRAINT name UNIQUE (col);
ALTER TABLE t ADD CONSTRAINT name CHECK (condition);
ALTER TABLE t DROP CONSTRAINT name;
ALTER TABLE t DROP PRIMARY KEY;              -- MySQL
ALTER TABLE t DROP FOREIGN KEY fk_name;     -- MySQL
```
