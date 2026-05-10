# Triggers

## Table of Contents
1. [What is a Trigger?](#1-what-is-a-trigger)
2. [Trigger Syntax](#2-trigger-syntax)
3. [BEFORE Triggers](#3-before-triggers)
4. [AFTER Triggers](#4-after-triggers)
5. [INSTEAD OF Triggers](#5-instead-of-triggers)
6. [NEW and OLD Reference Tables](#6-new-and-old-reference-tables)
7. [Statement-Level vs Row-Level Triggers](#7-statement-level-vs-row-level-triggers)
8. [Managing Triggers](#8-managing-triggers)
9. [Common Trigger Patterns](#9-common-trigger-patterns)

---

## 1. What is a Trigger?

A trigger is a stored procedure that automatically executes in response to specific DML events (INSERT, UPDATE, DELETE) on a table.

### Trigger Timing
- **BEFORE**: Executes before the DML operation (can modify NEW values or abort the operation)
- **AFTER**: Executes after the DML operation (good for auditing, cascade operations)
- **INSTEAD OF**: Replaces the DML operation (used on views — makes non-updatable views writable)

### Trigger Events
- `INSERT` — fires on INSERT
- `UPDATE` — fires on UPDATE
- `DELETE` — fires on DELETE
- `UPDATE OF column` — fires only when specific column is updated (PostgreSQL)

### When to Use Triggers
- **Audit logs**: Record who changed what and when
- **Derived values**: Auto-compute columns
- **Validation**: Enforce complex business rules
- **Cascade operations**: Maintain denormalized data
- **Notifications**: Queue events for external systems

### When NOT to Use Triggers
- Simple cascades — use FOREIGN KEY with ON DELETE CASCADE
- Simple defaults — use DEFAULT constraint
- Complex business logic — keep in application layer
- Performance-critical paths — triggers add overhead

---

## 2. Trigger Syntax

### MySQL Trigger Syntax
```sql
CREATE TRIGGER trigger_name
{BEFORE | AFTER} {INSERT | UPDATE | DELETE}
ON table_name
FOR EACH ROW
[trigger_order]
BEGIN
    trigger_body;
END;

-- trigger_order: FOLLOWS | PRECEDES other_trigger_name
```

### PostgreSQL Trigger Syntax
```sql
-- Step 1: Create the trigger function
CREATE OR REPLACE FUNCTION trigger_function_name()
RETURNS TRIGGER AS $$
BEGIN
    -- trigger body
    RETURN NEW;  -- or RETURN OLD; or RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Step 2: Create the trigger
CREATE TRIGGER trigger_name
{BEFORE | AFTER | INSTEAD OF} {INSERT | UPDATE | DELETE}
ON table_name
FOR EACH {ROW | STATEMENT}
[WHEN (condition)]
EXECUTE FUNCTION trigger_function_name();
```

### SQL Server Trigger Syntax
```sql
CREATE TRIGGER trigger_name
ON table_name
{AFTER | INSTEAD OF} {INSERT [,] UPDATE [,] DELETE}
AS
BEGIN
    -- trigger body using INSERTED and DELETED pseudo-tables
END;
```

---

## 3. BEFORE Triggers

BEFORE triggers fire before the row change. They can:
- Modify the NEW row values
- Raise an error to abort the operation
- Perform validation

### MySQL BEFORE INSERT
```sql
DELIMITER $$
CREATE TRIGGER before_employee_insert
BEFORE INSERT ON employees
FOR EACH ROW
BEGIN
    -- Auto-capitalize first name
    SET NEW.first_name = CONCAT(UPPER(SUBSTRING(NEW.first_name, 1, 1)),
                                LOWER(SUBSTRING(NEW.first_name, 2)));

    -- Set created_at timestamp
    SET NEW.created_at = NOW();

    -- Validate salary
    IF NEW.salary < 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Salary cannot be negative';
    END IF;

    -- Default dept_id if not provided
    IF NEW.dept_id IS NULL THEN
        SET NEW.dept_id = 1;  -- Default department
    END IF;
END$$
DELIMITER ;
```

### MySQL BEFORE UPDATE
```sql
DELIMITER $$
CREATE TRIGGER before_employee_update
BEFORE UPDATE ON employees
FOR EACH ROW
BEGIN
    -- Track the last update time
    SET NEW.updated_at = NOW();

    -- Prevent salary decrease below original
    IF NEW.salary < OLD.salary * 0.9 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Cannot reduce salary by more than 10%';
    END IF;

    -- Prevent changing primary key
    IF NEW.id != OLD.id THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Cannot change employee ID';
    END IF;
END$$
DELIMITER ;
```

### MySQL BEFORE DELETE
```sql
DELIMITER $$
CREATE TRIGGER before_employee_delete
BEFORE DELETE ON employees
FOR EACH ROW
BEGIN
    -- Check if employee has pending tasks
    IF EXISTS (SELECT 1 FROM tasks WHERE assignee_id = OLD.id AND status = 'pending') THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Cannot delete employee with pending tasks';
    END IF;
END$$
DELIMITER ;
```

### PostgreSQL BEFORE Trigger
```sql
CREATE OR REPLACE FUNCTION normalize_employee_data()
RETURNS TRIGGER AS $$
BEGIN
    -- Normalize data before insert or update
    NEW.first_name := INITCAP(TRIM(NEW.first_name));
    NEW.last_name  := INITCAP(TRIM(NEW.last_name));
    NEW.email      := LOWER(TRIM(NEW.email));
    NEW.updated_at := NOW();

    -- Validation
    IF NEW.salary < 0 THEN
        RAISE EXCEPTION 'Salary cannot be negative: %', NEW.salary;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_normalize_employee
BEFORE INSERT OR UPDATE ON employees
FOR EACH ROW
EXECUTE FUNCTION normalize_employee_data();
```

---

## 4. AFTER Triggers

AFTER triggers fire after the row change. Used for:
- Audit logging
- Updating related tables
- Sending notifications

### MySQL AFTER INSERT
```sql
DELIMITER $$
CREATE TRIGGER after_order_insert
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    -- Update customer's last_order_date
    UPDATE customers
    SET
        last_order_date = NEW.order_date,
        order_count = order_count + 1
    WHERE id = NEW.customer_id;

    -- Log the event
    INSERT INTO order_events (order_id, event_type, event_data, created_at)
    VALUES (NEW.id, 'ORDER_CREATED', CONCAT('Amount: ', NEW.total), NOW());
END$$
DELIMITER ;
```

### MySQL AFTER UPDATE — Audit Log
```sql
DELIMITER $$
CREATE TRIGGER after_salary_update
AFTER UPDATE ON employees
FOR EACH ROW
BEGIN
    -- Only log if salary changed
    IF OLD.salary != NEW.salary THEN
        INSERT INTO salary_history (
            employee_id,
            old_salary,
            new_salary,
            changed_by,
            changed_at
        )
        VALUES (
            NEW.id,
            OLD.salary,
            NEW.salary,
            USER(),
            NOW()
        );
    END IF;
END$$
DELIMITER ;
```

### MySQL AFTER DELETE
```sql
DELIMITER $$
CREATE TRIGGER after_employee_delete
AFTER DELETE ON employees
FOR EACH ROW
BEGIN
    -- Archive deleted employee
    INSERT INTO employees_archive (
        id, first_name, last_name, email, salary, dept_id, deleted_at, deleted_by
    )
    VALUES (
        OLD.id, OLD.first_name, OLD.last_name, OLD.email,
        OLD.salary, OLD.dept_id, NOW(), USER()
    );

    -- Update department headcount
    UPDATE departments
    SET employee_count = employee_count - 1
    WHERE id = OLD.dept_id;
END$$
DELIMITER ;
```

### PostgreSQL AFTER Trigger with Audit
```sql
-- Generic audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log (
        table_name,
        operation,
        old_data,
        new_data,
        changed_by,
        changed_at
    )
    VALUES (
        TG_TABLE_NAME,
        TG_OP,
        CASE WHEN TG_OP = 'DELETE' THEN row_to_json(OLD) ELSE NULL END,
        CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN row_to_json(NEW) ELSE NULL END,
        current_user,
        NOW()
    );

    RETURN NULL;  -- AFTER trigger, return value ignored for row triggers
END;
$$ LANGUAGE plpgsql;

-- Attach to multiple tables
CREATE TRIGGER audit_employees
AFTER INSERT OR UPDATE OR DELETE ON employees
FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_orders
AFTER INSERT OR UPDATE OR DELETE ON orders
FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
```

---

## 5. INSTEAD OF Triggers

Used exclusively on views. Replaces the DML operation on the view with custom logic.

### Making a View Updatable via INSTEAD OF Trigger
```sql
-- This view is not naturally updatable (has JOIN)
CREATE VIEW employee_dept_view AS
SELECT
    e.id,
    e.first_name,
    e.last_name,
    e.salary,
    d.name AS dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.id;

-- SQL Server INSTEAD OF trigger
CREATE TRIGGER tr_instead_of_update
ON employee_dept_view
INSTEAD OF UPDATE
AS
BEGIN
    -- Handle updates to employee columns
    UPDATE e
    SET
        e.first_name = i.first_name,
        e.last_name  = i.last_name,
        e.salary     = i.salary
    FROM employees e
    JOIN INSERTED i ON e.id = i.id;

    -- Handle department name change
    UPDATE d
    SET d.name = i.dept_name
    FROM departments d
    JOIN INSERTED i ON d.name = (SELECT dept_name FROM DELETED WHERE id = i.id);
END;
```

### PostgreSQL INSTEAD OF Trigger
```sql
CREATE OR REPLACE FUNCTION update_employee_view()
RETURNS TRIGGER AS $$
DECLARE
    v_dept_id INT;
BEGIN
    IF TG_OP = 'INSERT' THEN
        SELECT id INTO v_dept_id FROM departments WHERE name = NEW.dept_name;
        INSERT INTO employees (first_name, last_name, salary, dept_id)
        VALUES (NEW.first_name, NEW.last_name, NEW.salary, v_dept_id);
        RETURN NEW;

    ELSIF TG_OP = 'UPDATE' THEN
        SELECT id INTO v_dept_id FROM departments WHERE name = NEW.dept_name;
        UPDATE employees
        SET first_name = NEW.first_name, last_name = NEW.last_name,
            salary = NEW.salary, dept_id = v_dept_id
        WHERE id = OLD.id;
        RETURN NEW;

    ELSIF TG_OP = 'DELETE' THEN
        DELETE FROM employees WHERE id = OLD.id;
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_instead_of
INSTEAD OF INSERT OR UPDATE OR DELETE ON employee_dept_view
FOR EACH ROW EXECUTE FUNCTION update_employee_view();
```

---

## 6. NEW and OLD Reference Tables

### MySQL: NEW and OLD
```sql
-- NEW: the new row (available in INSERT and UPDATE)
-- OLD: the old row (available in UPDATE and DELETE)

CREATE TRIGGER example_trigger
AFTER UPDATE ON employees
FOR EACH ROW
BEGIN
    -- OLD.salary: value before update
    -- NEW.salary: value after update
    IF NEW.salary != OLD.salary THEN
        INSERT INTO salary_log VALUES (NEW.id, OLD.salary, NEW.salary, NOW());
    END IF;
END;
```

| Operation | NEW available | OLD available |
|-----------|--------------|--------------|
| INSERT | Yes (new values) | No |
| UPDATE | Yes (new values) | Yes (old values) |
| DELETE | No | Yes (deleted values) |

### PostgreSQL: NEW and OLD Records
```sql
CREATE OR REPLACE FUNCTION check_update()
RETURNS TRIGGER AS $$
BEGIN
    -- Both NEW and OLD are RECORD types
    IF NEW.salary < OLD.salary THEN
        RAISE EXCEPTION 'Cannot reduce salary from % to %', OLD.salary, NEW.salary;
    END IF;

    -- TG_OP: 'INSERT', 'UPDATE', or 'DELETE'
    -- TG_TABLE_NAME: name of the table
    -- TG_WHEN: 'BEFORE' or 'AFTER'

    RETURN NEW;  -- Must return NEW for BEFORE row trigger
                 -- Return NULL to cancel the operation
END;
$$ LANGUAGE plpgsql;
```

### SQL Server: INSERTED and DELETED Tables
```sql
CREATE TRIGGER tr_salary_audit
ON employees
AFTER UPDATE
AS
BEGIN
    -- INSERTED: new values (or inserted rows)
    -- DELETED: old values (or deleted rows)
    IF UPDATE(salary)  -- Check if salary column was updated
    BEGIN
        INSERT INTO salary_audit (emp_id, old_salary, new_salary, changed_at)
        SELECT d.id, d.salary, i.salary, GETDATE()
        FROM DELETED d
        JOIN INSERTED i ON d.id = i.id
        WHERE d.salary != i.salary;
    END
END;
```

---

## 7. Statement-Level vs Row-Level Triggers

### Row-Level Triggers (FOR EACH ROW)
Fires once per affected row. Has access to NEW and OLD.
```sql
-- MySQL: always row-level (only option)
-- PostgreSQL:
CREATE TRIGGER trg_row_level
AFTER UPDATE ON employees
FOR EACH ROW
EXECUTE FUNCTION my_function();
-- Fires 5 times if 5 rows updated
```

### Statement-Level Triggers (FOR EACH STATEMENT)
Fires once per SQL statement, regardless of rows affected. No NEW/OLD access.
```sql
-- PostgreSQL only
CREATE TRIGGER trg_stmt_level
AFTER UPDATE ON employees
FOR EACH STATEMENT
EXECUTE FUNCTION log_bulk_update();
-- Fires once even if 1000 rows updated

-- Use case: log that a bulk operation occurred (not individual rows)
CREATE OR REPLACE FUNCTION log_bulk_update()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO operation_log (table_name, operation, performed_by, performed_at)
    VALUES (TG_TABLE_NAME, TG_OP, current_user, NOW());
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;
```

---

## 8. Managing Triggers

### List Triggers
```sql
-- MySQL
SHOW TRIGGERS FROM mydb;
SHOW TRIGGERS FROM mydb LIKE 'employees%';

SELECT TRIGGER_NAME, EVENT_MANIPULATION, EVENT_OBJECT_TABLE,
       ACTION_TIMING, ACTION_STATEMENT
FROM information_schema.TRIGGERS
WHERE TRIGGER_SCHEMA = 'mydb';

-- PostgreSQL
SELECT tgname AS trigger_name, relname AS table_name,
       pg_get_triggerdef(t.oid) AS definition
FROM pg_trigger t
JOIN pg_class c ON t.tgrelid = c.oid
WHERE NOT t.tgisinternal;

\d tablename  -- Shows triggers in psql

-- SQL Server
SELECT t.name AS trigger_name, o.name AS table_name,
       t.is_disabled, t.is_instead_of_trigger
FROM sys.triggers t
JOIN sys.objects o ON t.parent_id = o.object_id;
```

### Drop Trigger
```sql
DROP TRIGGER trigger_name ON table_name;      -- PostgreSQL
DROP TRIGGER trigger_name;                    -- MySQL (trigger name is unique per schema)
DROP TRIGGER IF EXISTS trigger_name;          -- Both
```

### Enable / Disable Trigger
```sql
-- PostgreSQL
ALTER TABLE employees DISABLE TRIGGER trg_audit;
ALTER TABLE employees ENABLE  TRIGGER trg_audit;
ALTER TABLE employees DISABLE TRIGGER ALL;
ALTER TABLE employees ENABLE  TRIGGER ALL;

-- SQL Server
DISABLE TRIGGER tr_salary_audit ON employees;
ENABLE  TRIGGER tr_salary_audit ON employees;

-- MySQL: no direct enable/disable; must drop and recreate
```

---

## 9. Common Trigger Patterns

### Auto-Update Timestamp
```sql
-- MySQL
DELIMITER $$
CREATE TRIGGER set_updated_at
BEFORE UPDATE ON employees
FOR EACH ROW
BEGIN
    SET NEW.updated_at = NOW();
END$$
DELIMITER ;

-- PostgreSQL
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_set_updated_at
BEFORE UPDATE ON employees
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
```

### Audit Trail
```sql
-- PostgreSQL: comprehensive audit
CREATE OR REPLACE FUNCTION full_audit()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit (tbl, op, new_row, actor, ts)
        VALUES (TG_TABLE_NAME, 'I', row_to_json(NEW), current_user, NOW());
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit (tbl, op, old_row, new_row, actor, ts)
        VALUES (TG_TABLE_NAME, 'U', row_to_json(OLD), row_to_json(NEW), current_user, NOW());
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit (tbl, op, old_row, actor, ts)
        VALUES (TG_TABLE_NAME, 'D', row_to_json(OLD), current_user, NOW());
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

### Maintain Derived/Denormalized Column
```sql
-- Keep running order total in customer table
DELIMITER $$
CREATE TRIGGER after_order_change
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    UPDATE customers
    SET total_orders = total_orders + 1,
        total_spent  = total_spent + NEW.amount
    WHERE id = NEW.customer_id;
END$$
DELIMITER ;
```

### Enforce Business Rule
```sql
-- Cannot reassign order to different customer after shipping
DELIMITER $$
CREATE TRIGGER enforce_order_rules
BEFORE UPDATE ON orders
FOR EACH ROW
BEGIN
    IF OLD.status = 'shipped' AND NEW.customer_id != OLD.customer_id THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Cannot change customer on a shipped order';
    END IF;
END$$
DELIMITER ;
```

---

## Quick Reference

```sql
-- MySQL: Create trigger
CREATE TRIGGER name
{BEFORE|AFTER} {INSERT|UPDATE|DELETE}
ON table FOR EACH ROW
BEGIN ... END;

-- PostgreSQL: Create trigger function + trigger
CREATE OR REPLACE FUNCTION fname() RETURNS TRIGGER AS $$
BEGIN
    -- For BEFORE: RETURN NEW (proceed) or RETURN NULL (cancel)
    -- For AFTER: RETURN NULL (ignored)
    RETURN NEW;
END; $$ LANGUAGE plpgsql;

CREATE TRIGGER name
{BEFORE|AFTER|INSTEAD OF} {INSERT|UPDATE|DELETE}
ON table FOR EACH {ROW|STATEMENT}
EXECUTE FUNCTION fname();

-- SQL Server
CREATE TRIGGER name ON table
AFTER {INSERT,UPDATE,DELETE}
AS BEGIN ... END;

-- NEW/OLD references (MySQL, PostgreSQL)
NEW.column_name  -- New value (INSERT, UPDATE)
OLD.column_name  -- Old value (UPDATE, DELETE)

-- INSERTED/DELETED (SQL Server)
SELECT * FROM INSERTED;  -- New values
SELECT * FROM DELETED;   -- Old values

-- PostgreSQL trigger variables
TG_OP         -- 'INSERT', 'UPDATE', 'DELETE'
TG_TABLE_NAME -- Name of table
TG_WHEN       -- 'BEFORE', 'AFTER', 'INSTEAD OF'

-- Drop
DROP TRIGGER name ON table;    -- PostgreSQL
DROP TRIGGER name;             -- MySQL

-- Disable/enable (PostgreSQL)
ALTER TABLE t DISABLE TRIGGER name;
ALTER TABLE t ENABLE TRIGGER name;
```
