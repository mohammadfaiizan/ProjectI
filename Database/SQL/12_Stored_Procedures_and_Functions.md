# Stored Procedures and Functions

## Table of Contents
1. [Stored Procedures](#1-stored-procedures)
2. [User-Defined Functions (UDFs)](#2-user-defined-functions-udfs)
3. [Procedure vs Function](#3-procedure-vs-function)
4. [Variables and Control Flow](#4-variables-and-control-flow)
5. [Error Handling](#5-error-handling)
6. [Cursors](#6-cursors)
7. [PostgreSQL PL/pgSQL](#7-postgresql-plpgsql)
8. [MySQL Procedures and Functions](#8-mysql-procedures-and-functions)

---

## 1. Stored Procedures

A stored procedure is a precompiled set of SQL statements stored in the database that can be executed with a single call.

### Benefits
- Reduced network traffic (one call, many operations)
- Precompiled and cached (faster execution)
- Code reuse and centralization
- Security (grant EXECUTE, not table access)
- Encapsulation of business logic

### MySQL Stored Procedure Syntax
```sql
DELIMITER $$

CREATE PROCEDURE procedure_name([parameters])
BEGIN
    -- procedure body
    SQL statements;
END$$

DELIMITER ;
```

### Simple Procedure (No Parameters)
```sql
DELIMITER $$
CREATE PROCEDURE get_all_employees()
BEGIN
    SELECT id, first_name, last_name, salary, dept_id
    FROM employees
    WHERE is_active = TRUE
    ORDER BY last_name;
END$$
DELIMITER ;

-- Execute
CALL get_all_employees();
```

### Procedure with IN Parameters
```sql
DELIMITER $$
CREATE PROCEDURE get_employees_by_dept(IN p_dept_id INT)
BEGIN
    SELECT id, first_name, last_name, salary
    FROM employees
    WHERE dept_id = p_dept_id AND is_active = TRUE
    ORDER BY salary DESC;
END$$
DELIMITER ;

-- Execute
CALL get_employees_by_dept(10);
```

### Procedure with OUT Parameters
```sql
DELIMITER $$
CREATE PROCEDURE get_dept_salary_stats(
    IN  p_dept_id INT,
    OUT p_count   INT,
    OUT p_avg_sal DECIMAL(10,2),
    OUT p_total   DECIMAL(10,2)
)
BEGIN
    SELECT
        COUNT(*),
        AVG(salary),
        SUM(salary)
    INTO p_count, p_avg_sal, p_total
    FROM employees
    WHERE dept_id = p_dept_id AND is_active = TRUE;
END$$
DELIMITER ;

-- Execute and read output
CALL get_dept_salary_stats(10, @cnt, @avg, @total);
SELECT @cnt AS count, @avg AS avg_salary, @total AS total_payroll;
```

### Procedure with INOUT Parameters
```sql
DELIMITER $$
CREATE PROCEDURE apply_raise(INOUT p_salary DECIMAL(10,2), IN p_pct DECIMAL(5,2))
BEGIN
    SET p_salary = p_salary * (1 + p_pct / 100);
END$$
DELIMITER ;

-- Execute
SET @salary = 75000;
CALL apply_raise(@salary, 10);
SELECT @salary;  -- 82500
```

### Procedure with Multiple Result Sets
```sql
DELIMITER $$
CREATE PROCEDURE get_dept_summary(IN p_dept_id INT)
BEGIN
    -- First result set: department info
    SELECT * FROM departments WHERE id = p_dept_id;

    -- Second result set: employee list
    SELECT id, first_name, salary FROM employees WHERE dept_id = p_dept_id;

    -- Third result set: aggregate stats
    SELECT COUNT(*) AS cnt, AVG(salary) AS avg_sal
    FROM employees WHERE dept_id = p_dept_id;
END$$
DELIMITER ;
```

### Drop Procedure
```sql
DROP PROCEDURE IF EXISTS get_all_employees;
```

### List Procedures
```sql
-- MySQL
SHOW PROCEDURE STATUS WHERE Db = 'mydb';
SELECT ROUTINE_NAME, ROUTINE_TYPE, CREATED
FROM information_schema.ROUTINES
WHERE ROUTINE_SCHEMA = 'mydb' AND ROUTINE_TYPE = 'PROCEDURE';

-- PostgreSQL
SELECT proname, prosrc FROM pg_proc WHERE pronamespace = 'public'::regnamespace;
\df   -- psql command
```

---

## 2. User-Defined Functions (UDFs)

Functions return a value and can be used inside SQL expressions.

### Types of Functions
1. **Scalar Function**: Returns a single value
2. **Table-Valued Function**: Returns a table
3. **Aggregate Function**: Custom aggregate (PostgreSQL)

### MySQL Scalar Function
```sql
DELIMITER $$
CREATE FUNCTION calculate_bonus(p_salary DECIMAL(10,2), p_pct DECIMAL(5,2))
RETURNS DECIMAL(10,2)
DETERMINISTIC
BEGIN
    RETURN p_salary * p_pct / 100;
END$$
DELIMITER ;

-- Use in query
SELECT
    first_name,
    salary,
    calculate_bonus(salary, 10) AS bonus
FROM employees;
```

### MySQL Function Attributes
```sql
-- DETERMINISTIC: same inputs always return same output (allows caching)
-- NOT DETERMINISTIC (default): output may vary (e.g., uses NOW())
-- READS SQL DATA: reads from tables
-- MODIFIES SQL DATA: performs DML
-- NO SQL: no SQL statements

CREATE FUNCTION get_full_name(p_first VARCHAR(50), p_last VARCHAR(50))
RETURNS VARCHAR(101)
DETERMINISTIC
NO SQL
BEGIN
    RETURN CONCAT(p_first, ' ', p_last);
END;
```

### MySQL Table-Valued Function (via procedure workaround)
```sql
-- MySQL doesn't support table-valued functions directly
-- Use a temporary table approach in a procedure instead
DELIMITER $$
CREATE PROCEDURE get_dept_employees(IN p_dept_id INT)
BEGIN
    SELECT id, first_name, salary
    FROM employees
    WHERE dept_id = p_dept_id;
END$$
DELIMITER ;
```

### PostgreSQL Scalar Function
```sql
CREATE OR REPLACE FUNCTION calculate_annual_salary(p_monthly DECIMAL)
RETURNS DECIMAL AS $$
BEGIN
    RETURN p_monthly * 12;
END;
$$ LANGUAGE plpgsql;

-- Use in query
SELECT first_name, calculate_annual_salary(salary) AS annual
FROM employees;
```

### PostgreSQL Table-Returning Function
```sql
CREATE OR REPLACE FUNCTION get_dept_employees(p_dept_id INT)
RETURNS TABLE(id INT, name VARCHAR, salary DECIMAL) AS $$
BEGIN
    RETURN QUERY
    SELECT e.id, e.first_name || ' ' || e.last_name, e.salary
    FROM employees e
    WHERE e.dept_id = p_dept_id;
END;
$$ LANGUAGE plpgsql;

-- Use in FROM
SELECT * FROM get_dept_employees(10);
SELECT * FROM get_dept_employees(10) WHERE salary > 70000;
```

### PostgreSQL SETOF function
```sql
CREATE OR REPLACE FUNCTION get_high_earners(p_threshold DECIMAL)
RETURNS SETOF employees AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM employees WHERE salary > p_threshold;
END;
$$ LANGUAGE plpgsql;

SELECT id, first_name FROM get_high_earners(80000);
```

### SQL Server Scalar Function
```sql
CREATE FUNCTION dbo.GetFullName(@FirstName NVARCHAR(50), @LastName NVARCHAR(50))
RETURNS NVARCHAR(101)
AS
BEGIN
    RETURN @FirstName + ' ' + @LastName;
END;

-- Use
SELECT dbo.GetFullName(first_name, last_name) AS full_name FROM employees;
```

### SQL Server Inline Table-Valued Function
```sql
CREATE FUNCTION dbo.GetDeptEmployees(@DeptId INT)
RETURNS TABLE AS RETURN (
    SELECT id, first_name, last_name, salary
    FROM employees
    WHERE dept_id = @DeptId
);

-- Use
SELECT * FROM dbo.GetDeptEmployees(10) WHERE salary > 70000;
```

### SQL Server Multi-Statement Table-Valued Function
```sql
CREATE FUNCTION dbo.GetSalaryBands()
RETURNS @result TABLE (band VARCHAR(20), min_sal INT, max_sal INT, count INT)
AS
BEGIN
    INSERT INTO @result
    SELECT
        CASE
            WHEN salary < 50000  THEN 'Junior'
            WHEN salary < 80000  THEN 'Mid'
            WHEN salary < 120000 THEN 'Senior'
            ELSE 'Executive'
        END,
        MIN(salary), MAX(salary), COUNT(*)
    FROM employees
    GROUP BY CASE
        WHEN salary < 50000  THEN 'Junior'
        WHEN salary < 80000  THEN 'Mid'
        WHEN salary < 120000 THEN 'Senior'
        ELSE 'Executive'
    END;
    RETURN;
END;
```

### Drop Function
```sql
DROP FUNCTION IF EXISTS calculate_bonus;          -- MySQL
DROP FUNCTION IF EXISTS calculate_bonus(DECIMAL); -- PostgreSQL (specify arg types)
DROP FUNCTION dbo.GetFullName;                    -- SQL Server
```

---

## 3. Procedure vs Function

| Feature | Procedure | Function |
|---------|-----------|----------|
| Return value | OUT parameters or result sets | Returns a value (scalar or table) |
| Use in SQL | Called with CALL | Used in SELECT, WHERE, etc. |
| DML allowed | Yes (INSERT/UPDATE/DELETE) | Usually restricted |
| Transaction control | Yes | Usually no |
| Purpose | Perform actions | Compute and return values |

---

## 4. Variables and Control Flow

### Variables (MySQL)
```sql
DELIMITER $$
CREATE PROCEDURE demo_variables()
BEGIN
    -- Declare variables
    DECLARE v_count INT DEFAULT 0;
    DECLARE v_avg DECIMAL(10,2);
    DECLARE v_name VARCHAR(100);

    -- Assign with SET
    SET v_count = 5;
    SET v_name = 'Alice';

    -- Assign from query result
    SELECT COUNT(*), AVG(salary) INTO v_count, v_avg FROM employees;

    -- Use variables
    SELECT v_count AS total, v_avg AS average;
END$$
DELIMITER ;
```

### IF / ELSEIF / ELSE (MySQL)
```sql
DELIMITER $$
CREATE PROCEDURE classify_salary(IN p_salary DECIMAL(10,2))
BEGIN
    DECLARE v_band VARCHAR(20);

    IF p_salary < 40000 THEN
        SET v_band = 'Entry Level';
    ELSEIF p_salary < 70000 THEN
        SET v_band = 'Junior';
    ELSEIF p_salary < 100000 THEN
        SET v_band = 'Mid-Level';
    ELSEIF p_salary < 150000 THEN
        SET v_band = 'Senior';
    ELSE
        SET v_band = 'Executive';
    END IF;

    SELECT v_band AS salary_band;
END$$
DELIMITER ;
```

### CASE (MySQL)
```sql
DELIMITER $$
CREATE PROCEDURE get_day_type(IN p_date DATE)
BEGIN
    DECLARE v_weekday INT;
    DECLARE v_type VARCHAR(10);

    SET v_weekday = DAYOFWEEK(p_date);

    CASE v_weekday
        WHEN 1 THEN SET v_type = 'Sunday';
        WHEN 7 THEN SET v_type = 'Saturday';
        ELSE        SET v_type = 'Weekday';
    END CASE;

    SELECT v_type;
END$$
DELIMITER ;
```

### WHILE Loop (MySQL)
```sql
DELIMITER $$
CREATE PROCEDURE generate_numbers(IN p_max INT)
BEGIN
    DECLARE v_i INT DEFAULT 1;
    CREATE TEMPORARY TABLE IF NOT EXISTS numbers (n INT);

    WHILE v_i <= p_max DO
        INSERT INTO numbers VALUES (v_i);
        SET v_i = v_i + 1;
    END WHILE;

    SELECT * FROM numbers;
    DROP TEMPORARY TABLE numbers;
END$$
DELIMITER ;
```

### REPEAT ... UNTIL Loop (MySQL)
```sql
DELIMITER $$
CREATE PROCEDURE repeat_demo()
BEGIN
    DECLARE v_i INT DEFAULT 0;
    REPEAT
        SET v_i = v_i + 1;
    UNTIL v_i >= 10 END REPEAT;
    SELECT v_i;
END$$
DELIMITER ;
```

### LOOP with LEAVE (MySQL)
```sql
DELIMITER $$
CREATE PROCEDURE loop_demo()
BEGIN
    DECLARE v_i INT DEFAULT 0;

    my_loop: LOOP
        SET v_i = v_i + 1;
        IF v_i = 10 THEN
            LEAVE my_loop;
        END IF;
    END LOOP my_loop;

    SELECT v_i;
END$$
DELIMITER ;
```

---

## 5. Error Handling

### MySQL DECLARE HANDLER
```sql
DELIMITER $$
CREATE PROCEDURE safe_insert(IN p_name VARCHAR(100), IN p_email VARCHAR(100))
BEGIN
    DECLARE v_error INT DEFAULT 0;

    -- Declare a handler for duplicate key error
    DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
        SET v_error = 1;

    DECLARE CONTINUE HANDLER FOR 1062  -- Duplicate entry
        BEGIN
            SET v_error = 2;
        END;

    INSERT INTO employees (first_name, email) VALUES (p_name, p_email);

    IF v_error = 2 THEN
        SELECT 'Error: Email already exists' AS message;
    ELSEIF v_error = 1 THEN
        SELECT 'Error: Database error' AS message;
    ELSE
        SELECT 'Success' AS message;
    END IF;
END$$
DELIMITER ;
```

### MySQL SIGNAL (raise custom error)
```sql
DELIMITER $$
CREATE PROCEDURE validate_salary(IN p_salary DECIMAL(10,2))
BEGIN
    IF p_salary < 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Salary cannot be negative',
            MYSQL_ERRNO = 1644;
    END IF;

    SELECT 'Salary is valid' AS result;
END$$
DELIMITER ;
```

### PostgreSQL Exception Handling
```sql
CREATE OR REPLACE PROCEDURE safe_transfer(
    p_from_account INT,
    p_to_account   INT,
    p_amount       DECIMAL
) AS $$
BEGIN
    UPDATE accounts SET balance = balance - p_amount WHERE id = p_from_account;
    UPDATE accounts SET balance = balance + p_amount WHERE id = p_to_account;

    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE NOTICE 'Transfer failed: %', SQLERRM;
        RAISE;
END;
$$ LANGUAGE plpgsql;
```

---

## 6. Cursors

Cursors allow row-by-row processing of query results.

### MySQL Cursor
```sql
DELIMITER $$
CREATE PROCEDURE process_employees()
BEGIN
    DECLARE v_done INT DEFAULT 0;
    DECLARE v_id INT;
    DECLARE v_salary DECIMAL(10,2);

    -- Declare cursor
    DECLARE emp_cursor CURSOR FOR
        SELECT id, salary FROM employees WHERE is_active = TRUE;

    -- Handler for "no more rows"
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET v_done = 1;

    OPEN emp_cursor;

    read_loop: LOOP
        FETCH emp_cursor INTO v_id, v_salary;

        IF v_done THEN
            LEAVE read_loop;
        END IF;

        -- Process each row
        IF v_salary < 50000 THEN
            UPDATE employees SET salary = v_salary * 1.15 WHERE id = v_id;
        END IF;
    END LOOP;

    CLOSE emp_cursor;
END$$
DELIMITER ;
```

### PostgreSQL Cursor
```sql
CREATE OR REPLACE PROCEDURE process_with_cursor() AS $$
DECLARE
    emp_record RECORD;
    emp_cursor CURSOR FOR SELECT id, salary FROM employees WHERE is_active = TRUE;
BEGIN
    OPEN emp_cursor;
    LOOP
        FETCH emp_cursor INTO emp_record;
        EXIT WHEN NOT FOUND;

        IF emp_record.salary < 50000 THEN
            UPDATE employees SET salary = emp_record.salary * 1.15
            WHERE id = emp_record.id;
        END IF;
    END LOOP;
    CLOSE emp_cursor;
END;
$$ LANGUAGE plpgsql;
```

---

## 7. PostgreSQL PL/pgSQL

### Full PL/pgSQL Function Example
```sql
CREATE OR REPLACE FUNCTION calculate_department_budget(p_dept_id INT)
RETURNS JSONB AS $$
DECLARE
    v_dept_name  VARCHAR;
    v_headcount  INT;
    v_total_sal  DECIMAL;
    v_avg_sal    DECIMAL;
    v_result     JSONB;
BEGIN
    SELECT name INTO v_dept_name FROM departments WHERE id = p_dept_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Department % not found', p_dept_id;
    END IF;

    SELECT COUNT(*), SUM(salary), AVG(salary)
    INTO v_headcount, v_total_sal, v_avg_sal
    FROM employees
    WHERE dept_id = p_dept_id AND is_active = TRUE;

    v_result := jsonb_build_object(
        'department', v_dept_name,
        'headcount',  v_headcount,
        'total_salary', v_total_sal,
        'avg_salary', ROUND(v_avg_sal, 2)
    );

    RETURN v_result;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Error: %', SQLERRM;
        RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Usage
SELECT calculate_department_budget(10);
```

### PostgreSQL DO Block (Anonymous Procedure)
```sql
-- One-time execution without creating a persistent procedure
DO $$
DECLARE
    v_count INT;
BEGIN
    SELECT COUNT(*) INTO v_count FROM employees WHERE salary < 40000;
    RAISE NOTICE 'Low-salary employees: %', v_count;

    UPDATE employees SET salary = 40000 WHERE salary < 40000;
    RAISE NOTICE 'Updated % rows', v_count;
END;
$$ LANGUAGE plpgsql;
```

---

## 8. MySQL Procedures and Functions

### Batch Salary Update with Logging
```sql
DELIMITER $$
CREATE PROCEDURE apply_annual_raises()
BEGIN
    DECLARE v_emp_id INT;
    DECLARE v_old_salary DECIMAL(10,2);
    DECLARE v_new_salary DECIMAL(10,2);
    DECLARE v_dept_id INT;
    DECLARE v_done INT DEFAULT FALSE;

    DECLARE emp_cursor CURSOR FOR
        SELECT id, salary, dept_id FROM employees WHERE is_active = TRUE;
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET v_done = TRUE;

    START TRANSACTION;

    OPEN emp_cursor;
    process_loop: LOOP
        FETCH emp_cursor INTO v_emp_id, v_old_salary, v_dept_id;
        IF v_done THEN LEAVE process_loop; END IF;

        -- Calculate raise based on department
        SET v_new_salary = CASE v_dept_id
            WHEN 10 THEN v_old_salary * 1.12
            WHEN 20 THEN v_old_salary * 1.08
            ELSE         v_old_salary * 1.05
        END;

        UPDATE employees SET salary = v_new_salary WHERE id = v_emp_id;

        INSERT INTO salary_history (emp_id, old_salary, new_salary, change_date)
        VALUES (v_emp_id, v_old_salary, v_new_salary, CURRENT_DATE);
    END LOOP;
    CLOSE emp_cursor;

    COMMIT;
    SELECT ROW_COUNT() AS updated_rows;
END$$
DELIMITER ;
```

---

## Quick Reference

```sql
-- MySQL: Create procedure
DELIMITER $$
CREATE PROCEDURE name([IN|OUT|INOUT] param type, ...)
BEGIN ... END$$
DELIMITER ;

-- MySQL: Create function
DELIMITER $$
CREATE FUNCTION name(param type, ...) RETURNS type [DETERMINISTIC]
BEGIN ... RETURN value; END$$
DELIMITER ;

-- PostgreSQL: Create function
CREATE OR REPLACE FUNCTION name(param type) RETURNS type AS $$
BEGIN ... RETURN value; END; $$ LANGUAGE plpgsql;

-- PostgreSQL: Create procedure (13+)
CREATE OR REPLACE PROCEDURE name(param type) AS $$
BEGIN ... END; $$ LANGUAGE plpgsql;

-- Execute
CALL procedure_name(args);      -- Call procedure
SELECT function_name(args);     -- Use function in SQL

-- Variables
DECLARE var_name type [DEFAULT value];
SET var_name = value;
SELECT col INTO var_name FROM t WHERE ...;

-- Control flow
IF cond THEN ... ELSEIF cond THEN ... ELSE ... END IF;
CASE val WHEN x THEN ... END CASE;
WHILE cond DO ... END WHILE;
REPEAT ... UNTIL cond END REPEAT;
LOOP ... LEAVE label; ... END LOOP;

-- Error handling (MySQL)
DECLARE CONTINUE HANDLER FOR SQLEXCEPTION ...;
DECLARE EXIT HANDLER FOR 1062 ...;
SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'msg';

-- Drop
DROP PROCEDURE IF EXISTS name;
DROP FUNCTION  IF EXISTS name;
```
