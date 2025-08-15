"""
Python Context Managers: Advanced Context Protocol and Resource Management
Implementation-focused with minimal comments, maximum functionality coverage
"""

import contextlib
import tempfile
import os
import threading
import time
import sys
from typing import Any, Generator, Optional, IO
import sqlite3
import warnings
from pathlib import Path

# Basic context manager protocol
class FileManager:
    """Basic context manager for file operations"""
    
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        print(f"Opening {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Closing {self.filename}")
        if self.file:
            self.file.close()
        
        if exc_type:
            print(f"Exception occurred: {exc_type.__name__}: {exc_value}")
        
        # Return False to propagate exceptions
        return False

class DatabaseConnection:
    """Context manager with transaction support"""
    
    def __init__(self, db_path=":memory:"):
        self.db_path = db_path
        self.connection = None
        self.transaction_started = False
    
    def __enter__(self):
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("BEGIN")
        self.transaction_started = True
        return self.connection
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.connection:
            if exc_type:
                print("Rolling back transaction due to exception")
                self.connection.rollback()
            else:
                print("Committing transaction")
                self.connection.commit()
            self.connection.close()
        
        return False  # Don't suppress exceptions

def basic_context_manager_demo():
    # Create temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("Hello, World!")
        temp_filename = temp_file.name
    
    try:
        # Test file manager
        with FileManager(temp_filename, 'r') as file:
            content = file.read()
        
        # Test database connection
        with DatabaseConnection() as db:
            db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            db.execute("INSERT INTO users VALUES (1, 'Alice')")
            db.execute("INSERT INTO users VALUES (2, 'Bob')")
            
            cursor = db.execute("SELECT * FROM users")
            users = cursor.fetchall()
        
        # Test exception handling
        try:
            with DatabaseConnection() as db:
                db.execute("CREATE TABLE test (id INTEGER)")
                db.execute("INSERT INTO test VALUES (1)")
                raise ValueError("Simulated error")
        except ValueError:
            exception_handled = True
        
        return {
            "file_content": content,
            "database_users": users,
            "exception_handled": exception_handled
        }
    
    finally:
        os.unlink(temp_filename)

# Context manager with __call__ method
class TimingContext:
    """Context manager that measures execution time"""
    
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"Starting {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        print(f"{self.name} completed in {self.elapsed:.4f} seconds")
        return False
    
    def __call__(self, func):
        """Allow use as decorator"""
        def wrapper(*args, **kwargs):
            with TimingContext(f"Function {func.__name__}"):
                return func(*args, **kwargs)
        return wrapper

class ResourcePool:
    """Context manager for resource pooling"""
    
    def __init__(self, resource_factory, max_size=5):
        self.resource_factory = resource_factory
        self.max_size = max_size
        self.pool = []
        self.in_use = set()
        self.lock = threading.Lock()
    
    def acquire(self):
        with self.lock:
            if self.pool:
                resource = self.pool.pop()
            elif len(self.in_use) < self.max_size:
                resource = self.resource_factory()
            else:
                raise RuntimeError("Resource pool exhausted")
            
            self.in_use.add(resource)
            return resource
    
    def release(self, resource):
        with self.lock:
            if resource in self.in_use:
                self.in_use.remove(resource)
                self.pool.append(resource)
    
    def __enter__(self):
        self.resource = self.acquire()
        return self.resource
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.release(self.resource)
        return False

def advanced_context_manager_demo():
    # Test timing context
    timing_results = {}
    
    with TimingContext("Sleep operation") as timer:
        time.sleep(0.1)
    
    timing_results["sleep_time"] = timer.elapsed
    
    # Test timing as decorator
    @TimingContext("Calculation")
    def slow_calculation(n):
        return sum(i ** 2 for i in range(n))
    
    calc_result = slow_calculation(1000)
    
    # Test resource pool
    def create_connection():
        return f"Connection-{id(object())}"
    
    pool = ResourcePool(create_connection, max_size=2)
    
    pool_results = []
    with pool as conn1:
        pool_results.append(f"Using {conn1}")
        with pool as conn2:
            pool_results.append(f"Using {conn2}")
    
    return {
        "timing_results": timing_results,
        "calculation_result": calc_result,
        "pool_operations": pool_results
    }

# Contextlib utilities
@contextlib.contextmanager
def temporary_directory():
    """Generator-based context manager for temporary directory"""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        import shutil
        shutil.rmtree(temp_dir)

@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

@contextlib.contextmanager
def change_directory(path):
    """Context manager to temporarily change directory"""
    old_dir = os.getcwd()
    try:
        os.chdir(path)
        yield path
    finally:
        os.chdir(old_dir)

@contextlib.contextmanager
def timeout_context(seconds):
    """Context manager with timeout (simplified version)"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)  # Disable alarm
        signal.signal(signal.SIGALRM, old_handler)

def contextlib_demo():
    # Test temporary directory
    with temporary_directory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        file_exists_inside = os.path.exists(test_file)
        temp_dir_path = temp_dir
    
    file_exists_outside = os.path.exists(test_file)
    
    # Test stdout suppression
    print("This will be visible")
    with suppress_stdout():
        print("This will be suppressed")
    print("This will be visible again")
    
    # Test directory change
    original_dir = os.getcwd()
    with change_directory(os.path.expanduser("~")):
        home_dir = os.getcwd()
    current_dir = os.getcwd()
    
    return {
        "temporary_directory": {
            "temp_dir_path": temp_dir_path,
            "file_existed_inside": file_exists_inside,
            "file_exists_after": file_exists_outside,
            "cleanup_successful": not file_exists_outside
        },
        "directory_change": {
            "original_dir": original_dir,
            "home_dir": home_dir,
            "back_to_original": current_dir == original_dir
        }
    }

# Exception handling in context managers
class ErrorHandlingContext:
    """Context manager that handles specific exceptions"""
    
    def __init__(self, exception_types=(Exception,), default_return=None, log_errors=True):
        self.exception_types = exception_types
        self.default_return = default_return
        self.log_errors = log_errors
        self.exception_occurred = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type and issubclass(exc_type, self.exception_types):
            self.exception_occurred = (exc_type, exc_value, traceback)
            if self.log_errors:
                print(f"Handled exception: {exc_type.__name__}: {exc_value}")
            return True  # Suppress the exception
        return False

class RetryContext:
    """Context manager that retries operations"""
    
    def __init__(self, max_attempts=3, delay=1, backoff=2):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
        self.attempt = 0
        self.last_exception = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            self.attempt += 1
            self.last_exception = (exc_type, exc_value, traceback)
            
            if self.attempt < self.max_attempts:
                print(f"Attempt {self.attempt} failed, retrying in {self.delay}s...")
                time.sleep(self.delay)
                self.delay *= self.backoff
                return True  # Suppress exception and retry
            else:
                print(f"All {self.max_attempts} attempts failed")
                return False  # Let exception propagate
        return False

def exception_handling_demo():
    # Test error handling context
    with ErrorHandlingContext((ValueError, TypeError), default_return="Error occurred") as handler:
        raise ValueError("This will be handled")
    
    error_info = handler.exception_occurred
    
    # Test retry context
    retry_attempts = []
    
    class UnreliableOperation:
        def __init__(self):
            self.call_count = 0
        
        def execute(self):
            self.call_count += 1
            if self.call_count < 3:
                raise ConnectionError(f"Attempt {self.call_count} failed")
            return f"Success on attempt {self.call_count}"
    
    operation = UnreliableOperation()
    
    with RetryContext(max_attempts=4, delay=0.1) as retry:
        while True:
            try:
                result = operation.execute()
                break
            except ConnectionError:
                if retry.attempt >= retry.max_attempts - 1:
                    raise
    
    return {
        "error_handling": {
            "exception_type": error_info[0].__name__ if error_info else None,
            "exception_message": str(error_info[1]) if error_info else None,
            "handled_successfully": error_info is not None
        },
        "retry_mechanism": {
            "final_result": result,
            "total_attempts": operation.call_count,
            "success": "Success" in result
        }
    }

# Nested context managers
@contextlib.contextmanager
def nested_resource_manager():
    """Demonstrates proper nested resource management"""
    resources = []
    try:
        # Acquire multiple resources
        for i in range(3):
            resource = f"Resource-{i}"
            print(f"Acquiring {resource}")
            resources.append(resource)
        
        yield resources
    
    finally:
        # Release resources in reverse order
        for resource in reversed(resources):
            print(f"Releasing {resource}")

class TransactionContext:
    """Nested transaction context manager"""
    
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.committed = False
        self.rolled_back = False
        self.savepoint = None
    
    def __enter__(self):
        if self.parent:
            self.savepoint = f"savepoint_{self.name}"
            print(f"Creating savepoint {self.savepoint}")
        else:
            print(f"Starting transaction {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            if self.parent:
                print(f"Rolling back to savepoint {self.savepoint}")
                self.rolled_back = True
            else:
                print(f"Rolling back transaction {self.name}")
                self.rolled_back = True
        else:
            if self.parent:
                print(f"Releasing savepoint {self.savepoint}")
                self.committed = True
            else:
                print(f"Committing transaction {self.name}")
                self.committed = True
        
        return False

def nested_context_demo():
    # Test nested resource management
    resource_operations = []
    
    with nested_resource_manager() as resources:
        for resource in resources:
            resource_operations.append(f"Using {resource}")
    
    # Test nested transactions
    transaction_log = []
    
    with TransactionContext("main") as main_tx:
        transaction_log.append("Main transaction started")
        
        with TransactionContext("nested1", main_tx) as nested1:
            transaction_log.append("Nested transaction 1")
            
            with TransactionContext("nested2", nested1) as nested2:
                transaction_log.append("Nested transaction 2")
                # All succeed
    
    # Test nested transaction with failure
    try:
        with TransactionContext("main_with_error") as main_tx:
            with TransactionContext("nested_with_error", main_tx) as nested:
                raise ValueError("Simulated error in nested transaction")
    except ValueError:
        pass
    
    return {
        "resource_operations": resource_operations,
        "transaction_log": transaction_log,
        "nested_transaction_states": {
            "main_committed": main_tx.committed,
            "nested1_committed": nested1.committed,
            "nested2_committed": nested2.committed
        }
    }

# Async context managers (basic version)
class AsyncTimingContext:
    """Async context manager for timing"""
    
    def __init__(self, name="Async Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        print(f"Starting async {self.name}")
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.time() - self.start_time
        print(f"Async {self.name} completed in {self.elapsed:.4f} seconds")
        return False

# Custom context manager patterns
class StateManager:
    """Context manager that manages state changes"""
    
    def __init__(self, obj, **state_changes):
        self.obj = obj
        self.state_changes = state_changes
        self.original_state = {}
    
    def __enter__(self):
        # Save original state
        for attr in self.state_changes:
            if hasattr(self.obj, attr):
                self.original_state[attr] = getattr(self.obj, attr)
        
        # Apply new state
        for attr, value in self.state_changes.items():
            setattr(self.obj, attr, value)
        
        return self.obj
    
    def __exit__(self, exc_type, exc_value, traceback):
        # Restore original state
        for attr, value in self.original_state.items():
            setattr(self.obj, attr, value)
        return False

class ConfigurationContext:
    """Context manager for temporary configuration changes"""
    
    def __init__(self, config_dict, **changes):
        self.config = config_dict
        self.changes = changes
        self.original_values = {}
    
    def __enter__(self):
        # Save original values
        for key in self.changes:
            if key in self.config:
                self.original_values[key] = self.config[key]
        
        # Apply changes
        self.config.update(self.changes)
        return self.config
    
    def __exit__(self, exc_type, exc_value, traceback):
        # Restore original values
        for key, value in self.original_values.items():
            self.config[key] = value
        
        # Remove keys that weren't originally present
        for key in self.changes:
            if key not in self.original_values:
                self.config.pop(key, None)
        
        return False

def custom_patterns_demo():
    # Test state manager
    class TestObject:
        def __init__(self):
            self.value = 10
            self.name = "original"
    
    obj = TestObject()
    original_values = {"value": obj.value, "name": obj.name}
    
    with StateManager(obj, value=20, name="modified") as modified_obj:
        modified_values = {"value": modified_obj.value, "name": modified_obj.name}
    
    restored_values = {"value": obj.value, "name": obj.name}
    
    # Test configuration context
    config = {"debug": False, "level": "INFO"}
    original_config = config.copy()
    
    with ConfigurationContext(config, debug=True, level="DEBUG", new_key="value") as cfg:
        modified_config = cfg.copy()
    
    final_config = config.copy()
    
    return {
        "state_management": {
            "original_values": original_values,
            "modified_values": modified_values,
            "restored_values": restored_values,
            "restoration_successful": original_values == restored_values
        },
        "configuration_management": {
            "original_config": original_config,
            "modified_config": modified_config,
            "final_config": final_config,
            "restoration_successful": original_config == final_config
        }
    }

# Real-world context manager examples
@contextlib.contextmanager
def database_transaction(connection):
    """Real-world database transaction context manager"""
    transaction = connection.begin()
    try:
        yield connection
        transaction.commit()
    except Exception:
        transaction.rollback()
        raise

@contextlib.contextmanager
def file_lock(filename):
    """File locking context manager"""
    lock_file = f"{filename}.lock"
    
    # Simple file-based locking
    if os.path.exists(lock_file):
        raise RuntimeError(f"File {filename} is already locked")
    
    try:
        with open(lock_file, 'w') as f:
            f.write(str(os.getpid()))
        yield filename
    finally:
        try:
            os.remove(lock_file)
        except OSError:
            pass

@contextlib.contextmanager
def environment_variables(**env_vars):
    """Context manager for temporary environment variables"""
    original_values = {}
    
    # Save original values and set new ones
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = str(value)
    
    try:
        yield os.environ
    finally:
        # Restore original values
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

def real_world_demo():
    # Test file locking
    test_file = "test_lock_file.txt"
    
    try:
        with file_lock(test_file) as locked_file:
            with open(locked_file, 'w') as f:
                f.write("Locked file content")
            lock_successful = True
    except RuntimeError:
        lock_successful = False
    
    # Test environment variables
    original_path = os.environ.get("TEST_VAR", "not_set")
    
    with environment_variables(TEST_VAR="test_value", ANOTHER_VAR="another_value"):
        env_var_inside = os.environ.get("TEST_VAR")
        another_var_inside = os.environ.get("ANOTHER_VAR")
    
    env_var_after = os.environ.get("TEST_VAR", "not_set")
    another_var_after = os.environ.get("ANOTHER_VAR", "not_set")
    
    # Cleanup
    try:
        os.remove(test_file)
    except OSError:
        pass
    
    return {
        "file_locking": {
            "lock_successful": lock_successful,
            "file_created": os.path.exists(test_file)
        },
        "environment_variables": {
            "original_value": original_path,
            "value_inside_context": env_var_inside,
            "value_after_context": env_var_after,
            "another_var_inside": another_var_inside,
            "another_var_after": another_var_after,
            "restoration_successful": env_var_after == original_path
        }
    }

# Context manager best practices
class RobustContextManager:
    """Example of robust context manager implementation"""
    
    def __init__(self, resource_name, fail_on_enter=False, fail_on_exit=False):
        self.resource_name = resource_name
        self.fail_on_enter = fail_on_enter
        self.fail_on_exit = fail_on_exit
        self.resource = None
        self.entered = False
    
    def __enter__(self):
        if self.fail_on_enter:
            raise RuntimeError("Simulated enter failure")
        
        print(f"Acquiring resource: {self.resource_name}")
        self.resource = f"Resource({self.resource_name})"
        self.entered = True
        return self.resource
    
    def __exit__(self, exc_type, exc_value, traceback):
        if not self.entered:
            return False
        
        try:
            if self.fail_on_exit:
                raise RuntimeError("Simulated exit failure")
            
            print(f"Releasing resource: {self.resource_name}")
            self.resource = None
        except Exception as e:
            print(f"Error during cleanup: {e}")
            # Log error but don't suppress original exception
            if exc_type is None:
                # No original exception, so re-raise cleanup error
                raise
        
        # Return False to propagate any original exception
        return False

def best_practices_demo():
    # Test robust context manager with normal operation
    try:
        with RobustContextManager("normal_resource") as resource:
            normal_result = f"Used {resource}"
    except Exception as e:
        normal_result = f"Error: {e}"
    
    # Test context manager with enter failure
    try:
        with RobustContextManager("failing_resource", fail_on_enter=True) as resource:
            enter_fail_result = "This shouldn't execute"
    except RuntimeError as e:
        enter_fail_result = f"Enter failed: {e}"
    
    # Test context manager with exit failure
    try:
        with RobustContextManager("exit_fail_resource", fail_on_exit=True) as resource:
            exit_fail_resource = resource
    except RuntimeError as e:
        exit_fail_result = f"Exit failed: {e}"
    
    return {
        "normal_operation": normal_result,
        "enter_failure": enter_fail_result,
        "exit_failure": exit_fail_result,
        "best_practices": [
            "Always implement both __enter__ and __exit__",
            "Handle exceptions in __exit__ properly",
            "Use try/finally in __exit__ for cleanup",
            "Consider whether to suppress exceptions",
            "Log errors during cleanup",
            "Make context managers reusable when possible",
            "Use @contextlib.contextmanager for simple cases",
            "Test exception scenarios thoroughly"
        ]
    }

# Comprehensive testing
def run_all_context_manager_demos():
    """Execute all context manager demonstrations"""
    demo_functions = [
        ('basic_context_managers', basic_context_manager_demo),
        ('advanced_context_managers', advanced_context_manager_demo),
        ('contextlib_utilities', contextlib_demo),
        ('exception_handling', exception_handling_demo),
        ('nested_contexts', nested_context_demo),
        ('custom_patterns', custom_patterns_demo),
        ('real_world_examples', real_world_demo),
        ('best_practices', best_practices_demo)
    ]
    
    results = {}
    for name, func in demo_functions:
        try:
            result = func()
            results[name] = result
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    print("=== Python Context Managers Demo ===")
    
    # Run all demonstrations
    all_results = run_all_context_manager_demos()
    
    for category, data in all_results.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        
        if 'error' in data:
            print(f"  Error: {data['error']}")
            continue
        
        # Display results
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and len(value) > 3:
                    print(f"  {key}: {dict(list(value.items())[:3])}... (truncated)")
                elif isinstance(value, list) and len(value) > 5:
                    print(f"  {key}: {value[:5]}... (showing first 5)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Result: {data}")
    
    print("\n=== CONTEXT MANAGER PATTERNS ===")
    
    patterns = {
        "Basic Protocol": "__enter__ returns resource, __exit__ handles cleanup",
        "Generator-based": "@contextmanager decorator with yield statement",
        "Exception Handling": "__exit__ receives exception info and can suppress",
        "Nested Contexts": "Multiple context managers, proper resource ordering",
        "State Management": "Temporarily change object state and restore",
        "Resource Pooling": "Manage limited resources with acquire/release",
        "Transaction Support": "Database-style commit/rollback semantics",
        "Async Contexts": "__aenter__ and __aexit__ for async operations"
    }
    
    for pattern, description in patterns.items():
        print(f"  {pattern}: {description}")
    
    print("\n=== CONTEXTLIB UTILITIES ===")
    
    utilities = {
        "@contextmanager": "Decorator to create context manager from generator function",
        "closing()": "Ensure an object is closed when leaving the context",
        "suppress()": "Suppress specified exceptions within the context",
        "redirect_stdout()": "Redirect stdout to a file-like object",
        "redirect_stderr()": "Redirect stderr to a file-like object",
        "nullcontext()": "No-op context manager for conditional contexts",
        "ExitStack": "Manage multiple context managers dynamically"
    }
    
    for utility, description in utilities.items():
        print(f"  {utility}: {description}")
    
    print("\n=== BEST PRACTICES ===")
    
    best_practices = [
        "Always implement proper exception handling in __exit__",
        "Use @contextlib.contextmanager for simple context managers",
        "Consider whether exceptions should be suppressed",
        "Ensure proper resource cleanup even when exceptions occur",
        "Use try/finally blocks within __exit__ for critical cleanup",
        "Make context managers reusable when possible",
        "Document the behavior of your context managers",
        "Test both normal and exceptional flows",
        "Use nested context managers for complex resource management",
        "Consider async context managers for async operations"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== Context Managers Complete! ===")
    print("  Advanced context protocols and resource management mastered")
