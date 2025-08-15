"""
Python File Handling and Exception Management
Implementation-focused with minimal comments, maximum functionality coverage
"""

import os
import json
import csv
import pickle
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import time
import sys

# Basic file operations
def basic_file_operations():
    # Create temporary directory for demonstrations
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Writing files
        file_path = os.path.join(temp_dir, "sample.txt")
        
        # Different write modes
        with open(file_path, 'w') as f:
            f.write("Hello, World!\n")
            f.write("Python file handling\n")
        
        # Append mode
        with open(file_path, 'a') as f:
            f.write("Appended line\n")
        
        # Reading files
        read_results = {}
        
        # Read entire file
        with open(file_path, 'r') as f:
            read_results['full_content'] = f.read()
        
        # Read lines
        with open(file_path, 'r') as f:
            read_results['lines_list'] = f.readlines()
        
        # Read line by line
        with open(file_path, 'r') as f:
            read_results['first_line'] = f.readline().strip()
        
        # File iteration
        with open(file_path, 'r') as f:
            read_results['line_count'] = sum(1 for line in f)
        
        # Binary file operations
        binary_path = os.path.join(temp_dir, "binary.dat")
        binary_data = b"Binary data example"
        
        with open(binary_path, 'wb') as f:
            f.write(binary_data)
        
        with open(binary_path, 'rb') as f:
            read_results['binary_content'] = f.read()
        
        return {
            'write_operations': 'Files written successfully',
            'read_results': read_results,
            'file_exists': os.path.exists(file_path),
            'file_size': os.path.getsize(file_path)
        }
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

def file_modes_and_encoding():
    temp_dir = tempfile.mkdtemp()
    
    try:
        file_path = os.path.join(temp_dir, "encoding_test.txt")
        
        # Different encodings
        text_unicode = "Hello 世界 🌍 Héllo"
        
        encoding_results = {}
        
        # UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_unicode)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            encoding_results['utf8_read'] = f.read()
        
        # File position operations
        position_file = os.path.join(temp_dir, "position.txt")
        with open(position_file, 'w') as f:
            f.write("0123456789ABCDEF")
        
        with open(position_file, 'r') as f:
            encoding_results['position_start'] = f.tell()
            f.read(5)
            encoding_results['position_after_read'] = f.tell()
            f.seek(10)
            encoding_results['after_seek'] = f.read(3)
            f.seek(0, 2)  # Seek to end
            encoding_results['file_length'] = f.tell()
        
        # File modes demonstration
        modes_file = os.path.join(temp_dir, "modes.txt")
        
        mode_results = {}
        
        # Write modes
        with open(modes_file, 'x') as f:  # Exclusive creation
            f.write("Created exclusively")
        
        try:
            with open(modes_file, 'x') as f:  # This should fail
                f.write("This won't work")
        except FileExistsError:
            mode_results['exclusive_mode'] = "FileExistsError caught as expected"
        
        # Read/write mode
        with open(modes_file, 'r+') as f:
            original = f.read()
            f.seek(0)
            f.write("Modified")
            f.seek(0)
            mode_results['read_write_mode'] = f.read()
        
        return {
            'encoding_tests': encoding_results,
            'mode_tests': mode_results
        }
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Working with different file formats
def structured_file_formats():
    temp_dir = tempfile.mkdtemp()
    
    try:
        # JSON operations
        json_data = {
            "users": [
                {"name": "Alice", "age": 30, "active": True},
                {"name": "Bob", "age": 25, "active": False}
            ],
            "settings": {"theme": "dark", "notifications": True}
        }
        
        json_file = os.path.join(temp_dir, "data.json")
        
        # Write JSON
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Read JSON
        with open(json_file, 'r') as f:
            loaded_json = json.load(f)
        
        # CSV operations
        csv_data = [
            ["Name", "Age", "City"],
            ["Alice", "30", "NYC"],
            ["Bob", "25", "LA"],
            ["Charlie", "35", "Chicago"]
        ]
        
        csv_file = os.path.join(temp_dir, "data.csv")
        
        # Write CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
        
        # Read CSV
        csv_read_results = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            csv_read_results = list(reader)
        
        # CSV with DictReader/DictWriter
        dict_csv_file = os.path.join(temp_dir, "dict_data.csv")
        
        dict_data = [
            {"name": "Alice", "age": 30, "salary": 75000},
            {"name": "Bob", "age": 25, "salary": 65000}
        ]
        
        with open(dict_csv_file, 'w', newline='') as f:
            fieldnames = ['name', 'age', 'salary']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dict_data)
        
        with open(dict_csv_file, 'r') as f:
            reader = csv.DictReader(f)
            dict_csv_results = list(reader)
        
        # Pickle operations
        pickle_data = {
            "complex_object": {"nested": [1, 2, {"deep": True}]},
            "function": lambda x: x * 2,
            "set": {1, 2, 3, 4, 5}
        }
        
        pickle_file = os.path.join(temp_dir, "data.pickle")
        
        # Write pickle
        with open(pickle_file, 'wb') as f:
            pickle.dump(pickle_data, f)
        
        # Read pickle
        with open(pickle_file, 'rb') as f:
            loaded_pickle = pickle.load(f)
        
        return {
            'json_operations': {
                'original': json_data,
                'loaded': loaded_json,
                'data_matches': json_data == loaded_json
            },
            'csv_operations': {
                'written': csv_data,
                'read': csv_read_results,
                'dict_format': dict_csv_results
            },
            'pickle_operations': {
                'complex_preserved': loaded_pickle["complex_object"],
                'set_preserved': loaded_pickle["set"],
                'function_works': loaded_pickle["function"](5)
            }
        }
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Path operations using pathlib
def path_operations():
    # Current working directory operations
    cwd = Path.cwd()
    
    # Path construction
    paths = {
        'current_dir': cwd,
        'parent_dir': cwd.parent,
        'home_dir': Path.home(),
        'joined_path': cwd / "subdirectory" / "file.txt",
        'with_suffix': Path("file.txt").with_suffix(".json"),
        'with_name': Path("path/to/file.txt").with_name("newfile.txt")
    }
    
    # Create temporary directory for path operations
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Path creation and manipulation
        test_file = temp_dir / "test.txt"
        test_dir = temp_dir / "subdir"
        
        # Create directory
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file
        test_file.write_text("Hello from pathlib")
        
        # Path information
        path_info = {
            'exists': test_file.exists(),
            'is_file': test_file.is_file(),
            'is_dir': test_dir.is_dir(),
            'absolute': test_file.absolute(),
            'name': test_file.name,
            'stem': test_file.stem,
            'suffix': test_file.suffix,
            'parent': test_file.parent,
            'parts': test_file.parts,
            'stat': {
                'size': test_file.stat().st_size,
                'modified': test_file.stat().st_mtime
            }
        }
        
        # File operations with pathlib
        content = test_file.read_text()
        test_file.write_text(content + "\nAppended line")
        
        # Directory operations
        nested_dir = test_dir / "nested" / "deep"
        nested_dir.mkdir(parents=True, exist_ok=True)
        
        # List directory contents
        dir_contents = list(temp_dir.iterdir())
        all_files = list(temp_dir.rglob("*.txt"))
        
        return {
            'path_construction': {k: str(v) for k, v in paths.items()},
            'path_info': {k: str(v) if isinstance(v, Path) else v for k, v in path_info.items()},
            'file_content': test_file.read_text(),
            'directory_listing': [str(p.name) for p in dir_contents],
            'recursive_search': [str(p.relative_to(temp_dir)) for p in all_files]
        }
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Exception handling fundamentals
def exception_handling_basics():
    # Basic try-except patterns
    results = {}
    
    # Simple exception handling
    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        results['zero_division'] = f"Caught: {type(e).__name__}: {e}"
    
    # Multiple exception types
    def test_operations(operations):
        operation_results = []
        for op in operations:
            try:
                if op['type'] == 'division':
                    result = op['a'] / op['b']
                elif op['type'] == 'list_access':
                    result = op['list'][op['index']]
                elif op['type'] == 'dict_access':
                    result = op['dict'][op['key']]
                operation_results.append({'operation': op, 'result': result, 'status': 'success'})
            except (ZeroDivisionError, IndexError, KeyError) as e:
                operation_results.append({'operation': op, 'error': str(e), 'status': 'error'})
        return operation_results
    
    test_ops = [
        {'type': 'division', 'a': 10, 'b': 2},
        {'type': 'division', 'a': 10, 'b': 0},
        {'type': 'list_access', 'list': [1, 2, 3], 'index': 1},
        {'type': 'list_access', 'list': [1, 2, 3], 'index': 5},
        {'type': 'dict_access', 'dict': {'a': 1}, 'key': 'a'},
        {'type': 'dict_access', 'dict': {'a': 1}, 'key': 'b'}
    ]
    
    results['multiple_exceptions'] = test_operations(test_ops)
    
    # Try-except-else-finally
    def file_operation_demo():
        temp_file = None
        result = {}
        
        try:
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
            temp_file.write("Test content")
            result['write_success'] = True
        except IOError as e:
            result['error'] = str(e)
        else:
            result['else_executed'] = True
        finally:
            if temp_file:
                temp_file.close()
                os.unlink(temp_file.name)
            result['finally_executed'] = True
        
        return result
    
    results['try_else_finally'] = file_operation_demo()
    
    return results

def advanced_exception_handling():
    # Custom exceptions
    class CustomError(Exception):
        """Custom exception with additional attributes"""
        def __init__(self, message, error_code=None, details=None):
            super().__init__(message)
            self.error_code = error_code
            self.details = details
    
    class ValidationError(CustomError):
        """Specific validation error"""
        pass
    
    # Exception chaining
    def risky_operation():
        try:
            # Simulate a low-level error
            raise ValueError("Original error")
        except ValueError as e:
            # Re-raise with more context
            raise CustomError("High-level operation failed", error_code=500) from e
    
    # Exception handling with context
    def validate_user_data(data):
        errors = []
        
        if not data.get('name'):
            errors.append("Name is required")
        
        if not data.get('email') or '@' not in data.get('email', ''):
            errors.append("Valid email is required")
        
        age = data.get('age')
        if age is None or not isinstance(age, int) or age < 0:
            errors.append("Valid age is required")
        
        if errors:
            raise ValidationError(
                "Validation failed", 
                error_code=400, 
                details={"errors": errors, "data": data}
            )
        
        return True
    
    # Test exception scenarios
    test_results = {}
    
    # Custom exception handling
    try:
        risky_operation()
    except CustomError as e:
        test_results['custom_exception'] = {
            'message': str(e),
            'error_code': e.error_code,
            'cause': str(e.__cause__) if e.__cause__ else None
        }
    
    # Validation error handling
    invalid_data = [
        {},
        {"name": "Alice"},
        {"name": "Bob", "email": "invalid"},
        {"name": "Charlie", "email": "charlie@example.com", "age": -5}
    ]
    
    validation_results = []
    for data in invalid_data:
        try:
            validate_user_data(data)
            validation_results.append({"data": data, "status": "valid"})
        except ValidationError as e:
            validation_results.append({
                "data": data,
                "status": "invalid",
                "errors": e.details["errors"] if e.details else []
            })
    
    test_results['validation_tests'] = validation_results
    
    # Exception suppression
    from contextlib import suppress
    
    suppression_results = []
    
    # Without suppression
    try:
        result = 1 / 0
    except ZeroDivisionError:
        suppression_results.append("Exception caught manually")
    
    # With suppression
    with suppress(ZeroDivisionError):
        result = 1 / 0
    suppression_results.append("Exception suppressed")
    
    test_results['suppression'] = suppression_results
    
    return test_results

# Context managers and resource management
def context_manager_patterns():
    # Basic context manager usage
    temp_dir = tempfile.mkdtemp()
    
    try:
        # File context manager
        file_path = os.path.join(temp_dir, "context_test.txt")
        
        with open(file_path, 'w') as f:
            f.write("Context manager ensures file closure")
        
        # Custom context manager using contextlib
        from contextlib import contextmanager
        
        @contextmanager
        def timer_context():
            start = time.time()
            try:
                yield start
            finally:
                end = time.time()
                print(f"Operation took {end - start:.4f} seconds")
        
        # Multiple context managers
        results = {}
        
        with open(file_path, 'r') as input_file, \
             open(os.path.join(temp_dir, "output.txt"), 'w') as output_file:
            content = input_file.read()
            output_file.write(content.upper())
            results['multiple_contexts'] = "Files processed"
        
        # Custom context manager class
        class DatabaseConnection:
            def __init__(self, connection_string):
                self.connection_string = connection_string
                self.connected = False
            
            def __enter__(self):
                # Simulate connection
                self.connected = True
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Cleanup regardless of exceptions
                self.connected = False
                if exc_type:
                    # Log exception
                    return False  # Don't suppress exception
                return True
            
            def execute(self, query):
                if not self.connected:
                    raise RuntimeError("Not connected")
                return f"Executed: {query}"
        
        # Test custom context manager
        with DatabaseConnection("sqlite://memory") as db:
            results['custom_context'] = db.execute("SELECT * FROM users")
        
        # Exception handling in context managers
        try:
            with DatabaseConnection("sqlite://memory") as db:
                result = 1 / 0  # This will cause an exception
        except ZeroDivisionError:
            results['context_exception_handling'] = "Exception handled outside context"
        
        return results
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# File system operations
def filesystem_operations():
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Directory operations
        test_dir = os.path.join(temp_dir, "test_directory")
        nested_dir = os.path.join(test_dir, "nested", "deep")
        
        # Create directories
        os.makedirs(nested_dir, exist_ok=True)
        
        # Create files
        files_created = []
        for i in range(3):
            file_path = os.path.join(test_dir, f"file_{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"Content of file {i}")
            files_created.append(file_path)
        
        # File system information
        fs_info = {
            'directory_exists': os.path.exists(test_dir),
            'is_directory': os.path.isdir(test_dir),
            'directory_contents': os.listdir(test_dir),
            'walk_results': []
        }
        
        # Walk directory tree
        for root, dirs, files in os.walk(temp_dir):
            fs_info['walk_results'].append({
                'root': os.path.relpath(root, temp_dir),
                'directories': dirs,
                'files': files
            })
        
        # File operations
        source_file = files_created[0]
        dest_file = os.path.join(temp_dir, "copied_file.txt")
        
        # Copy file
        shutil.copy2(source_file, dest_file)
        
        # Move file
        moved_file = os.path.join(temp_dir, "moved_file.txt")
        shutil.move(dest_file, moved_file)
        
        # File permissions (Unix-like systems)
        if os.name != 'nt':  # Not Windows
            os.chmod(moved_file, 0o644)
            file_stat = os.stat(moved_file)
            fs_info['file_permissions'] = oct(file_stat.st_mode)[-3:]
        
        # File metadata
        file_stats = os.stat(moved_file)
        fs_info['file_metadata'] = {
            'size': file_stats.st_size,
            'modified_time': file_stats.st_mtime,
            'access_time': file_stats.st_atime
        }
        
        # Temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("Temporary content")
            temp_file_path = temp_file.name
        
        fs_info['temp_file_exists'] = os.path.exists(temp_file_path)
        
        # Cleanup temp file
        os.unlink(temp_file_path)
        fs_info['temp_file_cleaned'] = not os.path.exists(temp_file_path)
        
        return fs_info
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Interview problems involving file operations and exceptions
def file_exception_interview_problems():
    def safe_file_reader(file_path, encoding='utf-8', default_content=''):
        """Safely read file with error handling"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except FileNotFoundError:
            return default_content
        except PermissionError:
            return "Error: Permission denied"
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception:
                return "Error: Could not decode file"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def batch_file_processor(file_paths, processor_func):
        """Process multiple files with error handling"""
        results = []
        for file_path in file_paths:
            try:
                content = safe_file_reader(file_path)
                processed = processor_func(content)
                results.append({
                    'file': file_path,
                    'status': 'success',
                    'result': processed
                })
            except Exception as e:
                results.append({
                    'file': file_path,
                    'status': 'error',
                    'error': str(e)
                })
        return results
    
    def atomic_file_writer(file_path, content):
        """Write file atomically to prevent partial writes"""
        temp_path = file_path + '.tmp'
        try:
            with open(temp_path, 'w') as f:
                f.write(content)
            # Atomic rename
            os.rename(temp_path, file_path)
            return True
        except Exception as e:
            # Cleanup on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
    
    def config_file_manager(config_path, default_config=None):
        """Manage configuration file with validation"""
        if default_config is None:
            default_config = {}
        
        def load_config():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                # Validate config structure
                if not isinstance(config, dict):
                    raise ValueError("Config must be a dictionary")
                return config
            except FileNotFoundError:
                return default_config.copy()
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in config file: {e}")
        
        def save_config(config):
            try:
                atomic_file_writer(config_path, json.dumps(config, indent=2))
                return True
            except Exception as e:
                raise ValueError(f"Failed to save config: {e}")
        
        return load_config, save_config
    
    # Test implementations
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test safe file reader
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        reader_results = {
            'existing_file': safe_file_reader(test_file),
            'missing_file': safe_file_reader("nonexistent.txt", default_content="Default"),
            'missing_no_default': safe_file_reader("nonexistent.txt")
        }
        
        # Test batch processor
        processor_results = batch_file_processor(
            [test_file, "nonexistent.txt"],
            lambda content: content.upper()
        )
        
        # Test atomic writer
        atomic_file = os.path.join(temp_dir, "atomic.txt")
        atomic_success = atomic_file_writer(atomic_file, "Atomic content")
        
        # Test config manager
        config_file = os.path.join(temp_dir, "config.json")
        load_config, save_config = config_file_manager(
            config_file, 
            {"version": "1.0", "debug": False}
        )
        
        # Test config operations
        initial_config = load_config()
        initial_config["debug"] = True
        save_config(initial_config)
        reloaded_config = load_config()
        
        return {
            'safe_reader': reader_results,
            'batch_processor': processor_results,
            'atomic_writer': atomic_success,
            'config_manager': {
                'initial': initial_config,
                'reloaded': reloaded_config,
                'config_persisted': initial_config == reloaded_config
            }
        }
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Performance considerations
def file_performance_patterns():
    def time_operation(operation, iterations=1):
        start = time.time()
        for _ in range(iterations):
            result = operation()
        end = time.time()
        return (end - start) * 1000 / iterations, result
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Large file operations
        large_content = "Line of text\n" * 10000
        large_file = os.path.join(temp_dir, "large.txt")
        
        with open(large_file, 'w') as f:
            f.write(large_content)
        
        # Different reading strategies
        def read_all_at_once():
            with open(large_file, 'r') as f:
                return f.read()
        
        def read_line_by_line():
            lines = []
            with open(large_file, 'r') as f:
                for line in f:
                    lines.append(line.strip())
            return lines
        
        def read_with_readlines():
            with open(large_file, 'r') as f:
                return f.readlines()
        
        def read_in_chunks(chunk_size=8192):
            chunks = []
            with open(large_file, 'r') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    chunks.append(chunk)
            return ''.join(chunks)
        
        # Performance comparison
        read_all_time, _ = time_operation(read_all_at_once)
        read_lines_time, _ = time_operation(read_line_by_line)
        readlines_time, _ = time_operation(read_with_readlines)
        chunk_time, _ = time_operation(lambda: read_in_chunks(1024))
        
        # Memory usage patterns
        file_size = os.path.getsize(large_file)
        
        return {
            'file_size_bytes': file_size,
            'performance_ms': {
                'read_all': f"{read_all_time:.2f}",
                'line_by_line': f"{read_lines_time:.2f}",
                'readlines': f"{readlines_time:.2f}",
                'chunked': f"{chunk_time:.2f}"
            },
            'recommendations': [
                "Use read() for small files",
                "Use line iteration for large files",
                "Use chunked reading for very large files",
                "Consider memory constraints"
            ]
        }
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Comprehensive testing
def run_all_file_exception_demos():
    """Execute all file handling and exception demonstrations"""
    demo_functions = [
        ('basic_file_ops', basic_file_operations),
        ('file_modes', file_modes_and_encoding),
        ('structured_formats', structured_file_formats),
        ('path_operations', path_operations),
        ('exception_basics', exception_handling_basics),
        ('advanced_exceptions', advanced_exception_handling),
        ('context_managers', context_manager_patterns),
        ('filesystem_ops', filesystem_operations),
        ('interview_problems', file_exception_interview_problems),
        ('performance', file_performance_patterns)
    ]
    
    results = {}
    for name, func in demo_functions:
        try:
            start_time = time.time()
            result = func()
            execution_time = time.time() - start_time
            results[name] = {
                'result': result,
                'execution_time': f"{execution_time*1000:.2f}ms"
            }
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    print("=== Python File Handling and Exception Management Demo ===")
    
    # Run all demonstrations
    all_results = run_all_file_exception_demos()
    
    for category, data in all_results.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        
        if 'error' in data:
            print(f"  Error: {data['error']}")
            continue
            
        result = data['result']
        print(f"  Execution time: {data['execution_time']}")
        
        # Display results with appropriate formatting
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, (str, bytes)) and len(str(value)) > 100:
                    print(f"  {key}: {str(value)[:100]}... (truncated)")
                elif isinstance(value, list) and len(value) > 3:
                    print(f"  {key}: {value[:3]}... (showing first 3)")
                elif isinstance(value, dict) and len(value) > 3:
                    items = list(value.items())[:3]
                    print(f"  {key}: {dict(items)}... (showing first 3)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Result: {result}")
    
    print("\n=== FILE AND EXCEPTION BEST PRACTICES ===")
    
    best_practices = {
        'File Handling': 'Always use context managers (with statements)',
        'Exception Handling': 'Be specific with exception types',
        'Resource Management': 'Use try-finally or context managers',
        'Error Messages': 'Provide clear, actionable error messages',
        'File Paths': 'Use pathlib for cross-platform compatibility',
        'Large Files': 'Read in chunks to manage memory usage'
    }
    
    for category, practice in best_practices.items():
        print(f"  {category}: {practice}")
    
    print("\n=== PERFORMANCE SUMMARY ===")
    total_time = sum(float(data.get('execution_time', '0ms')[:-2]) 
                    for data in all_results.values() 
                    if 'execution_time' in data)
    print(f"  Total execution time: {total_time:.2f}ms")
    print(f"  Functions executed: {len(all_results)}")
    print(f"  Average per function: {total_time/len(all_results):.2f}ms")
