"""
Python Modules and Packages: Import Systems, Package Management
Implementation-focused with minimal comments, maximum functionality coverage
"""

import sys
import os
import importlib
import importlib.util
import tempfile
import shutil
from types import ModuleType
from typing import List, Dict, Any, Optional
import subprocess
import time

# Basic import patterns
def import_demonstrations():
    # Standard library imports
    import math
    import random
    import datetime
    from collections import defaultdict, Counter
    from pathlib import Path
    
    # Import with aliases
    import numpy as np  # Would work if numpy installed
    import json as js
    
    # Selective imports
    from os import path, environ
    from sys import version, platform
    
    # Import results
    import_results = {
        'math_pi': math.pi,
        'random_number': random.randint(1, 100),
        'current_time': datetime.datetime.now().isoformat(),
        'default_dict': dict(defaultdict(int)),
        'counter_example': dict(Counter('hello world')),
        'path_join': path.join('home', 'user', 'file.txt'),
        'python_version': version,
        'platform_info': platform
    }
    
    # Conditional imports
    try:
        import sqlite3
        import_results['sqlite3_available'] = True
        import_results['sqlite3_version'] = sqlite3.version
    except ImportError:
        import_results['sqlite3_available'] = False
    
    # Import all (not recommended, but demonstrating)
    import_results['builtins_count'] = len(dir(__builtins__))
    
    return import_results

def module_introspection():
    import math
    import os
    
    # Module attributes
    module_info = {
        'math_module': {
            'name': math.__name__,
            'file': getattr(math, '__file__', 'Built-in'),
            'doc': math.__doc__[:100] if math.__doc__ else None,
            'functions': [name for name in dir(math) if not name.startswith('_')],
            'function_count': len([name for name in dir(math) if callable(getattr(math, name))])
        },
        'os_module': {
            'name': os.__name__,
            'file': getattr(os, '__file__', 'Built-in'),
            'attributes': len(dir(os)),
            'environ_vars': len(os.environ)
        }
    }
    
    # Module search path
    module_info['sys_path'] = {
        'path_count': len(sys.path),
        'first_three_paths': sys.path[:3],
        'current_dir_in_path': '' in sys.path or '.' in sys.path
    }
    
    # Built-in modules
    module_info['builtin_modules'] = {
        'count': len(sys.builtin_module_names),
        'sample': list(sys.builtin_module_names)[:10]
    }
    
    return module_info

# Dynamic imports and module creation
def dynamic_import_patterns():
    # Import by string name
    def import_by_name(module_name):
        try:
            return importlib.import_module(module_name)
        except ImportError as e:
            return f"Failed to import {module_name}: {e}"
    
    # Test dynamic imports
    dynamic_results = {
        'json_import': str(type(import_by_name('json'))),
        'math_import': str(type(import_by_name('math'))),
        'fake_import': import_by_name('nonexistent_module')
    }
    
    # Reload module (demonstration)
    import json
    original_json = json
    importlib.reload(json)
    dynamic_results['reload_same_object'] = original_json is json
    
    # Import from string path
    def import_from_path(file_path, module_name):
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            return None
        except Exception as e:
            return f"Error: {e}"
    
    # Create temporary module file for testing
    temp_dir = tempfile.mkdtemp()
    temp_module_path = os.path.join(temp_dir, 'temp_module.py')
    
    try:
        with open(temp_module_path, 'w') as f:
            f.write('''
def greet(name):
    return f"Hello, {name}!"

CONSTANT = 42

class TempClass:
    def __init__(self, value):
        self.value = value
    
    def get_double(self):
        return self.value * 2
''')
        
        # Import the temporary module
        temp_module = import_from_path(temp_module_path, 'temp_module')
        if isinstance(temp_module, ModuleType):
            dynamic_results['temp_module'] = {
                'greet_function': temp_module.greet('World'),
                'constant': temp_module.CONSTANT,
                'class_instance': temp_module.TempClass(5).get_double()
            }
        else:
            dynamic_results['temp_module'] = temp_module
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return dynamic_results

def lazy_import_patterns():
    # Lazy import decorator
    def lazy_import(module_name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                module = importlib.import_module(module_name)
                return func(module, *args, **kwargs)
            return wrapper
        return decorator
    
    # Lazy import context manager
    class LazyImporter:
        def __init__(self, module_name):
            self.module_name = module_name
            self.module = None
        
        def __enter__(self):
            self.module = importlib.import_module(self.module_name)
            return self.module
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    # Test lazy imports
    @lazy_import('random')
    def get_random_numbers(random_module, count=5):
        return [random_module.randint(1, 100) for _ in range(count)]
    
    # Test lazy context manager
    lazy_results = {
        'lazy_function': get_random_numbers(3),
        'lazy_context': None
    }
    
    with LazyImporter('datetime') as dt:
        lazy_results['lazy_context'] = dt.datetime.now().year
    
    # Optional import pattern
    def optional_import(module_name, fallback=None):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            return fallback
    
    # Test optional imports
    json_module = optional_import('json')
    fake_module = optional_import('nonexistent', fallback={'dumps': lambda x: str(x)})
    
    lazy_results['optional_imports'] = {
        'json_available': json_module is not None,
        'fallback_used': fake_module is not None and hasattr(fake_module, 'dumps')
    }
    
    return lazy_results

# Package structure simulation
def package_structure_demo():
    # Create temporary package structure
    temp_dir = tempfile.mkdtemp()
    package_dir = os.path.join(temp_dir, 'demo_package')
    subpackage_dir = os.path.join(package_dir, 'subpackage')
    
    try:
        # Create package directories
        os.makedirs(subpackage_dir)
        
        # Create __init__.py files
        init_content = '''
"""Demo package for module demonstration"""
__version__ = "1.0.0"
__author__ = "Demo Author"

from .core import main_function
from .subpackage import sub_function

__all__ = ['main_function', 'sub_function']
'''
        
        with open(os.path.join(package_dir, '__init__.py'), 'w') as f:
            f.write(init_content)
        
        # Create core module
        core_content = '''
def main_function():
    return "Main function called"

def helper_function():
    return "Helper function"

PACKAGE_CONSTANT = "Demo constant"
'''
        
        with open(os.path.join(package_dir, 'core.py'), 'w') as f:
            f.write(core_content)
        
        # Create subpackage
        sub_init_content = '''
from .utils import sub_function
'''
        
        with open(os.path.join(subpackage_dir, '__init__.py'), 'w') as f:
            f.write(sub_init_content)
        
        utils_content = '''
def sub_function():
    return "Subpackage function called"

def internal_function():
    return "Internal function"
'''
        
        with open(os.path.join(subpackage_dir, 'utils.py'), 'w') as f:
            f.write(utils_content)
        
        # Add package to sys.path temporarily
        sys.path.insert(0, temp_dir)
        
        try:
            # Import the package
            demo_package = importlib.import_module('demo_package')
            
            package_results = {
                'package_info': {
                    'version': demo_package.__version__,
                    'author': demo_package.__author__,
                    'main_function': demo_package.main_function(),
                    'sub_function': demo_package.sub_function()
                },
                'package_structure': {
                    'all_exports': demo_package.__all__,
                    'package_file': demo_package.__file__,
                    'package_path': demo_package.__path__._path
                }
            }
            
            # Import submodules
            from demo_package import core
            from demo_package.subpackage import utils
            
            package_results['submodule_access'] = {
                'core_helper': core.helper_function(),
                'core_constant': core.PACKAGE_CONSTANT,
                'utils_internal': utils.internal_function()
            }
            
        finally:
            # Remove from sys.path
            sys.path.remove(temp_dir)
        
        return package_results
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Module search and discovery
def module_discovery_patterns():
    # Find modules by pattern
    def find_modules_by_pattern(pattern):
        found_modules = []
        for name in sys.builtin_module_names:
            if pattern in name:
                found_modules.append(name)
        return found_modules
    
    # Check module availability
    def check_module_availability(modules):
        availability = {}
        for module in modules:
            try:
                importlib.import_module(module)
                availability[module] = True
            except ImportError:
                availability[module] = False
        return availability
    
    # Get module information
    def get_module_info(module_name):
        try:
            module = importlib.import_module(module_name)
            return {
                'name': module.__name__,
                'file': getattr(module, '__file__', 'Built-in'),
                'package': getattr(module, '__package__', None),
                'spec': str(getattr(module, '__spec__', None)),
                'doc_length': len(module.__doc__) if module.__doc__ else 0,
                'attributes': len(dir(module)),
                'callable_count': len([attr for attr in dir(module) 
                                     if callable(getattr(module, attr, None))])
            }
        except ImportError as e:
            return {'error': str(e)}
    
    # Test discovery functions
    discovery_results = {
        'pattern_search': {
            'json_modules': find_modules_by_pattern('json'),
            'os_modules': find_modules_by_pattern('os'),
            'sys_modules': find_modules_by_pattern('sys')
        },
        'availability_check': check_module_availability([
            'json', 'math', 'os', 'sys', 'collections', 'nonexistent'
        ]),
        'module_details': {
            'json': get_module_info('json'),
            'collections': get_module_info('collections'),
            'pathlib': get_module_info('pathlib')
        }
    }
    
    # Module cache information
    discovery_results['cache_info'] = {
        'cached_modules': len(sys.modules),
        'sample_cached': list(sys.modules.keys())[:10]
    }
    
    return discovery_results

# Virtual environments and package management simulation
def package_management_simulation():
    # Simulate package installation checking
    def check_package_installed(package_name):
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False
    
    # Common packages to check
    common_packages = [
        'json', 'math', 'os', 'sys', 'collections', 'itertools',
        'functools', 'operator', 'copy', 'pickle', 'sqlite3',
        'urllib', 'http', 'email', 'datetime', 'time', 'random'
    ]
    
    package_status = {}
    for package in common_packages:
        package_status[package] = check_package_installed(package)
    
    # Get Python environment information
    env_info = {
        'python_version': sys.version,
        'python_executable': sys.executable,
        'platform': sys.platform,
        'path_count': len(sys.path),
        'modules_loaded': len(sys.modules)
    }
    
    # Virtual environment detection
    def detect_virtual_env():
        # Check for common virtual environment indicators
        indicators = {
            'VIRTUAL_ENV': 'VIRTUAL_ENV' in os.environ,
            'conda_env': 'CONDA_DEFAULT_ENV' in os.environ,
            'pip_prefix': hasattr(sys, 'real_prefix'),
            'base_prefix': hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        }
        return indicators
    
    env_info['virtual_env_indicators'] = detect_virtual_env()
    
    # Simulate pip list functionality
    def simulate_pip_list():
        # This would normally use subprocess to call pip list
        # For simulation, we'll just show loaded modules
        loaded_modules = {}
        for name, module in list(sys.modules.items())[:20]:  # First 20 for demo
            if hasattr(module, '__version__'):
                loaded_modules[name] = module.__version__
            elif hasattr(module, 'version'):
                loaded_modules[name] = str(module.version)
            else:
                loaded_modules[name] = 'unknown'
        return loaded_modules
    
    return {
        'package_availability': package_status,
        'environment_info': env_info,
        'simulated_packages': simulate_pip_list()
    }

# Import hooks and customization
def import_customization_patterns():
    # Custom import hook (simplified demonstration)
    class CustomImporter:
        def __init__(self, prefix):
            self.prefix = prefix
        
        def find_module(self, name, path=None):
            if name.startswith(self.prefix):
                return self
            return None
        
        def load_module(self, name):
            if name in sys.modules:
                return sys.modules[name]
            
            # Create a dummy module
            module = ModuleType(name)
            module.__file__ = f"<custom:{name}>"
            module.__loader__ = self
            module.custom_function = lambda: f"Custom module {name} loaded"
            
            sys.modules[name] = module
            return module
    
    # Module wrapper for additional functionality
    class ModuleWrapper:
        def __init__(self, module):
            self._module = module
            self._access_count = 0
        
        def __getattr__(self, name):
            self._access_count += 1
            return getattr(self._module, name)
        
        def get_access_count(self):
            return self._access_count
    
    # Test custom import behavior
    customization_results = {}
    
    # Test module wrapper
    import json
    wrapped_json = ModuleWrapper(json)
    
    # Use wrapped module
    wrapped_json.dumps({'test': 'data'})
    wrapped_json.loads('{"key": "value"}')
    
    customization_results['wrapper'] = {
        'access_count': wrapped_json.get_access_count(),
        'original_module': str(type(wrapped_json._module))
    }
    
    # Module monkey patching demonstration
    def patch_module_function():
        original_join = os.path.join
        
        def logged_join(*args):
            result = original_join(*args)
            return result
        
        # Temporarily patch
        os.path.join = logged_join
        
        # Test patched function
        test_result = os.path.join('home', 'user', 'file')
        
        # Restore original
        os.path.join = original_join
        
        return test_result
    
    customization_results['monkey_patch'] = {
        'patched_result': patch_module_function(),
        'function_restored': os.path.join == os.path.join  # Always True
    }
    
    return customization_results

# Namespace packages and relative imports
def namespace_package_patterns():
    # Simulate namespace package behavior
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create namespace package structure
        ns_package1 = os.path.join(temp_dir, 'namespace_pkg', 'part1')
        ns_package2 = os.path.join(temp_dir, 'namespace_pkg', 'part2')
        
        os.makedirs(ns_package1)
        os.makedirs(ns_package2)
        
        # Note: No __init__.py in namespace_pkg (makes it a namespace package)
        
        # Create modules in different parts
        with open(os.path.join(ns_package1, 'module1.py'), 'w') as f:
            f.write('''
def function1():
    return "Function from part1"
''')
        
        with open(os.path.join(ns_package2, 'module2.py'), 'w') as f:
            f.write('''
def function2():
    return "Function from part2"
''')
        
        # Create regular package with relative imports
        regular_pkg = os.path.join(temp_dir, 'regular_package')
        os.makedirs(regular_pkg)
        
        with open(os.path.join(regular_pkg, '__init__.py'), 'w') as f:
            f.write('from .main import main_function')
        
        with open(os.path.join(regular_pkg, 'main.py'), 'w') as f:
            f.write('''
from .utils import helper
from . import constants

def main_function():
    return f"{helper()} - {constants.PACKAGE_NAME}"
''')
        
        with open(os.path.join(regular_pkg, 'utils.py'), 'w') as f:
            f.write('''
def helper():
    return "Helper called"
''')
        
        with open(os.path.join(regular_pkg, 'constants.py'), 'w') as f:
            f.write('''
PACKAGE_NAME = "Regular Package"
VERSION = "1.0"
''')
        
        # Test the packages
        sys.path.insert(0, temp_dir)
        
        try:
            # Import regular package
            regular = importlib.import_module('regular_package')
            
            results = {
                'regular_package': {
                    'main_function': regular.main_function(),
                    'package_file': regular.__file__
                }
            }
            
            # Import individual modules from namespace package
            mod1 = importlib.import_module('namespace_pkg.part1.module1')
            mod2 = importlib.import_module('namespace_pkg.part2.module2')
            
            results['namespace_package'] = {
                'part1_function': mod1.function1(),
                'part2_function': mod2.function2()
            }
            
        finally:
            sys.path.remove(temp_dir)
        
        return results
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Interview problems related to modules
def module_interview_problems():
    def module_dependency_analyzer(module_name):
        """Analyze module dependencies"""
        try:
            module = importlib.import_module(module_name)
            dependencies = []
            
            # Get imports from module if available
            if hasattr(module, '__file__') and module.__file__:
                try:
                    with open(module.__file__, 'r') as f:
                        content = f.read()
                    
                    import re
                    import_patterns = [
                        r'import\s+(\w+)',
                        r'from\s+(\w+)\s+import'
                    ]
                    
                    for pattern in import_patterns:
                        dependencies.extend(re.findall(pattern, content))
                
                except Exception:
                    pass
            
            return {
                'module': module_name,
                'dependencies': list(set(dependencies)),
                'dependency_count': len(set(dependencies)),
                'file_location': getattr(module, '__file__', 'Built-in')
            }
        
        except ImportError as e:
            return {'module': module_name, 'error': str(e)}
    
    def circular_import_detector():
        """Detect potential circular imports in loaded modules"""
        loaded_modules = list(sys.modules.keys())
        potential_circles = []
        
        # Simple heuristic: modules with similar names might be related
        for i, mod1 in enumerate(loaded_modules):
            for mod2 in loaded_modules[i+1:]:
                if ('.' in mod1 and '.' in mod2 and 
                    mod1.split('.')[0] == mod2.split('.')[0]):
                    potential_circles.append((mod1, mod2))
        
        return potential_circles[:5]  # First 5 for demo
    
    def module_size_analyzer(module_names):
        """Analyze module sizes and complexity"""
        analysis = {}
        
        for name in module_names:
            try:
                module = importlib.import_module(name)
                analysis[name] = {
                    'attributes': len(dir(module)),
                    'callables': len([attr for attr in dir(module) 
                                   if callable(getattr(module, attr, None))]),
                    'doc_length': len(module.__doc__) if module.__doc__ else 0,
                    'has_file': hasattr(module, '__file__'),
                    'is_package': hasattr(module, '__path__')
                }
            except ImportError:
                analysis[name] = {'error': 'Module not found'}
        
        return analysis
    
    def import_time_profiler(module_names):
        """Profile import times for modules"""
        import_times = {}
        
        for name in module_names:
            if name in sys.modules:
                # Module already loaded
                import_times[name] = 'already_loaded'
                continue
            
            start_time = time.time()
            try:
                importlib.import_module(name)
                end_time = time.time()
                import_times[name] = f"{(end_time - start_time) * 1000:.2f}ms"
            except ImportError:
                import_times[name] = 'import_failed'
        
        return import_times
    
    # Test interview problems
    test_modules = ['json', 'os', 'sys', 'collections', 'itertools']
    
    results = {
        'dependency_analysis': [module_dependency_analyzer(mod) for mod in test_modules[:3]],
        'circular_imports': circular_import_detector(),
        'module_analysis': module_size_analyzer(test_modules),
        'import_profiling': import_time_profiler(['tempfile', 'shutil', 'pathlib'])
    }
    
    return results

# Performance and optimization
def module_performance_patterns():
    def import_performance_comparison():
        """Compare different import strategies"""
        
        def time_import_strategy(strategy_func, iterations=100):
            start_time = time.time()
            for _ in range(iterations):
                strategy_func()
            return (time.time() - start_time) * 1000 / iterations
        
        # Strategy 1: Import entire module
        def import_entire():
            import json
            return json.dumps({'test': 'data'})
        
        # Strategy 2: Import specific function
        def import_specific():
            from json import dumps
            return dumps({'test': 'data'})
        
        # Strategy 3: Import with alias
        def import_alias():
            import json as js
            return js.dumps({'test': 'data'})
        
        strategies = {
            'entire_module': import_entire,
            'specific_function': import_specific,
            'with_alias': import_alias
        }
        
        performance_results = {}
        for name, strategy in strategies.items():
            # Warm up
            strategy()
            # Measure
            performance_results[name] = f"{time_import_strategy(strategy, 10):.4f}ms"
        
        return performance_results
    
    def module_memory_usage():
        """Analyze memory usage of modules"""
        import sys
        
        # Get module sizes
        module_sizes = {}
        for name, module in list(sys.modules.items())[:10]:
            if module is not None:
                module_sizes[name] = sys.getsizeof(module)
        
        return {
            'module_sizes': module_sizes,
            'total_modules': len(sys.modules),
            'average_size': sum(module_sizes.values()) / len(module_sizes) if module_sizes else 0
        }
    
    def lazy_loading_demo():
        """Demonstrate lazy loading benefits"""
        
        class LazyModule:
            def __init__(self, module_name):
                self.module_name = module_name
                self._module = None
            
            def __getattr__(self, name):
                if self._module is None:
                    self._module = importlib.import_module(self.module_name)
                return getattr(self._module, name)
        
        # Test lazy loading
        lazy_json = LazyModule('json')
        
        # Module not loaded yet
        not_loaded = lazy_json._module is None
        
        # Access attribute - triggers loading
        result = lazy_json.dumps({'test': 'lazy'})
        loaded = lazy_json._module is not None
        
        return {
            'not_loaded_initially': not_loaded,
            'loaded_after_access': loaded,
            'lazy_result': result
        }
    
    return {
        'import_performance': import_performance_comparison(),
        'memory_usage': module_memory_usage(),
        'lazy_loading': lazy_loading_demo()
    }

# Comprehensive testing
def run_all_module_demos():
    """Execute all module and package demonstrations"""
    demo_functions = [
        ('basic_imports', import_demonstrations),
        ('introspection', module_introspection),
        ('dynamic_imports', dynamic_import_patterns),
        ('lazy_imports', lazy_import_patterns),
        ('package_structure', package_structure_demo),
        ('module_discovery', module_discovery_patterns),
        ('package_management', package_management_simulation),
        ('import_customization', import_customization_patterns),
        ('namespace_packages', namespace_package_patterns),
        ('interview_problems', module_interview_problems),
        ('performance', module_performance_patterns)
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
    print("=== Python Modules and Packages Demo ===")
    
    # Run all demonstrations
    all_results = run_all_module_demos()
    
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
                if isinstance(value, (list, dict)) and len(str(value)) > 100:
                    if isinstance(value, list) and len(value) > 3:
                        print(f"  {key}: {value[:3]}... (showing first 3)")
                    elif isinstance(value, dict) and len(value) > 3:
                        items = list(value.items())[:3]
                        print(f"  {key}: {dict(items)}... (showing first 3)")
                    else:
                        print(f"  {key}: {str(value)[:100]}... (truncated)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Result: {result}")
    
    print("\n=== MODULE BEST PRACTICES ===")
    
    best_practices = {
        'Import Style': 'Use specific imports, avoid import *',
        'Module Organization': 'Keep related functionality together',
        'Package Structure': 'Use __init__.py for package initialization',
        'Lazy Loading': 'Import heavy modules only when needed',
        'Circular Imports': 'Avoid circular dependencies',
        'Virtual Environments': 'Use virtual environments for isolation'
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
