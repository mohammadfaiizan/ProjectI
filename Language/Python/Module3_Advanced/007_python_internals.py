"""
Python Internals: CPython Implementation, Bytecode, and Virtual Machine Details
Implementation-focused with minimal comments, maximum functionality coverage
"""

import dis
import sys
import types
import inspect
import ast
import marshal
import importlib
import importlib.util
import threading
import time
import gc
from typing import Any, Dict, List, Optional, Callable
import opcode
import keyword
import token
import tokenize
import io
import traceback
import code

# Bytecode analysis and disassembly
def bytecode_analysis_demo():
    """Demonstrate Python bytecode generation and analysis"""
    
    def simple_function(x, y):
        result = x + y
        if result > 10:
            return result * 2
        else:
            return result
    
    def complex_function(data):
        total = 0
        for item in data:
            if isinstance(item, (int, float)):
                total += item
            elif isinstance(item, str):
                total += len(item)
        return total
    
    def generator_function(n):
        for i in range(n):
            yield i ** 2
    
    # Get bytecode information
    simple_code = simple_function.__code__
    complex_code = complex_function.__code__
    generator_code = generator_function.__code__
    
    # Disassemble functions
    simple_bytecode = []
    for instruction in dis.get_instructions(simple_function):
        simple_bytecode.append({
            'offset': instruction.offset,
            'opname': instruction.opname,
            'arg': instruction.arg,
            'argval': instruction.argval,
            'argrepr': instruction.argrepr
        })
    
    # Analyze code object attributes
    def analyze_code_object(code_obj, name):
        return {
            'name': name,
            'filename': code_obj.co_filename,
            'argcount': code_obj.co_argcount,
            'posonlyargcount': getattr(code_obj, 'co_posonlyargcount', 0),
            'kwonlyargcount': code_obj.co_kwonlyargcount,
            'nlocals': code_obj.co_nlocals,
            'stacksize': code_obj.co_stacksize,
            'flags': code_obj.co_flags,
            'code_size': len(code_obj.co_code),
            'constants': code_obj.co_consts,
            'names': code_obj.co_names,
            'varnames': code_obj.co_varnames,
            'freevars': code_obj.co_freevars,
            'cellvars': code_obj.co_cellvars
        }
    
    # Test function execution with different arguments
    simple_result_1 = simple_function(5, 3)
    simple_result_2 = simple_function(15, 2)
    
    complex_result = complex_function([1, 2, "hello", 3.5, "world"])
    
    gen = generator_function(5)
    generator_results = list(gen)
    
    return {
        'simple_function': analyze_code_object(simple_code, 'simple_function'),
        'complex_function': analyze_code_object(complex_code, 'complex_function'),
        'generator_function': analyze_code_object(generator_code, 'generator_function'),
        'bytecode_sample': simple_bytecode[:10],
        'execution_results': {
            'simple_result_1': simple_result_1,
            'simple_result_2': simple_result_2,
            'complex_result': complex_result,
            'generator_results': generator_results
        }
    }

# Dynamic code compilation and execution
def dynamic_compilation_demo():
    """Demonstrate dynamic code compilation and execution"""
    
    # Compile source code to bytecode
    source_code = """
def dynamic_function(x):
    '''Dynamically compiled function'''
    return x ** 2 + 2 * x + 1

result = dynamic_function(5)
"""
    
    # Method 1: compile() and exec()
    compiled_code = compile(source_code, '<string>', 'exec')
    namespace = {}
    exec(compiled_code, namespace)
    
    dynamic_result_1 = namespace['result']
    dynamic_function_1 = namespace['dynamic_function']
    
    # Method 2: Using ast module
    tree = ast.parse(source_code)
    compiled_ast = compile(tree, '<ast>', 'exec')
    namespace_2 = {}
    exec(compiled_ast, namespace_2)
    
    dynamic_result_2 = namespace_2['result']
    
    # Method 3: eval() for expressions
    expression = "sum(x**2 for x in range(10))"
    eval_result = eval(expression)
    
    # Method 4: Creating functions dynamically
    def create_function(name, args, body):
        """Create a function dynamically"""
        func_code = f"def {name}({', '.join(args)}):\n"
        for line in body:
            func_code += f"    {line}\n"
        
        local_namespace = {}
        exec(func_code, local_namespace)
        return local_namespace[name]
    
    dynamic_func = create_function(
        'multiply_add',
        ['a', 'b', 'c'],
        ['return a * b + c']
    )
    
    dynamic_func_result = dynamic_func(3, 4, 5)
    
    # Method 5: Code object manipulation
    def modify_constants(func, new_constants):
        """Create new function with modified constants"""
        code = func.__code__
        new_code = types.CodeType(
            code.co_argcount,
            code.co_posonlyargcount if hasattr(code, 'co_posonlyargcount') else 0,
            code.co_kwonlyargcount,
            code.co_nlocals,
            code.co_stacksize,
            code.co_flags,
            code.co_code,
            new_constants,
            code.co_names,
            code.co_varnames,
            code.co_filename,
            code.co_name,
            code.co_firstlineno,
            code.co_lnotab,
            code.co_freevars,
            code.co_cellvars
        )
        return types.FunctionType(new_code, func.__globals__)
    
    def original_func():
        return 42
    
    modified_func = modify_constants(original_func, (100,))
    
    # AST manipulation
    class ConstantTransformer(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, int):
                return ast.Constant(value=node.value * 2)
            return node
    
    source_with_constants = "result = 5 + 10 * 3"
    tree = ast.parse(source_with_constants)
    transformer = ConstantTransformer()
    new_tree = transformer.visit(tree)
    
    original_namespace = {}
    exec(compile(tree, '<original>', 'exec'), original_namespace)
    
    modified_namespace = {}
    exec(compile(new_tree, '<modified>', 'exec'), modified_namespace)
    
    return {
        'compilation_methods': {
            'compile_exec_result': dynamic_result_1,
            'ast_compile_result': dynamic_result_2,
            'eval_result': eval_result,
            'dynamic_function_result': dynamic_func_result
        },
        'code_modification': {
            'original_function_result': original_func(),
            'modified_function_result': modified_func()
        },
        'ast_transformation': {
            'original_result': original_namespace['result'],
            'modified_result': modified_namespace['result']
        }
    }

# Import system internals
def import_system_demo():
    """Demonstrate Python import system internals"""
    
    # Create a module dynamically
    module_source = """
CONSTANT = 42

def module_function(x):
    return x * CONSTANT

class ModuleClass:
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
"""
    
    # Method 1: Create module object directly
    module_spec = importlib.util.spec_from_loader(
        'dynamic_module',
        loader=None,
        origin='<string>'
    )
    dynamic_module = importlib.util.module_from_spec(module_spec)
    
    # Execute module code
    exec(module_source, dynamic_module.__dict__)
    
    # Add to sys.modules
    sys.modules['dynamic_module'] = dynamic_module
    
    # Test module usage
    module_func_result = dynamic_module.module_function(5)
    module_obj = dynamic_module.ModuleClass(100)
    module_obj_value = module_obj.get_value()
    
    # Method 2: Custom loader
    class StringLoader:
        def __init__(self, source_code):
            self.source_code = source_code
        
        def create_module(self, spec):
            return None  # Use default module creation
        
        def exec_module(self, module):
            exec(self.source_code, module.__dict__)
    
    # Create module with custom loader
    loader = StringLoader(module_source)
    spec = importlib.util.spec_from_loader('custom_loaded_module', loader)
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    
    # Method 3: Import hooks demonstration
    class CustomFinder:
        def find_spec(self, fullname, path, target=None):
            if fullname.startswith('generated_'):
                # Generate module source based on name
                suffix = fullname[10:]  # Remove 'generated_' prefix
                source = f"value = '{suffix}'\ndef get_value(): return value"
                
                loader = StringLoader(source)
                return importlib.util.spec_from_loader(fullname, loader)
            return None
    
    # Install custom finder
    custom_finder = CustomFinder()
    sys.meta_path.insert(0, custom_finder)
    
    try:
        # Import generated module
        import generated_test
        generated_value = generated_test.get_value()
    finally:
        # Remove custom finder
        sys.meta_path.remove(custom_finder)
    
    # Analyze import system state
    import_stats = {
        'modules_count': len(sys.modules),
        'meta_path_count': len(sys.meta_path),
        'path_hooks_count': len(sys.path_hooks),
        'path_importer_cache_count': len(sys.path_importer_cache)
    }
    
    # Module introspection
    module_info = {
        'dynamic_module_name': dynamic_module.__name__,
        'dynamic_module_dict_keys': list(dynamic_module.__dict__.keys()),
        'custom_module_name': custom_module.__name__,
        'generated_module_value': generated_value
    }
    
    return {
        'module_creation': {
            'dynamic_module_function_result': module_func_result,
            'dynamic_module_object_value': module_obj_value
        },
        'custom_loading': module_info,
        'import_system_stats': import_stats
    }

# Global Interpreter Lock (GIL) demonstration
def gil_demonstration():
    """Demonstrate GIL behavior and limitations"""
    
    import threading
    import queue
    import concurrent.futures
    
    # CPU-bound task
    def cpu_intensive_task(n, result_queue=None, thread_id=0):
        """CPU-intensive task that holds the GIL"""
        total = 0
        for i in range(n):
            total += i ** 2
        
        if result_queue:
            result_queue.put((thread_id, total))
        return total
    
    # I/O-bound task simulation
    def io_intensive_task(duration, result_queue=None, thread_id=0):
        """I/O simulation that releases the GIL"""
        import time
        time.sleep(duration)
        result = f"IO task {thread_id} completed"
        
        if result_queue:
            result_queue.put((thread_id, result))
        return result
    
    # Single-threaded CPU task
    start_time = time.time()
    single_thread_result = cpu_intensive_task(100000)
    single_thread_time = time.time() - start_time
    
    # Multi-threaded CPU tasks (GIL limited)
    start_time = time.time()
    cpu_queue = queue.Queue()
    cpu_threads = []
    
    for i in range(4):
        thread = threading.Thread(
            target=cpu_intensive_task,
            args=(25000, cpu_queue, i)
        )
        cpu_threads.append(thread)
        thread.start()
    
    for thread in cpu_threads:
        thread.join()
    
    multi_thread_cpu_time = time.time() - start_time
    
    # Collect CPU task results
    cpu_results = []
    while not cpu_queue.empty():
        cpu_results.append(cpu_queue.get())
    
    # Multi-threaded I/O tasks (GIL released)
    start_time = time.time()
    io_queue = queue.Queue()
    io_threads = []
    
    for i in range(4):
        thread = threading.Thread(
            target=io_intensive_task,
            args=(0.1, io_queue, i)
        )
        io_threads.append(thread)
        thread.start()
    
    for thread in io_threads:
        thread.join()
    
    multi_thread_io_time = time.time() - start_time
    
    # Collect I/O task results
    io_results = []
    while not io_queue.empty():
        io_results.append(io_queue.get())
    
    # Process pool for CPU tasks (bypasses GIL)
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        process_futures = [
            executor.submit(cpu_intensive_task, 25000, None, i)
            for i in range(4)
        ]
        process_results = [future.result() for future in process_futures]
    
    multi_process_time = time.time() - start_time
    
    # GIL switching demonstration
    switch_count_start = sys.getswitchinterval()
    
    # Temporarily change switch interval
    sys.setswitchinterval(0.001)  # Switch more frequently
    
    switch_count_modified = sys.getswitchinterval()
    
    # Reset to original
    sys.setswitchinterval(switch_count_start)
    
    return {
        'cpu_task_performance': {
            'single_thread_time': f"{single_thread_time:.4f}s",
            'multi_thread_time': f"{multi_thread_cpu_time:.4f}s",
            'multi_process_time': f"{multi_process_time:.4f}s",
            'threading_overhead': multi_thread_cpu_time > single_thread_time,
            'process_speedup': single_thread_time > multi_process_time
        },
        'io_task_performance': {
            'multi_thread_io_time': f"{multi_thread_io_time:.4f}s",
            'io_parallelism_effective': multi_thread_io_time < 0.4  # Should be ~0.1s
        },
        'gil_behavior': {
            'original_switch_interval': switch_count_start,
            'modified_switch_interval': switch_count_modified,
            'cpu_results_count': len(cpu_results),
            'io_results_count': len(io_results),
            'process_results_count': len(process_results)
        }
    }

# Frame and execution context analysis
def frame_analysis_demo():
    """Demonstrate frame objects and execution context"""
    
    def outer_function(x):
        outer_var = x * 2
        
        def inner_function(y):
            inner_var = y + outer_var
            
            # Get current frame
            current_frame = inspect.currentframe()
            frame_info = analyze_frame(current_frame)
            
            return inner_var, frame_info
        
        return inner_function(x + 1)
    
    def analyze_frame(frame):
        """Analyze a frame object"""
        info = {
            'function_name': frame.f_code.co_name,
            'filename': frame.f_code.co_filename,
            'line_number': frame.f_lineno,
            'local_vars': dict(frame.f_locals),
            'global_vars_count': len(frame.f_globals),
            'builtin_vars_count': len(frame.f_builtins) if frame.f_builtins else 0
        }
        
        # Get parent frame info
        if frame.f_back:
            info['parent_function'] = frame.f_back.f_code.co_name
            info['parent_locals'] = dict(frame.f_back.f_locals)
        
        return info
    
    def recursive_function(n, depth=0):
        """Recursive function to demonstrate stack frames"""
        if n <= 0:
            # Analyze call stack
            frames = []
            current_frame = inspect.currentframe()
            
            while current_frame:
                frames.append({
                    'function': current_frame.f_code.co_name,
                    'line': current_frame.f_lineno,
                    'locals': dict(current_frame.f_locals)
                })
                current_frame = current_frame.f_back
                
                # Prevent infinite loop
                if len(frames) > 20:
                    break
            
            return frames
        
        return recursive_function(n - 1, depth + 1)
    
    # Test frame analysis
    result, frame_info = outer_function(5)
    
    # Test recursive stack analysis
    stack_frames = recursive_function(3)
    
    # Traceback demonstration
    def error_function():
        raise ValueError("Intentional error for traceback demo")
    
    try:
        error_function()
    except ValueError:
        tb = traceback.format_exc()
        tb_info = {
            'traceback_length': len(tb.split('\n')),
            'error_type': 'ValueError',
            'error_caught': True
        }
    
    # Code object inspection
    def inspect_code_object():
        code_obj = inspect.currentframe().f_code
        return {
            'co_argcount': code_obj.co_argcount,
            'co_kwonlyargcount': code_obj.co_kwonlyargcount,
            'co_nlocals': code_obj.co_nlocals,
            'co_stacksize': code_obj.co_stacksize,
            'co_flags': code_obj.co_flags,
            'co_consts': code_obj.co_consts,
            'co_names': code_obj.co_names,
            'co_varnames': code_obj.co_varnames
        }
    
    code_info = inspect_code_object()
    
    return {
        'frame_analysis': {
            'function_result': result,
            'frame_info': frame_info
        },
        'stack_analysis': {
            'stack_depth': len(stack_frames),
            'bottom_frame': stack_frames[-1] if stack_frames else None,
            'top_frame': stack_frames[0] if stack_frames else None
        },
        'traceback_info': tb_info,
        'code_object_info': code_info
    }

# Object model and attribute access
def object_model_demo():
    """Demonstrate Python object model internals"""
    
    class CustomClass:
        class_attribute = "class_value"
        
        def __init__(self, value):
            self.instance_attribute = value
        
        def __getattribute__(self, name):
            print(f"Getting attribute: {name}")
            return super().__getattribute__(name)
        
        def __setattr__(self, name, value):
            print(f"Setting attribute: {name} = {value}")
            super().__setattr__(name, value)
        
        def __delattr__(self, name):
            print(f"Deleting attribute: {name}")
            super().__delattr__(name)
    
    class DescriptorClass:
        def __init__(self, name):
            self.name = name
        
        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            return f"Descriptor {self.name} accessed on {obj}"
        
        def __set__(self, obj, value):
            print(f"Descriptor {self.name} set to {value}")
            setattr(obj, f"_{self.name}", value)
        
        def __delete__(self, obj):
            print(f"Descriptor {self.name} deleted")
            delattr(obj, f"_{self.name}")
    
    class ClassWithDescriptor:
        descriptor_attr = DescriptorClass("test_descriptor")
        
        def __init__(self, value):
            self.normal_attr = value
    
    # Test attribute access
    obj = CustomClass(42)
    
    # Capture attribute operations
    class_attr = obj.class_attribute
    instance_attr = obj.instance_attribute
    
    obj.new_attribute = "new_value"
    new_attr = obj.new_attribute
    
    # Test descriptor
    desc_obj = ClassWithDescriptor(100)
    descriptor_get = desc_obj.descriptor_attr
    desc_obj.descriptor_attr = "descriptor_value"
    
    # Analyze object structure
    def analyze_object(obj):
        return {
            'type': type(obj).__name__,
            'id': id(obj),
            'dir_count': len(dir(obj)),
            'dict_keys': list(obj.__dict__.keys()) if hasattr(obj, '__dict__') else None,
            'class_name': obj.__class__.__name__,
            'mro': [cls.__name__ for cls in obj.__class__.__mro__],
            'has_dict': hasattr(obj, '__dict__'),
            'has_slots': hasattr(obj.__class__, '__slots__')
        }
    
    # Test metaclass behavior
    class MetaClass(type):
        def __new__(mcs, name, bases, namespace):
            # Add automatic property for any attribute ending with '_prop'
            for key, value in list(namespace.items()):
                if key.endswith('_prop') and not callable(value):
                    prop_name = key[:-5]  # Remove '_prop' suffix
                    
                    def make_property(attr_name):
                        def getter(self):
                            return getattr(self, f"_{attr_name}", None)
                        def setter(self, value):
                            setattr(self, f"_{attr_name}", value)
                        return property(getter, setter)
                    
                    namespace[prop_name] = make_property(prop_name)
            
            return super().__new__(mcs, name, bases, namespace)
    
    class AutoPropClass(metaclass=MetaClass):
        value_prop = None
        name_prop = None
        
        def __init__(self, value, name):
            self.value = value
            self.name = name
    
    auto_obj = AutoPropClass(123, "test")
    auto_value = auto_obj.value
    auto_name = auto_obj.name
    
    # Memory layout analysis
    import sys
    
    memory_info = {
        'custom_obj_size': sys.getsizeof(obj),
        'custom_obj_dict_size': sys.getsizeof(obj.__dict__) if hasattr(obj, '__dict__') else 0,
        'descriptor_obj_size': sys.getsizeof(desc_obj),
        'auto_obj_size': sys.getsizeof(auto_obj)
    }
    
    return {
        'attribute_access': {
            'class_attribute': class_attr,
            'instance_attribute': instance_attr,
            'new_attribute': new_attr
        },
        'descriptor_behavior': {
            'descriptor_get_result': descriptor_get,
            'descriptor_operations_logged': True
        },
        'object_analysis': {
            'custom_obj': analyze_object(obj),
            'descriptor_obj': analyze_object(desc_obj),
            'auto_prop_obj': analyze_object(auto_obj)
        },
        'auto_properties': {
            'auto_value': auto_value,
            'auto_name': auto_name,
            'has_value_property': hasattr(AutoPropClass, 'value'),
            'has_name_property': hasattr(AutoPropClass, 'name')
        },
        'memory_info': memory_info
    }

# Optimization internals
def optimization_internals_demo():
    """Demonstrate Python optimization internals"""
    
    # String interning
    def string_interning_test():
        # Short strings are automatically interned
        s1 = "hello"
        s2 = "hello"
        short_interned = s1 is s2
        
        # Longer strings may not be interned
        long1 = "this is a longer string that may not be interned"
        long2 = "this is a longer string that may not be interned"
        long_interned = long1 is long2
        
        # Force interning
        import sys
        forced1 = sys.intern("force_intern_test")
        forced2 = sys.intern("force_intern_test")
        forced_interned = forced1 is forced2
        
        return {
            'short_strings_interned': short_interned,
            'long_strings_interned': long_interned,
            'forced_interning_works': forced_interned
        }
    
    # Integer caching
    def integer_caching_test():
        # Small integers are cached
        small1 = 100
        small2 = 100
        small_cached = small1 is small2
        
        # Large integers are not cached
        large1 = 1000
        large2 = 1000
        large_cached = large1 is large2
        
        # Test the boundary
        boundary_tests = []
        for i in [-10, -1, 0, 1, 100, 256, 257, 1000]:
            a = i
            b = i
            boundary_tests.append({
                'value': i,
                'is_cached': a is b
            })
        
        return {
            'small_integers_cached': small_cached,
            'large_integers_cached': large_cached,
            'boundary_tests': boundary_tests
        }
    
    # Function call optimization
    def function_call_optimization():
        def simple_function(x):
            return x * 2
        
        def function_with_defaults(x, y=10, z=20):
            return x + y + z
        
        def function_with_kwargs(x, **kwargs):
            return x + sum(kwargs.values())
        
        # Measure call overhead
        import timeit
        
        simple_time = timeit.timeit(lambda: simple_function(5), number=100000)
        defaults_time = timeit.timeit(lambda: function_with_defaults(5), number=100000)
        kwargs_time = timeit.timeit(lambda: function_with_kwargs(5, a=1, b=2), number=100000)
        
        return {
            'simple_call_time': f"{simple_time:.6f}s",
            'defaults_call_time': f"{defaults_time:.6f}s",
            'kwargs_call_time': f"{kwargs_time:.6f}s",
            'defaults_overhead': defaults_time / simple_time,
            'kwargs_overhead': kwargs_time / simple_time
        }
    
    # List optimization
    def list_optimization_test():
        # Pre-allocation vs dynamic growth
        import timeit
        
        # Dynamic growth
        def dynamic_list(n):
            lst = []
            for i in range(n):
                lst.append(i)
            return lst
        
        # Pre-allocation
        def preallocated_list(n):
            lst = [None] * n
            for i in range(n):
                lst[i] = i
            return lst
        
        # List comprehension
        def comprehension_list(n):
            return [i for i in range(n)]
        
        n = 10000
        dynamic_time = timeit.timeit(lambda: dynamic_list(n), number=100)
        preallocated_time = timeit.timeit(lambda: preallocated_list(n), number=100)
        comprehension_time = timeit.timeit(lambda: comprehension_list(n), number=100)
        
        return {
            'dynamic_time': f"{dynamic_time:.6f}s",
            'preallocated_time': f"{preallocated_time:.6f}s",
            'comprehension_time': f"{comprehension_time:.6f}s",
            'fastest_method': min([
                ('dynamic', dynamic_time),
                ('preallocated', preallocated_time),
                ('comprehension', comprehension_time)
            ], key=lambda x: x[1])[0]
        }
    
    # Dictionary optimization
    def dict_optimization_test():
        # Dictionary creation methods
        import timeit
        
        def literal_dict():
            return {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
        
        def constructor_dict():
            return dict(a=1, b=2, c=3, d=4, e=5)
        
        def comprehension_dict():
            keys = ['a', 'b', 'c', 'd', 'e']
            values = [1, 2, 3, 4, 5]
            return {k: v for k, v in zip(keys, values)}
        
        literal_time = timeit.timeit(literal_dict, number=100000)
        constructor_time = timeit.timeit(constructor_dict, number=100000)
        comprehension_time = timeit.timeit(comprehension_dict, number=100000)
        
        return {
            'literal_time': f"{literal_time:.6f}s",
            'constructor_time': f"{constructor_time:.6f}s",
            'comprehension_time': f"{comprehension_time:.6f}s",
            'literal_fastest': literal_time < constructor_time and literal_time < comprehension_time
        }
    
    return {
        'string_interning': string_interning_test(),
        'integer_caching': integer_caching_test(),
        'function_calls': function_call_optimization(),
        'list_optimization': list_optimization_test(),
        'dict_optimization': dict_optimization_test()
    }

# Comprehensive testing
def run_all_internals_demos():
    """Execute all Python internals demonstrations"""
    demo_functions = [
        ('bytecode_analysis', bytecode_analysis_demo),
        ('dynamic_compilation', dynamic_compilation_demo),
        ('import_system', import_system_demo),
        ('gil_demonstration', gil_demonstration),
        ('frame_analysis', frame_analysis_demo),
        ('object_model', object_model_demo),
        ('optimization_internals', optimization_internals_demo)
    ]
    
    results = {}
    for name, func in demo_functions:
        try:
            result = func()
            results[name] = result
        except Exception as e:
            results[name] = {'error': str(e)}
    
    # Add system information
    results['python_info'] = {
        'version': sys.version,
        'implementation': sys.implementation.name,
        'platform': sys.platform,
        'executable': sys.executable,
        'recursion_limit': sys.getrecursionlimit(),
        'switch_interval': sys.getswitchinterval(),
        'thread_info': sys.thread_info._asdict() if hasattr(sys, 'thread_info') else None
    }
    
    return results

if __name__ == "__main__":
    print("=== Python Internals Demo ===")
    
    # Run all demonstrations
    all_results = run_all_internals_demos()
    
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
    
    print("\n=== PYTHON INTERNALS CONCEPTS ===")
    
    concepts = {
        "Bytecode": "Python source is compiled to bytecode for the virtual machine",
        "Code Objects": "Contain bytecode, constants, and metadata for functions",
        "Frame Objects": "Represent execution contexts with local/global namespaces",
        "Import System": "Modular system for loading and executing Python modules",
        "GIL": "Global Interpreter Lock prevents true parallelism in CPython",
        "Object Model": "Everything is an object with type, identity, and value",
        "Descriptor Protocol": "__get__, __set__, __delete__ for attribute access",
        "Metaclasses": "Classes that create classes, control class creation"
    }
    
    for concept, description in concepts.items():
        print(f"  {concept}: {description}")
    
    print("\n=== CPYTHON OPTIMIZATIONS ===")
    
    optimizations = {
        "String Interning": "Cache common strings to save memory",
        "Integer Caching": "Cache small integers (-5 to 256) for reuse",
        "Bytecode Caching": "Cache compiled bytecode in __pycache__",
        "Peephole Optimization": "Local optimizations during compilation",
        "List Pre-allocation": "Optimize list growth patterns",
        "Dictionary Order": "Maintain insertion order (Python 3.7+)",
        "Attribute Access": "Optimized lookups through method resolution order",
        "Function Calls": "Fast calling conventions for common patterns"
    }
    
    for optimization, description in optimizations.items():
        print(f"  {optimization}: {description}")
    
    print("\n=== DEBUGGING AND INTROSPECTION ===")
    
    tools = {
        "dis module": "Disassemble bytecode for analysis",
        "inspect module": "Runtime introspection of objects",
        "sys module": "System-specific parameters and functions",
        "gc module": "Garbage collection interface",
        "traceback module": "Format and print exception tracebacks",
        "ast module": "Abstract syntax tree manipulation",
        "types module": "Dynamic type creation utilities",
        "importlib": "Import system implementation and utilities"
    }
    
    for tool, description in tools.items():
        print(f"  {tool}: {description}")
    
    print("\n=== BEST PRACTICES ===")
    
    best_practices = [
        "Understand bytecode for performance optimization",
        "Use dis.dis() to analyze function performance",
        "Be aware of GIL limitations for CPU-bound threading",
        "Use multiprocessing for true parallelism",
        "Understand import system for better module design",
        "Profile frame overhead in recursive functions",
        "Leverage string interning for memory optimization",
        "Use __slots__ when appropriate for memory savings",
        "Understand descriptor protocol for advanced attribute handling",
        "Monitor object creation patterns for performance",
        "Use ast module for safe code analysis",
        "Understand metaclasses before using them"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== Python Internals Complete! ===")
    print("  CPython implementation details and virtual machine mastered")
