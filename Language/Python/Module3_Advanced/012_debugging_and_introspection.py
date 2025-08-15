"""
Python Debugging and Introspection: Advanced Debugging, Code Analysis, and Runtime Inspection
Implementation-focused with minimal comments, maximum functionality coverage
"""

import pdb
import inspect
import sys
import traceback
import logging
import functools
import types
import dis
import ast
import gc
import weakref
from typing import Any, Dict, List, Optional, Callable, Type, Union
import time
import threading
import warnings
import linecache
import code
import atexit
from dataclasses import dataclass
from contextlib import contextmanager

# Advanced debugging with pdb
def pdb_debugging_demo():
    """Demonstrate advanced pdb debugging techniques"""
    
    class DebugExample:
        def __init__(self, value: int):
            self.value = value
            self.history = []
        
        def process_data(self, data: List[int]) -> List[int]:
            results = []
            for i, item in enumerate(data):
                # Conditional breakpoint example
                if item < 0:
                    # pdb.set_trace()  # Would stop here if uncommented
                    self.history.append(f"Negative value at index {i}: {item}")
                
                processed = self._transform_item(item)
                results.append(processed)
            
            return results
        
        def _transform_item(self, item: int) -> int:
            # Complex transformation that might need debugging
            if item == 0:
                return 0
            
            result = item * self.value
            if result > 100:
                result = result // 2
            
            return result
        
        def debug_method(self, x: int, y: int) -> int:
            """Method specifically for debugging demonstration"""
            a = x * 2
            b = y + 5
            c = a + b
            
            # Post-mortem debugging setup
            try:
                if c > 50:
                    raise ValueError(f"Result too large: {c}")
                return c
            except Exception:
                # In real debugging, you'd call pdb.post_mortem()
                return -1
    
    # Programmatic debugging interface
    class DebugTracer:
        def __init__(self):
            self.trace_calls = []
            self.trace_lines = []
            self.trace_returns = []
        
        def trace_function(self, frame, event, arg):
            """Custom trace function for debugging"""
            if event == 'call':
                self.trace_calls.append({
                    'function': frame.f_code.co_name,
                    'filename': frame.f_code.co_filename,
                    'lineno': frame.f_lineno,
                    'locals': dict(frame.f_locals)
                })
            elif event == 'line':
                self.trace_lines.append({
                    'function': frame.f_code.co_name,
                    'lineno': frame.f_lineno
                })
            elif event == 'return':
                self.trace_returns.append({
                    'function': frame.f_code.co_name,
                    'return_value': arg
                })
            
            return self.trace_function
        
        def start_tracing(self):
            sys.settrace(self.trace_function)
        
        def stop_tracing(self):
            sys.settrace(None)
        
        def get_stats(self):
            return {
                'calls': len(self.trace_calls),
                'lines': len(self.trace_lines),
                'returns': len(self.trace_returns)
            }
    
    # Interactive debugging helpers
    def debug_decorator(func):
        """Decorator that adds debugging capabilities to functions"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Pre-execution debugging info
            debug_info = {
                'function_name': func.__name__,
                'args': args,
                'kwargs': kwargs,
                'start_time': time.time()
            }
            
            try:
                result = func(*args, **kwargs)
                debug_info['result'] = result
                debug_info['success'] = True
            except Exception as e:
                debug_info['exception'] = str(e)
                debug_info['success'] = False
                # In real debugging, might call pdb.post_mortem() here
                raise
            finally:
                debug_info['end_time'] = time.time()
                debug_info['duration'] = debug_info['end_time'] - debug_info['start_time']
                
                # Store debug info (in real app, might log or store differently)
                if not hasattr(wrapper, 'debug_history'):
                    wrapper.debug_history = []
                wrapper.debug_history.append(debug_info)
            
            return result
        return wrapper
    
    @debug_decorator
    def problematic_function(numbers: List[int]) -> int:
        total = 0
        for num in numbers:
            if num > 10:
                total += num * 2
            else:
                total += num
        return total
    
    # Test debugging features
    example = DebugExample(5)
    test_data = [1, -2, 15, 0, 25, -5]
    
    processed_results = example.process_data(test_data)
    debug_result = example.debug_method(10, 15)
    
    # Test tracer
    tracer = DebugTracer()
    tracer.start_tracing()
    
    traced_result = problematic_function([5, 15, 20])
    
    tracer.stop_tracing()
    trace_stats = tracer.get_stats()
    
    # Test decorated function
    decorated_result = problematic_function([1, 2, 3, 11, 12])
    debug_history = getattr(problematic_function, 'debug_history', [])
    
    return {
        'debug_example': {
            'processed_results': processed_results,
            'debug_method_result': debug_result,
            'history_count': len(example.history)
        },
        'tracing': {
            'traced_result': traced_result,
            'trace_stats': trace_stats
        },
        'decorated_debugging': {
            'result': decorated_result,
            'debug_calls': len(debug_history),
            'last_call_duration': debug_history[-1]['duration'] if debug_history else 0
        }
    }

# Code introspection and inspection
def introspection_demo():
    """Demonstrate advanced code introspection techniques"""
    
    class InspectionTarget:
        """Class for demonstrating inspection capabilities"""
        
        class_variable = "I'm a class variable"
        
        def __init__(self, name: str, value: int):
            self.name = name
            self.value = value
            self._private = "private data"
        
        def public_method(self, x: int) -> str:
            """A public method for testing"""
            return f"{self.name}: {x * self.value}"
        
        def _private_method(self) -> str:
            """A private method for testing"""
            return self._private
        
        @staticmethod
        def static_method(a: int, b: int) -> int:
            """A static method for testing"""
            return a + b
        
        @classmethod
        def class_method(cls, name: str) -> 'InspectionTarget':
            """A class method for testing"""
            return cls(name, 42)
        
        @property
        def computed_property(self) -> str:
            """A computed property for testing"""
            return f"computed_{self.name}_{self.value}"
    
    def analyze_object(obj: Any) -> Dict[str, Any]:
        """Comprehensive object analysis"""
        analysis = {
            'type': type(obj).__name__,
            'module': getattr(type(obj), '__module__', 'unknown'),
            'id': id(obj),
            'size': sys.getsizeof(obj),
            'is_callable': callable(obj),
            'has_dict': hasattr(obj, '__dict__'),
            'dict_size': len(obj.__dict__) if hasattr(obj, '__dict__') else 0
        }
        
        # Get all attributes
        all_attrs = dir(obj)
        analysis['total_attributes'] = len(all_attrs)
        
        # Categorize attributes
        public_attrs = [attr for attr in all_attrs if not attr.startswith('_')]
        private_attrs = [attr for attr in all_attrs if attr.startswith('_') and not attr.startswith('__')]
        dunder_attrs = [attr for attr in all_attrs if attr.startswith('__') and attr.endswith('__')]
        
        analysis['public_attributes'] = len(public_attrs)
        analysis['private_attributes'] = len(private_attrs)
        analysis['dunder_attributes'] = len(dunder_attrs)
        
        # Analyze methods
        methods = []
        for attr_name in all_attrs:
            try:
                attr = getattr(obj, attr_name)
                if callable(attr):
                    method_info = {
                        'name': attr_name,
                        'type': type(attr).__name__,
                        'is_method': inspect.ismethod(attr),
                        'is_function': inspect.isfunction(attr),
                        'is_builtin': inspect.isbuiltin(attr)
                    }
                    
                    # Get signature if possible
                    try:
                        sig = inspect.signature(attr)
                        method_info['signature'] = str(sig)
                        method_info['parameters'] = len(sig.parameters)
                    except (ValueError, TypeError):
                        method_info['signature'] = 'unavailable'
                        method_info['parameters'] = 0
                    
                    methods.append(method_info)
            except Exception:
                pass
        
        analysis['methods'] = methods[:5]  # Limit for brevity
        analysis['method_count'] = len(methods)
        
        return analysis
    
    def analyze_function(func: Callable) -> Dict[str, Any]:
        """Detailed function analysis"""
        try:
            signature = inspect.signature(func)
            source_lines = inspect.getsourcelines(func)
            
            analysis = {
                'name': func.__name__,
                'module': func.__module__,
                'doc': func.__doc__,
                'signature': str(signature),
                'parameter_count': len(signature.parameters),
                'source_file': inspect.getfile(func),
                'source_line_count': len(source_lines[0]),
                'starting_line': source_lines[1]
            }
            
            # Analyze parameters
            parameters = []
            for param_name, param in signature.parameters.items():
                param_info = {
                    'name': param_name,
                    'kind': param.kind.name,
                    'default': str(param.default) if param.default != param.empty else None,
                    'annotation': str(param.annotation) if param.annotation != param.empty else None
                }
                parameters.append(param_info)
            
            analysis['parameters'] = parameters
            
            # Get bytecode information
            code_obj = func.__code__
            analysis['bytecode_info'] = {
                'arg_count': code_obj.co_argcount,
                'kwonly_arg_count': code_obj.co_kwonlyargcount,
                'local_count': code_obj.co_nlocals,
                'stack_size': code_obj.co_stacksize,
                'code_size': len(code_obj.co_code),
                'constants': len(code_obj.co_consts),
                'names': len(code_obj.co_names)
            }
            
            return analysis
            
        except Exception as e:
            return {'error': str(e), 'name': getattr(func, '__name__', 'unknown')}
    
    def analyze_class(cls: Type) -> Dict[str, Any]:
        """Detailed class analysis"""
        analysis = {
            'name': cls.__name__,
            'module': cls.__module__,
            'doc': cls.__doc__,
            'bases': [base.__name__ for base in cls.__bases__],
            'mro': [c.__name__ for c in cls.__mro__],
            'is_abstract': inspect.isabstract(cls)
        }
        
        # Get class attributes
        class_attrs = {}
        for name, value in inspect.getmembers(cls):
            if not name.startswith('__') or name in ['__init__', '__str__', '__repr__']:
                attr_info = {
                    'type': type(value).__name__,
                    'is_method': inspect.ismethod(value),
                    'is_function': inspect.isfunction(value),
                    'is_property': isinstance(value, property),
                    'is_static': isinstance(inspect.getattr_static(cls, name), staticmethod),
                    'is_class_method': isinstance(inspect.getattr_static(cls, name), classmethod)
                }
                class_attrs[name] = attr_info
        
        analysis['attributes'] = class_attrs
        analysis['attribute_count'] = len(class_attrs)
        
        return analysis
    
    # Test introspection
    target = InspectionTarget("test", 10)
    
    object_analysis = analyze_object(target)
    function_analysis = analyze_function(target.public_method)
    class_analysis = analyze_class(InspectionTarget)
    
    # Stack frame inspection
    def get_stack_info() -> List[Dict[str, Any]]:
        """Get information about the current call stack"""
        stack_info = []
        for frame_info in inspect.stack():
            info = {
                'filename': frame_info.filename,
                'function': frame_info.function,
                'lineno': frame_info.lineno,
                'code': frame_info.code_context[0].strip() if frame_info.code_context else None
            }
            stack_info.append(info)
        return stack_info[:5]  # Limit to top 5 frames
    
    stack_analysis = get_stack_info()
    
    # Module introspection
    current_module = inspect.getmodule(introspection_demo)
    module_analysis = {
        'name': current_module.__name__ if current_module else 'unknown',
        'file': current_module.__file__ if current_module else 'unknown',
        'functions': len([name for name, obj in inspect.getmembers(current_module, inspect.isfunction)]) if current_module else 0,
        'classes': len([name for name, obj in inspect.getmembers(current_module, inspect.isclass)]) if current_module else 0
    }
    
    return {
        'object_analysis': object_analysis,
        'function_analysis': function_analysis,
        'class_analysis': class_analysis,
        'stack_analysis': stack_analysis,
        'module_analysis': module_analysis
    }

# Advanced logging and monitoring
def logging_monitoring_demo():
    """Demonstrate advanced logging and monitoring techniques"""
    
    # Custom log formatter with extra context
    class ContextFormatter(logging.Formatter):
        def format(self, record):
            # Add extra context to log records
            if not hasattr(record, 'context'):
                record.context = {}
            
            # Add current thread info
            record.context['thread_id'] = threading.current_thread().ident
            record.context['thread_name'] = threading.current_thread().name
            
            # Add function context
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_frame = frame.f_back.f_back
                record.context['caller_function'] = caller_frame.f_code.co_name
                record.context['caller_line'] = caller_frame.f_lineno
            
            # Format with context
            original_msg = super().format(record)
            context_str = ' | '.join(f"{k}={v}" for k, v in record.context.items())
            return f"{original_msg} | Context: {context_str}"
    
    # Performance monitoring decorator
    def monitor_performance(log_slow_calls=True, threshold_ms=100):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.time()
                    end_memory = get_memory_usage()
                    
                    duration_ms = (end_time - start_time) * 1000
                    memory_delta = end_memory - start_memory
                    
                    # Log performance data
                    perf_data = {
                        'function': func.__name__,
                        'duration_ms': duration_ms,
                        'memory_delta_mb': memory_delta,
                        'success': success,
                        'error': error
                    }
                    
                    if not hasattr(wrapper, 'performance_log'):
                        wrapper.performance_log = []
                    wrapper.performance_log.append(perf_data)
                    
                    # Log slow calls
                    if log_slow_calls and duration_ms > threshold_ms:
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Slow call detected: {func.__name__} took {duration_ms:.2f}ms")
                
                return result
            return wrapper
        return decorator
    
    def get_memory_usage() -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    # Error tracking and reporting
    class ErrorTracker:
        def __init__(self):
            self.errors = []
            self.error_counts = {}
        
        def track_error(self, error: Exception, context: Dict[str, Any] = None):
            error_info = {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc(),
                'timestamp': time.time(),
                'context': context or {}
            }
            
            self.errors.append(error_info)
            
            error_key = f"{error_info['type']}:{error_info['message']}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        def get_error_summary(self) -> Dict[str, Any]:
            return {
                'total_errors': len(self.errors),
                'unique_errors': len(self.error_counts),
                'most_common': max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None,
                'recent_errors': self.errors[-5:] if self.errors else []
            }
    
    # Custom logging handler for structured logging
    class StructuredLogHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.logs = []
        
        def emit(self, record):
            log_entry = {
                'timestamp': record.created,
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'thread': record.thread
            }
            
            # Add extra fields if present
            if hasattr(record, 'extra_data'):
                log_entry['extra'] = record.extra_data
            
            self.logs.append(log_entry)
    
    # Set up logging with custom formatter
    logger = logging.getLogger('monitoring_demo')
    logger.setLevel(logging.DEBUG)
    
    # Create handler with custom formatter
    handler = StructuredLogHandler()
    formatter = ContextFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Test performance monitoring
    @monitor_performance(threshold_ms=50)
    def monitored_function(n: int) -> int:
        # Simulate some work
        total = 0
        for i in range(n):
            total += i ** 2
        time.sleep(0.01)  # Simulate I/O
        return total
    
    @monitor_performance(threshold_ms=10)
    def fast_function(x: int) -> int:
        return x * 2
    
    # Error tracking demo
    error_tracker = ErrorTracker()
    
    def error_prone_function(value: int) -> float:
        try:
            if value == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            elif value < 0:
                raise ValueError("Negative values not allowed")
            else:
                return 100.0 / value
        except Exception as e:
            error_tracker.track_error(e, {'input_value': value})
            raise
    
    # Test monitoring and logging
    monitored_result1 = monitored_function(1000)
    monitored_result2 = fast_function(42)
    
    # Test error tracking
    error_results = []
    for test_value in [10, 0, -5, 20]:
        try:
            result = error_prone_function(test_value)
            error_results.append(f"Success: {result}")
        except Exception:
            error_results.append(f"Error with {test_value}")
    
    # Log some test messages
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message", extra={'extra_data': {'test': True}})
    
    # Get performance data
    perf_data_monitored = getattr(monitored_function, 'performance_log', [])
    perf_data_fast = getattr(fast_function, 'performance_log', [])
    
    return {
        'performance_monitoring': {
            'monitored_function_calls': len(perf_data_monitored),
            'fast_function_calls': len(perf_data_fast),
            'monitored_avg_duration': sum(p['duration_ms'] for p in perf_data_monitored) / len(perf_data_monitored) if perf_data_monitored else 0,
            'results': [monitored_result1, monitored_result2]
        },
        'error_tracking': {
            'error_summary': error_tracker.get_error_summary(),
            'test_results': error_results
        },
        'structured_logging': {
            'log_count': len(handler.logs),
            'log_levels': [log['level'] for log in handler.logs],
            'recent_messages': [log['message'] for log in handler.logs[-3:]]
        }
    }

# Memory debugging and leak detection
def memory_debugging_demo():
    """Demonstrate memory debugging and leak detection"""
    
    # Memory tracking decorator
    def track_memory(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import gc
            
            # Force garbage collection
            gc.collect()
            
            # Get initial state
            initial_objects = len(gc.get_objects())
            initial_memory = get_memory_usage() if 'psutil' in sys.modules else 0
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Force garbage collection again
                gc.collect()
                
                # Get final state
                final_objects = len(gc.get_objects())
                final_memory = get_memory_usage() if 'psutil' in sys.modules else 0
                
                # Store tracking info
                if not hasattr(wrapper, 'memory_tracking'):
                    wrapper.memory_tracking = []
                
                tracking_info = {
                    'function': func.__name__,
                    'object_delta': final_objects - initial_objects,
                    'memory_delta_mb': final_memory - initial_memory,
                    'initial_objects': initial_objects,
                    'final_objects': final_objects
                }
                wrapper.memory_tracking.append(tracking_info)
            
            return result
        return wrapper
    
    def get_memory_usage() -> float:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    # Leak detection utilities
    class LeakDetector:
        def __init__(self):
            self.snapshots = []
            self.tracked_objects = weakref.WeakSet()
        
        def take_snapshot(self, label: str):
            gc.collect()  # Force cleanup
            
            snapshot = {
                'label': label,
                'timestamp': time.time(),
                'object_count': len(gc.get_objects()),
                'memory_usage': get_memory_usage(),
                'tracked_objects': len(self.tracked_objects)
            }
            
            self.snapshots.append(snapshot)
        
        def track_object(self, obj):
            """Track an object for potential leaks"""
            self.tracked_objects.add(obj)
        
        def get_leak_report(self) -> Dict[str, Any]:
            if len(self.snapshots) < 2:
                return {'error': 'Need at least 2 snapshots to detect leaks'}
            
            first = self.snapshots[0]
            last = self.snapshots[-1]
            
            return {
                'time_span': last['timestamp'] - first['timestamp'],
                'object_growth': last['object_count'] - first['object_count'],
                'memory_growth_mb': last['memory_usage'] - first['memory_usage'],
                'tracked_objects_remaining': last['tracked_objects'],
                'snapshots_taken': len(self.snapshots),
                'growth_rate_objects_per_sec': (last['object_count'] - first['object_count']) / (last['timestamp'] - first['timestamp']) if last['timestamp'] != first['timestamp'] else 0
            }
    
    # Test classes for memory analysis
    class LeakyClass:
        """Class that demonstrates memory leaks"""
        _instances = []  # This will cause a leak
        
        def __init__(self, data):
            self.data = data
            LeakyClass._instances.append(self)  # Leak: never removed
    
    class CleanClass:
        """Class that doesn't leak memory"""
        _instances = weakref.WeakSet()  # Uses weak references
        
        def __init__(self, data):
            self.data = data
            CleanClass._instances.add(self)
    
    @track_memory
    def create_leaky_objects(count: int):
        """Function that creates objects with potential leaks"""
        objects = []
        for i in range(count):
            obj = LeakyClass(f"data_{i}")
            objects.append(obj)
        return len(objects)
    
    @track_memory
    def create_clean_objects(count: int):
        """Function that creates objects without leaks"""
        objects = []
        for i in range(count):
            obj = CleanClass(f"data_{i}")
            objects.append(obj)
        # Objects go out of scope and can be garbage collected
        return len(objects)
    
    @track_memory
    def memory_intensive_operation():
        """Operation that uses a lot of memory temporarily"""
        large_list = [i ** 2 for i in range(10000)]
        processed = [x * 2 for x in large_list if x % 2 == 0]
        return len(processed)
    
    # Garbage collection analysis
    def analyze_gc_state():
        """Analyze garbage collection state"""
        return {
            'gc_enabled': gc.isenabled(),
            'gc_counts': gc.get_count(),
            'gc_stats': gc.get_stats(),
            'gc_threshold': gc.get_threshold(),
            'tracked_objects': len(gc.get_objects()),
            'garbage_count': len(gc.garbage)
        }
    
    # Run memory debugging tests
    detector = LeakDetector()
    
    # Initial snapshot
    detector.take_snapshot("initial")
    initial_gc_state = analyze_gc_state()
    
    # Create some objects
    leaky_result = create_leaky_objects(100)
    detector.take_snapshot("after_leaky_objects")
    
    clean_result = create_clean_objects(100)
    detector.take_snapshot("after_clean_objects")
    
    # Memory intensive operation
    intensive_result = memory_intensive_operation()
    detector.take_snapshot("after_intensive_operation")
    
    # Force garbage collection
    gc.collect()
    detector.take_snapshot("after_gc")
    
    final_gc_state = analyze_gc_state()
    leak_report = detector.get_leak_report()
    
    # Get tracking data
    leaky_tracking = getattr(create_leaky_objects, 'memory_tracking', [])
    clean_tracking = getattr(create_clean_objects, 'memory_tracking', [])
    intensive_tracking = getattr(memory_intensive_operation, 'memory_tracking', [])
    
    return {
        'object_creation': {
            'leaky_objects_created': leaky_result,
            'clean_objects_created': clean_result,
            'intensive_operation_result': intensive_result
        },
        'memory_tracking': {
            'leaky_function': leaky_tracking[-1] if leaky_tracking else {},
            'clean_function': clean_tracking[-1] if clean_tracking else {},
            'intensive_function': intensive_tracking[-1] if intensive_tracking else {}
        },
        'gc_analysis': {
            'initial_state': initial_gc_state,
            'final_state': final_gc_state,
            'object_growth': final_gc_state['tracked_objects'] - initial_gc_state['tracked_objects']
        },
        'leak_detection': leak_report
    }

# Production debugging and monitoring
def production_debugging_demo():
    """Demonstrate production debugging techniques"""
    
    # Signal handling for debugging
    def setup_debug_signals():
        """Set up signal handlers for production debugging"""
        import signal
        
        def debug_handler(signum, frame):
            """Signal handler that dumps debug information"""
            debug_info = {
                'signal': signum,
                'timestamp': time.time(),
                'frame_info': {
                    'filename': frame.f_code.co_filename,
                    'function': frame.f_code.co_name,
                    'line': frame.f_lineno
                },
                'locals': dict(frame.f_locals),
                'stack_depth': len(inspect.stack())
            }
            
            # In production, would log this information
            print(f"Debug signal received: {debug_info}")
            return debug_info
        
        # Set up signal handlers (Unix only)
        try:
            signal.signal(signal.SIGUSR1, debug_handler)
            signal.signal(signal.SIGUSR2, debug_handler)
            return True
        except (AttributeError, OSError):
            # Windows or other platforms
            return False
    
    # Health check system
    class HealthChecker:
        def __init__(self):
            self.checks = {}
            self.history = []
        
        def register_check(self, name: str, check_func: Callable[[], bool]):
            """Register a health check function"""
            self.checks[name] = check_func
        
        def run_checks(self) -> Dict[str, Any]:
            """Run all health checks"""
            results = {}
            overall_healthy = True
            
            for name, check_func in self.checks.items():
                try:
                    start_time = time.time()
                    result = check_func()
                    duration = time.time() - start_time
                    
                    results[name] = {
                        'healthy': bool(result),
                        'duration_ms': duration * 1000,
                        'error': None
                    }
                    
                    if not result:
                        overall_healthy = False
                        
                except Exception as e:
                    results[name] = {
                        'healthy': False,
                        'duration_ms': 0,
                        'error': str(e)
                    }
                    overall_healthy = False
            
            health_report = {
                'timestamp': time.time(),
                'overall_healthy': overall_healthy,
                'checks': results,
                'total_checks': len(self.checks),
                'failed_checks': sum(1 for r in results.values() if not r['healthy'])
            }
            
            self.history.append(health_report)
            return health_report
        
        def get_health_summary(self) -> Dict[str, Any]:
            """Get summary of health check history"""
            if not self.history:
                return {'error': 'No health checks performed yet'}
            
            recent_checks = self.history[-10:]  # Last 10 checks
            healthy_count = sum(1 for check in recent_checks if check['overall_healthy'])
            
            return {
                'total_checks_performed': len(self.history),
                'recent_healthy_percentage': (healthy_count / len(recent_checks)) * 100,
                'last_check_time': self.history[-1]['timestamp'],
                'last_check_healthy': self.history[-1]['overall_healthy'],
                'average_check_count': sum(check['total_checks'] for check in recent_checks) / len(recent_checks)
            }
    
    # Runtime metrics collection
    class MetricsCollector:
        def __init__(self):
            self.metrics = {}
            self.counters = {}
            self.timers = {}
        
        def increment_counter(self, name: str, value: int = 1):
            """Increment a counter metric"""
            self.counters[name] = self.counters.get(name, 0) + value
        
        def record_timing(self, name: str, duration_ms: float):
            """Record a timing metric"""
            if name not in self.timers:
                self.timers[name] = []
            self.timers[name].append(duration_ms)
            
            # Keep only recent timings (last 100)
            if len(self.timers[name]) > 100:
                self.timers[name] = self.timers[name][-100:]
        
        def set_gauge(self, name: str, value: float):
            """Set a gauge metric"""
            self.metrics[name] = value
        
        def get_metrics_summary(self) -> Dict[str, Any]:
            """Get summary of all metrics"""
            timer_summary = {}
            for name, timings in self.timers.items():
                if timings:
                    timer_summary[name] = {
                        'count': len(timings),
                        'avg_ms': sum(timings) / len(timings),
                        'min_ms': min(timings),
                        'max_ms': max(timings),
                        'recent_ms': timings[-1]
                    }
            
            return {
                'counters': self.counters.copy(),
                'gauges': self.metrics.copy(),
                'timers': timer_summary,
                'collection_time': time.time()
            }
    
    @contextmanager
    def timing_context(metrics_collector: MetricsCollector, metric_name: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            metrics_collector.record_timing(metric_name, duration_ms)
    
    # Sample application components for monitoring
    def database_health_check() -> bool:
        """Simulate database health check"""
        # Simulate database connectivity check
        time.sleep(0.01)  # Simulate network delay
        return random.choice([True, True, True, False])  # 75% success rate
    
    def cache_health_check() -> bool:
        """Simulate cache health check"""
        time.sleep(0.005)  # Simulate cache check
        return True  # Cache always healthy in this demo
    
    def external_api_health_check() -> bool:
        """Simulate external API health check"""
        time.sleep(0.02)  # Simulate API call
        return random.choice([True, True, False])  # 66% success rate
    
    def simulate_application_work(metrics: MetricsCollector):
        """Simulate some application work with metrics"""
        # Increment request counter
        metrics.increment_counter('requests_total')
        
        # Simulate different operations with timing
        with timing_context(metrics, 'database_query'):
            time.sleep(0.01)  # Simulate DB query
        
        with timing_context(metrics, 'cache_lookup'):
            time.sleep(0.001)  # Simulate cache lookup
        
        # Set some gauge metrics
        metrics.set_gauge('active_connections', random.randint(10, 50))
        metrics.set_gauge('memory_usage_percent', random.uniform(40, 80))
        
        return "Work completed"
    
    # Set up production debugging
    signal_setup = setup_debug_signals()
    
    # Set up health checking
    health_checker = HealthChecker()
    health_checker.register_check('database', database_health_check)
    health_checker.register_check('cache', cache_health_check)
    health_checker.register_check('external_api', external_api_health_check)
    
    # Set up metrics collection
    metrics = MetricsCollector()
    
    # Simulate application running
    work_results = []
    for i in range(5):
        result = simulate_application_work(metrics)
        work_results.append(result)
        
        # Occasionally run health checks
        if i % 2 == 0:
            health_report = health_checker.run_checks()
    
    # Get final reports
    final_health_summary = health_checker.get_health_summary()
    final_metrics_summary = metrics.get_metrics_summary()
    
    return {
        'signal_handling': {
            'signals_setup': signal_setup
        },
        'health_monitoring': {
            'health_summary': final_health_summary,
            'checks_registered': len(health_checker.checks)
        },
        'metrics_collection': {
            'metrics_summary': final_metrics_summary,
            'work_iterations': len(work_results)
        },
        'application_simulation': {
            'work_completed': len([r for r in work_results if r == "Work completed"]),
            'total_operations': len(work_results)
        }
    }

# Comprehensive testing
def run_all_debugging_demos():
    """Execute all debugging and introspection demonstrations"""
    demo_functions = [
        ('pdb_debugging', pdb_debugging_demo),
        ('introspection', introspection_demo),
        ('logging_monitoring', logging_monitoring_demo),
        ('memory_debugging', memory_debugging_demo),
        ('production_debugging', production_debugging_demo)
    ]
    
    results = {}
    for name, func in demo_functions:
        try:
            result = func()
            results[name] = result
        except Exception as e:
            results[name] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    return results

if __name__ == "__main__":
    print("=== Python Debugging and Introspection Demo ===")
    
    # Run all demonstrations
    all_results = run_all_debugging_demos()
    
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
    
    print("\n=== DEBUGGING TOOLS ===")
    
    debugging_tools = {
        "pdb": "Interactive Python debugger for step-by-step debugging",
        "pdb++": "Enhanced pdb with syntax highlighting and better interface",
        "ipdb": "IPython-enhanced debugger with better features",
        "pudb": "Full-screen console debugger with visual interface",
        "VS Code Debugger": "Integrated debugging in VS Code IDE",
        "PyCharm Debugger": "Professional debugging tools in PyCharm",
        "remote-pdb": "Remote debugging over network connections",
        "web-pdb": "Web-based debugger interface"
    }
    
    for tool, description in debugging_tools.items():
        print(f"  {tool}: {description}")
    
    print("\n=== INTROSPECTION CAPABILITIES ===")
    
    introspection_features = {
        "inspect module": "Comprehensive object introspection and source analysis",
        "dir() function": "List object attributes and methods",
        "vars() function": "Get object's __dict__ attribute",
        "getattr/hasattr": "Dynamic attribute access and checking",
        "type() and isinstance()": "Type checking and identification",
        "sys module": "System-specific parameters and functions",
        "gc module": "Garbage collection monitoring and control",
        "traceback module": "Exception traceback formatting and analysis",
        "dis module": "Bytecode disassembly and analysis",
        "ast module": "Abstract syntax tree parsing and manipulation"
    }
    
    for feature, description in introspection_features.items():
        print(f"  {feature}: {description}")
    
    print("\n=== PRODUCTION DEBUGGING STRATEGIES ===")
    
    production_strategies = [
        "Comprehensive logging with structured data",
        "Health check endpoints for service monitoring",
        "Metrics collection and alerting",
        "Signal handlers for runtime debugging",
        "Memory leak detection and monitoring",
        "Performance profiling in production",
        "Error tracking and aggregation",
        "Distributed tracing for microservices",
        "Circuit breakers for resilience",
        "Graceful degradation mechanisms",
        "Rolling deployments with monitoring",
        "Automated rollback on errors"
    ]
    
    for strategy in production_strategies:
        print(f"  • {strategy}")
    
    print("\n=== MONITORING BEST PRACTICES ===")
    
    monitoring_practices = [
        "Use structured logging with consistent formats",
        "Implement health checks for all dependencies",
        "Monitor key performance indicators (KPIs)",
        "Set up alerting for critical failures",
        "Use distributed tracing for request flows",
        "Monitor memory usage and garbage collection",
        "Track error rates and types",
        "Implement circuit breakers for external services",
        "Use metrics dashboards for visualization",
        "Set up automated testing in production",
        "Monitor business metrics alongside technical metrics",
        "Implement gradual rollouts with monitoring"
    ]
    
    for practice in monitoring_practices:
        print(f"  • {practice}")
    
    print("\n=== DEBUGGING BEST PRACTICES ===")
    
    debugging_practices = [
        "Reproduce issues in development environment first",
        "Use logging strategically - not too much, not too little",
        "Implement proper exception handling and reporting",
        "Use debugger breakpoints instead of print statements",
        "Write tests that reproduce the bug",
        "Use version control to track when bugs were introduced",
        "Document debugging steps and solutions",
        "Use static analysis tools to catch issues early",
        "Implement comprehensive error handling",
        "Use profilers to identify performance bottlenecks",
        "Keep debug code out of production",
        "Practice defensive programming techniques"
    ]
    
    for practice in debugging_practices:
        print(f"  • {practice}")
    
    print("\n=== Debugging and Introspection Complete! ===")
    print("  Advanced debugging techniques and code analysis mastered")
