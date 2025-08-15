"""
Python Advanced Decorators: Function and Class Decorators with Complex Patterns
Implementation-focused with minimal comments, maximum functionality coverage
"""

import functools
import time
import threading
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import weakref
import inspect
from collections import defaultdict
import warnings

# Basic decorator patterns
def simple_decorator(func):
    """Basic decorator without functools.wraps"""
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__}")
        return result
    return wrapper

def proper_decorator(func):
    """Properly implemented decorator with metadata preservation"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Executing {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

def timing_decorator(func):
    """Measure execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@simple_decorator
def basic_function():
    return "Basic function result"

@proper_decorator
@timing_decorator
def complex_function(x, y):
    """Add two numbers after a delay"""
    time.sleep(0.1)
    return x + y

def basic_decorator_demo():
    result1 = basic_function()
    result2 = complex_function(5, 3)
    
    return {
        "basic_result": result1,
        "complex_result": result2,
        "basic_name": basic_function.__name__,  # Lost original name
        "complex_name": complex_function.__name__,  # Preserved
        "complex_doc": complex_function.__doc__  # Preserved
    }

# Decorators with arguments
def repeat(times):
    """Decorator factory that repeats function execution"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(times):
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator

def validate_types(**expected_types):
    """Decorator that validates argument types"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate types
            for param_name, expected_type in expected_types.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"{param_name} must be {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def cache_with_ttl(ttl_seconds=60):
    """Decorator that caches results with time-to-live"""
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < ttl_seconds:
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            return result
        
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {"size": len(cache), "ttl": ttl_seconds}
        return wrapper
    return decorator

@repeat(3)
def simple_greeting(name):
    return f"Hello, {name}!"

@validate_types(x=int, y=int)
def add_numbers(x, y):
    return x + y

@cache_with_ttl(5)
def expensive_operation(n):
    time.sleep(0.1)
    return n ** 2

def parametrized_decorator_demo():
    # Test repeat decorator
    greetings = simple_greeting("Alice")
    
    # Test type validation
    valid_sum = add_numbers(5, 3)
    
    try:
        invalid_sum = add_numbers("5", 3)
        type_error = False
    except TypeError as e:
        type_error = str(e)
    
    # Test caching
    start_time = time.time()
    result1 = expensive_operation(5)
    first_call_time = time.time() - start_time
    
    start_time = time.time()
    result2 = expensive_operation(5)  # Should be cached
    second_call_time = time.time() - start_time
    
    cache_info = expensive_operation.cache_info()
    
    return {
        "repeated_greetings": greetings,
        "valid_sum": valid_sum,
        "type_validation_error": type_error,
        "expensive_results": [result1, result2],
        "timing_comparison": {
            "first_call": f"{first_call_time:.4f}s",
            "second_call": f"{second_call_time:.4f}s",
            "cached": second_call_time < first_call_time
        },
        "cache_info": cache_info
    }

# Class decorators
def singleton(cls):
    """Class decorator for singleton pattern"""
    instances = {}
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

def add_methods(methods_dict):
    """Class decorator that adds methods dynamically"""
    def decorator(cls):
        for method_name, method_func in methods_dict.items():
            setattr(cls, method_name, method_func)
        return cls
    return decorator

def track_instances(cls):
    """Class decorator that tracks all instances"""
    original_init = cls.__init__
    cls._instances = weakref.WeakSet()
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        cls._instances.add(self)
    
    cls.__init__ = new_init
    cls.get_instances = classmethod(lambda cls: list(cls._instances))
    cls.instance_count = classmethod(lambda cls: len(cls._instances))
    return cls

@singleton
class DatabaseConnection:
    def __init__(self, connection_string="default"):
        self.connection_string = connection_string
        self.connected = False
    
    def connect(self):
        self.connected = True
        return f"Connected to {self.connection_string}"

@add_methods({
    'bark': lambda self: f"{self.name} says Woof!",
    'wag_tail': lambda self: f"{self.name} wags tail"
})
class Dog:
    def __init__(self, name):
        self.name = name

@track_instances
class User:
    def __init__(self, username):
        self.username = username

def class_decorator_demo():
    # Test singleton
    db1 = DatabaseConnection("db1")
    db2 = DatabaseConnection("db2")  # Should be same instance
    
    # Test method addition
    dog = Dog("Buddy")
    
    # Test instance tracking
    user1 = User("alice")
    user2 = User("bob")
    user3 = User("charlie")
    
    # Create weak reference to test cleanup
    temp_user = User("temp")
    temp_user_id = id(temp_user)
    del temp_user  # Should be removed from tracking
    
    return {
        "singleton_test": {
            "same_instance": db1 is db2,
            "connection1": db1.connect(),
            "connection_string": db2.connection_string
        },
        "method_addition": {
            "bark": dog.bark(),
            "wag_tail": dog.wag_tail(),
            "has_bark_method": hasattr(Dog, 'bark')
        },
        "instance_tracking": {
            "user_count": User.instance_count(),
            "user_list": [u.username for u in User.get_instances()],
            "temp_user_cleaned": temp_user_id not in [id(u) for u in User.get_instances()]
        }
    }

# Advanced decorator patterns
class CountCalls:
    """Decorator class that counts function calls"""
    def __init__(self, func):
        self.func = func
        self.call_count = 0
        functools.update_wrapper(self, func)
    
    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self.func(*args, **kwargs)
    
    def reset_count(self):
        self.call_count = 0

class RateLimiter:
    """Decorator class for rate limiting"""
    def __init__(self, max_calls=10, time_window=60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Remove old calls outside time window
            self.calls = [call_time for call_time in self.calls 
                         if current_time - call_time < self.time_window]
            
            if len(self.calls) >= self.max_calls:
                raise Exception(f"Rate limit exceeded: {self.max_calls} calls per {self.time_window}s")
            
            self.calls.append(current_time)
            return func(*args, **kwargs)
        
        wrapper.reset_rate_limit = lambda: self.calls.clear()
        return wrapper

def memoize_with_size_limit(max_size=128):
    """Advanced memoization with LRU-style size limit"""
    def decorator(func):
        cache = {}
        access_order = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                # Move to end (most recently used)
                access_order.remove(key)
                access_order.append(key)
                return cache[key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Check size limit
            if len(cache) >= max_size:
                # Remove least recently used
                lru_key = access_order.pop(0)
                del cache[lru_key]
            
            cache[key] = result
            access_order.append(key)
            return result
        
        wrapper.cache_clear = lambda: (cache.clear(), access_order.clear())
        wrapper.cache_info = lambda: {
            "hits": len([k for k in access_order if k in cache]),
            "size": len(cache),
            "max_size": max_size
        }
        return wrapper
    return decorator

@CountCalls
def counted_function(x):
    return x * 2

@RateLimiter(max_calls=3, time_window=2)
def rate_limited_function():
    return "Success"

@memoize_with_size_limit(max_size=3)
def fibonacci_with_limit(n):
    if n <= 1:
        return n
    return fibonacci_with_limit(n-1) + fibonacci_with_limit(n-2)

def advanced_decorator_demo():
    # Test call counting
    results = []
    for i in range(5):
        results.append(counted_function(i))
    
    call_count = counted_function.call_count
    
    # Test rate limiting
    rate_limit_results = []
    try:
        for i in range(5):
            rate_limit_results.append(rate_limited_function())
    except Exception as e:
        rate_limit_error = str(e)
    
    # Test LRU memoization
    fib_results = []
    for i in range(8):
        fib_results.append(fibonacci_with_limit(i))
    
    cache_info = fibonacci_with_limit.cache_info()
    
    return {
        "call_counting": {
            "results": results,
            "call_count": call_count
        },
        "rate_limiting": {
            "successful_calls": len(rate_limit_results),
            "rate_limit_error": rate_limit_error if 'rate_limit_error' in locals() else None
        },
        "lru_memoization": {
            "fibonacci_results": fib_results,
            "cache_info": cache_info
        }
    }

# Decorator chaining and composition
def log_calls(prefix="CALL"):
    """Decorator that logs function calls"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"{prefix}: {func.__name__}({args}, {kwargs})")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def handle_exceptions(exception_type=Exception, default_return=None):
    """Decorator that handles exceptions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                print(f"Exception in {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator

def deprecation_warning(message="This function is deprecated"):
    """Decorator that issues deprecation warnings"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@log_calls("API")
@handle_exceptions(ValueError, default_return="ERROR")
@timing_decorator
def complex_api_function(x, y):
    if x < 0:
        raise ValueError("x cannot be negative")
    return x ** y

@deprecation_warning("Use new_function() instead")
def old_function():
    return "Old implementation"

def decorator_composition_demo():
    # Test chained decorators
    result1 = complex_api_function(2, 3)
    result2 = complex_api_function(-1, 2)  # Should handle exception
    
    # Test deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deprecated_result = old_function()
        deprecation_warned = len(w) > 0 and issubclass(w[-1].category, DeprecationWarning)
    
    return {
        "chained_decorators": {
            "valid_result": result1,
            "error_handled": result2
        },
        "deprecation": {
            "result": deprecated_result,
            "warning_issued": deprecation_warned
        }
    }

# Property decorators revisited
class AdvancedProperty:
    """Advanced property implementation with validation and caching"""
    
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc
        self._name = None
        self._cache = weakref.WeakKeyDictionary()
    
    def __set_name__(self, owner, name):
        self._name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        
        if self.fget is None:
            raise AttributeError(f"unreadable attribute {self._name}")
        
        # Check cache first
        if obj in self._cache:
            return self._cache[obj]
        
        value = self.fget(obj)
        self._cache[obj] = value
        return value
    
    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError(f"can't set attribute {self._name}")
        
        # Clear cache when setting
        if obj in self._cache:
            del self._cache[obj]
        
        self.fset(obj, value)
    
    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError(f"can't delete attribute {self._name}")
        
        # Clear cache when deleting
        if obj in self._cache:
            del self._cache[obj]
        
        self.fdel(obj)
    
    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)
    
    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)
    
    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)

class ValidationProperty:
    """Property with automatic validation"""
    
    def __init__(self, validator=None, transformer=None):
        self.validator = validator
        self.transformer = transformer
        self._name = None
    
    def __set_name__(self, owner, name):
        self._name = f"_{name}"
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self._name, None)
    
    def __set__(self, obj, value):
        if self.transformer:
            value = self.transformer(value)
        
        if self.validator and not self.validator(value):
            raise ValueError(f"Invalid value for {self._name}: {value}")
        
        setattr(obj, self._name, value)

class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    @AdvancedProperty
    def area(self):
        print("Computing area...")  # Shows caching in action
        return 3.14159 * self.radius ** 2
    
    @AdvancedProperty
    def circumference(self):
        print("Computing circumference...")
        return 2 * 3.14159 * self.radius

class ValidatedUser:
    name = ValidationProperty(
        validator=lambda x: isinstance(x, str) and len(x) > 0,
        transformer=str.strip
    )
    age = ValidationProperty(
        validator=lambda x: isinstance(x, int) and 0 <= x <= 150
    )
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

def advanced_property_demo():
    # Test cached properties
    circle = Circle(5)
    area1 = circle.area  # Should compute
    area2 = circle.area  # Should use cache
    
    circumference1 = circle.circumference  # Should compute
    circumference2 = circle.circumference  # Should use cache
    
    # Test validated properties
    user = ValidatedUser("  Alice  ", 30)  # Name should be stripped
    
    try:
        invalid_user = ValidatedUser("", 25)  # Should fail validation
        validation_error = False
    except ValueError as e:
        validation_error = str(e)
    
    return {
        "cached_properties": {
            "area_values": [area1, area2],
            "circumference_values": [circumference1, circumference2],
            "values_same": area1 == area2 and circumference1 == circumference2
        },
        "validated_properties": {
            "user_name": user.name,  # Should be "Alice" (trimmed)
            "user_age": user.age,
            "validation_error": validation_error
        }
    }

# Real-world decorator examples
def api_endpoint(method="GET", path="/"):
    """Decorator for API endpoint registration"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Simulate request processing
            return {
                "method": method,
                "path": path,
                "function": func.__name__,
                "result": func(*args, **kwargs)
            }
        wrapper._api_method = method
        wrapper._api_path = path
        return wrapper
    return decorator

def transaction(isolation_level="READ_COMMITTED"):
    """Decorator for database transactions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Simulate transaction handling
            try:
                print(f"BEGIN TRANSACTION ({isolation_level})")
                result = func(*args, **kwargs)
                print("COMMIT")
                return {"success": True, "result": result}
            except Exception as e:
                print("ROLLBACK")
                return {"success": False, "error": str(e)}
        return wrapper
    return decorator

def circuit_breaker(failure_threshold=5, timeout=60):
    """Circuit breaker pattern decorator"""
    def decorator(func):
        failure_count = 0
        last_failure_time = 0
        state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal failure_count, last_failure_time, state
            current_time = time.time()
            
            if state == "OPEN":
                if current_time - last_failure_time > timeout:
                    state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                if state == "HALF_OPEN":
                    state = "CLOSED"
                    failure_count = 0
                return result
            except Exception as e:
                failure_count += 1
                last_failure_time = current_time
                
                if failure_count >= failure_threshold:
                    state = "OPEN"
                
                raise e
        
        wrapper.get_state = lambda: {
            "state": state,
            "failure_count": failure_count,
            "last_failure": last_failure_time
        }
        wrapper.reset = lambda: setattr(wrapper, 'failure_count', 0) or setattr(wrapper, 'state', "CLOSED")
        return wrapper
    return decorator

@api_endpoint("POST", "/users")
def create_user(username, email):
    return {"id": 123, "username": username, "email": email}

@transaction("SERIALIZABLE")
def transfer_money(from_account, to_account, amount):
    if amount <= 0:
        raise ValueError("Amount must be positive")
    return {"from": from_account, "to": to_account, "amount": amount}

@circuit_breaker(failure_threshold=3, timeout=5)
def unreliable_service():
    import random
    if random.random() < 0.7:  # 70% failure rate
        raise Exception("Service unavailable")
    return "Service response"

def real_world_decorator_demo():
    # Test API endpoint decorator
    api_result = create_user("alice", "alice@example.com")
    
    # Test transaction decorator
    transaction_success = transfer_money("ACC001", "ACC002", 100)
    transaction_failure = transfer_money("ACC001", "ACC002", -50)
    
    # Test circuit breaker
    circuit_results = []
    for i in range(10):
        try:
            result = unreliable_service()
            circuit_results.append({"attempt": i+1, "result": result})
        except Exception as e:
            circuit_results.append({"attempt": i+1, "error": str(e)})
    
    circuit_state = unreliable_service.get_state()
    
    return {
        "api_endpoint": api_result,
        "transaction_patterns": {
            "success": transaction_success,
            "failure": transaction_failure
        },
        "circuit_breaker": {
            "attempts": circuit_results,
            "final_state": circuit_state
        }
    }

# Comprehensive testing
def run_all_decorator_demos():
    """Execute all decorator demonstrations"""
    demo_functions = [
        ('basic_decorators', basic_decorator_demo),
        ('parametrized_decorators', parametrized_decorator_demo),
        ('class_decorators', class_decorator_demo),
        ('advanced_patterns', advanced_decorator_demo),
        ('decorator_composition', decorator_composition_demo),
        ('advanced_properties', advanced_property_demo),
        ('real_world_examples', real_world_decorator_demo)
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
    print("=== Python Advanced Decorators Demo ===")
    
    # Run all demonstrations
    all_results = run_all_decorator_demos()
    
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
    
    print("\n=== DECORATOR PATTERNS SUMMARY ===")
    
    patterns = {
        "Function Decorators": "Modify function behavior, preserve metadata with @functools.wraps",
        "Parametrized Decorators": "Decorator factories that return decorators based on arguments",
        "Class Decorators": "Modify classes, add methods, implement patterns like singleton",
        "Decorator Classes": "Use __call__ to make classes behave like decorators",
        "Property Decorators": "Advanced property patterns with caching and validation",
        "Decorator Chaining": "Multiple decorators applied in sequence, order matters",
        "Real-world Patterns": "API endpoints, transactions, circuit breakers, rate limiting"
    }
    
    for pattern, description in patterns.items():
        print(f"  {pattern}: {description}")
    
    print("\n=== BEST PRACTICES ===")
    
    best_practices = [
        "Always use @functools.wraps to preserve function metadata",
        "Handle edge cases like None arguments and keyword arguments",
        "Use weakref for caching to avoid memory leaks",
        "Consider thread safety for stateful decorators",
        "Document decorator behavior and side effects",
        "Use decorator factories for configurable decorators",
        "Test decorated functions thoroughly",
        "Be mindful of decorator order when chaining",
        "Use class decorators for complex modifications",
        "Consider performance implications of decorator overhead"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== Advanced Decorators Complete! ===")
    print("  Complex decorator patterns and real-world applications mastered")
