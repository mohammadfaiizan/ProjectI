"""
Python Functions and Scope: Definitions, Parameters, Closures, and Scope Management
Implementation-focused with minimal comments, maximum functionality coverage
"""

from typing import Callable, Any, List, Dict, Optional, Union
from functools import wraps, partial, reduce
import time
import math

# Basic function definitions
def basic_function_patterns():
    # Simple function
    def greet(name):
        return f"Hello, {name}!"
    
    # Function with default parameters
    def create_profile(name, age=25, city="Unknown"):
        return {"name": name, "age": age, "city": city}
    
    # Function with type hints
    def calculate_area(length: float, width: float) -> float:
        return length * width
    
    # Function with docstring
    def fibonacci(n: int) -> int:
        """Calculate nth Fibonacci number"""
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    return {
        'greet': greet("Alice"),
        'profile_default': create_profile("Bob"),
        'profile_full': create_profile("Charlie", 30, "NYC"),
        'area': calculate_area(5.0, 3.0),
        'fibonacci': fibonacci(7)
    }

# Parameter and argument patterns
def parameter_argument_patterns():
    # Positional arguments
    def add_numbers(a, b, c):
        return a + b + c
    
    # Keyword arguments
    def format_name(first, last, middle="", title=""):
        parts = [title, first, middle, last]
        return " ".join(part for part in parts if part)
    
    # Mixed arguments
    def calculate_price(base_price, tax_rate=0.08, discount=0.0):
        price_after_discount = base_price * (1 - discount)
        return price_after_discount * (1 + tax_rate)
    
    # Function calls with different argument styles
    results = {
        'positional': add_numbers(1, 2, 3),
        'keyword': format_name(first="John", last="Doe"),
        'mixed_1': format_name("Jane", "Smith", middle="Marie"),
        'mixed_2': format_name(last="Brown", first="Bob", title="Dr."),
        'price_basic': calculate_price(100),
        'price_discount': calculate_price(100, discount=0.1),
        'price_full': calculate_price(100, tax_rate=0.1, discount=0.15)
    }
    
    return results

def variable_length_arguments():
    # *args - variable positional arguments
    def sum_all(*args):
        return sum(args)
    
    def find_max(*numbers):
        return max(numbers) if numbers else None
    
    # **kwargs - variable keyword arguments
    def create_user(**kwargs):
        defaults = {"active": True, "role": "user"}
        defaults.update(kwargs)
        return defaults
    
    def process_data(required_param, *args, **kwargs):
        return {
            "required": required_param,
            "args": args,
            "kwargs": kwargs,
            "total_params": 1 + len(args) + len(kwargs)
        }
    
    # Unpacking arguments
    numbers = [1, 2, 3, 4, 5]
    user_data = {"name": "Alice", "email": "alice@example.com", "age": 30}
    
    return {
        'sum_args': sum_all(1, 2, 3, 4, 5),
        'max_args': find_max(10, 5, 8, 3, 12),
        'user_basic': create_user(),
        'user_custom': create_user(name="Bob", admin=True),
        'process_mixed': process_data("important", 1, 2, 3, flag=True, count=10),
        'unpacked_sum': sum_all(*numbers),
        'unpacked_user': create_user(**user_data)
    }

# Return statement patterns
def return_patterns():
    # Multiple return statements
    def classify_number(n):
        if n > 0:
            return "positive"
        elif n < 0:
            return "negative"
        else:
            return "zero"
    
    # Multiple return values
    def divide_with_remainder(dividend, divisor):
        if divisor == 0:
            return None, None
        quotient = dividend // divisor
        remainder = dividend % divisor
        return quotient, remainder
    
    # Early returns for validation
    def validate_email(email):
        if not email:
            return False, "Email is required"
        if "@" not in email:
            return False, "Invalid email format"
        if len(email) < 5:
            return False, "Email too short"
        return True, "Valid email"
    
    # Return complex data structures
    def analyze_text(text):
        words = text.split()
        return {
            "word_count": len(words),
            "char_count": len(text),
            "words": words,
            "longest_word": max(words, key=len) if words else "",
            "average_word_length": sum(len(w) for w in words) / len(words) if words else 0
        }
    
    # Tuple unpacking from returns
    q, r = divide_with_remainder(17, 5)
    valid, message = validate_email("test@example.com")
    
    return {
        'classifications': [classify_number(x) for x in [-5, 0, 3]],
        'division': (q, r),
        'email_validation': (valid, message),
        'text_analysis': analyze_text("Python is an amazing programming language")
    }

# Scope and variable resolution
def scope_demonstrations():
    global_var = "I'm global"
    
    def outer_function():
        outer_var = "I'm in outer"
        
        def inner_function():
            inner_var = "I'm in inner"
            return {
                "inner": inner_var,
                "outer": outer_var,
                "global": global_var
            }
        
        def modify_outer():
            nonlocal outer_var
            original = outer_var
            outer_var = "Modified outer"
            return original, outer_var
        
        inner_result = inner_function()
        before, after = modify_outer()
        
        return {
            "inner_scope": inner_result,
            "outer_modification": {"before": before, "after": after},
            "final_outer": outer_var
        }
    
    # Global variable modification
    global_counter = 0
    
    def increment_global():
        global global_counter
        global_counter += 1
        return global_counter
    
    # Local vs global name collision
    test_var = "global value"
    
    def name_collision_demo():
        test_var = "local value"
        return {"local": test_var, "global_access": globals().get('test_var')}
    
    return {
        "scope_chain": outer_function(),
        "global_counter": [increment_global() for _ in range(3)],
        "name_collision": name_collision_demo(),
        "builtin_access": len([1, 2, 3])  # Built-in function access
    }

# Closures and nested functions
def closure_patterns():
    # Basic closure
    def create_multiplier(factor):
        def multiply(number):
            return number * factor
        return multiply
    
    # Closure with state
    def create_counter(start=0):
        count = start
        
        def increment():
            nonlocal count
            count += 1
            return count
        
        def decrement():
            nonlocal count
            count -= 1
            return count
        
        def get_count():
            return count
        
        return increment, decrement, get_count
    
    # Factory pattern with closures
    def create_validator(min_length=0, max_length=100):
        def validate(text):
            if len(text) < min_length:
                return False, f"Too short (min: {min_length})"
            if len(text) > max_length:
                return False, f"Too long (max: {max_length})"
            return True, "Valid"
        return validate
    
    # Closures preserving state
    def create_accumulator():
        values = []
        
        def add(value):
            values.append(value)
            return sum(values)
        
        def get_values():
            return values.copy()
        
        def reset():
            values.clear()
            return "Reset"
        
        return add, get_values, reset
    
    # Test closures
    double = create_multiplier(2)
    triple = create_multiplier(3)
    
    inc, dec, get = create_counter(10)
    
    username_validator = create_validator(3, 20)
    password_validator = create_validator(8, 50)
    
    acc_add, acc_get, acc_reset = create_accumulator()
    
    return {
        'multipliers': [double(5), triple(5)],
        'counter_ops': [inc(), inc(), dec(), get()],
        'validators': [
            username_validator("bob"),
            username_validator("alice_smith"),
            password_validator("secret"),
            password_validator("very_secure_password_123")
        ],
        'accumulator': [acc_add(10), acc_add(20), acc_get(), acc_reset()]
    }

# Lambda functions and functional programming
def lambda_patterns():
    # Basic lambda functions
    square = lambda x: x ** 2
    add = lambda x, y: x + y
    is_even = lambda n: n % 2 == 0
    
    # Lambda with higher-order functions
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    squared = list(map(lambda x: x**2, numbers))
    evens = list(filter(lambda x: x % 2 == 0, numbers))
    product = reduce(lambda x, y: x * y, numbers[:4])
    
    # Lambda for sorting
    students = [
        {"name": "Alice", "grade": 85},
        {"name": "Bob", "grade": 92},
        {"name": "Charlie", "grade": 78}
    ]
    
    by_grade = sorted(students, key=lambda s: s["grade"])
    by_name = sorted(students, key=lambda s: s["name"])
    
    # Lambda in conditional expressions
    max_func = lambda a, b: a if a > b else b
    min_func = lambda a, b: a if a < b else b
    
    # Complex lambda expressions
    grade_letter = lambda score: (
        "A" if score >= 90 else
        "B" if score >= 80 else
        "C" if score >= 70 else
        "D" if score >= 60 else "F"
    )
    
    return {
        'basic_lambdas': [square(4), add(3, 7), is_even(6)],
        'higher_order': {
            'squared': squared[:5],
            'evens': evens,
            'product': product
        },
        'sorting': {
            'by_grade': by_grade,
            'by_name': by_name
        },
        'conditionals': [max_func(10, 5), min_func(10, 5)],
        'grade_letters': [grade_letter(score) for score in [95, 85, 75, 65, 55]]
    }

# Decorators and function modification
def decorator_patterns():
    # Basic decorator
    def timer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            return result, f"{(end-start)*1000:.2f}ms"
        return wrapper
    
    # Decorator with parameters
    def repeat(times):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                results = []
                for _ in range(times):
                    results.append(func(*args, **kwargs))
                return results
            return wrapper
        return decorator
    
    # Multiple decorators
    def validate_positive(func):
        @wraps(func)
        def wrapper(n):
            if n < 0:
                raise ValueError("Number must be positive")
            return func(n)
        return wrapper
    
    @timer
    def slow_calculation(n):
        total = 0
        for i in range(n):
            total += i ** 2
        return total
    
    @repeat(3)
    def random_number():
        import random
        return random.randint(1, 10)
    
    @validate_positive
    @timer
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n-1)[0]  # Extract result from timer
    
    # Property decorator simulation
    class Circle:
        def __init__(self, radius):
            self._radius = radius
        
        @property
        def radius(self):
            return self._radius
        
        @radius.setter
        def radius(self, value):
            if value < 0:
                raise ValueError("Radius must be positive")
            self._radius = value
        
        @property
        def area(self):
            return math.pi * self._radius ** 2
    
    # Test decorated functions
    slow_result, slow_time = slow_calculation(1000)
    random_results = random_number()
    
    try:
        fact_result, fact_time = factorial(5)
        factorial_error = None
    except ValueError as e:
        fact_result, fact_time = None, None
        factorial_error = str(e)
    
    circle = Circle(5)
    circle_area = circle.area
    
    return {
        'timer_result': (slow_result, slow_time),
        'repeat_result': random_results,
        'factorial': {
            'result': fact_result,
            'time': fact_time,
            'error': factorial_error
        },
        'property_demo': {
            'radius': circle.radius,
            'area': circle_area
        }
    }

# Recursive functions
def recursive_patterns():
    # Classic recursion examples
    def factorial_recursive(n):
        if n <= 1:
            return 1
        return n * factorial_recursive(n - 1)
    
    def fibonacci_recursive(n):
        if n <= 1:
            return n
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)
    
    # Tail recursion (Python doesn't optimize this)
    def factorial_tail_recursive(n, accumulator=1):
        if n <= 1:
            return accumulator
        return factorial_tail_recursive(n - 1, n * accumulator)
    
    # Tree traversal recursion
    def tree_sum(node):
        if isinstance(node, (int, float)):
            return node
        if isinstance(node, list):
            return sum(tree_sum(child) for child in node)
        return 0
    
    # String processing recursion
    def reverse_string(s):
        if len(s) <= 1:
            return s
        return reverse_string(s[1:]) + s[0]
    
    def is_palindrome(s):
        s = s.lower().replace(" ", "")
        if len(s) <= 1:
            return True
        if s[0] != s[-1]:
            return False
        return is_palindrome(s[1:-1])
    
    # Mathematical recursion
    def gcd(a, b):
        if b == 0:
            return a
        return gcd(b, a % b)
    
    def power(base, exp):
        if exp == 0:
            return 1
        if exp < 0:
            return 1 / power(base, -exp)
        if exp % 2 == 0:
            half_power = power(base, exp // 2)
            return half_power * half_power
        return base * power(base, exp - 1)
    
    # Test recursive functions
    tree_data = [1, [2, 3], [4, [5, 6]]]
    
    return {
        'factorial': factorial_recursive(6),
        'fibonacci': [fibonacci_recursive(i) for i in range(8)],
        'tail_factorial': factorial_tail_recursive(6),
        'tree_sum': tree_sum(tree_data),
        'string_ops': {
            'reverse': reverse_string("hello"),
            'palindrome_true': is_palindrome("A man a plan a canal Panama"),
            'palindrome_false': is_palindrome("hello world")
        },
        'math_ops': {
            'gcd': gcd(48, 18),
            'power': power(2, 10)
        }
    }

# Higher-order functions
def higher_order_patterns():
    # Functions that take functions as arguments
    def apply_operation(numbers, operation):
        return [operation(n) for n in numbers]
    
    def compose_functions(f, g):
        return lambda x: f(g(x))
    
    def conditional_apply(predicate, transform, default_transform):
        def apply_to_list(items):
            return [transform(item) if predicate(item) else default_transform(item) 
                   for item in items]
        return apply_to_list
    
    # Functions that return functions
    def create_validator_function(validator_type):
        validators = {
            'email': lambda s: '@' in s and '.' in s,
            'phone': lambda s: s.replace('-', '').replace(' ', '').isdigit(),
            'positive': lambda n: isinstance(n, (int, float)) and n > 0
        }
        return validators.get(validator_type, lambda x: True)
    
    def create_transformer(operation):
        operations = {
            'upper': str.upper,
            'lower': str.lower,
            'reverse': lambda s: s[::-1],
            'double': lambda n: n * 2,
            'square': lambda n: n ** 2
        }
        return operations.get(operation, lambda x: x)
    
    # Partial function application
    multiply = lambda x, y: x * y
    double = partial(multiply, 2)
    triple = partial(multiply, 3)
    
    # Function composition examples
    add_one = lambda x: x + 1
    square = lambda x: x ** 2
    add_one_then_square = compose_functions(square, add_one)
    square_then_add_one = compose_functions(add_one, square)
    
    # Conditional application
    process_positives = conditional_apply(
        lambda x: x > 0,
        lambda x: x * 2,
        lambda x: 0
    )
    
    # Test higher-order functions
    numbers = [1, 2, 3, 4, 5]
    mixed_numbers = [-2, -1, 0, 1, 2]
    
    email_validator = create_validator_function('email')
    upper_transformer = create_transformer('upper')
    
    return {
        'apply_operations': {
            'squared': apply_operation(numbers, lambda x: x**2),
            'doubled': apply_operation(numbers, lambda x: x*2)
        },
        'composition': {
            'add_then_square': add_one_then_square(5),
            'square_then_add': square_then_add_one(5)
        },
        'partial_application': {
            'double': double(7),
            'triple': triple(7)
        },
        'conditional': process_positives(mixed_numbers),
        'validators': {
            'valid_email': email_validator('test@example.com'),
            'invalid_email': email_validator('invalid')
        },
        'transformers': upper_transformer('hello world')
    }

# Interview problems using functions
def function_interview_problems():
    def memoize(func):
        """Memoization decorator for optimization"""
        cache = {}
        @wraps(func)
        def wrapper(*args):
            if args in cache:
                return cache[args]
            result = func(*args)
            cache[args] = result
            return result
        return wrapper
    
    @memoize
    def fibonacci_optimized(n):
        if n <= 1:
            return n
        return fibonacci_optimized(n-1) + fibonacci_optimized(n-2)
    
    def curry_function(func):
        """Convert function to curried version"""
        @wraps(func)
        def curried(*args, **kwargs):
            if len(args) + len(kwargs) >= func.__code__.co_argcount:
                return func(*args, **kwargs)
            return lambda *more_args, **more_kwargs: curried(*(args + more_args), **kwargs, **more_kwargs)
        return curried
    
    @curry_function
    def add_three_numbers(a, b, c):
        return a + b + c
    
    def function_pipeline(*functions):
        """Create pipeline of functions"""
        return lambda x: reduce(lambda acc, func: func(acc), functions, x)
    
    def retry_function(max_attempts=3, delay=0.1):
        """Retry decorator for unreliable functions"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            raise e
                        time.sleep(delay)
                return None
            return wrapper
        return decorator
    
    @retry_function(max_attempts=2)
    def unreliable_function(success_rate=0.7):
        import random
        if random.random() < success_rate:
            return "Success!"
        raise Exception("Random failure")
    
    # Test advanced function patterns
    pipeline = function_pipeline(
        lambda x: x * 2,
        lambda x: x + 10,
        lambda x: x ** 2
    )
    
    curried_add = add_three_numbers(1)
    partially_curried = curried_add(2)
    
    return {
        'memoized_fibonacci': [fibonacci_optimized(i) for i in range(10)],
        'currying': {
            'step1': str(curried_add),
            'step2': str(partially_curried),
            'final': partially_curried(3)
        },
        'pipeline_result': pipeline(5),  # ((5*2)+10)^2 = 400
        'retry_attempts': "Function with retry mechanism"
    }

# Performance and testing
def run_all_function_demos():
    """Execute all function demonstrations"""
    demo_functions = [
        ('basic_patterns', basic_function_patterns),
        ('parameters', parameter_argument_patterns),
        ('variable_args', variable_length_arguments),
        ('returns', return_patterns),
        ('scope', scope_demonstrations),
        ('closures', closure_patterns),
        ('lambdas', lambda_patterns),
        ('decorators', decorator_patterns),
        ('recursion', recursive_patterns),
        ('higher_order', higher_order_patterns),
        ('interview_problems', function_interview_problems)
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
    print("=== Python Functions and Scope Demo ===")
    
    # Run all demonstrations
    all_results = run_all_function_demos()
    
    for category, data in all_results.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        
        if 'error' in data:
            print(f"  Error: {data['error']}")
            continue
            
        result = data['result']
        print(f"  Execution time: {data['execution_time']}")
        
        # Display results based on type
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, (list, dict)) and len(str(value)) > 100:
                    print(f"  {key}: {str(value)[:100]}... (truncated)")
                else:
                    print(f"  {key}: {value}")
        elif isinstance(result, list):
            if len(result) > 5:
                print(f"  Results: {result[:5]}... (truncated)")
            else:
                print(f"  Results: {result}")
        else:
            print(f"  Result: {result}")
    
    print("\n=== PERFORMANCE SUMMARY ===")
    total_time = sum(float(data.get('execution_time', '0ms')[:-2]) 
                    for data in all_results.values() 
                    if 'execution_time' in data)
    print(f"  Total execution time: {total_time:.2f}ms")
    print(f"  Functions executed: {len(all_results)}")
    print(f"  Average per function: {total_time/len(all_results):.2f}ms")
    
    # Memory usage estimation
    import sys
    print(f"  Estimated memory usage: {sys.getsizeof(all_results)} bytes")
