"""
Python Fundamentals: Syntax and Basic Types
Implementation-focused with minimal comments, maximum functionality coverage
"""

from typing import Union, Any
import sys

# Variables and data types implementation
def demonstrate_variables():
    # Basic variable assignment
    name = "Python"
    age = 30
    height = 5.9
    is_programmer = True
    
    # Multiple assignment
    x, y, z = 1, 2, 3
    a = b = c = 0
    
    # Variable naming patterns
    user_name = "john_doe"
    userName = "camelCase"
    USER_NAME = "CONSTANT"
    _private = "underscore prefix"
    
    return locals()

def data_type_operations():
    # Integer operations
    num1 = 42
    num2 = 0x2A  # Hexadecimal
    num3 = 0o52  # Octal
    num4 = 0b101010  # Binary
    
    # Float operations
    pi = 3.14159
    scientific = 1.5e-4
    infinity = float('inf')
    not_a_number = float('nan')
    
    # String variations
    single_quote = 'Hello'
    double_quote = "World"
    triple_quote = """Multi
    line
    string"""
    raw_string = r"No escape \n sequences"
    
    # Boolean logic
    true_val = True
    false_val = False
    bool_from_int = bool(1)
    bool_from_str = bool("text")
    
    return {
        'integers': [num1, num2, num3, num4],
        'floats': [pi, scientific, infinity, not_a_number],
        'strings': [single_quote, double_quote, triple_quote, raw_string],
        'booleans': [true_val, false_val, bool_from_int, bool_from_str]
    }

# Type conversion implementations
def type_conversions():
    # Numeric conversions
    int_to_float = float(42)
    float_to_int = int(3.14)
    str_to_int = int("123")
    str_to_float = float("3.14")
    
    # String conversions
    int_to_str = str(42)
    float_to_str = str(3.14)
    bool_to_str = str(True)
    
    # Boolean conversions
    int_to_bool = bool(0)  # False
    str_to_bool = bool("")  # False
    list_to_bool = bool([])  # False
    
    # Complex conversions
    try:
        invalid = int("abc")
    except ValueError as e:
        invalid = f"Error: {e}"
    
    return {
        'to_numbers': [int_to_float, float_to_int, str_to_int, str_to_float],
        'to_strings': [int_to_str, float_to_str, bool_to_str],
        'to_booleans': [int_to_bool, str_to_bool, list_to_bool],
        'error_handling': invalid
    }

# Operator demonstrations
def arithmetic_operators():
    a, b = 10, 3
    
    return {
        'addition': a + b,
        'subtraction': a - b,
        'multiplication': a * b,
        'division': a / b,
        'floor_division': a // b,
        'modulus': a % b,
        'exponentiation': a ** b,
        'unary_minus': -a,
        'unary_plus': +a
    }

def comparison_operators():
    x, y = 5, 10
    
    return {
        'equal': x == y,
        'not_equal': x != y,
        'less_than': x < y,
        'less_equal': x <= y,
        'greater_than': x > y,
        'greater_equal': x >= y,
        'identity_is': x is y,
        'identity_is_not': x is not y,
        'membership_in': 5 in [1, 5, 10],
        'membership_not_in': 7 not in [1, 5, 10]
    }

def logical_operators():
    p, q = True, False
    
    return {
        'and_operation': p and q,
        'or_operation': p or q,
        'not_operation': not p,
        'short_circuit_and': False and (1/0),  # Won't raise error
        'short_circuit_or': True or (1/0),    # Won't raise error
        'chained_comparison': 1 < 5 < 10,
        'complex_logic': (True and False) or (True and True)
    }

def bitwise_operators():
    a, b = 12, 5  # 1100, 0101 in binary
    
    return {
        'and': a & b,      # 0100 = 4
        'or': a | b,       # 1101 = 13
        'xor': a ^ b,      # 1001 = 9
        'not': ~a,         # Two's complement
        'left_shift': a << 2,   # 110000 = 48
        'right_shift': a >> 2   # 11 = 3
    }

def assignment_operators():
    x = 10
    
    # Compound assignment
    x += 5   # x = x + 5
    y = x
    y -= 3   # y = y - 3
    z = y
    z *= 2   # z = z * 2
    w = z
    w //= 4  # w = w // 4
    
    return {'final_values': [x, y, z, w]}

# I/O operations and formatting
def input_output_operations():
    # String formatting methods
    name = "Alice"
    age = 25
    
    # Old style formatting
    old_format = "Name: %s, Age: %d" % (name, age)
    
    # .format() method
    new_format = "Name: {}, Age: {}".format(name, age)
    named_format = "Name: {name}, Age: {age}".format(name=name, age=age)
    
    # f-strings (Python 3.6+)
    f_string = f"Name: {name}, Age: {age}"
    f_expression = f"Next year: {age + 1}"
    f_formatting = f"Pi: {3.14159:.2f}"
    
    # Template strings
    from string import Template
    template = Template("Name: $name, Age: $age")
    template_result = template.substitute(name=name, age=age)
    
    return {
        'old_style': old_format,
        'format_method': [new_format, named_format],
        'f_strings': [f_string, f_expression, f_formatting],
        'template': template_result
    }

def advanced_formatting():
    value = 42.123456789
    
    return {
        'decimal_places': f"{value:.2f}",
        'padding': f"{value:10.2f}",
        'zero_padding': f"{value:010.2f}",
        'left_align': f"{value:<10.2f}",
        'right_align': f"{value:>10.2f}",
        'center_align': f"{value:^10.2f}",
        'percentage': f"{0.75:.1%}",
        'scientific': f"{value:.2e}",
        'binary': f"{42:b}",
        'hexadecimal': f"{42:x}",
        'octal': f"{42:o}"
    }

# Console I/O patterns
def console_operations():
    # Simulated input/output for demonstration
    inputs = ["Alice", "25", "yes"]
    outputs = []
    
    # Basic output
    outputs.append("Hello, World!")
    
    # Print with separators and endings
    result1 = f"Values: {1} {2} {3}"  # print(1, 2, 3, sep=' ')
    result2 = f"Line without newline"  # print("text", end='')
    
    # Formatted output
    data = {"name": "Bob", "score": 95.5}
    formatted = f"Student: {data['name']}, Score: {data['score']:.1f}"
    
    return {
        'basic_output': outputs[0],
        'formatted_values': result1,
        'no_newline': result2,
        'complex_format': formatted
    }

# Type checking and introspection
def type_operations():
    values = [42, 3.14, "hello", True, [1, 2, 3], {"key": "value"}]
    
    type_info = []
    for val in values:
        type_info.append({
            'value': val,
            'type': type(val).__name__,
            'isinstance_int': isinstance(val, int),
            'isinstance_str': isinstance(val, str),
            'isinstance_list': isinstance(val, list)
        })
    
    return type_info

def variable_introspection():
    x = 42
    y = "hello"
    z = [1, 2, 3]
    
    return {
        'id_x': id(x),
        'id_y': id(y),
        'size_of_x': sys.getsizeof(x),
        'size_of_y': sys.getsizeof(y),
        'size_of_z': sys.getsizeof(z),
        'dir_int': len(dir(int)),
        'dir_str': len(dir(str)),
        'hasattr_append': hasattr(z, 'append'),
        'getattr_default': getattr(x, 'nonexistent', 'default')
    }

# Memory and performance considerations
def memory_efficiency():
    # String interning
    a = "hello"
    b = "hello"
    interned = a is b
    
    # Small integer caching
    x = 100
    y = 100
    cached = x is y
    
    # Large integer comparison
    big_x = 1000
    big_y = 1000
    not_cached = big_x is big_y
    
    return {
        'string_interning': interned,
        'int_caching': cached,
        'large_int_separate': not_cached
    }

# Practical examples and patterns
def practical_patterns():
    # Swapping variables
    a, b = 10, 20
    a, b = b, a
    
    # Chained comparisons
    age = 25
    valid_age = 18 <= age <= 65
    
    # Default values with or operator
    name = None
    display_name = name or "Anonymous"
    
    # Ternary operator
    status = "adult" if age >= 18 else "minor"
    
    # Multiple conditions
    grade = 85
    letter = ("A" if grade >= 90 else
              "B" if grade >= 80 else
              "C" if grade >= 70 else
              "D" if grade >= 60 else "F")
    
    return {
        'swapped': [a, b],
        'age_validation': valid_age,
        'default_name': display_name,
        'ternary_result': status,
        'grade_letter': letter
    }

def number_base_conversions():
    num = 42
    
    return {
        'decimal': num,
        'binary': bin(num),
        'octal': oct(num),
        'hexadecimal': hex(num),
        'from_binary': int('101010', 2),
        'from_octal': int('52', 8),
        'from_hex': int('2A', 16),
        'custom_base': int('102', 3)  # Base 3
    }

# Interview-style problems
def solve_digit_sum(n: int) -> int:
    """Calculate sum of digits in a number"""
    return sum(int(digit) for digit in str(abs(n)))

def is_palindrome_number(n: int) -> bool:
    """Check if number is palindrome"""
    s = str(n)
    return s == s[::-1]

def reverse_integer(n: int) -> int:
    """Reverse digits of integer"""
    sign = -1 if n < 0 else 1
    reversed_num = int(str(abs(n))[::-1])
    return sign * reversed_num if -2**31 <= sign * reversed_num <= 2**31 - 1 else 0

def count_set_bits(n: int) -> int:
    """Count number of 1s in binary representation"""
    return bin(n).count('1')

def power_of_two(n: int) -> bool:
    """Check if number is power of 2"""
    return n > 0 and (n & (n - 1)) == 0

# Testing and validation
def run_all_demos():
    results = {}
    
    # Execute all demonstration functions
    demo_functions = [
        ('variables', demonstrate_variables),
        ('data_types', data_type_operations),
        ('type_conversions', type_conversions),
        ('arithmetic', arithmetic_operators),
        ('comparison', comparison_operators),
        ('logical', logical_operators),
        ('bitwise', bitwise_operators),
        ('assignment', assignment_operators),
        ('io_operations', input_output_operations),
        ('formatting', advanced_formatting),
        ('console', console_operations),
        ('type_ops', type_operations),
        ('introspection', variable_introspection),
        ('memory', memory_efficiency),
        ('patterns', practical_patterns),
        ('base_conversion', number_base_conversions)
    ]
    
    for name, func in demo_functions:
        try:
            results[name] = func()
        except Exception as e:
            results[name] = f"Error: {e}"
    
    return results

def interview_problems():
    test_cases = [
        (12345, solve_digit_sum),
        (121, is_palindrome_number),
        (-123, reverse_integer),
        (29, count_set_bits),
        (16, power_of_two)
    ]
    
    results = []
    for test_input, func in test_cases:
        try:
            result = func(test_input)
            results.append(f"{func.__name__}({test_input}) = {result}")
        except Exception as e:
            results.append(f"{func.__name__}({test_input}) = Error: {e}")
    
    return results

if __name__ == "__main__":
    # Run comprehensive demonstrations
    print("=== Python Syntax and Basic Types Demo ===")
    
    # Core syntax demonstrations
    demos = run_all_demos()
    for category, result in demos.items():
        print(f"\n{category.upper()}:")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {value}")
        elif isinstance(result, list):
            for i, item in enumerate(result):
                print(f"  [{i}]: {item}")
        else:
            print(f"  {result}")
    
    # Interview problem solutions
    print("\n=== INTERVIEW PROBLEMS ===")
    problems = interview_problems()
    for problem in problems:
        print(f"  {problem}")
    
    print("\n=== PERFORMANCE METRICS ===")
    import time
    start_time = time.time()
    
    # Performance test
    large_number = 123456789
    operations = [
        solve_digit_sum(large_number),
        is_palindrome_number(large_number),
        count_set_bits(large_number),
        power_of_two(1024)
    ]
    
    end_time = time.time()
    print(f"  Execution time: {(end_time - start_time) * 1000:.2f}ms")
    print(f"  Operations completed: {len(operations)}")
