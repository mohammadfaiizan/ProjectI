"""
Python Control Structures: Conditionals, Loops, and Flow Control
Implementation-focused with minimal comments, maximum functionality coverage
"""

from typing import List, Dict, Any, Iterator
import random
import time

# Conditional implementations
def basic_conditionals():
    age = 25
    score = 85
    has_license = True
    
    # Simple if-else
    status = "adult" if age >= 18 else "minor"
    
    # Multi-condition if-elif-else
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"
    
    # Complex conditions
    can_drive = age >= 16 and has_license
    can_vote = age >= 18
    can_drink = age >= 21
    
    return {
        'status': status,
        'grade': grade,
        'permissions': {
            'drive': can_drive,
            'vote': can_vote,
            'drink': can_drink
        }
    }

def boolean_logic_patterns():
    x, y, z = 5, 10, 15
    
    # Short-circuit evaluation
    safe_division = y != 0 and x / y > 2
    safe_access = z and len(str(z)) > 1
    
    # Chained comparisons
    in_range = 1 < x < 20
    valid_triangle = x + y > z and x + z > y and y + z > x
    
    # Boolean operations with different types
    truthy_values = [1, "hello", [1, 2], {"key": "value"}]
    falsy_values = [0, "", [], {}, None, False]
    
    all_truthy = all(truthy_values)
    any_falsy = any(falsy_values)
    
    return {
        'short_circuit': [safe_division, safe_access],
        'chained': [in_range, valid_triangle],
        'boolean_ops': [all_truthy, any_falsy]
    }

# Loop implementations
def for_loop_patterns():
    results = {}
    
    # Basic iteration
    numbers = list(range(1, 6))
    squares = [x**2 for x in numbers]
    
    # Enumerate pattern
    indexed_items = [(i, val) for i, val in enumerate(['a', 'b', 'c'])]
    
    # Zip pattern
    names = ['Alice', 'Bob', 'Charlie']
    ages = [25, 30, 35]
    paired = list(zip(names, ages))
    
    # Range variations
    countdown = list(range(10, 0, -1))
    evens = list(range(0, 21, 2))
    
    # Nested loops
    matrix = [[i*j for j in range(1, 4)] for i in range(1, 4)]
    
    # Loop with else clause
    search_value = 7
    found = False
    for num in range(1, 10):
        if num == search_value:
            found = True
            break
    else:
        # Executes if loop completes without break
        search_result = "not found"
    
    search_result = "found" if found else "not found"
    
    return {
        'squares': squares,
        'indexed': indexed_items,
        'paired': paired,
        'ranges': [countdown[:5], evens[:5]],
        'matrix': matrix,
        'search': search_result
    }

def while_loop_patterns():
    results = {}
    
    # Basic while loop
    count = 0
    powers_of_2 = []
    while count < 5:
        powers_of_2.append(2**count)
        count += 1
    
    # While with condition check
    user_input = "yes"
    attempts = 0
    while user_input.lower() not in ['quit', 'exit'] and attempts < 3:
        attempts += 1
        user_input = "quit" if attempts >= 3 else "continue"
    
    # Infinite loop with break (simulated)
    fibonacci = [0, 1]
    while True:
        next_fib = fibonacci[-1] + fibonacci[-2]
        if next_fib > 100:
            break
        fibonacci.append(next_fib)
    
    # While-else pattern
    target = 42
    guess = 0
    attempts = 0
    while guess != target and attempts < 5:
        guess = random.randint(1, 50)
        attempts += 1
    else:
        if guess == target:
            game_result = f"Found in {attempts} attempts"
        else:
            game_result = "Not found"
    
    return {
        'powers_of_2': powers_of_2,
        'attempts': attempts,
        'fibonacci': fibonacci,
        'game_result': game_result
    }

# Control flow statements
def control_flow_statements():
    results = {}
    
    # Break examples
    break_example = []
    for i in range(20):
        if i > 10:
            break
        break_example.append(i)
    
    # Continue examples
    continue_example = []
    for i in range(10):
        if i % 2 == 0:
            continue
        continue_example.append(i)
    
    # Pass examples (placeholder)
    pass_example = []
    for i in range(5):
        if i < 3:
            pass  # Placeholder - do nothing
        else:
            pass_example.append(i)
    
    # Nested loop control
    nested_break = []
    for i in range(3):
        for j in range(3):
            if i + j > 2:
                break  # Only breaks inner loop
            nested_break.append((i, j))
    
    # Multiple loop control
    found_pair = None
    for i in range(5):
        for j in range(5):
            if i * j == 12:
                found_pair = (i, j)
                break
        if found_pair:
            break
    
    return {
        'break_result': break_example,
        'continue_result': continue_example,
        'pass_result': pass_example,
        'nested_break': nested_break,
        'found_pair': found_pair
    }

# Match-case patterns (Python 3.10+)
def match_case_patterns():
    def process_data(data):
        match data:
            case int() if data > 0:
                return f"Positive integer: {data}"
            case int() if data < 0:
                return f"Negative integer: {data}"
            case 0:
                return "Zero"
            case str() if len(data) > 0:
                return f"Non-empty string: {data}"
            case []:
                return "Empty list"
            case [x] if isinstance(x, int):
                return f"Single integer list: {x}"
            case [x, y]:
                return f"Two-element list: {x}, {y}"
            case {"name": name, "age": age}:
                return f"Person: {name}, {age}"
            case _:
                return f"Unknown type: {type(data)}"
    
    test_cases = [
        42, -5, 0, "hello", "", [], [10], [1, 2], [1, 2, 3],
        {"name": "Alice", "age": 25}, {"key": "value"}
    ]
    
    return [process_data(case) for case in test_cases]

def advanced_match_patterns():
    def analyze_structure(obj):
        match obj:
            # Sequence patterns
            case []:
                return "empty_list"
            case [x]:
                return f"singleton: {x}"
            case [x, y] if x == y:
                return f"equal_pair: {x}"
            case [x, *rest]:
                return f"head_tail: {x}, {len(rest)} more"
            
            # Mapping patterns
            case {"type": "user", "name": str(name)}:
                return f"user: {name}"
            case {"type": "admin", "permissions": list(perms)}:
                return f"admin: {len(perms)} permissions"
            
            # Class patterns (would work with actual classes)
            case str() if obj.isdigit():
                return f"numeric_string: {obj}"
            case str() if obj.isalpha():
                return f"alpha_string: {obj}"
            
            # Guard patterns
            case x if isinstance(x, (int, float)) and x > 100:
                return f"large_number: {x}"
            
            case _:
                return f"unmatched: {type(obj).__name__}"
    
    test_data = [
        [], [1], [2, 2], [1, 2, 3, 4],
        {"type": "user", "name": "Alice"},
        {"type": "admin", "permissions": ["read", "write"]},
        "123", "abc", 150, "mixed123"
    ]
    
    return [analyze_structure(item) for item in test_data]

# Nested and complex control structures
def nested_control_examples():
    # Nested loops with multiple conditions
    prime_numbers = []
    for num in range(2, 50):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            prime_numbers.append(num)
    
    # Complex conditional logic
    def categorize_student(age, grade, attendance):
        if age < 18:
            if grade >= 90:
                if attendance >= 95:
                    return "excellent_minor"
                else:
                    return "good_minor"
            elif grade >= 70:
                return "average_minor"
            else:
                return "struggling_minor"
        else:
            if grade >= 90 and attendance >= 90:
                return "excellent_adult"
            elif grade >= 80 or attendance >= 85:
                return "good_adult"
            else:
                return "needs_improvement"
    
    students = [
        (16, 95, 98), (17, 85, 80), (19, 92, 95),
        (20, 75, 88), (18, 65, 70)
    ]
    
    categorized = [categorize_student(*student) for student in students]
    
    # Game simulation with nested control
    def simulate_game():
        score = 0
        level = 1
        lives = 3
        
        while lives > 0 and level <= 5:
            success = random.choice([True, False, True])  # 2/3 chance
            
            if success:
                score += level * 10
                level += 1
            else:
                lives -= 1
                if lives > 0:
                    continue
                else:
                    break
        
        return {"score": score, "level": level, "lives": lives}
    
    # Run multiple game simulations
    random.seed(42)  # For reproducible results
    game_results = [simulate_game() for _ in range(3)]
    
    return {
        'primes': prime_numbers[:10],
        'student_categories': categorized,
        'game_simulations': game_results
    }

# Loop optimization patterns
def loop_optimization_patterns():
    # List comprehension vs traditional loop
    data = range(1000)
    
    # Traditional loop timing
    start_time = time.time()
    squares_traditional = []
    for x in data:
        if x % 2 == 0:
            squares_traditional.append(x**2)
    traditional_time = time.time() - start_time
    
    # List comprehension timing
    start_time = time.time()
    squares_comprehension = [x**2 for x in data if x % 2 == 0]
    comprehension_time = time.time() - start_time
    
    # Generator expression
    start_time = time.time()
    squares_generator = (x**2 for x in data if x % 2 == 0)
    generator_list = list(squares_generator)
    generator_time = time.time() - start_time
    
    # Early termination optimization
    def find_first_match(data, condition):
        for item in data:
            if condition(item):
                return item
        return None
    
    large_data = range(10000)
    first_large = find_first_match(large_data, lambda x: x > 5000)
    
    return {
        'results_equal': squares_traditional[:5] == squares_comprehension[:5],
        'timing_comparison': {
            'traditional': f"{traditional_time:.6f}s",
            'comprehension': f"{comprehension_time:.6f}s", 
            'generator': f"{generator_time:.6f}s"
        },
        'early_termination': first_large,
        'sample_results': squares_comprehension[:10]
    }

# Advanced control flow patterns
def advanced_control_patterns():
    # State machine simulation
    def traffic_light_cycle():
        states = ['red', 'yellow', 'green']
        current = 0
        cycle = []
        
        for _ in range(10):
            cycle.append(states[current])
            current = (current + 1) % len(states)
        
        return cycle
    
    # Recursive-like iteration
    def iterative_factorial(n):
        result = 1
        while n > 1:
            result *= n
            n -= 1
        return result
    
    # Multiple exit conditions
    def search_2d_array(matrix, target):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == target:
                    return (i, j)
        return None
    
    test_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    # Exception-based control flow
    def safe_operations():
        operations = []
        
        for divisor in [2, 1, 0, 3]:
            try:
                result = 10 / divisor
                operations.append(f"10/{divisor} = {result}")
            except ZeroDivisionError:
                operations.append(f"10/{divisor} = undefined")
            continue
        
        return operations
    
    return {
        'traffic_cycle': traffic_light_cycle(),
        'factorial_5': iterative_factorial(5),
        'search_result': search_2d_array(test_matrix, 5),
        'safe_ops': safe_operations()
    }

# Interview problems using control structures
def control_structure_problems():
    def fizz_buzz(n: int) -> List[str]:
        """Classic FizzBuzz implementation"""
        result = []
        for i in range(1, n + 1):
            if i % 15 == 0:
                result.append("FizzBuzz")
            elif i % 3 == 0:
                result.append("Fizz")
            elif i % 5 == 0:
                result.append("Buzz")
            else:
                result.append(str(i))
        return result
    
    def find_peak_element(nums: List[int]) -> int:
        """Find peak element in array"""
        for i in range(len(nums)):
            left_ok = i == 0 or nums[i] > nums[i-1]
            right_ok = i == len(nums)-1 or nums[i] > nums[i+1]
            if left_ok and right_ok:
                return i
        return -1
    
    def spiral_matrix(n: int) -> List[List[int]]:
        """Generate spiral matrix"""
        matrix = [[0] * n for _ in range(n)]
        top, bottom, left, right = 0, n-1, 0, n-1
        num = 1
        
        while top <= bottom and left <= right:
            # Fill top row
            for j in range(left, right + 1):
                matrix[top][j] = num
                num += 1
            top += 1
            
            # Fill right column
            for i in range(top, bottom + 1):
                matrix[i][right] = num
                num += 1
            right -= 1
            
            # Fill bottom row
            if top <= bottom:
                for j in range(right, left - 1, -1):
                    matrix[bottom][j] = num
                    num += 1
                bottom -= 1
            
            # Fill left column
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    matrix[i][left] = num
                    num += 1
                left += 1
        
        return matrix
    
    def pascal_triangle(n: int) -> List[List[int]]:
        """Generate Pascal's triangle"""
        triangle = []
        for i in range(n):
            row = [1] * (i + 1)
            for j in range(1, i):
                row[j] = triangle[i-1][j-1] + triangle[i-1][j]
            triangle.append(row)
        return triangle
    
    return {
        'fizz_buzz_15': fizz_buzz(15),
        'peak_element': find_peak_element([1, 2, 3, 1]),
        'spiral_3x3': spiral_matrix(3),
        'pascal_5': pascal_triangle(5)
    }

# Performance and testing
def run_all_control_demos():
    """Execute all control structure demonstrations"""
    demo_functions = [
        ('basic_conditionals', basic_conditionals),
        ('boolean_logic', boolean_logic_patterns),
        ('for_loops', for_loop_patterns),
        ('while_loops', while_loop_patterns),
        ('control_flow', control_flow_statements),
        ('match_case', match_case_patterns),
        ('advanced_match', advanced_match_patterns),
        ('nested_control', nested_control_examples),
        ('optimization', loop_optimization_patterns),
        ('advanced_patterns', advanced_control_patterns),
        ('interview_problems', control_structure_problems)
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
    print("=== Python Control Structures Demo ===")
    
    # Run all demonstrations
    all_results = run_all_control_demos()
    
    for category, data in all_results.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        
        if 'error' in data:
            print(f"  Error: {data['error']}")
            continue
            
        result = data['result']
        print(f"  Execution time: {data['execution_time']}")
        
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, list) and len(value) > 5:
                    print(f"  {key}: {value[:5]}... (truncated)")
                else:
                    print(f"  {key}: {value}")
        elif isinstance(result, list):
            if len(result) > 10:
                print(f"  Results: {result[:10]}... (truncated)")
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
