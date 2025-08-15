"""
Python Functional Programming: Pure Functions, Higher-Order Functions, and Functional Patterns
Implementation-focused with minimal comments, maximum functionality coverage
"""

import functools
import itertools
import operator
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union
import math
import random
import collections
from dataclasses import dataclass
import copy

T = TypeVar('T')
U = TypeVar('U')

# Pure functions and side effects
def pure_functions_demo():
    """Demonstrate pure functions vs impure functions"""
    
    # Pure function - no side effects, deterministic
    def pure_add(x: int, y: int) -> int:
        return x + y
    
    def pure_multiply_list(numbers: List[int], factor: int) -> List[int]:
        return [n * factor for n in numbers]
    
    def pure_filter_even(numbers: List[int]) -> List[int]:
        return [n for n in numbers if n % 2 == 0]
    
    # Impure function - has side effects
    counter = 0
    
    def impure_add_with_counter(x: int, y: int) -> int:
        global counter
        counter += 1  # Side effect
        return x + y
    
    log_messages = []
    
    def impure_add_with_logging(x: int, y: int) -> int:
        result = x + y
        log_messages.append(f"Added {x} + {y} = {result}")  # Side effect
        return result
    
    def impure_modify_list(numbers: List[int], factor: int) -> List[int]:
        for i in range(len(numbers)):
            numbers[i] *= factor  # Modifies input
        return numbers
    
    # Test pure functions
    pure_result1 = pure_add(5, 3)
    pure_result2 = pure_add(5, 3)  # Same input, same output
    
    original_list = [1, 2, 3, 4, 5]
    pure_multiplied = pure_multiply_list(original_list, 2)
    pure_filtered = pure_filter_even([1, 2, 3, 4, 5, 6])
    
    # Test impure functions
    impure_result1 = impure_add_with_counter(5, 3)
    impure_result2 = impure_add_with_counter(5, 3)  # Same input, but counter changed
    
    logged_result1 = impure_add_with_logging(2, 3)
    logged_result2 = impure_add_with_logging(4, 6)
    
    test_list = [1, 2, 3, 4, 5]
    original_test_list = test_list.copy()
    impure_modified = impure_modify_list(test_list, 3)
    
    return {
        'pure_functions': {
            'add_results_equal': pure_result1 == pure_result2,
            'add_result': pure_result1,
            'original_list_unchanged': original_list == [1, 2, 3, 4, 5],
            'multiplied_result': pure_multiplied,
            'filtered_result': pure_filtered
        },
        'impure_functions': {
            'add_results_equal': impure_result1 == impure_result2,
            'counter_value': counter,
            'log_messages_count': len(log_messages),
            'list_was_modified': test_list != original_test_list,
            'modified_result': impure_modified
        }
    }

# Higher-order functions
def higher_order_functions_demo():
    """Demonstrate higher-order functions"""
    
    # Functions that take other functions as arguments
    def apply_function(func: Callable[[int], int], value: int) -> int:
        return func(value)
    
    def apply_to_list(func: Callable[[int], int], numbers: List[int]) -> List[int]:
        return [func(n) for n in numbers]
    
    def compose_functions(f: Callable[[int], int], g: Callable[[int], int]) -> Callable[[int], int]:
        return lambda x: f(g(x))
    
    def filter_and_transform(predicate: Callable[[int], bool], 
                           transform: Callable[[int], int], 
                           numbers: List[int]) -> List[int]:
        return [transform(n) for n in numbers if predicate(n)]
    
    # Functions that return other functions
    def make_multiplier(factor: int) -> Callable[[int], int]:
        def multiplier(x: int) -> int:
            return x * factor
        return multiplier
    
    def make_validator(min_val: int, max_val: int) -> Callable[[int], bool]:
        def validator(value: int) -> bool:
            return min_val <= value <= max_val
        return validator
    
    def make_accumulator(initial: int = 0) -> Callable[[int], int]:
        total = initial
        def accumulator(value: int) -> int:
            nonlocal total
            total += value
            return total
        return accumulator
    
    # Test functions
    def square(x: int) -> int:
        return x ** 2
    
    def double(x: int) -> int:
        return x * 2
    
    def is_even(x: int) -> bool:
        return x % 2 == 0
    
    # Test higher-order functions
    numbers = [1, 2, 3, 4, 5, 6]
    
    # Function application
    square_of_5 = apply_function(square, 5)
    squared_list = apply_to_list(square, numbers)
    
    # Function composition
    square_then_double = compose_functions(double, square)
    composed_result = square_then_double(3)  # double(square(3)) = double(9) = 18
    
    # Filter and transform
    even_squares = filter_and_transform(is_even, square, numbers)
    
    # Function factories
    triple = make_multiplier(3)
    validate_percentage = make_validator(0, 100)
    
    tripled_numbers = [triple(n) for n in numbers[:3]]
    validation_results = [validate_percentage(n) for n in [50, 150, -10, 75]]
    
    # Accumulator
    acc = make_accumulator(10)
    accumulator_results = [acc(n) for n in [1, 2, 3, 4]]
    
    # Built-in higher-order functions
    mapped_squares = list(map(square, numbers))
    filtered_evens = list(filter(is_even, numbers))
    
    # Reduce operation
    sum_result = functools.reduce(operator.add, numbers)
    product_result = functools.reduce(operator.mul, numbers)
    
    return {
        'function_application': {
            'square_of_5': square_of_5,
            'squared_list': squared_list
        },
        'function_composition': {
            'composed_result': composed_result,
            'expected_result': 18
        },
        'filter_transform': {
            'even_squares': even_squares,
            'original_evens': [2, 4, 6]
        },
        'function_factories': {
            'tripled_numbers': tripled_numbers,
            'validation_results': validation_results,
            'accumulator_results': accumulator_results
        },
        'builtin_functions': {
            'mapped_squares': mapped_squares,
            'filtered_evens': filtered_evens,
            'sum_result': sum_result,
            'product_result': product_result
        }
    }

# Closures and lexical scoping
def closures_demo():
    """Demonstrate closures and lexical scoping"""
    
    # Basic closure
    def outer_function(x: int):
        def inner_function(y: int) -> int:
            return x + y  # Captures x from outer scope
        return inner_function
    
    # Closure with mutable state
    def make_counter(start: int = 0):
        count = start
        
        def counter() -> int:
            nonlocal count
            count += 1
            return count
        
        def get_count() -> int:
            return count
        
        def reset() -> None:
            nonlocal count
            count = start
        
        # Return multiple functions that share state
        counter.get = get_count
        counter.reset = reset
        return counter
    
    # Closure for configuration
    def make_formatter(prefix: str, suffix: str):
        def format_string(text: str) -> str:
            return f"{prefix}{text}{suffix}"
        return format_string
    
    # Closure with multiple variables
    def make_calculator(operation: str):
        operators = {
            'add': operator.add,
            'sub': operator.sub,
            'mul': operator.mul,
            'div': operator.truediv
        }
        
        op_func = operators.get(operation, operator.add)
        
        def calculate(x: float, y: float) -> float:
            return op_func(x, y)
        
        def get_operation() -> str:
            return operation
        
        calculate.operation = get_operation
        return calculate
    
    # Closures in loops (common pitfall)
    def create_functions_wrong():
        functions = []
        for i in range(5):
            # This captures the variable i, not its value
            functions.append(lambda: i)
        return functions
    
    def create_functions_correct():
        functions = []
        for i in range(5):
            # This captures the value of i
            functions.append(lambda x=i: x)
        return functions
    
    # Test closures
    add_5 = outer_function(5)
    result1 = add_5(10)  # 5 + 10 = 15
    result2 = add_5(20)  # 5 + 20 = 25
    
    # Test counter closure
    counter1 = make_counter(10)
    counter2 = make_counter(0)
    
    counter1_results = [counter1() for _ in range(3)]
    counter2_results = [counter2() for _ in range(2)]
    
    current_count1 = counter1.get()
    counter1.reset()
    count_after_reset = counter1.get()
    
    # Test formatter closure
    html_formatter = make_formatter("<b>", "</b>")
    bracket_formatter = make_formatter("[", "]")
    
    html_result = html_formatter("bold text")
    bracket_result = bracket_formatter("bracketed")
    
    # Test calculator closure
    adder = make_calculator('add')
    multiplier = make_calculator('mul')
    
    add_result = adder(10, 5)
    mul_result = multiplier(10, 5)
    adder_operation = adder.operation()
    
    # Test loop closures
    wrong_functions = create_functions_wrong()
    correct_functions = create_functions_correct()
    
    wrong_results = [f() for f in wrong_functions]
    correct_results = [f() for f in correct_functions]
    
    return {
        'basic_closure': {
            'result1': result1,
            'result2': result2,
            'closure_works': result1 == 15 and result2 == 25
        },
        'stateful_closure': {
            'counter1_results': counter1_results,
            'counter2_results': counter2_results,
            'current_count1': current_count1,
            'count_after_reset': count_after_reset
        },
        'formatter_closure': {
            'html_result': html_result,
            'bracket_result': bracket_result
        },
        'calculator_closure': {
            'add_result': add_result,
            'mul_result': mul_result,
            'adder_operation': adder_operation
        },
        'loop_closures': {
            'wrong_results': wrong_results,
            'correct_results': correct_results,
            'wrong_all_same': len(set(wrong_results)) == 1,
            'correct_different': len(set(correct_results)) == 5
        }
    }

# Functional data transformations
def functional_transformations_demo():
    """Demonstrate functional data transformation patterns"""
    
    # Sample data
    @dataclass
    class Person:
        name: str
        age: int
        city: str
        salary: int
    
    people = [
        Person("Alice", 30, "New York", 70000),
        Person("Bob", 25, "San Francisco", 80000),
        Person("Charlie", 35, "New York", 60000),
        Person("Diana", 28, "Chicago", 75000),
        Person("Eve", 32, "San Francisco", 90000)
    ]
    
    # Map transformations
    def get_names(people_list: List[Person]) -> List[str]:
        return list(map(lambda p: p.name, people_list))
    
    def get_ages_in_5_years(people_list: List[Person]) -> List[int]:
        return list(map(lambda p: p.age + 5, people_list))
    
    def format_person_info(people_list: List[Person]) -> List[str]:
        return list(map(lambda p: f"{p.name} ({p.age}) from {p.city}", people_list))
    
    # Filter transformations
    def filter_by_age(people_list: List[Person], min_age: int) -> List[Person]:
        return list(filter(lambda p: p.age >= min_age, people_list))
    
    def filter_by_city(people_list: List[Person], city: str) -> List[Person]:
        return list(filter(lambda p: p.city == city, people_list))
    
    def filter_by_salary(people_list: List[Person], min_salary: int) -> List[Person]:
        return list(filter(lambda p: p.salary >= min_salary, people_list))
    
    # Reduce operations
    def total_salary(people_list: List[Person]) -> int:
        return functools.reduce(lambda acc, p: acc + p.salary, people_list, 0)
    
    def average_age(people_list: List[Person]) -> float:
        total_age = functools.reduce(lambda acc, p: acc + p.age, people_list, 0)
        return total_age / len(people_list) if people_list else 0
    
    def oldest_person(people_list: List[Person]) -> Optional[Person]:
        if not people_list:
            return None
        return functools.reduce(lambda p1, p2: p1 if p1.age > p2.age else p2, people_list)
    
    # Complex transformations (chaining)
    def analyze_city_data(people_list: List[Person], city: str) -> Dict[str, Any]:
        city_people = filter_by_city(people_list, city)
        
        if not city_people:
            return {'city': city, 'count': 0}
        
        return {
            'city': city,
            'count': len(city_people),
            'total_salary': total_salary(city_people),
            'average_age': average_age(city_people),
            'names': get_names(city_people)
        }
    
    # Functional composition
    def pipe(*functions):
        """Compose functions left to right"""
        return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)
    
    def compose(*functions):
        """Compose functions right to left"""
        return functools.reduce(lambda f, g: lambda x: f(g(x)), functions)
    
    # Pipeline example
    def get_high_earners_names(people_list: List[Person]) -> List[str]:
        high_earners = filter_by_salary(people_list, 75000)
        return get_names(high_earners)
    
    # Using partial application
    filter_adults = functools.partial(filter_by_age, min_age=30)
    filter_sf = functools.partial(filter_by_city, city="San Francisco")
    
    # Run transformations
    names = get_names(people)
    future_ages = get_ages_in_5_years(people)
    formatted_info = format_person_info(people)
    
    adults = filter_adults(people)
    sf_people = filter_sf(people)
    high_earners = filter_by_salary(people, 75000)
    
    total_sal = total_salary(people)
    avg_age = average_age(people)
    oldest = oldest_person(people)
    
    ny_analysis = analyze_city_data(people, "New York")
    sf_analysis = analyze_city_data(people, "San Francisco")
    
    high_earner_names = get_high_earners_names(people)
    
    # Composition example
    process_adults_in_sf = compose(get_names, filter_sf, filter_adults)
    adult_sf_names = process_adults_in_sf(people)
    
    return {
        'map_operations': {
            'names': names,
            'future_ages': future_ages,
            'formatted_info': formatted_info[:2]  # First 2 for brevity
        },
        'filter_operations': {
            'adults_count': len(adults),
            'sf_people_count': len(sf_people),
            'high_earners_count': len(high_earners)
        },
        'reduce_operations': {
            'total_salary': total_sal,
            'average_age': avg_age,
            'oldest_person_name': oldest.name if oldest else None
        },
        'complex_analysis': {
            'ny_analysis': ny_analysis,
            'sf_analysis': sf_analysis
        },
        'composition': {
            'high_earner_names': high_earner_names,
            'adult_sf_names': adult_sf_names
        }
    }

# Immutable data patterns
def immutable_data_demo():
    """Demonstrate immutable data patterns"""
    
    # Immutable data structures using tuples and namedtuples
    from collections import namedtuple
    
    Point = namedtuple('Point', ['x', 'y'])
    Person = namedtuple('Person', ['name', 'age', 'city'])
    
    # Creating immutable updates
    def update_point(point: Point, x: Optional[int] = None, y: Optional[int] = None) -> Point:
        return Point(
            x if x is not None else point.x,
            y if y is not None else point.y
        )
    
    def update_person(person: Person, **kwargs) -> Person:
        return person._replace(**kwargs)
    
    # Immutable list operations
    def append_immutable(lst: Tuple, item: Any) -> Tuple:
        return lst + (item,)
    
    def prepend_immutable(lst: Tuple, item: Any) -> Tuple:
        return (item,) + lst
    
    def remove_immutable(lst: Tuple, item: Any) -> Tuple:
        return tuple(x for x in lst if x != item)
    
    def update_at_index(lst: Tuple, index: int, new_value: Any) -> Tuple:
        return lst[:index] + (new_value,) + lst[index+1:]
    
    # Immutable dictionary operations
    def add_key(d: dict, key: str, value: Any) -> dict:
        new_dict = d.copy()
        new_dict[key] = value
        return new_dict
    
    def remove_key(d: dict, key: str) -> dict:
        new_dict = d.copy()
        new_dict.pop(key, None)
        return new_dict
    
    def update_dict(d: dict, **kwargs) -> dict:
        new_dict = d.copy()
        new_dict.update(kwargs)
        return new_dict
    
    # Deep immutable operations
    def deep_update_nested(data: dict, path: List[str], value: Any) -> dict:
        """Update nested dictionary immutably"""
        if not path:
            return value
        
        key = path[0]
        rest = path[1:]
        
        new_data = data.copy()
        if key in new_data and rest:
            new_data[key] = deep_update_nested(new_data[key], rest, value)
        else:
            new_data[key] = value if not rest else deep_update_nested({}, rest, value)
        
        return new_data
    
    # Immutable data class
    @dataclass(frozen=True)
    class ImmutablePerson:
        name: str
        age: int
        address: str
        
        def with_age(self, new_age: int) -> 'ImmutablePerson':
            return ImmutablePerson(self.name, new_age, self.address)
        
        def with_address(self, new_address: str) -> 'ImmutablePerson':
            return ImmutablePerson(self.name, self.age, new_address)
    
    # Test immutable operations
    original_point = Point(10, 20)
    updated_point = update_point(original_point, x=15)
    
    original_person = Person("Alice", 30, "New York")
    updated_person = update_person(original_person, age=31)
    
    # List operations
    original_list = (1, 2, 3, 4, 5)
    appended_list = append_immutable(original_list, 6)
    prepended_list = prepend_immutable(original_list, 0)
    removed_list = remove_immutable(original_list, 3)
    updated_list = update_at_index(original_list, 2, 99)
    
    # Dictionary operations
    original_dict = {'a': 1, 'b': 2, 'c': 3}
    dict_with_key = add_key(original_dict, 'd', 4)
    dict_without_key = remove_key(original_dict, 'b')
    updated_dict = update_dict(original_dict, a=10, e=5)
    
    # Nested dictionary update
    nested_data = {
        'user': {
            'profile': {
                'name': 'Alice',
                'age': 30
            },
            'preferences': {
                'theme': 'dark'
            }
        }
    }
    
    updated_nested = deep_update_nested(nested_data, ['user', 'profile', 'age'], 31)
    
    # Immutable data class
    immutable_person = ImmutablePerson("Bob", 25, "123 Main St")
    aged_person = immutable_person.with_age(26)
    moved_person = immutable_person.with_address("456 Oak Ave")
    
    # Test immutability
    try:
        immutable_person.age = 30  # This should fail
        immutable_class_works = False
    except Exception:
        immutable_class_works = True
    
    return {
        'namedtuple_updates': {
            'original_point': original_point,
            'updated_point': updated_point,
            'point_unchanged': original_point == Point(10, 20),
            'original_person': original_person,
            'updated_person': updated_person,
            'person_unchanged': original_person.age == 30
        },
        'list_operations': {
            'original': original_list,
            'appended': appended_list,
            'prepended': prepended_list,
            'removed': removed_list,
            'updated': updated_list,
            'original_unchanged': original_list == (1, 2, 3, 4, 5)
        },
        'dict_operations': {
            'original': original_dict,
            'with_key': dict_with_key,
            'without_key': dict_without_key,
            'updated': updated_dict,
            'original_unchanged': original_dict == {'a': 1, 'b': 2, 'c': 3}
        },
        'nested_update': {
            'original_age': nested_data['user']['profile']['age'],
            'updated_age': updated_nested['user']['profile']['age'],
            'original_data_unchanged': nested_data['user']['profile']['age'] == 30
        },
        'immutable_dataclass': {
            'original_person': (immutable_person.name, immutable_person.age, immutable_person.address),
            'aged_person': (aged_person.name, aged_person.age, aged_person.address),
            'moved_person': (moved_person.name, moved_person.age, moved_person.address),
            'immutability_enforced': immutable_class_works
        }
    }

# Recursive patterns and thinking
def recursive_patterns_demo():
    """Demonstrate recursive functional patterns"""
    
    # Basic recursion patterns
    def factorial(n: int) -> int:
        return 1 if n <= 1 else n * factorial(n - 1)
    
    def fibonacci(n: int) -> int:
        return n if n <= 1 else fibonacci(n - 1) + fibonacci(n - 2)
    
    def power(base: int, exp: int) -> int:
        return 1 if exp == 0 else base * power(base, exp - 1)
    
    # Tail recursion (Python doesn't optimize, but good pattern)
    def factorial_tail(n: int, acc: int = 1) -> int:
        return acc if n <= 1 else factorial_tail(n - 1, acc * n)
    
    def fibonacci_tail(n: int, a: int = 0, b: int = 1) -> int:
        return a if n == 0 else fibonacci_tail(n - 1, b, a + b)
    
    # List processing with recursion
    def sum_list(lst: List[int]) -> int:
        return 0 if not lst else lst[0] + sum_list(lst[1:])
    
    def reverse_list(lst: List[Any]) -> List[Any]:
        return [] if not lst else reverse_list(lst[1:]) + [lst[0]]
    
    def map_recursive(func: Callable[[T], U], lst: List[T]) -> List[U]:
        return [] if not lst else [func(lst[0])] + map_recursive(func, lst[1:])
    
    def filter_recursive(predicate: Callable[[T], bool], lst: List[T]) -> List[T]:
        if not lst:
            return []
        head, *tail = lst
        filtered_tail = filter_recursive(predicate, tail)
        return [head] + filtered_tail if predicate(head) else filtered_tail
    
    # Tree processing
    @dataclass
    class TreeNode:
        value: int
        left: Optional['TreeNode'] = None
        right: Optional['TreeNode'] = None
    
    def tree_sum(node: Optional[TreeNode]) -> int:
        if node is None:
            return 0
        return node.value + tree_sum(node.left) + tree_sum(node.right)
    
    def tree_height(node: Optional[TreeNode]) -> int:
        if node is None:
            return 0
        return 1 + max(tree_height(node.left), tree_height(node.right))
    
    def tree_map(func: Callable[[int], int], node: Optional[TreeNode]) -> Optional[TreeNode]:
        if node is None:
            return None
        return TreeNode(
            func(node.value),
            tree_map(func, node.left),
            tree_map(func, node.right)
        )
    
    def tree_find(predicate: Callable[[int], bool], node: Optional[TreeNode]) -> bool:
        if node is None:
            return False
        return (predicate(node.value) or 
                tree_find(predicate, node.left) or 
                tree_find(predicate, node.right))
    
    # Mutual recursion
    def is_even(n: int) -> bool:
        return True if n == 0 else is_odd(n - 1)
    
    def is_odd(n: int) -> bool:
        return False if n == 0 else is_even(n - 1)
    
    # Memoized recursion
    def memoize(func: Callable) -> Callable:
        cache = {}
        def wrapper(*args):
            if args in cache:
                return cache[args]
            result = func(*args)
            cache[args] = result
            return result
        return wrapper
    
    @memoize
    def fibonacci_memo(n: int) -> int:
        return n if n <= 1 else fibonacci_memo(n - 1) + fibonacci_memo(n - 2)
    
    # Test recursive functions
    test_numbers = [5, 10, 6]
    
    factorial_results = [factorial(n) for n in test_numbers]
    factorial_tail_results = [factorial_tail(n) for n in test_numbers]
    
    fibonacci_results = [fibonacci(n) for n in [5, 8, 10]]
    fibonacci_tail_results = [fibonacci_tail(n) for n in [5, 8, 10]]
    fibonacci_memo_results = [fibonacci_memo(n) for n in [5, 8, 10]]
    
    # List processing
    test_list = [1, 2, 3, 4, 5]
    sum_result = sum_list(test_list)
    reversed_result = reverse_list(test_list)
    
    mapped_result = map_recursive(lambda x: x * 2, test_list)
    filtered_result = filter_recursive(lambda x: x % 2 == 0, test_list)
    
    # Tree processing
    #       5
    #      / \
    #     3   8
    #    / \   \
    #   1   4   9
    tree = TreeNode(5,
                   TreeNode(3, TreeNode(1), TreeNode(4)),
                   TreeNode(8, None, TreeNode(9)))
    
    tree_sum_result = tree_sum(tree)
    tree_height_result = tree_height(tree)
    
    doubled_tree = tree_map(lambda x: x * 2, tree)
    doubled_tree_sum = tree_sum(doubled_tree)
    
    has_even = tree_find(lambda x: x % 2 == 0, tree)
    has_ten = tree_find(lambda x: x == 10, tree)
    
    # Mutual recursion
    even_odd_results = [(is_even(n), is_odd(n)) for n in [0, 1, 4, 7]]
    
    return {
        'basic_recursion': {
            'factorial_results': factorial_results,
            'factorial_tail_results': factorial_tail_results,
            'results_match': factorial_results == factorial_tail_results
        },
        'fibonacci_comparison': {
            'basic_results': fibonacci_results,
            'tail_results': fibonacci_tail_results,
            'memo_results': fibonacci_memo_results,
            'all_match': fibonacci_results == fibonacci_tail_results == fibonacci_memo_results
        },
        'list_processing': {
            'original_list': test_list,
            'sum_result': sum_result,
            'reversed_result': reversed_result,
            'mapped_result': mapped_result,
            'filtered_result': filtered_result
        },
        'tree_processing': {
            'tree_sum': tree_sum_result,
            'tree_height': tree_height_result,
            'doubled_tree_sum': doubled_tree_sum,
            'has_even_number': has_even,
            'has_ten': has_ten
        },
        'mutual_recursion': {
            'even_odd_results': even_odd_results
        }
    }

# Functional utilities and itertools
def functional_utilities_demo():
    """Demonstrate functional programming utilities"""
    
    # Itertools combinations
    def demonstrate_itertools():
        data = [1, 2, 3, 4]
        
        # Combinations and permutations
        combinations_2 = list(itertools.combinations(data, 2))
        permutations_2 = list(itertools.permutations(data, 2))
        combinations_with_replacement = list(itertools.combinations_with_replacement(data, 2))
        
        # Infinite iterators (limited for demo)
        count_from_10 = list(itertools.islice(itertools.count(10), 5))
        cycle_abc = list(itertools.islice(itertools.cycle(['a', 'b', 'c']), 10))
        repeat_x = list(itertools.islice(itertools.repeat('x'), 3))
        
        # Filtering iterators
        data_with_negatives = [-2, -1, 0, 1, 2, 3, 4]
        positives = list(itertools.filterfalse(lambda x: x <= 0, data_with_negatives))
        negatives_and_zero = list(itertools.filterfalse(lambda x: x > 0, data_with_negatives))
        
        # Grouping
        data_for_grouping = [1, 1, 2, 2, 2, 3, 1, 1]
        grouped = [(k, list(g)) for k, g in itertools.groupby(data_for_grouping)]
        
        # Accumulate
        accumulated = list(itertools.accumulate(data))
        accumulated_mul = list(itertools.accumulate(data, operator.mul))
        
        return {
            'combinations_2': combinations_2,
            'permutations_2': permutations_2[:6],  # Limit output
            'combinations_with_replacement': combinations_with_replacement,
            'count_from_10': count_from_10,
            'cycle_abc': cycle_abc,
            'repeat_x': repeat_x,
            'positives': positives,
            'negatives_and_zero': negatives_and_zero,
            'grouped': grouped,
            'accumulated': accumulated,
            'accumulated_mul': accumulated_mul
        }
    
    # Operator module functions
    def demonstrate_operators():
        data = [1, 2, 3, 4, 5]
        
        # Arithmetic operators
        sum_with_add = functools.reduce(operator.add, data)
        product_with_mul = functools.reduce(operator.mul, data)
        
        # Comparison operators
        max_with_gt = functools.reduce(lambda a, b: a if operator.gt(a, b) else b, data)
        
        # Attribute and item operators
        @dataclass
        class Item:
            name: str
            value: int
        
        items = [Item("a", 10), Item("b", 5), Item("c", 15)]
        
        names = list(map(operator.attrgetter('name'), items))
        values = list(map(operator.attrgetter('value'), items))
        
        # Dictionary operations
        dicts = [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}, {'x': 5, 'y': 6}]
        x_values = list(map(operator.itemgetter('x'), dicts))
        y_values = list(map(operator.itemgetter('y'), dicts))
        
        # Method operations
        strings = ['hello', 'world', 'python']
        upper_strings = list(map(operator.methodcaller('upper'), strings))
        lengths = list(map(operator.methodcaller('__len__'), strings))
        
        return {
            'arithmetic': {
                'sum': sum_with_add,
                'product': product_with_mul,
                'max': max_with_gt
            },
            'attribute_operations': {
                'names': names,
                'values': values
            },
            'item_operations': {
                'x_values': x_values,
                'y_values': y_values
            },
            'method_operations': {
                'upper_strings': upper_strings,
                'lengths': lengths
            }
        }
    
    # Functools utilities
    def demonstrate_functools():
        # Partial application
        multiply = lambda x, y: x * y
        double = functools.partial(multiply, 2)
        triple = functools.partial(multiply, 3)
        
        doubled_values = [double(x) for x in [1, 2, 3, 4, 5]]
        tripled_values = [triple(x) for x in [1, 2, 3, 4, 5]]
        
        # Single dispatch
        @functools.singledispatch
        def process(arg):
            return f"Processing unknown type: {type(arg).__name__}"
        
        @process.register
        def _(arg: int):
            return f"Processing integer: {arg * 2}"
        
        @process.register
        def _(arg: str):
            return f"Processing string: {arg.upper()}"
        
        @process.register
        def _(arg: list):
            return f"Processing list of {len(arg)} items"
        
        dispatch_results = [
            process(42),
            process("hello"),
            process([1, 2, 3]),
            process(3.14)
        ]
        
        # LRU Cache (already covered but including for completeness)
        @functools.lru_cache(maxsize=32)
        def expensive_computation(n):
            return sum(i ** 2 for i in range(n))
        
        # Wraps decorator
        def my_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                print(f"Calling {func.__name__}")
                return func(*args, **kwargs)
            return wrapper
        
        @my_decorator
        def sample_function():
            """Sample function docstring"""
            return "Hello from sample function"
        
        # Check if metadata is preserved
        metadata_preserved = (
            sample_function.__name__ == 'sample_function' and
            'Sample function' in sample_function.__doc__
        )
        
        return {
            'partial_application': {
                'doubled_values': doubled_values,
                'tripled_values': tripled_values
            },
            'single_dispatch': {
                'dispatch_results': dispatch_results
            },
            'metadata_preservation': {
                'function_name': sample_function.__name__,
                'metadata_preserved': metadata_preserved
            }
        }
    
    # Custom functional utilities
    def create_custom_utilities():
        # Compose function
        def compose(*functions):
            return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
        
        # Pipe function
        def pipe(value, *functions):
            return functools.reduce(lambda acc, func: func(acc), functions, value)
        
        # Curry function
        def curry(func, arity=None):
            if arity is None:
                arity = func.__code__.co_argcount
            
            def curried(*args):
                if len(args) >= arity:
                    return func(*args)
                return lambda *more_args: curried(*(args + more_args))
            
            return curried
        
        # Test custom utilities
        add = lambda x, y: x + y
        multiply = lambda x: x * 2
        square = lambda x: x ** 2
        
        # Composition
        composed_func = compose(square, multiply, lambda x: x + 1)
        composition_result = composed_func(3)  # ((3 + 1) * 2) ** 2 = 64
        
        # Pipeline
        pipeline_result = pipe(3, lambda x: x + 1, multiply, square)  # Same as above
        
        # Currying
        curried_add = curry(add)
        add_5 = curried_add(5)
        curry_result = add_5(10)  # 15
        
        return {
            'composition_result': composition_result,
            'pipeline_result': pipeline_result,
            'curry_result': curry_result,
            'results_match': composition_result == pipeline_result == 64
        }
    
    # Run all demonstrations
    itertools_results = demonstrate_itertools()
    operator_results = demonstrate_operators()
    functools_results = demonstrate_functools()
    custom_utilities_results = create_custom_utilities()
    
    return {
        'itertools': itertools_results,
        'operators': operator_results,
        'functools': functools_results,
        'custom_utilities': custom_utilities_results
    }

# Comprehensive testing
def run_all_functional_demos():
    """Execute all functional programming demonstrations"""
    demo_functions = [
        ('pure_functions', pure_functions_demo),
        ('higher_order_functions', higher_order_functions_demo),
        ('closures', closures_demo),
        ('functional_transformations', functional_transformations_demo),
        ('immutable_data', immutable_data_demo),
        ('recursive_patterns', recursive_patterns_demo),
        ('functional_utilities', functional_utilities_demo)
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
    print("=== Python Functional Programming Demo ===")
    
    # Run all demonstrations
    all_results = run_all_functional_demos()
    
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
    
    print("\n=== FUNCTIONAL PROGRAMMING CONCEPTS ===")
    
    concepts = {
        "Pure Functions": "Functions with no side effects that return same output for same input",
        "Higher-Order Functions": "Functions that take or return other functions",
        "Closures": "Functions that capture variables from their enclosing scope",
        "Immutability": "Data structures that cannot be modified after creation",
        "Recursion": "Functions that call themselves to solve problems",
        "Currying": "Transform multi-argument function into chain of single-argument functions",
        "Composition": "Combine simple functions to create complex operations",
        "Lazy Evaluation": "Delay computation until result is needed"
    }
    
    for concept, description in concepts.items():
        print(f"  {concept}: {description}")
    
    print("\n=== FUNCTIONAL PROGRAMMING BENEFITS ===")
    
    benefits = [
        "Easier to reason about and test",
        "More predictable behavior",
        "Better modularity and reusability",
        "Natural parallelization opportunities",
        "Reduced bugs from side effects",
        "Cleaner, more expressive code",
        "Better composition and abstraction",
        "Easier debugging and maintenance"
    ]
    
    for benefit in benefits:
        print(f"  • {benefit}")
    
    print("\n=== PYTHON FUNCTIONAL TOOLS ===")
    
    tools = {
        "map()": "Apply function to every item in iterable",
        "filter()": "Filter items based on predicate function",
        "functools.reduce()": "Apply function cumulatively to items",
        "functools.partial()": "Create partially applied functions",
        "functools.lru_cache()": "Memoization decorator",
        "itertools": "Efficient iterators for loops and combinations",
        "operator": "Functional interface to built-in operators",
        "collections.namedtuple": "Immutable data structures",
        "dataclasses (frozen=True)": "Immutable data classes"
    }
    
    for tool, description in tools.items():
        print(f"  {tool}: {description}")
    
    print("\n=== BEST PRACTICES ===")
    
    best_practices = [
        "Prefer pure functions over functions with side effects",
        "Use immutable data structures when possible",
        "Leverage higher-order functions for abstraction",
        "Use functional composition to build complex operations",
        "Apply memoization for expensive recursive functions",
        "Use generator expressions for memory efficiency",
        "Prefer map/filter/reduce over explicit loops when clear",
        "Use partial application to create specialized functions",
        "Avoid deep recursion due to Python's recursion limit",
        "Combine functional and object-oriented paradigms appropriately",
        "Use type hints for better functional interfaces",
        "Consider performance implications of functional approaches"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== Functional Programming Complete! ===")
    print("  Advanced functional programming patterns and techniques mastered")
