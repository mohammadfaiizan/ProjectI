"""
Python List and Dictionary Comprehensions: Advanced Patterns and Optimizations
Implementation-focused with minimal comments, maximum functionality coverage
"""

import time
import random
import math
from typing import List, Dict, Set, Tuple, Any, Iterator
from collections import defaultdict, Counter
import sys

# Basic comprehension patterns
def basic_list_comprehensions():
    # Simple list comprehensions
    numbers = list(range(20))
    
    basic_patterns = {
        'squares': [x**2 for x in range(10)],
        'evens': [x for x in numbers if x % 2 == 0],
        'odd_squares': [x**2 for x in numbers if x % 2 == 1],
        'string_lengths': [len(word) for word in ['python', 'java', 'javascript', 'go']],
        'uppercase': [word.upper() for word in ['hello', 'world', 'python']],
        'filtered_lengths': [len(word) for word in ['a', 'hello', 'world', 'python'] if len(word) > 2]
    }
    
    # Nested comprehensions
    matrix = [[i*j for j in range(1, 4)] for i in range(1, 4)]
    flattened = [item for row in matrix for item in row]
    
    # Conditional expressions in comprehensions
    categorized = ['positive' if x > 0 else 'negative' if x < 0 else 'zero' 
                  for x in [-2, -1, 0, 1, 2]]
    
    basic_patterns.update({
        'matrix_3x3': matrix,
        'flattened_matrix': flattened,
        'categorized_numbers': categorized
    })
    
    return basic_patterns

def advanced_list_comprehensions():
    # Multiple conditions
    data = list(range(50))
    
    advanced_patterns = {
        'multiple_conditions': [x for x in data if x % 2 == 0 and x % 3 == 0],
        'complex_filter': [x for x in data if x > 10 and x < 30 and x % 5 == 0],
        'function_filter': [x for x in data if str(x).isdigit() and int(str(x)[-1]) > 5]
    }
    
    # Nested loops in comprehensions
    pairs = [(x, y) for x in range(3) for y in range(3) if x != y]
    combinations = [(i, j, i+j) for i in range(5) for j in range(5) if i+j < 7]
    
    # String operations
    words = ['hello', 'world', 'python', 'programming', 'language']
    string_ops = {
        'first_letters': [word[0] for word in words],
        'vowel_words': [word for word in words if any(v in word for v in 'aeiou')],
        'long_words_upper': [word.upper() for word in words if len(word) > 5],
        'reversed_words': [word[::-1] for word in words]
    }
    
    # Mathematical operations
    math_ops = {
        'primes_check': [x for x in range(2, 30) 
                        if all(x % i != 0 for i in range(2, int(x**0.5) + 1))],
        'fibonacci_like': [a + b for a, b in zip(range(10), range(1, 11))],
        'factorials': [math.factorial(x) for x in range(8)],
        'perfect_squares': [x for x in range(100) if int(x**0.5)**2 == x]
    }
    
    advanced_patterns.update({
        'coordinate_pairs': pairs,
        'sum_combinations': combinations,
        'string_operations': string_ops,
        'mathematical_operations': math_ops
    })
    
    return advanced_patterns

def dictionary_comprehensions():
    # Basic dictionary comprehensions
    numbers = range(10)
    
    basic_dict_patterns = {
        'square_dict': {x: x**2 for x in numbers},
        'even_squares': {x: x**2 for x in numbers if x % 2 == 0},
        'word_lengths': {word: len(word) for word in ['python', 'java', 'go', 'rust']},
        'reversed_mapping': {v: k for k, v in enumerate(['a', 'b', 'c', 'd'])}
    }
    
    # Advanced dictionary comprehensions
    students = [
        ('Alice', 85), ('Bob', 92), ('Charlie', 78), ('Diana', 96), ('Eve', 88)
    ]
    
    advanced_dict_patterns = {
        'student_grades': {name: grade for name, grade in students},
        'high_grades': {name: grade for name, grade in students if grade > 85},
        'grade_categories': {name: 'A' if grade >= 90 else 'B' if grade >= 80 else 'C' 
                           for name, grade in students},
        'name_lengths': {name: len(name) for name, _ in students}
    }
    
    # Nested dictionary comprehensions
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    nested_dict = {f'row_{i}': {f'col_{j}': matrix[i][j] 
                               for j in range(len(matrix[i]))} 
                   for i in range(len(matrix))}
    
    # Dictionary from lists
    keys = ['name', 'age', 'city', 'job']
    values = ['Alice', 30, 'NYC', 'Engineer']
    zipped_dict = {k: v for k, v in zip(keys, values)}
    
    # Conditional dictionary creation
    data = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
    filtered_dict = {k: v for k, v in data.items() if v % 2 == 1}
    transformed_dict = {k.upper(): v*2 for k, v in data.items() if v > 2}
    
    return {
        'basic_patterns': basic_dict_patterns,
        'advanced_patterns': advanced_dict_patterns,
        'nested_structure': nested_dict,
        'from_zip': zipped_dict,
        'filtered': filtered_dict,
        'transformed': transformed_dict
    }

def set_comprehensions():
    # Basic set comprehensions
    numbers = [1, 2, 2, 3, 3, 3, 4, 4, 5]
    
    set_patterns = {
        'unique_squares': {x**2 for x in numbers},
        'even_numbers': {x for x in range(20) if x % 2 == 0},
        'vowels_in_words': {char for word in ['hello', 'world'] for char in word if char in 'aeiou'},
        'prime_factors': {x for x in range(2, 20) if any(x % i == 0 for i in range(2, x))}
    }
    
    # Advanced set operations
    words = ['python', 'java', 'javascript', 'python', 'go', 'rust', 'java']
    
    advanced_set_patterns = {
        'unique_words': {word for word in words},
        'long_words': {word for word in words if len(word) > 4},
        'first_letters': {word[0] for word in words},
        'word_endings': {word[-2:] for word in words if len(word) >= 2}
    }
    
    # Mathematical sets
    math_sets = {
        'perfect_squares': {x**2 for x in range(1, 11)},
        'cubes': {x**3 for x in range(1, 6)},
        'triangular_numbers': {x*(x+1)//2 for x in range(1, 11)},
        'multiples_of_3_or_5': {x for x in range(1, 50) if x % 3 == 0 or x % 5 == 0}
    }
    
    set_patterns.update({
        'advanced_patterns': advanced_set_patterns,
        'mathematical_sets': math_sets
    })
    
    return set_patterns

def generator_expressions():
    # Basic generator expressions
    def basic_generators():
        numbers_gen = (x**2 for x in range(1000000))  # Memory efficient
        evens_gen = (x for x in range(100) if x % 2 == 0)
        
        # Take first few values to demonstrate
        return {
            'first_10_squares': [next(numbers_gen) for _ in range(10)],
            'even_numbers': list(evens_gen),
            'generator_type': str(type(numbers_gen))
        }
    
    # Advanced generator patterns
    def advanced_generators():
        # Infinite generator
        def fibonacci_gen():
            a, b = 0, 1
            while True:
                yield a
                a, b = b, a + b
        
        fib_gen = fibonacci_gen()
        
        # File processing generator (simulated)
        def process_lines():
            lines = ['line 1', 'line 2', 'line 3', 'important line', 'line 5']
            return (line.upper() for line in lines if 'important' in line)
        
        processed = list(process_lines())
        
        # Chained generators
        def chain_generators():
            gen1 = (x for x in range(5))
            gen2 = (x for x in range(5, 10))
            return list(x for gen in [gen1, gen2] for x in gen)
        
        return {
            'fibonacci_first_10': [next(fib_gen) for _ in range(10)],
            'processed_lines': processed,
            'chained_result': chain_generators()
        }
    
    return {
        'basic': basic_generators(),
        'advanced': advanced_generators()
    }

def nested_comprehensions():
    # Matrix operations
    matrix_a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    matrix_b = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    
    matrix_operations = {
        'transpose': [[row[i] for row in matrix_a] for i in range(len(matrix_a[0]))],
        'element_wise_add': [[matrix_a[i][j] + matrix_b[i][j] 
                             for j in range(len(matrix_a[i]))] 
                            for i in range(len(matrix_a))],
        'flatten_nested': [element for row in matrix_a for element in row],
        'diagonal': [matrix_a[i][i] for i in range(len(matrix_a))]
    }
    
    # Complex nested structures
    data = {
        'users': [
            {'name': 'Alice', 'scores': [85, 92, 78]},
            {'name': 'Bob', 'scores': [90, 87, 85]},
            {'name': 'Charlie', 'scores': [75, 80, 82]}
        ]
    }
    
    nested_processing = {
        'all_scores': [score for user in data['users'] for score in user['scores']],
        'average_scores': {user['name']: sum(user['scores'])/len(user['scores']) 
                          for user in data['users']},
        'high_scorers': [user['name'] for user in data['users'] 
                        if max(user['scores']) > 85],
        'score_ranges': {user['name']: max(user['scores']) - min(user['scores']) 
                        for user in data['users']}
    }
    
    # Multi-level nesting
    three_d = [[[i+j+k for k in range(2)] for j in range(2)] for i in range(3)]
    flattened_3d = [element for matrix in three_d 
                   for row in matrix for element in row]
    
    return {
        'matrix_operations': matrix_operations,
        'nested_processing': nested_processing,
        'three_dimensional': three_d,
        'flattened_3d': flattened_3d
    }

def conditional_comprehensions():
    # Multiple conditions and complex logic
    numbers = list(range(1, 101))
    
    conditional_patterns = {
        'fizzbuzz': [
            'FizzBuzz' if x % 15 == 0 else
            'Fizz' if x % 3 == 0 else
            'Buzz' if x % 5 == 0 else x
            for x in range(1, 21)
        ],
        'categorize_numbers': [
            'small' if x < 25 else
            'medium' if x < 75 else 'large'
            for x in numbers[::10]  # Every 10th number
        ],
        'grade_letters': [
            'A' if score >= 90 else
            'B' if score >= 80 else
            'C' if score >= 70 else
            'D' if score >= 60 else 'F'
            for score in [95, 87, 76, 68, 54]
        ]
    }
    
    # Complex filtering
    words = ['python', 'java', 'javascript', 'go', 'rust', 'c++', 'swift']
    
    complex_filters = {
        'long_words_with_vowels': [
            word for word in words 
            if len(word) > 4 and any(v in word for v in 'aeiou')
        ],
        'short_or_starts_with_j': [
            word for word in words 
            if len(word) <= 3 or word.startswith('j')
        ],
        'contains_double_letters': [
            word for word in words 
            if any(word[i] == word[i+1] for i in range(len(word)-1))
        ]
    }
    
    # Nested conditions
    students = [
        {'name': 'Alice', 'age': 20, 'grades': [85, 90, 88]},
        {'name': 'Bob', 'age': 19, 'grades': [78, 85, 82]},
        {'name': 'Charlie', 'age': 21, 'grades': [92, 95, 90]}
    ]
    
    nested_conditions = {
        'honor_students': [
            student['name'] for student in students
            if student['age'] >= 20 and sum(student['grades'])/len(student['grades']) >= 85
        ],
        'grade_status': {
            student['name']: 'excellent' if min(student['grades']) >= 85 else
                           'good' if sum(student['grades'])/len(student['grades']) >= 80 else
                           'needs_improvement'
            for student in students
        }
    }
    
    return {
        'conditional_patterns': conditional_patterns,
        'complex_filters': complex_filters,
        'nested_conditions': nested_conditions
    }

def performance_optimizations():
    # Performance comparison between different approaches
    def time_operation(operation, iterations=1000):
        start = time.time()
        for _ in range(iterations):
            result = operation()
        return (time.time() - start) * 1000 / iterations, result
    
    data = list(range(1000))
    
    # Comparison: comprehension vs traditional loop
    def traditional_loop():
        result = []
        for x in data:
            if x % 2 == 0:
                result.append(x**2)
        return result
    
    def list_comprehension():
        return [x**2 for x in data if x % 2 == 0]
    
    def generator_expression():
        return list(x**2 for x in data if x % 2 == 0)
    
    # Timing comparisons
    traditional_time, traditional_result = time_operation(traditional_loop, 100)
    comprehension_time, comprehension_result = time_operation(list_comprehension, 100)
    generator_time, generator_result = time_operation(generator_expression, 100)
    
    # Memory efficiency test
    def memory_test():
        # Large list comprehension
        large_list = [x for x in range(100000)]
        list_size = sys.getsizeof(large_list)
        
        # Generator expression
        large_gen = (x for x in range(100000))
        gen_size = sys.getsizeof(large_gen)
        
        return {
            'list_size_bytes': list_size,
            'generator_size_bytes': gen_size,
            'memory_ratio': list_size / gen_size if gen_size > 0 else 0
        }
    
    # Optimization techniques
    optimization_techniques = {
        'early_filtering': [x**2 for x in data if x % 2 == 0 if x < 100],  # Double filtering
        'precomputed_values': [x**2 for x in data[:100] if x % 2 == 0],   # Limit data first
        'combined_operations': [(x, x**2, x**3) for x in range(50) if x % 3 == 0]  # Multiple operations
    }
    
    return {
        'performance_comparison_ms': {
            'traditional_loop': f"{traditional_time:.4f}",
            'list_comprehension': f"{comprehension_time:.4f}",
            'generator_expression': f"{generator_time:.4f}"
        },
        'results_equal': (traditional_result == comprehension_result == generator_result),
        'memory_analysis': memory_test(),
        'optimization_examples': optimization_techniques
    }

def real_world_applications():
    # Data processing scenarios
    def process_csv_data():
        # Simulated CSV data
        csv_rows = [
            ['Name', 'Age', 'Salary', 'Department'],
            ['Alice', '30', '75000', 'Engineering'],
            ['Bob', '25', '65000', 'Marketing'],
            ['Charlie', '35', '85000', 'Engineering'],
            ['Diana', '28', '70000', 'Sales']
        ]
        
        header = csv_rows[0]
        data_rows = csv_rows[1:]
        
        # Process using comprehensions
        processed_data = [
            {header[i]: row[i] for i in range(len(header))}
            for row in data_rows
        ]
        
        # Analytics using comprehensions
        analytics = {
            'engineers': [emp for emp in processed_data if emp['Department'] == 'Engineering'],
            'high_earners': [emp['Name'] for emp in processed_data if int(emp['Salary']) > 70000],
            'avg_age_by_dept': {
                dept: sum(int(emp['Age']) for emp in processed_data if emp['Department'] == dept) / 
                      len([emp for emp in processed_data if emp['Department'] == dept])
                for dept in set(emp['Department'] for emp in processed_data)
            }
        }
        
        return {
            'processed_data': processed_data,
            'analytics': analytics
        }
    
    def text_analysis():
        text = """
        Python is a powerful programming language. It is widely used for data science,
        web development, and automation. Python's syntax is clean and readable.
        Many developers choose Python for its simplicity and extensive libraries.
        """
        
        words = text.lower().split()
        
        text_stats = {
            'word_frequencies': {
                word: len([w for w in words if w == word])
                for word in set(words) if word.isalpha()
            },
            'long_words': [word for word in words if len(word) > 6 and word.isalpha()],
            'word_lengths': {word: len(word) for word in set(words) if word.isalpha()},
            'sentences_with_python': [
                sentence.strip() for sentence in text.split('.')
                if 'python' in sentence.lower()
            ]
        }
        
        return text_stats
    
    def web_scraping_simulation():
        # Simulated web data
        web_data = [
            {'url': 'https://example.com/page1', 'status': 200, 'size': 1024},
            {'url': 'https://example.com/page2', 'status': 404, 'size': 0},
            {'url': 'https://test.com/home', 'status': 200, 'size': 2048},
            {'url': 'https://demo.org/about', 'status': 500, 'size': 512}
        ]
        
        web_analysis = {
            'successful_pages': [item for item in web_data if item['status'] == 200],
            'domain_sizes': {
                item['url'].split('/')[2]: item['size'] 
                for item in web_data if item['status'] == 200
            },
            'error_urls': [item['url'] for item in web_data if item['status'] >= 400],
            'total_size_by_domain': {
                domain: sum(item['size'] for item in web_data 
                          if item['url'].split('/')[2] == domain)
                for domain in set(item['url'].split('/')[2] for item in web_data)
            }
        }
        
        return web_analysis
    
    return {
        'csv_processing': process_csv_data(),
        'text_analysis': text_analysis(),
        'web_scraping': web_scraping_simulation()
    }

def interview_problems():
    def transpose_matrix(matrix):
        """Transpose matrix using list comprehension"""
        return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
    
    def flatten_nested_list(nested):
        """Flatten arbitrarily nested list"""
        return [item for sublist in nested 
               for item in (flatten_nested_list(sublist) if isinstance(sublist, list) else [sublist])]
    
    def group_words_by_length(words):
        """Group words by their length"""
        lengths = set(len(word) for word in words)
        return {length: [word for word in words if len(word) == length] 
                for length in lengths}
    
    def find_common_elements(lists):
        """Find elements common to all lists"""
        if not lists:
            return []
        common = set(lists[0])
        return [item for item in common if all(item in lst for lst in lists[1:])]
    
    def pascal_triangle(n):
        """Generate Pascal's triangle using comprehensions"""
        return [[1 if j == 0 or j == i else 
                (triangle[i-1][j-1] + triangle[i-1][j] if i > 0 else 1)
                for j in range(i+1)]
               for i, triangle in enumerate([[[1]] + [[1 if j == 0 or j == i else 0 
                                                    for j in range(i+1)] 
                                                   for i in range(1, n)]]) 
               if i < n]
    
    def word_frequency_analysis(text):
        """Comprehensive word frequency analysis"""
        words = [word.lower().strip('.,!?";') for word in text.split() if word.isalpha()]
        
        return {
            'frequencies': {word: sum(1 for w in words if w == word) 
                          for word in set(words)},
            'by_length': {length: [word for word in set(words) if len(word) == length]
                         for length in set(len(word) for word in words)},
            'starting_letters': {letter: [word for word in set(words) if word.startswith(letter)]
                               for letter in set(word[0] for word in words)}
        }
    
    # Test problems
    test_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    test_nested = [[1, 2], [3, [4, 5]], [6, [7, [8, 9]]]]
    test_words = ['python', 'java', 'go', 'rust', 'javascript', 'c++']
    test_lists = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    test_text = "Python is great. Python is powerful. Great tools for Python development."
    
    # Generate Pascal's triangle (simplified version)
    def simple_pascal(n):
        triangle = []
        for i in range(n):
            row = [1 if j == 0 or j == i else 
                   triangle[i-1][j-1] + triangle[i-1][j] 
                   for j in range(i+1)]
            triangle.append(row)
        return triangle
    
    results = {
        'matrix_transpose': transpose_matrix(test_matrix),
        'flattened_list': flatten_nested_list(test_nested),
        'words_by_length': group_words_by_length(test_words),
        'common_elements': find_common_elements(test_lists),
        'pascal_triangle': simple_pascal(5),
        'word_analysis': word_frequency_analysis(test_text)
    }
    
    return results

def comprehension_best_practices():
    # Good vs bad practices
    good_practices = {
        'readable': [x**2 for x in range(10)],  # Clear and simple
        'meaningful_names': [student.upper() for student in ['alice', 'bob', 'charlie']],
        'appropriate_length': [word for word in ['hello', 'world'] if len(word) > 4]
    }
    
    # Practices to avoid (shown for educational purposes)
    practices_to_avoid = {
        'too_complex': 'Avoid: [complex_function(x, y, z) for x in data for y in other_data for z in third_data if condition1(x) and condition2(y) and condition3(z)]',
        'side_effects': 'Avoid: [print(x) or x for x in data]  # Side effects in comprehensions',
        'unclear_logic': 'Avoid: [x if y else z for x, y, z in data if some_complex_condition]'
    }
    
    # When to use alternatives
    alternatives = {
        'use_loop_when': 'Complex logic, multiple statements, error handling needed',
        'use_map_when': 'Simple function application: map(str.upper, words)',
        'use_filter_when': 'Simple filtering: filter(lambda x: x > 0, numbers)',
        'use_generator_when': 'Large datasets, memory efficiency important'
    }
    
    # Performance guidelines
    performance_tips = {
        'filter_early': [x**2 for x in range(1000) if x % 2 == 0],  # Filter before expensive operations
        'avoid_repeated_calls': [len(word) for word in ['hello', 'world']],  # Don't call len() multiple times
        'use_sets_for_membership': [x for x in range(100) if x in {2, 3, 5, 7, 11}]  # Set lookup is O(1)
    }
    
    return {
        'good_practices': good_practices,
        'avoid_these': practices_to_avoid,
        'when_to_use_alternatives': alternatives,
        'performance_tips': performance_tips
    }

# Comprehensive testing
def run_all_comprehension_demos():
    """Execute all comprehension demonstrations"""
    demo_functions = [
        ('basic_list', basic_list_comprehensions),
        ('advanced_list', advanced_list_comprehensions),
        ('dictionary', dictionary_comprehensions),
        ('set', set_comprehensions),
        ('generator', generator_expressions),
        ('nested', nested_comprehensions),
        ('conditional', conditional_comprehensions),
        ('performance', performance_optimizations),
        ('real_world', real_world_applications),
        ('interview_problems', interview_problems),
        ('best_practices', comprehension_best_practices)
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
    print("=== Python List and Dictionary Comprehensions Demo ===")
    
    # Run all demonstrations
    all_results = run_all_comprehension_demos()
    
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
                if isinstance(value, (list, dict, set)) and len(str(value)) > 100:
                    if isinstance(value, list) and len(value) > 5:
                        print(f"  {key}: {value[:5]}... (showing first 5)")
                    elif isinstance(value, dict) and len(value) > 3:
                        items = list(value.items())[:3]
                        print(f"  {key}: {dict(items)}... (showing first 3)")
                    else:
                        print(f"  {key}: {str(value)[:100]}... (truncated)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Result: {result}")
    
    print("\n=== COMPREHENSION GUIDELINES ===")
    
    guidelines = {
        'Readability': 'Keep comprehensions simple and readable',
        'Performance': 'Use generators for large datasets',
        'Complexity': 'Use regular loops for complex logic',
        'Filtering': 'Filter early in the comprehension',
        'Nesting': 'Limit nesting depth for maintainability',
        'Memory': 'Consider memory usage for large collections'
    }
    
    for category, guideline in guidelines.items():
        print(f"  {category}: {guideline}")
    
    print("\n=== PERFORMANCE SUMMARY ===")
    total_time = sum(float(data.get('execution_time', '0ms')[:-2]) 
                    for data in all_results.values() 
                    if 'execution_time' in data)
    print(f"  Total execution time: {total_time:.2f}ms")
    print(f"  Functions executed: {len(all_results)}")
    print(f"  Average per function: {total_time/len(all_results):.2f}ms")
