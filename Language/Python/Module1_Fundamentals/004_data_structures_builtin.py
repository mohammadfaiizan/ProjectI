"""
Python Built-in Data Structures: Lists, Tuples, Dictionaries, Sets
Implementation-focused with minimal comments, maximum functionality coverage
"""

from typing import List, Tuple, Dict, Set, Any, Union, Optional
from collections import defaultdict, Counter, deque, namedtuple
import time
import sys

# List operations and methods
def list_operations():
    # Creation and initialization
    empty_list = []
    number_list = [1, 2, 3, 4, 5]
    mixed_list = [1, "hello", 3.14, True, [1, 2]]
    range_list = list(range(10))
    repeated_list = [0] * 5
    nested_list = [[i*j for j in range(3)] for i in range(3)]
    
    # Basic operations
    original = [1, 2, 3]
    original.append(4)
    original.extend([5, 6])
    original.insert(0, 0)
    
    # Removal operations
    remove_list = [1, 2, 3, 2, 4, 2]
    remove_list.remove(2)  # Removes first occurrence
    popped = remove_list.pop()  # Removes and returns last
    popped_index = remove_list.pop(0)  # Removes and returns at index
    
    # List modification
    modify_list = [1, 2, 3, 4, 5]
    modify_list.reverse()
    sort_list = [3, 1, 4, 1, 5, 9, 2, 6]
    sort_list.sort()
    
    # Slicing operations
    slice_list = list(range(10))
    first_half = slice_list[:5]
    second_half = slice_list[5:]
    every_second = slice_list[::2]
    reversed_slice = slice_list[::-1]
    step_slice = slice_list[1:8:2]
    
    return {
        'creation': {
            'empty': empty_list,
            'numbers': number_list,
            'mixed': mixed_list[:3],  # Truncate for display
            'range': range_list[:5],
            'repeated': repeated_list,
            'nested': nested_list
        },
        'modification': {
            'after_append_extend': original,
            'after_removal': remove_list,
            'popped_values': [popped, popped_index],
            'reversed': modify_list,
            'sorted': sort_list
        },
        'slicing': {
            'first_half': first_half,
            'second_half': second_half,
            'every_second': every_second,
            'reversed': reversed_slice,
            'step_slice': step_slice
        }
    }

def advanced_list_operations():
    # List comprehensions
    squares = [x**2 for x in range(10)]
    evens = [x for x in range(20) if x % 2 == 0]
    nested_comp = [[x*y for y in range(3)] for x in range(3)]
    conditional_comp = [x if x > 0 else 0 for x in [-2, -1, 0, 1, 2]]
    
    # List methods and functions
    data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    count_ones = data.count(1)
    index_five = data.index(5)
    sorted_copy = sorted(data, reverse=True)
    
    # List concatenation and repetition
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    concatenated = list1 + list2
    repeated = list1 * 3
    
    # List copying (shallow vs deep)
    original = [[1, 2], [3, 4]]
    shallow_copy = original.copy()
    import copy
    deep_copy = copy.deepcopy(original)
    
    # Modify to show difference
    original[0][0] = 99
    
    # List as stack and queue operations
    stack = [1, 2, 3]
    stack.append(4)  # Push
    top = stack.pop()  # Pop
    
    queue = deque([1, 2, 3])
    queue.append(4)  # Enqueue
    front = queue.popleft()  # Dequeue
    
    return {
        'comprehensions': {
            'squares': squares[:5],
            'evens': evens[:5],
            'nested': nested_comp,
            'conditional': conditional_comp
        },
        'methods': {
            'count': count_ones,
            'index': index_five,
            'sorted_copy': sorted_copy
        },
        'operations': {
            'concatenated': concatenated,
            'repeated': repeated
        },
        'copying': {
            'original_after_modify': original,
            'shallow_copy': shallow_copy,
            'deep_copy': deep_copy
        },
        'stack_queue': {
            'stack_after_ops': stack,
            'popped': top,
            'queue_after_ops': list(queue),
            'dequeued': front
        }
    }

# Tuple implementations
def tuple_operations():
    # Creation and characteristics
    empty_tuple = ()
    single_tuple = (42,)  # Note the comma
    number_tuple = (1, 2, 3, 4, 5)
    mixed_tuple = (1, "hello", 3.14, True)
    nested_tuple = ((1, 2), (3, 4), (5, 6))
    
    # Tuple unpacking
    point = (3, 4)
    x, y = point
    
    # Multiple assignment
    a, b, c = (10, 20, 30)
    
    # Swapping variables
    x, y = y, x
    
    # Extended unpacking
    first, *middle, last = (1, 2, 3, 4, 5, 6)
    
    # Tuple methods
    data = (1, 2, 3, 2, 4, 2, 5)
    count_twos = data.count(2)
    index_four = data.index(4)
    
    # Tuple operations
    tuple1 = (1, 2, 3)
    tuple2 = (4, 5, 6)
    concatenated = tuple1 + tuple2
    repeated = tuple1 * 2
    
    # Named tuples
    Point = namedtuple('Point', ['x', 'y'])
    p1 = Point(1, 2)
    p2 = Point(x=3, y=4)
    
    # Accessing named tuple fields
    distance = ((p2.x - p1.x)**2 + (p2.y - p1.y)**2)**0.5
    
    return {
        'creation': {
            'empty': empty_tuple,
            'single': single_tuple,
            'numbers': number_tuple,
            'mixed': mixed_tuple,
            'nested': nested_tuple
        },
        'unpacking': {
            'point_coords': (x, y),
            'multiple_assign': (a, b, c),
            'after_swap': (x, y),
            'extended': (first, middle, last)
        },
        'methods': {
            'count': count_twos,
            'index': index_four
        },
        'operations': {
            'concatenated': concatenated,
            'repeated': repeated
        },
        'named_tuples': {
            'points': (p1, p2),
            'distance': distance
        }
    }

# Dictionary implementations
def dictionary_operations():
    # Creation methods
    empty_dict = {}
    literal_dict = {"name": "Alice", "age": 30, "city": "NYC"}
    dict_constructor = dict(name="Bob", age=25, city="LA")
    dict_from_pairs = dict([("a", 1), ("b", 2), ("c", 3)])
    dict_comprehension = {x: x**2 for x in range(5)}
    
    # Dictionary operations
    person = {"name": "Charlie", "age": 35}
    person["job"] = "Engineer"  # Add new key
    person["age"] = 36  # Update existing key
    
    # Dictionary methods
    keys_list = list(person.keys())
    values_list = list(person.values())
    items_list = list(person.items())
    
    # Safe access methods
    age = person.get("age", 0)
    salary = person.get("salary", "Not specified")
    
    # Dictionary modification methods
    person.update({"salary": 75000, "department": "IT"})
    removed_job = person.pop("job", "No job")
    
    # Dictionary copying
    original_dict = {"a": 1, "b": {"nested": 2}}
    shallow_copy = original_dict.copy()
    import copy
    deep_copy = copy.deepcopy(original_dict)
    
    # Modify to show difference
    original_dict["b"]["nested"] = 99
    
    # Dictionary merging (Python 3.9+)
    dict1 = {"a": 1, "b": 2}
    dict2 = {"b": 3, "c": 4}
    merged = {**dict1, **dict2}  # dict1 | dict2 in Python 3.9+
    
    return {
        'creation': {
            'literal': literal_dict,
            'constructor': dict_constructor,
            'from_pairs': dict_from_pairs,
            'comprehension': dict_comprehension
        },
        'modification': {
            'after_updates': person,
            'removed_value': removed_job
        },
        'access_methods': {
            'keys': keys_list,
            'values': values_list,
            'items': items_list,
            'safe_access': [age, salary]
        },
        'copying': {
            'original_after_modify': original_dict,
            'shallow_copy': shallow_copy,
            'deep_copy': deep_copy
        },
        'merging': merged
    }

def advanced_dictionary_operations():
    # Nested dictionaries
    nested_data = {
        "users": {
            "alice": {"age": 30, "role": "admin"},
            "bob": {"age": 25, "role": "user"}
        },
        "settings": {
            "theme": "dark",
            "notifications": True
        }
    }
    
    # Safe nested access
    alice_age = nested_data.get("users", {}).get("alice", {}).get("age", 0)
    
    # Dictionary with list values
    groups = {
        "admins": ["alice", "charlie"],
        "users": ["bob", "david", "eve"],
        "guests": []
    }
    
    # Adding to list values
    groups["users"].append("frank")
    groups.setdefault("moderators", []).append("alice")
    
    # Defaultdict usage
    dd = defaultdict(list)
    items = [("fruit", "apple"), ("fruit", "banana"), ("vegetable", "carrot")]
    for category, item in items:
        dd[category].append(item)
    
    # Counter usage
    text = "hello world hello python world"
    word_count = Counter(text.split())
    letter_count = Counter(text.replace(" ", ""))
    
    # Dictionary sorting
    scores = {"alice": 95, "bob": 87, "charlie": 92, "david": 88}
    sorted_by_name = dict(sorted(scores.items()))
    sorted_by_score = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    
    return {
        'nested_access': {
            'alice_age': alice_age,
            'nested_structure': nested_data["users"]
        },
        'list_values': {
            'groups_after_modify': groups,
            'defaultdict_result': dict(dd)
        },
        'counting': {
            'word_count': dict(word_count),
            'letter_count': dict(letter_count.most_common(5))
        },
        'sorting': {
            'by_name': sorted_by_name,
            'by_score': sorted_by_score
        }
    }

# Set operations and implementations
def set_operations():
    # Creation methods
    empty_set = set()
    literal_set = {1, 2, 3, 4, 5}
    set_from_list = set([1, 2, 2, 3, 3, 4])  # Duplicates removed
    set_from_string = set("hello")  # Unique characters
    set_comprehension = {x**2 for x in range(5)}
    
    # Set operations
    set1 = {1, 2, 3, 4, 5}
    set2 = {4, 5, 6, 7, 8}
    
    union = set1 | set2
    intersection = set1 & set2
    difference = set1 - set2
    symmetric_difference = set1 ^ set2
    
    # Set methods
    set_copy = set1.copy()
    set_copy.add(9)
    set_copy.update([10, 11])
    set_copy.discard(1)  # Doesn't raise error if not found
    
    try:
        set_copy.remove(1)  # Raises error if not found
    except KeyError:
        removed_error = "Key not found"
    else:
        removed_error = "Successfully removed"
    
    # Set relationships
    subset_check = {1, 2} <= {1, 2, 3, 4}
    superset_check = {1, 2, 3, 4} >= {1, 2}
    disjoint_check = {1, 2}.isdisjoint({3, 4})
    
    # Frozen sets (immutable)
    frozen = frozenset([1, 2, 3, 4])
    
    return {
        'creation': {
            'literal': literal_set,
            'from_list': set_from_list,
            'from_string': set_from_string,
            'comprehension': set_comprehension
        },
        'operations': {
            'union': union,
            'intersection': intersection,
            'difference': difference,
            'symmetric_difference': symmetric_difference
        },
        'modification': {
            'after_updates': set_copy,
            'removal_result': removed_error
        },
        'relationships': {
            'subset': subset_check,
            'superset': superset_check,
            'disjoint': disjoint_check
        },
        'frozen_set': frozen
    }

def advanced_set_operations():
    # Practical set applications
    def find_unique_words(text1, text2):
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        return {
            'unique_to_text1': words1 - words2,
            'unique_to_text2': words2 - words1,
            'common_words': words1 & words2,
            'all_words': words1 | words2
        }
    
    text1 = "Python is great for data science"
    text2 = "Python is excellent for web development"
    word_analysis = find_unique_words(text1, text2)
    
    # Set-based algorithms
    def remove_duplicates_preserve_order(items):
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    
    duplicated_list = [1, 2, 3, 2, 4, 1, 5, 3]
    unique_ordered = remove_duplicates_preserve_order(duplicated_list)
    
    # Set operations with custom objects
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
        
        def __hash__(self):
            return hash((self.name, self.age))
        
        def __eq__(self, other):
            return self.name == other.name and self.age == other.age
        
        def __repr__(self):
            return f"Person('{self.name}', {self.age})"
    
    people = {
        Person("Alice", 30),
        Person("Bob", 25),
        Person("Alice", 30),  # Duplicate
        Person("Charlie", 35)
    }
    
    return {
        'word_analysis': word_analysis,
        'deduplication': {
            'original': duplicated_list,
            'unique_ordered': unique_ordered
        },
        'custom_objects': list(people)
    }

# Performance comparisons
def performance_comparisons():
    # Time complexity demonstrations
    import time
    
    def time_operation(operation, iterations=1000):
        start = time.time()
        for _ in range(iterations):
            operation()
        return (time.time() - start) * 1000 / iterations
    
    # List vs Set lookup performance
    large_list = list(range(10000))
    large_set = set(range(10000))
    target = 9999
    
    list_lookup_time = time_operation(lambda: target in large_list, 100)
    set_lookup_time = time_operation(lambda: target in large_set, 100)
    
    # Dictionary vs List for key-value storage
    dict_data = {i: i*2 for i in range(1000)}
    list_data = [(i, i*2) for i in range(1000)]
    
    dict_access_time = time_operation(lambda: dict_data.get(999), 1000)
    list_search_time = time_operation(
        lambda: next((v for k, v in list_data if k == 999), None), 100
    )
    
    # Memory usage comparison
    list_memory = sys.getsizeof(large_list)
    set_memory = sys.getsizeof(large_set)
    dict_memory = sys.getsizeof(dict_data)
    
    return {
        'lookup_performance': {
            'list_time_ms': f"{list_lookup_time:.4f}",
            'set_time_ms': f"{set_lookup_time:.4f}",
            'speedup_factor': f"{list_lookup_time/set_lookup_time:.1f}x"
        },
        'access_performance': {
            'dict_time_ms': f"{dict_access_time:.4f}",
            'list_time_ms': f"{list_search_time:.4f}",
            'speedup_factor': f"{list_search_time/dict_access_time:.1f}x"
        },
        'memory_usage': {
            'list_bytes': list_memory,
            'set_bytes': set_memory,
            'dict_bytes': dict_memory
        }
    }

# Interview problems using data structures
def data_structure_interview_problems():
    def two_sum(nums: List[int], target: int) -> List[int]:
        """Find two numbers that sum to target"""
        seen = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i
        return []
    
    def group_anagrams(strs: List[str]) -> List[List[str]]:
        """Group strings that are anagrams"""
        anagram_groups = defaultdict(list)
        for s in strs:
            key = ''.join(sorted(s))
            anagram_groups[key].append(s)
        return list(anagram_groups.values())
    
    def intersection_of_arrays(nums1: List[int], nums2: List[int]) -> List[int]:
        """Find intersection of two arrays"""
        return list(set(nums1) & set(nums2))
    
    def top_k_frequent(nums: List[int], k: int) -> List[int]:
        """Find k most frequent elements"""
        count = Counter(nums)
        return [num for num, _ in count.most_common(k)]
    
    def valid_parentheses(s: str) -> bool:
        """Check if parentheses are valid"""
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        
        for char in s:
            if char in mapping:
                if not stack or stack.pop() != mapping[char]:
                    return False
            else:
                stack.append(char)
        
        return not stack
    
    def longest_consecutive(nums: List[int]) -> int:
        """Find longest consecutive sequence"""
        num_set = set(nums)
        longest = 0
        
        for num in num_set:
            if num - 1 not in num_set:  # Start of sequence
                current = num
                current_length = 1
                
                while current + 1 in num_set:
                    current += 1
                    current_length += 1
                
                longest = max(longest, current_length)
        
        return longest
    
    # Test problems
    test_results = {
        'two_sum': two_sum([2, 7, 11, 15], 9),
        'anagrams': group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]),
        'intersection': intersection_of_arrays([1, 2, 2, 1], [2, 2]),
        'top_k_frequent': top_k_frequent([1, 1, 1, 2, 2, 3], 2),
        'valid_parentheses': [
            valid_parentheses("()"),
            valid_parentheses("()[]{}"),
            valid_parentheses("(]")
        ],
        'longest_consecutive': longest_consecutive([100, 4, 200, 1, 3, 2])
    }
    
    return test_results

# Comprehensive testing and demonstration
def run_all_data_structure_demos():
    """Execute all data structure demonstrations"""
    demo_functions = [
        ('list_operations', list_operations),
        ('advanced_lists', advanced_list_operations),
        ('tuple_operations', tuple_operations),
        ('dictionary_operations', dictionary_operations),
        ('advanced_dictionaries', advanced_dictionary_operations),
        ('set_operations', set_operations),
        ('advanced_sets', advanced_set_operations),
        ('performance', performance_comparisons),
        ('interview_problems', data_structure_interview_problems)
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
    print("=== Python Built-in Data Structures Demo ===")
    
    # Run all demonstrations
    all_results = run_all_data_structure_demos()
    
    for category, data in all_results.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        
        if 'error' in data:
            print(f"  Error: {data['error']}")
            continue
            
        result = data['result']
        print(f"  Execution time: {data['execution_time']}")
        
        # Display results based on type and size
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, (list, dict, set)) and len(str(value)) > 150:
                    if isinstance(value, list) and len(value) > 5:
                        print(f"  {key}: {value[:5]}... (showing first 5)")
                    elif isinstance(value, dict) and len(value) > 5:
                        items = list(value.items())[:3]
                        print(f"  {key}: {dict(items)}... (showing first 3)")
                    else:
                        print(f"  {key}: {str(value)[:150]}... (truncated)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Result: {result}")
    
    print("\n=== DATA STRUCTURE COMPARISON ===")
    
    # Quick comparison of operations
    operations = {
        'List': 'O(n) search, O(1) append, ordered, mutable',
        'Tuple': 'O(n) search, immutable, ordered, hashable',
        'Dict': 'O(1) average search, O(1) insert, unordered (3.7+ insertion order)',
        'Set': 'O(1) average search, O(1) insert, unordered, unique elements'
    }
    
    for structure, complexity in operations.items():
        print(f"  {structure}: {complexity}")
    
    print("\n=== PERFORMANCE SUMMARY ===")
    total_time = sum(float(data.get('execution_time', '0ms')[:-2]) 
                    for data in all_results.values() 
                    if 'execution_time' in data)
    print(f"  Total execution time: {total_time:.2f}ms")
    print(f"  Functions executed: {len(all_results)}")
    print(f"  Average per function: {total_time/len(all_results):.2f}ms")
