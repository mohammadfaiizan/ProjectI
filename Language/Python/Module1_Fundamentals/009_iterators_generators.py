"""
Python Iterators and Generators: Iteration Protocols and Memory-Efficient Programming
Implementation-focused with minimal comments, maximum functionality coverage
"""

import time
import sys
from typing import Iterator, Generator, Iterable, Any, List, Dict
from collections.abc import Iterable as AbcIterable
from itertools import islice, chain, cycle, repeat, count, product, permutations, combinations
import random

# Basic iterator protocol
def iterator_fundamentals():
    # Manual iterator implementation
    class NumberIterator:
        def __init__(self, start, end):
            self.current = start
            self.end = end
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if self.current >= self.end:
                raise StopIteration
            else:
                self.current += 1
                return self.current - 1
    
    # Iterable class
    class NumberRange:
        def __init__(self, start, end):
            self.start = start
            self.end = end
        
        def __iter__(self):
            return NumberIterator(self.start, self.end)
    
    # Test custom iterator
    custom_range = NumberRange(1, 6)
    iterator_results = {
        'custom_range_list': list(custom_range),
        'multiple_iterations': [list(custom_range), list(custom_range)],  # Can iterate multiple times
        'iterator_type': str(type(iter(custom_range)))
    }
    
    # Built-in iterators
    builtin_examples = {
        'range_iterator': list(range(5)),
        'string_iterator': list('hello'),
        'dict_keys': list({'a': 1, 'b': 2}.keys()),
        'dict_values': list({'a': 1, 'b': 2}.values()),
        'dict_items': list({'a': 1, 'b': 2}.items())
    }
    
    # Iterator vs iterable demonstration
    my_list = [1, 2, 3, 4, 5]
    my_iterator = iter(my_list)
    
    protocol_demo = {
        'is_list_iterable': hasattr(my_list, '__iter__'),
        'is_iterator_iterable': hasattr(my_iterator, '__iter__'),
        'has_next_method': hasattr(my_iterator, '__next__'),
        'first_next': next(my_iterator),
        'second_next': next(my_iterator),
        'remaining_items': list(my_iterator)
    }
    
    return {
        'custom_iterator': iterator_results,
        'builtin_examples': builtin_examples,
        'protocol_demonstration': protocol_demo
    }

def advanced_iterators():
    # Reverse iterator
    class ReverseIterator:
        def __init__(self, data):
            self.data = data
            self.index = len(data)
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if self.index == 0:
                raise StopIteration
            self.index -= 1
            return self.data[self.index]
    
    # Filtered iterator
    class FilteredIterator:
        def __init__(self, data, condition):
            self.data = iter(data)
            self.condition = condition
        
        def __iter__(self):
            return self
        
        def __next__(self):
            while True:
                item = next(self.data)
                if self.condition(item):
                    return item
    
    # Batched iterator
    class BatchIterator:
        def __init__(self, data, batch_size):
            self.data = iter(data)
            self.batch_size = batch_size
        
        def __iter__(self):
            return self
        
        def __next__(self):
            batch = []
            try:
                for _ in range(self.batch_size):
                    batch.append(next(self.data))
                return batch
            except StopIteration:
                if batch:
                    return batch
                else:
                    raise StopIteration
    
    # Test advanced iterators
    test_data = list(range(20))
    
    advanced_results = {
        'reverse_iterator': list(ReverseIterator([1, 2, 3, 4, 5])),
        'filtered_evens': list(FilteredIterator(test_data, lambda x: x % 2 == 0)),
        'batched_data': list(BatchIterator(test_data, 3))
    }
    
    # Infinite iterator
    class InfiniteCounter:
        def __init__(self, start=0, step=1):
            self.current = start
            self.step = step
        
        def __iter__(self):
            return self
        
        def __next__(self):
            value = self.current
            self.current += self.step
            return value
    
    # Test infinite iterator (limited)
    infinite_counter = InfiniteCounter(10, 2)
    advanced_results['infinite_counter'] = [next(infinite_counter) for _ in range(5)]
    
    return advanced_results

def generator_fundamentals():
    # Basic generator function
    def simple_generator():
        yield 1
        yield 2
        yield 3
    
    # Generator with loop
    def count_up_to(max_count):
        count = 1
        while count <= max_count:
            yield count
            count += 1
    
    # Generator with conditions
    def even_numbers(limit):
        for i in range(limit):
            if i % 2 == 0:
                yield i
    
    # Fibonacci generator
    def fibonacci():
        a, b = 0, 1
        while True:
            yield a
            a, b = b, a + b
    
    # Test generators
    generator_results = {
        'simple_gen': list(simple_generator()),
        'count_gen': list(count_up_to(5)),
        'even_gen': list(even_numbers(10)),
        'fibonacci_first_10': [next(fibonacci()) for _ in range(10)]
    }
    
    # Generator expressions
    gen_expr_results = {
        'squares_gen': list(x**2 for x in range(5)),
        'filtered_gen': list(x for x in range(20) if x % 3 == 0),
        'string_gen': list(char.upper() for char in 'hello')
    }
    
    # Memory efficiency demonstration
    def memory_comparison():
        # List comprehension
        large_list = [x for x in range(100000)]
        list_size = sys.getsizeof(large_list)
        
        # Generator expression
        large_gen = (x for x in range(100000))
        gen_size = sys.getsizeof(large_gen)
        
        return {
            'list_size_bytes': list_size,
            'generator_size_bytes': gen_size,
            'memory_saved': list_size - gen_size,
            'ratio': list_size / gen_size if gen_size > 0 else 0
        }
    
    return {
        'generator_functions': generator_results,
        'generator_expressions': gen_expr_results,
        'memory_comparison': memory_comparison()
    }

def advanced_generators():
    # Generator with send method
    def echo_generator():
        value = None
        while True:
            value = yield value
            if value is not None:
                value = f"Echo: {value}"
    
    # Stateful generator
    def running_average():
        total = 0
        count = 0
        while True:
            value = yield total / count if count > 0 else 0
            if value is not None:
                total += value
                count += 1
    
    # Generator with exception handling
    def robust_generator():
        try:
            for i in range(5):
                yield i
        except GeneratorExit:
            print("Generator is being closed")
        finally:
            print("Cleanup in generator")
    
    # Coroutine-style generator
    def accumulator():
        total = 0
        while True:
            value = yield total
            if value is not None:
                total += value
    
    # Test advanced generators
    echo_gen = echo_generator()
    next(echo_gen)  # Prime the generator
    
    avg_gen = running_average()
    next(avg_gen)  # Prime the generator
    
    acc_gen = accumulator()
    next(acc_gen)  # Prime the generator
    
    advanced_results = {
        'echo_responses': [
            echo_gen.send("Hello"),
            echo_gen.send("World"),
            echo_gen.send("Python")
        ],
        'running_averages': [
            avg_gen.send(10),
            avg_gen.send(20),
            avg_gen.send(30)
        ],
        'accumulator_values': [
            acc_gen.send(5),
            acc_gen.send(10),
            acc_gen.send(15)
        ]
    }
    
    # Generator delegation with yield from
    def sub_generator():
        yield 1
        yield 2
        yield 3
    
    def main_generator():
        yield from sub_generator()
        yield 4
        yield 5
    
    advanced_results['yield_from'] = list(main_generator())
    
    return advanced_results

def itertools_demonstrations():
    # Infinite iterators
    infinite_examples = {
        'count_from_10': list(islice(count(10), 5)),
        'cycle_abc': list(islice(cycle(['a', 'b', 'c']), 8)),
        'repeat_hello': list(repeat('hello', 3))
    }
    
    # Combinatorial iterators
    letters = ['A', 'B', 'C']
    numbers = [1, 2]
    
    combinatorial_examples = {
        'product': list(product(letters, numbers)),
        'permutations_2': list(permutations(letters, 2)),
        'combinations_2': list(combinations(letters, 2)),
        'combinations_with_replacement': list(combinations_with_replacement(letters, 2))
    }
    
    # Chaining and grouping
    from itertools import combinations_with_replacement, groupby, takewhile, dropwhile
    
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    list3 = [7, 8, 9]
    
    chaining_examples = {
        'chain_lists': list(chain(list1, list2, list3)),
        'chain_from_iterable': list(chain.from_iterable([list1, list2, list3]))
    }
    
    # Filtering iterators
    numbers = range(20)
    
    filtering_examples = {
        'takewhile_less_10': list(takewhile(lambda x: x < 10, numbers)),
        'dropwhile_less_10': list(dropwhile(lambda x: x < 10, numbers)),
        'islice_2_to_10_step_2': list(islice(numbers, 2, 10, 2))
    }
    
    # Grouping data
    data = [('a', 1), ('a', 2), ('b', 3), ('b', 4), ('c', 5)]
    grouped = [(key, list(group)) for key, group in groupby(data, key=lambda x: x[0])]
    
    return {
        'infinite_iterators': infinite_examples,
        'combinatorial': combinatorial_examples,
        'chaining': chaining_examples,
        'filtering': filtering_examples,
        'grouped_data': grouped
    }

def custom_iteration_patterns():
    # File-like iterator
    class FileLineIterator:
        def __init__(self, content):
            self.lines = content.split('\n')
            self.index = 0
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if self.index >= len(self.lines):
                raise StopIteration
            line = self.lines[self.index]
            self.index += 1
            return line
    
    # Tree traversal iterator
    class TreeNode:
        def __init__(self, value, children=None):
            self.value = value
            self.children = children or []
        
        def __iter__(self):
            yield self.value
            for child in self.children:
                yield from child
    
    # Windowed iterator
    def windowed(iterable, n):
        """Return successive n-sized windows from iterable"""
        it = iter(iterable)
        window = list(islice(it, n))
        if len(window) == n:
            yield tuple(window)
        for x in it:
            window = window[1:] + [x]
            yield tuple(window)
    
    # Pairwise iterator
    def pairwise(iterable):
        """Return successive pairs from iterable"""
        it = iter(iterable)
        a = next(it, None)
        for b in it:
            yield (a, b)
            a = b
    
    # Chunked iterator
    def chunked(iterable, n):
        """Break iterable into chunks of size n"""
        it = iter(iterable)
        while True:
            chunk = list(islice(it, n))
            if not chunk:
                break
            yield chunk
    
    # Test custom patterns
    file_content = "Line 1\nLine 2\nLine 3\nLine 4"
    file_iter = FileLineIterator(file_content)
    
    # Create tree structure
    tree = TreeNode('root', [
        TreeNode('child1', [TreeNode('grandchild1'), TreeNode('grandchild2')]),
        TreeNode('child2')
    ])
    
    custom_results = {
        'file_lines': list(file_iter),
        'tree_traversal': list(tree),
        'windowed_data': list(windowed(range(10), 3)),
        'pairwise_data': list(pairwise('ABCDEFG')),
        'chunked_data': list(chunked(range(15), 4))
    }
    
    return custom_results

def generator_pipelines():
    # Data processing pipeline using generators
    def read_data():
        """Simulate reading data"""
        data = [
            {'name': 'Alice', 'age': 30, 'salary': 75000},
            {'name': 'Bob', 'age': 25, 'salary': 65000},
            {'name': 'Charlie', 'age': 35, 'salary': 85000},
            {'name': 'Diana', 'age': 28, 'salary': 70000},
            {'name': 'Eve', 'age': 32, 'salary': 80000}
        ]
        for item in data:
            yield item
    
    def filter_age(data_stream, min_age):
        """Filter by minimum age"""
        for item in data_stream:
            if item['age'] >= min_age:
                yield item
    
    def transform_salary(data_stream, factor):
        """Transform salary by factor"""
        for item in data_stream:
            item = item.copy()
            item['salary'] = item['salary'] * factor
            yield item
    
    def add_tax_info(data_stream, tax_rate):
        """Add tax information"""
        for item in data_stream:
            item = item.copy()
            item['tax'] = item['salary'] * tax_rate
            item['net_salary'] = item['salary'] - item['tax']
            yield item
    
    # Create processing pipeline
    pipeline = add_tax_info(
        transform_salary(
            filter_age(read_data(), 28),
            1.1  # 10% raise
        ),
        0.25  # 25% tax
    )
    
    pipeline_results = list(pipeline)
    
    # Async-style generator pipeline
    def pipeline_with_logging():
        """Pipeline with logging"""
        source = read_data()
        step1 = filter_age(source, 30)
        step2 = transform_salary(step1, 1.2)
        final = add_tax_info(step2, 0.3)
        
        results = []
        for item in final:
            results.append(item)
        
        return results
    
    return {
        'pipeline_results': pipeline_results,
        'logged_pipeline': pipeline_with_logging()
    }

def performance_comparisons():
    # Memory usage comparison
    def memory_usage_test(size=100000):
        # List approach
        def list_approach():
            data = list(range(size))
            processed = [x * 2 for x in data if x % 2 == 0]
            return processed[:10]  # Only need first 10
        
        # Generator approach
        def generator_approach():
            data = (x for x in range(size))
            processed = (x * 2 for x in data if x % 2 == 0)
            return list(islice(processed, 10))  # Only generate first 10
        
        # Timing comparison
        def time_function(func, iterations=10):
            start = time.time()
            for _ in range(iterations):
                result = func()
            return (time.time() - start) * 1000 / iterations, result
        
        list_time, list_result = time_function(list_approach)
        gen_time, gen_result = time_function(generator_approach)
        
        return {
            'list_time_ms': f"{list_time:.2f}",
            'generator_time_ms': f"{gen_time:.2f}",
            'results_equal': list_result == gen_result,
            'speedup': f"{list_time / gen_time:.2f}x" if gen_time > 0 else "N/A"
        }
    
    # Large dataset processing
    def large_dataset_simulation():
        # Simulate processing large CSV file
        def generate_csv_rows(num_rows):
            for i in range(num_rows):
                yield {
                    'id': i,
                    'value': random.randint(1, 1000),
                    'category': random.choice(['A', 'B', 'C'])
                }
        
        def process_rows(rows):
            for row in rows:
                if row['value'] > 500:
                    row['processed'] = row['value'] * 1.1
                    yield row
        
        # Process 1 million rows but only take first 5 results
        large_data = generate_csv_rows(1000000)
        processed = process_rows(large_data)
        results = list(islice(processed, 5))
        
        return {
            'processed_count': len(results),
            'sample_data': results
        }
    
    return {
        'memory_comparison': memory_usage_test(),
        'large_dataset': large_dataset_simulation()
    }

def iterator_interview_problems():
    def flatten_iterator(nested_list):
        """Flatten nested list using iterator"""
        for item in nested_list:
            if isinstance(item, list):
                yield from flatten_iterator(item)
            else:
                yield item
    
    def merge_sorted_iterators(*iterators):
        """Merge multiple sorted iterators"""
        import heapq
        heap = []
        
        # Initialize heap with first element from each iterator
        for i, it in enumerate(iterators):
            try:
                value = next(it)
                heapq.heappush(heap, (value, i, it))
            except StopIteration:
                pass
        
        while heap:
            value, iterator_index, iterator = heapq.heappop(heap)
            yield value
            
            try:
                next_value = next(iterator)
                heapq.heappush(heap, (next_value, iterator_index, iterator))
            except StopIteration:
                pass
    
    def sliding_window_max(arr, k):
        """Find maximum in each sliding window of size k"""
        from collections import deque
        
        def max_in_windows():
            dq = deque()
            
            for i in range(len(arr)):
                # Remove elements outside current window
                while dq and dq[0] <= i - k:
                    dq.popleft()
                
                # Remove smaller elements from back
                while dq and arr[dq[-1]] <= arr[i]:
                    dq.pop()
                
                dq.append(i)
                
                # Yield maximum when window is full
                if i >= k - 1:
                    yield arr[dq[0]]
        
        return list(max_in_windows())
    
    def unique_elements_iterator(iterable):
        """Yield unique elements while preserving order"""
        seen = set()
        for item in iterable:
            if item not in seen:
                seen.add(item)
                yield item
    
    def takewhile_inclusive(predicate, iterable):
        """Take elements while predicate is true, including the first false element"""
        for item in iterable:
            yield item
            if not predicate(item):
                break
    
    def round_robin(*iterables):
        """Round-robin through multiple iterables"""
        iterators = [iter(it) for it in iterables]
        while iterators:
            for it in list(iterators):
                try:
                    yield next(it)
                except StopIteration:
                    iterators.remove(it)
    
    # Test interview problems
    nested_data = [1, [2, 3], [4, [5, 6]], 7]
    sorted_lists = [
        [1, 4, 7, 10],
        [2, 5, 8, 11],
        [3, 6, 9, 12]
    ]
    
    test_results = {
        'flattened': list(flatten_iterator(nested_data)),
        'merged_sorted': list(merge_sorted_iterators(*[iter(lst) for lst in sorted_lists])),
        'sliding_max': sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3),
        'unique_elements': list(unique_elements_iterator([1, 2, 2, 3, 1, 4, 5, 3])),
        'takewhile_inclusive': list(takewhile_inclusive(lambda x: x < 5, range(10))),
        'round_robin': list(round_robin('ABC', '123', 'xyz'))
    }
    
    return test_results

def async_style_generators():
    # Simulate async behavior with generators
    def async_task_simulator():
        """Simulate async task execution"""
        tasks = [
            ('Task A', 2),
            ('Task B', 1),
            ('Task C', 3),
            ('Task D', 1)
        ]
        
        for task_name, duration in tasks:
            yield f"Starting {task_name}"
            # Simulate work
            for i in range(duration):
                yield f"{task_name} working... {i+1}/{duration}"
            yield f"Completed {task_name}"
    
    def event_driven_generator():
        """Simulate event-driven processing"""
        events = [
            ('login', 'user1'),
            ('click', 'button1'),
            ('logout', 'user1'),
            ('login', 'user2'),
            ('purchase', 'item1')
        ]
        
        state = {}
        
        for event_type, data in events:
            if event_type == 'login':
                state[data] = 'logged_in'
                yield f"User {data} logged in"
            elif event_type == 'logout':
                if data in state:
                    del state[data]
                yield f"User {data} logged out"
            elif event_type == 'click':
                yield f"Button {data} clicked"
            elif event_type == 'purchase':
                active_users = list(state.keys())
                yield f"Purchase {data} by users: {active_users}"
    
    # Coroutine-style data processor
    def data_processor():
        """Process data in coroutine style"""
        buffer = []
        while True:
            data = yield
            if data is None:
                # Flush buffer
                if buffer:
                    yield f"Processing batch: {buffer}"
                    buffer = []
            else:
                buffer.append(data)
                if len(buffer) >= 3:
                    yield f"Processing batch: {buffer}"
                    buffer = []
    
    # Test async-style generators
    async_results = {
        'task_simulation': list(async_task_simulator()),
        'event_processing': list(event_driven_generator())
    }
    
    # Test coroutine processor
    processor = data_processor()
    next(processor)  # Prime the generator
    
    processor_results = []
    test_data = ['item1', 'item2', 'item3', 'item4', 'item5']
    
    for item in test_data:
        result = processor.send(item)
        if result:
            processor_results.append(result)
    
    # Flush remaining
    final_result = processor.send(None)
    if final_result:
        processor_results.append(final_result)
    
    async_results['data_processor'] = processor_results
    
    return async_results

# Comprehensive testing
def run_all_iterator_demos():
    """Execute all iterator and generator demonstrations"""
    demo_functions = [
        ('iterator_fundamentals', iterator_fundamentals),
        ('advanced_iterators', advanced_iterators),
        ('generator_fundamentals', generator_fundamentals),
        ('advanced_generators', advanced_generators),
        ('itertools_demo', itertools_demonstrations),
        ('custom_patterns', custom_iteration_patterns),
        ('generator_pipelines', generator_pipelines),
        ('performance', performance_comparisons),
        ('interview_problems', iterator_interview_problems),
        ('async_style', async_style_generators)
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
    print("=== Python Iterators and Generators Demo ===")
    
    # Run all demonstrations
    all_results = run_all_iterator_demos()
    
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
                if isinstance(value, (list, dict)) and len(str(value)) > 120:
                    if isinstance(value, list) and len(value) > 5:
                        print(f"  {key}: {value[:5]}... (showing first 5)")
                    elif isinstance(value, dict) and len(value) > 3:
                        items = list(value.items())[:3]
                        print(f"  {key}: {dict(items)}... (showing first 3)")
                    else:
                        print(f"  {key}: {str(value)[:120]}... (truncated)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Result: {result}")
    
    print("\n=== ITERATOR BEST PRACTICES ===")
    
    best_practices = {
        'Memory Efficiency': 'Use generators for large datasets',
        'Lazy Evaluation': 'Process data on-demand with iterators',
        'Pipeline Processing': 'Chain generators for data transformation',
        'Iterator Protocol': 'Implement __iter__ and __next__ for custom iterators',
        'StopIteration': 'Always raise StopIteration when iterator is exhausted',
        'Generator Expressions': 'Use for simple transformations and filtering'
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
