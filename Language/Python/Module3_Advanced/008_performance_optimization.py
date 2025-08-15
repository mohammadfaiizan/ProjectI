"""
Python Performance Optimization: Profiling, Benchmarking, and Optimization Strategies
Implementation-focused with minimal comments, maximum functionality coverage
"""

import cProfile
import profile
import pstats
import io
import time
import timeit
import sys
import gc
import functools
import itertools
import operator
from typing import Any, Dict, List, Optional, Callable, Iterator
import threading
import multiprocessing
import concurrent.futures
import array
import collections
import bisect
import heapq
import math
import statistics

# Basic profiling and timing
def profiling_demo():
    """Demonstrate various profiling techniques"""
    
    # Sample functions to profile
    def cpu_intensive_function(n):
        """CPU-intensive function for profiling"""
        total = 0
        for i in range(n):
            total += math.sqrt(i) * math.sin(i)
        return total
    
    def memory_intensive_function(size):
        """Memory-intensive function"""
        data = []
        for i in range(size):
            data.append([j for j in range(100)])
        return sum(len(sublist) for sublist in data)
    
    def io_simulation(duration):
        """Simulate I/O operation"""
        time.sleep(duration)
        return "IO completed"
    
    # Method 1: timeit for micro-benchmarking
    timeit_result = timeit.timeit(
        lambda: cpu_intensive_function(1000),
        number=10
    )
    
    # Method 2: Manual timing
    start_time = time.perf_counter()
    manual_result = cpu_intensive_function(5000)
    manual_time = time.perf_counter() - start_time
    
    # Method 3: cProfile profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    profile_result = cpu_intensive_function(10000)
    memory_result = memory_intensive_function(100)
    
    profiler.disable()
    
    # Get profiling statistics
    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    profile_output = stats_stream.getvalue()
    
    # Method 4: Context manager for timing
    class TimeContext:
        def __init__(self):
            self.time = 0
        
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.time = time.perf_counter() - self.start
    
    with TimeContext() as timer:
        context_result = cpu_intensive_function(3000)
    
    # Function decorator for timing
    def timing_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"{func.__name__} took {end - start:.6f} seconds")
            return result
        return wrapper
    
    @timing_decorator
    def decorated_function(n):
        return sum(i ** 2 for i in range(n))
    
    decorated_result = decorated_function(1000)
    
    return {
        'timeit_result': f"{timeit_result:.6f}s",
        'manual_timing': f"{manual_time:.6f}s",
        'context_timing': f"{timer.time:.6f}s",
        'function_results': {
            'manual_result': manual_result,
            'profile_result': profile_result,
            'memory_result': memory_result,
            'context_result': context_result,
            'decorated_result': decorated_result
        },
        'profile_stats_length': len(profile_output.split('\n'))
    }

# Data structure optimization
def data_structure_optimization():
    """Demonstrate optimization through better data structures"""
    
    # List vs collections.deque for operations
    def test_list_operations(n):
        data = list(range(n))
        
        # Append operations
        start = time.perf_counter()
        for i in range(1000):
            data.append(i)
        append_time = time.perf_counter() - start
        
        # Pop from end
        start = time.perf_counter()
        for i in range(500):
            data.pop()
        pop_end_time = time.perf_counter() - start
        
        # Insert at beginning (slow for lists)
        start = time.perf_counter()
        for i in range(100):
            data.insert(0, i)
        insert_begin_time = time.perf_counter() - start
        
        return append_time, pop_end_time, insert_begin_time
    
    def test_deque_operations(n):
        from collections import deque
        data = deque(range(n))
        
        # Append operations
        start = time.perf_counter()
        for i in range(1000):
            data.append(i)
        append_time = time.perf_counter() - start
        
        # Pop from end
        start = time.perf_counter()
        for i in range(500):
            data.pop()
        pop_end_time = time.perf_counter() - start
        
        # Insert at beginning (fast for deques)
        start = time.perf_counter()
        for i in range(100):
            data.appendleft(i)
        insert_begin_time = time.perf_counter() - start
        
        return append_time, pop_end_time, insert_begin_time
    
    # Dictionary optimization
    def test_dict_vs_list_lookup(n):
        # List lookup (O(n))
        data_list = list(range(n))
        target = n // 2
        
        start = time.perf_counter()
        result_list = target in data_list
        list_lookup_time = time.perf_counter() - start
        
        # Dictionary lookup (O(1))
        data_dict = {i: True for i in range(n)}
        
        start = time.perf_counter()
        result_dict = target in data_dict
        dict_lookup_time = time.perf_counter() - start
        
        # Set lookup (O(1))
        data_set = set(range(n))
        
        start = time.perf_counter()
        result_set = target in data_set
        set_lookup_time = time.perf_counter() - start
        
        return list_lookup_time, dict_lookup_time, set_lookup_time
    
    # Array vs list for numeric data
    def test_array_vs_list(n):
        import array
        
        # List with integers
        start = time.perf_counter()
        data_list = [i for i in range(n)]
        list_sum = sum(data_list)
        list_time = time.perf_counter() - start
        
        # Array with integers
        start = time.perf_counter()
        data_array = array.array('i', range(n))
        array_sum = sum(data_array)
        array_time = time.perf_counter() - start
        
        # Memory usage comparison
        list_size = sys.getsizeof(data_list) + sum(sys.getsizeof(i) for i in data_list[:100])
        array_size = sys.getsizeof(data_array)
        
        return list_time, array_time, list_size, array_size
    
    # Binary search optimization
    def test_search_algorithms(n):
        import bisect
        data = sorted(range(0, n * 2, 2))  # Even numbers
        target = n
        
        # Linear search
        start = time.perf_counter()
        linear_result = target in data
        linear_time = time.perf_counter() - start
        
        # Binary search
        start = time.perf_counter()
        binary_pos = bisect.bisect_left(data, target)
        binary_result = binary_pos < len(data) and data[binary_pos] == target
        binary_time = time.perf_counter() - start
        
        return linear_time, binary_time, linear_result, binary_result
    
    # Run tests
    n = 10000
    
    list_times = test_list_operations(n)
    deque_times = test_deque_operations(n)
    
    lookup_times = test_dict_vs_list_lookup(n)
    
    array_results = test_array_vs_list(n)
    
    search_results = test_search_algorithms(n)
    
    return {
        'list_vs_deque': {
            'list_times': [f"{t:.6f}s" for t in list_times],
            'deque_times': [f"{t:.6f}s" for t in deque_times],
            'deque_faster_insert': deque_times[2] < list_times[2]
        },
        'lookup_comparison': {
            'list_lookup': f"{lookup_times[0]:.6f}s",
            'dict_lookup': f"{lookup_times[1]:.6f}s",
            'set_lookup': f"{lookup_times[2]:.6f}s",
            'dict_speedup': lookup_times[0] / lookup_times[1] if lookup_times[1] > 0 else float('inf')
        },
        'array_vs_list': {
            'list_time': f"{array_results[0]:.6f}s",
            'array_time': f"{array_results[1]:.6f}s",
            'list_size': array_results[2],
            'array_size': array_results[3],
            'memory_savings': (array_results[2] - array_results[3]) / array_results[2]
        },
        'search_comparison': {
            'linear_time': f"{search_results[0]:.6f}s",
            'binary_time': f"{search_results[1]:.6f}s",
            'binary_speedup': search_results[0] / search_results[1] if search_results[1] > 0 else float('inf')
        }
    }

# Algorithm optimization
def algorithm_optimization():
    """Demonstrate algorithmic optimizations"""
    
    # Fibonacci: recursive vs memoized vs iterative
    def fibonacci_recursive(n):
        if n <= 1:
            return n
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)
    
    def fibonacci_memoized():
        cache = {}
        def fib(n):
            if n in cache:
                return cache[n]
            if n <= 1:
                cache[n] = n
            else:
                cache[n] = fib(n-1) + fib(n-2)
            return cache[n]
        return fib
    
    def fibonacci_iterative(n):
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    # Matrix multiplication optimization
    def matrix_multiply_naive(A, B):
        n = len(A)
        C = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i][j] += A[i][k] * B[k][j]
        return C
    
    def matrix_multiply_optimized(A, B):
        """Optimized matrix multiplication with better cache locality"""
        n = len(A)
        C = [[0] * n for _ in range(n)]
        # Reorder loops for better cache performance
        for i in range(n):
            for k in range(n):
                for j in range(n):
                    C[i][j] += A[i][k] * B[k][j]
        return C
    
    # String operations optimization
    def string_concat_naive(strings):
        result = ""
        for s in strings:
            result += s
        return result
    
    def string_concat_optimized(strings):
        return "".join(strings)
    
    # Sorting algorithm comparison
    def bubble_sort(arr):
        n = len(arr)
        arr = arr.copy()
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr
    
    def quick_sort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)
    
    # Run algorithm tests
    def time_function(func, *args):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        return end - start, result
    
    # Fibonacci comparison
    fib_memo = fibonacci_memoized()
    
    fib_recursive_time, fib_recursive_result = time_function(fibonacci_recursive, 20)
    fib_memoized_time, fib_memoized_result = time_function(fib_memo, 20)
    fib_iterative_time, fib_iterative_result = time_function(fibonacci_iterative, 20)
    
    # Matrix multiplication comparison
    size = 50
    matrix_a = [[i + j for j in range(size)] for i in range(size)]
    matrix_b = [[i * j + 1 for j in range(size)] for i in range(size)]
    
    naive_time, _ = time_function(matrix_multiply_naive, matrix_a, matrix_b)
    optimized_time, _ = time_function(matrix_multiply_optimized, matrix_a, matrix_b)
    
    # String concatenation comparison
    strings = [f"string_{i}" for i in range(1000)]
    
    naive_concat_time, _ = time_function(string_concat_naive, strings)
    optimized_concat_time, _ = time_function(string_concat_optimized, strings)
    
    # Sorting comparison
    import random
    data = [random.randint(1, 1000) for _ in range(500)]
    
    bubble_time, _ = time_function(bubble_sort, data)
    quick_time, _ = time_function(quick_sort, data)
    builtin_time, _ = time_function(sorted, data)
    
    return {
        'fibonacci_comparison': {
            'recursive_time': f"{fib_recursive_time:.6f}s",
            'memoized_time': f"{fib_memoized_time:.6f}s",
            'iterative_time': f"{fib_iterative_time:.6f}s",
            'memoized_speedup': fib_recursive_time / fib_memoized_time if fib_memoized_time > 0 else float('inf'),
            'results_equal': fib_recursive_result == fib_memoized_result == fib_iterative_result
        },
        'matrix_multiplication': {
            'naive_time': f"{naive_time:.6f}s",
            'optimized_time': f"{optimized_time:.6f}s",
            'speedup': naive_time / optimized_time if optimized_time > 0 else float('inf')
        },
        'string_concatenation': {
            'naive_time': f"{naive_concat_time:.6f}s",
            'optimized_time': f"{optimized_concat_time:.6f}s",
            'speedup': naive_concat_time / optimized_concat_time if optimized_concat_time > 0 else float('inf')
        },
        'sorting_comparison': {
            'bubble_sort_time': f"{bubble_time:.6f}s",
            'quick_sort_time': f"{quick_time:.6f}s",
            'builtin_sort_time': f"{builtin_time:.6f}s",
            'builtin_fastest': builtin_time < min(bubble_time, quick_time)
        }
    }

# Memory optimization strategies
def memory_optimization():
    """Demonstrate memory optimization techniques"""
    
    # Generator vs list comprehension
    def test_generator_memory():
        import sys
        
        # List comprehension (eager)
        list_comp = [i ** 2 for i in range(10000)]
        list_memory = sys.getsizeof(list_comp)
        
        # Generator expression (lazy)
        gen_exp = (i ** 2 for i in range(10000))
        gen_memory = sys.getsizeof(gen_exp)
        
        # Calculate first 100 items from both
        list_first_100 = list_comp[:100]
        gen_first_100 = [next(gen_exp) for _ in range(100)]
        
        return list_memory, gen_memory, len(list_first_100), len(gen_first_100)
    
    # __slots__ optimization
    class RegularClass:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
    
    class SlottedClass:
        __slots__ = ['x', 'y', 'z']
        
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
    
    def test_slots_memory():
        import sys
        
        # Create instances
        regular_obj = RegularClass(1, 2, 3)
        slotted_obj = SlottedClass(1, 2, 3)
        
        # Measure memory
        regular_size = sys.getsizeof(regular_obj) + sys.getsizeof(regular_obj.__dict__)
        slotted_size = sys.getsizeof(slotted_obj)
        
        # Test many instances
        regular_instances = [RegularClass(i, i+1, i+2) for i in range(1000)]
        slotted_instances = [SlottedClass(i, i+1, i+2) for i in range(1000)]
        
        return regular_size, slotted_size, len(regular_instances), len(slotted_instances)
    
    # String interning
    def test_string_interning():
        import sys
        
        # Without interning
        strings1 = ["hello_world"] * 1000
        strings2 = ["hello_world"] * 1000
        
        without_intern_same = strings1[0] is strings2[0]
        
        # With interning
        interned_string = sys.intern("hello_world_interned")
        strings3 = [sys.intern("hello_world_interned") for _ in range(1000)]
        strings4 = [sys.intern("hello_world_interned") for _ in range(1000)]
        
        with_intern_same = strings3[0] is strings4[0]
        
        return without_intern_same, with_intern_same
    
    # Weak references
    def test_weak_references():
        import weakref
        import gc
        
        class TestObject:
            def __init__(self, value):
                self.value = value
        
        # Strong references
        objects = [TestObject(i) for i in range(100)]
        strong_refs = objects[:]
        
        object_count_before = len(gc.get_objects())
        
        # Weak references
        weak_refs = [weakref.ref(obj) for obj in objects]
        
        # Delete strong references
        del objects
        del strong_refs
        gc.collect()
        
        # Check weak references
        alive_weak_refs = sum(1 for ref in weak_refs if ref() is not None)
        
        object_count_after = len(gc.get_objects())
        
        return object_count_before, object_count_after, alive_weak_refs
    
    # Run memory tests
    gen_memory_results = test_generator_memory()
    slots_memory_results = test_slots_memory()
    string_intern_results = test_string_interning()
    weak_ref_results = test_weak_references()
    
    return {
        'generator_vs_list': {
            'list_memory': gen_memory_results[0],
            'generator_memory': gen_memory_results[1],
            'memory_savings': (gen_memory_results[0] - gen_memory_results[1]) / gen_memory_results[0],
            'both_functional': gen_memory_results[2] == gen_memory_results[3] == 100
        },
        'slots_optimization': {
            'regular_size': slots_memory_results[0],
            'slotted_size': slots_memory_results[1],
            'memory_savings': (slots_memory_results[0] - slots_memory_results[1]) / slots_memory_results[0],
            'instances_created': slots_memory_results[2] == slots_memory_results[3] == 1000
        },
        'string_interning': {
            'without_interning_same': string_intern_results[0],
            'with_interning_same': string_intern_results[1],
            'interning_effective': string_intern_results[1] and not string_intern_results[0]
        },
        'weak_references': {
            'objects_before': weak_ref_results[0],
            'objects_after': weak_ref_results[1],
            'weak_refs_alive': weak_ref_results[2],
            'memory_freed': weak_ref_results[0] > weak_ref_results[1]
        }
    }

# Parallel processing optimization
def parallel_processing_demo():
    """Demonstrate parallel processing for performance"""
    
    def cpu_bound_task(n):
        """CPU-intensive task"""
        total = 0
        for i in range(n):
            total += math.sqrt(i) * math.sin(i) * math.cos(i)
        return total
    
    def io_bound_task(duration):
        """I/O simulation"""
        time.sleep(duration)
        return f"Task completed in {duration}s"
    
    # Sequential processing
    def sequential_cpu_tasks(tasks):
        start = time.perf_counter()
        results = [cpu_bound_task(task) for task in tasks]
        end = time.perf_counter()
        return results, end - start
    
    def sequential_io_tasks(tasks):
        start = time.perf_counter()
        results = [io_bound_task(task) for task in tasks]
        end = time.perf_counter()
        return results, end - start
    
    # Threading (good for I/O bound)
    def threaded_cpu_tasks(tasks):
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(cpu_bound_task, tasks))
        end = time.perf_counter()
        return results, end - start
    
    def threaded_io_tasks(tasks):
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(io_bound_task, tasks))
        end = time.perf_counter()
        return results, end - start
    
    # Multiprocessing (good for CPU bound)
    def multiprocess_cpu_tasks(tasks):
        start = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(cpu_bound_task, tasks))
        end = time.perf_counter()
        return results, end - start
    
    # Test data
    cpu_tasks = [50000] * 4
    io_tasks = [0.1] * 4
    
    # Run tests
    seq_cpu_results, seq_cpu_time = sequential_cpu_tasks(cpu_tasks)
    thread_cpu_results, thread_cpu_time = threaded_cpu_tasks(cpu_tasks)
    process_cpu_results, process_cpu_time = multiprocess_cpu_tasks(cpu_tasks)
    
    seq_io_results, seq_io_time = sequential_io_tasks(io_tasks)
    thread_io_results, thread_io_time = threaded_io_tasks(io_tasks)
    
    return {
        'cpu_bound_performance': {
            'sequential_time': f"{seq_cpu_time:.4f}s",
            'threading_time': f"{thread_cpu_time:.4f}s",
            'multiprocessing_time': f"{process_cpu_time:.4f}s",
            'threading_speedup': seq_cpu_time / thread_cpu_time,
            'multiprocessing_speedup': seq_cpu_time / process_cpu_time,
            'multiprocessing_fastest': process_cpu_time < min(seq_cpu_time, thread_cpu_time)
        },
        'io_bound_performance': {
            'sequential_time': f"{seq_io_time:.4f}s",
            'threading_time': f"{thread_io_time:.4f}s",
            'threading_speedup': seq_io_time / thread_io_time,
            'threading_effective': thread_io_time < seq_io_time * 0.5
        }
    }

# Caching and memoization
def caching_optimization():
    """Demonstrate caching strategies for performance"""
    
    # Simple memoization
    def memoize(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args):
            if args in cache:
                return cache[args]
            result = func(*args)
            cache[args] = result
            return result
        
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        return wrapper
    
    # LRU Cache
    @functools.lru_cache(maxsize=128)
    def expensive_function_lru(n):
        """Expensive function with LRU cache"""
        time.sleep(0.01)  # Simulate expensive computation
        return sum(math.sqrt(i) for i in range(n))
    
    # Custom memoization
    @memoize
    def expensive_function_custom(n):
        """Expensive function with custom memoization"""
        time.sleep(0.01)  # Simulate expensive computation
        return sum(math.sqrt(i) for i in range(n))
    
    # Cache warming
    def warm_cache():
        """Pre-populate cache with common values"""
        for i in range(1, 11):
            expensive_function_lru(i * 100)
            expensive_function_custom(i * 100)
    
    # Property caching
    class CachedPropertyExample:
        def __init__(self, data):
            self.data = data
            self._expensive_computation = None
        
        @property
        def expensive_computation(self):
            if self._expensive_computation is None:
                print("Computing expensive property...")
                time.sleep(0.1)
                self._expensive_computation = sum(x ** 2 for x in self.data)
            return self._expensive_computation
    
    # Test caching performance
    def test_cache_performance():
        # Without cache
        def expensive_no_cache(n):
            time.sleep(0.01)
            return sum(math.sqrt(i) for i in range(n))
        
        # Test repeated calls
        test_values = [100, 200, 100, 300, 200, 100]
        
        # No cache
        start = time.perf_counter()
        no_cache_results = [expensive_no_cache(n) for n in test_values]
        no_cache_time = time.perf_counter() - start
        
        # LRU cache
        start = time.perf_counter()
        lru_cache_results = [expensive_function_lru(n) for n in test_values]
        lru_cache_time = time.perf_counter() - start
        
        # Custom cache
        start = time.perf_counter()
        custom_cache_results = [expensive_function_custom(n) for n in test_values]
        custom_cache_time = time.perf_counter() - start
        
        return no_cache_time, lru_cache_time, custom_cache_time
    
    # Class-level caching
    class DataProcessor:
        def __init__(self):
            self.cache = {}
        
        def process_data(self, data_id, expensive_param):
            cache_key = (data_id, expensive_param)
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Simulate expensive processing
            time.sleep(0.05)
            result = data_id * expensive_param + sum(range(expensive_param))
            
            self.cache[cache_key] = result
            return result
        
        def clear_cache(self):
            self.cache.clear()
    
    # Run caching tests
    warm_cache()
    
    cache_performance = test_cache_performance()
    
    # Test property caching
    cached_obj = CachedPropertyExample([1, 2, 3, 4, 5])
    
    start = time.perf_counter()
    first_access = cached_obj.expensive_computation
    first_time = time.perf_counter() - start
    
    start = time.perf_counter()
    second_access = cached_obj.expensive_computation
    second_time = time.perf_counter() - start
    
    # Test class-level caching
    processor = DataProcessor()
    
    start = time.perf_counter()
    result1 = processor.process_data(1, 100)
    result2 = processor.process_data(1, 100)  # Should be cached
    class_cache_time = time.perf_counter() - start
    
    # Get cache statistics
    lru_info = expensive_function_lru.cache_info()
    custom_cache_size = len(expensive_function_custom.cache)
    
    return {
        'cache_performance': {
            'no_cache_time': f"{cache_performance[0]:.4f}s",
            'lru_cache_time': f"{cache_performance[1]:.4f}s",
            'custom_cache_time': f"{cache_performance[2]:.4f}s",
            'lru_speedup': cache_performance[0] / cache_performance[1],
            'custom_speedup': cache_performance[0] / cache_performance[2]
        },
        'property_caching': {
            'first_access_time': f"{first_time:.4f}s",
            'second_access_time': f"{second_time:.4f}s",
            'caching_effective': second_time < first_time * 0.1,
            'results_equal': first_access == second_access
        },
        'cache_statistics': {
            'lru_cache_info': lru_info._asdict(),
            'custom_cache_size': custom_cache_size,
            'class_cache_time': f"{class_cache_time:.4f}s"
        }
    }

# Compiler optimizations and bytecode
def compiler_optimization_demo():
    """Demonstrate compiler-level optimizations"""
    
    # Constant folding
    def test_constant_folding():
        import dis
        
        def with_constants():
            return 2 + 3 * 4
        
        def with_variables():
            a, b, c = 2, 3, 4
            return a + b * c
        
        # Get bytecode
        constants_bytecode = list(dis.get_instructions(with_constants))
        variables_bytecode = list(dis.get_instructions(with_variables))
        
        return len(constants_bytecode), len(variables_bytecode)
    
    # Loop optimization
    def test_loop_optimization():
        # Inefficient loop
        def inefficient_loop(n):
            result = []
            for i in range(n):
                result.append(i ** 2)
            return result
        
        # List comprehension (optimized)
        def optimized_loop(n):
            return [i ** 2 for i in range(n)]
        
        # Generator (memory efficient)
        def generator_loop(n):
            return (i ** 2 for i in range(n))
        
        n = 10000
        
        # Time each approach
        inefficient_time = timeit.timeit(lambda: inefficient_loop(n), number=10)
        optimized_time = timeit.timeit(lambda: optimized_loop(n), number=10)
        
        # For generator, time creation only
        generator_time = timeit.timeit(lambda: generator_loop(n), number=10)
        
        return inefficient_time, optimized_time, generator_time
    
    # Function call optimization
    def test_function_call_overhead():
        def simple_function(x):
            return x + 1
        
        def inline_operation():
            x = 10
            return x + 1
        
        # Test function call vs inline
        function_time = timeit.timeit(lambda: simple_function(10), number=100000)
        inline_time = timeit.timeit(inline_operation, number=100000)
        
        return function_time, inline_time
    
    # String optimization
    def test_string_optimization():
        # String concatenation methods
        def plus_concatenation(strings):
            result = ""
            for s in strings:
                result = result + s
            return result
        
        def join_concatenation(strings):
            return "".join(strings)
        
        def format_concatenation(strings):
            return "{}{}{}{}{}".format(*strings[:5])
        
        def f_string_concatenation(strings):
            if len(strings) >= 5:
                return f"{strings[0]}{strings[1]}{strings[2]}{strings[3]}{strings[4]}"
            return "".join(strings)
        
        strings = ["hello", "world", "python", "optimization", "test"]
        
        plus_time = timeit.timeit(lambda: plus_concatenation(strings), number=10000)
        join_time = timeit.timeit(lambda: join_concatenation(strings), number=10000)
        format_time = timeit.timeit(lambda: format_concatenation(strings), number=10000)
        f_string_time = timeit.timeit(lambda: f_string_concatenation(strings), number=10000)
        
        return plus_time, join_time, format_time, f_string_time
    
    # Run optimization tests
    constant_results = test_constant_folding()
    loop_results = test_loop_optimization()
    function_results = test_function_call_overhead()
    string_results = test_string_optimization()
    
    return {
        'constant_folding': {
            'constants_instructions': constant_results[0],
            'variables_instructions': constant_results[1],
            'optimization_effective': constant_results[0] < constant_results[1]
        },
        'loop_optimization': {
            'inefficient_time': f"{loop_results[0]:.6f}s",
            'list_comp_time': f"{loop_results[1]:.6f}s",
            'generator_time': f"{loop_results[2]:.6f}s",
            'list_comp_speedup': loop_results[0] / loop_results[1],
            'generator_fastest_creation': loop_results[2] < min(loop_results[0], loop_results[1])
        },
        'function_call_overhead': {
            'function_time': f"{function_results[0]:.6f}s",
            'inline_time': f"{function_results[1]:.6f}s",
            'overhead_factor': function_results[0] / function_results[1]
        },
        'string_optimization': {
            'plus_time': f"{string_results[0]:.6f}s",
            'join_time': f"{string_results[1]:.6f}s",
            'format_time': f"{string_results[2]:.6f}s",
            'f_string_time': f"{string_results[3]:.6f}s",
            'fastest_method': ['plus', 'join', 'format', 'f_string'][string_results.index(min(string_results))]
        }
    }

# Comprehensive performance testing
def run_all_optimization_demos():
    """Execute all performance optimization demonstrations"""
    demo_functions = [
        ('profiling', profiling_demo),
        ('data_structures', data_structure_optimization),
        ('algorithms', algorithm_optimization),
        ('memory', memory_optimization),
        ('parallel_processing', parallel_processing_demo),
        ('caching', caching_optimization),
        ('compiler_optimization', compiler_optimization_demo)
    ]
    
    results = {}
    for name, func in demo_functions:
        try:
            result = func()
            results[name] = result
        except Exception as e:
            results[name] = {'error': str(e)}
    
    # Add system performance info
    import psutil
    results['system_performance'] = {
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'python_version': sys.version,
        'recursion_limit': sys.getrecursionlimit()
    }
    
    return results

if __name__ == "__main__":
    print("=== Python Performance Optimization Demo ===")
    
    # Run all demonstrations
    all_results = run_all_optimization_demos()
    
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
    
    print("\n=== PROFILING TOOLS ===")
    
    tools = {
        "cProfile": "Statistical profiler for production use",
        "profile": "Pure Python profiler (slower but more detailed)",
        "timeit": "Micro-benchmarking for small code snippets",
        "time.perf_counter()": "High-resolution timing for manual benchmarks",
        "memory_profiler": "Line-by-line memory usage profiling",
        "py-spy": "Sampling profiler for production systems",
        "pstats": "Statistics analysis for profiling results",
        "tracemalloc": "Memory allocation tracking"
    }
    
    for tool, description in tools.items():
        print(f"  {tool}: {description}")
    
    print("\n=== OPTIMIZATION STRATEGIES ===")
    
    strategies = {
        "Algorithm Choice": "Choose O(log n) over O(n) algorithms when possible",
        "Data Structure Selection": "Use appropriate data structures for operations",
        "Memory Management": "Use generators, __slots__, and weak references",
        "Caching and Memoization": "Cache expensive computations",
        "Parallel Processing": "Use threading for I/O, multiprocessing for CPU",
        "Compiler Optimizations": "Leverage built-in optimizations",
        "String Operations": "Use join() instead of concatenation",
        "Loop Optimization": "Use list comprehensions and built-in functions",
        "Function Call Overhead": "Minimize function calls in tight loops",
        "Lazy Evaluation": "Use generators for large datasets"
    }
    
    for strategy, description in strategies.items():
        print(f"  {strategy}: {description}")
    
    print("\n=== PERFORMANCE BEST PRACTICES ===")
    
    best_practices = [
        "Profile before optimizing - measure, don't guess",
        "Focus on algorithmic improvements first",
        "Use appropriate data structures for your use case",
        "Leverage built-in functions and libraries",
        "Use list comprehensions instead of explicit loops",
        "Minimize function call overhead in tight loops",
        "Use generators for memory-efficient iteration",
        "Cache expensive computations with @lru_cache",
        "Use multiprocessing for CPU-bound tasks",
        "Use threading for I/O-bound tasks",
        "Optimize string operations with join()",
        "Use __slots__ for classes with many instances",
        "Consider NumPy for numerical computations",
        "Use C extensions for performance-critical code",
        "Test performance changes with real data"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== Performance Optimization Complete! ===")
    print("  Advanced profiling and optimization techniques mastered")
