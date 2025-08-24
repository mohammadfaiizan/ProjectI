"""
346. Moving Average from Data Stream - Multiple Approaches
Difficulty: Easy

Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.

Implement the MovingAverage class:
- MovingAverage(int size) Initializes the object with the size of the window size.
- double next(int val) Returns the moving average of the last size values of the stream.
"""

from typing import List
from collections import deque

class MovingAverage1:
    """
    Approach 1: Circular Array Implementation
    
    Use circular array to maintain sliding window.
    
    Time: O(1) for next(), Space: O(size)
    """
    
    def __init__(self, size: int):
        self.size = size
        self.queue = [0] * size
        self.head = 0
        self.window_sum = 0
        self.count = 0
    
    def next(self, val: int) -> float:
        # Calculate the tail position
        tail = (self.head + 1) % self.size
        
        # Add new value
        self.window_sum += val
        
        # If window is full, subtract the oldest value
        if self.count == self.size:
            self.window_sum -= self.queue[tail]
        else:
            self.count += 1
        
        # Move head pointer and store new value
        self.head = tail
        self.queue[self.head] = val
        
        return self.window_sum / self.count


class MovingAverage2:
    """
    Approach 2: Deque Implementation
    
    Use collections.deque for sliding window.
    
    Time: O(1) for next(), Space: O(size)
    """
    
    def __init__(self, size: int):
        self.size = size
        self.queue = deque()
        self.window_sum = 0
    
    def next(self, val: int) -> float:
        self.queue.append(val)
        self.window_sum += val
        
        # Remove oldest element if window exceeds size
        if len(self.queue) > self.size:
            removed = self.queue.popleft()
            self.window_sum -= removed
        
        return self.window_sum / len(self.queue)


class MovingAverage3:
    """
    Approach 3: List Implementation
    
    Use Python list to maintain sliding window.
    
    Time: O(1) amortized for next(), Space: O(size)
    """
    
    def __init__(self, size: int):
        self.size = size
        self.queue = []
        self.window_sum = 0
    
    def next(self, val: int) -> float:
        self.queue.append(val)
        self.window_sum += val
        
        # Remove oldest element if window exceeds size
        if len(self.queue) > self.size:
            removed = self.queue.pop(0)  # O(n) operation
            self.window_sum -= removed
        
        return self.window_sum / len(self.queue)


class MovingAverage4:
    """
    Approach 4: Optimized List with Index Tracking
    
    Use list with index tracking to avoid expensive pop(0).
    
    Time: O(1) for next(), Space: O(size)
    """
    
    def __init__(self, size: int):
        self.size = size
        self.values = []
        self.start_idx = 0
        self.window_sum = 0
    
    def next(self, val: int) -> float:
        self.values.append(val)
        self.window_sum += val
        
        # Calculate current window size
        current_size = len(self.values) - self.start_idx
        
        # If window exceeds size, move start index
        if current_size > self.size:
            self.window_sum -= self.values[self.start_idx]
            self.start_idx += 1
            current_size -= 1
        
        return self.window_sum / current_size


class MovingAverage5:
    """
    Approach 5: Two Pointers Implementation
    
    Use two pointers to track window boundaries.
    
    Time: O(1) for next(), Space: O(n) where n is total elements
    """
    
    def __init__(self, size: int):
        self.size = size
        self.values = []
        self.window_sum = 0
    
    def next(self, val: int) -> float:
        self.values.append(val)
        self.window_sum += val
        
        # Calculate window boundaries
        window_start = max(0, len(self.values) - self.size)
        window_end = len(self.values)
        
        # Recalculate sum if needed (for first few elements)
        if len(self.values) <= self.size:
            current_sum = sum(self.values[window_start:window_end])
        else:
            # Remove the element that's no longer in window
            removed_idx = len(self.values) - self.size - 1
            self.window_sum -= self.values[removed_idx]
            current_sum = self.window_sum
        
        window_size = window_end - window_start
        return current_sum / window_size


class MovingAverage6:
    """
    Approach 6: Streaming Statistics Implementation
    
    Maintain streaming statistics with efficient updates.
    
    Time: O(1) for next(), Space: O(size)
    """
    
    def __init__(self, size: int):
        self.size = size
        self.buffer = [0] * size
        self.head = 0
        self.count = 0
        self.sum = 0.0
    
    def next(self, val: int) -> float:
        # If buffer is full, subtract the value being overwritten
        if self.count == self.size:
            self.sum -= self.buffer[self.head]
        else:
            self.count += 1
        
        # Add new value
        self.buffer[self.head] = val
        self.sum += val
        
        # Move head pointer
        self.head = (self.head + 1) % self.size
        
        return self.sum / self.count


class MovingAverage7:
    """
    Approach 7: Memory-Efficient Implementation
    
    Minimize memory usage while maintaining performance.
    
    Time: O(1) for next(), Space: O(size)
    """
    
    def __init__(self, size: int):
        self.size = size
        self.window = deque(maxlen=size)
        self.sum = 0
    
    def next(self, val: int) -> float:
        # If deque is at max capacity, the oldest element will be automatically removed
        if len(self.window) == self.size:
            self.sum -= self.window[0]
        
        self.window.append(val)
        self.sum += val
        
        return self.sum / len(self.window)


def test_moving_average_implementations():
    """Test all moving average implementations"""
    
    implementations = [
        ("Circular Array", MovingAverage1),
        ("Deque Implementation", MovingAverage2),
        ("List Implementation", MovingAverage3),
        ("Optimized List", MovingAverage4),
        ("Two Pointers", MovingAverage5),
        ("Streaming Statistics", MovingAverage6),
        ("Memory Efficient", MovingAverage7),
    ]
    
    test_cases = [
        (3, [1, 10, 3, 5], [1.0, 5.5, 4.666666666666667, 6.0], "Basic test"),
        (1, [1, 2, 3], [1.0, 2.0, 3.0], "Window size 1"),
        (5, [1, 2], [1.0, 1.5], "Fewer elements than window"),
        (2, [1, 2, 3, 4, 5], [1.0, 1.5, 2.5, 3.5, 4.5], "Sliding window"),
    ]
    
    print("=== Testing Moving Average Implementations ===")
    
    for window_size, values, expected, description in test_cases:
        print(f"\n--- {description} (Window size: {window_size}) ---")
        print(f"Values: {values}")
        print(f"Expected: {expected}")
        
        for impl_name, impl_class in implementations:
            try:
                ma = impl_class(window_size)
                results = []
                
                for val in values:
                    avg = ma.next(val)
                    results.append(avg)
                
                # Check if results match expected (with floating point tolerance)
                matches = all(abs(r - e) < 1e-10 for r, e in zip(results, expected))
                status = "✓" if matches else "✗"
                
                print(f"{impl_name:20} | {status} | Results: {[round(r, 2) for r in results]}")
            
            except Exception as e:
                print(f"{impl_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_sliding_window():
    """Demonstrate sliding window behavior step by step"""
    print("\n=== Sliding Window Demonstration ===")
    
    ma = MovingAverage2(3)  # Window size 3
    values = [1, 10, 3, 5, 8, 2]
    
    print("Window size: 3")
    print("Processing values:", values)
    
    for i, val in enumerate(values):
        avg = ma.next(val)
        window_values = list(ma.queue)
        print(f"Step {i+1}: next({val}) -> window: {window_values}, average: {avg:.2f}")


def benchmark_moving_average():
    """Benchmark different moving average implementations"""
    import time
    import random
    
    implementations = [
        ("Circular Array", MovingAverage1),
        ("Deque Implementation", MovingAverage2),
        ("Optimized List", MovingAverage4),
        ("Streaming Statistics", MovingAverage6),
        ("Memory Efficient", MovingAverage7),
    ]
    
    window_size = 100
    n_operations = 10000
    
    print("\n=== Moving Average Performance Benchmark ===")
    print(f"Window size: {window_size}, Operations: {n_operations}")
    
    # Generate test data
    test_values = [random.randint(1, 1000) for _ in range(n_operations)]
    
    for impl_name, impl_class in implementations:
        ma = impl_class(window_size)
        
        start_time = time.time()
        
        for val in test_values:
            ma.next(val)
        
        end_time = time.time()
        
        print(f"{impl_name:20} | Time: {end_time - start_time:.4f}s")


def test_edge_cases():
    """Test edge cases for moving average"""
    print("\n=== Testing Edge Cases ===")
    
    ma = MovingAverage2(3)
    
    edge_cases = [
        ("First element", 5, 5.0),
        ("Second element", 10, 7.5),
        ("Third element", 15, 10.0),
        ("Fourth element (sliding)", 20, 15.0),
        ("Negative number", -5, 10.0),
        ("Zero", 0, 5.0),
    ]
    
    for description, value, expected in edge_cases:
        result = ma.next(value)
        status = "✓" if abs(result - expected) < 1e-10 else "✗"
        print(f"{description:25} | {status} | next({value}) = {result:.2f}")


def memory_usage_analysis():
    """Analyze memory usage of different implementations"""
    print("\n=== Memory Usage Analysis ===")
    
    import sys
    
    window_size = 1000
    
    implementations = [
        ("Circular Array", MovingAverage1),
        ("Deque Implementation", MovingAverage2),
        ("List Implementation", MovingAverage3),
        ("Memory Efficient", MovingAverage7),
    ]
    
    for impl_name, impl_class in implementations:
        ma = impl_class(window_size)
        
        # Add some elements
        for i in range(window_size):
            ma.next(i)
        
        # Estimate memory usage
        memory_size = sys.getsizeof(ma)
        
        # Add size of internal data structures
        if hasattr(ma, 'queue'):
            memory_size += sys.getsizeof(ma.queue)
        if hasattr(ma, 'buffer'):
            memory_size += sys.getsizeof(ma.buffer)
        if hasattr(ma, 'values'):
            memory_size += sys.getsizeof(ma.values)
        if hasattr(ma, 'window'):
            memory_size += sys.getsizeof(ma.window)
        
        print(f"{impl_name:20} | Memory: ~{memory_size} bytes")


def test_different_window_sizes():
    """Test behavior with different window sizes"""
    print("\n=== Testing Different Window Sizes ===")
    
    test_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    window_sizes = [1, 3, 5, 10, 15]
    
    for window_size in window_sizes:
        print(f"\nWindow size: {window_size}")
        ma = MovingAverage2(window_size)
        
        results = []
        for val in test_values:
            avg = ma.next(val)
            results.append(avg)
        
        print(f"Values: {test_values}")
        print(f"Averages: {[round(r, 2) for r in results]}")


def compare_accuracy():
    """Compare accuracy of different implementations"""
    print("\n=== Accuracy Comparison ===")
    
    implementations = [
        ("Circular Array", MovingAverage1),
        ("Deque Implementation", MovingAverage2),
        ("Streaming Statistics", MovingAverage6),
    ]
    
    window_size = 5
    test_values = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]
    
    print(f"Window size: {window_size}")
    print(f"Test values: {test_values}")
    
    all_results = {}
    
    for impl_name, impl_class in implementations:
        ma = impl_class(window_size)
        results = []
        
        for val in test_values:
            avg = ma.next(val)
            results.append(avg)
        
        all_results[impl_name] = results
        print(f"{impl_name:20} | Results: {[round(r, 6) for r in results]}")
    
    # Check if all implementations give same results
    first_results = list(all_results.values())[0]
    all_same = all(
        all(abs(r1 - r2) < 1e-10 for r1, r2 in zip(first_results, results))
        for results in all_results.values()
    )
    
    print(f"\nAll implementations agree: {'✓' if all_same else '✗'}")


if __name__ == "__main__":
    test_moving_average_implementations()
    demonstrate_sliding_window()
    test_edge_cases()
    test_different_window_sizes()
    compare_accuracy()
    benchmark_moving_average()
    memory_usage_analysis()

"""
Moving Average from Data Stream demonstrates multiple approaches
for maintaining sliding window statistics including circular arrays,
deques, optimized lists, and streaming algorithms with performance analysis.
"""
