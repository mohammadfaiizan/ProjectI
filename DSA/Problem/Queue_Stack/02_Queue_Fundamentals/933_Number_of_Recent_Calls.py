"""
933. Number of Recent Calls - Multiple Approaches
Difficulty: Easy

You have a RecentCounter class which counts the number of recent requests within a certain time frame.

Implement the RecentCounter class:
- RecentCounter() Initializes the counter with zero recent requests.
- int ping(int t) Adds a new request at time t, where t represents some time in milliseconds, and returns the number of requests that has happened in the past 3000 milliseconds (including the new request). Specifically, return the number of requests that have happened in the inclusive range [t - 3000, t].

It is guaranteed that every call to ping uses a strictly larger value of t than the previous call.
"""

from typing import List
from collections import deque
import bisect

class RecentCounter1:
    """
    Approach 1: Deque Implementation
    
    Use deque to maintain sliding window of requests.
    
    Time: O(1) amortized for ping(), Space: O(n) where n is requests in window
    """
    
    def __init__(self):
        self.requests = deque()
    
    def ping(self, t: int) -> int:
        # Add current request
        self.requests.append(t)
        
        # Remove requests older than 3000ms
        while self.requests and self.requests[0] < t - 3000:
            self.requests.popleft()
        
        return len(self.requests)


class RecentCounter2:
    """
    Approach 2: List Implementation
    
    Use list to store requests and filter old ones.
    
    Time: O(n) for ping(), Space: O(n)
    """
    
    def __init__(self):
        self.requests = []
    
    def ping(self, t: int) -> int:
        # Add current request
        self.requests.append(t)
        
        # Count requests in valid time window
        count = 0
        for request_time in self.requests:
            if request_time >= t - 3000:
                count += 1
        
        return count


class RecentCounter3:
    """
    Approach 3: List with Cleanup
    
    Use list but periodically clean up old requests.
    
    Time: O(1) amortized for ping(), Space: O(n)
    """
    
    def __init__(self):
        self.requests = []
    
    def ping(self, t: int) -> int:
        # Add current request
        self.requests.append(t)
        
        # Remove old requests (cleanup)
        while self.requests and self.requests[0] < t - 3000:
            self.requests.pop(0)  # O(n) operation
        
        return len(self.requests)


class RecentCounter4:
    """
    Approach 4: Binary Search Implementation
    
    Use binary search to find valid range efficiently.
    
    Time: O(log n) for ping(), Space: O(n)
    """
    
    def __init__(self):
        self.requests = []
    
    def ping(self, t: int) -> int:
        # Add current request
        self.requests.append(t)
        
        # Find the leftmost position where time >= t - 3000
        left_bound = t - 3000
        left_idx = bisect.bisect_left(self.requests, left_bound)
        
        # Count requests from left_idx to end
        return len(self.requests) - left_idx


class RecentCounter5:
    """
    Approach 5: Circular Buffer Implementation
    
    Use circular buffer with fixed size for memory efficiency.
    
    Time: O(1) for ping(), Space: O(k) where k is buffer size
    """
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.buffer = [0] * buffer_size
        self.head = 0
        self.size = 0
    
    def ping(self, t: int) -> int:
        # Add new request to circular buffer
        self.buffer[self.head] = t
        self.head = (self.head + 1) % self.buffer_size
        
        if self.size < self.buffer_size:
            self.size += 1
        
        # Count valid requests in buffer
        count = 0
        for i in range(self.size):
            idx = (self.head - 1 - i) % self.buffer_size
            if self.buffer[idx] >= t - 3000:
                count += 1
            else:
                break  # Since requests are in chronological order
        
        return count


class RecentCounter6:
    """
    Approach 6: Two Pointers Implementation
    
    Use two pointers to maintain valid window.
    
    Time: O(1) amortized for ping(), Space: O(n)
    """
    
    def __init__(self):
        self.requests = []
        self.left = 0  # Points to first valid request
    
    def ping(self, t: int) -> int:
        # Add current request
        self.requests.append(t)
        
        # Move left pointer to maintain valid window
        while self.left < len(self.requests) and self.requests[self.left] < t - 3000:
            self.left += 1
        
        # Return count of valid requests
        return len(self.requests) - self.left


class RecentCounter7:
    """
    Approach 7: Optimized Deque with Size Limit
    
    Use deque with intelligent size management.
    
    Time: O(1) amortized for ping(), Space: O(min(n, 3000))
    """
    
    def __init__(self):
        self.requests = deque()
        self.max_requests = 3000  # Theoretical maximum in 3000ms window
    
    def ping(self, t: int) -> int:
        # Add current request
        self.requests.append(t)
        
        # Remove requests older than 3000ms
        while self.requests and self.requests[0] < t - 3000:
            self.requests.popleft()
        
        # Optional: limit deque size for memory efficiency
        if len(self.requests) > self.max_requests:
            # This shouldn't happen given the problem constraints
            self.requests.popleft()
        
        return len(self.requests)


def test_recent_counter_implementations():
    """Test all recent counter implementations"""
    
    implementations = [
        ("Deque Implementation", RecentCounter1),
        ("List Implementation", RecentCounter2),
        ("List with Cleanup", RecentCounter3),
        ("Binary Search", RecentCounter4),
        ("Circular Buffer", RecentCounter5),
        ("Two Pointers", RecentCounter6),
        ("Optimized Deque", RecentCounter7),
    ]
    
    test_cases = [
        ([1, 100, 3001, 3002], [1, 2, 3, 3], "Basic test case"),
        ([1], [1], "Single request"),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], "All within window"),
        ([1, 1000, 2000, 3000, 4000, 5000], [1, 2, 3, 4, 3, 2], "Sliding window"),
        ([1, 3001, 6001, 9001], [1, 1, 1, 1], "Sparse requests"),
    ]
    
    print("=== Testing Recent Counter Implementations ===")
    
    for requests, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Requests: {requests}")
        print(f"Expected: {expected}")
        
        for impl_name, impl_class in implementations:
            try:
                counter = impl_class()
                results = []
                
                for request_time in requests:
                    count = counter.ping(request_time)
                    results.append(count)
                
                status = "✓" if results == expected else "✗"
                print(f"{impl_name:20} | {status} | Results: {results}")
            
            except Exception as e:
                print(f"{impl_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_sliding_window_behavior():
    """Demonstrate sliding window behavior step by step"""
    print("\n=== Sliding Window Behavior Demonstration ===")
    
    counter = RecentCounter1()
    requests = [1, 100, 3001, 3002, 6000, 6001]
    
    print("Time window: 3000ms")
    print("Processing requests:", requests)
    
    for i, t in enumerate(requests):
        count = counter.ping(t)
        window_start = t - 3000
        window_end = t
        valid_requests = [req for req in list(counter.requests)]
        
        print(f"Step {i+1}: ping({t})")
        print(f"  Window: [{window_start}, {window_end}]")
        print(f"  Valid requests: {valid_requests}")
        print(f"  Count: {count}")
        print()


def benchmark_recent_counter():
    """Benchmark different recent counter implementations"""
    import time
    import random
    
    implementations = [
        ("Deque Implementation", RecentCounter1),
        ("Binary Search", RecentCounter4),
        ("Two Pointers", RecentCounter6),
        ("Optimized Deque", RecentCounter7),
    ]
    
    # Generate test data - increasing timestamps
    n_requests = 10000
    requests = []
    current_time = 1
    
    for _ in range(n_requests):
        current_time += random.randint(1, 100)  # Random intervals
        requests.append(current_time)
    
    print("\n=== Recent Counter Performance Benchmark ===")
    print(f"Number of requests: {n_requests}")
    
    for impl_name, impl_class in implementations:
        counter = impl_class()
        
        start_time = time.time()
        
        for request_time in requests:
            counter.ping(request_time)
        
        end_time = time.time()
        
        print(f"{impl_name:20} | Time: {end_time - start_time:.4f}s")


def test_edge_cases():
    """Test edge cases for recent counter"""
    print("\n=== Testing Edge Cases ===")
    
    counter = RecentCounter1()
    
    edge_cases = [
        ("First request", 1, 1),
        ("Request at boundary", 3001, 2),
        ("Request just outside", 3002, 2),
        ("Large time gap", 10000, 1),
        ("Multiple rapid requests", 10001, 2),
        ("Another rapid request", 10002, 3),
    ]
    
    for description, request_time, expected in edge_cases:
        result = counter.ping(request_time)
        status = "✓" if result == expected else "✗"
        print(f"{description:25} | {status} | ping({request_time}) = {result}")


def analyze_memory_usage():
    """Analyze memory usage patterns"""
    print("\n=== Memory Usage Analysis ===")
    
    import sys
    
    implementations = [
        ("Deque Implementation", RecentCounter1),
        ("List Implementation", RecentCounter2),
        ("Circular Buffer", RecentCounter5),
        ("Two Pointers", RecentCounter6),
    ]
    
    # Simulate requests over time
    requests = list(range(1, 5001, 10))  # 500 requests
    
    for impl_name, impl_class in implementations:
        counter = impl_class()
        
        # Process all requests
        for request_time in requests:
            counter.ping(request_time)
        
        # Estimate memory usage
        memory_size = sys.getsizeof(counter)
        
        if hasattr(counter, 'requests'):
            memory_size += sys.getsizeof(counter.requests)
        if hasattr(counter, 'buffer'):
            memory_size += sys.getsizeof(counter.buffer)
        
        print(f"{impl_name:20} | Memory: ~{memory_size} bytes")


def test_window_boundary_behavior():
    """Test behavior at window boundaries"""
    print("\n=== Window Boundary Behavior Test ===")
    
    counter = RecentCounter1()
    
    # Test requests at exact boundaries
    boundary_tests = [
        (1000, "Base request"),
        (4000, "Exactly 3000ms later"),
        (4001, "Just outside window"),
        (7000, "New base"),
        (7001, "Within new window"),
        (10000, "Another boundary"),
        (10001, "Just within"),
    ]
    
    for request_time, description in boundary_tests:
        count = counter.ping(request_time)
        window_requests = list(counter.requests)
        print(f"ping({request_time:5d}) | {description:20} | Count: {count} | Window: {window_requests}")


def compare_accuracy():
    """Compare accuracy of different implementations"""
    print("\n=== Accuracy Comparison ===")
    
    implementations = [
        ("Deque Implementation", RecentCounter1),
        ("Binary Search", RecentCounter4),
        ("Two Pointers", RecentCounter6),
    ]
    
    # Complex test case
    requests = [1, 500, 1500, 2500, 3500, 4000, 4500, 5000, 7000, 8000]
    
    print(f"Test requests: {requests}")
    
    all_results = {}
    
    for impl_name, impl_class in implementations:
        counter = impl_class()
        results = []
        
        for request_time in requests:
            count = counter.ping(request_time)
            results.append(count)
        
        all_results[impl_name] = results
        print(f"{impl_name:20} | Results: {results}")
    
    # Check if all implementations give same results
    first_results = list(all_results.values())[0]
    all_same = all(results == first_results for results in all_results.values())
    
    print(f"\nAll implementations agree: {'✓' if all_same else '✗'}")


def stress_test():
    """Stress test with many requests"""
    print("\n=== Stress Test ===")
    
    counter = RecentCounter1()
    
    # Generate many requests in a short time window
    base_time = 10000
    n_requests = 1000
    
    print(f"Processing {n_requests} requests in 3000ms window...")
    
    for i in range(n_requests):
        request_time = base_time + i * 3  # 3ms intervals
        count = counter.ping(request_time)
        
        if i % 100 == 0:  # Print every 100th request
            print(f"Request {i+1}: ping({request_time}) = {count}")
    
    final_count = len(counter.requests)
    print(f"Final count in window: {final_count}")


if __name__ == "__main__":
    test_recent_counter_implementations()
    demonstrate_sliding_window_behavior()
    test_edge_cases()
    test_window_boundary_behavior()
    compare_accuracy()
    benchmark_recent_counter()
    analyze_memory_usage()
    stress_test()

"""
Number of Recent Calls demonstrates multiple approaches for maintaining
sliding time windows including deque-based, binary search, two pointers,
and circular buffer implementations with comprehensive testing and analysis.
"""
