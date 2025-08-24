"""
362. Design Hit Counter - Multiple Approaches
Difficulty: Medium

Design a hit counter which counts the number of hits received in the past 5 minutes (i.e., the past 300 seconds).

Your system should accept a timestamp parameter (in seconds granularity), and you may assume that calls are being made to the system in chronological order (i.e., timestamp is monotonically increasing). Several hits may arrive roughly at the same time.

Implement the HitCounter class:
- HitCounter() Initializes the object of the hit counter system.
- void hit(int timestamp) Records a hit that happened at timestamp (in seconds). Several hits may happen at the same timestamp.
- int getHits(int timestamp) Returns the number of hits in the past 5 minutes from timestamp (i.e., from timestamp - 300 + 1 to timestamp).
"""

from typing import List, Deque
from collections import deque
import bisect

class HitCounter1:
    """
    Approach 1: Deque Implementation
    
    Use deque to maintain sliding window of hits.
    
    Time: hit O(1), getHits O(k) where k is hits to remove, Space: O(n)
    """
    
    def __init__(self):
        self.hits = deque()  # Store timestamps
    
    def hit(self, timestamp: int) -> None:
        self.hits.append(timestamp)
    
    def getHits(self, timestamp: int) -> int:
        # Remove hits older than 300 seconds
        while self.hits and self.hits[0] <= timestamp - 300:
            self.hits.popleft()
        
        return len(self.hits)


class HitCounter2:
    """
    Approach 2: Circular Array with Buckets
    
    Use circular array with 300 buckets for time windows.
    
    Time: hit O(1), getHits O(1), Space: O(300)
    """
    
    def __init__(self):
        self.times = [0] * 300  # Store timestamp for each bucket
        self.hits = [0] * 300   # Store hit count for each bucket
    
    def hit(self, timestamp: int) -> None:
        idx = timestamp % 300
        
        if self.times[idx] != timestamp:
            # New time window, reset the bucket
            self.times[idx] = timestamp
            self.hits[idx] = 1
        else:
            # Same time window, increment hits
            self.hits[idx] += 1
    
    def getHits(self, timestamp: int) -> int:
        total_hits = 0
        
        for i in range(300):
            # Check if this bucket is within the 5-minute window
            if timestamp - self.times[i] < 300:
                total_hits += self.hits[i]
        
        return total_hits


class HitCounter3:
    """
    Approach 3: List with Binary Search
    
    Use list to store timestamps and binary search for range queries.
    
    Time: hit O(1), getHits O(log n), Space: O(n)
    """
    
    def __init__(self):
        self.timestamps = []
    
    def hit(self, timestamp: int) -> None:
        self.timestamps.append(timestamp)
    
    def getHits(self, timestamp: int) -> int:
        # Find the leftmost position where time > timestamp - 300
        left_bound = timestamp - 300 + 1
        left_idx = bisect.bisect_left(self.timestamps, left_bound)
        
        # Count hits from left_idx to end
        return len(self.timestamps) - left_idx


class HitCounter4:
    """
    Approach 4: HashMap with Cleanup
    
    Use hashmap to count hits per timestamp with periodic cleanup.
    
    Time: hit O(1), getHits O(k) where k is unique timestamps, Space: O(k)
    """
    
    def __init__(self):
        self.hit_counts = {}  # timestamp -> count
    
    def hit(self, timestamp: int) -> None:
        self.hit_counts[timestamp] = self.hit_counts.get(timestamp, 0) + 1
    
    def getHits(self, timestamp: int) -> int:
        # Clean up old timestamps and count valid hits
        total_hits = 0
        to_remove = []
        
        for ts, count in self.hit_counts.items():
            if timestamp - ts < 300:
                total_hits += count
            else:
                to_remove.append(ts)
        
        # Remove old timestamps
        for ts in to_remove:
            del self.hit_counts[ts]
        
        return total_hits


class HitCounter5:
    """
    Approach 5: Two Pointers with List
    
    Use two pointers to maintain valid window.
    
    Time: hit O(1), getHits O(k) where k is expired hits, Space: O(n)
    """
    
    def __init__(self):
        self.hits = []
        self.left = 0  # Points to first valid hit
    
    def hit(self, timestamp: int) -> None:
        self.hits.append(timestamp)
    
    def getHits(self, timestamp: int) -> int:
        # Move left pointer to maintain valid window
        while self.left < len(self.hits) and self.hits[self.left] <= timestamp - 300:
            self.left += 1
        
        return len(self.hits) - self.left


class HitCounter6:
    """
    Approach 6: Optimized Circular Buffer
    
    Use circular buffer with intelligent bucketing.
    
    Time: hit O(1), getHits O(1), Space: O(300)
    """
    
    def __init__(self):
        self.buckets = [[0, 0] for _ in range(300)]  # [timestamp, count]
    
    def hit(self, timestamp: int) -> None:
        idx = timestamp % 300
        
        if self.buckets[idx][0] != timestamp:
            # New timestamp for this bucket
            self.buckets[idx] = [timestamp, 1]
        else:
            # Same timestamp, increment count
            self.buckets[idx][1] += 1
    
    def getHits(self, timestamp: int) -> int:
        total_hits = 0
        
        for bucket_timestamp, count in self.buckets:
            if timestamp - bucket_timestamp < 300:
                total_hits += count
        
        return total_hits


class HitCounter7:
    """
    Approach 7: Segment-based Tracking
    
    Track hits in time segments for efficient queries.
    
    Time: hit O(1), getHits O(1), Space: O(300)
    """
    
    def __init__(self):
        self.window_size = 300
        self.segments = {}  # segment_id -> hit_count
        self.current_segment = -1
        self.current_segment_hits = 0
    
    def hit(self, timestamp: int) -> None:
        segment_id = timestamp // self.window_size
        
        if segment_id != self.current_segment:
            # Save current segment if it has hits
            if self.current_segment_hits > 0:
                self.segments[self.current_segment] = self.current_segment_hits
            
            # Start new segment
            self.current_segment = segment_id
            self.current_segment_hits = 1
        else:
            self.current_segment_hits += 1
    
    def getHits(self, timestamp: int) -> int:
        # Clean up old segments
        cutoff_time = timestamp - 299
        cutoff_segment = cutoff_time // self.window_size
        
        # Remove segments that are completely outside the window
        to_remove = [seg_id for seg_id in self.segments.keys() if seg_id < cutoff_segment]
        for seg_id in to_remove:
            del self.segments[seg_id]
        
        # Count hits in valid segments
        total_hits = 0
        
        # Add hits from stored segments
        for seg_id, count in self.segments.items():
            if seg_id * self.window_size >= cutoff_time:
                total_hits += count
        
        # Add hits from current segment
        if self.current_segment * self.window_size >= cutoff_time:
            total_hits += self.current_segment_hits
        
        return total_hits


def test_hit_counter_implementations():
    """Test all hit counter implementations"""
    
    implementations = [
        ("Deque Implementation", HitCounter1),
        ("Circular Array Buckets", HitCounter2),
        ("Binary Search", HitCounter3),
        ("HashMap with Cleanup", HitCounter4),
        ("Two Pointers", HitCounter5),
        ("Optimized Circular Buffer", HitCounter6),
    ]
    
    test_cases = [
        ([("hit", 1), ("hit", 2), ("hit", 3), ("getHits", 4), ("hit", 300), ("getHits", 300), ("getHits", 301)], 
         [None, None, None, 3, None, 4, 3], "Basic test"),
        ([("hit", 1), ("getHits", 1), ("getHits", 300), ("getHits", 301)], 
         [None, 1, 1, 0], "Single hit test"),
        ([("getHits", 1)], [0], "No hits test"),
    ]
    
    print("=== Testing Hit Counter Implementations ===")
    
    for operations, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Operations: {operations}")
        print(f"Expected: {expected}")
        
        for impl_name, impl_class in implementations:
            try:
                counter = impl_class()
                results = []
                
                for op, *args in operations:
                    if op == "hit":
                        counter.hit(args[0])
                        results.append(None)
                    elif op == "getHits":
                        result = counter.getHits(args[0])
                        results.append(result)
                
                status = "✓" if results == expected else "✗"
                print(f"{impl_name:25} | {status} | Results: {results}")
            
            except Exception as e:
                print(f"{impl_name:25} | ERROR: {str(e)[:40]}")


def demonstrate_sliding_window():
    """Demonstrate sliding window behavior"""
    print("\n=== Sliding Window Behavior Demonstration ===")
    
    counter = HitCounter1()
    
    operations = [
        ("hit", 1),
        ("hit", 2),
        ("hit", 3),
        ("getHits", 4),
        ("hit", 300),
        ("getHits", 300),
        ("getHits", 301),
        ("hit", 600),
        ("getHits", 600),
    ]
    
    print("Window size: 300 seconds")
    
    for op, timestamp in operations:
        if op == "hit":
            counter.hit(timestamp)
            print(f"hit({timestamp})")
        else:
            hits = counter.getHits(timestamp)
            window_start = timestamp - 299
            window_end = timestamp
            print(f"getHits({timestamp}) = {hits} (window: [{window_start}, {window_end}])")


def benchmark_hit_counter():
    """Benchmark different hit counter implementations"""
    import time
    import random
    
    implementations = [
        ("Deque Implementation", HitCounter1),
        ("Circular Array Buckets", HitCounter2),
        ("Binary Search", HitCounter3),
        ("Two Pointers", HitCounter5),
        ("Optimized Circular Buffer", HitCounter6),
    ]
    
    # Generate test data
    n_operations = 10000
    operations = []
    current_time = 1
    
    for _ in range(n_operations):
        if random.random() < 0.8:  # 80% hits, 20% getHits
            operations.append(("hit", current_time))
        else:
            operations.append(("getHits", current_time))
        
        current_time += random.randint(1, 10)  # Time advances
    
    print("\n=== Hit Counter Performance Benchmark ===")
    print(f"Operations: {n_operations}")
    
    for impl_name, impl_class in implementations:
        counter = impl_class()
        
        start_time = time.time()
        
        for op, timestamp in operations:
            if op == "hit":
                counter.hit(timestamp)
            else:
                counter.getHits(timestamp)
        
        end_time = time.time()
        
        print(f"{impl_name:25} | Time: {end_time - start_time:.4f}s")


def test_edge_cases():
    """Test edge cases for hit counter"""
    print("\n=== Testing Edge Cases ===")
    
    counter = HitCounter2()  # Using circular array implementation
    
    edge_cases = [
        ("No hits", lambda: counter.getHits(100), 0),
        ("Single hit", lambda: (counter.hit(1), counter.getHits(1))[1], 1),
        ("Hit at boundary", lambda: counter.getHits(300), 1),
        ("Hit just outside", lambda: counter.getHits(301), 0),
        ("Multiple hits same time", lambda: (counter.hit(500), counter.hit(500), counter.getHits(500))[2], 2),
        ("Large timestamp", lambda: (counter.hit(1000000), counter.getHits(1000000))[1], 1),
    ]
    
    for description, operation, expected in edge_cases:
        try:
            result = operation()
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | Expected: {expected}, Got: {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def test_window_boundary():
    """Test behavior at window boundaries"""
    print("\n=== Window Boundary Test ===")
    
    counter = HitCounter1()
    
    # Test hits at exact boundaries
    boundary_tests = [
        (100, "Base hit"),
        (399, "Just within window"),
        (400, "Exactly at boundary"),
        (401, "Just outside window"),
    ]
    
    for timestamp, description in boundary_tests:
        counter.hit(timestamp)
        
        # Test getHits at different query times
        for query_time in [400, 401, 700]:
            hits = counter.getHits(query_time)
            window_start = query_time - 299
            in_window = "✓" if window_start <= timestamp <= query_time else "✗"
            print(f"hit({timestamp}) | {description:20} | getHits({query_time}) = {hits} {in_window}")


def analyze_memory_usage():
    """Analyze memory usage patterns"""
    print("\n=== Memory Usage Analysis ===")
    
    import sys
    
    implementations = [
        ("Deque Implementation", HitCounter1),
        ("Circular Array Buckets", HitCounter2),
        ("Binary Search", HitCounter3),
        ("HashMap with Cleanup", HitCounter4),
    ]
    
    # Simulate many hits
    n_hits = 5000
    
    for impl_name, impl_class in implementations:
        counter = impl_class()
        
        # Add many hits
        for i in range(n_hits):
            counter.hit(i)
        
        # Trigger cleanup with getHits
        counter.getHits(n_hits - 1)
        
        # Estimate memory usage
        memory_size = sys.getsizeof(counter)
        
        # Add size of internal data structures
        if hasattr(counter, 'hits'):
            memory_size += sys.getsizeof(counter.hits)
        if hasattr(counter, 'timestamps'):
            memory_size += sys.getsizeof(counter.timestamps)
        if hasattr(counter, 'hit_counts'):
            memory_size += sys.getsizeof(counter.hit_counts)
        if hasattr(counter, 'times'):
            memory_size += sys.getsizeof(counter.times)
            memory_size += sys.getsizeof(counter.hits)
        
        print(f"{impl_name:25} | Memory: ~{memory_size} bytes")


def test_concurrent_hits():
    """Test handling of concurrent hits at same timestamp"""
    print("\n=== Concurrent Hits Test ===")
    
    counter = HitCounter2()
    
    # Multiple hits at same timestamp
    timestamp = 1000
    n_hits = 10
    
    for _ in range(n_hits):
        counter.hit(timestamp)
    
    hits_count = counter.getHits(timestamp)
    print(f"Added {n_hits} hits at timestamp {timestamp}")
    print(f"getHits({timestamp}) = {hits_count}")
    print(f"Correct handling: {'✓' if hits_count == n_hits else '✗'}")
    
    # Test hits spread across time
    for i in range(5):
        counter.hit(timestamp + i)
    
    total_hits = counter.getHits(timestamp + 10)
    expected_total = n_hits + 5
    print(f"After adding 5 more hits: getHits({timestamp + 10}) = {total_hits}")
    print(f"Expected total: {expected_total}, Correct: {'✓' if total_hits == expected_total else '✗'}")


def stress_test():
    """Stress test with many operations"""
    print("\n=== Stress Test ===")
    
    counter = HitCounter2()  # Using circular array for efficiency
    
    n_operations = 100000
    hit_count = 0
    query_count = 0
    
    print(f"Performing {n_operations} operations...")
    
    import random
    current_time = 1
    
    for i in range(n_operations):
        if random.random() < 0.9:  # 90% hits
            counter.hit(current_time)
            hit_count += 1
        else:  # 10% queries
            hits = counter.getHits(current_time)
            query_count += 1
            
            if i % 10000 == 0:
                print(f"After {i+1} operations: {hit_count} hits, {query_count} queries, current hits: {hits}")
        
        current_time += random.randint(1, 5)
    
    final_hits = counter.getHits(current_time)
    print(f"Stress test completed: {hit_count} total hits, {query_count} queries")
    print(f"Final active hits: {final_hits}")


def compare_accuracy():
    """Compare accuracy of different implementations"""
    print("\n=== Accuracy Comparison ===")
    
    implementations = [
        ("Deque Implementation", HitCounter1),
        ("Circular Array Buckets", HitCounter2),
        ("Binary Search", HitCounter3),
        ("Two Pointers", HitCounter5),
    ]
    
    # Complex test sequence
    operations = [
        ("hit", 1), ("hit", 2), ("hit", 3), ("hit", 300), ("hit", 301),
        ("getHits", 301), ("getHits", 302), ("getHits", 600), ("getHits", 601)
    ]
    
    print(f"Test operations: {operations}")
    
    all_results = {}
    
    for impl_name, impl_class in implementations:
        counter = impl_class()
        results = []
        
        for op, timestamp in operations:
            if op == "hit":
                counter.hit(timestamp)
                results.append(None)
            else:
                hits = counter.getHits(timestamp)
                results.append(hits)
        
        all_results[impl_name] = results
        print(f"{impl_name:25} | Results: {results}")
    
    # Check consistency
    first_results = list(all_results.values())[0]
    all_consistent = all(results == first_results for results in all_results.values())
    
    print(f"\nAll implementations consistent: {'✓' if all_consistent else '✗'}")


if __name__ == "__main__":
    test_hit_counter_implementations()
    demonstrate_sliding_window()
    test_edge_cases()
    test_window_boundary()
    test_concurrent_hits()
    compare_accuracy()
    benchmark_hit_counter()
    analyze_memory_usage()
    stress_test()

"""
Design Hit Counter demonstrates multiple approaches for time-based
sliding window counting including deque, circular arrays, binary search,
and optimized bucketing strategies with comprehensive testing and analysis.
"""
