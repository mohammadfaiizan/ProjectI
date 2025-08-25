"""
362. Design Hit Counter - Multiple Approaches
Difficulty: Medium

Design a hit counter which counts the number of hits received in the past 5 minutes (300 seconds).

Your system should accept a timestamp parameter (in seconds granularity) and you may assume that calls are being made to the system in chronological order (i.e., timestamp is monotonically increasing). Several hits may arrive at the same timestamp.

Implement the HitCounter class:
- HitCounter() Initializes the object of the hit counter system.
- void hit(int timestamp) Records a hit that happened at timestamp (in seconds). Several hits may happen at the same timestamp.
- int getHits(int timestamp) Returns the number of hits in the past 5 minutes from timestamp (i.e., the past 300 seconds).
"""

from typing import List, Dict, Deque
from collections import deque, defaultdict
import bisect

class HitCounterSimple:
    """
    Approach 1: Simple List with Linear Search
    
    Store all timestamps and filter by time window.
    
    Time Complexity:
    - hit: O(1)
    - getHits: O(n) where n is total hits
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.hits = []
        self.window_size = 300
    
    def hit(self, timestamp: int) -> None:
        self.hits.append(timestamp)
    
    def getHits(self, timestamp: int) -> int:
        count = 0
        earliest_valid = timestamp - self.window_size + 1
        
        for hit_time in self.hits:
            if hit_time >= earliest_valid:
                count += 1
        
        return count

class HitCounterQueue:
    """
    Approach 2: Queue with Automatic Cleanup
    
    Use deque and remove expired entries automatically.
    
    Time Complexity:
    - hit: O(1)
    - getHits: O(k) where k is expired entries
    
    Space Complexity: O(m) where m is hits in window
    """
    
    def __init__(self):
        self.hits = deque()
        self.window_size = 300
    
    def hit(self, timestamp: int) -> None:
        self.hits.append(timestamp)
    
    def getHits(self, timestamp: int) -> int:
        # Remove expired hits
        earliest_valid = timestamp - self.window_size + 1
        
        while self.hits and self.hits[0] < earliest_valid:
            self.hits.popleft()
        
        return len(self.hits)

class HitCounterBucket:
    """
    Approach 3: Circular Buffer/Bucket Array
    
    Use fixed-size array with buckets for each second.
    
    Time Complexity:
    - hit: O(1)
    - getHits: O(1)
    
    Space Complexity: O(300) = O(1)
    """
    
    def __init__(self):
        self.window_size = 300
        self.buckets = [0] * self.window_size
        self.timestamps = [0] * self.window_size
    
    def hit(self, timestamp: int) -> None:
        index = timestamp % self.window_size
        
        if self.timestamps[index] != timestamp:
            # New timestamp, reset bucket
            self.buckets[index] = 0
            self.timestamps[index] = timestamp
        
        self.buckets[index] += 1
    
    def getHits(self, timestamp: int) -> int:
        total_hits = 0
        earliest_valid = timestamp - self.window_size + 1
        
        for i in range(self.window_size):
            if self.timestamps[i] >= earliest_valid:
                total_hits += self.buckets[i]
        
        return total_hits

class HitCounterHashMap:
    """
    Approach 4: HashMap with Timestamp Keys
    
    Use dictionary with timestamp as key and count as value.
    
    Time Complexity:
    - hit: O(1)
    - getHits: O(300) = O(1)
    
    Space Complexity: O(k) where k is unique timestamps in window
    """
    
    def __init__(self):
        self.hit_counts = defaultdict(int)
        self.window_size = 300
    
    def hit(self, timestamp: int) -> None:
        self.hit_counts[timestamp] += 1
    
    def getHits(self, timestamp: int) -> int:
        total_hits = 0
        earliest_valid = timestamp - self.window_size + 1
        
        # Clean up old entries and count valid ones
        expired_keys = []
        
        for hit_time, count in self.hit_counts.items():
            if hit_time < earliest_valid:
                expired_keys.append(hit_time)
            else:
                total_hits += count
        
        # Remove expired entries
        for key in expired_keys:
            del self.hit_counts[key]
        
        return total_hits

class HitCounterAdvanced:
    """
    Approach 5: Advanced with Multiple Features
    
    Support different window sizes and additional analytics.
    
    Time Complexity:
    - hit: O(1)
    - getHits: O(log n) with binary search optimization
    
    Space Complexity: O(n)
    """
    
    def __init__(self, window_size: int = 300):
        self.window_size = window_size
        self.timestamps = []  # Sorted list of timestamps
        self.counts = []      # Corresponding hit counts
        self.total_hits = 0
        self.last_cleanup = 0
    
    def hit(self, timestamp: int) -> None:
        # Binary search to find insertion point
        if not self.timestamps or timestamp > self.timestamps[-1]:
            # Common case: append to end
            self.timestamps.append(timestamp)
            self.counts.append(1)
        elif timestamp == self.timestamps[-1]:
            # Same timestamp as last hit
            self.counts[-1] += 1
        else:
            # Find insertion point
            index = bisect.bisect_left(self.timestamps, timestamp)
            
            if index < len(self.timestamps) and self.timestamps[index] == timestamp:
                # Timestamp already exists
                self.counts[index] += 1
            else:
                # Insert new timestamp
                self.timestamps.insert(index, timestamp)
                self.counts.insert(index, 1)
        
        self.total_hits += 1
        
        # Periodic cleanup
        if timestamp - self.last_cleanup > self.window_size:
            self._cleanup(timestamp)
            self.last_cleanup = timestamp
    
    def getHits(self, timestamp: int) -> int:
        earliest_valid = timestamp - self.window_size + 1
        
        # Binary search for earliest valid timestamp
        left_index = bisect.bisect_left(self.timestamps, earliest_valid)
        
        # Sum counts from valid range
        return sum(self.counts[left_index:])
    
    def _cleanup(self, current_timestamp: int) -> None:
        """Remove expired entries"""
        earliest_valid = current_timestamp - self.window_size + 1
        
        # Find first valid index
        valid_index = bisect.bisect_left(self.timestamps, earliest_valid)
        
        if valid_index > 0:
            # Remove expired entries
            expired_hits = sum(self.counts[:valid_index])
            self.timestamps = self.timestamps[valid_index:]
            self.counts = self.counts[valid_index:]
            self.total_hits -= expired_hits
    
    def get_total_hits(self) -> int:
        """Get total hits ever recorded"""
        return self.total_hits
    
    def get_hit_rate(self, timestamp: int) -> float:
        """Get hits per second in current window"""
        hits = self.getHits(timestamp)
        return hits / self.window_size
    
    def get_window_size(self) -> int:
        """Get current window size"""
        return self.window_size


def test_hit_counter_basic():
    """Test basic hit counter functionality"""
    print("=== Testing Basic Hit Counter Functionality ===")
    
    implementations = [
        ("Simple List", HitCounterSimple),
        ("Queue-based", HitCounterQueue),
        ("Bucket Array", HitCounterBucket),
        ("HashMap", HitCounterHashMap),
        ("Advanced", HitCounterAdvanced)
    ]
    
    for name, HitCounterClass in implementations:
        print(f"\n{name}:")
        
        counter = HitCounterClass()
        
        # Test sequence from problem description
        operations = [
            ("hit", 1), ("hit", 2), ("hit", 3),
            ("getHits", 4), ("hit", 300), ("getHits", 300),
            ("getHits", 301)
        ]
        
        for op, timestamp in operations:
            if op == "hit":
                counter.hit(timestamp)
                print(f"  hit({timestamp})")
            else:  # getHits
                result = counter.getHits(timestamp)
                print(f"  getHits({timestamp}): {result}")

def test_hit_counter_edge_cases():
    """Test hit counter edge cases"""
    print("\n=== Testing Hit Counter Edge Cases ===")
    
    counter = HitCounterQueue()
    
    # Test with no hits
    print("No hits recorded:")
    print(f"  getHits(100): {counter.getHits(100)}")
    
    # Test multiple hits at same timestamp
    print(f"\nMultiple hits at same timestamp:")
    for _ in range(5):
        counter.hit(200)
    
    print(f"  After 5 hits at timestamp 200:")
    print(f"  getHits(200): {counter.getHits(200)}")
    print(f"  getHits(499): {counter.getHits(499)}")
    print(f"  getHits(500): {counter.getHits(500)}")
    
    # Test window boundary
    print(f"\nWindow boundary test:")
    counter2 = HitCounterBucket()
    
    # Hits at boundaries
    counter2.hit(1)
    counter2.hit(300)
    counter2.hit(301)
    
    print(f"  Hits at: 1, 300, 301")
    print(f"  getHits(300): {counter2.getHits(300)}")  # Should include 1 and 300
    print(f"  getHits(301): {counter2.getHits(301)}")  # Should include 300 and 301
    print(f"  getHits(600): {counter2.getHits(600)}")  # Should include 301 only

def test_sliding_window():
    """Test sliding window behavior"""
    print("\n=== Testing Sliding Window Behavior ===")
    
    counter = HitCounterAdvanced()
    
    # Create hits across a larger time range
    hit_times = [1, 100, 200, 300, 400, 500, 600]
    
    print("Recording hits at:", hit_times)
    for timestamp in hit_times:
        counter.hit(timestamp)
    
    # Test window sliding
    test_times = [300, 400, 500, 600, 700, 800, 900]
    
    print(f"\nSliding window analysis:")
    for test_time in test_times:
        hits = counter.getHits(test_time)
        window_start = test_time - 299  # 300-second window
        print(f"  getHits({test_time}): {hits} (window: {window_start}-{test_time})")

def test_performance_comparison():
    """Test performance of different implementations"""
    print("\n=== Testing Performance Comparison ===")
    
    import time
    
    implementations = [
        ("Simple List", HitCounterSimple),
        ("Queue-based", HitCounterQueue),
        ("Bucket Array", HitCounterBucket),
        ("HashMap", HitCounterHashMap)
    ]
    
    num_operations = 10000
    
    for name, HitCounterClass in implementations:
        counter = HitCounterClass()
        
        # Test hit operations
        start_time = time.time()
        for i in range(num_operations):
            counter.hit(i)
        hit_time = (time.time() - start_time) * 1000
        
        # Test getHits operations
        start_time = time.time()
        for i in range(0, num_operations, 100):  # Every 100th timestamp
            counter.getHits(i + 300)
        get_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    {num_operations} hits: {hit_time:.2f}ms")
        print(f"    {num_operations//100} getHits: {get_time:.2f}ms")

def test_memory_efficiency():
    """Test memory efficiency"""
    print("\n=== Testing Memory Efficiency ===")
    
    # Simulate long-running system
    counter_queue = HitCounterQueue()
    counter_bucket = HitCounterBucket()
    counter_hashmap = HitCounterHashMap()
    
    # Generate hits over a long time period
    time_range = 10000  # 10000 seconds
    hit_frequency = 10  # 10 hits per second on average
    
    print(f"Simulating {time_range} seconds with ~{hit_frequency} hits/sec:")
    
    import random
    for timestamp in range(0, time_range, 1):
        # Random number of hits per timestamp
        hits_this_second = random.poisson(hit_frequency)
        
        for _ in range(hits_this_second):
            counter_queue.hit(timestamp)
            counter_bucket.hit(timestamp)
            counter_hashmap.hit(timestamp)
    
    # Check final state
    final_time = time_range - 1
    
    print(f"  Final getHits({final_time}):")
    print(f"    Queue-based: {counter_queue.getHits(final_time)}")
    print(f"    Bucket Array: {counter_bucket.getHits(final_time)}")
    print(f"    HashMap: {counter_hashmap.getHits(final_time)}")
    
    # Estimate memory usage
    print(f"  Estimated memory usage:")
    print(f"    Queue-based: {len(counter_queue.hits)} timestamps")
    print(f"    Bucket Array: 300 buckets (fixed)")
    print(f"    HashMap: {len(counter_hashmap.hit_counts)} unique timestamps")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    counter = HitCounterAdvanced(window_size=60)  # 1-minute window
    
    # Record hits with varying frequency
    timestamps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    hit_counts = [1, 3, 2, 5, 1, 4, 2, 3, 1, 2]
    
    print("Recording hits with varying frequency:")
    for timestamp, count in zip(timestamps, hit_counts):
        for _ in range(count):
            counter.hit(timestamp)
        print(f"  {count} hits at timestamp {timestamp}")
    
    # Test analytics
    test_time = 100
    print(f"\nAnalytics at timestamp {test_time}:")
    print(f"  Current window hits: {counter.getHits(test_time)}")
    print(f"  Total hits ever: {counter.get_total_hits()}")
    print(f"  Hit rate: {counter.get_hit_rate(test_time):.3f} hits/sec")
    print(f"  Window size: {counter.get_window_size()} seconds")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Rate limiting
    print("Application 1: API Rate Limiting")
    rate_limiter = HitCounterBucket()
    
    # Simulate API requests
    api_requests = [
        (100, "user1"), (101, "user1"), (102, "user1"),
        (150, "user1"), (200, "user1"), (250, "user1")
    ]
    
    rate_limit = 5  # 5 requests per 300 seconds
    
    for timestamp, user in api_requests:
        rate_limiter.hit(timestamp)
        current_requests = rate_limiter.getHits(timestamp)
        
        if current_requests <= rate_limit:
            print(f"  Request allowed for {user} at {timestamp} ({current_requests}/{rate_limit})")
        else:
            print(f"  Request DENIED for {user} at {timestamp} ({current_requests}/{rate_limit})")
    
    # Application 2: Traffic monitoring
    print(f"\nApplication 2: Website Traffic Monitoring")
    traffic_monitor = HitCounterAdvanced(window_size=300)
    
    # Simulate traffic patterns
    print("  Simulating traffic patterns:")
    
    # Normal traffic
    for t in range(100, 200):
        if t % 10 == 0:  # Spike every 10 seconds
            for _ in range(5):
                traffic_monitor.hit(t)
        else:
            traffic_monitor.hit(t)
    
    # Analyze traffic
    analysis_times = [150, 200, 250, 300]
    
    for analysis_time in analysis_times:
        hits = traffic_monitor.getHits(analysis_time)
        rate = traffic_monitor.get_hit_rate(analysis_time)
        
        print(f"    Time {analysis_time}: {hits} hits, {rate:.2f} hits/sec")
    
    # Application 3: System health monitoring
    print(f"\nApplication 3: Error Rate Monitoring")
    error_monitor = HitCounterQueue()
    
    # Simulate error occurrences
    error_times = [100, 105, 200, 205, 210, 300, 350, 380, 390, 395]
    
    print("  Recording error occurrences:")
    for error_time in error_times:
        error_monitor.hit(error_time)
        current_errors = error_monitor.getHits(error_time)
        
        if current_errors > 3:  # Alert threshold
            print(f"    ALERT: {current_errors} errors at time {error_time}")
        else:
            print(f"    Normal: {current_errors} errors at time {error_time}")

def test_concurrent_simulation():
    """Simulate concurrent access patterns"""
    print("\n=== Simulating Concurrent Access Patterns ===")
    
    counter = HitCounterHashMap()
    
    # Simulate multiple users hitting the system
    users = ["user1", "user2", "user3", "user4"]
    base_time = 1000
    
    print("Simulating concurrent user activity:")
    
    # Each user makes requests at different intervals
    for minute in range(5):  # 5 minutes
        current_time = base_time + minute * 60
        
        for user_id, user in enumerate(users):
            # Different request patterns per user
            requests = (user_id + 1) * 2  # user1: 2, user2: 4, etc.
            
            for req in range(requests):
                timestamp = current_time + req * 10  # Spread within minute
                counter.hit(timestamp)
        
        # Check load at end of each minute
        end_time = current_time + 59
        total_hits = counter.getHits(end_time)
        print(f"  Minute {minute + 1}: {total_hits} hits in last 5 minutes")

def benchmark_large_scale():
    """Benchmark large-scale operations"""
    print("\n=== Benchmarking Large-Scale Operations ===")
    
    import time
    
    implementations = [
        ("Bucket Array", HitCounterBucket),
        ("HashMap", HitCounterHashMap),
        ("Advanced", HitCounterAdvanced)
    ]
    
    # Large scale test
    total_hits = 100000
    time_span = 10000  # 10000 seconds
    
    for name, HitCounterClass in implementations:
        counter = HitCounterClass()
        
        start_time = time.time()
        
        # Distribute hits over time span
        import random
        for i in range(total_hits):
            timestamp = random.randint(0, time_span)
            counter.hit(timestamp)
        
        # Perform getHits operations
        for i in range(1000):
            query_time = random.randint(300, time_span)
            counter.getHits(query_time)
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {name}: {elapsed:.2f}ms for {total_hits} hits + 1000 queries")

def test_cleanup_efficiency():
    """Test cleanup efficiency of different approaches"""
    print("\n=== Testing Cleanup Efficiency ===")
    
    counter_queue = HitCounterQueue()
    counter_hashmap = HitCounterHashMap()
    
    # Generate many old hits followed by recent hits
    print("Generating old hits (will be expired):")
    for i in range(1000):
        counter_queue.hit(i)
        counter_hashmap.hit(i)
    
    print("Generating recent hits:")
    recent_start = 10000
    for i in range(recent_start, recent_start + 100):
        counter_queue.hit(i)
        counter_hashmap.hit(i)
    
    # Query recent time to trigger cleanup
    query_time = recent_start + 50
    
    import time
    
    # Test queue cleanup
    start_time = time.time()
    hits_queue = counter_queue.getHits(query_time)
    queue_time = (time.time() - start_time) * 1000
    
    # Test hashmap cleanup  
    start_time = time.time()
    hits_hashmap = counter_hashmap.getHits(query_time)
    hashmap_time = (time.time() - start_time) * 1000
    
    print(f"Cleanup performance:")
    print(f"  Queue-based: {hits_queue} hits, {queue_time:.3f}ms")
    print(f"  HashMap: {hits_hashmap} hits, {hashmap_time:.3f}ms")

if __name__ == "__main__":
    test_hit_counter_basic()
    test_hit_counter_edge_cases()
    test_sliding_window()
    test_performance_comparison()
    test_memory_efficiency()
    test_advanced_features()
    demonstrate_applications()
    test_concurrent_simulation()
    benchmark_large_scale()
    test_cleanup_efficiency()

"""
Hit Counter Design demonstrates key concepts:

Core Approaches:
1. Simple List - Store all hits, linear scan for counting
2. Queue-based - Automatic cleanup of expired entries
3. Bucket Array - Fixed-size circular buffer O(1) operations
4. HashMap - Timestamp->count mapping with cleanup
5. Advanced - Binary search optimization with analytics

Key Design Principles:
- Time window management and sliding windows
- Memory vs performance trade-offs
- Automatic cleanup vs on-demand cleanup
- Space optimization for fixed time windows

Performance Characteristics:
- Simple: O(1) hit, O(n) getHits, O(n) space
- Queue: O(1) hit, O(k) getHits, O(m) space where m is hits in window
- Bucket: O(1) hit, O(1) getHits, O(1) space
- HashMap: O(1) hit, O(1) getHits, O(k) space where k is unique timestamps
- Advanced: O(1) hit, O(log n) getHits with cleanup

Real-world Applications:
- API rate limiting and throttling
- Website traffic monitoring and analytics
- System health and error rate monitoring
- DDoS detection and mitigation
- Performance metrics and SLA monitoring

The bucket array approach is most commonly used for fixed
time windows due to its optimal O(1) performance and
constant space complexity.
"""
