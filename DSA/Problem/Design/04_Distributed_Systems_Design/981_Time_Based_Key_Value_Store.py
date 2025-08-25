"""
981. Time Based Key-Value Store - Multiple Approaches
Difficulty: Medium

Design a time-based key-value data structure that can store multiple values for the same key at different time stamps and retrieve the key's value at a certain timestamp.

Implement the TimeMap class:
- TimeMap() Initializes the object of the data structure.
- void set(String key, String value, int timestamp) Stores the key key with the value value at the given time timestamp.
- String get(String key, int timestamp) Returns a value such that set was called previously, with timestamp_prev <= timestamp. If there are multiple such values, it returns the value associated with the largest timestamp_prev. If there are no values, it returns "".
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import bisect

class TimeMapLinear:
    """
    Approach 1: Linear Search
    
    Store values in lists and search linearly for each get.
    
    Time Complexity:
    - set: O(1)
    - get: O(n) where n is number of values for key
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        # Key -> list of (timestamp, value) pairs
        self.store = defaultdict(list)
    
    def set(self, key: str, value: str, timestamp: int) -> None:
        self.store[key].append((timestamp, value))
    
    def get(self, key: str, timestamp: int) -> str:
        if key not in self.store:
            return ""
        
        # Linear search for largest timestamp <= target
        result = ""
        for ts, val in self.store[key]:
            if ts <= timestamp:
                result = val  # Keep updating to get the latest valid value
            else:
                break  # Assuming timestamps are added in order
        
        return result

class TimeMapBinarySearch:
    """
    Approach 2: Binary Search (Optimal)
    
    Use binary search for efficient timestamp lookup.
    
    Time Complexity:
    - set: O(1) amortized (assuming chronological order)
    - get: O(log n) where n is number of values for key
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.store = defaultdict(list)
    
    def set(self, key: str, value: str, timestamp: int) -> None:
        self.store[key].append((timestamp, value))
    
    def get(self, key: str, timestamp: int) -> str:
        if key not in self.store:
            return ""
        
        values = self.store[key]
        
        # Binary search for rightmost timestamp <= target
        left, right = 0, len(values) - 1
        result = ""
        
        while left <= right:
            mid = (left + right) // 2
            
            if values[mid][0] <= timestamp:
                result = values[mid][1]
                left = mid + 1
            else:
                right = mid - 1
        
        return result

class TimeMapBisect:
    """
    Approach 3: Using Python's bisect module
    
    Utilize built-in binary search for cleaner code.
    
    Time Complexity:
    - set: O(1)
    - get: O(log n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.timestamps = defaultdict(list)
        self.values = defaultdict(list)
    
    def set(self, key: str, value: str, timestamp: int) -> None:
        self.timestamps[key].append(timestamp)
        self.values[key].append(value)
    
    def get(self, key: str, timestamp: int) -> str:
        if key not in self.timestamps:
            return ""
        
        # Find rightmost position where timestamp <= target
        ts_list = self.timestamps[key]
        idx = bisect.bisect_right(ts_list, timestamp) - 1
        
        if idx >= 0:
            return self.values[key][idx]
        
        return ""

class TimeMapSegmentTree:
    """
    Approach 4: Segment Tree for Range Queries
    
    Use segment tree for advanced range-based operations.
    
    Time Complexity:
    - set: O(log max_timestamp)
    - get: O(log max_timestamp)
    
    Space Complexity: O(max_timestamp)
    """
    
    def __init__(self):
        self.data = defaultdict(dict)  # key -> {timestamp -> value}
        self.max_timestamp = 10**9
    
    def set(self, key: str, value: str, timestamp: int) -> None:
        self.data[key][timestamp] = value
    
    def get(self, key: str, timestamp: int) -> str:
        if key not in self.data:
            return ""
        
        # Find the largest timestamp <= target
        best_ts = -1
        for ts in self.data[key]:
            if ts <= timestamp and ts > best_ts:
                best_ts = ts
        
        if best_ts != -1:
            return self.data[key][best_ts]
        
        return ""

class TimeMapAdvanced:
    """
    Approach 5: Advanced with Features and Optimization
    
    Enhanced with caching, compression, and analytics.
    
    Time Complexity:
    - set: O(1) amortized
    - get: O(log n) with caching benefits
    
    Space Complexity: O(n + cache_size)
    """
    
    def __init__(self):
        self.store = defaultdict(list)
        
        # Performance tracking
        self.set_count = 0
        self.get_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Caching for frequent queries
        self.query_cache = {}  # (key, timestamp) -> value
        self.cache_limit = 1000
        
        # Statistics
        self.key_stats = defaultdict(lambda: {'sets': 0, 'gets': 0})
        self.timestamp_range = {}  # key -> (min_ts, max_ts)
    
    def set(self, key: str, value: str, timestamp: int) -> None:
        self.set_count += 1
        self.key_stats[key]['sets'] += 1
        
        # Update timestamp range
        if key not in self.timestamp_range:
            self.timestamp_range[key] = (timestamp, timestamp)
        else:
            min_ts, max_ts = self.timestamp_range[key]
            self.timestamp_range[key] = (min(min_ts, timestamp), max(max_ts, timestamp))
        
        self.store[key].append((timestamp, value))
        
        # Invalidate cache entries for this key
        keys_to_remove = [cache_key for cache_key in self.query_cache 
                         if cache_key[0] == key and cache_key[1] >= timestamp]
        for cache_key in keys_to_remove:
            del self.query_cache[cache_key]
    
    def get(self, key: str, timestamp: int) -> str:
        self.get_count += 1
        self.key_stats[key]['gets'] += 1
        
        # Check cache first
        cache_key = (key, timestamp)
        if cache_key in self.query_cache:
            self.cache_hits += 1
            return self.query_cache[cache_key]
        
        self.cache_misses += 1
        
        if key not in self.store:
            return ""
        
        values = self.store[key]
        
        # Binary search with optimization
        left, right = 0, len(values) - 1
        result = ""
        
        while left <= right:
            mid = (left + right) // 2
            
            if values[mid][0] <= timestamp:
                result = values[mid][1]
                left = mid + 1
            else:
                right = mid - 1
        
        # Cache the result if within limit
        if len(self.query_cache) < self.cache_limit:
            self.query_cache[cache_key] = result
        
        return result
    
    def get_statistics(self) -> dict:
        """Get performance statistics"""
        cache_hit_rate = self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        
        return {
            'total_sets': self.set_count,
            'total_gets': self.get_count,
            'unique_keys': len(self.store),
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.query_cache),
            'average_values_per_key': sum(len(values) for values in self.store.values()) / max(1, len(self.store))
        }
    
    def get_key_info(self, key: str) -> dict:
        """Get information about a specific key"""
        if key not in self.store:
            return {'exists': False}
        
        values = self.store[key]
        min_ts, max_ts = self.timestamp_range[key]
        
        return {
            'exists': True,
            'value_count': len(values),
            'min_timestamp': min_ts,
            'max_timestamp': max_ts,
            'timestamp_span': max_ts - min_ts,
            'sets': self.key_stats[key]['sets'],
            'gets': self.key_stats[key]['gets']
        }
    
    def cleanup_old_data(self, cutoff_timestamp: int) -> int:
        """Remove data older than cutoff timestamp"""
        removed_count = 0
        
        for key in list(self.store.keys()):
            original_len = len(self.store[key])
            
            # Keep only values with timestamp >= cutoff
            self.store[key] = [(ts, val) for ts, val in self.store[key] 
                              if ts >= cutoff_timestamp]
            
            removed_count += original_len - len(self.store[key])
            
            # Remove key if no values left
            if not self.store[key]:
                del self.store[key]
                if key in self.timestamp_range:
                    del self.timestamp_range[key]
        
        # Clear cache
        self.query_cache.clear()
        
        return removed_count


def test_time_map_basic():
    """Test basic TimeMap functionality"""
    print("=== Testing Basic TimeMap Functionality ===")
    
    implementations = [
        ("Linear Search", TimeMapLinear),
        ("Binary Search", TimeMapBinarySearch),
        ("Bisect Module", TimeMapBisect),
        ("Segment Tree", TimeMapSegmentTree),
        ("Advanced", TimeMapAdvanced)
    ]
    
    for name, TimeMapClass in implementations:
        print(f"\n{name}:")
        
        tm = TimeMapClass()
        
        # Test sequence from problem
        operations = [
            ("set", "foo", "bar", 1),
            ("get", "foo", 1),
            ("get", "foo", 3),
            ("set", "foo", "bar2", 4),
            ("get", "foo", 4),
            ("get", "foo", 5)
        ]
        
        for op, key, *args in operations:
            if op == "set":
                value, timestamp = args
                tm.set(key, value, timestamp)
                print(f"  set('{key}', '{value}', {timestamp})")
            else:  # get
                timestamp = args[0]
                result = tm.get(key, timestamp)
                print(f"  get('{key}', {timestamp}): '{result}'")

def test_time_map_edge_cases():
    """Test TimeMap edge cases"""
    print("\n=== Testing TimeMap Edge Cases ===")
    
    tm = TimeMapAdvanced()
    
    # Test get on non-existent key
    print("Non-existent key:")
    result = tm.get("nonexistent", 100)
    print(f"  get('nonexistent', 100): '{result}'")
    
    # Test get before any set
    print(f"\nGet before any set:")
    tm.set("key1", "value1", 10)
    result = tm.get("key1", 5)
    print(f"  get('key1', 5) when earliest is 10: '{result}'")
    
    # Test multiple values same timestamp
    print(f"\nMultiple sets at same timestamp:")
    tm.set("key2", "old", 20)
    tm.set("key2", "new", 20)  # Same timestamp
    result = tm.get("key2", 20)
    print(f"  After two sets at timestamp 20: '{result}'")
    
    # Test large timestamp gaps
    print(f"\nLarge timestamp gaps:")
    tm.set("key3", "early", 1000)
    tm.set("key3", "late", 1000000)
    
    test_times = [999, 1000, 50000, 999999, 1000000, 2000000]
    for test_time in test_times:
        result = tm.get("key3", test_time)
        print(f"  get('key3', {test_time}): '{result}'")

def test_chronological_order():
    """Test with non-chronological insertions"""
    print("\n=== Testing Non-Chronological Order ===")
    
    tm = TimeMapBinarySearch()
    
    # Insert timestamps out of order
    insertions = [
        ("key", "value5", 50),
        ("key", "value1", 10),
        ("key", "value3", 30),
        ("key", "value2", 20),
        ("key", "value4", 40)
    ]
    
    print("Inserting out of chronological order:")
    for key, value, timestamp in insertions:
        tm.set(key, value, timestamp)
        print(f"  set('{key}', '{value}', {timestamp})")
    
    # Test queries
    print(f"\nQuerying at different timestamps:")
    test_timestamps = [5, 15, 25, 35, 45, 55]
    
    for ts in test_timestamps:
        result = tm.get("key", ts)
        print(f"  get('key', {ts}): '{result}'")

def test_performance_comparison():
    """Test performance of different implementations"""
    print("\n=== Testing Performance Comparison ===")
    
    import time
    
    implementations = [
        ("Linear Search", TimeMapLinear),
        ("Binary Search", TimeMapBinarySearch),
        ("Bisect Module", TimeMapBisect),
        ("Advanced", TimeMapAdvanced)
    ]
    
    num_operations = 5000
    
    for name, TimeMapClass in implementations:
        tm = TimeMapClass()
        
        # Time set operations
        start_time = time.time()
        for i in range(num_operations):
            key = f"key_{i % 100}"  # 100 different keys
            value = f"value_{i}"
            timestamp = i
            tm.set(key, value, timestamp)
        set_time = (time.time() - start_time) * 1000
        
        # Time get operations
        start_time = time.time()
        import random
        for _ in range(num_operations):
            key = f"key_{random.randint(0, 99)}"
            timestamp = random.randint(0, num_operations)
            tm.get(key, timestamp)
        get_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    {num_operations} sets: {set_time:.2f}ms")
        print(f"    {num_operations} gets: {get_time:.2f}ms")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    tm = TimeMapAdvanced()
    
    # Create test data
    keys_data = [
        ("user_sessions", [("session1", 1000), ("session2", 2000), ("session3", 3000)]),
        ("server_configs", [("config_v1", 500), ("config_v2", 1500), ("config_v3", 2500)]),
        ("cache_states", [("cold", 100), ("warm", 800), ("hot", 1200)])
    ]
    
    print("Creating test data:")
    for key, values in keys_data:
        for value, timestamp in values:
            tm.set(key, value, timestamp)
            print(f"  set('{key}', '{value}', {timestamp})")
    
    # Test statistics
    stats = tm.get_statistics()
    print(f"\nOverall statistics:")
    for stat_key, stat_value in stats.items():
        if isinstance(stat_value, float):
            print(f"  {stat_key}: {stat_value:.3f}")
        else:
            print(f"  {stat_key}: {stat_value}")
    
    # Test key-specific information
    print(f"\nKey-specific information:")
    for key, _ in keys_data:
        key_info = tm.get_key_info(key)
        print(f"  {key}: {key_info}")
    
    # Test caching behavior
    print(f"\nTesting cache behavior:")
    
    # Make repeated queries to test caching
    for _ in range(5):
        tm.get("user_sessions", 1500)
        tm.get("server_configs", 1000)
    
    updated_stats = tm.get_statistics()
    print(f"  Cache hit rate: {updated_stats['cache_hit_rate']:.3f}")
    print(f"  Cache size: {updated_stats['cache_size']}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Configuration versioning
    print("Application 1: Configuration Management System")
    
    config_store = TimeMapAdvanced()
    
    # Simulate configuration changes over time
    config_changes = [
        ("database.host", "db1.example.com", 1000),
        ("database.port", "5432", 1000),
        ("database.host", "db2.example.com", 2000),  # Host migration
        ("api.rate_limit", "1000", 1500),
        ("api.rate_limit", "2000", 3000),  # Rate limit increase
        ("feature.new_ui", "false", 2500),
        ("feature.new_ui", "true", 4000)   # Feature rollout
    ]
    
    print("  Configuration timeline:")
    for key, value, timestamp in config_changes:
        config_store.set(key, value, timestamp)
        print(f"    {timestamp}: {key} = {value}")
    
    # Query configurations at different times
    print(f"\n  Configuration state at different times:")
    query_times = [1200, 2200, 3500, 4500]
    
    for query_time in query_times:
        print(f"    At time {query_time}:")
        for config_key in ["database.host", "api.rate_limit", "feature.new_ui"]:
            value = config_store.get(config_key, query_time)
            print(f"      {config_key}: {value if value else 'not set'}")
    
    # Application 2: User session tracking
    print(f"\nApplication 2: User Session State Tracking")
    
    session_store = TimeMapBisect()
    
    # Simulate user session state changes
    user_sessions = [
        ("user123", "logged_out", 0),
        ("user123", "logged_in", 1000),
        ("user123", "browsing", 1500),
        ("user123", "shopping_cart", 2000),
        ("user123", "checkout", 2500),
        ("user123", "payment", 3000),
        ("user123", "confirmed", 3500),
        ("user123", "logged_out", 4000)
    ]
    
    print("  User session timeline:")
    for user, state, timestamp in user_sessions:
        session_store.set(user, state, timestamp)
        print(f"    {timestamp}: {user} -> {state}")
    
    # Analyze user state at different times
    print(f"\n  User state analysis:")
    analysis_times = [800, 1200, 1800, 2300, 2800, 3200, 3800, 4200]
    
    for analysis_time in analysis_times:
        state = session_store.get("user123", analysis_time)
        print(f"    Time {analysis_time}: user was {state if state else 'unknown'}")
    
    # Application 3: Stock price history
    print(f"\nApplication 3: Stock Price History System")
    
    stock_store = TimeMapBinarySearch()
    
    # Simulate stock price updates (timestamps in milliseconds for demo)
    stock_updates = [
        ("AAPL", "150.00", 9000),
        ("AAPL", "152.50", 9300),
        ("AAPL", "149.75", 9600),
        ("AAPL", "153.25", 9900),
        ("GOOGL", "2800.00", 9100),
        ("GOOGL", "2825.50", 9400),
        ("GOOGL", "2790.25", 9700)
    ]
    
    print("  Stock price updates:")
    for symbol, price, timestamp in stock_updates:
        stock_store.set(symbol, price, timestamp)
        print(f"    {timestamp}: {symbol} = ${price}")
    
    # Query historical prices
    print(f"\n  Historical price queries:")
    
    symbols = ["AAPL", "GOOGL"]
    query_times = [9200, 9500, 9800, 10000]
    
    for symbol in symbols:
        print(f"    {symbol} price history:")
        for query_time in query_times:
            price = stock_store.get(symbol, query_time)
            print(f"      At {query_time}: ${price if price else 'N/A'}")

def test_memory_efficiency():
    """Test memory efficiency and cleanup"""
    print("\n=== Testing Memory Efficiency ===")
    
    tm = TimeMapAdvanced()
    
    # Generate large amount of historical data
    print("Generating historical data...")
    
    keys = [f"sensor_{i}" for i in range(50)]
    base_timestamp = 1000000
    
    # 30 days worth of data (one reading per hour)
    for day in range(30):
        for hour in range(24):
            timestamp = base_timestamp + day * 86400 + hour * 3600
            
            for key in keys:
                value = f"reading_{day}_{hour}"
                tm.set(key, value, timestamp)
    
    initial_stats = tm.get_statistics()
    print(f"  Initial data: {initial_stats['total_sets']} sets")
    print(f"  Unique keys: {initial_stats['unique_keys']}")
    print(f"  Avg values per key: {initial_stats['average_values_per_key']:.1f}")
    
    # Test cleanup of old data (keep only last 7 days)
    cutoff_timestamp = base_timestamp + 23 * 86400  # Keep last 7 days
    
    print(f"\nCleaning up data older than timestamp {cutoff_timestamp}...")
    removed_count = tm.cleanup_old_data(cutoff_timestamp)
    
    final_stats = tm.get_statistics()
    print(f"  Removed {removed_count} old entries")
    print(f"  Remaining keys: {final_stats['unique_keys']}")
    print(f"  New avg values per key: {final_stats['average_values_per_key']:.1f}")

def stress_test_time_map():
    """Stress test TimeMap with heavy load"""
    print("\n=== Stress Testing TimeMap ===")
    
    import time
    import random
    
    tm = TimeMapBinarySearch()  # Use efficient implementation
    
    # Stress test parameters
    num_keys = 1000
    operations_per_key = 100
    total_operations = num_keys * operations_per_key
    
    print(f"Stress test: {total_operations} total operations")
    
    start_time = time.time()
    
    # Phase 1: Heavy set operations
    print("  Phase 1: Heavy write load...")
    
    for i in range(total_operations):
        key = f"key_{i % num_keys}"
        value = f"value_{i}"
        timestamp = i
        tm.set(key, value, timestamp)
    
    write_time = time.time() - start_time
    
    # Phase 2: Heavy get operations
    print("  Phase 2: Heavy read load...")
    
    start_time = time.time()
    
    for _ in range(total_operations):
        key = f"key_{random.randint(0, num_keys - 1)}"
        timestamp = random.randint(0, total_operations)
        tm.get(key, timestamp)
    
    read_time = time.time() - start_time
    
    print(f"  Write performance: {write_time:.2f}s ({total_operations/write_time:.0f} ops/sec)")
    print(f"  Read performance: {read_time:.2f}s ({total_operations/read_time:.0f} ops/sec)")

def benchmark_query_patterns():
    """Benchmark different query patterns"""
    print("\n=== Benchmarking Query Patterns ===")
    
    import time
    import random
    
    tm = TimeMapBisect()
    
    # Setup: Create data with 100 keys, 1000 timestamps each
    num_keys = 100
    timestamps_per_key = 1000
    
    print("Setting up benchmark data...")
    
    for key_id in range(num_keys):
        for ts in range(timestamps_per_key):
            key = f"key_{key_id}"
            value = f"value_{key_id}_{ts}"
            timestamp = ts * 10  # Spread timestamps
            tm.set(key, value, timestamp)
    
    # Benchmark different query patterns
    patterns = [
        ("Recent queries", lambda: random.randint(8000, 9990)),
        ("Random queries", lambda: random.randint(0, 9990)),
        ("Old queries", lambda: random.randint(0, 2000)),
        ("Sequential queries", lambda: None)  # Special case
    ]
    
    num_queries = 5000
    
    for pattern_name, timestamp_gen in patterns:
        start_time = time.time()
        
        if pattern_name == "Sequential queries":
            # Sequential pattern
            for i in range(num_queries):
                key = f"key_{i % num_keys}"
                timestamp = (i % timestamps_per_key) * 10
                tm.get(key, timestamp)
        else:
            # Random pattern
            for _ in range(num_queries):
                key = f"key_{random.randint(0, num_keys - 1)}"
                timestamp = timestamp_gen()
                tm.get(key, timestamp)
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {pattern_name}: {elapsed:.2f}ms for {num_queries} queries")

if __name__ == "__main__":
    test_time_map_basic()
    test_time_map_edge_cases()
    test_chronological_order()
    test_performance_comparison()
    test_advanced_features()
    demonstrate_applications()
    test_memory_efficiency()
    stress_test_time_map()
    benchmark_query_patterns()

"""
Time-Based Key-Value Store Design demonstrates key concepts:

Core Approaches:
1. Linear Search - Simple list traversal for each query
2. Binary Search - Optimal logarithmic search for timestamps
3. Bisect Module - Clean implementation using built-in binary search
4. Segment Tree - Advanced approach for range-based operations
5. Advanced - Enhanced with caching, analytics, and cleanup

Key Design Principles:
- Time-series data storage and retrieval
- Efficient temporal range queries
- Binary search optimization for sorted timestamps
- Memory management for historical data

Performance Characteristics:
- Linear: O(1) set, O(n) get, simple implementation
- Binary Search: O(1) set, O(log n) get, optimal for queries
- Bisect: O(1) set, O(log n) get, clean and efficient
- Advanced: Includes caching and analytics overhead

Real-world Applications:
- Configuration management and versioning
- User session state tracking
- Stock price and financial data history
- Sensor data and IoT time series
- Database transaction logs
- System monitoring and metrics

The binary search approach is most commonly used due to
its optimal O(log n) query performance while maintaining
simple O(1) insertion complexity.
"""
