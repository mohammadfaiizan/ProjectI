"""
460. LFU Cache - Multiple Approaches
Difficulty: Hard

Design and implement a data structure for a Least Frequently Used (LFU) cache.

Implement the LFUCache class:
- LFUCache(int capacity) Initializes the object with the capacity of the data structure.
- int get(int key) Gets the value of the key if the key exists in the cache. Otherwise, returns -1.
- void put(int key, int value) Update the value of the key if the key exists. Otherwise, adds 
  the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, 
  evict the least frequently used key.

The functions get and put must each run in O(1) average time complexity.
"""

from typing import Dict, Optional
from collections import defaultdict, OrderedDict
import time

class LFUCacheBasic:
    """
    Approach 1: Basic LFU with HashMap + Frequency Tracking
    
    Use separate data structures for values, frequencies, and frequency groups.
    
    Time Complexity: O(1) for get and put operations
    Space Complexity: O(capacity)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_frequency = 0
        
        # key -> value mapping
        self.key_to_value = {}
        
        # key -> frequency mapping
        self.key_to_frequency = {}
        
        # frequency -> list of keys with that frequency
        self.frequency_to_keys = defaultdict(OrderedDict)
    
    def get(self, key: int) -> int:
        if key not in self.key_to_value:
            return -1
        
        # Update frequency
        self._update_frequency(key)
        
        return self.key_to_value[key]
    
    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return
        
        if key in self.key_to_value:
            # Update existing key
            self.key_to_value[key] = value
            self._update_frequency(key)
            return
        
        # Check if at capacity
        if len(self.key_to_value) >= self.capacity:
            self._evict_lfu()
        
        # Add new key
        self.key_to_value[key] = value
        self.key_to_frequency[key] = 1
        self.frequency_to_keys[1][key] = True
        self.min_frequency = 1
    
    def _update_frequency(self, key: int) -> None:
        """Update frequency of a key"""
        old_freq = self.key_to_frequency[key]
        new_freq = old_freq + 1
        
        # Remove from old frequency group
        del self.frequency_to_keys[old_freq][key]
        
        # Update min_frequency if needed
        if old_freq == self.min_frequency and not self.frequency_to_keys[old_freq]:
            self.min_frequency += 1
        
        # Add to new frequency group
        self.key_to_frequency[key] = new_freq
        self.frequency_to_keys[new_freq][key] = True
    
    def _evict_lfu(self) -> None:
        """Evict least frequently used key"""
        # Get least recently used key among least frequent keys
        key_to_evict, _ = self.frequency_to_keys[self.min_frequency].popitem(last=False)
        
        # Clean up
        del self.key_to_value[key_to_evict]
        del self.key_to_frequency[key_to_evict]

class LFUCacheOptimized:
    """
    Approach 2: Optimized LFU with Doubly Linked List
    
    Use doubly linked list for each frequency group for better performance.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(capacity)
    """
    
    class Node:
        def __init__(self, key: int = 0, value: int = 0):
            self.key = key
            self.value = value
            self.frequency = 1
            self.prev: Optional['LFUCacheOptimized.Node'] = None
            self.next: Optional['LFUCacheOptimized.Node'] = None
    
    class DoublyLinkedList:
        def __init__(self):
            self.head = LFUCacheOptimized.Node()
            self.tail = LFUCacheOptimized.Node()
            self.head.next = self.tail
            self.tail.prev = self.head
            self.size = 0
        
        def add_node(self, node: 'LFUCacheOptimized.Node') -> None:
            """Add node right after head"""
            node.prev = self.head
            node.next = self.head.next
            
            self.head.next.prev = node
            self.head.next = node
            self.size += 1
        
        def remove_node(self, node: 'LFUCacheOptimized.Node') -> None:
            """Remove node from list"""
            prev_node = node.prev
            next_node = node.next
            
            prev_node.next = next_node
            next_node.prev = prev_node
            self.size -= 1
        
        def remove_tail(self) -> 'LFUCacheOptimized.Node':
            """Remove node before tail"""
            last_node = self.tail.prev
            self.remove_node(last_node)
            return last_node
        
        def is_empty(self) -> bool:
            return self.size == 0
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_frequency = 0
        self.size = 0
        
        # key -> node mapping
        self.node_map = {}
        
        # frequency -> doubly linked list
        self.frequency_map = defaultdict(self.DoublyLinkedList)
    
    def get(self, key: int) -> int:
        if key not in self.node_map:
            return -1
        
        node = self.node_map[key]
        self._update_node(node)
        
        return node.value
    
    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return
        
        if key in self.node_map:
            # Update existing node
            node = self.node_map[key]
            node.value = value
            self._update_node(node)
            return
        
        # Check capacity
        if self.size >= self.capacity:
            self._evict_lfu()
        
        # Add new node
        new_node = self.Node(key, value)
        self.node_map[key] = new_node
        self.frequency_map[1].add_node(new_node)
        self.min_frequency = 1
        self.size += 1
    
    def _update_node(self, node: 'LFUCacheOptimized.Node') -> None:
        """Update node frequency"""
        old_freq = node.frequency
        new_freq = old_freq + 1
        
        # Remove from old frequency list
        self.frequency_map[old_freq].remove_node(node)
        
        # Update min_frequency if needed
        if old_freq == self.min_frequency and self.frequency_map[old_freq].is_empty():
            self.min_frequency += 1
        
        # Add to new frequency list
        node.frequency = new_freq
        self.frequency_map[new_freq].add_node(node)
    
    def _evict_lfu(self) -> None:
        """Evict least frequently used node"""
        # Remove LRU node from min frequency list
        lfu_list = self.frequency_map[self.min_frequency]
        node_to_evict = lfu_list.remove_tail()
        
        # Clean up
        del self.node_map[node_to_evict.key]
        self.size -= 1

class LFUCacheWithTimestamp:
    """
    Approach 3: LFU with Timestamp-based LRU
    
    Use timestamps to handle LRU among same frequency items.
    
    Time Complexity: O(1) for get and put operations
    Space Complexity: O(capacity)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.timestamp = 0
        
        # key -> (value, frequency, last_access_time)
        self.cache = {}
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        
        value, frequency, _ = self.cache[key]
        self.timestamp += 1
        
        # Update frequency and timestamp
        self.cache[key] = (value, frequency + 1, self.timestamp)
        
        return value
    
    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return
        
        self.timestamp += 1
        
        if key in self.cache:
            # Update existing key
            _, frequency, _ = self.cache[key]
            self.cache[key] = (value, frequency + 1, self.timestamp)
            return
        
        # Check capacity
        if self.size >= self.capacity:
            self._evict_lfu()
        
        # Add new key
        self.cache[key] = (value, 1, self.timestamp)
        self.size += 1
    
    def _evict_lfu(self) -> None:
        """Evict least frequently used key (LRU among ties)"""
        min_frequency = float('inf')
        oldest_time = float('inf')
        key_to_evict = None
        
        for key, (_, frequency, access_time) in self.cache.items():
            if (frequency < min_frequency or 
                (frequency == min_frequency and access_time < oldest_time)):
                min_frequency = frequency
                oldest_time = access_time
                key_to_evict = key
        
        if key_to_evict is not None:
            del self.cache[key_to_evict]
            self.size -= 1

class LFUCacheWithAnalytics:
    """
    Approach 4: Enhanced LFU with Analytics
    
    Track access patterns and provide analytics.
    
    Time Complexity: O(1) for core operations
    Space Complexity: O(capacity)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_frequency = 0
        
        self.key_to_value = {}
        self.key_to_frequency = {}
        self.frequency_to_keys = defaultdict(OrderedDict)
        
        # Analytics
        self.total_accesses = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        self.frequency_distribution = defaultdict(int)
    
    def get(self, key: int) -> int:
        self.total_accesses += 1
        
        if key not in self.key_to_value:
            self.cache_misses += 1
            return -1
        
        self.cache_hits += 1
        self._update_frequency(key)
        
        return self.key_to_value[key]
    
    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return
        
        if key in self.key_to_value:
            self.key_to_value[key] = value
            self._update_frequency(key)
            return
        
        if len(self.key_to_value) >= self.capacity:
            self._evict_lfu()
            self.evictions += 1
        
        self.key_to_value[key] = value
        self.key_to_frequency[key] = 1
        self.frequency_to_keys[1][key] = True
        self.frequency_distribution[1] += 1
        self.min_frequency = 1
    
    def _update_frequency(self, key: int) -> None:
        old_freq = self.key_to_frequency[key]
        new_freq = old_freq + 1
        
        # Update frequency distribution
        self.frequency_distribution[old_freq] -= 1
        self.frequency_distribution[new_freq] += 1
        
        del self.frequency_to_keys[old_freq][key]
        
        if old_freq == self.min_frequency and not self.frequency_to_keys[old_freq]:
            self.min_frequency += 1
        
        self.key_to_frequency[key] = new_freq
        self.frequency_to_keys[new_freq][key] = True
    
    def _evict_lfu(self) -> None:
        key_to_evict, _ = self.frequency_to_keys[self.min_frequency].popitem(last=False)
        
        # Update frequency distribution
        freq = self.key_to_frequency[key_to_evict]
        self.frequency_distribution[freq] -= 1
        
        del self.key_to_value[key_to_evict]
        del self.key_to_frequency[key_to_evict]
    
    def get_analytics(self) -> Dict[str, any]:
        """Get cache analytics"""
        hit_rate = self.cache_hits / max(1, self.total_accesses)
        
        return {
            'total_accesses': self.total_accesses,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'current_size': len(self.key_to_value),
            'capacity': self.capacity,
            'min_frequency': self.min_frequency,
            'frequency_distribution': dict(self.frequency_distribution)
        }


def test_lfu_cache_basic():
    """Test basic LFU cache functionality"""
    print("=== Testing Basic LFU Cache Functionality ===")
    
    implementations = [
        ("Basic LFU", LFUCacheBasic),
        ("Optimized LFU", LFUCacheOptimized),
        ("Timestamp-based", LFUCacheWithTimestamp),
        ("With Analytics", LFUCacheWithAnalytics)
    ]
    
    for name, LFUClass in implementations:
        print(f"\n{name}:")
        
        lfu = LFUClass(2)
        
        # Test basic operations
        lfu.put(1, 1)
        lfu.put(2, 2)
        print(f"  get(1): {lfu.get(1)}")  # Returns 1, freq of 1 becomes 2
        
        lfu.put(3, 3)  # Evicts key 2 (freq 1, least recent)
        print(f"  get(2): {lfu.get(2)}")  # Returns -1 (not found)
        print(f"  get(3): {lfu.get(3)}")  # Returns 3, freq of 3 becomes 2
        
        lfu.put(4, 4)  # Evicts key 1 (freq 2, least recent)
        print(f"  get(1): {lfu.get(1)}")  # Returns -1 (not found)
        print(f"  get(3): {lfu.get(3)}")  # Returns 3
        print(f"  get(4): {lfu.get(4)}")  # Returns 4

def test_lfu_cache_edge_cases():
    """Test LFU cache edge cases"""
    print("\n=== Testing LFU Cache Edge Cases ===")
    
    # Test capacity 0
    print("Testing capacity 0:")
    lfu = LFUCacheBasic(0)
    lfu.put(1, 1)
    print(f"  get(1) with capacity 0: {lfu.get(1)}")  # Should return -1
    
    # Test capacity 1
    print(f"\nTesting capacity 1:")
    lfu = LFUCacheBasic(1)
    lfu.put(1, 1)
    print(f"  get(1): {lfu.get(1)}")  # Returns 1
    
    lfu.put(2, 2)  # Evicts key 1
    print(f"  get(1): {lfu.get(1)}")  # Returns -1
    print(f"  get(2): {lfu.get(2)}")  # Returns 2
    
    # Test updating existing key
    print(f"\nTesting key updates:")
    lfu.put(2, 20)  # Update value
    print(f"  get(2): {lfu.get(2)}")  # Returns 20

def test_frequency_tracking():
    """Test frequency tracking accuracy"""
    print("\n=== Testing Frequency Tracking ===")
    
    lfu = LFUCacheWithAnalytics(3)
    
    # Add items and access with different frequencies
    lfu.put(1, 1)  # freq: 1
    lfu.put(2, 2)  # freq: 1
    lfu.put(3, 3)  # freq: 1
    
    # Access patterns
    lfu.get(1)  # freq: 2
    lfu.get(1)  # freq: 3
    lfu.get(2)  # freq: 2
    
    analytics = lfu.get_analytics()
    
    print("Analytics after access pattern:")
    print(f"  Total accesses: {analytics['total_accesses']}")
    print(f"  Hit rate: {analytics['hit_rate']:.2%}")
    print(f"  Frequency distribution: {analytics['frequency_distribution']}")
    print(f"  Min frequency: {analytics['min_frequency']}")

def test_lfu_performance():
    """Test LFU cache performance"""
    print("\n=== Testing LFU Cache Performance ===")
    
    import time
    import random
    
    implementations = [
        ("Basic LFU", LFUCacheBasic),
        ("Optimized LFU", LFUCacheOptimized),
        ("Timestamp-based", LFUCacheWithTimestamp)
    ]
    
    capacity = 1000
    operations = 50000
    
    for name, LFUClass in implementations:
        lfu = LFUClass(capacity)
        
        # Warm up cache
        for i in range(capacity):
            lfu.put(i, i * 2)
        
        # Measure performance
        start_time = time.time()
        
        for _ in range(operations):
            operation = random.choice(['get', 'put'])
            key = random.randint(0, capacity * 2)
            
            if operation == 'get':
                lfu.get(key)
            else:
                lfu.put(key, random.randint(1, 1000))
        
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000
        
        print(f"  {name}: {elapsed:.2f}ms for {operations} operations")

def test_lfu_vs_lru():
    """Compare LFU vs LRU behavior"""
    print("\n=== Comparing LFU vs LRU Behavior ===")
    
    # Simulate access pattern where LFU should perform better
    lfu = LFUCacheBasic(3)
    
    # Popular items (accessed frequently)
    popular_items = [1, 2]
    # Less popular item
    less_popular = [3, 4, 5, 6]
    
    # Add initial items
    for item in [1, 2, 3]:
        lfu.put(item, item * 10)
    
    # Create access pattern: frequent access to 1 and 2
    access_pattern = []
    for _ in range(10):
        access_pattern.extend(popular_items * 3)  # Access 1,2 frequently
        access_pattern.extend([3])  # Access 3 occasionally
    
    print("Access pattern simulation:")
    for key in access_pattern[:20]:  # Show first 20 accesses
        value = lfu.get(key)
        print(f"  get({key}): {value}")
    
    # Now add new items (should evict least frequent, not least recent)
    print(f"\nAdding new items:")
    for new_item in [4, 5]:
        lfu.put(new_item, new_item * 10)
        print(f"  Added {new_item}")
        
        # Check what's still in cache
        for test_key in [1, 2, 3]:
            result = lfu.get(test_key)
            status = "present" if result != -1 else "evicted"
            print(f"    Key {test_key}: {status}")

def demonstrate_lfu_applications():
    """Demonstrate real-world LFU applications"""
    print("\n=== Demonstrating LFU Applications ===")
    
    # Application 1: Web page caching
    print("Application 1: Web Page Caching")
    page_cache = LFUCacheWithAnalytics(5)
    
    # Simulate page requests
    page_requests = [
        "/home", "/about", "/products", "/home", "/home",
        "/contact", "/about", "/home", "/products", "/blog",
        "/home", "/pricing", "/about"
    ]
    
    for page in page_requests:
        page_hash = hash(page) % 1000  # Simple hash for demo
        
        # Check cache
        cached_content = page_cache.get(page_hash)
        
        if cached_content == -1:
            print(f"  Loading page: {page}")
            # Simulate page generation
            content_id = hash(f"content_{page}") % 1000
            page_cache.put(page_hash, content_id)
        else:
            print(f"  Serving cached: {page}")
    
    analytics = page_cache.get_analytics()
    print(f"  Cache hit rate: {analytics['hit_rate']:.2%}")
    
    # Application 2: Database query caching
    print(f"\nApplication 2: Database Query Caching")
    query_cache = LFUCacheOptimized(4)
    
    queries = [
        "SELECT * FROM users WHERE active=1",  # Frequent
        "SELECT * FROM products WHERE category='electronics'",  # Frequent
        "SELECT * FROM orders WHERE date=today",  # Less frequent
        "SELECT * FROM users WHERE active=1",  # Repeat
        "SELECT COUNT(*) FROM sessions",  # New
        "SELECT * FROM products WHERE category='electronics'",  # Repeat
        "SELECT * FROM audit_logs WHERE level='ERROR'"  # New, should evict least frequent
    ]
    
    for query in queries:
        query_hash = hash(query) % 1000
        result = query_cache.get(query_hash)
        
        if result == -1:
            print(f"  Executing: {query[:30]}...")
            # Simulate query execution
            query_cache.put(query_hash, hash(f"result_{query}") % 1000)
        else:
            print(f"  Using cached result for: {query[:30]}...")

def benchmark_memory_usage():
    """Benchmark memory usage of different LFU implementations"""
    print("\n=== Benchmarking Memory Usage ===")
    
    implementations = [
        ("Basic LFU", LFUCacheBasic),
        ("Optimized LFU", LFUCacheOptimized),
        ("With Analytics", LFUCacheWithAnalytics)
    ]
    
    capacity = 1000
    
    for name, LFUClass in implementations:
        lfu = LFUClass(capacity)
        
        # Fill cache
        for i in range(capacity):
            lfu.put(i, i * i)
        
        # Rough memory estimation
        print(f"  {name}:")
        print(f"    Capacity: {capacity}")
        
        if hasattr(lfu, 'get_analytics'):
            analytics = lfu.get_analytics()
            print(f"    Current size: {analytics['current_size']}")

if __name__ == "__main__":
    test_lfu_cache_basic()
    test_lfu_cache_edge_cases()
    test_frequency_tracking()
    test_lfu_performance()
    test_lfu_vs_lru()
    demonstrate_lfu_applications()
    benchmark_memory_usage()

"""
LFU Cache Design demonstrates advanced caching strategies:

Core Approaches:
1. Basic LFU - HashMap + frequency tracking with OrderedDict
2. Optimized LFU - Doubly linked lists for O(1) operations
3. Timestamp-based - Simple implementation using timestamps
4. With Analytics - Enhanced version with usage analytics

Key Design Principles:
- O(1) time complexity for all operations
- Accurate frequency tracking
- LRU tie-breaking among same frequencies
- Memory efficiency

Real-world Applications:
- Database query result caching
- Web page caching systems
- CPU cache replacement policies
- Content delivery networks
- Application-level caching

LFU is particularly effective when access patterns
show clear frequency differences over time.
"""
