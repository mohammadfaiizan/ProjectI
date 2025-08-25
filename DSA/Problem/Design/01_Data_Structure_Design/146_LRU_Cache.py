"""
146. LRU Cache - Multiple Approaches
Difficulty: Medium

Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:
- LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
- int get(int key) Return the value of the key if the key exists, otherwise return -1.
- void put(int key, int value) Update the value of the key if the key exists. 
  Otherwise, add the key-value pair to the cache. If the number of keys exceeds 
  the capacity from this operation, evict the least recently used key.

The functions get and put must each run in O(1) average time complexity.
"""

from typing import Optional, Dict, Any
from collections import OrderedDict
import time

class LRUCacheHashMapList:
    """
    Approach 1: HashMap + Doubly Linked List
    
    Use a combination of hash map and doubly linked list to achieve O(1) operations.
    
    Time Complexity: O(1) for both get and put operations
    Space Complexity: O(capacity)
    """
    
    class Node:
        def __init__(self, key: int = 0, value: int = 0):
            self.key = key
            self.value = value
            self.prev: Optional['LRUCacheHashMapList.Node'] = None
            self.next: Optional['LRUCacheHashMapList.Node'] = None
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[int, 'LRUCacheHashMapList.Node'] = {}
        
        # Create dummy head and tail nodes
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node: Node) -> None:
        """Add node right after head"""
        node.prev = self.head
        node.next = self.head.next
        
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node: Node) -> None:
        """Remove an existing node from the linked list"""
        prev_node = node.prev
        next_node = node.next
        
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _move_to_head(self, node: Node) -> None:
        """Move certain node to head"""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self) -> Node:
        """Pop the current tail"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node
    
    def get(self, key: int) -> int:
        node = self.cache.get(key)
        
        if node:
            # Move the accessed node to the head
            self._move_to_head(node)
            return node.value
        
        return -1
    
    def put(self, key: int, value: int) -> None:
        node = self.cache.get(key)
        
        if node:
            # Update the value and move to head
            node.value = value
            self._move_to_head(node)
        else:
            new_node = self.Node(key, value)
            
            if len(self.cache) >= self.capacity:
                # Remove LRU item
                tail = self._pop_tail()
                del self.cache[tail.key]
            
            # Add new node
            self.cache[key] = new_node
            self._add_node(new_node)

class LRUCacheOrderedDict:
    """
    Approach 2: Using OrderedDict
    
    Python's OrderedDict maintains insertion order and provides O(1) operations.
    
    Time Complexity: O(1) for both get and put operations
    Space Complexity: O(capacity)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: int) -> int:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update and move to end
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)
            
            self.cache[key] = value

class LRUCacheWithTimestamp:
    """
    Approach 3: HashMap with Timestamp
    
    Use timestamps to track access order. Less efficient but easier to understand.
    
    Time Complexity: O(n) for put when eviction is needed, O(1) for get
    Space Complexity: O(capacity)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[int, tuple] = {}  # key -> (value, timestamp)
        self.timestamp = 0
    
    def get(self, key: int) -> int:
        if key in self.cache:
            value, _ = self.cache[key]
            self.timestamp += 1
            self.cache[key] = (value, self.timestamp)
            return value
        return -1
    
    def put(self, key: int, value: int) -> None:
        self.timestamp += 1
        
        if key in self.cache:
            self.cache[key] = (value, self.timestamp)
        else:
            if len(self.cache) >= self.capacity:
                # Find LRU item (smallest timestamp)
                lru_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[lru_key]
            
            self.cache[key] = (value, self.timestamp)

class LRUCacheWithFrequency:
    """
    Approach 4: Advanced LRU with Access Frequency Tracking
    
    Enhanced version that also tracks access frequency for better eviction decisions.
    
    Time Complexity: O(1) for both operations
    Space Complexity: O(capacity)
    """
    
    class Node:
        def __init__(self, key: int = 0, value: int = 0):
            self.key = key
            self.value = value
            self.frequency = 1
            self.last_access = 0
            self.prev: Optional['LRUCacheWithFrequency.Node'] = None
            self.next: Optional['LRUCacheWithFrequency.Node'] = None
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[int, 'LRUCacheWithFrequency.Node'] = {}
        self.access_counter = 0
        
        # Create dummy head and tail
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node: Node) -> None:
        """Add node right after head"""
        node.prev = self.head
        node.next = self.head.next
        
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node: Node) -> None:
        """Remove node from linked list"""
        prev_node = node.prev
        next_node = node.next
        
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _move_to_head(self, node: Node) -> None:
        """Move node to head"""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self) -> Node:
        """Remove tail node"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node
    
    def get(self, key: int) -> int:
        node = self.cache.get(key)
        
        if node:
            # Update access statistics
            node.frequency += 1
            self.access_counter += 1
            node.last_access = self.access_counter
            
            # Move to head
            self._move_to_head(node)
            return node.value
        
        return -1
    
    def put(self, key: int, value: int) -> None:
        node = self.cache.get(key)
        
        if node:
            # Update existing node
            node.value = value
            node.frequency += 1
            self.access_counter += 1
            node.last_access = self.access_counter
            self._move_to_head(node)
        else:
            new_node = self.Node(key, value)
            self.access_counter += 1
            new_node.last_access = self.access_counter
            
            if len(self.cache) >= self.capacity:
                # Remove LRU node
                tail = self._pop_tail()
                del self.cache[tail.key]
            
            # Add new node
            self.cache[key] = new_node
            self._add_node(new_node)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.cache:
            return {"size": 0, "avg_frequency": 0}
        
        total_frequency = sum(node.frequency for node in self.cache.values())
        return {
            "size": len(self.cache),
            "capacity": self.capacity,
            "avg_frequency": total_frequency / len(self.cache),
            "total_accesses": self.access_counter
        }


def test_lru_cache_basic():
    """Test basic LRU cache functionality"""
    print("=== Testing Basic LRU Cache Functionality ===")
    
    # Test all implementations
    implementations = [
        ("HashMap + Doubly Linked List", LRUCacheHashMapList),
        ("OrderedDict", LRUCacheOrderedDict),
        ("Timestamp-based", LRUCacheWithTimestamp),
        ("With Frequency Tracking", LRUCacheWithFrequency)
    ]
    
    for name, LRUClass in implementations:
        print(f"\n{name}:")
        
        lru = LRUClass(2)
        
        # Test operations
        lru.put(1, 1)
        lru.put(2, 2)
        print(f"  get(1): {lru.get(1)}")  # Returns 1
        
        lru.put(3, 3)  # Evicts key 2
        print(f"  get(2): {lru.get(2)}")  # Returns -1 (not found)
        
        lru.put(4, 4)  # Evicts key 1
        print(f"  get(1): {lru.get(1)}")  # Returns -1 (not found)
        print(f"  get(3): {lru.get(3)}")  # Returns 3
        print(f"  get(4): {lru.get(4)}")  # Returns 4

def test_lru_cache_edge_cases():
    """Test LRU cache edge cases"""
    print("\n=== Testing LRU Cache Edge Cases ===")
    
    lru = LRUCacheHashMapList(1)
    
    # Test capacity 1
    print("Testing capacity 1:")
    lru.put(1, 1)
    print(f"  get(1): {lru.get(1)}")  # Returns 1
    
    lru.put(2, 2)  # Evicts key 1
    print(f"  get(1): {lru.get(1)}")  # Returns -1
    print(f"  get(2): {lru.get(2)}")  # Returns 2
    
    # Test updating existing key
    print(f"\nTesting key updates:")
    lru.put(2, 20)  # Update value
    print(f"  get(2): {lru.get(2)}")  # Returns 20

def test_lru_cache_performance():
    """Test LRU cache performance"""
    print("\n=== Testing LRU Cache Performance ===")
    
    import time
    import random
    
    implementations = [
        ("HashMap + Doubly Linked List", LRUCacheHashMapList),
        ("OrderedDict", LRUCacheOrderedDict),
        ("Timestamp-based", LRUCacheWithTimestamp)
    ]
    
    capacity = 1000
    operations = 10000
    
    for name, LRUClass in implementations:
        lru = LRUClass(capacity)
        
        # Warm up cache
        for i in range(capacity):
            lru.put(i, i * 2)
        
        # Measure performance
        start_time = time.time()
        
        for _ in range(operations):
            operation = random.choice(['get', 'put'])
            key = random.randint(0, capacity * 2)
            
            if operation == 'get':
                lru.get(key)
            else:
                lru.put(key, random.randint(1, 1000))
        
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000
        
        print(f"  {name}: {elapsed:.2f}ms for {operations} operations")

def test_advanced_lru_features():
    """Test advanced LRU cache features"""
    print("\n=== Testing Advanced LRU Features ===")
    
    lru = LRUCacheWithFrequency(3)
    
    # Test frequency tracking
    lru.put(1, 1)
    lru.put(2, 2)
    lru.put(3, 3)
    
    # Access patterns
    lru.get(1)  # frequency: 1
    lru.get(1)  # frequency: 2
    lru.get(2)  # frequency: 1
    
    stats = lru.get_cache_stats()
    print(f"Cache statistics:")
    print(f"  Size: {stats['size']}/{stats['capacity']}")
    print(f"  Average frequency: {stats['avg_frequency']:.2f}")
    print(f"  Total accesses: {stats['total_accesses']}")

def demonstrate_lru_cache_patterns():
    """Demonstrate LRU cache usage patterns"""
    print("\n=== Demonstrating LRU Cache Patterns ===")
    
    # Pattern 1: Web page caching
    print("Pattern 1: Web Page Caching")
    page_cache = LRUCacheHashMapList(3)
    
    pages = ["/home", "/about", "/contact", "/products", "/home"]
    
    for page in pages:
        # Simulate page request
        cached_content = page_cache.get(hash(page) % 1000)
        
        if cached_content == -1:
            print(f"  Loading page: {page}")
            # Simulate page generation
            content_id = hash(f"content_{page}") % 1000
            page_cache.put(hash(page) % 1000, content_id)
        else:
            print(f"  Serving cached: {page}")
    
    # Pattern 2: Database query caching
    print(f"\nPattern 2: Database Query Caching")
    query_cache = LRUCacheOrderedDict(5)
    
    queries = [
        "SELECT * FROM users WHERE id = 1",
        "SELECT * FROM products WHERE category = 'electronics'",
        "SELECT * FROM orders WHERE user_id = 1",
        "SELECT * FROM users WHERE id = 1",  # Cache hit
        "SELECT * FROM reviews WHERE product_id = 1"
    ]
    
    for query in queries:
        query_hash = hash(query) % 1000
        result = query_cache.get(query_hash)
        
        if result == -1:
            print(f"  Executing query: {query[:30]}...")
            # Simulate query execution
            query_cache.put(query_hash, hash(f"result_{query}") % 1000)
        else:
            print(f"  Using cached result for: {query[:30]}...")

def benchmark_lru_implementations():
    """Benchmark different LRU implementations"""
    print("\n=== Benchmarking LRU Implementations ===")
    
    implementations = [
        ("HashMap + Doubly Linked List", LRUCacheHashMapList),
        ("OrderedDict", LRUCacheOrderedDict),
        ("With Frequency Tracking", LRUCacheWithFrequency)
    ]
    
    test_sizes = [100, 1000, 5000]
    
    for size in test_sizes:
        print(f"\nCapacity: {size}")
        
        for name, LRUClass in implementations:
            lru = LRUClass(size)
            
            # Test put operations
            start_time = time.time()
            for i in range(size * 2):  # Trigger evictions
                lru.put(i, i * 2)
            put_time = (time.time() - start_time) * 1000
            
            # Test get operations
            start_time = time.time()
            hits = 0
            for i in range(size):
                if lru.get(i + size) != -1:
                    hits += 1
            get_time = (time.time() - start_time) * 1000
            
            print(f"  {name}:")
            print(f"    Put time: {put_time:.2f}ms")
            print(f"    Get time: {get_time:.2f}ms")
            print(f"    Cache hits: {hits}/{size}")

if __name__ == "__main__":
    test_lru_cache_basic()
    test_lru_cache_edge_cases()
    test_lru_cache_performance()
    test_advanced_lru_features()
    demonstrate_lru_cache_patterns()
    benchmark_lru_implementations()

"""
LRU Cache Design demonstrates key concepts:

Core Approaches:
1. HashMap + Doubly Linked List - Optimal O(1) solution
2. OrderedDict - Python-specific elegant solution
3. Timestamp-based - Conceptually simple but less efficient
4. Enhanced with frequency tracking - Advanced eviction strategies

Key Design Principles:
- Constant time operations through careful data structure choice
- Efficient memory management with capacity constraints
- Access pattern optimization for real-world scenarios

Real-world Applications:
- CPU cache management
- Web browser caching
- Database buffer pools
- Application-level caching layers
- CDN edge caching

The HashMap + Doubly Linked List approach is the gold standard,
providing true O(1) operations while maintaining LRU ordering.
"""
