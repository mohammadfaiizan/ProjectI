"""
460. LFU Cache - Multiple Approaches
Difficulty: Hard

Design and implement a data structure for a Least Frequently Used (LFU) cache.

Implement the LFUCache class:
- LFUCache(int capacity) Initializes the object with the capacity of the data structure.
- int get(int key) Gets the value of the key if the key exists in the cache. Otherwise, returns -1.
- void put(int key, int value) Update the value of the key if the key exists. Otherwise, adds the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least frequently used key.

The functions get and put must each run in O(1) average time complexity.
"""

from typing import Dict, Optional
from collections import defaultdict, OrderedDict

class LFUCacheOptimal:
    """
    Approach 1: HashMap + Frequency Groups (Optimal)
    
    Use HashMap for O(1) access and frequency groups for LFU tracking.
    
    Time: O(1) for both get and put, Space: O(capacity)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_freq = 0
        
        # key -> value
        self.key_to_val = {}
        # key -> frequency
        self.key_to_freq = {}
        # frequency -> OrderedDict of keys (insertion order = LRU within same frequency)
        self.freq_to_keys = defaultdict(OrderedDict)
    
    def _update_freq(self, key: int) -> None:
        """Update frequency of a key"""
        freq = self.key_to_freq[key]
        
        # Remove from current frequency group
        del self.freq_to_keys[freq][key]
        
        # If this was the only key with min_freq, increment min_freq
        if freq == self.min_freq and not self.freq_to_keys[freq]:
            self.min_freq += 1
        
        # Add to new frequency group
        self.key_to_freq[key] = freq + 1
        self.freq_to_keys[freq + 1][key] = None
    
    def get(self, key: int) -> int:
        """Get value by key"""
        if key not in self.key_to_val:
            return -1
        
        # Update frequency
        self._update_freq(key)
        
        return self.key_to_val[key]
    
    def put(self, key: int, value: int) -> None:
        """Put key-value pair"""
        if self.capacity <= 0:
            return
        
        if key in self.key_to_val:
            # Update existing key
            self.key_to_val[key] = value
            self._update_freq(key)
        else:
            # Add new key
            if len(self.key_to_val) >= self.capacity:
                # Evict LFU key (first in min frequency group)
                lfu_key = next(iter(self.freq_to_keys[self.min_freq]))
                
                del self.key_to_val[lfu_key]
                del self.key_to_freq[lfu_key]
                del self.freq_to_keys[self.min_freq][lfu_key]
            
            # Add new key with frequency 1
            self.key_to_val[key] = value
            self.key_to_freq[key] = 1
            self.freq_to_keys[1][key] = None
            self.min_freq = 1


class LFUCacheSimple:
    """
    Approach 2: Simple Implementation with Counter
    
    Use simple counter approach (less efficient but easier to understand).
    
    Time: O(n) for eviction, Space: O(capacity)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.key_to_val = {}
        self.key_to_freq = {}
        self.key_to_time = {}  # For LRU tie-breaking
        self.time = 0
    
    def get(self, key: int) -> int:
        """Get value by key"""
        if key not in self.key_to_val:
            return -1
        
        # Update frequency and time
        self.key_to_freq[key] += 1
        self.key_to_time[key] = self.time
        self.time += 1
        
        return self.key_to_val[key]
    
    def put(self, key: int, value: int) -> None:
        """Put key-value pair"""
        if self.capacity <= 0:
            return
        
        if key in self.key_to_val:
            # Update existing key
            self.key_to_val[key] = value
            self.key_to_freq[key] += 1
            self.key_to_time[key] = self.time
            self.time += 1
        else:
            # Add new key
            if len(self.key_to_val) >= self.capacity:
                # Find LFU key (with earliest time for tie-breaking)
                lfu_key = min(self.key_to_freq.keys(), 
                             key=lambda k: (self.key_to_freq[k], self.key_to_time[k]))
                
                del self.key_to_val[lfu_key]
                del self.key_to_freq[lfu_key]
                del self.key_to_time[lfu_key]
            
            # Add new key
            self.key_to_val[key] = value
            self.key_to_freq[key] = 1
            self.key_to_time[key] = self.time
            self.time += 1


class LFUCacheDoublyLinkedList:
    """
    Approach 3: Doubly Linked List Implementation
    
    Use doubly linked list for each frequency group.
    
    Time: O(1) for both operations, Space: O(capacity)
    """
    
    class Node:
        def __init__(self, key: int = 0, value: int = 0):
            self.key = key
            self.value = value
            self.freq = 1
            self.prev = None
            self.next = None
    
    class DoublyLinkedList:
        def __init__(self):
            self.head = LFUCacheDoublyLinkedList.Node()
            self.tail = LFUCacheDoublyLinkedList.Node()
            self.head.next = self.tail
            self.tail.prev = self.head
            self.size = 0
        
        def add_node(self, node):
            """Add node after head"""
            node.prev = self.head
            node.next = self.head.next
            self.head.next.prev = node
            self.head.next = node
            self.size += 1
        
        def remove_node(self, node):
            """Remove node from list"""
            node.prev.next = node.next
            node.next.prev = node.prev
            self.size -= 1
        
        def remove_last(self):
            """Remove last node"""
            if self.size > 0:
                last_node = self.tail.prev
                self.remove_node(last_node)
                return last_node
            return None
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_freq = 0
        self.key_to_node = {}
        self.freq_to_list = defaultdict(self.DoublyLinkedList)
    
    def _update_freq(self, node):
        """Update frequency of a node"""
        old_freq = node.freq
        new_freq = old_freq + 1
        
        # Remove from old frequency list
        self.freq_to_list[old_freq].remove_node(node)
        
        # Update min_freq if necessary
        if old_freq == self.min_freq and self.freq_to_list[old_freq].size == 0:
            self.min_freq += 1
        
        # Add to new frequency list
        node.freq = new_freq
        self.freq_to_list[new_freq].add_node(node)
    
    def get(self, key: int) -> int:
        """Get value by key"""
        if key not in self.key_to_node:
            return -1
        
        node = self.key_to_node[key]
        self._update_freq(node)
        
        return node.value
    
    def put(self, key: int, value: int) -> None:
        """Put key-value pair"""
        if self.capacity <= 0:
            return
        
        if key in self.key_to_node:
            # Update existing key
            node = self.key_to_node[key]
            node.value = value
            self._update_freq(node)
        else:
            # Add new key
            if len(self.key_to_node) >= self.capacity:
                # Remove LFU node
                lfu_list = self.freq_to_list[self.min_freq]
                lfu_node = lfu_list.remove_last()
                del self.key_to_node[lfu_node.key]
            
            # Add new node
            new_node = self.Node(key, value)
            self.key_to_node[key] = new_node
            self.freq_to_list[1].add_node(new_node)
            self.min_freq = 1


def test_lfu_cache_implementations():
    """Test LFU cache implementations"""
    
    implementations = [
        ("Optimal", LFUCacheOptimal),
        ("Simple", LFUCacheSimple),
        ("Doubly Linked List", LFUCacheDoublyLinkedList),
    ]
    
    test_cases = [
        {
            "capacity": 2,
            "operations": ["put", "put", "get", "put", "get", "get", "put", "get", "get", "get"],
            "values": [(1,1), (2,2), 1, (3,3), 2, 3, (4,4), 1, 3, 4],
            "expected": [None, None, 1, None, -1, 3, None, -1, 3, 4],
            "description": "Example 1"
        },
        {
            "capacity": 0,
            "operations": ["put", "get"],
            "values": [(1,1), 1],
            "expected": [None, -1],
            "description": "Zero capacity"
        },
        {
            "capacity": 1,
            "operations": ["put", "get", "put", "get", "get"],
            "values": [(1,1), 1, (2,2), 1, 2],
            "expected": [None, 1, None, -1, 2],
            "description": "Capacity 1"
        },
    ]
    
    print("=== Testing LFU Cache Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- {impl_name} Implementation ---")
        
        for test_case in test_cases:
            try:
                cache = impl_class(test_case["capacity"])
                results = []
                
                for i, op in enumerate(test_case["operations"]):
                    if op == "put":
                        key, value = test_case["values"][i]
                        cache.put(key, value)
                        results.append(None)
                    elif op == "get":
                        key = test_case["values"][i]
                        result = cache.get(key)
                        results.append(result)
                
                expected = test_case["expected"]
                status = "✓" if results == expected else "✗"
                
                print(f"  {test_case['description']:15} | {status} | {results}")
                if results != expected:
                    print(f"    Expected: {expected}")
                
            except Exception as e:
                print(f"  {test_case['description']:15} | ERROR: {str(e)[:40]}")


def demonstrate_lfu_mechanism():
    """Demonstrate LFU mechanism step by step"""
    print("\n=== LFU Mechanism Step-by-Step Demo ===")
    
    cache = LFUCacheOptimal(2)
    
    operations = [
        ("put", (1, 1)),
        ("put", (2, 2)),
        ("get", 1),       # freq: 1->2, 2->1
        ("put", (3, 3)),  # evict 2 (freq=1)
        ("get", 2),       # should return -1
        ("get", 3),       # freq: 3->2
        ("put", (4, 4)),  # evict 1 (freq=2, but older)
        ("get", 1),       # should return -1
        ("get", 3),       # should return 3
        ("get", 4),       # should return 4
    ]
    
    print("Strategy: Track frequency groups with OrderedDict for LRU within same frequency")
    print(f"Capacity: 2")
    
    def print_cache_state():
        """Helper to print current cache state"""
        print(f"  Keys: {list(cache.key_to_val.keys())}")
        print(f"  Frequencies: {dict(cache.key_to_freq)}")
        print(f"  Min frequency: {cache.min_freq}")
        freq_groups = {}
        for freq, keys in cache.freq_to_keys.items():
            if keys:
                freq_groups[freq] = list(keys.keys())
        print(f"  Frequency groups: {freq_groups}")
    
    print(f"\nInitial state:")
    print_cache_state()
    
    for i, (op, value) in enumerate(operations):
        print(f"\nStep {i+1}: {op}({value})")
        
        if op == "put":
            key, val = value
            cache.put(key, val)
            print(f"  Put key {key} with value {val}")
        elif op == "get":
            result = cache.get(value)
            print(f"  Get key {value} -> {result}")
        
        print_cache_state()


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: CDN cache
    print("1. CDN (Content Delivery Network) Cache:")
    cdn_cache = LFUCacheOptimal(3)
    
    content_requests = [
        ("request", "video1.mp4", "Popular video content"),
        ("request", "image1.jpg", "Profile image"),
        ("request", "video2.mp4", "Another video"),
        ("request", "video1.mp4", None),  # Popular content accessed again
        ("request", "document1.pdf", "New document"),  # Should evict image1.jpg (LFU)
        ("request", "image1.jpg", None),  # Should be cache miss
        ("request", "video1.mp4", None),  # Should be cache hit (most frequent)
    ]
    
    print("  CDN request simulation:")
    for action, content, description in content_requests:
        content_hash = hash(content) % 1000
        
        if action == "request":
            cached = cdn_cache.get(content_hash)
            
            if cached != -1:
                print(f"    Request {content} -> Cache HIT (served from edge)")
            else:
                print(f"    Request {content} -> Cache MISS (fetch from origin)")
                if description:  # New content
                    cdn_cache.put(content_hash, content_hash)
    
    # Application 2: Database query result cache
    print(f"\n2. Database Query Result Cache:")
    db_cache = LFUCacheOptimal(2)
    
    queries = [
        "SELECT * FROM users WHERE active = 1",  # Frequent query
        "SELECT * FROM products WHERE price > 100",
        "SELECT * FROM users WHERE active = 1",  # Frequent query again
        "SELECT * FROM orders WHERE date = TODAY()",  # Should evict products query
        "SELECT * FROM products WHERE price > 100",  # Cache miss
    ]
    
    print("  Database query caching:")
    for i, query in enumerate(queries):
        query_hash = hash(query) % 1000
        result = db_cache.get(query_hash)
        
        if result != -1:
            print(f"    Query {i+1}: Cache HIT (frequency increased)")
        else:
            print(f"    Query {i+1}: Cache MISS (execute and cache)")
            db_cache.put(query_hash, i * 100)


if __name__ == "__main__":
    test_lfu_cache_implementations()
    demonstrate_lfu_mechanism()
    demonstrate_real_world_applications()

"""
LFU Cache demonstrates advanced system design with frequency-based
eviction policies, including multiple implementation approaches
for efficient least frequently used cache management.
"""
