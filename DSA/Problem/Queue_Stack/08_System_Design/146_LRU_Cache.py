"""
146. LRU Cache - Multiple Approaches
Difficulty: Medium

Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:
- LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
- int get(int key) Return the value of the key if the key exists, otherwise return -1.
- void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.

The functions get and put must each run in O(1) average time complexity.
"""

from typing import Dict, Optional

class LRUCacheHashMapDoublyLinkedList:
    """
    Approach 1: HashMap + Doubly Linked List (Optimal)
    
    Use HashMap for O(1) access and Doubly Linked List for O(1) insertion/deletion.
    
    Time: O(1) for both get and put, Space: O(capacity)
    """
    
    class Node:
        def __init__(self, key: int = 0, value: int = 0):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> node
        
        # Create dummy head and tail nodes
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node: 'Node') -> None:
        """Add node right after head"""
        node.prev = self.head
        node.next = self.head.next
        
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node: 'Node') -> None:
        """Remove an existing node from the linked list"""
        prev_node = node.prev
        next_node = node.next
        
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _move_to_head(self, node: 'Node') -> None:
        """Move node to head (mark as recently used)"""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self) -> 'Node':
        """Pop the last node (least recently used)"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node
    
    def get(self, key: int) -> int:
        """Get value by key"""
        node = self.cache.get(key)
        
        if not node:
            return -1
        
        # Move to head (mark as recently used)
        self._move_to_head(node)
        
        return node.value
    
    def put(self, key: int, value: int) -> None:
        """Put key-value pair"""
        node = self.cache.get(key)
        
        if not node:
            # New key
            new_node = self.Node(key, value)
            
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                tail = self._pop_tail()
                del self.cache[tail.key]
            
            # Add new node
            self.cache[key] = new_node
            self._add_node(new_node)
        else:
            # Update existing key
            node.value = value
            self._move_to_head(node)


class LRUCacheOrderedDict:
    """
    Approach 2: OrderedDict Implementation
    
    Use Python's OrderedDict which maintains insertion order.
    
    Time: O(1) for both operations, Space: O(capacity)
    """
    
    def __init__(self, capacity: int):
        from collections import OrderedDict
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: int) -> int:
        """Get value by key"""
        if key not in self.cache:
            return -1
        
        # Move to end (mark as recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        """Put key-value pair"""
        if key in self.cache:
            # Update existing key
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Add new key
            if len(self.cache) >= self.capacity:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)
            
            self.cache[key] = value


class LRUCacheArrayBased:
    """
    Approach 3: Array-based Implementation
    
    Use array to maintain order and dictionary for fast access.
    
    Time: O(n) for operations due to array manipulation, Space: O(capacity)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.keys = []  # Maintain order (most recent at end)
        self.values = {}  # key -> value mapping
    
    def get(self, key: int) -> int:
        """Get value by key"""
        if key not in self.values:
            return -1
        
        # Move key to end (mark as recently used)
        self.keys.remove(key)
        self.keys.append(key)
        
        return self.values[key]
    
    def put(self, key: int, value: int) -> None:
        """Put key-value pair"""
        if key in self.values:
            # Update existing key
            self.values[key] = value
            self.keys.remove(key)
            self.keys.append(key)
        else:
            # Add new key
            if len(self.keys) >= self.capacity:
                # Remove least recently used (first in list)
                lru_key = self.keys.pop(0)
                del self.values[lru_key]
            
            self.keys.append(key)
            self.values[key] = value


class LRUCacheDeque:
    """
    Approach 4: Deque-based Implementation
    
    Use deque for order maintenance and dictionary for access.
    
    Time: O(n) for get due to deque search, Space: O(capacity)
    """
    
    def __init__(self, capacity: int):
        from collections import deque
        self.capacity = capacity
        self.keys = deque()  # Maintain order
        self.values = {}  # key -> value mapping
    
    def get(self, key: int) -> int:
        """Get value by key"""
        if key not in self.values:
            return -1
        
        # Move key to end (mark as recently used)
        self.keys.remove(key)  # O(n) operation
        self.keys.append(key)
        
        return self.values[key]
    
    def put(self, key: int, value: int) -> None:
        """Put key-value pair"""
        if key in self.values:
            # Update existing key
            self.values[key] = value
            self.keys.remove(key)  # O(n) operation
            self.keys.append(key)
        else:
            # Add new key
            if len(self.keys) >= self.capacity:
                # Remove least recently used
                lru_key = self.keys.popleft()
                del self.values[lru_key]
            
            self.keys.append(key)
            self.values[key] = value


def test_lru_cache_implementations():
    """Test LRU cache implementations"""
    
    implementations = [
        ("HashMap + DLL", LRUCacheHashMapDoublyLinkedList),
        ("OrderedDict", LRUCacheOrderedDict),
        ("Array-based", LRUCacheArrayBased),
        ("Deque-based", LRUCacheDeque),
    ]
    
    test_cases = [
        {
            "capacity": 2,
            "operations": ["put", "put", "get", "put", "get", "put", "get", "get", "get"],
            "values": [(1,1), (2,2), 1, (3,3), 2, (4,4), 1, 3, 4],
            "expected": [None, None, 1, None, -1, None, -1, 3, 4],
            "description": "Example 1"
        },
        {
            "capacity": 1,
            "operations": ["put", "get", "put", "get"],
            "values": [(1,1), 1, (2,2), 2],
            "expected": [None, 1, None, 2],
            "description": "Capacity 1"
        },
        {
            "capacity": 3,
            "operations": ["put", "put", "put", "get", "put", "get", "get", "get"],
            "values": [(1,1), (2,2), (3,3), 2, (4,4), 1, 2, 4],
            "expected": [None, None, None, 2, None, -1, 2, 4],
            "description": "Capacity 3"
        },
    ]
    
    print("=== Testing LRU Cache Implementations ===")
    
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


def demonstrate_lru_mechanism():
    """Demonstrate LRU mechanism step by step"""
    print("\n=== LRU Mechanism Step-by-Step Demo ===")
    
    cache = LRUCacheHashMapDoublyLinkedList(2)
    
    operations = [
        ("put", (1, 1)),
        ("put", (2, 2)),
        ("get", 1),
        ("put", (3, 3)),  # Should evict key 2
        ("get", 2),       # Should return -1
        ("put", (4, 4)),  # Should evict key 1
        ("get", 1),       # Should return -1
        ("get", 3),       # Should return 3
        ("get", 4),       # Should return 4
    ]
    
    print("Strategy: HashMap + Doubly Linked List for O(1) operations")
    print(f"Capacity: 2")
    
    def print_cache_state():
        """Helper to print current cache state"""
        keys = []
        current = cache.head.next
        while current != cache.tail:
            keys.append(f"{current.key}:{current.value}")
            current = current.next
        print(f"  Cache state: [{' -> '.join(keys)}] (head -> tail)")
    
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


def visualize_doubly_linked_list():
    """Visualize doubly linked list operations"""
    print("\n=== Doubly Linked List Operations Visualization ===")
    
    cache = LRUCacheHashMapDoublyLinkedList(3)
    
    operations = [
        ("put", (1, "A")),
        ("put", (2, "B")),
        ("put", (3, "C")),
        ("get", 1),  # Move 1 to head
        ("put", (4, "D")),  # Evict 2
    ]
    
    def visualize_list():
        """Visualize the doubly linked list"""
        nodes = []
        current = cache.head
        
        while current:
            if current == cache.head:
                nodes.append("HEAD")
            elif current == cache.tail:
                nodes.append("TAIL")
            else:
                nodes.append(f"{current.key}:{current.value}")
            current = current.next
        
        print(f"  List: {' <-> '.join(nodes)}")
    
    print("Visualizing doubly linked list structure:")
    
    for op, value in operations:
        print(f"\nOperation: {op}({value})")
        
        if op == "put":
            key, val = value
            cache.put(key, val)
        elif op == "get":
            result = cache.get(value)
            print(f"  Returned: {result}")
        
        visualize_list()


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Web page cache
    print("1. Web Page Cache:")
    page_cache = LRUCacheOrderedDict(3)
    
    web_requests = [
        ("visit", "home.html", "Home page content"),
        ("visit", "about.html", "About page content"),
        ("visit", "contact.html", "Contact page content"),
        ("visit", "home.html", None),  # Cache hit
        ("visit", "products.html", "Products page content"),  # Should evict about.html
        ("visit", "about.html", "About page content"),  # Cache miss, should evict contact.html
    ]
    
    print("  Web browsing simulation:")
    for action, page, content in web_requests:
        if action == "visit":
            cached_content = page_cache.get(hash(page) % 1000)  # Simulate key
            
            if cached_content != -1:
                print(f"    Visit {page} -> Cache HIT")
            else:
                print(f"    Visit {page} -> Cache MISS, loading content")
                if content:
                    page_cache.put(hash(page) % 1000, hash(content) % 1000)
    
    # Application 2: Database query cache
    print(f"\n2. Database Query Cache:")
    query_cache = LRUCacheHashMapDoublyLinkedList(2)
    
    queries = [
        "SELECT * FROM users WHERE id = 1",
        "SELECT * FROM products WHERE category = 'electronics'",
        "SELECT * FROM users WHERE id = 1",  # Cache hit
        "SELECT * FROM orders WHERE date > '2023-01-01'",  # Should evict products query
        "SELECT * FROM products WHERE category = 'electronics'",  # Cache miss
    ]
    
    print("  Database query simulation:")
    for i, query in enumerate(queries):
        query_hash = hash(query) % 1000
        result = query_cache.get(query_hash)
        
        if result != -1:
            print(f"    Query {i+1}: Cache HIT")
        else:
            print(f"    Query {i+1}: Cache MISS, executing query")
            # Simulate query result
            query_cache.put(query_hash, i * 100)
    
    # Application 3: Image cache for mobile app
    print(f"\n3. Mobile App Image Cache:")
    image_cache = LRUCacheOrderedDict(4)
    
    image_requests = [
        "profile_pic_1.jpg",
        "banner_ad_1.jpg", 
        "product_img_1.jpg",
        "profile_pic_2.jpg",
        "profile_pic_1.jpg",  # Cache hit
        "banner_ad_2.jpg",    # Should evict banner_ad_1.jpg
        "product_img_2.jpg",  # Should evict product_img_1.jpg
    ]
    
    print("  Image loading simulation:")
    for img in image_requests:
        img_hash = hash(img) % 1000
        cached_img = image_cache.get(img_hash)
        
        if cached_img != -1:
            print(f"    Load {img} -> From cache")
        else:
            print(f"    Load {img} -> Download from server")
            image_cache.put(img_hash, img_hash)


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("HashMap + DLL", "O(1)", "O(1)", "O(capacity)", "Optimal solution"),
        ("OrderedDict", "O(1)", "O(1)", "O(capacity)", "Python built-in optimization"),
        ("Array-based", "O(n)", "O(n)", "O(capacity)", "Array manipulation overhead"),
        ("Deque-based", "O(n)", "O(n)", "O(capacity)", "Deque search overhead"),
    ]
    
    print(f"{'Approach':<15} | {'Get':<8} | {'Put':<8} | {'Space':<12} | {'Notes'}")
    print("-" * 70)
    
    for approach, get_time, put_time, space, notes in approaches:
        print(f"{approach:<15} | {get_time:<8} | {put_time:<8} | {space:<12} | {notes}")
    
    print(f"\nHashMap + Doubly Linked List is the optimal approach")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    cache = LRUCacheHashMapDoublyLinkedList(2)
    
    edge_cases = [
        ("Get from empty cache", lambda: cache.get(1), -1),
        ("Put then get same key", lambda: (cache.put(1, 1), cache.get(1))[1], 1),
        ("Update existing key", lambda: (cache.put(1, 2), cache.get(1))[1], 2),
        ("Fill to capacity", lambda: (cache.put(2, 2), len(cache.cache)), 2),
        ("Exceed capacity", lambda: (cache.put(3, 3), cache.get(1))[1], -1),  # 1 should be evicted
    ]
    
    for description, operation, expected in edge_cases:
        try:
            # Reset cache for independent tests
            if "empty" in description:
                cache = LRUCacheHashMapDoublyLinkedList(2)
            
            result = operation()
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | Result: {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def benchmark_implementations():
    """Benchmark different implementations"""
    import time
    import random
    
    implementations = [
        ("HashMap + DLL", LRUCacheHashMapDoublyLinkedList),
        ("OrderedDict", LRUCacheOrderedDict),
    ]
    
    capacity = 1000
    n_operations = 10000
    
    print(f"\n=== Performance Benchmark ===")
    print(f"Capacity: {capacity}, Operations: {n_operations}")
    
    for impl_name, impl_class in implementations:
        try:
            cache = impl_class(capacity)
            
            start_time = time.time()
            
            # Mixed operations
            for i in range(n_operations):
                if random.random() < 0.7:  # 70% puts
                    cache.put(random.randint(1, capacity * 2), i)
                else:  # 30% gets
                    cache.get(random.randint(1, capacity * 2))
            
            end_time = time.time()
            
            print(f"{impl_name:15} | Time: {end_time - start_time:.4f}s")
            
        except Exception as e:
            print(f"{impl_name:15} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_lru_cache_implementations()
    demonstrate_lru_mechanism()
    visualize_doubly_linked_list()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    test_edge_cases()
    benchmark_implementations()

"""
LRU Cache demonstrates advanced system design with hash maps and
doubly linked lists, including multiple implementation approaches
for efficient cache management and real-world caching applications.
"""