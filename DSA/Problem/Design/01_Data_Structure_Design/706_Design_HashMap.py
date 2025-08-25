"""
706. Design HashMap - Multiple Approaches
Difficulty: Easy

Design a HashMap without using any built-in hash table libraries.

Implement the MyHashMap class:
- MyHashMap() initializes the object with an empty map.
- void put(int key, int value) inserts a (key, value) pair into the HashMap. 
  If the key already exists in the map, update the corresponding value.
- int get(int key) returns the value to which the specified key is mapped, 
  or -1 if this map contains no mapping for the key.
- void remove(int key) removes the key and its corresponding value if the map contains the mapping for the key.
"""

from typing import List, Optional, Tuple

class MyHashMapArray:
    """
    Approach 1: Simple Array Implementation
    
    Use a boolean array to track presence and value array for values.
    Assumes keys are in a limited range.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(range) where range is the maximum possible key value
    """
    
    def __init__(self):
        self.size = 1000001  # Assuming keys are <= 10^6
        self.data = [-1] * self.size  # -1 indicates empty slot
    
    def put(self, key: int, value: int) -> None:
        if 0 <= key < self.size:
            self.data[key] = value
    
    def get(self, key: int) -> int:
        if 0 <= key < self.size:
            return self.data[key]
        return -1
    
    def remove(self, key: int) -> None:
        if 0 <= key < self.size:
            self.data[key] = -1

class MyHashMapChaining:
    """
    Approach 2: Hash Table with Chaining
    
    Use separate chaining to handle collisions.
    
    Time Complexity: O(1) average, O(n) worst case for all operations
    Space Complexity: O(n) where n is the number of key-value pairs
    """
    
    def __init__(self):
        self.size = 1000
        self.buckets = [[] for _ in range(self.size)]
    
    def _hash(self, key: int) -> int:
        return key % self.size
    
    def put(self, key: int, value: int) -> None:
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        # Check if key already exists
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)  # Update existing
                return
        
        # Add new key-value pair
        bucket.append((key, value))
    
    def get(self, key: int) -> int:
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        return -1
    
    def remove(self, key: int) -> None:
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                return

class MyHashMapOpenAddressing:
    """
    Approach 3: Open Addressing with Linear Probing
    
    Use linear probing to handle collisions.
    
    Time Complexity: O(1) average case for all operations
    Space Complexity: O(capacity)
    """
    
    def __init__(self):
        self.capacity = 1000
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.deleted = [False] * self.capacity
        
    def _hash(self, key: int) -> int:
        return key % self.capacity
    
    def _find_slot(self, key: int) -> int:
        """Find slot for key using linear probing"""
        index = self._hash(key)
        original_index = index
        
        while (self.keys[index] is not None and 
               self.keys[index] != key and 
               not self.deleted[index]):
            index = (index + 1) % self.capacity
            
            # Prevent infinite loop
            if index == original_index:
                return -1
        
        return index
    
    def put(self, key: int, value: int) -> None:
        if self.size >= self.capacity * 0.7:  # Load factor threshold
            self._resize()
        
        index = self._find_slot(key)
        if index != -1:
            if self.keys[index] != key:
                self.size += 1
            
            self.keys[index] = key
            self.values[index] = value
            self.deleted[index] = False
    
    def get(self, key: int) -> int:
        index = self._find_slot(key)
        if (index != -1 and 
            self.keys[index] == key and 
            not self.deleted[index]):
            return self.values[index]
        return -1
    
    def remove(self, key: int) -> None:
        index = self._find_slot(key)
        if (index != -1 and 
            self.keys[index] == key and 
            not self.deleted[index]):
            self.deleted[index] = True
            self.size -= 1
    
    def _resize(self) -> None:
        """Resize hash table when load factor is too high"""
        old_keys = self.keys
        old_values = self.values
        old_deleted = self.deleted
        
        self.capacity *= 2
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.deleted = [False] * self.capacity
        
        # Rehash all elements
        for i, key in enumerate(old_keys):
            if key is not None and not old_deleted[i]:
                self.put(key, old_values[i])

class MyHashMapBST:
    """
    Approach 4: Binary Search Tree Implementation
    
    Use BST to maintain sorted order and handle any key range.
    
    Time Complexity: O(log n) average, O(n) worst case for all operations
    Space Complexity: O(n) where n is the number of key-value pairs
    """
    
    class TreeNode:
        def __init__(self, key: int, value: int):
            self.key = key
            self.value = value
            self.left: Optional['MyHashMapBST.TreeNode'] = None
            self.right: Optional['MyHashMapBST.TreeNode'] = None
    
    def __init__(self):
        self.root: Optional['MyHashMapBST.TreeNode'] = None
    
    def put(self, key: int, value: int) -> None:
        self.root = self._put_recursive(self.root, key, value)
    
    def _put_recursive(self, node: Optional['MyHashMapBST.TreeNode'], key: int, value: int) -> 'MyHashMapBST.TreeNode':
        if not node:
            return self.TreeNode(key, value)
        
        if key < node.key:
            node.left = self._put_recursive(node.left, key, value)
        elif key > node.key:
            node.right = self._put_recursive(node.right, key, value)
        else:
            # Key already exists, update value
            node.value = value
        
        return node
    
    def get(self, key: int) -> int:
        node = self._get_recursive(self.root, key)
        return node.value if node else -1
    
    def _get_recursive(self, node: Optional['MyHashMapBST.TreeNode'], key: int) -> Optional['MyHashMapBST.TreeNode']:
        if not node:
            return None
        
        if key == node.key:
            return node
        elif key < node.key:
            return self._get_recursive(node.left, key)
        else:
            return self._get_recursive(node.right, key)
    
    def remove(self, key: int) -> None:
        self.root = self._remove_recursive(self.root, key)
    
    def _remove_recursive(self, node: Optional['MyHashMapBST.TreeNode'], key: int) -> Optional['MyHashMapBST.TreeNode']:
        if not node:
            return None
        
        if key < node.key:
            node.left = self._remove_recursive(node.left, key)
        elif key > node.key:
            node.right = self._remove_recursive(node.right, key)
        else:
            # Node to be deleted found
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            else:
                # Node with two children
                min_node = self._find_min(node.right)
                node.key = min_node.key
                node.value = min_node.value
                node.right = self._remove_recursive(node.right, min_node.key)
        
        return node
    
    def _find_min(self, node: 'MyHashMapBST.TreeNode') -> 'MyHashMapBST.TreeNode':
        while node.left:
            node = node.left
        return node

class MyHashMapRobinHood:
    """
    Approach 5: Robin Hood Hashing
    
    Advanced open addressing with displacement optimization.
    
    Time Complexity: O(1) expected for all operations
    Space Complexity: O(capacity)
    """
    
    def __init__(self):
        self.capacity = 1000
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.distances = [0] * self.capacity  # Distance from ideal position
    
    def _hash(self, key: int) -> int:
        return key % self.capacity
    
    def put(self, key: int, value: int) -> None:
        if self.size >= self.capacity * 0.7:
            self._resize()
        
        index = self._hash(key)
        distance = 0
        
        while True:
            # Empty slot found
            if self.keys[index] is None:
                self.keys[index] = key
                self.values[index] = value
                self.distances[index] = distance
                self.size += 1
                return
            
            # Key already exists, update value
            if self.keys[index] == key:
                self.values[index] = value
                return
            
            # Robin Hood: if current element is closer to its ideal position,
            # displace it and continue with the displaced element
            if distance > self.distances[index]:
                # Swap current entry with the one we're inserting
                self.keys[index], key = key, self.keys[index]
                self.values[index], value = value, self.values[index]
                self.distances[index], distance = distance, self.distances[index]
            
            index = (index + 1) % self.capacity
            distance += 1
    
    def get(self, key: int) -> int:
        index = self._hash(key)
        distance = 0
        
        while self.keys[index] is not None:
            if self.keys[index] == key:
                return self.values[index]
            
            # If we've gone further than this key could be, it's not here
            if distance > self.distances[index]:
                break
            
            index = (index + 1) % self.capacity
            distance += 1
        
        return -1
    
    def remove(self, key: int) -> None:
        index = self._hash(key)
        distance = 0
        
        while self.keys[index] is not None:
            if self.keys[index] == key:
                # Found key to remove
                self._remove_at_index(index)
                self.size -= 1
                return
            
            if distance > self.distances[index]:
                break
            
            index = (index + 1) % self.capacity
            distance += 1
    
    def _remove_at_index(self, remove_index: int) -> None:
        """Remove entry at index and shift following entries back"""
        index = remove_index
        
        while True:
            next_index = (index + 1) % self.capacity
            
            # If next slot is empty or has distance 0, we can stop
            if (self.keys[next_index] is None or 
                self.distances[next_index] == 0):
                break
            
            # Shift the next entry back
            self.keys[index] = self.keys[next_index]
            self.values[index] = self.values[next_index]
            self.distances[index] = self.distances[next_index] - 1
            
            index = next_index
        
        # Clear the final position
        self.keys[index] = None
        self.values[index] = None
        self.distances[index] = 0
    
    def _resize(self) -> None:
        """Resize hash table"""
        old_keys = self.keys
        old_values = self.values
        
        self.capacity *= 2
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.distances = [0] * self.capacity
        
        # Rehash all elements
        for i, key in enumerate(old_keys):
            if key is not None:
                self.put(key, old_values[i])


def test_hashmap_basic_operations():
    """Test basic HashMap operations"""
    print("=== Testing Basic HashMap Operations ===")
    
    implementations = [
        ("Array-based", MyHashMapArray),
        ("Chaining", MyHashMapChaining),
        ("Open Addressing", MyHashMapOpenAddressing),
        ("BST-based", MyHashMapBST),
        ("Robin Hood", MyHashMapRobinHood)
    ]
    
    for name, HashMapClass in implementations:
        print(f"\n{name}:")
        
        hashmap = HashMapClass()
        
        # Test put and get
        pairs = [(1, 10), (2, 20), (3, 30), (2, 25)]  # Note: key 2 updated
        for key, value in pairs:
            hashmap.put(key, value)
            print(f"  put({key}, {value}), get({key}): {hashmap.get(key)}")
        
        # Test get for various keys
        test_keys = [1, 2, 3, 4, 5]
        for key in test_keys:
            result = hashmap.get(key)
            print(f"  get({key}): {result}")
        
        # Test remove
        hashmap.remove(2)
        print(f"  remove(2), get(2): {hashmap.get(2)}")

def test_hashmap_edge_cases():
    """Test HashMap edge cases"""
    print("\n=== Testing HashMap Edge Cases ===")
    
    hashmap = MyHashMapChaining()
    
    # Test operations on empty map
    print("Empty map operations:")
    print(f"  get(1): {hashmap.get(1)}")
    hashmap.remove(1)  # Should not crash
    print(f"  remove(1) on empty map: OK")
    
    # Test overwriting values
    print(f"\nOverwriting values:")
    hashmap.put(5, 100)
    hashmap.put(5, 200)
    hashmap.put(5, 300)
    print(f"  Put key 5 with values 100, 200, 300. Final get(5): {hashmap.get(5)}")
    
    # Test zero key and value
    print(f"\nZero key and value:")
    hashmap.put(0, 0)
    print(f"  put(0, 0), get(0): {hashmap.get(0)}")
    
    # Test negative values (value can be negative, but -1 is special)
    print(f"\nNegative values:")
    hashmap.put(7, -5)
    print(f"  put(7, -5), get(7): {hashmap.get(7)}")

def test_collision_scenarios():
    """Test collision handling scenarios"""
    print("\n=== Testing Collision Scenarios ===")
    
    hashmap = MyHashMapChaining()
    
    # Keys that will collide in a hash table of size 1000
    collision_pairs = [(1000, 100), (2000, 200), (3000, 300), (4000, 400)]
    
    print("Adding collision-prone key-value pairs:")
    for key, value in collision_pairs:
        hashmap.put(key, value)
        print(f"  put({key}, {value})")
    
    print("Retrieving all values:")
    for key, expected_value in collision_pairs:
        actual_value = hashmap.get(key)
        print(f"  get({key}): {actual_value} (expected: {expected_value})")
    
    # Test updating colliding keys
    hashmap.put(2000, 999)
    print(f"  Updated key 2000 to 999, get(2000): {hashmap.get(2000)}")
    
    # Ensure other colliding keys are unaffected
    unaffected = [(1000, 100), (3000, 300), (4000, 400)]
    all_correct = all(hashmap.get(k) == v for k, v in unaffected)
    print(f"  Other colliding keys unaffected: {all_correct}")

def test_hashmap_performance():
    """Test HashMap performance"""
    print("\n=== Testing HashMap Performance ===")
    
    import time
    
    implementations = [
        ("Chaining", MyHashMapChaining),
        ("Open Addressing", MyHashMapOpenAddressing),
        ("BST-based", MyHashMapBST),
        ("Robin Hood", MyHashMapRobinHood)
    ]
    
    operations = 10000
    
    for name, HashMapClass in implementations:
        hashmap = HashMapClass()
        
        # Test put performance
        start_time = time.time()
        for i in range(operations):
            hashmap.put(i, i * 2)
        put_time = (time.time() - start_time) * 1000
        
        # Test get performance
        start_time = time.time()
        for i in range(0, operations, 10):  # Get every 10th element
            hashmap.get(i)
        get_time = (time.time() - start_time) * 1000
        
        # Test remove performance
        start_time = time.time()
        for i in range(0, operations, 2):  # Remove every other element
            hashmap.remove(i)
        remove_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Put {operations}: {put_time:.2f}ms")
        print(f"    Get {operations//10}: {get_time:.2f}ms")
        print(f"    Remove {operations//2}: {remove_time:.2f}ms")

def test_update_semantics():
    """Test update semantics"""
    print("\n=== Testing Update Semantics ===")
    
    hashmap = MyHashMapOpenAddressing()
    
    # Test key update sequence
    key = 42
    values = [10, 20, 30, 40, 50]
    
    print(f"Testing updates for key {key}:")
    for value in values:
        hashmap.put(key, value)
        retrieved = hashmap.get(key)
        print(f"  put({key}, {value}) -> get({key}): {retrieved}")
    
    # Test mixed operations
    print(f"\nMixed operations on key {key}:")
    hashmap.remove(key)
    print(f"  remove({key}) -> get({key}): {hashmap.get(key)}")
    
    hashmap.put(key, 999)
    print(f"  put({key}, 999) -> get({key}): {hashmap.get(key)}")

def test_load_factor_behavior():
    """Test behavior under different load factors"""
    print("\n=== Testing Load Factor Behavior ===")
    
    hashmap = MyHashMapOpenAddressing()
    
    # Monitor performance as load factor increases
    for batch in range(1, 6):
        # Add batch of elements
        for i in range(100):
            key = (batch - 1) * 100 + i
            hashmap.put(key, key * 10)
        
        # Test lookup performance for this batch
        import time
        start_time = time.time()
        for i in range(50):  # Test 50 lookups
            key = random.randint(0, batch * 100 - 1)
            hashmap.get(key)
        lookup_time = (time.time() - start_time) * 1000
        
        # Estimate load factor
        if hasattr(hashmap, 'size') and hasattr(hashmap, 'capacity'):
            load_factor = hashmap.size / hashmap.capacity
            print(f"  Batch {batch}: Load factor {load_factor:.3f}, 50 lookups: {lookup_time:.2f}ms")

def demonstrate_hashmap_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating HashMap Applications ===")
    
    # Application 1: Frequency counting
    print("Application 1: Character Frequency Counter")
    freq_counter = MyHashMapChaining()
    
    text = "hello world"
    for char in text:
        if char != ' ':  # Skip spaces
            current_count = freq_counter.get(ord(char))
            new_count = (current_count if current_count != -1 else 0) + 1
            freq_counter.put(ord(char), new_count)
    
    print(f"  Text: '{text}'")
    print(f"  Character frequencies:")
    for char in "helo wrd":
        if char != ' ':
            count = freq_counter.get(ord(char))
            print(f"    '{char}': {count if count != -1 else 0}")
    
    # Application 2: Caching mechanism
    print(f"\nApplication 2: Simple Cache")
    cache = MyHashMapBST()
    
    def expensive_computation(n):
        """Simulate expensive computation"""
        return n * n * n
    
    def cached_computation(n):
        # Check cache first
        cached_result = cache.get(n)
        if cached_result != -1:
            print(f"    Cache hit for {n}: {cached_result}")
            return cached_result
        
        # Compute and cache
        result = expensive_computation(n)
        cache.put(n, result)
        print(f"    Computed and cached {n}: {result}")
        return result
    
    # Test caching behavior
    test_inputs = [3, 5, 3, 7, 5, 3]  # Some repeats
    for n in test_inputs:
        cached_computation(n)
    
    # Application 3: Two Sum problem solver
    print(f"\nApplication 3: Two Sum Solver")
    two_sum_map = MyHashMapRobinHood()
    
    def two_sum(nums, target):
        indices = []
        
        for i, num in enumerate(nums):
            complement = target - num
            complement_index = two_sum_map.get(complement)
            
            if complement_index != -1:
                indices = [complement_index, i]
                break
            
            two_sum_map.put(num, i)
        
        return indices
    
    nums = [2, 7, 11, 15]
    target = 9
    result = two_sum(nums, target)
    print(f"  Array: {nums}, Target: {target}")
    print(f"  Indices with sum {target}: {result}")

def test_robin_hood_efficiency():
    """Test Robin Hood hashing efficiency"""
    print("\n=== Testing Robin Hood Hashing Efficiency ===")
    
    import random
    
    # Compare Robin Hood with regular linear probing
    robin_hood = MyHashMapRobinHood()
    linear_probe = MyHashMapOpenAddressing()
    
    # Generate keys that would cause clustering in linear probing
    keys = [i * 1000 for i in range(100)]  # Keys that hash to similar positions
    random.shuffle(keys)
    
    # Add to both hash maps
    for key in keys:
        robin_hood.put(key, key * 2)
        linear_probe.put(key, key * 2)
    
    # Test lookup performance
    import time
    
    # Robin Hood lookup
    start_time = time.time()
    for key in keys:
        robin_hood.get(key)
    robin_hood_time = (time.time() - start_time) * 1000
    
    # Linear probing lookup
    start_time = time.time()
    for key in keys:
        linear_probe.get(key)
    linear_probe_time = (time.time() - start_time) * 1000
    
    print(f"Lookup performance comparison:")
    print(f"  Robin Hood hashing: {robin_hood_time:.2f}ms")
    print(f"  Linear probing: {linear_probe_time:.2f}ms")
    print(f"  Robin Hood improvement: {((linear_probe_time - robin_hood_time) / linear_probe_time * 100):.1f}%")

def benchmark_memory_efficiency():
    """Benchmark memory efficiency"""
    print("\n=== Benchmarking Memory Efficiency ===")
    
    implementations = [
        ("Chaining", MyHashMapChaining),
        ("Open Addressing", MyHashMapOpenAddressing),
        ("BST-based", MyHashMapBST)
    ]
    
    num_pairs = 1000
    
    for name, HashMapClass in implementations:
        hashmap = HashMapClass()
        
        # Add key-value pairs
        for i in range(num_pairs):
            hashmap.put(i, i * 2)
        
        # Estimate memory usage (simplified)
        if hasattr(hashmap, 'buckets'):
            if isinstance(hashmap.buckets[0], list):
                # Chaining: count total pairs in all buckets
                memory_pairs = sum(len(bucket) for bucket in hashmap.buckets)
            else:
                # Open addressing: count non-None slots
                memory_pairs = sum(1 for key in hashmap.keys if key is not None)
        else:
            # BST: approximate as number of nodes
            memory_pairs = num_pairs
        
        efficiency = (num_pairs / memory_pairs) * 100 if memory_pairs > 0 else 0
        print(f"  {name}: {memory_pairs} slots for {num_pairs} pairs ({efficiency:.1f}% efficiency)")

if __name__ == "__main__":
    test_hashmap_basic_operations()
    test_hashmap_edge_cases()
    test_collision_scenarios()
    test_hashmap_performance()
    test_update_semantics()
    test_load_factor_behavior()
    demonstrate_hashmap_applications()
    test_robin_hood_efficiency()
    benchmark_memory_efficiency()

"""
HashMap Design demonstrates key concepts:

Core Approaches:
1. Array-based - Simple but memory-intensive for large ranges
2. Chaining - Handles collisions with linked lists/arrays of pairs
3. Open Addressing - Linear probing for collision resolution
4. BST-based - Maintains sorted order, handles any key range
5. Robin Hood Hashing - Advanced open addressing with displacement optimization

Key Design Principles:
- Hash function design for uniform distribution
- Collision resolution strategies for key-value pairs
- Load factor management for performance
- Update semantics for existing keys

Advanced Techniques:
- Robin Hood Hashing: Minimizes variance in probe distances
- Dynamic resizing: Maintains optimal load factor
- Multiple hash functions: Better distribution

Performance Characteristics:
- Average case: O(1) for hash-based approaches
- Worst case: O(n) for hash-based, O(log n) for BST
- Robin Hood hashing: O(1) expected with lower variance

Real-world Applications:
- Frequency counting and analytics
- Caching mechanisms
- Database indexing
- Symbol tables in compilers
- Configuration management
- Algorithm optimization (Two Sum, etc.)

The chaining approach is most commonly used due to
its simplicity, while Robin Hood hashing provides
superior performance characteristics for demanding applications.
"""
