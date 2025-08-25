"""
705. Design HashSet - Multiple Approaches
Difficulty: Easy

Design a HashSet without using any built-in hash table libraries.

Implement MyHashSet class:
- MyHashSet() Initializes the HashSet object.
- void add(int key) Inserts the value key into the HashSet.
- bool contains(int key) Returns whether the value key exists in the HashSet or not.
- void remove(int key) Removes the value key in the HashSet. If key does not exist, do nothing.
"""

from typing import List, Optional

class MyHashSetArray:
    """
    Approach 1: Simple Array Implementation
    
    Use a boolean array to track presence of keys.
    Assumes keys are in a limited range.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(range) where range is the maximum possible key value
    """
    
    def __init__(self):
        self.size = 1000001  # Assuming keys are <= 10^6
        self.buckets = [False] * self.size
    
    def add(self, key: int) -> None:
        if 0 <= key < self.size:
            self.buckets[key] = True
    
    def remove(self, key: int) -> None:
        if 0 <= key < self.size:
            self.buckets[key] = False
    
    def contains(self, key: int) -> bool:
        if 0 <= key < self.size:
            return self.buckets[key]
        return False

class MyHashSetChaining:
    """
    Approach 2: Hash Table with Chaining
    
    Use separate chaining to handle collisions.
    
    Time Complexity: O(1) average, O(n) worst case for all operations
    Space Complexity: O(n) where n is the number of keys
    """
    
    def __init__(self):
        self.size = 1000
        self.buckets = [[] for _ in range(self.size)]
    
    def _hash(self, key: int) -> int:
        return key % self.size
    
    def add(self, key: int) -> None:
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        if key not in bucket:
            bucket.append(key)
    
    def remove(self, key: int) -> None:
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        if key in bucket:
            bucket.remove(key)
    
    def contains(self, key: int) -> bool:
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        return key in bucket

class MyHashSetOpenAddressing:
    """
    Approach 3: Open Addressing with Linear Probing
    
    Use linear probing to handle collisions.
    
    Time Complexity: O(1) average case for all operations
    Space Complexity: O(capacity)
    """
    
    def __init__(self):
        self.capacity = 1000
        self.size = 0
        self.buckets = [None] * self.capacity
        self.deleted = [False] * self.capacity  # Track deleted slots
    
    def _hash(self, key: int) -> int:
        return key % self.capacity
    
    def _find_slot(self, key: int) -> int:
        """Find slot for key using linear probing"""
        index = self._hash(key)
        original_index = index
        
        while (self.buckets[index] is not None and 
               self.buckets[index] != key and 
               not self.deleted[index]):
            index = (index + 1) % self.capacity
            
            # Prevent infinite loop
            if index == original_index:
                return -1
        
        return index
    
    def add(self, key: int) -> None:
        if self.size >= self.capacity * 0.7:  # Load factor threshold
            self._resize()
        
        index = self._find_slot(key)
        if index != -1:
            if self.buckets[index] != key:
                self.buckets[index] = key
                self.deleted[index] = False
                self.size += 1
    
    def remove(self, key: int) -> None:
        index = self._find_slot(key)
        if index != -1 and self.buckets[index] == key:
            self.deleted[index] = True
            self.size -= 1
    
    def contains(self, key: int) -> bool:
        index = self._find_slot(key)
        return (index != -1 and 
                self.buckets[index] == key and 
                not self.deleted[index])
    
    def _resize(self) -> None:
        """Resize hash table when load factor is too high"""
        old_buckets = self.buckets
        old_deleted = self.deleted
        
        self.capacity *= 2
        self.size = 0
        self.buckets = [None] * self.capacity
        self.deleted = [False] * self.capacity
        
        # Rehash all elements
        for i, key in enumerate(old_buckets):
            if key is not None and not old_deleted[i]:
                self.add(key)

class MyHashSetBST:
    """
    Approach 4: Binary Search Tree Implementation
    
    Use BST to maintain sorted order and handle any key range.
    
    Time Complexity: O(log n) average, O(n) worst case for all operations
    Space Complexity: O(n) where n is the number of keys
    """
    
    class TreeNode:
        def __init__(self, key: int):
            self.key = key
            self.left: Optional['MyHashSetBST.TreeNode'] = None
            self.right: Optional['MyHashSetBST.TreeNode'] = None
    
    def __init__(self):
        self.root: Optional['MyHashSetBST.TreeNode'] = None
    
    def add(self, key: int) -> None:
        self.root = self._add_recursive(self.root, key)
    
    def _add_recursive(self, node: Optional['MyHashSetBST.TreeNode'], key: int) -> 'MyHashSetBST.TreeNode':
        if not node:
            return self.TreeNode(key)
        
        if key < node.key:
            node.left = self._add_recursive(node.left, key)
        elif key > node.key:
            node.right = self._add_recursive(node.right, key)
        # If key == node.key, do nothing (already exists)
        
        return node
    
    def remove(self, key: int) -> None:
        self.root = self._remove_recursive(self.root, key)
    
    def _remove_recursive(self, node: Optional['MyHashSetBST.TreeNode'], key: int) -> Optional['MyHashSetBST.TreeNode']:
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
                node.right = self._remove_recursive(node.right, min_node.key)
        
        return node
    
    def _find_min(self, node: 'MyHashSetBST.TreeNode') -> 'MyHashSetBST.TreeNode':
        while node.left:
            node = node.left
        return node
    
    def contains(self, key: int) -> bool:
        return self._contains_recursive(self.root, key)
    
    def _contains_recursive(self, node: Optional['MyHashSetBST.TreeNode'], key: int) -> bool:
        if not node:
            return False
        
        if key == node.key:
            return True
        elif key < node.key:
            return self._contains_recursive(node.left, key)
        else:
            return self._contains_recursive(node.right, key)

class MyHashSetAdvanced:
    """
    Approach 5: Advanced Hash Set with Multiple Hash Functions
    
    Use multiple hash functions for better distribution.
    
    Time Complexity: O(1) average case for all operations
    Space Complexity: O(n) where n is the number of keys
    """
    
    def __init__(self):
        self.size = 1000
        self.buckets = [set() for _ in range(self.size)]
        self.count = 0
    
    def _hash1(self, key: int) -> int:
        return key % self.size
    
    def _hash2(self, key: int) -> int:
        return (key * 31) % self.size
    
    def _hash3(self, key: int) -> int:
        return ((key ^ (key >> 16)) * 0x45d9f3b) % self.size
    
    def _get_bucket_index(self, key: int) -> int:
        """Choose best bucket based on current load"""
        indices = [self._hash1(key), self._hash2(key), self._hash3(key)]
        
        # Choose bucket with minimum load
        min_load = float('inf')
        best_index = indices[0]
        
        for idx in indices:
            if len(self.buckets[idx]) < min_load:
                min_load = len(self.buckets[idx])
                best_index = idx
        
        return best_index
    
    def add(self, key: int) -> None:
        # Check if key already exists
        if self.contains(key):
            return
        
        bucket_index = self._get_bucket_index(key)
        self.buckets[bucket_index].add(key)
        self.count += 1
    
    def remove(self, key: int) -> None:
        # Search all possible buckets
        for hash_func in [self._hash1, self._hash2, self._hash3]:
            bucket_index = hash_func(key)
            if key in self.buckets[bucket_index]:
                self.buckets[bucket_index].remove(key)
                self.count -= 1
                return
    
    def contains(self, key: int) -> bool:
        # Search all possible buckets
        for hash_func in [self._hash1, self._hash2, self._hash3]:
            bucket_index = hash_func(key)
            if key in self.buckets[bucket_index]:
                return True
        return False
    
    def size(self) -> int:
        return self.count
    
    def load_factor(self) -> float:
        return self.count / self.size


def test_hashset_basic_operations():
    """Test basic HashSet operations"""
    print("=== Testing Basic HashSet Operations ===")
    
    implementations = [
        ("Array-based", MyHashSetArray),
        ("Chaining", MyHashSetChaining),
        ("Open Addressing", MyHashSetOpenAddressing),
        ("BST-based", MyHashSetBST),
        ("Advanced Multi-hash", MyHashSetAdvanced)
    ]
    
    for name, HashSetClass in implementations:
        print(f"\n{name}:")
        
        hashset = HashSetClass()
        
        # Test add and contains
        keys = [1, 2, 3, 2, 1, 4]
        for key in keys:
            hashset.add(key)
            print(f"  add({key}), contains({key}): {hashset.contains(key)}")
        
        # Test contains for various keys
        test_keys = [1, 2, 3, 4, 5]
        for key in test_keys:
            result = hashset.contains(key)
            print(f"  contains({key}): {result}")
        
        # Test remove
        hashset.remove(2)
        print(f"  remove(2), contains(2): {hashset.contains(2)}")

def test_hashset_edge_cases():
    """Test HashSet edge cases"""
    print("\n=== Testing HashSet Edge Cases ===")
    
    hashset = MyHashSetChaining()
    
    # Test operations on empty set
    print("Empty set operations:")
    print(f"  contains(1): {hashset.contains(1)}")
    hashset.remove(1)  # Should not crash
    print(f"  remove(1) on empty set: OK")
    
    # Test duplicate additions
    print(f"\nDuplicate additions:")
    hashset.add(5)
    hashset.add(5)
    hashset.add(5)
    print(f"  Added 5 three times, contains(5): {hashset.contains(5)}")
    
    # Test large numbers
    print(f"\nLarge numbers:")
    large_key = 999999
    hashset.add(large_key)
    print(f"  add({large_key}), contains({large_key}): {hashset.contains(large_key)}")
    
    # Test zero and negative (if supported)
    print(f"\nSpecial values:")
    hashset.add(0)
    print(f"  add(0), contains(0): {hashset.contains(0)}")

def test_collision_handling():
    """Test collision handling in hash implementations"""
    print("\n=== Testing Collision Handling ===")
    
    # Test with keys that are likely to collide
    hashset = MyHashSetChaining()
    
    # Keys that will collide in a hash table of size 1000
    collision_keys = [1000, 2000, 3000, 4000, 5000]  # All hash to index 0
    
    print("Adding collision-prone keys:")
    for key in collision_keys:
        hashset.add(key)
        print(f"  add({key})")
    
    print("Checking all keys are present:")
    all_present = all(hashset.contains(key) for key in collision_keys)
    print(f"  All collision keys present: {all_present}")
    
    # Test removal with collisions
    hashset.remove(3000)
    print(f"  remove(3000), contains(3000): {hashset.contains(3000)}")
    print(f"  Other keys still present: {all(hashset.contains(key) for key in [1000, 2000, 4000, 5000])}")

def test_hashset_performance():
    """Test HashSet performance"""
    print("\n=== Testing HashSet Performance ===")
    
    import time
    
    implementations = [
        ("Chaining", MyHashSetChaining),
        ("Open Addressing", MyHashSetOpenAddressing),
        ("BST-based", MyHashSetBST)
    ]
    
    operations = 10000
    
    for name, HashSetClass in implementations:
        hashset = HashSetClass()
        
        # Test add performance
        start_time = time.time()
        for i in range(operations):
            hashset.add(i)
        add_time = (time.time() - start_time) * 1000
        
        # Test contains performance
        start_time = time.time()
        for i in range(0, operations, 10):  # Check every 10th element
            hashset.contains(i)
        contains_time = (time.time() - start_time) * 1000
        
        # Test remove performance
        start_time = time.time()
        for i in range(0, operations, 2):  # Remove every other element
            hashset.remove(i)
        remove_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Add {operations}: {add_time:.2f}ms")
        print(f"    Contains {operations//10}: {contains_time:.2f}ms")
        print(f"    Remove {operations//2}: {remove_time:.2f}ms")

def test_load_factor_effects():
    """Test effects of load factor on performance"""
    print("\n=== Testing Load Factor Effects ===")
    
    hashset = MyHashSetAdvanced()
    
    # Add elements and monitor load factor
    for i in range(0, 2000, 200):
        # Add batch of elements
        for j in range(200):
            hashset.add(i + j)
        
        if hasattr(hashset, 'load_factor'):
            load_factor = hashset.load_factor()
            size = hashset.size()
            print(f"  Size: {size}, Load Factor: {load_factor:.3f}")

def test_memory_usage():
    """Test memory usage of different implementations"""
    print("\n=== Testing Memory Usage ===")
    
    num_elements = 1000
    
    implementations = [
        ("Chaining", MyHashSetChaining),
        ("Open Addressing", MyHashSetOpenAddressing),
        ("BST-based", MyHashSetBST)
    ]
    
    for name, HashSetClass in implementations:
        hashset = HashSetClass()
        
        # Add elements
        for i in range(num_elements):
            hashset.add(i)
        
        # Estimate memory usage (simplified)
        if hasattr(hashset, 'buckets'):
            if isinstance(hashset.buckets[0], list):
                # Chaining: count total list elements
                memory_estimate = sum(len(bucket) for bucket in hashset.buckets)
            else:
                # Open addressing: count non-None slots
                memory_estimate = sum(1 for bucket in hashset.buckets if bucket is not None)
        else:
            # BST: approximate as number of nodes
            memory_estimate = num_elements
        
        efficiency = (num_elements / memory_estimate) * 100 if memory_estimate > 0 else 0
        print(f"  {name}: {memory_estimate} memory units for {num_elements} elements ({efficiency:.1f}% efficiency)")

def demonstrate_hashset_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating HashSet Applications ===")
    
    # Application 1: Unique visitor tracking
    print("Application 1: Unique Visitor Tracking")
    visitor_tracker = MyHashSetChaining()
    
    # Simulate visitor IDs (some repeats)
    visitor_ids = [101, 102, 103, 101, 104, 102, 105, 103, 106]
    
    for visitor_id in visitor_ids:
        if not visitor_tracker.contains(visitor_id):
            visitor_tracker.add(visitor_id)
            print(f"  New visitor: {visitor_id}")
        else:
            print(f"  Returning visitor: {visitor_id}")
    
    # Application 2: Duplicate detection
    print(f"\nApplication 2: Duplicate Detection in Data Stream")
    duplicate_detector = MyHashSetOpenAddressing()
    
    data_stream = [1, 2, 3, 2, 4, 5, 1, 6, 3, 7]
    duplicates_found = []
    
    for data in data_stream:
        if duplicate_detector.contains(data):
            duplicates_found.append(data)
            print(f"  Duplicate detected: {data}")
        else:
            duplicate_detector.add(data)
            print(f"  New data: {data}")
    
    print(f"  Total duplicates found: {duplicates_found}")
    
    # Application 3: Permission checking
    print(f"\nApplication 3: Permission Checking")
    permissions = MyHashSetBST()
    
    # Set up permissions
    allowed_actions = [1001, 1002, 1003, 1005, 1007]  # Action IDs
    for action in allowed_actions:
        permissions.add(action)
    
    # Check various action requests
    action_requests = [1001, 1004, 1002, 1006, 1007, 1008]
    
    for action in action_requests:
        if permissions.contains(action):
            print(f"  Action {action}: ✓ Allowed")
        else:
            print(f"  Action {action}: ✗ Denied")

def test_stress_scenarios():
    """Test stress scenarios"""
    print("\n=== Testing Stress Scenarios ===")
    
    import random
    
    # Scenario 1: Random operations
    print("Scenario 1: Random Operations")
    hashset = MyHashSetChaining()
    
    operations = ["add", "remove", "contains"]
    
    for i in range(100):
        operation = random.choice(operations)
        key = random.randint(1, 50)  # Limited range to create collisions
        
        if operation == "add":
            hashset.add(key)
        elif operation == "remove":
            hashset.remove(key)
        elif operation == "contains":
            result = hashset.contains(key)
            # Don't print every result to avoid spam
    
    print(f"  Completed 100 random operations")
    
    # Scenario 2: Sequential vs Random access patterns
    print(f"\nScenario 2: Access Pattern Comparison")
    
    # Sequential access
    hashset_seq = MyHashSetOpenAddressing()
    import time
    
    start_time = time.time()
    for i in range(1000):
        hashset_seq.add(i)
    sequential_time = (time.time() - start_time) * 1000
    
    # Random access
    hashset_rand = MyHashSetOpenAddressing()
    random_keys = list(range(1000))
    random.shuffle(random_keys)
    
    start_time = time.time()
    for key in random_keys:
        hashset_rand.add(key)
    random_time = (time.time() - start_time) * 1000
    
    print(f"  Sequential insertion: {sequential_time:.2f}ms")
    print(f"  Random insertion: {random_time:.2f}ms")

def benchmark_different_sizes():
    """Benchmark performance with different data sizes"""
    print("\n=== Benchmarking Different Sizes ===")
    
    import time
    
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        print(f"\nSize: {size}")
        
        hashset = MyHashSetChaining()
        
        # Add elements
        start_time = time.time()
        for i in range(size):
            hashset.add(i)
        add_time = (time.time() - start_time) * 1000
        
        # Random lookups
        start_time = time.time()
        for _ in range(1000):
            key = random.randint(0, size - 1)
            hashset.contains(key)
        lookup_time = (time.time() - start_time) * 1000
        
        print(f"  Add {size} elements: {add_time:.2f}ms")
        print(f"  1000 random lookups: {lookup_time:.2f}ms")

if __name__ == "__main__":
    test_hashset_basic_operations()
    test_hashset_edge_cases()
    test_collision_handling()
    test_hashset_performance()
    test_load_factor_effects()
    test_memory_usage()
    demonstrate_hashset_applications()
    test_stress_scenarios()
    benchmark_different_sizes()

"""
HashSet Design demonstrates key concepts:

Core Approaches:
1. Array-based - Simple but memory-intensive for large ranges
2. Chaining - Handles collisions with linked lists/arrays
3. Open Addressing - Linear probing for collision resolution
4. BST-based - Maintains sorted order, handles any key range
5. Advanced Multi-hash - Multiple hash functions for better distribution

Key Design Principles:
- Hash function design for uniform distribution
- Collision resolution strategies
- Load factor management for performance
- Dynamic resizing capabilities

Collision Resolution Techniques:
- Separate Chaining: Each bucket contains a list of elements
- Linear Probing: Find next available slot sequentially
- Multiple Hashing: Use different hash functions

Performance Characteristics:
- Average case: O(1) for hash-based approaches
- Worst case: O(n) for hash-based, O(log n) for BST
- Space efficiency depends on load factor and collision handling

Real-world Applications:
- Unique visitor tracking
- Duplicate detection in data streams
- Permission and access control systems
- Caching mechanisms
- Database indexing

The chaining approach is most commonly used due to
its simplicity and predictable performance characteristics.
"""
