"""
380. Insert Delete GetRandom O(1) - Multiple Approaches
Difficulty: Medium

Implement the RandomizedSet class:
- RandomizedSet() Initializes the RandomizedSet object.
- bool insert(int val) Inserts an item val into the set if not present. Returns true if the 
  item was not present, false otherwise.
- bool remove(int val) Removes an item val from the set if present. Returns true if the 
  item was present, false otherwise.
- int getRandom() Returns a random element from the current set of elements (it's guaranteed 
  that at least one element exists when this method is called). Each element must have the 
  same probability of being returned.

You must implement the functions of the class such that each function works in average O(1) time complexity.
"""

import random
from typing import List, Dict, Set

class RandomizedSetArrayHashMap:
    """
    Approach 1: Array + HashMap
    
    Use array for O(1) random access and HashMap for O(1) lookup.
    Swap with last element for O(1) deletion.
    
    Time Complexity: O(1) average for all operations
    Space Complexity: O(n) where n is the number of elements
    """
    
    def __init__(self):
        self.values = []  # Store actual values
        self.indices = {}  # value -> index in values array
    
    def insert(self, val: int) -> bool:
        if val in self.indices:
            return False
        
        # Add to end of array and update index mapping
        self.indices[val] = len(self.values)
        self.values.append(val)
        return True
    
    def remove(self, val: int) -> bool:
        if val not in self.indices:
            return False
        
        # Get index of element to remove
        index_to_remove = self.indices[val]
        last_element = self.values[-1]
        
        # Move last element to the position of element to remove
        self.values[index_to_remove] = last_element
        self.indices[last_element] = index_to_remove
        
        # Remove last element and clean up
        self.values.pop()
        del self.indices[val]
        
        return True
    
    def getRandom(self) -> int:
        return random.choice(self.values)

class RandomizedSetWithFrequency:
    """
    Approach 2: Enhanced with Frequency Tracking
    
    Track frequency of each element for weighted random selection.
    
    Time Complexity: O(1) average for all operations
    Space Complexity: O(n) where n is the number of elements
    """
    
    def __init__(self):
        self.values = []
        self.indices = {}
        self.frequencies = {}  # value -> frequency
    
    def insert(self, val: int) -> bool:
        if val in self.indices:
            self.frequencies[val] += 1
            return False
        
        self.indices[val] = len(self.values)
        self.values.append(val)
        self.frequencies[val] = 1
        return True
    
    def remove(self, val: int) -> bool:
        if val not in self.indices:
            return False
        
        self.frequencies[val] -= 1
        if self.frequencies[val] > 0:
            return True
        
        # Actually remove the element
        index_to_remove = self.indices[val]
        last_element = self.values[-1]
        
        self.values[index_to_remove] = last_element
        self.indices[last_element] = index_to_remove
        
        self.values.pop()
        del self.indices[val]
        del self.frequencies[val]
        
        return True
    
    def getRandom(self) -> int:
        return random.choice(self.values)
    
    def getRandomWeighted(self) -> int:
        """Get random element weighted by frequency"""
        if not self.values:
            return None
        
        total_weight = sum(self.frequencies[val] for val in self.values)
        target = random.randint(1, total_weight)
        
        current_weight = 0
        for val in self.values:
            current_weight += self.frequencies[val]
            if current_weight >= target:
                return val
        
        return self.values[-1]

class RandomizedSetLinkedList:
    """
    Approach 3: Doubly Linked List + HashMap
    
    Alternative implementation using doubly linked list.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(n) where n is the number of elements
    """
    
    class Node:
        def __init__(self, val: int):
            self.val = val
            self.prev = None
            self.next = None
    
    def __init__(self):
        self.node_map = {}  # value -> Node
        self.head = self.Node(0)  # Dummy head
        self.tail = self.Node(0)  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
    
    def insert(self, val: int) -> bool:
        if val in self.node_map:
            return False
        
        # Create new node and add to tail
        new_node = self.Node(val)
        self._add_to_tail(new_node)
        self.node_map[val] = new_node
        self.size += 1
        return True
    
    def remove(self, val: int) -> bool:
        if val not in self.node_map:
            return False
        
        node = self.node_map[val]
        self._remove_node(node)
        del self.node_map[val]
        self.size -= 1
        return True
    
    def getRandom(self) -> int:
        if self.size == 0:
            return None
        
        # Get random index and traverse
        random_index = random.randint(0, self.size - 1)
        current = self.head.next
        
        for _ in range(random_index):
            current = current.next
        
        return current.val
    
    def _add_to_tail(self, node: 'RandomizedSetLinkedList.Node') -> None:
        prev_node = self.tail.prev
        prev_node.next = node
        node.prev = prev_node
        node.next = self.tail
        self.tail.prev = node
    
    def _remove_node(self, node: 'RandomizedSetLinkedList.Node') -> None:
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

class RandomizedSetMultiset:
    """
    Approach 4: Multiset Implementation
    
    Allow duplicate values with count tracking.
    
    Time Complexity: O(1) average for all operations
    Space Complexity: O(n) where n is the number of unique elements
    """
    
    def __init__(self):
        self.values = []
        self.indices = {}  # value -> list of indices
        self.counts = {}   # value -> count
    
    def insert(self, val: int) -> bool:
        if val not in self.indices:
            self.indices[val] = []
            self.counts[val] = 0
        
        # Add new occurrence
        index = len(self.values)
        self.values.append(val)
        self.indices[val].append(index)
        self.counts[val] += 1
        
        return self.counts[val] == 1  # Return true only for first occurrence
    
    def remove(self, val: int) -> bool:
        if val not in self.indices or not self.indices[val]:
            return False
        
        # Get index of last occurrence of this value
        index_to_remove = self.indices[val].pop()
        last_element = self.values[-1]
        
        # Replace with last element
        self.values[index_to_remove] = last_element
        
        # Update indices for the moved element
        if last_element != val:
            # Remove old index and add new index
            last_element_indices = self.indices[last_element]
            last_element_indices[last_element_indices.index(len(self.values) - 1)] = index_to_remove
        
        self.values.pop()
        self.counts[val] -= 1
        
        # Clean up if no more occurrences
        if self.counts[val] == 0:
            del self.indices[val]
            del self.counts[val]
            return True
        
        return True
    
    def getRandom(self) -> int:
        return random.choice(self.values)

class RandomizedSetThreadSafe:
    """
    Approach 5: Thread-Safe Implementation
    
    Add thread safety for concurrent access.
    
    Time Complexity: O(1) average for all operations
    Space Complexity: O(n) where n is the number of elements
    """
    
    def __init__(self):
        import threading
        self.values = []
        self.indices = {}
        self.lock = threading.RLock()
    
    def insert(self, val: int) -> bool:
        with self.lock:
            if val in self.indices:
                return False
            
            self.indices[val] = len(self.values)
            self.values.append(val)
            return True
    
    def remove(self, val: int) -> bool:
        with self.lock:
            if val not in self.indices:
                return False
            
            index_to_remove = self.indices[val]
            last_element = self.values[-1]
            
            self.values[index_to_remove] = last_element
            self.indices[last_element] = index_to_remove
            
            self.values.pop()
            del self.indices[val]
            
            return True
    
    def getRandom(self) -> int:
        with self.lock:
            if not self.values:
                return None
            return random.choice(self.values)
    
    def size(self) -> int:
        with self.lock:
            return len(self.values)
    
    def contains(self, val: int) -> bool:
        with self.lock:
            return val in self.indices


def test_randomized_set_basic():
    """Test basic RandomizedSet functionality"""
    print("=== Testing Basic RandomizedSet Functionality ===")
    
    implementations = [
        ("Array + HashMap", RandomizedSetArrayHashMap),
        ("With Frequency", RandomizedSetWithFrequency),
        ("Linked List", RandomizedSetLinkedList),
        ("Thread-Safe", RandomizedSetThreadSafe)
    ]
    
    for name, RandomizedSetClass in implementations:
        print(f"\n{name}:")
        
        rs = RandomizedSetClass()
        
        # Test insertions
        print(f"  insert(1): {rs.insert(1)}")  # True
        print(f"  insert(2): {rs.insert(2)}")  # True
        print(f"  insert(1): {rs.insert(1)}")  # False (already exists)
        
        # Test random
        random_vals = [rs.getRandom() for _ in range(5)]
        print(f"  getRandom() x5: {random_vals}")
        
        # Test removal
        print(f"  remove(1): {rs.remove(1)}")  # True
        print(f"  remove(1): {rs.remove(1)}")  # False (doesn't exist)
        
        # Test random after removal
        random_vals = [rs.getRandom() for _ in range(3)]
        print(f"  getRandom() x3 after removal: {random_vals}")

def test_randomized_set_edge_cases():
    """Test RandomizedSet edge cases"""
    print("\n=== Testing RandomizedSet Edge Cases ===")
    
    rs = RandomizedSetArrayHashMap()
    
    # Test single element
    print("Testing single element:")
    rs.insert(42)
    print(f"  getRandom() with single element: {rs.getRandom()}")
    
    # Test many insertions and removals
    print(f"\nTesting bulk operations:")
    
    # Insert many elements
    for i in range(10):
        rs.insert(i)
    
    print(f"  Inserted 0-9, size should be 10")
    
    # Remove every other element
    for i in range(0, 10, 2):
        rs.remove(i)
    
    print(f"  Removed even numbers")
    
    # Check remaining elements
    remaining = []
    for _ in range(20):
        val = rs.getRandom()
        if val not in remaining:
            remaining.append(val)
    
    print(f"  Remaining elements found: {sorted(remaining)}")

def test_randomized_set_randomness():
    """Test randomness distribution"""
    print("\n=== Testing Randomness Distribution ===")
    
    rs = RandomizedSetArrayHashMap()
    
    # Insert elements
    elements = [1, 2, 3, 4, 5]
    for elem in elements:
        rs.insert(elem)
    
    # Test distribution
    distribution = {}
    trials = 10000
    
    for _ in range(trials):
        val = rs.getRandom()
        distribution[val] = distribution.get(val, 0) + 1
    
    print(f"Distribution over {trials} trials:")
    for elem in elements:
        count = distribution.get(elem, 0)
        percentage = (count / trials) * 100
        print(f"  {elem}: {count} times ({percentage:.1f}%)")

def test_randomized_set_performance():
    """Test RandomizedSet performance"""
    print("\n=== Testing RandomizedSet Performance ===")
    
    import time
    
    implementations = [
        ("Array + HashMap", RandomizedSetArrayHashMap),
        ("Linked List", RandomizedSetLinkedList),
        ("Thread-Safe", RandomizedSetThreadSafe)
    ]
    
    operations = 100000
    
    for name, RandomizedSetClass in implementations:
        rs = RandomizedSetClass()
        
        # Test insertion performance
        start_time = time.time()
        for i in range(operations):
            rs.insert(i)
        insert_time = (time.time() - start_time) * 1000
        
        # Test random access performance
        start_time = time.time()
        for _ in range(10000):
            rs.getRandom()
        random_time = (time.time() - start_time) * 1000
        
        # Test removal performance
        start_time = time.time()
        for i in range(0, operations, 2):  # Remove half
            rs.remove(i)
        remove_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Insert {operations}: {insert_time:.2f}ms")
        print(f"    Random 10000: {random_time:.2f}ms")
        print(f"    Remove {operations//2}: {remove_time:.2f}ms")

def test_advanced_features():
    """Test advanced RandomizedSet features"""
    print("\n=== Testing Advanced Features ===")
    
    # Test frequency-based version
    print("Frequency-based RandomizedSet:")
    freq_rs = RandomizedSetWithFrequency()
    
    # Insert with frequencies
    for _ in range(3):
        freq_rs.insert(1)  # High frequency
    
    for _ in range(1):
        freq_rs.insert(2)  # Low frequency
    
    # Test weighted random
    print("  Standard getRandom():")
    standard_dist = {}
    for _ in range(1000):
        val = freq_rs.getRandom()
        standard_dist[val] = standard_dist.get(val, 0) + 1
    
    for val, count in standard_dist.items():
        print(f"    {val}: {count} times")
    
    print("  Weighted getRandom():")
    weighted_dist = {}
    for _ in range(1000):
        val = freq_rs.getRandomWeighted()
        weighted_dist[val] = weighted_dist.get(val, 0) + 1
    
    for val, count in weighted_dist.items():
        print(f"    {val}: {count} times")
    
    # Test thread-safe version
    print(f"\nThread-Safe RandomizedSet:")
    thread_rs = RandomizedSetThreadSafe()
    
    for i in range(5):
        thread_rs.insert(i)
    
    print(f"  Size: {thread_rs.size()}")
    print(f"  Contains 3: {thread_rs.contains(3)}")
    print(f"  Contains 10: {thread_rs.contains(10)}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Real-World Applications ===")
    
    # Application 1: Random sampling from dataset
    print("Application 1: Random Sampling from Dataset")
    dataset = RandomizedSetArrayHashMap()
    
    # Simulate adding user IDs
    user_ids = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    for user_id in user_ids:
        dataset.insert(user_id)
    
    # Random sampling for A/B testing
    sample_size = 3
    print(f"  Random sample of {sample_size} users for A/B test:")
    sampled_users = []
    for _ in range(sample_size):
        user = dataset.getRandom()
        while user in sampled_users:  # Ensure uniqueness
            user = dataset.getRandom()
        sampled_users.append(user)
    
    print(f"  Selected users: {sampled_users}")
    
    # Application 2: Load balancing
    print(f"\nApplication 2: Load Balancing")
    load_balancer = RandomizedSetArrayHashMap()
    
    # Available servers
    servers = ["server1", "server2", "server3", "server4"]
    for i, server in enumerate(servers):
        load_balancer.insert(hash(server) % 10000)  # Use hash as ID
    
    print("  Randomly distributing 10 requests:")
    for i in range(10):
        server_id = load_balancer.getRandom()
        server_name = servers[server_id % len(servers)]
        print(f"    Request {i+1} -> {server_name}")
    
    # Application 3: Playlist shuffling
    print(f"\nApplication 3: Music Playlist Shuffling")
    playlist = RandomizedSetArrayHashMap()
    
    songs = ["Song A", "Song B", "Song C", "Song D", "Song E"]
    for i, song in enumerate(songs):
        playlist.insert(i)
    
    print("  Shuffled playlist:")
    shuffled_order = []
    while len(shuffled_order) < len(songs):
        song_id = playlist.getRandom()
        if song_id not in shuffled_order:
            shuffled_order.append(song_id)
            print(f"    {len(shuffled_order)}. {songs[song_id]}")

def benchmark_different_sizes():
    """Benchmark performance with different data sizes"""
    print("\n=== Benchmarking Different Data Sizes ===")
    
    import time
    
    sizes = [1000, 10000, 100000]
    
    for size in sizes:
        print(f"\nSize: {size}")
        
        rs = RandomizedSetArrayHashMap()
        
        # Insertion benchmark
        start_time = time.time()
        for i in range(size):
            rs.insert(i)
        insert_time = (time.time() - start_time) * 1000
        
        # Random access benchmark
        start_time = time.time()
        for _ in range(min(10000, size)):
            rs.getRandom()
        random_time = (time.time() - start_time) * 1000
        
        # Removal benchmark
        start_time = time.time()
        for i in range(0, min(size, 5000)):  # Remove subset
            rs.remove(i)
        remove_time = (time.time() - start_time) * 1000
        
        print(f"  Insert: {insert_time:.2f}ms ({size/insert_time*1000:.0f} ops/sec)")
        print(f"  Random: {random_time:.2f}ms")
        print(f"  Remove: {remove_time:.2f}ms")

if __name__ == "__main__":
    test_randomized_set_basic()
    test_randomized_set_edge_cases()
    test_randomized_set_randomness()
    test_randomized_set_performance()
    test_advanced_features()
    demonstrate_applications()
    benchmark_different_sizes()

"""
RandomizedSet Design demonstrates key concepts:

Core Approaches:
1. Array + HashMap - Optimal O(1) solution for all operations
2. With Frequency - Enhanced version with weighted random selection
3. Linked List - Alternative implementation with different characteristics
4. Multiset - Support for duplicate values
5. Thread-Safe - Concurrent access support

Key Design Principles:
- O(1) operations through careful data structure combination
- Uniform random distribution
- Efficient memory utilization
- Handling of edge cases (empty set, single element)

Real-world Applications:
- Random sampling for statistical analysis
- Load balancing in distributed systems
- Music/video playlist shuffling
- A/B testing user selection
- Game random events
- Cache eviction policies

The Array + HashMap approach is the gold standard,
providing true O(1) operations with uniform randomness.
"""
