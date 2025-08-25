"""
432. All O(1) Data Structure - Multiple Approaches
Difficulty: Hard

Design a data structure to store the strings' count with the ability to return the strings with minimum and maximum counts.

Implement the AllOne class:
- AllOne() Initializes the object of the data structure.
- inc(String key) Increments the count of the string key by 1. If key does not exist in the data structure, insert it with count 1.
- dec(String key) Decrements the count of the string key by 1. If the count of key is 0 after the decrement, remove it from the data structure. It is guaranteed that key exists in the data structure before the decrement.
- getMaxKey() Returns one of the keys with the maximal count. If no element exists, return an empty string "".
- getMinKey() Returns one of the keys with the minimal count. If no element exists, return an empty string "".

Note: Each function must run in O(1) average time complexity.
"""

from typing import Dict, Set, Optional
from collections import defaultdict

class BucketNode:
    """Node representing a count bucket in doubly linked list"""
    def __init__(self, count: int):
        self.count = count
        self.keys = set()  # Set of keys with this count
        self.prev = None   # Previous bucket (lower count)
        self.next = None   # Next bucket (higher count)

class AllOneOptimal:
    """
    Approach 1: Doubly Linked List with Hash Maps
    
    Use doubly linked list of count buckets with hash maps for O(1) operations.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(n) where n is number of unique keys
    """
    
    def __init__(self):
        # Hash map: key -> count
        self.key_count = {}
        
        # Hash map: count -> BucketNode
        self.count_bucket = {}
        
        # Dummy head and tail for doubly linked list
        self.head = BucketNode(0)
        self.tail = BucketNode(float('inf'))
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_bucket_after(self, prev_bucket: BucketNode, count: int) -> BucketNode:
        """Add new bucket after given bucket"""
        new_bucket = BucketNode(count)
        new_bucket.prev = prev_bucket
        new_bucket.next = prev_bucket.next
        prev_bucket.next.prev = new_bucket
        prev_bucket.next = new_bucket
        
        self.count_bucket[count] = new_bucket
        return new_bucket
    
    def _remove_bucket(self, bucket: BucketNode) -> None:
        """Remove bucket from linked list"""
        bucket.prev.next = bucket.next
        bucket.next.prev = bucket.prev
        del self.count_bucket[bucket.count]
    
    def inc(self, key: str) -> None:
        if key in self.key_count:
            # Key exists, increment count
            old_count = self.key_count[key]
            new_count = old_count + 1
            
            # Update key count
            self.key_count[key] = new_count
            
            # Remove key from old bucket
            old_bucket = self.count_bucket[old_count]
            old_bucket.keys.remove(key)
            
            # Find or create new bucket
            if new_count in self.count_bucket:
                new_bucket = self.count_bucket[new_count]
            else:
                new_bucket = self._add_bucket_after(old_bucket, new_count)
            
            # Add key to new bucket
            new_bucket.keys.add(key)
            
            # Remove old bucket if empty
            if not old_bucket.keys:
                self._remove_bucket(old_bucket)
        else:
            # New key, count = 1
            self.key_count[key] = 1
            
            # Find or create bucket for count 1
            if 1 in self.count_bucket:
                bucket = self.count_bucket[1]
            else:
                bucket = self._add_bucket_after(self.head, 1)
            
            bucket.keys.add(key)
    
    def dec(self, key: str) -> None:
        if key not in self.key_count:
            return
        
        old_count = self.key_count[key]
        new_count = old_count - 1
        
        # Remove key from old bucket
        old_bucket = self.count_bucket[old_count]
        old_bucket.keys.remove(key)
        
        if new_count == 0:
            # Remove key completely
            del self.key_count[key]
        else:
            # Update key count and move to new bucket
            self.key_count[key] = new_count
            
            # Find or create new bucket
            if new_count in self.count_bucket:
                new_bucket = self.count_bucket[new_count]
            else:
                new_bucket = self._add_bucket_after(old_bucket.prev, new_count)
            
            new_bucket.keys.add(key)
        
        # Remove old bucket if empty
        if not old_bucket.keys:
            self._remove_bucket(old_bucket)
    
    def getMaxKey(self) -> str:
        if self.tail.prev == self.head:
            return ""
        return next(iter(self.tail.prev.keys))
    
    def getMinKey(self) -> str:
        if self.head.next == self.tail:
            return ""
        return next(iter(self.head.next.keys))

class AllOneSimple:
    """
    Approach 2: Simple with Separate Min/Max Tracking
    
    Simpler implementation that doesn't guarantee O(1) for all cases.
    
    Time Complexity: 
    - inc/dec: O(1)
    - getMin/getMax: O(n) in worst case
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.key_count = {}
        self.count_keys = defaultdict(set)
        self.min_count = float('inf')
        self.max_count = 0
    
    def inc(self, key: str) -> None:
        if key in self.key_count:
            old_count = self.key_count[key]
            new_count = old_count + 1
            
            # Update mappings
            self.count_keys[old_count].remove(key)
            if not self.count_keys[old_count]:
                del self.count_keys[old_count]
                if old_count == self.min_count:
                    self.min_count = min(self.count_keys.keys()) if self.count_keys else float('inf')
            
            self.key_count[key] = new_count
            self.count_keys[new_count].add(key)
            self.max_count = max(self.max_count, new_count)
        else:
            self.key_count[key] = 1
            self.count_keys[1].add(key)
            self.min_count = min(self.min_count, 1)
            self.max_count = max(self.max_count, 1)
    
    def dec(self, key: str) -> None:
        if key not in self.key_count:
            return
        
        old_count = self.key_count[key]
        new_count = old_count - 1
        
        # Remove from old count
        self.count_keys[old_count].remove(key)
        if not self.count_keys[old_count]:
            del self.count_keys[old_count]
            if old_count == self.max_count:
                self.max_count = max(self.count_keys.keys()) if self.count_keys else 0
        
        if new_count == 0:
            del self.key_count[key]
        else:
            self.key_count[key] = new_count
            self.count_keys[new_count].add(key)
            self.min_count = min(self.min_count, new_count)
        
        # Update min_count if needed
        if not self.count_keys:
            self.min_count = float('inf')
        elif old_count == self.min_count:
            self.min_count = min(self.count_keys.keys())
    
    def getMaxKey(self) -> str:
        if self.max_count == 0:
            return ""
        return next(iter(self.count_keys[self.max_count]))
    
    def getMinKey(self) -> str:
        if self.min_count == float('inf'):
            return ""
        return next(iter(self.count_keys[self.min_count]))

class AllOneAdvanced:
    """
    Approach 3: Advanced with Additional Features
    
    Enhanced version with statistics and debugging capabilities.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(n + k) where k is number of distinct counts
    """
    
    def __init__(self):
        self.key_count = {}
        self.count_bucket = {}
        
        # Dummy nodes
        self.head = BucketNode(0)
        self.tail = BucketNode(float('inf'))
        self.head.next = self.tail
        self.tail.prev = self.head
        
        # Statistics
        self.total_operations = 0
        self.inc_operations = 0
        self.dec_operations = 0
        self.query_operations = 0
    
    def _add_bucket_after(self, prev_bucket: BucketNode, count: int) -> BucketNode:
        new_bucket = BucketNode(count)
        new_bucket.prev = prev_bucket
        new_bucket.next = prev_bucket.next
        prev_bucket.next.prev = new_bucket
        prev_bucket.next = new_bucket
        
        self.count_bucket[count] = new_bucket
        return new_bucket
    
    def _remove_bucket(self, bucket: BucketNode) -> None:
        bucket.prev.next = bucket.next
        bucket.next.prev = bucket.prev
        del self.count_bucket[bucket.count]
    
    def inc(self, key: str) -> None:
        self.total_operations += 1
        self.inc_operations += 1
        
        if key in self.key_count:
            old_count = self.key_count[key]
            new_count = old_count + 1
            
            self.key_count[key] = new_count
            
            old_bucket = self.count_bucket[old_count]
            old_bucket.keys.remove(key)
            
            if new_count in self.count_bucket:
                new_bucket = self.count_bucket[new_count]
            else:
                new_bucket = self._add_bucket_after(old_bucket, new_count)
            
            new_bucket.keys.add(key)
            
            if not old_bucket.keys:
                self._remove_bucket(old_bucket)
        else:
            self.key_count[key] = 1
            
            if 1 in self.count_bucket:
                bucket = self.count_bucket[1]
            else:
                bucket = self._add_bucket_after(self.head, 1)
            
            bucket.keys.add(key)
    
    def dec(self, key: str) -> None:
        self.total_operations += 1
        self.dec_operations += 1
        
        if key not in self.key_count:
            return
        
        old_count = self.key_count[key]
        new_count = old_count - 1
        
        old_bucket = self.count_bucket[old_count]
        old_bucket.keys.remove(key)
        
        if new_count == 0:
            del self.key_count[key]
        else:
            self.key_count[key] = new_count
            
            if new_count in self.count_bucket:
                new_bucket = self.count_bucket[new_count]
            else:
                new_bucket = self._add_bucket_after(old_bucket.prev, new_count)
            
            new_bucket.keys.add(key)
        
        if not old_bucket.keys:
            self._remove_bucket(old_bucket)
    
    def getMaxKey(self) -> str:
        self.total_operations += 1
        self.query_operations += 1
        
        if self.tail.prev == self.head:
            return ""
        return next(iter(self.tail.prev.keys))
    
    def getMinKey(self) -> str:
        self.total_operations += 1
        self.query_operations += 1
        
        if self.head.next == self.tail:
            return ""
        return next(iter(self.head.next.keys))
    
    def get_statistics(self) -> dict:
        """Get operation statistics"""
        return {
            'total_operations': self.total_operations,
            'inc_operations': self.inc_operations,
            'dec_operations': self.dec_operations,
            'query_operations': self.query_operations,
            'unique_keys': len(self.key_count),
            'distinct_counts': len(self.count_bucket)
        }
    
    def get_state(self) -> dict:
        """Get current state for debugging"""
        buckets = []
        current = self.head.next
        
        while current != self.tail:
            buckets.append({
                'count': current.count,
                'keys': list(current.keys)
            })
            current = current.next
        
        return {
            'key_counts': dict(self.key_count),
            'buckets': buckets
        }


def test_all_one_basic():
    """Test basic AllOne functionality"""
    print("=== Testing Basic AllOne Functionality ===")
    
    implementations = [
        ("Optimal O(1)", AllOneOptimal),
        ("Simple", AllOneSimple),
        ("Advanced", AllOneAdvanced)
    ]
    
    for name, AllOneClass in implementations:
        print(f"\n{name}:")
        
        all_one = AllOneClass()
        
        # Test sequence from problem description
        operations = [
            ("inc", "hello"), ("inc", "hello"), ("getMaxKey", None),
            ("getMinKey", None), ("inc", "leet"), ("getMaxKey", None),
            ("getMinKey", None)
        ]
        
        for op, key in operations:
            if op == "inc":
                all_one.inc(key)
                print(f"  inc('{key}')")
            elif op == "dec":
                all_one.dec(key)
                print(f"  dec('{key}')")
            elif op == "getMaxKey":
                result = all_one.getMaxKey()
                print(f"  getMaxKey(): '{result}'")
            elif op == "getMinKey":
                result = all_one.getMinKey()
                print(f"  getMinKey(): '{result}'")

def test_all_one_edge_cases():
    """Test AllOne edge cases"""
    print("\n=== Testing AllOne Edge Cases ===")
    
    all_one = AllOneOptimal()
    
    # Test empty state
    print("Empty state:")
    print(f"  getMaxKey(): '{all_one.getMaxKey()}'")
    print(f"  getMinKey(): '{all_one.getMinKey()}'")
    
    # Test single key operations
    print(f"\nSingle key operations:")
    all_one.inc("single")
    print(f"  After inc('single'):")
    print(f"    getMaxKey(): '{all_one.getMaxKey()}'")
    print(f"    getMinKey(): '{all_one.getMinKey()}'")
    
    all_one.dec("single")
    print(f"  After dec('single'):")
    print(f"    getMaxKey(): '{all_one.getMaxKey()}'")
    print(f"    getMinKey(): '{all_one.getMinKey()}'")
    
    # Test multiple increments of same key
    print(f"\nMultiple increments:")
    for i in range(5):
        all_one.inc("test")
        max_key = all_one.getMaxKey()
        min_key = all_one.getMinKey()
        print(f"    After {i+1} inc('test'): max='{max_key}', min='{min_key}'")

def test_all_one_complex():
    """Test complex AllOne scenarios"""
    print("\n=== Testing Complex AllOne Scenarios ===")
    
    all_one = AllOneOptimal()
    
    # Create scenario with multiple keys at different counts
    keys_and_counts = [
        ("a", 3), ("b", 1), ("c", 2), ("d", 3), ("e", 1)
    ]
    
    print("Building complex state:")
    for key, count in keys_and_counts:
        for _ in range(count):
            all_one.inc(key)
        print(f"  Added '{key}' {count} times")
    
    print(f"\nAfter building:")
    print(f"  getMaxKey(): '{all_one.getMaxKey()}'")  # Should be 'a' or 'd'
    print(f"  getMinKey(): '{all_one.getMinKey()}'")  # Should be 'b' or 'e'
    
    # Modify counts and test
    print(f"\nModifying counts:")
    all_one.dec("a")  # a: 3->2
    all_one.inc("b")  # b: 1->2
    all_one.inc("e")  # e: 1->2
    all_one.inc("e")  # e: 2->3
    all_one.inc("e")  # e: 3->4
    
    print(f"  After modifications:")
    print(f"    getMaxKey(): '{all_one.getMaxKey()}'")  # Should be 'e'
    print(f"    getMinKey(): '{all_one.getMinKey()}'")  # Should be 'a', 'b', or 'c'

def test_performance():
    """Test performance of AllOne operations"""
    print("\n=== Testing AllOne Performance ===")
    
    import time
    
    implementations = [
        ("Optimal O(1)", AllOneOptimal),
        ("Simple", AllOneSimple)
    ]
    
    num_operations = 10000
    
    for name, AllOneClass in implementations:
        all_one = AllOneClass()
        
        start_time = time.time()
        
        # Mix of operations
        for i in range(num_operations):
            if i % 4 == 0:
                all_one.inc(f"key_{i % 100}")
            elif i % 4 == 1:
                all_one.getMaxKey()
            elif i % 4 == 2:
                all_one.getMinKey()
            else:
                if i > 100:  # Ensure some keys exist to decrement
                    all_one.dec(f"key_{(i-100) % 100}")
        
        elapsed = (time.time() - start_time) * 1000
        avg_time = elapsed / num_operations
        
        print(f"  {name}: {elapsed:.2f}ms total, {avg_time:.4f}ms per operation")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    all_one = AllOneAdvanced()
    
    # Perform various operations
    operations = [
        ("inc", "hello"), ("inc", "world"), ("inc", "hello"),
        ("getMaxKey", None), ("inc", "test"), ("dec", "world"),
        ("getMinKey", None), ("inc", "hello")
    ]
    
    print("Performing operations with statistics tracking:")
    
    for op, key in operations:
        if op == "inc":
            all_one.inc(key)
            print(f"  inc('{key}')")
        elif op == "dec":
            all_one.dec(key)
            print(f"  dec('{key}')")
        elif op == "getMaxKey":
            result = all_one.getMaxKey()
            print(f"  getMaxKey(): '{result}'")
        elif op == "getMinKey":
            result = all_one.getMinKey()
            print(f"  getMinKey(): '{result}'")
    
    # Get statistics
    stats = all_one.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get state
    state = all_one.get_state()
    print(f"\nCurrent state:")
    print(f"  Key counts: {state['key_counts']}")
    print(f"  Buckets: {state['buckets']}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Frequency-based cache eviction
    print("Application 1: Frequency-based Cache Management")
    
    cache_tracker = AllOneOptimal()
    
    # Simulate cache access patterns
    accesses = [
        "page1", "page2", "page1", "page3", "page2", "page1",
        "page4", "page2", "page1", "page5"
    ]
    
    print("  Simulating page accesses:")
    for page in accesses:
        cache_tracker.inc(page)
        print(f"    Access {page}")
    
    print(f"  Most accessed page: {cache_tracker.getMaxKey()}")
    print(f"  Least accessed page: {cache_tracker.getMinKey()}")
    
    # Application 2: Real-time leaderboard
    print(f"\nApplication 2: Gaming Leaderboard")
    
    leaderboard = AllOneOptimal()
    
    # Player scores (simplified as counts)
    score_updates = [
        ("Alice", 5), ("Bob", 3), ("Charlie", 7), ("Alice", 2),
        ("Bob", 4), ("David", 6), ("Charlie", 1)
    ]
    
    print("  Score updates:")
    for player, points in score_updates:
        for _ in range(points):
            leaderboard.inc(player)
        print(f"    {player} scored {points} points")
    
    print(f"  Current leader: {leaderboard.getMaxKey()}")
    print(f"  Lowest scorer: {leaderboard.getMinKey()}")
    
    # Application 3: Word frequency analysis
    print(f"\nApplication 3: Word Frequency Analysis")
    
    word_counter = AllOneOptimal()
    
    # Simulate text processing
    text = "the quick brown fox jumps over the lazy dog the fox is quick"
    words = text.split()
    
    print(f"  Analyzing text: '{text}'")
    
    for word in words:
        word_counter.inc(word)
    
    print(f"  Most frequent word: '{word_counter.getMaxKey()}'")
    print(f"  Least frequent word: '{word_counter.getMinKey()}'")

def test_stress_scenarios():
    """Test stress scenarios"""
    print("\n=== Testing Stress Scenarios ===")
    
    all_one = AllOneOptimal()
    
    # Stress test 1: Many keys with same count
    print("Stress test 1: Many keys with same count")
    
    num_keys = 1000
    for i in range(num_keys):
        all_one.inc(f"key_{i}")
    
    max_key = all_one.getMaxKey()
    min_key = all_one.getMinKey()
    
    print(f"  Added {num_keys} keys with count 1")
    print(f"  getMaxKey(): '{max_key[:10]}...' (showing first 10 chars)")
    print(f"  getMinKey(): '{min_key[:10]}...' (showing first 10 chars)")
    
    # Stress test 2: Rapid inc/dec cycles
    print(f"\nStress test 2: Rapid inc/dec cycles")
    
    all_one2 = AllOneOptimal()
    
    # Create some initial state
    for i in range(10):
        all_one2.inc(f"test_{i}")
    
    # Rapid cycles
    for cycle in range(100):
        for i in range(5):
            all_one2.inc(f"test_{i}")
        for i in range(5):
            all_one2.dec(f"test_{i}")
    
    print(f"  Completed 100 inc/dec cycles")
    print(f"  Final max key: '{all_one2.getMaxKey()}'")
    print(f"  Final min key: '{all_one2.getMinKey()}'")

def test_memory_efficiency():
    """Test memory efficiency"""
    print("\n=== Testing Memory Efficiency ===")
    
    implementations = [
        ("Optimal O(1)", AllOneOptimal),
        ("Simple", AllOneSimple)
    ]
    
    for name, AllOneClass in implementations:
        all_one = AllOneClass()
        
        # Add many keys
        num_keys = 1000
        for i in range(num_keys):
            key = f"key_{i:04d}"
            # Varying counts
            count = (i % 10) + 1
            for _ in range(count):
                all_one.inc(key)
        
        # Estimate memory usage
        if hasattr(all_one, 'key_count'):
            keys_memory = len(all_one.key_count)
        else:
            keys_memory = num_keys
        
        if hasattr(all_one, 'count_bucket'):
            buckets_memory = len(all_one.count_bucket)
        else:
            buckets_memory = 10  # Estimate based on count range
        
        total_memory = keys_memory + buckets_memory
        
        print(f"  {name}:")
        print(f"    Keys: {keys_memory}, Buckets: {buckets_memory}")
        print(f"    Total memory units: ~{total_memory}")

def benchmark_operations():
    """Benchmark individual operations"""
    print("\n=== Benchmarking Individual Operations ===")
    
    import time
    
    all_one = AllOneOptimal()
    
    # Prepare some data
    for i in range(100):
        all_one.inc(f"key_{i}")
    
    # Benchmark inc
    start_time = time.time()
    for i in range(10000):
        all_one.inc(f"bench_{i % 50}")
    inc_time = (time.time() - start_time) * 1000
    
    # Benchmark getMax
    start_time = time.time()
    for _ in range(10000):
        all_one.getMaxKey()
    max_time = (time.time() - start_time) * 1000
    
    # Benchmark getMin
    start_time = time.time()
    for _ in range(10000):
        all_one.getMinKey()
    min_time = (time.time() - start_time) * 1000
    
    # Benchmark dec
    start_time = time.time()
    for i in range(5000):  # Fewer to avoid removing all keys
        all_one.dec(f"bench_{i % 50}")
    dec_time = (time.time() - start_time) * 1000
    
    print(f"  10000 inc operations: {inc_time:.2f}ms ({inc_time/10000:.4f}ms each)")
    print(f"  10000 getMax operations: {max_time:.2f}ms ({max_time/10000:.4f}ms each)")
    print(f"  10000 getMin operations: {min_time:.2f}ms ({min_time/10000:.4f}ms each)")
    print(f"  5000 dec operations: {dec_time:.2f}ms ({dec_time/5000:.4f}ms each)")

def test_bucket_management():
    """Test bucket management in detail"""
    print("\n=== Testing Bucket Management ===")
    
    all_one = AllOneAdvanced()
    
    # Create and destroy buckets
    print("Testing bucket creation and destruction:")
    
    # Create buckets for counts 1, 2, 3
    keys = ["a", "b", "c"]
    for i, key in enumerate(keys):
        for _ in range(i + 1):
            all_one.inc(key)
    
    state = all_one.get_state()
    print(f"  After creating a:1, b:2, c:3")
    print(f"  Buckets: {[b['count'] for b in state['buckets']]}")
    
    # Remove middle bucket
    all_one.dec("b")
    all_one.dec("b")  # b now has count 0, should be removed
    
    state = all_one.get_state()
    print(f"  After removing b completely")
    print(f"  Buckets: {[b['count'] for b in state['buckets']]}")
    
    # Add new key with count that fills gap
    all_one.inc("d")
    all_one.inc("d")  # d now has count 2
    
    state = all_one.get_state()
    print(f"  After adding d with count 2")
    print(f"  Buckets: {[b['count'] for b in state['buckets']]}")

if __name__ == "__main__":
    test_all_one_basic()
    test_all_one_edge_cases()
    test_all_one_complex()
    test_performance()
    test_advanced_features()
    demonstrate_applications()
    test_stress_scenarios()
    test_memory_efficiency()
    benchmark_operations()
    test_bucket_management()

"""
All O(1) Data Structure Design demonstrates key concepts:

Core Approaches:
1. Optimal O(1) - Doubly linked list of count buckets with hash maps
2. Simple - Basic implementation with separate min/max tracking
3. Advanced - Enhanced with statistics and debugging features

Key Design Principles:
- Doubly linked list for ordered count buckets
- Hash maps for O(1) key lookup and count-to-bucket mapping
- Lazy bucket creation and eager bucket removal
- Maintaining pointers for constant-time min/max access

Data Structure Components:
- key_count: HashMap key -> count
- count_bucket: HashMap count -> BucketNode
- Doubly linked list: Ordered buckets by count
- Each bucket: Set of keys with same count

Performance Characteristics:
- All operations: O(1) average time complexity
- Space complexity: O(n) where n is number of unique keys
- Bucket operations: O(1) insertion, deletion, lookup

Real-world Applications:
- Frequency-based cache eviction (LFU cache support)
- Real-time leaderboards and ranking systems
- Word frequency analysis in text processing
- Access pattern monitoring in databases
- Gaming score tracking systems
- Social media engagement metrics

The optimal approach using doubly linked list with buckets
is the standard solution for achieving true O(1) complexity
for all operations while maintaining correct min/max tracking.
"""
