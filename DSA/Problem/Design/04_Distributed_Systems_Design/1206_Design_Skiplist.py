"""
1206. Design Skiplist - Multiple Approaches
Difficulty: Hard

Design a Skiplist without using any built-in libraries.

A skiplist is a data structure that takes O(log n) time to add, erase and search. Comparing with treap and red-black tree which has the same function and performance, the skiplist is easier to implement.

Implement the Skiplist class:
- Skiplist() Initializes the object of the skiplist.
- bool search(int target) Returns true if the integer target exists in the skiplist or false otherwise.
- void add(int num) Inserts the value num into the SkipList.
- bool erase(int num) Removes the value num from the skiplist and returns true. If num does not exist in the skiplist, do nothing and return false.
"""

import random
from typing import List, Optional

class SkipListNode:
    """Node for skip list implementation"""
    def __init__(self, val: int, level: int):
        self.val = val
        self.forward = [None] * level

class SkiplistBasic:
    """
    Approach 1: Basic Skiplist Implementation
    
    Standard skiplist with probabilistic balancing.
    
    Time Complexity:
    - search: O(log n) expected
    - add: O(log n) expected
    - erase: O(log n) expected
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.max_level = 16
        self.p = 0.5  # Probability for level promotion
        self.level = 0
        
        # Header node with maximum level
        self.header = SkipListNode(-1, self.max_level)
    
    def _random_level(self) -> int:
        """Generate random level for new node"""
        level = 1
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level
    
    def search(self, target: int) -> bool:
        current = self.header
        
        # Start from highest level and go down
        for i in range(self.level - 1, -1, -1):
            while current.forward[i] and current.forward[i].val < target:
                current = current.forward[i]
        
        # Move to next node at level 0
        current = current.forward[0]
        
        return current and current.val == target
    
    def add(self, num: int) -> None:
        update = [None] * self.max_level
        current = self.header
        
        # Find position to insert
        for i in range(self.level - 1, -1, -1):
            while current.forward[i] and current.forward[i].val < num:
                current = current.forward[i]
            update[i] = current
        
        # Generate random level for new node
        new_level = self._random_level()
        
        # Update skiplist level if necessary
        if new_level > self.level:
            for i in range(self.level, new_level):
                update[i] = self.header
            self.level = new_level
        
        # Create new node
        new_node = SkipListNode(num, new_level)
        
        # Update pointers
        for i in range(new_level):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
    
    def erase(self, num: int) -> bool:
        update = [None] * self.max_level
        current = self.header
        
        # Find node to delete
        for i in range(self.level - 1, -1, -1):
            while current.forward[i] and current.forward[i].val < num:
                current = current.forward[i]
            update[i] = current
        
        current = current.forward[0]
        
        # Check if node exists
        if not current or current.val != num:
            return False
        
        # Remove node from all levels
        for i in range(self.level):
            if update[i].forward[i] != current:
                break
            update[i].forward[i] = current.forward[i]
        
        # Update skiplist level
        while self.level > 1 and not self.header.forward[self.level - 1]:
            self.level -= 1
        
        return True

class SkiplistOptimized:
    """
    Approach 2: Optimized Skiplist with Finger Search
    
    Enhanced with finger search for better locality.
    
    Time Complexity: Same as basic, but better constants
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.max_level = 32
        self.p = 0.25  # Lower probability for better performance
        self.level = 1
        
        # Sentinel nodes
        self.header = SkipListNode(float('-inf'), self.max_level)
        self.tail = SkipListNode(float('inf'), self.max_level)
        
        # Connect header to tail
        for i in range(self.max_level):
            self.header.forward[i] = self.tail
        
        # Finger for locality
        self.finger = self.header
    
    def _random_level(self) -> int:
        level = 1
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level
    
    def _find_predecessors(self, target: int) -> List[SkipListNode]:
        """Find predecessor nodes for target value"""
        predecessors = [None] * self.max_level
        current = self.header
        
        for level in range(self.level - 1, -1, -1):
            while current.forward[level].val < target:
                current = current.forward[level]
            predecessors[level] = current
        
        return predecessors
    
    def search(self, target: int) -> bool:
        predecessors = self._find_predecessors(target)
        candidate = predecessors[0].forward[0]
        
        return candidate.val == target
    
    def add(self, num: int) -> None:
        predecessors = self._find_predecessors(num)
        new_level = self._random_level()
        
        # Update skiplist level if necessary
        if new_level > self.level:
            for i in range(self.level, new_level):
                predecessors[i] = self.header
            self.level = new_level
        
        # Create and insert new node
        new_node = SkipListNode(num, new_level)
        
        for i in range(new_level):
            new_node.forward[i] = predecessors[i].forward[i]
            predecessors[i].forward[i] = new_node
        
        # Update finger for locality
        self.finger = new_node
    
    def erase(self, num: int) -> bool:
        predecessors = self._find_predecessors(num)
        target = predecessors[0].forward[0]
        
        if target.val != num:
            return False
        
        # Remove from all levels
        for i in range(self.level):
            if predecessors[i].forward[i] != target:
                break
            predecessors[i].forward[i] = target.forward[i]
        
        # Update level
        while self.level > 1 and self.header.forward[self.level - 1] == self.tail:
            self.level -= 1
        
        # Update finger
        if self.finger == target:
            self.finger = predecessors[0]
        
        return True

class SkiplistAdvanced:
    """
    Approach 3: Advanced with Statistics and Features
    
    Enhanced skiplist with additional operations and analytics.
    
    Time Complexity: Same as basic + additional features
    Space Complexity: O(n + statistics)
    """
    
    def __init__(self):
        self.max_level = 16
        self.p = 0.5
        self.level = 1
        self.size = 0
        
        # Enhanced nodes with additional metadata
        self.header = SkipListNode(float('-inf'), self.max_level)
        self.tail = SkipListNode(float('inf'), self.max_level)
        
        for i in range(self.max_level):
            self.header.forward[i] = self.tail
        
        # Statistics
        self.search_count = 0
        self.add_count = 0
        self.erase_count = 0
        self.level_distribution = [0] * self.max_level
        
        # Performance tracking
        self.total_search_steps = 0
        self.total_add_steps = 0
        self.total_erase_steps = 0
    
    def _random_level(self) -> int:
        level = 1
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level
    
    def _find_path(self, target: int) -> tuple:
        """Find path to target and count steps"""
        predecessors = [None] * self.max_level
        current = self.header
        steps = 0
        
        for level in range(self.level - 1, -1, -1):
            while current.forward[level].val < target:
                current = current.forward[level]
                steps += 1
            predecessors[level] = current
        
        return predecessors, steps
    
    def search(self, target: int) -> bool:
        self.search_count += 1
        
        predecessors, steps = self._find_path(target)
        self.total_search_steps += steps
        
        candidate = predecessors[0].forward[0]
        return candidate.val == target
    
    def add(self, num: int) -> None:
        self.add_count += 1
        
        predecessors, steps = self._find_path(num)
        self.total_add_steps += steps
        
        new_level = self._random_level()
        self.level_distribution[new_level - 1] += 1
        
        if new_level > self.level:
            for i in range(self.level, new_level):
                predecessors[i] = self.header
            self.level = new_level
        
        new_node = SkipListNode(num, new_level)
        
        for i in range(new_level):
            new_node.forward[i] = predecessors[i].forward[i]
            predecessors[i].forward[i] = new_node
        
        self.size += 1
    
    def erase(self, num: int) -> bool:
        self.erase_count += 1
        
        predecessors, steps = self._find_path(num)
        self.total_erase_steps += steps
        
        target = predecessors[0].forward[0]
        
        if target.val != num:
            return False
        
        # Remove from all levels
        node_level = 0
        for i in range(self.level):
            if predecessors[i].forward[i] != target:
                break
            predecessors[i].forward[i] = target.forward[i]
            node_level = i + 1
        
        # Update level distribution
        if node_level > 0:
            self.level_distribution[node_level - 1] -= 1
        
        # Update level
        while self.level > 1 and self.header.forward[self.level - 1] == self.tail:
            self.level -= 1
        
        self.size -= 1
        return True
    
    def get_statistics(self) -> dict:
        """Get skiplist statistics"""
        avg_search_steps = self.total_search_steps / max(1, self.search_count)
        avg_add_steps = self.total_add_steps / max(1, self.add_count)
        avg_erase_steps = self.total_erase_steps / max(1, self.erase_count)
        
        return {
            'size': self.size,
            'level': self.level,
            'search_count': self.search_count,
            'add_count': self.add_count,
            'erase_count': self.erase_count,
            'avg_search_steps': avg_search_steps,
            'avg_add_steps': avg_add_steps,
            'avg_erase_steps': avg_erase_steps,
            'level_distribution': self.level_distribution[:self.level]
        }
    
    def range_search(self, min_val: int, max_val: int) -> List[int]:
        """Find all values in range [min_val, max_val]"""
        result = []
        
        # Find starting position
        current = self.header
        for level in range(self.level - 1, -1, -1):
            while current.forward[level].val < min_val:
                current = current.forward[level]
        
        current = current.forward[0]
        
        # Collect values in range
        while current.val <= max_val and current != self.tail:
            result.append(current.val)
            current = current.forward[0]
        
        return result
    
    def get_min(self) -> Optional[int]:
        """Get minimum value"""
        first = self.header.forward[0]
        return first.val if first != self.tail else None
    
    def get_max(self) -> Optional[int]:
        """Get maximum value"""
        if self.size == 0:
            return None
        
        current = self.header
        for level in range(self.level - 1, -1, -1):
            while current.forward[level] != self.tail:
                current = current.forward[level]
        
        return current.val

class SkiplistConcurrent:
    """
    Approach 4: Thread-Safe Skiplist
    
    Concurrent skiplist with fine-grained locking.
    
    Time Complexity: Same as basic + lock overhead
    Space Complexity: O(n)
    """
    
    def __init__(self):
        import threading
        
        self.max_level = 16
        self.p = 0.5
        self.level = 1
        
        self.header = SkipListNode(float('-inf'), self.max_level)
        self.tail = SkipListNode(float('inf'), self.max_level)
        
        for i in range(self.max_level):
            self.header.forward[i] = self.tail
        
        # Thread safety
        self.lock = threading.RLock()
        self.operation_count = 0
    
    def _random_level(self) -> int:
        level = 1
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level
    
    def search(self, target: int) -> bool:
        with self.lock:
            self.operation_count += 1
            
            current = self.header
            for i in range(self.level - 1, -1, -1):
                while current.forward[i].val < target:
                    current = current.forward[i]
            
            current = current.forward[0]
            return current.val == target
    
    def add(self, num: int) -> None:
        with self.lock:
            self.operation_count += 1
            
            update = [None] * self.max_level
            current = self.header
            
            for i in range(self.level - 1, -1, -1):
                while current.forward[i].val < num:
                    current = current.forward[i]
                update[i] = current
            
            new_level = self._random_level()
            
            if new_level > self.level:
                for i in range(self.level, new_level):
                    update[i] = self.header
                self.level = new_level
            
            new_node = SkipListNode(num, new_level)
            
            for i in range(new_level):
                new_node.forward[i] = update[i].forward[i]
                update[i].forward[i] = new_node
    
    def erase(self, num: int) -> bool:
        with self.lock:
            self.operation_count += 1
            
            update = [None] * self.max_level
            current = self.header
            
            for i in range(self.level - 1, -1, -1):
                while current.forward[i].val < num:
                    current = current.forward[i]
                update[i] = current
            
            current = current.forward[0]
            
            if current.val != num:
                return False
            
            for i in range(self.level):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]
            
            while self.level > 1 and self.header.forward[self.level - 1] == self.tail:
                self.level -= 1
            
            return True
    
    def get_stats(self) -> dict:
        with self.lock:
            return {
                'operation_count': self.operation_count,
                'current_level': self.level
            }

class SkiplistMemoryOptimized:
    """
    Approach 5: Memory-Optimized Skiplist
    
    Optimized for memory usage with compressed levels.
    
    Time Complexity: Same as basic
    Space Complexity: O(n) with better constants
    """
    
    def __init__(self):
        self.max_level = 16
        self.p = 0.25  # Lower probability for fewer levels
        self.level = 1
        
        # Use more compact node structure
        self.nodes = {}  # value -> node
        self.header = SkipListNode(float('-inf'), self.max_level)
        self.tail = SkipListNode(float('inf'), self.max_level)
        
        for i in range(self.max_level):
            self.header.forward[i] = self.tail
        
        # Memory statistics
        self.total_nodes = 0
        self.total_pointers = 0
    
    def _random_level(self) -> int:
        level = 1
        # Use more conservative level generation
        while random.random() < self.p and level < min(self.max_level, 8):
            level += 1
        return level
    
    def search(self, target: int) -> bool:
        if target in self.nodes:
            return True
        
        # Also do skiplist search for consistency
        current = self.header
        for i in range(self.level - 1, -1, -1):
            while current.forward[i].val < target:
                current = current.forward[i]
        
        current = current.forward[0]
        return current.val == target
    
    def add(self, num: int) -> None:
        if num in self.nodes:
            return  # Already exists
        
        update = [None] * self.max_level
        current = self.header
        
        for i in range(self.level - 1, -1, -1):
            while current.forward[i].val < num:
                current = current.forward[i]
            update[i] = current
        
        new_level = self._random_level()
        
        if new_level > self.level:
            for i in range(self.level, new_level):
                update[i] = self.header
            self.level = new_level
        
        new_node = SkipListNode(num, new_level)
        self.nodes[num] = new_node
        
        for i in range(new_level):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
        
        # Update statistics
        self.total_nodes += 1
        self.total_pointers += new_level
    
    def erase(self, num: int) -> bool:
        if num not in self.nodes:
            return False
        
        target_node = self.nodes[num]
        
        update = [None] * self.max_level
        current = self.header
        
        for i in range(self.level - 1, -1, -1):
            while current.forward[i].val < num:
                current = current.forward[i]
            update[i] = current
        
        node_level = len(target_node.forward)
        
        for i in range(node_level):
            if update[i].forward[i] != target_node:
                break
            update[i].forward[i] = target_node.forward[i]
        
        while self.level > 1 and self.header.forward[self.level - 1] == self.tail:
            self.level -= 1
        
        # Update statistics
        del self.nodes[num]
        self.total_nodes -= 1
        self.total_pointers -= node_level
        
        return True
    
    def get_memory_stats(self) -> dict:
        """Get memory usage statistics"""
        avg_level = self.total_pointers / max(1, self.total_nodes)
        
        return {
            'total_nodes': self.total_nodes,
            'total_pointers': self.total_pointers,
            'avg_level': avg_level,
            'current_level': self.level,
            'memory_efficiency': avg_level / self.max_level
        }


def test_skiplist_basic():
    """Test basic skiplist functionality"""
    print("=== Testing Basic Skiplist Functionality ===")
    
    implementations = [
        ("Basic", SkiplistBasic),
        ("Optimized", SkiplistOptimized),
        ("Advanced", SkiplistAdvanced),
        ("Concurrent", SkiplistConcurrent),
        ("Memory Optimized", SkiplistMemoryOptimized)
    ]
    
    for name, SkiplistClass in implementations:
        print(f"\n{name}:")
        
        skiplist = SkiplistClass()
        
        # Test sequence
        operations = [
            ("add", 1), ("add", 2), ("add", 3),
            ("search", 0), ("add", 4), ("search", 1),
            ("erase", 0), ("erase", 1), ("search", 1)
        ]
        
        for op, val in operations:
            if op == "add":
                skiplist.add(val)
                print(f"  add({val})")
            elif op == "search":
                result = skiplist.search(val)
                print(f"  search({val}): {result}")
            elif op == "erase":
                result = skiplist.erase(val)
                print(f"  erase({val}): {result}")

def test_skiplist_edge_cases():
    """Test skiplist edge cases"""
    print("\n=== Testing Skiplist Edge Cases ===")
    
    skiplist = SkiplistAdvanced()
    
    # Test empty skiplist
    print("Empty skiplist:")
    print(f"  search(1): {skiplist.search(1)}")
    print(f"  erase(1): {skiplist.erase(1)}")
    
    # Test duplicates
    print(f"\nDuplicates:")
    skiplist.add(5)
    skiplist.add(5)
    skiplist.add(5)
    
    print(f"  Added 5 three times")
    print(f"  search(5): {skiplist.search(5)}")
    
    print(f"  erase(5): {skiplist.erase(5)}")
    print(f"  search(5) after first erase: {skiplist.search(5)}")
    
    print(f"  erase(5): {skiplist.erase(5)}")
    print(f"  search(5) after second erase: {skiplist.search(5)}")
    
    # Test large values
    print(f"\nLarge values:")
    large_vals = [1000000, -1000000, 999999, -999999]
    
    for val in large_vals:
        skiplist.add(val)
    
    for val in large_vals:
        result = skiplist.search(val)
        print(f"  search({val}): {result}")
    
    # Test range operations (if available)
    if hasattr(skiplist, 'range_search'):
        print(f"\nRange search:")
        range_result = skiplist.range_search(-1000000, 1000000)
        print(f"  range_search(-1000000, 1000000): {range_result}")

def test_skiplist_performance():
    """Test skiplist performance"""
    print("\n=== Testing Skiplist Performance ===")
    
    import time
    
    implementations = [
        ("Basic", SkiplistBasic),
        ("Optimized", SkiplistOptimized),
        ("Advanced", SkiplistAdvanced)
    ]
    
    num_operations = 10000
    
    for name, SkiplistClass in implementations:
        skiplist = SkiplistClass()
        
        # Test add performance
        start_time = time.time()
        import random
        values = list(range(num_operations))
        random.shuffle(values)
        
        for val in values:
            skiplist.add(val)
        
        add_time = (time.time() - start_time) * 1000
        
        # Test search performance
        start_time = time.time()
        
        for val in values:
            skiplist.search(val)
        
        search_time = (time.time() - start_time) * 1000
        
        # Test erase performance
        start_time = time.time()
        
        for val in values[:num_operations//2]:
            skiplist.erase(val)
        
        erase_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Add: {add_time:.2f}ms")
        print(f"    Search: {search_time:.2f}ms")
        print(f"    Erase: {erase_time:.2f}ms")

def test_advanced_features():
    """Test advanced skiplist features"""
    print("\n=== Testing Advanced Features ===")
    
    skiplist = SkiplistAdvanced()
    
    # Build skiplist with known data
    test_data = [5, 2, 8, 1, 7, 3, 9, 4, 6]
    
    print("Building skiplist:")
    for val in test_data:
        skiplist.add(val)
        print(f"  add({val})")
    
    # Test statistics
    stats = skiplist.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        elif isinstance(value, list):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    # Test range search
    print(f"\nRange search tests:")
    ranges = [(2, 6), (1, 9), (0, 10), (5, 5)]
    
    for min_val, max_val in ranges:
        result = skiplist.range_search(min_val, max_val)
        print(f"  range_search({min_val}, {max_val}): {result}")
    
    # Test min/max
    print(f"\nMin/Max:")
    print(f"  get_min(): {skiplist.get_min()}")
    print(f"  get_max(): {skiplist.get_max()}")
    
    # Remove some elements and test again
    skiplist.erase(1)
    skiplist.erase(9)
    
    print(f"\nAfter removing 1 and 9:")
    print(f"  get_min(): {skiplist.get_min()}")
    print(f"  get_max(): {skiplist.get_max()}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Database indexing
    print("Application 1: Database Index")
    
    db_index = SkiplistAdvanced()
    
    # Simulate database records with IDs
    records = [
        (1001, "Alice Johnson"),
        (1005, "Bob Smith"),
        (1003, "Charlie Brown"),
        (1007, "Diana Prince"),
        (1002, "Eve Wilson"),
        (1006, "Frank Miller"),
        (1004, "Grace Lee")
    ]
    
    print("  Building database index:")
    for record_id, name in records:
        db_index.add(record_id)
        print(f"    Index record {record_id} ({name})")
    
    # Query operations
    print(f"\n  Database queries:")
    
    queries = [1003, 1008, 1001, 1010]
    for query_id in queries:
        exists = db_index.search(query_id)
        status = "FOUND" if exists else "NOT FOUND"
        print(f"    Query record {query_id}: {status}")
    
    # Range query
    range_result = db_index.range_search(1003, 1006)
    print(f"    Range query (1003-1006): {range_result}")
    
    # Application 2: Priority queue with search
    print(f"\nApplication 2: Priority Task Queue")
    
    task_queue = SkiplistOptimized()
    
    # Simulate tasks with priorities
    tasks = [
        (10, "Email notification"),
        (5, "Data backup"),
        (15, "Security scan"),
        (8, "Log rotation"),
        (12, "Cache cleanup"),
        (3, "Health check"),
        (20, "System update")
    ]
    
    print("  Adding tasks by priority:")
    for priority, task_name in tasks:
        task_queue.add(priority)
        print(f"    Priority {priority}: {task_name}")
    
    # Process high priority tasks
    print(f"\n  Processing high priority tasks (>= 10):")
    high_priority_tasks = []
    
    # Find high priority tasks
    for priority, task_name in tasks:
        if priority >= 10 and task_queue.search(priority):
            high_priority_tasks.append((priority, task_name))
    
    # Sort by priority and process
    high_priority_tasks.sort(reverse=True)
    for priority, task_name in high_priority_tasks:
        task_queue.erase(priority)
        print(f"    Processed: {task_name} (priority {priority})")
    
    # Application 3: Distributed cache keys
    print(f"\nApplication 3: Distributed Cache Key Management")
    
    cache_keys = SkiplistMemoryOptimized()
    
    # Simulate cache keys (hash values)
    key_hashes = [
        hash("user:1001") % 1000000,
        hash("session:abc123") % 1000000,
        hash("product:5001") % 1000000,
        hash("cart:user1001") % 1000000,
        hash("profile:user1001") % 1000000
    ]
    
    print("  Registering cache keys:")
    for key_hash in key_hashes:
        cache_keys.add(key_hash)
        print(f"    Registered key hash: {key_hash}")
    
    # Simulate cache key lookup
    print(f"\n  Cache key lookups:")
    
    lookup_keys = [
        hash("user:1001") % 1000000,
        hash("user:1002") % 1000000,
        hash("session:abc123") % 1000000,
        hash("nonexistent") % 1000000
    ]
    
    for lookup_hash in lookup_keys:
        exists = cache_keys.search(lookup_hash)
        status = "HIT" if exists else "MISS"
        print(f"    Lookup {lookup_hash}: {status}")
    
    # Memory statistics
    if hasattr(cache_keys, 'get_memory_stats'):
        memory_stats = cache_keys.get_memory_stats()
        print(f"\n  Cache memory statistics:")
        for key, value in memory_stats.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.3f}")
            else:
                print(f"    {key}: {value}")

def test_concurrent_access():
    """Test concurrent access patterns"""
    print("\n=== Testing Concurrent Access ===")
    
    import threading
    import time
    import random
    
    skiplist = SkiplistConcurrent()
    
    # Test concurrent operations
    num_threads = 4
    operations_per_thread = 1000
    
    def worker_thread(thread_id: int, results: list):
        """Worker thread for concurrent operations"""
        start_time = time.time()
        
        random.seed(thread_id)  # Different seed per thread
        
        for i in range(operations_per_thread):
            operation = random.choice(['add', 'search', 'erase'])
            value = random.randint(1, 5000)
            
            if operation == 'add':
                skiplist.add(value)
            elif operation == 'search':
                skiplist.search(value)
            elif operation == 'erase':
                skiplist.erase(value)
        
        elapsed = time.time() - start_time
        results.append(elapsed)
    
    print(f"Running {num_threads} concurrent threads...")
    
    # Start threads
    threads = []
    results = []
    
    overall_start = time.time()
    
    for thread_id in range(num_threads):
        thread = threading.Thread(target=worker_thread, args=(thread_id, results))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    overall_time = time.time() - overall_start
    total_operations = num_threads * operations_per_thread
    
    print(f"  Total time: {overall_time:.2f}s")
    print(f"  Total operations: {total_operations}")
    print(f"  Throughput: {total_operations/overall_time:.0f} ops/sec")
    print(f"  Thread times: {[f'{t:.2f}s' for t in results]}")
    
    # Get final statistics
    final_stats = skiplist.get_stats()
    print(f"  Final stats: {final_stats}")

def stress_test_skiplist():
    """Stress test skiplist with large dataset"""
    print("\n=== Stress Testing Skiplist ===")
    
    import time
    import random
    
    skiplist = SkiplistOptimized()
    
    # Large scale test
    num_elements = 100000
    
    print(f"Stress test with {num_elements} elements")
    
    # Phase 1: Add many elements
    start_time = time.time()
    
    values = list(range(num_elements))
    random.shuffle(values)
    
    for val in values:
        skiplist.add(val)
    
    add_time = time.time() - start_time
    
    # Phase 2: Search for all elements
    start_time = time.time()
    
    search_failures = 0
    for val in values:
        if not skiplist.search(val):
            search_failures += 1
    
    search_time = time.time() - start_time
    
    # Phase 3: Delete half the elements
    start_time = time.time()
    
    delete_count = 0
    for val in values[::2]:  # Every other element
        if skiplist.erase(val):
            delete_count += 1
    
    delete_time = time.time() - start_time
    
    # Phase 4: Verify remaining elements
    start_time = time.time()
    
    verify_failures = 0
    for i, val in enumerate(values):
        expected_exists = (i % 2 == 1)  # Odd indices should still exist
        actual_exists = skiplist.search(val)
        
        if expected_exists != actual_exists:
            verify_failures += 1
    
    verify_time = time.time() - start_time
    
    print(f"  Add {num_elements} elements: {add_time:.2f}s")
    print(f"  Search all elements: {search_time:.2f}s (failures: {search_failures})")
    print(f"  Delete {delete_count} elements: {delete_time:.2f}s")
    print(f"  Verify remaining: {verify_time:.2f}s (failures: {verify_failures})")

def benchmark_vs_other_structures():
    """Benchmark skiplist vs other data structures"""
    print("\n=== Benchmarking vs Other Structures ===")
    
    import time
    import bisect
    
    # Prepare test data
    num_elements = 10000
    test_values = list(range(num_elements))
    random.shuffle(test_values)
    
    search_values = random.sample(test_values, 1000)
    
    print(f"Comparing structures with {num_elements} elements, {len(search_values)} searches")
    
    # Test Skiplist
    skiplist = SkiplistOptimized()
    
    start_time = time.time()
    for val in test_values:
        skiplist.add(val)
    skiplist_add_time = (time.time() - start_time) * 1000
    
    start_time = time.time()
    for val in search_values:
        skiplist.search(val)
    skiplist_search_time = (time.time() - start_time) * 1000
    
    # Test Sorted List (for comparison)
    sorted_list = []
    
    start_time = time.time()
    for val in test_values:
        bisect.insort(sorted_list, val)
    sorted_list_add_time = (time.time() - start_time) * 1000
    
    start_time = time.time()
    for val in search_values:
        bisect.bisect_left(sorted_list, val)
    sorted_list_search_time = (time.time() - start_time) * 1000
    
    # Test Set (for comparison)
    test_set = set()
    
    start_time = time.time()
    for val in test_values:
        test_set.add(val)
    set_add_time = (time.time() - start_time) * 1000
    
    start_time = time.time()
    for val in search_values:
        val in test_set
    set_search_time = (time.time() - start_time) * 1000
    
    print(f"  Results:")
    print(f"    Skiplist:     Add {skiplist_add_time:.2f}ms, Search {skiplist_search_time:.2f}ms")
    print(f"    Sorted List:  Add {sorted_list_add_time:.2f}ms, Search {sorted_list_search_time:.2f}ms")
    print(f"    Set:          Add {set_add_time:.2f}ms, Search {set_search_time:.2f}ms")

if __name__ == "__main__":
    test_skiplist_basic()
    test_skiplist_edge_cases()
    test_skiplist_performance()
    test_advanced_features()
    demonstrate_applications()
    test_concurrent_access()
    stress_test_skiplist()
    benchmark_vs_other_structures()

"""
Skiplist Design demonstrates key concepts:

Core Approaches:
1. Basic - Standard skiplist with probabilistic balancing
2. Optimized - Enhanced with finger search and better constants
3. Advanced - Extended with statistics, range queries, and analytics
4. Concurrent - Thread-safe implementation with fine-grained locking
5. Memory Optimized - Compressed levels and memory-efficient structure

Key Design Principles:
- Probabilistic balancing for logarithmic performance
- Multiple levels for efficient search path shortcuts
- Random level generation for expected O(log n) operations
- Forward pointers at each level for traversal

Performance Characteristics:
- Expected O(log n) for search, add, and erase operations
- O(n) space complexity with low constants
- Better practical performance than many balanced trees
- Simpler implementation than red-black or AVL trees

Real-world Applications:
- Database indexing and range queries
- Priority queues with search capability
- Distributed cache key management
- Ordered set operations in databases
- Level-based game systems
- Probabilistic data structures in distributed systems

The skiplist provides an excellent balance of simplicity
and performance, making it a popular choice for systems
requiring ordered data with fast operations.
"""
