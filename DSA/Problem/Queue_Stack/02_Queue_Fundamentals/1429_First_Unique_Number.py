"""
1429. First Unique Number - Multiple Approaches
Difficulty: Medium

You have a queue of integers, you need to retrieve the first unique integer in the queue.

Implement the FirstUnique class:
- FirstUnique(int[] nums) Initializes the object with the numbers in the queue.
- int showFirstUnique() returns the value of the first unique integer of the queue, and returns -1 if there is no such integer.
- void add(int value) insert value to the queue.
"""

from typing import List, Dict, Optional
from collections import deque, OrderedDict, Counter

class FirstUnique1:
    """
    Approach 1: Queue + HashMap Implementation
    
    Use queue to maintain order and hashmap to track counts.
    
    Time: showFirstUnique O(n), add O(1), Space: O(n)
    """
    
    def __init__(self, nums: List[int]):
        self.queue = deque(nums)
        self.count = Counter(nums)
    
    def showFirstUnique(self) -> int:
        # Remove non-unique elements from front
        while self.queue and self.count[self.queue[0]] > 1:
            self.queue.popleft()
        
        return self.queue[0] if self.queue else -1
    
    def add(self, value: int) -> None:
        self.queue.append(value)
        self.count[value] += 1


class FirstUnique2:
    """
    Approach 2: OrderedDict Implementation
    
    Use OrderedDict to maintain insertion order of unique elements.
    
    Time: showFirstUnique O(1), add O(1), Space: O(n)
    """
    
    def __init__(self, nums: List[int]):
        self.unique_queue = OrderedDict()
        self.count = Counter()
        
        for num in nums:
            self.add(num)
    
    def showFirstUnique(self) -> int:
        if self.unique_queue:
            return next(iter(self.unique_queue))
        return -1
    
    def add(self, value: int) -> None:
        self.count[value] += 1
        
        if self.count[value] == 1:
            # First occurrence - add to unique queue
            self.unique_queue[value] = True
        elif self.count[value] == 2:
            # Second occurrence - remove from unique queue
            if value in self.unique_queue:
                del self.unique_queue[value]


class FirstUnique3:
    """
    Approach 3: Doubly Linked List + HashMap
    
    Use custom doubly linked list for O(1) operations.
    
    Time: showFirstUnique O(1), add O(1), Space: O(n)
    """
    
    class Node:
        def __init__(self, val: int):
            self.val = val
            self.prev = None
            self.next = None
    
    def __init__(self, nums: List[int]):
        # Create dummy head and tail
        self.head = self.Node(0)
        self.tail = self.Node(0)
        self.head.next = self.tail
        self.tail.prev = self.head
        
        self.count = {}
        self.node_map = {}  # value -> node mapping
        
        for num in nums:
            self.add(num)
    
    def _add_to_tail(self, node: 'Node') -> None:
        """Add node before tail"""
        prev_node = self.tail.prev
        prev_node.next = node
        node.prev = prev_node
        node.next = self.tail
        self.tail.prev = node
    
    def _remove_node(self, node: 'Node') -> None:
        """Remove node from list"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def showFirstUnique(self) -> int:
        if self.head.next == self.tail:
            return -1
        return self.head.next.val
    
    def add(self, value: int) -> None:
        if value not in self.count:
            # First occurrence
            self.count[value] = 1
            node = self.Node(value)
            self.node_map[value] = node
            self._add_to_tail(node)
        else:
            # Not first occurrence
            self.count[value] += 1
            if self.count[value] == 2:
                # Remove from unique list
                node = self.node_map[value]
                self._remove_node(node)
                del self.node_map[value]


class FirstUnique4:
    """
    Approach 4: Two Sets Implementation
    
    Use two sets to track unique and duplicate elements.
    
    Time: showFirstUnique O(n), add O(1), Space: O(n)
    """
    
    def __init__(self, nums: List[int]):
        self.queue = deque()
        self.unique = set()
        self.duplicates = set()
        
        for num in nums:
            self.add(num)
    
    def showFirstUnique(self) -> int:
        # Remove duplicates from front
        while self.queue and self.queue[0] in self.duplicates:
            self.queue.popleft()
        
        return self.queue[0] if self.queue else -1
    
    def add(self, value: int) -> None:
        if value not in self.unique and value not in self.duplicates:
            # First occurrence
            self.unique.add(value)
            self.queue.append(value)
        elif value in self.unique:
            # Second occurrence
            self.unique.remove(value)
            self.duplicates.add(value)


class FirstUnique5:
    """
    Approach 5: Lazy Deletion with Queue
    
    Use lazy deletion to avoid immediate cleanup.
    
    Time: showFirstUnique O(k) where k is duplicates, add O(1), Space: O(n)
    """
    
    def __init__(self, nums: List[int]):
        self.queue = deque()
        self.count = {}
        
        for num in nums:
            self.add(num)
    
    def showFirstUnique(self) -> int:
        # Lazy cleanup of non-unique elements
        while self.queue:
            front = self.queue[0]
            if self.count[front] == 1:
                return front
            else:
                self.queue.popleft()
        
        return -1
    
    def add(self, value: int) -> None:
        if value not in self.count:
            self.count[value] = 0
        
        self.count[value] += 1
        
        if self.count[value] == 1:
            self.queue.append(value)


class FirstUnique6:
    """
    Approach 6: Index-based Tracking
    
    Track indices to determine first unique element.
    
    Time: showFirstUnique O(n), add O(1), Space: O(n)
    """
    
    def __init__(self, nums: List[int]):
        self.elements = []
        self.count = {}
        self.first_index = {}  # Track first occurrence index
        
        for num in nums:
            self.add(num)
    
    def showFirstUnique(self) -> int:
        min_index = float('inf')
        result = -1
        
        for value, cnt in self.count.items():
            if cnt == 1:
                if self.first_index[value] < min_index:
                    min_index = self.first_index[value]
                    result = value
        
        return result
    
    def add(self, value: int) -> None:
        if value not in self.count:
            self.count[value] = 0
            self.first_index[value] = len(self.elements)
        
        self.count[value] += 1
        self.elements.append(value)


def test_first_unique_implementations():
    """Test all first unique implementations"""
    
    implementations = [
        ("Queue + HashMap", FirstUnique1),
        ("OrderedDict", FirstUnique2),
        ("Doubly Linked List", FirstUnique3),
        ("Two Sets", FirstUnique4),
        ("Lazy Deletion", FirstUnique5),
        ("Index Tracking", FirstUnique6),
    ]
    
    test_cases = [
        ([2, 3, 5], [(None, 2), (7, 2), (3, 7), (4, 7), (5, 4)], "Basic test"),
        ([7, 7, 7, 7, 7, 7], [(None, -1), (7, -1), (3, 3), (4, 3), (2, 3)], "All duplicates"),
        ([809], [(None, 809), (809, -1)], "Single element"),
        ([1, 2, 3, 4, 5], [(None, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, -1)], "All unique initially"),
    ]
    
    print("=== Testing First Unique Implementations ===")
    
    for initial_nums, operations, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Initial: {initial_nums}")
        print(f"Operations: {operations}")
        
        for impl_name, impl_class in implementations:
            try:
                fu = impl_class(initial_nums[:])  # Copy to avoid modification
                results = []
                
                for add_val, expected in operations:
                    if add_val is None:
                        # showFirstUnique operation
                        result = fu.showFirstUnique()
                    else:
                        # add operation followed by showFirstUnique
                        fu.add(add_val)
                        result = fu.showFirstUnique()
                    
                    results.append(result)
                
                expected_results = [expected for _, expected in operations]
                status = "✓" if results == expected_results else "✗"
                print(f"{impl_name:20} | {status} | Results: {results}")
            
            except Exception as e:
                print(f"{impl_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_first_unique_behavior():
    """Demonstrate first unique behavior step by step"""
    print("\n=== First Unique Behavior Demonstration ===")
    
    fu = FirstUnique2([2, 3, 5])  # Using OrderedDict implementation
    
    operations = [
        ("showFirstUnique", None),
        ("add", 7),
        ("showFirstUnique", None),
        ("add", 3),
        ("showFirstUnique", None),
        ("add", 4),
        ("showFirstUnique", None),
        ("add", 5),
        ("showFirstUnique", None),
    ]
    
    print("Initial queue: [2, 3, 5]")
    
    for i, (op, val) in enumerate(operations):
        print(f"\nStep {i+1}: {op}({val if val is not None else ''})")
        
        if op == "add":
            fu.add(val)
            print(f"  Added {val} to queue")
        
        result = fu.showFirstUnique()
        print(f"  First unique: {result}")
        
        # Show current state (for OrderedDict implementation)
        if hasattr(fu, 'unique_queue'):
            unique_elements = list(fu.unique_queue.keys())
            print(f"  Unique elements: {unique_elements}")


def benchmark_first_unique():
    """Benchmark different first unique implementations"""
    import time
    import random
    
    implementations = [
        ("Queue + HashMap", FirstUnique1),
        ("OrderedDict", FirstUnique2),
        ("Doubly Linked List", FirstUnique3),
        ("Two Sets", FirstUnique4),
        ("Lazy Deletion", FirstUnique5),
    ]
    
    # Generate test data
    initial_size = 1000
    n_operations = 5000
    
    initial_nums = [random.randint(1, 100) for _ in range(initial_size)]
    add_operations = [random.randint(1, 100) for _ in range(n_operations)]
    
    print("\n=== First Unique Performance Benchmark ===")
    print(f"Initial size: {initial_size}, Operations: {n_operations}")
    
    for impl_name, impl_class in implementations:
        fu = impl_class(initial_nums[:])
        
        start_time = time.time()
        
        for val in add_operations:
            fu.add(val)
            fu.showFirstUnique()  # Call both operations
        
        end_time = time.time()
        
        print(f"{impl_name:20} | Time: {end_time - start_time:.4f}s")


def test_edge_cases():
    """Test edge cases for first unique"""
    print("\n=== Testing Edge Cases ===")
    
    edge_cases = [
        ("Empty initial", [], -1),
        ("Single element", [42], 42),
        ("All same", [5, 5, 5], -1),
        ("Two different", [1, 2], 1),
        ("Large numbers", [1000000, 999999], 1000000),
    ]
    
    for description, initial_nums, expected_first in edge_cases:
        fu = FirstUnique2(initial_nums)
        result = fu.showFirstUnique()
        status = "✓" if result == expected_first else "✗"
        print(f"{description:20} | {status} | Initial: {initial_nums}, First unique: {result}")


def test_memory_efficiency():
    """Test memory efficiency of different implementations"""
    print("\n=== Memory Efficiency Test ===")
    
    import sys
    
    implementations = [
        ("Queue + HashMap", FirstUnique1),
        ("OrderedDict", FirstUnique2),
        ("Two Sets", FirstUnique4),
        ("Lazy Deletion", FirstUnique5),
    ]
    
    # Test with moderate amount of data
    initial_nums = list(range(1000))
    
    for impl_name, impl_class in implementations:
        fu = impl_class(initial_nums)
        
        # Add more elements
        for i in range(1000, 1500):
            fu.add(i)
        
        # Estimate memory usage
        memory_size = sys.getsizeof(fu)
        
        # Add size of internal data structures
        for attr_name in ['queue', 'count', 'unique_queue', 'unique', 'duplicates', 'elements']:
            if hasattr(fu, attr_name):
                attr = getattr(fu, attr_name)
                memory_size += sys.getsizeof(attr)
        
        print(f"{impl_name:20} | Memory: ~{memory_size} bytes")


def test_correctness_with_duplicates():
    """Test correctness when dealing with many duplicates"""
    print("\n=== Correctness Test with Duplicates ===")
    
    fu = FirstUnique2([])
    
    # Add elements with specific pattern
    test_sequence = [
        (1, 1),    # First unique: 1
        (2, 1),    # First unique: 1
        (1, 2),    # First unique: 2 (1 becomes duplicate)
        (3, 2),    # First unique: 2
        (2, 3),    # First unique: 3 (2 becomes duplicate)
        (4, 3),    # First unique: 3
        (3, 4),    # First unique: 4 (3 becomes duplicate)
        (5, 4),    # First unique: 4
    ]
    
    print("Testing sequence with duplicates:")
    
    for add_val, expected_first in test_sequence:
        fu.add(add_val)
        result = fu.showFirstUnique()
        status = "✓" if result == expected_first else "✗"
        print(f"add({add_val}) -> first unique: {result} {status} (expected: {expected_first})")


def stress_test():
    """Stress test with many operations"""
    print("\n=== Stress Test ===")
    
    fu = FirstUnique2([])
    
    # Add many elements
    n_elements = 10000
    unique_count = 0
    
    print(f"Adding {n_elements} elements...")
    
    for i in range(n_elements):
        fu.add(i % 100)  # Create many duplicates
        
        if i % 1000 == 0:
            first_unique = fu.showFirstUnique()
            if first_unique != -1:
                unique_count += 1
            print(f"After {i+1} additions: first unique = {first_unique}")
    
    final_first = fu.showFirstUnique()
    print(f"Final first unique: {final_first}")


def compare_implementation_behaviors():
    """Compare behaviors of different implementations"""
    print("\n=== Implementation Behavior Comparison ===")
    
    implementations = [
        ("OrderedDict", FirstUnique2),
        ("Doubly Linked List", FirstUnique3),
        ("Two Sets", FirstUnique4),
    ]
    
    test_sequence = [1, 2, 1, 3, 2, 4]
    
    print(f"Test sequence: {test_sequence}")
    
    all_results = {}
    
    for impl_name, impl_class in implementations:
        fu = impl_class([])
        results = []
        
        for val in test_sequence:
            fu.add(val)
            first_unique = fu.showFirstUnique()
            results.append(first_unique)
        
        all_results[impl_name] = results
        print(f"{impl_name:20} | Results: {results}")
    
    # Check consistency
    first_results = list(all_results.values())[0]
    all_consistent = all(results == first_results for results in all_results.values())
    
    print(f"\nAll implementations consistent: {'✓' if all_consistent else '✗'}")


if __name__ == "__main__":
    test_first_unique_implementations()
    demonstrate_first_unique_behavior()
    test_edge_cases()
    test_correctness_with_duplicates()
    compare_implementation_behaviors()
    benchmark_first_unique()
    test_memory_efficiency()
    stress_test()

"""
First Unique Number demonstrates multiple approaches for maintaining
first unique element including OrderedDict, doubly linked list,
two sets, lazy deletion, and index tracking with comprehensive testing.
"""
