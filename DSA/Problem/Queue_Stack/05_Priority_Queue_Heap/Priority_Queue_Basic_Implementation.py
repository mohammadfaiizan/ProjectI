"""
Priority Queue Basic Implementation - Multiple Approaches
Difficulty: Easy

Implement a priority queue (heap) data structure with the following operations:
- insert(val, priority): Insert element with given priority
- extract_max(): Remove and return element with highest priority
- peek_max(): Return element with highest priority without removing
- is_empty(): Check if priority queue is empty
- size(): Get number of elements
"""

from typing import Any, Optional, List, Tuple
import heapq

class PriorityQueueMaxHeap:
    """
    Approach 1: Max Heap Implementation using Array
    
    Implement max heap using array with heap property.
    
    Time: O(log n) for insert/extract, O(1) for peek, Space: O(n)
    """
    
    def __init__(self):
        self.heap = []
    
    def insert(self, val: Any, priority: int) -> None:
        """Insert element with priority"""
        # Store as (priority, val) for comparison
        self.heap.append((priority, val))
        self._heapify_up(len(self.heap) - 1)
    
    def extract_max(self) -> Optional[Tuple[Any, int]]:
        """Remove and return element with highest priority"""
        if not self.heap:
            return None
        
        if len(self.heap) == 1:
            priority, val = self.heap.pop()
            return val, priority
        
        # Replace root with last element
        max_item = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        
        return max_item[1], max_item[0]  # Return (val, priority)
    
    def peek_max(self) -> Optional[Tuple[Any, int]]:
        """Return element with highest priority without removing"""
        if not self.heap:
            return None
        return self.heap[0][1], self.heap[0][0]  # Return (val, priority)
    
    def is_empty(self) -> bool:
        """Check if priority queue is empty"""
        return len(self.heap) == 0
    
    def size(self) -> int:
        """Get number of elements"""
        return len(self.heap)
    
    def _heapify_up(self, idx: int) -> None:
        """Maintain heap property upward"""
        parent = (idx - 1) // 2
        
        if idx > 0 and self.heap[idx][0] > self.heap[parent][0]:
            self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
            self._heapify_up(parent)
    
    def _heapify_down(self, idx: int) -> None:
        """Maintain heap property downward"""
        largest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2
        
        if left < len(self.heap) and self.heap[left][0] > self.heap[largest][0]:
            largest = left
        
        if right < len(self.heap) and self.heap[right][0] > self.heap[largest][0]:
            largest = right
        
        if largest != idx:
            self.heap[idx], self.heap[largest] = self.heap[largest], self.heap[idx]
            self._heapify_down(largest)


class PriorityQueueBuiltinHeap:
    """
    Approach 2: Using Python's heapq (Min Heap)
    
    Use built-in heapq with negated priorities for max heap.
    
    Time: O(log n) for insert/extract, O(1) for peek, Space: O(n)
    """
    
    def __init__(self):
        self.heap = []
        self.counter = 0  # For tie-breaking
    
    def insert(self, val: Any, priority: int) -> None:
        """Insert element with priority"""
        # Negate priority for max heap behavior
        # Use counter for stable ordering
        heapq.heappush(self.heap, (-priority, self.counter, val))
        self.counter += 1
    
    def extract_max(self) -> Optional[Tuple[Any, int]]:
        """Remove and return element with highest priority"""
        if not self.heap:
            return None
        
        neg_priority, _, val = heapq.heappop(self.heap)
        return val, -neg_priority
    
    def peek_max(self) -> Optional[Tuple[Any, int]]:
        """Return element with highest priority without removing"""
        if not self.heap:
            return None
        
        neg_priority, _, val = self.heap[0]
        return val, -neg_priority
    
    def is_empty(self) -> bool:
        """Check if priority queue is empty"""
        return len(self.heap) == 0
    
    def size(self) -> int:
        """Get number of elements"""
        return len(self.heap)


class PriorityQueueSortedList:
    """
    Approach 3: Sorted List Implementation
    
    Maintain elements in sorted order by priority.
    
    Time: O(n) for insert, O(1) for extract/peek, Space: O(n)
    """
    
    def __init__(self):
        self.items = []  # List of (priority, val) tuples
    
    def insert(self, val: Any, priority: int) -> None:
        """Insert element with priority"""
        # Binary search for insertion point
        left, right = 0, len(self.items)
        
        while left < right:
            mid = (left + right) // 2
            if self.items[mid][0] < priority:  # Lower priority
                left = mid + 1
            else:
                right = mid
        
        self.items.insert(left, (priority, val))
    
    def extract_max(self) -> Optional[Tuple[Any, int]]:
        """Remove and return element with highest priority"""
        if not self.items:
            return None
        
        priority, val = self.items.pop()  # Remove last (highest priority)
        return val, priority
    
    def peek_max(self) -> Optional[Tuple[Any, int]]:
        """Return element with highest priority without removing"""
        if not self.items:
            return None
        
        priority, val = self.items[-1]
        return val, priority
    
    def is_empty(self) -> bool:
        """Check if priority queue is empty"""
        return len(self.items) == 0
    
    def size(self) -> int:
        """Get number of elements"""
        return len(self.items)


class PriorityQueueUnsortedList:
    """
    Approach 4: Unsorted List Implementation
    
    Store elements unsorted, find max when needed.
    
    Time: O(1) for insert, O(n) for extract/peek, Space: O(n)
    """
    
    def __init__(self):
        self.items = []  # List of (priority, val) tuples
    
    def insert(self, val: Any, priority: int) -> None:
        """Insert element with priority"""
        self.items.append((priority, val))
    
    def extract_max(self) -> Optional[Tuple[Any, int]]:
        """Remove and return element with highest priority"""
        if not self.items:
            return None
        
        # Find max priority
        max_idx = 0
        for i in range(1, len(self.items)):
            if self.items[i][0] > self.items[max_idx][0]:
                max_idx = i
        
        priority, val = self.items.pop(max_idx)
        return val, priority
    
    def peek_max(self) -> Optional[Tuple[Any, int]]:
        """Return element with highest priority without removing"""
        if not self.items:
            return None
        
        # Find max priority
        max_priority = max(self.items, key=lambda x: x[0])
        return max_priority[1], max_priority[0]
    
    def is_empty(self) -> bool:
        """Check if priority queue is empty"""
        return len(self.items) == 0
    
    def size(self) -> int:
        """Get number of elements"""
        return len(self.items)


class PriorityQueueBST:
    """
    Approach 5: Binary Search Tree Implementation
    
    Use BST to maintain elements sorted by priority.
    
    Time: O(log n) average for all operations, Space: O(n)
    """
    
    class TreeNode:
        def __init__(self, priority: int, val: Any):
            self.priority = priority
            self.val = val
            self.left = None
            self.right = None
            self.count = 1  # Count of nodes with this priority
    
    def __init__(self):
        self.root = None
        self.total_size = 0
    
    def insert(self, val: Any, priority: int) -> None:
        """Insert element with priority"""
        self.root = self._insert_node(self.root, priority, val)
        self.total_size += 1
    
    def extract_max(self) -> Optional[Tuple[Any, int]]:
        """Remove and return element with highest priority"""
        if not self.root:
            return None
        
        max_node = self._find_max(self.root)
        result = (max_node.val, max_node.priority)
        
        self.root = self._delete_max(self.root)
        self.total_size -= 1
        
        return result
    
    def peek_max(self) -> Optional[Tuple[Any, int]]:
        """Return element with highest priority without removing"""
        if not self.root:
            return None
        
        max_node = self._find_max(self.root)
        return max_node.val, max_node.priority
    
    def is_empty(self) -> bool:
        """Check if priority queue is empty"""
        return self.root is None
    
    def size(self) -> int:
        """Get number of elements"""
        return self.total_size
    
    def _insert_node(self, node, priority: int, val: Any):
        """Insert node into BST"""
        if not node:
            return self.TreeNode(priority, val)
        
        if priority > node.priority:
            node.right = self._insert_node(node.right, priority, val)
        elif priority < node.priority:
            node.left = self._insert_node(node.left, priority, val)
        else:
            # Same priority, just update count (simplified)
            node.count += 1
        
        return node
    
    def _find_max(self, node):
        """Find node with maximum priority"""
        while node.right:
            node = node.right
        return node
    
    def _delete_max(self, node):
        """Delete node with maximum priority"""
        if not node.right:
            return node.left
        
        node.right = self._delete_max(node.right)
        return node


def test_priority_queue_implementations():
    """Test all priority queue implementations"""
    
    implementations = [
        ("Max Heap", PriorityQueueMaxHeap),
        ("Builtin Heap", PriorityQueueBuiltinHeap),
        ("Sorted List", PriorityQueueSortedList),
        ("Unsorted List", PriorityQueueUnsortedList),
        ("BST", PriorityQueueBST),
    ]
    
    test_operations = [
        ("insert", "task1", 3),
        ("insert", "task2", 1),
        ("insert", "task3", 5),
        ("peek_max", None, None),
        ("extract_max", None, None),
        ("insert", "task4", 2),
        ("extract_max", None, None),
        ("extract_max", None, None),
        ("is_empty", None, None),
        ("extract_max", None, None),
        ("is_empty", None, None),
    ]
    
    print("=== Testing Priority Queue Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- {impl_name} ---")
        
        try:
            pq = impl_class()
            
            for op, val, priority in test_operations:
                if op == "insert":
                    pq.insert(val, priority)
                    print(f"  Insert({val}, {priority}) -> Size: {pq.size()}")
                
                elif op == "extract_max":
                    result = pq.extract_max()
                    print(f"  ExtractMax() -> {result}, Size: {pq.size()}")
                
                elif op == "peek_max":
                    result = pq.peek_max()
                    print(f"  PeekMax() -> {result}")
                
                elif op == "is_empty":
                    result = pq.is_empty()
                    print(f"  IsEmpty() -> {result}")
            
            print("  ✓ All operations completed")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")


def demonstrate_heap_operations():
    """Demonstrate heap operations step by step"""
    print("\n=== Heap Operations Step-by-Step Demo ===")
    
    pq = PriorityQueueMaxHeap()
    
    operations = [
        ("insert", "A", 10),
        ("insert", "B", 5),
        ("insert", "C", 15),
        ("insert", "D", 8),
        ("peek_max", None, None),
        ("extract_max", None, None),
        ("extract_max", None, None),
    ]
    
    for op, val, priority in operations:
        print(f"\nOperation: {op}({val}, {priority})" if val else f"\nOperation: {op}()")
        
        if op == "insert":
            pq.insert(val, priority)
            print(f"  Inserted ({val}, {priority})")
            print(f"  Heap: {[(p, v) for p, v in pq.heap]}")
        
        elif op == "extract_max":
            result = pq.extract_max()
            print(f"  Extracted: {result}")
            print(f"  Heap: {[(p, v) for p, v in pq.heap]}")
        
        elif op == "peek_max":
            result = pq.peek_max()
            print(f"  Max element: {result}")
        
        print(f"  Size: {pq.size()}")


def visualize_heap_structure():
    """Visualize heap structure"""
    print("\n=== Heap Structure Visualization ===")
    
    pq = PriorityQueueMaxHeap()
    
    # Insert elements
    elements = [("task1", 10), ("task2", 5), ("task3", 15), ("task4", 8), ("task5", 12)]
    
    for val, priority in elements:
        pq.insert(val, priority)
    
    print("Heap array representation:")
    print(f"  {[(p, v) for p, v in pq.heap]}")
    
    print("\nHeap tree structure:")
    print("       (15,task3)")
    print("      /          \\")
    print("  (12,task5)   (10,task1)")
    print("   /      \\")
    print("(8,task4) (5,task2)")
    
    print("\nHeap properties:")
    print("1. Parent >= Children (max heap)")
    print("2. Complete binary tree")
    print("3. Array indices: parent = (i-1)//2, left = 2*i+1, right = 2*i+2")


def benchmark_priority_queue_implementations():
    """Benchmark different implementations"""
    import time
    import random
    
    implementations = [
        ("Max Heap", PriorityQueueMaxHeap),
        ("Builtin Heap", PriorityQueueBuiltinHeap),
        ("Sorted List", PriorityQueueSortedList),
        ("Unsorted List", PriorityQueueUnsortedList),
    ]
    
    n_operations = 1000
    
    print(f"\n=== Priority Queue Performance Benchmark ===")
    print(f"Operations: {n_operations} inserts + {n_operations//2} extracts")
    
    for impl_name, impl_class in implementations:
        try:
            pq = impl_class()
            
            start_time = time.time()
            
            # Insert operations
            for i in range(n_operations):
                priority = random.randint(1, 100)
                pq.insert(f"task{i}", priority)
            
            # Extract operations
            for _ in range(n_operations // 2):
                pq.extract_max()
            
            end_time = time.time()
            
            print(f"{impl_name:15} | Time: {end_time - start_time:.4f}s")
            
        except Exception as e:
            print(f"{impl_name:15} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    pq = PriorityQueueMaxHeap()
    
    edge_cases = [
        ("Empty extract", lambda: pq.extract_max()),
        ("Empty peek", lambda: pq.peek_max()),
        ("Empty is_empty", lambda: pq.is_empty()),
        ("Empty size", lambda: pq.size()),
    ]
    
    for description, operation in edge_cases:
        try:
            result = operation()
            print(f"{description:20} | Result: {result}")
        except Exception as e:
            print(f"{description:20} | ERROR: {str(e)[:30]}")
    
    # Test with elements
    pq.insert("test", 5)
    
    print(f"\nAfter inserting one element:")
    print(f"Size: {pq.size()}")
    print(f"Peek: {pq.peek_max()}")
    print(f"Extract: {pq.extract_max()}")
    print(f"Empty after extract: {pq.is_empty()}")


def compare_time_complexities():
    """Compare time complexities of different approaches"""
    print("\n=== Time Complexity Comparison ===")
    
    approaches = [
        ("Max Heap", "O(log n)", "O(log n)", "O(1)", "Optimal for most cases"),
        ("Builtin Heap", "O(log n)", "O(log n)", "O(1)", "Python heapq optimized"),
        ("Sorted List", "O(n)", "O(1)", "O(1)", "Good for extract-heavy workloads"),
        ("Unsorted List", "O(1)", "O(n)", "O(n)", "Good for insert-heavy workloads"),
        ("BST", "O(log n)", "O(log n)", "O(log n)", "Average case, worst O(n)"),
    ]
    
    print(f"{'Approach':<15} | {'Insert':<10} | {'Extract':<10} | {'Peek':<8} | {'Notes'}")
    print("-" * 75)
    
    for approach, insert_time, extract_time, peek_time, notes in approaches:
        print(f"{approach:<15} | {insert_time:<10} | {extract_time:<10} | {peek_time:<8} | {notes}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Task scheduling
    print("1. Task Scheduling System:")
    scheduler = PriorityQueueMaxHeap()
    
    tasks = [
        ("Send email", 2),
        ("Backup database", 5),
        ("Process payments", 9),
        ("Generate reports", 3),
        ("System maintenance", 7),
    ]
    
    print("  Adding tasks to scheduler:")
    for task, priority in tasks:
        scheduler.insert(task, priority)
        print(f"    Task: {task} (Priority: {priority})")
    
    print("\n  Executing tasks by priority:")
    while not scheduler.is_empty():
        task, priority = scheduler.extract_max()
        print(f"    Executing: {task} (Priority: {priority})")
    
    # Application 2: Emergency room triage
    print("\n2. Emergency Room Triage System:")
    triage = PriorityQueueMaxHeap()
    
    patients = [
        ("John Doe", 3),      # Moderate
        ("Jane Smith", 8),    # Critical
        ("Bob Johnson", 1),   # Minor
        ("Alice Brown", 9),   # Life-threatening
        ("Charlie Wilson", 5), # Urgent
    ]
    
    print("  Patients arriving:")
    for patient, severity in patients:
        triage.insert(patient, severity)
        print(f"    {patient} - Severity: {severity}")
    
    print("\n  Treatment order:")
    while not triage.is_empty():
        patient, severity = triage.extract_max()
        print(f"    Treating: {patient} (Severity: {severity})")


def demonstrate_heap_properties():
    """Demonstrate heap properties"""
    print("\n=== Heap Properties Demonstration ===")
    
    print("Max Heap Properties:")
    print("1. Parent >= Children")
    print("2. Complete binary tree (filled left to right)")
    print("3. Root contains maximum element")
    print("4. Height = O(log n)")
    
    pq = PriorityQueueMaxHeap()
    values = [(10, "A"), (5, "B"), (15, "C"), (3, "D"), (8, "E")]
    
    print(f"\nInserting elements: {values}")
    
    for priority, val in values:
        pq.insert(val, priority)
        print(f"After inserting ({priority}, {val}): {[(p, v) for p, v in pq.heap]}")
        
        # Verify heap property
        valid = True
        for i in range(len(pq.heap)):
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < len(pq.heap) and pq.heap[i][0] < pq.heap[left][0]:
                valid = False
            if right < len(pq.heap) and pq.heap[i][0] < pq.heap[right][0]:
                valid = False
        
        print(f"  Heap property valid: {'✓' if valid else '✗'}")


if __name__ == "__main__":
    test_priority_queue_implementations()
    demonstrate_heap_operations()
    visualize_heap_structure()
    demonstrate_heap_properties()
    demonstrate_real_world_applications()
    test_edge_cases()
    compare_time_complexities()
    benchmark_priority_queue_implementations()

"""
Priority Queue Basic Implementation demonstrates multiple approaches
for implementing priority queues, including heap-based, list-based,
and tree-based implementations with comprehensive performance analysis.
"""
