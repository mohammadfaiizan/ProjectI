"""
Heap Fundamentals - Core Concepts and Implementation
===================================================

Topics: Heap definition, operations, implementations, heap properties
Companies: All tech companies test heap fundamentals
Difficulty: Easy to Medium
Time Complexity: O(log n) for insert/delete, O(1) for peek
Space Complexity: O(n) where n is number of elements
"""

from typing import List, Optional, Any, Generic, TypeVar, Callable
import math

T = TypeVar('T')

class HeapFundamentals:
    
    def __init__(self):
        """Initialize with demonstration tracking"""
        self.operation_count = 0
        self.demo_heap = []
    
    # ==========================================
    # 1. WHAT IS A HEAP?
    # ==========================================
    
    def explain_heap_concept(self) -> None:
        """
        Explain the fundamental concept of a heap data structure
        
        A heap is a complete binary tree that satisfies the heap property
        """
        print("=== WHAT IS A HEAP? ===")
        print("A Heap is a complete binary tree with special properties:")
        print()
        print("KEY CHARACTERISTICS:")
        print("• Complete Binary Tree: All levels filled except possibly the last")
        print("• Heap Property: Parent-child relationship follows specific order")
        print("• Array Representation: Can be efficiently stored in arrays")
        print("• Efficient Operations: Insert/Delete in O(log n), Peek in O(1)")
        print()
        print("HEAP TYPES:")
        print("• Max Heap: Parent ≥ Children (root is maximum)")
        print("• Min Heap: Parent ≤ Children (root is minimum)")
        print()
        print("COMPLETE BINARY TREE PROPERTIES:")
        print("• All levels completely filled except possibly the last")
        print("• Last level filled from left to right")
        print("• Height = ⌊log₂(n)⌋ where n is number of nodes")
        print("• No gaps in the tree structure")
        print()
        print("ARRAY REPRESENTATION:")
        print("• Root at index 0")
        print("• For node at index i:")
        print("  - Parent: (i-1)//2")
        print("  - Left child: 2*i + 1")
        print("  - Right child: 2*i + 2")
        print()
        print("Real-world Analogies:")
        print("• Priority queue: Most important task at top")
        print("• Tournament bracket: Winner bubbles to top")
        print("• Corporate hierarchy: CEO at top, structured levels")
        print("• Hospital triage: Most critical patients first")
    
    def heap_vs_other_structures(self) -> None:
        """Compare heap with other data structures"""
        print("=== HEAP VS OTHER DATA STRUCTURES ===")
        print()
        print("Heap vs Binary Search Tree (BST):")
        print("  BST: Maintains order (left < root < right)")
        print("  Heap: Only maintains parent-child relationship")
        print("  BST: O(log n) search, but can become unbalanced")
        print("  Heap: O(1) min/max access, O(n) search for arbitrary element")
        print("  Use Heap when: You only need min/max operations")
        print()
        print("Heap vs Sorted Array:")
        print("  Sorted Array: O(n) insertion, O(1) min/max access")
        print("  Heap: O(log n) insertion, O(1) min/max access")
        print("  Use Heap when: Frequent insertions with min/max queries")
        print()
        print("Heap vs Priority Queue (Abstract):")
        print("  Priority Queue: Abstract data type (interface)")
        print("  Heap: Concrete implementation of priority queue")
        print("  Heap is the most efficient implementation of priority queue")
        print()
        print("When to Use Heap:")
        print("• Priority queue implementations")
        print("• Heap sort algorithm")
        print("• Finding k largest/smallest elements")
        print("• Merge k sorted arrays")
        print("• Dijkstra's shortest path algorithm")
        print("• Huffman coding")
    
    # ==========================================
    # 2. HEAP OPERATIONS
    # ==========================================
    
    def demonstrate_heap_operations(self) -> None:
        """
        Demonstrate all basic heap operations with detailed explanation
        """
        print("=== BASIC HEAP OPERATIONS DEMONSTRATION ===")
        
        # Create a max heap for demonstration
        max_heap = MaxHeap()
        
        print("Creating a Max Heap (parent ≥ children)")
        print()
        
        # 1. INSERT Operations
        print("1. INSERT Operations (with heapify up):")
        elements_to_insert = [10, 20, 15, 30, 40]
        
        for element in elements_to_insert:
            print(f"\n--- Inserting {element} ---")
            max_heap.insert(element)
        
        print("\n" + "="*50)
        
        # 2. PEEK Operation
        print("\n2. PEEK Operation (view max without removing):")
        max_heap.peek()
        
        print("\n" + "="*50)
        
        # 3. EXTRACT Operations
        print("\n3. EXTRACT_MAX Operations (with heapify down):")
        for i in range(3):
            print(f"\n--- Extract operation {i+1} ---")
            max_heap.extract_max()
        
        print("\n" + "="*50)
        
        # 4. Build heap from array
        print("\n4. BUILD_HEAP Operation (heapify entire array):")
        array = [4, 10, 3, 5, 1, 20, 15]
        print(f"Original array: {array}")
        
        build_heap = MaxHeap()
        build_heap.build_heap(array)
    
    # ==========================================
    # 3. HEAP IMPLEMENTATIONS
    # ==========================================

class MaxHeap(Generic[T]):
    """
    Max Heap implementation where parent ≥ children
    
    Root contains the maximum element
    Array-based implementation for efficiency
    """
    
    def __init__(self):
        self.heap = []
        self.size = 0
        self.operation_count = 0
    
    def insert(self, item: T) -> None:
        """
        Insert element and maintain heap property
        
        Time: O(log n), Space: O(1)
        Algorithm: Add to end, then bubble up (heapify up)
        """
        self.operation_count += 1
        
        print(f"Insert operation #{self.operation_count}: Adding {item}")
        
        # Add element to end of heap
        self.heap.append(item)
        self.size += 1
        
        print(f"   Step 1: Add {item} to end of array")
        print(f"   Heap: {self.heap}")
        print(f"   Tree structure before heapify:")
        self._print_tree()
        
        # Bubble up to maintain heap property
        self._heapify_up(self.size - 1)
        
        print(f"   Final heap: {self.heap}")
        print(f"   Tree structure after heapify:")
        self._print_tree()
    
    def extract_max(self) -> Optional[T]:
        """
        Remove and return maximum element (root)
        
        Time: O(log n), Space: O(1)
        Algorithm: Replace root with last element, then bubble down
        """
        self.operation_count += 1
        
        print(f"Extract Max operation #{self.operation_count}:")
        
        if self.size == 0:
            print("   ✗ Heap is empty!")
            return None
        
        if self.size == 1:
            max_item = self.heap.pop()
            self.size = 0
            print(f"   ✓ Extracted {max_item} (last element)")
            return max_item
        
        # Store max element
        max_item = self.heap[0]
        print(f"   Maximum element to extract: {max_item}")
        
        # Replace root with last element
        self.heap[0] = self.heap.pop()
        self.size -= 1
        
        print(f"   Step 1: Replace root with last element")
        print(f"   Heap after replacement: {self.heap}")
        print(f"   Tree structure before heapify:")
        self._print_tree()
        
        # Bubble down to maintain heap property
        self._heapify_down(0)
        
        print(f"   ✓ Extracted {max_item}")
        print(f"   Final heap: {self.heap}")
        print(f"   Tree structure after heapify:")
        self._print_tree()
        
        return max_item
    
    def peek(self) -> Optional[T]:
        """
        Get maximum element without removing it
        
        Time: O(1), Space: O(1)
        """
        if self.size == 0:
            print("   ✗ Cannot peek: Heap is empty")
            return None
        
        max_element = self.heap[0]
        print(f"   Maximum element: {max_element}")
        return max_element
    
    def build_heap(self, array: List[T]) -> None:
        """
        Build heap from arbitrary array using heapify
        
        Time: O(n), Space: O(1)
        Algorithm: Start from last parent, heapify down each node
        """
        print(f"\n--- Building heap from array: {array} ---")
        
        self.heap = array[:]
        self.size = len(array)
        
        print(f"Step 1: Copy array to heap structure")
        print(f"Initial heap: {self.heap}")
        print(f"Tree structure before heapify:")
        self._print_tree()
        
        # Start from last parent node and heapify down
        last_parent = (self.size - 2) // 2
        print(f"\nStep 2: Heapify from last parent (index {last_parent}) to root")
        
        for i in range(last_parent, -1, -1):
            print(f"\n   Heapifying subtree rooted at index {i} (value: {self.heap[i]})")
            self._heapify_down(i)
            print(f"   Heap after heapifying index {i}: {self.heap}")
        
        print(f"\nFinal heap: {self.heap}")
        print(f"Final tree structure:")
        self._print_tree()
    
    def _heapify_up(self, index: int) -> None:
        """
        Bubble element up to maintain heap property
        
        Used after insertion
        """
        if index == 0:
            return
        
        parent_index = (index - 1) // 2
        
        print(f"      Heapify up: Comparing index {index} ({self.heap[index]}) with parent {parent_index} ({self.heap[parent_index]})")
        
        if self.heap[index] > self.heap[parent_index]:
            # Swap with parent
            self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
            print(f"      Swapped: {self.heap[parent_index]} ↔ {self.heap[index]}")
            print(f"      Heap: {self.heap}")
            
            # Continue heapifying up
            self._heapify_up(parent_index)
        else:
            print(f"      No swap needed: {self.heap[index]} ≤ {self.heap[parent_index]}")
    
    def _heapify_down(self, index: int) -> None:
        """
        Bubble element down to maintain heap property
        
        Used after extraction and during build_heap
        """
        left_child = 2 * index + 1
        right_child = 2 * index + 2
        largest = index
        
        print(f"      Heapify down from index {index} (value: {self.heap[index]})")
        
        # Find largest among node and its children
        if left_child < self.size and self.heap[left_child] > self.heap[largest]:
            largest = left_child
            print(f"        Left child {left_child} ({self.heap[left_child]}) is larger")
        
        if right_child < self.size and self.heap[right_child] > self.heap[largest]:
            largest = right_child
            print(f"        Right child {right_child} ({self.heap[right_child]}) is larger")
        
        if largest != index:
            # Swap with largest child
            print(f"        Swapping index {index} ({self.heap[index]}) with index {largest} ({self.heap[largest]})")
            self.heap[index], self.heap[largest] = self.heap[largest], self.heap[index]
            print(f"        Heap after swap: {self.heap}")
            
            # Continue heapifying down
            self._heapify_down(largest)
        else:
            print(f"        No swap needed: {self.heap[index]} is in correct position")
    
    def _print_tree(self) -> None:
        """Print heap as a tree structure for visualization"""
        if self.size == 0:
            print("        (empty)")
            return
        
        # Calculate tree height
        height = math.floor(math.log2(self.size)) + 1 if self.size > 0 else 0
        
        print(f"        Tree (height: {height}):")
        
        if self.size <= 15:  # Only print for small heaps to avoid clutter
            level = 0
            index = 0
            
            while index < self.size:
                level_size = min(2**level, self.size - index)
                level_elements = []
                
                for i in range(level_size):
                    if index + i < self.size:
                        level_elements.append(str(self.heap[index + i]))
                
                # Print level with indentation
                indent = "  " * (height - level)
                print(f"        {indent}Level {level}: {' '.join(level_elements)}")
                
                index += level_size
                level += 1
        else:
            print(f"        Root: {self.heap[0]} (tree too large to display)")
    
    def is_empty(self) -> bool:
        """Check if heap is empty"""
        return self.size == 0
    
    def get_size(self) -> int:
        """Get current size of heap"""
        return self.size
    
    def display(self) -> None:
        """Display heap in array and tree format"""
        print("Max Heap Display:")
        print(f"   Array representation: {self.heap}")
        print(f"   Size: {self.size}")
        self._print_tree()


class MinHeap(Generic[T]):
    """
    Min Heap implementation where parent ≤ children
    
    Root contains the minimum element
    Similar to MaxHeap but with reversed comparisons
    """
    
    def __init__(self):
        self.heap = []
        self.size = 0
    
    def insert(self, item: T) -> None:
        """Insert element and maintain min heap property"""
        print(f"Min Heap Insert: Adding {item}")
        
        self.heap.append(item)
        self.size += 1
        
        print(f"   Added to end: {self.heap}")
        
        # Bubble up
        self._heapify_up(self.size - 1)
        
        print(f"   Final heap: {self.heap}")
    
    def extract_min(self) -> Optional[T]:
        """Remove and return minimum element"""
        print(f"Min Heap Extract Min:")
        
        if self.size == 0:
            print("   ✗ Heap is empty!")
            return None
        
        if self.size == 1:
            min_item = self.heap.pop()
            self.size = 0
            print(f"   ✓ Extracted {min_item} (last element)")
            return min_item
        
        min_item = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.size -= 1
        
        print(f"   Minimum element: {min_item}")
        print(f"   After replacement: {self.heap}")
        
        self._heapify_down(0)
        
        print(f"   ✓ Extracted {min_item}")
        print(f"   Final heap: {self.heap}")
        
        return min_item
    
    def peek(self) -> Optional[T]:
        """Get minimum element without removing it"""
        if self.size == 0:
            print("   ✗ Cannot peek: Heap is empty")
            return None
        
        min_element = self.heap[0]
        print(f"   Minimum element: {min_element}")
        return min_element
    
    def _heapify_up(self, index: int) -> None:
        """Bubble element up for min heap"""
        if index == 0:
            return
        
        parent_index = (index - 1) // 2
        
        if self.heap[index] < self.heap[parent_index]:
            self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
            print(f"      Min heap up: Swapped {self.heap[parent_index]} ↔ {self.heap[index]}")
            self._heapify_up(parent_index)
    
    def _heapify_down(self, index: int) -> None:
        """Bubble element down for min heap"""
        left_child = 2 * index + 1
        right_child = 2 * index + 2
        smallest = index
        
        if left_child < self.size and self.heap[left_child] < self.heap[smallest]:
            smallest = left_child
        
        if right_child < self.size and self.heap[right_child] < self.heap[smallest]:
            smallest = right_child
        
        if smallest != index:
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            print(f"      Min heap down: Swapped {self.heap[smallest]} ↔ {self.heap[index]}")
            self._heapify_down(smallest)


# ==========================================
# 4. HEAP PROPERTIES AND VALIDATION
# ==========================================

class HeapValidator:
    """Utility class to validate heap properties"""
    
    @staticmethod
    def is_max_heap(array: List[T]) -> bool:
        """
        Check if array represents a valid max heap
        
        Time: O(n), Space: O(1)
        """
        n = len(array)
        
        print(f"Validating max heap property for: {array}")
        
        for i in range(n):
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            
            # Check left child
            if left_child < n:
                if array[i] < array[left_child]:
                    print(f"   ✗ Violation: Parent {array[i]} < Left child {array[left_child]} at indices {i}, {left_child}")
                    return False
                else:
                    print(f"   ✓ Parent {array[i]} ≥ Left child {array[left_child]}")
            
            # Check right child
            if right_child < n:
                if array[i] < array[right_child]:
                    print(f"   ✗ Violation: Parent {array[i]} < Right child {array[right_child]} at indices {i}, {right_child}")
                    return False
                else:
                    print(f"   ✓ Parent {array[i]} ≥ Right child {array[right_child]}")
        
        print(f"   ✓ Valid max heap!")
        return True
    
    @staticmethod
    def is_min_heap(array: List[T]) -> bool:
        """Check if array represents a valid min heap"""
        n = len(array)
        
        print(f"Validating min heap property for: {array}")
        
        for i in range(n):
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            
            if left_child < n and array[i] > array[left_child]:
                print(f"   ✗ Violation: Parent {array[i]} > Left child {array[left_child]}")
                return False
            
            if right_child < n and array[i] > array[right_child]:
                print(f"   ✗ Violation: Parent {array[i]} > Right child {array[right_child]}")
                return False
        
        print(f"   ✓ Valid min heap!")
        return True
    
    @staticmethod
    def is_complete_binary_tree(array: List[T]) -> bool:
        """
        Check if array represents a complete binary tree
        
        In array representation, a complete binary tree has no gaps
        """
        n = len(array)
        
        print(f"Checking complete binary tree property for array of size {n}")
        
        # In array representation, if any element is None/missing
        # before the last element, it's not complete
        for i in range(n):
            if array[i] is None:
                print(f"   ✗ Gap found at index {i}")
                return False
        
        print(f"   ✓ Complete binary tree (no gaps in array)")
        return True


# ==========================================
# 5. HEAP APPLICATIONS
# ==========================================

class BasicHeapApplications:
    
    def __init__(self):
        self.demo_count = 0
    
    def find_k_largest_elements(self, array: List[int], k: int) -> List[int]:
        """
        Find k largest elements using min heap
        
        Company: Amazon, Google, Microsoft
        Difficulty: Medium
        Time: O(n log k), Space: O(k)
        
        Algorithm: Use min heap of size k
        """
        if k <= 0 or not array:
            return []
        
        min_heap = MinHeap()
        
        print(f"Finding {k} largest elements from: {array}")
        print("Using min heap of size k for optimal space complexity")
        print()
        
        for i, num in enumerate(array):
            print(f"Step {i+1}: Processing {num}")
            
            if min_heap.size < k:
                # Heap not full, just add
                min_heap.insert(num)
                print(f"   Heap size < k, added {num}")
            else:
                # Heap full, compare with minimum
                current_min = min_heap.peek()
                if num > current_min:
                    print(f"   {num} > current min {current_min}, replacing")
                    min_heap.extract_min()
                    min_heap.insert(num)
                else:
                    print(f"   {num} ≤ current min {current_min}, skipping")
            
            print(f"   Current k largest: {sorted(min_heap.heap, reverse=True)}")
            print()
        
        # Extract all elements (they will be in min heap order)
        result = []
        while not min_heap.is_empty():
            result.append(min_heap.extract_min())
        
        # Reverse to get largest first
        result.reverse()
        
        print(f"Final {k} largest elements: {result}")
        return result
    
    def merge_k_sorted_arrays(self, arrays: List[List[int]]) -> List[int]:
        """
        Merge k sorted arrays using min heap
        
        Company: Google, Facebook, Amazon
        Difficulty: Hard
        Time: O(n log k), Space: O(k)
        where n is total elements, k is number of arrays
        """
        import heapq
        
        if not arrays:
            return []
        
        # Min heap to store (value, array_index, element_index)
        heap = []
        result = []
        
        print(f"Merging {len(arrays)} sorted arrays:")
        for i, arr in enumerate(arrays):
            print(f"   Array {i}: {arr}")
        print()
        
        # Initialize heap with first element of each array
        for i, array in enumerate(arrays):
            if array:  # Non-empty array
                heapq.heappush(heap, (array[0], i, 0))
                print(f"Initial: Added {array[0]} from array {i}")
        
        print(f"Initial heap: {heap}")
        print()
        
        step = 1
        while heap:
            print(f"Step {step}:")
            
            # Extract minimum
            value, array_idx, element_idx = heapq.heappop(heap)
            result.append(value)
            
            print(f"   Extracted minimum: {value} from array {array_idx}")
            print(f"   Current result: {result}")
            
            # Add next element from same array if exists
            if element_idx + 1 < len(arrays[array_idx]):
                next_value = arrays[array_idx][element_idx + 1]
                heapq.heappush(heap, (next_value, array_idx, element_idx + 1))
                print(f"   Added next element: {next_value} from array {array_idx}")
            
            print(f"   Heap after step: {heap}")
            print()
            step += 1
        
        print(f"Final merged array: {result}")
        return result
    
    def heap_as_priority_queue(self) -> None:
        """
        Demonstrate heap usage as priority queue
        
        Priority queue operations using heap
        """
        print("=== HEAP AS PRIORITY QUEUE ===")
        
        # Using max heap for priority queue (higher value = higher priority)
        pq = MaxHeap()
        
        print("Priority Queue using Max Heap (higher number = higher priority)")
        print()
        
        # Add tasks with priorities
        tasks = [
            (3, "Medium priority task"),
            (5, "High priority task"),
            (1, "Low priority task"),
            (4, "Above medium task"),
            (5, "Another high priority task")
        ]
        
        print("Adding tasks to priority queue:")
        for priority, task in tasks:
            print(f"\nAdding: Priority {priority} - {task}")
            pq.insert(priority)
        
        print("\n" + "="*50)
        print("Processing tasks by priority:")
        
        task_index = 0
        while not pq.is_empty():
            priority = pq.extract_max()
            # Find corresponding task (simplified mapping)
            task_desc = next(task for p, task in tasks if p == priority)
            
            print(f"\nProcessing: Priority {priority} - {task_desc}")
            task_index += 1


# ==========================================
# 6. PERFORMANCE ANALYSIS
# ==========================================

def analyze_heap_performance():
    """
    Analyze performance characteristics of heap operations
    """
    print("=== HEAP PERFORMANCE ANALYSIS ===")
    print()
    
    print("TIME COMPLEXITY:")
    print("• Insert:           O(log n) - Heapify up")
    print("• Extract Max/Min:  O(log n) - Heapify down")
    print("• Peek:             O(1)     - Access root")
    print("• Build Heap:       O(n)     - Bottom-up heapify")
    print("• Search:           O(n)     - No ordering except parent-child")
    print("• Delete arbitrary: O(log n) - If index known")
    print()
    
    print("SPACE COMPLEXITY:")
    print("• Array-based:      O(n) where n is number of elements")
    print("• Additional space: O(1) for operations (in-place)")
    print("• Recursive calls:  O(log n) stack space for heapify")
    print()
    
    print("HEAP VS OTHER DATA STRUCTURES:")
    print("┌─────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
    print("│ Operation       │ Heap        │ BST         │ Sorted Array│ Linked List │")
    print("├─────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤")
    print("│ Insert          │ O(log n)    │ O(log n)    │ O(n)        │ O(1)        │")
    print("│ Find Min/Max    │ O(1)        │ O(log n)    │ O(1)        │ O(n)        │")
    print("│ Extract Min/Max │ O(log n)    │ O(log n)    │ O(n)        │ O(n)        │")
    print("│ Search          │ O(n)        │ O(log n)    │ O(log n)    │ O(n)        │")
    print("│ Build from array│ O(n)        │ O(n log n)  │ O(n log n)  │ O(n)        │")
    print("└─────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘")
    print()
    
    print("WHEN TO USE HEAP:")
    print("• Need frequent min/max operations")
    print("• Implementing priority queue")
    print("• Finding k largest/smallest elements")
    print("• Heap sort algorithm")
    print("• Merging sorted sequences")
    print("• Graph algorithms (Dijkstra, Prim)")
    print()
    
    print("HEAP ADVANTAGES:")
    print("• Guaranteed O(log n) insert/delete")
    print("• O(1) access to min/max")
    print("• Space efficient (array-based)")
    print("• Complete binary tree structure")
    print("• Cache-friendly memory layout")
    print()
    
    print("HEAP LIMITATIONS:")
    print("• No efficient search for arbitrary elements")
    print("• Not suitable for range queries")
    print("• Cannot maintain sorted order of all elements")
    print("• Doesn't support efficient predecessor/successor operations")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_heap_fundamentals():
    """Demonstrate all heap fundamental concepts"""
    print("=== HEAP FUNDAMENTALS DEMONSTRATION ===\n")
    
    fundamentals = HeapFundamentals()
    
    # 1. Concept explanation
    fundamentals.explain_heap_concept()
    print("\n" + "="*60 + "\n")
    
    # 2. Comparison with other structures
    fundamentals.heap_vs_other_structures()
    print("\n" + "="*60 + "\n")
    
    # 3. Basic operations demonstration
    fundamentals.demonstrate_heap_operations()
    print("\n" + "="*60 + "\n")
    
    # 4. Min heap demonstration
    print("=== MIN HEAP DEMONSTRATION ===")
    min_heap = MinHeap[int]()
    
    elements = [20, 10, 30, 5, 15]
    print(f"Inserting elements into min heap: {elements}")
    
    for element in elements:
        print(f"\n--- Inserting {element} ---")
        min_heap.insert(element)
    
    print(f"\nExtracting elements from min heap:")
    while not min_heap.is_empty():
        print(f"\n--- Extract min ---")
        min_heap.extract_min()
    
    print("\n" + "="*60 + "\n")
    
    # 5. Heap validation
    print("=== HEAP VALIDATION ===")
    validator = HeapValidator()
    
    print("1. Valid Max Heap:")
    valid_max_heap = [50, 40, 30, 20, 10, 15, 25]
    validator.is_max_heap(valid_max_heap)
    print()
    
    print("2. Invalid Max Heap:")
    invalid_max_heap = [50, 60, 30, 20, 10, 15, 25]  # 60 > 50 violation
    validator.is_max_heap(invalid_max_heap)
    print()
    
    print("3. Valid Min Heap:")
    valid_min_heap = [5, 10, 15, 20, 25, 30, 35]
    validator.is_min_heap(valid_min_heap)
    print()
    
    print("\n" + "="*60 + "\n")
    
    # 6. Basic applications
    print("=== BASIC HEAP APPLICATIONS ===")
    apps = BasicHeapApplications()
    
    print("1. Find K Largest Elements:")
    array = [3, 1, 6, 5, 2, 4, 9, 8, 7]
    k = 3
    apps.find_k_largest_elements(array, k)
    print()
    
    print("2. Merge K Sorted Arrays:")
    sorted_arrays = [
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9]
    ]
    apps.merge_k_sorted_arrays(sorted_arrays)
    print()
    
    print("3. Priority Queue using Heap:")
    apps.heap_as_priority_queue()
    print()
    
    # 7. Performance analysis
    analyze_heap_performance()


if __name__ == "__main__":
    demonstrate_heap_fundamentals()
    
    print("\n" + "="*60)
    print("=== HEAP MASTERY GUIDE ===")
    print("="*60)
    
    print("\n🎯 WHEN TO USE HEAP:")
    print("✅ Priority queue implementation")
    print("✅ Finding k largest/smallest elements")
    print("✅ Heap sort algorithm")
    print("✅ Merging k sorted arrays/lists")
    print("✅ Graph algorithms (Dijkstra, Prim)")
    print("✅ Median finding in data stream")
    print("✅ Top k frequent elements")
    
    print("\n📋 HEAP IMPLEMENTATION CHECKLIST:")
    print("1. Choose max heap vs min heap based on requirements")
    print("2. Implement proper heapify up and down operations")
    print("3. Handle edge cases (empty heap, single element)")
    print("4. Maintain complete binary tree property")
    print("5. Use 0-based indexing for array representation")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Use build_heap() for O(n) construction from array")
    print("• Prefer iterative heapify over recursive for large heaps")
    print("• Consider cache-friendly memory access patterns")
    print("• Use appropriate heap size for k-element problems")
    print("• Implement custom comparators for complex objects")
    
    print("\n🚨 COMMON PITFALLS:")
    print("• Confusing max heap vs min heap operations")
    print("• Incorrect parent/child index calculations")
    print("• Not maintaining complete binary tree property")
    print("• Using heap for problems requiring sorted order")
    print("• Inefficient search operations on heap")
    
    print("\n🎓 LEARNING PROGRESSION:")
    print("1. Master heap properties and array representation")
    print("2. Implement basic operations (insert, extract, peek)")
    print("3. Learn heapify algorithms (up, down, build_heap)")
    print("4. Practice heap-based algorithms and applications")
    print("5. Study advanced heap variations and optimizations")
    
    print("\n📚 PROBLEM CATEGORIES TO PRACTICE:")
    print("• K largest/smallest elements problems")
    print("• Priority queue and scheduling algorithms")
    print("• Merge operations (k sorted arrays, lists)")
    print("• Graph algorithms using heaps")
    print("• Streaming data and online algorithms")
    print("• Heap sort and comparison-based sorting")
