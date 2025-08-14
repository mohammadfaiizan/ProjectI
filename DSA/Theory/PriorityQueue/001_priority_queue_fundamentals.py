"""
Priority Queue Fundamentals - Core Concepts and Implementation
=============================================================

Topics: Priority queue definition, operations, implementations, applications
Companies: All tech companies test priority queue fundamentals
Difficulty: Easy to Medium
Time Complexity: O(log n) for insert/delete, O(1) for peek
Space Complexity: O(n) where n is number of elements
"""

from typing import List, Optional, Any, Generic, TypeVar, Callable, Tuple
import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import time

T = TypeVar('T')

class PriorityQueueFundamentals:
    
    def __init__(self):
        """Initialize with demonstration tracking"""
        self.operation_count = 0
        self.demo_queue = []
    
    # ==========================================
    # 1. WHAT IS A PRIORITY QUEUE?
    # ==========================================
    
    def explain_priority_queue_concept(self) -> None:
        """
        Explain the fundamental concept of a priority queue data structure
        
        Priority Queue is an Abstract Data Type where elements have priorities
        """
        print("=== WHAT IS A PRIORITY QUEUE? ===")
        print("A Priority Queue is an Abstract Data Type (ADT) with these properties:")
        print()
        print("KEY CHARACTERISTICS:")
        print("• Each element has an associated priority")
        print("• Higher priority elements are served before lower priority")
        print("• Elements with same priority follow FIFO order")
        print("• Abstract interface - can be implemented in multiple ways")
        print()
        print("PRIORITY QUEUE vs REGULAR QUEUE:")
        print("• Regular Queue: FIFO (First In, First Out)")
        print("• Priority Queue: Priority-based ordering")
        print("• Regular Queue: All elements have equal importance")
        print("• Priority Queue: Elements have different importance levels")
        print()
        print("PRIORITY TYPES:")
        print("• Max Priority Queue: Higher values = higher priority")
        print("• Min Priority Queue: Lower values = higher priority")
        print("• Custom Priority: User-defined comparison function")
        print()
        print("ESSENTIAL OPERATIONS:")
        print("• insert(element, priority): Add element with priority")
        print("• extract_max/min(): Remove highest/lowest priority element")
        print("• peek(): View highest priority element without removing")
        print("• is_empty(): Check if queue is empty")
        print("• size(): Get number of elements")
        print()
        print("Real-world Analogies:")
        print("• Hospital triage: Critical patients treated first")
        print("• Airport boarding: First class boards before economy")
        print("• CPU scheduling: High priority processes get CPU time first")
        print("• Print queue: Urgent documents print before regular ones")
        print("• Traffic lights: Emergency vehicles get priority")
    
    def priority_queue_vs_other_structures(self) -> None:
        """Compare priority queue with other data structures"""
        print("=== PRIORITY QUEUE VS OTHER DATA STRUCTURES ===")
        print()
        print("Priority Queue vs Heap:")
        print("  Priority Queue: Abstract Data Type (interface)")
        print("  Heap: Concrete implementation of priority queue")
        print("  Relationship: Heap is the most efficient way to implement priority queue")
        print()
        print("Priority Queue vs Sorted Array:")
        print("  Sorted Array: O(n) insertion, O(1) extraction")
        print("  Priority Queue (heap): O(log n) insertion, O(log n) extraction")
        print("  Use Priority Queue when: Frequent insertions and extractions")
        print()
        print("Priority Queue vs BST:")
        print("  BST: Can find any element in O(log n)")
        print("  Priority Queue: Only access to highest priority element")
        print("  Use Priority Queue when: Only need min/max operations")
        print()
        print("Priority Queue vs Regular Queue:")
        print("  Regular Queue: FIFO ordering")
        print("  Priority Queue: Priority-based ordering")
        print("  Use Priority Queue when: Elements have different importance")
        print()
        print("When to Use Priority Queue:")
        print("• Task scheduling with priorities")
        print("• Graph algorithms (Dijkstra, Prim)")
        print("• Event simulation")
        print("• Huffman coding")
        print("• A* pathfinding algorithm")
        print("• Operating system process scheduling")
    
    # ==========================================
    # 2. PRIORITY QUEUE OPERATIONS
    # ==========================================
    
    def demonstrate_priority_queue_operations(self) -> None:
        """
        Demonstrate all basic priority queue operations with detailed explanation
        """
        print("=== BASIC PRIORITY QUEUE OPERATIONS ===")
        
        # Using Python's heapq for demonstration (min priority queue)
        pq = []
        
        print("Creating a Min Priority Queue using Python's heapq")
        print("Lower number = higher priority")
        print()
        
        # 1. INSERT Operations
        print("1. INSERT Operations (with priorities):")
        elements_with_priority = [
            (3, "Medium priority task"),
            (1, "High priority task"),
            (5, "Low priority task"),
            (2, "Above high priority task"),
            (4, "Below medium priority task")
        ]
        
        for priority, task in elements_with_priority:
            heapq.heappush(pq, (priority, task))
            print(f"   Insert: Priority {priority} - {task}")
            print(f"   Queue state: {pq}")
            print(f"   Next to serve: Priority {pq[0][0]} - {pq[0][1]}")
            print()
        
        print("=" * 50)
        
        # 2. PEEK Operation
        print("\n2. PEEK Operation (view highest priority without removing):")
        if pq:
            priority, task = pq[0]
            print(f"   Highest priority: {priority} - {task}")
            print(f"   Queue remains unchanged: {len(pq)} elements")
        
        print("\n" + "=" * 50)
        
        # 3. EXTRACT Operations
        print("\n3. EXTRACT Operations (remove by priority):")
        extraction_count = 1
        while pq and extraction_count <= 3:
            print(f"\n--- Extraction {extraction_count} ---")
            priority, task = heapq.heappop(pq)
            print(f"   Extracted: Priority {priority} - {task}")
            print(f"   Remaining queue: {pq}")
            if pq:
                print(f"   Next highest priority: {pq[0][0]} - {pq[0][1]}")
            extraction_count += 1
        
        print(f"\nRemaining elements in queue: {len(pq)}")
    
    # ==========================================
    # 3. PRIORITY QUEUE IMPLEMENTATIONS
    # ==========================================

# Priority definitions for better type safety
class Priority(Enum):
    """Priority levels for task scheduling"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class PriorityTask:
    """Task with priority and metadata"""
    name: str
    priority: Priority
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """Comparison for heap ordering"""
        # Lower priority value = higher priority
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        # If same priority, earlier timestamp wins (FIFO)
        return self.timestamp < other.timestamp

class HeapBasedPriorityQueue(Generic[T]):
    """
    Priority Queue implementation using binary heap
    
    Most efficient implementation for priority queue operations
    Uses Python's heapq module (min heap)
    """
    
    def __init__(self, max_priority: bool = False):
        self.heap = []
        self.size = 0
        self.max_priority = max_priority
        self.operation_count = 0
    
    def insert(self, item: T, priority: int) -> None:
        """
        Insert element with priority
        
        Time: O(log n), Space: O(1)
        """
        self.operation_count += 1
        
        print(f"Insert operation #{self.operation_count}: Adding {item} with priority {priority}")
        
        # For max priority queue, negate the priority
        heap_priority = -priority if self.max_priority else priority
        
        # Store as tuple (priority, insertion_order, item) for stable sorting
        entry = (heap_priority, self.operation_count, item)
        heapq.heappush(self.heap, entry)
        self.size += 1
        
        print(f"   ✓ Added to {'max' if self.max_priority else 'min'} priority queue")
        print(f"   Queue size: {self.size}")
        self._display_top_priorities()
    
    def extract(self) -> Optional[T]:
        """
        Extract highest priority element
        
        Time: O(log n), Space: O(1)
        """
        self.operation_count += 1
        
        print(f"Extract operation #{self.operation_count}:")
        
        if self.is_empty():
            print("   ✗ Priority queue is empty!")
            return None
        
        priority, insertion_order, item = heapq.heappop(self.heap)
        self.size -= 1
        
        # Convert back from heap priority
        actual_priority = -priority if self.max_priority else priority
        
        print(f"   ✓ Extracted: {item} (priority: {actual_priority})")
        print(f"   Queue size: {self.size}")
        
        if not self.is_empty():
            self._display_top_priorities()
        
        return item
    
    def peek(self) -> Optional[Tuple[T, int]]:
        """
        Get highest priority element without removing
        
        Time: O(1), Space: O(1)
        """
        if self.is_empty():
            print("   ✗ Cannot peek: Priority queue is empty")
            return None
        
        priority, insertion_order, item = self.heap[0]
        actual_priority = -priority if self.max_priority else priority
        
        print(f"   Highest priority: {item} (priority: {actual_priority})")
        return item, actual_priority
    
    def is_empty(self) -> bool:
        """Check if priority queue is empty"""
        return self.size == 0
    
    def get_size(self) -> int:
        """Get current size of priority queue"""
        return self.size
    
    def _display_top_priorities(self) -> None:
        """Display top 3 priorities for visualization"""
        if self.heap:
            top_3 = []
            for i in range(min(3, len(self.heap))):
                priority, _, item = self.heap[i]
                actual_priority = -priority if self.max_priority else priority
                top_3.append(f"{item}({actual_priority})")
            
            print(f"   Top priorities: {', '.join(top_3)}")


class ArrayBasedPriorityQueue(Generic[T]):
    """
    Priority Queue implementation using sorted array
    
    Simple but inefficient implementation
    Good for small datasets or learning purposes
    """
    
    def __init__(self, max_priority: bool = False):
        self.array = []
        self.max_priority = max_priority
        self.operation_count = 0
    
    def insert(self, item: T, priority: int) -> None:
        """
        Insert element maintaining sorted order
        
        Time: O(n), Space: O(1)
        """
        self.operation_count += 1
        
        print(f"Array-based insert #{self.operation_count}: Adding {item} with priority {priority}")
        
        # Find correct position to insert
        entry = (priority, self.operation_count, item)
        inserted = False
        
        for i in range(len(self.array)):
            existing_priority = self.array[i][0]
            
            # For max priority queue, insert before smaller priorities
            # For min priority queue, insert before larger priorities
            if (self.max_priority and priority > existing_priority) or \
               (not self.max_priority and priority < existing_priority):
                self.array.insert(i, entry)
                inserted = True
                print(f"   Inserted at position {i}")
                break
        
        if not inserted:
            self.array.append(entry)
            print(f"   Appended at end (position {len(self.array) - 1})")
        
        print(f"   Array state: {[(p, item) for p, _, item in self.array]}")
    
    def extract(self) -> Optional[T]:
        """
        Extract highest priority element (first in sorted array)
        
        Time: O(1), Space: O(1)
        """
        self.operation_count += 1
        
        print(f"Array-based extract #{self.operation_count}:")
        
        if not self.array:
            print("   ✗ Priority queue is empty!")
            return None
        
        priority, insertion_order, item = self.array.pop(0)
        
        print(f"   ✓ Extracted: {item} (priority: {priority})")
        print(f"   Remaining: {[(p, item) for p, _, item in self.array]}")
        
        return item
    
    def peek(self) -> Optional[Tuple[T, int]]:
        """Get highest priority element without removing"""
        if not self.array:
            print("   ✗ Cannot peek: Priority queue is empty")
            return None
        
        priority, insertion_order, item = self.array[0]
        print(f"   Highest priority: {item} (priority: {priority})")
        return item, priority
    
    def is_empty(self) -> bool:
        """Check if priority queue is empty"""
        return len(self.array) == 0
    
    def get_size(self) -> int:
        """Get current size of priority queue"""
        return len(self.array)


class CustomPriorityQueue:
    """
    Priority Queue with custom comparison function
    
    Allows for complex priority calculations and custom ordering
    """
    
    def __init__(self, priority_func: Callable[[Any], float], reverse: bool = False):
        self.heap = []
        self.priority_func = priority_func
        self.reverse = reverse
        self.counter = 0  # For stable sorting
    
    def insert(self, item: Any) -> None:
        """Insert item using custom priority function"""
        self.counter += 1
        priority = self.priority_func(item)
        
        # Negate priority if reverse order (max heap behavior)
        heap_priority = -priority if self.reverse else priority
        
        entry = (heap_priority, self.counter, item)
        heapq.heappush(self.heap, entry)
        
        print(f"Custom insert: {item} with computed priority {priority}")
        print(f"   Heap entry: {entry}")
    
    def extract(self) -> Optional[Any]:
        """Extract item with highest priority according to custom function"""
        if not self.heap:
            print("   Custom priority queue is empty!")
            return None
        
        heap_priority, counter, item = heapq.heappop(self.heap)
        actual_priority = -heap_priority if self.reverse else heap_priority
        
        print(f"Custom extract: {item} (priority: {actual_priority})")
        return item


# ==========================================
# 4. PRIORITY QUEUE VARIATIONS
# ==========================================

class BoundedPriorityQueue:
    """
    Priority Queue with maximum capacity
    
    Useful for keeping only top k elements or memory-constrained environments
    """
    
    def __init__(self, max_size: int, min_priority_queue: bool = True):
        self.max_size = max_size
        self.heap = []
        self.min_priority = min_priority_queue
        self.overflow_count = 0
    
    def insert(self, item: Any, priority: int) -> bool:
        """Insert with capacity constraint"""
        print(f"Bounded insert: {item} with priority {priority} (capacity: {self.max_size})")
        
        entry = (priority, item) if self.min_priority else (-priority, item)
        
        if len(self.heap) < self.max_size:
            # Queue not full, just add
            heapq.heappush(self.heap, entry)
            print(f"   Added to queue (size: {len(self.heap)}/{self.max_size})")
            return True
        else:
            # Queue full, check if new item should replace lowest priority
            if self.min_priority:
                # In min priority queue, replace if new priority is higher (larger number)
                if priority > self.heap[0][0]:
                    replaced = heapq.heapreplace(self.heap, entry)
                    print(f"   Replaced {replaced} with {entry}")
                    return True
                else:
                    self.overflow_count += 1
                    print(f"   Rejected (priority too low). Overflow count: {self.overflow_count}")
                    return False
            else:
                # In max priority queue, replace if new priority is higher (smaller negative number)
                if -priority < self.heap[0][0]:
                    replaced = heapq.heapreplace(self.heap, entry)
                    print(f"   Replaced {replaced} with {entry}")
                    return True
                else:
                    self.overflow_count += 1
                    print(f"   Rejected (priority too low). Overflow count: {self.overflow_count}")
                    return False
    
    def get_all_elements(self) -> List[Tuple[int, Any]]:
        """Get all elements sorted by priority"""
        if self.min_priority:
            return sorted([(p, item) for p, item in self.heap])
        else:
            return sorted([(-p, item) for p, item in self.heap], reverse=True)


# ==========================================
# 5. BASIC PRIORITY QUEUE APPLICATIONS
# ==========================================

class BasicPriorityQueueApplications:
    
    def __init__(self):
        self.demo_count = 0
    
    def task_scheduling_simulation(self) -> None:
        """
        Demonstrate task scheduling using priority queue
        
        Simulates operating system task scheduler
        """
        print("=== TASK SCHEDULING SIMULATION ===")
        
        # Create max priority queue (higher number = higher priority)
        scheduler = HeapBasedPriorityQueue[PriorityTask](max_priority=True)
        
        # Add tasks with different priorities
        tasks = [
            PriorityTask("System Update", Priority.CRITICAL),
            PriorityTask("User Application", Priority.MEDIUM),
            PriorityTask("Background Sync", Priority.BACKGROUND),
            PriorityTask("Antivirus Scan", Priority.LOW),
            PriorityTask("Emergency Shutdown", Priority.CRITICAL),
            PriorityTask("File Download", Priority.MEDIUM)
        ]
        
        print("Adding tasks to scheduler:")
        for task in tasks:
            scheduler.insert(task, task.priority.value)
            print()
        
        print("\nExecuting tasks by priority:")
        execution_order = []
        while not scheduler.is_empty():
            task = scheduler.extract()
            execution_order.append(task)
            print(f"   Executing: {task.name}")
            print()
        
        print("Task execution summary:")
        for i, task in enumerate(execution_order, 1):
            print(f"   {i}. {task.name} (Priority: {task.priority.name})")
    
    def hospital_triage_system(self) -> None:
        """
        Demonstrate hospital triage using priority queue
        
        Patients are treated based on severity of condition
        """
        print("=== HOSPITAL TRIAGE SYSTEM ===")
        
        # Define triage priorities (lower number = more urgent)
        TRIAGE_LEVELS = {
            "Critical": 1,
            "Urgent": 2,
            "Less Urgent": 3,
            "Non-Urgent": 4
        }
        
        # Min priority queue (lower number = higher priority)
        triage = HeapBasedPriorityQueue[str](max_priority=False)
        
        # Patients arriving
        patients = [
            ("John Doe - Chest Pain", "Critical"),
            ("Jane Smith - Flu Symptoms", "Non-Urgent"),
            ("Bob Johnson - Broken Arm", "Urgent"),
            ("Alice Brown - Minor Cut", "Non-Urgent"),
            ("Mike Wilson - Severe Bleeding", "Critical"),
            ("Sarah Davis - Sprained Ankle", "Less Urgent")
        ]
        
        print("Patients arriving at emergency room:")
        for patient, severity in patients:
            priority = TRIAGE_LEVELS[severity]
            triage.insert(patient, priority)
            print(f"   Added: {patient} ({severity})")
            print()
        
        print("Treating patients in order of urgency:")
        treatment_order = 1
        while not triage.is_empty():
            patient = triage.extract()
            print(f"   {treatment_order}. Now treating: {patient}")
            treatment_order += 1
            print()
    
    def event_simulation(self) -> None:
        """
        Demonstrate event-driven simulation using priority queue
        
        Events are processed in chronological order (timestamp priority)
        """
        print("=== EVENT-DRIVEN SIMULATION ===")
        
        # Min priority queue using timestamps (earlier time = higher priority)
        event_queue = HeapBasedPriorityQueue[str](max_priority=False)
        
        # Events with timestamps (seconds from start)
        events = [
            (5.0, "Customer 1 arrives"),
            (2.0, "System startup"),
            (8.0, "Customer 2 arrives"),
            (3.0, "Server becomes ready"),
            (12.0, "Customer 1 service complete"),
            (7.0, "Maintenance task starts"),
            (15.0, "Customer 2 service complete"),
            (10.0, "Maintenance task complete")
        ]
        
        print("Scheduling events:")
        for timestamp, event in events:
            # Use timestamp as priority (scaled to int for heap)
            priority = int(timestamp * 100)  # Scale for integer priorities
            event_queue.insert(f"t={timestamp}s: {event}", priority)
        
        print("\nProcessing events in chronological order:")
        while not event_queue.is_empty():
            event = event_queue.extract()
            print(f"   {event}")
            print()


# ==========================================
# 6. PERFORMANCE ANALYSIS
# ==========================================

def analyze_priority_queue_performance():
    """
    Analyze performance characteristics of different priority queue implementations
    """
    print("=== PRIORITY QUEUE PERFORMANCE ANALYSIS ===")
    print()
    
    print("TIME COMPLEXITY COMPARISON:")
    print("┌─────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Implementation  │ Insert       │ Extract      │ Peek         │")
    print("├─────────────────┼──────────────┼──────────────┼──────────────┤")
    print("│ Binary Heap     │ O(log n)     │ O(log n)     │ O(1)         │")
    print("│ Sorted Array    │ O(n)         │ O(1)         │ O(1)         │")
    print("│ Unsorted Array  │ O(1)         │ O(n)         │ O(n)         │")
    print("│ BST             │ O(log n)     │ O(log n)     │ O(log n)     │")
    print("│ Linked List     │ O(n)         │ O(1)         │ O(1)         │")
    print("└─────────────────┴──────────────┴──────────────┴──────────────┘")
    print()
    
    print("SPACE COMPLEXITY:")
    print("• Binary Heap:     O(n) - compact array representation")
    print("• Sorted Array:    O(n) - array storage")
    print("• Unsorted Array:  O(n) - array storage")
    print("• BST:             O(n) - node pointers overhead")
    print("• Linked List:     O(n) - pointer overhead")
    print()
    
    print("IMPLEMENTATION RECOMMENDATIONS:")
    print("• General Purpose: Binary Heap (best overall performance)")
    print("• Many Extractions: Sorted Array (if few insertions)")
    print("• Many Insertions: Unsorted Array (if few extractions)")
    print("• Range Queries: BST (if need arbitrary element access)")
    print("• Memory Constrained: Binary Heap (most space efficient)")
    print()
    
    print("WHEN TO USE PRIORITY QUEUE:")
    print("• Task scheduling with priorities")
    print("• Graph algorithms (Dijkstra, Prim, A*)")
    print("• Event-driven simulation")
    print("• Huffman coding")
    print("• Finding k largest/smallest elements")
    print("• Merge k sorted sequences")
    print("• Load balancing")
    print("• CPU scheduling in operating systems")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_priority_queue_fundamentals():
    """Demonstrate all priority queue fundamental concepts"""
    print("=== PRIORITY QUEUE FUNDAMENTALS DEMONSTRATION ===\n")
    
    fundamentals = PriorityQueueFundamentals()
    
    # 1. Concept explanation
    fundamentals.explain_priority_queue_concept()
    print("\n" + "="*60 + "\n")
    
    # 2. Comparison with other structures
    fundamentals.priority_queue_vs_other_structures()
    print("\n" + "="*60 + "\n")
    
    # 3. Basic operations demonstration
    fundamentals.demonstrate_priority_queue_operations()
    print("\n" + "="*60 + "\n")
    
    # 4. Heap-based implementation
    print("=== HEAP-BASED PRIORITY QUEUE ===")
    
    print("1. Max Priority Queue:")
    max_pq = HeapBasedPriorityQueue[str](max_priority=True)
    
    tasks = [("Low priority task", 1), ("High priority task", 5), 
             ("Medium priority task", 3), ("Urgent task", 4)]
    
    for task, priority in tasks:
        max_pq.insert(task, priority)
        print()
    
    print("Extracting from max priority queue:")
    while not max_pq.is_empty():
        task = max_pq.extract()
        print()
    
    print("\n" + "-"*40 + "\n")
    
    print("2. Min Priority Queue:")
    min_pq = HeapBasedPriorityQueue[str](max_priority=False)
    
    for task, priority in tasks:
        min_pq.insert(task, priority)
        print()
    
    print("Extracting from min priority queue:")
    while not min_pq.is_empty():
        task = min_pq.extract()
        print()
    
    print("\n" + "="*60 + "\n")
    
    # 5. Array-based implementation comparison
    print("=== ARRAY-BASED PRIORITY QUEUE COMPARISON ===")
    array_pq = ArrayBasedPriorityQueue[str](max_priority=True)
    
    print("Array-based implementation (for comparison):")
    for task, priority in [("Task A", 2), ("Task B", 5), ("Task C", 1)]:
        array_pq.insert(task, priority)
        print()
    
    print("Extracting from array-based queue:")
    while not array_pq.is_empty():
        task = array_pq.extract()
        print()
    
    print("\n" + "="*60 + "\n")
    
    # 6. Custom priority queue
    print("=== CUSTOM PRIORITY QUEUE ===")
    
    # Custom priority based on string length
    length_priority = CustomPriorityQueue(
        priority_func=lambda s: len(s),
        reverse=True  # Longer strings have higher priority
    )
    
    words = ["hello", "a", "beautiful", "day", "programming"]
    print("Custom priority queue (longer strings = higher priority):")
    
    for word in words:
        length_priority.insert(word)
    
    print("\nExtracting by string length priority:")
    while length_priority.heap:
        word = length_priority.extract()
    
    print("\n" + "="*60 + "\n")
    
    # 7. Bounded priority queue
    print("=== BOUNDED PRIORITY QUEUE ===")
    bounded_pq = BoundedPriorityQueue(max_size=3, min_priority_queue=False)
    
    items = [(5, "Item A"), (3, "Item B"), (8, "Item C"), 
             (1, "Item D"), (9, "Item E"), (4, "Item F")]
    
    print("Bounded priority queue (max size = 3, keeping highest priorities):")
    for priority, item in items:
        bounded_pq.insert(item, priority)
        print()
    
    print("Final elements in bounded queue:")
    final_elements = bounded_pq.get_all_elements()
    for priority, item in final_elements:
        print(f"   {item}: priority {priority}")
    
    print("\n" + "="*60 + "\n")
    
    # 8. Applications
    print("=== PRIORITY QUEUE APPLICATIONS ===")
    apps = BasicPriorityQueueApplications()
    
    print("1. Task Scheduling:")
    apps.task_scheduling_simulation()
    print("\n" + "-"*40 + "\n")
    
    print("2. Hospital Triage:")
    apps.hospital_triage_system()
    print("\n" + "-"*40 + "\n")
    
    print("3. Event Simulation:")
    apps.event_simulation()
    print("\n" + "="*60 + "\n")
    
    # 9. Performance analysis
    analyze_priority_queue_performance()


if __name__ == "__main__":
    demonstrate_priority_queue_fundamentals()
    
    print("\n" + "="*60)
    print("=== PRIORITY QUEUE MASTERY GUIDE ===")
    print("="*60)
    
    print("\n🎯 WHEN TO USE PRIORITY QUEUE:")
    print("✅ Task scheduling with different priorities")
    print("✅ Graph algorithms (Dijkstra, Prim, A*)")
    print("✅ Event-driven simulation")
    print("✅ Finding k largest/smallest elements")
    print("✅ Merge k sorted sequences")
    print("✅ Huffman coding and compression")
    print("✅ Load balancing and resource allocation")
    
    print("\n📋 IMPLEMENTATION CHOICE GUIDE:")
    print("• Frequent insert + extract → Binary Heap")
    print("• Mostly extractions → Sorted Array")
    print("• Mostly insertions → Unsorted Array")
    print("• Need arbitrary access → BST")
    print("• Memory constrained → Binary Heap")
    print("• Custom priorities → Heap with custom comparator")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Use binary heap for general-purpose priority queue")
    print("• Implement stable sorting for equal priorities")
    print("• Consider bounded priority queue for memory limits")
    print("• Use custom comparison functions for complex priorities")
    print("• Cache priority calculations if expensive")
    
    print("\n🚨 COMMON PITFALLS:")
    print("• Confusing min vs max priority semantics")
    print("• Not handling equal priorities consistently")
    print("• Inefficient priority calculation functions")
    print("• Memory leaks in custom implementations")
    print("• Not considering thread safety for concurrent access")
    
    print("\n🎓 LEARNING PROGRESSION:")
    print("1. Understand abstract data type concept")
    print("2. Master heap-based implementation")
    print("3. Learn to choose appropriate implementation")
    print("4. Practice with real-world applications")
    print("5. Study advanced variations and optimizations")
    
    print("\n📚 APPLICATION DOMAINS:")
    print("• Operating Systems: Process scheduling")
    print("• Network Systems: Packet routing")
    print("• Game Development: AI decision making")
    print("• Databases: Query optimization")
    print("• Simulation Systems: Event processing")
    print("• Algorithms: Graph traversal and optimization")
