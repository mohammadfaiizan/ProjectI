"""
Queue Fundamentals - Core Concepts and Implementation
====================================================

Topics: Queue definition, operations, implementations, basic applications
Companies: All tech companies test queue fundamentals
Difficulty: Easy to Medium
Time Complexity: O(1) for all basic operations
Space Complexity: O(n) where n is number of elements
"""

from typing import List, Optional, Any, Generic, TypeVar
from collections import deque
import threading

T = TypeVar('T')

class QueueFundamentals:
    
    def __init__(self):
        """Initialize with demonstration tracking"""
        self.operation_count = 0
        self.demo_queue = []
    
    # ==========================================
    # 1. WHAT IS A QUEUE?
    # ==========================================
    
    def explain_queue_concept(self) -> None:
        """
        Explain the fundamental concept of a queue data structure
        
        Queue is a FIFO (First In, First Out) data structure
        Think of it like a line of people - first person in line is served first
        """
        print("=== WHAT IS A QUEUE? ===")
        print("A Queue is a linear data structure that follows FIFO principle:")
        print("• FIFO = First In, First Out")
        print("• Elements are added at one end (rear/back) and removed from other end (front)")
        print("• Only the front element can be accessed for removal")
        print("• New elements are always added to the rear")
        print()
        print("Real-world Analogies:")
        print("• Line at a bank: First person in line is served first")
        print("• Print queue: Documents are printed in order they were sent")
        print("• Traffic queue: Cars follow first-come-first-served order")
        print("• Call center queue: Callers are answered in waiting order")
        print()
        print("Key Characteristics:")
        print("• Insertion (enqueue) happens at rear")
        print("• Deletion (dequeue) happens at front")
        print("• No random access to middle elements")
        print("• Essential for scheduling and buffering")
        print("• Used in BFS algorithms and level-order traversals")
    
    def queue_vs_other_structures(self) -> None:
        """Compare queue with other data structures"""
        print("=== QUEUE VS OTHER DATA STRUCTURES ===")
        print()
        print("Queue vs Stack:")
        print("  Stack: LIFO (Last In, First Out)")
        print("  Queue: FIFO (First In, First Out)")
        print("  Use Queue when: Order of processing matters")
        print()
        print("Queue vs Array:")
        print("  Array: Random access to any element")
        print("  Queue: Access only to front element")
        print("  Use Queue when: You need controlled sequential access")
        print()
        print("Queue vs Linked List:")
        print("  Linked List: Can insert/delete anywhere")
        print("  Queue: Insert at rear, delete at front only")
        print("  Use Queue when: You want FIFO behavior")
        print()
        print("When to Use Queue:")
        print("• Task scheduling (CPU scheduling, print jobs)")
        print("• Breadth-First Search (BFS) algorithms")
        print("• Handling requests in web servers")
        print("• Buffering data streams")
        print("• Managing shared resources")
    
    # ==========================================
    # 2. QUEUE OPERATIONS
    # ==========================================
    
    def demonstrate_basic_operations(self) -> None:
        """
        Demonstrate all basic queue operations with detailed explanation
        """
        print("=== BASIC QUEUE OPERATIONS ===")
        
        queue = deque()  # Using deque for demonstration
        print(f"Initial queue: {list(queue)}")
        print()
        
        # 1. ENQUEUE Operation (Add to rear)
        print("1. ENQUEUE Operation (Add element to rear):")
        elements_to_enqueue = [10, 20, 30, 40]
        
        for element in elements_to_enqueue:
            queue.append(element)
            print(f"   Enqueue {element}: {list(queue)} (front = {queue[0] if queue else 'None'}, rear = {queue[-1] if queue else 'None'})")
        print()
        
        # 2. DEQUEUE Operation (Remove from front)
        print("2. DEQUEUE Operation (Remove element from front):")
        for _ in range(2):
            if queue:
                dequeued = queue.popleft()
                print(f"   Dequeue {dequeued}: {list(queue)} (front = {queue[0] if queue else 'None'})")
        print()
        
        # 3. FRONT/PEEK Operation
        print("3. FRONT/PEEK Operation (View front element without removing):")
        if queue:
            front_element = queue[0]
            print(f"   Front: {front_element} (queue remains: {list(queue)})")
        print()
        
        # 4. REAR Operation
        print("4. REAR Operation (View rear element):")
        if queue:
            rear_element = queue[-1]
            print(f"   Rear: {rear_element} (queue remains: {list(queue)})")
        print()
        
        # 5. isEmpty Operation
        print("5. isEmpty Operation:")
        print(f"   Is queue empty? {len(queue) == 0}")
        print()
        
        # 6. size Operation
        print("6. size Operation:")
        print(f"   Queue size: {len(queue)}")
        print()
        
        # Empty the queue
        print("7. Emptying the queue:")
        while queue:
            dequeued = queue.popleft()
            print(f"   Dequeue {dequeued}: {list(queue)}")
        
        print(f"   Final queue: {list(queue)}")
        print(f"   Is empty now? {len(queue) == 0}")
    
    # ==========================================
    # 3. QUEUE IMPLEMENTATIONS
    # ==========================================

class ArrayBasedQueue(Generic[T]):
    """
    Array-based queue implementation with fixed capacity
    
    Uses circular array to avoid shifting elements
    
    Pros: Cache-friendly, predictable memory usage
    Cons: Fixed size, potential overflow
    """
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = 0
        self.rear = -1
        self.size = 0
        self.operation_count = 0
    
    def enqueue(self, item: T) -> bool:
        """
        Add element to rear of queue
        
        Time: O(1), Space: O(1)
        Returns: True if successful, False if queue overflow
        """
        self.operation_count += 1
        
        print(f"Enqueue operation #{self.operation_count}: Adding {item}")
        
        if self.is_full():
            print(f"   ✗ Queue Overflow! Cannot enqueue {item}")
            return False
        
        # Circular increment of rear pointer
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = item
        self.size += 1
        
        print(f"   ✓ Enqueued {item} at index {self.rear}")
        print(f"   Queue state: {self.get_elements()}")
        print(f"   Front index: {self.front}, Rear index: {self.rear}, Size: {self.size}")
        return True
    
    def dequeue(self) -> Optional[T]:
        """
        Remove element from front of queue
        
        Time: O(1), Space: O(1)
        Returns: Dequeued element or None if empty
        """
        self.operation_count += 1
        
        print(f"Dequeue operation #{self.operation_count}:")
        
        if self.is_empty():
            print(f"   ✗ Queue Underflow! Cannot dequeue from empty queue")
            return None
        
        item = self.queue[self.front]
        self.queue[self.front] = None  # Clear reference
        self.front = (self.front + 1) % self.capacity  # Circular increment
        self.size -= 1
        
        print(f"   ✓ Dequeued {item}")
        print(f"   Queue state: {self.get_elements()}")
        print(f"   Front index: {self.front}, Rear index: {self.rear}, Size: {self.size}")
        return item
    
    def peek_front(self) -> Optional[T]:
        """Get front element without removing it"""
        if self.is_empty():
            print("   ✗ Cannot peek: Queue is empty")
            return None
        
        front_element = self.queue[self.front]
        print(f"   Front element: {front_element}")
        return front_element
    
    def peek_rear(self) -> Optional[T]:
        """Get rear element without removing it"""
        if self.is_empty():
            print("   ✗ Cannot peek: Queue is empty")
            return None
        
        rear_element = self.queue[self.rear]
        print(f"   Rear element: {rear_element}")
        return rear_element
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self.size == 0
    
    def is_full(self) -> bool:
        """Check if queue is full"""
        return self.size == self.capacity
    
    def get_size(self) -> int:
        """Get current size of queue"""
        return self.size
    
    def get_elements(self) -> List[T]:
        """Get current elements in queue (for visualization)"""
        if self.is_empty():
            return []
        
        elements = []
        index = self.front
        for _ in range(self.size):
            elements.append(self.queue[index])
            index = (index + 1) % self.capacity
        return elements
    
    def display(self) -> None:
        """Display queue in visual format"""
        print("Array-based Queue visualization:")
        if self.is_empty():
            print("   (empty)")
        else:
            elements = self.get_elements()
            print(f"   Front -> {' <- '.join(map(str, elements))} <- Rear")
            print(f"   Indices: Front={self.front}, Rear={self.rear}")
            print(f"   Size: {self.size}/{self.capacity}")


class LinkedListBasedQueue(Generic[T]):
    """
    Linked list based queue implementation with dynamic size
    
    Pros: Dynamic size, no overflow issues
    Cons: Extra memory overhead for pointers
    """
    
    class Node:
        def __init__(self, data: T):
            self.data = data
            self.next = None
    
    def __init__(self):
        self.front = None
        self.rear = None
        self.queue_size = 0
        self.operation_count = 0
    
    def enqueue(self, item: T) -> None:
        """
        Add element to rear of queue
        
        Time: O(1), Space: O(1)
        """
        self.operation_count += 1
        
        print(f"Enqueue operation #{self.operation_count}: Adding {item}")
        
        # Create new node
        new_node = self.Node(item)
        
        if self.is_empty():
            # First element
            self.front = self.rear = new_node
        else:
            # Add to rear
            self.rear.next = new_node
            self.rear = new_node
        
        self.queue_size += 1
        
        print(f"   ✓ Enqueued {item}")
        print(f"   Queue state: {self.get_elements()}")
        print(f"   Size: {self.queue_size}")
    
    def dequeue(self) -> Optional[T]:
        """
        Remove element from front of queue
        
        Time: O(1), Space: O(1)
        """
        self.operation_count += 1
        
        print(f"Dequeue operation #{self.operation_count}:")
        
        if self.is_empty():
            print(f"   ✗ Queue Underflow! Cannot dequeue from empty queue")
            return None
        
        item = self.front.data
        self.front = self.front.next
        
        # If queue becomes empty, reset rear
        if self.front is None:
            self.rear = None
        
        self.queue_size -= 1
        
        print(f"   ✓ Dequeued {item}")
        print(f"   Queue state: {self.get_elements()}")
        print(f"   Size: {self.queue_size}")
        return item
    
    def peek_front(self) -> Optional[T]:
        """Get front element without removing it"""
        if self.is_empty():
            print("   ✗ Cannot peek: Queue is empty")
            return None
        
        front_element = self.front.data
        print(f"   Front element: {front_element}")
        return front_element
    
    def peek_rear(self) -> Optional[T]:
        """Get rear element without removing it"""
        if self.is_empty():
            print("   ✗ Cannot peek: Queue is empty")
            return None
        
        rear_element = self.rear.data
        print(f"   Rear element: {rear_element}")
        return rear_element
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self.front is None
    
    def size(self) -> int:
        """Get current size of queue"""
        return self.queue_size
    
    def get_elements(self) -> List[T]:
        """Get current elements in queue (for visualization)"""
        elements = []
        current = self.front
        while current:
            elements.append(current.data)
            current = current.next
        return elements
    
    def display(self) -> None:
        """Display queue in visual format"""
        print("Linked List Queue visualization:")
        if self.is_empty():
            print("   (empty)")
        else:
            elements = self.get_elements()
            print(f"   Front -> {' -> '.join(map(str, elements))} <- Rear")
            print(f"   Size: {self.queue_size}")


class DequeBasedQueue(Generic[T]):
    """
    Queue implementation using collections.deque
    
    Pros: Optimized for both ends, efficient operations
    Cons: Slight overhead, more memory than simple array
    """
    
    def __init__(self):
        self.queue = deque()
        self.operation_count = 0
    
    def enqueue(self, item: T) -> None:
        """Add element to rear of queue"""
        self.operation_count += 1
        self.queue.append(item)
        print(f"Enqueue operation #{self.operation_count}: Added {item}")
        print(f"   Queue state: {list(self.queue)}")
    
    def dequeue(self) -> Optional[T]:
        """Remove element from front of queue"""
        self.operation_count += 1
        
        if not self.queue:
            print(f"Dequeue operation #{self.operation_count}: Queue is empty")
            return None
        
        item = self.queue.popleft()
        print(f"Dequeue operation #{self.operation_count}: Removed {item}")
        print(f"   Queue state: {list(self.queue)}")
        return item
    
    def peek_front(self) -> Optional[T]:
        """Get front element without removing it"""
        if not self.queue:
            print("   ✗ Cannot peek: Queue is empty")
            return None
        
        front_element = self.queue[0]
        print(f"   Front element: {front_element}")
        return front_element
    
    def peek_rear(self) -> Optional[T]:
        """Get rear element without removing it"""
        if not self.queue:
            print("   ✗ Cannot peek: Queue is empty")
            return None
        
        rear_element = self.queue[-1]
        print(f"   Rear element: {rear_element}")
        return rear_element
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self.queue) == 0
    
    def size(self) -> int:
        """Get current size of queue"""
        return len(self.queue)


# ==========================================
# 4. QUEUE VARIATIONS
# ==========================================

class CircularQueue(Generic[T]):
    """
    Circular Queue implementation
    
    Fixed-size queue where rear wraps around to beginning
    Efficient use of space, no wasted slots
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = -1
        self.rear = -1
        self.size = 0
    
    def enqueue(self, item: T) -> bool:
        """Add element to circular queue"""
        print(f"Enqueue {item} to circular queue:")
        
        if self.is_full():
            print(f"   ✗ Circular queue is full")
            return False
        
        if self.is_empty():
            self.front = self.rear = 0
        else:
            self.rear = (self.rear + 1) % self.capacity
        
        self.queue[self.rear] = item
        self.size += 1
        
        print(f"   ✓ Added {item} at index {self.rear}")
        print(f"   Queue: {self.get_elements()}")
        print(f"   Front: {self.front}, Rear: {self.rear}")
        return True
    
    def dequeue(self) -> Optional[T]:
        """Remove element from circular queue"""
        print("Dequeue from circular queue:")
        
        if self.is_empty():
            print("   ✗ Circular queue is empty")
            return None
        
        item = self.queue[self.front]
        self.queue[self.front] = None
        
        if self.size == 1:
            # Queue becomes empty
            self.front = self.rear = -1
        else:
            self.front = (self.front + 1) % self.capacity
        
        self.size -= 1
        
        print(f"   ✓ Removed {item}")
        print(f"   Queue: {self.get_elements()}")
        print(f"   Front: {self.front}, Rear: {self.rear}")
        return item
    
    def is_empty(self) -> bool:
        return self.size == 0
    
    def is_full(self) -> bool:
        return self.size == self.capacity
    
    def get_elements(self) -> List[T]:
        """Get elements in queue order"""
        if self.is_empty():
            return []
        
        elements = []
        index = self.front
        for _ in range(self.size):
            elements.append(self.queue[index])
            index = (index + 1) % self.capacity
        return elements


class PriorityQueue:
    """
    Simple Priority Queue implementation using heaps
    
    Elements with higher priority are dequeued first
    """
    
    def __init__(self, max_heap: bool = False):
        import heapq
        self.heap = []
        self.max_heap = max_heap
        self.counter = 0  # For stable sorting
    
    def enqueue(self, item: Any, priority: int) -> None:
        """Add item with priority"""
        # For max heap, negate priority
        if self.max_heap:
            priority = -priority
        
        # Use counter for stable sorting
        heapq.heappush(self.heap, (priority, self.counter, item))
        self.counter += 1
        
        print(f"Enqueue: {item} with priority {priority if not self.max_heap else -priority}")
        print(f"   Heap: {[(p if not self.max_heap else -p, item) for p, _, item in self.heap]}")
    
    def dequeue(self) -> Optional[Any]:
        """Remove highest priority item"""
        if not self.heap:
            print("Priority queue is empty")
            return None
        
        priority, _, item = heapq.heappop(self.heap)
        if self.max_heap:
            priority = -priority
        
        print(f"Dequeue: {item} (priority {priority})")
        print(f"   Remaining: {[(p if not self.max_heap else -p, item) for p, _, item in self.heap]}")
        return item
    
    def peek(self) -> Optional[Any]:
        """Get highest priority item without removing"""
        if not self.heap:
            return None
        
        priority, _, item = self.heap[0]
        if self.max_heap:
            priority = -priority
        
        print(f"Highest priority: {item} (priority {priority})")
        return item


# ==========================================
# 5. BASIC QUEUE APPLICATIONS
# ==========================================

class BasicQueueApplications:
    
    def __init__(self):
        self.demo_count = 0
    
    def breadth_first_search_demo(self, graph: Dict[str, List[str]], start: str) -> List[str]:
        """
        Demonstrate BFS using queue
        
        Company: Every company that tests algorithms
        Difficulty: Medium
        Time: O(V + E), Space: O(V)
        """
        visited = set()
        queue = deque([start])
        bfs_order = []
        
        print(f"BFS traversal starting from '{start}':")
        print(f"Graph: {graph}")
        print()
        
        step = 1
        while queue:
            print(f"Step {step}: Queue = {list(queue)}")
            
            # Dequeue a vertex
            vertex = queue.popleft()
            
            if vertex not in visited:
                visited.add(vertex)
                bfs_order.append(vertex)
                print(f"   Visit '{vertex}' -> BFS order: {bfs_order}")
                
                # Enqueue unvisited neighbors
                neighbors = graph.get(vertex, [])
                for neighbor in neighbors:
                    if neighbor not in visited and neighbor not in queue:
                        queue.append(neighbor)
                        print(f"   Enqueue neighbor '{neighbor}'")
                
                print(f"   Updated queue: {list(queue)}")
            
            step += 1
            print()
        
        print(f"Final BFS order: {bfs_order}")
        return bfs_order
    
    def level_order_traversal_demo(self, tree_dict: Dict[str, List[str]], root: str) -> List[List[str]]:
        """
        Level-order traversal of tree using queue
        
        Time: O(n), Space: O(w) where w is maximum width
        """
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        print(f"Level-order traversal of tree starting from '{root}':")
        print(f"Tree structure: {tree_dict}")
        print()
        
        level = 1
        while queue:
            level_size = len(queue)
            current_level = []
            
            print(f"Level {level}: Processing {level_size} nodes")
            print(f"   Queue at start: {list(queue)}")
            
            for i in range(level_size):
                node = queue.popleft()
                current_level.append(node)
                print(f"   Process node '{node}'")
                
                # Add children to queue
                children = tree_dict.get(node, [])
                for child in children:
                    queue.append(child)
                    print(f"     Add child '{child}' to queue")
            
            result.append(current_level)
            print(f"   Level {level} nodes: {current_level}")
            print(f"   Queue after level: {list(queue)}")
            print()
            level += 1
        
        print(f"Level-order result: {result}")
        return result
    
    def task_scheduling_simulation(self, tasks: List[Tuple[str, int]]) -> None:
        """
        Simulate task scheduling using queue
        
        Tasks are processed in FIFO order
        Each task has a name and execution time
        """
        task_queue = deque(tasks)
        current_time = 0
        
        print("Task Scheduling Simulation (FIFO):")
        print(f"Initial tasks: {tasks}")
        print()
        
        task_num = 1
        while task_queue:
            print(f"Time {current_time}: Queue = {list(task_queue)}")
            
            # Get next task
            task_name, execution_time = task_queue.popleft()
            
            print(f"   Start task '{task_name}' (duration: {execution_time})")
            print(f"   Task {task_num} running from time {current_time} to {current_time + execution_time}")
            
            # Simulate task execution
            current_time += execution_time
            
            print(f"   Task '{task_name}' completed at time {current_time}")
            print(f"   Remaining tasks: {list(task_queue)}")
            print()
            task_num += 1
        
        print(f"All tasks completed at time {current_time}")


# ==========================================
# 6. PERFORMANCE ANALYSIS
# ==========================================

def analyze_queue_performance():
    """
    Analyze performance characteristics of different queue implementations
    """
    print("=== QUEUE PERFORMANCE ANALYSIS ===")
    print()
    
    print("TIME COMPLEXITY:")
    print("• Enqueue:   O(1) - Constant time insertion")
    print("• Dequeue:   O(1) - Constant time deletion")
    print("• Front:     O(1) - Constant time access to front")
    print("• Rear:      O(1) - Constant time access to rear")
    print("• isEmpty:   O(1) - Constant time check")
    print("• Size:      O(1) - Constant time if maintained")
    print()
    
    print("SPACE COMPLEXITY:")
    print("• Array-based:      O(n) where n is capacity")
    print("• Linked-list:      O(n) where n is number of elements")
    print("• Circular queue:   O(n) with efficient space utilization")
    print()
    
    print("IMPLEMENTATION COMPARISON:")
    print("┌─────────────────┬──────────────┬───────────────┬─────────────────┐")
    print("│ Implementation  │ Memory Usage │ Cache Perf.   │ Use Case        │")
    print("├─────────────────┼──────────────┼───────────────┼─────────────────┤")
    print("│ Array-based     │ Low          │ Excellent     │ Known max size  │")
    print("│ Linked-list     │ High         │ Good          │ Dynamic size    │")
    print("│ Circular        │ Very Low     │ Excellent     │ Fixed-size apps │")
    print("│ Deque-based     │ Medium       │ Very Good     │ General purpose │")
    print("└─────────────────┴──────────────┴───────────────┴─────────────────┘")
    print()
    
    print("WHEN TO USE EACH:")
    print("• Array-based: When size is bounded and memory is important")
    print("• Linked-list: When size varies greatly, frequent enqueue/dequeue")
    print("• Circular: When you need fixed-size buffer (streaming, caching)")
    print("• Deque-based: When you need both queue and stack operations")
    print("• Priority queue: When elements have different priorities")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_queue_fundamentals():
    """Demonstrate all queue fundamental concepts"""
    print("=== QUEUE FUNDAMENTALS DEMONSTRATION ===\n")
    
    fundamentals = QueueFundamentals()
    
    # 1. Concept explanation
    fundamentals.explain_queue_concept()
    print("\n" + "="*60 + "\n")
    
    # 2. Comparison with other structures
    fundamentals.queue_vs_other_structures()
    print("\n" + "="*60 + "\n")
    
    # 3. Basic operations demonstration
    fundamentals.demonstrate_basic_operations()
    print("\n" + "="*60 + "\n")
    
    # 4. Array-based queue implementation
    print("=== ARRAY-BASED QUEUE IMPLEMENTATION ===")
    array_queue = ArrayBasedQueue[int](capacity=5)
    
    # Test operations
    for value in [10, 20, 30, 40, 50]:
        array_queue.enqueue(value)
    
    print("\nTrying to enqueue to full queue:")
    array_queue.enqueue(60)  # Should fail
    
    print("\nPeek operations:")
    array_queue.peek_front()
    array_queue.peek_rear()
    
    print("\nDequeuing elements:")
    for _ in range(3):
        array_queue.dequeue()
    
    print("\nFinal display:")
    array_queue.display()
    
    print("\n" + "="*60 + "\n")
    
    # 5. Linked list based queue
    print("=== LINKED LIST BASED QUEUE IMPLEMENTATION ===")
    ll_queue = LinkedListBasedQueue[str]()
    
    for value in ['A', 'B', 'C', 'D']:
        ll_queue.enqueue(value)
    
    print("\nPeek operations:")
    ll_queue.peek_front()
    ll_queue.peek_rear()
    
    print("\nDequeuing elements:")
    for _ in range(2):
        ll_queue.dequeue()
    
    print("\nFinal display:")
    ll_queue.display()
    
    print("\n" + "="*60 + "\n")
    
    # 6. Circular queue demonstration
    print("=== CIRCULAR QUEUE DEMONSTRATION ===")
    circular_queue = CircularQueue[int](5)
    
    # Fill the queue
    for i in [1, 2, 3, 4, 5]:
        circular_queue.enqueue(i)
    
    print("\nDequeue 2 elements:")
    circular_queue.dequeue()
    circular_queue.dequeue()
    
    print("\nEnqueue 2 more elements (demonstrating circular nature):")
    circular_queue.enqueue(6)
    circular_queue.enqueue(7)
    
    print("\n" + "="*60 + "\n")
    
    # 7. Priority queue demonstration
    print("=== PRIORITY QUEUE DEMONSTRATION ===")
    pq = PriorityQueue(max_heap=True)  # Higher priority first
    
    tasks = [("Low priority task", 1), ("High priority task", 5), 
             ("Medium priority task", 3), ("Urgent task", 10)]
    
    for task, priority in tasks:
        pq.enqueue(task, priority)
    
    print("\nProcessing tasks by priority:")
    while pq.heap:
        pq.dequeue()
    
    print("\n" + "="*60 + "\n")
    
    # 8. Basic applications
    print("=== BASIC QUEUE APPLICATIONS ===")
    apps = BasicQueueApplications()
    
    # BFS demonstration
    print("1. Breadth-First Search:")
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    apps.breadth_first_search_demo(graph, 'A')
    print()
    
    # Level-order traversal
    print("2. Level-order Tree Traversal:")
    tree = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F', 'G'],
        'D': [],
        'E': ['H'],
        'F': [],
        'G': [],
        'H': []
    }
    apps.level_order_traversal_demo(tree, 'A')
    print()
    
    # Task scheduling
    print("3. Task Scheduling:")
    tasks = [("Task1", 3), ("Task2", 1), ("Task3", 4), ("Task4", 2)]
    apps.task_scheduling_simulation(tasks)
    print()
    
    # 9. Performance analysis
    analyze_queue_performance()


if __name__ == "__main__":
    demonstrate_queue_fundamentals()
    
    print("\n" + "="*60)
    print("=== QUEUE MASTERY GUIDE ===")
    print("="*60)
    
    print("\n🎯 WHEN TO USE QUEUE:")
    print("✅ Breadth-First Search (BFS) algorithms")
    print("✅ Level-order tree traversal")
    print("✅ Task scheduling and job processing")
    print("✅ Handling requests in web servers")
    print("✅ Buffer for data streams")
    print("✅ Print queue management")
    print("✅ Handling interrupts in operating systems")
    
    print("\n📋 QUEUE IMPLEMENTATION CHECKLIST:")
    print("1. Choose appropriate implementation based on requirements")
    print("2. Handle queue overflow and underflow conditions")
    print("3. Consider circular queue for fixed-size applications")
    print("4. Use priority queue when elements have different priorities")
    print("5. Implement proper error handling and edge cases")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Use circular array to avoid shifting elements")
    print("• Maintain front and rear pointers for O(1) operations")
    print("• Consider deque for better performance in Python")
    print("• Use linked list for unbounded queues")
    print("• Cache size information to avoid recalculation")
    
    print("\n🚨 COMMON PITFALLS:")
    print("• Not checking for empty queue before dequeue/front")
    print("• Integer overflow in array-based implementations")
    print("• Memory leaks in linked list implementations")
    print("• Not handling circular queue wraparound correctly")
    print("• Confusing queue with stack behavior (FIFO vs LIFO)")
    
    print("\n🎓 LEARNING PROGRESSION:")
    print("1. Master basic queue operations and concepts")
    print("2. Implement different queue variations")
    print("3. Practice queue-based algorithms (BFS)")
    print("4. Learn advanced applications (scheduling, buffering)")
    print("5. Study concurrent queues and thread safety")
    
    print("\n📚 PROBLEM CATEGORIES TO PRACTICE:")
    print("• Graph traversal (BFS, level-order)")
    print("• Task scheduling and resource management")
    print("• Stream processing and buffering")
    print("• Shortest path algorithms")
    print("• Multi-threaded producer-consumer problems")
    print("• Cache implementation and LRU algorithms")

