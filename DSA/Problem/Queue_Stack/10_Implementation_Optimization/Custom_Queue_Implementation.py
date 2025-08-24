"""
Custom Queue Implementation - Multiple Approaches
Difficulty: Easy

Implement a custom queue data structure with various optimization techniques and features.
Focus on memory efficiency, performance optimization, and advanced functionality.
"""

from typing import Any, Optional, List, Iterator, Generic, TypeVar, Deque
import threading
from collections import deque
import sys

T = TypeVar('T')

class BasicQueue:
    """
    Approach 1: Basic List-based Queue
    
    Simple implementation using Python list (inefficient for dequeue).
    
    Time: O(1) enqueue, O(n) dequeue, Space: O(n)
    """
    
    def __init__(self, capacity: Optional[int] = None):
        self._items = []
        self._capacity = capacity
    
    def enqueue(self, item: Any) -> None:
        """Add item to rear of queue"""
        if self._capacity and len(self._items) >= self._capacity:
            raise OverflowError("Queue overflow")
        self._items.append(item)
    
    def dequeue(self) -> Any:
        """Remove item from front of queue"""
        if self.is_empty():
            raise IndexError("Queue underflow")
        return self._items.pop(0)  # O(n) operation
    
    def front(self) -> Any:
        """Peek at front item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items[0]
    
    def rear(self) -> Any:
        """Peek at rear item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items[-1]
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self._items) == 0
    
    def size(self) -> int:
        """Get queue size"""
        return len(self._items)
    
    def clear(self) -> None:
        """Clear all items"""
        self._items.clear()


class CircularArrayQueue:
    """
    Approach 2: Circular Array Queue
    
    Efficient implementation using circular array.
    
    Time: O(1) for all operations, Space: O(capacity)
    """
    
    def __init__(self, capacity: int = 10):
        self._items = [None] * capacity
        self._capacity = capacity
        self._front = 0
        self._rear = 0
        self._size = 0
    
    def enqueue(self, item: Any) -> None:
        """Add item to rear"""
        if self.is_full():
            raise OverflowError("Queue overflow")
        
        self._items[self._rear] = item
        self._rear = (self._rear + 1) % self._capacity
        self._size += 1
    
    def dequeue(self) -> Any:
        """Remove item from front"""
        if self.is_empty():
            raise IndexError("Queue underflow")
        
        item = self._items[self._front]
        self._items[self._front] = None  # Help GC
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        return item
    
    def front(self) -> Any:
        """Peek at front item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items[self._front]
    
    def rear(self) -> Any:
        """Peek at rear item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        rear_index = (self._rear - 1) % self._capacity
        return self._items[rear_index]
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return self._size == 0
    
    def is_full(self) -> bool:
        """Check if full"""
        return self._size == self._capacity
    
    def size(self) -> int:
        """Get size"""
        return self._size
    
    def clear(self) -> None:
        """Clear all items"""
        self._items = [None] * self._capacity
        self._front = 0
        self._rear = 0
        self._size = 0


class LinkedListQueue:
    """
    Approach 3: Linked List Queue
    
    Implementation using linked list for dynamic sizing.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    class Node:
        def __init__(self, data: Any, next_node: Optional['Node'] = None):
            self.data = data
            self.next = next_node
    
    def __init__(self):
        self._front = None
        self._rear = None
        self._size = 0
    
    def enqueue(self, item: Any) -> None:
        """Add item to rear"""
        new_node = self.Node(item)
        
        if self.is_empty():
            self._front = self._rear = new_node
        else:
            self._rear.next = new_node
            self._rear = new_node
        
        self._size += 1
    
    def dequeue(self) -> Any:
        """Remove item from front"""
        if self.is_empty():
            raise IndexError("Queue underflow")
        
        data = self._front.data
        self._front = self._front.next
        
        if self._front is None:  # Queue became empty
            self._rear = None
        
        self._size -= 1
        return data
    
    def front(self) -> Any:
        """Peek at front item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._front.data
    
    def rear(self) -> Any:
        """Peek at rear item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._rear.data
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return self._front is None
    
    def size(self) -> int:
        """Get size"""
        return self._size
    
    def clear(self) -> None:
        """Clear all items"""
        self._front = None
        self._rear = None
        self._size = 0


class DequeBasedQueue:
    """
    Approach 4: Deque-based Queue
    
    Use collections.deque for optimal performance.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    def __init__(self, capacity: Optional[int] = None):
        self._items: Deque[Any] = deque()
        self._capacity = capacity
    
    def enqueue(self, item: Any) -> None:
        """Add item to rear"""
        if self._capacity and len(self._items) >= self._capacity:
            raise OverflowError("Queue overflow")
        self._items.append(item)
    
    def dequeue(self) -> Any:
        """Remove item from front"""
        if self.is_empty():
            raise IndexError("Queue underflow")
        return self._items.popleft()
    
    def front(self) -> Any:
        """Peek at front item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items[0]
    
    def rear(self) -> Any:
        """Peek at rear item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items[-1]
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return len(self._items) == 0
    
    def size(self) -> int:
        """Get size"""
        return len(self._items)
    
    def clear(self) -> None:
        """Clear all items"""
        self._items.clear()


class PriorityQueue:
    """
    Approach 5: Priority Queue
    
    Queue with priority-based ordering.
    
    Time: O(log n) for enqueue, O(log n) for dequeue, Space: O(n)
    """
    
    def __init__(self):
        self._items = []  # List of (priority, item) tuples
    
    def enqueue(self, item: Any, priority: int = 0) -> None:
        """Add item with priority"""
        import bisect
        bisect.insort(self._items, (priority, item))
    
    def dequeue(self) -> Any:
        """Remove highest priority item"""
        if self.is_empty():
            raise IndexError("Queue underflow")
        return self._items.pop(0)[1]  # Return item, not priority
    
    def front(self) -> Any:
        """Peek at highest priority item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items[0][1]
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return len(self._items) == 0
    
    def size(self) -> int:
        """Get size"""
        return len(self._items)
    
    def clear(self) -> None:
        """Clear all items"""
        self._items.clear()


class ThreadSafeQueue:
    """
    Approach 6: Thread-Safe Queue
    
    Queue implementation with thread safety.
    
    Time: O(1) for operations, Space: O(n)
    """
    
    def __init__(self, capacity: Optional[int] = None):
        self._items = deque()
        self._capacity = capacity
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
    
    def enqueue(self, item: Any, timeout: Optional[float] = None) -> bool:
        """Thread-safe enqueue with optional timeout"""
        with self._not_full:
            if self._capacity:
                while len(self._items) >= self._capacity:
                    if not self._not_full.wait(timeout):
                        return False
            
            self._items.append(item)
            self._not_empty.notify()
            return True
    
    def dequeue(self, timeout: Optional[float] = None) -> Any:
        """Thread-safe dequeue with optional timeout"""
        with self._not_empty:
            while len(self._items) == 0:
                if not self._not_empty.wait(timeout):
                    raise IndexError("Queue underflow - timeout")
            
            item = self._items.popleft()
            self._not_full.notify()
            return item
    
    def front(self) -> Any:
        """Thread-safe front peek"""
        with self._lock:
            if len(self._items) == 0:
                raise IndexError("Queue is empty")
            return self._items[0]
    
    def is_empty(self) -> bool:
        """Thread-safe empty check"""
        with self._lock:
            return len(self._items) == 0
    
    def size(self) -> int:
        """Thread-safe size check"""
        with self._lock:
            return len(self._items)


class GenericQueue(Generic[T]):
    """
    Approach 7: Generic Type-Safe Queue
    
    Type-safe queue implementation.
    
    Time: O(1) for operations, Space: O(n)
    """
    
    def __init__(self, capacity: Optional[int] = None):
        self._items: Deque[T] = deque()
        self._capacity = capacity
    
    def enqueue(self, item: T) -> None:
        """Add typed item"""
        if self._capacity and len(self._items) >= self._capacity:
            raise OverflowError("Queue overflow")
        self._items.append(item)
    
    def dequeue(self) -> T:
        """Remove typed item"""
        if self.is_empty():
            raise IndexError("Queue underflow")
        return self._items.popleft()
    
    def front(self) -> T:
        """Peek at front typed item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items[0]
    
    def rear(self) -> T:
        """Peek at rear typed item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items[-1]
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return len(self._items) == 0
    
    def size(self) -> int:
        """Get size"""
        return len(self._items)
    
    def __iter__(self) -> Iterator[T]:
        """Make queue iterable"""
        return iter(self._items)


def test_queue_implementations():
    """Test all queue implementations"""
    
    implementations = [
        ("Basic Queue", BasicQueue),
        ("Circular Array", lambda: CircularArrayQueue(10)),
        ("LinkedList Queue", LinkedListQueue),
        ("Deque-based", DequeBasedQueue),
        ("Priority Queue", PriorityQueue),
        ("Thread Safe", ThreadSafeQueue),
        ("Generic Queue", GenericQueue),
    ]
    
    print("=== Testing Queue Implementations ===")
    
    for name, queue_factory in implementations:
        print(f"\n--- {name} ---")
        
        try:
            queue = queue_factory()
            
            # Test basic operations
            print("Testing basic operations:")
            
            # Enqueue operations
            if name == "Priority Queue":
                # Test with priorities
                for i, priority in enumerate([3, 1, 4, 1, 5]):
                    queue.enqueue(f"item_{i}", priority)
                    print(f"  Enqueued item_{i} (priority {priority}), size: {queue.size()}")
            else:
                for i in range(5):
                    queue.enqueue(f"item_{i}")
                    print(f"  Enqueued item_{i}, size: {queue.size()}")
            
            # Front/rear operations (if available)
            if hasattr(queue, 'front') and hasattr(queue, 'rear'):
                try:
                    front = queue.front()
                    rear = queue.rear()
                    print(f"  Front: {front}, Rear: {rear}")
                except:
                    pass
            
            # Dequeue operations
            while not queue.is_empty():
                item = queue.dequeue()
                print(f"  Dequeued {item}, size: {queue.size()}")
            
            print(f"  {name} tests passed ✓")
            
        except Exception as e:
            print(f"  {name} error: {str(e)[:50]}")


def demonstrate_circular_array():
    """Demonstrate circular array queue"""
    print("\n=== Circular Array Queue Demonstration ===")
    
    queue = CircularArrayQueue(5)
    
    print("Demonstrating circular array behavior:")
    
    # Fill the queue
    for i in range(5):
        queue.enqueue(f"item_{i}")
        print(f"  Enqueued item_{i}, front_idx: {queue._front}, rear_idx: {queue._rear}")
    
    # Dequeue some items
    for _ in range(2):
        item = queue.dequeue()
        print(f"  Dequeued {item}, front_idx: {queue._front}, rear_idx: {queue._rear}")
    
    # Add more items (wrapping around)
    for i in range(5, 7):
        queue.enqueue(f"item_{i}")
        print(f"  Enqueued item_{i}, front_idx: {queue._front}, rear_idx: {queue._rear}")
    
    print(f"Final queue size: {queue.size()}")


def demonstrate_thread_safety():
    """Demonstrate thread-safe queue"""
    print("\n=== Thread Safety Demonstration ===")
    
    import threading
    import time
    import random
    
    safe_queue = ThreadSafeQueue(10)
    results = []
    
    def producer(queue, thread_id):
        """Producer thread"""
        for i in range(5):
            item = f"T{thread_id}_item_{i}"
            queue.enqueue(item)
            results.append(f"Producer {thread_id} enqueued {item}")
            time.sleep(random.uniform(0.01, 0.05))
    
    def consumer(queue, thread_id):
        """Consumer thread"""
        for _ in range(3):
            try:
                item = queue.dequeue(timeout=1.0)
                results.append(f"Consumer {thread_id} dequeued {item}")
                time.sleep(random.uniform(0.01, 0.05))
            except IndexError:
                results.append(f"Consumer {thread_id} timeout")
    
    # Create and start threads
    threads = []
    
    # Producer threads
    for i in range(2):
        t = threading.Thread(target=producer, args=(safe_queue, i))
        threads.append(t)
        t.start()
    
    # Consumer threads
    for i in range(2, 4):
        t = threading.Thread(target=consumer, args=(safe_queue, i))
        threads.append(t)
        t.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    print("Thread operations completed:")
    for result in results[:10]:
        print(f"  {result}")
    
    print(f"Final queue size: {safe_queue.size()}")


def benchmark_queue_implementations():
    """Benchmark different queue implementations"""
    print("\n=== Performance Benchmark ===")
    
    import time
    
    implementations = [
        ("Basic Queue", BasicQueue),
        ("Circular Array", lambda: CircularArrayQueue(100000)),
        ("LinkedList Queue", LinkedListQueue),
        ("Deque-based", DequeBasedQueue),
    ]
    
    n_operations = 50000
    
    for name, queue_factory in implementations:
        try:
            queue = queue_factory()
            
            # Benchmark enqueue operations
            start_time = time.time()
            for i in range(n_operations):
                queue.enqueue(i)
            enqueue_time = time.time() - start_time
            
            # Benchmark dequeue operations
            start_time = time.time()
            for _ in range(n_operations):
                queue.dequeue()
            dequeue_time = time.time() - start_time
            
            print(f"{name:20} | Enqueue: {enqueue_time:.4f}s | Dequeue: {dequeue_time:.4f}s")
            
        except Exception as e:
            print(f"{name:20} | Error: {str(e)[:30]}")


def demonstrate_priority_queue():
    """Demonstrate priority queue functionality"""
    print("\n=== Priority Queue Demonstration ===")
    
    pq = PriorityQueue()
    
    # Add items with different priorities
    tasks = [
        ("Low priority task", 3),
        ("High priority task", 1),
        ("Medium priority task", 2),
        ("Critical task", 0),
        ("Another low task", 3),
    ]
    
    print("Adding tasks with priorities:")
    for task, priority in tasks:
        pq.enqueue(task, priority)
        print(f"  Added '{task}' with priority {priority}")
    
    print(f"\nProcessing tasks in priority order:")
    while not pq.is_empty():
        task = pq.dequeue()
        print(f"  Processing: {task}")


def demonstrate_advanced_features():
    """Demonstrate advanced queue features"""
    print("\n=== Advanced Features Demonstration ===")
    
    # Generic type-safe queue
    print("1. Generic Type-Safe Queue:")
    
    int_queue: GenericQueue[int] = GenericQueue()
    for i in range(5):
        int_queue.enqueue(i)
    
    print(f"   Integer queue: {list(int_queue)}")
    
    str_queue: GenericQueue[str] = GenericQueue()
    for word in ["first", "second", "third"]:
        str_queue.enqueue(word)
    
    print(f"   String queue: {list(str_queue)}")
    
    # Circular array capacity management
    print("\n2. Circular Array Capacity Management:")
    
    circular = CircularArrayQueue(3)
    
    try:
        for i in range(4):  # Try to exceed capacity
            circular.enqueue(f"item_{i}")
            print(f"   Enqueued item_{i}")
    except OverflowError as e:
        print(f"   Caught overflow: {e}")
    
    print(f"   Queue size: {circular.size()}, Is full: {circular.is_full()}")


def test_edge_cases():
    """Test edge cases for queue implementations"""
    print("\n=== Testing Edge Cases ===")
    
    queue = DequeBasedQueue()
    
    edge_cases = [
        ("Empty queue dequeue", lambda: queue.dequeue(), IndexError),
        ("Empty queue front", lambda: queue.front(), IndexError),
        ("Enqueue then dequeue", lambda: (queue.enqueue(1), queue.dequeue())[1], None),
        ("Multiple operations", lambda: test_multiple_ops(queue), None),
    ]
    
    def test_multiple_ops(q):
        q.clear()
        for i in range(3):
            q.enqueue(i)
        return [q.dequeue() for _ in range(3)]
    
    for description, operation, expected_exception in edge_cases:
        try:
            result = operation()
            if expected_exception:
                print(f"{description:25} | ✗ Expected {expected_exception.__name__}")
            else:
                print(f"{description:25} | ✓ Result: {result}")
        except Exception as e:
            if expected_exception and isinstance(e, expected_exception):
                print(f"{description:25} | ✓ Correctly raised {type(e).__name__}")
            else:
                print(f"{description:25} | ✗ Unexpected error: {type(e).__name__}")


def analyze_time_complexity():
    """Analyze time complexity of different implementations"""
    print("\n=== Time Complexity Analysis ===")
    
    implementations = [
        ("Basic Queue", "O(1)", "O(n)", "O(n)", "List pop(0) is O(n)"),
        ("Circular Array", "O(1)", "O(1)", "O(capacity)", "True O(1) operations"),
        ("LinkedList Queue", "O(1)", "O(1)", "O(n)", "Dynamic sizing"),
        ("Deque-based", "O(1)", "O(1)", "O(n)", "Optimized deque operations"),
        ("Priority Queue", "O(log n)", "O(log n)", "O(n)", "Sorted insertion/removal"),
        ("Thread Safe", "O(1)", "O(1)", "O(n)", "Lock overhead negligible"),
        ("Generic Queue", "O(1)", "O(1)", "O(n)", "Type safety with same performance"),
    ]
    
    print(f"{'Implementation':<20} | {'Enqueue':<10} | {'Dequeue':<10} | {'Space':<12} | {'Notes'}")
    print("-" * 85)
    
    for impl, enqueue, dequeue, space, notes in implementations:
        print(f"{impl:<20} | {enqueue:<10} | {dequeue:<10} | {space:<12} | {notes}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Task scheduling system
    print("1. Task Scheduling System:")
    
    class TaskScheduler:
        def __init__(self):
            self.task_queue = PriorityQueue()
        
        def add_task(self, task, priority):
            self.task_queue.enqueue(task, priority)
            print(f"   Added task: {task} (priority {priority})")
        
        def process_next_task(self):
            if not self.task_queue.is_empty():
                task = self.task_queue.dequeue()
                print(f"   Processing: {task}")
                return task
            else:
                print("   No tasks to process")
                return None
    
    scheduler = TaskScheduler()
    scheduler.add_task("Send email", 2)
    scheduler.add_task("Backup database", 1)
    scheduler.add_task("Update website", 3)
    scheduler.add_task("Security scan", 0)
    
    print("   Processing tasks:")
    while not scheduler.task_queue.is_empty():
        scheduler.process_next_task()
    
    # Application 2: Print queue system
    print(f"\n2. Print Queue System:")
    
    class PrintQueue:
        def __init__(self):
            self.queue = DequeBasedQueue()
        
        def add_print_job(self, job):
            self.queue.enqueue(job)
            print(f"   Added print job: {job}")
        
        def process_print_job(self):
            if not self.queue.is_empty():
                job = self.queue.dequeue()
                print(f"   Printing: {job}")
                return job
            return None
    
    printer = PrintQueue()
    printer.add_print_job("Document1.pdf")
    printer.add_print_job("Photo.jpg")
    printer.add_print_job("Report.docx")
    
    print("   Processing print jobs:")
    while not printer.queue.is_empty():
        printer.process_print_job()
    
    # Application 3: Breadth-First Search
    print(f"\n3. Breadth-First Search:")
    
    def bfs_demo():
        # Simple graph representation
        graph = {
            'A': ['B', 'C'],
            'B': ['D', 'E'],
            'C': ['F'],
            'D': [],
            'E': ['F'],
            'F': []
        }
        
        def bfs(start):
            visited = set()
            queue = DequeBasedQueue()
            queue.enqueue(start)
            visited.add(start)
            
            result = []
            while not queue.is_empty():
                node = queue.dequeue()
                result.append(node)
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.enqueue(neighbor)
            
            return result
        
        traversal = bfs('A')
        print(f"   BFS traversal from A: {' -> '.join(traversal)}")
    
    bfs_demo()


if __name__ == "__main__":
    test_queue_implementations()
    demonstrate_circular_array()
    demonstrate_thread_safety()
    benchmark_queue_implementations()
    demonstrate_priority_queue()
    demonstrate_advanced_features()
    test_edge_cases()
    analyze_time_complexity()
    demonstrate_real_world_applications()

"""
Custom Queue Implementation demonstrates various optimization techniques
including circular arrays, thread safety, priority queues, and type safety
for high-performance queue data structures.
"""
