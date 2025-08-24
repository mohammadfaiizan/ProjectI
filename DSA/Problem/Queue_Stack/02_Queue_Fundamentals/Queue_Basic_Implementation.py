"""
Queue Basic Implementation - Multiple Approaches
Difficulty: Easy

Implement a queue data structure with basic operations:
- enqueue(x): Add element x to the rear of the queue
- dequeue(): Remove and return the front element
- front(): Return the front element without removing it
- rear(): Return the rear element without removing it
- isEmpty(): Check if the queue is empty
- size(): Return the number of elements in the queue
"""

from typing import Any, Optional
from collections import deque
import threading

class ArrayQueue:
    """
    Approach 1: Dynamic Array-based Queue
    
    Use dynamic array (list) for queue implementation.
    
    Time: enqueue O(1), dequeue O(n), Space: O(n)
    """
    
    def __init__(self, capacity: int = None):
        self.items = []
        self.capacity = capacity
    
    def enqueue(self, item: Any) -> bool:
        """Add item to rear of queue"""
        if self.capacity and len(self.items) >= self.capacity:
            raise OverflowError("Queue overflow")
        
        self.items.append(item)
        return True
    
    def dequeue(self) -> Any:
        """Remove and return front item"""
        if self.isEmpty():
            raise IndexError("Queue underflow")
        
        return self.items.pop(0)  # O(n) operation
    
    def front(self) -> Any:
        """Return front item without removing"""
        if self.isEmpty():
            raise IndexError("Queue is empty")
        
        return self.items[0]
    
    def rear(self) -> Any:
        """Return rear item without removing"""
        if self.isEmpty():
            raise IndexError("Queue is empty")
        
        return self.items[-1]
    
    def isEmpty(self) -> bool:
        """Check if queue is empty"""
        return len(self.items) == 0
    
    def size(self) -> int:
        """Return number of items in queue"""
        return len(self.items)
    
    def __str__(self) -> str:
        return f"ArrayQueue({self.items})"


class LinkedListQueue:
    """
    Approach 2: Linked List-based Queue
    
    Use linked list for efficient queue operations.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    class Node:
        def __init__(self, data: Any, next_node=None):
            self.data = data
            self.next = next_node
    
    def __init__(self):
        self.front_node = None
        self.rear_node = None
        self._size = 0
    
    def enqueue(self, item: Any) -> bool:
        """Add item to rear of queue"""
        new_node = self.Node(item)
        
        if self.rear_node is None:
            # First element
            self.front_node = new_node
            self.rear_node = new_node
        else:
            # Add to rear
            self.rear_node.next = new_node
            self.rear_node = new_node
        
        self._size += 1
        return True
    
    def dequeue(self) -> Any:
        """Remove and return front item"""
        if self.isEmpty():
            raise IndexError("Queue underflow")
        
        data = self.front_node.data
        self.front_node = self.front_node.next
        
        if self.front_node is None:
            # Queue became empty
            self.rear_node = None
        
        self._size -= 1
        return data
    
    def front(self) -> Any:
        """Return front item without removing"""
        if self.isEmpty():
            raise IndexError("Queue is empty")
        
        return self.front_node.data
    
    def rear(self) -> Any:
        """Return rear item without removing"""
        if self.isEmpty():
            raise IndexError("Queue is empty")
        
        return self.rear_node.data
    
    def isEmpty(self) -> bool:
        """Check if queue is empty"""
        return self.front_node is None
    
    def size(self) -> int:
        """Return number of items in queue"""
        return self._size
    
    def __str__(self) -> str:
        items = []
        current = self.front_node
        while current:
            items.append(current.data)
            current = current.next
        return f"LinkedQueue({items})"


class CircularArrayQueue:
    """
    Approach 3: Circular Array Queue
    
    Use circular array for memory-efficient queue.
    
    Time: O(1) for all operations, Space: O(capacity)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items = [None] * capacity
        self.front_idx = 0
        self.rear_idx = -1
        self._size = 0
    
    def enqueue(self, item: Any) -> bool:
        """Add item to rear of queue"""
        if self.isFull():
            raise OverflowError("Queue overflow")
        
        self.rear_idx = (self.rear_idx + 1) % self.capacity
        self.items[self.rear_idx] = item
        self._size += 1
        return True
    
    def dequeue(self) -> Any:
        """Remove and return front item"""
        if self.isEmpty():
            raise IndexError("Queue underflow")
        
        item = self.items[self.front_idx]
        self.items[self.front_idx] = None  # Clear reference
        self.front_idx = (self.front_idx + 1) % self.capacity
        self._size -= 1
        return item
    
    def front(self) -> Any:
        """Return front item without removing"""
        if self.isEmpty():
            raise IndexError("Queue is empty")
        
        return self.items[self.front_idx]
    
    def rear(self) -> Any:
        """Return rear item without removing"""
        if self.isEmpty():
            raise IndexError("Queue is empty")
        
        return self.items[self.rear_idx]
    
    def isEmpty(self) -> bool:
        """Check if queue is empty"""
        return self._size == 0
    
    def isFull(self) -> bool:
        """Check if queue is full"""
        return self._size >= self.capacity
    
    def size(self) -> int:
        """Return number of items in queue"""
        return self._size
    
    def __str__(self) -> str:
        if self.isEmpty():
            return "CircularQueue([])"
        
        items = []
        for i in range(self._size):
            idx = (self.front_idx + i) % self.capacity
            items.append(self.items[idx])
        return f"CircularQueue({items})"


class DequeQueue:
    """
    Approach 4: Deque-based Queue
    
    Use collections.deque for optimal performance.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    def __init__(self, capacity: int = None):
        self.items = deque()
        self.capacity = capacity
    
    def enqueue(self, item: Any) -> bool:
        """Add item to rear of queue"""
        if self.capacity and len(self.items) >= self.capacity:
            raise OverflowError("Queue overflow")
        
        self.items.append(item)
        return True
    
    def dequeue(self) -> Any:
        """Remove and return front item"""
        if self.isEmpty():
            raise IndexError("Queue underflow")
        
        return self.items.popleft()
    
    def front(self) -> Any:
        """Return front item without removing"""
        if self.isEmpty():
            raise IndexError("Queue is empty")
        
        return self.items[0]
    
    def rear(self) -> Any:
        """Return rear item without removing"""
        if self.isEmpty():
            raise IndexError("Queue is empty")
        
        return self.items[-1]
    
    def isEmpty(self) -> bool:
        """Check if queue is empty"""
        return len(self.items) == 0
    
    def size(self) -> int:
        """Return number of items in queue"""
        return len(self.items)
    
    def __str__(self) -> str:
        return f"DequeQueue({list(self.items)})"


class ThreadSafeQueue:
    """
    Approach 5: Thread-safe Queue
    
    Use locks to ensure thread safety.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    def __init__(self, capacity: int = None):
        self.items = deque()
        self.capacity = capacity
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
    
    def enqueue(self, item: Any, block: bool = True, timeout: float = None) -> bool:
        """Thread-safe enqueue operation"""
        with self.not_full:
            if self.capacity:
                while len(self.items) >= self.capacity:
                    if not block:
                        return False
                    if not self.not_full.wait(timeout):
                        return False
            
            self.items.append(item)
            self.not_empty.notify()
            return True
    
    def dequeue(self, block: bool = True, timeout: float = None) -> Any:
        """Thread-safe dequeue operation"""
        with self.not_empty:
            while len(self.items) == 0:
                if not block:
                    raise IndexError("Queue underflow")
                if not self.not_empty.wait(timeout):
                    raise IndexError("Queue underflow")
            
            item = self.items.popleft()
            self.not_full.notify()
            return item
    
    def front(self) -> Any:
        """Thread-safe front operation"""
        with self.lock:
            if len(self.items) == 0:
                raise IndexError("Queue is empty")
            return self.items[0]
    
    def rear(self) -> Any:
        """Thread-safe rear operation"""
        with self.lock:
            if len(self.items) == 0:
                raise IndexError("Queue is empty")
            return self.items[-1]
    
    def isEmpty(self) -> bool:
        """Thread-safe isEmpty check"""
        with self.lock:
            return len(self.items) == 0
    
    def size(self) -> int:
        """Thread-safe size check"""
        with self.lock:
            return len(self.items)


class PriorityQueue:
    """
    Approach 6: Priority Queue Implementation
    
    Queue with priority-based ordering.
    
    Time: enqueue O(log n), dequeue O(log n), Space: O(n)
    """
    
    def __init__(self):
        import heapq
        self.heap = []
        self.counter = 0  # For stable ordering
    
    def enqueue(self, item: Any, priority: int = 0) -> bool:
        """Add item with priority (lower number = higher priority)"""
        import heapq
        heapq.heappush(self.heap, (priority, self.counter, item))
        self.counter += 1
        return True
    
    def dequeue(self) -> Any:
        """Remove and return highest priority item"""
        if self.isEmpty():
            raise IndexError("Queue underflow")
        
        import heapq
        priority, counter, item = heapq.heappop(self.heap)
        return item
    
    def front(self) -> Any:
        """Return highest priority item without removing"""
        if self.isEmpty():
            raise IndexError("Queue is empty")
        
        return self.heap[0][2]
    
    def rear(self) -> Any:
        """Return lowest priority item without removing"""
        if self.isEmpty():
            raise IndexError("Queue is empty")
        
        # Find item with highest priority value (lowest priority)
        max_priority_item = max(self.heap, key=lambda x: x[0])
        return max_priority_item[2]
    
    def isEmpty(self) -> bool:
        """Check if queue is empty"""
        return len(self.heap) == 0
    
    def size(self) -> int:
        """Return number of items in queue"""
        return len(self.heap)
    
    def __str__(self) -> str:
        items = [item for priority, counter, item in sorted(self.heap)]
        return f"PriorityQueue({items})"


def test_queue_implementations():
    """Test all queue implementations"""
    
    implementations = [
        ("Array Queue", ArrayQueue),
        ("Linked List Queue", LinkedListQueue),
        ("Circular Array Queue", lambda: CircularArrayQueue(10)),
        ("Deque Queue", DequeQueue),
        ("Thread Safe Queue", ThreadSafeQueue),
    ]
    
    test_operations = [
        ("enqueue", 10),
        ("enqueue", 20),
        ("enqueue", 30),
        ("front", None, 10),
        ("rear", None, 30),
        ("size", None, 3),
        ("dequeue", None, 10),
        ("front", None, 20),
        ("enqueue", 40),
        ("rear", None, 40),
        ("size", None, 3),
        ("isEmpty", None, False),
        ("dequeue", None, 20),
        ("dequeue", None, 30),
        ("dequeue", None, 40),
        ("isEmpty", None, True),
    ]
    
    print("=== Testing Queue Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- Testing {impl_name} ---")
        
        try:
            queue = impl_class()
            
            for operation in test_operations:
                if operation[0] == "enqueue":
                    queue.enqueue(operation[1])
                    print(f"enqueue({operation[1]})")
                elif operation[0] == "dequeue":
                    result = queue.dequeue()
                    expected = operation[2]
                    status = "✓" if result == expected else "✗"
                    print(f"dequeue() = {result} {status}")
                elif operation[0] == "front":
                    result = queue.front()
                    expected = operation[2]
                    status = "✓" if result == expected else "✗"
                    print(f"front() = {result} {status}")
                elif operation[0] == "rear":
                    result = queue.rear()
                    expected = operation[2]
                    status = "✓" if result == expected else "✗"
                    print(f"rear() = {result} {status}")
                elif operation[0] == "size":
                    result = queue.size()
                    expected = operation[2]
                    status = "✓" if result == expected else "✗"
                    print(f"size() = {result} {status}")
                elif operation[0] == "isEmpty":
                    result = queue.isEmpty()
                    expected = operation[2]
                    status = "✓" if result == expected else "✗"
                    print(f"isEmpty() = {result} {status}")
        
        except Exception as e:
            print(f"ERROR: {str(e)}")


def test_priority_queue():
    """Test priority queue specific functionality"""
    print("\n=== Testing Priority Queue ===")
    
    pq = PriorityQueue()
    
    operations = [
        ("enqueue", "High Priority", 1),
        ("enqueue", "Low Priority", 5),
        ("enqueue", "Medium Priority", 3),
        ("enqueue", "Highest Priority", 0),
    ]
    
    for op, item, priority in operations:
        pq.enqueue(item, priority)
        print(f"enqueue('{item}', priority={priority}) | front: '{pq.front()}'")
    
    print("\nDequeuing by priority:")
    while not pq.isEmpty():
        item = pq.dequeue()
        print(f"dequeue() = '{item}'")


def demonstrate_fifo_behavior():
    """Demonstrate FIFO queue behavior"""
    print("\n=== FIFO Queue Behavior Demonstration ===")
    
    queue = LinkedListQueue()
    
    print("Adding elements: A, B, C, D")
    for item in ['A', 'B', 'C', 'D']:
        queue.enqueue(item)
        print(f"enqueue('{item}') | front: '{queue.front()}', rear: '{queue.rear()}'")
    
    print("\nRemoving elements (FIFO order):")
    while not queue.isEmpty():
        front_item = queue.front()
        dequeued = queue.dequeue()
        print(f"front: '{front_item}', dequeue() = '{dequeued}'")


def benchmark_queue_implementations():
    """Benchmark different queue implementations"""
    import time
    
    implementations = [
        ("Array Queue", ArrayQueue),
        ("Linked List Queue", LinkedListQueue),
        ("Circular Array Queue", lambda: CircularArrayQueue(20000)),
        ("Deque Queue", DequeQueue),
    ]
    
    n = 10000
    
    print("\n=== Queue Implementation Performance Benchmark ===")
    print(f"Operations: {n} enqueue + {n} dequeue")
    
    for impl_name, impl_class in implementations:
        queue = impl_class()
        
        start_time = time.time()
        
        # Enqueue operations
        for i in range(n):
            queue.enqueue(i)
        
        # Dequeue operations
        for i in range(n):
            queue.dequeue()
        
        end_time = time.time()
        
        print(f"{impl_name:25} | Time: {end_time - start_time:.4f}s")


def test_circular_queue_behavior():
    """Test circular queue wrap-around behavior"""
    print("\n=== Circular Queue Behavior Test ===")
    
    queue = CircularArrayQueue(4)
    
    print("Circular queue with capacity 4")
    
    # Fill queue
    for i in range(1, 5):
        queue.enqueue(i)
        print(f"enqueue({i}) | size: {queue.size()}")
    
    print(f"Queue full: {queue.isFull()}")
    
    # Remove some elements
    for _ in range(2):
        item = queue.dequeue()
        print(f"dequeue() = {item} | size: {queue.size()}")
    
    # Add more elements to show wrap-around
    for i in range(5, 7):
        queue.enqueue(i)
        print(f"enqueue({i}) | front: {queue.front()}, rear: {queue.rear()}")


def test_thread_safety():
    """Test thread-safe queue with multiple threads"""
    print("\n=== Thread Safety Test ===")
    
    import threading
    import time
    
    queue = ThreadSafeQueue(capacity=10)
    results = []
    
    def producer(name: str, start: int, count: int):
        """Producer thread function"""
        for i in range(start, start + count):
            queue.enqueue(f"{name}-{i}")
            time.sleep(0.001)  # Small delay
    
    def consumer(name: str, count: int):
        """Consumer thread function"""
        for _ in range(count):
            try:
                item = queue.dequeue(timeout=1.0)
                results.append(f"{name} consumed {item}")
            except:
                break
    
    # Create threads
    producers = [
        threading.Thread(target=producer, args=("P1", 1, 5)),
        threading.Thread(target=producer, args=("P2", 10, 5)),
    ]
    
    consumers = [
        threading.Thread(target=consumer, args=("C1", 5)),
        threading.Thread(target=consumer, args=("C2", 5)),
    ]
    
    # Start threads
    for t in producers + consumers:
        t.start()
    
    # Wait for completion
    for t in producers + consumers:
        t.join()
    
    print(f"Thread safety test completed. Results: {len(results)} items processed")
    for result in results[:10]:  # Show first 10 results
        print(f"  {result}")


def memory_usage_comparison():
    """Compare memory usage of different implementations"""
    print("\n=== Memory Usage Comparison ===")
    
    import sys
    
    implementations = [
        ("Array Queue", ArrayQueue),
        ("Linked List Queue", LinkedListQueue),
        ("Circular Array Queue", lambda: CircularArrayQueue(1000)),
        ("Deque Queue", DequeQueue),
    ]
    
    n_elements = 1000
    
    for impl_name, impl_class in implementations:
        queue = impl_class()
        
        # Add elements
        for i in range(n_elements):
            queue.enqueue(i)
        
        # Estimate memory usage
        memory_size = sys.getsizeof(queue)
        if hasattr(queue, 'items'):
            memory_size += sys.getsizeof(queue.items)
        
        print(f"{impl_name:25} | Memory: ~{memory_size} bytes")


if __name__ == "__main__":
    test_queue_implementations()
    test_priority_queue()
    demonstrate_fifo_behavior()
    test_circular_queue_behavior()
    test_thread_safety()
    benchmark_queue_implementations()
    memory_usage_comparison()

"""
Queue Basic Implementation demonstrates various queue implementation
approaches including array-based, linked list, circular array, deque-based,
thread-safe, and priority queue implementations with comprehensive testing.
"""
