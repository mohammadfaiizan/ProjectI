"""
622. Design Circular Queue - Multiple Approaches
Difficulty: Medium

Design your implementation of the circular queue. The circular queue is a linear data structure 
in which the operations are performed based on FIFO (First In First Out) principle and the last 
position is connected back to the first position to make a circle.

Implement the MyCircularQueue class:
- MyCircularQueue(k) Initializes the object with the size of the queue to be k.
- boolean enQueue(int value) Inserts an element into the circular queue. Return true if successful.
- boolean deQueue() Deletes an element from the circular queue. Return true if successful.
- int Front() Gets the front item from the queue. If the queue is empty, return -1.
- int Rear() Gets the last item from the queue. If the queue is empty, return -1.
- boolean isEmpty() Checks whether the circular queue is empty or not.
- boolean isFull() Checks whether the circular queue is full or not.
"""

from typing import List, Optional

class MyCircularQueueArray:
    """
    Approach 1: Array-based Implementation
    
    Use a fixed-size array with front and rear pointers.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(k) where k is the queue capacity
    """
    
    def __init__(self, k: int):
        self.capacity = k
        self.queue = [0] * k
        self.front_index = 0
        self.rear_index = -1
        self.size = 0
    
    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        
        self.rear_index = (self.rear_index + 1) % self.capacity
        self.queue[self.rear_index] = value
        self.size += 1
        return True
    
    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        
        self.front_index = (self.front_index + 1) % self.capacity
        self.size -= 1
        return True
    
    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.front_index]
    
    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.rear_index]
    
    def isEmpty(self) -> bool:
        return self.size == 0
    
    def isFull(self) -> bool:
        return self.size == self.capacity

class MyCircularQueueLinkedList:
    """
    Approach 2: Linked List Implementation
    
    Use a circular linked list to implement the queue.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(k) where k is the number of elements
    """
    
    class Node:
        def __init__(self, value: int):
            self.value = value
            self.next: Optional['MyCircularQueueLinkedList.Node'] = None
    
    def __init__(self, k: int):
        self.capacity = k
        self.size = 0
        self.front: Optional['MyCircularQueueLinkedList.Node'] = None
        self.rear: Optional['MyCircularQueueLinkedList.Node'] = None
    
    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        
        new_node = self.Node(value)
        
        if self.isEmpty():
            self.front = self.rear = new_node
            new_node.next = new_node  # Point to itself
        else:
            new_node.next = self.front
            self.rear.next = new_node
            self.rear = new_node
        
        self.size += 1
        return True
    
    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        
        if self.size == 1:
            self.front = self.rear = None
        else:
            self.rear.next = self.front.next
            self.front = self.front.next
        
        self.size -= 1
        return True
    
    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.front.value
    
    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.rear.value
    
    def isEmpty(self) -> bool:
        return self.size == 0
    
    def isFull(self) -> bool:
        return self.size == self.capacity

class MyCircularQueueTwoPointers:
    """
    Approach 3: Two Pointers without Size Counter
    
    Use two pointers and a sentinel value to distinguish empty from full.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(k + 1) where k is the queue capacity
    """
    
    def __init__(self, k: int):
        self.capacity = k + 1  # Extra space to distinguish empty from full
        self.queue = [0] * self.capacity
        self.front_index = 0
        self.rear_index = 0
    
    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        
        self.queue[self.rear_index] = value
        self.rear_index = (self.rear_index + 1) % self.capacity
        return True
    
    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        
        self.front_index = (self.front_index + 1) % self.capacity
        return True
    
    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.front_index]
    
    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        # Rear is at (rear_index - 1) % capacity
        rear_pos = (self.rear_index - 1 + self.capacity) % self.capacity
        return self.queue[rear_pos]
    
    def isEmpty(self) -> bool:
        return self.front_index == self.rear_index
    
    def isFull(self) -> bool:
        return (self.rear_index + 1) % self.capacity == self.front_index

class MyCircularQueueThreadSafe:
    """
    Approach 4: Thread-Safe Implementation
    
    Add thread safety for concurrent access.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(k) where k is the queue capacity
    """
    
    def __init__(self, k: int):
        import threading
        self.capacity = k
        self.queue = [0] * k
        self.front_index = 0
        self.rear_index = -1
        self.size = 0
        self.lock = threading.RLock()
    
    def enQueue(self, value: int) -> bool:
        with self.lock:
            if self.isFull():
                return False
            
            self.rear_index = (self.rear_index + 1) % self.capacity
            self.queue[self.rear_index] = value
            self.size += 1
            return True
    
    def deQueue(self) -> bool:
        with self.lock:
            if self.isEmpty():
                return False
            
            self.front_index = (self.front_index + 1) % self.capacity
            self.size -= 1
            return True
    
    def Front(self) -> int:
        with self.lock:
            if self.isEmpty():
                return -1
            return self.queue[self.front_index]
    
    def Rear(self) -> int:
        with self.lock:
            if self.isEmpty():
                return -1
            return self.queue[self.rear_index]
    
    def isEmpty(self) -> bool:
        with self.lock:
            return self.size == 0
    
    def isFull(self) -> bool:
        with self.lock:
            return self.size == self.capacity
    
    def getSize(self) -> int:
        with self.lock:
            return self.size

class MyCircularQueueDynamic:
    """
    Approach 5: Dynamic Resizing Circular Queue
    
    Allow the queue to resize when needed.
    
    Time Complexity: O(1) amortized for enQueue, O(1) for others
    Space Complexity: O(k) where k is the current capacity
    """
    
    def __init__(self, k: int):
        self.capacity = k
        self.queue = [0] * k
        self.front_index = 0
        self.rear_index = -1
        self.size = 0
        self.max_capacity = k * 4  # Limit for resizing
    
    def enQueue(self, value: int) -> bool:
        if self.isFull():
            if self.capacity < self.max_capacity:
                self._resize(self.capacity * 2)
            else:
                return False
        
        self.rear_index = (self.rear_index + 1) % self.capacity
        self.queue[self.rear_index] = value
        self.size += 1
        return True
    
    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        
        self.front_index = (self.front_index + 1) % self.capacity
        self.size -= 1
        
        # Shrink if utilization is low
        if self.size < self.capacity // 4 and self.capacity > 4:
            self._resize(self.capacity // 2)
        
        return True
    
    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.front_index]
    
    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.rear_index]
    
    def isEmpty(self) -> bool:
        return self.size == 0
    
    def isFull(self) -> bool:
        return self.size == self.capacity
    
    def _resize(self, new_capacity: int) -> None:
        """Resize the queue array"""
        new_queue = [0] * new_capacity
        
        # Copy elements to new array
        for i in range(self.size):
            old_index = (self.front_index + i) % self.capacity
            new_queue[i] = self.queue[old_index]
        
        self.queue = new_queue
        self.capacity = new_capacity
        self.front_index = 0
        self.rear_index = self.size - 1 if self.size > 0 else -1


def test_circular_queue_basic():
    """Test basic circular queue operations"""
    print("=== Testing Basic Circular Queue Operations ===")
    
    implementations = [
        ("Array-based", MyCircularQueueArray),
        ("Linked List", MyCircularQueueLinkedList),
        ("Two Pointers", MyCircularQueueTwoPointers),
        ("Thread-Safe", MyCircularQueueThreadSafe),
        ("Dynamic Resizing", MyCircularQueueDynamic)
    ]
    
    for name, QueueClass in implementations:
        print(f"\n{name}:")
        
        queue = QueueClass(3)
        
        # Test empty queue
        print(f"  isEmpty(): {queue.isEmpty()}")
        print(f"  isFull(): {queue.isFull()}")
        print(f"  Front(): {queue.Front()}")
        print(f"  Rear(): {queue.Rear()}")
        
        # Test enqueue operations
        print(f"  enQueue(1): {queue.enQueue(1)}")
        print(f"  enQueue(2): {queue.enQueue(2)}")
        print(f"  enQueue(3): {queue.enQueue(3)}")
        print(f"  enQueue(4): {queue.enQueue(4)}")  # Should fail
        
        print(f"  Front(): {queue.Front()}")
        print(f"  Rear(): {queue.Rear()}")
        print(f"  isFull(): {queue.isFull()}")
        
        # Test dequeue operations
        print(f"  deQueue(): {queue.deQueue()}")
        print(f"  Front(): {queue.Front()}")
        print(f"  enQueue(4): {queue.enQueue(4)}")  # Should succeed now
        print(f"  Rear(): {queue.Rear()}")

def test_circular_queue_edge_cases():
    """Test circular queue edge cases"""
    print("\n=== Testing Circular Queue Edge Cases ===")
    
    queue = MyCircularQueueArray(1)
    
    # Test capacity 1
    print("Testing capacity 1:")
    print(f"  isEmpty(): {queue.isEmpty()}")
    print(f"  enQueue(5): {queue.enQueue(5)}")
    print(f"  isFull(): {queue.isFull()}")
    print(f"  Front(): {queue.Front()}")
    print(f"  Rear(): {queue.Rear()}")
    print(f"  deQueue(): {queue.deQueue()}")
    print(f"  isEmpty(): {queue.isEmpty()}")
    
    # Test wraparound behavior
    print(f"\nTesting wraparound:")
    queue2 = MyCircularQueueArray(3)
    
    # Fill and empty multiple times
    for cycle in range(2):
        print(f"  Cycle {cycle + 1}:")
        for i in range(3):
            result = queue2.enQueue(i + cycle * 3)
            print(f"    enQueue({i + cycle * 3}): {result}")
        
        for i in range(3):
            front = queue2.Front()
            result = queue2.deQueue()
            print(f"    deQueue(): {result}, was front: {front}")

def test_circular_behavior():
    """Test the circular nature of the queue"""
    print("\n=== Testing Circular Behavior ===")
    
    queue = MyCircularQueueArray(4)
    
    # Fill the queue
    for i in range(4):
        queue.enQueue(i)
    
    print("Initial state (full):")
    print(f"  Front: {queue.Front()}, Rear: {queue.Rear()}")
    
    # Demonstrate circular behavior
    print(f"\nCircular operations:")
    for i in range(6):
        # Dequeue one, enqueue one
        old_front = queue.Front()
        queue.deQueue()
        queue.enQueue(10 + i)
        
        print(f"  Step {i+1}: Removed {old_front}, Added {10 + i}")
        print(f"    Front: {queue.Front()}, Rear: {queue.Rear()}")

def test_queue_performance():
    """Test circular queue performance"""
    print("\n=== Testing Circular Queue Performance ===")
    
    import time
    
    implementations = [
        ("Array-based", MyCircularQueueArray),
        ("Linked List", MyCircularQueueLinkedList),
        ("Two Pointers", MyCircularQueueTwoPointers)
    ]
    
    operations = 100000
    capacity = 1000
    
    for name, QueueClass in implementations:
        queue = QueueClass(capacity)
        
        # Fill queue to half capacity
        for i in range(capacity // 2):
            queue.enQueue(i)
        
        # Test mixed operations
        start_time = time.time()
        
        for i in range(operations):
            if i % 2 == 0:
                queue.enQueue(i)
            else:
                queue.deQueue()
        
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000
        
        print(f"  {name}: {elapsed:.2f}ms for {operations} operations")

def test_dynamic_resizing():
    """Test dynamic resizing feature"""
    print("\n=== Testing Dynamic Resizing ===")
    
    queue = MyCircularQueueDynamic(2)
    
    print(f"Initial capacity: {queue.capacity}")
    
    # Add elements to trigger resize
    for i in range(8):
        result = queue.enQueue(i)
        print(f"  enQueue({i}): {result}, capacity: {queue.capacity}, size: {queue.size}")
    
    # Remove elements to trigger shrinking
    print(f"\nRemoving elements:")
    for i in range(6):
        front = queue.Front()
        result = queue.deQueue()
        print(f"  deQueue(): {result}, removed: {front}, capacity: {queue.capacity}, size: {queue.size}")

def demonstrate_queue_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Queue Applications ===")
    
    # Application 1: Buffer for streaming data
    print("Application 1: Streaming Buffer")
    buffer = MyCircularQueueArray(5)
    
    # Simulate data streaming
    incoming_data = [10, 20, 30, 40, 50, 60, 70, 80]
    
    for data in incoming_data:
        if buffer.isFull():
            # Process oldest data
            processed = buffer.Front()
            buffer.deQueue()
            print(f"  Processed: {processed}")
        
        buffer.enQueue(data)
        print(f"  Received: {data}, buffer: front={buffer.Front()}, rear={buffer.Rear()}")
    
    # Application 2: Round-robin scheduling
    print(f"\nApplication 2: Round-Robin Task Scheduling")
    scheduler = MyCircularQueueArray(3)
    
    tasks = ["Task A", "Task B", "Task C"]
    task_ids = [hash(task) % 100 for task in tasks]
    
    # Add tasks to scheduler
    for i, task_id in enumerate(task_ids):
        scheduler.enQueue(task_id)
        print(f"  Added {tasks[i]} (ID: {task_id})")
    
    # Simulate round-robin execution
    print(f"  Round-robin execution:")
    for round_num in range(6):  # 2 full rounds
        current_task_id = scheduler.Front()
        scheduler.deQueue()
        scheduler.enQueue(current_task_id)  # Move to back
        
        # Find task name
        task_name = tasks[task_ids.index(current_task_id)]
        print(f"    Round {round_num + 1}: Executing {task_name}")

def test_thread_safety():
    """Test thread safety features"""
    print("\n=== Testing Thread Safety ===")
    
    import threading
    import time
    
    queue = MyCircularQueueThreadSafe(10)
    results = {"enqueued": 0, "dequeued": 0}
    
    def producer():
        for i in range(20):
            if queue.enQueue(i):
                results["enqueued"] += 1
            time.sleep(0.001)  # Small delay
    
    def consumer():
        for i in range(20):
            if queue.deQueue():
                results["dequeued"] += 1
            time.sleep(0.001)  # Small delay
    
    # Create and start threads
    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)
    
    producer_thread.start()
    consumer_thread.start()
    
    producer_thread.join()
    consumer_thread.join()
    
    print(f"Thread safety test results:")
    print(f"  Enqueued: {results['enqueued']}")
    print(f"  Dequeued: {results['dequeued']}")
    print(f"  Final queue size: {queue.getSize()}")

def test_memory_efficiency():
    """Test memory efficiency of different implementations"""
    print("\n=== Testing Memory Efficiency ===")
    
    implementations = [
        ("Array-based", MyCircularQueueArray),
        ("Linked List", MyCircularQueueLinkedList),
        ("Two Pointers", MyCircularQueueTwoPointers)
    ]
    
    capacity = 1000
    
    for name, QueueClass in implementations:
        queue = QueueClass(capacity)
        
        # Fill queue
        for i in range(capacity):
            queue.enQueue(i)
        
        # Estimate memory usage
        if hasattr(queue, 'queue'):
            array_size = len(queue.queue)
        else:
            array_size = 0
        
        if hasattr(queue, 'size'):
            actual_elements = queue.size
        else:
            actual_elements = capacity
        
        print(f"  {name}:")
        print(f"    Requested capacity: {capacity}")
        print(f"    Array size: {array_size}")
        print(f"    Actual elements: {actual_elements}")
        
        if array_size > 0:
            efficiency = (actual_elements / array_size) * 100
            print(f"    Space efficiency: {efficiency:.1f}%")

def benchmark_queue_scenarios():
    """Benchmark different usage scenarios"""
    print("\n=== Benchmarking Queue Scenarios ===")
    
    import time
    
    scenarios = [
        ("Mostly enqueue", lambda q, i: q.enQueue(i) if i % 10 != 0 else q.deQueue()),
        ("Mostly dequeue", lambda q, i: q.deQueue() if i % 10 != 0 else q.enQueue(i)),
        ("Balanced ops", lambda q, i: q.enQueue(i) if i % 2 == 0 else q.deQueue()),
    ]
    
    for scenario_name, operation in scenarios:
        queue = MyCircularQueueArray(1000)
        
        # Pre-fill for dequeue scenarios
        if "dequeue" in scenario_name:
            for i in range(500):
                queue.enQueue(i)
        
        start_time = time.time()
        
        for i in range(10000):
            operation(queue, i)
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {scenario_name}: {elapsed:.2f}ms")

if __name__ == "__main__":
    test_circular_queue_basic()
    test_circular_queue_edge_cases()
    test_circular_behavior()
    test_queue_performance()
    test_dynamic_resizing()
    demonstrate_queue_applications()
    test_thread_safety()
    test_memory_efficiency()
    benchmark_queue_scenarios()

"""
Circular Queue Design demonstrates key concepts:

Core Approaches:
1. Array-based - Most common and efficient implementation
2. Linked List - Dynamic memory allocation with circular connections
3. Two Pointers - Avoids size counter by using extra space
4. Thread-Safe - Concurrent access support with locks
5. Dynamic Resizing - Automatic capacity adjustment

Key Design Principles:
- Circular nature through modular arithmetic
- Fixed capacity with efficient space utilization
- O(1) operations for all queue operations
- FIFO ordering with wraparound behavior

Advantages of Circular Queue:
- Fixed memory usage (no dynamic allocation)
- Efficient space utilization
- No memory fragmentation
- Predictable performance

Real-world Applications:
- Streaming data buffers
- Round-robin task scheduling
- Producer-consumer scenarios
- Network packet buffering
- Audio/video processing pipelines
- CPU instruction queues

The array-based implementation is most commonly used
due to its simplicity and optimal performance characteristics.
"""
