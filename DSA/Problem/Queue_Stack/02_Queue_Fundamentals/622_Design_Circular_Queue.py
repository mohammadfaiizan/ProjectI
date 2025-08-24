"""
622. Design Circular Queue - Multiple Approaches
Difficulty: Medium

Design your implementation of the circular queue. The circular queue is a linear data structure in which the operations are performed based on FIFO (First In First Out) principle and the last position is connected back to the first position to make a circle. It is also called "Ring Buffer".

One of the benefits of the circular queue is that we can make use of the spaces in front of the queue. In a normal queue, once the queue becomes full, we cannot insert the next element even if there is a space in front of the queue. But using the circular queue, we can use the space to store new values.

Implement the MyCircularQueue class:
- MyCircularQueue(k) Initializes the object with the size of the queue to be k.
- boolean enQueue(int value) Inserts an element into the circular queue. Return true if the operation is successful.
- boolean deQueue() Deletes an element from the circular queue. Return true if the operation is successful.
- int Front() Gets the front item from the queue. If the queue is empty, return -1.
- int Rear() Gets the last item from the queue. If the queue is empty, return -1.
- boolean isEmpty() Checks whether the circular queue is empty or not.
- boolean isFull() Checks whether the circular queue is full or not.
"""

from typing import List, Optional

class MyCircularQueue1:
    """
    Approach 1: Array-based with Front and Rear Pointers
    
    Use array with front and rear pointers to track positions.
    
    Time: O(1) for all operations, Space: O(k)
    """
    
    def __init__(self, k: int):
        self.capacity = k
        self.queue = [0] * k
        self.front = 0
        self.rear = -1
        self.size = 0
    
    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = value
        self.size += 1
        return True
    
    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return True
    
    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.front]
    
    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.rear]
    
    def isEmpty(self) -> bool:
        return self.size == 0
    
    def isFull(self) -> bool:
        return self.size == self.capacity


class MyCircularQueue2:
    """
    Approach 2: Array-based without Size Counter
    
    Use only front and rear pointers without explicit size counter.
    
    Time: O(1) for all operations, Space: O(k)
    """
    
    def __init__(self, k: int):
        self.capacity = k + 1  # One extra space to distinguish full from empty
        self.queue = [0] * self.capacity
        self.front = 0
        self.rear = 0
    
    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        
        self.queue[self.rear] = value
        self.rear = (self.rear + 1) % self.capacity
        return True
    
    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        
        self.front = (self.front + 1) % self.capacity
        return True
    
    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.front]
    
    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[(self.rear - 1 + self.capacity) % self.capacity]
    
    def isEmpty(self) -> bool:
        return self.front == self.rear
    
    def isFull(self) -> bool:
        return (self.rear + 1) % self.capacity == self.front


class MyCircularQueue3:
    """
    Approach 3: Linked List Implementation
    
    Use circular linked list for dynamic implementation.
    
    Time: O(1) for all operations, Space: O(k)
    """
    
    class Node:
        def __init__(self, value: int, next_node=None):
            self.value = value
            self.next = next_node
    
    def __init__(self, k: int):
        self.capacity = k
        self.size = 0
        self.head = None
        self.tail = None
    
    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        
        new_node = self.Node(value)
        
        if self.isEmpty():
            self.head = new_node
            self.tail = new_node
            new_node.next = new_node  # Point to itself
        else:
            new_node.next = self.head
            self.tail.next = new_node
            self.tail = new_node
        
        self.size += 1
        return True
    
    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        
        if self.size == 1:
            self.head = None
            self.tail = None
        else:
            self.tail.next = self.head.next
            self.head = self.head.next
        
        self.size -= 1
        return True
    
    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.head.value
    
    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.tail.value
    
    def isEmpty(self) -> bool:
        return self.size == 0
    
    def isFull(self) -> bool:
        return self.size == self.capacity


class MyCircularQueue4:
    """
    Approach 4: List-based with Modular Arithmetic
    
    Use Python list with careful index management.
    
    Time: O(1) for all operations, Space: O(k)
    """
    
    def __init__(self, k: int):
        self.capacity = k
        self.queue = [None] * k
        self.front_idx = 0
        self.count = 0
    
    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        
        rear_idx = (self.front_idx + self.count) % self.capacity
        self.queue[rear_idx] = value
        self.count += 1
        return True
    
    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        
        self.queue[self.front_idx] = None  # Optional: clear the slot
        self.front_idx = (self.front_idx + 1) % self.capacity
        self.count -= 1
        return True
    
    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.front_idx]
    
    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        rear_idx = (self.front_idx + self.count - 1) % self.capacity
        return self.queue[rear_idx]
    
    def isEmpty(self) -> bool:
        return self.count == 0
    
    def isFull(self) -> bool:
        return self.count == self.capacity


class MyCircularQueue5:
    """
    Approach 5: Deque-based Implementation
    
    Use collections.deque with size limit.
    
    Time: O(1) for all operations, Space: O(k)
    """
    
    def __init__(self, k: int):
        from collections import deque
        self.capacity = k
        self.queue = deque(maxlen=k)
        self.current_size = 0
    
    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        
        self.queue.append(value)
        self.current_size += 1
        return True
    
    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        
        self.queue.popleft()
        self.current_size -= 1
        return True
    
    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[0]
    
    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[-1]
    
    def isEmpty(self) -> bool:
        return self.current_size == 0
    
    def isFull(self) -> bool:
        return self.current_size == self.capacity


class MyCircularQueue6:
    """
    Approach 6: Thread-Safe Circular Queue
    
    Add thread safety with locks.
    
    Time: O(1) for all operations, Space: O(k)
    """
    
    def __init__(self, k: int):
        import threading
        self.capacity = k
        self.queue = [0] * k
        self.front = 0
        self.rear = -1
        self.size = 0
        self.lock = threading.Lock()
    
    def enQueue(self, value: int) -> bool:
        with self.lock:
            if self.size >= self.capacity:
                return False
            
            self.rear = (self.rear + 1) % self.capacity
            self.queue[self.rear] = value
            self.size += 1
            return True
    
    def deQueue(self) -> bool:
        with self.lock:
            if self.size == 0:
                return False
            
            self.front = (self.front + 1) % self.capacity
            self.size -= 1
            return True
    
    def Front(self) -> int:
        with self.lock:
            if self.size == 0:
                return -1
            return self.queue[self.front]
    
    def Rear(self) -> int:
        with self.lock:
            if self.size == 0:
                return -1
            return self.queue[self.rear]
    
    def isEmpty(self) -> bool:
        with self.lock:
            return self.size == 0
    
    def isFull(self) -> bool:
        with self.lock:
            return self.size == self.capacity


def test_circular_queue_implementations():
    """Test all circular queue implementations"""
    
    implementations = [
        ("Array with Size Counter", MyCircularQueue1),
        ("Array without Size Counter", MyCircularQueue2),
        ("Linked List Implementation", MyCircularQueue3),
        ("List with Modular Arithmetic", MyCircularQueue4),
        ("Deque-based Implementation", MyCircularQueue5),
        ("Thread-Safe Implementation", MyCircularQueue6),
    ]
    
    def test_implementation(impl_class, impl_name):
        print(f"\n--- Testing {impl_name} ---")
        
        try:
            # Test with capacity 3
            queue = impl_class(3)
            
            operations = [
                ("isEmpty", None, True),
                ("enQueue", 1, True),
                ("enQueue", 2, True),
                ("enQueue", 3, True),
                ("isFull", None, True),
                ("enQueue", 4, False),  # Should fail - queue full
                ("Front", None, 1),
                ("Rear", None, 3),
                ("deQueue", None, True),
                ("Front", None, 2),
                ("enQueue", 4, True),  # Should succeed now
                ("Rear", None, 4),
                ("deQueue", None, True),
                ("deQueue", None, True),
                ("deQueue", None, True),
                ("isEmpty", None, True),
                ("deQueue", None, False),  # Should fail - queue empty
                ("Front", None, -1),
                ("Rear", None, -1),
            ]
            
            for op_name, value, expected in operations:
                if op_name == "enQueue":
                    result = queue.enQueue(value)
                    status = "✓" if result == expected else "✗"
                    print(f"enQueue({value}) = {result} {status}")
                elif op_name == "deQueue":
                    result = queue.deQueue()
                    status = "✓" if result == expected else "✗"
                    print(f"deQueue() = {result} {status}")
                elif op_name == "Front":
                    result = queue.Front()
                    status = "✓" if result == expected else "✗"
                    print(f"Front() = {result} {status}")
                elif op_name == "Rear":
                    result = queue.Rear()
                    status = "✓" if result == expected else "✗"
                    print(f"Rear() = {result} {status}")
                elif op_name == "isEmpty":
                    result = queue.isEmpty()
                    status = "✓" if result == expected else "✗"
                    print(f"isEmpty() = {result} {status}")
                elif op_name == "isFull":
                    result = queue.isFull()
                    status = "✓" if result == expected else "✗"
                    print(f"isFull() = {result} {status}")
        
        except Exception as e:
            print(f"ERROR: {str(e)}")
    
    print("=== Testing Circular Queue Implementations ===")
    
    for impl_name, impl_class in implementations:
        test_implementation(impl_class, impl_name)


def demonstrate_circular_behavior():
    """Demonstrate circular queue wrap-around behavior"""
    print("\n=== Circular Queue Wrap-around Demonstration ===")
    
    queue = MyCircularQueue1(4)
    
    print("Creating circular queue with capacity 4")
    print(f"Initial state: isEmpty={queue.isEmpty()}, isFull={queue.isFull()}")
    
    # Fill the queue
    print("\nFilling the queue:")
    for i in range(1, 5):
        success = queue.enQueue(i)
        print(f"enQueue({i}) = {success} | Front: {queue.Front()}, Rear: {queue.Rear()}")
    
    print(f"Queue full: {queue.isFull()}")
    
    # Remove some elements
    print("\nRemoving 2 elements:")
    for _ in range(2):
        success = queue.deQueue()
        front = queue.Front()
        rear = queue.Rear()
        print(f"deQueue() = {success} | Front: {front}, Rear: {rear}")
    
    # Add elements to demonstrate wrap-around
    print("\nAdding elements to demonstrate wrap-around:")
    for i in range(5, 7):
        success = queue.enQueue(i)
        print(f"enQueue({i}) = {success} | Front: {queue.Front()}, Rear: {queue.Rear()}")
    
    print(f"Final state: Front: {queue.Front()}, Rear: {queue.Rear()}")


def visualize_array_indices():
    """Visualize how array indices work in circular queue"""
    print("\n=== Array Indices Visualization ===")
    
    queue = MyCircularQueue1(5)
    
    def show_state():
        print(f"Array: {queue.queue}")
        print(f"Front: {queue.front}, Rear: {queue.rear}, Size: {queue.size}")
        print(f"Front value: {queue.Front()}, Rear value: {queue.Rear()}")
        print("-" * 40)
    
    print("Initial state:")
    show_state()
    
    # Add elements
    print("Adding elements 10, 20, 30:")
    for val in [10, 20, 30]:
        queue.enQueue(val)
        print(f"After enQueue({val}):")
        show_state()
    
    # Remove elements
    print("Removing 2 elements:")
    for _ in range(2):
        queue.deQueue()
        print("After deQueue():")
        show_state()
    
    # Add more to show wrap-around
    print("Adding 40, 50, 60 to show wrap-around:")
    for val in [40, 50, 60]:
        queue.enQueue(val)
        print(f"After enQueue({val}):")
        show_state()


def benchmark_circular_queue():
    """Benchmark different circular queue implementations"""
    import time
    import random
    
    implementations = [
        ("Array with Size Counter", MyCircularQueue1),
        ("Array without Size Counter", MyCircularQueue2),
        ("List with Modular Arithmetic", MyCircularQueue4),
        ("Deque-based Implementation", MyCircularQueue5),
    ]
    
    capacity = 1000
    n_operations = 10000
    
    print("\n=== Circular Queue Performance Benchmark ===")
    print(f"Capacity: {capacity}, Operations: {n_operations}")
    
    for impl_name, impl_class in implementations:
        queue = impl_class(capacity)
        
        start_time = time.time()
        
        # Perform random operations
        for _ in range(n_operations):
            operation = random.choice(['enQueue', 'deQueue', 'Front', 'Rear'])
            
            try:
                if operation == 'enQueue':
                    queue.enQueue(random.randint(1, 1000))
                elif operation == 'deQueue':
                    queue.deQueue()
                elif operation == 'Front':
                    queue.Front()
                elif operation == 'Rear':
                    queue.Rear()
            except:
                pass  # Ignore errors for benchmark
        
        end_time = time.time()
        
        print(f"{impl_name:30} | Time: {end_time - start_time:.4f}s")


def test_edge_cases():
    """Test edge cases for circular queue"""
    print("\n=== Testing Edge Cases ===")
    
    queue = MyCircularQueue1(1)  # Capacity 1
    
    edge_cases = [
        ("Single capacity enQueue", lambda: queue.enQueue(42), True),
        ("Single capacity isFull", lambda: queue.isFull(), True),
        ("Single capacity Front", lambda: queue.Front(), 42),
        ("Single capacity Rear", lambda: queue.Rear(), 42),
        ("Single capacity deQueue", lambda: queue.deQueue(), True),
        ("Single capacity isEmpty", lambda: queue.isEmpty(), True),
        ("Empty queue Front", lambda: queue.Front(), -1),
        ("Empty queue Rear", lambda: queue.Rear(), -1),
    ]
    
    for description, operation, expected in edge_cases:
        try:
            result = operation()
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | Expected: {expected}, Got: {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def memory_efficiency_analysis():
    """Analyze memory efficiency of different implementations"""
    print("\n=== Memory Efficiency Analysis ===")
    
    import sys
    
    capacity = 1000
    
    implementations = [
        ("Array with Size Counter", MyCircularQueue1),
        ("Array without Size Counter", MyCircularQueue2),
        ("Linked List Implementation", MyCircularQueue3),
        ("List with Modular Arithmetic", MyCircularQueue4),
    ]
    
    for impl_name, impl_class in implementations:
        queue = impl_class(capacity)
        
        # Fill half the queue
        for i in range(capacity // 2):
            queue.enQueue(i)
        
        # Estimate memory usage
        memory_size = sys.getsizeof(queue)
        
        # Add size of internal data structures
        if hasattr(queue, 'queue'):
            memory_size += sys.getsizeof(queue.queue)
        
        print(f"{impl_name:30} | Memory: ~{memory_size} bytes")


if __name__ == "__main__":
    test_circular_queue_implementations()
    demonstrate_circular_behavior()
    visualize_array_indices()
    test_edge_cases()
    benchmark_circular_queue()
    memory_efficiency_analysis()

"""
Design Circular Queue demonstrates multiple implementation approaches
including array-based, linked list, deque-based, and thread-safe
implementations with comprehensive testing and performance analysis.
"""
