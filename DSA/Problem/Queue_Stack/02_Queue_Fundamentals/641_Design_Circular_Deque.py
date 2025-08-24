"""
641. Design Circular Deque - Multiple Approaches
Difficulty: Medium

Design your implementation of the circular double-ended queue (deque).

Implement the MyCircularDeque class:
- MyCircularDeque(int k) Initializes the deque with a maximum size of k.
- boolean insertFront(int value) Adds an item at the front of Deque. Return true if the operation is successful.
- boolean insertLast(int value) Adds an item at the rear of Deque. Return true if the operation is successful.
- boolean deleteFront() Deletes an item from the front of Deque. Return true if the operation is successful.
- boolean deleteLast() Deletes an item from the rear of Deque. Return true if the operation is successful.
- int getFront() Gets the front item from the Deque. If the deque is empty, return -1.
- int getRear() Gets the last item from the Deque. If the deque is empty, return -1.
- boolean isEmpty() Checks whether Deque is empty or not.
- boolean isFull() Checks whether Deque is full or not.
"""

from typing import Optional

class MyCircularDeque1:
    """
    Approach 1: Array-based with Front and Rear Pointers
    
    Use circular array with separate front and rear pointers.
    
    Time: O(1) for all operations, Space: O(k)
    """
    
    def __init__(self, k: int):
        self.capacity = k
        self.deque = [0] * k
        self.front = 0
        self.rear = 0
        self.size = 0
    
    def insertFront(self, value: int) -> bool:
        if self.isFull():
            return False
        
        if self.size == 0:
            self.deque[self.front] = value
        else:
            self.front = (self.front - 1 + self.capacity) % self.capacity
            self.deque[self.front] = value
        
        self.size += 1
        return True
    
    def insertLast(self, value: int) -> bool:
        if self.isFull():
            return False
        
        if self.size == 0:
            self.deque[self.rear] = value
        else:
            self.rear = (self.rear + 1) % self.capacity
            self.deque[self.rear] = value
        
        self.size += 1
        return True
    
    def deleteFront(self) -> bool:
        if self.isEmpty():
            return False
        
        if self.size == 1:
            # Reset pointers when deque becomes empty
            pass
        else:
            self.front = (self.front + 1) % self.capacity
        
        self.size -= 1
        return True
    
    def deleteLast(self) -> bool:
        if self.isEmpty():
            return False
        
        if self.size == 1:
            # Reset pointers when deque becomes empty
            pass
        else:
            self.rear = (self.rear - 1 + self.capacity) % self.capacity
        
        self.size -= 1
        return True
    
    def getFront(self) -> int:
        if self.isEmpty():
            return -1
        return self.deque[self.front]
    
    def getRear(self) -> int:
        if self.isEmpty():
            return -1
        return self.deque[self.rear]
    
    def isEmpty(self) -> bool:
        return self.size == 0
    
    def isFull(self) -> bool:
        return self.size == self.capacity


class MyCircularDeque2:
    """
    Approach 2: Array-based without Size Counter
    
    Use only front and rear pointers without explicit size.
    
    Time: O(1) for all operations, Space: O(k+1)
    """
    
    def __init__(self, k: int):
        self.capacity = k + 1  # One extra space to distinguish full from empty
        self.deque = [0] * self.capacity
        self.front = 0
        self.rear = 0
    
    def insertFront(self, value: int) -> bool:
        if self.isFull():
            return False
        
        self.front = (self.front - 1 + self.capacity) % self.capacity
        self.deque[self.front] = value
        return True
    
    def insertLast(self, value: int) -> bool:
        if self.isFull():
            return False
        
        self.deque[self.rear] = value
        self.rear = (self.rear + 1) % self.capacity
        return True
    
    def deleteFront(self) -> bool:
        if self.isEmpty():
            return False
        
        self.front = (self.front + 1) % self.capacity
        return True
    
    def deleteLast(self) -> bool:
        if self.isEmpty():
            return False
        
        self.rear = (self.rear - 1 + self.capacity) % self.capacity
        return True
    
    def getFront(self) -> int:
        if self.isEmpty():
            return -1
        return self.deque[self.front]
    
    def getRear(self) -> int:
        if self.isEmpty():
            return -1
        return self.deque[(self.rear - 1 + self.capacity) % self.capacity]
    
    def isEmpty(self) -> bool:
        return self.front == self.rear
    
    def isFull(self) -> bool:
        return (self.rear + 1) % self.capacity == self.front


class MyCircularDeque3:
    """
    Approach 3: Doubly Linked List Implementation
    
    Use circular doubly linked list for dynamic implementation.
    
    Time: O(1) for all operations, Space: O(k)
    """
    
    class Node:
        def __init__(self, value: int):
            self.value = value
            self.prev = None
            self.next = None
    
    def __init__(self, k: int):
        self.capacity = k
        self.size = 0
        # Create dummy head and tail
        self.head = self.Node(0)
        self.tail = self.Node(0)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_after(self, node: 'Node', value: int) -> 'Node':
        """Add new node after given node"""
        new_node = self.Node(value)
        new_node.prev = node
        new_node.next = node.next
        node.next.prev = new_node
        node.next = new_node
        return new_node
    
    def _remove_node(self, node: 'Node') -> None:
        """Remove given node"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def insertFront(self, value: int) -> bool:
        if self.isFull():
            return False
        
        self._add_after(self.head, value)
        self.size += 1
        return True
    
    def insertLast(self, value: int) -> bool:
        if self.isFull():
            return False
        
        self._add_after(self.tail.prev, value)
        self.size += 1
        return True
    
    def deleteFront(self) -> bool:
        if self.isEmpty():
            return False
        
        self._remove_node(self.head.next)
        self.size -= 1
        return True
    
    def deleteLast(self) -> bool:
        if self.isEmpty():
            return False
        
        self._remove_node(self.tail.prev)
        self.size -= 1
        return True
    
    def getFront(self) -> int:
        if self.isEmpty():
            return -1
        return self.head.next.value
    
    def getRear(self) -> int:
        if self.isEmpty():
            return -1
        return self.tail.prev.value
    
    def isEmpty(self) -> bool:
        return self.size == 0
    
    def isFull(self) -> bool:
        return self.size == self.capacity


class MyCircularDeque4:
    """
    Approach 4: List-based Implementation
    
    Use Python list with index management.
    
    Time: O(1) for all operations, Space: O(k)
    """
    
    def __init__(self, k: int):
        self.capacity = k
        self.deque = [None] * k
        self.front_idx = 0
        self.count = 0
    
    def insertFront(self, value: int) -> bool:
        if self.isFull():
            return False
        
        if self.count == 0:
            self.deque[self.front_idx] = value
        else:
            self.front_idx = (self.front_idx - 1 + self.capacity) % self.capacity
            self.deque[self.front_idx] = value
        
        self.count += 1
        return True
    
    def insertLast(self, value: int) -> bool:
        if self.isFull():
            return False
        
        if self.count == 0:
            self.deque[self.front_idx] = value
        else:
            rear_idx = (self.front_idx + self.count) % self.capacity
            self.deque[rear_idx] = value
        
        self.count += 1
        return True
    
    def deleteFront(self) -> bool:
        if self.isEmpty():
            return False
        
        self.deque[self.front_idx] = None
        if self.count > 1:
            self.front_idx = (self.front_idx + 1) % self.capacity
        
        self.count -= 1
        return True
    
    def deleteLast(self) -> bool:
        if self.isEmpty():
            return False
        
        rear_idx = (self.front_idx + self.count - 1) % self.capacity
        self.deque[rear_idx] = None
        self.count -= 1
        return True
    
    def getFront(self) -> int:
        if self.isEmpty():
            return -1
        return self.deque[self.front_idx]
    
    def getRear(self) -> int:
        if self.isEmpty():
            return -1
        rear_idx = (self.front_idx + self.count - 1) % self.capacity
        return self.deque[rear_idx]
    
    def isEmpty(self) -> bool:
        return self.count == 0
    
    def isFull(self) -> bool:
        return self.count == self.capacity


class MyCircularDeque5:
    """
    Approach 5: Collections.deque with Size Limit
    
    Use built-in deque with manual size management.
    
    Time: O(1) for all operations, Space: O(k)
    """
    
    def __init__(self, k: int):
        from collections import deque
        self.capacity = k
        self.deque = deque()
    
    def insertFront(self, value: int) -> bool:
        if self.isFull():
            return False
        
        self.deque.appendleft(value)
        return True
    
    def insertLast(self, value: int) -> bool:
        if self.isFull():
            return False
        
        self.deque.append(value)
        return True
    
    def deleteFront(self) -> bool:
        if self.isEmpty():
            return False
        
        self.deque.popleft()
        return True
    
    def deleteLast(self) -> bool:
        if self.isEmpty():
            return False
        
        self.deque.pop()
        return True
    
    def getFront(self) -> int:
        if self.isEmpty():
            return -1
        return self.deque[0]
    
    def getRear(self) -> int:
        if self.isEmpty():
            return -1
        return self.deque[-1]
    
    def isEmpty(self) -> bool:
        return len(self.deque) == 0
    
    def isFull(self) -> bool:
        return len(self.deque) == self.capacity


def test_circular_deque_implementations():
    """Test all circular deque implementations"""
    
    implementations = [
        ("Array with Size Counter", MyCircularDeque1),
        ("Array without Size Counter", MyCircularDeque2),
        ("Doubly Linked List", MyCircularDeque3),
        ("List-based Implementation", MyCircularDeque4),
        ("Collections.deque", MyCircularDeque5),
    ]
    
    def test_implementation(impl_class, impl_name):
        print(f"\n--- Testing {impl_name} ---")
        
        try:
            # Test with capacity 3
            deque = impl_class(3)
            
            operations = [
                ("isEmpty", None, True),
                ("insertLast", 1, True),
                ("insertLast", 2, True),
                ("insertFront", 3, True),
                ("isFull", None, True),
                ("insertFront", 4, False),  # Should fail
                ("getRear", None, 2),
                ("getFront", None, 3),
                ("deleteLast", None, True),
                ("insertFront", 4, True),
                ("getFront", None, 4),
                ("deleteFront", None, True),
                ("getFront", None, 1),
                ("deleteFront", None, True),
                ("deleteFront", None, True),
                ("isEmpty", None, True),
                ("deleteFront", None, False),  # Should fail
                ("getFront", None, -1),
                ("getRear", None, -1),
            ]
            
            for op_name, value, expected in operations:
                if op_name == "insertFront":
                    result = deque.insertFront(value)
                    status = "✓" if result == expected else "✗"
                    print(f"insertFront({value}) = {result} {status}")
                elif op_name == "insertLast":
                    result = deque.insertLast(value)
                    status = "✓" if result == expected else "✗"
                    print(f"insertLast({value}) = {result} {status}")
                elif op_name == "deleteFront":
                    result = deque.deleteFront()
                    status = "✓" if result == expected else "✗"
                    print(f"deleteFront() = {result} {status}")
                elif op_name == "deleteLast":
                    result = deque.deleteLast()
                    status = "✓" if result == expected else "✗"
                    print(f"deleteLast() = {result} {status}")
                elif op_name == "getFront":
                    result = deque.getFront()
                    status = "✓" if result == expected else "✗"
                    print(f"getFront() = {result} {status}")
                elif op_name == "getRear":
                    result = deque.getRear()
                    status = "✓" if result == expected else "✗"
                    print(f"getRear() = {result} {status}")
                elif op_name == "isEmpty":
                    result = deque.isEmpty()
                    status = "✓" if result == expected else "✗"
                    print(f"isEmpty() = {result} {status}")
                elif op_name == "isFull":
                    result = deque.isFull()
                    status = "✓" if result == expected else "✗"
                    print(f"isFull() = {result} {status}")
        
        except Exception as e:
            print(f"ERROR: {str(e)}")
    
    print("=== Testing Circular Deque Implementations ===")
    
    for impl_name, impl_class in implementations:
        test_implementation(impl_class, impl_name)


def demonstrate_deque_behavior():
    """Demonstrate deque double-ended behavior"""
    print("\n=== Deque Double-ended Behavior Demonstration ===")
    
    deque = MyCircularDeque1(4)
    
    print("Creating circular deque with capacity 4")
    
    operations = [
        ("insertLast", 1),
        ("insertLast", 2),
        ("insertFront", 3),
        ("insertFront", 4),
        ("getFront", None),
        ("getRear", None),
        ("deleteFront", None),
        ("deleteLast", None),
        ("getFront", None),
        ("getRear", None),
    ]
    
    for op, val in operations:
        if op == "insertFront":
            success = deque.insertFront(val)
            print(f"insertFront({val}) = {success} | Front: {deque.getFront()}, Rear: {deque.getRear()}")
        elif op == "insertLast":
            success = deque.insertLast(val)
            print(f"insertLast({val}) = {success} | Front: {deque.getFront()}, Rear: {deque.getRear()}")
        elif op == "deleteFront":
            success = deque.deleteFront()
            print(f"deleteFront() = {success} | Front: {deque.getFront()}, Rear: {deque.getRear()}")
        elif op == "deleteLast":
            success = deque.deleteLast()
            print(f"deleteLast() = {success} | Front: {deque.getFront()}, Rear: {deque.getRear()}")
        elif op == "getFront":
            front = deque.getFront()
            print(f"getFront() = {front}")
        elif op == "getRear":
            rear = deque.getRear()
            print(f"getRear() = {rear}")


def visualize_circular_array():
    """Visualize circular array operations"""
    print("\n=== Circular Array Visualization ===")
    
    deque = MyCircularDeque1(5)
    
    def show_state():
        print(f"Array: {deque.deque}")
        print(f"Front: {deque.front}, Rear: {deque.rear}, Size: {deque.size}")
        print(f"Front value: {deque.getFront()}, Rear value: {deque.getRear()}")
        print("-" * 50)
    
    print("Initial state:")
    show_state()
    
    operations = [
        ("insertLast", 1),
        ("insertLast", 2),
        ("insertFront", 3),
        ("insertFront", 4),
        ("deleteFront", None),
        ("insertLast", 5),
        ("deleteLast", None),
        ("insertFront", 6),
    ]
    
    for op, val in operations:
        if op == "insertFront":
            deque.insertFront(val)
            print(f"After insertFront({val}):")
        elif op == "insertLast":
            deque.insertLast(val)
            print(f"After insertLast({val}):")
        elif op == "deleteFront":
            deque.deleteFront()
            print("After deleteFront():")
        elif op == "deleteLast":
            deque.deleteLast()
            print("After deleteLast():")
        
        show_state()


def benchmark_circular_deque():
    """Benchmark different circular deque implementations"""
    import time
    import random
    
    implementations = [
        ("Array with Size Counter", MyCircularDeque1),
        ("Array without Size Counter", MyCircularDeque2),
        ("Doubly Linked List", MyCircularDeque3),
        ("Collections.deque", MyCircularDeque5),
    ]
    
    capacity = 1000
    n_operations = 10000
    
    print("\n=== Circular Deque Performance Benchmark ===")
    print(f"Capacity: {capacity}, Operations: {n_operations}")
    
    for impl_name, impl_class in implementations:
        deque = impl_class(capacity)
        
        start_time = time.time()
        
        # Perform random operations
        for _ in range(n_operations):
            operation = random.choice(['insertFront', 'insertLast', 'deleteFront', 'deleteLast'])
            
            try:
                if operation == 'insertFront':
                    deque.insertFront(random.randint(1, 1000))
                elif operation == 'insertLast':
                    deque.insertLast(random.randint(1, 1000))
                elif operation == 'deleteFront':
                    deque.deleteFront()
                elif operation == 'deleteLast':
                    deque.deleteLast()
            except:
                pass  # Ignore errors for benchmark
        
        end_time = time.time()
        
        print(f"{impl_name:30} | Time: {end_time - start_time:.4f}s")


def test_edge_cases():
    """Test edge cases for circular deque"""
    print("\n=== Testing Edge Cases ===")
    
    # Test with capacity 1
    deque = MyCircularDeque1(1)
    
    edge_cases = [
        ("Single capacity insertFront", lambda: deque.insertFront(42), True),
        ("Single capacity isFull", lambda: deque.isFull(), True),
        ("Single capacity insertLast fail", lambda: deque.insertLast(43), False),
        ("Single capacity getFront", lambda: deque.getFront(), 42),
        ("Single capacity getRear", lambda: deque.getRear(), 42),
        ("Single capacity deleteFront", lambda: deque.deleteFront(), True),
        ("Single capacity isEmpty", lambda: deque.isEmpty(), True),
        ("Empty deque getFront", lambda: deque.getFront(), -1),
        ("Empty deque getRear", lambda: deque.getRear(), -1),
    ]
    
    for description, operation, expected in edge_cases:
        try:
            result = operation()
            status = "✓" if result == expected else "✗"
            print(f"{description:30} | {status} | Expected: {expected}, Got: {result}")
        except Exception as e:
            print(f"{description:30} | ERROR: {str(e)[:30]}")


def test_wraparound_behavior():
    """Test wraparound behavior in circular deque"""
    print("\n=== Testing Wraparound Behavior ===")
    
    deque = MyCircularDeque1(3)
    
    print("Testing wraparound with capacity 3:")
    
    # Fill deque
    deque.insertLast(1)
    deque.insertLast(2)
    deque.insertLast(3)
    print(f"After filling: Front={deque.getFront()}, Rear={deque.getRear()}")
    
    # Remove from front and add to rear
    deque.deleteFront()
    deque.insertLast(4)
    print(f"After deleteFront + insertLast(4): Front={deque.getFront()}, Rear={deque.getRear()}")
    
    # Remove from rear and add to front
    deque.deleteLast()
    deque.insertFront(5)
    print(f"After deleteLast + insertFront(5): Front={deque.getFront()}, Rear={deque.getRear()}")


def memory_usage_analysis():
    """Analyze memory usage of different implementations"""
    print("\n=== Memory Usage Analysis ===")
    
    import sys
    
    capacity = 1000
    
    implementations = [
        ("Array with Size Counter", MyCircularDeque1),
        ("Array without Size Counter", MyCircularDeque2),
        ("Doubly Linked List", MyCircularDeque3),
        ("Collections.deque", MyCircularDeque5),
    ]
    
    for impl_name, impl_class in implementations:
        deque = impl_class(capacity)
        
        # Fill half the deque
        for i in range(capacity // 2):
            deque.insertLast(i)
        
        # Estimate memory usage
        memory_size = sys.getsizeof(deque)
        
        # Add size of internal data structures
        if hasattr(deque, 'deque'):
            memory_size += sys.getsizeof(deque.deque)
        
        print(f"{impl_name:30} | Memory: ~{memory_size} bytes")


def stress_test():
    """Stress test with many operations"""
    print("\n=== Stress Test ===")
    
    deque = MyCircularDeque1(100)
    
    operations_count = 10000
    successful_ops = 0
    
    print(f"Performing {operations_count} random operations...")
    
    import random
    
    for i in range(operations_count):
        op = random.choice(['insertFront', 'insertLast', 'deleteFront', 'deleteLast'])
        
        if op == 'insertFront':
            if deque.insertFront(random.randint(1, 1000)):
                successful_ops += 1
        elif op == 'insertLast':
            if deque.insertLast(random.randint(1, 1000)):
                successful_ops += 1
        elif op == 'deleteFront':
            if deque.deleteFront():
                successful_ops += 1
        elif op == 'deleteLast':
            if deque.deleteLast():
                successful_ops += 1
        
        if i % 1000 == 0:
            print(f"After {i+1} operations: {successful_ops} successful, deque size: {deque.size if hasattr(deque, 'size') else 'unknown'}")
    
    print(f"Stress test completed: {successful_ops}/{operations_count} operations successful")


if __name__ == "__main__":
    test_circular_deque_implementations()
    demonstrate_deque_behavior()
    visualize_circular_array()
    test_edge_cases()
    test_wraparound_behavior()
    benchmark_circular_deque()
    memory_usage_analysis()
    stress_test()

"""
Design Circular Deque demonstrates multiple implementation approaches
including array-based, doubly linked list, and built-in deque solutions
with comprehensive testing and performance analysis for double-ended operations.
"""
