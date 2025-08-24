"""
Stack Basic Implementation - Multiple Approaches
Difficulty: Easy

Implement a stack data structure with basic operations:
- push(x): Add element x to the top of the stack
- pop(): Remove and return the top element
- peek()/top(): Return the top element without removing it
- isEmpty(): Check if the stack is empty
- size(): Return the number of elements in the stack
"""

from typing import List, Optional, Any
import threading

class ArrayStack:
    """
    Approach 1: Array-based Stack Implementation
    
    Use dynamic array (list) for stack implementation.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    def __init__(self, capacity: int = None):
        self.items = []
        self.capacity = capacity
    
    def push(self, item: Any) -> bool:
        """Push item to top of stack"""
        if self.capacity and len(self.items) >= self.capacity:
            raise OverflowError("Stack overflow")
        
        self.items.append(item)
        return True
    
    def pop(self) -> Any:
        """Remove and return top item"""
        if self.isEmpty():
            raise IndexError("Stack underflow")
        
        return self.items.pop()
    
    def peek(self) -> Any:
        """Return top item without removing"""
        if self.isEmpty():
            raise IndexError("Stack is empty")
        
        return self.items[-1]
    
    def isEmpty(self) -> bool:
        """Check if stack is empty"""
        return len(self.items) == 0
    
    def size(self) -> int:
        """Return number of items in stack"""
        return len(self.items)
    
    def __str__(self) -> str:
        return f"Stack({self.items})"


class LinkedListStack:
    """
    Approach 2: Linked List-based Stack Implementation
    
    Use linked list for dynamic stack implementation.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    class Node:
        def __init__(self, data: Any, next_node=None):
            self.data = data
            self.next = next_node
    
    def __init__(self):
        self.head = None
        self._size = 0
    
    def push(self, item: Any) -> bool:
        """Push item to top of stack"""
        new_node = self.Node(item, self.head)
        self.head = new_node
        self._size += 1
        return True
    
    def pop(self) -> Any:
        """Remove and return top item"""
        if self.isEmpty():
            raise IndexError("Stack underflow")
        
        data = self.head.data
        self.head = self.head.next
        self._size -= 1
        return data
    
    def peek(self) -> Any:
        """Return top item without removing"""
        if self.isEmpty():
            raise IndexError("Stack is empty")
        
        return self.head.data
    
    def isEmpty(self) -> bool:
        """Check if stack is empty"""
        return self.head is None
    
    def size(self) -> int:
        """Return number of items in stack"""
        return self._size
    
    def __str__(self) -> str:
        items = []
        current = self.head
        while current:
            items.append(current.data)
            current = current.next
        return f"LinkedStack({items})"


class FixedArrayStack:
    """
    Approach 3: Fixed-size Array Stack
    
    Use fixed-size array with explicit capacity management.
    
    Time: O(1) for all operations, Space: O(capacity)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items = [None] * capacity
        self.top_index = -1
    
    def push(self, item: Any) -> bool:
        """Push item to top of stack"""
        if self.top_index >= self.capacity - 1:
            raise OverflowError("Stack overflow")
        
        self.top_index += 1
        self.items[self.top_index] = item
        return True
    
    def pop(self) -> Any:
        """Remove and return top item"""
        if self.isEmpty():
            raise IndexError("Stack underflow")
        
        item = self.items[self.top_index]
        self.items[self.top_index] = None  # Clear reference
        self.top_index -= 1
        return item
    
    def peek(self) -> Any:
        """Return top item without removing"""
        if self.isEmpty():
            raise IndexError("Stack is empty")
        
        return self.items[self.top_index]
    
    def isEmpty(self) -> bool:
        """Check if stack is empty"""
        return self.top_index == -1
    
    def size(self) -> int:
        """Return number of items in stack"""
        return self.top_index + 1
    
    def isFull(self) -> bool:
        """Check if stack is full"""
        return self.top_index >= self.capacity - 1
    
    def __str__(self) -> str:
        items = self.items[:self.top_index + 1]
        return f"FixedStack({items})"


class ThreadSafeStack:
    """
    Approach 4: Thread-safe Stack Implementation
    
    Use locks to ensure thread safety.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    def __init__(self, capacity: int = None):
        self.items = []
        self.capacity = capacity
        self.lock = threading.Lock()
    
    def push(self, item: Any) -> bool:
        """Thread-safe push operation"""
        with self.lock:
            if self.capacity and len(self.items) >= self.capacity:
                raise OverflowError("Stack overflow")
            
            self.items.append(item)
            return True
    
    def pop(self) -> Any:
        """Thread-safe pop operation"""
        with self.lock:
            if len(self.items) == 0:
                raise IndexError("Stack underflow")
            
            return self.items.pop()
    
    def peek(self) -> Any:
        """Thread-safe peek operation"""
        with self.lock:
            if len(self.items) == 0:
                raise IndexError("Stack is empty")
            
            return self.items[-1]
    
    def isEmpty(self) -> bool:
        """Thread-safe isEmpty check"""
        with self.lock:
            return len(self.items) == 0
    
    def size(self) -> int:
        """Thread-safe size check"""
        with self.lock:
            return len(self.items)


class MinMaxStack:
    """
    Approach 5: Stack with Min/Max Tracking
    
    Track minimum and maximum elements efficiently.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    def __init__(self):
        self.items = []
        self.min_stack = []
        self.max_stack = []
    
    def push(self, item: Any) -> bool:
        """Push item and update min/max"""
        self.items.append(item)
        
        # Update min stack
        if not self.min_stack or item <= self.min_stack[-1]:
            self.min_stack.append(item)
        
        # Update max stack
        if not self.max_stack or item >= self.max_stack[-1]:
            self.max_stack.append(item)
        
        return True
    
    def pop(self) -> Any:
        """Pop item and update min/max"""
        if self.isEmpty():
            raise IndexError("Stack underflow")
        
        item = self.items.pop()
        
        # Update min stack
        if self.min_stack and item == self.min_stack[-1]:
            self.min_stack.pop()
        
        # Update max stack
        if self.max_stack and item == self.max_stack[-1]:
            self.max_stack.pop()
        
        return item
    
    def peek(self) -> Any:
        """Return top item"""
        if self.isEmpty():
            raise IndexError("Stack is empty")
        
        return self.items[-1]
    
    def getMin(self) -> Any:
        """Get minimum element in O(1)"""
        if not self.min_stack:
            raise IndexError("Stack is empty")
        
        return self.min_stack[-1]
    
    def getMax(self) -> Any:
        """Get maximum element in O(1)"""
        if not self.max_stack:
            raise IndexError("Stack is empty")
        
        return self.max_stack[-1]
    
    def isEmpty(self) -> bool:
        return len(self.items) == 0
    
    def size(self) -> int:
        return len(self.items)


class CircularStack:
    """
    Approach 6: Circular Array Stack
    
    Use circular array for memory-efficient implementation.
    
    Time: O(1) for all operations, Space: O(capacity)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items = [None] * capacity
        self.top = -1
        self.count = 0
    
    def push(self, item: Any) -> bool:
        """Push item in circular manner"""
        if self.count >= self.capacity:
            raise OverflowError("Stack overflow")
        
        self.top = (self.top + 1) % self.capacity
        self.items[self.top] = item
        self.count += 1
        return True
    
    def pop(self) -> Any:
        """Pop item in circular manner"""
        if self.isEmpty():
            raise IndexError("Stack underflow")
        
        item = self.items[self.top]
        self.items[self.top] = None
        self.top = (self.top - 1) % self.capacity
        self.count -= 1
        return item
    
    def peek(self) -> Any:
        """Return top item"""
        if self.isEmpty():
            raise IndexError("Stack is empty")
        
        return self.items[self.top]
    
    def isEmpty(self) -> bool:
        return self.count == 0
    
    def size(self) -> int:
        return self.count
    
    def isFull(self) -> bool:
        return self.count >= self.capacity


def test_stack_implementations():
    """Test all stack implementations"""
    
    implementations = [
        ("Array Stack", ArrayStack),
        ("Linked List Stack", LinkedListStack),
        ("Fixed Array Stack", lambda: FixedArrayStack(10)),
        ("Thread Safe Stack", ThreadSafeStack),
        ("Min/Max Stack", MinMaxStack),
        ("Circular Stack", lambda: CircularStack(10)),
    ]
    
    test_operations = [
        ("push", 10),
        ("push", 20),
        ("push", 5),
        ("peek", None, 5),
        ("size", None, 3),
        ("pop", None, 5),
        ("peek", None, 20),
        ("push", 30),
        ("size", None, 3),
        ("isEmpty", None, False),
        ("pop", None, 30),
        ("pop", None, 20),
        ("pop", None, 10),
        ("isEmpty", None, True),
    ]
    
    print("=== Testing Stack Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- Testing {impl_name} ---")
        
        try:
            stack = impl_class()
            
            for operation in test_operations:
                if operation[0] == "push":
                    stack.push(operation[1])
                    print(f"push({operation[1]})")
                elif operation[0] == "pop":
                    result = stack.pop()
                    expected = operation[2]
                    status = "✓" if result == expected else "✗"
                    print(f"pop() = {result} {status}")
                elif operation[0] == "peek":
                    result = stack.peek()
                    expected = operation[2]
                    status = "✓" if result == expected else "✗"
                    print(f"peek() = {result} {status}")
                elif operation[0] == "size":
                    result = stack.size()
                    expected = operation[2]
                    status = "✓" if result == expected else "✗"
                    print(f"size() = {result} {status}")
                elif operation[0] == "isEmpty":
                    result = stack.isEmpty()
                    expected = operation[2]
                    status = "✓" if result == expected else "✗"
                    print(f"isEmpty() = {result} {status}")
        
        except Exception as e:
            print(f"ERROR: {str(e)}")


def test_min_max_stack():
    """Test MinMaxStack specific functionality"""
    print("\n=== Testing Min/Max Stack ===")
    
    stack = MinMaxStack()
    
    operations = [
        ("push", 5),
        ("push", 2),
        ("push", 8),
        ("push", 1),
        ("push", 9),
    ]
    
    for op, val in operations:
        stack.push(val)
        print(f"push({val}) | min: {stack.getMin()}, max: {stack.getMax()}")
    
    print("\nPopping elements:")
    while not stack.isEmpty():
        top = stack.peek()
        min_val = stack.getMin()
        max_val = stack.getMax()
        popped = stack.pop()
        print(f"pop() = {popped} | was min: {min_val}, max: {max_val}")


def benchmark_stack_implementations():
    """Benchmark different stack implementations"""
    import time
    
    implementations = [
        ("Array Stack", ArrayStack),
        ("Linked List Stack", LinkedListStack),
        ("Fixed Array Stack", lambda: FixedArrayStack(20000)),
    ]
    
    n = 10000
    
    print("\n=== Stack Implementation Performance Benchmark ===")
    print(f"Operations: {n} push + {n} pop")
    
    for impl_name, impl_class in implementations:
        stack = impl_class()
        
        start_time = time.time()
        
        # Push operations
        for i in range(n):
            stack.push(i)
        
        # Pop operations
        for i in range(n):
            stack.pop()
        
        end_time = time.time()
        
        print(f"{impl_name:20} | Time: {end_time - start_time:.4f}s")


if __name__ == "__main__":
    test_stack_implementations()
    test_min_max_stack()
    benchmark_stack_implementations()

"""
Stack Basic Implementation demonstrates various stack implementation
approaches including array-based, linked list, fixed-size, thread-safe,
min/max tracking, and circular buffer implementations.
"""
