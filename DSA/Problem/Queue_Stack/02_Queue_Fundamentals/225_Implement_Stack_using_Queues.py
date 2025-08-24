"""
225. Implement Stack using Queues - Multiple Approaches
Difficulty: Easy

Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (push, top, pop, and empty).

Implement the MyStack class:
- void push(int x) Pushes element x to the top of the stack.
- int pop() Removes the element on the top of the stack and returns it.
- int top() Returns the element on the top of the stack.
- boolean empty() Returns true if the stack is empty, false otherwise.
"""

from collections import deque
from typing import List, Optional

class MyStack1:
    """
    Approach 1: Two Queues - Push Expensive
    
    Make push operation expensive by moving all elements.
    
    Time: push O(n), pop/top/empty O(1), Space: O(n)
    """
    
    def __init__(self):
        self.q1 = deque()  # Main queue
        self.q2 = deque()  # Auxiliary queue
    
    def push(self, x: int) -> None:
        # Add new element to q2
        self.q2.append(x)
        
        # Move all elements from q1 to q2
        while self.q1:
            self.q2.append(self.q1.popleft())
        
        # Swap q1 and q2
        self.q1, self.q2 = self.q2, self.q1
    
    def pop(self) -> int:
        if self.q1:
            return self.q1.popleft()
        return -1
    
    def top(self) -> int:
        if self.q1:
            return self.q1[0]
        return -1
    
    def empty(self) -> bool:
        return len(self.q1) == 0


class MyStack2:
    """
    Approach 2: Two Queues - Pop Expensive
    
    Make pop/top operations expensive.
    
    Time: push O(1), pop/top O(n), Space: O(n)
    """
    
    def __init__(self):
        self.q1 = deque()  # Main queue
        self.q2 = deque()  # Auxiliary queue
    
    def push(self, x: int) -> None:
        self.q1.append(x)
    
    def pop(self) -> int:
        if not self.q1:
            return -1
        
        # Move all but last element to q2
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        
        # Get the last element (top of stack)
        result = self.q1.popleft()
        
        # Swap queues
        self.q1, self.q2 = self.q2, self.q1
        
        return result
    
    def top(self) -> int:
        if not self.q1:
            return -1
        
        # Move all but last element to q2
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        
        # Get the last element (top of stack)
        result = self.q1[0]
        
        # Move the last element to q2 as well
        self.q2.append(self.q1.popleft())
        
        # Swap queues
        self.q1, self.q2 = self.q2, self.q1
        
        return result
    
    def empty(self) -> bool:
        return len(self.q1) == 0


class MyStack3:
    """
    Approach 3: Single Queue Implementation
    
    Use single queue with rotation technique.
    
    Time: push O(n), pop/top/empty O(1), Space: O(n)
    """
    
    def __init__(self):
        self.queue = deque()
    
    def push(self, x: int) -> None:
        size = len(self.queue)
        self.queue.append(x)
        
        # Rotate queue to bring new element to front
        for _ in range(size):
            self.queue.append(self.queue.popleft())
    
    def pop(self) -> int:
        if self.queue:
            return self.queue.popleft()
        return -1
    
    def top(self) -> int:
        if self.queue:
            return self.queue[0]
        return -1
    
    def empty(self) -> bool:
        return len(self.queue) == 0


class MyStack4:
    """
    Approach 4: List-based Queue Implementation
    
    Use Python list to simulate queue behavior.
    
    Time: push O(n), pop/top O(1), Space: O(n)
    """
    
    def __init__(self):
        self.queue = []
    
    def push(self, x: int) -> None:
        self.queue.append(x)
        
        # Rotate to bring new element to front
        for _ in range(len(self.queue) - 1):
            self.queue.append(self.queue.pop(0))
    
    def pop(self) -> int:
        if self.queue:
            return self.queue.pop(0)
        return -1
    
    def top(self) -> int:
        if self.queue:
            return self.queue[0]
        return -1
    
    def empty(self) -> bool:
        return len(self.queue) == 0


class MyStack5:
    """
    Approach 5: Optimized Two Queues with Size Tracking
    
    Track sizes to optimize operations.
    
    Time: push O(n), pop/top O(1), Space: O(n)
    """
    
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()
        self.size = 0
    
    def push(self, x: int) -> None:
        self.q2.append(x)
        
        # Move all elements from q1 to q2
        while self.q1:
            self.q2.append(self.q1.popleft())
        
        # Swap queues
        self.q1, self.q2 = self.q2, self.q1
        self.size += 1
    
    def pop(self) -> int:
        if self.size == 0:
            return -1
        
        self.size -= 1
        return self.q1.popleft()
    
    def top(self) -> int:
        if self.size == 0:
            return -1
        
        return self.q1[0]
    
    def empty(self) -> bool:
        return self.size == 0


class MyStack6:
    """
    Approach 6: Recursive Queue Implementation
    
    Use recursion to simulate stack behavior.
    
    Time: push O(n), pop O(1), Space: O(n)
    """
    
    def __init__(self):
        self.queue = deque()
    
    def push(self, x: int) -> None:
        self._push_recursive(x, len(self.queue))
    
    def _push_recursive(self, x: int, size: int) -> None:
        """Recursively push element to maintain stack order"""
        self.queue.append(x)
        
        if size > 0:
            # Move front element to back
            front = self.queue.popleft()
            self._push_recursive(front, size - 1)
    
    def pop(self) -> int:
        if self.queue:
            return self.queue.popleft()
        return -1
    
    def top(self) -> int:
        if self.queue:
            return self.queue[0]
        return -1
    
    def empty(self) -> bool:
        return len(self.queue) == 0


def test_stack_implementations():
    """Test all stack implementations"""
    
    implementations = [
        ("Two Queues - Push Expensive", MyStack1),
        ("Two Queues - Pop Expensive", MyStack2),
        ("Single Queue", MyStack3),
        ("List-based Queue", MyStack4),
        ("Optimized Two Queues", MyStack5),
        ("Recursive Queue", MyStack6),
    ]
    
    test_operations = [
        ("push", 1),
        ("push", 2),
        ("top", None, 2),
        ("pop", None, 2),
        ("empty", None, False),
        ("push", 3),
        ("push", 4),
        ("pop", None, 4),
        ("top", None, 3),
        ("pop", None, 3),
        ("pop", None, 1),
        ("empty", None, True),
    ]
    
    print("=== Testing Stack using Queues Implementations ===")
    
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
                elif operation[0] == "top":
                    result = stack.top()
                    expected = operation[2]
                    status = "✓" if result == expected else "✗"
                    print(f"top() = {result} {status}")
                elif operation[0] == "empty":
                    result = stack.empty()
                    expected = operation[2]
                    status = "✓" if result == expected else "✗"
                    print(f"empty() = {result} {status}")
        
        except Exception as e:
            print(f"ERROR: {str(e)}")


def demonstrate_stack_behavior():
    """Demonstrate LIFO stack behavior using queues"""
    print("\n=== Stack LIFO Behavior Demonstration ===")
    
    stack = MyStack1()  # Using push expensive implementation
    
    print("Creating stack and adding elements: 1, 2, 3, 4")
    for i in range(1, 5):
        stack.push(i)
        print(f"push({i}) -> top: {stack.top()}")
    
    print("\nRemoving elements (should be in LIFO order):")
    while not stack.empty():
        top_elem = stack.top()
        popped = stack.pop()
        print(f"top() = {top_elem}, pop() = {popped}")


def visualize_push_expensive():
    """Visualize push expensive approach step by step"""
    print("\n=== Push Expensive Approach Visualization ===")
    
    stack = MyStack1()
    
    elements = [1, 2, 3]
    
    for elem in elements:
        print(f"\nPushing {elem}:")
        print(f"Before: q1={list(stack.q1)}, q2={list(stack.q2)}")
        
        stack.push(elem)
        
        print(f"After:  q1={list(stack.q1)}, q2={list(stack.q2)}")
        print(f"Top element: {stack.top()}")


def benchmark_stack_implementations():
    """Benchmark different stack implementations"""
    import time
    import random
    
    implementations = [
        ("Two Queues - Push Expensive", MyStack1),
        ("Single Queue", MyStack3),
        ("Optimized Two Queues", MyStack5),
    ]
    
    n = 1000
    operations = []
    
    # Generate random operations
    for _ in range(n):
        op_type = random.choice(['push', 'pop', 'top', 'empty'])
        if op_type == 'push':
            operations.append(('push', random.randint(1, 1000)))
        else:
            operations.append((op_type,))
    
    print("\n=== Stack Implementation Performance Benchmark ===")
    print(f"Operations: {n}")
    
    for impl_name, impl_class in implementations:
        stack = impl_class()
        
        start_time = time.time()
        
        for operation in operations:
            try:
                if operation[0] == 'push':
                    stack.push(operation[1])
                elif operation[0] == 'pop':
                    stack.pop()
                elif operation[0] == 'top':
                    stack.top()
                elif operation[0] == 'empty':
                    stack.empty()
            except:
                pass  # Ignore errors for benchmark
        
        end_time = time.time()
        
        print(f"{impl_name:25} | Time: {end_time - start_time:.4f}s")


def compare_approaches():
    """Compare different approaches with complexity analysis"""
    print("\n=== Approach Comparison ===")
    
    approaches = [
        ("Push Expensive", "O(n)", "O(1)", "O(1)", "O(1)"),
        ("Pop Expensive", "O(1)", "O(n)", "O(n)", "O(1)"),
        ("Single Queue", "O(n)", "O(1)", "O(1)", "O(1)"),
        ("List-based", "O(n)", "O(1)", "O(1)", "O(1)"),
        ("Optimized Two Queues", "O(n)", "O(1)", "O(1)", "O(1)"),
        ("Recursive", "O(n)", "O(1)", "O(1)", "O(1)"),
    ]
    
    print(f"{'Approach':<25} | {'Push':<8} | {'Pop':<8} | {'Top':<8} | {'Empty':<8}")
    print("-" * 70)
    
    for name, push_time, pop_time, top_time, empty_time in approaches:
        print(f"{name:<25} | {push_time:<8} | {pop_time:<8} | {top_time:<8} | {empty_time:<8}")


def test_edge_cases():
    """Test edge cases for stack implementations"""
    print("\n=== Testing Edge Cases ===")
    
    stack = MyStack1()
    
    edge_cases = [
        ("Empty stack pop", lambda: stack.pop(), -1),
        ("Empty stack top", lambda: stack.top(), -1),
        ("Empty stack check", lambda: stack.empty(), True),
        ("Single element", lambda: (stack.push(42), stack.top())[1], 42),
        ("Single element pop", lambda: stack.pop(), 42),
        ("After pop empty", lambda: stack.empty(), True),
    ]
    
    for description, operation, expected in edge_cases:
        try:
            result = operation()
            status = "✓" if result == expected else "✗"
            print(f"{description:20} | {status} | Expected: {expected}, Got: {result}")
        except Exception as e:
            print(f"{description:20} | ERROR: {str(e)[:30]}")


def memory_usage_analysis():
    """Analyze memory usage of different implementations"""
    print("\n=== Memory Usage Analysis ===")
    
    import sys
    
    implementations = [
        ("Two Queues - Push Expensive", MyStack1),
        ("Single Queue", MyStack3),
        ("List-based Queue", MyStack4),
    ]
    
    n_elements = 1000
    
    for impl_name, impl_class in implementations:
        stack = impl_class()
        
        # Add elements
        for i in range(n_elements):
            stack.push(i)
        
        # Estimate memory usage
        memory_size = sys.getsizeof(stack)
        if hasattr(stack, 'q1'):
            memory_size += sys.getsizeof(stack.q1)
        if hasattr(stack, 'q2'):
            memory_size += sys.getsizeof(stack.q2)
        if hasattr(stack, 'queue'):
            memory_size += sys.getsizeof(stack.queue)
        
        print(f"{impl_name:25} | Memory: ~{memory_size} bytes")


if __name__ == "__main__":
    test_stack_implementations()
    demonstrate_stack_behavior()
    visualize_push_expensive()
    compare_approaches()
    test_edge_cases()
    benchmark_stack_implementations()
    memory_usage_analysis()

"""
Stack using Queues demonstrates multiple implementation strategies
including push-expensive, pop-expensive, single queue, and optimized
approaches with complexity analysis and performance comparisons.
"""
