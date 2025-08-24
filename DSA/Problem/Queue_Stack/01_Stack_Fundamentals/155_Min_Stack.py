"""
155. Min Stack - Multiple Approaches
Difficulty: Easy

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:
- MinStack() initializes the stack object.
- void push(int val) pushes the element val onto the stack.
- void pop() removes the element on the top of the stack.
- int top() gets the top element of the stack.
- int getMin() retrieves the minimum element in the stack.

You must implement a solution with O(1) time complexity for each function.
"""

from typing import List, Optional

class MinStack1:
    """
    Approach 1: Two Stacks - Main Stack + Min Stack
    
    Use separate stack to track minimum values.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val: int) -> None:
        self.stack.append(val)
        
        # Push to min_stack if it's empty or val is <= current minimum
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self) -> None:
        if self.stack:
            val = self.stack.pop()
            # Pop from min_stack if the popped value was the minimum
            if self.min_stack and val == self.min_stack[-1]:
                self.min_stack.pop()
    
    def top(self) -> int:
        return self.stack[-1] if self.stack else None
    
    def getMin(self) -> int:
        return self.min_stack[-1] if self.min_stack else None


class MinStack2:
    """
    Approach 2: Single Stack with Tuples
    
    Store (value, current_minimum) pairs in single stack.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    def __init__(self):
        self.stack = []  # Store (value, current_min) tuples
    
    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
        else:
            current_min = min(val, self.stack[-1][1])
            self.stack.append((val, current_min))
    
    def pop(self) -> None:
        if self.stack:
            self.stack.pop()
    
    def top(self) -> int:
        return self.stack[-1][0] if self.stack else None
    
    def getMin(self) -> int:
        return self.stack[-1][1] if self.stack else None


class MinStack3:
    """
    Approach 3: Difference Stack Approach
    
    Store differences from minimum to save space.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    def __init__(self):
        self.stack = []
        self.min_val = None
    
    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append(0)
            self.min_val = val
        else:
            diff = val - self.min_val
            self.stack.append(diff)
            if diff < 0:  # New minimum
                self.min_val = val
    
    def pop(self) -> None:
        if self.stack:
            diff = self.stack.pop()
            if diff < 0:  # We're popping the minimum
                self.min_val = self.min_val - diff
    
    def top(self) -> int:
        if not self.stack:
            return None
        
        diff = self.stack[-1]
        if diff < 0:
            return self.min_val
        else:
            return self.min_val + diff
    
    def getMin(self) -> int:
        return self.min_val


class MinStack4:
    """
    Approach 4: Linked List Implementation
    
    Use linked list with min tracking in each node.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    class Node:
        def __init__(self, val: int, min_val: int, next_node=None):
            self.val = val
            self.min_val = min_val
            self.next = next_node
    
    def __init__(self):
        self.head = None
    
    def push(self, val: int) -> None:
        if not self.head:
            self.head = self.Node(val, val)
        else:
            min_val = min(val, self.head.min_val)
            self.head = self.Node(val, min_val, self.head)
    
    def pop(self) -> None:
        if self.head:
            self.head = self.head.next
    
    def top(self) -> int:
        return self.head.val if self.head else None
    
    def getMin(self) -> int:
        return self.head.min_val if self.head else None


class MinStack5:
    """
    Approach 5: Optimized Min Stack (Space Efficient)
    
    Only store minimums when they change.
    
    Time: O(1) for all operations, Space: O(n) worst case, O(k) average where k is number of minimums
    """
    
    def __init__(self):
        self.stack = []
        self.min_stack = []  # Store (min_value, count) pairs
    
    def push(self, val: int) -> None:
        self.stack.append(val)
        
        if not self.min_stack or val < self.min_stack[-1][0]:
            self.min_stack.append((val, 1))
        elif val == self.min_stack[-1][0]:
            # Increment count of current minimum
            self.min_stack[-1] = (self.min_stack[-1][0], self.min_stack[-1][1] + 1)
    
    def pop(self) -> None:
        if self.stack:
            val = self.stack.pop()
            
            if self.min_stack and val == self.min_stack[-1][0]:
                if self.min_stack[-1][1] == 1:
                    self.min_stack.pop()
                else:
                    # Decrement count
                    self.min_stack[-1] = (self.min_stack[-1][0], self.min_stack[-1][1] - 1)
    
    def top(self) -> int:
        return self.stack[-1] if self.stack else None
    
    def getMin(self) -> int:
        return self.min_stack[-1][0] if self.min_stack else None


class MinStack6:
    """
    Approach 6: Single Variable Min Tracking
    
    Use single variable with stack manipulation for min tracking.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    def __init__(self):
        self.stack = []
        self.min_val = float('inf')
    
    def push(self, val: int) -> None:
        if val <= self.min_val:
            # Push old minimum before pushing new value
            self.stack.append(self.min_val)
            self.min_val = val
        self.stack.append(val)
    
    def pop(self) -> None:
        if self.stack:
            val = self.stack.pop()
            if val == self.min_val:
                # Restore previous minimum
                self.min_val = self.stack.pop() if self.stack else float('inf')
    
    def top(self) -> int:
        return self.stack[-1] if self.stack else None
    
    def getMin(self) -> int:
        return self.min_val if self.min_val != float('inf') else None


def test_min_stack():
    """Test all MinStack implementations"""
    
    implementations = [
        ("Two Stacks", MinStack1),
        ("Single Stack Tuples", MinStack2),
        ("Difference Stack", MinStack3),
        ("Linked List", MinStack4),
        ("Optimized Min Stack", MinStack5),
        ("Single Variable", MinStack6),
    ]
    
    test_operations = [
        ("push", -2),
        ("push", 0),
        ("push", -3),
        ("getMin", None, -3),
        ("pop", None),
        ("top", None, 0),
        ("getMin", None, -2),
    ]
    
    print("=== Testing Min Stack Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- Testing {impl_name} ---")
        
        try:
            min_stack = impl_class()
            
            for operation, *args in test_operations:
                if operation == "push":
                    min_stack.push(args[0])
                    print(f"push({args[0]})")
                elif operation == "pop":
                    min_stack.pop()
                    print("pop()")
                elif operation == "top":
                    result = min_stack.top()
                    expected = args[0] if args else None
                    status = "✓" if result == expected else "✗"
                    print(f"top() = {result} {status}")
                elif operation == "getMin":
                    result = min_stack.getMin()
                    expected = args[0] if args else None
                    status = "✓" if result == expected else "✗"
                    print(f"getMin() = {result} {status}")
        
        except Exception as e:
            print(f"ERROR: {str(e)}")


def benchmark_min_stack():
    """Benchmark different MinStack implementations"""
    import time
    import random
    
    implementations = [
        ("Two Stacks", MinStack1),
        ("Single Stack Tuples", MinStack2),
        ("Optimized Min Stack", MinStack5),
    ]
    
    n = 10000
    operations = []
    
    # Generate random operations
    for _ in range(n):
        op_type = random.choice(['push', 'pop', 'top', 'getMin'])
        if op_type == 'push':
            operations.append(('push', random.randint(-1000, 1000)))
        else:
            operations.append((op_type,))
    
    print("\n=== MinStack Performance Benchmark ===")
    print(f"Operations: {n}")
    
    for impl_name, impl_class in implementations:
        min_stack = impl_class()
        
        start_time = time.time()
        
        for operation in operations:
            try:
                if operation[0] == 'push':
                    min_stack.push(operation[1])
                elif operation[0] == 'pop':
                    min_stack.pop()
                elif operation[0] == 'top':
                    min_stack.top()
                elif operation[0] == 'getMin':
                    min_stack.getMin()
            except:
                pass  # Ignore errors for benchmark
        
        end_time = time.time()
        
        print(f"{impl_name:20} | Time: {end_time - start_time:.4f}s")


if __name__ == "__main__":
    test_min_stack()
    benchmark_min_stack()

"""
Min Stack demonstrates multiple approaches to maintain minimum
element in constant time, including space optimization techniques,
linked list implementations, and mathematical approaches.
"""
