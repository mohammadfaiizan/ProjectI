"""
1381. Design a Stack With Increment Operation - Multiple Approaches
Difficulty: Medium

Design a stack which supports the following operations.

Implement the CustomStack class:
- CustomStack(int maxSize) Initializes the object with maxSize which is the maximum number of elements in the stack or do nothing if the stack reached the maxSize.
- void push(int x) Adds x to the top of the stack if the stack hasn't reached the maxSize.
- int pop() Pops and returns the top of stack or -1 if the stack is empty.
- void inc(int k, int val) Increments the bottom k elements of the stack by val. If there are less than k elements in the stack, just increment all the elements in the stack.
"""

from typing import List

class CustomStackArray:
    """
    Approach 1: Array Implementation (Optimal)
    
    Use array with lazy increment propagation.
    
    Time: O(1) for all operations, Space: O(maxSize)
    """
    
    def __init__(self, maxSize: int):
        self.maxSize = maxSize
        self.stack = []
        self.increments = []  # Lazy increment array
    
    def push(self, x: int) -> None:
        """Push element to stack"""
        if len(self.stack) < self.maxSize:
            self.stack.append(x)
            self.increments.append(0)
    
    def pop(self) -> int:
        """Pop element from stack"""
        if not self.stack:
            return -1
        
        # Apply lazy increment
        val = self.stack.pop()
        inc = self.increments.pop()
        
        # Propagate increment to element below
        if self.increments:
            self.increments[-1] += inc
        
        return val + inc
    
    def increment(self, k: int, val: int) -> None:
        """Increment bottom k elements by val"""
        if self.increments:
            idx = min(k - 1, len(self.increments) - 1)
            self.increments[idx] += val


class CustomStackList:
    """
    Approach 2: List Implementation
    
    Use list with direct increment operations.
    
    Time: O(k) for increment, O(1) for push/pop, Space: O(maxSize)
    """
    
    def __init__(self, maxSize: int):
        self.maxSize = maxSize
        self.stack = []
    
    def push(self, x: int) -> None:
        """Push element to stack"""
        if len(self.stack) < self.maxSize:
            self.stack.append(x)
    
    def pop(self) -> int:
        """Pop element from stack"""
        if not self.stack:
            return -1
        return self.stack.pop()
    
    def increment(self, k: int, val: int) -> None:
        """Increment bottom k elements by val"""
        for i in range(min(k, len(self.stack))):
            self.stack[i] += val


class CustomStackDeque:
    """
    Approach 3: Deque Implementation
    
    Use deque for stack operations.
    
    Time: O(k) for increment, O(1) for push/pop, Space: O(maxSize)
    """
    
    def __init__(self, maxSize: int):
        from collections import deque
        self.maxSize = maxSize
        self.stack = deque()
    
    def push(self, x: int) -> None:
        """Push element to stack"""
        if len(self.stack) < self.maxSize:
            self.stack.append(x)
    
    def pop(self) -> int:
        """Pop element from stack"""
        if not self.stack:
            return -1
        return self.stack.pop()
    
    def increment(self, k: int, val: int) -> None:
        """Increment bottom k elements by val"""
        temp = []
        
        # Pop all elements
        while self.stack:
            temp.append(self.stack.pop())
        
        # Increment bottom k elements
        for i in range(len(temp)):
            if i >= len(temp) - k:  # Bottom k elements
                temp[i] += val
        
        # Push back in reverse order
        for i in range(len(temp) - 1, -1, -1):
            self.stack.append(temp[i])


class CustomStackLinkedList:
    """
    Approach 4: Linked List Implementation
    
    Use linked list for stack operations.
    
    Time: O(k) for increment, O(1) for push/pop, Space: O(maxSize)
    """
    
    class Node:
        def __init__(self, val: int):
            self.val = val
            self.next = None
    
    def __init__(self, maxSize: int):
        self.maxSize = maxSize
        self.head = None
        self.size = 0
    
    def push(self, x: int) -> None:
        """Push element to stack"""
        if self.size < self.maxSize:
            new_node = self.Node(x)
            new_node.next = self.head
            self.head = new_node
            self.size += 1
    
    def pop(self) -> int:
        """Pop element from stack"""
        if not self.head:
            return -1
        
        val = self.head.val
        self.head = self.head.next
        self.size -= 1
        return val
    
    def increment(self, k: int, val: int) -> None:
        """Increment bottom k elements by val"""
        # Convert to array for easier access
        elements = []
        current = self.head
        
        while current:
            elements.append(current.val)
            current = current.next
        
        # Increment bottom k elements
        elements.reverse()  # Bottom to top
        for i in range(min(k, len(elements))):
            elements[i] += val
        elements.reverse()  # Back to top to bottom
        
        # Rebuild linked list
        self.head = None
        self.size = 0
        
        for val in elements:
            self.push(val)


def test_custom_stack_implementations():
    """Test custom stack implementations"""
    
    implementations = [
        ("Array (Optimal)", CustomStackArray),
        ("List", CustomStackList),
        ("Deque", CustomStackDeque),
        ("Linked List", CustomStackLinkedList),
    ]
    
    test_cases = [
        {
            "maxSize": 3,
            "operations": ["push", "push", "pop", "push", "push", "push", "increment", "increment", "pop", "pop", "pop", "pop"],
            "values": [1, 2, None, 2, 3, 4, (5, 100), (2, 100), None, None, None, None],
            "expected": [None, None, 2, None, None, None, None, None, 103, 202, 201, -1],
            "description": "Example 1"
        },
        {
            "maxSize": 2,
            "operations": ["push", "push", "push", "pop", "increment", "pop", "pop"],
            "values": [1, 2, 3, None, (2, 10), None, None],
            "expected": [None, None, None, 2, None, 11, 11],
            "description": "Max size limit"
        },
        {
            "maxSize": 1,
            "operations": ["pop", "push", "increment", "pop"],
            "values": [None, 5, (1, 3), None],
            "expected": [-1, None, None, 8],
            "description": "Single element stack"
        },
    ]
    
    print("=== Testing Custom Stack Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- {impl_name} Implementation ---")
        
        for test_case in test_cases:
            try:
                stack = impl_class(test_case["maxSize"])
                results = []
                
                for i, op in enumerate(test_case["operations"]):
                    if op == "push":
                        stack.push(test_case["values"][i])
                        results.append(None)
                    elif op == "pop":
                        result = stack.pop()
                        results.append(result)
                    elif op == "increment":
                        k, val = test_case["values"][i]
                        stack.increment(k, val)
                        results.append(None)
                
                expected = test_case["expected"]
                status = "✓" if results == expected else "✗"
                
                print(f"  {test_case['description']:20} | {status} | {results}")
                if results != expected:
                    print(f"    Expected: {expected}")
                
            except Exception as e:
                print(f"  {test_case['description']:20} | ERROR: {str(e)[:40]}")


def demonstrate_lazy_increment():
    """Demonstrate lazy increment approach"""
    print("\n=== Lazy Increment Approach Demonstration ===")
    
    stack = CustomStackArray(4)
    
    operations = [
        ("push", 1),
        ("push", 2),
        ("push", 3),
        ("increment", (2, 10)),
        ("pop", None),
        ("increment", (3, 5)),
        ("pop", None),
        ("pop", None),
    ]
    
    print("Strategy: Use lazy increment array to defer increment operations")
    
    for op, val in operations:
        print(f"\nOperation: {op}({val if val else ''})")
        print(f"  Before: stack = {stack.stack}, increments = {stack.increments}")
        
        if op == "push":
            stack.push(val)
        elif op == "pop":
            result = stack.pop()
            print(f"  Popped: {result}")
        elif op == "increment":
            k, inc_val = val
            stack.increment(k, inc_val)
        
        print(f"  After:  stack = {stack.stack}, increments = {stack.increments}")


def visualize_increment_operations():
    """Visualize increment operations"""
    print("\n=== Increment Operations Visualization ===")
    
    stack = CustomStackList(5)
    
    # Build stack
    elements = [1, 2, 3, 4, 5]
    for elem in elements:
        stack.push(elem)
    
    print(f"Initial stack: {stack.stack}")
    print("Stack visualization (bottom to top):")
    for i, val in enumerate(stack.stack):
        print(f"  Index {i}: {val}")
    
    # Test different increment operations
    increment_tests = [
        (2, 10, "Increment bottom 2 elements by 10"),
        (4, 5, "Increment bottom 4 elements by 5"),
        (10, 1, "Increment all elements by 1 (k > stack size)"),
    ]
    
    for k, val, description in increment_tests:
        print(f"\n{description}:")
        print(f"  Before: {stack.stack}")
        stack.increment(k, val)
        print(f"  After:  {stack.stack}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Undo system with batch operations
    print("1. Undo System with Batch Operations:")
    undo_stack = CustomStackArray(10)
    
    # Simulate document editing operations
    operations = [
        ("Type 'A'", 1),
        ("Type 'B'", 2),
        ("Type 'C'", 3),
        ("Batch indent", (3, 10)),  # Add 10 to represent indentation
    ]
    
    for desc, op in operations:
        if isinstance(op, tuple):
            k, val = op
            undo_stack.increment(k, val)
            print(f"  {desc}: increment bottom {k} operations by {val}")
        else:
            undo_stack.push(op)
            print(f"  {desc}: operation {op}")
    
    print("  Undo operations:")
    while True:
        result = undo_stack.pop()
        if result == -1:
            break
        print(f"    Undo operation: {result}")
    
    # Application 2: Score tracking with bonuses
    print(f"\n2. Game Score Tracking with Bonuses:")
    score_stack = CustomStackArray(5)
    
    # Player scores
    scores = [100, 150, 200]
    for score in scores:
        score_stack.push(score)
        print(f"  Added score: {score}")
    
    # Apply bonus to recent scores
    score_stack.increment(2, 50)
    print(f"  Applied 50 point bonus to last 2 scores")
    
    print("  Final scores:")
    while True:
        score = score_stack.pop()
        if score == -1:
            break
        print(f"    Score: {score}")


if __name__ == "__main__":
    test_custom_stack_implementations()
    demonstrate_lazy_increment()
    visualize_increment_operations()
    demonstrate_real_world_applications()

"""
Design a Stack With Increment Operation demonstrates advanced stack
applications with lazy evaluation techniques and multiple implementation
approaches for efficient batch operations on stack elements.
"""
