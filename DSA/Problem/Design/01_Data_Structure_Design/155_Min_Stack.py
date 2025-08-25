"""
155. Min Stack - Multiple Approaches
Difficulty: Medium

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
import sys

class MinStackTwoStacks:
    """
    Approach 1: Two Stacks
    
    Use two stacks - one for regular elements and one for tracking minimums.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(n) where n is the number of elements
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
            
            # Pop from min_stack if the popped value is the current minimum
            if self.min_stack and val == self.min_stack[-1]:
                self.min_stack.pop()
    
    def top(self) -> int:
        return self.stack[-1] if self.stack else None
    
    def getMin(self) -> int:
        return self.min_stack[-1] if self.min_stack else None

class MinStackSingleStack:
    """
    Approach 2: Single Stack with Pairs
    
    Store pairs of (value, current_minimum) in a single stack.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(n) where n is the number of elements
    """
    
    def __init__(self):
        self.stack = []  # Each element is (value, min_so_far)
    
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

class MinStackOptimized:
    """
    Approach 3: Optimized Single Stack
    
    Only store the difference between current value and minimum when necessary.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(n) but more space-efficient in practice
    """
    
    def __init__(self):
        self.stack = []
        self.current_min = None
    
    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append(0)
            self.current_min = val
        else:
            # Store the difference
            diff = val - self.current_min
            self.stack.append(diff)
            
            # Update minimum if current value is smaller
            if val < self.current_min:
                self.current_min = val
    
    def pop(self) -> None:
        if self.stack:
            diff = self.stack.pop()
            
            # If diff < 0, the popped element was the minimum
            if diff < 0:
                self.current_min = self.current_min - diff
    
    def top(self) -> int:
        if not self.stack:
            return None
        
        diff = self.stack[-1]
        
        if diff < 0:
            return self.current_min
        else:
            return self.current_min + diff
    
    def getMin(self) -> int:
        return self.current_min

class MinStackWithLinkedList:
    """
    Approach 4: Linked List Implementation
    
    Use a linked list where each node stores value and minimum up to that point.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(n) where n is the number of elements
    """
    
    class Node:
        def __init__(self, val: int, min_val: int, next_node=None):
            self.val = val
            self.min_val = min_val
            self.next = next_node
    
    def __init__(self):
        self.head = None
    
    def push(self, val: int) -> None:
        if self.head is None:
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

class MinStackThreadSafe:
    """
    Approach 5: Thread-Safe Min Stack
    
    Add thread safety with locks for concurrent access.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(n) where n is the number of elements
    """
    
    def __init__(self):
        import threading
        self.stack = []
        self.min_stack = []
        self.lock = threading.RLock()
    
    def push(self, val: int) -> None:
        with self.lock:
            self.stack.append(val)
            
            if not self.min_stack or val <= self.min_stack[-1]:
                self.min_stack.append(val)
    
    def pop(self) -> None:
        with self.lock:
            if self.stack:
                val = self.stack.pop()
                
                if self.min_stack and val == self.min_stack[-1]:
                    self.min_stack.pop()
    
    def top(self) -> int:
        with self.lock:
            return self.stack[-1] if self.stack else None
    
    def getMin(self) -> int:
        with self.lock:
            return self.min_stack[-1] if self.min_stack else None
    
    def size(self) -> int:
        with self.lock:
            return len(self.stack)
    
    def is_empty(self) -> bool:
        with self.lock:
            return len(self.stack) == 0


def test_min_stack_basic():
    """Test basic MinStack functionality"""
    print("=== Testing Basic MinStack Functionality ===")
    
    implementations = [
        ("Two Stacks", MinStackTwoStacks),
        ("Single Stack with Pairs", MinStackSingleStack),
        ("Optimized Single Stack", MinStackOptimized),
        ("Linked List", MinStackWithLinkedList),
        ("Thread-Safe", MinStackThreadSafe)
    ]
    
    for name, MinStackClass in implementations:
        print(f"\n{name}:")
        
        min_stack = MinStackClass()
        
        # Test sequence from LeetCode example
        min_stack.push(-2)
        min_stack.push(0)
        min_stack.push(-3)
        
        print(f"  getMin(): {min_stack.getMin()}")  # Returns -3
        
        min_stack.pop()
        print(f"  top(): {min_stack.top()}")       # Returns 0
        print(f"  getMin(): {min_stack.getMin()}")  # Returns -2

def test_min_stack_edge_cases():
    """Test MinStack edge cases"""
    print("\n=== Testing MinStack Edge Cases ===")
    
    min_stack = MinStackTwoStacks()
    
    # Test empty stack
    print("Testing empty stack:")
    print(f"  top(): {min_stack.top()}")
    print(f"  getMin(): {min_stack.getMin()}")
    
    # Test single element
    print(f"\nTesting single element:")
    min_stack.push(5)
    print(f"  top(): {min_stack.top()}")
    print(f"  getMin(): {min_stack.getMin()}")
    
    # Test duplicate minimums
    print(f"\nTesting duplicate minimums:")
    min_stack.push(1)
    min_stack.push(1)
    min_stack.push(2)
    
    print(f"  getMin(): {min_stack.getMin()}")  # Should be 1
    min_stack.pop()  # Remove 2
    print(f"  getMin(): {min_stack.getMin()}")  # Should still be 1
    min_stack.pop()  # Remove one 1
    print(f"  getMin(): {min_stack.getMin()}")  # Should still be 1

def test_min_stack_performance():
    """Test MinStack performance"""
    print("\n=== Testing MinStack Performance ===")
    
    import time
    import random
    
    implementations = [
        ("Two Stacks", MinStackTwoStacks),
        ("Single Stack", MinStackSingleStack),
        ("Optimized", MinStackOptimized),
        ("Linked List", MinStackWithLinkedList)
    ]
    
    operations = 100000
    
    for name, MinStackClass in implementations:
        min_stack = MinStackClass()
        
        start_time = time.time()
        
        # Perform random operations
        for _ in range(operations):
            operation = random.choice(['push', 'pop', 'top', 'getMin'])
            
            if operation == 'push':
                min_stack.push(random.randint(-1000, 1000))
            elif operation == 'pop' and hasattr(min_stack, 'stack') and min_stack.stack:
                min_stack.pop()
            elif operation == 'pop' and hasattr(min_stack, 'head') and min_stack.head:
                min_stack.pop()
            elif operation == 'top':
                min_stack.top()
            elif operation == 'getMin':
                min_stack.getMin()
        
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000
        
        print(f"  {name}: {elapsed:.2f}ms for {operations} operations")

def test_min_stack_memory_usage():
    """Test MinStack memory usage"""
    print("\n=== Testing MinStack Memory Usage ===")
    
    import sys
    
    # Compare memory usage of different approaches
    test_data = list(range(1000, 0, -1))  # Decreasing sequence (worst case for min tracking)
    
    implementations = [
        ("Two Stacks", MinStackTwoStacks),
        ("Single Stack", MinStackSingleStack),
        ("Optimized", MinStackOptimized)
    ]
    
    for name, MinStackClass in implementations:
        min_stack = MinStackClass()
        
        # Add all elements
        for val in test_data:
            min_stack.push(val)
        
        # Estimate memory usage (rough approximation)
        if hasattr(min_stack, 'stack') and hasattr(min_stack, 'min_stack'):
            memory = len(min_stack.stack) + len(min_stack.min_stack)
        elif hasattr(min_stack, 'stack'):
            memory = len(min_stack.stack)
        else:
            memory = 1000  # Rough estimate for linked list
        
        print(f"  {name}: ~{memory} elements stored for {len(test_data)} pushed")

def demonstrate_min_stack_applications():
    """Demonstrate MinStack applications"""
    print("\n=== Demonstrating MinStack Applications ===")
    
    # Application 1: Stock price monitoring
    print("Application 1: Stock Price Monitoring")
    stock_monitor = MinStackTwoStacks()
    
    prices = [100, 95, 110, 85, 120, 80, 130]
    
    for price in prices:
        stock_monitor.push(price)
        print(f"  Price: ${price}, Lowest so far: ${stock_monitor.getMin()}")
    
    # Application 2: Temperature tracking
    print(f"\nApplication 2: Temperature Tracking")
    temp_tracker = MinStackSingleStack()
    
    temperatures = [72, 68, 75, 65, 80, 60, 85]
    
    for temp in temperatures:
        temp_tracker.push(temp)
        print(f"  Temp: {temp}°F, Minimum: {temp_tracker.getMin()}°F")

def test_min_stack_advanced_operations():
    """Test advanced MinStack operations"""
    print("\n=== Testing Advanced MinStack Operations ===")
    
    # Test with thread-safe version
    min_stack = MinStackThreadSafe()
    
    # Test size and empty operations
    print(f"Initially empty: {min_stack.is_empty()}")
    print(f"Initial size: {min_stack.size()}")
    
    # Add elements
    for i in [5, 2, 8, 1, 9]:
        min_stack.push(i)
        print(f"  Pushed {i}, size: {min_stack.size()}, min: {min_stack.getMin()}")
    
    # Remove elements
    while not min_stack.is_empty():
        top_val = min_stack.top()
        min_val = min_stack.getMin()
        min_stack.pop()
        print(f"  Popped {top_val}, size: {min_stack.size()}, min: {min_val}")

def benchmark_min_stack_scenarios():
    """Benchmark MinStack in different scenarios"""
    print("\n=== Benchmarking MinStack Scenarios ===")
    
    import time
    
    scenarios = [
        ("Ascending order", list(range(1000))),
        ("Descending order", list(range(1000, 0, -1))),
        ("Random order", [i for i in range(1000)])
    ]
    
    # Shuffle the random order
    import random
    random.shuffle(scenarios[2][1])
    
    for scenario_name, data in scenarios:
        print(f"\n{scenario_name}:")
        
        for impl_name, MinStackClass in [("Two Stacks", MinStackTwoStacks), ("Optimized", MinStackOptimized)]:
            min_stack = MinStackClass()
            
            start_time = time.time()
            
            # Push all elements
            for val in data:
                min_stack.push(val)
            
            # Get minimum (should be constant time)
            for _ in range(100):
                min_stack.getMin()
            
            # Pop all elements
            for _ in range(len(data)):
                min_stack.pop()
            
            end_time = time.time()
            elapsed = (end_time - start_time) * 1000
            
            print(f"  {impl_name}: {elapsed:.2f}ms")

if __name__ == "__main__":
    test_min_stack_basic()
    test_min_stack_edge_cases()
    test_min_stack_performance()
    test_min_stack_memory_usage()
    demonstrate_min_stack_applications()
    test_min_stack_advanced_operations()
    benchmark_min_stack_scenarios()

"""
Min Stack Design demonstrates several key concepts:

Core Approaches:
1. Two Stacks - Intuitive and clear separation of concerns
2. Single Stack with Pairs - Space-efficient while maintaining simplicity
3. Optimized Single Stack - Advanced space optimization using differences
4. Linked List - Alternative implementation with different memory characteristics
5. Thread-Safe - Production-ready with concurrency support

Key Design Principles:
- Constant time operations through auxiliary data structures
- Space-time tradeoffs in different implementations
- Thread safety considerations for concurrent environments

Real-world Applications:
- Function call stack with minimum tracking
- Stock price monitoring systems
- Temperature/sensor data tracking
- Undo/redo operations with constraints
- Browser history with minimum load times

The Two Stacks approach is most commonly used in interviews
due to its clarity and optimal performance characteristics.
"""
