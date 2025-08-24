"""
232. Implement Queue using Stacks - Multiple Approaches
Difficulty: Easy

Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (push, peek, pop, and empty).

Implement the MyQueue class:
- void push(int x) Pushes element x to the back of the queue.
- int pop() Removes the element from the front of the queue and returns it.
- int peek() Returns the element at the front of the queue.
- boolean empty() Returns true if the queue is empty, false otherwise.
"""

from typing import List, Optional

class MyQueue1:
    """
    Approach 1: Two Stacks - Push Expensive
    
    Make push operation expensive by transferring all elements.
    
    Time: push O(n), pop/peek/empty O(1), Space: O(n)
    """
    
    def __init__(self):
        self.stack1 = []  # Main stack
        self.stack2 = []  # Auxiliary stack
    
    def push(self, x: int) -> None:
        # Move all elements from stack1 to stack2
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        
        # Push new element to stack1
        self.stack1.append(x)
        
        # Move all elements back from stack2 to stack1
        while self.stack2:
            self.stack1.append(self.stack2.pop())
    
    def pop(self) -> int:
        if self.stack1:
            return self.stack1.pop()
        return -1
    
    def peek(self) -> int:
        if self.stack1:
            return self.stack1[-1]
        return -1
    
    def empty(self) -> bool:
        return len(self.stack1) == 0


class MyQueue2:
    """
    Approach 2: Two Stacks - Pop Expensive (Optimal)
    
    Make pop/peek operations expensive, amortized O(1).
    
    Time: push O(1), pop/peek O(1) amortized, Space: O(n)
    """
    
    def __init__(self):
        self.input_stack = []   # For push operations
        self.output_stack = []  # For pop/peek operations
    
    def push(self, x: int) -> None:
        self.input_stack.append(x)
    
    def pop(self) -> int:
        self._transfer_if_needed()
        if self.output_stack:
            return self.output_stack.pop()
        return -1
    
    def peek(self) -> int:
        self._transfer_if_needed()
        if self.output_stack:
            return self.output_stack[-1]
        return -1
    
    def empty(self) -> bool:
        return len(self.input_stack) == 0 and len(self.output_stack) == 0
    
    def _transfer_if_needed(self) -> None:
        """Transfer elements from input to output stack if output is empty"""
        if not self.output_stack:
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())


class MyQueue3:
    """
    Approach 3: Single Stack with Recursion
    
    Use recursion to simulate queue behavior with single stack.
    
    Time: push O(n), pop O(1), Space: O(n)
    """
    
    def __init__(self):
        self.stack = []
    
    def push(self, x: int) -> None:
        self._push_recursive(x)
    
    def _push_recursive(self, x: int) -> None:
        if not self.stack:
            self.stack.append(x)
            return
        
        # Remove top element
        temp = self.stack.pop()
        
        # Recursively push x
        self._push_recursive(x)
        
        # Put back the removed element
        self.stack.append(temp)
    
    def pop(self) -> int:
        if self.stack:
            return self.stack.pop()
        return -1
    
    def peek(self) -> int:
        if self.stack:
            return self.stack[-1]
        return -1
    
    def empty(self) -> bool:
        return len(self.stack) == 0


class MyQueue4:
    """
    Approach 4: Deque Simulation with Two Stacks
    
    Simulate deque behavior using two stacks.
    
    Time: All operations O(1) amortized, Space: O(n)
    """
    
    def __init__(self):
        self.front_stack = []  # Front half of queue
        self.back_stack = []   # Back half of queue
        self.front_size = 0
        self.back_size = 0
    
    def push(self, x: int) -> None:
        self.back_stack.append(x)
        self.back_size += 1
        self._rebalance()
    
    def pop(self) -> int:
        if self.empty():
            return -1
        
        if self.front_stack:
            self.front_size -= 1
            return self.front_stack.pop()
        else:
            self.back_size -= 1
            # Need to reverse back_stack to get front element
            temp = []
            while len(self.back_stack) > 1:
                temp.append(self.back_stack.pop())
            
            result = self.back_stack.pop() if self.back_stack else -1
            
            # Restore back_stack
            while temp:
                self.back_stack.append(temp.pop())
            
            return result
    
    def peek(self) -> int:
        if self.empty():
            return -1
        
        if self.front_stack:
            return self.front_stack[-1]
        else:
            # Need to find front element in back_stack
            temp = []
            while len(self.back_stack) > 1:
                temp.append(self.back_stack.pop())
            
            result = self.back_stack[-1] if self.back_stack else -1
            
            # Restore back_stack
            while temp:
                self.back_stack.append(temp.pop())
            
            return result
    
    def empty(self) -> bool:
        return self.front_size == 0 and self.back_size == 0
    
    def _rebalance(self) -> None:
        """Rebalance stacks to maintain efficiency"""
        total_size = self.front_size + self.back_size
        if total_size > 1 and self.front_size < total_size // 3:
            # Move half of back_stack to front_stack
            move_count = self.back_size // 2
            temp = []
            
            for _ in range(move_count):
                if self.back_stack:
                    temp.append(self.back_stack.pop())
            
            while temp:
                self.front_stack.append(temp.pop())
            
            self.front_size += move_count
            self.back_size -= move_count


class MyQueue5:
    """
    Approach 5: Lazy Transfer Strategy
    
    Transfer elements only when absolutely necessary.
    
    Time: All operations O(1) amortized, Space: O(n)
    """
    
    def __init__(self):
        self.newest_on_top = []  # Push stack
        self.oldest_on_top = []  # Pop stack
    
    def push(self, x: int) -> None:
        self.newest_on_top.append(x)
    
    def pop(self) -> int:
        self._shift_stacks()
        if self.oldest_on_top:
            return self.oldest_on_top.pop()
        return -1
    
    def peek(self) -> int:
        self._shift_stacks()
        if self.oldest_on_top:
            return self.oldest_on_top[-1]
        return -1
    
    def empty(self) -> bool:
        return len(self.newest_on_top) == 0 and len(self.oldest_on_top) == 0
    
    def _shift_stacks(self) -> None:
        """Move elements from newest_on_top to oldest_on_top if needed"""
        if not self.oldest_on_top:
            while self.newest_on_top:
                self.oldest_on_top.append(self.newest_on_top.pop())


def test_queue_implementations():
    """Test all queue implementations"""
    
    implementations = [
        ("Push Expensive", MyQueue1),
        ("Pop Expensive (Optimal)", MyQueue2),
        ("Single Stack Recursive", MyQueue3),
        ("Deque Simulation", MyQueue4),
        ("Lazy Transfer", MyQueue5),
    ]
    
    test_operations = [
        ("push", 1),
        ("push", 2),
        ("peek", None, 1),
        ("pop", None, 1),
        ("empty", None, False),
        ("push", 3),
        ("push", 4),
        ("pop", None, 2),
        ("peek", None, 3),
        ("pop", None, 3),
        ("pop", None, 4),
        ("empty", None, True),
    ]
    
    print("=== Testing Queue using Stacks Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- Testing {impl_name} ---")
        
        try:
            queue = impl_class()
            
            for operation in test_operations:
                if operation[0] == "push":
                    queue.push(operation[1])
                    print(f"push({operation[1]})")
                elif operation[0] == "pop":
                    result = queue.pop()
                    expected = operation[2]
                    status = "✓" if result == expected else "✗"
                    print(f"pop() = {result} {status}")
                elif operation[0] == "peek":
                    result = queue.peek()
                    expected = operation[2]
                    status = "✓" if result == expected else "✗"
                    print(f"peek() = {result} {status}")
                elif operation[0] == "empty":
                    result = queue.empty()
                    expected = operation[2]
                    status = "✓" if result == expected else "✗"
                    print(f"empty() = {result} {status}")
        
        except Exception as e:
            print(f"ERROR: {str(e)}")


def benchmark_queue_implementations():
    """Benchmark different queue implementations"""
    import time
    import random
    
    implementations = [
        ("Push Expensive", MyQueue1),
        ("Pop Expensive", MyQueue2),
        ("Lazy Transfer", MyQueue5),
    ]
    
    n = 5000
    operations = []
    
    # Generate random operations
    for _ in range(n):
        op_type = random.choice(['push', 'pop', 'peek', 'empty'])
        if op_type == 'push':
            operations.append(('push', random.randint(1, 1000)))
        else:
            operations.append((op_type,))
    
    print("\n=== Queue Implementation Performance Benchmark ===")
    print(f"Operations: {n}")
    
    for impl_name, impl_class in implementations:
        queue = impl_class()
        
        start_time = time.time()
        
        for operation in operations:
            try:
                if operation[0] == 'push':
                    queue.push(operation[1])
                elif operation[0] == 'pop':
                    queue.pop()
                elif operation[0] == 'peek':
                    queue.peek()
                elif operation[0] == 'empty':
                    queue.empty()
            except:
                pass  # Ignore errors for benchmark
        
        end_time = time.time()
        
        print(f"{impl_name:20} | Time: {end_time - start_time:.4f}s")


def demonstrate_queue_behavior():
    """Demonstrate queue FIFO behavior"""
    print("\n=== Queue FIFO Behavior Demonstration ===")
    
    queue = MyQueue2()  # Using optimal implementation
    
    print("Creating queue and adding elements: 1, 2, 3, 4")
    for i in range(1, 5):
        queue.push(i)
        print(f"push({i})")
    
    print("\nRemoving elements (should be in FIFO order):")
    while not queue.empty():
        front = queue.peek()
        popped = queue.pop()
        print(f"peek() = {front}, pop() = {popped}")


if __name__ == "__main__":
    test_queue_implementations()
    benchmark_queue_implementations()
    demonstrate_queue_behavior()

"""
Queue using Stacks demonstrates multiple implementation strategies
including push-expensive, pop-expensive, recursive, and optimized
approaches with amortized analysis and performance comparisons.
"""
