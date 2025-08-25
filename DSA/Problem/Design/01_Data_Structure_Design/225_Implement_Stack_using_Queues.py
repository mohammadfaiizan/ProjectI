"""
225. Implement Stack using Queues - Multiple Approaches
Difficulty: Easy

Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should 
support all the functions of a normal stack (push, top, pop, and empty).

Implement the MyStack class:
- MyStack() Initializes the stack object.
- void push(int x) Pushes element x to the top of the stack.
- int pop() Removes the element on the top of the stack and returns it.
- int top() Returns the element on the top of the stack.
- boolean empty() Returns true if the stack is empty, false otherwise.

Notes:
- You must use only standard operations of a queue, which means that only push to back, 
  peek/pop from front, size and is empty operations are valid.
"""

from collections import deque
from typing import Optional

class MyStackTwoQueues:
    """
    Approach 1: Two Queues (Push Heavy)
    
    Use two queues, make push operation O(n) to maintain LIFO order.
    
    Time Complexity: 
    - push: O(n)
    - pop: O(1)
    - top: O(1)
    - empty: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.q1 = deque()  # Main queue
        self.q2 = deque()  # Helper queue
    
    def push(self, x: int) -> None:
        # Add new element to q2
        self.q2.append(x)
        
        # Move all elements from q1 to q2
        while self.q1:
            self.q2.append(self.q1.popleft())
        
        # Swap queues
        self.q1, self.q2 = self.q2, self.q1
    
    def pop(self) -> int:
        if self.empty():
            return -1
        return self.q1.popleft()
    
    def top(self) -> int:
        if self.empty():
            return -1
        return self.q1[0]
    
    def empty(self) -> bool:
        return len(self.q1) == 0

class MyStackTwoQueuesPopHeavy:
    """
    Approach 2: Two Queues (Pop Heavy)
    
    Use two queues, make pop operation O(n) to maintain LIFO order.
    
    Time Complexity: 
    - push: O(1)
    - pop: O(n)
    - top: O(n)
    - empty: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.q1 = deque()  # Main queue
        self.q2 = deque()  # Helper queue
    
    def push(self, x: int) -> None:
        self.q1.append(x)
    
    def pop(self) -> int:
        if self.empty():
            return -1
        
        # Move all elements except last to q2
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        
        # Pop the last element (top of stack)
        result = self.q1.popleft()
        
        # Swap queues
        self.q1, self.q2 = self.q2, self.q1
        
        return result
    
    def top(self) -> int:
        if self.empty():
            return -1
        
        # Similar to pop but don't remove the element
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        
        result = self.q1[0]
        self.q2.append(self.q1.popleft())
        
        # Swap queues
        self.q1, self.q2 = self.q2, self.q1
        
        return result
    
    def empty(self) -> bool:
        return len(self.q1) == 0

class MyStackOneQueue:
    """
    Approach 3: Single Queue
    
    Use only one queue and rotate elements for LIFO behavior.
    
    Time Complexity: 
    - push: O(n)
    - pop: O(1)
    - top: O(1)
    - empty: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.queue = deque()
    
    def push(self, x: int) -> None:
        size = len(self.queue)
        self.queue.append(x)
        
        # Rotate the queue to bring new element to front
        for _ in range(size):
            self.queue.append(self.queue.popleft())
    
    def pop(self) -> int:
        if self.empty():
            return -1
        return self.queue.popleft()
    
    def top(self) -> int:
        if self.empty():
            return -1
        return self.queue[0]
    
    def empty(self) -> bool:
        return len(self.queue) == 0

class MyStackOptimized:
    """
    Approach 4: Optimized with Top Tracking
    
    Track the top element separately to avoid O(n) top operation.
    
    Time Complexity: 
    - push: O(1)
    - pop: O(n)
    - top: O(1)
    - empty: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()
        self.top_element = None
    
    def push(self, x: int) -> None:
        self.q1.append(x)
        self.top_element = x
    
    def pop(self) -> int:
        if self.empty():
            return -1
        
        # Move all elements except last to q2, track new top
        while len(self.q1) > 1:
            self.top_element = self.q1.popleft()
            self.q2.append(self.top_element)
        
        result = self.q1.popleft()
        
        # Swap queues
        self.q1, self.q2 = self.q2, self.q1
        
        # Update top element
        if self.empty():
            self.top_element = None
        
        return result
    
    def top(self) -> int:
        if self.empty():
            return -1
        return self.top_element
    
    def empty(self) -> bool:
        return len(self.q1) == 0

class MyStackList:
    """
    Approach 5: Using List as Queue (for comparison)
    
    Demonstrate the concept using Python list operations.
    Note: This is not using actual queue data structure.
    
    Time Complexity: 
    - push: O(n)
    - pop: O(1)
    - top: O(1)
    - empty: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.data = []
    
    def push(self, x: int) -> None:
        # Insert at beginning to simulate LIFO with FIFO operations
        self.data.insert(0, x)
    
    def pop(self) -> int:
        if self.empty():
            return -1
        return self.data.pop(0)
    
    def top(self) -> int:
        if self.empty():
            return -1
        return self.data[0]
    
    def empty(self) -> bool:
        return len(self.data) == 0


def test_stack_basic_operations():
    """Test basic stack operations"""
    print("=== Testing Basic Stack Operations ===")
    
    implementations = [
        ("Two Queues (Push Heavy)", MyStackTwoQueues),
        ("Two Queues (Pop Heavy)", MyStackTwoQueuesPopHeavy),
        ("Single Queue", MyStackOneQueue),
        ("Optimized Top Tracking", MyStackOptimized),
        ("List Implementation", MyStackList)
    ]
    
    for name, StackClass in implementations:
        print(f"\n{name}:")
        
        stack = StackClass()
        
        # Test empty stack
        print(f"  empty(): {stack.empty()}")
        
        # Test push operations
        for i in [1, 2, 3]:
            stack.push(i)
            print(f"  push({i}), top(): {stack.top()}")
        
        print(f"  empty(): {stack.empty()}")
        
        # Test pop operations
        while not stack.empty():
            top_val = stack.top()
            popped = stack.pop()
            print(f"  top(): {top_val}, pop(): {popped}")
        
        print(f"  empty(): {stack.empty()}")

def test_stack_edge_cases():
    """Test stack edge cases"""
    print("\n=== Testing Stack Edge Cases ===")
    
    stack = MyStackTwoQueues()
    
    # Test operations on empty stack
    print("Operations on empty stack:")
    print(f"  pop(): {stack.pop()}")
    print(f"  top(): {stack.top()}")
    print(f"  empty(): {stack.empty()}")
    
    # Test single element
    print(f"\nSingle element operations:")
    stack.push(42)
    print(f"  push(42), top(): {stack.top()}")
    print(f"  pop(): {stack.pop()}")
    print(f"  empty(): {stack.empty()}")
    
    # Test alternating push/pop
    print(f"\nAlternating push/pop:")
    stack.push(1)
    print(f"  push(1), top(): {stack.top()}")
    print(f"  pop(): {stack.pop()}")
    
    stack.push(2)
    stack.push(3)
    print(f"  push(2), push(3), top(): {stack.top()}")
    print(f"  pop(): {stack.pop()}")
    print(f"  top(): {stack.top()}")

def test_stack_lifo_behavior():
    """Test LIFO (Last In, First Out) behavior"""
    print("\n=== Testing LIFO Behavior ===")
    
    stack = MyStackOneQueue()
    
    # Push sequence
    sequence = [1, 2, 3, 4, 5]
    print(f"Push sequence: {sequence}")
    
    for num in sequence:
        stack.push(num)
    
    # Pop sequence should be reversed
    pop_sequence = []
    while not stack.empty():
        pop_sequence.append(stack.pop())
    
    print(f"Pop sequence: {pop_sequence}")
    print(f"Correct LIFO order: {pop_sequence == list(reversed(sequence))}")

def test_stack_performance():
    """Test performance of different implementations"""
    print("\n=== Testing Stack Performance ===")
    
    import time
    
    implementations = [
        ("Two Queues (Push Heavy)", MyStackTwoQueues),
        ("Two Queues (Pop Heavy)", MyStackTwoQueuesPopHeavy),
        ("Single Queue", MyStackOneQueue),
        ("Optimized", MyStackOptimized)
    ]
    
    operations = 1000
    
    for name, StackClass in implementations:
        stack = StackClass()
        
        # Test push performance
        start_time = time.time()
        for i in range(operations):
            stack.push(i)
        push_time = (time.time() - start_time) * 1000
        
        # Test pop performance
        start_time = time.time()
        for _ in range(operations):
            stack.pop()
        pop_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Push {operations} elements: {push_time:.2f}ms")
        print(f"    Pop {operations} elements: {pop_time:.2f}ms")

def demonstrate_queue_to_stack_concept():
    """Demonstrate the concept of using queues to implement stack"""
    print("\n=== Demonstrating Queue-to-Stack Concept ===")
    
    print("Queue behavior (FIFO - First In, First Out):")
    queue = deque([1, 2, 3, 4, 5])
    print(f"  Queue: {list(queue)}")
    print(f"  Remove from front: {queue.popleft()}")  # Removes 1
    print(f"  Queue after: {list(queue)}")
    
    print(f"\nStack behavior (LIFO - Last In, First Out):")
    stack = MyStackOneQueue()
    for i in [1, 2, 3, 4, 5]:
        stack.push(i)
    
    print(f"  Pushed: [1, 2, 3, 4, 5]")
    print(f"  Pop: {stack.pop()}")  # Should remove 5 (last in)
    print(f"  Top: {stack.top()}")  # Should show 4

def test_stack_with_duplicates():
    """Test stack behavior with duplicate elements"""
    print("\n=== Testing Stack with Duplicates ===")
    
    stack = MyStackOptimized()
    
    # Push duplicates
    elements = [1, 2, 2, 3, 2, 1]
    print(f"Pushing elements: {elements}")
    
    for elem in elements:
        stack.push(elem)
        print(f"  Pushed {elem}, top(): {stack.top()}")
    
    # Pop all elements
    print(f"Popping all elements:")
    popped = []
    while not stack.empty():
        val = stack.pop()
        popped.append(val)
        print(f"  Popped: {val}")
    
    print(f"Popped sequence: {popped}")
    print(f"Expected (reversed): {list(reversed(elements))}")
    print(f"Correct: {popped == list(reversed(elements))}")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Function call simulation
    print("Application 1: Function Call Stack Simulation")
    call_stack = MyStackTwoQueues()
    
    functions = ["main()", "processData()", "validateInput()", "parseJSON()"]
    
    print("  Function calls:")
    for func in functions:
        call_stack.push(func)
        print(f"    Called {func}, current: {call_stack.top()}")
    
    print("  Function returns:")
    while not call_stack.empty():
        returned = call_stack.pop()
        current = call_stack.top() if not call_stack.empty() else "program end"
        print(f"    Returned from {returned}, current: {current}")
    
    # Application 2: Undo operations
    print(f"\nApplication 2: Undo Operations")
    undo_stack = MyStackOneQueue()
    
    actions = ["Type 'Hello'", "Type ' World'", "Delete 5 chars", "Type '!'"]
    
    print("  User actions:")
    for action in actions:
        undo_stack.push(action)
        print(f"    Action: {action}")
    
    print("  Undo operations:")
    while not undo_stack.empty():
        last_action = undo_stack.pop()
        print(f"    Undo: {last_action}")

def benchmark_memory_usage():
    """Benchmark memory usage of different approaches"""
    print("\n=== Memory Usage Analysis ===")
    
    implementations = [
        ("Two Queues", MyStackTwoQueues),
        ("Single Queue", MyStackOneQueue),
        ("Optimized", MyStackOptimized)
    ]
    
    size = 1000
    
    for name, StackClass in implementations:
        stack = StackClass()
        
        # Fill stack
        for i in range(size):
            stack.push(i)
        
        # Estimate memory usage
        if hasattr(stack, 'q1') and hasattr(stack, 'q2'):
            memory_units = len(stack.q1) + len(stack.q2)
        elif hasattr(stack, 'queue'):
            memory_units = len(stack.queue)
        else:
            memory_units = len(stack.data) if hasattr(stack, 'data') else size
        
        print(f"  {name}: ~{memory_units} memory units for {size} elements")

def test_concurrent_operations():
    """Test concurrent-like operations pattern"""
    print("\n=== Testing Operation Patterns ===")
    
    stack = MyStackOptimized()
    
    # Pattern 1: Push multiple, pop one
    print("Pattern 1: Batch push, single pop")
    for i in range(5):
        stack.push(i)
    
    print(f"  After batch push, top: {stack.top()}")
    print(f"  Pop one: {stack.pop()}")
    print(f"  New top: {stack.top()}")
    
    # Pattern 2: Alternating operations
    print(f"\nPattern 2: Alternating push/pop")
    for i in range(3):
        stack.push(10 + i)
        print(f"  Push {10 + i}, top: {stack.top()}")
        if i > 0:  # Don't pop on first iteration
            popped = stack.pop()
            print(f"  Pop: {popped}, new top: {stack.top()}")

if __name__ == "__main__":
    test_stack_basic_operations()
    test_stack_edge_cases()
    test_stack_lifo_behavior()
    test_stack_performance()
    demonstrate_queue_to_stack_concept()
    test_stack_with_duplicates()
    demonstrate_real_world_applications()
    benchmark_memory_usage()
    test_concurrent_operations()

"""
Stack using Queues Design demonstrates key concepts:

Core Approaches:
1. Two Queues (Push Heavy) - O(n) push, O(1) pop/top
2. Two Queues (Pop Heavy) - O(1) push, O(n) pop/top  
3. Single Queue - O(n) push, O(1) pop/top with rotation
4. Optimized - Track top element separately for O(1) top operation
5. List Implementation - For conceptual understanding

Key Design Principles:
- Converting FIFO (queue) behavior to LIFO (stack) behavior
- Trade-offs between push and pop performance
- Memory efficiency considerations
- Maintaining stack semantics with queue operations

Real-world Applications:
- Function call stack simulation
- Undo/redo operations
- Expression evaluation
- Browser history (back button)
- Compiler parsing

The choice between approaches depends on whether
push or pop operations are more frequent in the use case.
"""
