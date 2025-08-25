"""
232. Implement Queue using Stacks - Multiple Approaches
Difficulty: Easy

Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should 
support all the functions of a normal queue (push, peek, pop, and empty).

Implement the MyQueue class:
- MyQueue() Initializes the queue object.
- void push(int x) Pushes element x to the back of the queue.
- int pop() Removes the element from the front of the queue and returns it.
- int peek() Returns the element at the front of the queue.
- boolean empty() Returns true if the queue is empty, false otherwise.

Notes:
- You must use only standard operations of a stack -- which means only push to top, 
  peek/pop from top, size, and is empty operations are valid.
"""

from typing import List, Optional

class MyQueueTwoStacks:
    """
    Approach 1: Two Stacks (Input/Output Pattern)
    
    Use two stacks: one for input operations, one for output operations.
    
    Time Complexity: 
    - push: O(1)
    - pop: O(1) amortized, O(n) worst case
    - peek: O(1) amortized, O(n) worst case
    - empty: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.input_stack = []   # For push operations
        self.output_stack = []  # For pop/peek operations
    
    def push(self, x: int) -> None:
        self.input_stack.append(x)
    
    def pop(self) -> int:
        if self.empty():
            return -1
        
        self._transfer_if_needed()
        return self.output_stack.pop()
    
    def peek(self) -> int:
        if self.empty():
            return -1
        
        self._transfer_if_needed()
        return self.output_stack[-1]
    
    def empty(self) -> bool:
        return len(self.input_stack) == 0 and len(self.output_stack) == 0
    
    def _transfer_if_needed(self) -> None:
        """Transfer elements from input to output stack if output is empty"""
        if not self.output_stack:
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())

class MyQueueSingleStack:
    """
    Approach 2: Single Stack with Recursion
    
    Use recursion to achieve FIFO behavior with a single stack.
    
    Time Complexity: 
    - push: O(n)
    - pop: O(n)
    - peek: O(n)
    - empty: O(1)
    
    Space Complexity: O(n) including recursion stack
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
        
        # Recursively push new element to bottom
        self._push_recursive(x)
        
        # Put back the removed element
        self.stack.append(temp)
    
    def pop(self) -> int:
        if self.empty():
            return -1
        return self.stack.pop()
    
    def peek(self) -> int:
        if self.empty():
            return -1
        return self.stack[-1]
    
    def empty(self) -> bool:
        return len(self.stack) == 0

class MyQueueTwoStacksPopHeavy:
    """
    Approach 3: Two Stacks (Pop Heavy)
    
    Make push operation expensive instead of pop/peek.
    
    Time Complexity: 
    - push: O(n)
    - pop: O(1)
    - peek: O(1)
    - empty: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.stack1 = []  # Main stack
        self.stack2 = []  # Helper stack
    
    def push(self, x: int) -> None:
        # Move all elements to stack2
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        
        # Push new element to stack1
        self.stack1.append(x)
        
        # Move all elements back to stack1
        while self.stack2:
            self.stack1.append(self.stack2.pop())
    
    def pop(self) -> int:
        if self.empty():
            return -1
        return self.stack1.pop()
    
    def peek(self) -> int:
        if self.empty():
            return -1
        return self.stack1[-1]
    
    def empty(self) -> bool:
        return len(self.stack1) == 0

class MyQueueOptimized:
    """
    Approach 4: Optimized Two Stacks with Front Tracking
    
    Track front element separately to optimize peek operation.
    
    Time Complexity: 
    - push: O(1)
    - pop: O(1) amortized
    - peek: O(1)
    - empty: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.input_stack = []
        self.output_stack = []
        self.front = None
    
    def push(self, x: int) -> None:
        if not self.input_stack:
            self.front = x
        self.input_stack.append(x)
    
    def pop(self) -> int:
        if self.empty():
            return -1
        
        if not self.output_stack:
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())
        
        return self.output_stack.pop()
    
    def peek(self) -> int:
        if self.empty():
            return -1
        
        if self.output_stack:
            return self.output_stack[-1]
        else:
            return self.front
    
    def empty(self) -> bool:
        return len(self.input_stack) == 0 and len(self.output_stack) == 0

class MyQueueWithList:
    """
    Approach 5: Using List Operations (for comparison)
    
    Demonstrate the concept using list operations that simulate stacks.
    
    Time Complexity: 
    - push: O(n)
    - pop: O(1)
    - peek: O(1)
    - empty: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.data = []
    
    def push(self, x: int) -> None:
        # Insert at beginning to maintain FIFO order
        self.data.insert(0, x)
    
    def pop(self) -> int:
        if self.empty():
            return -1
        return self.data.pop()
    
    def peek(self) -> int:
        if self.empty():
            return -1
        return self.data[-1]
    
    def empty(self) -> bool:
        return len(self.data) == 0


def test_queue_basic_operations():
    """Test basic queue operations"""
    print("=== Testing Basic Queue Operations ===")
    
    implementations = [
        ("Two Stacks (Input/Output)", MyQueueTwoStacks),
        ("Single Stack (Recursive)", MyQueueSingleStack),
        ("Two Stacks (Pop Heavy)", MyQueueTwoStacksPopHeavy),
        ("Optimized Front Tracking", MyQueueOptimized),
        ("List Implementation", MyQueueWithList)
    ]
    
    for name, QueueClass in implementations:
        print(f"\n{name}:")
        
        queue = QueueClass()
        
        # Test empty queue
        print(f"  empty(): {queue.empty()}")
        
        # Test push operations
        for i in [1, 2, 3]:
            queue.push(i)
            print(f"  push({i}), peek(): {queue.peek()}")
        
        print(f"  empty(): {queue.empty()}")
        
        # Test pop operations
        while not queue.empty():
            front = queue.peek()
            popped = queue.pop()
            print(f"  peek(): {front}, pop(): {popped}")
        
        print(f"  empty(): {queue.empty()}")

def test_queue_edge_cases():
    """Test queue edge cases"""
    print("\n=== Testing Queue Edge Cases ===")
    
    queue = MyQueueTwoStacks()
    
    # Test operations on empty queue
    print("Operations on empty queue:")
    print(f"  pop(): {queue.pop()}")
    print(f"  peek(): {queue.peek()}")
    print(f"  empty(): {queue.empty()}")
    
    # Test single element
    print(f"\nSingle element operations:")
    queue.push(42)
    print(f"  push(42), peek(): {queue.peek()}")
    print(f"  pop(): {queue.pop()}")
    print(f"  empty(): {queue.empty()}")
    
    # Test alternating push/pop
    print(f"\nAlternating push/pop:")
    queue.push(1)
    print(f"  push(1), peek(): {queue.peek()}")
    print(f"  pop(): {queue.pop()}")
    
    queue.push(2)
    queue.push(3)
    print(f"  push(2), push(3), peek(): {queue.peek()}")
    print(f"  pop(): {queue.pop()}")
    print(f"  peek(): {queue.peek()}")

def test_queue_fifo_behavior():
    """Test FIFO (First In, First Out) behavior"""
    print("\n=== Testing FIFO Behavior ===")
    
    queue = MyQueueOptimized()
    
    # Push sequence
    sequence = [1, 2, 3, 4, 5]
    print(f"Push sequence: {sequence}")
    
    for num in sequence:
        queue.push(num)
    
    # Pop sequence should be same order
    pop_sequence = []
    while not queue.empty():
        pop_sequence.append(queue.pop())
    
    print(f"Pop sequence: {pop_sequence}")
    print(f"Correct FIFO order: {pop_sequence == sequence}")

def test_queue_performance():
    """Test performance of different implementations"""
    print("\n=== Testing Queue Performance ===")
    
    import time
    
    implementations = [
        ("Two Stacks (Amortized)", MyQueueTwoStacks),
        ("Two Stacks (Pop Heavy)", MyQueueTwoStacksPopHeavy),
        ("Optimized", MyQueueOptimized)
    ]
    
    operations = 1000
    
    for name, QueueClass in implementations:
        queue = QueueClass()
        
        # Test push performance
        start_time = time.time()
        for i in range(operations):
            queue.push(i)
        push_time = (time.time() - start_time) * 1000
        
        # Test pop performance
        start_time = time.time()
        for _ in range(operations):
            queue.pop()
        pop_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Push {operations} elements: {push_time:.2f}ms")
        print(f"    Pop {operations} elements: {pop_time:.2f}ms")

def test_amortized_analysis():
    """Demonstrate amortized O(1) behavior"""
    print("\n=== Amortized Analysis Demonstration ===")
    
    import time
    
    queue = MyQueueTwoStacks()
    
    # Show that individual operations can be expensive but amortized cost is O(1)
    operation_times = []
    
    # Push elements
    for i in range(10):
        start = time.perf_counter()
        queue.push(i)
        end = time.perf_counter()
        operation_times.append((f"push({i})", (end - start) * 1000000))  # microseconds
    
    # Pop elements (first few will be expensive due to transfer)
    for i in range(10):
        start = time.perf_counter()
        queue.pop()
        end = time.perf_counter()
        operation_times.append((f"pop()", (end - start) * 1000000))
    
    print("Individual operation times (microseconds):")
    for operation, time_taken in operation_times:
        print(f"  {operation}: {time_taken:.2f}μs")

def demonstrate_stack_to_queue_concept():
    """Demonstrate the concept of using stacks to implement queue"""
    print("\n=== Demonstrating Stack-to-Queue Concept ===")
    
    print("Stack behavior (LIFO - Last In, First Out):")
    stack = [1, 2, 3, 4, 5]
    print(f"  Stack: {stack}")
    print(f"  Remove from top: {stack.pop()}")  # Removes 5
    print(f"  Stack after: {stack}")
    
    print(f"\nQueue behavior (FIFO - First In, First Out):")
    queue = MyQueueTwoStacks()
    for i in [1, 2, 3, 4, 5]:
        queue.push(i)
    
    print(f"  Pushed: [1, 2, 3, 4, 5]")
    print(f"  Pop: {queue.pop()}")  # Should remove 1 (first in)
    print(f"  Peek: {queue.peek()}")  # Should show 2

def test_queue_with_mixed_operations():
    """Test queue with mixed push/pop patterns"""
    print("\n=== Testing Mixed Operations ===")
    
    queue = MyQueueTwoStacks()
    
    operations = [
        ("push", 1), ("push", 2), ("pop", None), ("push", 3), 
        ("peek", None), ("push", 4), ("pop", None), ("pop", None)
    ]
    
    print("Mixed operations sequence:")
    for op, value in operations:
        if op == "push":
            queue.push(value)
            print(f"  {op}({value}) -> queue front: {queue.peek() if not queue.empty() else 'empty'}")
        elif op == "pop":
            result = queue.pop()
            print(f"  {op}() -> {result}, new front: {queue.peek() if not queue.empty() else 'empty'}")
        elif op == "peek":
            result = queue.peek()
            print(f"  {op}() -> {result}")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Task queue simulation
    print("Application 1: Task Queue")
    task_queue = MyQueueOptimized()
    
    tasks = ["Process Payment", "Send Email", "Update Database", "Generate Report"]
    
    print("  Adding tasks:")
    for task in tasks:
        task_queue.push(hash(task) % 1000)  # Use hash as task ID
        print(f"    Added: {task}")
    
    print("  Processing tasks (FIFO order):")
    task_index = 0
    while not task_queue.empty():
        task_id = task_queue.pop()
        print(f"    Processing: {tasks[task_index]} (ID: {task_id})")
        task_index += 1
    
    # Application 2: Print queue simulation
    print(f"\nApplication 2: Print Queue")
    print_queue = MyQueueTwoStacks()
    
    documents = ["Resume.pdf", "Report.docx", "Invoice.pdf", "Photos.jpg"]
    
    print("  Documents queued for printing:")
    for doc in documents:
        print_queue.push(hash(doc) % 1000)
        print(f"    Queued: {doc}")
    
    print("  Printing documents:")
    doc_index = 0
    while not print_queue.empty():
        doc_id = print_queue.pop()
        print(f"    Printing: {documents[doc_index]} (ID: {doc_id})")
        doc_index += 1

def test_memory_efficiency():
    """Test memory efficiency of different approaches"""
    print("\n=== Memory Efficiency Analysis ===")
    
    implementations = [
        ("Two Stacks", MyQueueTwoStacks),
        ("Optimized", MyQueueOptimized)
    ]
    
    size = 1000
    
    for name, QueueClass in implementations:
        queue = QueueClass()
        
        # Fill queue
        for i in range(size):
            queue.push(i)
        
        # Count total elements stored
        total_elements = 0
        if hasattr(queue, 'input_stack'):
            total_elements += len(queue.input_stack)
        if hasattr(queue, 'output_stack'):
            total_elements += len(queue.output_stack)
        if hasattr(queue, 'stack1'):
            total_elements += len(queue.stack1)
        
        efficiency = (size / total_elements) * 100 if total_elements > 0 else 0
        print(f"  {name}: {total_elements} slots for {size} elements ({efficiency:.1f}% efficient)")

def benchmark_worst_case_scenarios():
    """Benchmark worst-case scenarios"""
    print("\n=== Worst-Case Scenario Analysis ===")
    
    import time
    
    # Scenario 1: Alternating push/pop (worst for input/output pattern)
    print("Scenario 1: Alternating push/pop")
    queue = MyQueueTwoStacks()
    
    start_time = time.time()
    for i in range(500):
        queue.push(i)
        if i > 0:
            queue.pop()
    elapsed = (time.time() - start_time) * 1000
    
    print(f"  500 alternating operations: {elapsed:.2f}ms")
    
    # Scenario 2: Batch operations (best case)
    print(f"\nScenario 2: Batch operations")
    queue = MyQueueTwoStacks()
    
    start_time = time.time()
    # Batch push
    for i in range(500):
        queue.push(i)
    # Batch pop
    for i in range(500):
        queue.pop()
    elapsed = (time.time() - start_time) * 1000
    
    print(f"  500 batch operations: {elapsed:.2f}ms")

if __name__ == "__main__":
    test_queue_basic_operations()
    test_queue_edge_cases()
    test_queue_fifo_behavior()
    test_queue_performance()
    test_amortized_analysis()
    demonstrate_stack_to_queue_concept()
    test_queue_with_mixed_operations()
    demonstrate_real_world_applications()
    test_memory_efficiency()
    benchmark_worst_case_scenarios()

"""
Queue using Stacks Design demonstrates key concepts:

Core Approaches:
1. Two Stacks (Input/Output) - Amortized O(1) with transfer pattern
2. Single Stack (Recursive) - O(n) operations but conceptually simple
3. Two Stacks (Pop Heavy) - O(1) pop/peek, O(n) push
4. Optimized - Track front element for O(1) peek
5. List Implementation - For conceptual understanding

Key Design Principles:
- Converting LIFO (stack) behavior to FIFO (queue) behavior
- Amortized analysis for performance understanding
- Trade-offs between different operation costs
- Memory efficiency considerations

Amortized Analysis:
- Individual operations may be O(n) but average over sequence is O(1)
- Transfer cost is amortized across multiple operations
- Worst case: alternating push/pop operations

Real-world Applications:
- Task scheduling and job queues
- Print queue management
- BFS algorithm implementation
- Request processing in web servers
- Event handling systems

The Input/Output pattern (Approach 1) is most commonly used
due to its optimal amortized performance characteristics.
"""
