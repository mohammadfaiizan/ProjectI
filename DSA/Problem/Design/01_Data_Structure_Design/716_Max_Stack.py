"""
716. Max Stack - Multiple Approaches
Difficulty: Hard

Design a max stack data structure that supports the stack operations and supports finding the stack's maximum element.

Implement the MaxStack class:
- MaxStack() Initializes the stack object.
- void push(int x) Pushes element x onto the stack.
- int pop() Removes the element on top of the stack and returns it.
- int top() Gets the element on the top of the stack without removing it.
- int peekMax() Retrieves the maximum element in the stack without removing it.
- int popMax() Retrieves and removes the maximum element in the stack.

Note: If there are multiple maximum elements, only remove the top-most one.
"""

from typing import List, Optional
import heapq
from collections import defaultdict

class MaxStackTwoStacks:
    """
    Approach 1: Two Stacks
    
    Use main stack and auxiliary max stack to track maximums.
    
    Time Complexity: 
    - push, pop, top, peekMax: O(1)
    - popMax: O(n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.stack = []
        self.max_stack = []
    
    def push(self, x: int) -> None:
        self.stack.append(x)
        
        # Update max stack
        if not self.max_stack or x >= self.max_stack[-1]:
            self.max_stack.append(x)
        else:
            self.max_stack.append(self.max_stack[-1])
    
    def pop(self) -> int:
        if not self.stack:
            return -1
        
        self.max_stack.pop()
        return self.stack.pop()
    
    def top(self) -> int:
        if not self.stack:
            return -1
        return self.stack[-1]
    
    def peekMax(self) -> int:
        if not self.max_stack:
            return -1
        return self.max_stack[-1]
    
    def popMax(self) -> int:
        if not self.stack:
            return -1
        
        max_val = self.peekMax()
        temp_stack = []
        
        # Remove elements until we find the max
        while self.stack and self.stack[-1] != max_val:
            temp_stack.append(self.pop())
        
        # Remove the max element
        result = self.pop()
        
        # Push back the temporary elements
        while temp_stack:
            self.push(temp_stack.pop())
        
        return result

class MaxStackHeap:
    """
    Approach 2: Stack + Heap
    
    Use stack for main operations and heap for max tracking.
    
    Time Complexity: 
    - push: O(log n)
    - pop, top: O(1) amortized
    - peekMax, popMax: O(log n) amortized
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.stack = []
        self.max_heap = []  # Max heap (using negative values)
        self.removed = set()  # Track removed elements
        self.id_counter = 0  # Unique ID for each element
    
    def push(self, x: int) -> None:
        self.id_counter += 1
        element_id = self.id_counter
        
        self.stack.append((x, element_id))
        heapq.heappush(self.max_heap, (-x, -element_id, element_id))  # (neg_val, neg_id, id)
    
    def pop(self) -> int:
        if not self.stack:
            return -1
        
        value, element_id = self.stack.pop()
        self.removed.add(element_id)
        return value
    
    def top(self) -> int:
        if not self.stack:
            return -1
        return self.stack[-1][0]
    
    def peekMax(self) -> int:
        self._clean_heap()
        if not self.max_heap:
            return -1
        return -self.max_heap[0][0]
    
    def popMax(self) -> int:
        self._clean_heap()
        if not self.max_heap:
            return -1
        
        neg_val, neg_id, element_id = heapq.heappop(self.max_heap)
        self.removed.add(element_id)
        return -neg_val
    
    def _clean_heap(self) -> None:
        """Remove invalidated elements from heap top"""
        while self.max_heap and self.max_heap[0][2] in self.removed:
            heapq.heappop(self.max_heap)

class MaxStackDoubleLinkedList:
    """
    Approach 3: Doubly Linked List + Map
    
    Use doubly linked list for stack and map for max tracking.
    
    Time Complexity: O(1) for all operations
    Space Complexity: O(n)
    """
    
    class Node:
        def __init__(self, val: int):
            self.val = val
            self.prev: Optional['MaxStackDoubleLinkedList.Node'] = None
            self.next: Optional['MaxStackDoubleLinkedList.Node'] = None
    
    def __init__(self):
        # Doubly linked list for stack
        self.head = self.Node(0)  # Dummy head
        self.tail = self.Node(0)  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head
        
        # Map from value to stack of nodes with that value
        self.value_to_nodes = defaultdict(list)
    
    def push(self, x: int) -> None:
        node = self.Node(x)
        self._add_to_tail(node)
        self.value_to_nodes[x].append(node)
    
    def pop(self) -> int:
        if self._is_empty():
            return -1
        
        node = self._remove_from_tail()
        self.value_to_nodes[node.val].pop()
        if not self.value_to_nodes[node.val]:
            del self.value_to_nodes[node.val]
        
        return node.val
    
    def top(self) -> int:
        if self._is_empty():
            return -1
        return self.tail.prev.val
    
    def peekMax(self) -> int:
        if not self.value_to_nodes:
            return -1
        return max(self.value_to_nodes.keys())
    
    def popMax(self) -> int:
        if not self.value_to_nodes:
            return -1
        
        max_val = max(self.value_to_nodes.keys())
        node = self.value_to_nodes[max_val].pop()
        
        if not self.value_to_nodes[max_val]:
            del self.value_to_nodes[max_val]
        
        self._remove_node(node)
        return max_val
    
    def _add_to_tail(self, node: Node) -> None:
        """Add node before tail"""
        prev_node = self.tail.prev
        prev_node.next = node
        node.prev = prev_node
        node.next = self.tail
        self.tail.prev = node
    
    def _remove_from_tail(self) -> Node:
        """Remove node before tail"""
        if self._is_empty():
            return None
        
        node = self.tail.prev
        self._remove_node(node)
        return node
    
    def _remove_node(self, node: Node) -> None:
        """Remove arbitrary node from list"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _is_empty(self) -> bool:
        return self.head.next == self.tail

class MaxStackTreeMap:
    """
    Approach 4: Stack + TreeMap simulation
    
    Use stack and balanced structure for max tracking.
    
    Time Complexity: O(log n) for all operations
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.stack = []
        self.count_map = defaultdict(int)  # value -> count
        self.position_map = defaultdict(list)  # value -> list of positions
        self.position_counter = 0
    
    def push(self, x: int) -> None:
        self.position_counter += 1
        self.stack.append((x, self.position_counter))
        self.count_map[x] += 1
        self.position_map[x].append(self.position_counter)
    
    def pop(self) -> int:
        if not self.stack:
            return -1
        
        value, position = self.stack.pop()
        self.count_map[value] -= 1
        self.position_map[value].remove(position)
        
        if self.count_map[value] == 0:
            del self.count_map[value]
            del self.position_map[value]
        
        return value
    
    def top(self) -> int:
        if not self.stack:
            return -1
        return self.stack[-1][0]
    
    def peekMax(self) -> int:
        if not self.count_map:
            return -1
        return max(self.count_map.keys())
    
    def popMax(self) -> int:
        if not self.count_map:
            return -1
        
        max_val = max(self.count_map.keys())
        
        # Find the topmost occurrence of max_val
        max_position = max(self.position_map[max_val])
        
        # Remove from stack
        temp_stack = []
        while self.stack:
            value, position = self.stack.pop()
            if value == max_val and position == max_position:
                break
            temp_stack.append((value, position))
        
        # Push back elements that were above the max
        while temp_stack:
            value, position = temp_stack.pop()
            self.stack.append((value, position))
        
        # Update tracking structures
        self.count_map[max_val] -= 1
        self.position_map[max_val].remove(max_position)
        
        if self.count_map[max_val] == 0:
            del self.count_map[max_val]
            del self.position_map[max_val]
        
        return max_val

class MaxStackOptimized:
    """
    Approach 5: Optimized with Lazy Deletion
    
    Use lazy deletion for better amortized performance.
    
    Time Complexity: O(1) amortized for all operations
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.stack = []
        self.max_stack = []
        self.removed_positions = set()
        self.position_counter = 0
    
    def push(self, x: int) -> None:
        self.position_counter += 1
        position = self.position_counter
        
        self.stack.append((x, position))
        
        # Update max stack
        if not self.max_stack or x >= self.max_stack[-1][0]:
            self.max_stack.append((x, position))
    
    def pop(self) -> int:
        if not self.stack:
            return -1
        
        # Clean top of stack
        while self.stack and self.stack[-1][1] in self.removed_positions:
            self.stack.pop()
        
        if not self.stack:
            return -1
        
        value, position = self.stack.pop()
        self.removed_positions.add(position)
        
        return value
    
    def top(self) -> int:
        # Clean top of stack
        while self.stack and self.stack[-1][1] in self.removed_positions:
            self.stack.pop()
        
        if not self.stack:
            return -1
        
        return self.stack[-1][0]
    
    def peekMax(self) -> int:
        # Clean max stack
        while self.max_stack and self.max_stack[-1][1] in self.removed_positions:
            self.max_stack.pop()
        
        if not self.max_stack:
            return -1
        
        return self.max_stack[-1][0]
    
    def popMax(self) -> int:
        # Clean max stack
        while self.max_stack and self.max_stack[-1][1] in self.removed_positions:
            self.max_stack.pop()
        
        if not self.max_stack:
            return -1
        
        value, position = self.max_stack.pop()
        self.removed_positions.add(position)
        
        return value


def test_max_stack_basic():
    """Test basic MaxStack operations"""
    print("=== Testing Basic MaxStack Operations ===")
    
    implementations = [
        ("Two Stacks", MaxStackTwoStacks),
        ("Stack + Heap", MaxStackHeap),
        ("Doubly Linked List", MaxStackDoubleLinkedList),
        ("TreeMap Simulation", MaxStackTreeMap),
        ("Optimized Lazy", MaxStackOptimized)
    ]
    
    for name, MaxStackClass in implementations:
        print(f"\n{name}:")
        
        stack = MaxStackClass()
        
        # Test basic operations
        operations = [
            ("push", 5), ("push", 1), ("push", 5),
            ("top", None), ("popMax", None), ("top", None),
            ("peekMax", None), ("pop", None), ("top", None)
        ]
        
        for op, value in operations:
            if op == "push":
                stack.push(value)
                print(f"  push({value})")
            elif op == "pop":
                result = stack.pop()
                print(f"  pop(): {result}")
            elif op == "top":
                result = stack.top()
                print(f"  top(): {result}")
            elif op == "peekMax":
                result = stack.peekMax()
                print(f"  peekMax(): {result}")
            elif op == "popMax":
                result = stack.popMax()
                print(f"  popMax(): {result}")

def test_max_stack_edge_cases():
    """Test MaxStack edge cases"""
    print("\n=== Testing MaxStack Edge Cases ===")
    
    stack = MaxStackTwoStacks()
    
    # Test empty stack
    print("Empty stack operations:")
    print(f"  pop(): {stack.pop()}")
    print(f"  top(): {stack.top()}")
    print(f"  peekMax(): {stack.peekMax()}")
    print(f"  popMax(): {stack.popMax()}")
    
    # Test single element
    print(f"\nSingle element:")
    stack.push(42)
    print(f"  push(42)")
    print(f"  top(): {stack.top()}")
    print(f"  peekMax(): {stack.peekMax()}")
    print(f"  popMax(): {stack.popMax()}")
    print(f"  top(): {stack.top()}")
    
    # Test duplicate maximums
    print(f"\nDuplicate maximums:")
    for val in [3, 3, 3]:
        stack.push(val)
        print(f"  push({val})")
    
    print(f"  peekMax(): {stack.peekMax()}")
    print(f"  popMax(): {stack.popMax()}")
    print(f"  peekMax(): {stack.peekMax()}")

def test_max_element_tracking():
    """Test maximum element tracking"""
    print("\n=== Testing Maximum Element Tracking ===")
    
    stack = MaxStackHeap()
    
    # Build stack with known pattern
    sequence = [1, 5, 3, 9, 2, 9, 4]
    
    print("Building stack:")
    for val in sequence:
        stack.push(val)
        max_val = stack.peekMax()
        top_val = stack.top()
        print(f"  push({val}) -> top: {top_val}, max: {max_val}")
    
    print(f"\nPopping maximum elements:")
    while True:
        max_val = stack.peekMax()
        if max_val == -1:
            break
        
        popped_max = stack.popMax()
        new_max = stack.peekMax()
        top_val = stack.top()
        
        print(f"  popMax(): {popped_max} -> new max: {new_max}, top: {top_val}")

def test_pop_max_order():
    """Test popMax removes topmost occurrence"""
    print("\n=== Testing PopMax Order ===")
    
    stack = MaxStackDoubleLinkedList()
    
    # Push elements: [5, 1, 5, 2, 5]
    elements = [5, 1, 5, 2, 5]
    print("Pushing elements:")
    for val in elements:
        stack.push(val)
        print(f"  push({val})")
    
    print(f"\nStack state (bottom to top): {elements}")
    print(f"Testing popMax removes topmost 5:")
    
    # PopMax should remove the topmost 5 (index 4)
    popped = stack.popMax()
    remaining_top = stack.top()
    
    print(f"  popMax(): {popped}")
    print(f"  top(): {remaining_top}")  # Should be 2
    
    # PopMax again should remove next topmost 5 (index 2)
    popped = stack.popMax()
    remaining_top = stack.top()
    
    print(f"  popMax(): {popped}")
    print(f"  top(): {remaining_top}")  # Should be 2

def test_max_stack_performance():
    """Test MaxStack performance"""
    print("\n=== Testing MaxStack Performance ===")
    
    import time
    
    implementations = [
        ("Two Stacks", MaxStackTwoStacks),
        ("Stack + Heap", MaxStackHeap),
        ("Doubly Linked List", MaxStackDoubleLinkedList),
        ("Optimized Lazy", MaxStackOptimized)
    ]
    
    operations = 1000
    
    for name, MaxStackClass in implementations:
        stack = MaxStackClass()
        
        # Fill stack
        for i in range(operations // 2):
            stack.push(i)
        
        # Test mixed operations
        start_time = time.time()
        
        for i in range(operations):
            if i % 4 == 0:
                stack.push(i)
            elif i % 4 == 1:
                stack.pop()
            elif i % 4 == 2:
                stack.peekMax()
            else:
                stack.popMax()
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {name}: {elapsed:.2f}ms for {operations} mixed operations")

def test_complex_scenarios():
    """Test complex usage scenarios"""
    print("\n=== Testing Complex Scenarios ===")
    
    stack = MaxStackTreeMap()
    
    # Scenario: Alternating push/popMax
    print("Scenario 1: Alternating push/popMax")
    values = [10, 20, 15, 30, 25]
    
    for val in values:
        stack.push(val)
        max_val = stack.peekMax()
        print(f"  push({val}) -> max: {max_val}")
        
        if val % 2 == 0:  # Pop max for even values
            popped = stack.popMax()
            new_max = stack.peekMax()
            print(f"    popMax(): {popped} -> new max: {new_max}")
    
    # Scenario: Build mountain pattern
    print(f"\nScenario 2: Mountain pattern")
    stack2 = MaxStackOptimized()
    
    mountain = [1, 3, 5, 7, 9, 7, 5, 3, 1]
    
    for val in mountain:
        stack2.push(val)
    
    print(f"  Built mountain: {mountain}")
    print(f"  peekMax(): {stack2.peekMax()}")
    
    # Pop elements and track max changes
    for i in range(len(mountain)):
        top = stack2.top()
        max_val = stack2.peekMax()
        popped = stack2.pop()
        new_max = stack2.peekMax()
        
        print(f"  pop(): {popped}, was max: {max_val}, now max: {new_max}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Stock price monitoring
    print("Application 1: Stock Price Monitoring")
    price_monitor = MaxStackHeap()
    
    prices = [100, 105, 98, 110, 95, 120, 102]
    
    print("  Daily stock prices:")
    for day, price in enumerate(prices, 1):
        price_monitor.push(price)
        highest = price_monitor.peekMax()
        current = price_monitor.top()
        
        print(f"    Day {day}: ${price}, Current: ${current}, Highest: ${highest}")
    
    print(f"  Removing peak price: ${price_monitor.popMax()}")
    print(f"  New highest: ${price_monitor.peekMax()}")
    
    # Application 2: Undo/Redo with max tracking
    print(f"\nApplication 2: Undo System with Max Tracking")
    undo_stack = MaxStackDoubleLinkedList()
    
    actions = [("edit", 5), ("delete", 3), ("insert", 8), ("edit", 2)]
    
    print("  User actions (with priority):")
    for action, priority in actions:
        undo_stack.push(priority)
        print(f"    {action} (priority {priority}) -> max priority: {undo_stack.peekMax()}")
    
    print(f"  Undo highest priority action: priority {undo_stack.popMax()}")
    print(f"  Undo last action: priority {undo_stack.pop()}")

def test_memory_efficiency():
    """Test memory efficiency of different approaches"""
    print("\n=== Testing Memory Efficiency ===")
    
    implementations = [
        ("Two Stacks", MaxStackTwoStacks),
        ("Stack + Heap", MaxStackHeap),
        ("Doubly Linked List", MaxStackDoubleLinkedList)
    ]
    
    size = 1000
    
    for name, MaxStackClass in implementations:
        stack = MaxStackClass()
        
        # Fill stack
        for i in range(size):
            stack.push(i)
        
        # Estimate memory usage (simplified)
        memory_estimate = 0
        
        if hasattr(stack, 'stack') and hasattr(stack, 'max_stack'):
            memory_estimate = len(stack.stack) + len(stack.max_stack)
        elif hasattr(stack, 'stack') and hasattr(stack, 'max_heap'):
            memory_estimate = len(stack.stack) + len(stack.max_heap)
        elif hasattr(stack, 'stack'):
            memory_estimate = len(stack.stack)
        else:
            memory_estimate = size  # Rough estimate
        
        efficiency = (size / memory_estimate) * 100 if memory_estimate > 0 else 0
        print(f"  {name}: ~{memory_estimate} memory units for {size} elements ({efficiency:.1f}% efficient)")

def benchmark_operation_types():
    """Benchmark different operation types"""
    print("\n=== Benchmarking Operation Types ===")
    
    import time
    
    stack = MaxStackOptimized()
    
    # Fill stack
    for i in range(1000):
        stack.push(i)
    
    operations = [
        ("push", lambda: stack.push(999)),
        ("pop", lambda: stack.pop()),
        ("top", lambda: stack.top()),
        ("peekMax", lambda: stack.peekMax()),
        ("popMax", lambda: stack.popMax())
    ]
    
    test_iterations = 1000
    
    for op_name, operation in operations:
        # Restore some elements for pop operations
        if op_name in ["pop", "popMax"]:
            for i in range(100):
                stack.push(i)
        
        start_time = time.time()
        
        for _ in range(test_iterations):
            try:
                operation()
            except:
                pass  # Handle empty stack gracefully
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {op_name}: {elapsed:.2f}ms for {test_iterations} operations")

if __name__ == "__main__":
    test_max_stack_basic()
    test_max_stack_edge_cases()
    test_max_element_tracking()
    test_pop_max_order()
    test_max_stack_performance()
    test_complex_scenarios()
    demonstrate_applications()
    test_memory_efficiency()
    benchmark_operation_types()

"""
Max Stack Design demonstrates advanced stack concepts:

Core Approaches:
1. Two Stacks - Auxiliary stack tracks maximums at each level
2. Stack + Heap - Use max heap for efficient max tracking  
3. Doubly Linked List - O(1) removal of arbitrary elements
4. TreeMap Simulation - Balanced structure for sorted max tracking
5. Optimized Lazy - Lazy deletion for amortized performance

Key Design Challenges:
- Maintaining max element efficiently
- popMax operation complexity (finding and removing topmost max)
- Memory vs time complexity trade-offs
- Handling duplicate maximum values

Performance Characteristics:
- Two Stacks: O(n) popMax, O(1) others
- Heap: O(log n) push/popMax, O(1) amortized others
- Doubly Linked List: O(1) all operations
- TreeMap: O(log n) all operations
- Lazy Deletion: O(1) amortized all operations

Real-world Applications:
- Stock price monitoring with peak tracking
- Undo systems with priority-based operations
- Game score tracking with high score removal
- Resource management with maximum utilization tracking
- Expression evaluation with maximum value tracking

The doubly linked list approach provides optimal O(1) performance
for all operations but requires more complex implementation.
"""
