"""
1172. Dinner Plate Stacks - Multiple Approaches
Difficulty: Hard

You have an infinite number of stacks arranged in a row and numbered (left to right) from 0, 1, 2, ..., each of the stacks has the same maximum capacity.

Implement the DinnerPlates class:
- DinnerPlates(int capacity) Initializes the object with the maximum capacity of the stacks.
- void push(int val) Pushes the given integer val into the leftmost stack with size less than capacity.
- int pop() Returns the value at the top of the rightmost non-empty stack and removes it. If there is no non-empty stack, return -1.
- int popAtStack(int index) Returns the value at the top of the stack with the given index and removes it. If the stack at the given index is empty, return -1.
"""

from typing import List
import heapq

class DinnerPlatesHeap:
    """
    Approach 1: Heap-based Implementation (Optimal)
    
    Use min heap to track available stacks for push operations.
    
    Time: O(log n) for push, O(1) for pop, O(log n) for popAtStack, Space: O(n)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.stacks = []  # List of stacks
        self.available = []  # Min heap of available stack indices
    
    def push(self, val: int) -> None:
        """Push value to leftmost available stack"""
        # Clean up invalid indices from heap
        while self.available and (
            self.available[0] >= len(self.stacks) or 
            len(self.stacks[self.available[0]]) >= self.capacity
        ):
            heapq.heappop(self.available)
        
        if self.available:
            # Use existing stack with space
            idx = heapq.heappop(self.available)
            self.stacks[idx].append(val)
            
            # If stack still has space, add back to heap
            if len(self.stacks[idx]) < self.capacity:
                heapq.heappush(self.available, idx)
        else:
            # Create new stack
            self.stacks.append([val])
            
            # If new stack has space, add to heap
            if self.capacity > 1:
                heapq.heappush(self.available, len(self.stacks) - 1)
    
    def pop(self) -> int:
        """Pop from rightmost non-empty stack"""
        # Find rightmost non-empty stack
        while self.stacks and not self.stacks[-1]:
            self.stacks.pop()
        
        if not self.stacks:
            return -1
        
        val = self.stacks[-1].pop()
        
        # Add stack back to available heap if it now has space
        if len(self.stacks[-1]) < self.capacity:
            heapq.heappush(self.available, len(self.stacks) - 1)
        
        return val
    
    def popAtStack(self, index: int) -> int:
        """Pop from specific stack"""
        if index >= len(self.stacks) or not self.stacks[index]:
            return -1
        
        val = self.stacks[index].pop()
        
        # Add stack to available heap if it now has space
        if len(self.stacks[index]) < self.capacity:
            heapq.heappush(self.available, index)
        
        return val


class DinnerPlatesSimple:
    """
    Approach 2: Simple Implementation
    
    Use linear search for available stacks.
    
    Time: O(n) for push, O(1) for pop, O(1) for popAtStack, Space: O(n)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.stacks = []
    
    def push(self, val: int) -> None:
        """Push value to leftmost available stack"""
        # Find leftmost stack with space
        for i, stack in enumerate(self.stacks):
            if len(stack) < self.capacity:
                stack.append(val)
                return
        
        # No available stack, create new one
        self.stacks.append([val])
    
    def pop(self) -> int:
        """Pop from rightmost non-empty stack"""
        # Remove empty stacks from right
        while self.stacks and not self.stacks[-1]:
            self.stacks.pop()
        
        if not self.stacks:
            return -1
        
        return self.stacks[-1].pop()
    
    def popAtStack(self, index: int) -> int:
        """Pop from specific stack"""
        if index >= len(self.stacks) or not self.stacks[index]:
            return -1
        
        return self.stacks[index].pop()


class DinnerPlatesSet:
    """
    Approach 3: Set-based Implementation
    
    Use set to track available stack indices.
    
    Time: O(n) for push, O(1) for pop, O(1) for popAtStack, Space: O(n)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.stacks = []
        self.available = set()  # Set of available stack indices
    
    def push(self, val: int) -> None:
        """Push value to leftmost available stack"""
        if self.available:
            # Find leftmost available stack
            idx = min(self.available)
            self.stacks[idx].append(val)
            
            # Remove from available if now full
            if len(self.stacks[idx]) >= self.capacity:
                self.available.remove(idx)
        else:
            # Create new stack
            self.stacks.append([val])
            
            # Add to available if has space
            if self.capacity > 1:
                self.available.add(len(self.stacks) - 1)
    
    def pop(self) -> int:
        """Pop from rightmost non-empty stack"""
        # Remove empty stacks from right
        while self.stacks and not self.stacks[-1]:
            self.stacks.pop()
            # Remove from available set
            if len(self.stacks) in self.available:
                self.available.remove(len(self.stacks))
        
        if not self.stacks:
            return -1
        
        val = self.stacks[-1].pop()
        
        # Add to available if now has space
        if len(self.stacks[-1]) < self.capacity:
            self.available.add(len(self.stacks) - 1)
        
        return val
    
    def popAtStack(self, index: int) -> int:
        """Pop from specific stack"""
        if index >= len(self.stacks) or not self.stacks[index]:
            return -1
        
        val = self.stacks[index].pop()
        
        # Add to available if now has space
        if len(self.stacks[index]) < self.capacity:
            self.available.add(index)
        
        return val


def test_dinner_plates_implementations():
    """Test dinner plates implementations"""
    
    implementations = [
        ("Heap-based", DinnerPlatesHeap),
        ("Simple", DinnerPlatesSimple),
        ("Set-based", DinnerPlatesSet),
    ]
    
    test_cases = [
        {
            "capacity": 2,
            "operations": ["push", "push", "push", "push", "push", "popAtStack", "push", "push", "popAtStack", "popAtStack", "pop", "pop", "pop", "pop", "pop"],
            "values": [1, 2, 3, 4, 5, 0, 20, 21, 0, 2, None, None, None, None, None],
            "expected": [None, None, None, None, None, 2, None, None, 20, 21, 5, 4, 3, 1, -1],
            "description": "Example 1"
        },
        {
            "capacity": 1,
            "operations": ["push", "push", "popAtStack", "popAtStack", "push", "pop"],
            "values": [1, 2, 1, 0, 3, None],
            "expected": [None, None, 2, 1, None, 3],
            "description": "Capacity 1"
        },
        {
            "capacity": 3,
            "operations": ["push", "push", "push", "pop", "pop", "pop", "pop"],
            "values": [1, 2, 3, None, None, None, None],
            "expected": [None, None, None, 3, 2, 1, -1],
            "description": "Single stack"
        },
    ]
    
    print("=== Testing Dinner Plates Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- {impl_name} Implementation ---")
        
        for test_case in test_cases:
            try:
                dp = impl_class(test_case["capacity"])
                results = []
                
                for i, op in enumerate(test_case["operations"]):
                    if op == "push":
                        dp.push(test_case["values"][i])
                        results.append(None)
                    elif op == "pop":
                        result = dp.pop()
                        results.append(result)
                    elif op == "popAtStack":
                        result = dp.popAtStack(test_case["values"][i])
                        results.append(result)
                
                expected = test_case["expected"]
                status = "✓" if results == expected else "✗"
                
                print(f"  {test_case['description']:15} | {status} | {results}")
                if results != expected:
                    print(f"    Expected: {expected}")
                
            except Exception as e:
                print(f"  {test_case['description']:15} | ERROR: {str(e)[:40]}")


def demonstrate_heap_approach():
    """Demonstrate heap approach step by step"""
    print("\n=== Heap Approach Step-by-Step Demo ===")
    
    dp = DinnerPlatesHeap(2)
    
    operations = [
        ("push", 1),
        ("push", 2),
        ("push", 3),
        ("push", 4),
        ("popAtStack", 0),
        ("push", 5),
        ("pop", None),
    ]
    
    print("Strategy: Use min heap to track leftmost available stacks")
    
    for op, val in operations:
        print(f"\nOperation: {op}({val if val is not None else ''})")
        print(f"  Before: stacks = {dp.stacks}")
        print(f"  Available heap: {sorted(dp.available)}")
        
        if op == "push":
            dp.push(val)
        elif op == "pop":
            result = dp.pop()
            print(f"  Returned: {result}")
        elif op == "popAtStack":
            result = dp.popAtStack(val)
            print(f"  Returned: {result}")
        
        print(f"  After:  stacks = {dp.stacks}")
        print(f"  Available heap: {sorted(dp.available)}")


def visualize_stack_operations():
    """Visualize stack operations"""
    print("\n=== Stack Operations Visualization ===")
    
    dp = DinnerPlatesSimple(3)
    
    # Build multiple stacks
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    
    print("Building stacks with capacity 3:")
    for val in values:
        dp.push(val)
        print(f"  Pushed {val}: {dp.stacks}")
    
    print(f"\nTesting popAtStack operations:")
    
    # Test popAtStack at different indices
    pop_tests = [1, 0, 2, 1]
    
    for idx in pop_tests:
        result = dp.popAtStack(idx)
        print(f"  popAtStack({idx}): returned {result}, stacks = {dp.stacks}")
    
    print(f"\nTesting regular pop operations:")
    
    # Test regular pop
    for _ in range(3):
        result = dp.pop()
        print(f"  pop(): returned {result}, stacks = {dp.stacks}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Document page management
    print("1. Document Page Management System:")
    page_manager = DinnerPlatesHeap(5)  # 5 pages per section
    
    # Add pages to sections
    pages = [f"Page{i}" for i in range(1, 13)]
    
    print("  Adding pages to sections:")
    for i, page in enumerate(pages):
        page_manager.push(hash(page) % 100)  # Use hash for demo
        print(f"    Added {page}")
    
    print(f"  Document structure: {page_manager.stacks}")
    
    # Remove specific page
    removed = page_manager.popAtStack(1)
    print(f"  Removed page from section 1: {removed}")
    print(f"  Updated structure: {page_manager.stacks}")
    
    # Application 2: Server load balancing
    print(f"\n2. Server Load Balancing:")
    load_balancer = DinnerPlatesHeap(3)  # 3 connections per server
    
    # Simulate incoming connections
    connections = list(range(1, 10))
    
    print("  Distributing connections to servers:")
    for conn in connections:
        load_balancer.push(conn)
    
    print(f"  Server loads: {load_balancer.stacks}")
    
    # Remove connection from specific server
    removed_conn = load_balancer.popAtStack(0)
    print(f"  Removed connection {removed_conn} from server 0")
    print(f"  Updated loads: {load_balancer.stacks}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Heap-based", "O(log n)", "O(1)", "O(log n)", "Optimal for push-heavy workloads"),
        ("Simple", "O(n)", "O(1)", "O(1)", "Simple but slow push operations"),
        ("Set-based", "O(n)", "O(1)", "O(1)", "Better than simple for sparse stacks"),
    ]
    
    print(f"{'Approach':<15} | {'Push':<10} | {'Pop':<8} | {'PopAt':<10} | {'Notes'}")
    print("-" * 70)
    
    for approach, push_time, pop_time, pop_at_time, notes in approaches:
        print(f"{approach:<15} | {push_time:<10} | {pop_time:<8} | {pop_at_time:<10} | {notes}")
    
    print(f"\nHeap-based approach is optimal for most use cases")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    dp = DinnerPlatesHeap(2)
    
    edge_cases = [
        ("Empty pop", lambda: dp.pop(), -1),
        ("Invalid popAtStack", lambda: dp.popAtStack(5), -1),
        ("Capacity 1", None, None),
        ("Large capacity", None, None),
    ]
    
    # Test basic edge cases
    for description, operation, expected in edge_cases[:2]:
        try:
            result = operation()
            status = "✓" if result == expected else "✗"
            print(f"{description:20} | {status} | Result: {result}")
        except Exception as e:
            print(f"{description:20} | ERROR: {str(e)[:30]}")
    
    # Test capacity 1
    print(f"\nCapacity 1 test:")
    dp1 = DinnerPlatesHeap(1)
    
    for i in range(3):
        dp1.push(i)
    
    print(f"  After pushing 0,1,2: {dp1.stacks}")
    
    result = dp1.popAtStack(1)
    print(f"  popAtStack(1): {result}")
    print(f"  Stacks: {dp1.stacks}")
    
    # Test large capacity
    print(f"\nLarge capacity test:")
    dp_large = DinnerPlatesHeap(100)
    
    for i in range(5):
        dp_large.push(i)
    
    print(f"  Single stack with 5 elements: {dp_large.stacks}")


if __name__ == "__main__":
    test_dinner_plates_implementations()
    demonstrate_heap_approach()
    visualize_stack_operations()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    test_edge_cases()

"""
Dinner Plate Stacks demonstrates advanced queue/stack applications
for multi-stack management systems, including heap-based optimization
and multiple approaches for efficient leftmost insertion strategies.
"""
