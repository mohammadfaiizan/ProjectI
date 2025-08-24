"""
1670. Design Front Middle Back Queue - Multiple Approaches
Difficulty: Medium

Design a queue that supports push and pop operations in the front, middle, and back.

Implement the FrontMiddleBack class:
- FrontMiddleBack() Initializes the queue.
- void pushFront(int val) Adds val to the front of the queue.
- void pushMiddle(int val) Adds val to the middle of the queue.
- void pushBack(int val) Adds val to the back of the queue.
- int popFront() Removes the front element of the queue and returns it. If the queue is empty, return -1.
- int popMiddle() Removes the middle element of the queue and returns it. If the queue is empty, return -1.
- int popBack() Removes the back element of the queue and returns it. If the queue is empty, return -1.

Notice that when there are two middle position choices, the operation is performed on the frontmost middle position choice.
"""

from typing import List
from collections import deque

class FrontMiddleBackQueueDeque:
    """
    Approach 1: Single Deque Implementation
    
    Use single deque with index calculations for middle operations.
    
    Time: O(n) for middle operations, O(1) for front/back, Space: O(n)
    """
    
    def __init__(self):
        self.queue = deque()
    
    def pushFront(self, val: int) -> None:
        """Add element to front"""
        self.queue.appendleft(val)
    
    def pushMiddle(self, val: int) -> None:
        """Add element to middle"""
        mid_idx = len(self.queue) // 2
        # Convert deque to list, insert, convert back
        temp_list = list(self.queue)
        temp_list.insert(mid_idx, val)
        self.queue = deque(temp_list)
    
    def pushBack(self, val: int) -> None:
        """Add element to back"""
        self.queue.append(val)
    
    def popFront(self) -> int:
        """Remove and return front element"""
        if not self.queue:
            return -1
        return self.queue.popleft()
    
    def popMiddle(self) -> int:
        """Remove and return middle element"""
        if not self.queue:
            return -1
        
        mid_idx = (len(self.queue) - 1) // 2
        # Convert to list, pop, convert back
        temp_list = list(self.queue)
        result = temp_list.pop(mid_idx)
        self.queue = deque(temp_list)
        return result
    
    def popBack(self) -> int:
        """Remove and return back element"""
        if not self.queue:
            return -1
        return self.queue.pop()


class FrontMiddleBackQueueTwoDeques:
    """
    Approach 2: Two Deques Implementation (Optimal)
    
    Use two deques to efficiently handle middle operations.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    def __init__(self):
        self.left = deque()   # First half
        self.right = deque()  # Second half
        # Invariant: len(left) <= len(right) <= len(left) + 1
    
    def _balance(self) -> None:
        """Maintain balance between left and right deques"""
        if len(left := self.left) > len(right := self.right):
            # Move from left to right
            self.right.appendleft(self.left.pop())
        elif len(right) > len(left) + 1:
            # Move from right to left
            self.left.append(self.right.popleft())
    
    def pushFront(self, val: int) -> None:
        """Add element to front"""
        self.left.appendleft(val)
        self._balance()
    
    def pushMiddle(self, val: int) -> None:
        """Add element to middle"""
        if len(self.left) == len(self.right):
            self.left.append(val)
        else:
            self.right.appendleft(val)
    
    def pushBack(self, val: int) -> None:
        """Add element to back"""
        self.right.append(val)
        self._balance()
    
    def popFront(self) -> int:
        """Remove and return front element"""
        if not self.left and not self.right:
            return -1
        
        if self.left:
            result = self.left.popleft()
        else:
            result = self.right.popleft()
        
        self._balance()
        return result
    
    def popMiddle(self) -> int:
        """Remove and return middle element"""
        if not self.left and not self.right:
            return -1
        
        if len(self.left) == len(self.right):
            result = self.left.pop()
        else:
            result = self.right.popleft()
        
        self._balance()
        return result
    
    def popBack(self) -> int:
        """Remove and return back element"""
        if not self.left and not self.right:
            return -1
        
        if self.right:
            result = self.right.pop()
        else:
            result = self.left.pop()
        
        self._balance()
        return result


class FrontMiddleBackQueueList:
    """
    Approach 3: List Implementation
    
    Use Python list with direct indexing.
    
    Time: O(n) for all operations, Space: O(n)
    """
    
    def __init__(self):
        self.queue = []
    
    def pushFront(self, val: int) -> None:
        """Add element to front"""
        self.queue.insert(0, val)
    
    def pushMiddle(self, val: int) -> None:
        """Add element to middle"""
        mid_idx = len(self.queue) // 2
        self.queue.insert(mid_idx, val)
    
    def pushBack(self, val: int) -> None:
        """Add element to back"""
        self.queue.append(val)
    
    def popFront(self) -> int:
        """Remove and return front element"""
        if not self.queue:
            return -1
        return self.queue.pop(0)
    
    def popMiddle(self) -> int:
        """Remove and return middle element"""
        if not self.queue:
            return -1
        
        mid_idx = (len(self.queue) - 1) // 2
        return self.queue.pop(mid_idx)
    
    def popBack(self) -> int:
        """Remove and return back element"""
        if not self.queue:
            return -1
        return self.queue.pop()


class FrontMiddleBackQueueLinkedList:
    """
    Approach 4: Doubly Linked List Implementation
    
    Use doubly linked list for efficient insertions/deletions.
    
    Time: O(n) for middle operations, O(1) for front/back, Space: O(n)
    """
    
    class Node:
        def __init__(self, val: int):
            self.val = val
            self.prev = None
            self.next = None
    
    def __init__(self):
        self.head = self.Node(0)  # Dummy head
        self.tail = self.Node(0)  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
    
    def _find_middle_node(self):
        """Find middle node for operations"""
        if self.size == 0:
            return None
        
        mid_idx = self.size // 2
        current = self.head.next
        
        for _ in range(mid_idx):
            current = current.next
        
        return current
    
    def _insert_after(self, node, val: int) -> None:
        """Insert new node after given node"""
        new_node = self.Node(val)
        new_node.next = node.next
        new_node.prev = node
        node.next.prev = new_node
        node.next = new_node
        self.size += 1
    
    def _remove_node(self, node) -> int:
        """Remove and return value of given node"""
        if node == self.head or node == self.tail:
            return -1
        
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1
        return node.val
    
    def pushFront(self, val: int) -> None:
        """Add element to front"""
        self._insert_after(self.head, val)
    
    def pushMiddle(self, val: int) -> None:
        """Add element to middle"""
        if self.size == 0:
            self._insert_after(self.head, val)
        else:
            mid_node = self._find_middle_node()
            self._insert_after(mid_node.prev, val)
    
    def pushBack(self, val: int) -> None:
        """Add element to back"""
        self._insert_after(self.tail.prev, val)
    
    def popFront(self) -> int:
        """Remove and return front element"""
        if self.size == 0:
            return -1
        return self._remove_node(self.head.next)
    
    def popMiddle(self) -> int:
        """Remove and return middle element"""
        if self.size == 0:
            return -1
        
        mid_idx = (self.size - 1) // 2
        current = self.head.next
        
        for _ in range(mid_idx):
            current = current.next
        
        return self._remove_node(current)
    
    def popBack(self) -> int:
        """Remove and return back element"""
        if self.size == 0:
            return -1
        return self._remove_node(self.tail.prev)


def test_front_middle_back_queue_implementations():
    """Test front middle back queue implementations"""
    
    implementations = [
        ("Single Deque", FrontMiddleBackQueueDeque),
        ("Two Deques", FrontMiddleBackQueueTwoDeques),
        ("List", FrontMiddleBackQueueList),
        ("Linked List", FrontMiddleBackQueueLinkedList),
    ]
    
    test_cases = [
        {
            "operations": ["pushFront", "pushBack", "pushMiddle", "pushMiddle", "popFront", "popMiddle", "popMiddle", "popBack", "popFront"],
            "values": [1, 2, 3, 4, None, None, None, None, None],
            "expected": [None, None, None, None, 1, 3, 4, 2, -1],
            "description": "Example 1"
        },
        {
            "operations": ["pushMiddle", "pushMiddle", "popMiddle", "popMiddle"],
            "values": [1, 2, None, None],
            "expected": [None, None, 2, 1],
            "description": "Middle operations only"
        },
        {
            "operations": ["pushFront", "popBack", "pushBack", "popFront"],
            "values": [1, None, 2, None],
            "expected": [None, 1, None, 2],
            "description": "Front and back operations"
        },
    ]
    
    print("=== Testing Front Middle Back Queue Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- {impl_name} Implementation ---")
        
        for test_case in test_cases:
            try:
                queue = impl_class()
                results = []
                
                for i, op in enumerate(test_case["operations"]):
                    if "push" in op:
                        getattr(queue, op)(test_case["values"][i])
                        results.append(None)
                    else:  # pop operation
                        result = getattr(queue, op)()
                        results.append(result)
                
                expected = test_case["expected"]
                status = "✓" if results == expected else "✗"
                
                print(f"  {test_case['description']:20} | {status} | {results}")
                if results != expected:
                    print(f"    Expected: {expected}")
                
            except Exception as e:
                print(f"  {test_case['description']:20} | ERROR: {str(e)[:40]}")


def demonstrate_two_deques_approach():
    """Demonstrate two deques approach step by step"""
    print("\n=== Two Deques Approach Step-by-Step Demo ===")
    
    queue = FrontMiddleBackQueueTwoDeques()
    
    operations = [
        ("pushFront", 1),
        ("pushBack", 2),
        ("pushMiddle", 3),
        ("pushMiddle", 4),
        ("popFront", None),
        ("popMiddle", None),
        ("popBack", None),
    ]
    
    print("Strategy: Use two deques with balance invariant")
    print("Invariant: len(left) <= len(right) <= len(left) + 1")
    
    for op, val in operations:
        print(f"\nOperation: {op}({val if val is not None else ''})")
        print(f"  Before: left = {list(queue.left)}, right = {list(queue.right)}")
        
        if "push" in op:
            getattr(queue, op)(val)
            result = None
        else:
            result = getattr(queue, op)()
        
        print(f"  After:  left = {list(queue.left)}, right = {list(queue.right)}")
        if result is not None:
            print(f"  Returned: {result}")
        
        # Show combined queue
        combined = list(queue.left) + list(queue.right)
        print(f"  Combined queue: {combined}")


def visualize_middle_operations():
    """Visualize middle operations"""
    print("\n=== Middle Operations Visualization ===")
    
    queue = FrontMiddleBackQueueList()
    
    print("Understanding middle position:")
    print("- For even length n: middle index = n // 2")
    print("- For odd length n: middle index = n // 2")
    print("- For pop: middle index = (n - 1) // 2")
    
    # Build queue step by step
    elements = [1, 2, 3, 4, 5]
    
    for elem in elements:
        queue.pushBack(elem)
        n = len(queue.queue)
        push_mid_idx = n // 2
        pop_mid_idx = (n - 1) // 2
        
        print(f"\nQueue: {queue.queue} (length: {n})")
        print(f"  Push middle index: {push_mid_idx}")
        print(f"  Pop middle index: {pop_mid_idx}")
        
        if n > 0:
            print(f"  Middle element for pop: {queue.queue[pop_mid_idx]}")
    
    print(f"\nTesting middle operations:")
    
    # Test middle operations
    for i in range(3):
        print(f"\nBefore popMiddle: {queue.queue}")
        result = queue.popMiddle()
        print(f"After popMiddle: {queue.queue}, returned: {result}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Media player playlist
    print("1. Media Player Playlist Management:")
    playlist = FrontMiddleBackQueueTwoDeques()
    
    # Add songs
    songs = ["Song A", "Song B", "Song C", "Song D"]
    operations = [
        ("Add to front (priority)", "pushFront", "Priority Song"),
        ("Add to back (normal)", "pushBack", "Song A"),
        ("Add to back", "pushBack", "Song B"),
        ("Insert in middle", "pushMiddle", "Featured Song"),
        ("Add to back", "pushBack", "Song C"),
    ]
    
    for desc, op, song in operations:
        if "push" in op:
            # Simulate with numbers for simplicity
            song_id = hash(song) % 100
            getattr(playlist, op)(song_id)
            print(f"  {desc}: {song} (ID: {song_id})")
    
    combined = list(playlist.left) + list(playlist.right)
    print(f"  Final playlist order: {combined}")
    
    # Application 2: Task queue with priorities
    print(f"\n2. Task Queue with Priority Levels:")
    task_queue = FrontMiddleBackQueueTwoDeques()
    
    tasks = [
        ("High priority", "pushFront", 1),
        ("Normal task", "pushBack", 2),
        ("Medium priority", "pushMiddle", 3),
        ("Urgent task", "pushFront", 4),
    ]
    
    for desc, op, task_id in tasks:
        getattr(task_queue, op)(task_id)
        print(f"  {desc}: Task {task_id}")
    
    combined = list(task_queue.left) + list(task_queue.right)
    print(f"  Task execution order: {combined}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    operations = ["pushFront", "pushMiddle", "pushBack", "popFront", "popMiddle", "popBack"]
    
    approaches = [
        ("Single Deque", ["O(1)", "O(n)", "O(1)", "O(1)", "O(n)", "O(1)"]),
        ("Two Deques", ["O(1)", "O(1)", "O(1)", "O(1)", "O(1)", "O(1)"]),
        ("List", ["O(n)", "O(n)", "O(1)", "O(n)", "O(n)", "O(1)"]),
        ("Linked List", ["O(1)", "O(n)", "O(1)", "O(1)", "O(n)", "O(1)"]),
    ]
    
    print(f"{'Approach':<15} | {' | '.join(f'{op:<12}' for op in operations)}")
    print("-" * (15 + 13 * len(operations)))
    
    for approach, complexities in approaches:
        complexity_str = " | ".join(f"{comp:<12}" for comp in complexities)
        print(f"{approach:<15} | {complexity_str}")
    
    print(f"\nTwo Deques approach is optimal with O(1) for all operations")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    queue = FrontMiddleBackQueueTwoDeques()
    
    edge_cases = [
        ("Empty queue pops", ["popFront", "popMiddle", "popBack"], [-1, -1, -1]),
        ("Single element", ["pushMiddle", "popMiddle"], [None, 1]),
        ("Two elements", ["pushFront", "pushBack", "popMiddle", "popMiddle"], [None, None, 1, 2]),
        ("All front operations", ["pushFront", "pushFront", "popFront", "popFront"], [None, None, 2, 1]),
        ("All back operations", ["pushBack", "pushBack", "popBack", "popBack"], [None, None, 2, 1]),
    ]
    
    for description, ops, expected in edge_cases:
        queue = FrontMiddleBackQueueTwoDeques()
        results = []
        
        print(f"\n{description}:")
        
        for i, op in enumerate(ops):
            if "push" in op:
                getattr(queue, op)(i + 1)  # Use index as value
                results.append(None)
                print(f"  {op}({i + 1})")
            else:
                result = getattr(queue, op)()
                results.append(result)
                print(f"  {op}() -> {result}")
        
        status = "✓" if results == expected else "✗"
        print(f"  Result: {results} (expected: {expected}) {status}")


def benchmark_implementations():
    """Benchmark different implementations"""
    import time
    
    implementations = [
        ("Two Deques", FrontMiddleBackQueueTwoDeques),
        ("Single Deque", FrontMiddleBackQueueDeque),
        ("List", FrontMiddleBackQueueList),
    ]
    
    n_operations = 1000
    
    print(f"\n=== Performance Benchmark ===")
    print(f"Operations: {n_operations} mixed push/pop operations")
    
    for impl_name, impl_class in implementations:
        try:
            queue = impl_class()
            
            start_time = time.time()
            
            # Mixed operations
            for i in range(n_operations):
                op_type = i % 6
                
                if op_type == 0:
                    queue.pushFront(i)
                elif op_type == 1:
                    queue.pushMiddle(i)
                elif op_type == 2:
                    queue.pushBack(i)
                elif op_type == 3:
                    queue.popFront()
                elif op_type == 4:
                    queue.popMiddle()
                else:
                    queue.popBack()
            
            end_time = time.time()
            
            print(f"{impl_name:15} | Time: {end_time - start_time:.4f}s")
            
        except Exception as e:
            print(f"{impl_name:15} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_front_middle_back_queue_implementations()
    demonstrate_two_deques_approach()
    visualize_middle_operations()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    test_edge_cases()
    benchmark_implementations()

"""
Design Front Middle Back Queue demonstrates advanced queue applications
for multi-access data structures, including optimal two-deque implementation
and multiple approaches for efficient front/middle/back operations.
"""
