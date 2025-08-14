"""
Queue Interview Problems - Most Asked Questions
===============================================

Topics: Common queue interview questions with solutions and explanations
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix, Uber, LinkedIn
Difficulty: Easy to Hard
Time Complexity: Usually O(n) with queue optimization
Space Complexity: O(n) for queue storage
"""

from typing import List, Optional, Dict, Tuple, Any
from collections import deque
import heapq

class QueueInterviewProblems:
    
    def __init__(self):
        """Initialize with problem tracking"""
        self.problem_count = 0
        self.solution_steps = []
    
    # ==========================================
    # 1. CLASSIC QUEUE INTERVIEW PROBLEMS
    # ==========================================
    
    def implement_queue_using_stacks(self):
        """
        Implement Queue using Two Stacks
        
        Company: Amazon, Microsoft, Google, Facebook
        Difficulty: Easy
        Time: O(1) amortized for all operations, Space: O(n)
        
        Key insight: Use two stacks to simulate FIFO behavior
        """
        
        class QueueUsingStacks:
            def __init__(self):
                self.input_stack = []
                self.output_stack = []
                self.operation_count = 0
            
            def enqueue(self, x: int) -> None:
                """Add element to rear of queue"""
                self.operation_count += 1
                self.input_stack.append(x)
                
                print(f"Enqueue operation #{self.operation_count}: Add {x}")
                print(f"   Input stack: {self.input_stack}")
                print(f"   Output stack: {self.output_stack}")
            
            def dequeue(self) -> int:
                """Remove element from front of queue"""
                self.operation_count += 1
                print(f"Dequeue operation #{self.operation_count}:")
                
                # If output stack is empty, transfer all from input stack
                if not self.output_stack:
                    if not self.input_stack:
                        print("   ✗ Queue is empty!")
                        return -1
                    
                    print("   Output stack empty, transferring from input stack:")
                    while self.input_stack:
                        element = self.input_stack.pop()
                        self.output_stack.append(element)
                        print(f"     Move {element} from input to output")
                    
                    print(f"   After transfer - Input: {self.input_stack}, Output: {self.output_stack}")
                
                # Pop from output stack (FIFO order)
                result = self.output_stack.pop()
                print(f"   Dequeue: {result}")
                print(f"   Input stack: {self.input_stack}")
                print(f"   Output stack: {self.output_stack}")
                
                return result
            
            def peek(self) -> int:
                """Get front element without removing"""
                if not self.output_stack:
                    if not self.input_stack:
                        print("   Queue is empty!")
                        return -1
                    
                    # Transfer to output stack
                    while self.input_stack:
                        self.output_stack.append(self.input_stack.pop())
                
                front = self.output_stack[-1]
                print(f"   Front element: {front}")
                return front
            
            def empty(self) -> bool:
                """Check if queue is empty"""
                is_empty = len(self.input_stack) == 0 and len(self.output_stack) == 0
                print(f"   Is empty: {is_empty}")
                return is_empty
        
        print("=== QUEUE USING STACKS IMPLEMENTATION ===")
        queue = QueueUsingStacks()
        
        # Test operations
        operations = [
            ('enqueue', 1), ('enqueue', 2), ('enqueue', 3),
            ('peek', None), ('dequeue', None),
            ('enqueue', 4), ('dequeue', None), ('dequeue', None),
            ('dequeue', None), ('empty', None)
        ]
        
        for op, value in operations:
            print(f"\nOperation: {op}" + (f"({value})" if value is not None else "()"))
            if op == 'enqueue':
                queue.enqueue(value)
            elif op == 'dequeue':
                queue.dequeue()
            elif op == 'peek':
                queue.peek()
            elif op == 'empty':
                queue.empty()
        
        return queue
    
    def implement_stack_using_queues(self):
        """
        Implement Stack using Two Queues
        
        Company: Amazon, Microsoft
        Difficulty: Easy
        Time: O(n) for push, O(1) for pop, Space: O(n)
        """
        
        class StackUsingQueues:
            def __init__(self):
                self.q1 = deque()
                self.q2 = deque()
                self.operation_count = 0
            
            def push(self, x: int) -> None:
                """Add element to top of stack"""
                self.operation_count += 1
                print(f"Push operation #{self.operation_count}: Add {x}")
                
                # Add to q2
                self.q2.append(x)
                print(f"   Add {x} to q2: {list(self.q2)}")
                
                # Move all elements from q1 to q2
                while self.q1:
                    element = self.q1.popleft()
                    self.q2.append(element)
                    print(f"   Move {element} from q1 to q2")
                
                # Swap q1 and q2
                self.q1, self.q2 = self.q2, self.q1
                print(f"   After swap - q1: {list(self.q1)}, q2: {list(self.q2)}")
            
            def pop(self) -> int:
                """Remove element from top of stack"""
                self.operation_count += 1
                print(f"Pop operation #{self.operation_count}:")
                
                if not self.q1:
                    print("   ✗ Stack is empty!")
                    return -1
                
                element = self.q1.popleft()
                print(f"   Pop {element}: q1 becomes {list(self.q1)}")
                return element
            
            def top(self) -> int:
                """Get top element without removing"""
                if not self.q1:
                    print("   Stack is empty!")
                    return -1
                
                element = self.q1[0]
                print(f"   Top element: {element}")
                return element
            
            def empty(self) -> bool:
                """Check if stack is empty"""
                is_empty = len(self.q1) == 0
                print(f"   Is empty: {is_empty}")
                return is_empty
        
        print("=== STACK USING QUEUES IMPLEMENTATION ===")
        stack = StackUsingQueues()
        
        # Test operations
        operations = [
            ('push', 1), ('push', 2), ('push', 3),
            ('top', None), ('pop', None),
            ('push', 4), ('top', None), ('pop', None),
            ('pop', None), ('pop', None), ('empty', None)
        ]
        
        for op, value in operations:
            print(f"\nOperation: {op}" + (f"({value})" if value is not None else "()"))
            if op == 'push':
                stack.push(value)
            elif op == 'pop':
                stack.pop()
            elif op == 'top':
                stack.top()
            elif op == 'empty':
                stack.empty()
        
        return stack
    
    def circular_queue_design(self):
        """
        Design Circular Queue
        
        Company: Facebook, Amazon
        Difficulty: Medium
        Time: O(1) for all operations, Space: O(k)
        
        Efficient implementation of circular queue with fixed capacity
        """
        
        class CircularQueue:
            def __init__(self, k: int):
                self.capacity = k
                self.queue = [0] * k
                self.front = 0
                self.rear = -1
                self.size = 0
                self.operation_count = 0
            
            def enQueue(self, value: int) -> bool:
                """Insert element into circular queue"""
                self.operation_count += 1
                print(f"EnQueue operation #{self.operation_count}: Add {value}")
                
                if self.isFull():
                    print(f"   ✗ Queue is full! Cannot add {value}")
                    return False
                
                self.rear = (self.rear + 1) % self.capacity
                self.queue[self.rear] = value
                self.size += 1
                
                print(f"   ✓ Added {value} at index {self.rear}")
                print(f"   Queue: {self.get_elements()}")
                print(f"   Front: {self.front}, Rear: {self.rear}, Size: {self.size}")
                return True
            
            def deQueue(self) -> bool:
                """Delete element from circular queue"""
                self.operation_count += 1
                print(f"DeQueue operation #{self.operation_count}:")
                
                if self.isEmpty():
                    print(f"   ✗ Queue is empty! Cannot dequeue")
                    return False
                
                removed_value = self.queue[self.front]
                self.front = (self.front + 1) % self.capacity
                self.size -= 1
                
                print(f"   ✓ Removed {removed_value}")
                print(f"   Queue: {self.get_elements()}")
                print(f"   Front: {self.front}, Rear: {self.rear}, Size: {self.size}")
                return True
            
            def Front(self) -> int:
                """Get front element"""
                if self.isEmpty():
                    print("   Queue is empty!")
                    return -1
                
                front_value = self.queue[self.front]
                print(f"   Front element: {front_value}")
                return front_value
            
            def Rear(self) -> int:
                """Get rear element"""
                if self.isEmpty():
                    print("   Queue is empty!")
                    return -1
                
                rear_value = self.queue[self.rear]
                print(f"   Rear element: {rear_value}")
                return rear_value
            
            def isEmpty(self) -> bool:
                """Check if queue is empty"""
                is_empty = self.size == 0
                print(f"   Is empty: {is_empty}")
                return is_empty
            
            def isFull(self) -> bool:
                """Check if queue is full"""
                is_full = self.size == self.capacity
                print(f"   Is full: {is_full}")
                return is_full
            
            def get_elements(self) -> List[int]:
                """Get current elements for visualization"""
                if self.isEmpty():
                    return []
                
                elements = []
                index = self.front
                for _ in range(self.size):
                    elements.append(self.queue[index])
                    index = (index + 1) % self.capacity
                return elements
        
        print("=== CIRCULAR QUEUE DESIGN ===")
        cq = CircularQueue(3)
        
        # Test operations
        operations = [
            ('enQueue', 1), ('enQueue', 2), ('enQueue', 3),
            ('enQueue', 4), ('Rear', None), ('isFull', None),
            ('deQueue', None), ('enQueue', 4), ('Rear', None)
        ]
        
        for op, value in operations:
            print(f"\nOperation: {op}" + (f"({value})" if value is not None else "()"))
            if op == 'enQueue':
                cq.enQueue(value)
            elif op == 'deQueue':
                cq.deQueue()
            elif op == 'Front':
                cq.Front()
            elif op == 'Rear':
                cq.Rear()
            elif op == 'isEmpty':
                cq.isEmpty()
            elif op == 'isFull':
                cq.isFull()
        
        return cq
    
    # ==========================================
    # 2. BFS AND GRAPH PROBLEMS
    # ==========================================
    
    def binary_tree_level_order_traversal(self, tree: Dict[str, List[str]], root: str) -> List[List[str]]:
        """
        Binary Tree Level Order Traversal
        
        Company: Amazon, Microsoft, Apple
        Difficulty: Medium
        Time: O(n), Space: O(w) where w is maximum width
        
        Return nodes level by level from left to right
        """
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        print(f"Level Order Traversal of tree from root '{root}'")
        print(f"Tree structure: {tree}")
        print()
        
        level = 1
        while queue:
            level_size = len(queue)
            current_level = []
            
            print(f"Level {level}: Processing {level_size} nodes")
            print(f"   Queue: {list(queue)}")
            
            for i in range(level_size):
                node = queue.popleft()
                current_level.append(node)
                print(f"   Process node '{node}'")
                
                # Add children to queue
                children = tree.get(node, [])
                for child in children:
                    queue.append(child)
                    print(f"     Add child '{child}' to queue")
            
            result.append(current_level)
            print(f"   Level {level} result: {current_level}")
            print(f"   Queue for next level: {list(queue)}")
            print()
            level += 1
        
        print(f"Final level order result: {result}")
        return result
    
    def binary_tree_zigzag_traversal(self, tree: Dict[str, List[str]], root: str) -> List[List[str]]:
        """
        Binary Tree Zigzag Level Order Traversal
        
        Company: Amazon, Microsoft, Facebook
        Difficulty: Medium
        Time: O(n), Space: O(w)
        
        Alternate direction for each level (left-to-right, then right-to-left)
        """
        if not root:
            return []
        
        result = []
        queue = deque([root])
        left_to_right = True
        
        print(f"Zigzag Level Order Traversal from root '{root}'")
        print(f"Tree structure: {tree}")
        print()
        
        level = 1
        while queue:
            level_size = len(queue)
            current_level = []
            
            direction = "left-to-right" if left_to_right else "right-to-left"
            print(f"Level {level}: {level_size} nodes, direction: {direction}")
            print(f"   Queue: {list(queue)}")
            
            for i in range(level_size):
                node = queue.popleft()
                current_level.append(node)
                
                # Add children to queue
                children = tree.get(node, [])
                for child in children:
                    queue.append(child)
            
            # Reverse if going right-to-left
            if not left_to_right:
                current_level.reverse()
                print(f"   Processed (then reversed): {current_level}")
            else:
                print(f"   Processed: {current_level}")
            
            result.append(current_level)
            left_to_right = not left_to_right
            
            print(f"   Next level direction: {'left-to-right' if left_to_right else 'right-to-left'}")
            print()
            level += 1
        
        print(f"Zigzag traversal result: {result}")
        return result
    
    def word_ladder(self, begin_word: str, end_word: str, word_list: List[str]) -> int:
        """
        Word Ladder - Shortest Transformation Sequence
        
        Company: Amazon, Facebook, Google
        Difficulty: Hard
        Time: O(M^2 * N), Space: O(M^2 * N)
        where M = length of words, N = number of words
        
        Find shortest path from begin_word to end_word, changing one letter at a time
        """
        if end_word not in word_list:
            return 0
        
        word_set = set(word_list)
        queue = deque([(begin_word, 1)])  # (word, length)
        visited = set([begin_word])
        
        print(f"Word Ladder: '{begin_word}' -> '{end_word}'")
        print(f"Word list: {word_list}")
        print()
        
        step = 1
        while queue:
            current_word, length = queue.popleft()
            
            print(f"Step {step}: Processing '{current_word}' (path length: {length})")
            
            if current_word == end_word:
                print(f"   ✓ Reached target word '{end_word}'!")
                print(f"   Shortest transformation length: {length}")
                return length
            
            # Try changing each character
            neighbors_found = 0
            for i in range(len(current_word)):
                for char in 'abcdefghijklmnopqrstuvwxyz':
                    if char != current_word[i]:
                        new_word = current_word[:i] + char + current_word[i+1:]
                        
                        if new_word in word_set and new_word not in visited:
                            visited.add(new_word)
                            queue.append((new_word, length + 1))
                            neighbors_found += 1
                            print(f"   Found neighbor: '{new_word}' (length: {length + 1})")
            
            if neighbors_found == 0:
                print(f"   No valid neighbors found for '{current_word}'")
            
            print(f"   Queue size: {len(queue)}")
            step += 1
            print()
        
        print(f"No transformation sequence found from '{begin_word}' to '{end_word}'")
        return 0
    
    # ==========================================
    # 3. SLIDING WINDOW WITH QUEUE
    # ==========================================
    
    def sliding_window_maximum(self, nums: List[int], k: int) -> List[int]:
        """
        Sliding Window Maximum using Deque
        
        Company: Amazon, Google, Microsoft
        Difficulty: Hard
        Time: O(n), Space: O(k)
        
        Find maximum element in each sliding window of size k
        """
        if not nums or k == 0:
            return []
        
        dq = deque()  # Store indices
        result = []
        
        print(f"Sliding Window Maximum: nums={nums}, k={k}")
        print("Using monotonic deque for O(n) solution")
        print()
        
        for i in range(len(nums)):
            print(f"Step {i+1}: Processing nums[{i}] = {nums[i]}")
            
            # Remove indices outside current window
            while dq and dq[0] <= i - k:
                removed_idx = dq.popleft()
                print(f"   Remove index {removed_idx} (outside window)")
            
            # Remove indices with smaller values
            while dq and nums[dq[-1]] < nums[i]:
                removed_idx = dq.pop()
                print(f"   Remove index {removed_idx} (value {nums[removed_idx]} < {nums[i]})")
            
            # Add current index
            dq.append(i)
            print(f"   Add index {i}")
            print(f"   Deque indices: {list(dq)}")
            print(f"   Deque values: {[nums[idx] for idx in dq]}")
            
            # If window is complete, record maximum
            if i >= k - 1:
                maximum = nums[dq[0]]
                result.append(maximum)
                window_start = i - k + 1
                window = nums[window_start:i+1]
                print(f"   Window [{window_start}:{i+1}]: {window} -> Maximum = {maximum}")
            
            print()
        
        print(f"Sliding window maximums: {result}")
        return result
    
    def perfect_squares(self, n: int) -> int:
        """
        Perfect Squares - Minimum number of perfect square numbers that sum to n
        
        Company: Google, Facebook
        Difficulty: Medium
        Time: O(n * sqrt(n)), Space: O(n)
        
        Use BFS to find shortest path (minimum steps)
        """
        if n <= 0:
            return 0
        
        # Generate perfect squares up to n
        perfect_squares = []
        i = 1
        while i * i <= n:
            perfect_squares.append(i * i)
            i += 1
        
        print(f"Finding minimum perfect squares that sum to {n}")
        print(f"Perfect squares <= {n}: {perfect_squares}")
        print()
        
        # BFS to find minimum steps
        queue = deque([(n, 0)])  # (remaining_sum, steps)
        visited = set([n])
        
        step = 1
        while queue:
            current_sum, steps = queue.popleft()
            
            print(f"Step {step}: Processing sum={current_sum}, steps={steps}")
            
            # Try subtracting each perfect square
            neighbors_added = 0
            for square in perfect_squares:
                if square > current_sum:
                    break
                
                new_sum = current_sum - square
                
                if new_sum == 0:
                    result = steps + 1
                    print(f"   ✓ Found solution: {current_sum} - {square} = 0")
                    print(f"   Minimum perfect squares needed: {result}")
                    return result
                
                if new_sum not in visited:
                    visited.add(new_sum)
                    queue.append((new_sum, steps + 1))
                    neighbors_added += 1
                    print(f"   Added: sum={new_sum}, steps={steps + 1} (subtracted {square})")
            
            if neighbors_added == 0:
                print(f"   No new neighbors added")
            
            step += 1
            if step > 10:  # Limit output for demo
                print("   ... (continuing search)")
                break
        
        # Fallback to complete search
        queue = deque([(n, 0)])
        visited = set([n])
        
        while queue:
            current_sum, steps = queue.popleft()
            
            for square in perfect_squares:
                if square > current_sum:
                    break
                
                new_sum = current_sum - square
                
                if new_sum == 0:
                    return steps + 1
                
                if new_sum not in visited:
                    visited.add(new_sum)
                    queue.append((new_sum, steps + 1))
        
        return -1  # Should never reach here for valid input
    
    # ==========================================
    # 4. DESIGN PROBLEMS
    # ==========================================
    
    def design_hit_counter(self):
        """
        Design Hit Counter
        
        Company: Google, Facebook
        Difficulty: Medium
        Time: O(1) for hit, O(s) for getHits where s is time window
        Space: O(s)
        
        Record hits and return hits in past 5 minutes (300 seconds)
        """
        
        class HitCounter:
            def __init__(self):
                self.hits = deque()  # Store (timestamp, count) pairs
                self.total_hits = 0
            
            def hit(self, timestamp: int) -> None:
                """Record a hit at given timestamp"""
                print(f"Hit at timestamp {timestamp}")
                
                # Remove hits older than 300 seconds
                self._cleanup(timestamp)
                
                # Add current hit
                if self.hits and self.hits[-1][0] == timestamp:
                    # Same timestamp, increment count
                    old_count = self.hits[-1][1]
                    self.hits[-1] = (timestamp, old_count + 1)
                    self.total_hits += 1
                    print(f"   Updated timestamp {timestamp}: count = {old_count + 1}")
                else:
                    # New timestamp
                    self.hits.append((timestamp, 1))
                    self.total_hits += 1
                    print(f"   New timestamp {timestamp}: count = 1")
                
                print(f"   Total hits in window: {self.total_hits}")
                print(f"   Hits queue: {list(self.hits)}")
            
            def getHits(self, timestamp: int) -> int:
                """Get hits in past 300 seconds from given timestamp"""
                print(f"Get hits at timestamp {timestamp}")
                
                # Remove hits older than 300 seconds
                self._cleanup(timestamp)
                
                print(f"   Hits in past 300 seconds: {self.total_hits}")
                print(f"   Hits queue: {list(self.hits)}")
                
                return self.total_hits
            
            def _cleanup(self, timestamp: int) -> None:
                """Remove hits older than 300 seconds"""
                removed_count = 0
                while self.hits and self.hits[0][0] <= timestamp - 300:
                    old_timestamp, count = self.hits.popleft()
                    self.total_hits -= count
                    removed_count += count
                    print(f"   Removed {count} hits from timestamp {old_timestamp}")
                
                if removed_count > 0:
                    print(f"   Total removed: {removed_count} hits")
        
        print("=== HIT COUNTER DESIGN ===")
        hc = HitCounter()
        
        # Test operations
        operations = [
            ('hit', 1), ('hit', 2), ('hit', 3),
            ('getHits', 4), ('hit', 300), ('getHits', 300),
            ('getHits', 301)
        ]
        
        for op, timestamp in operations:
            print(f"\nOperation: {op}({timestamp})")
            if op == 'hit':
                hc.hit(timestamp)
            elif op == 'getHits':
                result = hc.getHits(timestamp)
                print(f"   Result: {result}")
        
        return hc
    
    # ==========================================
    # 5. INTERVIEW STRATEGY AND TIPS
    # ==========================================
    
    def interview_approach_guide(self) -> None:
        """
        Comprehensive guide for tackling queue interview problems
        """
        print("=== QUEUE INTERVIEW APPROACH GUIDE ===")
        print()
        
        print("🎯 STEP 1: PROBLEM RECOGNITION")
        print("Look for these keywords/patterns:")
        print("• 'Level by level', 'layer by layer'")
        print("• 'Breadth-first', 'shortest path' (unweighted)")
        print("• 'First come, first served', 'FIFO'")
        print("• 'Process in order', 'preserve order'")
        print("• 'Sliding window', 'moving window'")
        print("• 'Multi-source', 'spreading'")
        print()
        
        print("🎯 STEP 2: CHOOSE QUEUE APPROACH")
        print("Decision framework:")
        print("• FIFO processing needed? → Standard Queue")
        print("• Level-by-level traversal? → BFS with Queue")
        print("• Shortest path (unweighted)? → BFS")
        print("• Sliding window maximum? → Monotonic Deque")
        print("• Multi-source shortest path? → Multi-source BFS")
        print("• Need both ends access? → Deque")
        print()
        
        print("🎯 STEP 3: IMPLEMENTATION TEMPLATES")
        print()
        print("A) Basic BFS Template:")
        print("   queue = deque([start])")
        print("   visited = set([start])")
        print("   while queue:")
        print("       current = queue.popleft()")
        print("       for neighbor in get_neighbors(current):")
        print("           if neighbor not in visited:")
        print("               visited.add(neighbor)")
        print("               queue.append(neighbor)")
        print()
        print("B) Level-order Template:")
        print("   queue = deque([root])")
        print("   while queue:")
        print("       level_size = len(queue)")
        print("       for _ in range(level_size):")
        print("           node = queue.popleft()")
        print("           # Process node")
        print("           # Add children to queue")
        print()
        print("C) Sliding Window Template:")
        print("   dq = deque()  # monotonic deque")
        print("   for i, element in enumerate(array):")
        print("       # Remove outside window")
        print("       # Maintain monotonic property")
        print("       # Add current element")
        print("       # Record result if window complete")
        print()
        
        print("🎯 STEP 4: OPTIMIZATION CONSIDERATIONS")
        print("• Use deque for efficient front/rear operations")
        print("• Store additional info in queue (distance, level, path)")
        print("• Consider space optimization with in-place modifications")
        print("• Use monotonic deque for sliding window problems")
        print("• Multi-source BFS for problems with multiple starting points")
        print()
        
        print("🎯 STEP 5: TESTING STRATEGY")
        print("Test cases to consider:")
        print("• Empty input or single element")
        print("• All elements same vs all different")
        print("• Minimum and maximum constraints")
        print("• Disconnected components (for graphs)")
        print("• Cycles and special graph structures")
    
    def common_mistakes(self) -> None:
        """
        Common mistakes in queue interview problems
        """
        print("=== COMMON QUEUE INTERVIEW MISTAKES ===")
        print()
        
        print("❌ MISTAKE 1: Using wrong end for dequeue")
        print("Problem: Using pop() instead of popleft() for FIFO")
        print("Solution: Always use popleft() for standard queue behavior")
        print()
        
        print("❌ MISTAKE 2: Not handling level-by-level processing correctly")
        print("Problem: Processing all nodes in queue instead of current level")
        print("Solution: Store level_size = len(queue) before processing")
        print()
        
        print("❌ MISTAKE 3: Forgetting to mark nodes as visited")
        print("Problem: Infinite loops in graph traversal")
        print("Solution: Always maintain visited set for BFS")
        print()
        
        print("❌ MISTAKE 4: Incorrect sliding window deque maintenance")
        print("Problem: Not maintaining monotonic property correctly")
        print("Solution: Carefully handle both ends of deque")
        print()
        
        print("❌ MISTAKE 5: Not optimizing space for large inputs")
        print("Problem: Storing unnecessary information in queue")
        print("Solution: Store minimal required information (indices vs values)")
        print()
        
        print("❌ MISTAKE 6: Misunderstanding problem requirements")
        print("Problem: Using BFS when DFS is needed or vice versa")
        print("Solution: Understand if you need shortest path vs any path")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_queue_interview_problems():
    """Demonstrate key queue interview problems"""
    print("=== QUEUE INTERVIEW PROBLEMS DEMONSTRATION ===\n")
    
    problems = QueueInterviewProblems()
    
    # 1. Classic design problems
    print("=== CLASSIC DESIGN PROBLEMS ===")
    
    print("1. Queue Using Stacks:")
    problems.implement_queue_using_stacks()
    print("\n" + "-" * 60 + "\n")
    
    print("2. Stack Using Queues:")
    problems.implement_stack_using_queues()
    print("\n" + "-" * 60 + "\n")
    
    print("3. Circular Queue Design:")
    problems.circular_queue_design()
    print("\n" + "=" * 60 + "\n")
    
    # 2. BFS and tree problems
    print("=== BFS AND TREE PROBLEMS ===")
    
    tree = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F', 'G'],
        'D': [], 'E': ['H'], 'F': [], 'G': [], 'H': []
    }
    
    print("1. Binary Tree Level Order Traversal:")
    problems.binary_tree_level_order_traversal(tree, 'A')
    print("\n" + "-" * 60 + "\n")
    
    print("2. Binary Tree Zigzag Traversal:")
    problems.binary_tree_zigzag_traversal(tree, 'A')
    print("\n" + "-" * 60 + "\n")
    
    print("3. Word Ladder:")
    word_list = ["hot","dot","dog","lot","log","cog"]
    problems.word_ladder("hit", "cog", word_list)
    print("\n" + "=" * 60 + "\n")
    
    # 3. Sliding window problems
    print("=== SLIDING WINDOW PROBLEMS ===")
    
    print("1. Sliding Window Maximum:")
    problems.sliding_window_maximum([1,3,-1,-3,5,3,6,7], 3)
    print("\n" + "-" * 60 + "\n")
    
    print("2. Perfect Squares:")
    problems.perfect_squares(12)
    print("\n" + "=" * 60 + "\n")
    
    # 4. Design problems
    print("=== ADVANCED DESIGN PROBLEMS ===")
    
    print("1. Hit Counter Design:")
    problems.design_hit_counter()
    print("\n" + "=" * 60 + "\n")
    
    # 5. Interview guidance
    problems.interview_approach_guide()
    print("\n" + "=" * 60 + "\n")
    
    problems.common_mistakes()


if __name__ == "__main__":
    demonstrate_queue_interview_problems()
    
    print("\n=== QUEUE INTERVIEW SUCCESS STRATEGY ===")
    
    print("\n🎯 PREPARATION ROADMAP:")
    print("Week 1: Master queue operations and basic design problems")
    print("Week 2: Practice BFS algorithms and tree traversals")
    print("Week 3: Tackle sliding window and deque problems")
    print("Week 4: Solve complex design and optimization problems")
    
    print("\n📚 MUST-PRACTICE PROBLEMS:")
    print("• Queue Using Stacks (Easy)")
    print("• Binary Tree Level Order Traversal (Medium)")
    print("• Word Ladder (Hard)")
    print("• Sliding Window Maximum (Hard)")
    print("• Perfect Squares (Medium)")
    print("• Design Hit Counter (Medium)")
    print("• Rotting Oranges (Medium)")
    print("• Open the Lock (Medium)")
    
    print("\n⚡ QUICK PROBLEM IDENTIFICATION:")
    print("• Level-by-level processing → BFS with queue")
    print("• Shortest path unweighted → BFS")
    print("• FIFO behavior → Standard queue")
    print("• Sliding window maximum → Monotonic deque")
    print("• Design with time windows → Queue with cleanup")
    
    print("\n🏆 INTERVIEW DAY TIPS:")
    print("• Clarify if you need shortest path or any path")
    print("• Ask about graph properties (directed, weighted, cycles)")
    print("• Consider space optimization for large inputs")
    print("• Draw examples for tree/graph traversals")
    print("• Explain BFS vs DFS trade-offs")
    print("• Test with edge cases (empty, single node, disconnected)")
    
    print("\n📊 COMPLEXITY GOALS:")
    print("• BFS: O(V + E) time, O(V) space")
    print("• Level-order: O(n) time, O(w) space (w = max width)")
    print("• Sliding window: O(n) time with deque optimization")
    print("• Design problems: O(1) amortized for operations")
    
    print("\n🎓 ADVANCED PREPARATION:")
    print("• Study different queue implementations and trade-offs")
    print("• Practice with memory and time constraints")
    print("• Learn bidirectional BFS for optimization")
    print("• Understand when to use BFS vs DFS vs other approaches")
    print("• Practice explaining solutions clearly and concisely")

