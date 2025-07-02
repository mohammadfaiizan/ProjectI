"""
Array Stack and Queue Operations - Implementation & Problems
===========================================================

Topics: Stack using arrays, queue using arrays, monotonic stack/queue
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Medium to Hard
"""

from typing import List, Optional
from collections import deque

class ArrayStackQueueOperations:
    
    # ==========================================
    # 1. STACK USING ARRAYS
    # ==========================================
    
    class ArrayStack:
        """Stack implementation using dynamic array"""
        
        def __init__(self, capacity: int = 1000):
            self.data = [0] * capacity
            self.top_index = -1
            self.capacity = capacity
        
        def push(self, val: int) -> bool:
            if self.top_index + 1 >= self.capacity:
                return False
            self.top_index += 1
            self.data[self.top_index] = val
            return True
        
        def pop(self) -> bool:
            if self.top_index == -1:
                return False
            self.top_index -= 1
            return True
        
        def top(self) -> int:
            return self.data[self.top_index] if self.top_index != -1 else -1
        
        def empty(self) -> bool:
            return self.top_index == -1
    
    # ==========================================
    # 2. QUEUE USING ARRAYS
    # ==========================================
    
    class ArrayQueue:
        """Circular queue implementation using array"""
        
        def __init__(self, k: int):
            self.data = [0] * k
            self.head = -1
            self.tail = -1
            self.size = k
        
        def enqueue(self, value: int) -> bool:
            if self.is_full():
                return False
            
            if self.is_empty():
                self.head = 0
            
            self.tail = (self.tail + 1) % self.size
            self.data[self.tail] = value
            return True
        
        def dequeue(self) -> bool:
            if self.is_empty():
                return False
            
            if self.head == self.tail:
                self.head = -1
                self.tail = -1
                return True
            
            self.head = (self.head + 1) % self.size
            return True
        
        def front(self) -> int:
            return self.data[self.head] if not self.is_empty() else -1
        
        def rear(self) -> int:
            return self.data[self.tail] if not self.is_empty() else -1
        
        def is_empty(self) -> bool:
            return self.head == -1
        
        def is_full(self) -> bool:
            return ((self.tail + 1) % self.size) == self.head
    
    # ==========================================
    # 3. MONOTONIC STACK PROBLEMS
    # ==========================================
    
    def next_greater_element(self, nums: List[int]) -> List[int]:
        """LC 496: Next Greater Element using monotonic stack
        Time: O(n), Space: O(n)
        """
        result = [-1] * len(nums)
        stack = []
        
        for i in range(len(nums)):
            while stack and nums[stack[-1]] < nums[i]:
                idx = stack.pop()
                result[idx] = nums[i]
            stack.append(i)
        
        return result
    
    def daily_temperatures(self, temperatures: List[int]) -> List[int]:
        """LC 739: Daily Temperatures
        Time: O(n), Space: O(n)
        """
        result = [0] * len(temperatures)
        stack = []
        
        for i, temp in enumerate(temperatures):
            while stack and temperatures[stack[-1]] < temp:
                idx = stack.pop()
                result[idx] = i - idx
            stack.append(i)
        
        return result
    
    def largest_rectangle_histogram(self, heights: List[int]) -> int:
        """LC 84: Largest Rectangle in Histogram
        Time: O(n), Space: O(n)
        """
        stack = []
        max_area = 0
        
        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        
        while stack:
            height = heights[stack.pop()]
            width = len(heights) if not stack else len(heights) - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        return max_area
    
    def sliding_window_maximum(self, nums: List[int], k: int) -> List[int]:
        """LC 239: Sliding Window Maximum using deque
        Time: O(n), Space: O(k)
        """
        if not nums or k == 0:
            return []
        
        dq = deque()
        result = []
        
        for i in range(len(nums)):
            # Remove indices outside current window
            while dq and dq[0] <= i - k:
                dq.popleft()
            
            # Remove indices of smaller elements
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop()
            
            dq.append(i)
            
            if i >= k - 1:
                result.append(nums[dq[0]])
        
        return result
    
    # ==========================================
    # 4. ADVANCED STACK PROBLEMS
    # ==========================================
    
    def valid_parentheses(self, s: str) -> bool:
        """LC 20: Valid Parentheses
        Time: O(n), Space: O(n)
        """
        stack = []
        mapping = {")": "(", "}": "{", "]": "["}
        
        for char in s:
            if char in mapping:
                if not stack or stack.pop() != mapping[char]:
                    return False
            else:
                stack.append(char)
        
        return not stack
    
    def min_stack(self):
        """LC 155: Min Stack implementation
        """
        class MinStack:
            def __init__(self):
                self.stack = []
                self.min_stack = []
            
            def push(self, val: int) -> None:
                self.stack.append(val)
                if not self.min_stack or val <= self.min_stack[-1]:
                    self.min_stack.append(val)
            
            def pop(self) -> None:
                if self.stack:
                    if self.stack[-1] == self.min_stack[-1]:
                        self.min_stack.pop()
                    self.stack.pop()
            
            def top(self) -> int:
                return self.stack[-1] if self.stack else 0
            
            def get_min(self) -> int:
                return self.min_stack[-1] if self.min_stack else 0
        
        return MinStack()
    
    def evaluate_rpn(self, tokens: List[str]) -> int:
        """LC 150: Evaluate Reverse Polish Notation
        Time: O(n), Space: O(n)
        """
        stack = []
        operators = {'+', '-', '*', '/'}
        
        for token in tokens:
            if token in operators:
                b = stack.pop()
                a = stack.pop()
                
                if token == '+':
                    stack.append(a + b)
                elif token == '-':
                    stack.append(a - b)
                elif token == '*':
                    stack.append(a * b)
                else:  # token == '/'
                    stack.append(int(a / b))
            else:
                stack.append(int(token))
        
        return stack[0]
    
    # ==========================================
    # 5. QUEUE-BASED PROBLEMS
    # ==========================================
    
    def moving_average(self, size: int):
        """LC 346: Moving Average from Data Stream
        """
        class MovingAverage:
            def __init__(self, size: int):
                self.size = size
                self.queue = deque()
                self.window_sum = 0
            
            def next(self, val: int) -> float:
                self.queue.append(val)
                self.window_sum += val
                
                if len(self.queue) > self.size:
                    self.window_sum -= self.queue.popleft()
                
                return self.window_sum / len(self.queue)
        
        return MovingAverage(size)
    
    def walls_and_gates(self, rooms: List[List[int]]) -> None:
        """LC 286: Walls and Gates (Multi-source BFS)
        Time: O(m*n), Space: O(m*n)
        """
        if not rooms or not rooms[0]:
            return
        
        m, n = len(rooms), len(rooms[0])
        queue = deque()
        INF = 2147483647
        
        # Find all gates
        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0:
                    queue.append((i, j))
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            row, col = queue.popleft()
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < m and 0 <= new_col < n and 
                    rooms[new_row][new_col] == INF):
                    rooms[new_row][new_col] = rooms[row][col] + 1
                    queue.append((new_row, new_col))
    
    def perfect_squares(self, n: int) -> int:
        """LC 279: Perfect Squares using BFS
        Time: O(n * sqrt(n)), Space: O(n)
        """
        if n <= 0:
            return 0
        
        perfect_squares = []
        i = 1
        while i * i <= n:
            perfect_squares.append(i * i)
            i += 1
        
        queue = deque([(n, 0)])
        visited = {n}
        
        while queue:
            num, steps = queue.popleft()
            
            for square in perfect_squares:
                if square > num:
                    break
                
                next_num = num - square
                if next_num == 0:
                    return steps + 1
                
                if next_num not in visited:
                    visited.add(next_num)
                    queue.append((next_num, steps + 1))
        
        return 0

# Test Examples
def run_examples():
    asqo = ArrayStackQueueOperations()
    
    print("=== ARRAY STACK AND QUEUE OPERATIONS ===\n")
    
    # Stack operations
    print("1. STACK OPERATIONS:")
    stack = asqo.ArrayStack(5)
    operations = [("push", 1), ("push", 2), ("top", None), ("pop", None), ("empty", None)]
    
    for op, val in operations:
        if op == "push":
            result = stack.push(val)
            print(f"Push {val}: {result}")
        elif op == "pop":
            result = stack.pop()
            print(f"Pop: {result}")
        elif op == "top":
            result = stack.top()
            print(f"Top: {result}")
        elif op == "empty":
            result = stack.empty()
            print(f"Empty: {result}")
    
    # Queue operations
    print("\n2. QUEUE OPERATIONS:")
    queue = asqo.ArrayQueue(3)
    queue.enqueue(1)
    queue.enqueue(2)
    print(f"Front: {queue.front()}, Rear: {queue.rear()}")
    queue.dequeue()
    print(f"After dequeue - Front: {queue.front()}")
    
    # Monotonic stack problems
    print("\n3. MONOTONIC STACK PROBLEMS:")
    temps = [73, 74, 75, 71, 69, 72, 76, 73]
    result = asqo.daily_temperatures(temps)
    print(f"Daily temperatures: {result}")
    
    heights = [2, 1, 5, 6, 2, 3]
    area = asqo.largest_rectangle_histogram(heights)
    print(f"Largest rectangle area: {area}")
    
    # Advanced problems
    print("\n4. ADVANCED PROBLEMS:")
    s = "()[]{}"
    valid = asqo.valid_parentheses(s)
    print(f"Valid parentheses '{s}': {valid}")
    
    tokens = ["2", "1", "+", "3", "*"]
    result = asqo.evaluate_rpn(tokens)
    print(f"RPN evaluation {tokens}: {result}")

if __name__ == "__main__":
    run_examples() 